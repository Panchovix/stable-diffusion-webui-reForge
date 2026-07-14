"""
jax_pipeline/vae.py — pure-functional JAX KL-F8 VAE decoder for SDXL.

Architecture (SDXL KL-F8), mirrors mlx_pipeline/vae.py:
  ddconfig = {
    'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
    'z_channels': 4, 'out_ch': 3, 'attn_resolutions': [],
    'double_z': True,
  }

Decoder pipeline:
  post_quant_conv  Conv(4->4, 1x1)
  conv_in          Conv(4->512, 3x3)
  mid              ResNet(512) -> Attn(512) -> ResNet(512)
  up[3]            3xResNet(512->512) + Upsample(512)
  up[2]            3xResNet(512->512) + Upsample(512)
  up[1]            ResNet(512->256) + 2xResNet(256) + Upsample(256)
  up[0]            ResNet(256->128) + 2xResNet(128)  [no upsample]
  norm_out         GroupNorm(32, 128) + SiLU
  conv_out         Conv(128->3, 3x3)

State-dict key mapping
-----------------------
All weights are loaded from ``first_stage_model.state_dict()``, which uses
``decoder.*`` / ``encoder.*`` / ``quant_conv.*`` / ``post_quant_conv.*``
prefixes directly (ldm-native naming, no HF renaming needed, unlike the
UNet). Every 4-D weight tensor in this state dict is a conv weight (no
ambiguity the way UNet has Linear-vs-Conv), so every 4-D tensor is
unconditionally transposed OIHW -> HWIO for jax.lax.conv_general_dilated's
('NHWC','HWIO','NHWC') layout.

Integration
-----------
Wraps ``first_stage_model.decode()`` to use the JAX implementation when
active. Falls back to torch for unusual decoder kwargs (3-D conv / video
VAE) or on any runtime error. The encoder is intentionally left as torch —
same rationale as mlx_pipeline: img2img encode runs once per generation and
needs the reparameterization code path intact.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)

_NORM_GROUPS = 32
_NORM_EPS = 1e-6  # VAE GroupNorm eps differs from UNet's 1e-5


# -- conversion ----------------------------------------------------------------

def _tensor_to_jax(tensor: "torch.Tensor", dtype):
    import jax.numpy as jnp
    import numpy as np

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = np.transpose(arr, (2, 3, 1, 0))  # OIHW -> HWIO
    return jnp.asarray(arr, dtype=dtype)


def load_vae_params(first_stage_model, dtype=None) -> Dict[str, "jnp.ndarray"]:
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.bfloat16
    sd = first_stage_model.state_dict()
    return {k: _tensor_to_jax(v, dtype) for k, v in sd.items()}


def _detect_decoder_arch(sd_state: Dict[str, "torch.Tensor"]):
    max_up_level = max(int(k.split(".")[2]) for k in sd_state if k.startswith("decoder.up."))
    num_up = max_up_level + 1
    num_res = max(
        int(k.split(".")[4]) for k in sd_state if k.startswith("decoder.up.") and ".block." in k
    ) + 1
    return num_up, num_res


# -- primitive layers ------------------------------------------------------------

def _silu(x):
    import jax
    return jax.nn.silu(x)


def conv2d(params, prefix, x, stride: int = 1, padding: int = 1):
    import jax

    w = params[f"{prefix}.weight"]  # [kH, kW, I, O]
    b = params.get(f"{prefix}.bias")
    out = jax.lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if b is not None:
        out = out + b[None, None, None, :]
    return out


def group_norm(params, prefix, x, num_groups: int = _NORM_GROUPS, eps: float = _NORM_EPS):
    import jax.numpy as jnp

    w = params[f"{prefix}.weight"]
    b = params[f"{prefix}.bias"]
    B, H, W, C = x.shape
    G = num_groups
    x32 = x.astype(jnp.float32).reshape(B, H, W, G, C // G)
    mean = jnp.mean(x32, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(x32, axis=(1, 2, 4), keepdims=True)
    normed = (x32 - mean) / jnp.sqrt(var + eps)
    normed = normed.reshape(B, H, W, C)
    out = normed * w.astype(jnp.float32)[None, None, None, :] + b.astype(jnp.float32)[None, None, None, :]
    return out.astype(x.dtype)


def _upsample2x(x):
    import jax.numpy as jnp
    x = jnp.repeat(x, 2, axis=1)
    x = jnp.repeat(x, 2, axis=2)
    return x


# -- blocks ------------------------------------------------------------------

def resnet_block(params, prefix, x):
    """ResNet block without timestep embedding (VAE has no time conditioning)."""
    has_shortcut = f"{prefix}.nin_shortcut.weight" in params
    residual = x

    h = _silu(group_norm(params, f"{prefix}.norm1", x))
    h = conv2d(params, f"{prefix}.conv1", h, stride=1, padding=1)
    h = _silu(group_norm(params, f"{prefix}.norm2", h))
    h = conv2d(params, f"{prefix}.conv2", h, stride=1, padding=1)

    if has_shortcut:
        residual = conv2d(params, f"{prefix}.nin_shortcut", residual, stride=1, padding=0)

    return h + residual


def attn_block(params, prefix, x):
    """Spatial single-head self-attention (mid block only)."""
    import jax
    import jax.numpy as jnp

    N, H, W, C = x.shape
    h = group_norm(params, f"{prefix}.norm", x)
    q = conv2d(params, f"{prefix}.q", h, stride=1, padding=0).reshape(N, H * W, C)
    k = conv2d(params, f"{prefix}.k", h, stride=1, padding=0).reshape(N, H * W, C)
    v = conv2d(params, f"{prefix}.v", h, stride=1, padding=0).reshape(N, H * W, C)

    scale = C ** -0.5
    attn = jnp.einsum("nqc,nkc->nqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
    out = jnp.einsum("nqk,nkc->nqc", attn, v)
    out = out.reshape(N, H, W, C)
    out = conv2d(params, f"{prefix}.proj_out", out, stride=1, padding=0)
    return x + out


def upsample_2d(params, prefix, x):
    return conv2d(params, f"{prefix}.conv", _upsample2x(x), stride=1, padding=1)


def mid_block(params, prefix, x):
    x = resnet_block(params, f"{prefix}.block_1", x)
    x = attn_block(params, f"{prefix}.attn_1", x)
    x = resnet_block(params, f"{prefix}.block_2", x)
    return x


# -- top-level decode ----------------------------------------------------------

def vae_decode(params, z, num_up: int, num_res: int):
    """z: [B,4,H,W] NCHW -> image [B,3,H*8,W*8] NCHW."""
    import jax.numpy as jnp

    x = jnp.transpose(z, (0, 2, 3, 1))  # NCHW -> NHWC

    x = conv2d(params, "post_quant_conv", x, stride=1, padding=0)
    x = conv2d(params, "decoder.conv_in", x, stride=1, padding=1)
    x = mid_block(params, "decoder.mid", x)

    # ldm stores up blocks low-res-first (up[0]..up[num_up-1]); decode order
    # is reversed (highest index = lowest resolution = decoded first).
    for i in reversed(range(num_up)):
        for j in range(num_res):
            x = resnet_block(params, f"decoder.up.{i}.block.{j}", x)
        if f"decoder.up.{i}.upsample.conv.weight" in params:
            x = upsample_2d(params, f"decoder.up.{i}.upsample", x)

    x = _silu(group_norm(params, "decoder.norm_out", x))
    x = conv2d(params, "decoder.conv_out", x, stride=1, padding=1)

    return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW


def make_decode_jit(num_up: int, num_res: int):
    """Bake the static architecture config into a closure, then jit only
    over (params, z) — avoids needing static_argnames for plain Python ints.
    """
    import jax

    fn = functools.partial(vae_decode, num_up=num_up, num_res=num_res)
    return jax.jit(fn)


# -- integration hook ------------------------------------------------------------

def install_vae_hooks(sd_model) -> bool:
    """Monkey-patch ``first_stage_model.decode()`` to use the JAX decoder.

    Only patches the standard KL-F8 path (no 3-D conv / video VAE). Falls
    back to torch on unsupported decoder kwargs or any runtime error.

    The encoder is intentionally left as torch: img2img encode happens once
    per generation, and reparameterization requires keeping the
    regularization code path intact (same rationale as mlx_pipeline).
    """
    try:
        import jax.numpy as jnp

        fst = sd_model.first_stage_model
        sd_state = fst.state_dict()
        if "decoder.conv_in.weight" not in sd_state:
            log.debug("[JAX VAE] Non-standard decoder; skipping hook")
            return False
        if sd_state["decoder.conv_in.weight"].ndim != 4:
            log.debug("[JAX VAE] 3-D (video) VAE detected; skipping hook")
            return False

        num_up, num_res = _detect_decoder_arch(sd_state)
        params = load_vae_params(fst)
        decode_jit = make_decode_jit(num_up, num_res)

        _orig_decode = fst.decode

        def _jax_decode(z: torch.Tensor, **kwargs) -> torch.Tensor:
            if kwargs:
                return _orig_decode(z, **kwargs)
            try:
                z_np = z.detach().float().cpu().numpy()
                z_jax = jnp.asarray(z_np, dtype=jnp.bfloat16)
                out = decode_jit(params, z_jax)
                out_np = np.asarray(jnp.asarray(out, dtype=jnp.float32))
                return torch.from_numpy(out_np.copy()).float().to(z.device)
            except Exception as e:
                log.warning("[JAX VAE] Decoder error: %s - falling back to torch", e, exc_info=True)
                return _orig_decode(z)

        fst.decode = _jax_decode

        # Keep references alive (prevents GC)
        sd_model._jax_vae_params = params
        sd_model._jax_vae_decode_jit = decode_jit

        log.info(
            "[JAX VAE] Hooked first_stage_model.decode - JAX jit VAE decoding active "
            "(up_levels=%d res_blocks=%d)", num_up, num_res,
        )
        return True

    except Exception as e:
        log.warning("[JAX VAE] Hook installation failed: %s", e, exc_info=True)
        return False
