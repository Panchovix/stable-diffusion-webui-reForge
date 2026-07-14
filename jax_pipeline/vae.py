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

Tiled decode
------------
Large images can exceed available VRAM even though the UNet sampling loop
fit comfortably (VAE decode activations scale with *pixel*-space size, not
latent size). Two things make this cheap to handle well:

1. ``should_tile_decode()`` estimates peak decode memory the same way
   ``diff_pipeline/adapter.py::_decode_needs_tiling`` already does for the
   diffusers backend (peak activation ~= upsampled_H * upsampled_W *
   max_channels * elem_bytes * 4, checked against a fraction of currently
   free VRAM) and decides *before* attempting a decode, instead of
   reactively catching a CUDA OOM from a doomed full-resolution attempt
   and retrying (which is what happens by default: this hook's fallback on
   any exception is the original torch ``decode()``, whose own OOM handler
   in ``ldm_patched.modules.sd.VAE.decode()`` then retries tiled — that
   still works, but wastes two full-resolution attempts first).
2. When tiling is needed, ``jax_decode_tiled()`` reuses
   ``ldm_patched.modules.utils.tiled_scale`` — the same slice/dispatch/
   blend engine the torch path already uses — driven by a JAX-backed
   per-tile decode function, so tiles get the same jax.jit-compiled path
   as a full decode (and share one compiled executable across tiles, since
   every tile has the same shape).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)

_NORM_GROUPS = 32
_NORM_EPS = 1e-6  # VAE GroupNorm eps differs from UNet's 1e-5

# SDXL KL-F8 ddconfig: ch=128, ch_mult=[1,2,4,4] -> per-level channel counts.
# Matches diff_pipeline/adapter.py's fallback default for the same formula.
_BLOCK_OUT_CHANNELS: Tuple[int, ...] = (128, 256, 512, 512)


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


# -- VRAM estimation / tiled decode ---------------------------------------------

def estimate_decode_peak_bytes(
    latent_shape,
    elem_bytes: int,
    block_out_channels: Tuple[int, ...] = _BLOCK_OUT_CHANNELS,
) -> int:
    """Rough peak-activation estimate for a full-resolution VAE decode.

    Same formula as ``diff_pipeline/adapter.py::DiffusersModelAdapter.
    _decode_needs_tiling`` (this repo's other backend already validated
    this heuristic in production): the largest intermediate tensor in the
    decoder is roughly the final upsampled spatial size at the widest
    channel count, times 4 (accounts for input + output of the block's
    largest conv/resnet intermediate).
    """
    _, _, lh, lw = latent_shape
    n_up = len(block_out_channels)
    scale = 2 ** (n_up - 1)  # each up level doubles spatial resolution
    peak_h = lh * scale
    peak_w = lw * scale
    peak_ch = max(block_out_channels)
    return peak_h * peak_w * peak_ch * elem_bytes * 4


def should_tile_decode(device, latent_shape, elem_bytes: int = 4, threshold_frac: float = 0.8) -> bool:
    """Decide, before attempting a decode, whether it should go straight to
    tiled decoding rather than risk (and waste time on) a doomed
    full-resolution attempt.
    """
    try:
        from ldm_patched.modules import model_management
        free_bytes = model_management.get_free_memory(device)
    except Exception:
        return False

    peak = estimate_decode_peak_bytes(latent_shape, elem_bytes)
    return peak > free_bytes * threshold_frac


def _jax_tile_decode_fn(decode_jit, params, tile_h: int, tile_w: int):
    """Return a decode_fn(tile: torch.Tensor) -> torch.Tensor for
    ldm_patched.modules.utils.tiled_scale to call once per tile.

    Zero-pads every tile up to the FIXED (tile_h, tile_w) latent shape
    before decoding, and crops the output back down afterward, so every
    call within one tiled_scale pass reuses the SAME jax.jit-compiled
    executable.

    Without this, edge tiles (any latent whose H/W isn't an exact multiple
    of the tile size — the common case) are narrower/shorter than the
    requested tile size (``tiled_scale_multidim`` truncates via
    ``.narrow()`` rather than padding), so each distinct edge shape
    triggers its own fresh XLA compile + cuDNN algorithm autotune. That is
    slow (autotuning benchmarks several candidate conv algorithms per new
    shape) and, with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (this
    backend's default — see ``jax_pipeline.__init__``), can itself OOM:
    autotuning needs its own scratch-memory allocation on top of the real
    convolution, and there's no large reserved arena to draw it from.

    Padding is safe here: convolutions already implicitly zero-pad at
    tensor boundaries, so an explicitly zero-padded region beyond the real
    tile behaves the same, near the real/pad boundary, as the tensor
    simply ending there — the padded region itself is discarded by the
    crop and never feeds into ``tiled_scale``'s blend.
    """
    import jax.numpy as jnp
    import torch.nn.functional as F

    def decode_fn(tile: "torch.Tensor") -> "torch.Tensor":
        _, _, th, tw = tile.shape
        pad_h = max(0, tile_h - th)
        pad_w = max(0, tile_w - tw)
        if pad_h or pad_w:
            tile = F.pad(tile, (0, pad_w, 0, pad_h))  # zero-pad right/bottom only

        z_np = tile.detach().float().cpu().numpy()
        z_jax = jnp.asarray(z_np, dtype=jnp.bfloat16)
        out = decode_jit(params, z_jax)
        out_np = np.asarray(jnp.asarray(out, dtype=jnp.float32))
        out_t = torch.from_numpy(out_np.copy()).float()

        if pad_h or pad_w:
            out_t = out_t[:, :, : th * 8, : tw * 8]  # crop back to the real tile's output size

        return out_t

    return decode_fn


def jax_decode_tiled(
    decode_jit, params, z: "torch.Tensor",
    tile_x: int = None, tile_y: int = None, overlap: int = 16,
) -> "torch.Tensor":
    """Tiled VAE decode driven entirely by JAX per tile.

    Reuses ``ldm_patched.modules.utils.tiled_scale`` — the same proven
    slice/dispatch/feathered-blend engine ``VAE.decode_tiled_()`` uses —
    instead of hand-rolling tiling/blending. Every tile within a given
    pass is padded to that pass's fixed (tile_h, tile_w) shape (see
    ``_jax_tile_decode_fn``), so each pass reuses exactly one
    jax.jit-compiled executable rather than recompiling per distinct edge
    shape.

    Replicates ``decode_tiled_()``'s 3-pass-average-over-aspect-ratios
    trick (this repo has no GroupNorm-sync "real" Tiled VAE, so blending
    result from 3 differently-shaped tile grids is the seam-reduction
    substitute already in use here — kept for consistency with the torch
    tiled path's established output quality). This still means 3 distinct
    compiled shapes total (one per aspect ratio), not 1 — a further
    speed/robustness trade would be dropping to a single pass, at a small
    cost to seam quality.

    ``tile_x``/``tile_y`` default to ``model_management.VAE_DECODE_TILE_SIZE_X/Y``
    (the same globals the "Never OOM" UI script's sliders write to) so this
    respects whatever the user has already configured there.
    """
    from ldm_patched.modules import model_management
    from ldm_patched.modules import utils as ldm_utils

    if tile_x is None:
        tile_x = model_management.VAE_DECODE_TILE_SIZE_X
    if tile_y is None:
        tile_y = model_management.VAE_DECODE_TILE_SIZE_Y

    def _pass(tx: int, ty: int) -> "torch.Tensor":
        decode_fn = _jax_tile_decode_fn(decode_jit, params, tile_h=ty, tile_w=tx)
        return ldm_utils.tiled_scale(z, decode_fn, tx, ty, overlap, upscale_amount=8, output_device=z.device)

    out = (
        _pass(tile_x // 2, tile_y * 2) +
        _pass(tile_x * 2, tile_y // 2) +
        _pass(tile_x, tile_y)
    ) / 3.0
    return out


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

        # Circuit breaker: a JAX VAE decode failure (of any kind) has been
        # observed to leave GPU memory in a bad state rather than cleanly
        # release it — a RESOURCE_EXHAUSTED error from decode_jit() has
        # shown up followed by torch itself reporting almost no free VRAM
        # immediately after, even for a tiny subsequent allocation. Retrying
        # JAX on the next call (or the next tile, or the next generation)
        # only compounds this. So: on the first failure, disable JAX decode
        # for the rest of this model's lifetime and use the proven torch
        # path (including its own battle-tested OOM -> tiled fallback) from
        # then on, rather than repeatedly re-triggering a path that gets
        # the GPU into a worse state each time it fails.
        _state = {"disabled": False}

        def _jax_decode(z: torch.Tensor, **kwargs) -> torch.Tensor:
            if kwargs or _state["disabled"]:
                return _orig_decode(z, **kwargs)
            try:
                if should_tile_decode(z.device, tuple(z.shape), elem_bytes=z.element_size()):
                    log.info(
                        "[JAX VAE] Latent %s estimated to exceed available VRAM - "
                        "decoding tiled via JAX directly (skipping a doomed full-res attempt)",
                        list(z.shape),
                    )
                    return jax_decode_tiled(decode_jit, params, z).to(z.device)

                z_np = z.detach().float().cpu().numpy()
                z_jax = jnp.asarray(z_np, dtype=jnp.bfloat16)
                out = decode_jit(params, z_jax)
                out_np = np.asarray(jnp.asarray(out, dtype=jnp.float32))
                return torch.from_numpy(out_np.copy()).float().to(z.device)
            except Exception as e:
                log.warning(
                    "[JAX VAE] Decoder error on latent %s: %s - falling back to torch for "
                    "this call and disabling JAX VAE decode for the rest of this model's "
                    "session (a JAX decode failure has been observed to leave GPU memory "
                    "in a bad state; retrying would likely compound it).",
                    list(z.shape), e, exc_info=True,
                )
                _state["disabled"] = True
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
