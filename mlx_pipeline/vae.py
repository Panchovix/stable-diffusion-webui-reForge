"""
mlx_pipeline/vae.py — MLX-native KL-F8 VAE decoder and encoder for SDXL.

Architecture (SDXL KL-F8)
--------------------------
  ddconfig = {
    'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
    'z_channels': 4, 'out_ch': 3, 'attn_resolutions': [],
    'double_z': True,
  }

Decoder pipeline:
  post_quant_conv  Conv(4→4, 1×1)
  conv_in          Conv(4→512, 3×3)
  mid              ResNet(512) → Attn(512) → ResNet(512)
  up[3]            3×ResNet(512→512) + Upsample(512)
  up[2]            3×ResNet(512→512) + Upsample(512)
  up[1]            ResNet(512→256) + 2×ResNet(256) + Upsample(256)
  up[0]            ResNet(256→128) + 2×ResNet(128)  [no upsample]
  norm_out         GroupNorm(32, 128) + SiLU
  conv_out         Conv(128→3, 3×3)

State-dict key mapping
----------------------
All weights are loaded from ``first_stage_model.state_dict()``, which uses
``decoder.*`` / ``encoder.*`` / ``quant_conv.*`` / ``post_quant_conv.*`` prefixes.
Conv2d weights are transposed OIHW→OHWI for MLX's NHWC layout.

Integration
-----------
Wraps ``first_stage_model.decode()`` and ``first_stage_model.encode()`` to
use the MLX implementation when active.  Falls back to torch for unusual
decoder kwargs (3D convolution, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    import mlx.core as mx

log = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _t2m(t: torch.Tensor) -> "mx.array":
    """Torch float32 → MLX float32 array (bfloat16 cast at use site)."""
    import mlx.core as mx
    return mx.array(t.detach().float().cpu().numpy())


def _conv_w(t: torch.Tensor) -> "mx.array":
    """Conv2d weight OIHW → OHWI for MLX."""
    import mlx.core as mx
    arr = t.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr.transpose(0, 2, 3, 1)   # [O, I, H, W] → [O, H, W, I]
    return mx.array(arr).astype(mx.bfloat16)


def _silu(x: "mx.array") -> "mx.array":
    import mlx.nn as nn
    return nn.silu(x)


def _group_norm(x: "mx.array", w: "mx.array", b: "mx.array",
                num_groups: int = 32, eps: float = 1e-6) -> "mx.array":
    """
    Group normalization on NHWC tensor [N, H, W, C].

    Splits C into num_groups groups along the channel axis (last axis),
    normalises within each group, then applies element-wise scale (w) and
    bias (b).
    """
    import mlx.core as mx
    N, H, W, C = x.shape
    G = num_groups
    x32 = x.astype(mx.float32).reshape(N, H, W, G, C // G)
    mean = mx.mean(x32, axis=(1, 2, 4), keepdims=True)                  # [N,1,1,G,1]
    var  = mx.mean((x32 - mean) ** 2, axis=(1, 2, 4), keepdims=True)    # [N,1,1,G,1]
    x32  = (x32 - mean) / mx.sqrt(var + eps)
    x32  = x32.reshape(N, H, W, C)
    return (x32 * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


# ── MLX VAE modules ─────────────────────────────────────────────────────────────

class _Conv2d:
    """
    Thin wrapper around Conv2d weights.

    Padding is inferred automatically from the kernel size:
      1×1 → padding=0
      3×3 → padding=1  (same-padding, used by all VAE 3×3 convolutions)

    The Downsample path (stride-2 with asymmetric pre-pad) bypasses this
    class and calls mx.conv2d directly.
    """
    def __init__(self, w: "mx.array", b: "mx.array"):
        import mlx.core as mx
        self.w = w.astype(mx.bfloat16)
        self.b = b.astype(mx.bfloat16)
        # weight shape: [O, kH, kW, I] (OHWI for MLX)
        kH = self.w.shape[1]
        self._pad = (kH - 1) // 2    # 0 for 1×1, 1 for 3×3

    def __call__(self, x: "mx.array") -> "mx.array":
        import mlx.core as mx
        return mx.conv2d(x, self.w, padding=self._pad) + self.b


class _ResnetBlock:
    """
    ResNet block without timestep embedding.
      norm1 → SiLU → conv1 → norm2 → SiLU → conv2 + skip

    Handles both same-channel (identity shortcut) and different-channel
    (nin_shortcut 1×1 conv) cases.
    """
    def __init__(
        self,
        norm1_w, norm1_b,
        conv1_w, conv1_b,
        norm2_w, norm2_b,
        conv2_w, conv2_b,
        nin_shortcut_w=None, nin_shortcut_b=None,
    ):
        import mlx.core as mx
        self.norm1_w = norm1_w.astype(mx.float32)
        self.norm1_b = norm1_b.astype(mx.float32)
        self.conv1   = _Conv2d(conv1_w, conv1_b)
        self.norm2_w = norm2_w.astype(mx.float32)
        self.norm2_b = norm2_b.astype(mx.float32)
        self.conv2   = _Conv2d(conv2_w, conv2_b)
        if nin_shortcut_w is not None:
            self.skip = _Conv2d(nin_shortcut_w, nin_shortcut_b)
        else:
            self.skip = None

    def __call__(self, x: "mx.array") -> "mx.array":
        h = _group_norm(x, self.norm1_w, self.norm1_b)
        h = _silu(h)
        h = self.conv1(h)
        h = _group_norm(h, self.norm2_w, self.norm2_b)
        h = _silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class _AttnBlock:
    """
    Spatial single-head self-attention (mid block only).
    Operates on NHWC tensors; reshapes spatial dims to sequence for SDPA.
    """
    def __init__(self, norm_w, norm_b, q_w, q_b, k_w, k_b, v_w, v_b, po_w, po_b):
        import mlx.core as mx
        self.norm_w = norm_w.astype(mx.float32)
        self.norm_b = norm_b.astype(mx.float32)
        # 1×1 conv weights [O, 1, 1, I] → equivalent to linear on C axis
        self.q  = _Conv2d(q_w, q_b)
        self.k  = _Conv2d(k_w, k_b)
        self.v  = _Conv2d(v_w, v_b)
        self.po = _Conv2d(po_w, po_b)

    def __call__(self, x: "mx.array") -> "mx.array":
        import mlx.core as mx
        N, H, W, C = x.shape
        h = _group_norm(x, self.norm_w, self.norm_b)
        q = self.q(h).reshape(N, 1, H * W, C)   # [N, 1, HW, C]
        k = self.k(h).reshape(N, 1, H * W, C)
        v = self.v(h).reshape(N, 1, H * W, C)
        scale = C ** -0.5
        out = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale,
        )   # [N, 1, HW, C]
        out = out.astype(mx.bfloat16).reshape(N, H, W, C)
        return x + self.po(out)


class _Upsample:
    """Nearest-neighbour ×2 upsample followed by an optional Conv2d."""
    def __init__(self, conv_w, conv_b):
        import mlx.nn as nn
        self._up  = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _Conv2d(conv_w, conv_b)

    def __call__(self, x: "mx.array") -> "mx.array":
        return self.conv(self._up(x))


class _Downsample:
    """Stride-2 Conv2d downsample (used in encoder)."""
    def __init__(self, conv_w, conv_b):
        import mlx.core as mx
        self.conv_w = conv_w.astype(mx.bfloat16)
        self.conv_b = conv_b.astype(mx.bfloat16)

    def __call__(self, x: "mx.array") -> "mx.array":
        import mlx.core as mx
        # Pad right/bottom by 1 to match PyTorch asymmetric padding
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return mx.conv2d(x, self.conv_w, stride=(2, 2)) + self.conv_b


# ── high-level blocks ───────────────────────────────────────────────────────────

class _MidBlock:
    def __init__(self, block_1, attn_1, block_2):
        self.block_1 = block_1
        self.attn_1  = attn_1
        self.block_2 = block_2

    def __call__(self, x):
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x


class _UpBlock:
    def __init__(self, blocks, upsample=None):
        self.blocks    = blocks
        self.upsample  = upsample

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _DownBlock:
    def __init__(self, blocks, downsample=None):
        self.blocks     = blocks
        self.downsample = downsample

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ── full decoder ────────────────────────────────────────────────────────────────

class MLXVAEDecoder:
    """
    MLX implementation of the KL-F8 VAE decoder.

    Input:  z [N, 4, H, W]  torch float32  (post_quant_conv applied here)
    Output: x [N, 3, H, W]  torch float32
    """

    def __init__(
        self,
        post_quant_conv: _Conv2d,
        conv_in:         _Conv2d,
        mid:             _MidBlock,
        up:              list,           # up[0]..up[3], forward order is reversed
        norm_out_w:      "mx.array",
        norm_out_b:      "mx.array",
        conv_out:        _Conv2d,
    ):
        import mlx.core as mx
        self.post_quant_conv = post_quant_conv
        self.conv_in         = conv_in
        self.mid             = mid
        self.up              = up
        self.norm_out_w      = norm_out_w.astype(mx.float32)
        self.norm_out_b      = norm_out_b.astype(mx.float32)
        self.conv_out        = conv_out

    def decode(self, z_t: torch.Tensor) -> torch.Tensor:
        """z_t: [N, 4, H, W] torch float32 → image [N, 3, H, W] torch float32."""
        import mlx.core as mx
        # NCHW → NHWC
        z_np = z_t.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
        x    = mx.array(z_np).astype(mx.bfloat16)

        x = self.post_quant_conv(x)
        x = self.conv_in(x)
        x = self.mid(x)

        # Iterate up blocks in decode order: up[3] → up[2] → up[1] → up[0]
        for blk in reversed(self.up):
            x = blk(x)

        x = _group_norm(x, self.norm_out_w, self.norm_out_b)
        x = _silu(x)
        x = self.conv_out(x)

        mx.eval(x)
        # NHWC → NCHW, float32
        out_np = np.array(x.astype(mx.float32)).transpose(0, 3, 1, 2)
        return torch.from_numpy(out_np.copy()).float()


class MLXVAEEncoder:
    """
    MLX implementation of the KL-F8 VAE encoder.

    Input:  x [N, 3, H, W]  torch float32
    Output: z [N, 8, H//8, W//8]  torch float32  (pre-quant_conv, double_z=True)
    """

    def __init__(
        self,
        conv_in:      _Conv2d,
        down:         list,           # down[0]..down[3] forward order
        mid:          _MidBlock,
        norm_out_w:   "mx.array",
        norm_out_b:   "mx.array",
        conv_out:     _Conv2d,
        quant_conv:   _Conv2d,
    ):
        import mlx.core as mx
        self.conv_in   = conv_in
        self.down      = down
        self.mid       = mid
        self.norm_out_w = norm_out_w.astype(mx.float32)
        self.norm_out_b = norm_out_b.astype(mx.float32)
        self.conv_out  = conv_out
        self.quant_conv = quant_conv

    def encode(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: [N, 3, H, W] → moments [N, 8, H//8, W//8] float32."""
        import mlx.core as mx
        x_np = x_t.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
        x    = mx.array(x_np).astype(mx.bfloat16)

        x = self.conv_in(x)
        for blk in self.down:
            x = blk(x)
        x = self.mid(x)
        x = _group_norm(x, self.norm_out_w, self.norm_out_b)
        x = _silu(x)
        x = self.conv_out(x)
        x = self.quant_conv(x)

        mx.eval(x)
        out_np = np.array(x.astype(mx.float32)).transpose(0, 3, 1, 2)
        return torch.from_numpy(out_np.copy()).float()


# ── weight loading ──────────────────────────────────────────────────────────────

def _load_resnet(sd: dict, prefix: str) -> _ResnetBlock:
    """Load a ResnetBlock from state dict with given prefix (without trailing dot)."""
    p = prefix + "."
    nin_w = sd.get(p + "nin_shortcut.weight")
    nin_b = sd.get(p + "nin_shortcut.bias")
    return _ResnetBlock(
        norm1_w = _t2m(sd[p + "norm1.weight"]),
        norm1_b = _t2m(sd[p + "norm1.bias"]),
        conv1_w = _conv_w(sd[p + "conv1.weight"]),
        conv1_b = _t2m(sd[p + "conv1.bias"]),
        norm2_w = _t2m(sd[p + "norm2.weight"]),
        norm2_b = _t2m(sd[p + "norm2.bias"]),
        conv2_w = _conv_w(sd[p + "conv2.weight"]),
        conv2_b = _t2m(sd[p + "conv2.bias"]),
        nin_shortcut_w = _conv_w(nin_w) if nin_w is not None else None,
        nin_shortcut_b = _t2m(nin_b)   if nin_b is not None else None,
    )


def _load_attn(sd: dict, prefix: str) -> _AttnBlock:
    p = prefix + "."
    return _AttnBlock(
        norm_w = _t2m(sd[p + "norm.weight"]),
        norm_b = _t2m(sd[p + "norm.bias"]),
        q_w    = _conv_w(sd[p + "q.weight"]),
        q_b    = _t2m(sd[p + "q.bias"]),
        k_w    = _conv_w(sd[p + "k.weight"]),
        k_b    = _t2m(sd[p + "k.bias"]),
        v_w    = _conv_w(sd[p + "v.weight"]),
        v_b    = _t2m(sd[p + "v.bias"]),
        po_w   = _conv_w(sd[p + "proj_out.weight"]),
        po_b   = _t2m(sd[p + "proj_out.bias"]),
    )


def _load_conv(sd: dict, prefix: str) -> _Conv2d:
    return _Conv2d(_conv_w(sd[prefix + ".weight"]), _t2m(sd[prefix + ".bias"]))


def _load_upsample(sd: dict, prefix: str) -> _Upsample:
    return _Upsample(
        _conv_w(sd[prefix + ".conv.weight"]),
        _t2m(   sd[prefix + ".conv.bias"]),
    )


def _load_downsample(sd: dict, prefix: str) -> _Downsample:
    return _Downsample(
        _conv_w(sd[prefix + ".conv.weight"]),
        _t2m(   sd[prefix + ".conv.bias"]),
    )


def _load_mid(sd: dict, prefix: str) -> _MidBlock:
    return _MidBlock(
        block_1 = _load_resnet(sd, f"{prefix}.mid.block_1"),
        attn_1  = _load_attn( sd, f"{prefix}.mid.attn_1"),
        block_2 = _load_resnet(sd, f"{prefix}.mid.block_2"),
    )


def build_vae_decoder(first_stage_model) -> MLXVAEDecoder:
    """
    Build an MLXVAEDecoder from the ldm_patched first_stage_model.

    Parameters
    ----------
    first_stage_model : AutoencoderKL (ldm_patched)

    Returns
    -------
    MLXVAEDecoder ready for use.
    """
    sd = {k: v for k, v in first_stage_model.state_dict().items()}

    # Detect architecture from weight shapes
    conv_in_w    = sd["decoder.conv_in.weight"]          # [block_in, z_ch, 3, 3]
    block_in     = conv_in_w.shape[0]                     # 512 for SDXL
    z_channels   = conv_in_w.shape[1]                     # 4

    # Determine number of up-blocks and resnet-blocks from keys
    max_up_level = max(
        int(k.split(".")[2]) for k in sd if k.startswith("decoder.up.")
    )                                                      # 3 for SDXL
    num_up       = max_up_level + 1                        # 4

    num_res = max(
        int(k.split(".")[4]) for k in sd if k.startswith("decoder.up.") and ".block." in k
    ) + 1                                                  # 3 for SDXL (num_res_blocks+1)

    # Build up blocks (stored in ldm order: up[0]=lowest resolution)
    up_blocks = []
    for i in range(num_up):
        blocks = [_load_resnet(sd, f"decoder.up.{i}.block.{j}") for j in range(num_res)]
        upsample = None
        if f"decoder.up.{i}.upsample.conv.weight" in sd:
            upsample = _load_upsample(sd, f"decoder.up.{i}.upsample")
        up_blocks.append(_UpBlock(blocks, upsample))

    decoder = MLXVAEDecoder(
        post_quant_conv = _load_conv(sd, "post_quant_conv"),
        conv_in         = _load_conv(sd, "decoder.conv_in"),
        mid             = _load_mid(sd, "decoder"),
        up              = up_blocks,
        norm_out_w      = _t2m(sd["decoder.norm_out.weight"]),
        norm_out_b      = _t2m(sd["decoder.norm_out.bias"]),
        conv_out        = _load_conv(sd, "decoder.conv_out"),
    )

    log.debug(
        "[MLX VAE] Decoder built: z_ch=%d block_in=%d up_levels=%d res_blocks=%d",
        z_channels, block_in, num_up, num_res,
    )
    return decoder


def build_vae_encoder(first_stage_model) -> MLXVAEEncoder:
    """Build an MLXVAEEncoder from the ldm_patched first_stage_model."""
    sd = {k: v for k, v in first_stage_model.state_dict().items()}

    max_down = max(
        int(k.split(".")[2]) for k in sd if k.startswith("encoder.down.")
    )
    num_down = max_down + 1

    num_res_enc = max(
        int(k.split(".")[4]) for k in sd if k.startswith("encoder.down.") and ".block." in k
    ) + 1

    down_blocks = []
    for i in range(num_down):
        blocks = [_load_resnet(sd, f"encoder.down.{i}.block.{j}") for j in range(num_res_enc)]
        downsample = None
        if f"encoder.down.{i}.downsample.conv.weight" in sd:
            downsample = _load_downsample(sd, f"encoder.down.{i}.downsample")
        down_blocks.append(_DownBlock(blocks, downsample))

    encoder = MLXVAEEncoder(
        conv_in     = _load_conv(sd, "encoder.conv_in"),
        down        = down_blocks,
        mid         = _load_mid(sd, "encoder"),
        norm_out_w  = _t2m(sd["encoder.norm_out.weight"]),
        norm_out_b  = _t2m(sd["encoder.norm_out.bias"]),
        conv_out    = _load_conv(sd, "encoder.conv_out"),
        quant_conv  = _load_conv(sd, "quant_conv"),
    )
    return encoder


# ── integration hooks ───────────────────────────────────────────────────────────

def install_vae_hooks(sd_model) -> bool:
    """
    Monkey-patch ``first_stage_model.decode()`` to use the MLX decoder.

    Only patches the standard KL-F8 path (no 3D conv / video VAE).
    Falls back to torch on unsupported decoder kwargs.

    The encoder is intentionally left as torch: img2img encode happens once
    per generation, and reparameterisation requires keeping the regularisation
    code path intact.
    """
    try:
        fst = sd_model.first_stage_model
        # Sanity check: standard 2-D decoder keys only
        sd_state = fst.state_dict()
        if "decoder.conv_in.weight" not in sd_state:
            log.debug("[MLX VAE] Non-standard decoder; skipping hook")
            return False
        if sd_state["decoder.conv_in.weight"].ndim != 4:
            log.debug("[MLX VAE] 3-D (video) VAE detected; skipping hook")
            return False

        mlx_decoder = build_vae_decoder(fst)
        _orig_decode = fst.decode

        def _mlx_decode(z: torch.Tensor, **kwargs) -> torch.Tensor:
            # Fall back if extra decoder kwargs present (rare: image conditioning)
            if kwargs:
                return _orig_decode(z, **kwargs)
            try:
                out = mlx_decoder.decode(z)
                # MLX always outputs CPU tensors; restore to the same device
                # as the input latent so the decoded image stays on MPS/CUDA.
                return out.to(z.device)
            except Exception as e:
                log.warning("[MLX VAE] Decoder error: %s — falling back to torch", e,
                            exc_info=True)
                return _orig_decode(z)

        fst.decode = _mlx_decode

        # Keep reference alive (prevents GC)
        sd_model._mlx_vae_decoder = mlx_decoder

        log.info("[MLX VAE] Hooked first_stage_model.decode — Metal bfloat16 VAE decoding active")
        return True

    except Exception as e:
        log.warning("[MLX VAE] Hook installation failed: %s", e, exc_info=True)
        return False
