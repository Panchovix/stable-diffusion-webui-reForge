"""
mlx_pipeline/unet.py — SDXL UNet2DConditionModel in Apple MLX BFloat16.

Architecture mirrors HuggingFace diffusers UNet2DConditionModel (SDXL config)
so that weights loaded via unet_to_diffusers() key mapping drop in without any
further transposition of Linear weights.  Conv2d weights ARE transposed from
PyTorch's OIHW to MLX's OHWI format in convert.py.

Forward-pass tensor format: NHWC throughout (MLX-native).
  - Input:  NCHW torch tensor  →  permuted to NHWC at entry
  - Output: NHWC MLX array    →  permuted to NCHW at exit (in pipeline.py)

Weight naming convention: exact HF diffusers paths, e.g.
  down_blocks.1.resnets.0.conv1.weight
  down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight
  up_blocks.0.resnets.2.norm1.weight
  mid_block.attentions.0.transformer_blocks.5.ff.net.2.weight

SDXL fixed config (stabilityai/stable-diffusion-xl-base-1.0):
  block_out_channels      : [320, 640, 1280]
  layers_per_block        : 2
  transformer_layers      : [1, 2, 10]  (per CrossAttn level / mid)
  attention_head_dim      : [5, 10, 20]  (ignored — dim_per_head=64 always)
  cross_attention_dim     : 2048
  addition_embed_type     : "text_time"
  addition_time_embed_dim : 256
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ── activation shorthands (both verified present in MLX 0.31+) ────────────────

def _silu(x: mx.array) -> mx.array:
    return nn.silu(x)


def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sinusoidal_embedding(
    timesteps: mx.array,
    dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
) -> mx.array:
    """Sinusoidal (Fourier) timestep embedding — no learnable parameters.

    Matches HuggingFace ``Timesteps`` / ``get_timestep_embedding``.
    """
    half = dim // 2
    freqs = mx.arange(0, half, dtype=mx.float32)
    freqs = mx.exp(-math.log(10000) * freqs / (half - downscale_freq_shift))
    # timesteps: [B] → [B, 1];  freqs: [half] → [1, half]
    x = timesteps[:, None].astype(mx.float32) * freqs[None, :]  # [B, half]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(x), mx.sin(x)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)
    return emb  # [B, dim]


def _upsample2x(x: mx.array) -> mx.array:
    """Nearest-neighbour 2× upsampling for NHWC tensors."""
    x = mx.repeat(x, 2, axis=1)
    x = mx.repeat(x, 2, axis=2)
    return x


def _sdp_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
) -> mx.array:
    """Scaled dot-product attention — uses MLX fast path when available.

    All tensors in [B, heads, seq, dim_per_head] format.
    """
    try:
        # mlx >= 0.22: fused Metal kernel, much faster than naive matmul path
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    except AttributeError:
        pass
    # Fallback: explicit softmax (fp32 for numerical stability, cast back)
    attn = (q @ mx.swapaxes(k, -1, -2)) * scale      # [B, h, Sq, Sk]
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    return attn @ v                                    # [B, h, Sq, d]


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention
# ═══════════════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Single multi-head attention block (self or cross).

    Parameter names match HF diffusers Attention exactly so load_weights works:
      to_q.weight / to_q.bias
      to_k.weight / to_k.bias
      to_v.weight / to_v.bias
      to_out.0.weight / to_out.0.bias
    """

    def __init__(self, query_dim: int, num_heads: int, cross_attention_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = query_dim // num_heads
        inner_dim = query_dim  # HF Attention inner_dim == query_dim for SDXL

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim)
        # to_out is a list (HF uses nn.ModuleList) — to_out.0 is the projection
        self.to_out = [nn.Linear(inner_dim, query_dim)]

    def __call__(
        self,
        hidden_states: mx.array,           # [B, Sq, D]
        encoder_hidden_states: Optional[mx.array] = None,  # [B, Sk, ctx]
    ) -> mx.array:
        B, Sq, _ = hidden_states.shape
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        Sk = context.shape[1]

        q = self.to_q(hidden_states)  # [B, Sq, D]
        k = self.to_k(context)        # [B, Sk, D]
        v = self.to_v(context)        # [B, Sk, D]

        H, dh = self.num_heads, self.dim_per_head
        # Reshape → [B, H, S, dh]
        q = mx.transpose(q.reshape(B, Sq, H, dh), (0, 2, 1, 3))
        k = mx.transpose(k.reshape(B, Sk, H, dh), (0, 2, 1, 3))
        v = mx.transpose(v.reshape(B, Sk, H, dh), (0, 2, 1, 3))

        out = _sdp_attention(q, k, v, scale=dh ** -0.5)  # [B, H, Sq, dh]

        # Merge heads → [B, Sq, D]
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, Sq, H * dh)
        return self.to_out[0](out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feed-Forward (GEGLU)
# ═══════════════════════════════════════════════════════════════════════════════

class _GEGLU(nn.Module):
    """Gated-GELU: proj outputs 2×dim, splits into value and gate halves."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = self.proj(x).split(2, axis=-1)
        return x * _gelu(gate)


class _Identity(nn.Module):
    """No-op placeholder for Dropout (always disabled at inference)."""
    def __call__(self, x: mx.array) -> mx.array:
        return x


class FeedForward(nn.Module):
    """Two-layer feed-forward with GEGLU, matching HF FeedForward.

    Weight paths:
      ff.net.0.proj.weight / .bias   ← GEGLU projection
      ff.net.2.weight / .bias        ← output Linear
    (ff.net.1 is Dropout — no weights, fine with strict=False)
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        # net is a list so indices match HF's net.0, net.1, net.2
        self.net = [_GEGLU(dim, inner), _Identity(), nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[1](self.net[0](x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  BasicTransformerBlock
# ═══════════════════════════════════════════════════════════════════════════════

class BasicTransformerBlock(nn.Module):
    """Self-attention + cross-attention + feed-forward, matching HF naming.

    Weight paths (example for block index 0):
      transformer_blocks.0.norm1.weight
      transformer_blocks.0.attn1.to_q.weight   ← self-attn
      transformer_blocks.0.norm2.weight
      transformer_blocks.0.attn2.to_k.weight   ← cross-attn
      transformer_blocks.0.norm3.weight
      transformer_blocks.0.ff.net.0.proj.weight
      transformer_blocks.0.ff.net.2.weight
    """

    def __init__(self, dim: int, num_heads: int, cross_attention_dim: int = 2048):
        super().__init__()
        # LayerNorm defaults: affine=True, bias=True in MLX 0.31+
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, num_heads, dim)              # self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, num_heads, cross_attention_dim)  # cross-attention
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(
        self,
        hidden_states: mx.array,                         # [B, seq, dim]
        encoder_hidden_states: Optional[mx.array] = None,  # [B, ctx, 2048]
    ) -> mx.array:
        # Self-attention (pre-norm)
        h = self.norm1(hidden_states)
        h = self.attn1(h)
        hidden_states = hidden_states + h

        # Cross-attention (pre-norm)
        h = self.norm2(hidden_states)
        h = self.attn2(h, encoder_hidden_states)
        hidden_states = hidden_states + h

        # Feed-forward (pre-norm)
        h = self.norm3(hidden_states)
        h = self.ff(h)
        hidden_states = hidden_states + h

        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer2DModel  (spatial transformer)
# ═══════════════════════════════════════════════════════════════════════════════

class Transformer2DModel(nn.Module):
    """Spatial transformer wrapping N BasicTransformerBlocks.

    Flattens HW into a sequence, runs transformer blocks, reshapes back.
    Uses linear proj_in/proj_out (SDXL uses use_linear_projection=True).

    Weight paths (example):
      attentions.0.norm.weight
      attentions.0.proj_in.weight
      attentions.0.transformer_blocks.0.norm1.weight
      ...
      attentions.0.proj_out.weight
    """

    def __init__(self, channels: int, num_heads: int, num_layers: int):
        super().__init__()
        # GroupNorm: pytorch_compatible=True matches PyTorch's grouping order (normalise over H*W*C/G)
        self.norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks: List[BasicTransformerBlock] = [
            BasicTransformerBlock(channels, num_heads)
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(
        self,
        hidden_states: mx.array,                         # [B, H, W, C]  NHWC
        encoder_hidden_states: Optional[mx.array] = None,  # [B, ctx, 2048]
    ) -> mx.array:
        B, H, W, C = hidden_states.shape
        residual = hidden_states

        # GroupNorm then flatten spatial → sequence
        h = self.norm(hidden_states)                     # [B, H, W, C]
        h = h.reshape(B, H * W, C)                      # [B, HW, C]
        h = self.proj_in(h)

        for block in self.transformer_blocks:
            h = block(h, encoder_hidden_states)

        h = self.proj_out(h)
        h = h.reshape(B, H, W, C)
        return h + residual


# ═══════════════════════════════════════════════════════════════════════════════
#  ResnetBlock2D
# ═══════════════════════════════════════════════════════════════════════════════

class ResnetBlock2D(nn.Module):
    """Two-conv residual block with time embedding injection.

    Weight paths:
      resnets.N.norm1.weight
      resnets.N.conv1.weight
      resnets.N.time_emb_proj.weight
      resnets.N.norm2.weight
      resnets.N.conv2.weight
      resnets.N.conv_shortcut.weight   ← only when in_channels != out_channels
    """

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int = 1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        # hidden_states: [B, H, W, C_in]   temb: [B, temb_channels]
        residual = hidden_states

        h = nn.silu(self.norm1(hidden_states))
        h = self.conv1(h)

        # Inject time embedding: SiLU → project → broadcast over HW
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]

        h = nn.silu(self.norm2(h))
        h = self.conv2(h)

        if hasattr(self, "conv_shortcut"):
            residual = self.conv_shortcut(residual)

        return h + residual


# ═══════════════════════════════════════════════════════════════════════════════
#  Downsample / Upsample
# ═══════════════════════════════════════════════════════════════════════════════

class Downsample2D(nn.Module):
    """Stride-2 convolution downsampler.

    Weight path:  downsamplers.0.conv.weight
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbour 2× upsample followed by conv (no learned transpose-conv).

    Weight path:  upsamplers.0.conv.weight
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(_upsample2x(x))


# ═══════════════════════════════════════════════════════════════════════════════
#  Down Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class DownBlock2D(nn.Module):
    """Pure-ResNet down block (no attention) — SDXL level 0 (ch=320)."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(ResnetBlock2D(
                in_channels if i == 0 else out_channels,
                out_channels,
            ))
        self.resnets: List[ResnetBlock2D] = resnets
        self.downsamplers = [Downsample2D(out_channels)]

    def __call__(
        self, x: mx.array, temb: mx.array
    ) -> Tuple[mx.array, List[mx.array]]:
        skips = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            skips.append(x)
        x = self.downsamplers[0](x)
        skips.append(x)
        return x, skips


class CrossAttnDownBlock2D(nn.Module):
    """ResNet + Transformer down block — SDXL levels 1 & 2.

    Parameters
    ----------
    in_channels   : input channel count
    out_channels  : output channel count
    num_layers    : number of (ResNet, Transformer) pairs (= 2 for SDXL)
    num_attn_layers : transformer layers inside each Transformer2DModel
    num_heads     : attention heads in each transformer block
    add_downsample: whether to append a Downsample2D (False for the last level)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_attn_layers: int = 2,
        num_heads: int = 10,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(ch_in, out_channels))
            attentions.append(Transformer2DModel(out_channels, num_heads, num_attn_layers))
        self.resnets: List[ResnetBlock2D] = resnets
        self.attentions: List[Transformer2DModel] = attentions
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(
        self,
        x: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array,
    ) -> Tuple[mx.array, List[mx.array]]:
        skips = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            x = attn(x, encoder_hidden_states)
            skips.append(x)
        if self.downsamplers:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


# ═══════════════════════════════════════════════════════════════════════════════
#  Mid Block
# ═══════════════════════════════════════════════════════════════════════════════

class UNetMidBlock2DCrossAttn(nn.Module):
    """SDXL mid block: ResNet → Transformer(10 layers) → ResNet.

    Weight paths:
      mid_block.resnets.0.*
      mid_block.attentions.0.*
      mid_block.resnets.1.*
    """

    def __init__(self, channels: int = 1280, num_heads: int = 20, num_attn_layers: int = 10):
        super().__init__()
        self.resnets: List[ResnetBlock2D] = [
            ResnetBlock2D(channels, channels),
            ResnetBlock2D(channels, channels),
        ]
        self.attentions: List[Transformer2DModel] = [
            Transformer2DModel(channels, num_heads, num_attn_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, encoder_hidden_states)
        x = self.resnets[1](x, temb)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Up Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class CrossAttnUpBlock2D(nn.Module):
    """ResNet + Transformer up block with skip-connection concatenation.

    Each ResNet takes ``(prev_output, skip_from_down)`` concatenated on the
    channel axis, so ``in_channels`` must be the sum of both.

    Parameters
    ----------
    res_skip_channels  : list of skip channel sizes for each of the 3 resnets
    out_channels       : output channels (= channels for this level)
    prev_output_channels : channels from the previous up/mid block
    """

    def __init__(
        self,
        out_channels: int,
        prev_output_channels: int,
        res_skip_channels: List[int],   # 3 entries for SDXL (layers_per_block+1)
        num_attn_layers: int = 10,
        num_heads: int = 20,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        prev_ch = prev_output_channels
        for skip_ch in res_skip_channels:
            resnets.append(ResnetBlock2D(prev_ch + skip_ch, out_channels))
            attentions.append(Transformer2DModel(out_channels, num_heads, num_attn_layers))
            prev_ch = out_channels
        self.resnets: List[ResnetBlock2D] = resnets
        self.attentions: List[Transformer2DModel] = attentions
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(
        self,
        x: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array,
        res_hidden_states_tuple: Tuple[mx.array, ...],
    ) -> mx.array:
        for resnet, attn, skip in zip(
            self.resnets, self.attentions, res_hidden_states_tuple
        ):
            x = mx.concatenate([x, skip], axis=-1)  # NHWC → concat on C
            x = resnet(x, temb)
            x = attn(x, encoder_hidden_states)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class UpBlock2D(nn.Module):
    """Pure-ResNet up block — SDXL level 2 (ch=320), no attention."""

    def __init__(
        self,
        out_channels: int,
        prev_output_channels: int,
        res_skip_channels: List[int],
        add_upsample: bool = False,
    ):
        super().__init__()
        resnets = []
        prev_ch = prev_output_channels
        for skip_ch in res_skip_channels:
            resnets.append(ResnetBlock2D(prev_ch + skip_ch, out_channels))
            prev_ch = out_channels
        self.resnets: List[ResnetBlock2D] = resnets
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(
        self,
        x: mx.array,
        temb: mx.array,
        res_hidden_states_tuple: Tuple[mx.array, ...],
    ) -> mx.array:
        for resnet, skip in zip(self.resnets, res_hidden_states_tuple):
            x = mx.concatenate([x, skip], axis=-1)
            x = resnet(x, temb)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Time / Add Embedding modules
# ═══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """Sinusoidal → 2-layer MLP time embedding.

    Weight paths:
      time_embedding.linear_1.weight
      time_embedding.linear_2.weight
    """

    def __init__(self, in_dim: int = 320, out_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class AddEmbedding(nn.Module):
    """SDXL additional conditioning (text_embeds + Fourier(time_ids)) → MLP.

    Weight paths:
      add_embedding.linear_1.weight
      add_embedding.linear_2.weight
    (add_time_proj has no learnable weights — it is a sinusoidal Timesteps)
    """

    def __init__(
        self,
        in_dim: int = 2816,   # 1536 (6×256 Fourier) + 1280 (pooled CLIP-G)
        out_dim: int = 1280,
        time_embed_dim: int = 256,  # addition_time_embed_dim in SDXL config
    ):
        super().__init__()
        self._time_embed_dim = time_embed_dim
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, text_embeds: mx.array, time_ids: mx.array) -> mx.array:
        # Fourier-embed each of the 6 time_id scalars (256 dims each)
        B, num_ids = time_ids.shape
        time_ids_flat = time_ids.reshape(-1)  # [B*6]
        fourier = _sinusoidal_embedding(
            time_ids_flat,
            self._time_embed_dim,
            flip_sin_to_cos=True,
        )                                     # [B*6, 256]
        fourier = fourier.reshape(B, -1)      # [B, 6*256=1536]

        # Concat with pooled text embeds → [B, 2816]
        cond = mx.concatenate([text_embeds, fourier], axis=-1)
        return self.linear_2(nn.silu(self.linear_1(cond)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level SDXLUNet
# ═══════════════════════════════════════════════════════════════════════════════

class SDXLUNet(nn.Module):
    """SDXL UNet2DConditionModel in Apple MLX BFloat16.

    Fixed to the standard SDXL-base-1.0 architecture; weights are loaded
    post-construction via ``load_weights(params, strict=False)``.

    Call signature mirrors HF UNet2DConditionModel.__call__:
        out = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs)

    Tensor format
    -------------
    * sample                 : [B, 4, H, W]  PyTorch-style NCHW  (permuted internally)
    * timestep               : [B]           integer or float sigma-based timesteps
    * encoder_hidden_states  : [B, S, 2048]  text token embeddings
    * added_cond_kwargs       : dict with keys "text_embeds" [B,1280], "time_ids" [B,6]

    All internal computation is NHWC (MLX-native).
    Output: [B, 4, H, W] in NCHW, matching UNet input convention.
    """

    # ── SDXL fixed config ─────────────────────────────────────────────────────
    _BLOCK_OUT_CHANNELS = [320, 640, 1280]
    _LAYERS_PER_BLOCK = 2
    _TRANSFORMER_LAYERS = [1, 2, 10]   # [level-1 CrossAttn, level-2 CrossAttn, mid]
    _NUM_HEADS = [10, 20, 20]           # [level-1, level-2, mid]  (64 dim/head each)
    _TIME_DIM = 320
    _EMBD_DIM = 1280                    # time MLP output
    _CROSS_ATT_DIM = 2048
    _IN_CHANNELS = 4
    _OUT_CHANNELS = 4
    _NORM_GROUPS = 32

    def __init__(self):
        super().__init__()
        ch = self._BLOCK_OUT_CHANNELS  # [320, 640, 1280]

        # ── Input projection ──
        self.conv_in = nn.Conv2d(self._IN_CHANNELS, ch[0], 3, padding=1)

        # ── Time embedding ──
        self.time_embedding = TimestepEmbedding(self._TIME_DIM, self._EMBD_DIM)

        # ── SDXL add embedding ──
        self.add_embedding = AddEmbedding()

        # ── Down blocks ──
        self.down_blocks = [
            # Level 0: DownBlock2D (320, no attention)
            DownBlock2D(ch[0], ch[0], self._LAYERS_PER_BLOCK),
            # Level 1: CrossAttnDownBlock2D (640, 1 transformer layer, add_downsample)
            CrossAttnDownBlock2D(
                ch[0], ch[1],
                num_layers=self._LAYERS_PER_BLOCK,
                num_attn_layers=self._TRANSFORMER_LAYERS[0],
                num_heads=self._NUM_HEADS[0],
                add_downsample=True,
            ),
            # Level 2: CrossAttnDownBlock2D (1280, 2 transformer layers, NO downsample)
            CrossAttnDownBlock2D(
                ch[1], ch[2],
                num_layers=self._LAYERS_PER_BLOCK,
                num_attn_layers=self._TRANSFORMER_LAYERS[1],
                num_heads=self._NUM_HEADS[1],
                add_downsample=False,
            ),
        ]

        # ── Mid block ──
        self.mid_block = UNetMidBlock2DCrossAttn(
            ch[2],
            num_heads=self._NUM_HEADS[2],
            num_attn_layers=self._TRANSFORMER_LAYERS[2],
        )

        # ── Up blocks ──
        # Skip channel counts (popped from end of down_block_res_samples):
        #   UpBlock0 (1280): skips = [1280, 1280, 640]
        #   UpBlock1 (640):  skips = [640,  640,  320]
        #   UpBlock2 (320):  skips = [320,  320,  320]
        self.up_blocks = [
            CrossAttnUpBlock2D(
                out_channels=ch[2],
                prev_output_channels=ch[2],
                res_skip_channels=[ch[2], ch[2], ch[1]],
                num_attn_layers=self._TRANSFORMER_LAYERS[2],
                num_heads=self._NUM_HEADS[2],
                add_upsample=True,
            ),
            CrossAttnUpBlock2D(
                out_channels=ch[1],
                prev_output_channels=ch[2],
                res_skip_channels=[ch[1], ch[1], ch[0]],
                num_attn_layers=self._TRANSFORMER_LAYERS[1],
                num_heads=self._NUM_HEADS[0],
                add_upsample=True,
            ),
            UpBlock2D(
                out_channels=ch[0],
                prev_output_channels=ch[1],
                res_skip_channels=[ch[0], ch[0], ch[0]],
                add_upsample=False,
            ),
        ]

        # ── Output projection ──
        self.conv_norm_out = nn.GroupNorm(self._NORM_GROUPS, ch[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(ch[0], self._OUT_CHANNELS, 3, padding=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def __call__(
        self,
        sample: mx.array,                      # [B, 4, H, W]  NCHW (from torch)
        timestep: mx.array,                    # [B]
        encoder_hidden_states: mx.array,       # [B, S, 2048]
        added_cond_kwargs: dict,               # {"text_embeds": ..., "time_ids": ...}
    ) -> mx.array:
        # ── 1. NCHW → NHWC ───────────────────────────────────────────────────
        sample = mx.transpose(sample, (0, 2, 3, 1))  # [B, H, W, 4]

        # ── 2. Time embedding ─────────────────────────────────────────────────
        timestep_emb = _sinusoidal_embedding(
            timestep.astype(mx.float32), self._TIME_DIM, flip_sin_to_cos=True
        ).astype(sample.dtype)                       # [B, 320]
        temb = self.time_embedding(timestep_emb)     # [B, 1280]

        # ── 3. SDXL additional conditioning ──────────────────────────────────
        text_embeds = added_cond_kwargs["text_embeds"].astype(sample.dtype)
        time_ids    = added_cond_kwargs["time_ids"].astype(mx.float32)
        aug_emb = self.add_embedding(text_embeds, time_ids).astype(sample.dtype)
        temb = temb + aug_emb                        # [B, 1280]

        # ── 4. Input conv ─────────────────────────────────────────────────────
        sample = self.conv_in(sample)                # [B, H, W, 320]

        # ── 5. Down path ──────────────────────────────────────────────────────
        # Collect ALL skip connections (including initial conv_in output)
        down_block_res_samples: List[mx.array] = [sample]

        for i, block in enumerate(self.down_blocks):
            if i == 0:  # DownBlock2D
                sample, skips = block(sample, temb)
            else:       # CrossAttnDownBlock2D
                sample, skips = block(sample, temb, encoder_hidden_states)
            down_block_res_samples.extend(skips)

        # ── 6. Mid block ──────────────────────────────────────────────────────
        sample = self.mid_block(sample, temb, encoder_hidden_states)

        # ── 7. Up path ────────────────────────────────────────────────────────
        for i, block in enumerate(self.up_blocks):
            # Each up block consumes layers_per_block+1 = 3 skip tensors
            n = len(block.resnets)
            res_samples = tuple(
                down_block_res_samples.pop() for _ in range(n)
            )
            if isinstance(block, CrossAttnUpBlock2D):
                sample = block(sample, temb, encoder_hidden_states, res_samples)
            else:
                sample = block(sample, temb, res_samples)

        # ── 8. Output projection ──────────────────────────────────────────────
        sample = nn.silu(self.conv_norm_out(sample))  # [B, H, W, 320]
        sample = self.conv_out(sample)                 # [B, H, W, 4]

        # ── 9. NHWC → NCHW ───────────────────────────────────────────────────
        return mx.transpose(sample, (0, 3, 1, 2))      # [B, 4, H, W]
