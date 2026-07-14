"""
jax_pipeline/unet.py — SDXL UNet2DConditionModel as a pure-functional JAX forward pass.

Architecture mirrors HuggingFace diffusers UNet2DConditionModel (SDXL config),
matching mlx_pipeline/unet.py block-for-block, so weights loaded via
jax_pipeline.convert (which reuses the same unet_to_diffusers() key mapping)
drop in directly.

There is no module/parameter-tree class here — JAX has no built-in stateful
layer objects and we deliberately avoid adding a Flax/Equinox dependency for
this. Instead every building block is a pure function taking
``(params, prefix, *inputs) -> output``, where ``params`` is the flat
``{hf_dotted_key: jnp.ndarray}`` dict produced by ``jax_pipeline.convert``.

Forward-pass tensor format: NHWC throughout (matches XLA's preferred conv
layout on GPU/TPU).
  - Input:  NCHW torch-derived array  →  permuted to NHWC at entry
  - Output: NHWC array                →  permuted to NCHW at exit

Weight naming convention: exact HF diffusers dotted paths, e.g.
  down_blocks.1.resnets.0.conv1.weight
  down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight
  up_blocks.0.resnets.2.norm1.weight
  mid_block.attentions.0.transformer_blocks.5.ff.net.2.weight

SDXL fixed config (stabilityai/stable-diffusion-xl-base-1.0), copied
verbatim from mlx_pipeline/unet.py rather than re-derived:
  block_out_channels      : [320, 640, 1280]
  layers_per_block        : 2
  transformer_layers      : [1, 2, 10]  (per CrossAttn level / mid — NOTE:
                             up_blocks use this list REVERSED per-block, so
                             up_blocks.0 (1280ch) gets 10 layers while
                             down_blocks.1 (640ch) gets 1 and up_blocks.1
                             (also 640ch) gets 2 — this asymmetry is the real
                             SDXL config, not a bug; do not "simplify" it)
  attention_head_dim      : dim_per_head=64 fixed => num_heads = channels // 64
  cross_attention_dim     : 2048
  addition_embed_type     : "text_time"
  addition_time_embed_dim : 256
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp

Params = Dict[str, "jnp.ndarray"]

# ── SDXL fixed config ─────────────────────────────────────────────────────────
_BLOCK_OUT_CHANNELS = [320, 640, 1280]
_LAYERS_PER_BLOCK = 2
_TRANSFORMER_LAYERS = [1, 2, 10]  # [down level-1, down level-2, mid]  (see module docstring)
_DIM_PER_HEAD = 64
_TIME_DIM = 320
_ADD_TIME_EMBED_DIM = 256
_NORM_GROUPS = 32
_NORM_EPS = 1e-5


def _num_heads(channels: int) -> int:
    return channels // _DIM_PER_HEAD


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _silu(x: "jnp.ndarray") -> "jnp.ndarray":
    return jax.nn.silu(x)


def _gelu(x: "jnp.ndarray") -> "jnp.ndarray":
    # approximate=False => exact erf-based GELU, matching PyTorch's default
    # nn.GELU() (no "tanh" approximation) used by HF diffusers' GEGLU.
    return jax.nn.gelu(x, approximate=False)


def _sinusoidal_embedding(
    timesteps: "jnp.ndarray",
    dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
) -> "jnp.ndarray":
    """Sinusoidal (Fourier) timestep embedding — no learnable parameters.

    Matches HuggingFace ``Timesteps`` / ``get_timestep_embedding``.
    """
    half = dim // 2
    freqs = jnp.arange(0, half, dtype=jnp.float32)
    freqs = jnp.exp(-math.log(10000) * freqs / (half - downscale_freq_shift))
    x = timesteps[:, None].astype(jnp.float32) * freqs[None, :]  # [B, half]
    if flip_sin_to_cos:
        emb = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)
    else:
        emb = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    return emb  # [B, dim]


def _upsample2x(x: "jnp.ndarray") -> "jnp.ndarray":
    """Nearest-neighbour 2x upsampling for NHWC tensors."""
    x = jnp.repeat(x, 2, axis=1)
    x = jnp.repeat(x, 2, axis=2)
    return x


def _softmax_attention(q, k, v, scale: float):
    """Explicit scaled dot-product attention, softmax computed in float32.

    All tensors in [B, heads, seq, dim_per_head] format.
    """
    orig_dtype = q.dtype
    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
    attn = jax.nn.softmax(attn, axis=-1).astype(orig_dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v)


# ═══════════════════════════════════════════════════════════════════════════════
#  Primitive layers
# ═══════════════════════════════════════════════════════════════════════════════

def conv2d(params: Params, prefix: str, x, stride: int = 1, padding: int = 1):
    """2-D conv. Weight expected in HWIO layout (see jax_pipeline.convert)."""
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


def linear(params: Params, prefix: str, x):
    """y = x @ W^T + b, W stored in PyTorch/HF [out, in] layout (unchanged by convert)."""
    w = params[f"{prefix}.weight"]  # [out, in]
    b = params.get(f"{prefix}.bias")
    out = x @ w.T
    if b is not None:
        out = out + b
    return out


def group_norm(params: Params, prefix: str, x, num_groups: int = _NORM_GROUPS, eps: float = _NORM_EPS):
    """PyTorch-compatible GroupNorm over NHWC: groups split the channel axis,
    normalization is over (H, W, C/G) per group per sample.
    """
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


def layer_norm(params: Params, prefix: str, x, eps: float = _NORM_EPS):
    w = params[f"{prefix}.weight"]
    b = params[f"{prefix}.bias"]
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - mean) / jnp.sqrt(var + eps)
    out = normed * w.astype(jnp.float32) + b.astype(jnp.float32)
    return out.astype(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention / Feed-forward / Transformer block
# ═══════════════════════════════════════════════════════════════════════════════

def attention(params: Params, prefix: str, hidden_states, num_heads: int, encoder_hidden_states=None):
    """Weight paths: to_q / to_k / to_v / to_out.0 (matches HF Attention)."""
    B, Sq, D = hidden_states.shape
    context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    Sk = context.shape[1]
    dh = D // num_heads

    q = linear(params, f"{prefix}.to_q", hidden_states)
    k = linear(params, f"{prefix}.to_k", context)
    v = linear(params, f"{prefix}.to_v", context)

    q = jnp.transpose(q.reshape(B, Sq, num_heads, dh), (0, 2, 1, 3))
    k = jnp.transpose(k.reshape(B, Sk, num_heads, dh), (0, 2, 1, 3))
    v = jnp.transpose(v.reshape(B, Sk, num_heads, dh), (0, 2, 1, 3))

    out = _softmax_attention(q, k, v, scale=dh ** -0.5)  # [B, H, Sq, dh]

    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, Sq, num_heads * dh)
    return linear(params, f"{prefix}.to_out.0", out)


def geglu(params: Params, prefix: str, x):
    proj = linear(params, f"{prefix}.proj", x)
    x_part, gate = jnp.split(proj, 2, axis=-1)
    return x_part * _gelu(gate)


def feed_forward(params: Params, prefix: str, x):
    """net.0 = GEGLU, net.1 = Dropout (no-op at inference), net.2 = output Linear."""
    h = geglu(params, f"{prefix}.net.0", x)
    h = linear(params, f"{prefix}.net.2", h)
    return h


def basic_transformer_block(params: Params, prefix: str, hidden_states, encoder_hidden_states, num_heads: int):
    h = layer_norm(params, f"{prefix}.norm1", hidden_states)
    h = attention(params, f"{prefix}.attn1", h, num_heads)  # self-attention
    hidden_states = hidden_states + h

    h = layer_norm(params, f"{prefix}.norm2", hidden_states)
    h = attention(params, f"{prefix}.attn2", h, num_heads, encoder_hidden_states=encoder_hidden_states)  # cross-attention
    hidden_states = hidden_states + h

    h = layer_norm(params, f"{prefix}.norm3", hidden_states)
    h = feed_forward(params, f"{prefix}.ff", h)
    hidden_states = hidden_states + h

    return hidden_states


def transformer_2d(params: Params, prefix: str, hidden_states, encoder_hidden_states, num_heads: int, num_layers: int):
    """Spatial transformer: GroupNorm -> flatten HW -> linear proj_in -> N blocks -> linear proj_out -> residual."""
    B, H, W, C = hidden_states.shape
    residual = hidden_states

    h = group_norm(params, f"{prefix}.norm", hidden_states)
    h = h.reshape(B, H * W, C)
    h = linear(params, f"{prefix}.proj_in", h)

    for i in range(num_layers):
        h = basic_transformer_block(params, f"{prefix}.transformer_blocks.{i}", h, encoder_hidden_states, num_heads)

    h = linear(params, f"{prefix}.proj_out", h)
    h = h.reshape(B, H, W, C)
    return h + residual


# ═══════════════════════════════════════════════════════════════════════════════
#  ResnetBlock2D / Downsample / Upsample
# ═══════════════════════════════════════════════════════════════════════════════

def resnet_block(params: Params, prefix: str, hidden_states, temb):
    """Weight paths: norm1/conv1, time_emb_proj, norm2/conv2, optional conv_shortcut."""
    residual = hidden_states
    has_shortcut = f"{prefix}.conv_shortcut.weight" in params

    h = _silu(group_norm(params, f"{prefix}.norm1", hidden_states))
    h = conv2d(params, f"{prefix}.conv1", h, stride=1, padding=1)

    temb_proj = linear(params, f"{prefix}.time_emb_proj", _silu(temb))
    h = h + temb_proj[:, None, None, :]

    h = _silu(group_norm(params, f"{prefix}.norm2", h))
    h = conv2d(params, f"{prefix}.conv2", h, stride=1, padding=1)

    if has_shortcut:
        residual = conv2d(params, f"{prefix}.conv_shortcut", residual, stride=1, padding=0)

    return h + residual


def downsample_2d(params: Params, prefix: str, x):
    return conv2d(params, f"{prefix}.conv", x, stride=2, padding=1)


def upsample_2d(params: Params, prefix: str, x):
    return conv2d(params, f"{prefix}.conv", _upsample2x(x), stride=1, padding=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Down / Mid / Up blocks
# ═══════════════════════════════════════════════════════════════════════════════

def down_block2d(params: Params, prefix: str, x, temb, num_layers: int):
    """Pure-ResNet down block (no attention) — SDXL level 0 (ch=320)."""
    skips: List = []
    for i in range(num_layers):
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
        skips.append(x)
    x = downsample_2d(params, f"{prefix}.downsamplers.0", x)
    skips.append(x)
    return x, skips


def cross_attn_down_block2d(
    params: Params, prefix: str, x, temb, encoder_hidden_states,
    num_layers: int, num_attn_layers: int, num_heads: int, add_downsample: bool,
):
    skips: List = []
    for i in range(num_layers):
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
        x = transformer_2d(params, f"{prefix}.attentions.{i}", x, encoder_hidden_states, num_heads, num_attn_layers)
        skips.append(x)
    if add_downsample:
        x = downsample_2d(params, f"{prefix}.downsamplers.0", x)
        skips.append(x)
    return x, skips


def mid_block(params: Params, prefix: str, x, temb, encoder_hidden_states, num_heads: int, num_attn_layers: int):
    """SDXL mid block: ResNet -> Transformer -> ResNet."""
    x = resnet_block(params, f"{prefix}.resnets.0", x, temb)
    x = transformer_2d(params, f"{prefix}.attentions.0", x, encoder_hidden_states, num_heads, num_attn_layers)
    x = resnet_block(params, f"{prefix}.resnets.1", x, temb)
    return x


def cross_attn_up_block2d(
    params: Params, prefix: str, x, temb, encoder_hidden_states, res_samples: List,
    num_layers: int, num_attn_layers: int, num_heads: int, add_upsample: bool,
):
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
        x = transformer_2d(params, f"{prefix}.attentions.{i}", x, encoder_hidden_states, num_heads, num_attn_layers)
    if add_upsample:
        x = upsample_2d(params, f"{prefix}.upsamplers.0", x)
    return x


def up_block2d(params: Params, prefix: str, x, temb, res_samples: List, num_layers: int, add_upsample: bool):
    """Pure-ResNet up block — SDXL level 2 (ch=320), no attention."""
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
    if add_upsample:
        x = upsample_2d(params, f"{prefix}.upsamplers.0", x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Time / Add embedding
# ═══════════════════════════════════════════════════════════════════════════════

def timestep_embedding_mlp(params: Params, prefix: str, x):
    """Weight paths: time_embedding.linear_1 / linear_2."""
    h = linear(params, f"{prefix}.linear_1", x)
    h = _silu(h)
    h = linear(params, f"{prefix}.linear_2", h)
    return h


def add_embedding(params: Params, prefix: str, text_embeds, time_ids, time_embed_dim: int = _ADD_TIME_EMBED_DIM):
    """SDXL additional conditioning (text_embeds + Fourier(time_ids)) -> MLP.

    Weight paths: add_embedding.linear_1 / linear_2 (add_time_proj has no
    learnable weights — it is a sinusoidal Timesteps).
    """
    B, _num_ids = time_ids.shape
    time_ids_flat = time_ids.reshape(-1)
    fourier = _sinusoidal_embedding(time_ids_flat, time_embed_dim, flip_sin_to_cos=True)
    fourier = fourier.reshape(B, -1)

    cond = jnp.concatenate([text_embeds, fourier], axis=-1)
    h = linear(params, f"{prefix}.linear_1", cond)
    h = _silu(h)
    h = linear(params, f"{prefix}.linear_2", h)
    return h


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def unet_forward(
    params: Params,
    sample: "jnp.ndarray",                 # [B, 4, H, W]  NCHW
    timestep: "jnp.ndarray",               # [B]
    encoder_hidden_states: "jnp.ndarray",  # [B, S, 2048]
    added_cond_kwargs: Dict[str, "jnp.ndarray"],  # {"text_embeds": [B,1280], "time_ids": [B,6]}
) -> "jnp.ndarray":
    """SDXL UNet2DConditionModel forward pass. Call signature mirrors HF
    UNet2DConditionModel.__call__. Not jit-wrapped itself — see
    ``unet_forward_jit`` (no host offload) or ``jax_pipeline.host_offload``
    (offload-aware jit wrapper).

    Output: [B, 4, H, W] NCHW, matching input convention.
    """
    ch = _BLOCK_OUT_CHANNELS  # [320, 640, 1280]

    # 1. NCHW -> NHWC
    sample = jnp.transpose(sample, (0, 2, 3, 1))

    # 2. Time embedding
    timestep_emb = _sinusoidal_embedding(
        timestep.astype(jnp.float32), _TIME_DIM, flip_sin_to_cos=True
    ).astype(sample.dtype)
    temb = timestep_embedding_mlp(params, "time_embedding", timestep_emb)

    # 3. SDXL additional conditioning
    text_embeds = added_cond_kwargs["text_embeds"].astype(sample.dtype)
    time_ids = added_cond_kwargs["time_ids"].astype(jnp.float32)
    aug_emb = add_embedding(params, "add_embedding", text_embeds, time_ids).astype(sample.dtype)
    temb = temb + aug_emb

    # 4. Input conv
    sample = conv2d(params, "conv_in", sample, stride=1, padding=1)

    # 5. Down path — collect ALL skip connections (including conv_in output)
    down_block_res_samples: List = [sample]

    sample, skips = down_block2d(params, "down_blocks.0", sample, temb, _LAYERS_PER_BLOCK)
    down_block_res_samples.extend(skips)

    sample, skips = cross_attn_down_block2d(
        params, "down_blocks.1", sample, temb, encoder_hidden_states,
        num_layers=_LAYERS_PER_BLOCK,
        num_attn_layers=_TRANSFORMER_LAYERS[0],
        num_heads=_num_heads(ch[1]),
        add_downsample=True,
    )
    down_block_res_samples.extend(skips)

    sample, skips = cross_attn_down_block2d(
        params, "down_blocks.2", sample, temb, encoder_hidden_states,
        num_layers=_LAYERS_PER_BLOCK,
        num_attn_layers=_TRANSFORMER_LAYERS[1],
        num_heads=_num_heads(ch[2]),
        add_downsample=False,
    )
    down_block_res_samples.extend(skips)

    # 6. Mid block
    sample = mid_block(
        params, "mid_block", sample, temb, encoder_hidden_states,
        num_heads=_num_heads(ch[2]), num_attn_layers=_TRANSFORMER_LAYERS[2],
    )

    # 7. Up path. Skip counts per up block = layers_per_block + 1 = 3.
    #    Transformer-layer counts intentionally mirror _TRANSFORMER_LAYERS
    #    REVERSED per up-block index, not derived from channel count (see
    #    module docstring) — copied verbatim from mlx_pipeline/unet.py.
    up_block_specs = [
        dict(name="up_blocks.0", num_layers=3, attn=True,
             num_attn_layers=_TRANSFORMER_LAYERS[2], num_heads=_num_heads(ch[2]), add_upsample=True),
        dict(name="up_blocks.1", num_layers=3, attn=True,
             num_attn_layers=_TRANSFORMER_LAYERS[1], num_heads=_num_heads(ch[1]), add_upsample=True),
        dict(name="up_blocks.2", num_layers=3, attn=False, add_upsample=False),
    ]
    for spec in up_block_specs:
        n = spec["num_layers"]
        res_samples = [down_block_res_samples.pop() for _ in range(n)]
        if spec["attn"]:
            sample = cross_attn_up_block2d(
                params, spec["name"], sample, temb, encoder_hidden_states, res_samples,
                num_layers=n, num_attn_layers=spec["num_attn_layers"],
                num_heads=spec["num_heads"], add_upsample=spec["add_upsample"],
            )
        else:
            sample = up_block2d(params, spec["name"], sample, temb, res_samples, n, spec["add_upsample"])

    # 8. Output projection
    sample = _silu(group_norm(params, "conv_norm_out", sample))
    sample = conv2d(params, "conv_out", sample, stride=1, padding=1)

    # 9. NHWC -> NCHW
    return jnp.transpose(sample, (0, 3, 1, 2))


unet_forward_jit = jax.jit(unet_forward)
