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
  transformer_layers      : [1, 2, 10] — indexed by down_block index (0/1/2),
                             matching ldm_patched's own SDXL reference config
                             (supported_models.py: transformer_depth=
                             [0,0,2,2,10,10], i.e. 2 per resnet at the 640ch
                             level, 10 at the 1280ch level). Index 0 (=1) is
                             DEAD/unused — down_blocks.0 has no attention at
                             all, so nothing ever reads it; down_blocks.1
                             uses index 1 (=2), down_blocks.2 uses index 2
                             (=10). mid_block reuses index 2 (=10, matching
                             the deepest/1280ch level). up_blocks mirror this
                             REVERSED per-block: up_blocks.0 (1280ch) gets
                             index 2 (=10), up_blocks.1 (640ch) gets index 1
                             (=2) — matching down_blocks.1 exactly, no
                             asymmetry. (A prior version of this file mis-
                             indexed down_blocks.1/.2 to use index 0/1
                             instead of 1/2 — silently ran too few
                             transformer_blocks at both cross-attention down
                             levels; caught via jax_pipeline.block_cache's
                             strict per-block parameter coverage check
                             raising on real checkpoint weights it found
                             unassigned. Fixed 2026-07-19 — see git history.)
  attention_head_dim      : dim_per_head=64 fixed => num_heads = channels // 64
  cross_attention_dim     : 2048
  addition_embed_type     : "text_time"
  addition_time_embed_dim : 256
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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


def upsample_2d(params: Params, prefix: str, x, target_hw: Optional[tuple] = None):
    """Upsample, then conv.

    SDXL/SD1.5 assume pixel resolution is a multiple of 64 (latent
    divisible by 8, so the UNet's 3 internal 2x downsample stages divide
    evenly) — but nothing actually enforces that: this webui's width/
    height sliders only step by 8 (matching just the VAE's factor), so a
    latent dimension not divisible by 8 is a real, reachable case.

    The reference PyTorch UNet (ldm_patched/ldm/modules/diffusionmodules/
    openaimodel.py) handles this not via padding tricks on Downsample (its
    stride-2/padding-1 conv already rounds odd inputs up to ceil(L/2), the
    same formula our `conv2d` uses), but by having each Upsample target
    the *exact* shape of the next skip-connection tensor about to be
    concatenated (`output_shape = hs[-1].shape`), rather than blindly
    doubling. Skipping that step means our naive always-2x upsample
    silently diverges from the skip tensor's actual shape whenever a
    latent dim isn't divisible by 8 at that level, which crashes
    jnp.concatenate at the next skip connection (shapes must match exactly
    on non-concat axes) rather than merely producing wrong output.

    `target_hw`, when given, is used only if it differs from the naive 2x
    result — the already-verified jnp.repeat-based 2x path is untouched
    for the common (resolution divisible by 64) case. jax.image.resize's
    nearest-neighbor index mapping may not be bit-identical to PyTorch's
    F.interpolate(mode="nearest") in the non-2x case, but any working
    output beats a hard crash for what both this webui's own step=8 UI and
    the reference implementation already treat as a valid resolution.
    """
    B, H, W, C = x.shape
    if target_hw is not None and tuple(target_hw) != (H * 2, W * 2):
        target_h, target_w = target_hw
        x = jax.image.resize(x, (B, target_h, target_w, C), method="nearest")
    else:
        x = _upsample2x(x)
    return conv2d(params, f"{prefix}.conv", x, stride=1, padding=1)


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
    target_hw: Optional[tuple] = None,
):
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
        x = transformer_2d(params, f"{prefix}.attentions.{i}", x, encoder_hidden_states, num_heads, num_attn_layers)
    if add_upsample:
        x = upsample_2d(params, f"{prefix}.upsamplers.0", x, target_hw=target_hw)
    return x


def up_block2d(
    params: Params, prefix: str, x, temb, res_samples: List, num_layers: int, add_upsample: bool,
    target_hw: Optional[tuple] = None,
):
    """Pure-ResNet up block — SDXL level 2 (ch=320), no attention."""
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        x = resnet_block(params, f"{prefix}.resnets.{i}", x, temb)
    if add_upsample:
        x = upsample_2d(params, f"{prefix}.upsamplers.0", x, target_hw=target_hw)
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
        num_attn_layers=_TRANSFORMER_LAYERS[1],
        num_heads=_num_heads(ch[1]),
        add_downsample=True,
    )
    down_block_res_samples.extend(skips)

    sample, skips = cross_attn_down_block2d(
        params, "down_blocks.2", sample, temb, encoder_hidden_states,
        num_layers=_LAYERS_PER_BLOCK,
        num_attn_layers=_TRANSFORMER_LAYERS[2],
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
        # Mirrors the reference UNet's `output_shape = hs[-1].shape if hs else None`
        # (openaimodel.py): after popping this block's own skips, whatever
        # is left on top of the stack is the exact shape the NEXT up_block
        # will consume — that's what THIS block's own trailing upsample (if
        # any) must land on, not a naive doubling, so the following
        # concatenate sees matching shapes even when the latent dims aren't
        # divisible by 8 (pixel resolution not a multiple of 64).
        target_hw = down_block_res_samples[-1].shape[1:3] if down_block_res_samples else None
        if spec["attn"]:
            sample = cross_attn_up_block2d(
                params, spec["name"], sample, temb, encoder_hidden_states, res_samples,
                num_layers=n, num_attn_layers=spec["num_attn_layers"],
                num_heads=spec["num_heads"], add_upsample=spec["add_upsample"],
                target_hw=target_hw,
            )
        else:
            sample = up_block2d(
                params, spec["name"], sample, temb, res_samples, n, spec["add_upsample"],
                target_hw=target_hw,
            )

    # 8. Output projection
    sample = _silu(group_norm(params, "conv_norm_out", sample))
    sample = conv2d(params, "conv_out", sample, stride=1, padding=1)

    # 9. NHWC -> NCHW
    return jnp.transpose(sample, (0, 3, 1, 2))


unet_forward_jit = jax.jit(unet_forward)


# ═══════════════════════════════════════════════════════════════════════════════
#  Block-level streaming forward pass (VRAM-constrained cards)
# ═══════════════════════════════════════════════════════════════════════════════
#
# unet_forward/unet_forward_jit above compile the ENTIRE UNet as one XLA
# program, which requires the whole ~2.6B-parameter (bf16: ~5-6GB
# including XLA's own compiled-executable/compilation state) params
# pytree to already be device-resident before the call — fine for
# large-VRAM cards, but on constrained cards (<20GB, see
# host_offload.PhaseManager) that single ~5-6GB residency requirement
# alone can leave too little headroom for the forward pass's own
# activation memory. Extensive VRAM profiling in this repo's commit
# history (~2026-07-19) showed a whole-UNet-resident approach
# consistently landing within a few hundred MB of an 8GB card's ceiling,
# no matter how aggressively everything ELSE (torch's redundant copies,
# CLIP, VAE) was evicted first.
#
# unet_forward_streaming below is architecturally different, not just a
# smaller jit: instead of ONE jax.jit covering the whole network, each
# block — each ResnetBlock2D, each individual transformer sub-block (NOT
# the whole bundled N-layer Transformer2D — SDXL's deepest attention
# stacks bundle up to 10 transformer_blocks per call site, ~700MB if kept
# as one streaming unit, so those are split further), each
# Downsample/Upsample, both embedding MLPs, conv_in, conv_norm_out,
# conv_out — is jitted SEPARATELY, and a Python-level
# jax_pipeline.block_cache.BlockParamCache decides, block by block,
# whether to stage that block's weights onto device or reuse an
# already-resident copy. This mirrors the layer-streaming lowvram
# approach ldm_patched's own PyTorch model_patcher.py already uses for
# the torch UNet (move_weight_functions per nn.Module via forward hooks),
# reimplemented for JAX's array/pytree model — JAX has no stateful
# modules or hooks to hang a callback off, hence the explicit Python-level
# orchestration below instead of one big traced function.
#
# Trade-off, and why it's the right one here: a full forward pass visits
# every block exactly once (no cross-block weight reuse within one pass,
# unlike e.g. attention KV caching), so a cache budget smaller than the
# whole UNet means every block's weights get re-transferred from pinned
# host on EVERY denoising step, not just once per generation — that's the
# deliberate trade this cache makes (bounded, small peak VRAM for real
# PCIe transfer time), not an oversight.
#
# Numerical correctness: every block-level jitted function below wraps
# the EXACT SAME per-block function unet_forward already uses
# (resnet_block, basic_transformer_block, downsample_2d, upsample_2d,
# conv2d, linear, group_norm, timestep_embedding_mlp, add_embedding) —
# this section only changes WHERE each call's params come from (a
# per-block cache lookup instead of a slice of one big resident dict) and
# HOW each call is jit-compiled (individually, not as part of one big
# trace). The math executed, and the control flow (down blocks -> mid ->
# up blocks, skip-connection bookkeeping, target_hw shape-matching for
# non-64-multiple resolutions), are copied verbatim from unet_forward
# above.

import functools


def _timed_jit(jitted_fn):
    """TEMPORARY compute-timing instrumentation wrapper (see
    jax_pipeline/_debug_profile.py's "speed phase") around an
    ALREADY-``jax.jit``-wrapped callable — jax.jit's own tracing/caching
    happens inside ``jitted_fn``, unaffected by this wrapper.

    Measures each call's wall time via ``jax.block_until_ready()`` on the
    result (real completion time, not async-dispatch time) and records
    it under "compute", keyed by the ``prefix`` kwarg every call site
    already passes (so timing is broken down per exact block_id, not
    just per function).

    Near-zero overhead when disabled (one ``if _debug_profile.ENABLED``
    check, no timing/import work) — safe to leave wrapping every call
    site unconditionally.
    """
    @functools.wraps(jitted_fn)
    def wrapper(*args, **kwargs):
        from jax_pipeline import _debug_profile
        if not _debug_profile.ENABLED:
            return jitted_fn(*args, **kwargs)
        import time
        t0 = time.perf_counter()
        out = jitted_fn(*args, **kwargs)
        jax.block_until_ready(out)
        block_id = kwargs.get("prefix", "?")
        _debug_profile.record_timing("compute", block_id, time.perf_counter() - t0)
        return out
    return wrapper


_jit_conv2d = _timed_jit(jax.jit(conv2d, static_argnames=("prefix", "stride", "padding")))
_jit_linear = _timed_jit(jax.jit(linear, static_argnames=("prefix",)))
_jit_group_norm = _timed_jit(jax.jit(group_norm, static_argnames=("prefix", "num_groups", "eps")))
_jit_timestep_embedding_mlp = _timed_jit(jax.jit(timestep_embedding_mlp, static_argnames=("prefix",)))
_jit_add_embedding = _timed_jit(jax.jit(add_embedding, static_argnames=("prefix", "time_embed_dim")))
_jit_resnet_block = _timed_jit(jax.jit(resnet_block, static_argnames=("prefix",)))
_jit_basic_transformer_block = _timed_jit(
    jax.jit(basic_transformer_block, static_argnames=("prefix", "num_heads"))
)
_jit_downsample_2d = _timed_jit(jax.jit(downsample_2d, static_argnames=("prefix",)))
_jit_upsample_2d = _timed_jit(jax.jit(upsample_2d, static_argnames=("prefix", "target_hw")))


# ── block registry ──────────────────────────────────────────────────────────

def _transformer_2d_block_ids(prefix: str, num_layers: int) -> List[str]:
    """Every streaming block_id one transformer_2d call site needs — the
    norm+proj_in and proj_out wrapper stages, plus one id per individual
    basic_transformer_block (the fine-grained split that keeps SDXL's
    deepest attention stacks, which bundle up to 10 transformer_blocks
    per call site, from becoming a single ~700MB streaming unit).
    """
    ids = [f"{prefix}.norm", f"{prefix}.proj_in"]
    for i in range(num_layers):
        ids.append(f"{prefix}.transformer_blocks.{i}")
    ids.append(f"{prefix}.proj_out")
    return ids


def build_block_ids() -> List[str]:
    """Enumerate every streaming block_id covering the UNet's full
    parameter space, walked in the SAME down -> mid -> up structure
    unet_forward/unet_forward_streaming use. Must stay in exact sync with
    that traversal — jax_pipeline.block_cache.partition_params_by_block's
    coverage check (every param key claimed by exactly one block_id,
    every block_id claims at least one key) is the safety net if it ever
    drifts.

    ``"class_embedding"``: ldm_patched.modules.utils.unet_to_diffusers's
    own key-mapping table (jax_pipeline.convert reuses it directly) maps
    the SAME source ldm weights (``label_emb.0.0/2.*``) to BOTH
    ``add_embedding.linear_1/2.*`` AND ``class_embedding.linear_1/2.*`` —
    intentional duplication for cross-compatibility with different HF
    diffusers naming conventions, not a jax_pipeline bug. The forward
    pass only ever reads ``add_embedding.*`` (see ``unet_forward``'s
    step 3) — ``class_embedding`` is real, present, but permanently
    unused data that still needs somewhere to live in the registry so
    partition_params_by_block's coverage check doesn't (correctly) flag
    it as unassigned. Small (2 tiny Linear layers) — parked in pinned
    host memory, never staged to device since nothing ever calls
    ``cache.get("class_embedding")``.
    """
    ch = _BLOCK_OUT_CHANNELS
    ids = ["time_embedding", "add_embedding", "class_embedding", "conv_in"]

    # down_blocks.0: pure-resnet, no attention
    for i in range(_LAYERS_PER_BLOCK):
        ids.append(f"down_blocks.0.resnets.{i}")
    ids.append("down_blocks.0.downsamplers.0")

    # down_blocks.1 / down_blocks.2: cross-attention
    for level, num_attn, add_downsample in (
        (1, _TRANSFORMER_LAYERS[1], True),
        (2, _TRANSFORMER_LAYERS[2], False),
    ):
        for i in range(_LAYERS_PER_BLOCK):
            ids.append(f"down_blocks.{level}.resnets.{i}")
            ids += _transformer_2d_block_ids(f"down_blocks.{level}.attentions.{i}", num_attn)
        if add_downsample:
            ids.append(f"down_blocks.{level}.downsamplers.0")

    # mid_block
    ids.append("mid_block.resnets.0")
    ids += _transformer_2d_block_ids("mid_block.attentions.0", _TRANSFORMER_LAYERS[2])
    ids.append("mid_block.resnets.1")

    # up_blocks (3 layers each; up_blocks.0/.1 have attention, up_blocks.2 doesn't —
    # attn layer counts mirror _TRANSFORMER_LAYERS reversed per up-block index,
    # matching unet_forward's up_block_specs exactly)
    up_specs = [
        ("up_blocks.0", 3, _TRANSFORMER_LAYERS[2], True),
        ("up_blocks.1", 3, _TRANSFORMER_LAYERS[1], True),
        ("up_blocks.2", 3, None, False),
    ]
    for name, n, num_attn, add_upsample in up_specs:
        for i in range(n):
            ids.append(f"{name}.resnets.{i}")
            if num_attn is not None:
                ids += _transformer_2d_block_ids(f"{name}.attentions.{i}", num_attn)
        if add_upsample:
            ids.append(f"{name}.upsamplers.0")

    ids += ["conv_norm_out", "conv_out"]
    return ids


# ── streaming block functions ────────────────────────────────────────────────

def _transformer_2d_streaming(cache, prefix: str, hidden_states, encoder_hidden_states, num_heads: int, num_layers: int):
    B, H, W, C = hidden_states.shape
    residual = hidden_states

    norm_id = f"{prefix}.norm"
    h = _jit_group_norm(cache.get(norm_id), x=hidden_states, prefix=norm_id)
    h = h.reshape(B, H * W, C)
    proj_in_id = f"{prefix}.proj_in"
    h = _jit_linear(cache.get(proj_in_id), x=h, prefix=proj_in_id)

    for i in range(num_layers):
        block_id = f"{prefix}.transformer_blocks.{i}"
        h = _jit_basic_transformer_block(
            cache.get(block_id), hidden_states=h, encoder_hidden_states=encoder_hidden_states,
            prefix=block_id, num_heads=num_heads,
        )

    proj_out_id = f"{prefix}.proj_out"
    h = _jit_linear(cache.get(proj_out_id), x=h, prefix=proj_out_id)
    h = h.reshape(B, H, W, C)
    return h + residual


def _down_block2d_streaming(cache, prefix: str, x, temb, num_layers: int):
    skips: List = []
    for i in range(num_layers):
        rp = f"{prefix}.resnets.{i}"
        x = _jit_resnet_block(cache.get(rp), hidden_states=x, temb=temb, prefix=rp)
        skips.append(x)
    dp = f"{prefix}.downsamplers.0"
    x = _jit_downsample_2d(cache.get(dp), x=x, prefix=dp)
    skips.append(x)
    return x, skips


def _cross_attn_down_block2d_streaming(
    cache, prefix: str, x, temb, encoder_hidden_states,
    num_layers: int, num_attn_layers: int, num_heads: int, add_downsample: bool,
):
    skips: List = []
    for i in range(num_layers):
        rp = f"{prefix}.resnets.{i}"
        x = _jit_resnet_block(cache.get(rp), hidden_states=x, temb=temb, prefix=rp)
        ap = f"{prefix}.attentions.{i}"
        x = _transformer_2d_streaming(cache, ap, x, encoder_hidden_states, num_heads, num_attn_layers)
        skips.append(x)
    if add_downsample:
        dp = f"{prefix}.downsamplers.0"
        x = _jit_downsample_2d(cache.get(dp), x=x, prefix=dp)
        skips.append(x)
    return x, skips


def _mid_block_streaming(cache, prefix: str, x, temb, encoder_hidden_states, num_heads: int, num_attn_layers: int):
    p0 = f"{prefix}.resnets.0"
    x = _jit_resnet_block(cache.get(p0), hidden_states=x, temb=temb, prefix=p0)
    ap = f"{prefix}.attentions.0"
    x = _transformer_2d_streaming(cache, ap, x, encoder_hidden_states, num_heads, num_attn_layers)
    p1 = f"{prefix}.resnets.1"
    x = _jit_resnet_block(cache.get(p1), hidden_states=x, temb=temb, prefix=p1)
    return x


def _cross_attn_up_block2d_streaming(
    cache, prefix: str, x, temb, encoder_hidden_states, res_samples: List,
    num_layers: int, num_attn_layers: int, num_heads: int, add_upsample: bool,
    target_hw: Optional[Tuple[int, int]] = None,
):
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        rp = f"{prefix}.resnets.{i}"
        x = _jit_resnet_block(cache.get(rp), hidden_states=x, temb=temb, prefix=rp)
        ap = f"{prefix}.attentions.{i}"
        x = _transformer_2d_streaming(cache, ap, x, encoder_hidden_states, num_heads, num_attn_layers)
    if add_upsample:
        up = f"{prefix}.upsamplers.0"
        x = _jit_upsample_2d(cache.get(up), x=x, prefix=up, target_hw=target_hw)
    return x


def _up_block2d_streaming(
    cache, prefix: str, x, temb, res_samples: List, num_layers: int, add_upsample: bool,
    target_hw: Optional[Tuple[int, int]] = None,
):
    for i in range(num_layers):
        x = jnp.concatenate([x, res_samples[i]], axis=-1)
        rp = f"{prefix}.resnets.{i}"
        x = _jit_resnet_block(cache.get(rp), hidden_states=x, temb=temb, prefix=rp)
    if add_upsample:
        up = f"{prefix}.upsamplers.0"
        x = _jit_upsample_2d(cache.get(up), x=x, prefix=up, target_hw=target_hw)
    return x


def unet_forward_streaming(
    cache,
    sample: "jnp.ndarray",
    timestep: "jnp.ndarray",
    encoder_hidden_states: "jnp.ndarray",
    added_cond_kwargs: Dict[str, "jnp.ndarray"],
) -> "jnp.ndarray":
    """Block-streaming counterpart of unet_forward — identical math and
    control flow, sourced from ``cache`` (a
    jax_pipeline.block_cache.BlockParamCache already loaded with this
    UNet's params — see jax_pipeline.pipeline/lora for construction) one
    block at a time instead of from one big resident params dict.

    Deliberately NOT itself wrapped in jax.jit — cache.get(...) has to
    run as plain eager Python between the individually-jitted block
    calls below; that Python-level per-block residency decision is the
    entire point (see the module-level "Block-level streaming forward
    pass" section docstring above).

    Output: [B, 4, H, W] NCHW, matching unet_forward's convention.
    """
    ch = _BLOCK_OUT_CHANNELS

    sample = jnp.transpose(sample, (0, 2, 3, 1))

    timestep_emb = _sinusoidal_embedding(
        timestep.astype(jnp.float32), _TIME_DIM, flip_sin_to_cos=True
    ).astype(sample.dtype)
    temb = _jit_timestep_embedding_mlp(cache.get("time_embedding"), x=timestep_emb, prefix="time_embedding")

    text_embeds = added_cond_kwargs["text_embeds"].astype(sample.dtype)
    time_ids = added_cond_kwargs["time_ids"].astype(jnp.float32)
    aug_emb = _jit_add_embedding(
        cache.get("add_embedding"), text_embeds=text_embeds, time_ids=time_ids,
        prefix="add_embedding", time_embed_dim=_ADD_TIME_EMBED_DIM,
    ).astype(sample.dtype)
    temb = temb + aug_emb

    sample = _jit_conv2d(cache.get("conv_in"), x=sample, prefix="conv_in", stride=1, padding=1)

    down_block_res_samples: List = [sample]

    sample, skips = _down_block2d_streaming(cache, "down_blocks.0", sample, temb, _LAYERS_PER_BLOCK)
    down_block_res_samples.extend(skips)

    sample, skips = _cross_attn_down_block2d_streaming(
        cache, "down_blocks.1", sample, temb, encoder_hidden_states,
        num_layers=_LAYERS_PER_BLOCK,
        num_attn_layers=_TRANSFORMER_LAYERS[1],
        num_heads=_num_heads(ch[1]),
        add_downsample=True,
    )
    down_block_res_samples.extend(skips)

    sample, skips = _cross_attn_down_block2d_streaming(
        cache, "down_blocks.2", sample, temb, encoder_hidden_states,
        num_layers=_LAYERS_PER_BLOCK,
        num_attn_layers=_TRANSFORMER_LAYERS[2],
        num_heads=_num_heads(ch[2]),
        add_downsample=False,
    )
    down_block_res_samples.extend(skips)

    sample = _mid_block_streaming(
        cache, "mid_block", sample, temb, encoder_hidden_states,
        num_heads=_num_heads(ch[2]), num_attn_layers=_TRANSFORMER_LAYERS[2],
    )

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
        target_hw = down_block_res_samples[-1].shape[1:3] if down_block_res_samples else None
        if spec["attn"]:
            sample = _cross_attn_up_block2d_streaming(
                cache, spec["name"], sample, temb, encoder_hidden_states, res_samples,
                num_layers=n, num_attn_layers=spec["num_attn_layers"],
                num_heads=spec["num_heads"], add_upsample=spec["add_upsample"],
                target_hw=target_hw,
            )
        else:
            sample = _up_block2d_streaming(
                cache, spec["name"], sample, temb, res_samples, n, spec["add_upsample"],
                target_hw=target_hw,
            )

    norm_out_id = "conv_norm_out"
    sample = _silu(_jit_group_norm(cache.get(norm_out_id), x=sample, prefix=norm_out_id))
    sample = _jit_conv2d(cache.get("conv_out"), x=sample, prefix="conv_out", stride=1, padding=1)

    return jnp.transpose(sample, (0, 3, 1, 2))
