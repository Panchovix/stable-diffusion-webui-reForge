"""
mlx_pipeline/clip.py — MLX-native CLIP text encoders for SDXL.

Implements CLIP-L (768d, 12 layers, quick_gelu) and CLIP-G (1280d, 32 layers,
standard gelu) running on Apple Silicon Metal via MLX bfloat16.

Architecture
------------
Both encoders use the ldm_patched CLIPTextModel weight format (HF-compatible):

  text_model.embeddings.token_embedding.weight   [V, d]
  text_model.embeddings.position_embedding.weight [77, d]
  text_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
  text_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
  text_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}
  text_model.final_layer_norm.{weight,bias}
  text_projection.weight  [d, d]  (for pooled output)

Integration
-----------
Hooks ``encode_with_transformers`` on forge_clip.CLIP_SD_XL_L and CLIP_SD_XL_G
instances so the CLIP computation runs entirely on Metal via MLX.

Output contract:
  CLIP-L: torch.Tensor [B, 77, 768]  — penultimate hidden state, NOT normed
  CLIP-G: torch.Tensor [B, 77, 1280] — same, with `.pooled` = [B, 1280] projected

Falls back to the original torch implementation when textual inversion is
detected (any token id ≥ 49408).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    import mlx.core as mx

log = logging.getLogger(__name__)

_CLIP_VOCAB_SIZE = 49408  # Standard CLIP vocab; tokens above this = textual inversion
_EOS_TOKEN_ID    = 49407  # <|endoftext|>


# ── activation functions ────────────────────────────────────────────────────────

def _quick_gelu(x: "mx.array") -> "mx.array":
    """CLIP-L uses quick GELU: x * sigmoid(1.702 * x)."""
    import mlx.core as mx
    return x * mx.sigmoid(1.702 * x.astype(mx.float32)).astype(x.dtype)


def _gelu(x: "mx.array") -> "mx.array":
    """CLIP-G uses standard GELU."""
    import mlx.nn as nn
    return nn.gelu(x)


# ── MLX CLIP building blocks ────────────────────────────────────────────────────

class _CLIPSelfAttn:
    """
    Single-head or multi-head self-attention for CLIP.

    Stores weights as flat arrays for fast matmul; no nn.Module overhead.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        q_weight: "mx.array", q_bias: "mx.array",
        k_weight: "mx.array", k_bias: "mx.array",
        v_weight: "mx.array", v_bias: "mx.array",
        out_weight: "mx.array", out_bias: "mx.array",
    ):
        import mlx.core as mx
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.scale     = self.head_dim ** -0.5
        # Weights: [out, in] → bias: [out]
        self.q_w = q_weight.astype(mx.bfloat16)
        self.q_b = q_bias.astype(mx.bfloat16)
        self.k_w = k_weight.astype(mx.bfloat16)
        self.k_b = k_bias.astype(mx.bfloat16)
        self.v_w = v_weight.astype(mx.bfloat16)
        self.v_b = v_bias.astype(mx.bfloat16)
        self.o_w = out_weight.astype(mx.bfloat16)
        self.o_b = out_bias.astype(mx.bfloat16)

    def __call__(self, x: "mx.array", mask: "mx.array") -> "mx.array":
        """x: [B, T, d]; mask: [T, T] additive causal mask → [B, T, d]"""
        import mlx.core as mx
        B, T, d = x.shape
        H, Hd = self.n_heads, self.head_dim

        def proj(w, b):
            return (x @ w.T + b).reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

        q = proj(self.q_w, self.q_b)   # [B, H, T, Hd]
        k = proj(self.k_w, self.k_b)
        v = proj(self.v_w, self.v_b)

        attn = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=self.scale,
            mask=mask.astype(mx.float32),     # [T, T] broadcasts
        )   # [B, H, T, Hd] float32

        out = attn.astype(mx.bfloat16).transpose(0, 2, 1, 3).reshape(B, T, d)
        return out @ self.o_w.T + self.o_b


class _CLIPMLP:
    """Two-layer feed-forward network for CLIP."""

    def __init__(
        self,
        fc1_w: "mx.array", fc1_b: "mx.array",
        fc2_w: "mx.array", fc2_b: "mx.array",
        activation,
    ):
        import mlx.core as mx
        self.fc1_w = fc1_w.astype(mx.bfloat16)
        self.fc1_b = fc1_b.astype(mx.bfloat16)
        self.fc2_w = fc2_w.astype(mx.bfloat16)
        self.fc2_b = fc2_b.astype(mx.bfloat16)
        self.act   = activation

    def __call__(self, x: "mx.array") -> "mx.array":
        h = x @ self.fc1_w.T + self.fc1_b
        h = self.act(h)
        return h @ self.fc2_w.T + self.fc2_b


class _CLIPLayer:
    """One transformer block: LN1 → Attn → residual → LN2 → MLP → residual."""

    def __init__(
        self,
        ln1_w: "mx.array", ln1_b: "mx.array",
        ln2_w: "mx.array", ln2_b: "mx.array",
        attn: _CLIPSelfAttn,
        mlp:  _CLIPMLP,
    ):
        import mlx.core as mx
        self.ln1_w = ln1_w.astype(mx.float32)
        self.ln1_b = ln1_b.astype(mx.float32)
        self.ln2_w = ln2_w.astype(mx.float32)
        self.ln2_b = ln2_b.astype(mx.float32)
        self.attn  = attn
        self.mlp   = mlp

    def __call__(self, x: "mx.array", mask: "mx.array") -> "mx.array":
        import mlx.core as mx
        # LN1 → Attention → residual
        h = _layer_norm(x, self.ln1_w, self.ln1_b)
        h = self.attn(h.astype(mx.bfloat16), mask)
        x = x + h.astype(x.dtype)
        # LN2 → MLP → residual
        h = _layer_norm(x, self.ln2_w, self.ln2_b)
        h = self.mlp(h.astype(mx.bfloat16))
        return x + h.astype(x.dtype)


def _layer_norm(x: "mx.array", w: "mx.array", b: "mx.array") -> "mx.array":
    """Manual layer norm (runs in float32 for stability)."""
    import mlx.core as mx
    x = x.astype(mx.float32)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var  = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / mx.sqrt(var + 1e-5)) * w + b


class MLXCLIPTextEncoder:
    """
    Full CLIP text encoder: embedding + N transformer layers + final LN.

    Pre-compiled from the ldm_patched CLIPTextModel state dict.  Runs entirely
    on Metal in bfloat16; falls back to torch for textual inversion tokens.
    """

    def __init__(
        self,
        tok_emb:    "mx.array",        # [V, d]
        pos_emb:    "mx.array",        # [77, d]
        layers:     list,              # list of _CLIPLayer
        ln_f_w:     "mx.array",        # [d]
        ln_f_b:     "mx.array",        # [d]
        tp_w:       "mx.array",        # [d, d]  text_projection (no bias)
        n_layers:   int,
        layer_idx:  int,               # intermediate output (negative = from end)
    ):
        import mlx.core as mx
        self.tok_emb   = tok_emb.astype(mx.bfloat16)
        self.pos_emb   = pos_emb.astype(mx.bfloat16)
        self.layers    = layers
        self.ln_f_w    = ln_f_w.astype(mx.float32)
        self.ln_f_b    = ln_f_b.astype(mx.float32)
        self.tp_w      = tp_w.astype(mx.bfloat16) if tp_w is not None else None
        self.n_layers  = n_layers
        # Resolve negative index (e.g. -2 with 12 layers → 10)
        self._inter_idx = n_layers + layer_idx if layer_idx < 0 else layer_idx

        # Build causal mask [T, T] once
        T = 77
        self._causal = mx.triu(mx.full((T, T), float("-inf")), k=1)
        mx.eval(self.tok_emb, self.pos_emb, self.ln_f_w, self.ln_f_b, self._causal)
        if self.tp_w is not None:
            mx.eval(self.tp_w)

    def encode(
        self,
        tokens: torch.Tensor,          # [B, 77] long
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        z       : [B, 77, d]  penultimate hidden state (NOT layer-normed)
        pooled  : [B, d] projected pooled output (None if no text_projection)
        """
        import mlx.core as mx

        B, T = tokens.shape

        # Token + position embeddings
        tok_ids  = mx.array(tokens.cpu().numpy().astype(np.int32))   # [B, T]
        x        = self.tok_emb[tok_ids] + self.pos_emb[None, :T]    # [B, T, d]

        intermediate: Optional["mx.array"] = None
        for i, layer in enumerate(self.layers):
            x = layer(x, self._causal[:T, :T])
            if i == self._inter_idx:
                intermediate = x          # capture before final LN, not normed

        # Final layer norm for the full output (used for pooled)
        x_normed = _layer_norm(x, self.ln_f_w, self.ln_f_b).astype(mx.bfloat16)

        # Pooled: hidden state at EOS position
        pooled = None
        if self.tp_w is not None:
            eos_col  = tokens.long().cpu() == _EOS_TOKEN_ID
            eos_pos  = eos_col.to(torch.int).argmax(dim=-1).numpy()    # [B]
            batch_idx = np.arange(B)
            # MLX does not yet support numpy-array fancy indices; gather via numpy.
            # x_normed is [B, T, d] — we pick one token per batch element.
            mx.eval(x_normed)
            x_np       = np.array(x_normed.astype(mx.float32))         # [B, T, d]
            raw_pooled = mx.array(x_np[batch_idx, eos_pos])            # [B, d]
            # tp_w is stored pre-transposed to [in, out] so we multiply directly:
            # pooled = eos_hidden @ tp_w  (matches SDClipModel convention:
            # clip_g.text_projection is an nn.Parameter used as x @ param)
            pooled = (raw_pooled.astype(mx.bfloat16) @ self.tp_w)      # [B, d]

        # Convert intermediate and pooled back to torch
        assert intermediate is not None, "intermediate not captured — check layer_idx"
        mx.eval(intermediate)
        z_np = np.array(intermediate.astype(mx.float32))
        z    = torch.from_numpy(z_np.copy()).float()

        pooled_t: Optional[torch.Tensor] = None
        if pooled is not None:
            mx.eval(pooled)
            pooled_t = torch.from_numpy(np.array(pooled.astype(mx.float32)).copy()).float()

        return z, pooled_t


# ── weight loading ──────────────────────────────────────────────────────────────

def _t2m_f32(t: torch.Tensor) -> "mx.array":
    """torch tensor → MLX float32 array (bfloat16 cast happens at use site)."""
    import mlx.core as mx
    return mx.array(t.detach().float().cpu().numpy())


def _load_layer(sd: dict, prefix: str, d_model: int, n_heads: int,
                d_ff: int, act) -> "_CLIPLayer":
    """Load one CLIP transformer layer from the state dict."""
    p = prefix  # e.g.  "text_model.encoder.layers.0"
    attn = _CLIPSelfAttn(
        d_model, n_heads,
        _t2m_f32(sd[f"{p}.self_attn.q_proj.weight"]),
        _t2m_f32(sd[f"{p}.self_attn.q_proj.bias"]),
        _t2m_f32(sd[f"{p}.self_attn.k_proj.weight"]),
        _t2m_f32(sd[f"{p}.self_attn.k_proj.bias"]),
        _t2m_f32(sd[f"{p}.self_attn.v_proj.weight"]),
        _t2m_f32(sd[f"{p}.self_attn.v_proj.bias"]),
        _t2m_f32(sd[f"{p}.self_attn.out_proj.weight"]),
        _t2m_f32(sd[f"{p}.self_attn.out_proj.bias"]),
    )
    mlp = _CLIPMLP(
        _t2m_f32(sd[f"{p}.mlp.fc1.weight"]),
        _t2m_f32(sd[f"{p}.mlp.fc1.bias"]),
        _t2m_f32(sd[f"{p}.mlp.fc2.weight"]),
        _t2m_f32(sd[f"{p}.mlp.fc2.bias"]),
        act,
    )
    return _CLIPLayer(
        _t2m_f32(sd[f"{p}.layer_norm1.weight"]),
        _t2m_f32(sd[f"{p}.layer_norm1.bias"]),
        _t2m_f32(sd[f"{p}.layer_norm2.weight"]),
        _t2m_f32(sd[f"{p}.layer_norm2.bias"]),
        attn, mlp,
    )


def build_clip_encoder(
    transformer,                          # HuggingFace CLIPTextModel instance
    layer_idx: int = -2,
    has_text_projection: bool = True,
    activation = None,
    text_projection_param: Optional[torch.Tensor] = None,
) -> "MLXCLIPTextEncoder":
    """
    Build an MLXCLIPTextEncoder from an ldm_patched CLIPTextModel.

    Parameters
    ----------
    transformer            : HuggingFace CLIPTextModel (stored as SDClipModel.transformer)
    layer_idx              : which layer's output to use as 'z' (default -2 = penultimate)
    has_text_projection    : whether to compute pooled output
    activation             : callable — defaults to quick_gelu
    text_projection_param  : the raw nn.Parameter from SDClipModel (``clip_g.text_projection``).
                             Shape [d, d].  Used directly as: pooled = eos_hidden @ tp_w.
                             If None, falls back to ``"text_projection.weight"`` in the
                             transformer state dict (Linear weight convention → transposed).
    """
    import mlx.core as mx

    if activation is None:
        activation = _quick_gelu

    sd = transformer.state_dict()

    d_model   = sd["text_model.embeddings.token_embedding.weight"].shape[1]
    n_layers  = sum(1 for k in sd if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    d_ff_key  = "text_model.encoder.layers.0.mlp.fc1.weight"
    d_ff      = sd[d_ff_key].shape[0]
    n_heads   = d_model // 64     # all CLIP variants use 64-d heads

    layers = [
        _load_layer(sd, f"text_model.encoder.layers.{i}", d_model, n_heads, d_ff, activation)
        for i in range(n_layers)
    ]

    # ── text_projection (for CLIP-G pooled output) ────────────────────────────
    # The weight is stored as a raw [d, d] parameter on SDClipModel (NOT inside
    # the transformer).  It is used as:  pooled = eos_hidden @ text_projection
    # (right-multiply, no transpose).
    #
    # Fallback: if "text_projection.weight" exists in the transformer state_dict
    # (Linear weight convention [out, in]), we transpose it so the same
    # "pooled = eos_hidden @ tp_w" formula still holds.
    tp_w = None
    if has_text_projection:
        if text_projection_param is not None:
            # Preferred path: raw nn.Parameter [d, d], used as-is
            tp_w = _t2m_f32(text_projection_param)
        elif "text_projection.weight" in sd:
            # Fallback: Linear weight [out, in] → transpose to [in, out]
            tp_w = _t2m_f32(sd["text_projection.weight"].t().contiguous())

    log.debug(
        "[MLX CLIP] Built encoder: d=%d heads=%d layers=%d ff=%d",
        d_model, n_heads, n_layers, d_ff,
    )

    return MLXCLIPTextEncoder(
        tok_emb   = _t2m_f32(sd["text_model.embeddings.token_embedding.weight"]),
        pos_emb   = _t2m_f32(sd["text_model.embeddings.position_embedding.weight"]),
        layers    = layers,
        ln_f_w    = _t2m_f32(sd["text_model.final_layer_norm.weight"]),
        ln_f_b    = _t2m_f32(sd["text_model.final_layer_norm.bias"]),
        tp_w      = tp_w,
        n_layers  = n_layers,
        layer_idx = layer_idx,
    )


# ── hook helpers ────────────────────────────────────────────────────────────────

def _make_clip_l_hook(
    mlx_enc: "MLXCLIPTextEncoder",
    orig_fn,
    clip_l_model=None,
):
    """Return a drop-in replacement for CLIP_SD_XL_L.encode_with_transformers.

    When *clip_l_model* is supplied the hook tracks a cheap weight fingerprint
    and rebuilds the MLX encoder when the PyTorch CLIP-L weights change (e.g.
    after a CLIP LoRA is applied via ``patch_model()``).  The rebuild happens
    at most **once per prompt** — negligible compared to the sampler cost.
    """
    from mlx_pipeline.lora import clip_weight_fingerprint

    # Mutable state shared across calls for this hook instance
    _state = {
        "enc": mlx_enc,
        "fp":  clip_weight_fingerprint(clip_l_model.transformer) if clip_l_model is not None else (),
    }

    def _hook(tokens: torch.Tensor) -> torch.Tensor:
        # Textual inversion fallback: any token id >= standard vocab
        if tokens.max().item() >= _CLIP_VOCAB_SIZE:
            log.debug("[MLX CLIP-L] Textual inversion detected, falling back to torch")
            return orig_fn(tokens)

        # LoRA-change detection: compare weight fingerprint
        if clip_l_model is not None:
            current_fp = clip_weight_fingerprint(clip_l_model.transformer)
            if current_fp != _state["fp"]:
                log.info("[MLX CLIP-L] Weight change detected (LoRA?), rebuilding …")
                try:
                    _state["enc"] = build_clip_encoder(
                        clip_l_model.transformer,
                        layer_idx          = clip_l_model.layer_idx if clip_l_model.layer_idx is not None else -2,
                        has_text_projection= False,
                        activation         = _quick_gelu,
                    )
                    _state["fp"] = current_fp
                except Exception as _rb_exc:
                    log.warning("[MLX CLIP-L] Rebuild failed (%s), using torch fallback", _rb_exc)
                    return orig_fn(tokens)

        z, _ = _state["enc"].encode(tokens)
        # MLX always outputs CPU tensors; move to the same device as the
        # input tokens so downstream ops (emphasis multiply, etc.) don't
        # hit cross-device errors on MPS.
        return z.to(tokens.device)

    return _hook


def _make_clip_g_hook(
    mlx_enc: "MLXCLIPTextEncoder",
    orig_fn,
    clip_g_model=None,
):
    """Return a drop-in replacement for CLIP_SD_XL_G.encode_with_transformers.

    When *clip_g_model* is supplied the hook tracks a cheap weight fingerprint
    (including ``text_projection``) and rebuilds the MLX encoder on change.
    """
    from mlx_pipeline.lora import clip_weight_fingerprint

    tp_param = getattr(clip_g_model, "text_projection", None) if clip_g_model is not None else None
    tp_data  = tp_param.data if tp_param is not None else None

    _state = {
        "enc": mlx_enc,
        "fp":  clip_weight_fingerprint(clip_g_model.transformer, tp_data) if clip_g_model is not None else (),
    }

    def _hook(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.max().item() >= _CLIP_VOCAB_SIZE:
            log.debug("[MLX CLIP-G] Textual inversion detected, falling back to torch")
            return orig_fn(tokens)

        # LoRA-change detection
        if clip_g_model is not None:
            _tp_param = getattr(clip_g_model, "text_projection", None)
            _tp_data  = _tp_param.data if _tp_param is not None else None
            current_fp = clip_weight_fingerprint(clip_g_model.transformer, _tp_data)
            if current_fp != _state["fp"]:
                log.info("[MLX CLIP-G] Weight change detected (LoRA?), rebuilding …")
                try:
                    _state["enc"] = build_clip_encoder(
                        clip_g_model.transformer,
                        layer_idx              = clip_g_model.layer_idx if clip_g_model.layer_idx is not None else -2,
                        has_text_projection    = _tp_data is not None,
                        activation             = _gelu,
                        text_projection_param  = _tp_data,
                    )
                    _state["fp"] = current_fp
                except Exception as _rb_exc:
                    log.warning("[MLX CLIP-G] Rebuild failed (%s), using torch fallback", _rb_exc)
                    return orig_fn(tokens)

        z, pooled = _state["enc"].encode(tokens)
        z = z.to(tokens.device)
        if pooled is not None:
            z.pooled = pooled.to(tokens.device)
        return z

    return _hook


def install_clip_hooks(sd_model, forge_objects) -> bool:
    """
    Install MLX hooks on the CLIP-L and CLIP-G encode_with_transformers methods.

    Parameters
    ----------
    sd_model      : SDXL sd_model
    forge_objects : ForgeObjects (has .clip with .cond_stage_model)

    Returns
    -------
    True on success, False if prerequisites not found.
    """
    try:
        clip_model = forge_objects.clip.cond_stage_model
        clip_l = clip_model.clip_l      # SDClipModel
        clip_g = clip_model.clip_g      # SDXLClipG

        # Build MLX encoders from the ldm_patched transformer weights
        mlx_clip_l = build_clip_encoder(
            clip_l.transformer,
            layer_idx          = clip_l.layer_idx if clip_l.layer_idx is not None else -2,
            has_text_projection= False,   # CLIP-L pooled not used in SDXL
            activation         = _quick_gelu,
        )
        # text_projection is an nn.Parameter on SDClipModel itself, NOT inside
        # the transformer.  Pass it directly so build_clip_encoder doesn't try
        # to find "text_projection.weight" in the HuggingFace CLIPTextModel
        # state dict (where it does not exist).
        tp_param = getattr(clip_g, "text_projection", None)
        tp_data  = tp_param.data if tp_param is not None else None
        mlx_clip_g = build_clip_encoder(
            clip_g.transformer,
            layer_idx              = clip_g.layer_idx if clip_g.layer_idx is not None else -2,
            has_text_projection    = tp_data is not None,
            activation             = _gelu,
            text_projection_param  = tp_data,
        )

        # Locate the forge_clip instances in the conditioner
        # They are the embedder objects in sd_model's conditioner
        from modules_forge import forge_clip as fc
        conditioner = getattr(sd_model, "conditioner", None) or getattr(sd_model.cond_stage_model, "conditioner", None)
        if conditioner is None and hasattr(sd_model, "cond_stage_model"):
            conditioner = sd_model.cond_stage_model

        # Patch all CLIP_SD_XL_L and CLIP_SD_XL_G instances
        n_patched = 0
        def _patch_embedder(obj):
            nonlocal n_patched
            if isinstance(obj, fc.CLIP_SD_XL_L):
                orig = obj.encode_with_transformers.__func__ if hasattr(obj.encode_with_transformers, '__func__') else obj.encode_with_transformers
                # Pass clip_l_model so the hook can detect LoRA weight changes
                # and rebuild the MLX encoder automatically.
                obj.encode_with_transformers = _make_clip_l_hook(
                    mlx_clip_l,
                    lambda t: orig(obj, t),
                    clip_l_model=clip_l,
                )
                n_patched += 1
                log.debug("[MLX CLIP] Patched CLIP_SD_XL_L instance (LoRA-aware)")
            elif isinstance(obj, fc.CLIP_SD_XL_G):
                orig = obj.encode_with_transformers.__func__ if hasattr(obj.encode_with_transformers, '__func__') else obj.encode_with_transformers
                # Pass clip_g_model so the hook can detect LoRA weight changes
                # (including text_projection LoRA) and rebuild automatically.
                obj.encode_with_transformers = _make_clip_g_hook(
                    mlx_clip_g,
                    lambda t: orig(obj, t),
                    clip_g_model=clip_g,
                )
                n_patched += 1
                log.debug("[MLX CLIP] Patched CLIP_SD_XL_G instance (LoRA-aware)")

        # Walk conditioner embedders
        if hasattr(conditioner, 'embedders'):
            for emb in conditioner.embedders:
                _patch_embedder(emb)
        # Also check the top-level cond_stage_model
        if hasattr(sd_model, 'cond_stage_model'):
            _patch_embedder(sd_model.cond_stage_model)

        if n_patched > 0:
            log.info(
                "[MLX CLIP] Hooked %d CLIP encoder(s) — Metal bfloat16 text encoding active",
                n_patched,
            )
            # Store references to prevent GC
            sd_model._mlx_clip_l = mlx_clip_l
            sd_model._mlx_clip_g = mlx_clip_g
            return True
        else:
            log.debug("[MLX CLIP] No forge_clip CLIP_SD_XL_L/G instances found to patch")
            return False

    except Exception as e:
        log.warning("[MLX CLIP] Hook installation failed: %s", e, exc_info=True)
        return False
