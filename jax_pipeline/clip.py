"""
jax_pipeline/clip.py — pure-functional JAX CLIP text encoders for SDXL.

Implements CLIP-L (768d, 12 layers, quick_gelu) and CLIP-G (1280d, 32 layers,
standard gelu), mirroring mlx_pipeline/clip.py's architecture and hook
mechanism but as jax.jit-compiled pure functions over a flat params dict
(consistent with jax_pipeline/unet.py's style), rather than stateful class
instances.

Architecture
------------
Both encoders use the ldm_patched CLIPTextModel weight format (already
HF-compatible, so no key renaming is needed — unlike the UNet):

  text_model.embeddings.token_embedding.weight    [V, d]
  text_model.embeddings.position_embedding.weight [77, d]
  text_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
  text_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
  text_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}
  text_model.final_layer_norm.{weight,bias}
  text_projection.weight  [d, d]  (for pooled output)

None of these weights are 4-D conv tensors, so weight conversion here is a
straight dtype cast — no OIHW->HWIO transpose needed (contrast with unet.py
and vae.py).

Integration
-----------
Hooks ``encode_with_transformers`` on forge_clip.CLIP_SD_XL_L and
CLIP_SD_XL_G instances so CLIP text encoding runs as a jax.jit-compiled
function instead of PyTorch.

Output contract:
  CLIP-L: torch.Tensor [B, 77, 768]  — penultimate hidden state, NOT normed
  CLIP-G: torch.Tensor [B, 77, 1280] — same, with `.pooled` = [B, 1280]

Falls back to the original torch implementation when textual inversion is
detected (any token id >= 49408) or on any runtime error.
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)

_CLIP_VOCAB_SIZE = 49408  # Standard CLIP vocab; tokens above this = textual inversion
_EOS_TOKEN_ID = 49407     # <|endoftext|>
_NORM_EPS = 1e-5


# -- activation functions ------------------------------------------------------

def _quick_gelu(x):
    """CLIP-L uses quick GELU: x * sigmoid(1.702 * x)."""
    return x * jax.nn.sigmoid(1.702 * x.astype(jnp.float32)).astype(x.dtype)


def _gelu(x):
    """CLIP-G uses standard (exact, erf-based) GELU."""
    return jax.nn.gelu(x, approximate=False)


# -- primitive layers ------------------------------------------------------------

def linear(params: Dict[str, "jnp.ndarray"], prefix: str, x):
    w = params[f"{prefix}.weight"]  # [out, in]
    b = params.get(f"{prefix}.bias")
    out = x @ w.T
    if b is not None:
        out = out + b
    return out


def layer_norm(params, prefix, x, eps: float = _NORM_EPS):
    w = params[f"{prefix}.weight"]
    b = params[f"{prefix}.bias"]
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - mean) / jnp.sqrt(var + eps)
    out = normed * w.astype(jnp.float32) + b.astype(jnp.float32)
    return out.astype(x.dtype)


def self_attn(params, prefix, x, mask, num_heads: int):
    """x: [B, T, D]; mask: [T, T] additive causal mask -> [B, T, D]."""
    B, T, D = x.shape
    dh = D // num_heads

    q = linear(params, f"{prefix}.q_proj", x).reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)
    k = linear(params, f"{prefix}.k_proj", x).reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)
    v = linear(params, f"{prefix}.v_proj", x).reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)

    scale = dh ** -0.5
    attn = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    attn = attn + mask[None, None, :, :]
    attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return linear(params, f"{prefix}.out_proj", out)


def mlp(params, prefix, x, activation):
    h = linear(params, f"{prefix}.fc1", x)
    h = activation(h)
    return linear(params, f"{prefix}.fc2", h)


def clip_layer(params, prefix, x, mask, num_heads: int, activation):
    h = layer_norm(params, f"{prefix}.layer_norm1", x)
    h = self_attn(params, f"{prefix}.self_attn", h, mask, num_heads)
    x = x + h

    h = layer_norm(params, f"{prefix}.layer_norm2", x)
    h = mlp(params, f"{prefix}.mlp", h, activation)
    x = x + h

    return x


# -- top-level encode ------------------------------------------------------------

def clip_encode(
    params: Dict[str, "jnp.ndarray"],
    tokens,
    num_layers: int,
    num_heads: int,
    layer_idx: int,
    activation,
    has_text_projection: bool,
):
    """tokens: [B, 77] int32 -> (intermediate [B,77,D], pooled [B,D] or None)."""
    B, T = tokens.shape

    tok_emb = params["text_model.embeddings.token_embedding.weight"]
    pos_emb = params["text_model.embeddings.position_embedding.weight"]
    x = tok_emb[tokens] + pos_emb[None, :T]

    mask = jnp.triu(jnp.full((T, T), -jnp.inf, dtype=jnp.float32), k=1)

    inter_idx = num_layers + layer_idx if layer_idx < 0 else layer_idx
    intermediate = None
    for i in range(num_layers):
        x = clip_layer(params, f"text_model.encoder.layers.{i}", x, mask, num_heads, activation)
        if i == inter_idx:
            intermediate = x

    pooled = None
    if has_text_projection:
        x_normed = layer_norm(params, "text_model.final_layer_norm", x)
        eos_mask = tokens == _EOS_TOKEN_ID
        eos_pos = jnp.argmax(eos_mask, axis=-1)  # [B]
        raw_pooled = jnp.take_along_axis(
            x_normed.astype(jnp.float32), eos_pos[:, None, None], axis=1
        )[:, 0, :]  # [B, D]
        tp_w = params["text_projection.weight"]  # [d_in, d_out], used as pooled = eos_hidden @ tp_w
        pooled = raw_pooled @ tp_w.astype(jnp.float32)

    return intermediate, pooled


# -- weight loading ----------------------------------------------------------

def _tensor_to_jax(tensor: "torch.Tensor", dtype, sharding=None):
    """``sharding``, when given, is passed as ``jnp.asarray``'s ``device=``
    argument so the array lands directly where the caller wants it (e.g.
    pinned host memory, on a phase-managed card) instead of defaulting to
    GPU device memory and needing a second device_put to relocate it
    later. See ``jax_pipeline.convert._tensor_to_jax``'s docstring for
    the same reasoning applied to the UNet -- this is the CLIP-side
    counterpart of that fix.
    """
    arr = tensor.detach().float().cpu().numpy()
    return jnp.asarray(arr, dtype=dtype, device=sharding)


def load_clip_params(transformer, dtype=None, sharding=None) -> Dict[str, "jnp.ndarray"]:
    if dtype is None:
        dtype = jnp.bfloat16
    sd = transformer.state_dict()
    return {k: _tensor_to_jax(v, dtype, sharding=sharding) for k, v in sd.items()}


class JAXCLIPTextEncoder:
    """Holds converted params + a jax.jit-compiled encode function for one
    CLIP text tower. Constructed once at model load / LoRA-rebuild time;
    ``.encode()`` is called once per prompt (not per sampling step).
    """

    def __init__(
        self,
        params: Dict[str, "jnp.ndarray"],
        num_layers: int,
        num_heads: int,
        layer_idx: int,
        activation,
        has_text_projection: bool,
    ):
        self.params = params
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.has_text_projection = has_text_projection

        fn = functools.partial(
            clip_encode,
            num_layers=num_layers, num_heads=num_heads, layer_idx=layer_idx,
            activation=activation, has_text_projection=has_text_projection,
        )
        self._encode_jit = jax.jit(fn)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """tokens: [B, 77] torch long -> (z [B,77,D] float32, pooled [B,D] float32 or None)."""
        tok_np = tokens.detach().cpu().numpy().astype(np.int32)
        tok_jax = jnp.asarray(tok_np)

        intermediate, pooled = self._encode_jit(self.params, tok_jax)

        assert intermediate is not None, "intermediate not captured - check layer_idx"
        z_np = np.asarray(jnp.asarray(intermediate, dtype=jnp.float32))
        z = torch.from_numpy(z_np.copy()).float()

        pooled_t: Optional[torch.Tensor] = None
        if pooled is not None:
            pooled_np = np.asarray(jnp.asarray(pooled, dtype=jnp.float32))
            pooled_t = torch.from_numpy(pooled_np.copy()).float()

        return z, pooled_t


def build_clip_text_encoder(
    transformer,
    layer_idx: int = -2,
    has_text_projection: bool = True,
    activation=None,
    text_projection_param: Optional["torch.Tensor"] = None,
    dtype=None,
    sharding=None,
) -> "JAXCLIPTextEncoder":
    """Build a JAXCLIPTextEncoder from an ldm_patched CLIPTextModel.

    Parameters mirror mlx_pipeline.clip.build_clip_encoder exactly,
    including the text_projection resolution priority:
      1. ``text_projection_param`` (raw nn.Parameter [d,d] from SDClipModel,
         e.g. ``clip_g.text_projection``) - used as-is, no transpose.
      2. ``"text_projection.weight"`` in the transformer state dict (Linear
         convention [out,in]) - transposed to [in,out] so the same
         ``pooled = eos_hidden @ tp_w`` formula holds either way.

    ``sharding``: forwarded to every weight conversion (see
    ``_tensor_to_jax``'s docstring) — ``install_clip_hooks`` passes a
    pinned-host sharding here on a phase-managed card so CLIP-L/G never
    touch device memory during conversion, only via
    ``PhaseManager.activate("clip")`` on demand.
    """
    if dtype is None:
        dtype = jnp.bfloat16
    if activation is None:
        activation = _quick_gelu

    sd = transformer.state_dict()
    d_model = sd["text_model.embeddings.token_embedding.weight"].shape[1]
    n_layers = sum(
        1 for k in sd
        if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight")
    )
    n_heads = d_model // 64  # all CLIP variants use 64-d heads

    params = load_clip_params(transformer, dtype=dtype, sharding=sharding)

    if has_text_projection:
        if text_projection_param is not None:
            params["text_projection.weight"] = _tensor_to_jax(text_projection_param, dtype, sharding=sharding)
        elif "text_projection.weight" in params:
            params["text_projection.weight"] = params["text_projection.weight"].T

    log.debug(
        "[JAX CLIP] Built encoder: d=%d heads=%d layers=%d", d_model, n_heads, n_layers,
    )

    return JAXCLIPTextEncoder(
        params, num_layers=n_layers, num_heads=n_heads, layer_idx=layer_idx,
        activation=activation, has_text_projection=has_text_projection,
    )


# -- hook helpers ----------------------------------------------------------------

def _make_clip_l_hook(clip_state: dict, orig_fn, clip_l_model=None, clip_patcher=None, phase_manager=None):
    """Return a drop-in replacement for CLIP_SD_XL_L.encode_with_transformers.

    ``clip_state`` is the SHARED mutable dict created once in
    ``install_clip_hooks`` (holds both "l" and "g" encoders) — using it
    instead of a private closure-local dict means a LoRA-triggered rebuild
    here (which replaces ``clip_state["l"]`` with a brand new
    ``JAXCLIPTextEncoder`` instance) stays visible to the phase manager's
    registered ``get_params``/``set_params`` for the "clip" component,
    which read/write through this same dict.

    ``clip_patcher`` (``forge_objects.clip.patcher``) is used both for the
    cheap ``patches_uuid``-based change-detection signal (see
    ``jax_pipeline.lora.clip_lora_sig``) and to reload the torch CLIP back
    onto GPU on demand — see ``install_clip_hooks``'s docstring for why
    the torch copy is evicted after JAX has its own.
    """
    from jax_pipeline.lora import clip_lora_sig
    from jax_pipeline.host_offload import ensure_torch_model_on_gpu, evict_torch_model_from_gpu

    clip_state.setdefault("l_sig", clip_lora_sig(clip_patcher) if clip_patcher is not None else "")

    def _hook(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.max().item() >= _CLIP_VOCAB_SIZE:
            log.debug("[JAX CLIP-L] Textual inversion detected, falling back to torch")
            if phase_manager is not None:
                phase_manager.escape_to_pytorch()
            if clip_patcher is not None:
                ensure_torch_model_on_gpu(clip_patcher)
            return orig_fn(tokens)

        if clip_l_model is not None and clip_patcher is not None:
            current_sig = clip_lora_sig(clip_patcher)
            if current_sig != clip_state["l_sig"]:
                log.info("[JAX CLIP-L] Weight change detected (LoRA?), rebuilding...")
                try:
                    ensure_torch_model_on_gpu(clip_patcher)
                    clip_state["l"] = build_clip_text_encoder(
                        clip_l_model.transformer,
                        layer_idx=clip_l_model.layer_idx if clip_l_model.layer_idx is not None else -2,
                        has_text_projection=False,
                        activation=_quick_gelu,
                    )
                    clip_state["l_sig"] = current_sig
                    if phase_manager is not None and phase_manager.enabled:
                        evict_torch_model_from_gpu(clip_patcher, label="CLIP")
                except Exception as rb_exc:
                    log.warning("[JAX CLIP-L] Rebuild failed (%s), using torch fallback", rb_exc)
                    if phase_manager is not None:
                        phase_manager.escape_to_pytorch()
                    ensure_torch_model_on_gpu(clip_patcher)
                    return orig_fn(tokens)

        if phase_manager is not None:
            phase_manager.activate("clip")

        try:
            z, _ = clip_state["l"].encode(tokens)
        except Exception as exc:
            from jax_pipeline.host_offload import _is_jax_oom_exception, handle_jax_oom
            if _is_jax_oom_exception(exc):
                handle_jax_oom(phase_manager, "CLIP-L encode", exc)  # always raises
            raise
        return z.to(tokens.device)

    return _hook


def _make_clip_g_hook(clip_state: dict, orig_fn, clip_g_model=None, clip_patcher=None, phase_manager=None):
    """Return a drop-in replacement for CLIP_SD_XL_G.encode_with_transformers.

    See ``_make_clip_l_hook`` for why ``clip_state`` is a shared dict and
    what ``clip_patcher`` is used for.
    """
    from jax_pipeline.lora import clip_lora_sig
    from jax_pipeline.host_offload import ensure_torch_model_on_gpu, evict_torch_model_from_gpu

    clip_state.setdefault("g_sig", clip_lora_sig(clip_patcher) if clip_patcher is not None else "")

    def _hook(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.max().item() >= _CLIP_VOCAB_SIZE:
            log.debug("[JAX CLIP-G] Textual inversion detected, falling back to torch")
            if phase_manager is not None:
                phase_manager.escape_to_pytorch()
            if clip_patcher is not None:
                ensure_torch_model_on_gpu(clip_patcher)
            return orig_fn(tokens)

        if clip_g_model is not None and clip_patcher is not None:
            current_sig = clip_lora_sig(clip_patcher)
            if current_sig != clip_state["g_sig"]:
                log.info("[JAX CLIP-G] Weight change detected (LoRA?), rebuilding...")
                try:
                    ensure_torch_model_on_gpu(clip_patcher)
                    _tp_param = getattr(clip_g_model, "text_projection", None)
                    _tp_data = _tp_param.data if _tp_param is not None else None
                    clip_state["g"] = build_clip_text_encoder(
                        clip_g_model.transformer,
                        layer_idx=clip_g_model.layer_idx if clip_g_model.layer_idx is not None else -2,
                        has_text_projection=_tp_data is not None,
                        activation=_gelu,
                        text_projection_param=_tp_data,
                    )
                    clip_state["g_sig"] = current_sig
                    if phase_manager is not None and phase_manager.enabled:
                        evict_torch_model_from_gpu(clip_patcher, label="CLIP")
                except Exception as rb_exc:
                    log.warning("[JAX CLIP-G] Rebuild failed (%s), using torch fallback", rb_exc)
                    if phase_manager is not None:
                        phase_manager.escape_to_pytorch()
                    ensure_torch_model_on_gpu(clip_patcher)
                    return orig_fn(tokens)

        if phase_manager is not None:
            phase_manager.activate("clip")

        try:
            z, pooled = clip_state["g"].encode(tokens)
        except Exception as exc:
            from jax_pipeline.host_offload import _is_jax_oom_exception, handle_jax_oom
            if _is_jax_oom_exception(exc):
                handle_jax_oom(phase_manager, "CLIP-G encode", exc)  # always raises
            raise
        z = z.to(tokens.device)
        if pooled is not None:
            z.pooled = pooled.to(tokens.device)
        return z

    return _hook


def install_clip_hooks(sd_model, forge_objects, phase_manager=None) -> bool:
    """Install JAX hooks on the CLIP-L and CLIP-G encode_with_transformers methods.

    Parameters
    ----------
    sd_model      : SDXL sd_model
    forge_objects : ForgeObjects (has .clip with .cond_stage_model)
    phase_manager : optional jax_pipeline.host_offload.PhaseManager shared
                    with the UNet/VAE hooks — when its ``enabled`` is True,
                    CLIP-L+G params are treated as ONE combined "clip"
                    component that gets moved to device together right
                    before encoding and offloaded to pinned host memory
                    whenever a different phase (unet/vae) activates.

    Returns
    -------
    True on success, False if prerequisites not found.
    """
    try:
        clip_model = forge_objects.clip.cond_stage_model
        clip_l = clip_model.clip_l  # SDClipModel
        clip_g = clip_model.clip_g  # SDXLClipG

        # On a phase-managed (VRAM-constrained) card, convert CLIP-L/G
        # weights DIRECTLY to pinned host instead of the default GPU
        # placement -- phase_manager.register("clip", ...) below would
        # move them to pinned host anyway, but doing it AT conversion
        # time avoids a transient extra device allocation for the whole
        # ~1-2GB combined CLIP-L+G footprint stacking on top of whatever
        # else (notably the UNet's own conversion) is happening around
        # the same point in activation. Same fix as
        # jax_pipeline.convert.load_weights_from_ldm's sharding param —
        # see its docstring for the real-hardware OOM this class of bug
        # caused.
        clip_sharding = None
        if phase_manager is not None and phase_manager.enabled:
            import jax
            clip_sharding = jax.sharding.SingleDeviceSharding(phase_manager.device, memory_kind="pinned_host")

        jax_clip_l = build_clip_text_encoder(
            clip_l.transformer,
            layer_idx=clip_l.layer_idx if clip_l.layer_idx is not None else -2,
            has_text_projection=False,  # CLIP-L pooled not used in SDXL
            activation=_quick_gelu,
            sharding=clip_sharding,
        )

        tp_param = getattr(clip_g, "text_projection", None)
        tp_data = tp_param.data if tp_param is not None else None
        jax_clip_g = build_clip_text_encoder(
            clip_g.transformer,
            layer_idx=clip_g.layer_idx if clip_g.layer_idx is not None else -2,
            has_text_projection=tp_data is not None,
            activation=_gelu,
            text_projection_param=tp_data,
            sharding=clip_sharding,
        )

        # Shared mutable holder for both encoders — see _make_clip_l_hook's
        # docstring for why this (rather than two private per-hook dicts)
        # is what the phase manager registers against.
        clip_state = {"l": jax_clip_l, "g": jax_clip_g}
        clip_patcher = getattr(forge_objects.clip, "patcher", None)

        # JAX now has its own independent copy of these weights (just
        # built above) — the torch-side CLIP the checkpoint loader force-
        # loaded onto GPU is redundant until the next LoRA change or a
        # torch-fallback path (textual inversion, rebuild failure) needs
        # it again, both of which reload it on demand via
        # ensure_torch_model_on_gpu (see _make_clip_l_hook/_make_clip_g_hook).
        # Unlike the UNet, this is a ONE-TIME saving, not per-generation —
        # CLIP's torch copy is only ever force-loaded once, at checkpoint
        # load, not re-loaded every generation the way sampling_prepare()
        # does for the UNet.
        if phase_manager is not None and phase_manager.enabled and clip_patcher is not None:
            from jax_pipeline.host_offload import evict_torch_model_from_gpu
            evict_torch_model_from_gpu(clip_patcher, label="CLIP")

        if phase_manager is not None:
            def _get_clip_params():
                return {"l": clip_state["l"].params, "g": clip_state["g"].params}

            def _set_clip_params(new):
                clip_state["l"].params = new["l"]
                clip_state["g"].params = new["g"]

            phase_manager.register("clip", get_params=_get_clip_params, set_params=_set_clip_params)

        from modules_forge import forge_clip as fc
        conditioner = getattr(sd_model, "conditioner", None) or getattr(sd_model.cond_stage_model, "conditioner", None)
        if conditioner is None and hasattr(sd_model, "cond_stage_model"):
            conditioner = sd_model.cond_stage_model

        n_patched = 0

        def _patch_embedder(obj):
            nonlocal n_patched
            if isinstance(obj, fc.CLIP_SD_XL_L):
                orig = obj.encode_with_transformers.__func__ if hasattr(obj.encode_with_transformers, "__func__") else obj.encode_with_transformers
                obj.encode_with_transformers = _make_clip_l_hook(
                    clip_state, lambda t: orig(obj, t), clip_l_model=clip_l,
                    clip_patcher=clip_patcher, phase_manager=phase_manager,
                )
                n_patched += 1
                log.debug("[JAX CLIP] Patched CLIP_SD_XL_L instance (LoRA-aware)")
            elif isinstance(obj, fc.CLIP_SD_XL_G):
                orig = obj.encode_with_transformers.__func__ if hasattr(obj.encode_with_transformers, "__func__") else obj.encode_with_transformers
                obj.encode_with_transformers = _make_clip_g_hook(
                    clip_state, lambda t: orig(obj, t), clip_g_model=clip_g,
                    clip_patcher=clip_patcher, phase_manager=phase_manager,
                )
                n_patched += 1
                log.debug("[JAX CLIP] Patched CLIP_SD_XL_G instance (LoRA-aware)")

        if hasattr(conditioner, "embedders"):
            for emb in conditioner.embedders:
                _patch_embedder(emb)
        if hasattr(sd_model, "cond_stage_model"):
            _patch_embedder(sd_model.cond_stage_model)

        if n_patched > 0:
            log.info(
                "[JAX CLIP] Hooked %d CLIP encoder(s) - jax.jit text encoding active", n_patched,
            )
            sd_model._jax_clip_state = clip_state
            return True
        else:
            log.debug("[JAX CLIP] No forge_clip CLIP_SD_XL_L/G instances found to patch")
            return False

    except Exception as e:
        log.warning("[JAX CLIP] Hook installation failed: %s", e, exc_info=True)
        return False
