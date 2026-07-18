"""
jax_pipeline/lora.py — LoRA / DoRA detection and weight-reload for the JAX
UNet and CLIP.

Mirrors mlx_pipeline/lora.py's UNet-side mechanism:

* ``unet_lora_sig(unet_patcher)`` / ``clip_lora_sig(clip_patcher)`` read
  ``<patcher>.patches_uuid`` — a UUID4 that ldm_patched regenerates
  atomically on every LoRA/DoRA add/remove/strength change, on ANY
  ``ModelPatcher`` (UNet and CLIP alike — this is generic, not UNet-
  specific) — as a cheap change-detection token. Cheap and device-
  independent: it's a plain Python attribute read, so it works correctly
  even while the torch-side copy is evicted to CPU (see
  ``jax_pipeline.host_offload.evict_torch_model_from_gpu``) — unlike
  reading live weight values, which would silently see stale/reverted
  data once evicted.
* ``reload_jax_unet_weights(pipeline, unet_patcher)`` re-reads ALL
  diffusion model weights from the currently-patched PyTorch model (LoRA /
  DoRA already merged in by ``patch_model()``/``patch_weight_to_device()``)
  and replaces ``pipeline._params`` in one shot. No incremental diffing —
  every patch type ``ldm_patched.modules.lora.calculate_weight`` supports
  (LoRA, DoRA, LoCon, LoKr, LoHa, IA3, full-diff, ...) is handled
  automatically because we simply read the already-merged PyTorch weights.

Cost: one full weight reconversion per generation where the LoRA config
changed; zero per-step overhead afterward (JAXSDXLPipeline caches
``_lora_sig`` in ``pipeline.py`` and only reconverts on mismatch).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def unet_lora_sig(unet_patcher) -> str:
    """Return ``str(patches_uuid)`` — the canonical LoRA-change token."""
    return str(getattr(unet_patcher, "patches_uuid", ""))


def clip_lora_sig(clip_patcher) -> str:
    """Return ``str(patches_uuid)`` for the CLIP ``ModelPatcher`` —
    ``forge_objects.clip.patcher``, which wraps BOTH CLIP-L and CLIP-G as
    one combined ``cond_stage_model``, so a single shared signal correctly
    reflects a LoRA change touching either (or both) text towers.
    """
    return str(getattr(clip_patcher, "patches_uuid", ""))


def reload_jax_unet_weights(pipeline, unet_patcher) -> None:
    """Reconvert PyTorch UNet weights (LoRA/DoRA already merged) into JAX
    and replace ``pipeline._params`` in place.

    When ``pipeline._phase_manager`` is enabled, ``pipeline._params`` is a
    ``jax_pipeline.block_cache.BlockParamCache`` (block-level streaming —
    see ``pipeline.py``'s ``JAXSDXLPipeline.__init__``), not a flat dict —
    re-partition the freshly-converted weights and ``cache.load(...)``
    them, which replaces the cache's pinned-host store (and drops any
    stale device-resident blocks from the OLD weights) in one call.
    Otherwise (large-VRAM cards), re-place the flat dict exactly as
    before.
    """
    from jax_pipeline import convert, host_offload

    log.info("[JAX LoRA] LoRA/DoRA config changed - reloading JAX UNet weights...")
    params = convert.load_weights_from_ldm(unet_patcher.model, report=False)

    phase_manager = getattr(pipeline, "_phase_manager", None)
    if phase_manager is not None and phase_manager.enabled:
        from jax_pipeline.block_cache import partition_params_by_block
        from jax_pipeline.unet import build_block_ids

        cache = pipeline._block_cache
        cache.load(partition_params_by_block(params, build_block_ids()))
        pipeline._params = cache
    else:
        pipeline._params = host_offload.place_params(params, pipeline._device, pipeline.host_offload)

    log.info("[JAX LoRA] JAX UNet weights reloaded (all LoRA / DoRA deltas merged).")
