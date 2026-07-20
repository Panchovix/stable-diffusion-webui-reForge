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

    ``pipeline._raw_params`` (the flat, un-partitioned dict — the single
    canonical source ``ensure_unet_built``/autotune's benchmarking read
    from) is always refreshed first.

    When ``pipeline._phase_manager`` is enabled, ``pipeline._params`` is
    either a ``jax_pipeline.block_cache.BlockParamCache`` (fine/coarse
    streaming) or a flat dict ("whole" — see ``pipeline.py``'s
    ``ensure_unet_built``/``_build_unet_execution``, chosen by
    ``jax_pipeline.autotune``), or possibly not built at all yet (``None``
    — no generation has run since checkpoint load, so no shape/fusion-
    level decision exists yet):

      * cache already built: re-partition the fresh weights with the SAME
        fusion level already chosen (a LoRA change doesn't alter the
        VRAM/speed trade-off that decided it — no need to re-autotune)
        and ``cache.load(...)`` them, which replaces the cache's
        pinned-host store (dropping any stale device-resident blocks from
        the OLD weights) in one call.
      * "whole" already built (``pipeline._block_cache is None`` but
        ``pipeline._forward is not None``): re-place the flat dict, same
        as the non-phase-managed path below.
      * nothing built yet: just leave the refreshed ``_raw_params`` in
        place — the next ``ensure_unet_built()`` call (first apply_model/
        sampler-hook call of the next generation) builds fresh from it,
        picking up these weights automatically.

    Non-phase-managed (large-VRAM cards): re-place the flat dict exactly
    as before — unchanged.
    """
    from jax_pipeline import convert, host_offload

    log.info("[JAX LoRA] LoRA/DoRA config changed - reloading JAX UNet weights...")
    params = convert.load_weights_from_ldm(unet_patcher.model, report=False)
    pipeline._raw_params = params

    phase_manager = getattr(pipeline, "_phase_manager", None)
    if phase_manager is not None and phase_manager.enabled:
        block_cache = getattr(pipeline, "_block_cache", None)
        level = getattr(pipeline, "_fusion_level", None) or "fine"
        if level == "segmented" and block_cache is not None:
            # block_cache is actually a jax_pipeline.unet_segments.SegmentedUnetState
            # here (see pipeline.py's _build_unet_execution) -- its own
            # BlockParamCache re-partitions against the SAME atom sequence
            # (and thus the SAME segmentation/spill_schedule) already
            # chosen for this shape; a LoRA change doesn't alter the
            # weight-byte-driven segmentation decision enough to be worth
            # re-searching (weight SHAPES, which drive the search, never
            # change from a LoRA merge -- only weight VALUES do).
            from jax_pipeline import unet_segments as unet_segs

            block_cache.param_cache.load(unet_segs.partition_params_for_atoms(params, block_cache.atoms))
            pipeline._params = block_cache
        elif block_cache is not None:
            from jax_pipeline.block_cache import partition_params_by_block
            from jax_pipeline.unet import build_block_ids, build_block_ids_coarse

            block_ids = build_block_ids() if level == "fine" else build_block_ids_coarse()
            block_cache.load(partition_params_by_block(params, block_ids))
            pipeline._params = block_cache
        elif getattr(pipeline, "_forward", None) is not None:
            # "whole" was chosen for this session's shape — no BlockParamCache,
            # just a flat device-resident (or offloaded) dict to re-place.
            pipeline._params = host_offload.place_params(params, pipeline._device, pipeline.host_offload)
        # else: nothing built yet this session — _raw_params refresh above
        # is enough; ensure_unet_built() picks it up on the next call.
    else:
        pipeline._params = host_offload.place_params(params, pipeline._device, pipeline.host_offload)

    log.info("[JAX LoRA] JAX UNet weights reloaded (all LoRA / DoRA deltas merged).")
