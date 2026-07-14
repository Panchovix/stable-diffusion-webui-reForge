"""
jax_pipeline/lora.py — LoRA / DoRA detection and weight-reload for the JAX UNet.

Mirrors mlx_pipeline/lora.py's UNet-side mechanism:

* ``unet_lora_sig(unet_patcher)`` reads ``unet_patcher.patches_uuid`` — a
  UUID4 that ldm_patched regenerates atomically on every LoRA/DoRA
  add/remove/strength change — as a cheap change-detection token.
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


def reload_jax_unet_weights(pipeline, unet_patcher) -> None:
    """Reconvert PyTorch UNet weights (LoRA/DoRA already merged) into JAX
    and replace ``pipeline._params`` in place, re-applying the same
    host-offload placement decision made at construction time.
    """
    from jax_pipeline import convert, host_offload

    log.info("[JAX LoRA] LoRA/DoRA config changed - reloading JAX UNet weights...")
    params = convert.load_weights_from_ldm(unet_patcher.model, report=False)
    pipeline._params = host_offload.place_params(params, pipeline._device, pipeline.host_offload)
    log.info("[JAX LoRA] JAX UNet weights reloaded (all LoRA / DoRA deltas merged).")
