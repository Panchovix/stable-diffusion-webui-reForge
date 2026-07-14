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
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import torch

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


# -- CLIP LoRA fingerprint ----------------------------------------------------

def clip_weight_fingerprint(
    transformer,
    text_projection: Optional["torch.Tensor"] = None,
) -> Tuple[float, ...]:
    """Return a cheap fingerprint of CLIP transformer weights.

    Samples a handful of representative values from the first and last
    attention projection layers. The fingerprint changes when LoRA (or any
    other external modification) alters the transformer weights. CLIP has
    no patches_uuid-style signal exposed to jax_pipeline, so this sampled
    fingerprint is the change-detection proxy instead (identical approach
    to mlx_pipeline.lora.clip_weight_fingerprint).

    Parameters
    ----------
    transformer       : HuggingFace CLIPTextModel whose weights to sample
    text_projection    : optional raw nn.Parameter for CLIP-G pooled
                         projection (``clip_g.text_projection``). Included
                         in the fingerprint so CLIP-G text_projection LoRA
                         is detected too.

    Returns
    -------
    A short tuple of floats, or an empty tuple on error. An empty tuple is
    treated as "always rebuild" by the caller - safe but potentially
    wasteful.
    """
    try:
        layers = transformer.text_model.encoder.layers
        q0 = layers[0].self_attn.q_proj.weight
        qN = layers[-1].self_attn.q_proj.weight
        fp: Tuple[float, ...] = (
            float(q0[0, 0].item()),
            float(q0[-1, -1].item()),
            float(qN[0, 0].item()),
            float(qN[-1, -1].item()),
        )
        if text_projection is not None:
            tp = text_projection
            fp = fp + (float(tp[0, 0].item()), float(tp[-1, -1].item()))
        return fp
    except Exception as exc:
        log.debug("[JAX LoRA] clip_weight_fingerprint error: %s", exc)
        return ()
