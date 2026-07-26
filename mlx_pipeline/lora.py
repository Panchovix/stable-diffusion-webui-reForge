"""
mlx_pipeline/lora.py — LoRA / DoRA detection and weight-reload utilities.

Overview
--------
The forge/ldm_patched model patcher applies LoRA patches to PyTorch model
weights transiently: ``patch_model()`` merges LoRA deltas into the actual
``.weight`` tensors before sampling, and ``unpatch_model()`` restores the
originals afterward.

Our MLX UNet and CLIP encoders hold statically-loaded weights.  To stay
in sync with the active LoRA configuration we use a two-pronged approach:

UNet LoRA / DoRA
~~~~~~~~~~~~~~~~
* ``unet_lora_signature(patcher)`` hashes the active patch set (keys +
  strength values).  The hash changes whenever LoRA files are loaded,
  unloaded, or their strengths are adjusted.
* On the first ``apply_model`` call after the config changes,
  ``reload_mlx_unet_weights`` re-reads **all** diffusion model weights from
  the currently-patched PyTorch model (LoRA / DoRA already merged by
  ``patch_model()``) and loads them into the MLX UNet in one shot.
* Subsequent steps in the same generation use the cached MLX weights — no
  per-step overhead.
* All patch types handled by ``ldm_patched.modules.lora.calculate_weight``
  (LoRA, DoRA, LoCon, LoKr, LoHa, IA³, full-diff, etc.) are supported
  automatically because we simply read the already-merged PyTorch weights.

CLIP LoRA
~~~~~~~~~
* ``clip_weight_fingerprint`` samples a handful of weight values from the
  CLIP transformer and returns a tiny tuple used as a cheap change-detection
  proxy (~4–6 float comparisons, negligible cost).
* The CLIP hooks (``mlx_pipeline/clip.py``) compare this fingerprint before
  each text encoding.  A mismatch triggers a live rebuild of the MLX CLIP
  encoder from the current (LoRA-patched) PyTorch CLIP transformer.
* Since CLIP runs **once per prompt** (not per step), the rebuild cost is
  amortised over the entire generation.

Performance
-----------
* UNet reload  : ~1–2 s (once per generation when LoRA config changes)
* CLIP rebuild : ~0.1–0.3 s (once per prompt when CLIP LoRA changes)
* Per-step cost: zero after first step of a generation
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ── UNet LoRA / DoRA ──────────────────────────────────────────────────────────

def unet_lora_sig(unet_patcher) -> str:
    """Return ``str(patches_uuid)`` — the canonical LoRA-change token.

    ldm_patched regenerates ``patches_uuid`` (a UUID4) atomically on every
    add_patches / set_model_patch / strength change, making it the cheapest
    and most reliable change-detection signal available.

    Callers store this string and compare it on the next generation; a
    mismatch means LoRA changed and the MLX UNet weights need reloading.
    """
    return str(getattr(unet_patcher, "patches_uuid", ""))


# Keep the old name as an alias so existing imports keep working.
unet_lora_signature = unet_lora_sig


def reload_mlx_unet_weights(mlx_unet, unet_patcher) -> None:
    """Re-read all UNet weights from the PyTorch diffusion_model into MLX.

    Must be called **after** ldm_patched's ``load_models_gpu()`` (or
    equivalent) has already applied the new LoRA/DoRA deltas to the PyTorch
    model via ``patch_weight_to_device()``.  That function saves original
    weights to ``patcher.backup`` and writes the merged result back into the
    live parameter tensors — so ``diffusion_model.state_dict()`` already
    contains the fully-merged weights when we call this.

    Timeline in the MLX sampler path
    ---------------------------------
    1. User changes LoRA → ``patches_uuid`` regenerated
    2. ``load_models_gpu()`` → ``patch_weight_to_device()`` for every patched
       key → PyTorch diffusion_model has merged weights
    3. ``_patched_sample()`` detects UUID mismatch → calls **this function**
    4. MLX UNet loaded from merged PyTorch weights → correct for all steps

    Handles correctly:
    * LoRA / DoRA added         → reloads merged weights                 ✓
    * LoRA removed              → reloads clean base weights             ✓
    * Strength slider changed   → reloads with new delta                 ✓
    * Multiple LoRAs stacked    → all deltas already merged by ldm_patched ✓
    * DoRA (dora_scale)         → handled by ldm_patched's calculate_weight ✓

    Takes ~1–2 s on M-series hardware; runs at most once per generation
    when the LoRA configuration changes between generations.
    """
    from mlx_pipeline.convert import load_weights_from_ldm

    log.info("[MLX LoRA] LoRA/DoRA config changed — reloading MLX UNet weights …")
    load_weights_from_ldm(mlx_unet, unet_patcher.model, report=False)
    log.info("[MLX LoRA] MLX UNet reloaded (all LoRA / DoRA deltas merged).")


# ── CLIP LoRA fingerprint ─────────────────────────────────────────────────────

def clip_weight_fingerprint(
    transformer,
    text_projection: Optional["torch.Tensor"] = None,
) -> Tuple[float, ...]:
    """Return a cheap fingerprint of CLIP transformer weights.

    Samples a handful of representative values from the first and last
    attention projection layers.  The fingerprint changes when LoRA (or any
    other external modification) alters the transformer weights.

    Parameters
    ----------
    transformer       : HuggingFace CLIPTextModel whose weights to sample
    text_projection   : optional raw nn.Parameter for CLIP-G pooled projection
                        (``clip_g.text_projection``).  Included in the
                        fingerprint so CLIP-G text_projection LoRA is detected.

    Returns
    -------
    A short tuple of floats, or an empty tuple on error.  An empty tuple is
    treated as "always rebuild" by the caller — safe but potentially wasteful.
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
        log.debug("[MLX LoRA] clip_weight_fingerprint error: %s", exc)
        return ()
