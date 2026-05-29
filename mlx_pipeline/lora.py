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

import hashlib
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ── UNet LoRA / DoRA ──────────────────────────────────────────────────────────

def unet_lora_signature(unet_patcher) -> str:
    """Return a hash that changes whenever the active UNet LoRA config changes.

    Only considers patches whose key starts with ``"diffusion_model."`` —
    CLIP patches in the same patcher do not affect this hash.

    Returns an empty string when no UNet patches are registered (matches the
    initial state right after model load with no LoRA selected).
    """
    patches = getattr(unet_patcher, "patches", {})
    unet_patches = {
        k: v for k, v in patches.items()
        if k.startswith("diffusion_model.")
    }
    if not unet_patches:
        return ""

    h = hashlib.md5(usedforsecurity=False)
    for key in sorted(unet_patches):
        for patch in unet_patches[key]:
            # patch tuple: (strength, v, strength_model, offset, function)
            strength = patch[0] if len(patch) > 0 else 1.0
            sm       = patch[2] if len(patch) > 2 else 1.0
            try:
                h.update(f"{key}:{float(strength):.8f}:{float(sm):.8f}\n".encode())
            except (TypeError, ValueError):
                h.update(f"{key}:?:?\n".encode())
    return h.hexdigest()


def reload_mlx_unet_weights(mlx_unet, unet_patcher) -> None:
    """Re-read all UNet weights from the (LoRA-patched) PyTorch model into MLX.

    Called when the LoRA signature changes.  At call time ``patch_model()``
    has already run, so ``unet_patcher.model.diffusion_model.state_dict()``
    returns tensors with **all** LoRA / DoRA / LoCon deltas already merged.
    We simply re-read them through the same conversion pipeline used at init.

    This correctly handles:
    * LoRA/DoRA added   — reloads merged weights          ✓
    * LoRA removed      — reloads clean base weights      ✓
    * Strength changed  — reloads with new merged delta   ✓
    * Multiple LoRAs    — all deltas already merged by ldm_patched  ✓

    The reload takes ~1–2 s on M-series hardware; it happens at most once
    per generation when the LoRA configuration changes between generations.
    """
    from mlx_pipeline.convert import load_weights_from_ldm

    log.info("[MLX LoRA] UNet LoRA config changed — reloading weights into MLX …")
    load_weights_from_ldm(mlx_unet, unet_patcher.model, report=False)
    log.info("[MLX LoRA] MLX UNet reloaded (LoRA / DoRA merged).")


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
