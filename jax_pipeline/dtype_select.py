"""
jax_pipeline/dtype_select.py — hardware-Tensor-Core-aware compute dtype
selection.

Every weight-conversion call site in this package (convert.py, clip.py,
vae.py) previously defaulted unconditionally to ``jnp.bfloat16`` — fine
on Ampere-or-newer NVIDIA GPUs (compute capability >= 8.0), which have
native BF16 Tensor Core support, but not automatically the right choice
everywhere:

  * Turing/Volta (compute capability 7.0-7.9, e.g. every RTX 20-series
    card, Tesla T4/V100): Tensor Cores accelerate FP16 natively but do
    NOT support BF16 at all — confirmed against NVIDIA's own Turing
    architecture deep dive, which lists FP16/INT8/INT4 as Turing Tensor
    Core precisions with bf16 absent. Requesting bf16 here doesn't
    error (unlike some other inference frameworks, which hard-refuse);
    XLA's GemmRewriter/FloatNormalization compiler passes fall back to
    upcasting bf16 operands to fp16 (which IS accelerated) for the
    actual matmul, then downcast the result back to bf16 for storage —
    the same documented pattern XLA uses for fp8 on GPUs below sm_89.
    That works, but pays a real conversion cost on both sides of every
    single matmul. Computing NATIVELY in fp16 instead avoids that
    round-trip entirely.
  * Anything else (compute capability < 7.0, ROCm, CPU, TPU, or compute
    capability unreadable for any reason): no researched generation-
    specific override here — keep whatever the caller requested
    (defaults to bfloat16, matching this package's original behavior).

The trade fp16 makes for that Tensor-Core speedup: its dynamic range
(~6.5e4 max magnitude) is drastically narrower than bfloat16's
(~3.4e38, matching fp32's exponent range) — precisely the problem
bfloat16 was invented to avoid. SDXL is known to occasionally produce
large-magnitude activations (most commonly reported in VAE decode
across the wider PyTorch fp16-SD ecosystem) that fp16 can overflow to
inf/NaN on where bf16 would not. ``select_compute_dtype`` warns loudly
about this rather than silently making the swap.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)

#: Fires the detailed fp16-fallback explanation at most once per process
#: (every UNet/CLIP/VAE call site independently calls select_compute_
#: dtype — without this, activation would print the same multi-line
#: warning three times in a row).
_warned_fp16_fallback = {"shown": False}

#: Increasing numerical safety (widest-dynamic-range-last), NOT
#: increasing speed — the order host_offload.handle_jax_nan walks a
#: component's dtype through after a real NaN/Inf is observed in its
#: output. float32 is the ceiling: full range AND full precision, no
#: further escalation possible.
DTYPE_SAFETY_LADDER: Tuple[str, ...] = ("float16", "bfloat16", "float32")


def escalate_dtype(current_name: str) -> Optional[str]:
    """Return the next-safer dtype name after ``current_name`` on
    ``DTYPE_SAFETY_LADDER``, or None if already at the safest (float32)
    — meaning there's nothing left to escalate TO; the caller should
    treat that as "this component needs a different fix entirely, not
    just a bigger dtype."
    """
    if current_name not in DTYPE_SAFETY_LADDER:
        return "float32"  # unknown/unrecognized name -- jump straight to the safest
    idx = DTYPE_SAFETY_LADDER.index(current_name)
    if idx + 1 >= len(DTYPE_SAFETY_LADDER):
        return None
    return DTYPE_SAFETY_LADDER[idx + 1]


def dtype_by_name(name: str):
    import jax.numpy as jnp
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[name]


def _cuda_compute_capability(torch_device) -> Optional[Tuple[int, int]]:
    """(major, minor) NVIDIA compute capability for ``torch_device``, or
    None if this isn't a genuine NVIDIA CUDA device.

    The compute-capability-based Tensor Core generation matrix in
    ``select_compute_dtype`` is NVIDIA-specific — ROCm's AMD GPUs have
    their own, entirely different generational bf16/fp16 support story
    that isn't researched here, so a ROCm build of torch (``torch
    .version.hip`` set) is deliberately excluded even though ``torch
    .cuda.is_available()`` also returns True for it.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None) is not None:
        return None
    try:
        return torch.cuda.get_device_capability(torch_device)
    except Exception:
        return None


def select_compute_dtype(torch_device, requested: str = "bfloat16", prefer_safety: bool = False):
    """Return ``(jnp_dtype, dtype_name)`` — the compute dtype to
    actually use for JAX weight conversion, chosen for the ACTUAL
    Tensor Core generation on ``torch_device`` rather than blindly
    using ``requested`` everywhere. See module docstring for the
    reasoning; always logs which dtype was picked and why (the "warn
    the user what dtype is in use, for best performance" half of this
    — not just the fallback-specific warning below).

    ``prefer_safety``: skip the Turing/Volta bf16->fp16 fallback for
    THIS caller specifically, keeping ``requested`` (bfloat16 by
    default) even on hardware that would otherwise get the fp16
    speedup. Intended for components where the narrower fp16 range is
    a bad trade even before any actual NaN is observed — VAE decode
    runs ONCE per generation (not once per denoising step like the
    UNet), so the fp16 Tensor-Core speedup there is a much smaller
    absolute win than the risk it introduces (real-hardware report:
    fp16 VAE decode producing NaN/Inf output on Turing from an
    already-large-magnitude latent). Still overridden by
    JAX_PIPELINE_COMPUTE_DTYPE if set.
    """
    import os
    import jax.numpy as jnp

    dtype_map = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}
    default_dtype = dtype_map.get(requested, jnp.bfloat16)

    forced_name = os.environ.get("JAX_PIPELINE_COMPUTE_DTYPE", "").strip().lower()
    if forced_name:
        if forced_name not in dtype_map:
            log.warning(
                "[JAX Pipeline] JAX_PIPELINE_COMPUTE_DTYPE=%r is not one of %s — ignoring, "
                "falling through to automatic hardware-based selection.",
                forced_name, sorted(dtype_map),
            )
        else:
            log.info(
                "[JAX Pipeline] Compute dtype: %s (forced via JAX_PIPELINE_COMPUTE_DTYPE, "
                "skipping automatic hardware-based selection).",
                forced_name,
            )
            return dtype_map[forced_name], forced_name

    cc = _cuda_compute_capability(torch_device)
    if cc is None:
        log.info(
            "[JAX Pipeline] Compute dtype: %s (non-NVIDIA-CUDA device, or compute "
            "capability unavailable — no hardware-specific override applied).",
            requested,
        )
        return default_dtype, requested

    major, minor = cc
    cc_val = major + minor / 10.0

    if cc_val >= 8.0:
        log.info(
            "[JAX Pipeline] Compute dtype: bfloat16 (compute capability %d.%d, "
            "Ampere-or-newer) — native BF16 Tensor Core support: best performance "
            "AND full dynamic range, no trade-off here.",
            major, minor,
        )
        return jnp.bfloat16, "bfloat16"

    if cc_val >= 7.0 and requested == "bfloat16" and prefer_safety:
        log.info(
            "[JAX Pipeline] Compute dtype: bfloat16 (compute capability %d.%d, "
            "Volta/Turing -- pinned to bfloat16 for this component regardless of "
            "the fp16 Tensor-Core speedup other components use here, since the "
            "narrower fp16 dynamic range is a worse trade for it -- see "
            "select_compute_dtype's prefer_safety docstring).",
            major, minor,
        )
        return jnp.bfloat16, "bfloat16"

    if cc_val >= 7.0 and requested == "bfloat16":
        if not _warned_fp16_fallback["shown"]:
            log.warning(
                "[JAX Pipeline] Compute dtype: falling back from bfloat16 to "
                "float16 — compute capability %d.%d (Volta/Turing) Tensor Cores "
                "accelerate FP16 natively but do NOT support bfloat16 at all. "
                "Requesting bf16 on this GPU would still run correctly (XLA "
                "upcasts to fp16 internally for every matmul, then downcasts the "
                "result back), but pays that conversion twice per matmul instead "
                "of computing in fp16 directly — switching to native fp16 avoids "
                "the round trip.\n"
                "CAVEAT: float16's dynamic range (~6.5e4 max magnitude) is far "
                "narrower than bfloat16's (~3.4e38, matching fp32) — SDXL can "
                "produce large-magnitude activations (most commonly reported in "
                "VAE decode across the wider PyTorch fp16-SD ecosystem) that "
                "overflow float16 to inf/NaN where bfloat16 would not. If you see "
                "NaN or black-image output, this dtype fallback is the likely "
                "cause — set JAX_PIPELINE_COMPUTE_DTYPE=bfloat16 to force the "
                "original (slower but numerically safer) behavior.",
                major, minor,
            )
            _warned_fp16_fallback["shown"] = True
        else:
            log.info(
                "[JAX Pipeline] Compute dtype: float16 (compute capability %d.%d, "
                "Volta/Turing fallback from bfloat16 — see the earlier warning "
                "for why).",
                major, minor,
            )
        return jnp.float16, "float16"

    log.info(
        "[JAX Pipeline] Compute dtype: %s (compute capability %d.%d)%s",
        requested, major, minor,
        " — no Tensor Cores on this GPU generation; dtype choice affects memory "
        "bandwidth more than compute throughput here" if cc_val < 7.0 else "",
    )
    return default_dtype, requested
