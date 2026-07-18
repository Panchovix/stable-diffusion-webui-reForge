"""
jax_pipeline/_debug_profile.py — TEMPORARY VRAM profiling instrumentation.

Added to diagnose a ~4000MB VRAM baseline that exists BEFORE the sampler
loop's first step even runs — i.e. it survives all of jax_pipeline's
UNet/CLIP/VAE torch-side eviction work (see host_offload.py), so the
question is whether that baseline is JAX's own activation-time footprint,
torch's cache-based allocator holding onto something we haven't evicted
yet, or fixed overhead (CUDA context, XLA's own reserve).

Opt-in via env var ``JAX_PIPELINE_PROFILE=1`` — zero effect on a normal
run. Call sites live in ``jax_pipeline/__init__.py`` at each phase
boundary (activation complete, generation start, per-step, generation
end). Each checkpoint prints BOTH:

  * torch's own view (``torch.cuda.memory_allocated``/``memory_reserved``)
  * JAX's own view (``jax.live_arrays()`` byte total) — jax.live_arrays()
    enumerates every array JAX's runtime currently considers live, so a
    non-zero total here even when the UNet/CLIP/VAE "phase" should all be
    offloaded to pinned host would point at something jax_pipeline itself
    is holding onto (e.g. a stray reference keeping an array alive) rather
    than a torch-side or fixed-overhead cause.

Also wires up torch's memory-history recorder and JAX's pprof-format
device memory profile dumper, both dumped to a temp directory at the end
of a profiled generation for deeper offline inspection (torch's snapshot
viewable at https://pytorch.org/memory_viz, JAX's via `go tool pprof`).

REMOVE this file and its call sites once the baseline issue is
understood/fixed — this is a debugging aid, not a permanent feature.
"""

from __future__ import annotations

import logging
import os
import tempfile

log = logging.getLogger(__name__)

ENABLED = os.environ.get("JAX_PIPELINE_PROFILE", "").strip().lower() in ("1", "true")

_torch_history_started = False
_dump_dir = os.path.join(tempfile.gettempdir(), "jax_pipeline_profile")


def _torch_stats() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "torch: cuda unavailable"
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1e6
        reserved = torch.cuda.memory_reserved(dev) / 1e6
        free, total = torch.cuda.mem_get_info(dev)
        return (
            f"torch: allocated={allocated:.1f}MB reserved={reserved:.1f}MB "
            f"driver_free={free/1e6:.1f}MB driver_total={total/1e6:.1f}MB"
        )
    except Exception as e:
        return f"torch: error ({e})"


def _jax_stats() -> str:
    try:
        import jax
        arrays = jax.live_arrays()
        total = sum(int(a.nbytes) for a in arrays) / 1e6
        return f"jax: live_arrays={len(arrays)} bytes={total:.1f}MB"
    except Exception as e:
        return f"jax: error ({e})"


def checkpoint(label: str) -> None:
    """Log a labeled torch+JAX VRAM snapshot. No-op unless JAX_PIPELINE_PROFILE=1."""
    if not ENABLED:
        return
    msg = f"[JAX Profile] {label} | {_torch_stats()} | {_jax_stats()}"
    log.warning(msg)
    print(msg)


def note(message: str) -> None:
    """Log a plain diagnostic line (no torch/JAX stats attached). No-op
    unless JAX_PIPELINE_PROFILE=1. For one-off yes/no findings (e.g. "did
    the eviction actually find its target?") that don't need a full
    memory snapshot attached.
    """
    if not ENABLED:
        return
    msg = f"[JAX Profile] {message}"
    log.warning(msg)
    print(msg)


def start_torch_history() -> None:
    """Start torch's CUDA memory-history recorder. No-op unless enabled or
    already started; safe to call repeatedly.
    """
    global _torch_history_started
    if not ENABLED or _torch_history_started:
        return
    try:
        import torch
        # stacks="all" (Python + C++) is required for the dumped snapshot's
        # per-block `frames` to include Python call sites — without it,
        # torch only records C++ unwind frames (torch::unwind::unwind()),
        # which makes the snapshot useless for telling jax_pipeline's own
        # code apart from ldm_patched/torch internals. Older torch builds
        # don't accept stacks/context kwargs, so fall back to the bare call.
        try:
            torch.cuda.memory._record_memory_history(
                max_entries=200_000, context="all", stacks="all",
            )
        except TypeError:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        _torch_history_started = True
        log.warning("[JAX Profile] torch.cuda.memory._record_memory_history() started")
    except Exception as e:
        log.warning("[JAX Profile] Could not start torch memory history: %s", e)


def dump_snapshots(tag: str) -> None:
    """Dump both a torch memory-history snapshot and a JAX pprof-format
    device memory profile, timestamp-free (one file per ``tag``, so
    re-running overwrites rather than accumulating). No-op unless enabled.
    """
    if not ENABLED:
        return
    try:
        os.makedirs(_dump_dir, exist_ok=True)
    except Exception as e:
        log.warning("[JAX Profile] Could not create dump dir %s: %s", _dump_dir, e)
        return

    torch_path = os.path.join(_dump_dir, f"torch_{tag}.pickle")
    try:
        import torch
        torch.cuda.memory._dump_snapshot(torch_path)
        print(f"[JAX Profile] torch memory snapshot -> {torch_path}  (view at https://pytorch.org/memory_viz)")
    except Exception as e:
        log.warning("[JAX Profile] Could not dump torch memory snapshot: %s", e)

    jax_path = os.path.join(_dump_dir, f"jax_{tag}.prof")
    try:
        import jax
        jax.profiler.save_device_memory_profile(jax_path)
        print(f"[JAX Profile] JAX device memory profile -> {jax_path}  (inspect with: go tool pprof -http=:8081 {jax_path})")
    except Exception as e:
        log.warning("[JAX Profile] Could not dump JAX memory profile: %s", e)
