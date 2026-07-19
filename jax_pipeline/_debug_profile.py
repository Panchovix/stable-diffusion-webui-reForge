"""
jax_pipeline/_debug_profile.py — TEMPORARY profiling instrumentation
(VRAM + speed).

VRAM phase (original motivation): diagnose a ~4000MB VRAM baseline that
existed BEFORE the sampler loop's first step even ran. Resolved — see
git history (torch redundant-copy eviction + block-level UNet streaming).

Speed phase (current): block-level streaming trades VRAM headroom for
real per-step PCIe transfer cost (every block's weights re-stage from
pinned host on every denoising step — see block_cache.py's module
docstring). Now that peak VRAM is well under budget (~3.8GB observed on
an 8GB card, "overkill" per the user), the open question is where the
resulting ~7s/it on DPM++ 2M actually goes: transfer, XLA compute, or
Python/dispatch overhead across ~130 blocks x N steps. Timing
instrumentation below answers this directly instead of guessing —
record_timing()/timing_summary() aggregate wall-clock time by phase
("transfer" = block_cache.py's cache-miss host->device stage, "compute"
= each jitted block call, both bucketed by block_id too) and print a
report (+ dump raw records to JSON) at generation end.

Opt-in via env var ``JAX_PIPELINE_PROFILE=1`` — zero effect on a normal
run. Call sites live in ``jax_pipeline/__init__.py`` at each phase
boundary (activation complete, generation start, per-step, generation
end) and in ``block_cache.py``/``unet.py`` for timing.

VRAM checkpoints print BOTH:
  * torch's own view (``torch.cuda.memory_allocated``/``memory_reserved``)
  * JAX's own view (``jax.live_arrays()`` byte total)

Also wires up torch's memory-history recorder, JAX's pprof-format device
memory profile dumper, and (new) a real ``jax.profiler`` trace capture
(XLA op-level timeline, viewable in Perfetto UI at ui.perfetto.dev or
TensorBoard's Trace Viewer with ``tensorboard-plugin-profile`` installed)
— all dumped to a temp directory at generation end for deeper offline
inspection.

REMOVE this file and its call sites once the VRAM/speed investigation is
done — this is a debugging aid, not a permanent feature.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

log = logging.getLogger(__name__)

ENABLED = os.environ.get("JAX_PIPELINE_PROFILE", "").strip().lower() in ("1", "true")

_torch_history_started = False
_jax_trace_active = False
_dump_dir = os.path.join(tempfile.gettempdir(), "jax_pipeline_profile")

#: (phase, block_id, seconds) tuples. phase in {"transfer", "compute", "hit"}.
#: "transfer" = block_cache.py cache-miss host->device stage,
#: "compute" = a jitted block call's wall time (jax.block_until_ready'd,
#: so it's real completion time, not async-dispatch time),
#: "hit" = a cache-hit block_cache.py get() (no transfer — recorded for
#: hit-rate visibility, not because the time itself is meaningful).
_timing_records: list = []


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


def reset_timing() -> None:
    """Clear accumulated timing records. Call at the start of a profiled
    generation so multiple test runs don't blend together. No-op unless
    enabled.
    """
    if not ENABLED:
        return
    _timing_records.clear()


def record_timing(phase: str, block_id: str, seconds: float) -> None:
    """Record one timed event. No-op (near-zero overhead: one `if`) unless
    enabled — safe to call unconditionally from hot paths like
    BlockParamCache.get() and every jitted block call.
    """
    if not ENABLED:
        return
    _timing_records.append((phase, block_id, seconds))


def timing_summary(top_n: int = 20) -> str:
    """Human-readable breakdown of accumulated timing records: total time
    and count per phase (transfer/compute/hit), plus the top_n slowest
    individual block_ids by total accumulated time. Returns "" if
    disabled or nothing recorded yet.
    """
    if not ENABLED or not _timing_records:
        return ""

    import collections
    by_phase: dict = collections.defaultdict(lambda: [0.0, 0])
    by_block: dict = collections.defaultdict(lambda: [0.0, 0])
    for phase, block_id, seconds in _timing_records:
        by_phase[phase][0] += seconds
        by_phase[phase][1] += 1
        by_block[(phase, block_id)][0] += seconds
        by_block[(phase, block_id)][1] += 1

    lines = ["[JAX Profile] ── timing summary ──"]
    total = sum(v[0] for v in by_phase.values())
    for phase in ("transfer", "compute", "hit"):
        if phase in by_phase:
            secs, n = by_phase[phase]
            pct = 100.0 * secs / total if total else 0.0
            lines.append(f"  {phase:10s}: {secs:8.3f}s  ({n:5d} calls, {pct:5.1f}%, {secs/max(n,1)*1000:.2f}ms/call avg)")
    lines.append(f"  {'TOTAL':10s}: {total:8.3f}s")

    top = sorted(by_block.items(), key=lambda kv: -kv[1][0])[:top_n]
    if top:
        lines.append(f"  -- top {len(top)} slowest (phase, block_id) by total time --")
        for (phase, block_id), (secs, n) in top:
            lines.append(f"  {secs:7.3f}s  x{n:3d}  [{phase}] {block_id}")

    return "\n".join(lines)


def print_timing_summary(top_n: int = 20) -> None:
    """Log + print timing_summary()'s report. No-op unless enabled."""
    if not ENABLED:
        return
    s = timing_summary(top_n=top_n)
    if s:
        log.warning(s)
        print(s)


def dump_timing(tag: str) -> None:
    """Dump raw (phase, block_id, seconds) timing records to a JSON file
    for offline analysis. No-op unless enabled.
    """
    if not ENABLED or not _timing_records:
        return
    try:
        os.makedirs(_dump_dir, exist_ok=True)
        path = os.path.join(_dump_dir, f"timing_{tag}.json")
        with open(path, "w") as f:
            json.dump(
                [{"phase": p, "block_id": b, "seconds": s} for p, b, s in _timing_records],
                f, indent=1,
            )
        print(f"[JAX Profile] raw timing records -> {path}")
    except Exception as e:
        log.warning("[JAX Profile] Could not dump timing records: %s", e)


def start_jax_profiler_trace(tag: str) -> None:
    """Start a real jax.profiler trace (XLA op-level timeline) — a
    complement to the hand-rolled timing above, viewable in Perfetto UI
    (ui.perfetto.dev) or TensorBoard's Trace Viewer
    (tensorboard-plugin-profile) for a full op-by-op breakdown including
    anything the coarse transfer/compute buckets above miss. No-op unless
    enabled; safe to call when already active (no-ops).
    """
    global _jax_trace_active
    if not ENABLED or _jax_trace_active:
        return
    try:
        import jax
        os.makedirs(_dump_dir, exist_ok=True)
        trace_dir = os.path.join(_dump_dir, f"jax_trace_{tag}")
        jax.profiler.start_trace(trace_dir)
        _jax_trace_active = True
        log.warning("[JAX Profile] jax.profiler trace started -> %s", trace_dir)
    except Exception as e:
        log.warning("[JAX Profile] Could not start jax.profiler trace: %s", e)


def stop_jax_profiler_trace() -> None:
    """Stop the jax.profiler trace started by start_jax_profiler_trace().
    No-op if not active."""
    global _jax_trace_active
    if not ENABLED or not _jax_trace_active:
        return
    try:
        import jax
        jax.profiler.stop_trace()
        print(f"[JAX Profile] jax.profiler trace saved under -> {_dump_dir}  (open with Perfetto UI or TensorBoard)")
    except Exception as e:
        log.warning("[JAX Profile] Could not stop jax.profiler trace: %s", e)
    finally:
        _jax_trace_active = False


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
