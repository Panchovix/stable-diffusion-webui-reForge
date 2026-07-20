"""
jax_pipeline/autotune.py — TensorRT-style benchmark-driven UNet fusion-
level selection. Fourth (and integrating) step of the autotuning effort:
jax_pipeline.unet provides the candidate fusion levels ("fine"
per-micro-block streaming, "coarse" per-stage streaming, "whole"
single-jit — see unet.py's module docstrings), jax_pipeline.engine_cache
persists which one won for a given (component, device, shape) point.
This module is the part in between: for a shape with no cached decision,
build each candidate with EMPTY (zero-filled) tensors — no real
conditioning needed, mirroring TensorRT's builder phase exploring
tactics before a single real inference runs — measure its peak VRAM and
average per-call time, and pick the fastest one that fits.

Only UNet has multiple fusion levels today (CLIP/VAE stay at their
existing whole-component PhaseManager-managed granularity — there was
never a reason to build fine-grained streaming for components that small
in the first place), so ``autotune_unet`` is the only real entry point;
``FUSION_LEVELS``/the DB schema are kept component-generic in
engine_cache.py so CLIP/VAE could gain their own candidate levels later
without a schema change.

Caveat on the VRAM measurement: peak usage is approximated as
(free-VRAM-before-benchmark - free-VRAM-right-after-the-timed-calls,
before cleanup) — with the "platform" XLA allocator (this backend's
default; see jax_pipeline/__init__.py) memory is freed immediately on
delete rather than cached, so this resting-state delta is a reasonable
stand-in for true peak, but it can still UNDER-estimate a transient
spike that occurred mid-computation and was already freed by the time we
measure (e.g. a large intermediate activation buffer). ``select_fusion_
level``'s safety factor exists specifically to compensate for that.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

#: fine/coarse/whole are fixed points, empirically compared against each
#: other by benchmarking all three (see autotune_unet). "segmented" is
#: NOT part of that same comparison loop -- it's already VRAM-aware by
#: construction (jax_pipeline.segment_search builds it against the
#: CURRENT free-VRAM budget directly), so autotune_unet tries it FIRST,
#: standalone, and short-circuits on success rather than ALSO running it
#: back-to-back with three other full candidate builds. Stacking a 4th
#: heavy candidate (its own benchmark alone does up to ~118 sequential
#: fresh JIT compiles for per-atom footprint tracing) onto an already-
#: tight VRAM-constrained card's benchmarking pass is real, avoidable
#: extra peak/cumulative memory churn on exactly the hardware this
#: system exists for -- see git history for a real-hardware OOM this
#: caused when it was a naive 4th entry in this tuple.
FUSION_LEVELS: Tuple[str, ...] = ("fine", "coarse", "whole")

_WARMUP_CALLS = 1
_TIMED_CALLS = 2

#: Multiply a candidate's measured peak by this before comparing against
#: free VRAM — compensates for the resting-state-delta measurement above
#: possibly under-counting a transient mid-computation spike.
_VRAM_SAFETY_FACTOR = 1.35


def _empty_unet_inputs(latent_h: int, latent_w: int, batch: int, seq_len: int, dtype=None):
    """Dummy empty-tensor inputs for benchmarking. ``dtype`` should match
    whatever the REAL params were converted to (bfloat16, or the fp16
    fallback on Turing/Volta — see jax_pipeline.dtype_select) — a
    mismatched dummy-input dtype would benchmark a different
    weight/activation precision combination than what real generation
    actually runs, making the measured peak-VRAM/timing numbers
    meaningless for the decision they're used to make.
    """
    import jax.numpy as jnp

    compute_dtype = dtype if dtype is not None else jnp.bfloat16
    sample = jnp.zeros((batch, 4, latent_h, latent_w), dtype=compute_dtype)
    timestep = jnp.zeros((batch,), dtype=jnp.float32)
    encoder_hidden_states = jnp.zeros((batch, seq_len, 2048), dtype=compute_dtype)
    added_cond_kwargs = {
        "text_embeds": jnp.zeros((batch, 1280), dtype=compute_dtype),
        "time_ids": jnp.zeros((batch, 6), dtype=jnp.float32),
    }
    return sample, timestep, encoder_hidden_states, added_cond_kwargs


def device_name_for(device) -> str:
    """A stable-ish string identifying the GPU/backend, for engine_cache
    signatures. jax.Device exposes ``device_kind`` on real accelerator
    backends (e.g. the actual GPU model name); fall back to str(device)
    if that's unavailable (CPU backend, unusual plugin, etc.).
    """
    return str(getattr(device, "device_kind", None) or device)


def _free_vram_bytes(torch_device) -> int:
    from ldm_patched.modules import model_management
    return model_management.get_free_memory(torch_device)


def _benchmark_candidate(
    level: str, params: Dict[str, Any], jax_device, latent_h: int, latent_w: int,
    batch: int, seq_len: int, torch_device_for_vram_query,
    vram_budget_override: Optional[int] = None,
) -> Optional[Tuple[int, float]]:
    """Build ``level``'s UNet execution path fresh, run it a few times
    with empty tensors, measure (peak_vram_bytes, avg_seconds_per_call),
    then tear the whole thing down again. Returns None if this level
    OOMs or otherwise fails to build/run — a legitimate outcome (the
    level just doesn't fit / doesn't work here), not something to
    propagate as an error.

    ``vram_budget_override`` only applies to ``level == "segmented"`` —
    forwarded to ``segment_search.search_unet_segmentation`` so
    ``_benchmark_segmented_with_regret``'s backtrack loop can ask for a
    progressively smaller, more conservative plan after a larger one
    failed to actually build/run here.
    """
    import jax
    from jax_pipeline import block_cache as block_cache_mod
    from jax_pipeline import unet as unet_mod
    from jax_pipeline import _debug_profile
    from jax_pipeline.host_offload import _is_jax_oom_exception

    free_before = _free_vram_bytes(torch_device_for_vram_query)
    _debug_profile.note(
        f"autotune[{level}]: benchmark starting, free_vram={free_before/1e9:.3f}GB "
        f"(shape={latent_h}x{latent_w} batch={batch} seq={seq_len})"
    )
    phase = "setup"  # updated as we go so a failure's log line always says WHERE it died
    cache = None
    device_params = None
    pending_store = None
    out = None
    try:
        # Infer the compute dtype straight from the real params rather
        # than assuming bfloat16 -- see _empty_unet_inputs's docstring.
        params_dtype = next(iter(params.values())).dtype if params else None
        sample, timestep, enc, added = _empty_unet_inputs(latent_h, latent_w, batch, seq_len, dtype=params_dtype)

        if level == "whole":
            phase = "build (stage full params to device)"
            device_sharding = jax.sharding.SingleDeviceSharding(jax_device, memory_kind="device")
            device_params = jax.tree.map(lambda p: jax.device_put(p, device_sharding), params)
            forward = jax.jit(unet_mod.unet_forward)
            call = lambda: forward(device_params, sample, timestep, enc, added)
        elif level == "segmented":
            from jax_pipeline import segment_search, unet_segments as unet_segs

            phase = "build (segment search: atoms + footprint estimation + greedy plan)"
            atoms, plan = segment_search.search_unet_segmentation(
                params, jax_device, torch_device_for_vram_query, latent_h, latent_w, batch, seq_len,
                vram_budget_override=vram_budget_override,
            )
            _debug_profile.note(
                f"autotune[segmented]: plan ready -- {len(plan.ranges)} segments over {len(atoms)} atoms, "
                f"{sum(len(v) for v in plan.spill_schedule.values())} boundary spills, "
                f"free_vram={_free_vram_bytes(torch_device_for_vram_query)/1e9:.3f}GB"
            )
            phase = "build (compile segments + load block cache)"
            atom_groups = segment_search.atom_groups_from_plan(atoms, plan)
            compiled = unet_segs.compile_segments(atom_groups)
            # BlockParamCache's default budget (None -> 3x the largest
            # SINGLE atom) is far too small here: unet_forward_segmented's
            # driver stages an ENTIRE segment's blocks onto device before
            # that segment's fused jit call runs. plan.cache_budget_bytes
            # is sized by the search itself to hold as MANY segments
            # simultaneously resident as its own VRAM budget allows (up
            # to literally everything, when it all fits) -- not just one
            # segment's bare minimum -- see SegmentPlan's docstring for
            # the real-hardware "100% cache-miss, full weight retransfer
            # every denoising step" a smaller budget caused.
            _debug_profile.note(
                f"autotune[segmented]: sizing BlockParamCache budget="
                f"{plan.cache_budget_bytes/1e9:.3f}GB (from the search's own plan)"
            )
            cache = block_cache_mod.BlockParamCache(jax_device, budget_bytes=plan.cache_budget_bytes)
            cache.load(unet_segs.partition_params_for_atoms(params, atoms))
            pending_store = unet_segs.PendingValueStore(jax_device)
            call = lambda: unet_segs.unet_forward_segmented(
                cache, pending_store, compiled, dict(plan.spill_schedule), sample, timestep, enc, added,
            )
        else:
            phase = "build (partition + load block cache)"
            block_ids = unet_mod.build_block_ids() if level == "fine" else unet_mod.build_block_ids_coarse()
            partitioned = block_cache_mod.partition_params_by_block(params, block_ids)
            cache = block_cache_mod.BlockParamCache(jax_device, budget_bytes=None)
            cache.load(partitioned)
            forward = unet_mod.unet_forward_streaming if level == "fine" else unet_mod.unet_forward_streaming_coarse
            call = lambda: forward(cache, sample, timestep, enc, added)

        _debug_profile.checkpoint(f"autotune[{level}]: after build, before warmup")

        phase = f"warmup ({_WARMUP_CALLS} call(s))"
        for i in range(_WARMUP_CALLS):
            out = call()
            jax.block_until_ready(out)
            _debug_profile.note(
                f"autotune[{level}]: warmup call {i+1}/{_WARMUP_CALLS} done, "
                f"free_vram={_free_vram_bytes(torch_device_for_vram_query)/1e9:.3f}GB"
            )

        _debug_profile.checkpoint(f"autotune[{level}]: after warmup, before timed calls")

        phase = f"timed ({_TIMED_CALLS} call(s))"
        t0 = time.perf_counter()
        for i in range(_TIMED_CALLS):
            out = call()
            jax.block_until_ready(out)
        avg_seconds = (time.perf_counter() - t0) / _TIMED_CALLS

        free_after = _free_vram_bytes(torch_device_for_vram_query)  # measured BEFORE cleanup below
        peak_used = max(free_before - free_after, 0)
        _debug_profile.note(
            f"autotune[{level}]: benchmark SUCCEEDED -- peak={peak_used/1e9:.3f}GB "
            f"avg={avg_seconds*1000:.1f}ms/call free_after={free_after/1e9:.3f}GB"
        )
        return peak_used, avg_seconds

    except Exception as e:
        free_at_failure = _free_vram_bytes(torch_device_for_vram_query)
        if _is_jax_oom_exception(e):
            log.info(
                "[JAX Pipeline] autotune: level=%s OOM'd during benchmark (phase=%s, "
                "free_vram_at_failure=%.3fGB, free_vram_before_attempt=%.3fGB) — doesn't fit here. %s",
                level, phase, free_at_failure / 1e9, free_before / 1e9, e,
            )
        else:
            log.warning(
                "[JAX Pipeline] autotune: level=%s benchmark failed (phase=%s, "
                "free_vram_at_failure=%.3fGB) — skipping.",
                level, phase, free_at_failure / 1e9, exc_info=True,
            )
        _debug_profile.note(
            f"autotune[{level}]: benchmark FAILED at phase='{phase}' -- {type(e).__name__}: {e} "
            f"(free_before={free_before/1e9:.3f}GB free_at_failure={free_at_failure/1e9:.3f}GB "
            f"delta_consumed_before_failing={max(free_before - free_at_failure, 0)/1e9:.3f}GB)"
        )
        return None
    finally:
        try:
            if cache is not None:
                cache.clear()
            if pending_store is not None:
                pending_store.clear()
            if device_params is not None:
                jax.tree.map(lambda p: p.delete() if hasattr(p, "delete") else None, device_params)
            if out is not None:
                jax.tree.map(lambda p: p.delete() if hasattr(p, "delete") else None, out)
        except Exception:
            pass


#: Each regret retry uses this fraction of the PREVIOUS attempt's
#: budget — geometric shrink (100%, 75%, 56%, 42%, 32% of the original
#: over 5 attempts), converging quickly without being so aggressive
#: that one failed attempt jumps straight to "fine"-equivalent
#: granularity.
_SEGMENTED_REGRET_SHRINK = 0.75

#: How many progressively-smaller budgets to try before giving up on
#: segmented entirely and falling through to fine/coarse/whole.
_SEGMENTED_REGRET_MAX_ATTEMPTS = 5


def _benchmark_segmented_with_regret(
    params: Dict[str, Any], jax_device, latent_h: int, latent_w: int,
    batch: int, seq_len: int, torch_device_for_vram_query,
) -> Optional[Tuple[int, float]]:
    """Try the segmented candidate; if it fails to actually build/run,
    "regret" the budget the search used and try again with a smaller
    one, producing a more conservative (more, smaller segments) plan
    each time — up to ``_SEGMENTED_REGRET_MAX_ATTEMPTS`` attempts.

    Without this, a single failed attempt (the search's footprint
    estimate was too optimistic, or free VRAM was simply tighter than
    expected once XLA's own compiled-executable overhead for a large
    fused segment is included) abandoned segmentation outright in favor
    of the fine/coarse/whole fallback — even though a SMALLER, safer
    segmentation might well have fit. This is what makes the search an
    actual Best-Fit BACKTRACKING search rather than a single blind
    guess: infeasible attempts are retried against a tighter constraint
    instead of given up on.

    Each retry's ``search_unet_segmentation`` call still benefits from
    ``estimate_activation_bytes``'s own DB warm start (the expensive
    per-atom trace only ever runs once per shape/device, regardless of
    how many regret attempts follow) — only the cheap O(atoms) greedy
    pass re-runs on each retry.
    """
    from jax_pipeline import _debug_profile

    free_vram = _free_vram_bytes(torch_device_for_vram_query)
    budget = free_vram
    for attempt in range(1, _SEGMENTED_REGRET_MAX_ATTEMPTS + 1):
        result = _benchmark_candidate(
            "segmented", params, jax_device, latent_h, latent_w, batch, seq_len,
            torch_device_for_vram_query, vram_budget_override=budget,
        )
        if result is not None:
            if attempt > 1:
                log.info(
                    "[JAX Pipeline] autotune: segmented succeeded on regret attempt %d/%d "
                    "with a shrunk budget=%.3fGB (original free_vram=%.3fGB)",
                    attempt, _SEGMENTED_REGRET_MAX_ATTEMPTS, budget / 1e9, free_vram / 1e9,
                )
            return result
        if attempt == _SEGMENTED_REGRET_MAX_ATTEMPTS:
            break
        next_budget = int(budget * _SEGMENTED_REGRET_SHRINK)
        _debug_profile.note(
            f"autotune: segmented regret attempt {attempt}/{_SEGMENTED_REGRET_MAX_ATTEMPTS} failed at "
            f"budget={budget/1e9:.3f}GB -- shrinking to {next_budget/1e9:.3f}GB and retrying"
        )
        log.info(
            "[JAX Pipeline] autotune: segmented attempt %d/%d failed at budget=%.3fGB — "
            "regretting, retrying with a smaller budget=%.3fGB",
            attempt, _SEGMENTED_REGRET_MAX_ATTEMPTS, budget / 1e9, next_budget / 1e9,
        )
        budget = next_budget

    log.info(
        "[JAX Pipeline] autotune: segmented exhausted all %d regret attempts (final budget=%.3fGB) "
        "— giving up on segmentation for this shape, falling back to fine/coarse/whole.",
        _SEGMENTED_REGRET_MAX_ATTEMPTS, budget / 1e9,
    )
    return None


def select_fusion_level(
    candidates: Dict[str, Tuple[int, float]], free_vram_bytes: int,
    safety_factor: float = _VRAM_SAFETY_FACTOR,
) -> str:
    """Pick the FASTEST candidate whose (peak_vram * safety_factor) fits
    within free_vram_bytes. If none comfortably fit, fall back to
    whichever used the LEAST VRAM — better to run slow-but-safe than not
    run at all. ``candidates`` must be non-empty.
    """
    fitting = {lvl: stats for lvl, stats in candidates.items() if stats[0] * safety_factor <= free_vram_bytes}
    if fitting:
        return min(fitting.items(), key=lambda kv: kv[1][1])[0]  # fastest avg_seconds among those that fit
    return min(candidates.items(), key=lambda kv: kv[1][0])[0]  # smallest peak_vram as last resort


def autotune_unet(
    params: Dict[str, Any], jax_device, torch_device_for_vram_query,
    latent_h: int, latent_w: int, batch: int, seq_len: int,
) -> str:
    """Return the fusion level ("fine"/"coarse"/"whole") to use for the
    UNet at this shape — from the persistent DB if there's an exact or
    close-enough match, otherwise by benchmarking every candidate fresh
    with empty tensors and saving the winner for next time.

    Best-effort: any failure anywhere in this function falls back to
    "fine" (the original, most VRAM-conservative, most battle-tested
    level in this session) rather than raising — autotuning is an
    optimization, not something a bug in it should be allowed to block
    generation over.
    """
    try:
        from ldm_patched.modules import model_management
        from jax_pipeline import engine_cache

        device_name = device_name_for(jax_device)
        total_vram = model_management.get_total_memory(torch_device_for_vram_query)
        signature = engine_cache.compute_signature(
            "unet", device_name, total_vram, latent_h, latent_w, batch, seq_len,
        )

        entry = engine_cache.lookup_exact(signature)
        if entry is None:
            entry = engine_cache.lookup_closest(
                "unet", device_name, total_vram, latent_h, latent_w, batch, seq_len,
            )
        if entry is not None:
            chosen = entry.get("decision", "fine")
            log.info(
                "[JAX Pipeline] autotune: DB match for unet (sig=%s) -> '%s'", signature, chosen,
            )
            return chosen

        from jax_pipeline import _debug_profile
        free_at_start = model_management.get_free_memory(torch_device_for_vram_query)
        _debug_profile.checkpoint(
            f"autotune: starting for unet sig={signature} "
            f"(free_vram={free_at_start/1e9:.3f}GB total_vram={total_vram/1e9:.3f}GB)"
        )
        log.info(
            "[JAX Pipeline] autotune: no DB match for unet (sig=%s, %dx%d latent, batch=%d, "
            "seq=%d, free_vram=%.3fGB) — trying the Best-Fit segmentation search first...",
            signature, latent_h, latent_w, batch, seq_len, free_at_start / 1e9,
        )
        segmented_result = _benchmark_segmented_with_regret(
            params, jax_device, latent_h, latent_w, batch, seq_len, torch_device_for_vram_query,
        )
        if segmented_result is not None:
            peak, avg = segmented_result
            log.info(
                "[JAX Pipeline] autotune: level=segmented peak=%.2fGB avg=%.1fms/call — "
                "accepted outright (already built against the current VRAM budget), "
                "skipping fine/coarse/whole benchmarking.",
                peak / 1e9, avg * 1000,
            )
            engine_cache.save(
                "unet", signature, device_name, total_vram, latent_h, latent_w, batch, seq_len,
                "segmented", {"segmented": {"peak_vram_bytes": peak, "avg_seconds": avg}},
            )
            return "segmented"

        log.info(
            "[JAX Pipeline] autotune: segmented search unavailable for this shape — "
            "falling back to benchmarking fine/coarse/whole with empty tensors...",
        )
        free_vram = model_management.get_free_memory(torch_device_for_vram_query)
        stats: Dict[str, Tuple[int, float]] = {}
        approx_total_bytes = sum(int(v.nbytes) for v in params.values())

        for level in FUSION_LEVELS:
            if level == "whole" and approx_total_bytes * _VRAM_SAFETY_FACTOR > free_vram:
                log.info(
                    "[JAX Pipeline] autotune: skipping 'whole' (needs ~%.2fGB, only %.2fGB free)",
                    approx_total_bytes / 1e9, free_vram / 1e9,
                )
                continue
            result = _benchmark_candidate(
                level, params, jax_device, latent_h, latent_w, batch, seq_len, torch_device_for_vram_query,
            )
            if result is not None:
                stats[level] = result
                log.info(
                    "[JAX Pipeline] autotune: level=%s peak=%.2fGB avg=%.1fms/call",
                    level, result[0] / 1e9, result[1] * 1000,
                )

        if not stats:
            free_at_end = model_management.get_free_memory(torch_device_for_vram_query)
            # Distinguishes "genuinely out of luck" (free VRAM was already
            # critically low before we even started trying, and stayed
            # roughly flat across every attempt -- nothing to fit no
            # matter what) from "something's off" (free VRAM DROPPED
            # significantly across the attempts and never came back, i.e.
            # a leak/cleanup bug, not a capacity problem). Set
            # JAX_PIPELINE_PROFILE=1 for the full per-candidate/per-atom
            # trail (autotune[<level>]/segment_search notes above) that
            # pinpoints exactly which phase of which candidate failed.
            leaked = free_at_start - free_at_end
            log.warning(
                "[JAX Pipeline] autotune: every candidate (segmented, fine, coarse, whole) "
                "failed/OOM'd during benchmarking — defaulting to 'fine' (known safest). "
                "free_vram: start=%.3fGB end=%.3fGB (net change=%.3fGB%s). "
                "Set JAX_PIPELINE_PROFILE=1 for a per-candidate/per-atom failure trail.",
                free_at_start / 1e9, free_at_end / 1e9, leaked / 1e9,
                " -- did NOT recover, suggests a cleanup/leak bug rather than plain capacity"
                if leaked > 256 * 1024 * 1024 else " -- recovered fine, looks like a genuine capacity limit",
            )
            return "fine"

        chosen = select_fusion_level(stats, free_vram)
        engine_cache.save(
            "unet", signature, device_name, total_vram, latent_h, latent_w, batch, seq_len,
            chosen, {lvl: {"peak_vram_bytes": p, "avg_seconds": a} for lvl, (p, a) in stats.items()},
        )
        log.info("[JAX Pipeline] autotune: chose '%s' for unet (sig=%s)", chosen, signature)
        return chosen

    except Exception as e:
        log.warning("[JAX Pipeline] autotune failed (%s) — defaulting to 'fine'.", e, exc_info=True)
        return "fine"
