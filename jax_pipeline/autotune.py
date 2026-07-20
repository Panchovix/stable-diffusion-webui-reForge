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

FUSION_LEVELS: Tuple[str, ...] = ("fine", "coarse", "whole", "segmented")

_WARMUP_CALLS = 1
_TIMED_CALLS = 2

#: Multiply a candidate's measured peak by this before comparing against
#: free VRAM — compensates for the resting-state-delta measurement above
#: possibly under-counting a transient mid-computation spike.
_VRAM_SAFETY_FACTOR = 1.35


def _empty_unet_inputs(latent_h: int, latent_w: int, batch: int, seq_len: int):
    import jax.numpy as jnp

    sample = jnp.zeros((batch, 4, latent_h, latent_w), dtype=jnp.bfloat16)
    timestep = jnp.zeros((batch,), dtype=jnp.float32)
    encoder_hidden_states = jnp.zeros((batch, seq_len, 2048), dtype=jnp.bfloat16)
    added_cond_kwargs = {
        "text_embeds": jnp.zeros((batch, 1280), dtype=jnp.bfloat16),
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
) -> Optional[Tuple[int, float]]:
    """Build ``level``'s UNet execution path fresh, run it a few times
    with empty tensors, measure (peak_vram_bytes, avg_seconds_per_call),
    then tear the whole thing down again. Returns None if this level
    OOMs or otherwise fails to build/run — a legitimate outcome (the
    level just doesn't fit / doesn't work here), not something to
    propagate as an error.
    """
    import jax
    from jax_pipeline import block_cache as block_cache_mod
    from jax_pipeline import unet as unet_mod
    from jax_pipeline.host_offload import _is_jax_oom_exception

    free_before = _free_vram_bytes(torch_device_for_vram_query)
    cache = None
    device_params = None
    pending_store = None
    out = None
    try:
        sample, timestep, enc, added = _empty_unet_inputs(latent_h, latent_w, batch, seq_len)

        if level == "whole":
            device_sharding = jax.sharding.SingleDeviceSharding(jax_device, memory_kind="device")
            device_params = jax.tree.map(lambda p: jax.device_put(p, device_sharding), params)
            forward = jax.jit(unet_mod.unet_forward)
            call = lambda: forward(device_params, sample, timestep, enc, added)
        elif level == "segmented":
            from jax_pipeline import segment_search, unet_segments as unet_segs

            atoms, plan = segment_search.search_unet_segmentation(
                params, jax_device, torch_device_for_vram_query, latent_h, latent_w, batch, seq_len,
            )
            compiled = unet_segs.compile_segments(segment_search.atom_groups_from_plan(atoms, plan))
            cache = block_cache_mod.BlockParamCache(jax_device, budget_bytes=None)
            cache.load(unet_segs.partition_params_for_atoms(params, atoms))
            pending_store = unet_segs.PendingValueStore(jax_device)
            call = lambda: unet_segs.unet_forward_segmented(
                cache, pending_store, compiled, dict(plan.spill_schedule), sample, timestep, enc, added,
            )
        else:
            block_ids = unet_mod.build_block_ids() if level == "fine" else unet_mod.build_block_ids_coarse()
            partitioned = block_cache_mod.partition_params_by_block(params, block_ids)
            cache = block_cache_mod.BlockParamCache(jax_device, budget_bytes=None)
            cache.load(partitioned)
            forward = unet_mod.unet_forward_streaming if level == "fine" else unet_mod.unet_forward_streaming_coarse
            call = lambda: forward(cache, sample, timestep, enc, added)

        for _ in range(_WARMUP_CALLS):
            out = call()
            jax.block_until_ready(out)

        t0 = time.perf_counter()
        for _ in range(_TIMED_CALLS):
            out = call()
            jax.block_until_ready(out)
        avg_seconds = (time.perf_counter() - t0) / _TIMED_CALLS

        free_after = _free_vram_bytes(torch_device_for_vram_query)  # measured BEFORE cleanup below
        peak_used = max(free_before - free_after, 0)
        return peak_used, avg_seconds

    except Exception as e:
        if _is_jax_oom_exception(e):
            log.info("[JAX Pipeline] autotune: level=%s OOM'd during benchmark — doesn't fit here.", level)
        else:
            log.warning("[JAX Pipeline] autotune: level=%s benchmark failed (%s) — skipping.", level, e)
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

        log.info(
            "[JAX Pipeline] autotune: no DB match for unet (sig=%s, %dx%d latent, batch=%d, "
            "seq=%d) — benchmarking every fusion level with empty tensors...",
            signature, latent_h, latent_w, batch, seq_len,
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
            log.warning(
                "[JAX Pipeline] autotune: every candidate failed/OOM'd during benchmarking — "
                "defaulting to 'fine' (known safest)."
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
