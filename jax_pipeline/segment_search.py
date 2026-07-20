"""
jax_pipeline/segment_search.py — Best-Fit-style search over
jax_pipeline.unet_segments's atom IR, replacing the discrete "fine"/
"coarse"/"whole" fusion-level choice with an arbitrary, per-shape-
optimal contiguous segmentation.

Problem framing
-----------------
Given the UNet's ~118 atoms in FIXED execution order (see unet_segments
.build_atom_specs — the sampler only ever runs the UNet forward, never
backward, so every atom's access order is completely static), choose
where to cut them into contiguous jitted segments such that:

  1. each segment fills as much of the VRAM budget as possible (fewer,
     bigger segments -> less per-call dispatch overhead -> faster),
  2. no segment (plus whatever named values are still pending across
     its boundary) ever exceeds the VRAM budget (no OOM),
  3. subject to 1+2, minimize dispatch count (fastest).

This is the classic "partition a sequence of positive-weight items into
the fewest contiguous groups, each under a capacity" problem — greedily
packing left-to-right, closing a group only when the NEXT item would
overflow it, is provably optimal for that problem (same principle as
TeX's line-breaking / optimal bin-packing-on-a-sequence): it can never
use more groups than any other valid packing, and each group is packed
as full as it can be before overflowing, which simultaneously satisfies
"fill VRAM" and "fewest segments" (fastest). ``greedy_best_fit_search``
below is exactly that greedy walk, extended with one addition unique to
this problem: some of the "weight" crossing a boundary is SKIP-
CONNECTION / ATTENTION-RESIDUAL state (jax_pipeline.unet_segments's
named values) that doesn't have to stay VRAM-resident — it can be
SPILLED to pinned host RAM at a boundary (jax_pipeline.unet_segments
.PendingValueStore) to buy room for more atoms in the segments that
follow, at the cost of a transfer both ways. Deciding which pending
values to spill, and when, is the "boundary storage" feature described
below.

Boundary storage (skip-connection spill) policy
-------------------------------------------------
Spilling is a real device<->host copy, so it can only happen BETWEEN
segments (once a jitted segment call has returned control to Python),
never mid-segment — see jax_pipeline.unet_segments.unet_forward_segmented.
That means a segment's OWN footprint, while it's still being greedily
grown, must always assume every currently-pending value is fully
device-resident (spilling it wouldn't free anything until the segment
closes anyway). The search therefore makes spill decisions only at the
moment it closes a segment (i.e. exactly the boundary storage points):
if the pending device-resident total at that boundary exceeds
``_MAX_PENDING_FRACTION`` of the VRAM budget, it spills the pending
values whose CONSUMING atom is farthest away first (values about to be
consumed by the very next segment are deliberately left alone — round-
tripping them to host and immediately back would only add transfer time
for no VRAM benefit), stopping as soon as the pending total is back
under the threshold. This directly gives later segments more headroom
to pack more atoms, which is the whole point of exposing this as a
first-class search feature rather than incidental plumbing: without it,
a handful of long-lived skip connections (conv_in's output, alive from
atom 0 until the very last up-block) would permanently tax every single
segment's budget for the entire forward pass.

Footprint accounting
-----------------------
  * WEIGHT bytes are exact, no tracing needed (jax_pipeline.unet_segments
    .compute_weight_bytes just sums each atom's own param slice's
    nbytes) — SUMMED across every atom that joins a segment, since
    jax_pipeline.unet_segments's driver stages a segment's ENTIRE
    block_id set to device before running it (no early release within
    one segment call).
  * NAMED VALUE (skip/residual) bytes are exact too, no tracing needed
    (jax_pipeline.unet_segments.named_value_shapes — every intermediate
    tensor's shape is architecture-determined given the input latent
    resolution, not data-dependent).
  * ACTIVATION bytes (each atom's own transient intermediate-buffer
    cost — attention's QK^T matrix, conv's workspace, etc.) genuinely
    need empirical measurement (``trace_atom_activation_bytes``, one
    jit call per atom with empty tensors, same empty-tensor philosophy
    as jax_pipeline.autotune) OR a close-match warm start from
    jax_pipeline.engine_cache — taken as the PEAK (max), not sum, across
    a segment's atoms, since they're transient per-atom buffers freed
    before the next atom's own computation, not simultaneously live for
    the whole segment the way weights are.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

log = logging.getLogger(__name__)

#: Once a boundary's pending device-resident total exceeds this fraction
#: of the VRAM budget, start spilling (farthest-consumer-first) until
#: back under it.
_MAX_PENDING_FRACTION = 0.4

#: Headroom reserved OUTSIDE the search's own budget for state that's
#: concurrently resident during real sampling but isn't part of any
#: UNet segment's own footprint: JAXCFGDenoiser's pre-converted
#: conditioning tensors (encoder_hidden_states / pooled embeddings),
#: the k-diffusion sampler loop's own per-step state (e.g. DPM++2M's
#: ``old_denoised``, kept resident across steps — see samplers.py), the
#: latent tensor itself, and general JAX/XLA compiled-executable
#: overhead. A search that spends 100% of currently-free VRAM packing
#: UNet atoms into one giant segment (observed on real hardware: 118
#: atoms in ONE segment reported "fits" at the benchmark's isolated-
#: UNet-call granularity) leaves nothing for any of that, which then
#: OOMs once the full sampler loop is actually running around it even
#: though the isolated benchmark succeeded. Mirrors
#: jax_pipeline.host_offload.should_offload's existing 1GB headroom
#: convention for the identical "leave room for everything else"
#: reasoning.
_SAMPLER_RESERVE_BYTES = 1024 * 1024 * 1024  # 1GB

#: Don't bother spilling a value whose consumer is closer than this many
#: atoms away — the round-trip transfer cost isn't worth it for
#: something about to be re-staged almost immediately.
_MIN_SPILL_DISTANCE = 3

#: Same spirit as jax_pipeline.autotune's _VRAM_SAFETY_FACTOR — the
#: activation-bytes measurement (peak-across-segment, not sum) can still
#: under-count a transient spike; leave headroom rather than target the
#: budget exactly.
_VRAM_SAFETY_FACTOR = 1.15


class SegmentPlan(NamedTuple):
    """A concrete, ready-to-compile segmentation: which atom index
    ranges form each segment, which pending named values to spill to
    host right after which segment closes, and how large a
    BlockParamCache the runtime should build for it.

    ``cache_budget_bytes`` matters more than it looks: a cache sized to
    hold only ONE segment's own weights (the bare minimum needed to
    avoid the segment's own gather loop evicting itself) means that
    whenever there's more than one segment, EVERY segment evicts every OTHER
    segment's blocks each time it runs — so on the very next denoising
    step, the previous segment's blocks are gone and get re-transferred
    from pinned host all over again. A real-hardware run hit this
    directly: a 2-segment plan (peak usage only ~3.9GB on an 8GB card,
    plenty of headroom left) still re-transferred its ENTIRE ~5GB of
    weights on every single step (100% cache-miss "transfer" time in
    the profiler, ~14.5s of pure PCIe traffic over a 10-step generation)
    purely because the cache could only ever hold one segment's worth
    at a time. Sizing the cache to hold AS MANY segments simultaneously
    as the search's own VRAM budget allows (up to holding literally
    everything, when it all fits) turns that into a ONE-TIME transfer
    at the first step and zero further transfers for the rest of the
    generation, without changing the segmentation itself at all.
    """
    ranges: Tuple[Tuple[int, int], ...]           # [(start, end), ...] over atoms, half-open
    spill_schedule: Dict[int, Tuple[str, ...]]     # segment_index -> value_ids to spill after it
    cache_budget_bytes: int                        # how much weight data BlockParamCache may hold resident at once


def greedy_best_fit_search(
    atoms: List[Any],
    weight_bytes: Dict[str, int],
    activation_bytes: Dict[str, int],
    value_bytes: Dict[str, int],
    vram_budget_bytes: int,
    max_pending_fraction: float = _MAX_PENDING_FRACTION,
    min_spill_distance: int = _MIN_SPILL_DISTANCE,
) -> SegmentPlan:
    """Greedy left-to-right Best-Fit segmentation with spill-aware
    boundary storage. O(N) in the number of atoms. See module docstring
    for why greedy is optimal here and how spill decisions are made.
    """
    if not atoms:
        return SegmentPlan(ranges=(), spill_schedule={}, cache_budget_bytes=0)

    consumer_idx: Dict[str, int] = {}
    for i, atom in enumerate(atoms):
        for vid in atom.consumes:
            consumer_idx[vid] = i

    ranges: List[Tuple[int, int]] = []
    spill_schedule: Dict[int, Tuple[str, ...]] = {}

    seg_start = 0
    seg_weight = 0
    seg_peak_activation = 0
    pending: Dict[str, str] = {}  # value_id -> "device" | "host"

    def pending_device_bytes() -> int:
        return sum(value_bytes.get(vid, 0) for vid, loc in pending.items() if loc == "device")

    def close_segment(before_idx: int) -> None:
        nonlocal seg_start, seg_weight, seg_peak_activation
        ranges.append((seg_start, before_idx))
        seg_idx = len(ranges) - 1

        budget_used = pending_device_bytes()
        if budget_used > max_pending_fraction * vram_budget_bytes:
            candidates = [vid for vid, loc in pending.items() if loc == "device"]
            # farthest-consumer-first: spilling something about to be
            # used right away would just add a pointless round trip.
            candidates.sort(key=lambda vid: -(consumer_idx.get(vid, before_idx) - before_idx))
            spilled: List[str] = []
            for vid in candidates:
                if consumer_idx.get(vid, before_idx) - before_idx < min_spill_distance:
                    continue
                pending[vid] = "host"
                spilled.append(vid)
                if pending_device_bytes() <= max_pending_fraction * vram_budget_bytes:
                    break
            if spilled:
                spill_schedule[seg_idx] = tuple(spilled)

        seg_start = before_idx
        seg_weight = 0
        seg_peak_activation = 0

    for i, atom in enumerate(atoms):
        a_weight = weight_bytes.get(atom.block_id, 0)
        a_act = int(activation_bytes.get(atom.block_id, 0) * _VRAM_SAFETY_FACTOR)

        projected = (
            (seg_weight + a_weight)
            + max(seg_peak_activation, a_act)
            + pending_device_bytes()
        )
        if projected > vram_budget_bytes and i > seg_start:
            close_segment(i)

        seg_weight += a_weight
        seg_peak_activation = max(seg_peak_activation, a_act)
        for vid in atom.produces:
            pending[vid] = "device"
        for vid in atom.consumes:
            pending.pop(vid, None)

    ranges.append((seg_start, len(atoms)))

    # Size the cache to hold as MANY segments simultaneously resident as
    # the search's own budget allows -- up to literally everything, if
    # it all fits -- rather than just the bare minimum for one segment
    # at a time (min(total_weight, vram_budget_bytes) is always >= any
    # single segment's own weight sum by construction of the loop above,
    # so this can never under-size below what compute_segment_cache_
    # budget's floor would require). See SegmentPlan's docstring for the
    # real-hardware "100% cache-miss, full retransfer every step" this
    # directly fixes.
    total_weight = sum(weight_bytes.get(a.block_id, 0) for a in atoms)
    cache_budget_bytes = min(total_weight, vram_budget_bytes)

    return SegmentPlan(ranges=tuple(ranges), spill_schedule=spill_schedule, cache_budget_bytes=cache_budget_bytes)


def atom_groups_from_plan(atoms: List[Any], plan: SegmentPlan) -> List[List[Any]]:
    return [atoms[start:end] for start, end in plan.ranges]


# ═══════════════════════════════════════════════════════════════════════════════
#  Activation-footprint estimation (tracing) + DB warm start
# ═══════════════════════════════════════════════════════════════════════════════

def trace_atom_activation_bytes(
    atoms: List[Any], flat_params: Dict[str, Any], jax_device, latent_h: int, latent_w: int,
    batch: int, seq_len: int, torch_device_for_vram_query,
) -> Dict[str, int]:
    """Empirically measure each atom's OWN peak activation footprint —
    one jit call per atom (``unet_segments.compile_segments`` at fully-
    fine granularity, already verified numerically equivalent to a
    single fused segment), empty tensors, free-VRAM delta measured
    AFTER that atom's own weights are already staged (so the delta is
    activation-only, matching jax_pipeline.autotune's empty-tensor
    benchmark philosophy but at atom instead of whole-UNet granularity).

    This is the potentially-slow full trace (~one XLA compile per atom)
    the user's warm-start request is meant to amortize — see
    ``estimate_activation_bytes`` below for the DB-backed fast path.
    """
    import jax
    import jax.numpy as jnp
    from jax_pipeline import block_cache as block_cache_mod
    from jax_pipeline import unet_segments as segs
    from jax_pipeline import _debug_profile
    from jax_pipeline.autotune import _free_vram_bytes, _empty_unet_inputs
    from jax_pipeline.unet import _sinusoidal_embedding, _TIME_DIM as _TD

    block_ids = segs.block_ids_in_order(atoms)
    partitioned = segs.partition_params_for_atoms(flat_params, atoms)
    cache = block_cache_mod.BlockParamCache(jax_device, budget_bytes=None)
    cache.load(partitioned)

    # Infer the compute dtype straight from the real params rather than
    # assuming bfloat16 -- see autotune._empty_unet_inputs's docstring.
    params_dtype = next(iter(flat_params.values())).dtype if flat_params else None
    sample, timestep, enc, added = _empty_unet_inputs(latent_h, latent_w, batch, seq_len, dtype=params_dtype)
    pending_store = segs.PendingValueStore(jax_device)
    compiled = segs.compile_segments([[a] for a in atoms])

    free_at_trace_start = _free_vram_bytes(torch_device_for_vram_query)
    _debug_profile.note(
        f"segment_search.trace: starting per-atom trace over {len(atoms)} atoms "
        f"(free_vram={free_at_trace_start/1e9:.3f}GB, cache_budget="
        f"{cache.budget_bytes/1e9 if cache.budget_bytes else 0:.3f}GB)"
    )

    footprints: Dict[str, int] = {}
    try:
        s = jnp.transpose(sample, (0, 2, 3, 1))
        temb_sin = _sinusoidal_embedding(timestep.astype(jnp.float32), _TD, flip_sin_to_cos=True).astype(s.dtype)
        ctx = {
            "x": s, "temb": None, "encoder_hidden_states": enc,
            "timestep_emb": temb_sin,
            "text_embeds": added["text_embeds"].astype(s.dtype),
            "time_ids": added["time_ids"].astype(jnp.float32),
            "values": {},
        }
        for idx, (atom, segment) in enumerate(zip(atoms, compiled)):
            needed = set(atom.consumes) - set(atom.produces)
            for vid in needed:
                if vid in ctx["values"]:
                    continue
                ctx["values"][vid] = pending_store.get(vid)
            params_by_block = {bid: cache.get(bid) for bid in segment.block_ids}

            free_before = _free_vram_bytes(torch_device_for_vram_query)
            # Logged BEFORE the call runs, not after -- if this specific
            # atom's first-ever compile OOMs or hangs, this is the last
            # line printed, pinpointing exactly which atom/block_id died
            # instead of leaving "every candidate failed" as the only clue.
            _debug_profile.note(
                f"segment_search.trace: atom {idx+1}/{len(atoms)} block_id={atom.block_id!r} "
                f"kind={atom.kind} pending_values={len(ctx['values'])} "
                f"resident_bytes={cache._resident_bytes/1e6:.1f}MB free_vram={free_before/1e9:.3f}GB"
            )
            ctx = segment.fn(params_by_block, ctx)
            jax.block_until_ready(ctx["x"])
            free_after = _free_vram_bytes(torch_device_for_vram_query)
            footprints[atom.block_id] = max(free_before - free_after, 0)

            still_pending = dict(ctx["values"])
            ctx["values"] = {}
            for vid, arr in still_pending.items():
                pending_store.put(vid, arr)

            if (idx + 1) % 20 == 0 or idx + 1 == len(atoms):
                _debug_profile.checkpoint(
                    f"segment_search.trace: {idx+1}/{len(atoms)} atoms done, "
                    f"cumulative_footprint_sum={sum(footprints.values())/1e6:.1f}MB "
                    f"vs_started_at_free={free_at_trace_start/1e9:.3f}GB"
                )
    except Exception:
        free_now = _free_vram_bytes(torch_device_for_vram_query)
        log.warning(
            "[JAX Pipeline] segment_search.trace: FAILED partway through the atom trace "
            "(free_vram_now=%.3fGB, free_vram_at_trace_start=%.3fGB, %d/%d atoms measured "
            "before failing) -- see the 'atom N/M block_id=...' note just above this for "
            "exactly which atom was in flight.",
            free_now / 1e9, free_at_trace_start / 1e9, len(footprints), len(atoms),
            exc_info=True,
        )
        raise
    finally:
        try:
            cache.clear()
        except Exception:
            pass
        try:
            pending_store.clear()
        except Exception:
            pass
    return footprints


def estimate_activation_bytes(
    atoms: List[Any], flat_params: Dict[str, Any], jax_device, torch_device_for_vram_query,
    device_name: str, total_vram_bytes: int, latent_h: int, latent_w: int, batch: int, seq_len: int,
) -> Tuple[Dict[str, int], bool]:
    """DB-backed fast path for per-atom activation footprint: exact-
    signature hit reuses the stored measurements outright; a close-
    match hit reuses them as a WARM START (skips the expensive full
    trace); no match at all falls back to the full empirical trace and
    persists the result. Returns ``(activation_bytes, was_traced)``.

    Warm start is exact-by-block_id, not by segmentation: activation
    footprint is a per-op property of (op kind, weight shape, activation
    shape) — all identical between two shapes close enough to pass
    engine_cache's tolerance bands, so reusing the matched entry's raw
    per-atom numbers is safe even though the OPTIMAL segmentation for
    the current (possibly different) VRAM budget is then recomputed
    fresh by ``greedy_best_fit_search`` against the current budget, not
    copied from the matched entry.
    """
    from jax_pipeline import engine_cache

    signature = engine_cache.compute_signature(
        "unet_atom_footprints", device_name, total_vram_bytes, latent_h, latent_w, batch, seq_len,
    )
    entry = engine_cache.lookup_exact(signature)
    if entry is None:
        entry = engine_cache.lookup_closest(
            "unet_atom_footprints", device_name, total_vram_bytes, latent_h, latent_w, batch, seq_len,
        )
    if entry is not None:
        stored = entry.get("decision", {}).get("activation_bytes", {})
        # Only trust the warm start if it covers every atom this UNet
        # actually has (a stale/partial entry from a code change would
        # otherwise silently under-account newly-added/renamed atoms).
        if stored and all(a.block_id in stored for a in atoms):
            log.info(
                "[JAX Pipeline] segment_search: warm-starting atom activation footprints "
                "from engine_db (sig=%s, matched=%s)", signature, entry.get("signature"),
            )
            return stored, False

    log.info(
        "[JAX Pipeline] segment_search: no usable DB match for atom footprints (sig=%s) -- "
        "tracing all %d atoms with empty tensors (one-time cost, cached for next time)...",
        signature, len(atoms),
    )
    traced = trace_atom_activation_bytes(
        atoms, flat_params, jax_device, latent_h, latent_w, batch, seq_len, torch_device_for_vram_query,
    )
    engine_cache.save(
        "unet_atom_footprints", signature, device_name, total_vram_bytes, latent_h, latent_w, batch, seq_len,
        {"activation_bytes": traced}, {"num_atoms": len(atoms)},
    )
    return traced, True


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level entry point
# ═══════════════════════════════════════════════════════════════════════════════

def search_unet_segmentation(
    flat_params: Dict[str, Any], jax_device, torch_device_for_vram_query,
    latent_h: int, latent_w: int, batch: int, seq_len: int, dtype_bytes: int = 2,
    vram_budget_override: Optional[int] = None,
) -> Tuple[List[Any], SegmentPlan]:
    """Full pipeline: build atoms for this shape, get weight/value/
    activation footprints (DB warm start or fresh trace), run the
    Best-Fit search against the available VRAM, persist the resulting
    plan. Returns ``(atoms, plan)`` — the caller (jax_pipeline.pipeline /
    autotune) compiles the plan via
    ``jax_pipeline.unet_segments.compile_segments``.

    ``vram_budget_override``: when given, skips both the free-VRAM query
    AND the DB-reuse-first shortcut below, and searches fresh against
    exactly this budget — this is how ``jax_pipeline.autotune``'s
    regret/backtrack loop asks for a progressively more conservative
    plan after a previous (larger-budget) attempt failed to actually
    build/run; it must bypass the cache, not reuse the failed attempt's
    stale saved plan. When omitted (the common case), this function
    checks for an EXACT signature match in engine_cache first and reuses
    it verbatim rather than re-deriving a plan from whatever free VRAM
    happens to be available at the moment — critical for consistency
    with ``jax_pipeline.pipeline._build_unet_execution``'s OWN call to
    this same function moments later to actually build the chosen plan:
    without this, the plan that gets BUILT (and used for real sampling)
    could differ from whichever plan ``autotune`` just spent time
    verifying actually fits, since free VRAM can shift slightly between
    the two calls and total_vram (the only thing the signature buckets
    on) never does.
    """
    from ldm_patched.modules import model_management
    from jax_pipeline import unet_segments as segs
    from jax_pipeline import engine_cache
    from jax_pipeline import _debug_profile
    from jax_pipeline.autotune import device_name_for

    device_name = device_name_for(jax_device)
    total_vram = model_management.get_total_memory(torch_device_for_vram_query)
    signature = engine_cache.compute_signature(
        "unet_segmentation", device_name, total_vram, latent_h, latent_w, batch, seq_len,
    )
    atoms = segs.build_atom_specs(latent_h, latent_w)

    if vram_budget_override is None:
        entry = engine_cache.lookup_exact(signature)
        if entry is not None:
            decision = entry.get("decision", {})
            ranges = tuple(tuple(r) for r in decision.get("ranges", []))
            spill_schedule = {int(k): tuple(v) for k, v in decision.get("spill_schedule", {}).items()}
            cache_budget_bytes = decision.get("cache_budget_bytes")
            if (
                ranges and ranges[-1][1] == len(atoms)  # sanity: plan must cover exactly this atom count
                and cache_budget_bytes is not None  # schema_version bump (v3) guarantees this field exists
            ):
                _debug_profile.note(
                    f"segment_search: reusing already-verified plan from engine_db "
                    f"(sig={signature}) -- {len(ranges)} segments, "
                    f"cache_budget={cache_budget_bytes/1e9:.3f}GB, skipping re-search"
                )
                return atoms, SegmentPlan(
                    ranges=ranges, spill_schedule=spill_schedule, cache_budget_bytes=cache_budget_bytes,
                )

    weight_bytes = segs.compute_weight_bytes(flat_params, atoms)

    shapes = segs.named_value_shapes(latent_h, latent_w)
    value_bytes = {vid: h * w * c * batch * dtype_bytes for vid, (h, w, c) in shapes.items()}

    activation_bytes, _traced = estimate_activation_bytes(
        atoms, flat_params, jax_device, torch_device_for_vram_query,
        device_name, total_vram, latent_h, latent_w, batch, seq_len,
    )

    if vram_budget_override is not None:
        raw_budget = vram_budget_override
    else:
        raw_budget = model_management.get_free_memory(torch_device_for_vram_query)
    usable_budget = max(raw_budget - _SAMPLER_RESERVE_BYTES, 0)

    total_weight_bytes = sum(weight_bytes.values())
    total_value_bytes = sum(value_bytes.values())
    max_atom_activation = max(activation_bytes.values()) if activation_bytes else 0
    _debug_profile.note(
        f"segment_search: shape={latent_h}x{latent_w} batch={batch} seq={seq_len} -- "
        f"{len(atoms)} atoms, sum(weight_bytes)={total_weight_bytes/1e9:.3f}GB "
        f"(sanity check: should be close to the UNet's total param footprint -- a much "
        f"SMALLER number here means partition_params_for_atoms silently dropped weights), "
        f"sum(named_value_bytes)={total_value_bytes/1e9:.3f}GB, "
        f"max_single_atom_activation={max_atom_activation/1e6:.1f}MB, "
        f"activation_source={'traced just now' if _traced else 'DB warm start'}, "
        f"raw_budget={raw_budget/1e9:.3f}GB minus {_SAMPLER_RESERVE_BYTES/1e9:.3f}GB reserved for "
        f"sampler-loop/denoiser/XLA overhead -> usable_budget_for_search={usable_budget/1e9:.3f}GB"
    )

    plan = greedy_best_fit_search(atoms, weight_bytes, activation_bytes, value_bytes, usable_budget)

    seg_sizes = [e - s for s, e in plan.ranges]
    _debug_profile.note(
        f"segment_search: greedy plan -- {len(plan.ranges)} segments over {len(atoms)} atoms "
        f"(sizes: min={min(seg_sizes) if seg_sizes else 0} max={max(seg_sizes) if seg_sizes else 0} "
        f"avg={sum(seg_sizes)/len(seg_sizes) if seg_sizes else 0:.1f}), "
        f"{sum(len(v) for v in plan.spill_schedule.values())} boundary spills across "
        f"{len(plan.spill_schedule)} boundaries"
    )

    engine_cache.save(
        "unet_segmentation", signature, device_name, total_vram, latent_h, latent_w, batch, seq_len,
        {
            "mode": "segmented",
            "ranges": [list(r) for r in plan.ranges],
            "spill_schedule": {str(k): list(v) for k, v in plan.spill_schedule.items()},
            "cache_budget_bytes": plan.cache_budget_bytes,
        },
        {"num_segments": len(plan.ranges), "num_atoms": len(atoms), "usable_budget_bytes": usable_budget},
    )
    log.info(
        "[JAX Pipeline] segment_search: %d atoms -> %d segments (%d boundary spills) for sig=%s",
        len(atoms), len(plan.ranges), sum(len(v) for v in plan.spill_schedule.values()), signature,
    )
    return atoms, plan
