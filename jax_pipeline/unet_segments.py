"""
jax_pipeline/unet_segments.py — atom-level IR + generic segment fusion
for the streaming UNet. Replaces the discrete "fine"/"coarse" fusion
levels (jax_pipeline.unet's unet_forward_streaming /
unet_forward_streaming_coarse) with an arbitrary, per-shape-optimal
contiguous segmentation found by jax_pipeline.segment_search's Best-Fit
-style algorithm.

Why a new IR instead of parameterizing the existing streaming
functions: "fine" and "coarse" are two FIXED points in a much larger
space of ways to group the UNet's ~129 elementary operations into
jitted calls. Searching that space requires every contiguous range of
operations — not just "one micro-block" or "one whole stage" — to be
fusable into a single jax.jit call on demand. This module provides that:
a flat, ordered sequence of ``AtomSpec``s (same granularity as
``jax_pipeline.unet.build_block_ids()``'s 129 fine blocks, reusing the
exact same raw math functions) plus a generic compiler that fuses any
contiguous range of them into one jitted function.

Static shapes, no runtime shape threading
------------------------------------------
Every atom is built for a SPECIFIC, KNOWN (latent_h, latent_w, batch,
seq_len) — same granularity jax_pipeline.autotune already keys its
signatures on. That means every intermediate tensor's shape in the
whole forward pass is known analytically at atom-construction time (the
UNet's 3 downsample stages are a fixed stride-2/pad-1/kernel-3 conv, so
each level's (H, W) is a pure function of the input latent's (H, W) —
see ``_downsample_out``). Two things the original ``unet_forward``
derives from a live tensor's ``.shape`` at runtime are therefore baked
in as compile-time Python constants here instead of threaded as dynamic
values:
  * each ``upsample_2d`` call's ``target_hw`` (needed only for latent
    dims not divisible by 8 — see unet.py's ``upsample_2d`` docstring),
  * each attention residual's ``(H, W, C)`` reshape target.
This keeps the only genuinely dynamic cross-atom state to the tensors
that actually flow across a boundary the search might cut:

  * SKIP CONNECTIONS — 9 total (conv_in's output + each down-block
    resnet/downsample output), produced during the down path, consumed
    LIFO during the up path. A skip produced early (conv_in's) can stay
    pending across many atoms/segments before up_blocks.2 finally
    consumes it.
  * ATTENTION RESIDUALS — each attention sub-stage's norm+proj_in atom
    stashes the pre-norm ``hidden_states`` for the final residual add;
    its own proj_out atom (possibly several transformer_block atoms
    later, if a segment boundary falls in between) consumes it.

Both are modeled identically as named entries in a ``ctx["values"]``
dict — the segment search doesn't need to special-case skips vs
residuals; it just decides, at each segment boundary, which pending
entries to keep device-resident vs spill to pinned host (the "boundary
storage" feature — see ``PendingValueStore`` below, which owns that
device<->host movement, mirroring ``jax_pipeline.block_cache.
BlockParamCache``'s pattern but for a handful of live activation
tensors instead of weights).

Segment compilation
--------------------
``compile_segment(atoms)`` builds ONE ``jax.jit`` call that threads a
plain dict ``ctx`` (``{"x", "temb", "encoder_hidden_states", "values"}``,
all arrays / dicts-of-arrays — valid JAX pytrees) through every atom in
the range via an ordinary Python loop, INSIDE the trace. Each atom
closes over its own static kwargs (``prefix``, ``num_heads``, target
shapes, ...) as plain Python constants rather than ``static_argnames``,
so fusing atoms never hits the "non-hashable static argument" class of
bug the original per-block ``_jit_*`` wrappers had to work around (see
unet.py's git history) — there simply are no jit-level static kwargs
here at all, only closure constants baked in once at atom-construction
time.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

log = logging.getLogger(__name__)

_BLOCK_OUT_CHANNELS = [320, 640, 1280]
_LAYERS_PER_BLOCK = 2
_TRANSFORMER_LAYERS = [1, 2, 10]
_DIM_PER_HEAD = 64
_TIME_DIM = 320
_ADD_TIME_EMBED_DIM = 256


def _num_heads(channels: int) -> int:
    return channels // _DIM_PER_HEAD


def _downsample_out(n: int) -> int:
    """Spatial size after one stride-2/pad-1/kernel-3 conv (unet.py's
    downsample_2d) — the exact formula XLA's conv will compute, so
    precomputing this analytically matches the runtime shape exactly,
    it does not approximate it.
    """
    return (n - 1) // 2 + 1


class AtomSpec(NamedTuple):
    """One elementary UNet op, in exact execution order.

    ``run(params_slice, ctx) -> ctx`` — ``params_slice`` is this atom's
    own sub-dict of the flat UNet params (``BlockParamCache.get(block_id)``,
    same source ``jax_pipeline.unet``'s existing streaming paths use).
    ``ctx`` is a plain dict with keys "x" (running activation), "temb",
    "encoder_hidden_states" (both constant through one forward pass),
    and "values" (dict of pending named values — skips / residuals).
    """
    block_id: str
    kind: str
    run: Any
    produces: Tuple[str, ...] = ()
    consumes: Tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
#  Atom constructors — reuse jax_pipeline.unet's raw (non-jitted) functions
# ═══════════════════════════════════════════════════════════════════════════════

def _make_time_embedding_atom() -> AtomSpec:
    from jax_pipeline.unet import timestep_embedding_mlp

    def run(params, ctx):
        ctx["temb"] = timestep_embedding_mlp(params, "time_embedding", ctx["timestep_emb"])
        return ctx

    return AtomSpec("time_embedding", "time_embedding", run)


def _make_add_embedding_atom() -> AtomSpec:
    from jax_pipeline.unet import add_embedding

    def run(params, ctx):
        aug_emb = add_embedding(
            params, "add_embedding", ctx["text_embeds"], ctx["time_ids"], _ADD_TIME_EMBED_DIM,
        ).astype(ctx["temb"].dtype)
        ctx["temb"] = ctx["temb"] + aug_emb
        return ctx

    return AtomSpec("add_embedding", "add_embedding", run)


def _make_conv_atom(block_id: str, stride: int, padding: int) -> AtomSpec:
    from jax_pipeline.unet import conv2d

    def run(params, ctx):
        ctx["x"] = conv2d(params, block_id, ctx["x"], stride=stride, padding=padding)
        return ctx

    return AtomSpec(block_id, "conv", run)


def _make_resnet_atom(block_id: str) -> AtomSpec:
    from jax_pipeline.unet import resnet_block

    def run(params, ctx):
        ctx["x"] = resnet_block(params, block_id, ctx["x"], ctx["temb"])
        return ctx

    return AtomSpec(block_id, "resnet", run)


def _make_downsample_atom(block_id: str) -> AtomSpec:
    from jax_pipeline.unet import downsample_2d

    def run(params, ctx):
        ctx["x"] = downsample_2d(params, block_id, ctx["x"])
        return ctx

    return AtomSpec(block_id, "downsample", run)


def _make_upsample_atom(block_id: str, target_hw: Optional[Tuple[int, int]]) -> AtomSpec:
    from jax_pipeline.unet import upsample_2d

    def run(params, ctx):
        ctx["x"] = upsample_2d(params, block_id, ctx["x"], target_hw=target_hw)
        return ctx

    return AtomSpec(block_id, "upsample", run)


def _make_attn_norm_proj_in_atom(prefix: str, residual_id: str) -> AtomSpec:
    from jax_pipeline.unet import group_norm, linear

    norm_id = f"{prefix}.norm"
    proj_in_id = f"{prefix}.proj_in"

    def run(params, ctx):
        hidden_states = ctx["x"]
        B, H, W, C = hidden_states.shape
        h = group_norm(params, norm_id, hidden_states)
        h = h.reshape(B, H * W, C)
        h = linear(params, proj_in_id, h)
        ctx["x"] = h
        ctx["values"][residual_id] = hidden_states
        return ctx

    return AtomSpec(f"{prefix}.norm_proj_in", "attn_norm_proj_in", run, produces=(residual_id,))


def _make_transformer_block_atom(block_id: str, num_heads: int) -> AtomSpec:
    from jax_pipeline.unet import basic_transformer_block

    def run(params, ctx):
        ctx["x"] = basic_transformer_block(params, block_id, ctx["x"], ctx["encoder_hidden_states"], num_heads)
        return ctx

    return AtomSpec(block_id, "transformer_block", run)


def _make_attn_proj_out_atom(prefix: str, residual_id: str, H: int, W: int, C: int) -> AtomSpec:
    from jax_pipeline.unet import linear

    proj_out_id = f"{prefix}.proj_out"

    def run(params, ctx):
        residual = ctx["values"].pop(residual_id)
        h = linear(params, proj_out_id, ctx["x"])
        B = h.shape[0]
        h = h.reshape(B, H, W, C)
        ctx["x"] = h + residual
        return ctx

    return AtomSpec(f"{prefix}.proj_out", "attn_proj_out", run, consumes=(residual_id,))


def _make_norm_out_atom() -> AtomSpec:
    from jax_pipeline.unet import group_norm, _silu

    def run(params, ctx):
        ctx["x"] = _silu(group_norm(params, "conv_norm_out", ctx["x"]))
        return ctx

    return AtomSpec("conv_norm_out", "norm_silu", run)


def _with_skip_push(inner: AtomSpec, skip_id: str) -> AtomSpec:
    """Wrap an atom so it also pushes its OWN output (``ctx["x"]``) as a
    named skip value after running.
    """
    def run(params, ctx):
        ctx = inner.run(params, ctx)
        ctx["values"][skip_id] = ctx["x"]
        return ctx

    return AtomSpec(inner.block_id, inner.kind, run, produces=inner.produces + (skip_id,), consumes=inner.consumes)


def _with_skip_concat(inner: AtomSpec, skip_id: str) -> AtomSpec:
    """Wrap an atom so it FIRST concatenates a pending named skip value
    onto ``ctx["x"]`` before running — used for each up-block loop
    iteration's first sub-op (always a resnet).
    """
    import jax.numpy as jnp

    def run(params, ctx):
        skip_val = ctx["values"].pop(skip_id)
        ctx["x"] = jnp.concatenate([ctx["x"], skip_val], axis=-1)
        return inner.run(params, ctx)

    return AtomSpec(inner.block_id, inner.kind, run, produces=inner.produces, consumes=inner.consumes + (skip_id,))


# ═══════════════════════════════════════════════════════════════════════════════
#  Full atom sequence — mirrors jax_pipeline.unet.unet_forward exactly
# ═══════════════════════════════════════════════════════════════════════════════

def _transformer_2d_atoms(prefix: str, num_layers: int, num_heads: int, H: int, W: int, C: int) -> List[AtomSpec]:
    residual_id = f"_residual::{prefix}"
    atoms = [_make_attn_norm_proj_in_atom(prefix, residual_id)]
    for i in range(num_layers):
        atoms.append(_make_transformer_block_atom(f"{prefix}.transformer_blocks.{i}", num_heads))
    atoms.append(_make_attn_proj_out_atom(prefix, residual_id, H, W, C))
    return atoms


def build_atom_specs(latent_h: int, latent_w: int) -> List[AtomSpec]:
    """Build the full ~129-atom sequence for one specific input latent
    resolution — exact execution order, exact math, matching
    ``jax_pipeline.unet.unet_forward``/``build_block_ids()``.

    ``latent_h``/``latent_w`` are needed only to precompute the static
    shape constants described in the module docstring (upsample
    target_hw, attention residual reshape dims) — nothing about the
    weights or the math depends on them; the SAME atom list works for
    any batch/seq_len at this (H, W).
    """
    ch = _BLOCK_OUT_CHANNELS
    H0, W0 = latent_h, latent_w
    H1, W1 = _downsample_out(H0), _downsample_out(W0)
    H2, W2 = _downsample_out(H1), _downsample_out(W1)

    atoms: List[AtomSpec] = [
        _make_time_embedding_atom(),
        _make_add_embedding_atom(),
    ]

    conv_in = _make_conv_atom("conv_in", stride=1, padding=1)
    atoms.append(_with_skip_push(conv_in, "skip_0"))

    # down_blocks.0 — pure resnet, no attention. H0,W0 -> H0,W0 -> (downsample) H1,W1
    atoms.append(_with_skip_push(_make_resnet_atom("down_blocks.0.resnets.0"), "skip_1"))
    atoms.append(_with_skip_push(_make_resnet_atom("down_blocks.0.resnets.1"), "skip_2"))
    atoms.append(_with_skip_push(_make_downsample_atom("down_blocks.0.downsamplers.0"), "skip_3"))

    # down_blocks.1 — cross-attn, 2 iters, H1,W1 -> (downsample) H2,W2
    n1 = _num_heads(ch[1])
    atoms.append(_make_resnet_atom("down_blocks.1.resnets.0"))
    atoms += _with_skip_push_last(_transformer_2d_atoms(
        "down_blocks.1.attentions.0", _TRANSFORMER_LAYERS[1], n1, H1, W1, ch[1]), "skip_4")
    atoms.append(_make_resnet_atom("down_blocks.1.resnets.1"))
    atoms += _with_skip_push_last(_transformer_2d_atoms(
        "down_blocks.1.attentions.1", _TRANSFORMER_LAYERS[1], n1, H1, W1, ch[1]), "skip_5")
    atoms.append(_with_skip_push(_make_downsample_atom("down_blocks.1.downsamplers.0"), "skip_6"))

    # down_blocks.2 — cross-attn, 2 iters, no downsample. Stays at H2,W2
    n2 = _num_heads(ch[2])
    atoms.append(_make_resnet_atom("down_blocks.2.resnets.0"))
    atoms += _with_skip_push_last(_transformer_2d_atoms(
        "down_blocks.2.attentions.0", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2]), "skip_7")
    atoms.append(_make_resnet_atom("down_blocks.2.resnets.1"))
    atoms += _with_skip_push_last(_transformer_2d_atoms(
        "down_blocks.2.attentions.1", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2]), "skip_8")

    # mid_block — resnet, transformer, resnet. Stays at H2,W2
    atoms.append(_make_resnet_atom("mid_block.resnets.0"))
    atoms += _transformer_2d_atoms("mid_block.attentions.0", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2])
    atoms.append(_make_resnet_atom("mid_block.resnets.1"))

    # up_blocks.0 — consumes skip_8, skip_7, skip_6 (LIFO); upsamples H2,W2 -> H1,W1
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.0.resnets.0"), "skip_8"))
    atoms += _transformer_2d_atoms("up_blocks.0.attentions.0", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2])
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.0.resnets.1"), "skip_7"))
    atoms += _transformer_2d_atoms("up_blocks.0.attentions.1", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2])
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.0.resnets.2"), "skip_6"))
    atoms += _transformer_2d_atoms("up_blocks.0.attentions.2", _TRANSFORMER_LAYERS[2], n2, H2, W2, ch[2])
    atoms.append(_make_upsample_atom("up_blocks.0.upsamplers.0", target_hw=(H1, W1)))

    # up_blocks.1 — consumes skip_5, skip_4, skip_3; upsamples H1,W1 -> H0,W0
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.1.resnets.0"), "skip_5"))
    atoms += _transformer_2d_atoms("up_blocks.1.attentions.0", _TRANSFORMER_LAYERS[1], n1, H1, W1, ch[1])
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.1.resnets.1"), "skip_4"))
    atoms += _transformer_2d_atoms("up_blocks.1.attentions.1", _TRANSFORMER_LAYERS[1], n1, H1, W1, ch[1])
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.1.resnets.2"), "skip_3"))
    atoms += _transformer_2d_atoms("up_blocks.1.attentions.2", _TRANSFORMER_LAYERS[1], n1, H1, W1, ch[1])
    atoms.append(_make_upsample_atom("up_blocks.1.upsamplers.0", target_hw=(H0, W0)))

    # up_blocks.2 — pure resnet, consumes skip_2, skip_1, skip_0; no upsample
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.2.resnets.0"), "skip_2"))
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.2.resnets.1"), "skip_1"))
    atoms.append(_with_skip_concat(_make_resnet_atom("up_blocks.2.resnets.2"), "skip_0"))

    atoms.append(_make_norm_out_atom())
    atoms.append(_make_conv_atom("conv_out", stride=1, padding=1))
    return atoms


def _with_skip_push_last(atom_list: List[AtomSpec], skip_id: str) -> List[AtomSpec]:
    """Attach a skip-push to the LAST atom of an already-built list
    (used for cross-attn down-block iterations, where the skip is
    pushed after the whole attention sub-stack, not after the resnet).
    """
    return atom_list[:-1] + [_with_skip_push(atom_list[-1], skip_id)]


def named_value_shapes(latent_h: int, latent_w: int) -> Dict[str, Tuple[int, int, int]]:
    """(H, W, C) for every named value ``build_atom_specs`` can produce,
    at this latent resolution — batch-independent, so
    ``jax_pipeline.segment_search`` can derive exact byte sizes as
    ``H*W*C*batch*dtype_size`` without tracing (shapes are architecture-
    determined, not data-dependent — see the module docstring). Must stay
    in sync with ``build_atom_specs``'s producer sites; there is no
    automatic cross-check the way ``block_cache.partition_params_by_block``
    checks weight coverage, so a drift here would silently mis-estimate
    footprint (not corrupt output — the actual arrays are still whatever
    the atoms really produced) rather than crash.
    """
    ch = _BLOCK_OUT_CHANNELS
    H0, W0 = latent_h, latent_w
    H1, W1 = _downsample_out(H0), _downsample_out(W0)
    H2, W2 = _downsample_out(H1), _downsample_out(W1)
    shapes = {
        "skip_0": (H0, W0, ch[0]), "skip_1": (H0, W0, ch[0]), "skip_2": (H0, W0, ch[0]),
        "skip_3": (H1, W1, ch[0]),
        "skip_4": (H1, W1, ch[1]), "skip_5": (H1, W1, ch[1]),
        "skip_6": (H2, W2, ch[1]),
        "skip_7": (H2, W2, ch[2]), "skip_8": (H2, W2, ch[2]),
    }
    for prefix in ("down_blocks.1.attentions.0", "down_blocks.1.attentions.1"):
        shapes[f"_residual::{prefix}"] = (H1, W1, ch[1])
    for prefix in (
        "down_blocks.2.attentions.0", "down_blocks.2.attentions.1", "mid_block.attentions.0",
        "up_blocks.0.attentions.0", "up_blocks.0.attentions.1", "up_blocks.0.attentions.2",
    ):
        shapes[f"_residual::{prefix}"] = (H2, W2, ch[2])
    for prefix in ("up_blocks.1.attentions.0", "up_blocks.1.attentions.1", "up_blocks.1.attentions.2"):
        shapes[f"_residual::{prefix}"] = (H1, W1, ch[1])
    return shapes


def partition_params_for_atoms(flat_params: Dict[str, Any], atoms: List[AtomSpec]) -> Dict[str, Dict[str, Any]]:
    """Partition a flat ``{hf_key: array}`` params dict into per-atom
    sub-dicts, keyed by each atom's ``block_id`` — same strict-coverage
    philosophy as ``jax_pipeline.block_cache.partition_params_by_block``
    (raise loudly on any unassigned key rather than silently drop
    weights), specialized for this module's block_id conventions: most
    atoms' block_id is a literal prefix of its own params' HF keys, but
    attention atoms use ``"{prefix}.norm_proj_in"``/``"{prefix}.proj_out"``
    as their block_id (a labeling choice for per-atom timing/footprint
    bookkeeping), which don't literally prefix-match their actual HF key
    paths (``"{prefix}.norm.*"``/``"{prefix}.proj_in.*"`` and
    ``"{prefix}.proj_out.*"`` respectively) the way every other atom's
    does.
    """
    by_block: Dict[str, Dict[str, Any]] = {}
    claimed = set()
    for atom in atoms:
        bid = atom.block_id
        if bid in by_block:
            continue
        if bid.endswith(".norm_proj_in"):
            prefix = bid[: -len(".norm_proj_in")]
            keys = [f"{prefix}.norm.weight", f"{prefix}.norm.bias", f"{prefix}.proj_in.weight", f"{prefix}.proj_in.bias"]
        elif bid.endswith(".proj_out"):
            keys = [f"{bid}.weight", f"{bid}.bias"]
        else:
            keys = [k for k in flat_params if k == f"{bid}.weight" or k == f"{bid}.bias" or k.startswith(bid + ".")]
        sub = {}
        for k in keys:
            if k in flat_params:
                sub[k] = flat_params[k]
                claimed.add(k)
        by_block[bid] = sub

    unclaimed = set(flat_params.keys()) - claimed
    # "class_embedding.*" is real-but-permanently-unused duplicate weight
    # data (see jax_pipeline.unet.build_block_ids's docstring) -- every
    # atom here only ever reads add_embedding.*, so it's expected to be
    # unclaimed. Anything else unclaimed is a real coverage bug.
    unexpected = sorted(k for k in unclaimed if not k.startswith("class_embedding."))
    if unexpected:
        raise ValueError(f"partition_params_for_atoms: unassigned param keys: {unexpected[:10]}"
                          f"{'...' if len(unexpected) > 10 else ''}")
    return by_block


def compute_weight_bytes(flat_params: Dict[str, Any], atoms: List[AtomSpec]) -> Dict[str, int]:
    """Exact (no tracing needed) weight footprint per atom block_id."""
    by_block = partition_params_for_atoms(flat_params, atoms)
    return {bid: sum(int(v.nbytes) for v in sub.values()) for bid, sub in by_block.items()}


def block_ids_in_order(atoms: List[AtomSpec]) -> List[str]:
    """Unique block_ids in first-appearance order — for building/
    partitioning a BlockParamCache the same way build_block_ids() does.
    """
    seen: Dict[str, None] = {}
    for a in atoms:
        seen.setdefault(a.block_id, None)
    return list(seen.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  Boundary storage — spill pending named values (skips / residuals) to
#  pinned host memory between segments, freeing device VRAM for atoms
#  further down the sequence at the cost of an extra transfer both ways.
# ═══════════════════════════════════════════════════════════════════════════════

class PendingValueStore:
    """Owns every named value (skip / attention residual) that has been
    produced by an earlier segment but not yet consumed by a later one.
    Mirrors ``jax_pipeline.block_cache.BlockParamCache``'s device<->
    pinned-host movement pattern, applied to a handful of small live
    activation tensors instead of the whole weight pytree.

    Values are always PRODUCED on-device (they're a jitted segment's own
    output). ``spill(value_id)`` is the boundary-storage decision the
    segment search makes to fit more atoms per segment without
    exceeding the VRAM budget; ``get(value_id)`` transparently re-stages
    a spilled value back to device on demand (the caller doesn't need to
    know whether it was spilled).
    """

    def __init__(self, device):
        self._device = device
        self._entries: Dict[str, Any] = {}
        self._on_host: Dict[str, bool] = {}

    def put(self, value_id: str, array) -> None:
        self._entries[value_id] = array
        self._on_host[value_id] = False

    def spill(self, value_id: str) -> None:
        import jax

        if self._on_host.get(value_id):
            return
        sharding = jax.sharding.SingleDeviceSharding(self._device, memory_kind="pinned_host")
        old = self._entries[value_id]
        self._entries[value_id] = jax.device_put(old, sharding)
        old.delete()
        self._on_host[value_id] = True

    def get(self, value_id: str):
        import jax

        arr = self._entries.pop(value_id)
        on_host = self._on_host.pop(value_id)
        if on_host:
            sharding = jax.sharding.SingleDeviceSharding(self._device, memory_kind="device")
            staged = jax.device_put(arr, sharding)
            arr.delete()
            return staged
        return arr

    def pending_ids(self) -> List[str]:
        return list(self._entries.keys())

    def device_resident_ids(self) -> List[str]:
        return [vid for vid, on_host in self._on_host.items() if not on_host]

    def nbytes(self, value_id: str) -> int:
        return int(self._entries[value_id].nbytes)

    def clear(self) -> None:
        for arr in self._entries.values():
            try:
                arr.delete()
            except Exception:
                pass
        self._entries.clear()
        self._on_host.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Segment compiler + driver
# ═══════════════════════════════════════════════════════════════════════════════

def compile_segment(atoms: List[AtomSpec]):
    """Fuse a contiguous range of atoms into ONE jax.jit call. Every
    atom's static kwargs (prefix, num_heads, target_hw, reshape dims,
    ...) are already baked in as closure constants at atom-construction
    time (see the constructors above) — this function itself needs no
    static_argnames at all, sidestepping the whole "positional arg
    silently fills a static slot" class of bug the per-block fine/coarse
    jitted wrappers had to work around explicitly.
    """
    import jax

    def segment_fn(params_by_block, ctx):
        for atom in atoms:
            ctx = atom.run(params_by_block[atom.block_id], ctx)
        return ctx

    return jax.jit(segment_fn)


class CompiledSegment(NamedTuple):
    atoms: List[AtomSpec]
    block_ids: List[str]
    fn: Any  # jitted segment_fn(params_by_block, ctx) -> ctx


def compile_segments(atom_groups: List[List[AtomSpec]]) -> List[CompiledSegment]:
    return [CompiledSegment(atoms, block_ids_in_order(atoms), compile_segment(atoms)) for atoms in atom_groups]


class SegmentedUnetState:
    """Bundles everything ``unet_forward_segmented`` needs behind the
    same single-``params``-argument calling convention
    ``jax_pipeline.pipeline.JAXSDXLPipeline.apply_model`` already uses
    for every other fusion level (``self._forward(self._params, sample,
    timestep, enc, added)``) — so wiring in the "segmented" level
    (jax_pipeline.segment_search's Best-Fit search) needs no change to
    that call site, only a new branch in ``_build_unet_execution`` that
    constructs one of these instead of a bare ``BlockParamCache``.

    ``evict_all_resident``/``clear`` give ``jax_pipeline.host_offload
    .PhaseManager`` the same two-tier cleanup hook every other streaming
    level's ``BlockParamCache`` already exposes, extended to also drop
    the ``PendingValueStore``'s in-flight skip/residual tensors (weights
    and boundary-storage activations are two independent pools of
    device memory, both need releasing on eviction/teardown).
    """
    def __init__(self, param_cache, pending_store: PendingValueStore, atoms: List[AtomSpec],
                 segments: List["CompiledSegment"], spill_schedule: Dict[int, Tuple[str, ...]]):
        self.param_cache = param_cache
        self.pending_store = pending_store
        self.atoms = atoms
        self.segments = segments
        self.spill_schedule = spill_schedule

    def evict_all_resident(self) -> None:
        self.param_cache.evict_all_resident()
        self.pending_store.clear()

    def clear(self) -> None:
        self.param_cache.clear()
        self.pending_store.clear()


def segmented_forward(state: SegmentedUnetState, sample, timestep, encoder_hidden_states, added_cond_kwargs):
    return unet_forward_segmented(
        state.param_cache, state.pending_store, state.segments, state.spill_schedule,
        sample, timestep, encoder_hidden_states, added_cond_kwargs,
    )


def unet_forward_segmented(
    param_cache,
    pending_store: PendingValueStore,
    segments: List[CompiledSegment],
    spill_schedule: Dict[int, Tuple[str, ...]],
    sample: "Any",
    timestep: "Any",
    encoder_hidden_states: "Any",
    added_cond_kwargs: Dict[str, Any],
):
    """Run the UNet forward pass as a sequence of fused segment calls.

    ``spill_schedule``: ``{segment_index: (value_id, ...)}`` — value_ids
    the search decided to spill to pinned host RIGHT AFTER that segment
    produces them (the "boundary storage" the search can use to fit
    more atoms into LATER segments without exceeding the VRAM budget).
    Re-staging back to device happens transparently in
    ``PendingValueStore.get`` whichever later segment consumes it.

    Not itself jit-wrapped — ``param_cache.get``/``pending_store.get``/
    ``.spill`` must run as plain eager Python between segment calls,
    exactly like ``unet_forward_streaming``'s block-level cache lookups.
    """
    import jax.numpy as jnp
    from jax_pipeline.unet import _sinusoidal_embedding, _TIME_DIM as _TD

    sample = jnp.transpose(sample, (0, 2, 3, 1))
    timestep_emb = _sinusoidal_embedding(
        timestep.astype(jnp.float32), _TD, flip_sin_to_cos=True
    ).astype(sample.dtype)
    text_embeds = added_cond_kwargs["text_embeds"].astype(sample.dtype)
    time_ids = added_cond_kwargs["time_ids"].astype(jnp.float32)

    ctx = {
        "x": sample,
        "temb": None,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep_emb": timestep_emb,
        "text_embeds": text_embeds,
        "time_ids": time_ids,
        "values": {},
    }

    for seg_idx, segment in enumerate(segments):
        produced_within = set()
        consumed_within = set()
        for atom in segment.atoms:
            produced_within.update(atom.produces)
            consumed_within.update(atom.consumes)
        # Only values consumed by this segment but produced by an EARLIER
        # segment need to come from the store -- anything produced AND
        # consumed within this same segment stays purely local to its own
        # trace (never needs to leave ctx at all).
        needed_from_store = consumed_within - produced_within
        for value_id in needed_from_store:
            ctx["values"][value_id] = pending_store.get(value_id)

        params_by_block = {bid: param_cache.get(bid) for bid in segment.block_ids}
        ctx = segment.fn(params_by_block, ctx)

        still_pending = dict(ctx["values"])
        ctx["values"] = {}
        for value_id, arr in still_pending.items():
            pending_store.put(value_id, arr)
        for value_id in spill_schedule.get(seg_idx, ()):
            if value_id in still_pending:
                pending_store.spill(value_id)

    return jnp.transpose(ctx["x"], (0, 3, 1, 2))
