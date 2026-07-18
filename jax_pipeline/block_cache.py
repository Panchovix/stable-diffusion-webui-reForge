"""
jax_pipeline/block_cache.py — Python-orchestrated, block-level LRU weight
cache for the JAX UNet.

Motivation (see the extensive VRAM-profiling trail in this repo's commit
history around 2026-07-19): compiling the whole SDXL UNet as ONE
``jax.jit`` program requires every one of its ~2.6B bf16 parameters
(~5-6GB, once XLA's own compilation/execution state is included) to
already be device-resident before the call — on an 8GB card that alone
left too little headroom for the forward pass's own activation memory,
consistently landing within a few hundred MB of the card's ceiling no
matter how aggressively everything ELSE (torch's redundant copies, CLIP,
VAE) was evicted first.

This mirrors ldm_patched's own PyTorch lowvram layer-streaming
(``model_patcher.py``'s ``move_weight_functions``, swapping each
``nn.Module``'s weights onto GPU just before its own ``forward()`` runs,
via forward hooks) — reimplemented for JAX, which has no stateful
modules or hook mechanism to hang a callback off. Instead,
``jax_pipeline.unet``'s streaming forward pass explicitly calls
``BlockParamCache.get(block_id)`` before each block-level jitted call,
and Python (not XLA) decides whether that block's weights are already
staged on device or need to come from pinned host — trading a real,
deliberate PCIe transfer cost (paid on every block, every denoising step,
since a full forward pass visits each block exactly once — there's no
intra-pass reuse to cache) for a bounded, small peak VRAM footprint.

Eviction is cheap: params are pure/immutable (no training here), so the
pinned-host copy is a permanent, unchanging source of truth. "Evicting" a
device-resident block is just ``jax.Array.delete()`` on its device
buffers — nothing needs to be written back; the next ``get()`` for that
block simply re-stages fresh copies from the still-intact host store.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)


def partition_params_by_block(
    params: Dict[str, "jnp.ndarray"], block_ids: List[str],
) -> Dict[str, Dict[str, "jnp.ndarray"]]:
    """Split a flat ``{absolute_hf_key: array}`` dict into
    ``{block_id: {absolute_hf_key: array}}`` sub-dicts, one per entry in
    ``block_ids`` — a key belongs to ``block_id`` when it equals it or
    starts with ``block_id + "."``.

    Deliberately strict: every key in ``params`` must be claimed by
    EXACTLY one block_id, and every block_id must claim at least one key.
    A block registry that's out of sync with the model's actual parameter
    keys (a typo'd prefix, a block the registry forgot) would otherwise
    silently drop weights from the computation instead of raising —
    silently-wrong image generation is a much worse failure mode than a
    loud error at pipeline-build time.
    """
    partitioned: Dict[str, Dict[str, "jnp.ndarray"]] = {bid: {} for bid in block_ids}
    # Longest-prefix-first so a more specific block_id is preferred over a
    # shorter one that could also technically match (defensive — the
    # actual registry is disjoint by construction, but this keeps the
    # function correct even if that ever changes).
    sorted_ids = sorted(block_ids, key=len, reverse=True)

    unassigned: List[str] = []
    for key, value in params.items():
        matched: Optional[str] = None
        for bid in sorted_ids:
            if key == bid or key.startswith(bid + "."):
                matched = bid
                break
        if matched is None:
            unassigned.append(key)
        else:
            partitioned[matched][key] = value

    if unassigned:
        raise ValueError(
            f"[JAX Pipeline] {len(unassigned)} UNet parameter key(s) not covered "
            f"by any streaming block_id (e.g. {unassigned[:5]}) — the block "
            f"registry (jax_pipeline.unet.build_block_ids) is out of sync with "
            f"the model's actual parameter keys."
        )
    empty = [bid for bid, sub in partitioned.items() if not sub]
    if empty:
        raise ValueError(
            f"[JAX Pipeline] {len(empty)} streaming block_id(s) matched ZERO "
            f"parameter keys (e.g. {empty[:5]}) — likely a typo'd prefix in "
            f"jax_pipeline.unet.build_block_ids."
        )
    return partitioned


class BlockParamCache:
    """LRU cache of block-level UNet param sub-dicts, bounding how much
    GPU-device-resident weight data exists at any moment.

    Every block's params live PERMANENTLY in pinned-host memory (the
    single source of truth) once ``load()`` is called. ``get(block_id)``
    stages that block's params onto device on demand — a no-op if
    already resident — evicting the least-recently-used OTHER resident
    block(s) first if that would exceed ``budget_bytes``.
    """

    #: Default budget when none is given: enough for a few of the
    #: largest single block to be simultaneously resident — small and
    #: deliberately conservative (favors VRAM headroom over cache hit
    #: rate), matching the whole point of this cache.
    _DEFAULT_BUDGET_BLOCK_MULTIPLE = 3

    def __init__(self, device, budget_bytes: Optional[int] = None):
        self.device = device
        self.budget_bytes = budget_bytes  # resolved to a concrete value in load() if None
        self._host_store: Dict[str, Dict[str, "jnp.ndarray"]] = {}
        self._host_bytes: Dict[str, int] = {}
        self._resident: "OrderedDict[str, Dict[str, jnp.ndarray]]" = OrderedDict()
        self._resident_bytes = 0

    def load(self, block_params: Dict[str, Dict[str, "jnp.ndarray"]]) -> None:
        """(Re)populate the pinned-host store from freshly partitioned
        params (see ``partition_params_by_block``) — call at pipeline
        build time and again after any LoRA reload. Drops whatever was
        previously resident/hosted first (``clear()``), so a reload never
        leaves stale arrays reachable.
        """
        import jax

        self.clear()
        pinned_sharding = jax.sharding.SingleDeviceSharding(self.device, memory_kind="pinned_host")
        for block_id, params in block_params.items():
            hosted = jax.tree.map(lambda p: jax.device_put(p, pinned_sharding), params)
            nbytes = sum(int(v.nbytes) for v in hosted.values())
            self._host_store[block_id] = hosted
            self._host_bytes[block_id] = nbytes

        if self.budget_bytes is None:
            largest = max(self._host_bytes.values()) if self._host_bytes else 0
            self.budget_bytes = largest * self._DEFAULT_BUDGET_BLOCK_MULTIPLE

        log.info(
            "[JAX Pipeline] UNet block cache loaded: %d blocks, total=%.2fGB, "
            "budget=%.2fGB (largest single block=%.2fMB)",
            len(self._host_store), sum(self._host_bytes.values()) / 1e9,
            self.budget_bytes / 1e9, (max(self._host_bytes.values()) if self._host_bytes else 0) / 1e6,
        )

    def clear(self) -> None:
        """Delete every device-resident block and drop all host-store
        references. Used before ``load()`` replaces everything (LoRA
        reload) and by ``jax_pipeline.host_offload.PhaseManager``'s
        checkpoint-switch teardown (``force_release_all``).
        """
        self.evict_all_resident()
        self._host_store.clear()
        self._host_bytes.clear()

    def evict_all_resident(self) -> None:
        """Delete every currently device-resident block WITHOUT touching
        the pinned-host store — the cheap, recoverable eviction used by
        event-driven cleanup (PyTorch escape, JAX OOM cleanup; see
        ``host_offload.PhaseManager.force_offload_all``). The next
        ``get(block_id)`` simply re-stages from the still-intact host
        copy.
        """
        for params in self._resident.values():
            for p in params.values():
                if hasattr(p, "delete"):
                    try:
                        p.delete()
                    except Exception:
                        pass
        self._resident.clear()
        self._resident_bytes = 0

    def get(self, block_id: str) -> Dict[str, "jnp.ndarray"]:
        if block_id in self._resident:
            self._resident.move_to_end(block_id)  # mark most-recently-used
            return self._resident[block_id]

        import jax

        needed = self._host_bytes[block_id]
        self._evict_until_room(needed)

        device_sharding = jax.sharding.SingleDeviceSharding(self.device, memory_kind="device")
        device_params = jax.tree.map(lambda p: jax.device_put(p, device_sharding), self._host_store[block_id])
        self._resident[block_id] = device_params
        self._resident_bytes += needed
        return device_params

    def _evict_until_room(self, needed_bytes: int) -> None:
        budget = self.budget_bytes if self.budget_bytes is not None else needed_bytes
        while self._resident and self._resident_bytes + needed_bytes > budget:
            victim_id, victim_params = self._resident.popitem(last=False)  # oldest = LRU
            for p in victim_params.values():
                if hasattr(p, "delete"):
                    try:
                        p.delete()
                    except Exception:
                        pass
            self._resident_bytes -= self._host_bytes[victim_id]

    def total_bytes(self) -> int:
        return sum(self._host_bytes.values())
