"""
diff_pipeline/vram_allocator.py — Generalized VRAM pool allocator.

Treats GPU VRAM as a pool/heap.  Every significant GPU memory consumer
(UNet blocks, VAE, CLIP, latent tensors) registers as an *Allocation*.

Eviction policy — generation-based adaptive LRU
------------------------------------------------
Each allocation carries a 4-state generation counter (0 = COLD … 3 = HOT):

  * On any ``activate()`` that moves a block from CPU → GPU the block's
    generation is incremented mod GEN_STATES:
      gen = (gen + 1) % GEN_STATES
    This means a block that wraps from HOT (3) back to COLD (0) has been
    loaded four times in quick succession, signalling a thrashing victim
    that should eventually yield to fresher data.

  * After each individual eviction, every *resident* block's generation is
    decremented by 1 (clamped at GEN_COLD = 0).  Blocks that are not
    accessed across many eviction cycles drift toward COLD and become
    higher-priority eviction candidates.

  * Eviction priority order (lowest-priority-to-evict last):
      1.  Free-list  (gen = GEN_FREE = -1): module GC'd or explicitly dead
      2.  GEN_COLD (0) — LRU order within tier
      3.  GEN_COOL (1) — LRU order within tier
      4.  GEN_WARM (2) — LRU order within tier
      5.  GEN_HOT  (3) — LRU order within tier (last resort)

Free list
---------
Allocations whose backing ``nn.Module`` has been detected as orphaned
(weakref finalizer fired) or that are explicitly marked via
``mark_free(name)`` are placed in the free list (gen = GEN_FREE) and
evicted first on any capacity pressure.

Unmanaged tensor tracking (TorchDispatchMode)
---------------------------------------------
``VRAMTrackerDispatch`` is a ``TorchDispatchMode`` subclass that can be
used as a context manager via ``VRAMAllocator.tracking_context()``.

Within the context, every CUDA tensor allocation above ``_TRACK_BYTES``
(1 MiB) is registered with a ``weakref`` finalizer.  When Python GC
collects the tensor the reservation is automatically released, so
``_effective_free()`` reflects live activations in real time rather than
relying solely on a static ``inference_headroom`` estimate.

    with allocator.tracking_context():
        output = unet(x, t, encoder_hidden_states=enc)  # activations tracked

Managed ``.to("cuda")`` calls inside ``activate()`` are excluded from
tracking via the ``_suppressing_tracking`` flag so parameter tensors are
not double-counted.

Singleton per device
--------------------
Use ``VRAMAllocator.get(device)`` — one instance per CUDA device is
shared across ``DiffPipeline`` and ``DiffusersModelAdapter``.
"""

from __future__ import annotations

import contextlib
import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Defragment when inactive_split / reserved exceeds this ratio.
FRAG_THRESHOLD: float = 0.25

# Empirical CUDA page-alignment overhead: numel*element_size underestimates
# actual VRAM by ~15% due to 2 MiB block boundaries.
_CUDA_OVERHEAD: float = 1.15

# Generation states — 4-state RRIP-inspired adaptive eviction counter.
GEN_FREE: int = -1   # orphaned / explicitly dead → evict first
GEN_COLD: int =  0   # not accessed across many eviction cycles
GEN_COOL: int =  1
GEN_WARM: int =  2
GEN_HOT:  int =  3   # recently loaded / accessed
GEN_STATES: int = 4  # active states used in the (gen+1)%GEN_STATES wrap

# Minimum tensor size tracked by VRAMTrackerDispatch.
_TRACK_BYTES: int = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# TorchDispatchMode — unmanaged tensor tracking
# ---------------------------------------------------------------------------

try:
    from torch.utils._python_dispatch import TorchDispatchMode as _TorchDispatchMode
    _HAS_DISPATCH_MODE = True
except ImportError:
    _TorchDispatchMode = object  # type: ignore[assignment,misc]
    _HAS_DISPATCH_MODE = False


class VRAMTrackerDispatch(_TorchDispatchMode):  # type: ignore[misc]
    """Intercept large CUDA tensor allocations and track them via weakref.

    When the tensor is freed by Python GC the corresponding ``_effective_free``
    reservation is automatically released, giving the allocator real-time
    accounting of live activations instead of a static estimate.

    Only used while ``VRAMAllocator.tracking_context()`` is active.
    Allocations during managed ``module.to(device)`` calls are suppressed
    via ``_suppressing_tracking`` to avoid double-counting parameters.
    """

    def __init__(self, allocator: "VRAMAllocator") -> None:
        if _HAS_DISPATCH_MODE:
            super().__init__()
        self._allocator = allocator

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
        out = func(*args, **(kwargs or {}))
        if not self._allocator._suppressing_tracking:
            try:
                self._register(out)
            except Exception:
                pass
        return out

    def _register(self, result: Any) -> None:
        """Walk result tensors and register large new CUDA allocations."""
        if isinstance(result, torch.Tensor):
            batch = (result,)
        elif isinstance(result, (list, tuple)):
            batch = tuple(r for r in result if isinstance(r, torch.Tensor))
        else:
            return

        alloc = self._allocator
        for t in batch:
            if not (t.is_cuda and t.device == alloc._device):
                continue
            nbytes = t.nbytes
            if nbytes < _TRACK_BYTES:
                continue
            # Use the data pointer as a cheap unique key per storage.
            # Views sharing storage share a data_ptr, so subsequent views
            # are skipped (already tracked).
            key = t.data_ptr()
            if key in alloc._tracked_tensor_refs:
                continue

            res_key = f"_tk_{key}"
            alloc._reservations[res_key] = nbytes

            def _on_free(ref, _k=key, _rk=res_key, _a=alloc):
                _a._reservations.pop(_rk, None)
                _a._tracked_tensor_refs.pop(_k, None)

            try:
                ref = weakref.ref(t, _on_free)
                alloc._tracked_tensor_refs[key] = ref
            except TypeError:
                # Tensor not weakref-able (rare; skip silently).
                alloc._reservations.pop(res_key, None)


# ---------------------------------------------------------------------------
# Allocation descriptor
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """Descriptor for a single managed VRAM allocation."""

    name: str
    load_fn: Callable[[], None]
    evict_fn: Callable[[], None]
    size_fn: Callable[[], int]
    pinned: bool = False
    on_device: bool = False
    gen: int = GEN_HOT           # generation counter: higher = warmer
    in_free_list: bool = False   # True when backing object is orphaned / dead
    _hook_handles: List[Any] = field(default_factory=list, repr=False)
    _module_ref: Any = field(default=None, repr=False)  # weakref to module

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()


# ---------------------------------------------------------------------------
# VRAMAllocator
# ---------------------------------------------------------------------------

class VRAMAllocator:
    """VRAM pool allocator with generation-based adaptive LRU eviction.

    Use ``VRAMAllocator.get(device)`` — one shared instance per CUDA device.
    """

    _instances: Dict[str, "VRAMAllocator"] = {}

    def __init__(self, device: torch.device) -> None:
        self._device = device
        # LRU-ordered residents: insertion order = recency (newest at end).
        self._lru: OrderedDict[str, Allocation] = OrderedDict()
        # All registered allocations (resident or not).
        self._registry: Dict[str, Allocation] = {}
        # Byte reservations for unmanaged on-device memory.
        # Key "_tk_<ptr>" entries are managed by VRAMTrackerDispatch.
        self._reservations: Dict[str, int] = {}
        # Keep weakrefs alive so finalizers fire when tensors are GC'd.
        self._tracked_tensor_refs: Dict[int, "weakref.ref[torch.Tensor]"] = {}
        # Suppresses VRAMTrackerDispatch tracking during module .to() calls.
        self._suppressing_tracking: bool = False
        log.info("VRAMAllocator: created for device %s", device)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, device: torch.device) -> "VRAMAllocator":
        """Return the shared VRAMAllocator for *device*, creating if needed."""
        key = f"{device.type}:{device.index if device.index is not None else 0}"
        if key not in cls._instances:
            cls._instances[key] = cls(device)
        return cls._instances[key]

    @classmethod
    def reset(cls, device: torch.device) -> None:
        """Destroy the singleton for *device* (flush + unregister all)."""
        key = f"{device.type}:{device.index if device.index is not None else 0}"
        inst = cls._instances.pop(key, None)
        if inst is not None:
            inst.flush_all()
            for alloc in list(inst._registry.values()):
                alloc.remove_hooks()
            inst._registry.clear()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_module(
        self,
        name: str,
        module: nn.Module,
        *,
        pinned: bool = False,
        hook_submodules: Optional[List[str]] = None,
    ) -> "Allocation":
        """Register an ``nn.Module`` as a managed VRAM allocation.

        A ``forward_pre_hook`` is installed so ``activate(name)`` fires
        automatically before every forward pass.

        Parameters
        ----------
        name:
            Logical allocation name (e.g. ``"unet.down_blocks.0"``).
        module:
            The ``nn.Module`` whose parameters are managed.
        pinned:
            If True the allocation is never auto-evicted.
        hook_submodules:
            Dotted sub-module paths to hook instead of the module itself.
            Useful when the module is not called directly (e.g. VAE decoder).
        """
        _dev = self._device

        def load_fn() -> None:
            self._suppressing_tracking = True
            try:
                module.to(device=_dev)
            finally:
                self._suppressing_tracking = False

        def evict_fn() -> None:
            module.to(device="cpu")

        def size_fn() -> int:
            n = sum(p.numel() * p.element_size() for p in module.parameters())
            n += sum(
                b.numel() * b.element_size()
                for b in module.buffers()
                if b is not None
            )
            return int(n * _CUDA_OVERHEAD)

        first_param = next(module.parameters(), None)
        on_device = (
            first_param is not None
            and first_param.device.type == _dev.type
            and (first_param.device.index or 0) == (_dev.index or 0)
        )

        alloc = Allocation(
            name=name,
            load_fn=load_fn,
            evict_fn=evict_fn,
            size_fn=size_fn,
            pinned=pinned,
            on_device=on_device,
            gen=GEN_COLD,
        )

        # Weakref: if the module is ever garbage-collected by external code
        # the finalizer marks this allocation as a free-list candidate.
        # (Our load_fn/evict_fn closures hold a strong reference, so the
        # finalizer only fires after unregister() releases those closures
        # and some other reference also drops — effectively it is a safety net.)
        def _on_module_gc(ref, _n=name, _a=self):
            a = _a._registry.get(_n)
            if a is not None and not a.in_free_list:
                a.in_free_list = True
                a.gen = GEN_FREE
        alloc._module_ref = weakref.ref(module, _on_module_gc)

        # Collect hook targets.
        hook_targets: List[nn.Module] = []
        if hook_submodules:
            for dotted in hook_submodules:
                sub = module
                for part in dotted.split("."):
                    sub = getattr(sub, part, None)  # type: ignore[assignment]
                    if sub is None:
                        break
                if sub is not None:
                    hook_targets.append(sub)  # type: ignore[arg-type]
        if not hook_targets:
            hook_targets = [module]

        for target in hook_targets:
            def _make_hook(n: str = name) -> Any:
                def _hook(m: nn.Module, inp: Any) -> None:
                    # Detect stale on_device flag: external code (e.g.
                    # model_management) may have moved the module to CPU.
                    alloc_ = self._registry.get(n)
                    if alloc_ is not None and alloc_.on_device:
                        p = next(m.parameters(), None)
                        if p is not None:
                            actual, exp = p.device, self._device
                            if actual.type != exp.type or (actual.index or 0) != (exp.index or 0):
                                self._lru.pop(n, None)
                                alloc_.on_device = False
                                # Treat as a free-list candidate — this data
                                # was evicted without going through our path.
                                alloc_.in_free_list = True
                                alloc_.gen = GEN_FREE
                    self.activate(n)
                return _hook
            alloc._hook_handles.append(
                target.register_forward_pre_hook(_make_hook())
            )

        self._registry[name] = alloc
        if on_device:
            self._lru[name] = alloc

        log.info(
            "VRAMAllocator: registered '%s' | %d MB | gen=%d | pinned=%s | on_device=%s",
            name, alloc.size_fn() >> 20, alloc.gen, pinned, on_device,
        )
        return alloc

    def register(
        self,
        name: str,
        load_fn: Callable[[], None],
        evict_fn: Callable[[], None],
        size_fn: Callable[[], int],
        *,
        pinned: bool = False,
        on_device: bool = False,
    ) -> "Allocation":
        """Register a custom load/evict pair (non-module allocations)."""
        alloc = Allocation(
            name=name,
            load_fn=load_fn,
            evict_fn=evict_fn,
            size_fn=size_fn,
            pinned=pinned,
            on_device=on_device,
            gen=GEN_COLD,
        )
        self._registry[name] = alloc
        if on_device:
            self._lru[name] = alloc
        log.info("VRAMAllocator: registered custom '%s' | %d MB", name, alloc.size_fn() >> 20)
        return alloc

    def unregister(self, name: str) -> None:
        """Remove *name*; evicts first if currently on device."""
        self.evict(name)
        alloc = self._registry.pop(name, None)
        if alloc is not None:
            alloc.remove_hooks()

    def mark_free(self, name: str) -> None:
        """Explicitly place *name* in the free list (evicted before cold data).

        Use when you know an allocation will no longer be needed this session.
        """
        alloc = self._registry.get(name)
        if alloc is not None:
            alloc.in_free_list = True
            alloc.gen = GEN_FREE

    # ------------------------------------------------------------------
    # Reservations — unmanaged on-device memory
    # ------------------------------------------------------------------

    def reserve(self, name: str, size_bytes: int) -> None:
        """Account for *size_bytes* of unmanaged on-device memory.

        Subtracted from ``_effective_free()`` to prevent over-commit.
        Call ``free_reservation()`` when the tensor is freed or moved to CPU.
        """
        self._reservations[name] = size_bytes

    def free_reservation(self, name: str) -> None:
        """Remove a VRAM reservation."""
        self._reservations.pop(name, None)

    # ------------------------------------------------------------------
    # TorchDispatch tracking context
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def tracking_context(self):
        """Enable dynamic unmanaged tensor tracking for the duration of this block.

        Large CUDA tensors (≥ 1 MiB) allocated within the context are tracked
        via weakref; when Python GC collects them the reservation is released
        automatically, giving ``_effective_free()`` real-time accounting.

        Example::

            with allocator.tracking_context():
                out = unet(x, t, encoder_hidden_states=enc)

        Skip-connection activations that remain alive across block boundaries
        are tracked as reserved VRAM, so the allocator won't over-commit when
        deciding whether to evict for the next block.
        """
        if not _HAS_DISPATCH_MODE:
            yield
            return
        with VRAMTrackerDispatch(self):
            yield

    # ------------------------------------------------------------------
    # Core activate / prepare / evict
    # ------------------------------------------------------------------

    def activate(self, name: str) -> None:
        """Ensure allocation *name* is resident on device.

        Evicts LRU cold allocations (generation-ordered) until effective free
        VRAM ≥ the allocation's estimated weight size.  On each eviction all
        resident generations are decremented (cooling step).  On load the
        allocation's generation is incremented mod GEN_STATES.
        """
        alloc = self._registry.get(name)
        if alloc is None:
            return

        if alloc.on_device:
            # Warm up the generation counter on repeated use — capped at GEN_HOT
            # so frequently-accessed residents are deprioritised for eviction.
            if alloc.gen < GEN_HOT:
                alloc.gen += 1
            self._lru.move_to_end(name)
            return

        needed = alloc.size_fn()

        evicted: List[str] = []
        while self._effective_free() < needed:
            victim = self._lru_candidate()
            if victim is None:
                print(
                    f"[VRAM] WARNING: no evictable candidate for '{name}' "
                    f"(need {needed >> 20} MB, free {self._effective_free() >> 20} MB)"
                    " — loading anyway"
                )
                break
            self._do_evict(victim)
            evicted.append(victim)

        if evicted:
            print(f"[VRAM] '{name}': evicted {len(evicted)} block(s) to make room")
            # Cooling step: age all remaining residents ONCE per activation round,
            # not once per individual eviction.  This prevents cascade aging when
            # multiple blocks are evicted in sequence to fit one large allocation.
            self._age_all()

        # Load — suppress tracking so parameter tensors aren't counted as
        # unmanaged by an active VRAMTrackerDispatch.
        self._suppressing_tracking = True
        try:
            alloc.load_fn()
        finally:
            self._suppressing_tracking = False

        # Generation increment on load: cycles through 0→1→2→3→0.
        # A block that wraps from 3→0 has been loaded four times in quick
        # succession; it briefly appears COLD until the next access warms it.
        alloc.gen = (alloc.gen + 1) % GEN_STATES
        alloc.on_device = True
        alloc.in_free_list = False   # loading clears dead-data flag
        self._lru[name] = alloc

    def prepare_for(self, name: str) -> None:
        """Pin *name*, evict all unpinned, then activate *name*."""
        alloc = self._registry.get(name)
        if alloc is None:
            log.warning("VRAMAllocator.prepare_for '%s': NOT registered — skipping", name)
            return
        was_pinned = alloc.pinned
        alloc.pinned = True
        try:
            self.evict_all_unpinned()
        finally:
            alloc.pinned = was_pinned
        self.activate(name)

    def prepare_for_prefix(self, prefix: str) -> None:
        """Evict all allocations NOT matching *prefix*; matching blocks load on demand.

        Temporarily pins every allocation whose name starts with *prefix* so
        ``evict_all_unpinned`` skips them.  Forward pre-hooks will activate
        matching blocks on demand during the subsequent forward pass.
        """
        free_before = self._effective_free()
        n_before = len(self._lru)
        matching = [n for n in self._registry if n.startswith(prefix)]
        for n in matching:
            self._registry[n].pinned = True
        try:
            self.evict_all_unpinned()
        finally:
            for n in matching:
                self._registry[n].pinned = False
        free_after = self._effective_free()
        print(
            f"[VRAM] prepare_for_prefix '{prefix}': evicted {n_before - len(self._lru)} block(s) | "
            f"free {free_before >> 20} MB → {free_after >> 20} MB | residents {len(self._lru)}"
        )

    def evict(self, name: str) -> None:
        """Explicitly evict *name* to CPU."""
        if name in self._lru:
            self._do_evict(name)

    def evict_by_prefix(self, prefix: str) -> None:
        """Evict all resident allocations whose name starts with *prefix*."""
        victims = [n for n in list(self._lru.keys()) if n.startswith(prefix)]
        for name in victims:
            self._do_evict(name)
        if victims:
            print(f"[VRAM] evicted {len(victims)} '{prefix}*' block(s)")
        self._maybe_empty_cache()

    def evict_all_unpinned(self) -> None:
        """Evict all unpinned residents and release the CUDA cache."""
        for name in list(self._lru.keys()):
            if not self._lru[name].pinned:
                self._do_evict(name)
        self._maybe_empty_cache()

    def flush_all(self) -> None:
        """Evict ALL residents (including pinned) and release the CUDA cache."""
        for name in list(self._lru.keys()):
            self._do_evict(name)
        self._maybe_empty_cache()

    # ------------------------------------------------------------------
    # Pin / unpin
    # ------------------------------------------------------------------

    def pin(self, name: str) -> None:
        if name in self._registry:
            self._registry[name].pinned = True

    def unpin(self, name: str) -> None:
        if name in self._registry:
            self._registry[name].pinned = False

    # ------------------------------------------------------------------
    # VRAM accounting
    # ------------------------------------------------------------------

    def _effective_free(self) -> int:
        """Bytes available for a new allocation, fragmentation-aware.

        Formula::

            cuda_free + reclaimable_pytorch_cache - reservations
            reclaimable = reserved - active - inactive_split (fragmented)

        ``_reservations`` includes both static entries (e.g.
        ``inference_headroom``) and dynamic entries injected by
        ``VRAMTrackerDispatch`` for live unmanaged tensors.
        """
        if self._device.type != "cuda":
            return 2 ** 62
        try:
            cuda_free, _ = torch.cuda.mem_get_info(self._device)
            s = torch.cuda.memory_stats(self._device)
            frag        = s.get("inactive_split_bytes.all.current", 0)
            reserved    = s.get("reserved_bytes.all.current",     0)
            active      = s.get("active_bytes.all.current",       0)
            reclaimable = max(0, reserved - active - frag)
            return max(0, cuda_free + reclaimable - sum(self._reservations.values()))
        except Exception:
            return 0

    def stats(self) -> dict:
        """Snapshot of allocator state for logging / debugging."""
        return {
            "device": str(self._device),
            "effective_free_mb": self._effective_free() >> 20,
            "n_residents": len(self._lru),
            "n_registered": len(self._registry),
            "residents": [
                {
                    "name": n,
                    "size_mb": a.size_fn() >> 20,
                    "gen": a.gen,
                    "pinned": a.pinned,
                    "in_free_list": a.in_free_list,
                }
                for n, a in self._lru.items()
            ],
            "reservations_mb": sum(self._reservations.values()) >> 20,
            "tracked_tensors": len(self._tracked_tensor_refs),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lru_candidate(self) -> Optional[str]:
        """Select the best eviction victim.

        Priority:
          1. Free-list items (``in_free_list`` or ``gen == GEN_FREE``)
          2. Coldest generation tier (GEN_COLD=0 first)
          3. Within the same tier: LRU order (OrderedDict insertion order)
        """
        # Pass 1: free list — orphaned / explicitly dead data
        for name, alloc in self._lru.items():
            if not alloc.pinned and (alloc.in_free_list or alloc.gen == GEN_FREE):
                return name
        # Pass 2: ascending generation, LRU within each tier
        for tier in range(GEN_COLD, GEN_HOT + 1):
            for name, alloc in self._lru.items():
                if not alloc.pinned and alloc.gen == tier:
                    return name
        return None

    def _do_evict(self, name: str) -> None:
        alloc = self._lru.pop(name, None)
        if alloc is None:
            return
        try:
            alloc.evict_fn()
        except Exception as e:
            log.warning("VRAMAllocator: evict '%s' error: %s", name, e)
        alloc.on_device = False
        self._check_defrag()

    def _age_all(self) -> None:
        """Decrement generation of every resident allocation (cooling step).

        Called once per ``activate()`` round (after all evictions needed to fit
        one block), NOT once per individual eviction.  This prevents cascade
        aging when a sequence of blocks is evicted for a single large allocation.
        Blocks that are not accessed across many eviction rounds drift toward
        GEN_COLD and become prime eviction candidates.  Free-list items stay at
        GEN_FREE.
        """
        for alloc in self._lru.values():
            if alloc.gen > GEN_COLD:
                alloc.gen -= 1

    def sync_device_state(self) -> None:
        """Fix stale ``on_device`` flags after external VRAM reclaims.

        ``model_management.free_memory()`` may move tracked modules to CPU
        without notifying the allocator.  This method walks every registered
        allocation whose ``on_device`` flag is True, checks the actual device
        of the first parameter via the stored ``_module_ref`` weakref, and
        corrects the flag if the module has been silently moved.
        """
        for name, alloc in list(self._registry.items()):
            if not alloc.on_device:
                continue
            module = alloc._module_ref() if alloc._module_ref else None
            if module is None:
                continue
            p = next(module.parameters(), None)
            if p is None:
                continue
            actual = p.device
            exp = self._device
            if actual.type != exp.type or (actual.index or 0) != (exp.index or 0):
                self._lru.pop(name, None)
                alloc.on_device = False
                print(f"[VRAM] sync: '{name}' stale on_device fixed ({actual} → expected {exp})")

    def _check_defrag(self) -> None:
        """Call ``empty_cache()`` when fragmentation ratio exceeds threshold."""
        if self._device.type != "cuda":
            return
        try:
            s = torch.cuda.memory_stats(self._device)
            frag     = s.get("inactive_split_bytes.all.current", 0)
            reserved = s.get("reserved_bytes.all.current",       0)
            if reserved > 0 and frag / reserved > FRAG_THRESHOLD:
                torch.cuda.empty_cache()
                print(
                    f"[VRAM] defrag: empty_cache() fired "
                    f"(frag {frag >> 20} MB / reserved {reserved >> 20} MB = "
                    f"{frag / reserved * 100:.0f}% > {FRAG_THRESHOLD * 100:.0f}%)"
                )
        except Exception:
            pass

    def _maybe_empty_cache(self) -> None:
        if self._device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latent_vram_bytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    """Estimate VRAM cost for a latent tensor of *shape* and *dtype*."""
    numel = 1
    for s in shape:
        numel *= s
    return numel * torch.zeros(1, dtype=dtype).element_size()
