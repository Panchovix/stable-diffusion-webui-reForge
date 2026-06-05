"""
diff_pipeline/_lru_blocks.py — LRU block cache for UNet dynamic block loading.

Keeps at most `capacity` UNet blocks resident on the compute device at any time,
using a Least-Recently-Used eviction policy.  Implements the same block-level
dynamic loading that reForge's model_patcher.load() does for ldm models, adapted
for a HF UNet2DConditionModel.

Design: decouple structure from data
-------------------------------------
nn.Module objects (the structural graph) never move.  Only the *data*
(parameter and buffer tensors) is transferred between CPU and device.  Each
managed block keeps a persistent pinned-CPU copy of its weights; GPU storage is
allocated on first load and freed on eviction.  Redirecting ``param.data``
replaces the storage pointer without touching the Module tree.

CUDA stream transfer
---------------------
When ``device`` is ``cuda``, a dedicated transfer stream is created.  H→D
copies run on that stream concurrently with compute on the default stream.
A ``torch.cuda.Event`` handshake ensures the compute stream waits for the
transfer to finish before a block's forward pass begins.

Pinned memory
--------------
CPU copies are kept in pinned (page-locked) memory when CUDA is available so
the DMA engine can copy them directly to the device without a CPU-bounce.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class LRUBlockCache:
    """LRU cache keeping at most `capacity` UNet blocks on-device.

    Parameters
    ----------
    blocks:
        ``[(path, module)]`` — the managed blocks in forward-pass order.
        ``path`` is the dotted module name (e.g. ``"down_blocks.0"``).
    device:
        Compute device.  Only parameters / buffers on this device are live at
        any time; everything else sits in CPU memory.
    capacity:
        Maximum number of blocks that may reside on ``device`` simultaneously.
        Must be ≥ 1.  Use :func:`estimate_capacity` to pick a good value.
    """

    def __init__(
        self,
        blocks: List[Tuple[str, nn.Module]],
        device: torch.device,
        capacity: int,
    ) -> None:
        self._device = device
        self._capacity = max(1, capacity)
        # LRU order: key=path, value=None.  Oldest (LRU) is first.
        self._lru: OrderedDict[str, None] = OrderedDict()

        # Transfer stream for async H→D copies (CUDA only).
        self._xfer_stream: Optional[Any] = None
        if device.type == "cuda":
            try:
                self._xfer_stream = torch.cuda.Stream(device)
                log.info("LRUBlockCache: dedicated CUDA transfer stream created.")
            except Exception as exc:
                log.warning("LRUBlockCache: could not create transfer stream: %s", exc)

        # Per-block storage
        self._modules: Dict[str, nn.Module] = {}
        # path → {param_name: cpu_tensor}  (always resident, pinned when possible)
        self._cpu_params: Dict[str, Dict[str, torch.Tensor]] = {}
        # path → {buf_name: cpu_tensor}
        self._cpu_bufs: Dict[str, Dict[str, torch.Tensor]] = {}
        # path → {param_name: gpu_tensor}  (only while block is cached)
        self._gpu_params: Dict[str, Dict[str, torch.Tensor]] = {}
        # path → {buf_name: gpu_tensor}
        self._gpu_bufs: Dict[str, Dict[str, torch.Tensor]] = {}

        for path, module in blocks:
            self._modules[path] = module
            cpu_p: Dict[str, torch.Tensor] = {}
            cpu_b: Dict[str, torch.Tensor] = {}

            for name, param in module.named_parameters():
                t = param.data.cpu().clone()
                if device.type == "cuda":
                    try:
                        t = t.pin_memory()
                    except Exception:
                        pass
                cpu_p[name] = t
                param.data = t  # redirect to our CPU copy

            for name, buf in module.named_buffers():
                if buf is None:
                    continue
                t = buf.data.cpu().clone()
                if device.type == "cuda":
                    try:
                        t = t.pin_memory()
                    except Exception:
                        pass
                cpu_b[name] = t
                buf.data = t

            self._cpu_params[path] = cpu_p
            self._cpu_bufs[path] = cpu_b

        n = len(blocks)
        log.info(
            "LRUBlockCache: %d blocks managed | capacity=%d | device=%s | "
            "xfer_stream=%s",
            n,
            self._capacity,
            device,
            "yes" if self._xfer_stream else "no",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def activate(self, path: str) -> None:
        """Ensure the block at *path* is resident on device.

        If already cached: bump to MRU position.
        If not cached and cache is full: evict the LRU block, then load.
        """
        if path in self._lru:
            self._lru.move_to_end(path)
            return

        if len(self._lru) >= self._capacity:
            evict_path, _ = self._lru.popitem(last=False)
            self._evict(evict_path)

        self._load(path)
        self._lru[path] = None

    def install_hooks(self) -> List[Any]:
        """Register forward pre-hooks on every managed block.

        Each hook calls :meth:`activate` so the block's parameters are on
        device before its forward pass begins.  Returns the hook handles
        (call ``handle.remove()`` to uninstall).
        """
        handles: List[Any] = []
        for path in self._modules:
            def _make_hook(p: str):
                def _pre_hook(m: nn.Module, inp: Any) -> None:
                    self.activate(p)
                return _pre_hook
            handles.append(
                self._modules[path].register_forward_pre_hook(_make_hook(path))
            )
        log.info("LRUBlockCache: %d forward pre-hooks installed.", len(handles))
        return handles

    # ------------------------------------------------------------------
    # Internal load / evict
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        """Copy CPU params/bufs → freshly allocated GPU tensors and redirect
        the module's storage pointers to the GPU tensors."""
        cpu_p = self._cpu_params[path]
        cpu_b = self._cpu_bufs[path]
        gpu_p: Dict[str, torch.Tensor] = {}
        gpu_b: Dict[str, torch.Tensor] = {}

        if self._xfer_stream is not None:
            # Allocate + copy on the transfer stream.
            with torch.cuda.stream(self._xfer_stream):
                for name, t in cpu_p.items():
                    g = torch.empty(t.shape, dtype=t.dtype, device=self._device)
                    g.copy_(t, non_blocking=True)
                    gpu_p[name] = g
                for name, t in cpu_b.items():
                    g = torch.empty(t.shape, dtype=t.dtype, device=self._device)
                    g.copy_(t, non_blocking=True)
                    gpu_b[name] = g
            # Insert an event so the compute stream waits for the copies.
            _evt = torch.cuda.Event()
            self._xfer_stream.record_event(_evt)
            torch.cuda.current_stream(self._device).wait_event(_evt)
        else:
            for name, t in cpu_p.items():
                gpu_p[name] = t.to(self._device)
            for name, t in cpu_b.items():
                gpu_b[name] = t.to(self._device)

        # Redirect storage pointers.
        module = self._modules[path]
        for name, param in module.named_parameters():
            if name in gpu_p:
                param.data = gpu_p[name]
        for name, buf in module.named_buffers():
            if name in gpu_b:
                buf.data = gpu_b[name]

        self._gpu_params[path] = gpu_p
        self._gpu_bufs[path] = gpu_b
        log.debug("LRUBlockCache: loaded '%s' → %s", path, self._device)

    def _evict(self, path: str) -> None:
        """Redirect module storage back to CPU and free GPU tensors."""
        cpu_p = self._cpu_params[path]
        cpu_b = self._cpu_bufs[path]
        module = self._modules[path]

        for name, param in module.named_parameters():
            if name in cpu_p:
                param.data = cpu_p[name]
        for name, buf in module.named_buffers():
            if name in cpu_b:
                buf.data = cpu_b[name]

        # Free GPU tensors (del removes last Python ref; CUDA allocator reclaims).
        self._gpu_params.pop(path, None)
        self._gpu_bufs.pop(path, None)
        log.debug("LRUBlockCache: evicted '%s' → cpu", path)


# ---------------------------------------------------------------------------
# Capacity estimation helper
# ---------------------------------------------------------------------------

def estimate_capacity(
    blocks: List[Tuple[str, nn.Module]],
    device: torch.device,
    headroom_bytes: int = 512 * 1024 * 1024,
) -> int:
    """Return how many blocks from *blocks* fit in available device memory.

    Uses a greedy largest-first fit so the biggest blocks are always cached
    (they benefit most from staying on device).

    Parameters
    ----------
    blocks:
        ``[(path, module)]`` list — same order as passed to :class:`LRUBlockCache`.
    device:
        Compute device.  Returns ``len(blocks)`` for non-CUDA devices.
    headroom_bytes:
        Memory to reserve for activations, KV cache, etc.  Default 512 MiB.
    """
    if device.type != "cuda":
        return len(blocks)

    # usable VRAM = free + allocator cache − headroom
    try:
        stats = torch.cuda.memory_stats(device)
        cuda_free, _ = torch.cuda.mem_get_info(device)
        free = int(
            cuda_free
            + stats.get("reserved_bytes.all.current", 0)
            - stats.get("active_bytes.all.current", 0)
        )
    except Exception:
        free = 0

    usable = max(0, free - headroom_bytes)
    log.info(
        "LRUBlockCache capacity estimate: free_vram=%.0f MB  headroom=%.0f MB  "
        "usable=%.0f MB",
        free / (1024 ** 2),
        headroom_bytes / (1024 ** 2),
        usable / (1024 ** 2),
    )

    block_sizes = []
    for path, module in blocks:
        size = sum(p.numel() * p.element_size() for p in module.parameters())
        size += sum(
            b.numel() * b.element_size()
            for b in module.buffers()
            if b is not None
        )
        block_sizes.append((size, path))

    block_sizes.sort(key=lambda x: x[0], reverse=True)

    capacity = 0
    remaining = usable
    for size, path in block_sizes:
        if remaining >= size:
            remaining -= size
            capacity += 1
        # Don't break — smaller blocks might still fit.

    capacity = max(1, min(capacity, len(blocks)))
    log.info(
        "LRUBlockCache capacity estimate: %d / %d blocks fit in usable VRAM",
        capacity,
        len(blocks),
    )
    return capacity
