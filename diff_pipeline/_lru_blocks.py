"""
diff_pipeline/_lru_blocks.py — LRU block cache for UNet dynamic block loading.

Keeps at most `capacity` UNet blocks resident on the compute device at any time,
using a Least-Recently-Used eviction policy.  Mirrors the dynamic block loading
that reForge's model_patcher.load() does for ldm models.

Design
------
``module.to(device)`` / ``module.to("cpu")`` are the only primitives used for
moving parameters.  This keeps the implementation simple and provably correct —
the same primitives the existing auto-offload hooks use.

CUDA stream transfer
--------------------
When ``device`` is CUDA and a dedicated transfer stream is available, H→D copies
run on that stream.  Because ``module.to(device)`` (non_blocking=False) is
CPU-synchronous, the copy COMPLETES before the pre-hook returns, so the compute
stream sees consistent GPU memory without needing an explicit CUDA event
handshake.  Running the copy on a dedicated stream keeps the default stream free
for other work while the CPU stalls on the transfer.

LRU policy
----------
An ``OrderedDict`` tracks which blocks are currently on-device.  The oldest
(least-recently-used) entry is evicted when the cache is full.  For a UNet with
a fixed forward-pass block order, LRU naturally converges to an optimal resident
set after the first step.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LRUBlockCache:
    """LRU cache keeping at most ``capacity`` UNet blocks on-device.

    Parameters
    ----------
    blocks:
        ``[(path, module)]`` — the managed blocks in forward-pass order.
        ``path`` is the dotted module name (e.g. ``"down_blocks.0"``).
    device:
        Compute device.
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

        # Dedicated transfer stream for H→D copies (CUDA only).
        # Placing the copy on a separate stream lets the default stream continue
        # with CPU-side work while the DMA engine transfers the block.
        # module.to(device) is still CPU-synchronous so no CUDA event is needed.
        self._xfer_stream: Optional[Any] = None
        if device.type == "cuda":
            try:
                self._xfer_stream = torch.cuda.Stream(device)
                log.info("LRUBlockCache: dedicated CUDA transfer stream created.")
            except Exception as exc:
                log.warning("LRUBlockCache: could not create transfer stream: %s", exc)

        self._modules: Dict[str, nn.Module] = {}
        for path, module in blocks:
            self._modules[path] = module
            # Ensure block starts on CPU — if it's already on CPU this is a no-op.
            module.to("cpu")

        log.info(
            "LRUBlockCache: %d blocks managed | capacity=%d | device=%s | "
            "xfer_stream=%s",
            len(blocks),
            self._capacity,
            device,
            "yes" if self._xfer_stream else "no",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def activate(self, path: str) -> None:
        """Ensure the block at *path* is resident on device.

        * LRU hit  — bump to MRU position (no transfer needed).
        * LRU miss — evict the LRU block if at capacity, then load.
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
        """Register a forward pre-hook on every managed block.

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
        """Move block parameters from CPU → device.

        Uses the dedicated transfer stream when available.  ``module.to()``
        (non_blocking=False) is CPU-synchronous: it stalls the CPU until the
        copy completes, guaranteeing parameters are valid before the pre-hook
        returns and the block's forward begins.
        """
        module = self._modules[path]
        if self._xfer_stream is not None:
            with torch.cuda.stream(self._xfer_stream):
                module.to(device=self._device)
        else:
            module.to(device=self._device)
        log.debug("LRUBlockCache: loaded '%s' → %s", path, self._device)

    def _evict(self, path: str) -> None:
        """Move block parameters from device → CPU to free VRAM."""
        self._modules[path].to(device="cpu")
        log.debug("LRUBlockCache: evicted '%s' → cpu", path)


# ---------------------------------------------------------------------------
# Capacity estimation helper
# ---------------------------------------------------------------------------

def estimate_capacity(
    blocks: List[Tuple[str, nn.Module]],
    device: torch.device,
    headroom_bytes: Optional[int] = None,
    x_shape: Optional[Tuple[int, ...]] = None,
) -> int:
    """Return how many blocks from *blocks* fit in available device memory.

    Uses a greedy largest-first fit so the biggest blocks benefit most from
    staying on-device.

    Parameters
    ----------
    blocks:
        ``[(path, module)]`` list — same order as passed to :class:`LRUBlockCache`.
    device:
        Compute device.  Returns ``len(blocks)`` for non-CUDA devices.
    headroom_bytes:
        Memory to reserve for activations, KV cache, etc.  When ``None``
        (default) the value is derived from reForge's
        ``minimum_inference_memory()`` (1 GiB).
    x_shape:
        Shape of the latent tensor ``(B, C, H, W)`` used to scale the
        activation headroom estimate.  Pass ``x.shape`` when available.
    """
    if device.type != "cuda":
        return len(blocks)

    if headroom_bytes is None:
        try:
            from ldm_patched.modules.model_management import minimum_inference_memory
            headroom_bytes = int(minimum_inference_memory())
        except Exception:
            headroom_bytes = 1 * 1024 * 1024 * 1024  # 1 GiB fallback

    if x_shape is not None:
        b, _c, h, w = x_shape
        headroom_bytes += b * 320 * h * w * 2 * 4

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
        "LRUBlockCache capacity: free_vram=%.0f MB  headroom=%.0f MB  "
        "usable=%.0f MB",
        free / (1024 ** 2),
        headroom_bytes / (1024 ** 2),
        usable / (1024 ** 2),
    )

    # Block size = parameters + buffers (on CPU at this point for LRU path;
    # numel * element_size is device-independent).
    block_sizes: List[Tuple[int, str]] = []
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
    for size, _ in block_sizes:
        if remaining >= size:
            remaining -= size
            capacity += 1

    capacity = max(1, min(capacity, len(blocks)))
    log.info(
        "LRUBlockCache capacity: %d / %d blocks fit in usable VRAM",
        capacity,
        len(blocks),
    )
    return capacity
