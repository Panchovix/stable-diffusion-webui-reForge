"""
jax_pipeline/_async.py — epoll-style non-blocking readiness polling for
JAX values whose computation may still be in flight.

JAX exposes no native non-blocking "is this ready yet" check at the
Python level: ``jax.Array`` only has ``block_until_ready()`` (confirmed
by reading ``jax/_src/array.py`` directly — the only readiness-related
methods are ``block_until_ready`` and ``is_deleted``, no ``is_ready``).
This module fills that gap with a small background-thread-based
primitive: spawn a lightweight daemon thread that DOES block (on
``block_until_ready()``) and signals completion via a
``threading.Event``, so the CALLING thread can poll non-blockingly
(``.ready()``) instead of stalling every step waiting for GPU
completion that hasn't happened yet.

Why this matters for jax_pipeline specifically: with per-step
synchronization removed from the sampler loop (see
jax_pipeline.samplers's module docstring) and no explicit sync added at
PhaseManager transitions either, the whole generation — CLIP encode,
every denoising step, VAE decode, device<->host phase moves — forms one
continuous async dispatch chain from "user clicks generate" through to
the final image. Python races through dispatching all of it almost
instantly while the GPU is still working through it. Progress
reporting / live preview / "new image ready" notifications can no
longer be tied to the Python dispatch loop's iteration position
(meaningless once dispatch outruns execution) — they have to poll
actual GPU completion instead, on whatever cadence the caller wants,
without ever blocking the dispatch loop itself. That's what
``AsyncReadiness`` is for.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class AsyncReadiness:
    """Wraps a JAX value (or pytree of JAX values) whose computation may
    still be in flight.

    ``.ready()`` polls non-blockingly. ``.get()`` blocks (only when the
    caller genuinely needs the materialized value — e.g. the final
    result, an interrupt request, or an actual "push this image to the
    WebUI" moment) and re-raises any exception the background wait hit.

    One background thread per instance — cheap to abandon (a superseded,
    still-unresolved instance just finishes on its own with its result
    discarded; nothing needs to cancel it).
    """

    __slots__ = ("_value", "_event", "_result", "_thread", "_error")

    def __init__(self, value: Any):
        self._value = value
        self._event = threading.Event()
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def _wait(self) -> None:
        import jax
        try:
            self._result = jax.block_until_ready(self._value)
        except BaseException as e:  # noqa: BLE001 -- surfaced via .get(), never swallowed
            self._error = e
        finally:
            self._event.set()

    def ready(self) -> bool:
        """Non-blocking poll — True once the value is fully materialized."""
        return self._event.is_set()

    def get(self, timeout: Optional[float] = None) -> Any:
        """Block (if not already ready) until the value is materialized,
        then return it."""
        self._event.wait(timeout=timeout)
        if self._error is not None:
            raise self._error
        return self._result
