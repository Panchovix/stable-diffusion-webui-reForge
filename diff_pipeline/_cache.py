"""
diff_pipeline/_cache.py — lazy LRU-cache decorator for diff_pipeline functions.

The ``@lru_cached`` decorator behaves exactly like
``@functools.lru_cache(maxsize=N)`` but defers the ``maxsize`` lookup until the
*first call* so that the value of ``--forge-diffusers-lru-cache-size`` from
``cmd_args`` is respected even though decorators are applied at module-import
time (before argument parsing).

Usage
-----
    from diff_pipeline._cache import lru_cached

    @lru_cached
    def my_expensive_function(x: int) -> str:
        ...

The wrapped function gains two extra helpers (mirroring ``functools.lru_cache``):

* ``my_expensive_function.cache_clear()``  — clears cached results and resets
  the internal cache so the next call re-reads ``maxsize`` from opts.
* ``my_expensive_function.cache_info()``   — returns the ``CacheInfo`` named
  tuple from the underlying ``lru_cache``, or ``None`` when no calls have been
  made yet or caching is disabled (``maxsize == 0``).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _read_lru_maxsize() -> int:
    """Return the LRU cache maxsize from ``--cache-lru`` (``args_parser.args``).

    ``--cache-lru N`` sets the node-result LRU cache size for the ldm_patched
    backend.  We reuse the same value so that diff_pipeline caches follow the
    same user-controlled knob.

    Returns 0 when ``--cache-lru`` was not set (i.e. its default of 0), which
    disables caching.  Falls back to 0 on any import error so the module can be
    imported without a running webui environment.
    """
    try:
        from ldm_patched.modules import args_parser  # type: ignore[import-not-found]
        return int(getattr(args_parser.args, "cache_lru", 0))
    except Exception:
        return 0


def lru_cached(fn: F) -> F:
    """Lazy LRU-cache decorator.

    The ``maxsize`` is resolved from ``cmd_opts`` on the *first* call so that
    command-line arguments are already parsed by then.  When ``maxsize`` is 0,
    the function is called directly without any caching.

    The returned wrapper exposes ``cache_clear()`` and ``cache_info()``
    methods that mirror those of ``functools.lru_cache``.
    """
    # _inner holds either the lru_cache-wrapped function, or the raw function
    # when caching is disabled (maxsize == 0).
    _inner: Optional[Callable] = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _inner
        if _inner is None:
            maxsize = _read_lru_maxsize()
            if maxsize == 0:
                log.debug(
                    "diff_pipeline LRU cache disabled (maxsize=0) for %s",
                    fn.__qualname__,
                )
                _inner = fn
            else:
                log.debug(
                    "diff_pipeline LRU cache initialised (maxsize=%d) for %s",
                    maxsize,
                    fn.__qualname__,
                )
                _inner = functools.lru_cache(maxsize=maxsize)(fn)
        return _inner(*args, **kwargs)

    def cache_clear() -> None:
        """Clear cached results and reset the cache so the next call
        re-reads ``maxsize`` from opts (picks up any runtime changes)."""
        nonlocal _inner
        if _inner is not None and hasattr(_inner, "cache_clear"):
            _inner.cache_clear()
        _inner = None

    def cache_info():
        """Return ``CacheInfo`` from the underlying ``lru_cache``, or ``None``
        when no calls have been made yet or caching is disabled."""
        if _inner is not None and hasattr(_inner, "cache_info"):
            return _inner.cache_info()
        return None

    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = cache_info    # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
