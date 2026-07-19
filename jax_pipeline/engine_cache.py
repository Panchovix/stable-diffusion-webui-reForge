"""
jax_pipeline/engine_cache.py — persistent "which fusion level worked
best" database, third step of a TensorRT-style autotuning effort (see
jax_pipeline/autotune.py for the benchmark/selection logic this backs,
and jax_pipeline/unet.py for the fusion levels themselves: "fine"
per-micro-block streaming, "coarse" per-stage streaming, "whole"
single-jit — mirroring how TensorRT's builder explores multiple kernel
implementations for the same op and caches whichever wins for a given
input profile).

Persisted at ``jax_pipeline/.cache/engine_db.json`` (same ``.cache/``
directory as the JAX persistent compilation cache, gitignored) as a flat
JSON list of entries — one per (component, device, VRAM, shape) point
that's ever been benchmarked. A modest number of entries is expected
(distinct resolution/prompt-length/device combinations a user actually
runs), so a linear scan for the closest match is plenty fast; no need
for anything more sophisticated than that.

Two lookup modes, matching the workflow the user asked for:
  * ``lookup_exact`` — same signature, instant hit, no computation.
  * ``lookup_closest`` — no exact match, but something "close enough"
    (same component/device/VRAM tier, resolution and prompt length
    within a tolerance) — reuse that decision rather than re-benchmarking
    for a trivially different shape.
If neither hits, the caller (autotune.py) benchmarks fresh and calls
``save()`` to record the result for next time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "engine_db.json")

# Tolerances for "close enough" matching — deliberately generous (a
# fusion-level decision doesn't need to be shape-exact the way a
# compiled XLA executable does; it only needs the VRAM/speed trade-off
# ballpark to still hold).
_RES_TOLERANCE_LATENT_PX = 32   # ~256 real pixels
_SEQ_TOLERANCE_TOKENS = 154     # 2 CLIP chunks
_VRAM_TOLERANCE_BYTES = 512 * 1024 * 1024  # 512MB


def _load_db() -> List[Dict[str, Any]]:
    try:
        with open(_DB_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        log.warning("[JAX Pipeline] engine_db.json unreadable (%s) — starting fresh.", e)
        return []


def _save_db(entries: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        tmp_path = _DB_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(entries, f, indent=1)
        os.replace(tmp_path, _DB_PATH)  # atomic on POSIX — avoids a torn/corrupt file on crash mid-write
    except Exception as e:
        log.warning("[JAX Pipeline] Could not persist engine_db.json: %s", e)


def compute_signature(
    component: str, device_name: str, total_vram_bytes: int,
    latent_h: int, latent_w: int, batch: int, seq_len: int,
) -> str:
    """A stable exact-match key. Bucketed just enough to avoid one
    signature per single-pixel jitter (resolution rounded to the nearest
    8 latent px = ~64 real px, VRAM rounded to the nearest GB, sequence
    length rounded up to the nearest 77-token CLIP chunk) while staying
    exact enough that two genuinely different setups don't collide.
    """
    vram_gb = round(total_vram_bytes / 1e9)
    res_h = round(latent_h / 8) * 8
    res_w = round(latent_w / 8) * 8
    seq_bucket = ((max(seq_len, 1) - 1) // 77 + 1) * 77
    return f"{component}|{device_name}|vram{vram_gb}GB|res{res_h}x{res_w}|b{batch}|seq{seq_bucket}"


def lookup_exact(signature: str) -> Optional[Dict[str, Any]]:
    for entry in _load_db():
        if entry.get("signature") == signature:
            return entry
    return None


def lookup_closest(
    component: str, device_name: str, total_vram_bytes: int,
    latent_h: int, latent_w: int, batch: int, seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Best-effort "close enough" match when there's no exact signature
    hit — same component/device/batch (those never fuzzy-match: a
    decision benchmarked for one GPU or one component isn't meaningful
    for another), resolution and sequence length within tolerance,
    picking whichever candidate is numerically closest if several match.
    Returns None if nothing qualifies — caller should benchmark fresh.
    """
    best = None
    best_distance = None
    for entry in _load_db():
        if entry.get("component") != component:
            continue
        if entry.get("device_name") != device_name:
            continue
        if entry.get("batch") != batch:
            continue
        if abs(entry.get("total_vram_bytes", 0) - total_vram_bytes) > _VRAM_TOLERANCE_BYTES:
            continue
        dh = abs(entry.get("latent_h", 0) - latent_h)
        dw = abs(entry.get("latent_w", 0) - latent_w)
        if dh > _RES_TOLERANCE_LATENT_PX or dw > _RES_TOLERANCE_LATENT_PX:
            continue
        dseq = abs(entry.get("seq_len", 0) - seq_len)
        if dseq > _SEQ_TOLERANCE_TOKENS:
            continue
        distance = dh + dw + dseq  # simple combined distance; good enough for a small candidate pool
        if best_distance is None or distance < best_distance:
            best, best_distance = entry, distance
    return best


def save(
    component: str, signature: str, device_name: str, total_vram_bytes: int,
    latent_h: int, latent_w: int, batch: int, seq_len: int,
    decision: str, stats: Dict[str, Any],
) -> None:
    """Record a fusion-level decision (+ the benchmark stats that
    justified it, kept for debugging/inspection) — replaces any existing
    entry with the same exact signature.
    """
    entries = _load_db()
    entries = [e for e in entries if e.get("signature") != signature]
    entries.append({
        "signature": signature,
        "component": component,
        "device_name": device_name,
        "total_vram_bytes": total_vram_bytes,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "batch": batch,
        "seq_len": seq_len,
        "decision": decision,
        "stats": stats,
        "saved_at": time.time(),
    })
    _save_db(entries)
    log.info(
        "[JAX Pipeline] engine_db: saved decision=%s for %s (sig=%s)",
        decision, component, signature,
    )
