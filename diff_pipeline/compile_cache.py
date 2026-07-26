"""
diff_pipeline/compile_cache.py — Persistent torch.compile artifact cache.

Points TORCHINDUCTOR_CACHE_DIR at a model-specific subdirectory of the WebUI
cache folder so that compiled kernel artifacts survive process restarts.

Cache directory layout
----------------------
    {cache_dir}/torch-compile/{model_shorthash}/
        fingerprint.json   — device + version info; written on first use,
                             compared on subsequent uses.  A mismatch voids
                             the entire subdirectory so stale artifacts cannot
                             corrupt output.

Model identity
--------------
The model shorthash (first 10 characters of sha256) comes from the WebUI
checkpoint system (``sd_model.sd_model_hash``), the same identifier shown in
the UI title bar.  One subdirectory is kept per unique model file.

LoRA hotswap
------------
LoRA adapters structurally modify the UNet forward pass, so each LoRA
combination produces a different compute graph.  The torch.inductor cache
automatically assigns distinct cache entries to distinct graphs within the
same ``TORCHINDUCTOR_CACHE_DIR``.  All LoRA variants of a model therefore
share one directory; the inductor cache manages the per-variant entries
transparently.  After the first inference with a given base+LoRA combo,
subsequent swaps to that same combo reuse the cached kernel without a new
compile — "most of the compile" is skipped, which is the accepted behaviour.

Cross-machine portability
-------------------------
The fingerprint check covers PyTorch version, CUDA version, and GPU model.
The torch.inductor cache validates the same information internally, but our
fingerprint provides an early coarse check and a human-readable record of
what the cache was built for.  Caches are not portable across different GPU
families or PyTorch major/minor versions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from typing import Optional

import torch  # type: ignore[import-not-found]

log = logging.getLogger(__name__)

_SUBSECTION = "torch-compile"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _cache_root() -> str:
    """Return the torch-compile subdir inside the WebUI cache directory."""
    try:
        from modules.cache import cache_dir
        return os.path.join(cache_dir, _SUBSECTION)
    except Exception:
        # Standalone / test context: fall back to a predictable location.
        return os.path.join(os.path.expanduser("~/.cache"), "reforge", _SUBSECTION)


def model_cache_dir(model_shorthash: str) -> str:
    """Return the absolute path for *model_shorthash*'s compile cache."""
    return os.path.join(_cache_root(), model_shorthash)


# ---------------------------------------------------------------------------
# Device fingerprint
# ---------------------------------------------------------------------------

def _build_fingerprint(device: torch.device) -> dict:
    """Return a dict that uniquely identifies the current compute environment.

    The fingerprint covers:
    - PyTorch version (graph serialisation format changes between releases)
    - CUDA version    (triton kernels are ABI-coupled to CUDA)
    - GPU name        (ISA / compute-cap differences mean a cache built on an
                       RTX 3090 cannot be used on an RTX 4090)
    - Platform OS     (kernel ABI differs between Linux / Windows / macOS)
    """
    fp: dict = {
        "torch_version": torch.__version__,
        "platform": sys.platform,
    }
    if device.type == "cuda":
        fp["cuda_version"] = torch.version.cuda or "unknown"
        try:
            fp["device_name"] = torch.cuda.get_device_name(device)
        except Exception:
            fp["device_name"] = "unknown"
    else:
        # MPS / CPU: record device type; no driver version to capture.
        fp["device_type"] = device.type
    return fp


def _fingerprint_matches(stored: dict, current: dict) -> bool:
    return stored == current


# ---------------------------------------------------------------------------
# Cache voiding
# ---------------------------------------------------------------------------

def _void_cache(cache_dir: str) -> None:
    """Delete every file/dir inside *cache_dir*, keeping the directory itself.

    Called when the fingerprint check fails so that stale inductor artifacts
    (wrong GPU ISA, wrong CUDA ABI, wrong PyTorch graph format) are never used.
    """
    if not os.path.isdir(cache_dir):
        return
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as exc:
            log.warning("compile_cache: could not remove %s: %s", path, exc)
    log.info("compile_cache: cache voided — %s", cache_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def activate(model_shorthash: str, device: torch.device) -> Optional[str]:
    """Set TORCHINDUCTOR_CACHE_DIR to the model-specific compile cache dir.

    Steps
    -----
    1. Compute the target directory: ``{cache_dir}/torch-compile/{hash}/``.
    2. Read ``fingerprint.json`` if it exists and compare it against the
       current environment.  A mismatch means the cached kernels were built
       for a different GPU or PyTorch version — void the directory so the
       fresh compile is stored cleanly.
    3. Write (or overwrite) ``fingerprint.json`` with current environment info.
    4. Set ``TORCHINDUCTOR_CACHE_DIR`` to the target directory so that
       torch.inductor stores and retrieves all compiled artifacts there.
    5. Enable ``TORCHINDUCTOR_FX_GRAPH_CACHE`` so the FX graph layer (the
       most expensive part of compilation) is also persisted across processes.

    Parameters
    ----------
    model_shorthash:
        10-character sha256 prefix from ``sd_model.sd_model_hash``.  If empty
        the function is a no-op and returns None.
    device:
        The compute device torch.compile will target (used for fingerprinting).

    Returns
    -------
    The activated cache directory path, or None if activation was skipped.
    """
    if not model_shorthash:
        log.debug("compile_cache: no model hash — using default inductor cache location")
        return None

    target_dir = model_cache_dir(model_shorthash)
    os.makedirs(target_dir, exist_ok=True)

    fp_path = os.path.join(target_dir, "fingerprint.json")
    current_fp = _build_fingerprint(device)

    # --- 1. Validate existing fingerprint ---
    if os.path.exists(fp_path):
        try:
            with open(fp_path, "r", encoding="utf-8") as fh:
                stored_fp = json.load(fh)
        except Exception as exc:
            log.warning(
                "compile_cache: could not read fingerprint (%s) — voiding cache", exc
            )
            stored_fp = {}

        if not _fingerprint_matches(stored_fp, current_fp):
            log.warning(
                "compile_cache: fingerprint mismatch for model %s — voiding cache.\n"
                "  stored : %s\n"
                "  current: %s",
                model_shorthash,
                json.dumps(stored_fp),
                json.dumps(current_fp),
            )
            _void_cache(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    # --- 2. Write current fingerprint ---
    try:
        with open(fp_path, "w", encoding="utf-8") as fh:
            json.dump(current_fp, fh, indent=2)
    except Exception as exc:
        log.warning("compile_cache: could not write fingerprint: %s", exc)

    # --- 3. Point inductor at the persistent model-specific directory ---
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target_dir

    # Enable FX graph cache — persists the (most expensive) graph compilation
    # step across Python processes.  Set both the env var (older PyTorch) and
    # the config object (newer PyTorch) for broadest compatibility.
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    try:
        import torch._inductor.config as _ind_cfg  # type: ignore[import-not-found]
        _ind_cfg.fx_graph_cache = True
    except Exception:
        pass

    log.info(
        "compile_cache: activated — model=%s  dir=%s  fingerprint=%s",
        model_shorthash,
        target_dir,
        json.dumps(current_fp),
    )
    return target_dir
