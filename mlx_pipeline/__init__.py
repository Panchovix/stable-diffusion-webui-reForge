"""
mlx_pipeline — Apple MLX BFloat16 sampling pipeline for SDXL.

Auto-activates on Apple Silicon Macs when the ``mlx`` package is installed.
Replaces the UNet forward pass (apply_model) with a bfloat16 MLX implementation
that runs natively on the Metal backend via Apple's unified memory.

Requirements
------------
* macOS on Apple Silicon (M-series chip)
* ``mlx`` Python package installed  (``pip install mlx``)
* SDXL checkpoint (detected automatically by is_sdxl flag)

Activation
----------
Called automatically from modules_forge/forge_loader.py after the normal ldm
model loading completes.  No flags or user action required on compatible hardware.
If MLX is unavailable the function returns False silently and the standard
PyTorch / MPS pipeline is used instead.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── activation banner ────────────────────────────────────────────────────────
_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║          🍎  Apple MLX Pipeline Activated  🍎                       ║
║                                                                      ║
║  SDXL UNet → BFloat16 on Apple Silicon Metal via MLX                ║
║  Unified memory — zero-overhead torch ↔ MLX tensor bridge           ║
║  Native BF16: no fp16 overflow, full fp32 exponent range            ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ── hardware / package detection ─────────────────────────────────────────────

def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon macOS (M-series chip)."""
    if sys.platform != "darwin":
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        return "Apple" in result.stdout
    except Exception:
        return False


def is_mlx_available() -> bool:
    """Return True if the mlx package is importable and Metal is active."""
    try:
        import mlx.core as mx
        import mlx.nn  # noqa: F401
        # Confirm Metal backend is active (returns False on x86 or CPU-only MLX)
        return mx.metal.is_available()
    except Exception:
        return False


# ── public entry point ───────────────────────────────────────────────────────

def maybe_activate(sd_model, forge_objects) -> bool:
    """Auto-activate the MLX pipeline for SDXL on Apple Silicon.

    Called from modules_forge/forge_loader.py after normal ldm loading.

    Parameters
    ----------
    sd_model      : the loaded sd_model (needs is_sdxl attribute)
    forge_objects : ForgeObjects namedtuple (needs .unet UnetPatcher)

    Returns
    -------
    True  — MLX pipeline installed and wrapper registered.
    False — conditions not met or installation failed (no-op, standard path used).
    """
    if not getattr(sd_model, "is_sdxl", False):
        return False

    if not is_apple_silicon():
        return False

    if not is_mlx_available():
        log.info(
            "[MLX Pipeline] Apple Silicon detected but Metal MLX unavailable. "
            "Install with: pip install mlx"
        )
        return False

    try:
        from mlx_pipeline.pipeline import MLXSDXLPipeline
        from ldm_patched.modules.patcher_extension import WrappersMP

        mlx_pipe = MLXSDXLPipeline(forge_objects.unet, sd_model)

        # ── register wrapper (highest-priority — wraps any prior apply_model) ──
        def _mlx_apply_model_wrapper(
            executor, x, t,
            c_concat=None, c_crossattn=None,
            control=None, transformer_options=None, **kwargs
        ):
            # We own the full forward pass; executor is intentionally not called.
            return mlx_pipe.apply_model(
                x, t,
                c_concat=c_concat, c_crossattn=c_crossattn,
                control=control,
                transformer_options=transformer_options or {},
                **kwargs,
            )

        forge_objects.unet.add_wrapper_with_key(
            WrappersMP.APPLY_MODEL, "forge_mlx", _mlx_apply_model_wrapper
        )

        # Store reference so the pipe is not GC'd
        sd_model.mlx_pipeline = mlx_pipe

        print(_BANNER)
        log.info("[MLX Pipeline] SDXL UNet registered on Metal in bfloat16")
        return True

    except Exception as exc:
        log.warning("[MLX Pipeline] Activation failed: %s", exc, exc_info=True)
        print(
            f"\n[MLX Pipeline] ⚠  Could not activate ({exc})\n"
            "  Falling back to standard PyTorch/MPS pipeline.\n"
        )
        return False
