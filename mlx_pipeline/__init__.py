"""
mlx_pipeline — Apple MLX BFloat16 sampling pipeline for SDXL.

Auto-activates on Apple Silicon Macs when the ``mlx`` package is installed.
Replaces both the UNet forward pass *and* the sampler loop with native
MLX bfloat16 implementations that run on the Metal GPU via Apple's
unified memory.

Optimisations over the default PyTorch/MPS path
------------------------------------------------
1. **BFloat16 UNet on Metal** — no fp16 overflow, full fp32 exponent range.
2. **Conditioning pre-conversion** — cross-attention & pooled CLIP-G tensors
   are converted to MLX *once* per generation, not every denoising step.
3. **Batched CFG** — cond and uncond concatenated into a single UNet forward
   (batch=2) instead of two separate calls.
4. **MLX sampler loop** — Euler, Euler-a, Heun, DPM++ 2M / SDE / 2M-SDE /
   3M-SDE run entirely in MLX; no torch tensors are created mid-loop.

Requirements
------------
* macOS on Apple Silicon (M-series chip)
* ``mlx`` Python package installed  (``pip install mlx``)
* SDXL checkpoint (detected automatically via ``sd_model.is_sdxl``)

Activation
----------
Called automatically from ``modules_forge/forge_loader.py`` after ldm model
loading completes.  No flags or user action required on compatible hardware.
If MLX is unavailable the function returns False silently and the standard
PyTorch/MPS pipeline is used.
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
║  SDXL UNet   → BFloat16 on Apple Silicon Metal via MLX              ║
║  CLIP L+G    → BFloat16 text encoding on Metal                      ║
║  VAE Decoder → BFloat16 decoding on Metal                           ║
║  Samplers    → Euler / Euler-a / Heun / DPM++ 2M·SDE·3M-SDE        ║
║  CFG         → single batched UNet forward (batch=2)                ║
║  Conditioning → pre-converted once per generation                   ║
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
        return mx.metal.is_available()
    except Exception:
        return False


# ── sampler hook ──────────────────────────────────────────────────────────────

_sampler_hook_installed: bool = False


def _install_mlx_sampler_hook() -> None:
    """
    Monkey-patch ``KDiffusionSampler.sample`` *once* so that whenever an
    MLX pipeline is active and the chosen sampler has an MLX equivalent,
    the sampler loop runs entirely in MLX.

    The patch is idempotent — calling this multiple times is safe.

    Strategy
    --------
    Rather than rewriting all the sigma-scheduling and callback plumbing
    in ``KDiffusionSampler.sample``, we let the original method do its
    setup and only swap ``self.func`` (the inner sampler function) for the
    duration of the call::

        orig.sample(…)
            └── self.func = MLX wrapper  ← swapped in for this call
                    ├── build MLXCFGDenoiser  (cond/uncond pre-converted)
                    └── MLX sampler loop
            └── self.func = restored      ← always restored in finally
    """
    global _sampler_hook_installed
    if _sampler_hook_installed:
        return

    try:
        from modules.sd_samplers_kdiffusion import KDiffusionSampler
        from mlx_pipeline.samplers import MLX_SAMPLER_MAP, make_mlx_func
        from mlx_pipeline.mlx_denoiser import MLXCFGDenoiser
        import modules.shared as _shared
    except ImportError as e:
        log.debug("[MLX Pipeline] Sampler hook skipped (import error): %s", e)
        return

    _orig_sample = KDiffusionSampler.sample

    def _patched_sample(
        self,
        p,
        x,
        conditioning,
        unconditional_conditioning,
        steps=None,
        image_conditioning=None,
    ):
        # Check if MLX pipeline is active on the current model
        mlx_pipe = getattr(getattr(_shared, "sd_model", None), "mlx_pipeline", None)
        funcname = self.funcname if isinstance(self.funcname, str) else None

        if mlx_pipe is None or funcname not in MLX_SAMPLER_MAP:
            # Fall through to original (non-MLX) sampler
            return _orig_sample(
                self, p, x, conditioning, unconditional_conditioning,
                steps, image_conditioning,
            )

        # Build MLXCFGDenoiser lazily inside the sample call so it has
        # access to the final sampler_extra_args (set by initialize()).
        # We use a sentinel to detect the first self.func call.
        latent_shape = tuple(x.shape)
        mlx_fn       = MLX_SAMPLER_MAP[funcname]
        orig_func    = self.func
        _built       = [False]

        def _lazy_func(model_wrap_cfg, x_torch, sigmas=None, extra_args=None,
                        callback=None, disable=None, **kwargs):
            if not _built[0]:
                _built[0] = True
                denoiser = MLXCFGDenoiser(
                    mlx_pipe._mlx_unet,
                    mlx_pipe.model_sampling,
                    extra_args or {},
                    latent_shape,
                )
                wrapped_fn = make_mlx_func(mlx_fn, denoiser)
                # Cache on the wrapper so it is not rebuilt on re-entry
                _lazy_func._wrapped = wrapped_fn
            return _lazy_func._wrapped(
                model_wrap_cfg, x_torch, sigmas=sigmas,
                extra_args=extra_args, callback=callback,
                disable=disable, **kwargs,
            )

        self.func = _lazy_func
        try:
            result = _orig_sample(
                self, p, x, conditioning, unconditional_conditioning,
                steps, image_conditioning,
            )
        finally:
            self.func = orig_func   # always restore

        return result

    _patched_sample.__wrapped__ = _orig_sample
    KDiffusionSampler.sample    = _patched_sample
    _sampler_hook_installed     = True
    log.info(
        "[MLX Pipeline] KDiffusionSampler hooked — MLX samplers: %s",
        ", ".join(sorted(MLX_SAMPLER_MAP)),
    )


# ── public entry point ───────────────────────────────────────────────────────

def maybe_activate(sd_model, forge_objects) -> bool:
    """Auto-activate the MLX pipeline for SDXL on Apple Silicon.

    Called from ``modules_forge/forge_loader.py`` after normal ldm loading.

    Parameters
    ----------
    sd_model      : loaded sd_model (needs ``is_sdxl`` attribute)
    forge_objects : ForgeObjects namedtuple (needs ``.unet`` UnetPatcher)

    Returns
    -------
    True  — MLX pipeline installed and wrappers registered.
    False — conditions not met or installation failed (standard path used).
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

        # ── 1. apply_model wrapper (UNet forward pass) ────────────────────────
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

        # ── 2. sampler loop hook (eliminates per-step torch↔MLX conversions) ──
        _install_mlx_sampler_hook()

        # ── 3. CLIP text encoders (bfloat16 on Metal) ────────────────────────
        try:
            from mlx_pipeline.clip import install_clip_hooks
            install_clip_hooks(sd_model, forge_objects)
        except Exception as _clip_exc:
            log.warning("[MLX Pipeline] CLIP hook skipped: %s", _clip_exc)

        # ── 4. VAE decoder (bfloat16 on Metal) ───────────────────────────────
        try:
            from mlx_pipeline.vae import install_vae_hooks
            install_vae_hooks(sd_model)
        except Exception as _vae_exc:
            log.warning("[MLX Pipeline] VAE hook skipped: %s", _vae_exc)

        # Keep a reference so the pipeline is not garbage-collected
        sd_model.mlx_pipeline = mlx_pipe

        print(_BANNER)
        log.info("[MLX Pipeline] SDXL registered — Metal bfloat16 + MLX samplers + CLIP + VAE")
        return True

    except Exception as exc:
        log.warning("[MLX Pipeline] Activation failed: %s", exc, exc_info=True)
        print(
            f"\n[MLX Pipeline] ⚠  Could not activate ({exc})\n"
            "  Falling back to standard PyTorch/MPS pipeline.\n"
        )
        return False
