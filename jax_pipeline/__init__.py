"""
jax_pipeline — JAX JIT SDXL UNet backend.

Opt-in via --forge-jax-pipeline. Replaces only the UNet forward pass
(apply_model) with a jax.jit-compiled implementation; CLIP, VAE, and the
sampler loop remain on PyTorch. JAX's tracing model recompiles/caches
automatically per input shape (no manual guard bookkeeping), and its
Memories API (pinned_host sharding) can stream UNet weights from host RAM
to device on demand to relieve VRAM pressure.

Requirements
------------
* ``jax`` (and a matching ``jaxlib``) installed — see requirements_jax.txt
* --forge-jax-pipeline launch flag
* SDXL checkpoint (detected via ``sd_model.is_sdxl``)

Activation
----------
Called from ``modules_forge/forge_loader.py`` after ldm model loading
completes, gated behind ``cmd_opts.forge_jax_pipeline``. If jax is
unavailable or activation fails, returns False silently (loud log message)
and the standard PyTorch pipeline is used.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║           JAX Pipeline Activated                                    ║
║                                                                      ║
║  SDXL UNet   → JAX jax.jit compiled forward pass                    ║
║  Backend     → {backend:<54}║
║  Host offload→ {offload:<54}║
║  CLIP / VAE  → unchanged (PyTorch)                                   ║
║  Samplers    → unchanged (PyTorch / k-diffusion)                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def is_jax_available() -> bool:
    """Return True if the jax package is importable and reports a usable device."""
    try:
        import jax
        return len(jax.devices()) > 0
    except Exception:
        return False


def get_jax_backend() -> str:
    """Return jax's default backend string (e.g. 'gpu', 'tpu', 'cpu'), or '' if unavailable."""
    try:
        import jax
        return jax.default_backend()
    except Exception:
        return ""


def maybe_activate(sd_model, forge_objects) -> bool:
    """Auto-activate the JAX pipeline for SDXL when --forge-jax-pipeline is set.

    Called from ``modules_forge/forge_loader.py`` after normal ldm loading,
    gated by ``cmd_opts.forge_jax_pipeline``.

    Parameters
    ----------
    sd_model      : loaded sd_model (needs ``is_sdxl`` attribute)
    forge_objects : ForgeObjects namedtuple (needs ``.unet`` UnetPatcher)

    Returns
    -------
    True  — JAX pipeline installed and apply_model wrapper registered.
    False — conditions not met or installation failed (standard path used).
    """
    if not getattr(sd_model, "is_sdxl", False):
        return False

    if not is_jax_available():
        log.info(
            "[JAX Pipeline] --forge-jax-pipeline set but jax is not available. "
            "Install with: pip install -r requirements_jax.txt"
        )
        return False

    try:
        from jax_pipeline.pipeline import JAXSDXLPipeline
        from ldm_patched.modules.patcher_extension import WrappersMP

        jax_pipe = JAXSDXLPipeline(forge_objects.unet, sd_model)

        def _jax_apply_model_wrapper(
            executor, x, t,
            c_concat=None, c_crossattn=None,
            control=None, transformer_options=None, **kwargs
        ):
            result = jax_pipe.apply_model(
                x, t,
                c_concat=c_concat, c_crossattn=c_crossattn,
                control=control,
                transformer_options=transformer_options or {},
                **kwargs,
            )
            if result is None:
                # Unsupported conditioning for this call (e.g. active ControlNet) —
                # fall through to the standard PyTorch UNet for this call only.
                return executor(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                 control=control, transformer_options=transformer_options,
                                 **kwargs)
            return result

        forge_objects.unet.add_wrapper_with_key(
            WrappersMP.APPLY_MODEL, "forge_jax", _jax_apply_model_wrapper
        )

        sd_model.jax_pipeline = jax_pipe

        print(_BANNER.format(
            backend=get_jax_backend() or "unknown",
            offload="enabled" if jax_pipe.host_offload else "disabled",
        ))
        log.info("[JAX Pipeline] SDXL UNet registered on backend=%s", get_jax_backend())
        return True

    except Exception as exc:
        log.warning("[JAX Pipeline] Activation failed: %s", exc, exc_info=True)
        print(
            f"\n[JAX Pipeline] Could not activate ({exc})\n"
            "  Falling back to standard PyTorch pipeline.\n"
        )
        return False
