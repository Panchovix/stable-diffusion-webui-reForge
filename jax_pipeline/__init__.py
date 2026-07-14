"""
jax_pipeline — JAX JIT SDXL UNet backend.

Opt-in via --forge-jax-pipeline. Replaces the UNet forward pass
(apply_model), CLIP-L/G text encoding, and VAE decode with jax.jit-compiled
implementations; the sampler loop remains on PyTorch/k-diffusion. Routing
CLIP+VAE through JAX too (not just the UNet) keeps everything in one XLA
memory pool instead of two allocators competing for VRAM — PyTorch's
allocator and JAX's default `XLA_PYTHON_CLIENT_PREALLOCATE` pool fighting
over the same GPU was the original motivation for porting these. JAX's
tracing model recompiles/caches automatically per input shape (no manual
guard bookkeeping), and its Memories API (pinned_host sharding) can stream
UNet weights from host RAM to device on demand to relieve VRAM pressure
further.

CLIP and VAE hooks are independent of UNet activation and of each other:
if either fails to install, that component falls back to PyTorch while the
others stay on JAX.

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

import importlib.util
import logging
import os

log = logging.getLogger(__name__)


def _apply_cuda_nvcc_namespace_package_workaround() -> None:
    """Work around a jax==0.4.35 import-time crash.

    jax._src.lib._cuda_path() -> _try_cuda_nvcc_import() does
    ``pathlib.Path(cuda_nvcc.__file__).parent`` to locate the pip-installed
    ``nvidia-cuda-nvcc-cu12`` package. That package ships as a PEP 420
    namespace package (no ``__init__.py``), so ``__file__`` is None and the
    pathlib call raises ``TypeError: argument should be a str or an
    os.PathLike object ... not 'NoneType'`` — unconditionally, at
    ``import jax`` time, even for CPU-only use (this is not gated behind
    actually using the CUDA backend).

    ``_cuda_path()`` checks the ``CUDA_ROOT`` env var FIRST and returns
    immediately if set, short-circuiting before the buggy code path runs.
    We locate the same package via ``importlib.util.find_spec`` (safe,
    since it only reads ``submodule_search_locations``, never touches the
    missing ``__file__``) and set ``CUDA_ROOT`` ourselves if not already
    set by the user/launcher.
    """
    if os.environ.get("CUDA_ROOT"):
        return
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except Exception:
        return
    if spec is None or not spec.submodule_search_locations:
        return
    os.environ["CUDA_ROOT"] = next(iter(spec.submodule_search_locations))


def _apply_vram_preallocation_default() -> None:
    """Default JAX to on-demand GPU memory growth instead of its 75%
    upfront preallocation.

    Per JAX's own GPU memory docs, XLA preallocates 75% of total GPU memory
    on first use and never gives it back for the life of the process —
    fine when JAX owns the whole GPU, but this backend shares the card with
    PyTorch (other webui components: ControlNet preprocessors, upscalers,
    extensions, and — if CLIP/VAE hooks fail to install — the CLIP/VAE
    fallback path itself). On small-VRAM cards (the exact audience for
    --forge-jax-pipeline's host-offload feature) a permanent 75% land-grab
    starves everything else.

    Only sets the env var if the user/launcher hasn't already configured
    JAX's allocator themselves (respects XLA_PYTHON_CLIENT_PREALLOCATE or
    XLA_PYTHON_CLIENT_MEM_FRACTION if either is already present).
    """
    if "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        return
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║           JAX Pipeline Activated                                    ║
║                                                                      ║
║  SDXL UNet   → JAX jax.jit compiled forward pass                    ║
║  Backend     → {backend:<54}║
║  Host offload→ {offload:<54}║
║  CLIP        → {clip:<54}║
║  VAE decode  → {vae:<54}║
║  Samplers    → unchanged (PyTorch / k-diffusion)                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _jax_probe():
    """Return (available: bool, reason: str) — reason is empty on success.

    Bare ``except Exception: return False`` hides exactly the information
    needed to debug a failed activation (jax not installed vs. import
    succeeding but the CUDA plugin failing to initialize vs. jax.devices()
    returning no devices), so this is deliberately verbose.
    """
    _apply_cuda_nvcc_namespace_package_workaround()
    _apply_vram_preallocation_default()
    try:
        import jax
    except Exception as e:
        return False, f"import jax failed: {e!r}"
    try:
        devices = jax.devices()
    except Exception as e:
        return False, f"jax.devices() raised: {e!r}"
    if not devices:
        return False, "jax.devices() returned no devices"
    return True, ""


def is_jax_available() -> bool:
    """Return True if the jax package is importable and reports a usable device."""
    available, _reason = _jax_probe()
    return available


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

    available, reason = _jax_probe()
    if not available:
        msg = (
            f"[JAX Pipeline] --forge-jax-pipeline set but jax is not usable: {reason}. "
            "Install with: pip install -r requirements_jax.txt"
        )
        log.warning(msg)
        print(f"\n{msg}\n")
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

        # CLIP and VAE hooks are independent of the UNet wrapper above and of
        # each other — a failure in either falls back to the standard
        # PyTorch path for that component only, it does not undo UNet
        # activation (same pattern as mlx_pipeline).
        clip_active = False
        try:
            from jax_pipeline.clip import install_clip_hooks
            clip_active = install_clip_hooks(sd_model, forge_objects)
        except Exception as clip_exc:
            log.warning("[JAX Pipeline] CLIP hook skipped: %s", clip_exc)

        vae_active = False
        try:
            from jax_pipeline.vae import install_vae_hooks
            vae_active = install_vae_hooks(sd_model)
        except Exception as vae_exc:
            log.warning("[JAX Pipeline] VAE hook skipped: %s", vae_exc)

        print(_BANNER.format(
            backend=get_jax_backend() or "unknown",
            offload="enabled" if jax_pipe.host_offload else "disabled",
            clip="active (jax.jit)" if clip_active else "PyTorch (hook failed, see log)",
            vae="active (jax.jit)" if vae_active else "PyTorch (hook failed, see log)",
        ))
        log.info(
            "[JAX Pipeline] SDXL UNet registered on backend=%s clip_active=%s vae_active=%s",
            get_jax_backend(), clip_active, vae_active,
        )
        return True

    except Exception as exc:
        log.warning("[JAX Pipeline] Activation failed: %s", exc, exc_info=True)
        print(
            f"\n[JAX Pipeline] Could not activate ({exc})\n"
            "  Falling back to standard PyTorch pipeline.\n"
        )
        return False
