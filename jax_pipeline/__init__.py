"""
jax_pipeline — JAX JIT SDXL UNet backend.

Opt-in via --forge-jax-pipeline. Replaces the UNet forward pass
(apply_model), CLIP-L/G text encoding, and VAE decode with jax.jit-compiled
implementations. Routing CLIP+VAE through JAX too (not just the UNet)
keeps everything in one XLA memory pool instead of two allocators
competing for VRAM — PyTorch's allocator and JAX's default
`XLA_PYTHON_CLIENT_PREALLOCATE` pool fighting over the same GPU was the
original motivation for porting these. JAX's tracing model recompiles/
caches automatically per input shape (no manual guard bookkeeping), and
its Memories API (pinned_host sharding) can stream UNet weights from host
RAM to device on demand to relieve VRAM pressure further.

For samplers with a JAX-native implementation (jax_pipeline.samplers —
Euler, Euler ancestral, Heun, DPM++ 2M/SDE/2M SDE/3M SDE), the entire
sampling loop runs in JAX end to end, not just the per-step UNet forward:
the latent stays resident in JAX arrays for all 20-50 steps instead of
round-tripping through torch<->JAX conversion every single step (see
jax_pipeline.samplers's module docstring for the data-flow diagram). Other
samplers fall back to the standard per-step apply_model path (still
JAX-accelerated per step, just without the whole-loop residency).

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

Environment variables
----------------------
Set automatically (unless already set by you): ``CUDA_ROOT``,
``XLA_PYTHON_CLIENT_PREALLOCATE=false``, ``XLA_PYTHON_CLIENT_ALLOCATOR=
platform`` (the only JAX GPU allocator that actually returns freed memory
to the CUDA driver — without it, PhaseManager's device<->host moves don't
translate into real free VRAM; see ``_apply_platform_allocator_default``),
``XLA_FLAGS`` (adds ``--xla_gpu_strict_conv_algorithm_picker=false``). See
``_apply_*_workaround``/``_apply_*_default`` below for why.

Opt-in only (NOT set automatically — see ``_apply_vmm_allocator_default``
for why): ``JAX_PIPELINE_VMM_ALLOCATOR=1`` switches the GPU allocator to
JAX's experimental CUDA VMM allocator. Confirmed to break CUDA backend
registration entirely (silent fallback to CPU) against this project's
pinned jaxlib==0.4.34 — only set this if you've upgraded jaxlib and
confirmed it works for you.
"""

from __future__ import annotations

import importlib.util
import logging
import os

log = logging.getLogger(__name__)

_sampler_hook_installed: bool = False


def _find_cuda_nvcc_spec():
    """Locate the nvidia-cuda-nvcc-cu12 package spec, if installed.

    Shared by the CUDA_ROOT workaround and the CUDA-backend detection used
    to decide on the VMM allocator default — both need an answer before
    ``import jax`` runs, so neither can use ``jax.default_backend()`` yet.
    Presence of this package (pulled in by the ``jax[cuda12]`` extra) is
    used as the CUDA-backend proxy signal.
    """
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return spec


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
    spec = _find_cuda_nvcc_spec()
    if spec is None:
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


def _apply_platform_allocator_default() -> None:
    """Default JAX's GPU allocator to "platform" — the only one that
    actually returns freed memory to the CUDA driver.

    This is the fix for a problem PhaseManager alone can't solve: JAX's
    default allocator (BFC, best-fit-with-coalescing — still the default
    even with XLA_PYTHON_CLIENT_PREALLOCATE=false) is a *caching*
    allocator. It recycles freed device buffers for JAX's own future
    allocations, but per JAX's own docs, "platform" is "the only
    configuration that deallocates memory" — BFC never gives memory back
    to the driver. That means PhaseManager's device<->host moves
    (jax.device_put(params, pinned_host_sharding)) relocate the *logical*
    data, but the underlying VRAM arena XLA claimed while UNet/CLIP were
    active does not shrink back down — torch (or JAX's own next, unrelated
    allocation) sees no more free VRAM than before the "offload" happened.
    Symptom observed without this: torch reporting under 1% free VRAM
    immediately after a JAX component was supposedly offloaded, and a
    single JAX allocation failure being enough to permanently starve the
    rest of the process — nothing was ever actually being freed.

    JAX's docs call "platform" "very slow" (it does a real cudaMalloc/
    cudaFree per allocation instead of reusing a cache) — a real tradeoff,
    but on a card small enough for PhaseManager to have engaged at all
    (<20GB total VRAM), correctness/stability matters far more than the
    per-call allocator overhead.

    Ordered before ``_apply_vmm_allocator_default`` and respects the same
    opt-in: if you've set JAX_PIPELINE_VMM_ALLOCATOR=1 yourself, this
    backs off and lets that apply instead. Also backs off if you've set
    XLA_PYTHON_CLIENT_ALLOCATOR yourself, or if preallocation ends up
    enabled (this only matters when it's disabled).
    """
    if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
        return
    if os.environ.get("JAX_PIPELINE_VMM_ALLOCATOR", "").strip().lower() in ("1", "true"):
        return
    preallocate_off = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").strip().lower() in ("false", "0")
    if not preallocate_off:
        return
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _apply_vmm_allocator_default() -> None:
    """NOT auto-applied — see ``JAX_PIPELINE_VMM_ALLOCATOR`` below. Kept as
    an explicit opt-in, not a default.

    JAX's experimental ``XLA_PYTHON_CLIENT_ALLOCATOR=vmm`` (CUDA virtual
    memory management allocator, cuMemAddressReserve/cuMemMap) is the
    right tool for the fragmentation ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
    reintroduces — JAX's default BFC allocator is built around one large
    preallocated arena and fragments piecemeal without it. But this
    setting's documentation is only current for JAX's main/latest branch;
    it broke CUDA backend registration entirely (jax.devices() silently
    fell back to a CPU-only device list, no error surfaced) when tried
    against this project's pinned jaxlib==0.4.34 — that build most likely
    predates "vmm" support, and there is no way to detect that in advance
    or recover within the same process (env vars controlling the XLA
    client's allocator only take effect before the very first
    ``import jax`` / client creation; once that's happened, re-importing
    with a different XLA_PYTHON_CLIENT_ALLOCATOR value has no effect).

    Because a silent CPU fallback is a much worse failure mode than
    forgoing this optimization, it is not auto-applied. Set
    ``JAX_PIPELINE_VMM_ALLOCATOR=1`` yourself (e.g. if you've upgraded to
    a newer jaxlib you've confirmed supports it) to opt in explicitly.
    """
    if os.environ.get("JAX_PIPELINE_VMM_ALLOCATOR", "").strip().lower() not in ("1", "true"):
        return
    preallocate_off = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").strip().lower() in ("false", "0")
    if not preallocate_off:
        return
    if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
        return
    if _find_cuda_nvcc_spec() is None:
        return
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"


def _apply_xla_conv_autotune_workaround() -> None:
    """Relax XLA's GPU convolution-algorithm autotuner so it accepts a
    fallback algorithm instead of failing outright.

    Symptom without this: ``RESOURCE_EXHAUSTED: Out of memory while trying
    to allocate N bytes`` the first time a convolution runs at a new
    shape — even though the convolution itself would fit comfortably.
    XLA's cuDNN autotuner benchmarks several candidate algorithms per
    (shape, dtype) and needs its own scratch-memory allocation for each
    candidate; on small-VRAM cards this scratch allocation can itself OOM.
    This is more likely with XLA_PYTHON_CLIENT_PREALLOCATE=false (see
    ``_apply_vram_preallocation_default`` above) since there's no large
    reserved arena to draw the scratch allocation from — exactly the
    tradeoff that setting buys back some of. XLA's own error message
    recommends this exact flag.

    Only appends the flag if it isn't already present in XLA_FLAGS,
    preserving whatever else the user/launcher already set there.
    """
    flag = "--xla_gpu_strict_conv_algorithm_picker=false"
    existing = os.environ.get("XLA_FLAGS", "")
    if flag in existing:
        return
    os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║           JAX Pipeline Activated                                    ║
║                                                                      ║
║  SDXL UNet   → JAX jax.jit compiled forward pass                    ║
║  Backend     → {backend:<54}║
║  Host offload→ {offload:<54}║
║  CLIP        → {clip:<54}║
║  VAE decode  → {vae:<54}║
║  Samplers    → {samplers:<54}║
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
    _apply_platform_allocator_default()
    _apply_vmm_allocator_default()
    _apply_xla_conv_autotune_workaround()
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


def _install_jax_sampler_hook() -> None:
    """Monkey-patch ``KDiffusionSampler.sample`` *once* so that whenever a
    JAX pipeline is active and the chosen sampler has a JAX-native
    implementation, the ENTIRE sampling loop runs in JAX — not just the
    per-step UNet forward — eliminating the torch<->JAX round trip at
    every single step.

    The patch is idempotent — calling this multiple times is safe.

    Strategy — identical to mlx_pipeline's ``_install_mlx_sampler_hook``:
    rather than rewriting all the sigma-scheduling and callback plumbing
    in ``KDiffusionSampler.sample``, let the original method do its setup
    and only swap ``self.func`` (the inner sampler function) for the
    duration of the call::

        orig.sample(...)
            +-- self.func = JAX wrapper  <- swapped in for this call
                    +-- build JAXCFGDenoiser (cond/uncond pre-converted)
                    +-- JAX sampler loop
            +-- self.func = restored     <- always restored in finally

    If the chosen sampler has no JAX implementation, or building the
    denoiser fails for a non-OOM reason, this falls back to the standard
    per-call path — which, since ``WrappersMP.APPLY_MODEL`` is a separate,
    independent hook from this ``self.func`` swap, still routes each
    step's UNet forward through JAX via ``apply_model()`` (with its own
    phase-activation and OOM handling); only the whole-loop residency
    optimization is lost, not JAX acceleration entirely. An OOM anywhere
    in this path aborts via ``host_offload.handle_jax_oom`` like every
    other JAX entry point.
    """
    global _sampler_hook_installed
    if _sampler_hook_installed:
        return

    try:
        from modules.sd_samplers_kdiffusion import KDiffusionSampler
        from jax_pipeline.samplers import JAX_SAMPLER_MAP, make_jax_func
        from jax_pipeline.jax_denoiser import JAXCFGDenoiser
        import modules.shared as _shared
    except ImportError as e:
        log.debug("[JAX Pipeline] Sampler hook skipped (import error): %s", e)
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
        # Check if a JAX pipeline is active on the current model
        jax_pipe = getattr(getattr(_shared, "sd_model", None), "jax_pipeline", None)
        funcname = self.funcname if isinstance(self.funcname, str) else None

        if jax_pipe is None or funcname not in JAX_SAMPLER_MAP:
            # Fall through to original (non-whole-loop) path
            return _orig_sample(
                self, p, x, conditioning, unconditional_conditioning,
                steps, image_conditioning,
            )

        phase_manager = getattr(jax_pipe, "_phase_manager", None)

        # -- LoRA / DoRA hot-sync + phase activation (sampler-loop path) --
        # When the JAX sampler loop is active, jax_pipe.apply_model() is
        # never called — JAXCFGDenoiser goes straight to the UNet forward
        # function, bypassing both the LoRA reload AND the
        # phase_manager.activate("unet") call that normally live inside
        # apply_model(). Both need to happen here instead, before the
        # denoiser is built, for exactly the same reason mlx_pipeline's
        # hook re-syncs LoRA here: patch_weight_to_device() has already
        # merged any new LoRA/DoRA deltas into the PyTorch diffusion_model
        # by this point; unet_patcher.patches_uuid is a cheap UUID4
        # comparison, not a weight hash.
        try:
            from jax_pipeline.lora import reload_jax_unet_weights, unet_lora_sig
            cur_uuid = unet_lora_sig(jax_pipe.unet_patcher)
            if cur_uuid != jax_pipe._lora_sig:
                reload_jax_unet_weights(jax_pipe, jax_pipe.unet_patcher)
                jax_pipe._lora_sig = cur_uuid
        except Exception as lora_exc:
            log.warning("[JAX LoRA] Sampler-hook LoRA sync failed: %s", lora_exc)

        if phase_manager is not None:
            phase_manager.activate("unet")

        # Build JAXCFGDenoiser lazily inside the sample call so it has
        # access to the final sampler_extra_args (set by initialize()).
        # We use sentinels to detect the first self.func call and to
        # remember a failed build across the remaining steps of this call
        # (rebuilding — and re-failing — on every step would be wasteful).
        latent_shape = tuple(x.shape)
        jax_fn = JAX_SAMPLER_MAP[funcname]
        orig_func = self.func
        _built = [False]
        _build_failed = [False]

        def _lazy_func(model_wrap_cfg, x_torch, sigmas=None, extra_args=None,
                        callback=None, disable=None, **kwargs):
            if _build_failed[0]:
                return orig_func(model_wrap_cfg, x_torch, sigmas=sigmas,
                                  extra_args=extra_args, callback=callback,
                                  disable=disable, **kwargs)

            if not _built[0]:
                _built[0] = True
                try:
                    denoiser = JAXCFGDenoiser(
                        jax_pipe._forward, jax_pipe._params,
                        jax_pipe.model_sampling, extra_args or {}, latent_shape,
                        phase_manager=phase_manager,
                    )
                    wrapped_fn = make_jax_func(jax_fn, denoiser)
                    # Cache on the wrapper so it is not rebuilt on re-entry
                    _lazy_func._wrapped = wrapped_fn
                except Exception as build_exc:
                    from jax_pipeline.host_offload import _is_jax_oom_exception, handle_jax_oom
                    if _is_jax_oom_exception(build_exc):
                        handle_jax_oom(phase_manager, "sampler-loop denoiser build", build_exc)  # raises
                    log.warning(
                        "[JAX Pipeline] Sampler-loop denoiser build failed (%s), "
                        "falling back to the standard per-step path.", build_exc,
                    )
                    _build_failed[0] = True
                    return orig_func(model_wrap_cfg, x_torch, sigmas=sigmas,
                                      extra_args=extra_args, callback=callback,
                                      disable=disable, **kwargs)

            try:
                return _lazy_func._wrapped(
                    model_wrap_cfg, x_torch, sigmas=sigmas,
                    extra_args=extra_args, callback=callback,
                    disable=disable, **kwargs,
                )
            except Exception as run_exc:
                from jax_pipeline.host_offload import _is_jax_oom_exception, handle_jax_oom
                if _is_jax_oom_exception(run_exc):
                    handle_jax_oom(phase_manager, "JAX sampler loop", run_exc)  # raises
                raise

        self.func = _lazy_func
        try:
            result = _orig_sample(
                self, p, x, conditioning, unconditional_conditioning,
                steps, image_conditioning,
            )
        finally:
            self.func = orig_func  # always restore

        return result

    _patched_sample.__wrapped__ = _orig_sample
    KDiffusionSampler.sample = _patched_sample
    _sampler_hook_installed = True
    log.info(
        "[JAX Pipeline] KDiffusionSampler hooked — JAX samplers: %s",
        ", ".join(sorted(JAX_SAMPLER_MAP)),
    )


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
        from jax_pipeline import host_offload
        from jax_pipeline.pipeline import JAXSDXLPipeline
        from ldm_patched.modules.patcher_extension import WrappersMP

        # One PhaseManager per model load, shared by UNet/CLIP/VAE below —
        # on VRAM-constrained cards (<20GB total) it keeps only whichever
        # component is actively in use resident on device, offloading the
        # others to pinned host memory, instead of all three (UNet+CLIP-L+
        # CLIP-G+VAE) sitting on device simultaneously for the entire
        # session. No-ops on generously-sized cards.
        phase_manager = host_offload.PhaseManager(host_offload.get_default_device())
        sd_model._jax_phase_manager = phase_manager  # keep alive

        jax_pipe = JAXSDXLPipeline(forge_objects.unet, sd_model, phase_manager=phase_manager)

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

        # Whole-loop sampler hook: independent of the per-step apply_model
        # wrapper above — installed unconditionally (idempotent, and it
        # self-gates per generation on whether the chosen sampler has a
        # JAX implementation), never blocks UNet activation if it fails.
        sampler_hook_active = False
        try:
            _install_jax_sampler_hook()
            sampler_hook_active = _sampler_hook_installed
        except Exception as sampler_exc:
            log.warning("[JAX Pipeline] Sampler hook skipped: %s", sampler_exc)

        # CLIP and VAE hooks are independent of the UNet wrapper above and of
        # each other — a failure in either falls back to the standard
        # PyTorch path for that component only, it does not undo UNet
        # activation (same pattern as mlx_pipeline).
        clip_active = False
        try:
            from jax_pipeline.clip import install_clip_hooks
            clip_active = install_clip_hooks(sd_model, forge_objects, phase_manager=phase_manager)
        except Exception as clip_exc:
            log.warning("[JAX Pipeline] CLIP hook skipped: %s", clip_exc)

        vae_active = False
        try:
            from jax_pipeline.vae import install_vae_hooks
            vae_active = install_vae_hooks(sd_model, phase_manager=phase_manager)
        except Exception as vae_exc:
            log.warning("[JAX Pipeline] VAE hook skipped: %s", vae_exc)

        offload_desc = "enabled" if jax_pipe.host_offload else "disabled"
        if phase_manager.enabled:
            offload_desc = "phase-managed (<20GB VRAM, cycles per phase)"

        print(_BANNER.format(
            backend=get_jax_backend() or "unknown",
            offload=offload_desc,
            clip="active (jax.jit)" if clip_active else "PyTorch (hook failed, see log)",
            vae="active (jax.jit)" if vae_active else "PyTorch (hook failed, see log)",
            samplers=(
                "whole-loop in JAX (Euler/Heun/DPM++...)" if sampler_hook_active
                else "per-step only (hook failed, see log)"
            ),
        ))
        log.info(
            "[JAX Pipeline] SDXL UNet registered on backend=%s clip_active=%s vae_active=%s "
            "sampler_hook_active=%s phase_managed=%s",
            get_jax_backend(), clip_active, vae_active, sampler_hook_active, phase_manager.enabled,
        )
        return True

    except Exception as exc:
        log.warning("[JAX Pipeline] Activation failed: %s", exc, exc_info=True)
        print(
            f"\n[JAX Pipeline] Could not activate ({exc})\n"
            "  Falling back to standard PyTorch pipeline.\n"
        )
        return False


def release_model_memory(sd_model, release: bool = True) -> None:
    """Force-cleanup any JAX-resident memory held by ``sd_model``'s
    jax_pipeline components (UNet/CLIP/VAE), if present.

    Call this from a checkpoint-switch / model-teardown path (see
    ``modules/sd_models.py``'s ``complete_model_teardown`` and
    ``unload_model_weights``). Those functions only know how to walk
    torch.nn.Module parameter/buffer trees — jax_pipeline's state lives in
    plain Python attributes holding JAX arrays, which that traversal never
    touches, so without this the old checkpoint's JAX-side weights would
    only get cleaned up whenever Python's garbage collector eventually
    gets to them (if anything, e.g. a monkey-patched closure, keeps a
    stray reference alive, that might be never).

    ``release=True`` (default): permanently delete the arrays
    (``PhaseManager.force_release_all``) — for a checkpoint switch, where
    a fresh JAXSDXLPipeline/CLIP/VAE gets built from scratch for whatever
    loads next and there's no reason to keep the old checkpoint's weights
    around in any tier.

    ``release=False``: move the arrays to pinned host memory instead
    (``PhaseManager.force_offload_all``) — for "unload model to RAM"
    (recoverable): the model may be reactivated later via
    ``load_model_to_device``, and PhaseManager's own ``activate()`` will
    transparently bring whichever component is needed back on-device on
    its next real use, so no explicit "reload" hook is needed on that
    path.

    Safe to call on any sd_model, JAX-backed or not — no-ops if
    ``_jax_phase_manager`` isn't present, and never raises (best-effort).
    """
    phase_manager = getattr(sd_model, "_jax_phase_manager", None)
    if phase_manager is None:
        return
    try:
        if release:
            phase_manager.force_release_all()
            log.info("[JAX Pipeline] Released JAX GPU memory for %s", getattr(sd_model, "filename", sd_model))
        else:
            phase_manager.force_offload_all()
            log.info("[JAX Pipeline] Offloaded JAX GPU memory for %s", getattr(sd_model, "filename", sd_model))
    except Exception as e:
        log.warning("[JAX Pipeline] Error during JAX memory cleanup: %s", e)
