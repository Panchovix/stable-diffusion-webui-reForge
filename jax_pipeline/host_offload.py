"""
jax_pipeline/host_offload.py — VRAM-relief helper using JAX's Memories API.

Lets the UNet weight pytree live in pinned host RAM and stream to device
automatically as part of a jax.jit-compiled call, via
``jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")`` +
``jax.device_put`` — the pattern documented in JAX's own host-offloading
guide (docs/notebooks/host-offloading.ipynb, "Hybrid Activation and
Parameter Offloading"). This is an alternative to ldm_patched's manual
lowvram/weight-swap machinery, applied only to the JAX-side UNet copy — it
has no effect on the PyTorch UNet, CLIP, or VAE.

v1 scope: single JAX device only (``jax.devices()[0]``); no multi-device
sharding.

Force-cleanup policy
---------------------
With ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` (see jax_pipeline.__init__ —
the only JAX GPU allocator that actually returns freed memory to the CUDA
driver instead of caching it for JAX's own reuse), moving a component off
device via ``PhaseManager._move(..., "pinned_host")`` now genuinely frees
that VRAM rather than just relocating logical data within an arena XLA
never gives back. That makes it worth forcing cleanup at specific,
bounded, event-driven points — rather than the routine per-phase cycling
alone — for three situations research/production use turned up:

1. Checkpoint switch — old JAX arrays are about to be discarded entirely
   (see ``force_release_all`` and ``jax_pipeline.release_model_memory``,
   hooked into ``modules/sd_models.py``'s model-teardown paths).
2. Escape from JAX to a PyTorch code path (see ``escape_to_pytorch``) —
   specifically when JAX hands control BACK to PyTorch for something PyTorch
   itself needs GPU memory for (ControlNet, CLIP textual-inversion/LoRA-
   rebuild-failure fallback, VAE fallback) — not routine phase transitions
   between JAX's own components. PyTorch's allocator has no visibility into
   what JAX/XLA currently holds, so this must be forced proactively rather
   than relying on PyTorch's own OOM handling to somehow negotiate with an
   allocator it can't see.
3. A JAX-raised OOM exception (see ``handle_jax_oom``) — force a cleanup and
   abort the generation with a clear message, rather than silently falling
   back to PyTorch. Earlier debugging showed a failed JAX allocation can
   leave the GPU in a state where an immediate PyTorch fallback attempt
   fails too; aborting and telling the user to retry is more honest than a
   silent continue that's likely to fail again anyway.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)


class JAXOutOfMemoryError(RuntimeError):
    """Raised when a JAX/XLA operation runs out of GPU memory.

    By the time this is raised, JAX's GPU memory has already been
    force-offloaded (see ``handle_jax_oom``) — the caller should NOT
    automatically retry. Surface this to the user and let them decide
    whether to try again (e.g. at a lower resolution, or with less else
    running on the GPU).
    """


def _is_jax_oom_exception(exc: BaseException) -> bool:
    """Detect JAX/XLA-specific out-of-memory exceptions.

    Deliberately narrow — checks for jaxlib's XlaRuntimeError plus a
    RESOURCE_EXHAUSTED/out-of-memory message — so this policy doesn't
    misfire on unrelated bugs (shape errors, missing keys, etc.), which
    should keep falling back gracefully rather than aborting the whole
    generation.
    """
    type_name = type(exc).__name__
    module_name = type(exc).__module__ or ""
    if "XlaRuntimeError" not in type_name and "xla_extension" not in module_name:
        return False
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()


def handle_jax_oom(phase_manager, component: str, exc: BaseException) -> None:
    """Force-cleanup JAX GPU memory and raise ``JAXOutOfMemoryError``.

    Call from an ``except`` block once ``_is_jax_oom_exception(exc)`` is
    True. Always raises — never returns — so callers should place any
    non-OOM fallback logic in a separate branch, not after this call.
    """
    log.warning(
        "[JAX Pipeline] GPU out of memory during %s — forcing JAX memory "
        "cleanup and aborting this generation (not retrying automatically, "
        "not silently falling back to PyTorch for this attempt): %s",
        component, exc,
    )
    if phase_manager is not None:
        try:
            phase_manager.force_offload_all()
        except Exception as cleanup_exc:
            log.warning("[JAX Pipeline] Cleanup after OOM also failed: %s", cleanup_exc)
    raise JAXOutOfMemoryError(
        f"[JAX Pipeline] Ran out of GPU memory during {component}. JAX's GPU "
        "memory has been released — please try again (e.g. at a lower "
        "resolution, or with fewer other things using the GPU right now)."
    ) from exc


def evict_torch_unet_from_gpu(unet_patcher) -> None:
    """Move the torch-side SDXL UNet weights off GPU via ldm_patched's own
    model_management unload path, once JAX no longer needs to read them
    for this generation.

    Only ever called from the JAX whole-loop sampler-hook path (see
    ``jax_pipeline.__init__._install_jax_sampler_hook``) — NOT from
    ``JAXSDXLPipeline.apply_model()``'s per-step path, which can fall back
    to the real torch UNet mid-generation whenever ControlNet conditioning
    is present, and must therefore never assume the torch copy is gone.
    The whole-loop path has no such fallback (``JAXCFGDenoiser`` calls the
    JAX forward function directly, every step, for the whole loop), so
    nothing needs the torch UNet GPU-resident again until the NEXT
    generation's own setup reloads it.

    Why this is needed at all: Forge's ``sampling_prepare()`` (called at
    the start of every generation, including each hires-fix pass)
    unconditionally re-loads and re-patches the torch UNet onto GPU
    regardless of what we did last time — so without this, JAX ends up
    holding a full second copy of the same ~2.5-5GB of weights on GPU
    every single generation, on top of its own copy, which is enough by
    itself to OOM a small (<8GB) card even after trimming JAX's own peak
    usage.

    A raw ``unet_patcher.model.to("cpu")`` would leave
    ``model_management.current_loaded_models``'s bookkeeping stale, so the
    next ``sampling_prepare()`` would think the model is still
    GPU-resident and skip reloading it (silently leaving stale/CPU-only
    weights in place instead of the fresh, correctly LoRA-patched ones).
    Removing the matching ``LoadedModel`` entry after calling its own
    ``model_unload()`` — mirroring exactly what
    ``model_management.free_memory``/``load_models_gpu`` do internally —
    keeps that bookkeeping correct, so the torch UNet is transparently
    reloaded (with fresh LoRA patches) the next time ``sampling_prepare()``
    runs, same as if we'd never touched it.

    Best-effort: never raises. If anything about model_management's
    internals doesn't match what's expected here, this silently no-ops
    and the torch UNet just stays resident (today's behavior) rather than
    risk a half-broken unload.
    """
    try:
        from ldm_patched.modules import model_management
    except Exception:
        return
    try:
        for i, lm in enumerate(model_management.current_loaded_models):
            if lm.model is unet_patcher:
                lm.model_unload()
                model_management.current_loaded_models.pop(i)
                model_management.soft_empty_cache(force=True)
                log.debug(
                    "[JAX Pipeline] Evicted torch-side UNet weights from GPU "
                    "(JAX sampler loop has its own copy)."
                )
                break
    except Exception as e:
        log.debug(
            "[JAX Pipeline] Torch UNet eviction skipped (%s) — torch copy stays resident.", e,
        )


def get_default_device():
    """The single JAX device this pipeline targets (v1: no multi-device sharding)."""
    import jax
    return jax.devices()[0]


def params_nbytes(params: Dict[str, "jnp.ndarray"]) -> int:
    return sum(int(v.nbytes) for v in params.values())


def should_offload(unet_patcher, params: Dict[str, "jnp.ndarray"], headroom_bytes: int = 1024 * 1024 * 1024) -> bool:
    """Decide whether the JAX UNet params should live in pinned host memory.

    Reuses ldm_patched's existing free-VRAM probe (the same one the rest of
    the webui's lowvram logic relies on:
    ``ldm_patched.modules.model_management.get_free_memory``) rather than
    adding a new one. Offload when the UNet's own footprint wouldn't
    comfortably fit in currently-free VRAM alongside the working headroom
    other tensors need.
    """
    try:
        from ldm_patched.modules import model_management
        free_bytes = model_management.get_free_memory(unet_patcher.load_device)
    except Exception as e:
        log.warning("[JAX Pipeline] Could not query free VRAM (%s); defaulting to no host offload.", e)
        return False

    footprint = params_nbytes(params)
    offload = footprint + headroom_bytes > free_bytes
    log.info(
        "[JAX Pipeline] UNet footprint=%.2fGB free_vram=%.2fGB headroom=%.2fGB -> host_offload=%s",
        footprint / 1e9, free_bytes / 1e9, headroom_bytes / 1e9, offload,
    )
    return offload


def place_params(params: Dict[str, "jnp.ndarray"], device, offload: bool) -> Dict[str, "jnp.ndarray"]:
    """Place every array in ``params`` on either pinned host or device memory."""
    import jax

    memory_kind = "pinned_host" if offload else "device"
    sharding = jax.sharding.SingleDeviceSharding(device, memory_kind=memory_kind)
    return jax.tree.map(lambda p: jax.device_put(p, sharding), params)


def make_forward(device, offload: bool):
    """Return a jax.jit-compiled forward function matching
    ``jax_pipeline.unet.unet_forward``'s signature.

    When ``offload`` is True, params are expected to already live in pinned
    host memory (via ``place_params(..., offload=True)``); the returned
    function re-stages them to device memory at the top of its jitted body
    so XLA schedules the host->device transfer as part of the compiled
    program — the same pattern JAX's host-offloading docs use for parameter
    offloading. When False, this is just ``jax.jit(unet_forward)``.

    Used only when ``PhaseManager`` is disabled (large-VRAM cards). When
    enabled, ``PhaseManager`` moves params to device once per phase
    transition instead of re-streaming from host on every single call —
    see ``PhaseManager`` below.
    """
    import jax
    from jax_pipeline.unet import unet_forward

    if not offload:
        return jax.jit(unet_forward)

    device_sharding = jax.sharding.SingleDeviceSharding(device, memory_kind="device")

    def _offloaded_forward(params, sample, timestep, encoder_hidden_states, added_cond_kwargs):
        device_params = jax.tree.map(lambda p: jax.device_put(p, device_sharding), params)
        return unet_forward(device_params, sample, timestep, encoder_hidden_states, added_cond_kwargs)

    return jax.jit(_offloaded_forward)


# -- phase-based component offload --------------------------------------------

class PhaseManager:
    """Coordinates which JAX-resident component (unet / clip / vae) lives on
    GPU device memory at any given moment, offloading every other
    registered component to pinned host memory when a different phase
    becomes active.

    Why this exists: UNet + CLIP-L + CLIP-G + VAE simultaneously resident on
    device can exceed VRAM even when each is individually placed sensibly
    (``should_offload``'s footprint-vs-free-VRAM check is per-component, not
    aware of what ELSE is currently resident). A normal generation only
    ever needs ONE of these three phases active at a time — CLIP encodes
    the prompt, then UNet samples for many steps, then VAE decodes once —
    so cycling components between device and pinned host memory at phase
    boundaries (not per-call — ``activate()`` is a no-op if the same phase
    is already active, so a 20-50 step sampling loop pays the UNet
    host->device transfer exactly once) trades a bounded, small number of
    PCIe transfers per generation for materially lower peak VRAM.

    Only engages when total device VRAM is below LARGE_VRAM_THRESHOLD_BYTES
    — on generously-sized cards there's no benefit to paying that transfer
    cost, so ``enabled`` is False and every method becomes a no-op,
    preserving today's "everything stays resident" behavior.
    """

    LARGE_VRAM_THRESHOLD_BYTES = 20 * 1024 ** 3  # 20GB

    def __init__(self, device):
        self.device = device
        self.enabled = self._detect_should_manage()
        self._components: Dict[str, dict] = {}
        self._active: str | None = None

    def _detect_should_manage(self) -> bool:
        try:
            from ldm_patched.modules import model_management
            total = model_management.get_total_memory(self.device)
        except Exception as e:
            log.warning(
                "[JAX Pipeline] Could not query total VRAM (%s); enabling "
                "phase-based component offload defensively.", e,
            )
            return True
        managed = total < self.LARGE_VRAM_THRESHOLD_BYTES
        log.info(
            "[JAX Pipeline] Total VRAM=%.2fGB -> phase-based component offload %s",
            total / 1e9, "enabled" if managed else "disabled (large-VRAM card)",
        )
        return managed

    def register(self, name: str, get_params, set_params) -> None:
        """Register a component.

        ``get_params()`` returns its current params pytree; ``set_params(p)``
        replaces it (params are new array objects after every device<->host
        move, since ``jax.device_put`` doesn't mutate in place).

        Newly registered components are immediately offloaded to pinned
        host memory (when enabled) rather than left on-device until the
        next ``activate()`` call — components are constructed with normal
        device-resident arrays (matching how each currently loads itself),
        and without this, installing all three components in sequence at
        model-load time would leave all three simultaneously resident on
        device for the entire loading window, before a single generation
        has even started — exactly the peak-VRAM scenario this exists to
        avoid. The first real ``activate()`` call brings whichever phase
        runs first back on device.
        """
        self._components[name] = {"get": get_params, "set": set_params, "on_device": True}
        if self.enabled:
            self._move(name, "pinned_host")

    def activate(self, name: str) -> None:
        """Ensure ``name`` is device-resident; offload every other
        registered, currently-on-device component to pinned host memory
        first. No-op if disabled, ``name`` isn't registered, or ``name`` is
        already the active phase.
        """
        if not self.enabled or name not in self._components:
            return
        if self._active == name:
            return
        for other, info in self._components.items():
            if other != name and info["on_device"]:
                self._move(other, "pinned_host")
        self._move(name, "device")
        self._active = name

    _ESCAPE_PHASE = "__pytorch_escape__"

    def escape_to_pytorch(self) -> None:
        """Call when handing control to a PyTorch code path that needs to
        see the VRAM JAX is holding, but that isn't itself a JAX phase
        (ControlNet fallback, CLIP textual-inversion/LoRA-rebuild-failure
        fallback, VAE fallback-to-torch) — as opposed to routine
        transitions between JAX's own unet/clip/vae phases, which
        ``activate()`` already handles.

        No-op if already escaped (mirrors ``activate()``'s
        no-op-if-same-phase behavior): a ControlNet-active generation
        calls this on every single sampling step, but only the first call
        actually forces a transfer.

        Unlike ``activate()``, NOT gated on ``self.enabled`` — event-
        driven cleanup (see module docstring) applies regardless of card
        size, and is cheap when there's nothing to move.
        """
        if self._active == self._ESCAPE_PHASE:
            return
        self.force_offload_all()
        self._active = self._ESCAPE_PHASE

    def force_offload_all(self) -> None:
        """Move every registered, currently-on-device component to pinned
        host memory, unconditionally (not gated on ``self.enabled`` — see
        module docstring: checkpoint switch / escape / OOM cleanup are
        event-driven, not part of the routine large-VRAM-card opt-out).

        With the platform allocator, this genuinely returns the freed
        device buffers to the CUDA driver rather than just relocating
        logical data within a cache XLA would otherwise never give back.
        """
        for name, info in self._components.items():
            if info["on_device"]:
                self._move(name, "pinned_host")
        self._active = None

    def force_release_all(self) -> None:
        """Explicitly delete every registered component's arrays (not just
        relocate to pinned host) and clear the registry.

        Use when the underlying weights are about to be discarded entirely
        — a checkpoint switch, where a fresh JAXSDXLPipeline/CLIP/VAE will
        be constructed from scratch for the new checkpoint and there's no
        reason to keep the old checkpoint's weights resident in ANY memory
        tier. Calls ``jax.Array.delete()`` explicitly rather than just
        dropping the Python reference and waiting for garbage collection —
        deterministic and immediate, rather than depending on GC timing.
        """
        import jax

        for name, info in self._components.items():
            try:
                params = info["get"]()
                jax.tree.map(lambda p: p.delete() if hasattr(p, "delete") else None, params)
            except Exception as e:
                log.warning("[JAX Pipeline] Error releasing '%s' params: %s", name, e)
        self._components.clear()
        self._active = None

    def _move(self, name: str, memory_kind: str) -> None:
        import jax

        info = self._components[name]
        params = info["get"]()
        sharding = jax.sharding.SingleDeviceSharding(self.device, memory_kind=memory_kind)
        new_params = jax.tree.map(lambda p: jax.device_put(p, sharding), params)
        info["set"](new_params)
        info["on_device"] = (memory_kind == "device")
        log.debug("[JAX Pipeline] Moved '%s' params to %s", name, memory_kind)
