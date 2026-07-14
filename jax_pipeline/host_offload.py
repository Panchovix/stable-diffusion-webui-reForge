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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)


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
