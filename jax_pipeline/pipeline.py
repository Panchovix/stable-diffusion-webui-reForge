"""
jax_pipeline/pipeline.py — Torch <-> JAX bridge for SDXL UNet inference.

JAXSDXLPipeline wraps the JAX UNet and presents the same
``apply_model(x, t, ...)`` interface as ``DiffPipeline``/``MLXSDXLPipeline``
so the forge wrapper system and the rest of the webui are unaware of the
backend change.

Data-flow
---------
::

    apply_model(x, t, c_crossattn, y, ...)           <- forge sampler call
        |
        +-- LoRA / DoRA hot-reload check   (unet_patcher.patches_uuid)
        +-- sigma preconditioning          (model_sampling.calculate_input /
        |                                   model_sampling.timestep)
        +-- torch -> JAX conversion        (detach, cpu, numpy, jnp.asarray)
        |     +-- xc   [B,4,H,W]
        |     +-- ts   [B]
        |     +-- enc  [B,S,2048]
        |     +-- added_cond_kwargs
        |
        +-- unet_forward_jit() / offloaded forward   (jax.jit-compiled)
        |
        +-- JAX -> torch conversion        (np.asarray, torch.from_numpy)
        |
        +-- sigma postconditioning         (model_sampling.calculate_denoised)

Conditioning resolution
------------------------
Follows the same priority logic as DiffPipeline/MLXSDXLPipeline:
  * text_embeds  : adm_text_embeds kwarg  ->  y[:1280]  ->  zeros fallback
  * time_ids     : adm_time_ids kwarg     ->  derived from latent shape

ControlNet
----------
Unlike mlx_pipeline (which silently drops ControlNet conditioning),
``apply_model`` returns ``None`` when ``control`` is not None. The
``WrappersMP.APPLY_MODEL`` wrapper registered in ``jax_pipeline.__init__``
treats a ``None`` return as "fall through to the standard PyTorch UNet for
this call" so ControlNet conditioning is never silently ignored.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from modules_forge.unet_patcher import UnetPatcher

log = logging.getLogger(__name__)


# -- torch <-> JAX conversion helpers ----------------------------------------

def _torch_to_jax(t: torch.Tensor, dtype=None):
    """Convert a PyTorch tensor to a JAX array.

    Handles CUDA/ROCm/CPU tensors uniformly via numpy. ``dtype`` defaults to
    jnp.bfloat16 if not given.
    """
    import jax.numpy as jnp
    arr = t.detach().float().cpu().numpy()
    return jnp.asarray(arr, dtype=dtype if dtype is not None else jnp.bfloat16)


def _jax_to_torch(a, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor on *device* with *dtype*."""
    import jax.numpy as jnp
    arr = np.asarray(a.astype(jnp.float32))
    return torch.from_numpy(arr.copy()).to(device=device, dtype=dtype)


# -- main pipeline class ------------------------------------------------------

class JAXSDXLPipeline:
    """SDXL UNet forward-pass engine backed by JAX (jax.jit compiled).

    Constructed once at model load time; ``apply_model()`` is called every
    sampling step by the forge wrapper registered in ``jax_pipeline.__init__``.
    """

    def __init__(self, unet_patcher: "UnetPatcher", sd_model: Any, phase_manager=None):
        self.unet_patcher = unet_patcher
        self.sd_model = sd_model
        self._phase_manager = phase_manager

        # Sigma preconditioning from the loaded model's model_sampling object
        # (same object DiffPipeline and MLXSDXLPipeline use).
        self.model_sampling = unet_patcher.model.model_sampling

        log.info("[JAX Pipeline] Converting SDXL UNet weights to JAX...")
        from jax_pipeline import convert, host_offload

        self._device = host_offload.get_default_device()
        params = convert.load_weights_from_ldm(unet_patcher.model, report=True)

        if phase_manager is not None and phase_manager.enabled:
            # Block-level LRU streaming (jax_pipeline.block_cache), NOT
            # PhaseManager's coarse whole-component device<->host moves —
            # bringing the whole ~5-6GB bf16 UNet resident at once (even
            # just once per sampling phase, not per call) left too little
            # headroom for the forward pass's own activation memory on an
            # 8GB card; see the VRAM-profiling trail in this repo's commit
            # history (~2026-07-19). self._params is a BlockParamCache
            # here, not a flat dict — self._forward's signature is
            # unchanged either way (see make_streaming_forward), so
            # apply_model()/JAXCFGDenoiser's call sites don't need to
            # know which mode built it.
            #
            # NOT registered with phase_manager.register("unet", ...) —
            # the cache manages the UNet's own device residency entirely
            # independently, at a much finer (per-block) granularity than
            # PhaseManager's per-component moves. PhaseManager continues
            # to coarsely manage CLIP/VAE exactly as before; the two
            # mechanisms coexist without overlapping responsibility.
            self.host_offload = False
            from jax_pipeline.block_cache import BlockParamCache, partition_params_by_block
            from jax_pipeline.unet import build_block_ids

            self._block_cache = BlockParamCache(self._device, budget_bytes=None)
            self._block_cache.load(partition_params_by_block(params, build_block_ids()))
            self._params = self._block_cache
            self._forward = host_offload.make_streaming_forward()

            phase_manager.register_evict_callback(self._block_cache.evict_all_resident)
            phase_manager.register_release_callback(self._block_cache.clear)
        else:
            self.host_offload: bool = host_offload.should_offload(unet_patcher, params)
            self._params = host_offload.place_params(params, self._device, self.host_offload)
            self._forward = host_offload.make_forward(self._device, self.host_offload)

        # LoRA / DoRA tracking: the ldm_patched patches_uuid (a UUID4
        # regenerated on every LoRA add/remove/strength change) that was in
        # effect when the JAX params were last loaded. Initialized now so the
        # first generation doesn't trigger a spurious reload.
        self._lora_sig: str = str(unet_patcher.patches_uuid)

        log.info(
            "[JAX Pipeline] Ready — backend=%s host_offload=%s phase_managed=%s",
            self._device.platform, self.host_offload,
            phase_manager.enabled if phase_manager is not None else False,
        )

    # -- forward pass ---------------------------------------------------------

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[torch.Tensor] = None,
        c_crossattn: Optional[torch.Tensor] = None,
        control: Optional[Dict] = None,
        transformer_options: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Drop-in replacement for ``BaseModel.apply_model()``.

        Parameters match the ldm_patched calling convention; the return
        value is a *denoised* latent tensor in the same dtype/device as
        *x*, or ``None`` when this call's conditioning isn't supported yet
        (currently: an active ControlNet), signalling the caller to fall
        back to the standard PyTorch UNet for this call only.
        """
        if control is not None:
            if not getattr(self, "_warned_controlnet", False):
                log.warning(
                    "[JAX Pipeline] ControlNet is not yet supported on the JAX backend; "
                    "falling back to the standard PyTorch UNet while ControlNet is active."
                )
                self._warned_controlnet = True
            # PyTorch is about to need GPU memory JAX may be holding, and
            # PyTorch's allocator can't see JAX's — free it proactively.
            # No-op after the first call in a ControlNet-active generation
            # (every sampling step reaches here; only the first pays the
            # transfer cost).
            if self._phase_manager is not None:
                self._phase_manager.escape_to_pytorch()
            return None

        if transformer_options is None:
            transformer_options = {}

        # -- 0. LoRA / DoRA hot-reload -----------------------------------------
        # patch_weight_to_device() has already merged LoRA/DoRA deltas into
        # the PyTorch diffusion_model weights before this wrapper is reached;
        # patches_uuid is a cheap UUID4 comparison, not a weight hash.
        try:
            from jax_pipeline.lora import reload_jax_unet_weights, unet_lora_sig
            cur_uuid = unet_lora_sig(self.unet_patcher)
            if cur_uuid != self._lora_sig:
                reload_jax_unet_weights(self, self.unet_patcher)
                self._lora_sig = cur_uuid
        except Exception as lora_exc:
            log.warning("[JAX LoRA] LoRA reload skipped: %s", lora_exc)

        # -- 0.5. Phase activation ----------------------------------------------
        # When block-streaming is active (self._block_cache set — see
        # __init__), "unet" was never registered with phase_manager (the
        # cache manages UNet residency itself, per-block); the only thing
        # left for phase_manager to do here is make sure CLIP/VAE aren't
        # sitting on device competing with the UNet block cache's budget.
        # force_offload_all() is cheap/no-op when nothing's resident (e.g.
        # every step after the first, since nothing else re-activates
        # CLIP/VAE mid-sampling-loop). On large-VRAM cards (phase_manager
        # disabled) this whole block is a no-op either way.
        if self._phase_manager is not None:
            if self._phase_manager.enabled:
                self._phase_manager.force_offload_all()
            else:
                self._phase_manager.activate("unet")

        import jax.numpy as jnp

        sigma = t  # alias (t carries sigma values from k-diffusion)
        device = x.device
        out_dtype = x.dtype  # preserve caller's dtype on output

        # -- 1. Sigma preconditioning (matches ldm BaseModel / DiffPipeline) --
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        timestep = self.model_sampling.timestep(sigma).float()

        # -- 2. Resolve SDXL conditioning --------------------------------------
        adm_text_embeds = kwargs.get("adm_text_embeds", None)
        adm_time_ids = kwargs.get("adm_time_ids", None)
        y = kwargs.get("y", None)

        if adm_text_embeds is not None:
            text_embeds = adm_text_embeds
        elif y is not None:
            text_embeds = y[:, :1280]
        else:
            log.warning(
                "[JAX Pipeline] No pooled conditioning (adm_text_embeds/y absent) "
                "- using zero fallback. Quality will be degraded."
            )
            text_embeds = torch.zeros(x.shape[0], 1280, device=device, dtype=torch.float32)

        if adm_time_ids is not None:
            time_ids = adm_time_ids
        else:
            h_px, w_px = x.shape[2] * 8, x.shape[3] * 8
            time_ids = torch.tensor(
                [[h_px, w_px, 0, 0, h_px, w_px]], device=device, dtype=torch.float32,
            ).expand(text_embeds.shape[0], -1)

        # -- 3. Convert to JAX --------------------------------------------------
        jx_sample = _torch_to_jax(xc)  # [B,4,H,W]
        jx_timestep = _torch_to_jax(timestep, dtype=jnp.float32)  # [B]
        jx_enc_hs = (
            _torch_to_jax(c_crossattn) if c_crossattn is not None
            else jnp.zeros((x.shape[0], 1, 2048), dtype=jnp.bfloat16)
        )
        jx_text_emb = _torch_to_jax(text_embeds)  # [B, 1280]
        jx_time_ids = _torch_to_jax(time_ids, dtype=jnp.float32)  # [B, 6]

        added_cond_kwargs = {
            "text_embeds": jx_text_emb,
            "time_ids": jx_time_ids,
        }

        # -- 4. JAX forward pass -------------------------------------------------
        try:
            jx_output = self._forward(
                self._params, jx_sample, jx_timestep, jx_enc_hs, added_cond_kwargs,
            )  # [B, 4, H, W]
        except Exception as exc:
            from jax_pipeline.host_offload import _is_jax_oom_exception, handle_jax_oom
            if _is_jax_oom_exception(exc):
                handle_jax_oom(self._phase_manager, "UNet forward", exc)  # always raises
            raise

        # -- 5. Convert back to torch ---------------------------------------------
        model_output = _jax_to_torch(jx_output, device=device, dtype=torch.float32)

        # -- 6. Sigma postconditioning ----------------------------------------------
        denoised = self.model_sampling.calculate_denoised(sigma, model_output, x)

        return denoised.to(dtype=out_dtype)
