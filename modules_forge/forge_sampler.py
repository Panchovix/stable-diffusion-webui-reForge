import logging
import torch
from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn
from ldm_patched.modules.samplers import sampling_function
from ldm_patched.modules import model_management
from ldm_patched.modules.ops import cleanup_cache
import ldm_patched.modules.patcher_extension as patcher_extension

_log = logging.getLogger(__name__)


def cond_from_a1111_to_patched_ldm(cond):
    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=CONDCrossAttn(cond),
            )
        )
        return [result, ]

    cross_attn = cond['crossattn']
    pooled_output = cond.get('vector', None)
    if pooled_output is None:
        pooled_output = torch.zeros(cross_attn.shape[0], 1280, device=cross_attn.device, dtype=cross_attn.dtype)

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=CONDCrossAttn(cross_attn),
            y=CONDRegular(pooled_output),
            # adm_text_embeds carries the raw pooled text embedding (1280-dim for SDXL
            # CLIP-G) so DiffPipeline.apply_model can use the primary branch instead of
            # the y-split fallback.  The ldm path ignores unknown model_conds keys.
            adm_text_embeds=CONDRegular(pooled_output),
        )
    )

    return [result, ]


def cond_from_a1111_to_patched_ldm_weighted(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = cond_from_a1111_to_patched_ldm(feed)
        h[0]['strength'] = current_weight
        results += h

    return results


def forge_sample(self, denoiser_params, cond_scale, cond_composition):
    model = self.inner_model.inner_model.forge_objects.unet.model
    control = self.inner_model.inner_model.forge_objects.unet.controlnet_linked_list
    extra_concat_condition = self.inner_model.inner_model.forge_objects.unet.extra_concat_condition
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
    cond = cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)
    model_options = self.inner_model.inner_model.forge_objects.unet.model_options
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            for i in range(len(uncond)):
                uncond[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)
            for i in range(len(cond)):
                cond[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)

    if control is not None:
        for h in cond + uncond:
            h['control'] = control

    # Handle skip_uncond
    skip_uncond = getattr(self, 'skip_uncond', False)
    if skip_uncond:
        uncond = None

    # Handle is_edit_model
    is_edit_model = getattr(self, 'is_edit_model', False)
    if is_edit_model:
        image_cfg_scale = getattr(self, 'image_cfg_scale', None)
        model_options['image_cfg_scale'] = image_cfg_scale

    # Handle mask and init_latent
    mask = getattr(self, 'mask', None)
    init_latent = getattr(self, 'init_latent', None)
    if mask is not None and init_latent is not None:
        model_options['mask'] = mask
        model_options['init_latent'] = init_latent

    for modifier in model_options.get('conditioning_modifiers', []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised = sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    # Handle mask_before_denoising
    if getattr(self, 'mask_before_denoising', False) and mask is not None:
        denoised = denoised * (1 - mask) + init_latent * mask

    return denoised



def _ensure_diffusers_wrapper(unet):
    """Register the DiffPipeline apply_model wrapper on *this* unet if missing.

    The sampler may hold a stale sd_model reference (inner_model cached before a
    model reload), so the wrapper registered at load time may live on a different
    UnetPatcher instance.  Re-registering here is idempotent — add_wrapper_with_key
    appends, so we guard with an explicit presence check.
    """
    key = patcher_extension.WrappersMP.APPLY_MODEL
    if "forge_diffusers" in unet.wrappers.get(key, {}):
        return  # already present on this patcher instance

    # Lazy import to avoid circular dependency (modules → modules_forge → modules)
    from modules import sd_models
    sd_model = sd_models.model_data.get_sd_model()
    if sd_model is None:
        return
    _dp = getattr(sd_model, "diff_pipeline", None)
    if _dp is None:
        return
    # Guard: only register the wrapper for a real DiffPipeline instance.
    # DiffusersModelAdapter sets diff_pipeline to the raw diffusers pipe when
    # DiffPipeline construction fails; calling .apply_model() on it would crash.
    from diff_pipeline.pipeline import DiffPipeline
    if not isinstance(_dp, DiffPipeline):
        return

    def _diff_apply_model_wrapper(executor, x, t, c_concat=None, c_crossattn=None,
                                   control=None, transformer_options={}, **kwargs):
        return _dp.apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                               control=control, transformer_options=transformer_options,
                               **kwargs)

    unet.add_wrapper_with_key(key, "forge_diffusers", _diff_apply_model_wrapper)
    _log.info("[forge_sampler] forge_diffusers wrapper registered on unet id=%d", id(unet))


def sampling_prepare(unet, x):
    B, C, H, W = x.shape

    memory_estimation_function = unet.model_options.get('memory_peak_estimation_modifier', unet.memory_required)

    unet_inference_memory = memory_estimation_function([B * 2, C, H, W])
    additional_inference_memory = unet.extra_preserved_memory_during_sampling
    additional_model_patchers = unet.extra_model_patchers_during_sampling

    if unet.controlnet_linked_list is not None:
        additional_inference_memory += unet.controlnet_linked_list.inference_memory_requirements(unet.model_dtype())
        additional_model_patchers += unet.controlnet_linked_list.get_models()

    model_management.load_models_gpu(
        models=[unet] + additional_model_patchers,
        memory_required=unet_inference_memory + additional_inference_memory)

    # Lazy-register the DiffPipeline apply_model wrapper on whichever unet object
    # the sampler actually uses. The sampler can hold a stale sd_model reference
    # (inner_model cached before model reload), so the wrapper added at load time
    # may be on a different UnetPatcher instance. Re-register here if missing.
    _ensure_diffusers_wrapper(unet)

    # Propagate wrappers from the UnetPatcher into transformer_options so that
    # wrappers registered via add_wrapper_with_key() (e.g. forge_diffusers) are
    # visible to model_base.apply_model() during calc_cond_batch().
    unet.model_options.setdefault("transformer_options", {})["wrappers"] = \
        patcher_extension.copy_nested_dicts(unet.wrappers)
    _log.debug("[sampling_prepare] unet id=%d apply_model wrappers=%s",
               id(unet), list(unet.wrappers.get(patcher_extension.WrappersMP.APPLY_MODEL, {}).keys()))

    real_model = unet.model

    percent_to_timestep_function = lambda p: real_model.model_sampling.percent_to_sigma(p)

    for cnet in unet.list_controlnets():
        cnet.pre_run(real_model, percent_to_timestep_function)

    return


def sampling_cleanup(unet):
    for cnet in unet.list_controlnets():
        cnet.cleanup()
    # Clear wrappers injected by sampling_prepare so stale state
    # doesn't persist if a different model is loaded next call.
    unet.model_options.get("transformer_options", {}).pop("wrappers", None)
    cleanup_cache()
    return
