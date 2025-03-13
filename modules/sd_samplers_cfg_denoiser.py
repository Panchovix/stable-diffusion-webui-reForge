from modules.shared import opts
import torch
from modules import prompt_parser, sd_samplers_common
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback
from modules_forge import forge_sampler
from ldm_patched.modules.conds import CONDRegular
from ldm_patched.modules.samplers import sampling_function


def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]

    return {key: vec[a:b] for key, vec in cond.items()}


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    tensor['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return tensor

class CFGDenoiser(torch.nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        self.total_steps = None
        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.sampler = sampler
        self.model_wrap = None
        self.p = None
        self.mask_before_denoising = False
        self.classic_ddim_eps_estimation = False
        self.cond_scale_miltiplier = 1.0
        self.need_last_noise_uncond = False
        self.last_noise_uncond = None

    @property
    def inner_model(self):
        raise NotImplementedError()

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)
        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None

        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc

    def pad_cond_uncond(self, cond, uncond):
        empty = shared.sd_model.cond_stage_model_empty_prompt
        num_repeats = (cond.shape[1] - uncond.shape[1]) // empty.shape[1]

        if num_repeats < 0:
            cond = pad_cond(cond, -num_repeats, empty)
            self.padded_cond_uncond = True
        elif num_repeats > 0:
            uncond = pad_cond(uncond, num_repeats, empty)
            self.padded_cond_uncond = True

        return cond, uncond

    def pad_cond_uncond_v0(self, cond, uncond):
        is_dict_cond = isinstance(uncond, dict)
        uncond_vec = uncond['crossattn'] if is_dict_cond else uncond

        if uncond_vec.shape[1] < cond.shape[1]:
            last_vector = uncond_vec[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond_vec.shape[1], 1])
            uncond_vec = torch.hstack([uncond_vec, last_vector_repeated])
            self.padded_cond_uncond_v0 = True
        elif uncond_vec.shape[1] > cond.shape[1]:
            uncond_vec = uncond_vec[:, :cond.shape[1]]
            self.padded_cond_uncond_v0 = True

        if is_dict_cond:
            uncond['crossattn'] = uncond_vec
        else:
            uncond = uncond_vec

        return cond, uncond

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond, **kwargs):
        if shared.state.interrupted or shared.state.skipped:
            raise sd_samplers_common.InterruptedException

        original_x_device = x.device
        original_x_dtype = x.dtype

        if self.classic_ddim_eps_estimation:
            acd = self.inner_model.inner_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma ** 2.0 + real_sigma_data ** 2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma

        if sd_samplers_common.apply_refiner(self, x):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']

        cond_composition, cond = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0

        # If we use masks, blending between the denoised and original latent images occurs here.
        def apply_blend(current_latent, noisy_initial_latent=None):
            if noisy_initial_latent is None:
                noisy_initial_latent = self.init_latent
            blended_latent = current_latent * self.nmask + noisy_initial_latent * self.mask

            if self.p.scripts is not None:
                from modules import scripts
                mba = scripts.MaskBlendArgs(current_latent, self.nmask, self.init_latent, self.mask, blended_latent, denoiser=self, sigma=sigma)
                self.p.scripts.on_mask_blend(self.p, mba)
                blended_latent = mba.blended_latent

            return blended_latent

        if self.mask_before_denoising and self.mask is not None:
            noisy_initial_latent = self.init_latent + sigma[:, None, None, None] * torch.randn_like(self.init_latent).to(self.init_latent)
            x = apply_blend(x, noisy_initial_latent)

        denoiser_params = CFGDenoiserParams(x, image_cond, sigma, shared.state.sampling_step, shared.state.sampling_steps, cond, uncond, self)
        cfg_denoiser_callback(denoiser_params)

        # Initialize skip_uncond
        skip_uncond = False

        # NGMS logic

        if self.p.is_hr_pass == True:
            cond_scale = self.p.hr_cfg
        s_min_uncond = getattr(shared.opts, 's_min_uncond', 0.0)
        if s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            if self.step % 2 == 0 or shared.opts.s_min_uncond_all:
                skip_uncond = True
                self.p.extra_generation_params["NGMS"] = s_min_uncond
                if shared.opts.s_min_uncond_all:
                    self.p.extra_generation_params["NGMS all steps"] = shared.opts.s_min_uncond_all
                print(f"Applying NGMS at step {self.step}: s_min_uncond = {s_min_uncond}, sigma = {sigma[0]}")

        # Existing skip_early_cond logic
        if not skip_uncond:
            if self.step < shared.opts.skip_early_cond:
                skip_uncond = True
            elif shared.opts.skip_early_cond != 0. and self.step / self.total_steps <= shared.opts.skip_early_cond:
                skip_uncond = True
                self.p.extra_generation_params["Skip Early CFG"] = shared.opts.skip_early_cond

        # Implement padding logic
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        if shared.opts.pad_cond_uncond_v0 and cond.shape[1] != uncond.shape[1]:
            cond, uncond = self.pad_cond_uncond_v0(cond, uncond)
        elif shared.opts.pad_cond_uncond and cond.shape[1] != uncond.shape[1]:
            cond, uncond = self.pad_cond_uncond(cond, uncond)

        # Use forge_sample
        model = self.inner_model.inner_model.forge_objects.unet.model
        control = self.inner_model.inner_model.forge_objects.unet.controlnet_linked_list
        extra_concat_condition = self.inner_model.inner_model.forge_objects.unet.extra_concat_condition
        model_options = self.inner_model.inner_model.forge_objects.unet.model_options
        if kwargs.get('model_options', None):
            model_options = model_options.copy()
            model_options.update(kwargs['model_options'])
        seed = self.p.seeds[0]

        uncond_patched = forge_sampler.cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
        cond_patched = forge_sampler.cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)

        if extra_concat_condition is not None:
            image_cond_in = extra_concat_condition
        else:
            image_cond_in = denoiser_params.image_cond

        if isinstance(image_cond_in, torch.Tensor):
            if image_cond_in.shape[0] == x.shape[0] \
                    and image_cond_in.shape[2] == x.shape[2] \
                    and image_cond_in.shape[3] == x.shape[3]:
                for i in range(len(uncond_patched)):
                    uncond_patched[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)
                for i in range(len(cond_patched)):
                    cond_patched[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)

        if control is not None:
            for h in cond_patched + uncond_patched:
                h['control'] = control

        for modifier in model_options.get('conditioning_modifiers', []):
            model, x, sigma, uncond_patched, cond_patched, cond_scale, model_options, seed = modifier(model, x, sigma, uncond_patched, cond_patched, cond_scale, model_options, seed)

        if skip_uncond:
            # Only use the conditional input when skipping unconditional
            denoised = sampling_function(model, x, sigma, None, cond_patched, 1.0, model_options, seed)
        else:
            denoised = sampling_function(model, x, sigma, uncond_patched, cond_patched, cond_scale, model_options, seed)

        if self.need_last_noise_uncond:
            if skip_uncond:
                self.last_noise_uncond = torch.zeros_like(x)  # or another appropriate default
            else:
                self.last_noise_uncond = torch.clone(denoised[-uncond.shape[0]:])

        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(denoised, cond_scale * self.cond_scale_miltiplier)
        elif skip_uncond:
            # No need to combine, just use the conditional result
            pass
        else:
            denoised = self.combine_denoised(denoised, cond_composition, uncond, cond_scale * self.cond_scale_miltiplier)
        
        if not self.mask_before_denoising and self.mask is not None:
            denoised = apply_blend(denoised)
        preview = self.sampler.last_latent = denoised
        sd_samplers_common.store_latent(preview)
        after_cfg_callback_params = AfterCFGCallbackParams(denoised, shared.state.sampling_step, shared.state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x
        self.step += 1
        if self.classic_ddim_eps_estimation:
            eps = (x - denoised) / sigma[:, None, None, None]
            return eps
        return denoised.to(device=original_x_device, dtype=original_x_dtype)
    