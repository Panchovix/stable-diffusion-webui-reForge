import inspect
from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from modules import devices, images, sd_vae_approx, sd_samplers, sd_vae_taesd, shared
from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup
from modules import extra_networks
from modules.sd_sampling_backend import get_sampling

SamplerDataTuple = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])


class SamplerData(SamplerDataTuple):
    def total_steps(self, steps):
        if self.options.get("second_order", False):
            steps = steps * 2

        return steps


def setup_img2img_steps(p, steps=None):
    if shared.opts.img2img_fix_steps or steps is not None:
        requested_steps = (steps or p.steps)
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3, "Approx PCA": 4}


_pca_cache = {}

def _pca_project_latent(sample):
    """Cheap projection of arbitrary-channel latents to RGB using a cached random orthonormal basis."""
    # sample: B,C,H,W
    key = (sample.shape[1], sample.device.type)
    proj = _pca_cache.get(key, None)
    if proj is None:
        with torch.no_grad():
            rand = torch.randn(sample.shape[1], 3, device=sample.device, dtype=torch.float32)
            # Orthonormalize for stable scaling.
            q, _ = torch.linalg.qr(rand, mode="reduced")
            proj = q.to(sample.dtype)
        _pca_cache[key] = proj

    # Project channels -> 3
    x = torch.einsum("bchw,cp->bhwp", sample, proj)
    # Normalize to [-1,1] to mimic decoded output range.
    max_abs = torch.amax(torch.abs(x), dim=(1,2,3), keepdim=True).clamp_min(1e-6)
    x = (x / max_abs).clamp_(-1, 1)
    x = x.permute(0, 3, 1, 2)  # B,3,H,W
    return x

def samples_to_images_tensor(sample, approximation=None, model=None):
    """Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1]."""

    if approximation is None or (shared.state.interrupted and shared.opts.live_preview_fast_interrupt):
        approximation = approximation_indexes.get(shared.opts.show_progress_type, 0)
        if approximation == 0:
            approximation = 1

    # For non-4-channel latents (e.g., SD3/Flux2), use PCA projection as a cheap preview.
    if sample.shape[1] != 4 and approximation in (1, 2, 3):
        approximation = 4

    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        x_sample = sd_vae_taesd.decoder_model()(sample.to(devices.device, devices.dtype)).detach()
        x_sample = x_sample * 2 - 1
    elif approximation == 4:
        x_sample = _pca_project_latent(sample)
    else:
        if model is None:
            model = shared.sd_model
        x_sample = model.decode_first_stage(sample)

    return x_sample


def single_sample_to_image(sample, approximation=None):
    x_sample = samples_to_images_tensor(sample.unsqueeze(0), approximation)[0] * 0.5 + 0.5

    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)

    return Image.fromarray(x_sample)


def decode_first_stage(model, x):
    approx_index = approximation_indexes.get(shared.opts.sd_vae_decode_method, 0)
    return samples_to_images_tensor(x, approx_index, model)


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def images_tensor_to_samples(image, approximation=None, model=None):
    '''image[0, 1] -> latent'''
    if approximation is None:
        approximation = approximation_indexes.get(shared.opts.sd_vae_encode_method, 0)

    if approximation == 3:
        image = image.to(devices.device, devices.dtype)
        x_latent = sd_vae_taesd.encoder_model()(image)
    else:
        if model is None:
            model = shared.sd_model

        image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image * 2 - 1
        if len(image) > 1:
            x_latent = torch.stack([
                model.get_first_stage_encoding(
                    model.encode_first_stage(torch.unsqueeze(img, 0))
                )[0]
                for img in image
            ])
        else:
            x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))

    return x_latent


def store_latent(decoded):
    shared.state.current_latent = decoded

    if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % shared.opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.assign_current_image(sample_to_image(decoded))


def is_sampler_using_eta_noise_seed_delta(p):
    """returns whether sampler from config will use eta noise seed delta for image creation"""

    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)

    eta = p.eta

    if eta is None and p.sampler is not None:
        eta = p.sampler.eta

    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0

    if eta == 0:
        return False

    return sampler_config.options.get("uses_ensd", False)


class InterruptedException(BaseException):
    pass


def replace_torchsde_browinan():
    import torchsde._brownian.brownian_interval

    def torchsde_randn(size, dtype, device, seed):
        return devices.randn_local(seed, size).to(device=device, dtype=dtype)

    torchsde._brownian.brownian_interval._randn = torchsde_randn


replace_torchsde_browinan()


def apply_refiner(cfg_denoiser, x, sigma=None):
    from modules import sd_models  # local import to avoid circular dependency
    if shared.opts.refiner_switch_by_sample_steps or sigma is None:
        completed_ratio = cfg_denoiser.step / cfg_denoiser.total_steps
    else:
        # Ensure sigma is on the same device as cfg_denoiser.inner_model.sigmas
        device = cfg_denoiser.inner_model.sigmas.device
        sigma = sigma.to(device)

        try:
            timestep = torch.argmin(torch.abs(cfg_denoiser.inner_model.sigmas - torch.max(sigma)))
        except AttributeError:  # for samplers that don't use sigmas (DDIM) sigma is actually the timestep
            timestep = torch.max(sigma).to(dtype=int)
        completed_ratio = (999 - timestep) / 1000
    refiner_switch_at = cfg_denoiser.p.refiner_switch_at
    refiner_checkpoint_info = cfg_denoiser.p.refiner_checkpoint_info

    if refiner_switch_at is not None and completed_ratio < refiner_switch_at:
        return False

    if refiner_checkpoint_info is None or shared.sd_model.sd_checkpoint_info == refiner_checkpoint_info:
        return False

    if getattr(cfg_denoiser.p, "enable_hr", False):
        is_second_pass = cfg_denoiser.p.is_hr_pass

        if shared.opts.hires_fix_refiner_pass == "first pass" and is_second_pass:
            return False

        if shared.opts.hires_fix_refiner_pass == "second pass" and not is_second_pass:
            return False

        if shared.opts.hires_fix_refiner_pass != "second pass":
            cfg_denoiser.p.extra_generation_params['Hires refiner'] = shared.opts.hires_fix_refiner_pass

    cfg_denoiser.p.extra_generation_params['Refiner'] = refiner_checkpoint_info.short_title
    cfg_denoiser.p.extra_generation_params['Refiner switch at'] = refiner_switch_at

    sampling_cleanup(sd_models.model_data.get_sd_model().forge_objects.unet)

    with sd_models.SkipWritingToConfig():
        sd_models.reload_model_weights(info=refiner_checkpoint_info)

    if not cfg_denoiser.p.disable_extra_networks:
        extra_networks.activate(cfg_denoiser.p, cfg_denoiser.p.extra_network_data)

    cfg_denoiser.p.setup_conds()
    cfg_denoiser.update_inner_model()

    sampling_prepare(sd_models.model_data.get_sd_model().forge_objects.unet, x=x)
    return True


class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, p):
        self.rng = p.rng

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return self.rng.next()


class Sampler:
    def __init__(self, funcname):
        self.funcname = funcname
        self.func = funcname
        self.extra_params = []
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config: SamplerData = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None
        # Default values for sampler parameters
        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')
        self.s_noise = 1.0
        self.dpmpp_sde_r = 0.5
        self.dpmpp_2m_sde_solver = 'midpoint'

        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0

        self.conditioning_key = getattr(shared.sd_model.model, 'conditioning_key', 'crossattn')

        self.p = None
        self.model_wrap_cfg = None
        self.sampler_extra_args = None
        self.options = {}

    def callback_state(self, d):
        if self.p is not None and self.p.scripts is not None:
            self.p.scripts.process_before_every_step(p=self.p, d=d)

        step = d['i']

        if self.stop_at is not None and step > self.stop_at:
            raise InterruptedException

        shared.state.sampling_step = step
        shared.total_tqdm.update()

    def launch_sampling(self, steps, func):
        self.model_wrap_cfg.steps = steps
        self.model_wrap_cfg.total_steps = self.config.total_steps(steps)
        shared.state.sampling_steps = steps
        shared.state.sampling_step = 0

        try:
            return func()
        except RecursionError:
            print(
                'Encountered RecursionError during sampling, returning last latent. '
                'rho >5 with a polyexponential scheduler may cause this error. '
                'You should try to use a smaller rho value instead.'
            )
            return self.last_latent
        except InterruptedException:
            return self.last_latent

    def number_of_needed_noises(self, p):
        return p.steps

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p = p
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.eta = p.eta if p.eta is not None else getattr(shared.opts, self.eta_option_field, 0.0)
        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)

        sampling = get_sampling()
        sampling.torch = TorchHijack(p)

        # AlterSamplers (modules_forge/forge_alter_samplers.py) unconditionally
        # bind to functions from ldm_patched.k_diffusion.sampling regardless of
        # opts.sd_sampling, so when the A1111 backend is selected above the
        # hijack lands on k_diff.k_diffusion.sampling and AlterSamplers miss
        # it. Also install the hijack on the ldm_patched module when it is a
        # distinct module object, so the randn_like intercept applies to the
        # sampler functions AlterSamplers actually call. When the Comfy
        # backend is selected `sampling` already IS that module and this is a
        # no-op.
        try:
            from ldm_patched.k_diffusion import sampling as _ldm_patched_sampling
        except ImportError:
            _ldm_patched_sampling = None
        if _ldm_patched_sampling is not None and _ldm_patched_sampling is not sampling:
            _ldm_patched_sampling.torch = TorchHijack(p)

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        # Handle eta parameter
        if 'eta' in inspect.signature(self.func).parameters:
            if self.eta != self.eta_default:
                p.extra_generation_params[self.eta_infotext_field] = self.eta
            extra_params_kwargs['eta'] = self.eta

        # Handle special parameters for DPM++ samplers
        if self.funcname == 'sample_dpmpp_sde':
            r = getattr(shared.opts,'dpmpp_sde_r', self.dpmpp_sde_r)
            if r != self.dpmpp_sde_r:
                extra_params_kwargs['r'] = r
                p.extra_generation_params['DPM++ SDE r'] = r

        if self.funcname == 'sample_dpmpp_2m_sde':
            solver_type = getattr(shared.opts,'dpmpp_2m_sde_solver', self.dpmpp_2m_sde_solver)
            if solver_type != self.dpmpp_2m_sde_solver:
                extra_params_kwargs['solver_type'] = solver_type
                p.extra_generation_params['DPM++ 2M solver'] = solver_type

        _SURE_SAMPLERS = {
            'sample_sure', 'sample_sure_wavelet', 'sample_sure_wavelet_auto',
            'sample_sure_wavelet_converge', 'sample_sure_wavelet_auto_converge',
            'sample_sure_adaptive',
            'sample_dpmpp_2m_sure', 'sample_dpmpp_2m_sde_sure',
            'sample_dpmpp_3m_sde_sure', 'sample_dpmpp_2m_sde_sure_adaptive',
            'sample_dpmpp_2s_a_sure', 'sample_dpmpp_2s_a_sure_adaptive',
        }
        if self.funcname in _SURE_SAMPLERS:
            import inspect as _inspect
            _sure_sig = _inspect.signature(self.func).parameters
            sure_alpha = getattr(shared.opts, 'sure_alpha', 0.05)
            sure_n_mc  = getattr(shared.opts, 'sure_n_mc', 1)
            sure_eps   = getattr(shared.opts, 'sure_eps', 1e-3)
            sure_jac_interval   = getattr(shared.opts, 'sure_jac_interval', 2)
            sure_preheat_steps  = getattr(shared.opts, 'sure_preheat_steps', -1)
            sure_grad_mode  = getattr(shared.opts, 'sure_grad_mode', 'vjp')
            sure_adam_mode  = getattr(shared.opts, 'sure_adam_mode', 'none')
            sure_adam_beta1 = getattr(shared.opts, 'sure_adam_beta1', 0.9)
            sure_adam_beta2 = getattr(shared.opts, 'sure_adam_beta2', 0.999)
            sure_adam_wd    = getattr(shared.opts, 'sure_adam_wd', 0.01)
            if 'sure_alpha'         in _sure_sig: extra_params_kwargs['sure_alpha']         = sure_alpha
            if 'sure_n_mc'          in _sure_sig: extra_params_kwargs['sure_n_mc']          = sure_n_mc
            if 'sure_eps'           in _sure_sig: extra_params_kwargs['sure_eps']           = sure_eps
            if 'sure_jac_interval'  in _sure_sig: extra_params_kwargs['sure_jac_interval']  = sure_jac_interval
            if 'sure_preheat_steps' in _sure_sig: extra_params_kwargs['sure_preheat_steps'] = sure_preheat_steps
            if 'sure_preheat_frac'  in _sure_sig:
                # adaptive variants use a fraction instead of a fixed count
                frac = max(0.0, min(1.0, sure_preheat_steps / 20.0)) if sure_preheat_steps >= 0 else 0.3
                extra_params_kwargs['sure_preheat_frac'] = frac
            if 'sure_grad_mode'  in _sure_sig: extra_params_kwargs['sure_grad_mode']  = sure_grad_mode
            if 'sure_adam_mode'  in _sure_sig: extra_params_kwargs['sure_adam_mode']  = sure_adam_mode
            if 'sure_adam_beta1' in _sure_sig: extra_params_kwargs['sure_adam_beta1'] = sure_adam_beta1
            if 'sure_adam_beta2' in _sure_sig: extra_params_kwargs['sure_adam_beta2'] = sure_adam_beta2
            if 'sure_adam_wd'    in _sure_sig: extra_params_kwargs['sure_adam_wd']    = sure_adam_wd
            sure_approx_coeff = getattr(shared.opts, 'sure_approx_coeff', 2.0)
            sure_csv_path     = getattr(shared.opts, 'sure_csv_path', '').strip() or None
            sure_sigma_ema    = getattr(shared.opts, 'sure_sigma_ema', 0.0)
            if 'sure_approx_coeff' in _sure_sig: extra_params_kwargs['sure_approx_coeff'] = sure_approx_coeff
            if 'sure_csv_path'     in _sure_sig: extra_params_kwargs['sure_csv_path']     = sure_csv_path
            if 'sure_sigma_ema'    in _sure_sig: extra_params_kwargs['sure_sigma_ema']    = sure_sigma_ema
            # SURE-Wavelet specific params — only forwarded when the sampler accepts them
            if 'sure_wavelet'       in _sure_sig:
                sure_wavelet       = getattr(shared.opts, 'sure_wavelet',       'db4')
                extra_params_kwargs['sure_wavelet'] = sure_wavelet
            if 'sure_wavelet_level' in _sure_sig:
                sure_wavelet_level = getattr(shared.opts, 'sure_wavelet_level', 3)
                extra_params_kwargs['sure_wavelet_level'] = sure_wavelet_level
            if 'sure_wavelet_warmup_steps' in _sure_sig:
                sure_wavelet_warmup_steps = getattr(shared.opts, 'sure_wavelet_warmup_steps', 0)
                extra_params_kwargs['sure_wavelet_warmup_steps'] = sure_wavelet_warmup_steps
            if 'sure_wavelet_lp_frac' in _sure_sig:
                sure_wavelet_lp_frac = getattr(shared.opts, 'sure_wavelet_lp_frac', 1.0)
                extra_params_kwargs['sure_wavelet_lp_frac'] = sure_wavelet_lp_frac
            # Calibration params for SURE Wavelet Auto
            if 'sure_wavelet_cal_steps' in _sure_sig:
                extra_params_kwargs['sure_wavelet_cal_steps'] = int(getattr(shared.opts, 'sure_wavelet_cal_steps', 5))
            if 'sure_wavelet_cal_wavelets' in _sure_sig:
                _raw = getattr(shared.opts, 'sure_wavelet_cal_wavelets', 'haar,db2,db4,db6,sym4')
                extra_params_kwargs['sure_wavelet_cal_wavelets'] = tuple(s.strip() for s in _raw.split(',') if s.strip())
            if 'sure_wavelet_cal_levels' in _sure_sig:
                # Accepts "min,max" or a single value; passes as int tuple to the sampler
                _raw = getattr(shared.opts, 'sure_wavelet_cal_levels', '1,6')
                extra_params_kwargs['sure_wavelet_cal_levels'] = tuple(int(s.strip()) for s in _raw.split(',') if s.strip())
            if 'sure_wavelet_bo_trials' in _sure_sig:
                extra_params_kwargs['sure_wavelet_bo_trials'] = int(getattr(shared.opts, 'sure_wavelet_bo_trials', 40))
            if 'sure_wavelet_bo_patience' in _sure_sig:
                extra_params_kwargs['sure_wavelet_bo_patience'] = int(getattr(shared.opts, 'sure_wavelet_bo_patience', 8))
            if 'sure_wavelet_bo_cv_warn' in _sure_sig:
                extra_params_kwargs['sure_wavelet_bo_cv_warn'] = float(getattr(shared.opts, 'sure_wavelet_bo_cv_warn', 0.4))
            if 'sure_inner_steps' in _sure_sig:
                extra_params_kwargs['sure_inner_steps'] = int(getattr(shared.opts, 'sure_inner_steps', 4))
            if 'sure_inner_tol' in _sure_sig:
                extra_params_kwargs['sure_inner_tol'] = float(getattr(shared.opts, 'sure_inner_tol', 1e-4))
            if 'sure_alpha_bo_trials' in _sure_sig:
                extra_params_kwargs['sure_alpha_bo_trials'] = int(getattr(shared.opts, 'sure_alpha_bo_trials', 0))
            if 'sure_alpha_bo_patience' in _sure_sig:
                extra_params_kwargs['sure_alpha_bo_patience'] = int(getattr(shared.opts, 'sure_alpha_bo_patience', 4))
            p.extra_generation_params['SURE alpha']         = sure_alpha
            p.extra_generation_params['SURE n_mc']          = sure_n_mc
            p.extra_generation_params['SURE eps']           = sure_eps
            p.extra_generation_params['SURE jac_interval']  = sure_jac_interval
            p.extra_generation_params['SURE preheat_steps'] = sure_preheat_steps
            if sure_grad_mode != 'vjp':
                p.extra_generation_params['SURE grad_mode'] = sure_grad_mode
            if sure_adam_mode != 'none':
                p.extra_generation_params['SURE adam_mode']  = sure_adam_mode
                p.extra_generation_params['SURE adam_beta1'] = sure_adam_beta1
                p.extra_generation_params['SURE adam_beta2'] = sure_adam_beta2
                if sure_adam_mode == 'adamw':
                    p.extra_generation_params['SURE adam_wd'] = sure_adam_wd
            _alpha_bo_trials_val = int(getattr(shared.opts, 'sure_alpha_bo_trials', 0))
            if _alpha_bo_trials_val > 0:
                p.extra_generation_params['SURE alpha_bo_trials']   = _alpha_bo_trials_val
                p.extra_generation_params['SURE alpha_bo_patience'] = int(getattr(shared.opts, 'sure_alpha_bo_patience', 4))

        if self.funcname == 'sample_dc_solver':
            import inspect as _inspect
            _dc_sig = _inspect.signature(self.func).parameters
            dc_order = getattr(shared.opts, 'dc_solver_order', 2)
            dc_ratio  = getattr(shared.opts, 'dc_solver_ratio', 0.5)
            if 'order' in _dc_sig:
                extra_params_kwargs['order'] = dc_order
            if 'dc_ratios' in _dc_sig:
                extra_params_kwargs['dc_ratios'] = None  # will be filled per-step by the sampler
                # we pass ratio via a list built here so every step uses the configured value;
                # length is unknown at this point so we use None and let the sampler default to 1.0,
                # then override with a uniform list once sigmas are known
                # — instead, store on p so KDiffusionSampler.sample() can inject it
                p._dc_solver_ratio = dc_ratio
            p.extra_generation_params['DC-Solver order'] = dc_order
            p.extra_generation_params['DC-Solver ratio'] = dc_ratio

        # Handle standard sigma parameters
        if len(self.extra_params) > 0:
            s_churn = getattr(shared.opts,'s_churn', p.s_churn)
            s_tmin = getattr(shared.opts,'s_tmin', p.s_tmin)
            s_tmax = getattr(shared.opts,'s_tmax', p.s_tmax) or self.s_tmax  # 0 = inf
            s_noise = getattr(shared.opts,'s_noise', p.s_noise)

            if 's_churn' in extra_params_kwargs and s_churn != self.s_churn:
                extra_params_kwargs['s_churn'] = s_churn
                p.s_churn = s_churn
                p.extra_generation_params['Sigma churn'] = s_churn
            if 's_tmin' in extra_params_kwargs and s_tmin != self.s_tmin:
                extra_params_kwargs['s_tmin'] = s_tmin
                p.s_tmin = s_tmin
                p.extra_generation_params['Sigma tmin'] = s_tmin
            if 's_tmax' in extra_params_kwargs and s_tmax != self.s_tmax:
                extra_params_kwargs['s_tmax'] = s_tmax
                p.s_tmax = s_tmax
                p.extra_generation_params['Sigma tmax'] = s_tmax
            if 's_noise' in extra_params_kwargs and s_noise != self.s_noise:
                extra_params_kwargs['s_noise'] = s_noise
                p.s_noise = s_noise
                p.extra_generation_params['Sigma noise'] = s_noise

        return extra_params_kwargs

    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None

        BrownianTreeNoiseSampler = get_sampling().BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def add_infotext(self, p):
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        if self.model_wrap_cfg.padded_cond_uncond_v0:
            p.extra_generation_params["Pad conds v0"] = True
