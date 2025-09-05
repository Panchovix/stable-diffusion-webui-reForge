import torch
from ldm_patched.modules import model_sampling, latent_formats

class LCM(model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input

class X0(model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output
    
class Lotus(X0):
    def calculate_input(self, sigma, noise):
        return noise

class ModelSamplingDiscreteDistilled(model_sampling.ModelSamplingDiscrete):
    original_timesteps = 50

    def __init__(self, model_config=None):
        super().__init__(model_config)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

class ModelSamplingDiscrete:
    def patch(self, model, sampling, zsnr):
        m = model.clone()

        sampling_base = model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = model_sampling.V_PREDICTION
        elif sampling == "lcm":
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled
        elif sampling == "x0":
            sampling_type = X0
        elif sampling == "lotus":
            sampling_type = Lotus

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        if zsnr:
            model_sampling_obj.set_sigmas(rescale_zero_terminal_snr_sigmas(model_sampling_obj.sigmas))

        m.add_object_patch("model_sampling", model_sampling_obj)
        return (m, )

class ModelSamplingStableCascade:
    def patch(self, model, shift):
        m = model.clone()

        sampling_base = model_sampling.StableCascadeSampling
        sampling_type = model_sampling.EPS

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        model_sampling_obj.set_parameters(shift)
        m.add_object_patch("model_sampling", model_sampling_obj)
        return (m, )

class ModelSamplingSD3:
    def patch(self, model, shift, multiplier=1000):
        import logging
        
        logging.info(f"[ModelSamplingSD3 Debug] Starting patch with shift={shift}, multiplier={multiplier}")
        
        m = model.clone()

        # Debug model info
        if hasattr(model.model, 'model_config'):
            config = model.model.model_config
            logging.info(f"[ModelSamplingSD3 Debug] Model config type: {type(config).__name__}")
            if hasattr(config, 'sampling_settings'):
                logging.info(f"[ModelSamplingSD3 Debug] Original sampling_settings: {config.sampling_settings}")
            if hasattr(config, 'unet_config'):
                unet_keys = list(config.unet_config.keys()) if config.unet_config else []
                logging.info(f"[ModelSamplingSD3 Debug] UNet config keys: {unet_keys}")
        else:
            logging.warning("[ModelSamplingSD3 Debug] No model_config found!")

        sampling_base = model_sampling.ModelSamplingDiscreteFlow
        sampling_type = model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            def calculate_input(self, sigma, noise):
                logging.debug(f"[CONST Debug] calculate_input called with sigma shape: {sigma.shape}, noise shape: {noise.shape}")
                result = super().calculate_input(sigma, noise)
                logging.debug(f"[CONST Debug] calculate_input result shape: {result.shape}")
                return result
            
            def calculate_denoised(self, sigma, model_output, model_input):
                logging.debug(f"[CONST Debug] calculate_denoised called with sigma: {sigma.mean().item():.4f}")
                result = super().calculate_denoised(sigma, model_output, model_input)
                return result
            
            def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                logging.debug(f"[CONST Debug] noise_scaling called with sigma: {sigma.mean().item():.4f}, max_denoise: {max_denoise}")
                result = super().noise_scaling(sigma, noise, latent_image, max_denoise)
                return result

        logging.info(f"[ModelSamplingSD3 Debug] Creating ModelSamplingAdvanced with base={sampling_base.__name__}, type={sampling_type.__name__}")
        
        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        logging.info(f"[ModelSamplingSD3 Debug] Created model_sampling_obj: {type(model_sampling_obj).__name__}")
        logging.info(f"[ModelSamplingSD3 Debug] Before set_parameters - shift: {getattr(model_sampling_obj, 'shift', 'N/A')}, multiplier: {getattr(model_sampling_obj, 'multiplier', 'N/A')}")
        
        model_sampling_obj.set_parameters(shift=shift, multiplier=multiplier)
        logging.info(f"[ModelSamplingSD3 Debug] After set_parameters - shift: {getattr(model_sampling_obj, 'shift', 'N/A')}, multiplier: {getattr(model_sampling_obj, 'multiplier', 'N/A')}")
        logging.info(f"[ModelSamplingSD3 Debug] Sigma range: {model_sampling_obj.sigma_min} - {model_sampling_obj.sigma_max}")
        
        m.add_object_patch("model_sampling", model_sampling_obj)
        logging.info("[ModelSamplingSD3 Debug] Added object_patch successfully")
        
        return (m, )

class ModelSamplingAuraFlow(ModelSamplingSD3):
    def patch_aura(self, model, shift):
        return self.patch(model, shift, multiplier=1.0)

class ModelSamplingFlux:
    def patch(self, model, max_shift, base_shift, width, height):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = model_sampling.ModelSamplingFlux
        sampling_type = model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        model_sampling_obj.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling_obj)
        return (m, )

class ModelSamplingContinuousEDM:
    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        latent_format = None
        sigma_data = 1.0
        if sampling == "eps":
            sampling_type = model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = model_sampling.V_PREDICTION
        elif sampling == "edm_playground_v2.5":
            sampling_type = model_sampling.EDM
            sigma_data = 0.5
            latent_format = latent_formats.SDXL_Playground_2_5()

        class ModelSamplingAdvanced(model_sampling.ModelSamplingContinuousEDM, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        model_sampling_obj.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling_obj)
        if latent_format is not None:
            m.add_object_patch("latent_format", latent_format)
        return (m, )

class ModelSamplingContinuousV:
    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()
        sigma_data = 1.0
        if sampling == "v_prediction":
            sampling_type = model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(model_sampling.ModelSamplingContinuousV, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(model.model.model_config)
        model_sampling_obj.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling_obj)
        return (m, )
