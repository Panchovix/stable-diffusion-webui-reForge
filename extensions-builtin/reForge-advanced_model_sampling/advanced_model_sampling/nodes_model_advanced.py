import torch
from backend.modules import k_prediction

class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_dimensions = 2
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SDXL_Playground_2_5(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(1, 4, 1, 1)
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(1, 4, 1, 1)

        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3920,  0.4054,  0.4549],
                    [-0.2634, -0.0196,  0.0653],
                    [ 0.0568,  0.1687, -0.0755],
                    [-0.3112, -0.2359, -0.2076]
                ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

class LCM(k_prediction.Prediction):
    def __init__(self, *args, **kwargs):
        super().__init__(sigma_data=0.5, prediction_type='eps', *args, **kwargs)
        
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input

class X0(k_prediction.AbstractPrediction):
    def __init__(self, *args, **kwargs):
        super().__init__(sigma_data=1.0, prediction_type='eps', *args, **kwargs)
        
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output
    
class Lotus(X0):
    def calculate_input(self, sigma, noise):
        return noise

class ModelSamplingDiscreteDistilled(k_prediction.Prediction):
    original_timesteps = 50

    def __init__(self, model_config=None, zsnr=None, **kwargs):
        # Get sampling settings from model_config if available
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}
            
        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        timesteps = sampling_settings.get("timesteps", 1000)
        
        super().__init__(sigma_data=1.0, prediction_type='eps', beta_schedule=beta_schedule,
                         linear_start=linear_start, linear_end=linear_end, timesteps=timesteps)
        
        if zsnr:
            sigmas = k_prediction.rescale_zero_terminal_snr_sigmas(self.sigmas)
            self.set_sigmas(sigmas)

        self.skip_steps = timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[timesteps - 1 - x * self.skip_steps]

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


class ModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["eps", "v_prediction", "lcm", "x0", "img_to_img"],),
                              "zsnr": ("BOOLEAN", {"default": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, zsnr):
        m = model.clone()

        # Get model config for sampling settings
        model_config = model.model.config if hasattr(model.model, 'config') else None
        
        if sampling == "eps":
            # Get sampling settings from model config for proper initialization
            if model_config and hasattr(model_config, 'sampling_settings'):
                sampling_settings = model_config.sampling_settings
                predictor = k_prediction.Prediction(
                    sigma_data=1.0, 
                    prediction_type='eps',
                    beta_schedule=sampling_settings.get("beta_schedule", "linear"),
                    linear_start=sampling_settings.get("linear_start", 0.00085),
                    linear_end=sampling_settings.get("linear_end", 0.012),
                    timesteps=sampling_settings.get("timesteps", 1000)
                )
            else:
                predictor = k_prediction.Prediction(sigma_data=1.0, prediction_type='eps')
        elif sampling == "v_prediction":
            # Get sampling settings from model config for proper initialization  
            if model_config and hasattr(model_config, 'sampling_settings'):
                sampling_settings = model_config.sampling_settings
                predictor = k_prediction.Prediction(
                    sigma_data=1.0, 
                    prediction_type='v_prediction',
                    beta_schedule=sampling_settings.get("beta_schedule", "linear"),
                    linear_start=sampling_settings.get("linear_start", 0.00085),
                    linear_end=sampling_settings.get("linear_end", 0.012),
                    timesteps=sampling_settings.get("timesteps", 1000)
                )
            else:
                predictor = k_prediction.Prediction(sigma_data=1.0, prediction_type='v_prediction')
        elif sampling == "lcm":
            predictor = ModelSamplingDiscreteDistilled(model_config=model_config, zsnr=zsnr)
        elif sampling == "x0":
            predictor = X0()
        elif sampling == "img_to_img":
            predictor = X0()  # IMG_TO_IMG is essentially X0 with different input handling
        
        # Apply ZSNR if requested and not LCM (LCM handles it internally)
        if zsnr and sampling != "lcm":
            if hasattr(predictor, 'sigmas') and hasattr(predictor, 'set_sigmas'):
                sigmas = k_prediction.rescale_zero_terminal_snr_sigmas(predictor.sigmas)
                predictor.set_sigmas(sigmas)

        m.add_object_patch("predictor", predictor)
        return (m, )

class ModelSamplingStableCascade:
    def patch(self, model, shift):
        m = model.clone()

        # Use a custom predictor for StableCascade that mimics the old behavior
        # For now, we'll use a basic discrete predictor
        predictor = k_prediction.Prediction(sigma_data=1.0, prediction_type='eps')
        
        m.add_object_patch("predictor", predictor)
        return (m, )

class ModelSamplingSD3:
    def patch(self, model, shift, multiplier=1000):
        m = model.clone()

        predictor = k_prediction.PredictionFlow(sigma_data=1.0, prediction_type='const', 
                                                shift=shift, multiplier=multiplier)
        
        m.add_object_patch("predictor", predictor)
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

        # Calculate sequence length for Flux predictor
        seq_len = (width * height) // (8 * 8)  # Assuming 8x downscaling
        
        predictor = k_prediction.PredictionFlux(seq_len=seq_len, base_seq_len=256, max_seq_len=4096,
                                                base_shift=base_shift, max_shift=max_shift)
        
        m.add_object_patch("predictor", predictor)
        return (m, )

class ModelSamplingContinuousEDM:
    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        latent_format = None
        sigma_data = 1.0
        prediction_type = 'eps'
        
        if sampling == "eps":
            prediction_type = 'eps'
        elif sampling == "edm":
            prediction_type = 'edm'
            sigma_data = 0.5
        elif sampling == "v_prediction":
            prediction_type = 'v_prediction'
        elif sampling == "edm_playground_v2.5":
            prediction_type = 'edm'
            sigma_data = 0.5
            latent_format = SDXL_Playground_2_5()
        elif sampling == "cosmos_rflow":
            # For cosmos_rflow, we'll use a basic continuous EDM for now
            prediction_type = 'eps'

        predictor = k_prediction.PredictionContinuousEDM(sigma_data=sigma_data, 
                                                         prediction_type=prediction_type,
                                                         sigma_min=sigma_min, 
                                                         sigma_max=sigma_max)

        m.add_object_patch("predictor", predictor)
        if latent_format is not None:
            m.add_object_patch("latent_format", latent_format)
        return (m, )

class ModelSamplingContinuousV:
    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()
        sigma_data = 1.0
        prediction_type = 'v_prediction'
        
        if sampling == "v_prediction":
            prediction_type = 'v_prediction'

        predictor = k_prediction.PredictionContinuousV(sigma_data=sigma_data, 
                                                       prediction_type=prediction_type,
                                                       sigma_min=sigma_min, 
                                                       sigma_max=sigma_max)

        m.add_object_patch("predictor", predictor)
        return (m, )
