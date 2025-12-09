import logging
import gradio as gr
import torch
from modules import scripts
from modules.shared import opts
from ldm_patched.modules import model_sampling
from advanced_model_sampling.nodes_model_advanced import (
    ModelSamplingDiscrete, ModelSamplingContinuousEDM, ModelSamplingContinuousV,
    ModelSamplingStableCascade, ModelSamplingSD3, ModelSamplingAuraFlow, ModelSamplingFlux
)
#checkpoint 1

class FlowMatchingDenoiser(torch.nn.Module):
    """Custom denoiser for flow matching models that uses CONST prediction and patched model_sampling sigmas"""

    def __init__(self, model, unet_patcher):
        super().__init__()
        self.inner_model = model
        self.unet = unet_patcher
        self.sigma_data = 1.0
        self._cached_device = None

        # CRITICAL: Don't cache sigmas here! Always fetch from unet.get_model_object("model_sampling")
        # This allows the denoiser to pick up changes from patching that happens later
        logging.info(f"[FlowMatchingDenoiser] Created, will use model_sampling from unet_patcher dynamically")
        logging.info(f"[FlowMatchingDenoiser] Current sigma range: {self.sigma_min} - {self.sigma_max}")
        logging.info(f"[FlowMatchingDenoiser] Current model_sampling type: {type(self.model_sampling).__name__}")

        # CRITICAL: Patch inner_model.model_sampling to use the patched version
        # This ensures noise_scaling and inverse_noise_scaling use correct formulas
        self.inner_model.model_sampling = self.model_sampling
        logging.info(f"[FlowMatchingDenoiser] Patched inner_model.model_sampling")

        # Add compatibility for Comfy samplers that expect model_patcher
        # This allows DPM++ SDE and other advanced samplers to access model_sampling
        self.model_patcher = self.unet
        logging.info(f"[FlowMatchingDenoiser] Exposed model_patcher for Comfy sampler compatibility")

    @property
    def model_sampling(self):
        """Always get the latest model_sampling from the unet - this picks up patches"""
        # CRITICAL: Use get_model_object to get the patched version!
        return self.unet.get_model_object("model_sampling")

    @property
    def sigmas(self):
        """Always get fresh sigmas from model_sampling"""
        return self.model_sampling.sigmas

    @sigmas.setter
    def sigmas(self, value):
        """Allow setting sigmas for device transfers - update the model_sampling"""
        # When A1111 sampler moves sigmas to device, we need to handle it
        if isinstance(value, torch.Tensor):
            ms = self.model_sampling
            # Only update if device is different or if we need to update the buffer
            if hasattr(ms, 'register_buffer'):
                ms.register_buffer('sigmas', value)
                ms.register_buffer('log_sigmas', value.log())
                logging.debug(f"[FlowMatchingDenoiser] Updated sigmas on device: {value.device}")
            elif hasattr(ms, 'sigmas') and value.device != ms.sigmas.device:
                ms.sigmas = value
                if hasattr(ms, 'log_sigmas'):
                    ms.log_sigmas = value.log()
                logging.debug(f"[FlowMatchingDenoiser] Moved sigmas to device: {value.device}")

    @property
    def log_sigmas(self):
        """Always get fresh log_sigmas from model_sampling"""
        return self.sigmas.log()

    @log_sigmas.setter
    def log_sigmas(self, value):
        """Allow setting log_sigmas for device transfers - update the model_sampling"""
        # When A1111 sampler moves log_sigmas to device, we need to handle it
        if isinstance(value, torch.Tensor):
            ms = self.model_sampling
            # Only update if device is different or if we need to update the buffer
            if hasattr(ms, 'register_buffer'):
                ms.register_buffer('log_sigmas', value)
                ms.register_buffer('sigmas', value.exp())
                logging.debug(f"[FlowMatchingDenoiser] Updated log_sigmas on device: {value.device}")
            elif hasattr(ms, 'log_sigmas') and value.device != ms.log_sigmas.device:
                ms.log_sigmas = value
                if hasattr(ms, 'sigmas'):
                    ms.sigmas = value.exp()
                logging.debug(f"[FlowMatchingDenoiser] Moved log_sigmas to device: {value.device}")

    @property
    def sigma_min(self):
        return self.model_sampling.sigma_min

    @property
    def sigma_max(self):
        return self.model_sampling.sigma_max

    def get_sigmas(self, n=None):
        """Generate sigma schedule for n steps"""
        if n is None:
            # Flip and append zero
            return torch.cat([self.sigmas.flip(0), self.sigmas.new_zeros([1])])
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return torch.cat([self.t_to_sigma(t), self.sigmas.new_zeros([1])])

    def sigma_to_t(self, sigma, quantize=True):
        """Convert sigma to timestep"""
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        """Convert timestep to sigma"""
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def noise_latent(self, x, noise, sigma):
        """
        Add noise to latent for img2img/hires fix using CONST/flow matching formula.
        This is CRITICAL for flow models - they use a different noise formula than EPS models!

        EPS formula (WRONG for flow):  xi = x + noise * sigma
        CONST formula (CORRECT):       xi = sigma * noise + (1 - sigma) * x
        """
        sigma_reshaped = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        noisy = sigma_reshaped * noise + (1.0 - sigma_reshaped) * x
        logging.debug(f"[FlowMatchingDenoiser] noise_latent: sigma={sigma.item():.4f}, x range: [{x.min():.3f}, {x.max():.3f}], noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
        return noisy

    def forward(self, x, sigma, **kwargs):
        """
        Forward pass for flow matching / CONST prediction.
        For flow models: denoised = x - sigma * model_output

        This denoiser wraps the model for k-diffusion samplers.
        Unlike EPS prediction, flow matching uses CONST prediction where
        the model output is the velocity field.
        """
        # For flow models, sigma values are in [0, 1] range
        # The input x is NOT scaled (unlike EPS models)
        # Get model output - the model handles sigma->timestep conversion internally
        model_output = self.inner_model.apply_model(x, sigma, **kwargs)

        # For CONST/flow matching: denoised = x - sigma * model_output
        # This matches the CONST.calculate_denoised in ldm_patched/modules/model_sampling.py
        sigma_reshaped = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        denoised = x - model_output * sigma_reshaped

        # Debug extreme values
        if torch.isnan(denoised).any() or torch.isinf(denoised).any():
            logging.warning(f"[FlowMatchingDenoiser] NaN/Inf detected! sigma range: {sigma.min()}-{sigma.max()}, output range: {model_output.min()}-{model_output.max()}")

        return denoised

class AdvancedModelSamplingScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.sampling_mode = "Discrete"
        self.discrete_sampling = "v_prediction"
        self.discrete_zsnr = True
        self.continuous_edm_sampling = "v_prediction"
        self.continuous_edm_sigma_max = 120.0
        self.continuous_edm_sigma_min = 0.002
        self.continuous_v_sigma_max = 500.0
        self.continuous_v_sigma_min = 0.03
        self.stable_cascade_shift = 2.0
        self.sd3_shift = 2.5
        self.aura_flow_shift = 1.73
        self.flux_max_shift = 1.15
        self.flux_base_shift = 0.5
        self.flux_width = 1024
        self.flux_height = 1024

    sorting_priority = 15

    def title(self):
        return "Advanced Model Sampling for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Advanced Model Sampling.</i></p>")

            enabled = gr.Checkbox(label="Enable Advanced Model Sampling", value=self.enabled)

            sampling_mode = gr.Radio(
                ["Discrete", "Continuous EDM", "Continuous V", "Stable Cascade", "SD3", "Aura Flow", "Flux"],
                label="Sampling Mode",
                value=self.sampling_mode
            )

            with gr.Group(visible=True) as discrete_group:
                discrete_sampling = gr.Radio(
                    ["eps", "v_prediction", "lcm", "x0"],
                    label="Discrete Sampling Type",
                    value=self.discrete_sampling
                )
                discrete_zsnr = gr.Checkbox(label="Zero SNR", value=self.discrete_zsnr)

            with gr.Group(visible=False) as continuous_edm_group:
                continuous_edm_sampling = gr.Radio(
                    ["v_prediction", "edm_playground_v2.5", "eps"],
                    label="Continuous EDM Sampling Type",
                    value=self.continuous_edm_sampling
                )
                continuous_edm_sigma_max = gr.Slider(label="Sigma Max", minimum=0.0, maximum=1000.0, step=0.001, value=self.continuous_edm_sigma_max)
                continuous_edm_sigma_min = gr.Slider(label="Sigma Min", minimum=0.0, maximum=1000.0, step=0.001, value=self.continuous_edm_sigma_min)

            with gr.Group(visible=False) as continuous_v_group:
                continuous_v_sigma_max = gr.Slider(label="Sigma Max", minimum=0.0, maximum=1000.0, step=0.001, value=self.continuous_v_sigma_max)
                continuous_v_sigma_min = gr.Slider(label="Sigma Min", minimum=0.0, maximum=1000.0, step=0.001, value=self.continuous_v_sigma_min)

            with gr.Group(visible=False) as stable_cascade_group:
                stable_cascade_shift = gr.Slider(label="Shift", minimum=0.0, maximum=100.0, step=0.01, value=self.stable_cascade_shift)

            with gr.Group(visible=False) as sd3_group:
                sd3_shift = gr.Slider(label="Shift", minimum=0.0, maximum=100.0, step=0.01, value=self.sd3_shift)

            with gr.Group(visible=False) as aura_flow_group:
                aura_flow_shift = gr.Slider(label="Shift", minimum=0.0, maximum=100.0, step=0.01, value=self.aura_flow_shift)

            with gr.Group(visible=False) as flux_group:
                flux_max_shift = gr.Slider(label="Max Shift", minimum=0.0, maximum=100.0, step=0.01, value=self.flux_max_shift)
                flux_base_shift = gr.Slider(label="Base Shift", minimum=0.0, maximum=100.0, step=0.01, value=self.flux_base_shift)
                flux_width = gr.Slider(label="Width", minimum=16, maximum=8192, step=8, value=self.flux_width)
                flux_height = gr.Slider(label="Height", minimum=16, maximum=8192, step=8, value=self.flux_height)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Discrete")),
                    gr.Group.update(visible=(mode == "Continuous EDM")),
                    gr.Group.update(visible=(mode == "Continuous V")),
                    gr.Group.update(visible=(mode == "Stable Cascade")),
                    gr.Group.update(visible=(mode == "SD3")),
                    gr.Group.update(visible=(mode == "Aura Flow")),
                    gr.Group.update(visible=(mode == "Flux"))
                )

            sampling_mode.change(
                update_visibility,
                inputs=[sampling_mode],
                outputs=[discrete_group, continuous_edm_group, continuous_v_group, stable_cascade_group, sd3_group, aura_flow_group, flux_group]
            )

        return (enabled, sampling_mode, discrete_sampling, discrete_zsnr, continuous_edm_sampling, continuous_edm_sigma_max, continuous_edm_sigma_min,
                continuous_v_sigma_max, continuous_v_sigma_min, stable_cascade_shift, sd3_shift, aura_flow_shift,
                flux_max_shift, flux_base_shift, flux_width, flux_height)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 16:
            (self.enabled, self.sampling_mode, self.discrete_sampling, self.discrete_zsnr, self.continuous_edm_sampling,
             self.continuous_edm_sigma_max, self.continuous_edm_sigma_min, self.continuous_v_sigma_max, self.continuous_v_sigma_min,
             self.stable_cascade_shift, self.sd3_shift, self.aura_flow_shift, self.flux_max_shift, self.flux_base_shift,
             self.flux_width, self.flux_height) = args[:16]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        # Debug: Print original model info
        original_model_sampling = getattr(unet.model, 'model_sampling', None)
        if original_model_sampling:
            logging.info(f"[Advanced Sampling Debug] Original model_sampling type: {type(original_model_sampling).__name__}")
            logging.info(f"[Advanced Sampling Debug] Original sigma_min: {getattr(original_model_sampling, 'sigma_min', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] Original sigma_max: {getattr(original_model_sampling, 'sigma_max', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] Model config: {type(unet.model.model_config).__name__ if hasattr(unet.model, 'model_config') else 'No model_config'}")

        # Debug: Print model type detection
        model_type = getattr(unet.model.model_config, 'unet_config', {}) if hasattr(unet.model, 'model_config') else {}
        logging.info(f"[Advanced Sampling Debug] UNet config keys: {list(model_type.keys()) if model_type else 'No unet_config'}")

        if self.sampling_mode == "Discrete":
            unet = ModelSamplingDiscrete().patch(unet, self.discrete_sampling, self.discrete_zsnr)[0]
        elif self.sampling_mode == "Continuous EDM":
            unet = ModelSamplingContinuousEDM().patch(unet, self.continuous_edm_sampling, self.continuous_edm_sigma_max, self.continuous_edm_sigma_min)[0]
        elif self.sampling_mode == "Continuous V":
            unet = ModelSamplingContinuousV().patch(unet, "v_prediction", self.continuous_v_sigma_max, self.continuous_v_sigma_min)[0]
        elif self.sampling_mode == "Stable Cascade":
            unet = ModelSamplingStableCascade().patch(unet, self.stable_cascade_shift)[0]
        elif self.sampling_mode == "SD3":
            logging.info(f"[Advanced Sampling Debug] Applying SD3 sampling with shift={self.sd3_shift}")
            unet = ModelSamplingSD3().patch(unet, self.sd3_shift)[0]
        elif self.sampling_mode == "Aura Flow":
            unet = ModelSamplingAuraFlow().patch_aura(unet, self.aura_flow_shift)[0]
        elif self.sampling_mode == "Flux":
            unet = ModelSamplingFlux().patch(unet, self.flux_max_shift, self.flux_base_shift, self.flux_width, self.flux_height)[0]

        p.sd_model.forge_objects.unet = unet

        # CRITICAL FIX: Update the model's create_denoiser to use the patched model_sampling
        # This ensures A1111 backend samplers use the correct sigmas and prediction type
        self._patch_model_denoiser(p, unet)

        p.extra_generation_params.update({
            "advanced_sampling_enabled": self.enabled,
            "advanced_sampling_mode": self.sampling_mode,
            "discrete_sampling": self.discrete_sampling if self.sampling_mode == "Discrete" else None,
            "discrete_zsnr": self.discrete_zsnr if self.sampling_mode == "Discrete" else None,
            "continuous_edm_sampling": self.continuous_edm_sampling if self.sampling_mode == "Continuous EDM" else None,
            "continuous_edm_sigma_max": self.continuous_edm_sigma_max if self.sampling_mode == "Continuous EDM" else None,
            "continuous_edm_sigma_min": self.continuous_edm_sigma_min if self.sampling_mode == "Continuous EDM" else None,
            "continuous_v_sigma_max": self.continuous_v_sigma_max if self.sampling_mode == "Continuous V" else None,
            "continuous_v_sigma_min": self.continuous_v_sigma_min if self.sampling_mode == "Continuous V" else None,
            "stable_cascade_shift": self.stable_cascade_shift if self.sampling_mode == "Stable Cascade" else None,
            "sd3_shift": self.sd3_shift if self.sampling_mode == "SD3" else None,
            "aura_flow_shift": self.aura_flow_shift if self.sampling_mode == "Aura Flow" else None,
            "flux_max_shift": self.flux_max_shift if self.sampling_mode == "Flux" else None,
            "flux_base_shift": self.flux_base_shift if self.sampling_mode == "Flux" else None,
            "flux_width": self.flux_width if self.sampling_mode == "Flux" else None,
            "flux_height": self.flux_height if self.sampling_mode == "Flux" else None,
        })

        logging.debug(f"Advanced Model Sampling: Enabled: {self.enabled}, Mode: {self.sampling_mode}")

        return

    def _patch_model_denoiser(self, p, unet_patcher):
        """
        Patch the sampler's denoiser to use the patched model_sampling.
        This is crucial for flow matching models when using A1111 backend samplers.
        """
        try:
            sd_model = p.sd_model
            # Get the patched model_sampling using get_model_object
            patched_sampling = unet_patcher.get_model_object("model_sampling")

            # Check if this is a flow-based sampling mode that needs special handling
            is_flow_model = isinstance(patched_sampling, model_sampling.ModelSamplingDiscreteFlow)
            is_flux_model = isinstance(patched_sampling, model_sampling.ModelSamplingFlux)

            logging.info(f"[Advanced Sampling] Checking patched_sampling: {type(patched_sampling).__name__}, sigma_min: {patched_sampling.sigma_min}, sigma_max: {patched_sampling.sigma_max}")

            if is_flow_model or is_flux_model or self.sampling_mode in ["SD3", "Aura Flow", "Flux"]:
                logging.info(f"[Advanced Sampling] Patching denoiser for flow matching model (mode: {self.sampling_mode})")

                # Create the FlowMatchingDenoiser
                denoiser = FlowMatchingDenoiser(sd_model, unet_patcher)
                logging.info(f"[Advanced Sampling] Created FlowMatchingDenoiser")
                logging.info(f"[Advanced Sampling] Current sigma range: {denoiser.sigma_min} - {denoiser.sigma_max}")
                logging.info(f"[Advanced Sampling] Current model_sampling type: {type(denoiser.model_sampling).__name__}")

                # Patch the sampler's model_wrap directly
                # For A1111 KDiffusion samplers
                if hasattr(p, 'sampler') and hasattr(p.sampler, 'model_wrap_cfg'):
                    logging.info(f"[Advanced Sampling] Patching A1111 KDiffusion sampler")
                    p.sampler.model_wrap_cfg.model_wrap = denoiser
                    p.sampler.model_wrap = denoiser
                    logging.info(f"[Advanced Sampling] Replaced model_wrap in A1111 sampler")

                    # CRITICAL: Patch sample_img2img to use CONST noise scaling
                    self._patch_sampler_img2img(p.sampler, denoiser)

                # Also patch create_denoiser for future calls
                def create_flow_denoiser():
                    new_denoiser = FlowMatchingDenoiser(sd_model, unet_patcher)
                    logging.info(f"[Advanced Sampling] create_flow_denoiser called - sigma range: {new_denoiser.sigma_min} - {new_denoiser.sigma_max}")
                    return new_denoiser

                sd_model.create_denoiser = create_flow_denoiser

                # Also patch shared.sd_model if different
                try:
                    from modules import shared
                    if shared.sd_model is not sd_model:
                        logging.info(f"[Advanced Sampling] Also patching shared.sd_model.create_denoiser")
                        shared.sd_model.create_denoiser = create_flow_denoiser
                except Exception as e:
                    logging.debug(f"[Advanced Sampling] Could not patch shared.sd_model: {e}")

                logging.info(f"[Advanced Sampling] Successfully patched denoiser")
            else:
                # For non-flow models, we still need to update the sigmas
                # This ensures the A1111 backend uses the correct sigma range
                logging.info(f"[Advanced Sampling] Updating model sigmas for non-flow sampling mode")

                # Update alphas_cumprod to match the patched sigmas
                # alphas_cumprod = 1 / (sigmas^2 + 1)
                sigmas_sq = patched_sampling.sigmas ** 2
                new_alphas_cumprod = 1.0 / (sigmas_sq + 1.0)

                # Update the model's alphas_cumprod
                if hasattr(sd_model, 'alphas_cumprod'):
                    sd_model.alphas_cumprod = new_alphas_cumprod.cpu()
                    logging.info(f"[Advanced Sampling] Updated alphas_cumprod, new sigma range: {patched_sampling.sigma_min} - {patched_sampling.sigma_max}")

        except Exception as e:
            logging.error(f"[Advanced Sampling] Error patching model denoiser: {e}", exc_info=True)

    def _patch_sampler_img2img(self, sampler, denoiser):
        """
        Patch the sampler's sample_img2img to use CONST noise scaling instead of EPS.
        This fixes oversaturation issues in hires fix and img2img.
        """
        try:
            from modules import sd_samplers_common
            import inspect

            # Store the original method
            original_sample_img2img = sampler.sample_img2img

            def patched_sample_img2img(p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
                """Patched img2img that uses CONST noise scaling for flow models"""
                logging.info(f"[Advanced Sampling] patched_sample_img2img called")

                # Call the original method but intercept the noise addition
                # We need to monkey-patch the specific line where noise is added

                # Get the unet_patcher
                unet_patcher = sampler.model_wrap.inner_model.forge_objects.unet
                from modules_forge.forge_sampler import sampling_prepare

                sampling_prepare(sampler.model_wrap.inner_model.forge_objects.unet, x=x)

                sampler.model_wrap.log_sigmas = sampler.model_wrap.log_sigmas.to(x.device)
                sampler.model_wrap.sigmas = sampler.model_wrap.sigmas.to(x.device)

                steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

                sigmas = sampler.get_sigmas(p, steps).to(x.device)
                sigma_sched = sigmas[steps - t_enc - 1:]

                x = x.to(noise)

                # CRITICAL FIX: Use CONST noise scaling instead of EPS
                # Old (EPS): xi = x + noise * sigma_sched[0]
                # New (CONST): xi = sigma * noise + (1 - sigma) * x
                if hasattr(denoiser, 'noise_latent'):
                    xi = denoiser.noise_latent(x, noise, sigma_sched[0])
                    logging.info(f"[Advanced Sampling] Using CONST noise scaling: sigma={sigma_sched[0].item():.4f}")
                else:
                    # Fallback to EPS-style
                    xi = x + noise * sigma_sched[0]
                    logging.warning(f"[Advanced Sampling] Falling back to EPS noise scaling")

                # Handle extra noise
                from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
                from modules.shared import opts
                if opts.img2img_extra_noise > 0:
                    p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
                    extra_noise_params = ExtraNoiseParams(noise, x, xi)
                    extra_noise_callback(extra_noise_params)
                    noise = extra_noise_params.noise
                    xi += noise * opts.img2img_extra_noise

                # Continue with the rest of the sampling setup
                extra_params_kwargs = sampler.initialize(p)
                parameters = inspect.signature(sampler.func).parameters

                if 'sigma_min' in parameters:
                    extra_params_kwargs['sigma_min'] = sigma_sched[-2]
                if 'sigma_max' in parameters:
                    extra_params_kwargs['sigma_max'] = sigma_sched[0]
                if 'n' in parameters:
                    extra_params_kwargs['n'] = len(sigma_sched) - 1
                if 'sigma_sched' in parameters:
                    extra_params_kwargs['sigma_sched'] = sigma_sched
                # NOTE: Don't add 'sigmas' to extra_params_kwargs since we pass sigma_sched as positional arg

                if sampler.config.options.get('brownian_noise', False):
                    noise_sampler = sampler.create_noise_sampler(x, sigmas, p)
                    extra_params_kwargs['noise_sampler'] = noise_sampler

                if opts.sd_sampling == "A1111":
                    if sampler.config.options.get('solver_type', None) == 'heun':
                        extra_params_kwargs['solver_type'] = 'heun'

                sampler.model_wrap_cfg.init_latent = x
                sampler.last_latent = x
                sampler.sampler_extra_args = {
                    'cond': conditioning,
                    'image_cond': image_conditioning,
                    'uncond': unconditional_conditioning,
                    'cond_scale': p.cfg_scale,
                    's_min_uncond': sampler.s_min_uncond
                }

                samples = sampler.launch_sampling(t_enc + 1, lambda: sampler.func(sampler.model_wrap_cfg, xi, sigma_sched, extra_args=sampler.sampler_extra_args, callback=sampler.callback_state, disable=False, **extra_params_kwargs))

                from modules_forge.forge_sampler import sampling_cleanup
                sampling_cleanup(sampler.model_wrap.inner_model.forge_objects.unet)

                return samples

            # Replace the method
            sampler.sample_img2img = patched_sample_img2img
            logging.info(f"[Advanced Sampling] Successfully patched sample_img2img for CONST noise scaling")

        except Exception as e:
            logging.error(f"[Advanced Sampling] Error patching sample_img2img: {e}", exc_info=True)
