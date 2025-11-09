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


class FlowMatchingDenoiser(torch.nn.Module):
    """Custom denoiser for flow matching models that uses CONST prediction and patched model_sampling sigmas"""

    def __init__(self, model, unet_patcher):
        super().__init__()
        self.inner_model = model
        self.unet = unet_patcher
        # Get sigmas from the patched model_sampling
        self.model_sampling = self.unet.model.model_sampling
        self.register_buffer('sigmas', self.model_sampling.sigmas.clone())
        self.register_buffer('log_sigmas', self.model_sampling.sigmas.log().clone())
        self.sigma_data = 1.0

        logging.info(f"[FlowMatchingDenoiser] Created with sigma range: {self.sigma_min} - {self.sigma_max}")
        logging.info(f"[FlowMatchingDenoiser] Model sampling type: {type(self.model_sampling).__name__}")

        # Also patch the inner_model to expose the patched model_sampling
        # This ensures noise_scaling and other operations use the correct sampling
        if not hasattr(self.inner_model, 'forge_objects'):
            # For non-forge models, add model_sampling directly
            self.inner_model.model_sampling = self.model_sampling

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

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
        self.sd3_shift = 3.0
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
        self._patch_model_denoiser(p.sd_model, unet)

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

    def _patch_model_denoiser(self, sd_model, unet_patcher):
        """
        Patch the model's create_denoiser method to use the patched model_sampling.
        This is crucial for flow matching models when using A1111 backend samplers.
        """
        try:
            patched_sampling = unet_patcher.model.model_sampling

            # Check if this is a flow-based sampling mode that needs special handling
            is_flow_model = isinstance(patched_sampling, model_sampling.ModelSamplingDiscreteFlow)
            is_flux_model = isinstance(patched_sampling, model_sampling.ModelSamplingFlux)

            if is_flow_model or is_flux_model or self.sampling_mode in ["SD3", "Aura Flow", "Flux"]:
                logging.info(f"[Advanced Sampling] Patching create_denoiser for flow matching model (mode: {self.sampling_mode})")

                # Create a closure that captures the unet_patcher
                def create_flow_denoiser():
                    denoiser = FlowMatchingDenoiser(sd_model, unet_patcher)
                    logging.info(f"[Advanced Sampling] Created FlowMatchingDenoiser with sigma range: {denoiser.sigma_min} - {denoiser.sigma_max}")
                    return denoiser

                # Replace the create_denoiser method
                sd_model.create_denoiser = create_flow_denoiser

                logging.info(f"[Advanced Sampling] Successfully patched create_denoiser method")
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
