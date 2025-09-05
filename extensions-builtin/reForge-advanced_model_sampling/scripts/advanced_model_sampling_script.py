import logging
import gradio as gr
from modules import scripts
from modules.shared import opts
from ldm_patched.modules import model_sampling
from advanced_model_sampling.nodes_model_advanced import (
    ModelSamplingDiscrete, ModelSamplingContinuousEDM, ModelSamplingContinuousV,
    ModelSamplingStableCascade, ModelSamplingSD3, ModelSamplingAuraFlow, ModelSamplingFlux
)

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

        # Debug: Print patched model info
        patched_model_sampling = getattr(unet.model, 'model_sampling', None)
        if patched_model_sampling:
            logging.info(f"[Advanced Sampling Debug] Patched model_sampling type: {type(patched_model_sampling).__name__}")
            logging.info(f"[Advanced Sampling Debug] Patched sigma_min: {getattr(patched_model_sampling, 'sigma_min', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] Patched sigma_max: {getattr(patched_model_sampling, 'sigma_max', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] Shift: {getattr(patched_model_sampling, 'shift', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] Multiplier: {getattr(patched_model_sampling, 'multiplier', 'N/A')}")
            
        # Debug: Check object_patches
        if hasattr(unet, 'object_patches'):
            logging.info(f"[Advanced Sampling Debug] Object patches: {list(unet.object_patches.keys())}")
            if 'model_sampling' in unet.object_patches:
                obj_patch = unet.object_patches['model_sampling']
                logging.info(f"[Advanced Sampling Debug] model_sampling patch type: {type(obj_patch).__name__}")
                logging.info(f"[Advanced Sampling Debug] model_sampling patch shift: {getattr(obj_patch, 'shift', 'N/A')}")
                logging.info(f"[Advanced Sampling Debug] model_sampling patch multiplier: {getattr(obj_patch, 'multiplier', 'N/A')}")

        # POTENTIAL FIX: Force apply the model_sampling patch immediately
        # The issue seems to be that object patches aren't applied until sampling starts
        # But we need model_sampling to be applied immediately for the sigma values to be correct
        if 'model_sampling' in unet.object_patches:
            logging.info(f"[Advanced Sampling Debug] APPLYING FIX - Force applying model_sampling patch immediately")
            patch_obj = unet.object_patches['model_sampling']
            # Store the original in backup if not already done
            if 'model_sampling' not in unet.object_patches_backup:
                unet.object_patches_backup['model_sampling'] = unet.model.model_sampling
            # Apply the patch immediately
            unet.model.model_sampling = patch_obj
            logging.info(f"[Advanced Sampling Debug] FIX APPLIED - type: {type(unet.model.model_sampling).__name__}")
            logging.info(f"[Advanced Sampling Debug] FIX APPLIED - sigma_min: {unet.model.model_sampling.sigma_min}")
            logging.info(f"[Advanced Sampling Debug] FIX APPLIED - sigma_max: {unet.model.model_sampling.sigma_max}")
            logging.info(f"[Advanced Sampling Debug] FIX APPLIED - shift: {getattr(unet.model.model_sampling, 'shift', 'N/A')}")
            logging.info(f"[Advanced Sampling Debug] FIX APPLIED - multiplier: {getattr(unet.model.model_sampling, 'multiplier', 'N/A')}")

        # SCHEDULER FIX: Set global state for WebUI scheduler to detect
        from modules import shared
        shared.forge_advanced_sampling_state = {
            'enabled': self.enabled,
            'mode': self.sampling_mode,
            'sd3_shift': self.sd3_shift,
            'aura_flow_shift': self.aura_flow_shift,
            'discrete_sampling': self.discrete_sampling,
            'discrete_zsnr': self.discrete_zsnr
        }
        logging.info(f"[Advanced Sampling Debug] Set global state for scheduler: {shared.forge_advanced_sampling_state}")

        p.sd_model.forge_objects.unet = unet

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
