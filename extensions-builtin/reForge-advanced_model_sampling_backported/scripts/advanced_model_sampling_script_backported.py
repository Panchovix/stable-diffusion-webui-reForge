import logging
import gradio as gr
from modules import scripts, shared_options as opts

class AdvancedModelSamplingScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.sampling_mode = "Discrete"
        self.discrete_sampling = "v_prediction"
        self.discrete_zsnr = True  # FIXME: This produces incorrect results when disabled. Need to investigate.
        self.continuous_edm_sampling = "v_prediction"
        self.continuous_edm_sigma_max = 120.0
        self.continuous_edm_sigma_min = 0.002
        self.goofysampling_betascale = False        #hey this might fix that fixme above! also this is useless unless you're implementing sigma shifting re: simplediffusion *at train time*.
        self.hardcodebetas_linear_start = 0.00085   # these
        self.hardcodebetas_linear_end = 0.012       # are like,
        self.hardcodebetas_timesteps = 1000         # the defaults enumerated elsewhere in model base code and correspond to the sd15->sd3 betas schedule. 

    sorting_priority = 15

    def title(self):
        return "Advanced Model Sampling for reForge (Backported)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Advanced Model Sampling.</i></p>")

            enabled = gr.Checkbox(label="Enable Advanced Model Sampling", value=self.enabled)

            sampling_mode = gr.Radio(
                ["Discrete", "Continuous EDM"],
                label="Sampling Mode",
                value=self.sampling_mode
            )

            with gr.Group(visible=True) as discrete_group:
                discrete_sampling = gr.Radio(
                    ["eps", "v_prediction", "lcm"],
                    label="Discrete Sampling Type",
                    value=self.discrete_sampling                    
                )
                discrete_zsnr = gr.Checkbox(label="Zero SNR", value=self.discrete_zsnr)
            
            with gr.Group(visible=False) as continuous_edm_group:
                continuous_edm_sampling = gr.Radio(
                    ["v_prediction", "eps"],
                    label="Continuous EDM Sampling Type",
                    value=self.continuous_edm_sampling
                )
                continuous_edm_sigma_max = gr.Slider(
                    label="Sigma Max",
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.001,
                    value=self.continuous_edm_sigma_max
                )
                continuous_edm_sigma_min = gr.Slider(
                    label="Sigma Min",
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.001,
                    value=self.continuous_edm_sigma_min
                )

            goofysampling_betascale = gr.Checkbox(label="GoofySampling Betascale [x]", value=self.goofysampling_betascale)

            with gr.Group(visible=False) as betascale_group:
                hardcodebetas_linear_start = gr.Slider(label="betas_linear_start", minimum=0.0, maximum=1.0, step=0.00005, value=self.hardcodebetas_linear_start)
                hardcodebetas_linear_end = gr.Slider(label="betas_linear_end", minimum=0.0, maximum=1.0, step=0.00005, value=self.hardcodebetas_linear_end)
                hardcodebetas_timesteps = gr.Slider(label="timesteps", minimum=1, maximum=10000.0, step=1, value=self.hardcodebetas_timesteps)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Discrete")),
                    gr.Group.update(visible=(mode == "Continuous EDM"))
                )
            def update_goofysampling_visibility(checkbox):
                return ( gr.Group.update(visible=checkbox) )

            sampling_mode.change(
                update_visibility,
                inputs=[sampling_mode],
                outputs=[discrete_group, continuous_edm_group]
            )

            goofysampling_betascale.change(
                update_goofysampling_visibility,
                inputs=goofysampling_betascale,
                outputs=[betascale_group]
            )

        return (enabled, sampling_mode, discrete_sampling, discrete_zsnr,
                continuous_edm_sampling, continuous_edm_sigma_max, continuous_edm_sigma_min, 
                goofysampling_betascale, hardcodebetas_linear_start, hardcodebetas_linear_end, hardcodebetas_timesteps)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 11:
            (self.enabled, self.sampling_mode, self.discrete_sampling, self.discrete_zsnr,
             self.continuous_edm_sampling, self.continuous_edm_sigma_max, self.continuous_edm_sigma_min, 
             self.goofysampling_betascale, self.hardcodebetas_linear_start, self.hardcodebetas_linear_end, self.hardcodebetas_timesteps) = args[:11]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        from advanced_model_sampling.nodes_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM

        # Store original UNet for restoration
        if not hasattr(p.sd_model.forge_objects.unet, '_original_model'):
            p.sd_model.forge_objects.unet._original_model = p.sd_model.forge_objects.unet.model.model_sampling

        unet = p.sd_model.forge_objects.unet.clone()

        if self.sampling_mode == "Discrete":
            sampler = ModelSamplingDiscrete()
            unet = sampler.patch(unet, self.discrete_sampling, self.discrete_zsnr, 
            patch_timesteps=self.hardcodebetas_timesteps, patch_linear_start=self.hardcodebetas_linear_start, patch_linear_end=self.hardcodebetas_linear_end)[0]
        elif self.sampling_mode == "Continuous EDM":
            sampler = ModelSamplingContinuousEDM()
            unet = sampler.patch(unet, self.continuous_edm_sampling, 
                               self.continuous_edm_sigma_max, 
                               self.continuous_edm_sigma_min)[0]

        p.sd_model.forge_objects.unet = unet

        # Add sampling info to generation parameters
        p.extra_generation_params.update({
            "advanced_sampling_enabled": self.enabled,
            "advanced_sampling_mode": self.sampling_mode,
            "discrete_sampling": self.discrete_sampling if self.sampling_mode == "Discrete" else None,
            "discrete_zsnr": self.discrete_zsnr if self.sampling_mode == "Discrete" else None,
            "continuous_edm_sampling": self.continuous_edm_sampling if self.sampling_mode == "Continuous EDM" else None,
            "continuous_edm_sigma_max": self.continuous_edm_sigma_max if self.sampling_mode == "Continuous EDM" else None,
            "continuous_edm_sigma_min": self.continuous_edm_sigma_min if self.sampling_mode == "Continuous EDM" else None,
            "noise density function override β(0)": self.hardcodebetas_linear_start if self.goofysampling_betascale == True else None,
            "noise density function override β(T)": self.hardcodebetas_linear_end if self.goofysampling_betascale == True else None,
            "noise density function override T domain": self.hardcodebetas_timesteps if self.goofysampling_betascale == True else None
        })

    def postprocess(self, p, processed, *args):
        """Restore original sampling after generation"""
        if hasattr(p.sd_model.forge_objects.unet, '_original_model'):
            p.sd_model.forge_objects.unet.model.model_sampling = p.sd_model.forge_objects.unet._original_model
            del p.sd_model.forge_objects.unet._original_model

        logging.debug(f"Advanced Model Sampling: Enabled: {self.enabled}, Mode: {self.sampling_mode}")

        return
		