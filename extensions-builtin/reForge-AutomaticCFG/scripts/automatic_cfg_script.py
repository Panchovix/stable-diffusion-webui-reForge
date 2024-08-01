import gradio as gr
from modules import scripts
from automaticCFG.nodes import advancedDynamicCFG, simpleDynamicCFG, simpleDynamicCFGlerpUncond, simpleDynamicCFGwarpDrive, presetLoader, simpleDynamicCFGExcellentattentionPatch, simpleDynamicCFGCustomAttentionPatch, postCFGrescaleOnly
import os
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
json_preset_path = os.path.join(current_dir, 'automaticCFG', 'presets')

class AutomaticCFGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.advcfg = advancedDynamicCFG()

    def title(self):
        return "Automatic CFG for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            with gr.Tab("Simple CFG"):
                simple_cfg_enabled = gr.Checkbox(label="Enable Simple CFG", value=False)
                hard_mode = gr.Checkbox(label="Hard Mode", value=True)
                boost = gr.Checkbox(label="Boost", value=True)

            with gr.Tab("Dynamic CFG"):
                dynamic_cfg_enabled = gr.Checkbox(label="Enable Dynamic CFG", value=False)
                boost_dynamic = gr.Checkbox(label="Boost", value=True)
                negative_strength = gr.Slider(label="Negative Strength", minimum=0.0, maximum=5.0, step=0.1, value=1.0)

            with gr.Tab("Warp Drive"):
                warp_drive_enabled = gr.Checkbox(label="Enable Warp Drive", value=False)
                uncond_sigma_start = gr.Slider(label="Uncond Sigma Start", minimum=0.0, maximum=10000.0, step=0.1, value=5.5)
                uncond_sigma_end = gr.Slider(label="Uncond Sigma End", minimum=0.0, maximum=10000.0, step=0.1, value=1.0)
                fake_uncond_sigma_end = gr.Slider(label="Fake Uncond Sigma End", minimum=0.0, maximum=10000.0, step=0.1, value=1.0)

            with gr.Tab("Preset Loader"):
                preset_enabled = gr.Checkbox(label="Enable Preset", value=False)
                presets = [pj.replace(".json", "") for pj in os.listdir(json_preset_path) if ".json" in pj]
                preset_name = gr.Dropdown(label="Preset", choices=presets, value="Excellent_attention" if "Excellent_attention" in presets else presets[0])
                use_uncond_sigma_end_from_preset = gr.Checkbox(label="Use Uncond Sigma End from Preset", value=True)
                preset_uncond_sigma_end = gr.Slider(label="Preset Uncond Sigma End", minimum=0.0, maximum=10000.0, step=0.1, value=0.0)
                preset_automatic_cfg = gr.Dropdown(label="Automatic CFG", choices=["From preset", "None", "soft", "hard", "hard_squared", "range"], value="From preset")

            with gr.Tab("Post CFG Rescale"):
                post_cfg_rescale_enabled = gr.Checkbox(label="Enable Post CFG Rescale", value=False)
                subtract_latent_mean = gr.Checkbox(label="Subtract Latent Mean", value=True)
                subtract_latent_mean_sigma_start = gr.Slider(label="Subtract Latent Mean Sigma Start", minimum=0.0, maximum=10000.0, step=0.1, value=1000.0)
                subtract_latent_mean_sigma_end = gr.Slider(label="Subtract Latent Mean Sigma End", minimum=0.0, maximum=10000.0, step=0.1, value=7.5)
                latent_intensity_rescale = gr.Checkbox(label="Latent Intensity Rescale", value=True)
                latent_intensity_rescale_method = gr.Dropdown(label="Latent Intensity Rescale Method", choices=["soft", "hard", "range"], value="hard")
                latent_intensity_rescale_cfg = gr.Slider(label="Latent Intensity Rescale CFG", minimum=0.0, maximum=100.0, step=0.1, value=8.0)
                latent_intensity_rescale_sigma_start = gr.Slider(label="Latent Intensity Rescale Sigma Start", minimum=0.0, maximum=10000.0, step=0.1, value=1000.0)
                latent_intensity_rescale_sigma_end = gr.Slider(label="Latent Intensity Rescale Sigma End", minimum=0.0, maximum=10000.0, step=0.1, value=5.0)

        return (enabled,simple_cfg_enabled, hard_mode, boost, 
            dynamic_cfg_enabled, boost_dynamic, negative_strength,
            warp_drive_enabled, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end,
            preset_enabled, preset_name, use_uncond_sigma_end_from_preset, preset_uncond_sigma_end, preset_automatic_cfg,
            post_cfg_rescale_enabled, subtract_latent_mean, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end,
            latent_intensity_rescale, latent_intensity_rescale_method, latent_intensity_rescale_cfg,
            latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (enabled,simple_cfg_enabled, hard_mode, boost, 
        dynamic_cfg_enabled, boost_dynamic, negative_strength,
        warp_drive_enabled, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end,
        preset_enabled, preset_name, use_uncond_sigma_end_from_preset, preset_uncond_sigma_end, preset_automatic_cfg,
        post_cfg_rescale_enabled, subtract_latent_mean, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end,
        latent_intensity_rescale, latent_intensity_rescale_method, latent_intensity_rescale_cfg,
        latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end) = script_args

        if not enabled:
            return
        
        unet = p.sd_model.forge_objects.unet

        def wrap_pre_cfg_function(func):
            def wrapper(model, cond, uncond, x, timestep, model_options):
                sigma = timestep
                cond_scale = model_options.get('cond_scale', 1.0)
                uncond, cond, cond_scale = func(sigma, uncond, cond, cond_scale)
                return model, cond, uncond, x, timestep, model_options
            return wrapper

        def safe_patch(patch_func, *args):
            result = patch_func(*args)
            if isinstance(result, tuple):
                return result[0]
            return result

        def update_model_options(unet):
            if hasattr(unet, 'model_options') and 'sampler_pre_cfg_function' in unet.model_options:
                if callable(unet.model_options['sampler_pre_cfg_function']):
                    unet.model_options['sampler_pre_cfg_function'] = [wrap_pre_cfg_function(unet.model_options['sampler_pre_cfg_function'])]
                elif isinstance(unet.model_options['sampler_pre_cfg_function'], list):
                    unet.model_options['sampler_pre_cfg_function'] = [wrap_pre_cfg_function(func) for func in unet.model_options['sampler_pre_cfg_function']]
            return unet

        if simple_cfg_enabled:
            unet = safe_patch(simpleDynamicCFG().patch, unet, hard_mode, boost)
            unet = update_model_options(unet)
        
        if dynamic_cfg_enabled:
            unet = safe_patch(simpleDynamicCFGlerpUncond().patch, unet, boost_dynamic, negative_strength)
            unet = update_model_options(unet)
        
        if warp_drive_enabled:
            unet = safe_patch(simpleDynamicCFGwarpDrive().patch, unet, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end)
            unet = update_model_options(unet)
        
        if preset_enabled:
            unet = safe_patch(presetLoader().patch, unet, preset_name, 
                                        preset_uncond_sigma_end if not use_uncond_sigma_end_from_preset else 0.0, 
                                        use_uncond_sigma_end_from_preset, 
                                        preset_automatic_cfg)
            unet = update_model_options(unet)
        
        if post_cfg_rescale_enabled:
            unet = safe_patch(postCFGrescaleOnly().patch, unet, 
                                            subtract_latent_mean, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end,
                                            latent_intensity_rescale, latent_intensity_rescale_method, latent_intensity_rescale_cfg,
                                            latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end)
            unet = update_model_options(unet)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            automatic_cfg_simple_enabled=simple_cfg_enabled,
            automatic_cfg_dynamic_enabled=dynamic_cfg_enabled,
            automatic_cfg_warp_drive_enabled=warp_drive_enabled,
            automatic_cfg_preset_enabled=preset_enabled,
            automatic_cfg_post_cfg_rescale_enabled=post_cfg_rescale_enabled,
        ))

        return
