import gradio as gr
import copy

from modules import scripts, shared
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.ui_components import InputAccordion


class PerturbedAttentionGuidanceForForge(scripts.Script):
    sorting_priority = 13

    def title(self):
        return "PerturbedAttentionGuidance Integrated (SD1.x, SD2.x, SDXL)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=100.0, step=0.1, value=3.0)
                attenuation = gr.Slider(label='Attenuation (linear, % of scale)', minimum=0.0, maximum=100.0, step=0.1, value=0.0)
            with gr.Row():
                start_step = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.15)
                end_step = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=0.6)


        self.infotext_fields = [
            (enabled, lambda d: d.get("pagi_enabled", False)),
            (scale,         "pagi_scale"),
            (attenuation,   "pagi_attenuation"),
            (start_step,    "pagi_start_step"),
            (end_step,      "pagi_end_step"),
        ]

        return enabled, scale, attenuation, start_step, end_step

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, attenuation, start_step, end_step = script_args

        if enabled:
            PerturbedAttentionGuidanceForForge.scale = scale
            PerturbedAttentionGuidanceForForge.PAG_start = start_step
            PerturbedAttentionGuidanceForForge.PAG_end = end_step

            unet = p.sd_model.forge_objects.unet.clone()

            def attn_proc(q, k, v, to):
                return v

            def post_cfg_function(args):
                denoised = args["denoised"]

                if PerturbedAttentionGuidanceForForge.scale <= 0.0:
                    return denoised

                thisStep = shared.state.sampling_step / (shared.state.sampling_steps - 1)
                if thisStep < PerturbedAttentionGuidanceForForge.PAG_start or thisStep > PerturbedAttentionGuidanceForForge.PAG_end:
                    return denoised

                model, cond_denoised, cond, sigma, x, options = \
                    args["model"], args["cond_denoised"], args["cond"], args["sigma"], args["input"], args["model_options"].copy()

                pag_options = set_model_options_patch_replace(options, attn_proc, "attn1", "middle", 0, 0)
               
                degraded, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, pag_options)

                result = denoised + (cond_denoised - degraded) * PerturbedAttentionGuidanceForForge.scale
                PerturbedAttentionGuidanceForForge.scale -= scale * attenuation / 100.0

                return result

            unet.set_model_sampler_post_cfg_function(post_cfg_function)

            p.sd_model.forge_objects.unet = unet

        return

    def process(self, p, *script_args, **kwargs):
        enabled, scale, attenuation, start_step, end_step = script_args

        if enabled:
            p.extra_generation_params.update(dict(
                pagi_enabled     = enabled,
                pagi_scale       = scale,
                pagi_attenuation = attenuation,
                pagi_start_step  = start_step,
                pagi_end_step    = end_step,
            ))

        return
