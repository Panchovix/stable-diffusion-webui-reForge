import gradio as gr

from modules import scripts, shared
from modules.ui_components import InputAccordion
from backend.misc.image_resize import adaptive_resize


class PatchModelAddDownscale:
    def patch(self, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, phase2):
        sigma_start = model.model.predictor.percent_to_sigma(start_percent)
        sigma_end = model.model.predictor.percent_to_sigma(end_percent)
        sigma_p2 = model.model.predictor.percent_to_sigma(min(1.0, end_percent * 1.5))
        downscale_p2 = 0.5 * (downscale_factor + 1.0)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                if sigma <= sigma_start and sigma >= sigma_end:
                    h = adaptive_resize(h, round(h.shape[-1] * (1.0 / downscale_factor)), round(h.shape[-2] * (1.0 / downscale_factor)), downscale_method, "disabled")
                elif sigma <= sigma_start and sigma >= sigma_p2 and phase2:
                    h = adaptive_resize(h, round(h.shape[-1] * (1.0 / downscale_p2)), round(h.shape[-2] * (1.0 / downscale_p2)), downscale_method, "disabled")

            shared.kohya_shrink_shape = (h.shape[-1], h.shape[-2])
            shared.kohya_shrink_shape_out = None
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[2] != hsp.shape[2]:
                h = adaptive_resize(h, hsp.shape[-1], hsp.shape[-2], upscale_method, "disabled")

            shared.kohya_shrink_shape_out = (h.shape[-1], h.shape[-2])
            return h, hsp

        m = model.clone()
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m,)


opPatchModelAddDownscale = PatchModelAddDownscale()


class KohyaHRFixForForge(scripts.Script):
    sorting_priority = 14

    def title(self):
        return "Kohya HRFix Integrated (SD1.x, SD2.x, SDXL)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp", "adaptive_avgpool"]
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                block_number = gr.Slider(label='Block Number', value=3, minimum=1, maximum=32, step=1)
                downscale_factor = gr.Slider(label='Downscale Factor (0: auto calculated)', value=2.0, minimum=0.0, maximum=9.0, step=0.001)
            with gr.Row():
                start_percent = gr.Slider(label='Start Percent', value=0.0, minimum=0.0, maximum=1.0, step=0.001)
                end_percent = gr.Slider(label='End Percent', value=0.35, minimum=0.0, maximum=1.0, step=0.001)
            downscale_after_skip = gr.Checkbox(label='Downscale After Skip', value=True)
            with gr.Row():
                downscale_method = gr.Dropdown(label='Downscale Method', choices=upscale_methods, value=upscale_methods[0])
                upscale_method = gr.Dropdown(label='Upscale Method', choices=upscale_methods, value=upscale_methods[0])
            phase2 = gr.Checkbox(label='Do second phase (auto calculated)', value=False)

        self.infotext_fields = [
            (enabled, lambda d: d.get("kohya_hrfix_enabled", False)),
            (block_number,          "kohya_hrfix_block_number"),
            (downscale_factor,      "kohya_hrfix_downscale_factor"),
            (start_percent,         "kohya_hrfix_start_percent"),
            (end_percent,           "kohya_hrfix_end_percent"),
            (downscale_after_skip,  "kohya_hrfix_downscale_after_skip"),
            (downscale_method,      "kohya_hrfix_downscale_method"),
            (upscale_method,        "kohya_hrfix_upscale_method"),
            (phase2,                "kohya_hrfix_phase2"),
        ]

        return enabled, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, phase2

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, phase2 = script_args
        block_number = int(block_number)

        if enabled:
            if p.is_hr_pass:
                return
            if not p.sd_model.is_webui_legacy_model() or p.sd_model.is_sd3:
                return

            #auto calc downscale factor based on model and width/height?
            if downscale_factor == 0.0:
                width = p.width
                height = p.height
                if p.sd_model.is_sd1:
                    downscale_factor = ((width * height) ** 0.5) / 512
                elif p.sd_model.is_sd2:
                    downscale_factor = ((width * height) ** 0.5) / 768
                elif p.sd_model.is_sdxl:
                    downscale_factor = ((width * height) ** 0.5) / 1024

                p.extra_generation_params.update(dict(
                    kohya_hrfix_downscale_factor=round(downscale_factor, 3)
                ))  # 'params.txt' already has previous value written

            unet = p.sd_model.forge_objects.unet

            unet = opPatchModelAddDownscale.patch(unet, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, phase2)[0]

            p.sd_model.forge_objects.unet = unet
        return

    def process(self, p, *script_args, **kwargs):
        enabled, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, phase2 = script_args
        block_number = int(block_number)

        if enabled:
            if p.sd_model.is_webui_legacy_model() or p.sd_model.is_sd3:
                return

            p.extra_generation_params.update(dict(
                kohya_hrfix_enabled=enabled,
                kohya_hrfix_block_number=block_number,
                kohya_hrfix_downscale_factor=downscale_factor,
                kohya_hrfix_start_percent=start_percent,
                kohya_hrfix_end_percent=end_percent,
                kohya_hrfix_downscale_after_skip=downscale_after_skip,
                kohya_hrfix_downscale_method=downscale_method,
                kohya_hrfix_upscale_method=upscale_method,
                kohya_hrfix_phase2=phase2,
            ))

        return

