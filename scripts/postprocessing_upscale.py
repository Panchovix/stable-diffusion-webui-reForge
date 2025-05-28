from PIL import Image
import numpy
import gradio as gr

from modules import scripts_postprocessing, shared
from modules.processing import setup_color_correction, apply_color_correction

from modules.ui_components import FormRow, ToolButton, InputAccordion
from modules.ui import switch_values_symbol

upscale_cache = {}


def limit_size_by_one_dimension(w, h, upscale, limit):
    w *= upscale
    h *= upscale
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w > limit:
        h = limit * h // w
        w = limit

    return int(w), int(h)


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def ui(self):
        selected_tab = gr.State(value=0)

        with InputAccordion(True, label="Upscale", elem_id="extras_upscale") as upscale_enabled:
            with FormRow():
                extras_upscaler_1 = gr.Dropdown(label='Upscaler 1', elem_id="extras_upscaler_1", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
                extras_upscaler_2 = gr.Dropdown(label='Upscaler 2', elem_id="extras_upscaler_2", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)

            with FormRow():
                extras_color_correction = gr.Checkbox(label="Color correction", elem_id="extras_color_correction", value=False)
                extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=0.0, elem_id="extras_upscaler_2_visibility")

            with FormRow():
                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.TabItem('Scale by', id="extras_scale_by_tab", elem_id="extras_scale_by_tab") as tab_scale_by:
                        with gr.Row():
                            with gr.Column(scale=4):
                                upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4, elem_id="extras_upscaling_resize")
                            with gr.Column(scale=1, min_width=160):
                                max_side_length = gr.Number(label="Max side length", value=0, elem_id="extras_upscale_max_side_length", tooltip="If any of two sides of the image ends up larger than specified, will downscale it to fit. 0 = no limit.", min_width=160, step=8, minimum=0)

                    with gr.TabItem('Scale to', id="extras_scale_to_tab", elem_id="extras_scale_to_tab") as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id="upscaling_column_size", scale=4):
                                upscaling_resize_w = gr.Slider(minimum=64, maximum=8192, step=8, label="Width", value=512, elem_id="extras_upscaling_resize_w")
                                upscaling_resize_h = gr.Slider(minimum=64, maximum=8192, step=8, label="Height", value=512, elem_id="extras_upscaling_resize_h")
                            with gr.Column(elem_id="upscaling_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                upscaling_res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="upscaling_res_switch_btn", tooltip="Switch width/height")
                                upscaling_crop = gr.Checkbox(label='Crop to fit', value=True, elem_id="extras_upscaling_crop")

        upscaling_res_switch_btn.click(lambda w, h: (h, w), inputs=[upscaling_resize_w, upscaling_resize_h], outputs=[upscaling_resize_w, upscaling_resize_h], show_progress=False)
        tab_scale_by.select(fn=lambda: 0, inputs=None, outputs=[selected_tab])
        tab_scale_to.select(fn=lambda: 1, inputs=None, outputs=[selected_tab])

        return {
            "upscale_enabled": upscale_enabled,
            "upscale_mode": selected_tab,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_cc": extras_color_correction,
        }

    def upscale(self, image, upscaler, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop):
        if upscale_mode == 1:
            upscale_by = max(upscale_to_width/image.width, upscale_to_height/image.height)
        else:
            if max_side_length != 0 and max(*image.size)*upscale_by > max_side_length:
                upscale_mode = 1
                upscale_crop = False
                upscale_to_width, upscale_to_height = limit_size_by_one_dimention(image.width*upscale_by, image.height*upscale_by, max_side_length)
                upscale_by = max(upscale_to_width/image.width, upscale_to_height/image.height)

        cache_key = (hash(numpy.array(image.getdata()).tobytes()), upscaler.name, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
        cached_image = upscale_cache.pop(cache_key, None)

        if cached_image is not None:
            image = cached_image
        else:
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        upscale_cache[cache_key] = image
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        if upscale_mode == 1 and upscale_crop:
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
            image = cropped

        return image

    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, upscale_enabled=True, upscale_mode=1, upscale_by=2.0, max_side_length=0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0, upscale_cc=False,):
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        else:
            pp.shared.target_width, pp.shared.target_height = limit_size_by_one_dimension(pp.image.width, pp.image.height, upscale_by, max_side_length)

        if upscale_cc:
            upscale_cache["cc"] = setup_color_correction(pp.image)

    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscale_enabled=True, upscale_mode=1, upscale_by=2.0, max_side_length=0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0, upscale_cc=False):

        if not upscale_enabled or upscaler_1_name == "None":
            return

        # print (upscaler_1_name, upscaler_2_name, [x.name for x in shared.sd_upscalers])
        for x in shared.sd_upscalers:
            if upscaler_1_name == x.name:
                upscaler_1 = x
                break

        upscaled_image = self.upscale(pp.image, upscaler_1, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)
        info = 'Upscaler: ' + upscaler_1.name

        if upscaler_2_name != "None" and upscaler_2_visibility > 0:
            for x in shared.sd_upscalers:
                if upscaler_2_name == x.name:
                    upscaler_2 = x
                    break

            second_upscale = self.upscale(pp.image, upscaler_2, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)

            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)

            info += f' {upscaler_2.name} ({str(upscaler_2_visibility)})'

        if upscale_cc and "cc" in upscale_cache:  # postprocess during txt2img
            pp.image = apply_color_correction(upscale_cache["cc"], upscaled_image)
            info += ' [cc]'
        else:
            pp.image = upscaled_image

        pp.info['PostProcess'] = info

    def image_changed(self):
        upscale_cache.clear()
