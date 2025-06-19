from PIL import Image
import gradio as gr

from modules import scripts_postprocessing
from modules.ui_components import InputAccordion


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 500     # should run first - before any upscale, face restore

    def ui(self):
        with InputAccordion(False, label="Rotate", elem_id="extras_rotate") as rotate_enabled:
            rotate = gr.Radio(label='Rotation (clockwise)', elem_id="extras_rotation", choices=["None", "90", "180", "270"], value="None")
            flip = gr.Radio(label='Flip', elem_id="extras_flip", choices=["None", "Horizontal", "Vertical"], value="None")

        return {
            "rotate_enabled": rotate_enabled,
            "rotate": rotate,
            "flip": flip,
        }


    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, rotate_enabled=False, rotate="None", flip="None",):
        if rotate_enabled and (rotate != "None" or flip != "None"):
            match rotate:
                case "90":
                    pp.image = pp.image.transpose(Image.ROTATE_270)
                case "180":
                    pp.image = pp.image.transpose(Image.ROTATE_180)
                case "270":
                    pp.image = pp.image.transpose(Image.ROTATE_90)
                case _:
                    pass

            match flip:
                case "Horizontal":
                    pp.image = pp.image.transpose(Image.FLIP_LEFT_RIGHT)
                case "Vertical":
                    pp.image = pp.image.transpose(Image.FLIP_TOP_BOTTOM)
                case _:
                    pass

            if rotate != "None":
                pp.info['Rotate'] = rotate
            if flip != "None":
                pp.info['Flip'] = flip
