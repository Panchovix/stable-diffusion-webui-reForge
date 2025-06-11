import gradio as gr

from modules import scripts, shared, sd_models, sd_samplers
from modules.infotext_utils import PasteField
from modules.ui_common import create_refresh_button
from modules.ui_components import InputAccordion


class ScriptRefiner(scripts.ScriptBuiltinUI):
    section = "accordions"
    create_group = False

    def __init__(self):
        pass

    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Refiner", elem_id=self.elem_id("enable")) as enable_refiner:
            with gr.Row():
                refiner_checkpoint = gr.Dropdown(label='Checkpoint', info='(use model of same architecture)', elem_id=self.elem_id("checkpoint"), choices=["", *sd_models.checkpoint_tiles()], value='', tooltip="switch to another model in the middle of generation")
                create_refresh_button(refiner_checkpoint, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, self.elem_id("checkpoint_refresh"))
        
                refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01, elem_id=self.elem_id("switch_at"), tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")

            with gr.Row():
                refiner_cfg = gr.Slider(label='Refiner CFG', elem_id="refiner_cfg", value=0, minimum=0, maximum=16, step=0.1)


        def lookup_checkpoint(title):
            info = sd_models.get_closet_checkpoint_match(title)
            return None if info is None else info.name
        
        self.infotext_fields = [
            PasteField(enable_refiner, lambda d: 'Refiner' in d),
            PasteField(refiner_checkpoint, lambda d: lookup_checkpoint(d.get('Refiner')), api="refiner_checkpoint"),
            PasteField(refiner_switch_at, 'Refiner switch at', api="refiner_switch_at"),
            PasteField(refiner_cfg, 'Refiner CFG', api="refiner_cfg"),
        ]

        return enable_refiner, refiner_checkpoint, refiner_switch_at, refiner_cfg

    def setup(self, p, enable_refiner, refiner_checkpoint, refiner_switch_at, refiner_cfg):
        # the actual implementation is in sd_samplers_common.py, apply_refiner
        if not enable_refiner or refiner_checkpoint in (None, "", "None"):
            p.refiner_checkpoint = None
            p.refiner_switch_at = None
            p.refiner_cfg = None
        else:
            p.refiner_checkpoint = refiner_checkpoint
            p.refiner_switch_at = refiner_switch_at
            p.refiner_cfg = refiner_cfg
