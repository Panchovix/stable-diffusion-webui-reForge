import gradio as gr

from modules import scripts
from ldm_patched.contrib.nodes_sure_ag import SureAttentionGuidance


opSureAG = SureAttentionGuidance()


class SureAGForForge(scripts.Script):
    sorting_priority = 12.6

    def title(self):
        return "SURE Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            with gr.Row():
                alpha = gr.Slider(
                    label="Alpha",
                    minimum=0.001, maximum=0.49, step=0.001, value=0.05,
                    info="Step size. Auto-clamped to 1/(2*(1+attn_weight)).",
                )
                attn_weight = gr.Slider(
                    label="Attention Weight",
                    minimum=0.0, maximum=4.0, step=0.05, value=1.0,
                    info="Entropy amplification. 0 = uniform SURE.",
                )
            with gr.Row():
                attn_blocks = gr.Radio(
                    label="Attention Blocks",
                    choices=["all", "middle", "mid+out", "input", "output"],
                    value="middle",
                    info="Which UNet blocks to capture entropy from. 'middle' is cheapest.",
                )
                approx_coeff = gr.Slider(
                    label="Gradient Coeff",
                    minimum=0.5, maximum=4.0, step=0.1, value=2.0,
                    info="Gradient approximation scale (2.0 = theory).",
                )

        return enabled, alpha, attn_weight, attn_blocks, approx_coeff

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, alpha, attn_weight, attn_blocks, approx_coeff = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        unet = opSureAG.patch(unet, alpha, attn_weight, attn_blocks, approx_coeff)[0]
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            sure_ag_enabled=True,
            sure_ag_alpha=alpha,
            sure_ag_attn_weight=attn_weight,
            sure_ag_attn_blocks=attn_blocks,
            sure_ag_approx_coeff=approx_coeff,
        ))
