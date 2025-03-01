import gradio as gr

import modules.infotext_utils as parameters_copypaste
from modules.call_queue import wrap_gradio_call_no_job
from modules.core_ui.components import ResizeHandleRow
from modules.extras import run_pnginfo


def create_interface():
    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with ResizeHandleRow(equal_height=False):
            with gr.Column(variant="panel"):
                image = gr.Image(
                    elem_id="pnginfo_image",
                    label="Source",
                    sources="upload",
                    interactive=True,
                    height="50vh",
                    type="pil",
                    image_mode="RGBA",
                )

            with gr.Column(variant="panel"):
                html = gr.HTML()
                generation_info = gr.Textbox(
                    visible=False, elem_id="pnginfo_generation_info"
                )
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(
                        ["txt2img", "img2img", "inpaint", "extras"]
                    )

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(
                        parameters_copypaste.ParamBinding(
                            paste_button=button,
                            tabname=tabname,
                            source_text_component=generation_info,
                            source_image_component=image,
                        )
                    )

        image.change(
            fn=wrap_gradio_call_no_job(run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )
    return pnginfo_interface
