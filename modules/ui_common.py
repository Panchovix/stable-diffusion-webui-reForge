import dataclasses
import json
import html
import os

import gradio as gr

from modules import shared, ui_tempdir, util
import modules.images
from modules.ui_components import ToolButton
import modules.infotext_utils as parameters_copypaste

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄


def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, gr.update()
        return plaintext_to_html(generation_info["infotexts"][img_index]), gr.update()
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info, gr.update()


def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


@dataclasses.dataclass
class OutputPanel:
    gallery = None
    generation_info = None
    infotext = None
    html_log = None
    button_upscale = None


def create_output_panel(tabname, outdir, toprow=None):
    res = OutputPanel()

    def open_folder(f, images=None, index=None):
        if shared.cmd_opts.hide_ui_dir_config:
            return

        try:
            if 'Sub' in shared.opts.open_dir_button_choice:
                image_dir = os.path.split(images[index]["name"].rsplit('?', 1)[0])[0]
                if 'temp' in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                    f = image_dir
        except Exception:
            pass

        util.open_folder(f)

    with gr.Column(elem_id=f"{tabname}_results"):
        if toprow:
            toprow.create_inline_toprow_image()

        with gr.Column(variant='panel', elem_id=f"{tabname}_results_panel"):
            with gr.Group(elem_id=f"{tabname}_gallery_container"):
                res.gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery", columns=4, preview=True, height=shared.opts.gallery_height or None, interactive=False, type="pil", object_fit="contain")

            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                open_folder_button = ToolButton(folder_symbol, elem_id=f'{tabname}_open_folder', visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open images output directory.")

                buttons = {
                    'img2img': ToolButton('🖼️', elem_id=f'{tabname}_send_to_img2img', tooltip="Send image and generation parameters to img2img tab."),
                    'inpaint': ToolButton('🎨️', elem_id=f'{tabname}_send_to_inpaint', tooltip="Send image and generation parameters to img2img inpaint tab."),
                    'extras': ToolButton('📐', elem_id=f'{tabname}_send_to_extras', tooltip="Send image and generation parameters to extras tab.")
                }

                if tabname == 'txt2img':
                    res.button_upscale = ToolButton('✨', elem_id=f'{tabname}_upscale', tooltip="Create an upscaled version of the current image using hires fix settings.")

            open_folder_button.click(
                fn=lambda images, index: open_folder(shared.opts.outdir_samples or outdir, images, index),
                _js="(y, w) => [y, selected_gallery_index()]",
                inputs=[
                    res.gallery,
                    open_folder_button,  # placeholder for index
                ],
                outputs=None,
            )

            if tabname != "extras":
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')

                with gr.Group():
                    res.infotext = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
                    res.html_log = gr.HTML(elem_id=f'html_log_{tabname}', elem_classes="html-log")

                    res.generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                    if tabname == 'txt2img' or tabname == 'img2img':
                        generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                        generation_info_button.click(
                            fn=update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[res.generation_info, res.infotext, res.infotext],
                            outputs=[res.infotext, res.infotext],
                            show_progress=False,
                        )

            else:
                res.generation_info = gr.HTML(elem_id=f'html_info_x_{tabname}')
                res.infotext = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
                res.html_log = gr.HTML(elem_id=f'html_log_{tabname}')

            paste_field_names = []
            if tabname == "txt2img":
                paste_field_names = modules.scripts.scripts_txt2img.paste_field_names
            elif tabname == "img2img":
                paste_field_names = modules.scripts.scripts_img2img.paste_field_names

            for paste_tabname, paste_button in buttons.items():
                parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                    paste_button=paste_button, tabname=paste_tabname, source_tabname="txt2img" if tabname == "txt2img" else None, source_image_component=res.gallery,
                    paste_field_names=paste_field_names
                ))

    return res


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(
        fn=refresh,
        inputs=None,
        outputs=refresh_components
    )
    return refresh_button


def setup_dialog(button_show, dialog, *, button_close=None):
    """Sets up the UI so that the dialog (gr.Box) is invisible, and is only shown when buttons_show is clicked, in a fullscreen modal window."""

    dialog.visible = False

    button_show.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[dialog],
    ).then(fn=None, _js="function(){ popupId('" + dialog.elem_id + "'); }")

    if button_close:
        button_close.click(fn=None, _js="closePopup")

