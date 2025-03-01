import datetime
import mimetypes
import os
import sys
import warnings

import gradio as gr
import gradio.analytics
import gradio.utils

from modules import gradio_extensions
from modules import (
    script_callbacks,
    ui_extensions,
    ui_loadsave,
    ui_settings,
    timer,
    sysinfo,
    ui_checkpoint_merger,
    scripts,
    launch_utils,
)
from modules.core_ui import common_elements
from modules.paths import script_path
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.infotext_utils as parameters_copypaste
import modules.shared as shared
from modules_forge.forge_canvas.canvas import canvas_head

from modules.core_ui.txt2img_block import create_interface as create_txt2img_interface
from modules.core_ui.img2img_block import create_interface as create_img2img_interface
from modules.core_ui.extras_block import create_interface as create_extras_interface
from modules.core_ui.pnginfo_block import create_interface as create_pnginfo_interface
from modules.core_ui.train_block import create_interface as create_train_interface

create_setting_component = ui_settings.create_setting_component

warnings.filterwarnings(
    "default" if opts.show_warnings else "ignore", category=UserWarning
)
warnings.filterwarnings(
    "default" if opts.show_gradio_deprecation_warnings else "ignore",
    category=gradio_extensions.GradioDeprecationWarning,
)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/avif", ".avif")

# override potentially incorrect mimetypes
mimetypes.add_type("text/css", ".css")

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.analytics.version_check = lambda: None
    # TODO: Check if needed?
    # gradio.utils.get_local_ip_address = lambda: "127.0.0.1"

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok

    print("ngrok authtoken detected, trying to connect...")
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options,
    )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.


plaintext_to_html = common_elements.plaintext_to_html


# def send_gradio_gallery_to_image(x):
#     if len(x) == 0:
#         return None
#     return image_from_url_text(x[0])


# def connect_clear_prompt(button):
#     """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
#     button.click(
#         js="clear_prompt",
#         fn=None,
#         inputs=[],
#         outputs=[],
#     )


# def apply_setting(key: str, value: Any):
#     if value is None:
#         return gr.update()

#     if shared.cmd_opts.freeze_settings:
#         return gr.update()

#     # dont allow model to be swapped when model hash exists in prompt
#     if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
#         return gr.update()

#     if key == "sd_model_checkpoint":
#         ckpt_info = sd_models.get_closet_checkpoint_match(value)

#         if ckpt_info is not None:
#             value = ckpt_info.title
#         else:
#             return gr.update()

#     comp_args = opts.data_labels[key].component_args
#     if comp_args and isinstance(comp_args, dict) and comp_args.get("visible") is False:
#         return

#     valtype = type(opts.data_labels[key].default)
#     oldval = opts.data.get(key, None)
#     opts.data[key] = valtype(value) if valtype != type(None) else value
#     if oldval != value and opts.data_labels[key].onchange is not None:
#         opts.data_labels[key].onchange()

#     opts.save(shared.config_filename)
#     return getattr(opts, key)


def create_ui():
    reload_javascript()

    parameters_copypaste.reset()

    settings = ui_settings.UiSettings()
    settings.register_settings()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    txt2img_interface, txt2img_preview_params, dummy_component = (
        create_txt2img_interface()
    )

    scripts.scripts_current = scripts.scripts_img2img
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    img2img_interface, image_cfg_scale = create_img2img_interface()

    scripts.scripts_current = None

    extras_interface = create_extras_interface()

    pnginfo_interface = create_pnginfo_interface()

    modelmerger_ui = ui_checkpoint_merger.UiCheckpointMerger()

    train_interface = create_train_interface(txt2img_preview_params)

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()

    settings.create_ui(loadsave, dummy_component)

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "train"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings.interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(
        theme=shared.gradio_theme,
        analytics_enabled=False,
        title="Stable Diffusion",
        head=canvas_head,
    ) as demo:
        settings.add_quicksettings()

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(
                interfaces, key=lambda x: tab_order.get(x[1], 9999)
            )

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    if interface is None:
                        raise Exception(f"interface is None for {label}")
                    interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            loadsave.setup_ui()

        if (
            os.path.exists(os.path.join(script_path, "notification.mp3"))
            and shared.opts.notification_audio
        ):
            gr.Audio(
                interactive=False,
                value=os.path.join(script_path, "notification.mp3"),
                elem_id="audio_notification",
                visible=False,
            )

        footer = shared.html("footer.html")
        footer = footer.format(
            versions=versions_html(),
            api_docs="/docs"
            if shared.cmd_opts.api
            else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API",
        )
        gr.HTML(footer, elem_id="footer")

        settings.add_functionality(demo)

        update_image_cfg_scale_visibility = lambda: gr.update(
            visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit"
        )
        settings.text_settings.change(
            fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale]
        )
        demo.load(
            fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale]
        )

        modelmerger_ui.setup_ui(
            dummy_component=dummy_component,
            sd_model_checkpoint_component=settings.component_dict[
                "sd_model_checkpoint"
            ],
        )

    if ui_settings_from_file != loadsave.ui_settings:
        loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers

        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/Panchovix/stable-diffusion-webui-reForge/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, "__long_version__", torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [
            QuicksettingsHint(name=k, label=v.label)
            for k, v in opts.data_labels.items()
        ]

    app.add_api_route(
        "/internal/quicksettings-hint",
        quicksettings_hint,
        methods=["GET"],
        response_model=list[QuicksettingsHint],
    )

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route(
        "/internal/profile-startup", lambda: timer.startup_record, methods=["GET"]
    )

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = (
            f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"
        )

        return PlainTextResponse(
            text,
            headers={
                "Content-Disposition": f'{"attachment" if attachment else "inline"}; filename="{filename}"'
            },
        )

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route(
        "/internal/sysinfo-download",
        lambda: download_sysinfo(attachment=True),
        methods=["GET"],
    )

    import fastapi.staticfiles

    app.mount(
        "/webui-assets",
        fastapi.staticfiles.StaticFiles(
            directory=launch_utils.repo_dir("stable-diffusion-webui-assets")
        ),
        name="webui-assets",
    )
