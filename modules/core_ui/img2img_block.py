import os
from contextlib import ExitStack
from pathlib import Path

import gradio as gr
from PIL import Image

import modules.infotext_utils as parameters_copypaste
import modules.shared as shared
from modules import deepbooru, img2img, progress, scripts, ui_extra_networks
from modules.call_queue import (
    wrap_gradio_gpu_call,
    wrap_queued_call,
)
from modules.core_ui.common_elements import (
    create_output_panel,
    create_override_settings_dropdown,
    ordered_ui_categories,
)
from modules.core_ui.components import (
    FormGroup,
    FormHTML,
    FormRow,
    ResizeHandleRow,
    ToolButton,
)
from modules.core_ui.token_counters import (
    update_negative_prompt_token_counter,
    update_token_counter,
)
from modules.core_ui.toolbutton_symbols import (
    detect_image_size_symbol,
    switch_values_symbol,
)
from modules.core_ui.toprow import Toprow
from modules.shared import opts
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head


def process_interrogate(
    interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles
):
    mode = int(mode)
    if mode in (0, 1, 3, 4):
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, (
            "Launched with --hide-ui-dir-config, batch img2img disabled"
        )
        images = Path(ii_input_dir).iterdir()
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(
                interrogation_function(img),
                file=open(
                    os.path.join(ii_output_dir, f"{left}.txt"), "a", encoding="utf-8"
                ),
            )

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def resize_from_to_html(width, height, scale_by):
    if not scale_by:
        return "Resize: Invalid Scale"
    target_width = int(float(width) * scale_by)
    target_height = int(float(height) * scale_by)

    if not target_width or not target_height:
        return "Resize: Zero Target Width / Height"

    target_width -= target_width % 8  #   note: hardcoded latent size 8
    target_height -= target_height % 8

    return f"Resize from: <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def create_interface():
    with gr.Blocks(analytics_enabled=False, head=canvas_head) as img2img_interface:
        dummy_component = gr.Textbox(visible=False)

        toprow = Toprow(is_img2img=True, is_compact=shared.opts.compact_prompt_box)

        extra_tabs = gr.Tabs(
            elem_id="img2img_extra_tabs", elem_classes=["extra-networks"]
        )
        extra_tabs.__enter__()

        with (
            gr.Tab("Generation", id="img2img_generation") as img2img_generation_tab,
            ResizeHandleRow(equal_height=False),
        ):
            with ExitStack() as stack:
                if shared.opts.img2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(
                    gr.Column(variant="compact", elem_id="img2img_settings")
                )

                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(
                        variant="compact", elem_id=f"img2img_copy_to_{tab_name}"
                    ):
                        gr.HTML(
                            "Copy image to: ",
                            elem_id=f"img2img_label_copy_to_{tab_name}",
                        )

                        for title, name in zip(
                            ["img2img", "sketch", "inpaint", "inpaint sketch"],
                            ["img2img", "sketch", "inpaint", "inpaint_sketch"],
                        ):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                scripts.scripts_img2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    if category == "image":
                        with gr.Tabs(elem_id="mode_img2img"):
                            img2img_selected_tab = gr.Number(value=0, visible=False)

                            with gr.TabItem(
                                "img2img", id="img2img", elem_id="img2img_img2img_tab"
                            ) as tab_img2img:
                                init_img = ForgeCanvas(
                                    elem_id="img2img_image",
                                    height=512,
                                    no_scribbles=True,
                                )
                                add_copy_image_controls("img2img", init_img)

                            with gr.TabItem(
                                "Sketch",
                                id="img2img_sketch",
                                elem_id="img2img_img2img_sketch_tab",
                            ) as tab_sketch:
                                sketch = ForgeCanvas(
                                    elem_id="img2img_sketch",
                                    height=512,
                                    scribble_color=opts.img2img_sketch_default_brush_color,
                                )
                                add_copy_image_controls("sketch", sketch)

                            with gr.TabItem(
                                "Inpaint", id="inpaint", elem_id="img2img_inpaint_tab"
                            ) as tab_inpaint:
                                init_img_with_mask = ForgeCanvas(
                                    elem_id="img2maskimg",
                                    height=512,
                                    contrast_scribbles=opts.img2img_inpaint_mask_high_contrast,
                                    scribble_color=opts.img2img_inpaint_mask_brush_color,
                                    scribble_color_fixed=True,
                                    scribble_alpha=opts.img2img_inpaint_mask_scribble_alpha,
                                    scribble_alpha_fixed=True,
                                    scribble_softness_fixed=True,
                                )
                                add_copy_image_controls("inpaint", init_img_with_mask)

                            with gr.TabItem(
                                "Inpaint sketch",
                                id="inpaint_sketch",
                                elem_id="img2img_inpaint_sketch_tab",
                            ) as tab_inpaint_color:
                                inpaint_color_sketch = ForgeCanvas(
                                    elem_id="inpaint_sketch",
                                    height=512,
                                    scribble_color=opts.img2img_inpaint_sketch_default_brush_color,
                                )
                                add_copy_image_controls(
                                    "inpaint_sketch", inpaint_color_sketch
                                )

                            with gr.TabItem(
                                "Inpaint upload",
                                id="inpaint_upload",
                                elem_id="img2img_inpaint_upload_tab",
                            ) as tab_inpaint_upload:
                                init_img_inpaint = gr.Image(
                                    label="Image for img2img",
                                    show_label=False,
                                    sources="upload",
                                    interactive=True,
                                    type="pil",
                                    elem_id="img_inpaint_base",
                                )
                                init_mask_inpaint = gr.Image(
                                    label="Mask",
                                    sources="upload",
                                    interactive=True,
                                    type="pil",
                                    image_mode="RGBA",
                                    elem_id="img_inpaint_mask",
                                )

                            with gr.TabItem(
                                "Batch", id="batch", elem_id="img2img_batch_tab"
                            ) as tab_batch:
                                with gr.Tabs(elem_id="img2img_batch_source"):
                                    img2img_batch_source_type = gr.Textbox(
                                        visible=False, value="upload"
                                    )
                                    with gr.TabItem(
                                        "Upload",
                                        id="batch_upload",
                                        elem_id="img2img_batch_upload_tab",
                                    ) as tab_batch_upload:
                                        img2img_batch_upload = gr.Files(
                                            label="Files",
                                            interactive=True,
                                            elem_id="img2img_batch_upload",
                                        )
                                    with gr.TabItem(
                                        "From directory",
                                        id="batch_from_dir",
                                        elem_id="img2img_batch_from_dir_tab",
                                    ) as tab_batch_from_dir:
                                        hidden = (
                                            "<br>Disabled when launched with --hide-ui-dir-config."
                                            if shared.cmd_opts.hide_ui_dir_config
                                            else ""
                                        )
                                        gr.HTML(
                                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running."
                                            + "<br>Use an empty output directory to save pictures normally instead of writing to the output directory."
                                            + f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
                                            f"{hidden}</p>"
                                        )
                                        img2img_batch_input_dir = gr.Textbox(
                                            label="Input directory",
                                            **shared.hide_dirs,
                                            elem_id="img2img_batch_input_dir",
                                        )
                                        img2img_batch_output_dir = gr.Textbox(
                                            label="Output directory",
                                            **shared.hide_dirs,
                                            elem_id="img2img_batch_output_dir",
                                        )
                                        img2img_batch_inpaint_mask_dir = gr.Textbox(
                                            label="Inpaint batch mask directory (required for inpaint batch processing only)",
                                            **shared.hide_dirs,
                                            elem_id="img2img_batch_inpaint_mask_dir",
                                        )
                                tab_batch_upload.select(
                                    fn=lambda: "upload",
                                    inputs=[],
                                    outputs=[img2img_batch_source_type],
                                )
                                tab_batch_from_dir.select(
                                    fn=lambda: "from dir",
                                    inputs=[],
                                    outputs=[img2img_batch_source_type],
                                )
                                with gr.Accordion("PNG info", open=False):
                                    img2img_batch_use_png_info = gr.Checkbox(
                                        label="Append png info to prompts",
                                        elem_id="img2img_batch_use_png_info",
                                    )
                                    img2img_batch_png_info_dir = gr.Textbox(
                                        label="PNG info directory",
                                        **shared.hide_dirs,
                                        placeholder="Leave empty to use input directory",
                                        elem_id="img2img_batch_png_info_dir",
                                    )
                                    img2img_batch_png_info_props = gr.CheckboxGroup(
                                        [
                                            "Prompt",
                                            "Negative prompt",
                                            "Seed",
                                            "CFG scale",
                                            "Sampler",
                                            "Steps",
                                            "Model hash",
                                        ],
                                        label="Parameters to take from png info",
                                        info="Prompts from png info will be appended to prompts set in ui.",
                                    )

                            img2img_tabs = [
                                tab_img2img,
                                tab_sketch,
                                tab_inpaint,
                                tab_inpaint_color,
                                tab_inpaint_upload,
                                tab_batch,
                            ]

                            for i, tab in enumerate(img2img_tabs):
                                tab.select(
                                    fn=lambda tabnum=i: tabnum,
                                    inputs=[],
                                    outputs=[img2img_selected_tab],
                                )

                        def copy_image(img):
                            if isinstance(img, dict) and "image" in img:
                                return img["image"]

                            return img

                        def copyCanvas_img2img(background, foreground, source):
                            if (
                                source == 1 or source == 3
                            ):  #   1 is sketch, 3 is Inpaint sketch
                                bg = Image.alpha_composite(background, foreground)
                                return bg, None
                            return background, None

                        for button, name, elem in copy_image_buttons:
                            button.click(
                                fn=copyCanvas_img2img,
                                inputs=[
                                    elem.background,
                                    elem.foreground,
                                    img2img_selected_tab,
                                ],
                                outputs=[
                                    copy_image_destinations[name].background,
                                    copy_image_destinations[name].foreground,
                                ],
                            )
                            button.click(
                                fn=None,
                                js=f"switch_to_{name.replace(' ', '_')}",
                                inputs=[],
                                outputs=[],
                            )

                        with FormRow():
                            resize_mode = gr.Radio(
                                label="Resize mode",
                                elem_id="resize_mode",
                                choices=[
                                    "Just resize",
                                    "Crop and resize",
                                    "Resize and fill",
                                    "Just resize (latent upscale)",
                                ],
                                type="index",
                                value="Just resize",
                            )

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.Number(value=0, visible=False)

                                with gr.Tabs(elem_id="img2img_tabs_resize"):
                                    with gr.Tab(
                                        label="Resize to",
                                        id="to",
                                        elem_id="img2img_tab_resize_to",
                                    ) as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(
                                                elem_id="img2img_column_size", scale=4
                                            ):
                                                width = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Width",
                                                    value=512,
                                                    elem_id="img2img_width",
                                                )
                                                height = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Height",
                                                    value=512,
                                                    elem_id="img2img_height",
                                                )
                                            with gr.Column(
                                                elem_id="img2img_dimensions_row",
                                                scale=1,
                                                elem_classes="dimensions-tools",
                                            ):
                                                res_switch_btn = ToolButton(
                                                    value=switch_values_symbol,
                                                    elem_id="img2img_res_switch_btn",
                                                    tooltip="Switch width/height",
                                                )
                                                detect_image_size_btn = ToolButton(
                                                    value=detect_image_size_symbol,
                                                    elem_id="img2img_detect_image_size_btn",
                                                    tooltip="Auto detect size from img2img",
                                                )

                                    with gr.Tab(
                                        label="Resize by",
                                        id="by",
                                        elem_id="img2img_tab_resize_by",
                                    ) as tab_scale_by:
                                        scale_by = gr.Slider(
                                            minimum=0.05,
                                            maximum=4.0,
                                            step=0.01,
                                            label="Scale",
                                            value=1.0,
                                            elem_id="img2img_scale",
                                        )

                                        with FormRow():
                                            scale_by_html = FormHTML(
                                                resize_from_to_html(0, 0, 0.0),
                                                elem_id="img2img_scale_resolution_preview",
                                            )
                                            gr.Slider(
                                                label="Unused",
                                                elem_id="img2img_unused_scale_by_slider",
                                            )
                                            button_update_resize_to = gr.Button(
                                                visible=False,
                                                elem_id="img2img_update_resize_to",
                                            )

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        js="currentImg2imgSourceResolution",
                                        inputs=[
                                            dummy_component,
                                            dummy_component,
                                            scale_by,
                                        ],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.change(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                                    def updateWH(img, w, h):
                                        if img and shared.opts.img2img_autosize:
                                            return img.size[0], img.size[1]
                                        else:
                                            return w, h

                                    img_sources = [
                                        init_img.background,
                                        sketch.background,
                                        init_img_with_mask.background,
                                        inpaint_color_sketch.background,
                                        init_img_inpaint,
                                    ]
                                    for i in img_sources:
                                        i.change(
                                            fn=updateWH,
                                            inputs=[i, width, height],
                                            outputs=[width, height],
                                            show_progress="hidden",
                                        )
                                        i.change(**on_change_args)

                            tab_scale_to.select(
                                fn=lambda: 0, inputs=[], outputs=[selected_scale_tab]
                            )
                            tab_scale_by.select(
                                fn=lambda: 1, inputs=[], outputs=[selected_scale_tab]
                            )

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="img2img_column_batch"):
                                    batch_count = gr.Slider(
                                        minimum=1,
                                        step=1,
                                        label="Batch count",
                                        value=1,
                                        elem_id="img2img_batch_count",
                                    )
                                    batch_size = gr.Slider(
                                        minimum=1,
                                        maximum=8,
                                        step=1,
                                        label="Batch size",
                                        value=1,
                                        elem_id="img2img_batch_size",
                                    )

                    elif category == "denoising":
                        denoising_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            label="Denoising strength",
                            value=0.75,
                            elem_id="img2img_denoising_strength",
                        )

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=30.0,
                                step=0.5,
                                label="CFG Scale",
                                value=7.0,
                                elem_id="img2img_cfg_scale",
                            )
                            image_cfg_scale = gr.Slider(
                                minimum=0,
                                maximum=3.0,
                                step=0.05,
                                label="Image CFG Scale",
                                value=1.5,
                                elem_id="img2img_image_cfg_scale",
                                visible=False,
                            )

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(
                            elem_id="img2img_accordions", elem_classes="accordions"
                        ):
                            scripts.scripts_img2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(
                                    minimum=1,
                                    step=1,
                                    label="Batch count",
                                    value=1,
                                    elem_id="img2img_batch_count",
                                )
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    step=1,
                                    label="Batch size",
                                    value=1,
                                    elem_id="img2img_batch_size",
                                )

                    elif category == "override_settings":
                        with FormRow(elem_id="img2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown(
                                "img2img", row
                            )

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

                    elif category == "inpaint":
                        with FormGroup(
                            elem_id="inpaint_controls", visible=False
                        ) as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(
                                    label="Mask blur",
                                    minimum=0,
                                    maximum=64,
                                    step=1,
                                    value=4,
                                    elem_id="img2img_mask_blur",
                                )
                                mask_alpha = gr.Slider(
                                    label="Mask transparency",
                                    visible=False,
                                    elem_id="img2img_mask_alpha",
                                )

                            with FormRow():
                                inpainting_mask_invert = gr.Radio(
                                    label="Mask mode",
                                    choices=["Inpaint masked", "Inpaint not masked"],
                                    value="Inpaint masked",
                                    type="index",
                                    elem_id="img2img_mask_mode",
                                )

                            with FormRow():
                                inpainting_fill = gr.Radio(
                                    label="Masked content",
                                    choices=[
                                        "fill",
                                        "original",
                                        "latent noise",
                                        "latent nothing",
                                    ],
                                    value="original",
                                    type="index",
                                    elem_id="img2img_inpainting_fill",
                                )

                            with FormRow():
                                with gr.Column():
                                    inpaint_full_res = gr.Radio(
                                        label="Inpaint area",
                                        choices=["Whole picture", "Only masked"],
                                        type="index",
                                        value="Whole picture",
                                        elem_id="img2img_inpaint_full_res",
                                    )

                                with gr.Column(scale=4):
                                    inpaint_full_res_padding = gr.Slider(
                                        label="Only masked padding, pixels",
                                        minimum=0,
                                        maximum=256,
                                        step=4,
                                        value=32,
                                        elem_id="img2img_inpaint_full_res_padding",
                                    )

                    if category not in {"accordions"}:
                        scripts.scripts_img2img.setup_ui_for_section(category)

            def select_img2img_tab(tab):
                return (
                    gr.update(visible=tab in [2, 3, 4]),
                    gr.update(visible=tab == 3),
                )

            for i, elem in enumerate(img2img_tabs):
                elem.select(
                    fn=lambda tab=i: select_img2img_tab(tab),
                    inputs=[],
                    outputs=[inpaint_controls, mask_alpha],
                )

            output_panel = create_output_panel(
                "img2img", opts.outdir_img2img_samples, toprow
            )

            submit_img2img_inputs = [
                dummy_component,
                img2img_selected_tab,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                init_img.background,
                sketch.background,
                sketch.foreground,
                init_img_with_mask.background,
                init_img_with_mask.foreground,
                inpaint_color_sketch.background,
                inpaint_color_sketch.foreground,
                init_img_inpaint,
                init_mask_inpaint,
                mask_blur,
                mask_alpha,
                inpainting_fill,
                batch_count,
                batch_size,
                cfg_scale,
                image_cfg_scale,
                denoising_strength,
                selected_scale_tab,
                height,
                width,
                scale_by,
                resize_mode,
                inpaint_full_res,
                inpaint_full_res_padding,
                inpainting_mask_invert,
                img2img_batch_input_dir,
                img2img_batch_output_dir,
                img2img_batch_inpaint_mask_dir,
                override_settings,
                img2img_batch_use_png_info,
                img2img_batch_png_info_props,
                img2img_batch_png_info_dir,
                img2img_batch_source_type,
                img2img_batch_upload,
            ] + custom_inputs

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(img2img.img2img, extra_outputs=[None, "", ""]),
                js="submit_img2img",
                inputs=submit_img2img_inputs,
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            interrogate_args = dict(
                js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    init_img.background,
                    sketch.background,
                    init_img_with_mask.background,
                    inpaint_color_sketch.background,
                    init_img_inpaint,
                ],
                outputs=[toprow.prompt, dummy_component],
            )

            toprow.prompt.submit(**img2img_args)
            toprow.submit.click(**img2img_args)

            res_switch_btn.click(
                lambda w, h: (h, w),
                inputs=[width, height],
                outputs=[width, height],
                show_progress="hidden",
            )

            detect_image_size_btn.click(
                fn=lambda w, h: (w or gr.update(), h or gr.update()),
                js="currentImg2imgSourceResolution",
                inputs=[dummy_component, dummy_component],
                outputs=[width, height],
                show_progress="hidden",
            )

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                js="restoreProgressImg2img",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress="hidden",
            )

            toprow.button_interrogate.click(
                fn=lambda *args: process_interrogate(interrogate, *args),
                **interrogate_args,
            )

            toprow.button_deepbooru.click(
                fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
                **interrogate_args,
            )

            steps = scripts.scripts_img2img.script("Sampler").steps

            toprow.ui_styles.dropdown.change(
                fn=wrap_queued_call(update_token_counter),
                inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.token_counter],
            )
            toprow.ui_styles.dropdown.change(
                fn=wrap_queued_call(update_negative_prompt_token_counter),
                inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.negative_token_counter],
            )
            toprow.token_button.click(
                fn=update_token_counter,
                inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.token_counter],
            )
            toprow.negative_token_button.click(
                fn=wrap_queued_call(update_negative_prompt_token_counter),
                inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.negative_token_counter],
            )

            img2img_paste_fields = [
                (toprow.prompt, "Prompt"),
                (toprow.negative_prompt, "Negative prompt"),
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (
                    toprow.ui_styles.dropdown,
                    lambda d: d["Styles array"]
                    if isinstance(d.get("Styles array"), list)
                    else gr.update(),
                ),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                (inpainting_mask_invert, "Mask mode"),
                (inpainting_fill, "Masked content"),
                (inpaint_full_res, "Inpaint area"),
                (inpaint_full_res_padding, "Masked area padding"),
                *scripts.scripts_img2img.infotext_fields,
            ]
            parameters_copypaste.add_paste_fields(
                "img2img", init_img.background, img2img_paste_fields, override_settings
            )
            parameters_copypaste.add_paste_fields(
                "inpaint",
                init_img_with_mask.background,
                img2img_paste_fields,
                override_settings,
            )
            parameters_copypaste.register_paste_params_button(
                parameters_copypaste.ParamBinding(
                    paste_button=toprow.paste,
                    tabname="img2img",
                    source_text_component=toprow.prompt,
                    source_image_component=None,
                )
            )

        extra_networks_ui_img2img = ui_extra_networks.create_ui(
            img2img_interface, [img2img_generation_tab], "img2img"
        )
        ui_extra_networks.setup_ui(extra_networks_ui_img2img, output_panel.gallery)

        extra_tabs.__exit__()
    return img2img_interface, image_cfg_scale
