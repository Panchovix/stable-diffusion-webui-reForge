from contextlib import ExitStack

import gradio as gr
from modules.call_queue import (
    wrap_gradio_gpu_call,
    wrap_queued_call,
)

from modules import sd_schedulers
from modules import (
    progress,
    scripts,
    sd_samplers,
    ui_extra_networks,
    sd_models,
    txt2img,
    processing
)

from modules.core_ui.components import (
    FormRow,
    FormGroup,
    ToolButton,
    FormHTML,
    ResizeHandleRow,
    InputAccordion,
)
from modules.core_ui.common_elements import create_refresh_button, create_output_panel, create_override_settings_dropdown,ordered_ui_categories
from modules.core_ui.toolbutton_symbols import switch_values_symbol

from modules.shared import opts

import modules.infotext_utils as parameters_copypaste
import modules.shared as shared
from modules.infotext_utils import PasteField
from modules_forge.forge_canvas.canvas import canvas_head

from modules.core_ui.toprow import Toprow
from modules.core_ui.token_counters import (
    update_negative_prompt_token_counter,
    update_token_counter,
)


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(
        width=width,
        height=height,
        enable_hr=True,
        hr_scale=hr_scale,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
    )
    p.calculate_target_resolution()

    new_width = p.hr_resize_x or p.hr_upscale_to_x
    new_height = p.hr_resize_y or p.hr_upscale_to_y

    new_width -= new_width % 8  #   note: hardcoded latent size 8
    new_height -= new_height % 8

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{new_width}x{new_height}</span>"


def create_accordions():
    category = "accordions"
    with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
        with InputAccordion(
            False, label="Hires. fix", elem_id="txt2img_hr"
        ) as enable_hr:
            with enable_hr.extra():
                hr_final_resolution = FormHTML(
                    value="",
                    elem_id="txtimg_hr_finalres",
                    label="Upscaled resolution",
                    interactive=False,
                    min_width=0,
                )

            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                hr_upscaler = gr.Dropdown(
                    label="Upscaler",
                    elem_id="txt2img_hr_upscaler",
                    choices=[
                        *shared.latent_upscale_modes,
                        *[x.name for x in shared.sd_upscalers],
                    ],
                    value=shared.latent_upscale_default_mode,
                )
                hr_second_pass_steps = gr.Slider(
                    minimum=0,
                    maximum=150,
                    step=1,
                    label="Hires steps",
                    value=0,
                    elem_id="txt2img_hires_steps",
                )
                denoising_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Denoising strength",
                    value=0.7,
                    elem_id="txt2img_denoising_strength",
                )

            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                hr_scale = gr.Slider(
                    minimum=1.0,
                    maximum=4.0,
                    step=0.05,
                    label="Upscale by",
                    value=2.0,
                    elem_id="txt2img_hr_scale",
                )
                hr_resize_x = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    step=8,
                    label="Resize width to",
                    value=0,
                    elem_id="txt2img_hr_resize_x",
                )
                hr_resize_y = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    step=8,
                    label="Resize height to",
                    value=0,
                    elem_id="txt2img_hr_resize_y",
                )

            with FormRow(
                elem_id="txt2img_hires_fix_row_cfg",
                variant="compact",
            ):
                hr_cfg = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    step=0.1,
                    label="Hires CFG Scale",
                    value=7.0,
                    elem_id="txt2img_hr_cfg",
                )

            with FormRow(
                elem_id="txt2img_hires_fix_row3",
                variant="compact",
                visible=opts.hires_fix_show_sampler,
            ) as hr_sampler_container:
                hr_checkpoint_name = gr.Dropdown(
                    label="Checkpoint",
                    elem_id="hr_checkpoint",
                    choices=["Use same checkpoint"]
                    + sd_models.checkpoint_tiles(use_short=True),
                    value="Use same checkpoint",
                )
                create_refresh_button(
                    hr_checkpoint_name,
                    sd_models.list_models,
                    lambda: {
                        "choices": ["Use same checkpoint"]
                        + sd_models.checkpoint_tiles(use_short=True)
                    },
                    "hr_checkpoint_refresh",
                )

                hr_sampler_name = gr.Dropdown(
                    label="Hires sampling method",
                    elem_id="hr_sampler",
                    choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                    value="Use same sampler",
                )
                hr_scheduler = gr.Dropdown(
                    label="Hires schedule type",
                    elem_id="hr_scheduler",
                    choices=["Use same scheduler"]
                    + [x.label for x in sd_schedulers.schedulers],
                    value="Use same scheduler",
                )

            with FormRow(
                elem_id="txt2img_hires_fix_row4",
                variant="compact",
                visible=opts.hires_fix_show_prompts,
            ) as hr_prompts_container:
                with gr.Column(scale=80):
                    with gr.Row():
                        hr_prompt = gr.Textbox(
                            label="Hires prompt",
                            elem_id="hires_prompt",
                            show_label=False,
                            lines=3,
                            placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.",
                            elem_classes=["prompt"],
                        )
                with gr.Column(scale=80):
                    with gr.Row():
                        hr_negative_prompt = gr.Textbox(
                            label="Hires negative prompt",
                            elem_id="hires_neg_prompt",
                            show_label=False,
                            lines=3,
                            placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.",
                            elem_classes=["prompt"],
                        )

            hr_cfg.change(
                lambda x: gr.update(interactive=(x != 1)),
                inputs=[hr_cfg],
                outputs=[hr_negative_prompt],
                queue=False,
                show_progress=False,
            )

        scripts.scripts_txt2img.setup_ui_for_section(category)

    return {
        "enable_hr": enable_hr,
        "denoising_strength": denoising_strength,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_second_pass_steps,
        "hr_resize_x": hr_resize_x,
        "hr_resize_y": hr_resize_y,
        "hr_checkpoint_name": hr_checkpoint_name,
        "hr_sampler_name": hr_sampler_name,
        "hr_scheduler": hr_scheduler,
        "hr_prompt": hr_prompt,
        "hr_negative_prompt": hr_negative_prompt,
        "hr_final_resolution": hr_final_resolution,
        "hr_sampler_container": hr_sampler_container,
        "hr_prompts_container": hr_prompts_container,
        "hr_cfg": hr_cfg,
    }


def create_interface():
    with gr.Blocks(analytics_enabled=False, head=canvas_head) as txt2img_interface:
        toprow = Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

        dummy_component = gr.Textbox(visible=False)

        extra_tabs = gr.Tabs(
            elem_id="txt2img_extra_tabs", elem_classes=["extra-networks"]
        )
        extra_tabs.__enter__()

        with (
            gr.Tab("Generation", id="txt2img_generation") as txt2img_generation_tab,
            ResizeHandleRow(equal_height=False),
        ):
            with ExitStack() as stack:
                if shared.opts.txt2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(
                    gr.Column(variant="compact", elem_id="txt2img_settings")
                )

                scripts.scripts_txt2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(
                                    minimum=64,
                                    maximum=2048,
                                    step=8,
                                    label="Width",
                                    value=512,
                                    elem_id="txt2img_width",
                                )
                                height = gr.Slider(
                                    minimum=64,
                                    maximum=2048,
                                    step=8,
                                    label="Height",
                                    value=512,
                                    elem_id="txt2img_height",
                                )

                            with gr.Column(
                                elem_id="txt2img_dimensions_row",
                                scale=1,
                                elem_classes="dimensions-tools",
                            ):
                                res_switch_btn = ToolButton(
                                    value=switch_values_symbol,
                                    elem_id="txt2img_res_switch_btn",
                                    tooltip="Switch width/height",
                                )

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(
                                        minimum=1,
                                        step=1,
                                        label="Batch count",
                                        value=1,
                                        elem_id="txt2img_batch_count",
                                    )
                                    batch_size = gr.Slider(
                                        minimum=1,
                                        maximum=8,
                                        step=1,
                                        label="Batch size",
                                        value=1,
                                        elem_id="txt2img_batch_size",
                                    )

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=0.0,
                                maximum=30.0,
                                step=0.1,
                                label="CFG Scale",
                                value=7.0,
                                elem_id="txt2img_cfg_scale",
                            )

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        accordion_elements: dict[str, gr.components.Component] = (
                            create_accordions()
                        )

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(
                                    minimum=1,
                                    step=1,
                                    label="Batch count",
                                    value=1,
                                    elem_id="txt2img_batch_count",
                                )
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    step=1,
                                    label="Batch size",
                                    value=1,
                                    elem_id="txt2img_batch_size",
                                )

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown(
                                "txt2img", row
                            )

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = scripts.scripts_txt2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_txt2img.setup_ui_for_section(category)

            # Extract out Hires Upscale elements from dict.
            # A bit ugly to do this, but ensures that typing would be compatible.
            denoising_strength = accordion_elements["denoising_strength"]
            enable_hr = accordion_elements["enable_hr"]
            hr_cfg = accordion_elements["hr_cfg"]
            hr_checkpoint_name = accordion_elements["hr_checkpoint_name"]
            hr_final_resolution = accordion_elements["hr_final_resolution"]
            hr_negative_prompt = accordion_elements["hr_negative_prompt"]
            hr_prompt = accordion_elements["hr_prompt"]
            hr_resize_x = accordion_elements["hr_resize_x"]
            hr_resize_y = accordion_elements["hr_resize_y"]
            hr_sampler_name = accordion_elements["hr_sampler_name"]
            hr_scale = accordion_elements["hr_scale"]
            hr_scheduler = accordion_elements["hr_scheduler"]
            hr_sampler_container = accordion_elements["hr_sampler_container"]
            hr_prompts_container = accordion_elements["hr_prompts_container"]
            hr_second_pass_steps = accordion_elements["hr_second_pass_steps"]
            hr_upscaler = accordion_elements["hr_upscaler"]

            hr_resolution_preview_inputs = [
                enable_hr,
                width,
                height,
                hr_scale,
                hr_resize_x,
                hr_resize_y,
            ]

            for component in hr_resolution_preview_inputs:
                event = (
                    component.release
                    if isinstance(component, gr.Slider)
                    else component.change
                )

                event(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress="hidden",
                )
                event(
                    None,
                    js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress="hidden",
                )

            output_panel = create_output_panel(
                "txt2img", opts.outdir_txt2img_samples, toprow
            )

            if (
                not toprow.ui_styles
                or not toprow.prompt
                or not toprow.negative_prompt
                or not toprow.submit
            ):
                raise Exception("Toprow UI Styles is not ready.")

            txt2img_inputs = [
                dummy_component,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                batch_count,
                batch_size,
                cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_checkpoint_name,
                hr_sampler_name,
                hr_scheduler,
                hr_prompt,
                hr_negative_prompt,
                hr_cfg,
                override_settings,
            ] + custom_inputs

            txt2img_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(txt2img.txt2img, extra_outputs=[None, "", ""]),
                js="submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress="hidden",
            )

            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)

            def select_gallery_image(index):
                index = int(index)
                if getattr(shared.opts, "hires_button_gallery_insert", False):
                    index += 1
                return gr.update(selected_index=index)

            txt2img_upscale_inputs = (
                txt2img_inputs[0:1]
                + [output_panel.gallery, output_panel.generation_info]
                + txt2img_inputs[1:]
            )
            output_panel.button_upscale.click(
                fn=wrap_gradio_gpu_call(
                    txt2img.txt2img_upscale, extra_outputs=[None, "", ""]
                ),
                js="submit_txt2img_upscale",
                inputs=txt2img_upscale_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            ).then(
                fn=select_gallery_image,
                js="selected_gallery_index",
                inputs=[dummy_component],
                outputs=[output_panel.gallery],
            )

            res_switch_btn.click(
                lambda w, h: (h, w),
                inputs=[width, height],
                outputs=[width, height],
                show_progress="hidden",
            )

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                js="restoreProgressTxt2img",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress="hidden",
            )

            txt2img_paste_fields = [
                PasteField(toprow.prompt, "Prompt", api="prompt"),
                PasteField(
                    toprow.negative_prompt, "Negative prompt", api="negative_prompt"
                ),
                PasteField(cfg_scale, "CFG scale", api="cfg_scale"),
                PasteField(width, "Size-1", api="width"),
                PasteField(height, "Size-2", api="height"),
                PasteField(batch_size, "Batch size", api="batch_size"),
                PasteField(
                    toprow.ui_styles.dropdown,
                    lambda d: d["Styles array"]
                    if isinstance(d.get("Styles array"), list)
                    else gr.update(),
                    api="styles",
                ),
                PasteField(
                    denoising_strength, "Denoising strength", api="denoising_strength"
                ),
                PasteField(
                    enable_hr,
                    lambda d: "Denoising strength" in d
                    and (
                        "Hires upscale" in d
                        or "Hires upscaler" in d
                        or "Hires resize-1" in d
                    ),
                    api="enable_hr",
                ),
                PasteField(hr_scale, "Hires upscale", api="hr_scale"),
                PasteField(hr_upscaler, "Hires upscaler", api="hr_upscaler"),
                PasteField(
                    hr_second_pass_steps, "Hires steps", api="hr_second_pass_steps"
                ),
                PasteField(hr_resize_x, "Hires resize-1", api="hr_resize_x"),
                PasteField(hr_resize_y, "Hires resize-2", api="hr_resize_y"),
                PasteField(
                    hr_checkpoint_name, "Hires checkpoint", api="hr_checkpoint_name"
                ),
                PasteField(
                    hr_sampler_name,
                    sd_samplers.get_hr_sampler_from_infotext,
                    api="hr_sampler_name",
                ),
                PasteField(
                    hr_scheduler,
                    sd_samplers.get_hr_scheduler_from_infotext,
                    api="hr_scheduler",
                ),
                PasteField(
                    hr_sampler_container,
                    lambda d: gr.update(visible=True)
                    if d.get("Hires sampler", "Use same sampler") != "Use same sampler"
                    or d.get("Hires checkpoint", "Use same checkpoint")
                    != "Use same checkpoint"
                    or d.get("Hires schedule type", "Use same scheduler")
                    != "Use same scheduler"
                    else gr.update(),
                ),
                PasteField(hr_prompt, "Hires prompt", api="hr_prompt"),
                PasteField(
                    hr_negative_prompt,
                    "Hires negative prompt",
                    api="hr_negative_prompt",
                ),
                PasteField(hr_cfg, "Hires CFG Scale", api="hr_cfg"),
                PasteField(
                    hr_prompts_container,
                    lambda d: gr.update(visible=True)
                    if d.get("Hires prompt", "") != ""
                    or d.get("Hires negative prompt", "") != ""
                    else gr.update(),
                ),
                *scripts.scripts_txt2img.infotext_fields,
            ]
            parameters_copypaste.add_paste_fields(
                "txt2img", None, txt2img_paste_fields, override_settings
            )
            parameters_copypaste.register_paste_params_button(
                parameters_copypaste.ParamBinding(
                    paste_button=toprow.paste,
                    tabname="txt2img",
                    source_text_component=toprow.prompt,
                    source_image_component=None,
                )
            )

            steps = scripts.scripts_txt2img.script("Sampler").steps

            txt2img_preview_params = [
                toprow.prompt,
                toprow.negative_prompt,
                steps,
                scripts.scripts_txt2img.script("Sampler").sampler_name,
                cfg_scale,
                scripts.scripts_txt2img.script("Seed").seed,
                width,
                height,
            ]
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
                fn=wrap_queued_call(update_token_counter),
                inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.token_counter],
            )
            toprow.negative_token_button.click(
                fn=wrap_queued_call(update_negative_prompt_token_counter),
                inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown],
                outputs=[toprow.negative_token_counter],
            )

        extra_networks_ui = ui_extra_networks.create_ui(
            txt2img_interface, [txt2img_generation_tab], "txt2img"
        )
        ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)

        extra_tabs.__exit__()
    return txt2img_interface, txt2img_preview_params
