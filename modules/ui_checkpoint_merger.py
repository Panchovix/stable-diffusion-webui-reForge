import os
import gradio as gr

from modules import sd_models, sd_vae, errors, extras, call_queue
from modules.ui_components import FormRow
from modules.ui_common import ToolButton, refresh_symbol
from modules_forge.main_entry import module_list, refresh_models


def update_interp_description(value, choices):
    interp_descriptions = {
        "None"                      : (1, "Allows for format conversion and VAE baking."),
        "Weighted sum"              : (2, "Requires two models: A and B. The result is calculated as A * (1 - M) + B * M"),
        "Add difference"            : (3, "Requires three models: A, B and C. The result is calculated as A + (B - C) * M"),
        "Extract Unet"              : (1, "Takes one model (A) as input. Only output name option is relevant."),
        "Extract VAE"               : (1, "Takes one model (A) as input. Only output name option is relevant."),
        "Extract Text encoder(s)"   : (1, "Takes one model (A) as input. Only output name option is relevant."),
    }

    description = interp_descriptions[value][1]
    count = interp_descriptions[value][0]
    return gr.Dropdown(info=description), gr.Dropdown(max_choices=count, value=choices[0:count])


def modelmerger(*args):
    try:
        results = extras.run_modelmerger(*args)
    except Exception as e:
        errors.report("Error loading/saving model file", exc_info=True)
        sd_models.list_models()  # to remove the potentially missing models from the list
        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
    return results


class UiCheckpointMerger:
    vae_list = []
    te_list = []

    def refresh_additional (fromUI=True):
        refresh_models()
        sd_vae.refresh_vae_list()

        vae_list = list(sd_vae.vae_dict)
        te_list = list(module_list.keys())
        for vae in vae_list:
            if vae in te_list:
                te_list.remove(vae)

        vae_list = [""] + vae_list

        if fromUI:
            return gr.Dropdown(choices=vae_list), gr.Dropdown(choices=te_list)
        else:
            return vae_list, te_list

    vae_list, te_list = refresh_additional (fromUI=False)

    def __init__(self):
        with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
            with gr.Accordion(open=True, label='Save Current Checkpoint (including all quantization)'):
                with gr.Row():
                    textbox_file_name_forge = gr.Textbox(label="Filename (will save in /models/Stable-diffusion)", value='my_model.safetensors')
                    btn_save_unet_forge = gr.Button('Save UNet')
                    btn_save_ckpt_forge = gr.Button('Save Checkpoint')

                with gr.Row():
                    result_html = gr.Markdown('Ready to save ...')

                    def save_unet(filename):
                        from modules.paths import models_path
                        long_filename = os.path.join(models_path, 'Stable-diffusion', filename)
                        os.makedirs(os.path.dirname(long_filename), exist_ok=True)
                        from modules import shared, sd_models
                        sd_models.forge_model_reload()
                        p = shared.sd_model.save_unet(long_filename)
                        print(f'Saved UNet at: {p}')
                        return f'Saved UNet at: {p}'

                    def save_checkpoint(filename):
                        from modules.paths import models_path
                        long_filename = os.path.join(models_path, 'Stable-diffusion', filename)
                        os.makedirs(os.path.dirname(long_filename), exist_ok=True)
                        from modules import shared
                        sd_models.forge_model_reload()
                        p = shared.sd_model.save_checkpoint(long_filename)
                        print(f'Saved checkpoint at: {p}')
                        return f'Saved checkpoint at: {p}'

                    btn_save_unet_forge.click(save_unet, inputs=textbox_file_name_forge, outputs=result_html)
                    btn_save_ckpt_forge.click(save_checkpoint, inputs=textbox_file_name_forge, outputs=result_html)

# add checkbox to specifiy checkbox is vpred (add vpred key)
# similar possible for cos? etc

            with gr.Row(equal_height=False):
                with gr.Column(variant='compact'):
                    with FormRow():
                        self.interp_method = gr.Dropdown(choices=["None", "Extract Unet", "Extract VAE", "Extract Text encoder(s)", "Weighted sum", "Add difference"], value="None", label="Interpolation method / Function", elem_id="modelmerger_interp_method", info="Allows for format conversion and VAE baking.")
                        self.custom_name = gr.Textbox(label="Custom output name", info="Optional", max_lines=1, elem_id="modelmerger_custom_name")

                    with FormRow(elem_id="modelmerger_models"):
                        self.model_names = gr.Dropdown(sd_models.checkpoint_tiles(), multiselect=True, max_choices=1, elem_id="modelmerger_model_names", label="Models (select in order: A; optional B, C)", value=[])

                        def refresh_checkpoints():
                            sd_models.list_models()
                            newlist = sd_models.checkpoint_tiles()
                            return gr.Dropdown(choices=newlist)

                        self.refresh_button = ToolButton(value=refresh_symbol)
                        self.refresh_button.click(fn=refresh_checkpoints, inputs=None, outputs=[self.model_names])

                    self.interp_method.change(fn=update_interp_description, inputs=[self.interp_method, self.model_names], outputs=[self.interp_method, self.model_names], show_progress=False)

                    self.interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M)', value=0.5, elem_id="modelmerger_interp_amount")

                    with FormRow():
                        self.bake_in_vae = gr.Dropdown(choices=self.vae_list, value="", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                        self.bake_in_te = gr.Dropdown(choices=self.te_list, value=[], label="Bake in Text encoder(s)", elem_id="modelmerger_bake_in_te", multiselect=True, max_choices=3)

                        self.refresh_buttonM = ToolButton(value=refresh_symbol, elem_id="modelmerger_refresh_vaete")
                        self.refresh_buttonM.click(fn=UiCheckpointMerger.refresh_additional, inputs=None, outputs=[self.bake_in_vae, self.bake_in_te])

                    with FormRow():
                        self.save_u = gr.Dropdown(label="Unet precision", choices=["None (remove)", "No change", "float32", "bfloat16", "float16", "fp8e4m3", "fp8e5m2"], value="float16")
                        self.save_v = gr.Dropdown(label="VAE precision", choices=["None (remove)", "No change", "float32", "bfloat16", "float16", "fp8e4m3", "fp8e5m2"], value="float16")
                        self.save_t = gr.Dropdown(label="Text encoder(s) precision", choices=["None (remove)", "No change", "float32", "bfloat16", "float16", "fp8e4m3", "fp8e5m2"], value="float16")
                        self.calc_fp32 = gr.Checkbox(value=False, label="Calculate merge in float32")
# if want to save fp32, must also set calc_fp32 for non-fp32 models

                    with FormRow():
                        self.discard_weights = gr.Textbox(value="", label="Discard weights with matching name; e.g. Use 'model_ema' to discard EMA weights.", elem_id="modelmerger_discard_weights")

                    with FormRow():
                        with gr.Column():
                            self.config_source = gr.Radio(choices=["A, B or C", "B", "C", "Don't"], value="Don't", label="Copy config from", type="index", elem_id="modelmerger_config_method")

                    with gr.Accordion("Metadata", open=False) as metadata_editor:
                        with FormRow():
                            self.save_metadata = gr.Checkbox(value=False, label="Save metadata", elem_id="modelmerger_save_metadata")
                            self.add_merge_recipe = gr.Checkbox(value=False, label="Add merge recipe metadata", elem_id="modelmerger_add_recipe")
                            self.copy_metadata_fields = gr.Checkbox(value=False, label="Copy metadata from merged models", elem_id="modelmerger_copy_metadata")

                        self.metadata_json = gr.TextArea('{}', label="Metadata in JSON format")
                        self.read_metadata = gr.Button("Read metadata from selected checkpoints")


                with gr.Column(variant='compact', elem_id="modelmerger_results_container"):
                    self.modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant='primary')
                    with gr.Group(elem_id="modelmerger_results_panel"):
                        self.modelmerger_result = gr.HTML(elem_id="modelmerger_result", show_label=False)

        self.metadata_editor = metadata_editor
        self.blocks = modelmerger_interface

        return

    def setup_ui(self, dummy_component, sd_model_checkpoint_component):
        self.read_metadata.click(extras.read_metadata, inputs=[self.model_names], outputs=[self.metadata_json])

        self.modelmerger_merge.click(fn=lambda: '', inputs=None, outputs=[self.modelmerger_result])
        self.modelmerger_merge.click(
            fn=call_queue.wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
            _js='modelmerger',
            inputs=[
                dummy_component,
                self.model_names,
                self.interp_method,
                self.interp_amount,
                self.save_u,
                self.save_v,
                self.save_t,
                self.calc_fp32,
                self.custom_name,
                self.config_source,
                self.bake_in_vae,
                self.bake_in_te,
                self.discard_weights,
                self.save_metadata,
                self.add_merge_recipe,
                self.copy_metadata_fields,
                self.metadata_json,
            ],
            outputs=[
                self.model_names,
                sd_model_checkpoint_component,
                self.modelmerger_result,
            ]
        ).then(fn=UiCheckpointMerger.refresh_additional, inputs=None, outputs=[self.bake_in_vae, self.bake_in_te])

