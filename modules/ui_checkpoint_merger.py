import os
import gradio as gr

from modules import sd_models, sd_vae, errors, extras, call_queue
from modules.ui_components import FormRow
from modules.ui_common import ToolButton, refresh_symbol
from modules_forge.main_entry import module_list, module_vae_list, module_te_list, refresh_models


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

def convert_embeds(one, many, output_dir):
    # https://github.com/nArn0/sdxl-embedding-converter
    import safetensors.torch
    import torch
    from tqdm import tqdm
    from modules.util import load_file_from_url, walk_files

    converter_model = torch.nn.Sequential(
        torch.nn.Linear(768, 3072),
        torch.nn.ReLU(),
        torch.nn.Linear(3072, 3072),
        torch.nn.ReLU(),
        torch.nn.Linear(3072, 1280),
    )

    try:
        converter_model_path = load_file_from_url("https://github.com/nArn0/sdxl-embedding-converter/releases/download/v1.0/model.safetensors", 
                                                  model_dir="models", file_name="embedding_converter.safetensors")
        safetensors.torch.load_model(converter_model, converter_model_path)
    except:
        return "ERROR: could not load 'models/embedding_converter.safetensors'"

    success_count = 0
    fail_count = 0

    if many != "":
        embeds = walk_files(many, allowed_extensions=[".pt", ".safetensors"])
    elif one != "":
        embeds = [one]
    else:
        return "Nothing to process"

    if output_dir == "":
        output_dir = ".\\embeddings"
    
    os.makedirs(output_dir, exist_ok=True)

    for embed in embeds:
        print (f"Embedding converter: {embed}", end="\r", flush=True)

        loaded = False
        if embed.endswith(".pt"):
            try:
                e = torch.load(embed, map_location=torch.device('cpu'))
                emb = e['string_to_param']['*']
                loaded = True
            except:
                print(f"Embedding converter: {embed} ERROR: could not load")
        elif embed.endswith(".safetensors"):
            try:
                e = safetensors.torch.load_file(embed)
                emb = e["emb_params"]
                loaded = True
            except:
                print(f"Embedding converter: {embed} ERROR: could not load")
        else:
            print (f"Embedding converter: {embed} ERROR: unknown filetype")

        if loaded and emb.shape[-1] == 768:
            length = emb.shape[0]
            print (f"Embedding converter: {embed} CONVERTING")

            clip_l = []
            clip_g = []
            for i in tqdm(range(length)):
                clip_l.append(emb[i])
                clip_g.append(converter_model(emb[i]))

            output = {}
            output['clip_l'] = torch.stack(clip_l, dim=0)
            output['clip_g'] = torch.stack(clip_g, dim=0)

            input_filename, _ = os.path.splitext(embed.split('\\')[-1])

            output_filename = os.path.join(output_dir, input_filename + "_SDXL.safetensors")

            if os.path.exists(output_filename):
                print (f"Embedding converter: {embed} NOT SAVED - output name already exists")
                fail_count += 1
            else:
                safetensors.torch.save_file(output, output_filename)
                print (f"Embedding converter: {embed} SAVED to {output_filename}")
                success_count += 1
        else:
            if loaded:
                print(f"Embedding converter: {embed} ERROR: wrong shape, probably SD2 embedding")
            fail_count += 1

    del converter_model

    return f"DONE: {success_count} processed; {fail_count} failures"
    


class UiCheckpointMerger:
    vae_list = []
    te_list = []

    def refresh_additional (fromUI=True):
        refresh_models()

        te_list = list(module_te_list.keys())
        vae_list = [""] + list(module_vae_list.keys())

        if fromUI:
            return gr.Dropdown(choices=vae_list), gr.Dropdown(choices=te_list)
        else:
            return vae_list, te_list

    vae_list, te_list = refresh_additional (fromUI=False)

    def __init__(self):
        with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
            with gr.Accordion(open=False, label='Save current checkpoint (including all quantization)'):
                with gr.Row():
                    textbox_file_name_forge = gr.Textbox(label="Filename (will save in /models/Stable-diffusion)", value='my_model.safetensors')
                    btn_save_unet_forge = gr.Button('Save UNet')
                    btn_save_ckpt_forge = gr.Button('Save checkpoint')

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

            with gr.Accordion(open=False, label='Convert SD1 embedding to SDXL'):
                with gr.Row():
                    embeds_dir = gr.Textbox(label='Directory to convert', value='')
                    embeds_one = gr.Textbox(label='Single file to convert (if no Directory set)', value='')
                    output_dir = gr.Textbox(label='Save results to directory', value='.\embeddings')
                    convert = gr.Button('Convert', variant='primary', scale=0)
                with gr.Row():
                    message = gr.Markdown("")

                convert.click(fn=convert_embeds, inputs=[embeds_one, embeds_dir, output_dir], outputs=[message])


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
                        self.discard_weights = gr.Textbox(value="", label="Discard weights with matching name; e.g. Use 'model_ema' to discard EMA weights. 'embedding_manager|lora|control_model'", elem_id="modelmerger_discard_weights")

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

