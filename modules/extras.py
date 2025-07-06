import os
import gc
import re
import json

import torch
import numpy

from modules import shared, images, sd_models, errors, paths
from modules.ui_common import plaintext_to_html
import gradio as gr
import safetensors.torch
from modules_forge.main_entry import module_list, module_vae_list
from backend.loader import replace_state_dict
from backend.utils import load_torch_file

import huggingface_guess

def run_pnginfo(image):
    if image is None:
        return '', '', ''

    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
<div class="infotext">
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', geninfo, info


checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

def to_bfloat16(tensor):
    return tensor.to(torch.bfloat16)

def to_float16(tensor):
    return tensor.to(torch.float16)

def to_fp8e4m3(tensor):
    return tensor.to(torch.float8_e4m3fn)

def to_fp8e5m2(tensor):
    return tensor.to(torch.float8_e5m2)

def read_metadata(model_names):
    metadata = {}

    for checkpoint_name in [model_names]:
        checkpoint_info = sd_models.checkpoints_list.get(checkpoint_name, None)
        if checkpoint_info is None:
            continue

        metadata.update(checkpoint_info.metadata)

    return json.dumps(metadata, indent=4, ensure_ascii=False)

@torch.no_grad()
def run_modelmerger(id_task, model_names, interp_method, multiplier, save_u, save_v, save_t, calc_fp32, custom_name, config_source, bake_in_vae, bake_in_te, discard_weights, save_metadata, add_merge_recipe, copy_metadata_fields, metadata_json):

    shared.state.begin(job="model-merge")

    if len(model_names) > 2:
        tertiary_model_name = model_names[2]
    if len(model_names) > 1:
        secondary_model_name = model_names[1]
    primary_model_name = model_names[0]


    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return [*[gr.update() for _ in range(4)], message]

    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    def filename_nothing(): # avoid overwrite original checkpoint
        return "[]" + primary_model_info.model_name

    def filename_weighted_sum():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    def filename_in_out():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name

        return f"{a}-{b}({multiplier})"

    def filename_add_difference():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        M = round(multiplier, 2)

        return f"{a} + {M}({b} - {c})"

    def filename_unet():
        return "[UNET]-" + primary_model_info.model_name

    def filename_vae():
        return "[VAE]-" + primary_model_info.model_name

    def filename_te():
        return "[TE]-" + primary_model_info.model_name

    theta_funcs = {
        "None": (filename_nothing, None, None),
        "Weighted sum": (filename_weighted_sum, None, weighted_sum),
        "Add difference": (filename_add_difference, get_difference, add_difference),
        "SmoothBlend": (filename_in_out, None, weighted_sum),
        "Extract Unet": (filename_unet, None, None),
        "Extract VAE": (filename_vae, None, None),
        "Extract Text encoder(s)" : (filename_te, None, None),
    }
    filename_generator, theta_func1, theta_func2 = theta_funcs[interp_method]
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    if bake_in_vae != "":
        shared.state.job_count += 1
    if bake_in_te != []:
        shared.state.job_count += 1

    if (save_u != "None (remove)" and save_u != "No change"):
        shared.state.job_count += 1
    if (save_v != "None (remove)" and save_v != "No change"):
        shared.state.job_count += 1
    if (save_t != "None (remove)" and save_t != "No change"):
        shared.state.job_count += 1

    if not primary_model_name:
        return fail("Failed: Merging requires a primary model.")

    primary_model_info = sd_models.checkpoint_aliases[primary_model_name]

    if theta_func2 and not secondary_model_name:
        return fail("Failed: Merging requires a secondary model.")

    secondary_model_info = sd_models.checkpoint_aliases[secondary_model_name] if theta_func2 else None

    if theta_func1 and not tertiary_model_name:
        return fail(f"Failed: Interpolation method ({interp_method}) requires a tertiary model.")

    tertiary_model_info = sd_models.checkpoint_aliases[tertiary_model_name] if theta_func1 else None

    result_is_inpainting_model = False
    result_is_instruct_pix2pix_model = False

    def load_model (filename, message):
        shared.state.textinfo = f"Loading {message}: {filename} ..."

        theta = load_torch_file(filename)

        #   strip unwanted keys immediately - reduce memory use and processing
        if interp_method == "Extract Unet":
            regex = re.compile(r'^(?!.*(model\.diffusion_model)\.)')
            strip = 8
        elif interp_method == "Extract VAE":
            regex = re.compile(r'^(?!.*(first_stage_model|vae)\.)')
            strip = 8
        elif interp_method == "Extract Text encoder(s)":
            regex = re.compile(r'^(?!.*(text_model|conditioner\.embedders|cond_stage_model|text_encoders)\.)')
            strip = 8
        else:
            strip = 0
            if save_t == "None (remove)" or (bake_in_te != []):
                strip += 1
            if save_v == "None (remove)" or (bake_in_vae != ""):
                strip += 2
            if save_u == "None (remove)":
                strip += 4

            match strip:
                case 1:
                    regex = re.compile(r'(text_model|conditioner\.embedders|cond_stage_model|text_encoders)\.')
                case 2:
                    regex = re.compile(r'(first_stage_model|vae)\.')
                case 3:
                    regex = re.compile(r'(text_model|conditioner\.embedders|cond_stage_model|text_encoders|first_stage_model|vae)\.')
                case 4:
                    regex = re.compile(r'(model\.diffusion_model)\.')
                case 5:
                    regex = re.compile(r'(text_model|conditioner\.embedders|cond_stage_model|text_encoders|model\.diffusion_model)\.')
                case 6:
                    regex = re.compile(r'(first_stage_model|vae|model\.diffusion_model)\.')
                case 7:
                    regex = re.compile(r'(text_model|conditioner\.embedders|cond_stage_model|text_encoders|first_stage_model|vae|model\.diffusion_model)\.')
                case _:
                    pass

        if strip > 0:
            for key in list(theta):
                if re.search(regex, key):
                    theta.pop(key)

        if discard_weights:
            regex = re.compile(discard_weights)
            for key in list(theta):
                if re.search(regex, key):
                    theta.pop(key)

        if calc_fp32:
            for k,v in theta.items():
                theta[k] = v.to(torch.float32)

        return theta

    if theta_func2:
        shared.state.textinfo = 'Loading B'
        theta_1 = load_model(secondary_model_info.filename, "B")
    else:
        theta_1 = None

    if theta_func1:
        shared.state.textinfo = 'Loading C'
        theta_2 = load_model(tertiary_model_info.filename, "C")

        shared.state.textinfo = 'Merging B and C'
        for key in theta_1.keys():
            if key in checkpoint_dict_skip_on_merge:
                continue

            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.pop(key)   # .get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])

        del theta_2
        shared.state.nextjob()

    shared.state.textinfo = 'Loading A'
    theta_0 = load_model(primary_model_info.filename, "A")

    if "Extract" in interp_method:
        filename = filename_generator() if custom_name == '' else custom_name
        filename += ".safetensors"

# should these paths be hardcoded?
        if interp_method == "Extract Text encoder(s)":
            type = "Text encoder(s)"
            te_dir = os.path.abspath(os.path.join(paths.models_path, "text_encoder"))
            output_modelname = os.path.join(te_dir, filename)
        elif interp_method == "Extract VAE":
            type = "VAE"
            vae_dir = os.path.abspath(os.path.join(paths.models_path, "VAE"))
            output_modelname = os.path.join(vae_dir, filename)
        elif interp_method == "Extract Unet":
            type = "Unet"
            unet_dir = os.path.abspath(os.path.join(paths.models_path, "Stable-diffusion"))
            output_modelname = os.path.join(unet_dir, filename)
        else:
            type = None

        if type:
            shared.state.textinfo = f"Saving to {output_modelname} ..."

            safetensors.torch.save_file(theta_0, output_modelname, metadata=None)

            shared.state.textinfo = f"{type} saved to {output_modelname}"
            shared.state.end()

        return [gr.Dropdown.update(), gr.Dropdown.update(), "Checkpoint saved to " + output_modelname]

    if theta_1:
        shared.state.textinfo = 'Merging A and B'

        if interp_method == "SmoothBlend":
            total_key_count = 24.0 if 'model.diffusion_model.output_blocks.11.0.emb_layers.1.bias' in theta_0.keys() else 19.0

            for key in theta_0.keys():
                if 'model' in key and key in theta_1:

                    if key in checkpoint_dict_skip_on_merge:
                        continue

                    a = theta_0[key]
                    b = theta_1.pop(key)

#input 0-11, middle 0 output 0-11 : 12 + 1 + 12 = 25 (0->24) for sd1.5
#input 0-8, middle 0 output 0-8 : 9 + 1 + 9 = 19 (0->18) for sdxl?

                    if 'input_blocks' in key:
                        key_count = 0
                    elif 'middle_block' in key:
                        key_count = 12 if total_key_count == 24.0 else 9
                    elif 'output_blocks' in key:
                        key_count = 13 if total_key_count == 24.0 else 10
                    elif '.out.' in key:
                        keycount = total_key_count
                    else:
                        continue

                    if not 'middle_block' in key:
                        for i in range(11, -1, -1):
                            if f'_blocks.{i}.' in key:
                                key_count += i
                                break

                    muli = float(key_count) / total_key_count
                    # muli *= multiplier
                    muli = min(multiplier, muli)
                    theta_0[key] = theta_func2(a, b, muli)

        else:
            for key in theta_0.keys():
                if 'model' in key and key in theta_1:

                    if key in checkpoint_dict_skip_on_merge:
                        continue

                    a = theta_0[key]
                    b = theta_1.pop(key)

                    # this enables merging an inpainting model (A) with another one (B);
                    # where normal model would have 4 channels, for latent space, inpainting model would
                    # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
                    if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                        if a.shape[1] == 4 and b.shape[1] == 9:
                            raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")
                        if a.shape[1] == 4 and b.shape[1] == 8:
                            raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")

                        if a.shape[1] == 8 and b.shape[1] == 4:#If we have an Instruct-Pix2Pix model...
                            theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)#Merge only the vectors the models have in common.  Otherwise we get an error due to dimension mismatch.
                            result_is_instruct_pix2pix_model = True
                        else:
                            assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
                            theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                            result_is_inpainting_model = True
                    else:
                        theta_0[key] = theta_func2(a, b, multiplier)

        del theta_1
        shared.state.nextjob()
    else:
        shared.state.textinfo = 'Copying A'

    if save_u != "None (remove)" and ("" != bake_in_vae or [] != bake_in_te):
        guess = huggingface_guess.guess(theta_0)
    else:
        guess = None

    # bake in vae
    if "" != bake_in_vae:
        shared.state.textinfo = f'Baking in VAE from {bake_in_vae}'
        vae_dict = load_torch_file(module_vae_list[bake_in_vae])

        if guess:
            theta_0 = replace_state_dict (theta_0, vae_dict, guess)
        else:
            for key in vae_dict.keys():
                theta_0[key] = vae_dict[key]   # precision convert later

        del vae_dict

        shared.state.nextjob()

    # bake in text encoders
    if bake_in_te != []:
        for te in bake_in_te:
            shared.state.textinfo = f'Baking in Text encoder from {te}'
            te_dict = load_torch_file(module_list[te])

            if guess:
                theta_0 = replace_state_dict (theta_0, te_dict, guess)
            else:
                for key in te_dict.keys():
                    theta_0[key] = te_dict[key]     # precision convert later

            del te_dict

        shared.state.nextjob()

    if discard_weights:     # this is repeated from load_model() in case baking vae/te put unwanted keys back
                            # for example, could have VAE decoder only by discarding "first_stage_model.encoder."
                            # (but will then get warning about missing keys)
        regex = re.compile(discard_weights)
        for key in list(theta_0):
            if re.search(regex, key):
                theta_0.pop(key)

    shared.state.textinfo = 'Converting keys to selected dtypes'
    saves = [0, save_u, 1, save_v, 2, save_t]
    for save in saves:
        if save != "None" and save != "No change":
            match save:
                case 0:
                    regex = re.compile("model.diffusion_model.|double_block.|single_block.")    #   untested if this hits inpaint, pix2pix keys
                case 1:
                    regex = re.compile(r'\b(first_stage_model|vae)\.\b')
                case 2:
                    regex = re.compile(r'\b(text_model|conditioner\.embedders)\.\b')

                case "bfloat16":
                    for key in theta_0.keys():
                        if re.search(regex, key):
                            theta_0[key] = to_bfloat16(theta_0[key])
                case "float16":
                    for key in theta_0.keys():
                        if re.search(regex, key):
                            theta_0[key] = to_float16(theta_0[key])
                case "fp8e4m3":
                    for key in theta_0.keys():
                        if re.search(regex, key):
                            theta_0[key] = to_fp8e4m3(theta_0[key])
                case "fp8e5m2":
                    for key in theta_0.keys():
                        if re.search(regex, key):
                            theta_0[key] = to_fp8e5m2(theta_0[key])

                case "None (remove)":
                    for key in theta_0.keys():
                        if re.search(regex, key):
                            theta_0.pop(key)
                case _:
                    pass

            shared.state.nextjob()


    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    filename = filename_generator() if custom_name == '' else custom_name
    filename += ".inpainting" if result_is_inpainting_model else ""
    filename += ".instruct-pix2pix" if result_is_instruct_pix2pix_model else ""
    filename += ".safetensors"

    output_modelname = os.path.join(ckpt_dir, filename)

    shared.state.textinfo = f"Saving to {output_modelname} ..."

    metadata = {}

    if save_metadata and copy_metadata_fields:
        if primary_model_info:
            metadata.update(primary_model_info.metadata)
        if secondary_model_info:
            metadata.update(secondary_model_info.metadata)
        if tertiary_model_info:
            metadata.update(tertiary_model_info.metadata)

    if save_metadata:
        try:
            metadata.update(json.loads(metadata_json))
        except Exception as e:
            errors.display(e, "readin metadata from json")

        metadata["format"] = "pt"

    if save_metadata and add_merge_recipe:
        save_as = f"Unet: {save_u}, VAE: {save_v}, Text encoder(s): {save_t}"
        merge_recipe = {
            "type": "webui", # indicate this model was merged with webui's built-in merger
            "primary_model_hash": primary_model_info.sha256,
            "secondary_model_hash": secondary_model_info.sha256 if secondary_model_info else None,
            "tertiary_model_hash": tertiary_model_info.sha256 if tertiary_model_info else None,
            "interp_method": interp_method,
            "multiplier": multiplier,
            "save_as": save_as,
            "custom_name": custom_name,
            "config_source": config_source,
            "bake_in_vae": bake_in_vae,
            "bake_in_te": bake_in_te,
            "discard_weights": discard_weights,
            "is_inpainting": result_is_inpainting_model,
            "is_instruct_pix2pix": result_is_instruct_pix2pix_model
        }

        sd_merge_models = {}

        def add_model_metadata(checkpoint_info):
            checkpoint_info.calculate_shorthash()
            sd_merge_models[checkpoint_info.sha256] = {
                "name": checkpoint_info.name,
                "legacy_hash": checkpoint_info.hash,
                "sd_merge_recipe": checkpoint_info.metadata.get("sd_merge_recipe", None)
            }

            sd_merge_models.update(checkpoint_info.metadata.get("sd_merge_models", {}))

        add_model_metadata(primary_model_info)
        if secondary_model_info:
            add_model_metadata(secondary_model_info)
        if tertiary_model_info:
            add_model_metadata(tertiary_model_info)

        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
        metadata["sd_merge_models"] = json.dumps(sd_merge_models)

    gc.collect()
    torch.cuda.empty_cache()

    safetensors.torch.save_file(theta_0, output_modelname, metadata=metadata if len(metadata)>0 else None)

    sd_models.list_models()
    created_model = next((ckpt for ckpt in sd_models.checkpoints_list.values() if ckpt.name == filename), None)
    if created_model:
        created_model.calculate_shorthash()

    shared.state.textinfo = f"Checkpoint saved to {output_modelname}"
    shared.state.end()

    new_model_list = sd_models.checkpoint_tiles()
    return [gr.Dropdown(value=model_names, choices=new_model_list), gr.Dropdown(choices=new_model_list), "Checkpoint saved to " + output_modelname]
