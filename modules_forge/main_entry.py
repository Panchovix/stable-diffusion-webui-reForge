import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared_items, shared, ui_common, sd_models, processing, infotext_utils, paths, ui_loadsave
from backend import memory_management, stream
from backend.args import dynamic_args
from modules.shared import cmd_opts

total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Dropdown = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None



forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'float16 (fp16 LoRA)': (torch.float16, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_ckpt():
    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)
    return ckpt_list
    
    
def refresh_vaete():
    global module_list
    
    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors', 'gguf']

    module_list.clear()

    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
        os.path.abspath(os.path.join(paths.models_path, "other_module")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return module_list.keys()


def refresh_models():
    checkpoint_list = refresh_ckpt()
    modules_list = refresh_vaete()

    return checkpoint_list, modules_list

ckpt_list = refresh_ckpt()
if module_list == {}:
    _ = refresh_vaete()


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset, ckpt_list

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(label="UI", value=lambda: shared.opts.forge_preset, choices=['sd', 'xl', 'sd3', 'flux', 'all'], elem_id="forge_ui_preset", scale=0)

    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="Additional modules",
        choices=list(module_list.keys())
    )

    def gr_refresh_models():
        a, b = refresh_models()
        return gr.update(choices=a), gr.update(choices=b)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        inputs=None,
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()), filterable=False)
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: shared.opts.forge_async_loading, choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: shared.opts.forge_pin_shared_memory, choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change_ui, inputs=[ui_checkpoint, ui_vae], outputs=[ui_vae], show_progress=False)
    ui_vae.change(modules_change, inputs=[ui_vae], queue=False, show_progress=False)

    return


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    """ Passes precalculated 'model_memory' from "GPU Weights" UI slider (skip redundant calculation) """
    refresh_memory_management_settings(
        async_loading=async_loading,
        pin_shared_memory=pin_shared_memory,
        model_memory=model_memory  # Use model_memory directly from UI slider value
    )

def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    # Fallback to defaults if values are not passed
    async_loading = async_loading if async_loading is not None else shared.opts.forge_async_loading
    inference_memory = inference_memory if inference_memory is not None else shared.opts.forge_inference_memory
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else shared.opts.forge_pin_shared_memory

    # If model_memory is provided, calculate inference memory accordingly, otherwise use inference_memory directly
    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    print(f'Environment vars changed: {log_dict}')

    if inference_memory < min(512, total_vram * 0.05):
        print('------------------')
        print(f'[Low VRAM Warning] You just set Forge to use 100% GPU memory ({model_memory:.2f} MB) to load model weights.')
        print('[Low VRAM Warning] This means you will have 0% GPU memory (0.00 MB) to do matrix computation. Computations may fallback to CPU or go Out of Memory.')
        print('[Low VRAM Warning] In many cases, image generation will be 10x slower.')
        print("[Low VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
        print("[Low VRAM Warning] If you cannot find 'GPU Weights', you can click the 'all' option in the 'UI' area on the left-top corner of the webpage.")
        print('[Low VRAM Warning] Make sure that you know what you are testing.')
        print('------------------')
    else:
        compute_percentage = (inference_memory / total_vram) * 100.0
        print(f'[GPU Setting] You will use {(100 - compute_percentage):.2f}% GPU memory ({model_memory:.2f} MB) to load weights, and use {compute_percentage:.2f}% GPU memory ({inference_memory:.2f} MB) to do matrix computation.')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    dynamic_args['online_lora'] = lora_fp16

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')
    processing.need_global_unload = True

    return



def checkpoint_change_ui(ckpt_name:str, vae_te:list):
    result = vae_te

    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info != current_ckpt_info:
        if shared.opts.sd_vae_overrides_per_model_preferences:
            from modules import extra_networks

            metadata = extra_networks.get_user_metadata(new_ckpt_info.filename)
            vae_metadata = metadata.get("vae_te", None)
            if vae_metadata is None:
                vae_metadata = metadata.get("vae", None)

            if vae_metadata is not None:
                if isinstance(vae_metadata, str):
                    vae_metadata = [vae_metadata]

                if "Built in" in vae_metadata:  # this means use models built in to checkpoint, so clear the selection
                    vae_metadata = []

                if vae_metadata != ['']:        # ['']  means 'no change', keep whatever is already set
                    modules_change(vae_metadata, save=False, refresh=False)
                    result = vae_metadata

        shared.opts.set('sd_model_checkpoint', ckpt_name)

        shared.opts.save(shared.config_filename)
        refresh_model_loading_parameters()

    return result


def checkpoint_change(ckpt_name:str, save=True, refresh=True):
    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set('sd_model_checkpoint', ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values:list, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])

    # skip further processing if value unchanged
    if sorted(modules) == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set('forge_additional_modules', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'Hires CFG Scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')

    output_targets = [
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)
    ui_forge_preset.change(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)

    Context.root_block.load(on_preset_change, inputs=None, outputs=output_targets, queue=True, show_progress=False)

    refresh_model_loading_parameters()
    return


def on_preset_change(preset=None):
    if preset is not None:
        shared.opts.set('forge_preset', preset)
        shared.opts.save(shared.config_filename)
    else:
        preset = getattr(shared.opts, 'forge_preset', None)

    if shared.opts.forge_preset == 'sd':
        return [
            gr.update(value=getattr(shared.opts, "sd_vae_te", [""])),
            gr.update(visible=True),                                                    # ui_clip_skip
            gr.update(value=getattr(shared.opts, "sd_unet_dtype", 'Automatic')),
            gr.update(visible=False),                                                   # ui_forge_async_loading
            gr.update(visible=False),                                                   # ui_forge_pin_shared_memory
            gr.update(visible=False, value=total_vram - 1024),                          # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 512)),
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 512)),
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 640)),
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 512)),
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),
            gr.update(visible=False),                                                   # ui_txt2img_distilled_cfg
            gr.update(visible=False),                                                   # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler a')),
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'Euler a')),
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Automatic')),
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Automatic')),
            gr.update(value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),
            gr.update(visible=False),                                                   # ui_txt2img_hr_distilled_cfg
        ]

    if shared.opts.forge_preset == 'xl':
        model_mem = getattr(shared.opts, "xl_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(value=getattr(shared.opts, "xl_vae_te", [""])),
            gr.update(visible=True),                                                    # ui_clip_skip
            gr.update(value=getattr(shared.opts, "xl_unet_dtype", 'Automatic')),
            gr.update(visible=False),                                                   # ui_forge_async_loading
            gr.update(visible=False),                                                   # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 896)),
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 1024)),
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1152)),
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1024)),
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),
            gr.update(visible=False),                                                   # ui_txt2img_distilled_cfg
            gr.update(visible=False),                                                   # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler a')),
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler a')),
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Automatic')),
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Automatic')),
            gr.update(value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),
            gr.update(visible=False),                                                   # ui_txt2img_hr_distilled_cfg
        ]

    if shared.opts.forge_preset == 'sd3':
        model_mem = getattr(shared.opts, "sd3_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(value=getattr(shared.opts, "sd3_vae_te", [""])),
            gr.update(visible=True),                                                    # ui_clip_skip
            gr.update(value=getattr(shared.opts, "sd3_unet_dtype", 'Automatic')),
            gr.update(visible=True),                                                    # ui_forge_async_loading
            gr.update(visible=True),                                                    # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd3_t2i_width", 896)),
            gr.update(value=getattr(shared.opts, "sd3_i2i_width", 1024)),
            gr.update(value=getattr(shared.opts, "sd3_t2i_height", 1152)),
            gr.update(value=getattr(shared.opts, "sd3_i2i_height", 1024)),
            gr.update(value=getattr(shared.opts, "sd3_t2i_cfg", 1)),
            gr.update(value=getattr(shared.opts, "sd3_i2i_cfg", 1)),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=getattr(shared.opts, "sd3_t2i_sampler", 'Euler')),
            gr.update(value=getattr(shared.opts, "sd3_i2i_sampler", 'Euler')),
            gr.update(value=getattr(shared.opts, "sd3_t2i_scheduler", 'Simple')),
            gr.update(value=getattr(shared.opts, "sd3_i2i_scheduler", 'Simple')),
            gr.update(value=getattr(shared.opts, "sd3_t2i_hr_cfg", 1.0)),
            gr.update(visible=False),
        ]

    if shared.opts.forge_preset == 'flux':
        model_mem = getattr(shared.opts, "flux_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(value=getattr(shared.opts, "flux_vae_te", [""])),
            gr.update(visible=False),                                                   # ui_clip_skip
            gr.update(value=getattr(shared.opts, "flux_unet_dtype", 'Automatic')),
            gr.update(visible=True),                                                    # ui_forge_async_loading
            gr.update(visible=True),                                                    # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 896)),
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 1024)),
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1152)),
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1024)),
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)),
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)),
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),
            gr.update(value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),
        ]

    if shared.opts.use_ui_config_json:
        loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
        ui_settings_from_file = loadsave.ui_settings.copy()

        return [
            gr.update(),                                                                    # ui_vae
            gr.update(visible=True),                                                        # ui_clip_skip
            gr.update(),                                                                    # ui_forge_unet_storage_dtype_options
            gr.update(visible=True),                                                        # ui_forge_async_loading
            gr.update(visible=True),                                                        # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 1024),                               # ui_forge_inference_memory
            gr.update(value=ui_settings_from_file['txt2img/Width/value']),
            gr.update(value=ui_settings_from_file['img2img/Width/value']),
            gr.update(value=ui_settings_from_file['txt2img/Height/value']),
            gr.update(value=ui_settings_from_file['img2img/Height/value']),
            gr.update(value=ui_settings_from_file['txt2img/CFG Scale/value']),
            gr.update(value=ui_settings_from_file['img2img/CFG Scale/value']),
            gr.update(visible=True, value=ui_settings_from_file['txt2img/Distilled CFG Scale/value']),
            gr.update(visible=True, value=ui_settings_from_file['img2img/Distilled CFG Scale/value']),
            gr.update(value=ui_settings_from_file['customscript/sampler.py/txt2img/Sampling method/value']),
            gr.update(value=ui_settings_from_file['customscript/sampler.py/img2img/Sampling method/value']),
            gr.update(value=ui_settings_from_file['customscript/sampler.py/txt2img/Schedule type/value']),
            gr.update(value=ui_settings_from_file['customscript/sampler.py/img2img/Schedule type/value']),
            gr.update(visible=True, value=ui_settings_from_file['txt2img/Hires CFG Scale/value']),
            gr.update(visible=True, value=ui_settings_from_file['txt2img/Hires Distilled CFG Scale/value']),
        ]

    return [
        gr.update(),                                                                    # ui_vae
        gr.update(visible=True),                                                        # ui_clip_skip
        gr.update(),                                                                    # ui_forge_unet_storage_dtype_options
        gr.update(visible=True),                                                        # ui_forge_async_loading
        gr.update(visible=True),                                                        # ui_forge_pin_shared_memory
        gr.update(visible=True, value=total_vram - 1024),                               # ui_forge_inference_memory
        gr.update(value=1024),
        gr.update(value=1024),
        gr.update(value=1024),
        gr.update(value=1024),
        gr.update(value=5),
        gr.update(value=5),
        gr.update(visible=True, value=3.5),
        gr.update(visible=True, value=3.5),
        gr.update(value='Euler'),
        gr.update(value='Euler'),
        gr.update(value='Simple'),
        gr.update(value='simple'),
        gr.update(value=3),
        gr.update(visible=True, value=3.5),
    ]


shared.options_templates.update(shared.options_section(('ui_sd', "UI defaults 'sd'", "ui"), {
    "sd_t2i_width":  shared.OptionInfo(512,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_t2i_height": shared.OptionInfo(640,  "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_t2i_cfg":    shared.OptionInfo(7,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_t2i_hr_cfg": shared.OptionInfo(7,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_i2i_width":  shared.OptionInfo(512,  "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_i2i_height": shared.OptionInfo(512,  "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd_i2i_cfg":    shared.OptionInfo(7,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "sd_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
    "xl_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "xl_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "xl_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "xl_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "xl_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_sd3', "UI defaults 'sd3'", "ui"), {
    "sd3_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "sd3_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "sd3_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
    "sd3_vae_te":     shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "sd3_unet_dtype": shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))
shared.options_templates.update(shared.options_section(('ui_flux', "UI defaults 'flux'", "ui"), {
    "flux_t2i_width":    shared.OptionInfo(896,  "txt2img width",                gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_t2i_height":   shared.OptionInfo(1152, "txt2img height",               gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_t2i_cfg":      shared.OptionInfo(1,    "txt2img CFG",                  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_cfg":   shared.OptionInfo(1,    "txt2img HiRes CFG",            gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_i2i_width":    shared.OptionInfo(1024, "img2img width",                gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_i2i_height":   shared.OptionInfo(1024, "img2img height",               gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8}),
    "flux_i2i_cfg":      shared.OptionInfo(1,    "img2img CFG",                  gr.Slider, {"minimum": 1,   "maximum": 30,   "step": 0.1}),
    "flux_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,   "maximum": 30,   "step": 0.1}),
    "flux_GPU_MB":       shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)",gr.Slider, {"minimum": 0,   "maximum": total_vram,   "step": 1}),
    "flux_vae_te":       shared.OptionInfo([""], "VAE / Text Encoder", gr.Dropdown,{"multiselect": True, "choices": list(module_list.keys())}),
    "flux_unet_dtype":   shared.OptionInfo("Automatic", "Diffusion in Low Bits", gr.Dropdown, {"choices": list(forge_unet_storage_dtype_options.keys())}),
}))

