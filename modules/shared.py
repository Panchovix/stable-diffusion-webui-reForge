import os
import sys

import torch
import gradio as gr

from modules import shared_cmd_options, options, shared_items
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import devices, util, modelloader
from modules_forge.utils import torch_bgr_to_pil_image, pil_image_to_torch_bgr

from typing import TYPE_CHECKING
from backend import memory_management

if TYPE_CHECKING:
    from modules import shared_state, styles, interrogate, shared_total_tqdm, memmon

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file = cmd_opts.styles_file if len(cmd_opts.styles_file) > 0 else [os.path.join(data_path, 'styles.csv'), os.path.join(data_path, 'styles_integrated.csv')]
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo: gr.Blocks = None

device: str = None

xformers_available = (memory_management.xformers_enabled(), memory_management.XFORMERS_VERSION)
torch_version = getattr(torch, '__long_version__',torch.__version__)

hypernetworks = {}

loaded_hypernetworks = []

state: 'shared_state.State' = None

prompt_styles: 'styles.StyleDatabase' = None

interrogator: 'interrogate.InterrogateModels' = None

face_restorers = []

lama_model = None
def process_lama(image, mask):
    global lama_model
    # https://github.com/advimman/lama; github/Sanster for the model download; github.com/light-and-ray for some implementation
    if lama_model is None:
        lama_path = os.path.join(models_path, 'big-lama.pt')
        if not os.path.exists(lama_path):
            lama_path = modelloader.load_file_from_url(
                'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
                model_dir=models_path,
            )

        try:
            lama_model = modelloader.load_spandrel_model(lama_path, device=devices.device)
        except:
            print ("Inpaint (lama) error: could not find model 'big-lama.pt'")
    else:
        lama_model.to(devices.device, dtype=torch.float32)

    with torch.no_grad():
        tensor_image = pil_image_to_torch_bgr(image).unsqueeze(0)  # add batch dimension
        tensor_image = tensor_image.to(device=devices.device, dtype=torch.float32)

        tensor_mask = pil_image_to_torch_bgr(mask.convert('1', dither=False)).unsqueeze(0)[:, 0:1, :, :]
        tensor_mask = tensor_mask.to(device=devices.device, dtype=torch.float32)

        image = torch_bgr_to_pil_image(lama_model(tensor_image, mask=tensor_mask))

    lama_model.to('cpu')
    return image


MAT_model = None
def process_MAT(image, mask):
    global MAT_model
    # https://github.com/fenglinglwb/MAT; Acly for this version of the model; github.com/light-and-ray for some implementation
    if MAT_model is None:
        MAT_path = os.path.join(models_path, 'MAT_Places512_G_fp16.safetensors')
        if not os.path.exists(MAT_path):
            MAT_path = modelloader.load_file_from_url(
                'https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors',
                model_dir=models_path,
            )

        try:
            MAT_model = modelloader.load_spandrel_model(MAT_path, device=devices.device)
        except:
            print ("Inpaint fill (MAT) error: could not find model 'MAT_Places512_G_fp16.safetensors'")
    else:
        MAT_model.to(devices.device, dtype=torch.float32)

    with torch.no_grad():
        tensor_image = pil_image_to_torch_bgr(image).unsqueeze(0)  # add batch dimension
        tensor_image = tensor_image.to(device=devices.device, dtype=torch.float32)

        tensor_mask = pil_image_to_torch_bgr(mask.convert('1', dither=False)).unsqueeze(0)[:, 0:1, :, :]
        tensor_mask = tensor_mask.to(device=devices.device, dtype=torch.float32)

        image = torch_bgr_to_pil_image(MAT_model(tensor_image, mask=tensor_mask))

    MAT_model.to('cpu')
    return image


options_templates: dict = None
opts: options.Options = None
restricted_opts: set[str] = None

sd_model = None

settings_components: dict = None
"""assigned from ui.py, a mapping on setting names to gradio components responsible for those settings"""

tab_names = []

latent_upscale_modes = {
    # "Latent":                       {"mode": "bilinear",        "antialias": False},
    "Latent (antialiased)":         {"mode": "bilinear",        "antialias": True}, # identical?
    # "Latent (bicubic)":             {"mode": "bicubic",         "antialias": False},
    # "Latent (bicubic antialiased)": {"mode": "bicubic",         "antialias": True},
    # "Latent (nearest)":             {"mode": "nearest",         "antialias": False},
    # "Latent (nearest-exact)":       {"mode": "nearest-exact",   "antialias": False},
    "Latent (NeuralNet)":           {"mode": "NNet",            "antialias": None},
}

sd_upscalers = []

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm: 'shared_total_tqdm.TotalTQDM' = None

mem_mon: 'memmon.MemUsageMonitor' = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files

# list_checkpoint_tiles = shared_items.list_checkpoint_tiles #sd_models.checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

hf_endpoint = os.getenv('HF_ENDPOINT', 'https://huggingface.co')
