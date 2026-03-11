from __future__ import annotations

import importlib
import logging
import os
from urllib.parse import urlparse

import torch

from modules import shared
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone
from modules.util import load_file_from_url  # noqa, backwards compatibility
from ldm_patched.modules.utils import load_torch_file
import spandrel

logger = logging.getLogger(__name__)


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None, hash_prefix=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @param hash_prefix: the expected sha256 of the model_url
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=places[0], file_name=download_name, hash_prefix=hash_prefix))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name


def load_upscalers():
    from modules.esrgan_model   import UpscalerESRGAN
    from modules.rcan_model     import UpscalerRCAN
    from modules.plksr_model    import UpscalerPLKSR
    from modules.dat_model      import UpscalerDAT
    from modules.hat_model      import UpscalerHAT
    from modules.srformer_model import UpscalerSRFormer
    from modules.grl_model      import UpscalerGRL
    from modules.omnisr_model   import UpscalerOmniSR
    from modules.span_model     import UpscalerSPAN
    from modules.compact_model  import UpscalerCOMPACT

    del shared.sd_upscalers

    upscaler_classes = [
        (UpscalerESRGAN,   'esrgan_models_path'),
        (UpscalerRCAN,     'rcan_models_path'),
        (UpscalerPLKSR,    'plksr_models_path'),
        (UpscalerDAT,      'dat_models_path'),
        (UpscalerHAT,      'hat_models_path'),
        (UpscalerSRFormer, 'srformer_models_path'),
        (UpscalerGRL,      'grl_models_path'),
        (UpscalerOmniSR,   'omnisr_models_path'),
        (UpscalerSPAN,     'span_models_path'),
        (UpscalerCOMPACT,  'compact_models_path'),
    ]

    all_scalers = []
    for cls, cmd_opt_name in upscaler_classes:
        commandline_model_path = getattr(shared.cmd_opts, cmd_opt_name, None)
        upscaler = cls(commandline_model_path)
        upscaler.user_path = commandline_model_path
        upscaler.model_download_path = commandline_model_path or upscaler.model_path
        all_scalers.extend(upscaler.scalers)

    shared.sd_upscalers = [
        *UpscalerNone().scalers,
        *UpscalerLanczos().scalers,
        *UpscalerNearest().scalers,
        *sorted(all_scalers, key=lambda s: s.name.lower()),
    ]


def load_spandrel_model(path: os.PathLike, device: torch.device | None, prefer_half: bool = False, *args, **kwargs) -> spandrel.ImageModelDescriptor:
    sd = load_torch_file(path, safe_load=True, device=device)
    model_descriptor = spandrel.ModelLoader(device=device).load_from_state_dict(sd)

    arch = model_descriptor.architecture
    logger.info(f'Loaded {arch.name} Model: "{os.path.basename(path)}"')

    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.half()
        elif model_descriptor.supports_bfloat16:
            model_descriptor.bfloat16()
        else:
            logger.warning(f"Model {path} does not support half precision...")

    model_descriptor.eval()
    return model_descriptor
