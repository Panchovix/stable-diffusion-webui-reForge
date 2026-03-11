import os
import re
from functools import lru_cache
from modules import modelloader, devices, errors
from modules.shared import opts, cmd_opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.forge_util import prepare_free_memory

PREFER_HALF = opts.prefer_fp16_upscalers
if PREFER_HALF:
    print("[Upscalers] Prefer Half-Precision:", PREFER_HALF)


class UpscalerHAT(Upscaler):
    def __init__(self, dirname):
        self.name = "HAT"
        self.scalers = []
        self.user_path = dirname
        super().__init__()

        for file in self.find_models(ext_filter=[".pt", ".pth", ".safetensors"]):
            if file.startswith("http"):
                name = modelloader.friendly_name(file)
            else:
                name = modelloader.friendly_name(file)

            if match := re.search(r"(\d)[xX]|[xX](\d)", name):
                scale = int(match.group(1) or match.group(2))
            else:
                scale = 4

            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        prepare_free_memory()
        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load HAT model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_hat)
        return upscale_with_model(
            model=model,
            img=img,
            tile_size=opts.HAT_tile,
            tile_overlap=opts.HAT_tile_overlap,
        )

    @lru_cache(maxsize=4)
    def load_model(self, path: str):
        if not path.startswith("http"):
            filename = path
        else:
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=path.rsplit("/", 1)[-1],
            )

        model = modelloader.load_spandrel_model(filename, device=devices.cpu, prefer_half=PREFER_HALF)
        model.to(devices.device_hat)
        return model
    