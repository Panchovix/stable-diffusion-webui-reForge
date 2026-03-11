import os, re
from functools import lru_cache
from modules import modelloader, devices, errors
from modules.shared import opts, cmd_opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.forge_util import prepare_free_memory

PREFER_HALF = opts.prefer_fp16_upscalers
if PREFER_HALF:
    print("[Upscalers] Prefer Half-Precision:", PREFER_HALF)


class UpscalerESRGAN(Upscaler):
    def __init__(self, dirname: str):
        self.user_path = dirname
        self.model_path = dirname
        super().__init__(True)

        self.name = "ESRGAN"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        self.model_name = "ESRGAN"
        self.scalers = []

        model_paths = self.find_models(ext_filter=[".pt", ".pth", ".safetensors"])
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            self.scalers.append(scaler_data)

        for file in model_paths:
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)

            if match := re.search(r"(\d)[xX]|[xX](\d)", name):
                scale = int(match.group(1) or match.group(2))
            else:
                scale = 4

            scaler_data = UpscalerData(name, file, self, scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        prepare_free_memory()
        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load ESRGAN model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_esrgan)
        return upscale_with_model(
            model=model,
            img=img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
        )

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        else:
            filename = path

        return modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_esrgan.type == 'mps' else None),
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture='ESRGAN',
        )


    @lru_cache(maxsize=4, typed=False)
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
        model.to(devices.device_esrgan)
        return model
