import os
import re
from functools import lru_cache
from modules import modelloader, devices, errors
from modules.shared import cmd_opts, opts, hf_endpoint
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.forge_util import prepare_free_memory

PREFER_HALF = opts.prefer_fp16_upscalers
if PREFER_HALF:
    print("[Upscalers] Prefer Half-Precision:", PREFER_HALF)


class UpscalerDAT(Upscaler):
    def __init__(self, user_path):
        self.name = "DAT"
        self.user_path = user_path
        self.scalers = []
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

        for model in get_dat_models(self):
            if model.name in opts.dat_enabled_models:
                self.scalers.append(model)

    def do_upscale(self, img, path):
        prepare_free_memory()
        try:
            model = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load DAT model {path}", exc_info=True)
            return img
        model.to(devices.device_esrgan)
        return upscale_with_model(
            model=model,
            img=img,
            tile_size=opts.DAT_tile,
            tile_overlap=opts.DAT_tile_overlap,
        )

    @lru_cache(maxsize=4)
    def load_model(self, path: str):
        # Resolve UpscalerData entry (needed for bundled DAT models with sha256)
        local_path = path
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                        hash_prefix=scaler.sha256,
                    )
                    if os.path.getsize(scaler.local_data_path) < 200:
                        # Re-download if the file is too small (LFS pointer)
                        scaler.local_data_path = modelloader.load_file_from_url(
                            scaler.data_path,
                            model_dir=self.model_download_path,
                            hash_prefix=scaler.sha256,
                            re_download=True,
                        )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"DAT data missing: {scaler.local_data_path}")
                local_path = scaler.local_data_path
                break
        else:
            # Plain local file passed directly
            if path.startswith("http"):
                local_path = modelloader.load_file_from_url(
                    url=path,
                    model_dir=self.model_download_path,
                    file_name=path.rsplit("/", 1)[-1],
                )
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f"Model file {local_path} not found")

        model = modelloader.load_spandrel_model(local_path, device=devices.cpu, prefer_half=PREFER_HALF)
        model.to(devices.device_esrgan)
        return model


def get_dat_models(scaler):
    return [
        UpscalerData(
            name="DAT x2",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x2.pth",
            scale=2,
            upscaler=scaler,
            sha256='7760aa96e4ee77e29d4f89c3a4486200042e019461fdb8aa286f49aa00b89b51',
        ),
        UpscalerData(
            name="DAT x3",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x3.pth",
            scale=3,
            upscaler=scaler,
            sha256='581973e02c06f90d4eb90acf743ec9604f56f3c2c6f9e1e2c2b38ded1f80d197',
        ),
        UpscalerData(
            name="DAT x4",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth",
            scale=4,
            upscaler=scaler,
            sha256='391a6ce69899dff5ea3214557e9d585608254579217169faf3d4c353caff049e',
        ),
    ]
