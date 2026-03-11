import os
from functools import lru_cache
from modules import modelloader, devices, errors
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.forge_util import prepare_free_memory

PREFER_HALF = opts.prefer_fp16_upscalers
if PREFER_HALF:
    print("[Upscalers] Prefer Half-Precision:", PREFER_HALF)
class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN"
        self.user_path = path
        super().__init__()
        self.enable = True
        self.scalers = []

        scalers = get_realesrgan_models(self)
        local_model_paths = self.find_models(ext_filter=[".pth", ".safetensors"])

        for scaler in scalers:
            if scaler.local_data_path.startswith("http"):
                filename = modelloader.friendly_name(scaler.local_data_path)
                local_model_candidates = [p for p in local_model_paths if p.endswith(f"{filename}.pth")]
                if local_model_candidates:
                    scaler.local_data_path = local_model_candidates[0]
            if scaler.name in opts.realesrgan_enabled_models:
                self.scalers.append(scaler)

    def do_upscale(self, img, path):
        prepare_free_memory()
        if not self.enable:
            return img
        try:
            local_path = self.load_model(path)
            model_descriptor = self._load_spandrel(local_path)
        except Exception:
            errors.report(f"Unable to load RealESRGAN model {path}", exc_info=True)
            return img
        return upscale_with_model(
            model_descriptor,
            img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
        )

    def load_model(self, path) -> str:
        """Resolve data_path → local file path, downloading if needed. Returns local path string."""
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"RealESRGAN data missing: {scaler.local_data_path}")
                return scaler.local_data_path
        raise ValueError(f"Unable to find model info: {path}")

    @lru_cache(maxsize=4)
    def _load_spandrel(self, local_path: str):
        model = modelloader.load_spandrel_model(
            local_path,
            device=devices.cpu,
            prefer_half=opts.prefer_fp16_upscalers,
        )
        model.to(devices.device_esrgan)
        return model


def get_realesrgan_models(scaler: UpscalerRealESRGAN):
    return [
        UpscalerData(
            name="R-ESRGAN General 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN General WDN 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN AnimeVideo",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 4x+ Anime6B",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4,
            upscaler=scaler,
        ),
        UpscalerData(
            name="R-ESRGAN 2x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            scale=2,
            upscaler=scaler,
        ),
    ]