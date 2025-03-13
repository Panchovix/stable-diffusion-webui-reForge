import os
from modules import modelloader, devices, errors
from modules.shared import opts, cmd_opts, models_path
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.forge_util import prepare_free_memory


class UpscalerPLKSR(Upscaler):
    def __init__(self, dirname=None):
        self.name = "PLKSR"
        self.model_path = os.path.join(models_path, "PLKSR")
        self.model_name = None
        self.model_url = None
        self.scalers = []
        super().__init__(create_dirs=True)
        
        model_paths = self.find_models(ext_filter=[".pt", ".pth", ".safetensors"])
        for file in model_paths:
            if "http" in file:
                continue
            name = modelloader.friendly_name(file)
            scale = None  # We could try to detect scale from filename if needed
            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        prepare_free_memory()
        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load PLKSR model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_esrgan)
        return plksr_upscale(model, img)

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        else:
            filename = path

        return modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_esrgan.type == 'mps' else None),
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture='PLKSR',
        )


def plksr_upscale(model, img):
    return upscale_with_model(
        model,
        img,
        tile_size=opts.PLKSR_tile,
        tile_overlap=opts.PLKSR_tile_overlap,
    )