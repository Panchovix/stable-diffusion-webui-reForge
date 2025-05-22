from modules import modelloader
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules_forge.utils import prepare_free_memory

# this module for various Spandrel supported upscalers, that don't need their own special loader
# default storage location: 'models/upscaler'
# replaces 'hat_model' and 'esrgan_model'

class UpscalerGeneric(Upscaler):
    def __init__(self):
        self.name = "upscaler"
        self.scalers = []
        super().__init__()
        for file in self.find_models():
            name = modelloader.friendly_name(file)
            scale = 4  # TODO: scale might not be 4, but we can't know without loading the model
            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        prepare_free_memory()
        try:
            model = self.load_model(selected_model)
        except Exception as e:
            print(f"Unable to load upscaler model {selected_model}: {e}")
            return img
        model.to(self.device)
        return upscale_with_model(
            model,
            img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
        )

    def load_model(self, path: str):
        return modelloader.load_spandrel_model(
            path,
            device=self.device,
        )
