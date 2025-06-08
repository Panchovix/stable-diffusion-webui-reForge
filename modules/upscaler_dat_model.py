import os

from modules import modelloader, errors
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_2
from modules_forge.utils import prepare_free_memory


class UpscalerDAT(Upscaler):
    def __init__(self):
        self.name = "DAT"
        self.scalers = []
        super().__init__()

        gotDAT2 = False
        gotDAT3 = False
        gotDAT4 = False
        for file in self.find_models():
            name = modelloader.friendly_name(file)
            if name == "DAT_x2":
                gotDAT2 = True
            if name == "DAT_x3":
                gotDAT3 = True
            if name == "DAT_x4":
                gotDAT4 = True
            scaler_data = UpscalerData(name, file, upscaler=self, scale=None)
            self.scalers.append(scaler_data)

        if not gotDAT2:
            DAT2 = UpscalerData(
                name="DAT_x2",
                path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x2.pth",
                scale=2,
                upscaler=self,
                sha256='7760aa96e4ee77e29d4f89c3a4486200042e019461fdb8aa286f49aa00b89b51',
            )
            self.scalers.append(DAT2)
        if not gotDAT3:
            DAT3 = UpscalerData(
                name="DAT_x3",
                path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x3.pth",
                scale=3,
                upscaler=self,
                sha256='581973e02c06f90d4eb90acf743ec9604f56f3c2c6f9e1e2c2b38ded1f80d197',
            )
            self.scalers.append(DAT3)
        if not gotDAT4:
            DAT4 = UpscalerData(
                name="DAT_x4",
                path="https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth",
                scale=4,
                upscaler=self,
                sha256='391a6ce69899dff5ea3214557e9d585608254579217169faf3d4c353caff049e',
            )
            self.scalers.append(DAT4)

    def do_upscale(self, img, path):
        prepare_free_memory()
        try:
            info = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load DAT model {path}", exc_info=True)
            return img

        model_descriptor = modelloader.load_spandrel_model(
            info.local_data_path,
            device=self.device,
        )
        return upscale_2(
            img,
            model_descriptor,
            tile_size=opts.DAT_tile,
            tile_overlap=opts.DAT_tile_overlap,
            scale=model_descriptor.scale,
            desc="Tiled upscale",
        )

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                        hash_prefix=scaler.sha256,
                    )

                    if os.path.getsize(scaler.local_data_path) < 200:
                        # Re-download if the file is too small, probably an LFS pointer
                        scaler.local_data_path = modelloader.load_file_from_url(
                            scaler.data_path,
                            model_dir=self.model_download_path,
                            hash_prefix=scaler.sha256,
                            re_download=True,
                        )

                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"DAT data missing: {scaler.local_data_path}")
                return scaler
        raise ValueError(f"Unable to find model info: {path}")
