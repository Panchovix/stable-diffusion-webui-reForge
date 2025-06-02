from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices

import os
import torch
import numpy
import contextlib

Extra = lambda x: contextlib.nullcontext()
def torch_handler(module: str, name: str):
    """ Allow all torch access. Bypass A1111 safety whitelist. """
    if module == 'torch':
        return getattr(torch, name)
    if module == 'torch._tensor':
        # depth_anything dep.
        return getattr(torch._tensor, name)

class PreprocessorHandRefiner(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'depth_hand_refiner'
        self.tags = ['Depth']
        self.model_filename_filters = ['depth']
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100
        self.model = None
        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "hand_refiner")

        from hand_refiner import MeshGraphormerDetector  # installed via hand_refiner_portable
        with Extra(torch_handler):
            self.model = MeshGraphormerDetector.from_pretrained(
                "hr16/ControlNet-HandRefiner-pruned",
                cache_dir=model_dir,
                device=self.device,
            )

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)
        with Extra(torch_handler):
            depth_map, mask, info = self.model(
                image, output_type="np",
                detect_resolution=resolution,
                mask_bbox_padding=30,
            )

        self.model.to('cpu')
        torch.cuda.empty_cache()

        return HWC3(remove_pad(depth_map))


add_supported_preprocessor(PreprocessorHandRefiner())
