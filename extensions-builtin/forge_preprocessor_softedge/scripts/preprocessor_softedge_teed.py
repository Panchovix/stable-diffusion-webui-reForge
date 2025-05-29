from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices
from modules.modelloader import load_file_from_url

import os
import torch
import numpy
import cv2
from einops import rearrange
from teed import TED  # TEED architecture

## MIT License : Copyright (c) 2022 Xavier Soria Poma (https://github.com/xavysp/TEED)

def safe_step(x, step=2):
    y = x.astype(numpy.float32) * float(step + 1)
    y = y.astype(numpy.int32).astype(numpy.float32) / float(step)
    return y


class PreprocessorSoftEdgeTEED(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'softedge_teed'
        self.tags = ['SoftEdge']
        self.model_filename_filters = ['softedge']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(minimum=0, maximum=10, step=1, value=2, label='Safe steps', visible=True)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.model = None
        self.device = devices.get_device_for('controlnet')

    def load_model(self, name):
        model_dir = os.path.join(preprocessor_dir, 'TEED')
        remote_model_path = 'https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth'
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = TED()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        self.model = model


    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            self.load_model('7_model.pth')
        self.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)
        safe_steps = int(slider_1)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image).to(torch.float32).to(self.device)
            image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(numpy.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = numpy.stack(edges, axis=2)
            edge = 1 / (1 + numpy.exp(-numpy.mean(edges, axis=2).astype(numpy.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            result = (edge * 255.0).clip(0, 255).astype(numpy.uint8)

        self.model.cpu()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorSoftEdgeTEED())

