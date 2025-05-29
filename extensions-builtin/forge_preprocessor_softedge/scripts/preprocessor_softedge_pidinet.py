from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3, nms
from modules import devices
from modules.modelloader import load_file_from_url

import os
import torch
import numpy
import cv2
from einops import rearrange
from pidinet import pidinet

## MIT License  : Copyright (c) 2021 Zhuo Su
## ? MIT text was included, but license started with:
##   'It is just for research purpose, and commercial use should be contacted with authors first.'


PiDiNet_model = None

class PreprocessorSoftEdgePiDiNet(Preprocessor):
    def __init__(self, name, apply_filter, safe_steps):
        super().__init__()
        self.name = name
        self.apply_filter = apply_filter
        self.safe_steps = safe_steps
        if 't2ia' in name:
            self.tags = ['T2I-Adapter']
            self.model_filename_filters = ['t2iadapter_sketch']
        elif 'scribble' in name:
            self.tags = ['Scribble']
            self.model_filename_filters = ['scribble']
        else:
            self.tags = ['SoftEdge']
            self.model_filename_filters = ['softedge']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        global PiDiNet_model
        model_dir = os.path.join(preprocessor_dir, 'pidinet')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth'
        model_path = os.path.join(model_dir, 'table5_pidinet.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        net = pidinet()
        ckpt = torch.load(model_path)
        ckpt = ckpt['state_dict']
        for key in list(ckpt.keys()):
            if key.startswith('module.'):
                ckpt[key[7:]] = ckpt.pop(key)
        net.load_state_dict(ckpt)

        net.eval()
        PiDiNet_model = net

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        global PiDiNet_model
        if PiDiNet_model is None:
            self.load_model()
        PiDiNet_model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)
        steps = int(slider_1)

        image = image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(image).to(torch.float32).to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = PiDiNet_model(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if self.apply_filter:
                edge = edge > 0.5 
            if self.safe_steps:
                y = edge.astype(numpy.float32) * float(2 + 1)
                edge = y.astype(numpy.int32).astype(numpy.float32) / float(2)

            edge = (edge * 255.0).clip(0, 255).astype(numpy.uint8)
            
        result = edge[0][0] 

        PiDiNet_model.cpu()

        if 'scribble' in self.name:
            result = nms(result, 127, 3.0)
            result = cv2.GaussianBlur(result, (0, 0), 3.0)
            result[result > 4] = 255
            result[result < 255] = 0

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorSoftEdgePiDiNet('softedge_pidinet',  False, False))
add_supported_preprocessor(PreprocessorSoftEdgePiDiNet('softedge_pidisafe', False, True))
add_supported_preprocessor(PreprocessorSoftEdgePiDiNet('t2ia_sketch_pidi',  True, False))
add_supported_preprocessor(PreprocessorSoftEdgePiDiNet('scribble_pidinet',  False, False))

