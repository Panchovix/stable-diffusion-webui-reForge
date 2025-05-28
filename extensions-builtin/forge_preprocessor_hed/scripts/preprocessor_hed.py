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


HED_model = None

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class PreprocessorHED(Preprocessor):
    def __init__(self, name, safe_steps):
        super().__init__()
        self.name = name
        self.safe_steps = safe_steps
        if 'softedge' in name:
            self.tags = ['SoftEdge']
            self.model_filename_filters = ['softedge']
        else:
            self.tags = ['Scribble']
            self.model_filename_filters = ['scribble']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.device = devices.get_device_for('controlnet')

    def load_model(self):
        global HED_model
        model_dir = os.path.join(preprocessor_dir, 'hed')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth'
        model_path = os.path.join(model_dir, 'ControlNetHED.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        netNetwork = ControlNetHED_Apache2()
        netNetwork.load_state_dict(torch.load(model_path, map_location='cpu'))
        netNetwork.to(torch.float32)

        netNetwork.eval()
        HED_model = netNetwork


    def __call__(self, input_image, resolution, slider_1=32, slider_2=None, slider_3=None, **kwargs):
        global HED_model
        if HED_model is None:
            self.load_model()
        HED_model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        H, W, C = image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(image).to(torch.float32).to(self.device)
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edges = HED_model(image_hed)
            edges = [e.detach().cpu().numpy().astype(numpy.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = numpy.stack(edges, axis=2)
            edge = 1 / (1 + numpy.exp(-numpy.mean(edges, axis=2).astype(numpy.float64)))
            if self.safe_steps:
                y = edge.astype(numpy.float32) * float(2 + 1)
                edge = y.astype(numpy.int32).astype(numpy.float32) / float(2)
            result = (edge * 255.0).clip(0, 255).astype(numpy.uint8)

        if self.name == 'scribble_hed':
            result = nms(result, 127, 3.0)
            result = cv2.GaussianBlur(result, (0, 0), 3.0)
            result[result > 4] = 255
            result[result < 255] = 0

        HED_model.cpu()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorHED('scribble_hed',     False))
add_supported_preprocessor(PreprocessorHED('softedge_hed',     False))
add_supported_preprocessor(PreprocessorHED('softedge_hedsafe', True))
