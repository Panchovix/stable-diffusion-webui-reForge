from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from backend.memory_management import get_torch_device, should_use_fp16

import torch
import numpy as np

from diffusers import MarigoldDepthPipeline, MarigoldNormalsPipeline


class PreprocessorMarigoldDepth(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'depth_marigold'
        self.tags = ['Depth']
        self.model_filename_filters = ['depth']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=256, maximum=2048, value=768, step=64, visible=True)
        self.slider_1 = PreprocessorParameter(
            label='Steps', minimum=4, maximum=32, value=4, step=1, visible=True)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list
        self.pipeline = None
        self.keep_loaded = False # make an option?

    def load_model(self):
        device = get_torch_device()
        if self.pipeline is None:
            dtype = torch.float16 if should_use_fp16(device=device, prioritize_performance=False, manual_cast=True) else torch.float32
            self.pipeline = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=dtype)
            self.pipeline.enable_model_cpu_offload()

        return

    def __call__(self, input_image, resolution, slider_1=4, slider_2=None, slider_3=None, **kwargs):
        self.load_model()
        
        resolution = 8 * (resolution // 8)

        with torch.no_grad():
            depth = self.pipeline(input_image, num_inference_steps=slider_1, processing_resolution=resolution)

            depth_image = self.pipeline.image_processor.visualize_depth(depth.prediction, color_map="binary")
            # depth_16bit = self.pipeline.image_processor.export_depth_to_16bit_png(depth.prediction, color_map="binary")

        if self.keep_loaded:
            self.pipeline.to("cpu")
        else:
            self.pipeline = None
        torch.cuda.empty_cache()
        
        return np.array(depth_image[0])


class PreprocessorMarigoldNormal(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normal_marigold'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=256, maximum=2048, value=768, step=64, visible=True)
        self.slider_1 = PreprocessorParameter(
            label='Steps', minimum=4, maximum=32, value=4, step=1, visible=True)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100
        self.pipeline = None
        self.keep_loaded = False

    def load_model(self):
        device = get_torch_device()
        if self.pipeline is None:
            dtype = torch.float16 if should_use_fp16(device=device, prioritize_performance=False, manual_cast=True) else torch.float32
            self.pipeline = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=dtype)
            self.pipeline.enable_model_cpu_offload()

        return

    def __call__(self, input_image, resolution, slider_1=4, slider_2=None, slider_3=None, **kwargs):
        self.load_model()

        resolution = 8 * (resolution // 8)

        with torch.no_grad():
            normal = self.pipeline(input_image, num_inference_steps=slider_1, processing_resolution=resolution)

            normal_image = self.pipeline.image_processor.visualize_normals(normal.prediction)

        if self.keep_loaded:
            self.pipeline.to("cpu")
        else:
            self.pipeline = None
        torch.cuda.empty_cache()
        
        return np.array(normal_image[0])



add_supported_preprocessor(PreprocessorMarigoldDepth())
add_supported_preprocessor(PreprocessorMarigoldNormal())
