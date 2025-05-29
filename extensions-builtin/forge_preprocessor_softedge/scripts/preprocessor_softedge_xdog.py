from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import resize_image_with_pad, HWC3

import numpy
import cv2


class PreprocessorSoftEdgexDoG(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'softedge_xdog'
        self.tags = ['SoftEdge']
        self.model_filename_filters = ['softedge']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)
        
        image = image.astype(numpy.float32) ** 1.1557 # this is somewhat arbitrary
        g1 = cv2.GaussianBlur(image, (0, 0), 0.5)
        g2 = cv2.GaussianBlur(image, (0, 0), 4.0)
        g3 = cv2.GaussianBlur(g2-g1, (0, 0), 1.0)
        result = (numpy.min(g3, axis=2)).clip(0, 255).astype(numpy.uint8)

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorSoftEdgexDoG())
