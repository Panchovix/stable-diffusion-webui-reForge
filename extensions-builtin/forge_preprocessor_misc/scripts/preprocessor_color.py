from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import HWC3

import cv2


def cv2_resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


class PreprocessorColour(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 't2ia_color_grid'
        self.tags = ['T2I-Adapter']
        self.model_filename_filters = ['T2IAdapter_Color']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image = cv2_resize_shortest_edge(input_image, resolution)
        h, w = image.shape[:2]

        image  = cv2.resize(image, (w//64, h//64), interpolation=cv2.INTER_CUBIC)  
        result = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

        return HWC3(result)


add_supported_preprocessor(PreprocessorColour())
