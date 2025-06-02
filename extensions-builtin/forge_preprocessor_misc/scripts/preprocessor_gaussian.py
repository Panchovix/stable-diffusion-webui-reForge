from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import resize_image_with_pad, HWC3

import cv2


class PreprocessorBlur(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'blur_gaussian'
        self.tags = ['Tile']
        self.model_filename_filters = ['tile']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(label='Sigma', minimum=0.01, maximum=64.0, step=0.01, value=9.0, visible=True)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=9.0, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)
        image = remove_pad(image)

        result = cv2.GaussianBlur(image, (0, 0), slider_1)

        return HWC3(result)


add_supported_preprocessor(PreprocessorBlur())
