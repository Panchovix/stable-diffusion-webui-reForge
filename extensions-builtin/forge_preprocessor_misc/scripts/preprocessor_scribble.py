from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import resize_image_with_pad, HWC3

import numpy
import cv2


class PreprocessorScribble(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['Scribble']
        self.model_filename_filters = ['scribble']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(minimum=1, maximum=64, step=1, value=32, label='xDoG threshold', visible=True)
            
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=32, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)

        if 'inverted' in self.name:
            image = 255 - image

        g1 = cv2.GaussianBlur(image.astype(numpy.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(image.astype(numpy.float32), (0, 0), 5.0)
        dog = (255 - numpy.min(g2 - g1, axis=2)).clip(0, 255).astype(numpy.uint8)
        result = numpy.zeros_like(image, dtype=numpy.uint8)
        result[2 * (255 - dog) > slider_1] = 255

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorScribble('scribble_xdog'))
add_supported_preprocessor(PreprocessorScribble('scribble_inverted'))
