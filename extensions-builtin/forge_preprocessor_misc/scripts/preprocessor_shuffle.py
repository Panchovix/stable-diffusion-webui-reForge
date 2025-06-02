from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.utils import resize_image_with_pad, HWC3

import cv2
import numpy


def make_noise_disk(H, W, C, F):
    noise = numpy.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= numpy.min(noise)
    noise /= numpy.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

class PreprocessorShuffle(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'shuffle'
        self.tags = ['Misc']
        self.model_filename_filters = ['shuffle']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(minimum=16, maximum=512, step=16, value=256, label='f', visible=True)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)
        image = remove_pad(image)

        H, W, C = image.shape
        f = slider_1

        x = make_noise_disk(H, W, 1, f) * float(W - 1)
        y = make_noise_disk(H, W, 1, f) * float(H - 1)
        flow = numpy.concatenate([x, y], axis=2).astype(numpy.float32)
        result = cv2.remap(image, flow, None, cv2.INTER_LINEAR)

        return HWC3(result)


add_supported_preprocessor(PreprocessorShuffle())
