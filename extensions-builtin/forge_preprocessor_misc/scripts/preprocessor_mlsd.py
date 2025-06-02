from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices
from modules.modelloader import load_file_from_url

import os
import torch
import numpy
import cv2


'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
first modified by lihaoweicv (pytorch version)
'''

def decode_output_score_and_ptss(tpMap, topk_n = 200, ksize = 5):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert  b==1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = torch.nn.functional.max_pool2d( heat, (ksize, ksize), stride=1, padding=(ksize-1)//2)
    keep = (hmax == heat).to(torch.float32)
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx),dim=-1)

    ptss   = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1,2,0))
    return  ptss, scores, displacement


def pred_lines(image, model, device,
               score_thr=0.10,
               dist_thr=20.0):
    h, w, _ = image.shape

    resized_image = numpy.concatenate([image, numpy.ones([h, w, 1])], axis=-1)

    resized_image = resized_image.transpose((2,0,1))
    batch_image = numpy.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).to(torch.float32).to(device)
    outputs = model(batch_image)
    pts, pts_score, vmap = decode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = numpy.sqrt(numpy.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * numpy.array(segments_list)  # 256 > 512

    return lines


class BlockTypeA(torch.nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c2, out_c2, kernel_size=1),
            torch.nn.BatchNorm2d(out_c2),
            torch.nn.ReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c1, out_c1, kernel_size=1),
            torch.nn.BatchNorm2d(out_c1),
            torch.nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
             b = torch.nn.functional.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(in_c),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            torch.nn.BatchNorm2d(in_c),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(in_c),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride

        # TFLite uses slightly different padding than PyTorch
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU6(inplace=True)
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=stride, stride=stride)


    def forward(self, x):
        # TFLite uses  different padding
        if self.stride == 2:
            x = torch.nn.functional.pad(x, (0, 1, 0, 1), "constant", 0)

        for module in self:
            if not isinstance(module, torch.nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        ])
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(torch.nn.Module):
    def __init__(self):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            inverted_residual_setting: Network structure
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        # building first layer
        features = [ConvBNReLU(4, 32, stride=2)]
        # building inverted residual blocks
        features.append(InvertedResidual(32, 16, 1, expand_ratio=1))

        features.append(InvertedResidual(16, 24, 2, expand_ratio=6))
        features.append(InvertedResidual(24, 24, 1, expand_ratio=6))

        features.append(InvertedResidual(24, 32, 2, expand_ratio=6))
        features.append(InvertedResidual(32, 32, 1, expand_ratio=6))
        features.append(InvertedResidual(32, 32, 1, expand_ratio=6))

        features.append(InvertedResidual(32, 64, 2, expand_ratio=6))
        features.append(InvertedResidual(64, 64, 1, expand_ratio=6))
        features.append(InvertedResidual(64, 64, 1, expand_ratio=6))
        features.append(InvertedResidual(64, 64, 1, expand_ratio=6))

        features.append(InvertedResidual(64, 96, 1, expand_ratio=6))
        features.append(InvertedResidual(96, 96, 1, expand_ratio=6))
        features.append(InvertedResidual(96, 96, 1, expand_ratio=6))

        self.features = torch.nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]

        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        return fpn_features


    def forward(self, x):
        return self._forward_impl(x)


class MobileV2_MLSD_Large(torch.nn.Module):
    def __init__(self):
        super(MobileV2_MLSD_Large, self).__init__()

        self.backbone = MobileNetV2()
        ## A, B
        self.block15 = BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)

        ## A, B
        self.block17 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)

        ## A, B
        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)

        ## A, B, C
        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)

        self.block23 = BlockTypeC(64, 16)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)

        x = self.block15(c4, c5)
        x = self.block16(x)

        x = self.block17(c3, x)
        x = self.block18(x)

        x = self.block19(c2, x)
        x = self.block20(x)

        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]

        return x

class PreprocessorMLSD(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['MLSD']
        self.model_filename_filters = ['mlsd']
        self.slider_1 = PreprocessorParameter(label='Value threshold',    minimum=0.01, maximum=2.0,  value=0.1, step=0.01, visible=True)
        self.slider_2 = PreprocessorParameter(label='Distance threshold', minimum=0.01, maximum=20.0, value=0.1, step=0.01, visible=True)
        self.sorting_priority = 100
        self.model = None
        self.device = devices.get_device_for('controlnet')
        self.use_soft_projection_in_hr_fix = True

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "mlsd")
        remote_model_path = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth'
        model_path = os.path.join(model_dir, 'mlsd_large_512_fp32.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = MobileV2_MLSD_Large()

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        self.model = model

    def __call__(self, input_image, resolution=518, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        # if self.name == 'mlsd_inverted':
            # image = 255 - image

        output = numpy.zeros_like(image)
        with torch.no_grad():
            lines = pred_lines(image, self.model, self.device, slider_1, slider_2)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        result = output[:, :, 0]

        self.model.cpu()
        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorMLSD('mlsd'))
# add_supported_preprocessor(PreprocessorMLSD('mlsd_inverted'))
