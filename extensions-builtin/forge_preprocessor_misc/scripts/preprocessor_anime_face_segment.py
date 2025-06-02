from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices
from modules.modelloader import load_file_from_url

import os
import torch
import numpy
import torchvision


##  MIT License : Copyright (c) 2021 Miaomiao Li


class AnimeFaceNet(torch.nn.Module):
    def __init__(self):
        super(AnimeFaceNet, self).__init__()
        self.NUM_SEG_CLASSES = 7 # Background, hair, face, eye, mouth, skin, clothes
        
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mob_blocks = mobilenet_v2.features
        
        # Encoder
        self.en_block0 = torch.nn.Sequential(    # in_ch=3 out_ch=16
            mob_blocks[0],  mob_blocks[1]
        )
        self.en_block1 = torch.nn.Sequential(    # in_ch=16 out_ch=24
            mob_blocks[2],  mob_blocks[3],
        )
        self.en_block2 = torch.nn.Sequential(    # in_ch=24 out_ch=32
            mob_blocks[4],  mob_blocks[5],  mob_blocks[6],
        )
        self.en_block3 = torch.nn.Sequential(    # in_ch=32 out_ch=96
            mob_blocks[7],  mob_blocks[8],  mob_blocks[9],  mob_blocks[10],
            mob_blocks[11], mob_blocks[12], mob_blocks[13],
        )
        self.en_block4 = torch.nn.Sequential(    # in_ch=96 out_ch=160
            mob_blocks[14], mob_blocks[15], mob_blocks[16],
        )
        
        # Decoder
        self.de_block4 = torch.nn.Sequential(     # in_ch=160 out_ch=96
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(160, 96, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(96),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.2)
        )
        self.de_block3 = torch.nn.Sequential(     # in_ch=96x2 out_ch=32
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(96*2, 32, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.2)
        )
        self.de_block2 = torch.nn.Sequential(     # in_ch=32x2 out_ch=24
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(32*2, 24, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(24),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.2)
        )
        self.de_block1 = torch.nn.Sequential(     # in_ch=24x2 out_ch=16
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(24*2, 16, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(16),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.2)
        )
        
        self.de_block0 = torch.nn.Sequential(     # in_ch=16x2 out_ch=7
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(16*2, self.NUM_SEG_CLASSES, kernel_size=3, padding=1),
            torch.nn.Softmax2d()
        )
        
    def forward(self, x):
        e0 = self.en_block0(x)
        e1 = self.en_block1(e0)
        e2 = self.en_block2(e1)
        e3 = self.en_block3(e2)
        e4 = self.en_block4(e3)
        
        d4 = self.de_block4(e4)
        d4 = torch.nn.functional.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        c4 = torch.cat((d4,e3),1)

        d3 = self.de_block3(c4)
        d3 = torch.nn.functional.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        c3 = torch.cat((d3,e2),1)

        d2 = self.de_block2(c3)
        d2 = torch.nn.functional.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        c2 =torch.cat((d2,e1),1)

        d1 = self.de_block1(c2)
        d1 = torch.nn.functional.interpolate(d1, size=e0.size()[2:], mode='bilinear', align_corners=True)
        c1 = torch.cat((d1,e0),1)
        y = self.de_block0(c1)
        
        return y


class PreprocessorSegmentation_AnimeFace(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'seg_anime_face'
        self.tags = ['Segmentation']
        self.model_filename_filters = ['segment']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100
        self.model = None
        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        model_dir = os.path.join(preprocessor_dir, "anime_face_segment")
        remote_model_path = 'https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/UNet.pth'
        model_path = os.path.join(model_dir, 'UNet.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = AnimeFaceNet()
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if key.startswith('module.'):
                ckpt[key[7:]] = ckpt.pop(key)
        model.load_state_dict(ckpt)
        model.eval()
        self.model = model

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, 512)
        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32).to(self.device)
        image /= 255.0

        with torch.no_grad():
            seg = self.model(image).squeeze(dim=0).permute(1, 2, 0)

        image = seg.cpu().detach().numpy()

        COLOR_BACKGROUND = (255,255,0)
        COLOR_HAIR       = (0,0,255)
        COLOR_EYE        = (255,0,0)
        COLOR_MOUTH      = (255,255,255)
        COLOR_FACE       = (0,255,0)
        COLOR_SKIN       = (0,255,255)
        COLOR_CLOTHES    = (255,0,255)
        PALETTE = [COLOR_BACKGROUND, COLOR_HAIR, COLOR_EYE, COLOR_MOUTH, COLOR_FACE, COLOR_SKIN, COLOR_CLOTHES]

        image = [[PALETTE[numpy.argmax(val)] for val in buf]for buf in image]
        result = numpy.array(image).astype(numpy.uint8)

        self.model.cpu()
        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorSegmentation_AnimeFace())
