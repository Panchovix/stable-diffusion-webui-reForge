from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules import devices
from modules_forge.utils import resize_image_with_pad, HWC3
from modules.modelloader import load_file_from_url

import torch
import numpy
import cv2
import os
from einops import rearrange


##  realistic and coarse models
## MIT License : Copyright (c) 2022 Caroline Chan
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(torch.nn.ReflectionPad2d(1),
                                              torch.nn.Conv2d(in_features, in_features, 3),
                                              torch.nn.InstanceNorm2d(in_features),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.ReflectionPad2d(1),
                                              torch.nn.Conv2d(in_features, in_features, 3),
                                              torch.nn.InstanceNorm2d(in_features)
                                             )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        self.model0 = torch.nn.Sequential(torch.nn.ReflectionPad2d(3),
                                          torch.nn.Conv2d(input_nc, 64, 7),
                                          torch.nn.InstanceNorm2d(64),
                                          torch.nn.ReLU(inplace=True)
                                         )

        # Downsampling
        self.model1 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                          torch.nn.InstanceNorm2d(128),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                          torch.nn.InstanceNorm2d(256),
                                          torch.nn.ReLU(inplace=True)
                                         )

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(256)]
        self.model2 = torch.nn.Sequential(*model2)

        # Upsampling
        self.model3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                          torch.nn.InstanceNorm2d(128),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                          torch.nn.InstanceNorm2d(64),
                                          torch.nn.ReLU(inplace=True)
                                         )

        # Output layer
        model4 = [  torch.nn.ReflectionPad2d(3),
                    torch.nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [torch.nn.Sigmoid()]

        self.model4 = torch.nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out
##  end: realistic and coarse models


## anime
## MIT License : Copyright (c) 2022 Caroline Chan
class UnetGenerator(torch.nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)  # add the innermost layer
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(torch.nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        use_bias = True

        if input_nc is None:
            input_nc = outer_nc
        downconv = torch.nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.InstanceNorm2d(inner_nc)
        uprelu = torch.nn.ReLU(True)
        upnorm = torch.nn.InstanceNorm2d(outer_nc)

        if outermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, torch.nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
##  end: anime


## manga (anime_denoised)
## MIT License: Copyright (c) 2021 Miaomiao Li
class _bn_relu_conv(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_filters, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)


class _u_bn_relu_conv(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_filters, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            torch.nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)


class _shortcut(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        if self.process:
            y0 = self.model(x)
            return y0 + y
        else:
            return x + y


class _u_shortcut(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                torch.nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)


class _u_basic_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(torch.nn.Module):
    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y
##  end: manga


class PreprocessorLineart(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['Lineart']
        self.model_filename_filters = ['lineart']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=100, label='Low Threshold', visible=True)
        self.slider_2 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=200, label='High Threshold', visible=True)
        self.sorting_priority = 100

        self.model = None
        self.device = devices.get_device_for('controlnet')
        
        self.cache = None
        self.cacheHash = None

    def load_model(self, name):
        model_dir = os.path.join(preprocessor_dir, 'lineart')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/' + name
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        self.model = model

    def load_anime_model(self):
        model_dir = os.path.join(preprocessor_dir, 'lineart_anime')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth'
        model_path = os.path.join(model_dir, 'netG.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        net = UnetGenerator(3, 1, 8, 64, use_dropout=False)
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if key.startswith('module.'):
                ckpt[key[7:]] = ckpt.pop(key)
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net

    def load_manga_model(self):
        model_dir = os.path.join(preprocessor_dir, 'manga_line')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth'
        model_path = os.path.join(model_dir, 'erika.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        net = res_skip()
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if key.startswith('module.'):
                ckpt[key[7:]] = ckpt.pop(key)
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net


    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)

        match self.name:
            case 'lineart_standard (from white bg & black line)':
                x = image.astype(numpy.float32)
                g = cv2.GaussianBlur(x, (0, 0), 6.0)
                intensity = numpy.min(g - x, axis=2).clip(0, 255)
                intensity /= max(16, numpy.median(intensity[intensity > 8]))
                intensity *= 127
                result = intensity.clip(0, 255).astype(numpy.uint8)

            case 'lineart_inverted (from black bg & white line)':
                image = 255 - image
                x = image.astype(numpy.float32)
                g = cv2.GaussianBlur(x, (0, 0), 6.0)
                intensity = numpy.min(g - x, axis=2).clip(0, 255)
                intensity /= max(16, numpy.median(intensity[intensity > 8]))
                intensity *= 127
                result = intensity.clip(0, 255).astype(numpy.uint8)

            case 'lineart_coarse':
                if self.model is None:
                    self.load_model('sk_model2.pth')
                self.model.to(self.device)

                with torch.no_grad():
                    image = torch.from_numpy(image).to(torch.float32).to(self.device)
                    image = image / 255.0
                    image = rearrange(image, 'h w c -> 1 c h w')
                    line = self.model(image)[0][0]
                    line = line.cpu().numpy()
                    result = 255 - (line * 255.0).clip(0, 255).astype(numpy.uint8)

                self.model.to('cpu')

            case 'lineart_realistic':
                if self.model is None:
                    self.load_model('sk_model.pth')
                self.model.to(self.device)

                with torch.no_grad():
                    image = torch.from_numpy(image).to(torch.float32).to(self.device)
                    image = image / 255.0
                    image = rearrange(image, 'h w c -> 1 c h w')
                    line = self.model(image)[0][0]
                    line = line.cpu().numpy()
                    result = 255 - (line * 255.0).clip(0, 255).astype(numpy.uint8)

                self.model.to('cpu')

            case 'lineart_anime':
                if self.model is None:
                    self.load_anime_model()
                self.model.to(self.device)

                H, W, C = image.shape
                Hn = 256 * int(numpy.ceil(float(H) / 256.0))
                Wn = 256 * int(numpy.ceil(float(W) / 256.0))
                image = cv2.resize(image, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
                with torch.no_grad():
                    image = torch.from_numpy(image).to(torch.float32).to(self.device)
                    image = image / 127.5 - 1.0
                    image = rearrange(image, 'h w c -> 1 c h w')
                    line = self.model(image)[0, 0] * 127.5 + 127.5
                    line = line.cpu().numpy()
                    line = cv2.resize(line, (W, H), interpolation=cv2.INTER_CUBIC)
                    result = 255 - line.clip(0, 255).astype(numpy.uint8)

                self.model.cpu()

            case 'lineart_anime_denoised':
                if self.model is None:
                    self.load_manga_model()
                self.model.to(self.device)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = numpy.ascontiguousarray(image)
                with torch.no_grad():
                    image = torch.from_numpy(image).to(torch.float32).to(self.device)
                    image = rearrange(image, 'h w -> 1 1 h w')
                    line = self.model(image)[0, 0]
                    line = line.cpu().numpy()
                    result = 255 - line.clip(0, 255).astype(numpy.uint8)


            case _:
                return input_image

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorLineart('lineart_standard (from white bg & black line)'))
add_supported_preprocessor(PreprocessorLineart('lineart_inverted (from black bg & white line)'))
add_supported_preprocessor(PreprocessorLineart('lineart_realistic'))
add_supported_preprocessor(PreprocessorLineart('lineart_coarse'))
add_supported_preprocessor(PreprocessorLineart('lineart_anime'))
add_supported_preprocessor(PreprocessorLineart('lineart_anime_denoised'))
