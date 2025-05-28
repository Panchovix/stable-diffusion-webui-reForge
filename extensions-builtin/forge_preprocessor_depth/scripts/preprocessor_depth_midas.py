from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.utils import resize_image_with_pad, HWC3
from modules import devices
from modules.modelloader import load_file_from_url

import os
import torch
import numpy
import cv2
import timm
import types
from einops import rearrange

##  MIT License : Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)


class Interpolate(torch.nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit_custom(torch.nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn
        self.groups = 1
        self.conv1 = torch.nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = torch.nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = torch.nn.BatchNorm2d(features)
            self.bn2 = torch.nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(torch.nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = torch.nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = torch.nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class ProjectReadout(torch.nn.Module):
    def __init__(self):
        super(ProjectReadout, self).__init__()
        self.start_index = 1
        self.project = torch.nn.Sequential(torch.nn.Linear(2 * 768, 768), torch.nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(len(posemb_grid) ** 0.5)

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def _make_vit_b_rn50_backbone(model):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output

        return hook

    pretrained = torch.nn.Module()

    pretrained.model = model

    pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(get_activation("1"))
    pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(get_activation("2"))

    pretrained.model.blocks[8].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[11].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = [ProjectReadout(), ProjectReadout(), ProjectReadout(), ProjectReadout()]

    pretrained.act_postprocess1 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity())
    pretrained.act_postprocess2 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity())

    pretrained.act_postprocess3 = torch.nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        torch.nn.Unflatten(2, torch.Size([384 // 16, 384 // 16])),
        torch.nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = torch.nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        torch.nn.Unflatten(2, torch.Size([384 // 16, 384 // 16])),
        torch.nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        torch.nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = 1
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def forward_vit(pretrained, x):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = torch.nn.Sequential(
        torch.nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    layer_3 = unflatten(layer_3)
    layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4

class DPT(torch.nn.Module):
    def __init__(
        self,
        head,
        features=256,
        use_bn=False,
    ):
        super(DPT, self).__init__()

        # Instantiate backbone and reassemble blocks
        model = timm.create_model("vit_base_resnet50_384", pretrained=False)
        self.pretrained = _make_vit_b_rn50_backbone(model)

        self.scratch = torch.nn.Module()

        self.scratch.layer1_rn = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.scratch.layer2_rn = torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.scratch.layer3_rn = torch.nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.scratch.layer4_rn = torch.nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

        self.scratch.refinenet1 = FeatureFusionBlock_custom(features, torch.nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features, torch.nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features, torch.nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        self.scratch.refinenet4 = FeatureFusionBlock_custom(features, torch.nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)

        self.scratch.output_conv = head

    def forward(self, x):
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out

class DPTDepthModel(DPT):
    def __init__(self, path=None, **kwargs):
        head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Identity(),
        )

        super().__init__(head, **kwargs)

        parameters = torch.load(path, map_location=torch.device('cpu'))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)


MiDas_model = None


class PreprocessorMiDas(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'depth_midas':
            self.tags = ['Depth']
            self.model_filename_filters = ['depth']
            self.slider_1 = PreprocessorParameter(visible=False)
        elif name == 'normal_midas':
            self.tags = ['NormalMap']
            self.model_filename_filters = ['normal']
            self.slider_1 = PreprocessorParameter(label='Background threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=True)

        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.device = devices.get_device_for('controlnet')

        self.cache = None
        self.cacheHash = None

    def load_model(self):
        global MiDas_model
        model_dir = os.path.join(preprocessor_dir, "midas")
        remote_model_path = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt'
        model_path = os.path.join(model_dir, 'dpt_hybrid-midas-501f0c75.pt')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = DPTDepthModel(path=model_path,
            # backbone="vitb_rn50_384",
        )

        model.eval()
        MiDas_model = model

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        global MiDas_model
        if MiDas_model is None:
            self.load_model()
        MiDas_model.to(self.device)

        image, remove_pad = resize_image_with_pad(input_image, resolution)

        image = torch.from_numpy(image).to(torch.float32).to(self.device)
        image = image / 127.5 - 1.0
        image = rearrange(image, 'h w c -> 1 c h w')

        with torch.no_grad():
            depth = MiDas_model(image)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()

        if self.name == 'depth_midas':
            result = (depth_pt * 255.0).clip(0, 255).astype(numpy.uint8)

        elif self.name == 'normal_midas':
            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = numpy.ones_like(x) * numpy.pi * 2.0
            x[depth_pt < slider_1] = 0
            y[depth_pt < slider_1] = 0
            normal = numpy.stack([x, y, z], axis=2)
            normal /= numpy.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            result = (normal * 127.5 + 127.5).clip(0, 255).astype(numpy.uint8)#[:, :, ::-1]

        MiDas_model.cpu()
        torch.cuda.empty_cache()

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorMiDas('depth_midas'))
add_supported_preprocessor(PreprocessorMiDas('normal_midas'))
