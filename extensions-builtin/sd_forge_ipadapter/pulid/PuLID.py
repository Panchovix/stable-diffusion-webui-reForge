# lightly modified from https://github.com/cubiq/PuLID_ComfyUI/
# Apache License 2.0

# v1.1 support from PR #125, by vitrun (github.com/vitrun)

import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import math

from modules.paths_internal import models_path

from backend import memory_management
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from backend.attention import attention_basic
from backend.misc.image_resize import contrast_adaptive_sharpening

from pulid.IDEncoder import IDEncoder
from pulid.IDFormer import IDFormer


class PulidModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        self.ip_layers = To_KV(model["ip_adapter"])
    
    def init_id_adapter(self):
        image_proj_model = IDEncoder()
        return image_proj_model

    def get_image_embeds(self, face_embed, clip_embeds):
        embeds = self.image_proj_model(face_embed, clip_embeds)
        return embeds

class PulidModelV1_1(PulidModel):

    def init_id_adapter(self):
        image_proj_model = IDFormer()
        return image_proj_model


class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    out = out[..., [2, 1, 0]]
    out = out.numpy()

    return out

def NPToTensor(image):
    out = torch.from_numpy(image)
    out = torch.clamp(out.to(torch.float) / 255., 0.0, 1.0)
    out = out[..., [2, 1, 0]]

    return out



def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]
    
    return source

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]#.copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    # else:
        # to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    # else:
        # to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(pulid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(pulid_attention, **patch_kwargs)

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]
    
    def add(self, callback, **kwargs):          
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = attention_basic(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])
        
        return out.to(dtype=dtype)

def pulid_attention(out, q, k, v, extra_options, module_key='', pulid=None, cond=None, uncond=None, weight=1.0, ortho=False, ortho_v2=False, mask=None, **kwargs):
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    dtype = q.dtype
    seq_len = q.shape[1]
    cond_or_uncond = extra_options["cond_or_uncond"]
    b = q.shape[0]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    k_cond = pulid.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
    k_uncond = pulid.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
    v_cond = pulid.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
    v_uncond = pulid.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)
    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    out_ip = attention_basic(q, ip_k, ip_v, extra_options["n_heads"])

    if ortho:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        divisor = 1e-6 + torch.sum((out * out), dim=-2, keepdim=True)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / divisor * out)
        orthogonal = out_ip - projection
        out_ip = weight * orthogonal
    elif ortho_v2:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        attn_map = q @ ip_k.transpose(-2, -1)
        attn_mean = attn_map.softmax(dim=-1).mean(dim=1, keepdim=True)
        attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
        divisor = 1e-6 + torch.sum((out * out), dim=-2, keepdim=True)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / divisor * out)
        orthogonal = out_ip + (attn_mean - 1) * projection
        out_ip = weight * orthogonal
    else:
        out_ip = out_ip * weight

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
        mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    return out_ip.to(dtype=dtype)

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


class ApplyPulid:
    def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, sharpening=0.0, method=None, noise=0.0, attn_mask=None):
        work_model = model.clone()
        
        device = memory_management.get_torch_device()
        dtype = torch.float16 if memory_management.should_use_fp16() else torch.float32

        eva_clip.to(device, dtype=dtype)

        if len(pulid["image_proj"].keys()) == 172:
            pulid_model = PulidModelV1_1(pulid).to(device, dtype=dtype)
        elif len(pulid["image_proj"].keys()) == 110:
            pulid_model = PulidModel(pulid).to(device, dtype=dtype)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if method == "fidelity":
            num_zero = 8
            ortho = False
            ortho_v2 = True
        elif method == "style":
            num_zero = 16
            ortho = True
            ortho_v2 = False
        else:
            num_zero = 0
            ortho = False
            ortho_v2 = False
        
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []
        uncond = []

        for i in range(len(image)):
            face_img = self.prep_image(image[i], sharpening)
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 128, -64)]:
                face_analysis.det_model.input_size = size
                face = face_analysis.get(face_img[0])
                if face:
                    face = sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                print('Warning: No face detected in image', i)
                continue

            # get eva_clip embeddings
            face_helper.clean_all()
            face_helper.read_image(face_img[0])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                # No face detected, skip this image
                continue
            
            face = face_helper.cropped_faces[0]
            face = NPToTensor(face).unsqueeze(0).permute(0,3,1,2).to(device)
            parsing_out = face_helper.face_parse(T.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(face)
            face_features_image = torch.where(bg, white_image, to_gray(face))
            # apparently MPS only supports NEAREST interpolation?
            face_features_image = T.functional.resize(face_features_image, eva_clip.image_size, T.InterpolationMode.BICUBIC if 'cuda' in device.type else T.InterpolationMode.NEAREST).to(device, dtype=dtype)
            face_features_image = T.functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)
            
            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            if noise == 0:
                id_uncond = torch.zeros_like(id_cond)
            else:
                id_uncond = torch.rand_like(id_cond) * noise
            id_vit_hidden_uncond = []
            for idx in range(len(id_vit_hidden)):
                if noise == 0:
                    id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
                else:
                    id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)
            
            cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
            uncond.append(pulid_model.get_image_embeds(id_uncond, id_vit_hidden_uncond))

        eva_clip.to('cpu')

        if not cond:
            # No faces detected, return the original model
            print("pulid warning: No faces detected in any of the given images, returning unmodified model.")
            return (work_model,)
        
        # average embeddings
        cond = torch.cat(cond).to(device, dtype=dtype)
        uncond = torch.cat(uncond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)
            uncond = torch.mean(uncond, dim=0, keepdim=True)

        if num_zero > 0:
            if noise == 0:
                zero_tensor = torch.zeros((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device)
            else:
                zero_tensor = torch.rand((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device) * noise
            cond = torch.cat([cond, zero_tensor], dim=1)
            uncond = torch.cat([uncond, zero_tensor], dim=1)

        sigma_start = model.model.predictor.percent_to_sigma(start_at)
        sigma_end = model.model.predictor.percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid": pulid_model,
            "weight": weight,
            "cond": cond,
            "uncond": uncond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "ortho": ortho,
            "ortho_v2": ortho_v2,
            "mask": attn_mask,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1

        return (work_model,)

    def prep_image(self, image, sharpening, convertNP=True, channels_last=True):
        if sharpening > 0.0:
            image = image ** 2.2
            image = contrast_adaptive_sharpening(image, sharpening, channels_last)
            image = image ** (1/2.2)
        
        if convertNP:
            return tensorToNP(image)
        else:
            return image
