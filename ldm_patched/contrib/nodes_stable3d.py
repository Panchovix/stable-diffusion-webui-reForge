# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 

import torch
import ldm_patched.contrib.nodes
import ldm_patched.modules.utils

def camera_embeddings(elevation, azimuth):
    elevation = torch.as_tensor([elevation])
    azimuth = torch.as_tensor([azimuth])
    embeddings = torch.stack(
        [
                torch.deg2rad(
                    (90 - elevation) - (90)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth)),
                torch.cos(torch.deg2rad(azimuth)),
                torch.deg2rad(
                    90 - torch.full_like(elevation, 0)
                ),
        ], dim=-1).unsqueeze(1)

    return embeddings


class StableZero123_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/3d_models"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = ldm_patched.modules.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.to(pooled.device).repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

class StableZero123_Conditioning_Batched:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "elevation_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/3d_models"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth, elevation_batch_increment, azimuth_batch_increment):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = ldm_patched.modules.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        cam_embeds = []
        for i in range(batch_size):
            cam_embeds.append(camera_embeddings(elevation, azimuth))
            elevation += elevation_batch_increment
            azimuth += azimuth_batch_increment

        cam_embeds = torch.cat(cam_embeds, dim=0)
        cond = torch.cat([ldm_patched.modules.utils.repeat_to_batch_size(pooled, batch_size), cam_embeds], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent, "batch_index": [0] * batch_size})


# Original code and file from ComfyUI, https://github.com/comfyanonymous/ComfyUI
NODE_CLASS_MAPPINGS = {
    "StableZero123_Conditioning": StableZero123_Conditioning,
    "StableZero123_Conditioning_Batched": StableZero123_Conditioning_Batched,
}
