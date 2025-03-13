from ldm_patched.modules.model_base import BaseModel
from ldm_patched.modules import model_management
from ldm_patched.modules.sd import CLIPType
import ldm_patched.modules.utils

class SD3ModelConfig:
    def __init__(self):
        self.name = 'SD3'
        self.unet_config = {
            "in_channels": 16,
            "pos_embed_scaling_factor": None,
        }
        self.latent_format = "SD3"

    def get_model(self, state_dict, prefix="model.diffusion_model.", device=None):
        if device is None:
            device = model_management.get_torch_device()
        shift = 3.0  # SD3 specific shift value
        return BaseModel(shift=shift, state_dict=state_dict, prefix=prefix, device=device)

    def process_clip_state_dict(self, state_dict):
        clip_l = ldm_patched.modules.utils.state_dict_prefix_replace(state_dict, {"clip_l.": ""})
        clip_g = ldm_patched.modules.utils.state_dict_prefix_replace(state_dict, {"clip_g.": ""})
        t5xxl = ldm_patched.modules.utils.state_dict_prefix_replace(state_dict, {"t5xxl.": ""})
        return {"clip_l": clip_l, "clip_g": clip_g, "t5xxl": t5xxl}

    def clip_target(self, state_dict=None):
        from ldm_patched.modules.text_encoders import sd3_clip
        clip_l = sd3_clip.SD1ClipModel(layer="hidden", layer_idx=-2)
        clip_g = sd3_clip.SDXLClipG()
        t5xxl = sd3_clip.T5XXLModel()
        return (clip_l, clip_g, t5xxl)