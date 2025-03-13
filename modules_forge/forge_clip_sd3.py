import torch
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from ldm_patched.modules import model_management
from modules import sd_models
from modules.shared import opts

class SD3CLIP(torch.nn.Module):
    def __init__(self, clip_components, embedding_directory=None):
        super().__init__()
        self.clip_l, self.clip_g, self.t5xxl = clip_components
        self.tokenizer = SD3Tokenizer(embedding_directory)
        self.patcher = model_management.ModelPatcher(self)

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        
        # Process tokens for each component
        z_l, pooled_l = self.clip_l(tokens['l'])
        z_g, pooled_g = self.clip_g(tokens['g'])
        z_t5, _ = self.t5xxl(tokens['t5xxl'])

        # Combine outputs
        z = torch.cat([z_l, z_g, z_t5], dim=-2)
        pooled = torch.cat([pooled_l, pooled_g], dim=-1)

        return z, pooled

    def encode_from_tokens(self, tokens, return_pooled=False):
        z, pooled = self.encode_with_transformers(tokens)
        if return_pooled:
            return z, pooled
        return z

class SD3Tokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = self.clip_g.clip_l.tokenizer
        self.clip_g = self.clip_g.clip_g.tokenizer
        self.t5xxl = self.t5xxl.tokenizer

    def tokenize(self, text):
        return {
            "l": self.clip_l.tokenize(text),
            "g": self.clip_g.tokenize(text),
            "t5xxl": self.t5xxl.tokenize(text)
        }

def move_clip_to_gpu():
    if sd_models.model_data.sd_model is None:
        print('Error: CLIP called before SD is loaded!')
        return
    model_management.load_model_gpu(sd_models.model_data.sd_model.forge_objects.clip.patcher)