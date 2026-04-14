from __future__ import annotations

import torch

from ldm_patched.modules import model_base
from ldm_patched.modules import model_sampling
from modules import devices, shared, prompt_parser
from modules import torch_utils

import ldm_patched.modules.model_management as model_management
from modules_forge.forge_clip import move_clip_to_gpu


def get_learned_conditioning(self: model_base.BaseModel, batch: prompt_parser.SdConditioning | list[str]):
    move_clip_to_gpu()

    # Check if this is actually an SDXL model - if not, don't override
    if not (hasattr(self, 'is_sdxl') and self.is_sdxl):
        # For non-SDXL models, fall back to the original method
        if hasattr(model_base.BaseModel, '_original_get_learned_conditioning') and model_base.BaseModel._original_get_learned_conditioning is not None:
            return model_base.BaseModel._original_get_learned_conditioning(self, batch)
        else:
            # If no original method, use the forge_objects directly or cond_stage_model
            if hasattr(self, 'cond_stage_model') and hasattr(self.cond_stage_model, 'encode'):
                # Standard approach - use the cond_stage_model.encode method
                if isinstance(batch, str):
                    batch = [batch]
                elif not isinstance(batch, list):
                    batch = list(batch)
                return self.cond_stage_model.encode(batch)
            elif hasattr(self, 'forge_objects') and hasattr(self.forge_objects, 'clip'):
                clip_model = self.forge_objects.clip
                
                # Handle the text input properly for SD1.5
                if isinstance(batch, str):
                    texts = [batch]
                elif isinstance(batch, list):
                    texts = batch
                else:
                    texts = list(batch)
                
                # For SD1.5, tokenize each text individually
                results = []
                for text in texts:
                    tokens = clip_model.tokenize(text)
                    cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
                    results.append(cond)
                
                # Return as tensor if single item, or stack if multiple
                if len(results) == 1:
                    return results[0]
                else:
                    return torch.cat(results, dim=0)
            else:
                raise NotImplementedError("No valid conditioning method found for this model")

    # Handle both ldm_patched models and legacy SGM models (SDXL only)
    if hasattr(self, 'conditioner') and hasattr(self.conditioner, 'embedders'):
        # Legacy SGM-based conditioner path
        for embedder in self.conditioner.embedders:
            embedder.ucg_rate = 0.0

        width = getattr(batch, 'width', 1024) or 1024
        height = getattr(batch, 'height', 1024) or 1024
        is_negative_prompt = getattr(batch, 'is_negative_prompt', False)
        aesthetic_score = shared.opts.sdxl_refiner_low_aesthetic_score if is_negative_prompt else shared.opts.sdxl_refiner_high_aesthetic_score

        devices_args = dict(device=self.forge_objects.clip.patcher.model.device, dtype=model_management.text_encoder_dtype())

        sdxl_conds = {
            "txt": batch,
            "original_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
            "crop_coords_top_left": torch.tensor([shared.opts.sdxl_crop_top, shared.opts.sdxl_crop_left], **devices_args).repeat(len(batch), 1),
            "target_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
            "aesthetic_score": torch.tensor([aesthetic_score], **devices_args).repeat(len(batch), 1),
        }

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in batch)
        c = self.conditioner(sdxl_conds, force_zero_embeddings=['txt'] if force_zero_negative_prompt else [])
        return c
    else:
        # ldm_patched model path - use the forge_objects directly (SDXL)
        if hasattr(self, 'forge_objects') and hasattr(self.forge_objects, 'clip'):
            # Use the ldm_patched CLIP implementation
            clip_model = self.forge_objects.clip
            
            # Convert batch to proper format if needed
            if isinstance(batch, str):
                batch = [batch]
            elif not isinstance(batch, list):
                batch = list(batch)
            
            # Get conditioning from the ldm_patched CLIP model
            tokens = clip_model.tokenize(batch)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            
            # For SDXL, we need to handle the pooled output and additional conditioning
            width = getattr(batch, 'width', 1024) if hasattr(batch, 'width') else 1024
            height = getattr(batch, 'height', 1024) if hasattr(batch, 'height') else 1024
            
            # Create SDXL-style conditioning dict
            return {
                'c_crossattn': [cond],
                'c_adm': pooled  # This will be processed by the model's encode_adm method
            }
        else:
            # Fallback - should not happen in normal operation
            raise NotImplementedError("No valid conditioning method found for this model")


def apply_model(self: model_base.BaseModel, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    # Check if this is actually an SDXL model - if not, use original method
    if not (hasattr(self, 'is_sdxl') and self.is_sdxl):
        # For non-SDXL models, call the original apply_model
        if hasattr(model_base.BaseModel, '_original_apply_model'):
            return model_base.BaseModel._original_apply_model(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)
        else:
            # Fallback to the ldm_patched original
            orig_apply_model = model_base.BaseModel._apply_model
            return orig_apply_model(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)
    
    # SDXL-specific logic
    # Convert conditioning format from ldm_patched to SGM format
    if c_crossattn is not None or c_concat is not None:
        cond = {}
        if c_crossattn is not None:
            cond['c_crossattn'] = c_crossattn
        if c_concat is not None:
            cond['c_concat'] = c_concat
    else:
        # Fallback for older format
        cond = kwargs.get('cond', {})

    if self.diffusion_model.in_channels == 9 and 'c_concat' in cond:
        x = torch.cat([x] + cond['c_concat'], dim=1)

    # In ldm_patched, we need to call the original apply_model which expects the right signature
    # Remove our method temporarily and call the parent
    orig_apply_model = model_base.BaseModel._apply_model
    return orig_apply_model(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)


def get_first_stage_encoding(self, x):  # SDXL's encode_first_stage does everything so get_first_stage_encoding is just there for compatibility
    return x


# Save the original methods before overriding
if not hasattr(model_base.BaseModel, '_original_get_learned_conditioning'):
    model_base.BaseModel._original_get_learned_conditioning = getattr(model_base.BaseModel, 'get_learned_conditioning', None)
if not hasattr(model_base.BaseModel, '_original_apply_model'):
    model_base.BaseModel._original_apply_model = getattr(model_base.BaseModel, 'apply_model', None)

model_base.BaseModel.get_learned_conditioning = get_learned_conditioning
model_base.BaseModel.apply_model = apply_model
model_base.BaseModel.get_first_stage_encoding = get_first_stage_encoding


def encode_embedding_init_text(self, init_text, nvpt):
    res = []

    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'encode_embedding_init_text')]:
        encoded = embedder.encode_embedding_init_text(init_text, nvpt)
        res.append(encoded)

    return torch.cat(res, dim=1)


def tokenize(self, texts):
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'tokenize')]:
        return embedder.tokenize(texts)

    raise AssertionError('no tokenizer available')



def process_texts(self, texts):
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'process_texts')]:
        return embedder.process_texts(texts)


def get_target_prompt_token_count(self, token_count):
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'get_target_prompt_token_count')]:
        return embedder.get_target_prompt_token_count(token_count)


# Add conditioning methods to BaseModel for compatibility with existing code
# In ldm_patched, conditioning is handled differently through the model's built-in methods
# We'll add these methods to BaseModel to maintain compatibility
model_base.BaseModel.encode_embedding_init_text = encode_embedding_init_text
model_base.BaseModel.tokenize = tokenize
model_base.BaseModel.process_texts = process_texts
model_base.BaseModel.get_target_prompt_token_count = get_target_prompt_token_count


def extend_sdxl(model):
    """this adds a bunch of parameters to make SDXL model look a bit more like SD1.5 to the rest of the codebase."""

    dtype = torch_utils.get_param(model.model.diffusion_model).dtype
    model.model.diffusion_model.dtype = dtype
    model.model.conditioning_key = 'crossattn'
    model.cond_stage_key = 'txt'
    # model.cond_stage_model will be set in sd_hijack

    # Determine parameterization based on model type in ldm_patched
    model.parameterization = "v" if hasattr(model, 'model_type') and model.model_type in [model_base.ModelType.V_PREDICTION, model_base.ModelType.V_PREDICTION_EDM, model_base.ModelType.V_PREDICTION_CONTINUOUS] else "eps"

    # Use model sampling from ldm_patched instead of SGM discretization
    if hasattr(model, 'model_sampling') and hasattr(model.model_sampling, 'alphas_cumprod'):
        model.alphas_cumprod = model.model_sampling.alphas_cumprod.to(device=devices.device, dtype=torch.float32)
    else:
        # Fallback: create alphas_cumprod similar to LegacyDDPMDiscretization
        from ldm_patched.ldm.modules.diffusionmodules.util import make_beta_schedule
        betas = make_beta_schedule("linear", 1000, 0.00085, 0.012)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        model.alphas_cumprod = torch.asarray(alphas_cumprod, device=devices.device, dtype=torch.float32)

    # Create a conditioner compatibility wrapper for forge_loader
    # In ldm_patched, conditioning is handled differently, but we need to maintain compatibility
    if not hasattr(model, 'conditioner'):
        # Create a simple conditioner wrapper that mimics SGM's GeneralConditioner interface
        class ConditionerWrapper:
            def __init__(self):
                self.embedders = []
                self.wrapped = torch.nn.Module()
            
            def __call__(self, *args, **kwargs):
                # This should not be called in the normal flow with ldm_patched
                # The actual conditioning is handled in the ldm_patched model
                raise NotImplementedError("This conditioner wrapper is for compatibility only")
        
        model.conditioner = ConditionerWrapper()
    else:
        model.conditioner.wrapped = torch.nn.Module()


# Set print functions for ldm_patched modules instead of SGM
try:
    import ldm_patched.ldm.modules.attention
    ldm_patched.ldm.modules.attention.print = shared.ldm_print
except (ImportError, AttributeError):
    pass

try:
    import ldm_patched.ldm.modules.diffusionmodules.model
    ldm_patched.ldm.modules.diffusionmodules.model.print = shared.ldm_print
except (ImportError, AttributeError):
    pass

try:
    import ldm_patched.ldm.modules.diffusionmodules.openaimodel
    ldm_patched.ldm.modules.diffusionmodules.openaimodel.print = shared.ldm_print
except (ImportError, AttributeError):
    pass

# Configure attention settings for ldm_patched
try:
    import ldm_patched.ldm.modules.attention
    ldm_patched.ldm.modules.attention.SDP_IS_AVAILABLE = True
    ldm_patched.ldm.modules.attention.XFORMERS_IS_AVAILABLE = False
except (ImportError, AttributeError):
    pass
