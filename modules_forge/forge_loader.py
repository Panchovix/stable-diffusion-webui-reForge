import torch
import contextlib
import os
from ldm_patched.modules import model_management
from ldm_patched.modules import model_detection

from ldm_patched.modules.sd import VAE, CLIP, load_model_weights
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.clip_vision

from omegaconf import OmegaConf
from modules.sd_models_config import find_checkpoint_config
from modules.shared import cmd_opts
from modules import sd_hijack
from modules.sd_models_xl import extend_sdxl
from ldm.util import instantiate_from_config
from modules_forge import forge_clip
from modules_forge.unet_patcher import UnetPatcher
from ldm_patched.modules.model_base import model_sampling, ModelType, SD3
import logging
import types

import open_clip
from transformers import CLIPTextModel, CLIPTokenizer
from ldm_patched.modules.args_parser import args


def maybe_override_text_encoder(forge_objects, checkpoint_info):
    try:
        from modules import sd_text_encoder, shared
        import gc
        import torch

        text_encoder_option = getattr(shared.opts, 'sd_text_encoder', 'Automatic')
        
        # Check if we should use a separate text encoder
        should_use_separate_te = False
        selected_option = None
        
        # If CLIP is None (was skipped during loading), we must load a separate TE
        if forge_objects.clip is None:
            should_use_separate_te = True
            print("No checkpoint CLIP loaded - must use separate text encoder")
            if text_encoder_option == 'None':
                raise RuntimeError("Cannot use 'None' text encoder option when checkpoint CLIP was not loaded")
        else:
            # Normal case: checkpoint CLIP was loaded, decide whether to replace it
            if text_encoder_option == 'None':
                # User explicitly wants no separate TE - use checkpoint's built-in
                return forge_objects
            elif text_encoder_option == 'Automatic':
                # In Automatic mode, don't use separate TE - just use checkpoint's built-in
                return forge_objects
            else:
                # User explicitly selected a specific TE
                matching_te = next((x for x in sd_text_encoder.text_encoder_options if x.label == text_encoder_option), None)
                if matching_te:
                    # Check if the selected TE is the same as the checkpoint
                    if checkpoint_info and matching_te.te_info.model_name == checkpoint_info.model_name:
                        # Same model - use checkpoint's built-in TE
                        print(f"Selected text encoder matches checkpoint model - using checkpoint's built-in text encoder")
                        return forge_objects
                    else:
                        # Different model - use separate TE
                        should_use_separate_te = True
                        selected_option = matching_te
                else:
                    print(f"Warning: Text encoder '{text_encoder_option}' not found, using checkpoint's built-in text encoder")
                    return forge_objects
        
        # If we reach here and should_use_separate_te is False, return original
        if not should_use_separate_te:
            return forge_objects
            
        # Find the text encoder to use if not already selected
        if selected_option is None:
            if text_encoder_option == 'Automatic' and checkpoint_info:
                model_name = checkpoint_info.model_name
                matching_options = [x for x in sd_text_encoder.text_encoder_options if x.model_name == model_name]
                if not matching_options:
                    # Fallback: use the first available text encoder if none match
                    if sd_text_encoder.text_encoder_options:
                        selected_option = sd_text_encoder.text_encoder_options[0]
                        print(f"No matching TE found, using first available: {selected_option.label}")
                    else:
                        raise RuntimeError("No text encoders available and checkpoint CLIP was not loaded")
                else:
                    selected_option = matching_options[0]
            else:
                selected_option = next((x for x in sd_text_encoder.text_encoder_options if x.label == text_encoder_option), None)
                if selected_option is None:
                    raise RuntimeError(f"Text encoder '{text_encoder_option}' not found and checkpoint CLIP was not loaded")
        
        # Store reference to original CLIP for cleanup
        original_clip = forge_objects.clip
        
        # Load the separate text encoder
        print(f"Loading separate text encoder: {selected_option.label}")
        separate_te = selected_option.create_text_encoder()
        separate_te.option = selected_option  # Add option reference
        separate_te._load_clip_from_checkpoint()  # Force load
        
        # Replace the CLIP in forge_objects
        if separate_te.clip_model is not None:
            print(f"Replacing text encoder with: {selected_option.label}")
            forge_objects.clip = separate_te.clip_model
            # Mark that we've loaded a separate TE to avoid double-loading later
            forge_objects._separate_te_loaded = True
            
            # Update the global text encoder state to match what we loaded
            from modules import sd_text_encoder
            sd_text_encoder.current_text_encoder_option = selected_option
            sd_text_encoder.current_text_encoder = separate_te
            print(f"Updated global TE state to: {selected_option.label}")
            
            # Properly unload the original CLIP model to free VRAM (only if it exists)
            if original_clip is not None and original_clip != separate_te.clip_model:
                print("Unloading original text encoder to free VRAM")
                
                # Unload model from VRAM if it has a patcher
                if hasattr(original_clip, 'patcher'):
                    try:
                        # Move model to CPU/offload device
                        original_clip.patcher.unpatch_model()
                        if hasattr(original_clip.patcher, 'model_unload'):
                            original_clip.patcher.model_unload()
                        elif hasattr(original_clip.patcher, 'to'):
                            original_clip.patcher.to('cpu')
                    except Exception as e:
                        print(f"Warning: Error during original CLIP patcher cleanup: {e}")
                
                # Clear the conditional stage model
                if hasattr(original_clip, 'cond_stage_model'):
                    try:
                        if hasattr(original_clip.cond_stage_model, 'to'):
                            original_clip.cond_stage_model.to('cpu')
                        # Clear parameters to free memory
                        for param in original_clip.cond_stage_model.parameters():
                            if hasattr(param, 'data'):
                                param.data = param.data.to('cpu')
                                del param.data
                    except Exception as e:
                        print(f"Warning: Error during original CLIP model cleanup: {e}")
                
                # Clear references
                del original_clip
                
                # Force garbage collection and CUDA cache cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        return forge_objects
        
    except Exception as e:
        print(f"Error loading separate text encoder: {e}")
        return forge_objects  # Fall back to original


class FakeObject:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.visual = None
        return

    def eval(self, *args, **kwargs):
        return self

    def parameters(self, *args, **kwargs):
        return []


class ForgeSD:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeSD(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


@contextlib.contextmanager
def no_clip():
    backup_openclip = open_clip.create_model_and_transforms
    backup_CLIPTextModel = CLIPTextModel.from_pretrained
    backup_CLIPTokenizer = CLIPTokenizer.from_pretrained

    try:
        open_clip.create_model_and_transforms = lambda *args, **kwargs: (FakeObject(), None, None)
        CLIPTextModel.from_pretrained = lambda *args, **kwargs: FakeObject()
        CLIPTokenizer.from_pretrained = lambda *args, **kwargs: FakeObject()
        yield

    finally:
        open_clip.create_model_and_transforms = backup_openclip
        CLIPTextModel.from_pretrained = backup_CLIPTextModel
        CLIPTokenizer.from_pretrained = backup_CLIPTokenizer
    return

def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded via <lora:loraname:lorastrength>..."
    return ""

def load_checkpoint_guess_config(ckpt, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    if isinstance(ckpt, str) and os.path.isfile(ckpt):
        # If ckpt is a string and a valid file path, load it
        sd, metadata = ldm_patched.modules.utils.load_torch_file(ckpt, return_metadata=True)
        ckpt_path = ckpt  # Store the path for error reporting
    elif isinstance(ckpt, dict):
        # If ckpt is already a state dictionary, use it directly
        sd = ckpt
        metadata = None  # No metadata available for directly provided state dict
        ckpt_path = "provided state dict"  # Generic description for error reporting
    else:
        raise ValueError("Input must be either a file path or a state dictionary")

    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options, metadata=metadata)
    if out is None:
        # Include helpful error hint
        error_hint = model_detection_error_hint(ckpt_path, sd)
        raise RuntimeError(f"ERROR: Could not detect model type of: {ckpt_path}{error_hint}")
    return out

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, metadata=None):
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = ldm_patched.modules.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = ldm_patched.modules.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
        diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
        if diffusion_model is None:
            return None
        return (diffusion_model, None, VAE(sd={}), None)  # The VAE object is there to throw an exception if it's actually used'

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = ldm_patched.modules.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd, metadata=metadata)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = ldm_patched.modules.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = UnetPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            print("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return ForgeSD(model_patcher, clip, vae, clipvision)

def load_diffusion_model_state_dict(sd, model_options={}):
    dtype = model_options.get("dtype", None)
    metadata = model_options.get("metadata", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = ldm_patched.modules.utils.calculate_parameters(sd)
    weight_dtype = ldm_patched.modules.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "", metadata=metadata)
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    
    # Return ForgeSD with just the UNet
    model_patcher = UnetPatcher(model, load_device=load_device, offload_device=offload_device)
    return ForgeSD(model_patcher, None, None, None)

def load_diffusion_model(unet_path, model_options={}):
    sd, metadata = ldm_patched.modules.utils.load_torch_file(unet_path, return_metadata=True)
    model_options["metadata"] = metadata
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model


@torch.no_grad()
def load_model_for_a1111(timer, checkpoint_info=None, state_dict=None):
    is_sd3 = 'model.diffusion_model.x_embedder.proj.weight' in state_dict
    ztsnr = 'ztsnr' in state_dict
    timer.record("forge solving config")
    
    if not is_sd3:
        a1111_config_filename = find_checkpoint_config(state_dict, checkpoint_info)
        a1111_config = OmegaConf.load(a1111_config_filename)
        if hasattr(a1111_config.model.params, 'network_config'):
            a1111_config.model.params.network_config.target = 'modules_forge.forge_loader.FakeObject'
        if hasattr(a1111_config.model.params, 'unet_config'):
            a1111_config.model.params.unet_config.target = 'modules_forge.forge_loader.FakeObject'
        if hasattr(a1111_config.model.params, 'first_stage_config'):
            a1111_config.model.params.first_stage_config.target = 'modules_forge.forge_loader.FakeObject'
        with no_clip():
            sd_model = instantiate_from_config(a1111_config.model)
    else:
        sd_model = torch.nn.Module() 
    
    timer.record("forge instantiate config")
    
    # Check if we should skip loading the checkpoint's CLIP to save memory
    should_skip_clip = False
    try:
        from modules import sd_text_encoder, shared
        text_encoder_option = getattr(shared.opts, 'sd_text_encoder', 'Automatic')
        
        # If we're in Automatic mode and there's a matching separate TE, skip loading checkpoint CLIP
        if text_encoder_option == 'Automatic' and checkpoint_info:
            model_name = checkpoint_info.model_name
            matching_options = [x for x in sd_text_encoder.text_encoder_options if x.model_name == model_name]
            if matching_options:
                should_skip_clip = True
                print(f"Skipping checkpoint CLIP loading - will use separate TE: {matching_options[0].label}")
        # If explicitly set to use a separate TE, also skip
        elif text_encoder_option != 'None' and text_encoder_option != 'Automatic':
            separate_option = next((x for x in sd_text_encoder.text_encoder_options if x.label == text_encoder_option), None)
            if separate_option:
                should_skip_clip = True
                print(f"Skipping checkpoint CLIP loading - will use separate TE: {text_encoder_option}")
    except Exception as e:
        print(f"Warning: Error checking text encoder options: {e}")
        should_skip_clip = False
    
    forge_objects = load_checkpoint_guess_config(
        state_dict,
        output_vae=True,
        output_clip=not should_skip_clip,  # Skip CLIP loading if we'll replace it anyway
        output_clipvision=True,
        embedding_directory=cmd_opts.embeddings_dir,
        output_model=True
    )
    
    forge_objects = maybe_override_text_encoder(forge_objects, checkpoint_info)
    sd_model.first_stage_model = forge_objects.vae.first_stage_model
    sd_model.model.diffusion_model = forge_objects.unet.model.diffusion_model
    sd_model.forge_objects = forge_objects
    sd_model.forge_objects_original = forge_objects.shallow_copy()
    sd_model.forge_objects_after_applying_lora = forge_objects.shallow_copy()
    
    # Transfer the separate TE flag to the sd_model
    if hasattr(forge_objects, '_separate_te_loaded'):
        sd_model._separate_te_loaded = True
    if args.torch_compile:
        timer.record("start model compilation")
        if forge_objects.unet is not None:
            forge_objects.unet.compile_model(backend=args.torch_compile_backend)
        timer.record("model compilation complete")
    timer.record("forge load real models")
    
    conditioner = getattr(sd_model, 'conditioner', None)

    if conditioner:
        text_cond_models = []
        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__
            if typename == 'FrozenCLIPEmbedder':  # SDXL Clip L
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
                    model_embeddings.token_embedding, sd_hijack.model_hijack)
                embedder = forge_clip.CLIP_SD_XL_L(embedder, sd_hijack.model_hijack)
                conditioner.embedders[i] = embedder
                text_cond_models.append(embedder)
            elif typename == 'FrozenOpenCLIPEmbedder2':  # SDXL Clip G
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_g.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_g.transformer
                embedder.text_projection = forge_objects.clip.cond_stage_model.clip_g.text_projection
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
                    model_embeddings.token_embedding, sd_hijack.model_hijack, textual_inversion_key='clip_g')
                embedder = forge_clip.CLIP_SD_XL_G(embedder, sd_hijack.model_hijack)
                conditioner.embedders[i] = embedder
                text_cond_models.append(embedder)
        if len(text_cond_models) == 1:
            sd_model.cond_stage_model = text_cond_models[0]
        else:
            sd_model.cond_stage_model = conditioner
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenCLIPEmbedder':  # SD15 Clip
        sd_model.cond_stage_model.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
        sd_model.cond_stage_model.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
        model_embeddings = sd_model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
            model_embeddings.token_embedding, sd_hijack.model_hijack)
        sd_model.cond_stage_model = forge_clip.CLIP_SD_15_L(sd_model.cond_stage_model, sd_hijack.model_hijack)
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenOpenCLIPEmbedder':  # SD21 Clip
        sd_model.cond_stage_model.tokenizer = forge_objects.clip.tokenizer.clip_h.tokenizer
        sd_model.cond_stage_model.transformer = forge_objects.clip.cond_stage_model.clip_h.transformer
        model_embeddings = sd_model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
            model_embeddings.token_embedding, sd_hijack.model_hijack)
        sd_model.cond_stage_model = forge_clip.CLIP_SD_21_H(sd_model.cond_stage_model, sd_hijack.model_hijack)
    else:
        raise NotImplementedError('Bad Clip Class Name:' + type(sd_model.cond_stage_model).__name__)

    timer.record("forge set components")
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if getattr(sd_model, 'parameterization', None) == 'v':
        sd_model.forge_objects.unet.model.model_sampling = model_sampling(sd_model.forge_objects.unet.model.model_config, ModelType.V_PREDICTION)
    
    sd_model.ztsnr = ztsnr

    sd_model.is_sd3 = is_sd3
    sd_model.latent_channels = 16 if is_sd3 else 4
    sd_model.is_sdxl = conditioner is not None and not is_sd3
    sd_model.is_sdxl_inpaint = sd_model.is_sdxl and forge_objects.unet.model.diffusion_model.in_channels == 9
    sd_model.is_sd2 = not sd_model.is_sdxl and not is_sd3 and hasattr(sd_model.cond_stage_model, 'model')
    sd_model.is_sd1 = not sd_model.is_sdxl and not sd_model.is_sd2 and not is_sd3
    sd_model.is_ssd = sd_model.is_sdxl and 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in sd_model.state_dict().keys()
    
    if sd_model.is_sdxl:
        extend_sdxl(sd_model)
    
    sd_model.sd_model_hash = sd_model_hash
    sd_model.sd_model_checkpoint = checkpoint_info.filename
    sd_model.sd_checkpoint_info = checkpoint_info

    @torch.inference_mode()
    def patched_decode_first_stage(x):
        sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_out(x)
        sample = sd_model.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    @torch.inference_mode()
    def patched_encode_first_stage(x):
        sample = sd_model.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_in(sample)
        return sample.to(x)

    sd_model.ema_scope = lambda *args, **kwargs: contextlib.nullcontext()
    sd_model.get_first_stage_encoding = lambda x: x
    sd_model.decode_first_stage = patched_decode_first_stage
    sd_model.encode_first_stage = patched_encode_first_stage
    sd_model.clip = sd_model.cond_stage_model
    sd_model.tiling_enabled = False
    timer.record("forge finalize")
    sd_model.current_lora_hash = str([])
    return sd_model
