import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from modules import script_callbacks, shared, paths
from modules.timer import Timer
import ldm_patched.modules.utils
import ldm_patched.modules.model_detection
from ldm_patched.modules.sd import CLIP

# Global state for text encoder management
text_encoder_options = []
current_text_encoder_option = None
current_text_encoder = None

@dataclass
class TextEncoderInfo:
    """Information about a text encoder file"""
    filename: str
    title: str
    model_name: str
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def name_for_extra(self):
        return os.path.splitext(os.path.basename(self.filename))[0]

class SdTextEncoderOption:
    """Base class for text encoder options - similar to SdUnetOption"""
    model_name = None
    """name of related checkpoint - this option will be selected automatically if checkpoint name matches"""
    
    label = None
    """name of the text encoder in UI"""
    
    def create_text_encoder(self):
        """returns SdTextEncoder object to be used instead of built-in text encoder"""
        raise NotImplementedError()

class SdTextEncoder:
    """Base class for text encoders - similar to SdUnet"""
    def __init__(self, clip_model):
        self.clip_model = clip_model
        
    def encode_prompt(self, prompt, *args, **kwargs):
        """Encode a text prompt into conditioning"""
        raise NotImplementedError()
        
    def activate(self):
        """Called when this text encoder becomes active"""
        pass
        
    def deactivate(self):
        """Called when this text encoder is deactivated"""
        pass

class CheckpointTextEncoderOption(SdTextEncoderOption):
    """Text encoder option that loads from a checkpoint file"""
    def __init__(self, te_info: TextEncoderInfo):
        self.te_info = te_info
        self.model_name = te_info.model_name
        self.label = te_info.title
        
    def create_text_encoder(self):
        return CheckpointTextEncoder(self.te_info)

class CheckpointTextEncoder(SdTextEncoder):
    """Text encoder loaded from a checkpoint file"""
    def __init__(self, te_info: TextEncoderInfo):
        self.te_info = te_info
        self.clip_model = None
        
    def _load_clip_from_checkpoint(self):
        """Load CLIP model from checkpoint file"""
        if self.clip_model is not None:
            return self.clip_model
            
        timer = Timer()
        
        try:
            # Load state dict from file
            state_dict, metadata = ldm_patched.modules.utils.load_torch_file(
                self.te_info.filename, return_metadata=True
            )
            timer.record("load state dict")
            
            # Detect model configuration
            diffusion_model_prefix = ldm_patched.modules.model_detection.unet_prefix_from_state_dict(state_dict)
            model_config = ldm_patched.modules.model_detection.model_config_from_unet(
                state_dict, diffusion_model_prefix, metadata=metadata
            )
            
            if model_config is None:
                raise RuntimeError(f"Could not detect model type for text encoder: {self.te_info.filename}")
            
            timer.record("detect model type")
            
            # Get clip target and process state dict
            clip_target = model_config.clip_target(state_dict=state_dict)
            if clip_target is None:
                raise RuntimeError(f"No text encoder found in: {self.te_info.filename}")
                
            clip_sd = model_config.process_clip_state_dict(state_dict)
            if len(clip_sd) == 0:
                raise RuntimeError(f"No CLIP weights found in: {self.te_info.filename}")
                
            timer.record("process clip state dict")
            
            # Create CLIP model
            parameters = ldm_patched.modules.utils.calculate_parameters(clip_sd)
            embedding_directory = shared.cmd_opts.embeddings_dir
            
            clip = CLIP(
                clip_target, 
                embedding_directory=embedding_directory, 
                tokenizer_data=clip_sd, 
                parameters=parameters
            )
            
            # Load weights
            m, u = clip.load_sd(clip_sd, full_model=True)
            if len(m) > 0:
                # Filter out expected missing keys
                m_filter = list(filter(
                    lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, 
                    m
                ))
                if len(m_filter) > 0:
                    logging.warning(f"Text encoder missing keys: {m_filter}")
                    
            if len(u) > 0:
                logging.debug(f"Text encoder unexpected keys: {u}")
                
            timer.record("load clip weights")
            
            self.clip_model = clip
            print(f"Loaded text encoder {self.te_info.title} in {timer.summary()}")
            
        except Exception as e:
            logging.error(f"Error loading text encoder {self.te_info.filename}: {e}")
            raise
            
        return self.clip_model
        
    def encode_prompt(self, prompt, *args, **kwargs):
        """Encode prompt using this text encoder"""
        if self.clip_model is None:
            self._load_clip_from_checkpoint()
            
        # Use the same encoding logic as the original CLIP model
        return self.clip_model.encode(prompt, *args, **kwargs)
        
    def activate(self):
        """Activate this text encoder"""
        if self.clip_model is None:
            self._load_clip_from_checkpoint()
            
    def deactivate(self):
        """Deactivate this text encoder and free VRAM"""
        if self.clip_model is not None:
            print(f"Deactivating and unloading text encoder: {self.te_info.title}")
            try:
                # Unload from VRAM if it has a patcher
                if hasattr(self.clip_model, 'patcher'):
                    self.clip_model.patcher.unpatch_model()
                    if hasattr(self.clip_model.patcher, 'model_unload'):
                        self.clip_model.patcher.model_unload()
                    elif hasattr(self.clip_model.patcher, 'to'):
                        self.clip_model.patcher.to('cpu')
                
                # Move the model to CPU
                if hasattr(self.clip_model, 'cond_stage_model'):
                    if hasattr(self.clip_model.cond_stage_model, 'to'):
                        self.clip_model.cond_stage_model.to('cpu')
                    
                    # Clear parameters to free memory
                    for param in self.clip_model.cond_stage_model.parameters():
                        if hasattr(param, 'data'):
                            param.data = param.data.to('cpu')
                
                # Force cleanup
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Error during text encoder deactivation: {e}")
            
            # Clear the reference
            self.clip_model = None

def list_text_encoders():
    """Scan for available text encoder files and update the options list"""
    text_encoder_options.clear()
    
    # Look in the same directory as checkpoints and dedicated text encoder directory
    checkpoint_dirs = []
    
    if shared.cmd_opts.ckpt_dir:
        checkpoint_dirs.append(shared.cmd_opts.ckpt_dir)
    else:
        checkpoint_dirs.append(os.path.join(paths.models_path, "Stable-diffusion"))
    
    # Add dedicated text encoder directory
    te_dir = os.path.join(paths.models_path, "text_encoders")
    if not os.path.exists(te_dir):
        try:
            os.makedirs(te_dir, exist_ok=True)
        except Exception:
            pass  # Ignore if we can't create the directory
    checkpoint_dirs.append(te_dir)
    
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            continue
            
        # Use shared.walk_files for recursive scanning like checkpoints do
        for full_path in shared.walk_files(checkpoint_dir, allowed_extensions=['.safetensors', '.ckpt', '.pth', '.bin']):
            if not os.path.isfile(full_path):
                continue
                
            # Extract model name from filename for matching  
            filename = os.path.basename(full_path)
            model_name = os.path.splitext(filename)[0]
            
            # For files in subdirectories, include the subdirectory in the title
            rel_path = os.path.relpath(full_path, checkpoint_dir)
            rel_dir = os.path.dirname(rel_path)
            if rel_dir and rel_dir != '.':
                title = f"TE: {rel_dir}/{model_name}"
            else:
                title = f"TE: {model_name}"
            
            te_info = TextEncoderInfo(
                filename=full_path,
                title=title,
                model_name=model_name
            )
            
            option = CheckpointTextEncoderOption(te_info)
            text_encoder_options.append(option)
    
    # Allow extensions to add their own text encoder options
    new_options = script_callbacks.list_text_encoders_callback()
    text_encoder_options.extend(new_options)

def get_text_encoder_option(option=None):
    """Get text encoder option by name or automatic selection"""
    option = option or getattr(shared.opts, 'sd_text_encoder', 'Automatic')
    
    if option == "None":
        return None
        
    if option == "Automatic":
        # Try to find a text encoder with matching name to current checkpoint
        if shared.sd_model and hasattr(shared.sd_model, 'sd_checkpoint_info'):
            checkpoint_name = shared.sd_model.sd_checkpoint_info.model_name
            
            matching_options = [x for x in text_encoder_options if x.model_name == checkpoint_name]
            if matching_options:
                return matching_options[0]
                
        return None  # Use checkpoint's built-in text encoder
        
    # Find by label
    return next((x for x in text_encoder_options if x.label == option), None)

def apply_text_encoder(option=None):
    """Apply a text encoder option"""
    global current_text_encoder_option, current_text_encoder
    
    new_option = get_text_encoder_option(option)
    if new_option == current_text_encoder_option:
        return
        
    # Deactivate current text encoder and ensure proper cleanup
    if current_text_encoder is not None:
        print(f"Deactivating text encoder: {current_text_encoder.option.label}")
        current_text_encoder.deactivate()
        # Clear the reference to help with garbage collection
        current_text_encoder = None
        
        # Force garbage collection after deactivation
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    current_text_encoder_option = new_option
    
    if current_text_encoder_option is None:
        current_text_encoder = None
        print("Using checkpoint's built-in text encoder")
        return
        
    # Activate new text encoder
    current_text_encoder = current_text_encoder_option.create_text_encoder()
    current_text_encoder.option = current_text_encoder_option
    print(f"Activating text encoder: {current_text_encoder.option.label}")
    current_text_encoder.activate()

def get_current_text_encoder():
    """Get currently active text encoder, or None if using checkpoint's built-in"""
    return current_text_encoder

def reload_text_encoder_list():
    """Reload the list of available text encoders"""
    list_text_encoders()

# Initialize on module load
list_text_encoders()