import os
import logging
import torch.nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

from modules import script_callbacks, shared, devices, paths
from modules.timer import Timer
import ldm_patched.modules.utils
import ldm_patched.modules.model_detection
from modules_forge.forge_loader import load_diffusion_model
from modules_forge.unet_patcher import UnetPatcher

# Global state for UNET management
unet_options = []
current_unet_option = None
current_unet = None
original_forward = None  # not used, only left temporarily for compatibility

@dataclass
class UnetInfo:
    """Information about a UNET file"""
    filename: str
    title: str
    model_name: str
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def name_for_extra(self):
        return os.path.splitext(os.path.basename(self.filename))[0]

def list_unets():
    """Scan for available UNET files and update the options list"""
    unet_options.clear()
    
    # Look in the same directory as checkpoints and dedicated UNET directory
    checkpoint_dirs = []
    
    if shared.cmd_opts.ckpt_dir:
        checkpoint_dirs.append(shared.cmd_opts.ckpt_dir)
    else:
        checkpoint_dirs.append(os.path.join(paths.models_path, "Stable-diffusion"))
    
    # Add dedicated UNET directory
    unet_dir = os.path.join(paths.models_path, "unet")
    if not os.path.exists(unet_dir):
        try:
            os.makedirs(unet_dir, exist_ok=True)
        except Exception:
            pass  # Ignore if we can't create the directory
    checkpoint_dirs.append(unet_dir)
    
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
                title = f"UNET: {rel_dir}/{model_name}"
            else:
                title = f"UNET: {model_name}"
            
            unet_info = UnetInfo(
                filename=full_path,
                title=title,
                model_name=model_name
            )
            
            option = CheckpointUnetOption(unet_info)
            unet_options.append(option)
    
    # Allow extensions to add their own UNET options
    new_unets = script_callbacks.list_unets_callback()
    unet_options.extend(new_unets)


class SdUnetOption:
    """Base class for UNET options - similar to SdTextEncoderOption"""
    model_name = None
    """name of related checkpoint - this option will be selected automatically if checkpoint name matches"""
    
    label = None
    """name of the UNET in UI"""
    
    def create_unet(self):
        """returns SdUnet object to be used instead of built-in UNET"""
        raise NotImplementedError()

class SdUnet:
    """Base class for UNETs - similar to SdTextEncoder"""
    def __init__(self, unet_patcher):
        self.unet_patcher = unet_patcher
        
    def activate(self):
        """Called when this UNET becomes active"""
        pass
        
    def deactivate(self):
        """Called when this UNET is deactivated"""
        pass

class CheckpointUnetOption(SdUnetOption):
    """UNET option that loads from a checkpoint file"""
    def __init__(self, unet_info: UnetInfo):
        self.unet_info = unet_info
        self.model_name = unet_info.model_name
        self.label = unet_info.title
        
    def create_unet(self):
        return CheckpointUnet(self.unet_info)

class CheckpointUnet(SdUnet):
    """UNET loaded from a checkpoint file"""
    def __init__(self, unet_info: UnetInfo):
        self.unet_info = unet_info
        self.unet_patcher = None
        
    def _load_unet_from_checkpoint(self):
        """Load UNET model from checkpoint file"""
        if self.unet_patcher is not None:
            return self.unet_patcher
            
        timer = Timer()
        
        try:
            # Load the UNET using forge_loader functionality
            forge_sd = load_diffusion_model(self.unet_info.filename)
            if forge_sd is None or forge_sd.unet is None:
                raise RuntimeError(f"Could not load UNET from: {self.unet_info.filename}")
            
            timer.record("load unet")
            
            self.unet_patcher = forge_sd.unet
            print(f"Loaded UNET {self.unet_info.title} in {timer.summary()}")
            
        except Exception as e:
            logging.error(f"Error loading UNET {self.unet_info.filename}: {e}")
            raise
            
        return self.unet_patcher
        
    def activate(self):
        """Activate this UNET"""
        if self.unet_patcher is None:
            self._load_unet_from_checkpoint()
            
    def deactivate(self):
        """Deactivate this UNET"""
        # Could optionally unload from VRAM here
        pass

def get_unet_option(option=None):
    """Get UNET option by name or automatic selection"""
    option = option or getattr(shared.opts, 'sd_unet', 'Automatic')

    if option == "None":
        return None

    if option == "Automatic":
        # Try to find a UNET with matching name to current checkpoint
        if shared.sd_model and hasattr(shared.sd_model, 'sd_checkpoint_info'):
            checkpoint_name = shared.sd_model.sd_checkpoint_info.model_name
            
            matching_options = [x for x in unet_options if hasattr(x, 'model_name') and x.model_name == checkpoint_name]
            if matching_options:
                return matching_options[0]
                
        return None  # Use checkpoint's built-in UNET

    # Find by label
    return next((x for x in unet_options if hasattr(x, 'label') and x.label == option), None)


def apply_unet(option=None):
    """Apply a UNET option"""
    global current_unet_option, current_unet

    new_option = get_unet_option(option)
    if new_option == current_unet_option:
        return

    # Deactivate current UNET
    if current_unet is not None:
        print(f"Deactivating UNET: {current_unet.option.label}")
        current_unet.deactivate()

    current_unet_option = new_option
    
    if current_unet_option is None:
        current_unet = None
        print("Using checkpoint's built-in UNET")
        return

    # Activate new UNET
    current_unet = current_unet_option.create_unet()
    current_unet.option = current_unet_option
    print(f"Activating UNET: {current_unet.option.label}")
    current_unet.activate()

def get_current_unet():
    """Get currently active UNET, or None if using checkpoint's built-in"""
    return current_unet

def reload_unet_list():
    """Reload the list of available UNETs"""
    list_unets()

# Legacy classes for backward compatibility
class SdUnetOption_Legacy:
    model_name = None
    """name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this"""

    label = None
    """name of the unet in UI"""

    def create_unet(self):
        """returns SdUnet object to be used as a Unet instead of built-in unet when making pictures"""
        raise NotImplementedError()

class SdUnet_Legacy(torch.nn.Module):
    def forward(self, x, timesteps, context, *args, **kwargs):
        raise NotImplementedError()

    def activate(self):
        pass

    def deactivate(self):
        pass

def create_unet_forward(original_forward):
    def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
        if current_unet is not None:
            return current_unet.forward(x, timesteps, context, *args, **kwargs)

        return original_forward(self, x, timesteps, context, *args, **kwargs)

    return UNetModel_forward

# Initialize on module load
list_unets()

