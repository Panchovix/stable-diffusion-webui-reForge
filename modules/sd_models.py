import collections
import importlib
import os
import sys
import threading
import enum

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from urllib import request
import ldm.modules.midas as midas
import gc

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches
from modules.timer import Timer
import numpy as np
from modules_forge import forge_loader
import modules_forge.ops as forge_ops
from ldm_patched.modules.ops import manual_cast
from ldm_patched.modules import model_management as model_management
import ldm_patched.modules.model_patcher
import weakref
import logging as log

import ldm_patched.modules.utils
from ldm_patched.modules.patcher_extension import PatcherInjection

# Store original functions
original_set_attr = ldm_patched.modules.utils.set_attr
original_set_attr_param = ldm_patched.modules.utils.set_attr_param

# Create VAE structure preservation class
class VAEStructurePreserver(PatcherInjection):
    """Injection that preserves VAE structure during model unloading"""
    
    def __init__(self):
        self.preserved_vae_refs = set()
        
    def inject(self, model_patcher):
        # Track important VAE references when we inject
        self.preserved_vae_refs = set()
        model = model_patcher.model
        
        if hasattr(model, 'forge_objects') and hasattr(model.forge_objects, 'vae'):
            vae = model.forge_objects.vae
            self.preserved_vae_refs.add(id(vae))
            
            if hasattr(vae, 'model'):
                vae_model = vae.model
                self.preserved_vae_refs.add(id(vae_model))
                
                if hasattr(vae_model, 'encoder'):
                    self.preserved_vae_refs.add(id(vae_model.encoder))
                
                if hasattr(vae_model, 'decoder'):
                    self.preserved_vae_refs.add(id(vae_model.decoder))
                    
            if hasattr(vae, 'patcher'):
                self.preserved_vae_refs.add(id(vae.patcher))
    
    def eject(self, model_patcher):
        # Nothing needed on ejection
        pass

# Safer versions of the attribute setting functions
def safer_set_attr(obj, attr, value):
    """A safer version of set_attr that checks for None objects and missing attributes"""
    try:
        attrs = attr.split(".")
        for name in attrs[:-1]:
            if obj is None:
                return None
            if not hasattr(obj, name):
                return None
            obj = getattr(obj, name)
            
        if obj is None:
            return None
            
        prev = getattr(obj, attrs[-1], None)
        setattr(obj, attrs[-1], value)  # Just set the value directly, don't convert to Parameter
        return prev  # Return previous value instead of deleting it
    except Exception as e:
        # print(f"Error in safer_set_attr for {attr}: {str(e)}")
        return None

def safer_set_attr_param(obj, attr, value):
    """A safer version of set_attr_param that handles None values and missing attributes"""
    try:
        if value is None:
            return None
        return safer_set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))
    except Exception as e:
        # print(f"Error in safer_set_attr_param for {attr}: {str(e)}")
        return None

# Replace the original functions
ldm_patched.modules.utils.set_attr = safer_set_attr
ldm_patched.modules.utils.set_attr_param = safer_set_attr_param

# Add a validation function for the VAE
def validate_and_fix_vae(sd_model):
    """Checks if VAE has required components and attempts to fix if not"""
    if not hasattr(sd_model, 'forge_objects'):
        return
        
    if not hasattr(sd_model.forge_objects, 'vae'):
        print("Warning: Model has no VAE object")
        return
        
    vae = sd_model.forge_objects.vae
    
    # Check model attribute exists
    if not hasattr(vae, 'model'):
        print("Reloading VAE")
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(sd_model.sd_checkpoint_info.filename).tuple()
        sd_vae.load_vae(sd_model, vae_file, vae_source)
        return
        
    # Check encoder/decoder exist
    missing_components = []
    if not hasattr(vae.model, 'encoder') or vae.model.encoder is None:
        missing_components.append('encoder')
        
    if not hasattr(vae.model, 'decoder') or vae.model.decoder is None:
        missing_components.append('decoder')
        
    if len(missing_components) > 0:
        print(f"Warning: VAE missing components: {', '.join(missing_components)}, attempting to reload VAE")
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(sd_model.sd_checkpoint_info.filename).tuple()
        sd_vae.load_vae(sd_model, vae_file, vae_source)

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded = collections.OrderedDict()


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging, CLIPModel  # noqa: F401

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    """called once at startup to do various one-time tasks related to SD models"""

    os.makedirs(model_path, exist_ok=True)

    enable_midas_autodownload()
    patch_given_betas()


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
        expected_sha256 = None
    else:
        model_url = "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors"

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="NoobAI-XL-v1.1.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = { # Converts SD 2.1 Turbo from SGM to LDM format.
    'conditioner.embedders.0.': 'cond_stage_model.',
}


def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    sd = {}
    for k, v in pl_sd.items():
        if is_sd2_turbo:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
        else:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
             errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

def complete_model_teardown(model):
    """Completely tear down a model by breaking all references to its components"""
    if model is None:
        return
        
    model_name = "Unknown"
    if hasattr(model, 'sd_checkpoint_info') and hasattr(model.sd_checkpoint_info, 'title'):
        model_name = model.sd_checkpoint_info.title
    elif hasattr(model, 'filename'):
        model_name = model.filename
        
    print(f"Performing complete teardown of model: {model_name}")
    
    # Create a set of objects to preserve (don't nullify these)
    preserve_attributes = set()
    
    # Preserve critical VAE structure
    if hasattr(model, 'forge_objects'):
        preserve_attributes.add(id(model.forge_objects))
        
        if hasattr(model.forge_objects, 'vae'):
            vae = model.forge_objects.vae
            preserve_attributes.add(id(vae))
            
            if hasattr(vae, 'model'):
                preserve_attributes.add(id(vae.model))
                
                # Preserve encoder/decoder structure but not their internal weights
                if hasattr(vae.model, 'encoder'):
                    preserve_attributes.add(id(vae.model.encoder))
                    
                if hasattr(vae.model, 'decoder'):
                    preserve_attributes.add(id(vae.model.decoder))
                    
                # Also preserve quantizer if it exists
                if hasattr(vae.model, 'quantize'):
                    preserve_attributes.add(id(vae.model.quantize))
                    
            # Preserve patcher for VRAM offloading
            if hasattr(vae, 'patcher'):
                preserve_attributes.add(id(vae.patcher))
    
    # Safer implementation that avoids errors and preserves critical structures
    def replace_attributes(obj, path="", visited=None, depth=0):
        if visited is None:
            visited = set()
            
        # Limit recursion depth for safety
        if depth > 10:
            return
            
        # Don't process the same object twice
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        # Skip objects that need to be preserved
        if obj_id in preserve_attributes:
            # Process children of preserved objects, but don't nullify the structure
            pass
        
        try:
            # Handle torch.nn.Module
            if hasattr(obj, 'parameters') and hasattr(obj, 'named_parameters'):
                # Handle parameters
                try:
                    for name, param in list(obj.named_parameters(recurse=False)):
                        try:
                            if hasattr(param, 'data'):
                                # Don't nullify parameters of preserved objects
                                if obj_id not in preserve_attributes:
                                    param.data = None
                        except:
                            pass
                except:
                    pass
                    
                # Handle buffers
                try:
                    for name, buffer in list(obj.named_buffers(recurse=False)):
                        try:
                            if hasattr(buffer, 'data'):
                                if obj_id not in preserve_attributes:
                                    buffer.data = None
                        except:
                            pass
                except:
                    pass
                    
                # Process child modules
                try:
                    for name, module in list(obj.named_children()):
                        try:
                            replace_attributes(module, f"{path}.{name}", visited, depth+1)
                            # Only nullify if not in preserve list
                            if id(obj) not in preserve_attributes and id(module) not in preserve_attributes:
                                setattr(obj, name, None)
                        except:
                            pass
                except:
                    pass
                
            # Handle dictionaries
            elif isinstance(obj, dict):
                for key in list(obj.keys()):
                    try:
                        val = obj[key]
                        if hasattr(val, 'parameters') or hasattr(val, 'numel'):
                            replace_attributes(val, f"{path}[{key}]", visited, depth+1)
                            # Only nullify if not in preserve list
                            if obj_id not in preserve_attributes and id(val) not in preserve_attributes:
                                obj[key] = None
                    except:
                        pass
                # Don't clear dictionaries that might contain preserved objects
                if obj_id not in preserve_attributes:
                    try:
                        obj.clear()
                    except:
                        pass
                    
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                for i, item in enumerate(obj):
                    try:
                        if hasattr(item, 'parameters') or hasattr(item, 'numel'):
                            replace_attributes(item, f"{path}[{i}]", visited, depth+1)
                    except:
                        pass
        except:
            pass
                    
    # Safely process the model's attribute tree
    for attr_name in dir(model):
        if attr_name.startswith('__'):
            continue
            
        try:
            attr = getattr(model, attr_name)
            if attr is not None:
                if isinstance(attr, (dict, list, tuple)) or hasattr(attr, 'parameters') or hasattr(attr, 'numel'):
                    replace_attributes(attr, attr_name)
                    # Only nullify attributes that aren't in the preserve list
                    if id(attr) not in preserve_attributes:
                        setattr(model, attr_name, None)
        except:
            pass
    
    # Clear CUDA cache forcefully
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            pass
    
    # Run GC multiple times
    for _ in range(3):
        gc.collect()
        
    print(f"Model teardown completed")

# Global tracking to find leaks
models_loaded_count = 0
peak_memory_usage = 0

def force_memory_deallocation():
    """Force deallocation of memory using memory profiling and weak references"""
    import gc
    import psutil
    
    # 1. Get pre-cleanup memory for comparison
    process = psutil.Process()
    pre_mem = process.memory_info().rss
    
    # 2. Clear all caches that might hold model references
    global checkpoints_loaded
    if len(checkpoints_loaded) > 0:
        print(f"Clearing {len(checkpoints_loaded)} cached state dictionaries")
        checkpoints_loaded.clear()
    
    # 3. Clear checkpoints_list references to only keep essential information
    for key in list(checkpoints_list):
        info = checkpoints_list[key]
        if hasattr(info, 'metadata') and info.metadata:
            # Save only minimal metadata to preserve functionality
            minimal_metadata = {}
            if 'ss_sd_model_name' in info.metadata:
                minimal_metadata['ss_sd_model_name'] = info.metadata['ss_sd_model_name']
            info.metadata = minimal_metadata
    
    # 4. Clear torch CUDA caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 5. Use specialized weakref to identify and break circular references
    # This trick helps identify objects that aren't being collected
    tracked_objects = []
    
    def on_delete(ref):
        tracked_objects.remove(ref)
    
    # Track large objects for collection
    large_objects = [obj for obj in gc.get_objects() 
                     if (isinstance(obj, torch.Tensor) and 
                         not obj.is_cuda and 
                         obj.numel() > 1e6)]
    
    for obj in large_objects:
        ref = weakref.ref(obj, on_delete)
        tracked_objects.append(ref)
    
    # Force collection
    del large_objects
    gc.collect()
    
    # Report uncollected objects
    if tracked_objects:
        log.debug(f"Warning: {len(tracked_objects)} large tensor objects were not collected")
    
    # 6. Run several garbage collection passes
    for i in range(3):
        count = gc.collect()
        if count == 0:
            break
        print(f"GC pass {i+1}: collected {count} objects")
    
    # 7. Report memory change
    post_mem = process.memory_info().rss
    mem_diff = (post_mem - pre_mem) / (1024 * 1024)
    print(f"Memory change: {mem_diff:.2f} MB ({post_mem/(1024*1024*1024):.2f} GB total)")
    
    # Special measure for large leaks
    global peak_memory_usage, models_loaded_count
    models_loaded_count += 1
    peak_memory_usage = max(peak_memory_usage, post_mem)
    
    return f"Memory cleanup: {post_mem/(1024*1024*1024):.2f} GB used"

disable_checkpoint_caching = True  # Global flag to completely disable checkpoint caching

def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    global disable_checkpoint_caching
    
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")
    
    # Completely disable checkpoint caching - always load fresh
    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    
    # Use a more direct loading approach to avoid duplicate copies
    if checkpoint_info.is_safetensors:
        import safetensors.torch
        # Load directly to the appropriate device
        device = shared.weight_load_location or model_management.get_torch_device()
        
        if shared.opts.disable_mmap_load_safetensors:
            with torch.no_grad():
                data = open(checkpoint_info.filename, 'rb').read()
                res = safetensors.torch.load(data)
                # Move tensors to target device
                res = {k: v.to(device) for k, v in res.items()}
                del data  # Immediately delete raw data
        else:
            # Direct file loading to device
            res = safetensors.torch.load_file(
                checkpoint_info.filename, 
                device=device
            )
    else:
        # For regular checkpoints
        res = torch.load(
            checkpoint_info.filename, 
            map_location=shared.weight_load_location or model_management.get_torch_device()
        )
        res = get_state_dict_from_checkpoint(res)
    
    timer.record("load weights from disk")
    return res


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


def check_fp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_fp8 = False
    elif shared.opts.fp8_storage == "Enable":
        enable_fp8 = True
    elif getattr(model, "is_sdxl", False) and shared.opts.fp8_storage == "Enable for SDXL":
        enable_fp8 = True
    else:
        enable_fp8 = False
    return enable_fp8


def set_model_type(model, state_dict):
    model.is_sd1 = False
    model.is_sd2 = False
    model.is_sdxl = False
    model.is_ssd = False
    model.is_sd3 = False

    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        model.is_sd3 = True
        model.model_type = ModelType.SD3
    elif hasattr(model, 'conditioner'):
        model.is_sdxl = True

        if 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys():
            model.is_ssd = True
            model.model_type = ModelType.SSD
        else:
            model.model_type = ModelType.SDXL
    elif hasattr(model.cond_stage_model, 'model'):
        model.is_sd2 = True
        model.model_type = ModelType.SD2
    else:
        model.is_sd1 = True
        model.model_type = ModelType.SD1


def set_model_fields(model):
    if not hasattr(model, 'latent_channels'):
        model.latent_channels = 4


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    return


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                os.mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def patch_given_betas():
    import ldm.models.diffusion.ddpm

    def patched_register_schedule(*args, **kwargs):
        """a modified version of register_schedule function that converts plain list from Omegaconf into numpy"""

        if isinstance(args[1], ListConfig):
            args = (args[0], np.array(args[1]), *args[2:])

        original_register_schedule(*args, **kwargs)

    original_register_schedule = patches.patch(__name__, ldm.models.diffusion.ddpm.DDPM, 'register_schedule', patched_register_schedule)


def repair_config(sd_config, state_dict=None):
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if hasattr(sd_config.model.params, 'unet_config'):
        if shared.cmd_opts.no_half:
            sd_config.model.params.unet_config.params.use_fp16 = False
        elif shared.cmd_opts.upcast_sampling or shared.cmd_opts.precision == "half":
            sd_config.model.params.unet_config.params.use_fp16 = True

    if hasattr(sd_config.model.params, 'first_stage_config'):
        if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
            sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # For UnCLIP-L, override the hardcoded karlo directory
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params, "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)

    # Do not use checkpoint for inference.
    # This helps prevent extra performance overhead on checking parameters.
    # The perf overhead is about 100ms/it on 4090 for SDXL.
    if hasattr(sd_config.model.params, "network_config"):
        sd_config.model.params.network_config.params.use_checkpoint = False
    if hasattr(sd_config.model.params, "unet_config"):
        sd_config.model.params.unet_config.params.use_checkpoint = False



def rescale_zero_terminal_snr_abar(alphas_cumprod):
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def apply_alpha_schedule_override(sd_model, p=None, force_apply=False):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if shared.opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params['Downcast alphas_cumprod'] = shared.opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    if shared.opts.sd_noise_schedule == "Zero Terminal SNR" or (hasattr(sd_model, 'ztsnr') and sd_model.ztsnr) or force_apply:
        if p is not None and shared.opts.sd_noise_schedule != "Default":
            p.extra_generation_params['Noise Schedule'] = shared.opts.sd_noise_schedule
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    load_model()

                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, v, already_loaded=False):
        self.sd_model = v
        if already_loaded:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info


model_data = SdModelData()


def get_empty_cond(sd_model):

    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, 'get_learned_conditioning'):
        d = sd_model.get_learned_conditioning([""])
    else:
        d = sd_model.cond_stage_model([""])

    if isinstance(d, dict):
        d = d['crossattn']

    return d


def send_model_to_cpu(m):
    pass


def model_target_device(m):
    return devices.device


def send_model_to_device(m):
    pass


def send_model_to_trash(m):
    pass


def instantiate_from_config(config, state_dict=None):
    constructor = get_obj_from_str(config["target"])

    params = {**config.get("params", {})}

    if state_dict and "state_dict" in params and params["state_dict"] is None:
        params["state_dict"] = state_dict

    return constructor(**params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def clear_python_cache():
    """Clear Python's internal module cache to reduce memory usage"""
    # Clear module cache - can help with memory leaks
    count = 0
    for module_name in list(sys.modules.keys()):
        # Don't remove essential modules
        if module_name in ('sys', 'os', 'gc', 'torch', 'numpy'):
            continue
        # Don't remove main modules
        if module_name.startswith('__main__') or module_name == '__main__':
            continue
        # Focus on model-related modules that might hold large tensors
        if 'model' in module_name or 'unet' in module_name or 'vae' in module_name or 'tensor' in module_name:
            try:
                del sys.modules[module_name]
                count += 1
            except:
                pass
    
    print(f"Cleared {count} modules from Python module cache")
    return count

def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup to address RAM leaks - safer approach"""
    global checkpoints_loaded
    
    print("Performing aggressive memory cleanup...")
    
    # 1. Clear checkpoint cache
    if len(checkpoints_loaded) > 0:
        print(f"Clearing {len(checkpoints_loaded)} cached checkpoints from memory")
        checkpoints_loaded.clear()
    
    # 2. Clear torch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 3. Run garbage collection
    collected = gc.collect()
    print(f"GC: collected {collected} objects")
    
    # 4. Get current memory usage for logging
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current memory usage: RSS={mem_info.rss/(1024*1024*1024):.2f} GB, VMS={mem_info.vms/(1024*1024*1024):.2f} GB")
    
    return f"Memory cleanup complete. Current usage: {mem_info.rss/(1024*1024*1024):.2f} GB"

def load_model(checkpoint_info=None, already_loaded_state_dict=None):
    import logging as log
    global model_data

    checkpoint_info = checkpoint_info or select_checkpoint()
    timer = Timer()

    # Check if the model is already loaded
    for i, loaded_model in enumerate(model_data.loaded_sd_models):
        if loaded_model.filename == checkpoint_info.filename:
            log.debug(f"Using already loaded model {loaded_model.sd_checkpoint_info.title}")
            # Set this model as active by moving it to the front
            model_data.loaded_sd_models.remove(loaded_model)
            model_data.loaded_sd_models.insert(0, loaded_model)
            model_data.set_sd_model(loaded_model, already_loaded=True)
            return loaded_model

    # Enforce model limit
    while len(model_data.loaded_sd_models) >= shared.opts.sd_checkpoints_limit:
        unload_first_loaded_model()
    
    # Force memory deallocation
    force_memory_deallocation()
    timer.record("memory cleanup")

    current_loaded_models = len(model_data.loaded_sd_models)
    print(f"Loading model {checkpoint_info.title} ({current_loaded_models + 1} of {shared.opts.sd_checkpoints_limit})")

    # State dict handling with explicit scoping
    sd_model = None
    try:
        if already_loaded_state_dict is not None:
            state_dict = already_loaded_state_dict
        else:
            state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

        # Load the model
        sd_model = forge_loader.load_model_for_a1111(timer=timer, checkpoint_info=checkpoint_info, state_dict=state_dict)
        sd_model.filename = checkpoint_info.filename
    finally:
        # Always clear state dict, even on failure
        if 'state_dict' in locals():
            del state_dict
            gc.collect()

    # Only proceed if model loaded successfully
    if sd_model is not None:
        model_data.loaded_sd_models.insert(0, sd_model)  # Add new model to the front
        model_data.set_sd_model(sd_model)
        model_data.was_loaded_at_least_once = True
        vae_preserver = VAEStructurePreserver()
        sd_model.forge_objects.vae_preserver = vae_preserver
        # sd_model.set_injections("vae_protection", [vae_preserver]) #This seems to give issues, have to check if injection is needed for the fix

        shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
        sd_vae.load_vae(sd_model, vae_file, vae_source)
        timer.record("load VAE")
        validate_and_fix_vae(sd_model)
        timer.record("load textual inversion embeddings")

        script_callbacks.model_loaded_callback(sd_model)
        timer.record("scripts callbacks")

        with torch.no_grad():
            sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)
        timer.record("calculate empty prompt")

        print(f"Model {checkpoint_info.title} loaded in {timer.summary()}.")
        
        # One final cleanup to release any temporary objects
        force_memory_deallocation()
    else:
        print("Error: Model failed to load")

    return sd_model

def set_model_active(model_index):
    """Set a specific model as active based on its index in loaded_sd_models"""
    global model_data
    
    try:
        model_index = int(model_index)
    except:
        return "Model index must be a number"
    
    if not model_data.loaded_sd_models:
        return "No models currently loaded"
    
    if model_index < 0 or model_index >= len(model_data.loaded_sd_models):
        return f"Invalid model index: {model_index}, valid range is 0-{len(model_data.loaded_sd_models)-1}"
    
    # Get the model we want to activate
    model_to_activate = model_data.loaded_sd_models[model_index]
    
    # If it's already active, no need to do anything
    if model_data.sd_model == model_to_activate:
        return f"Model {model_to_activate.sd_checkpoint_info.title} is already active"
    
    # Move the model to the front of the list and set it as active
    model_data.loaded_sd_models.remove(model_to_activate)
    model_data.loaded_sd_models.insert(0, model_to_activate)
    model_data.set_sd_model(model_to_activate, already_loaded=True)
    
    return f"Activated model: {model_to_activate.sd_checkpoint_info.title}"

def unload_first_loaded_model():
    """Completely unload the first loaded model using aggressive teardown"""
    global model_data
    if not model_data.loaded_sd_models:
        return

    first_loaded_model = model_data.loaded_sd_models.pop(-1)  # Remove the last item (first loaded)
    
    # Get the model name safely
    if hasattr(first_loaded_model, 'sd_checkpoint_info'):
        if hasattr(first_loaded_model.sd_checkpoint_info, 'title'):
            model_name = first_loaded_model.sd_checkpoint_info.title
        else:
            model_name = str(first_loaded_model.sd_checkpoint_info)
    elif hasattr(first_loaded_model, 'filename'):
        model_name = first_loaded_model.filename
    else:
        model_name = "Unknown"
        
    print(f"Unloading first loaded model: {model_name}")
    
    # Complete teardown of the model
    complete_model_teardown(first_loaded_model)
    
    # Force reference removal
    first_loaded_model = None
    
    # Force GC
    gc.collect()
    gc.collect()
    
    # Get memory statistics
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current memory usage after unload: RSS={mem_info.rss/(1024*1024*1024):.2f} GB")
    
    return None  # Return None instead of the model to ensure no references remain

def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    pass


def reload_model_weights(sd_model=None, info=None, forced_reload=False):
    return load_model(info)


def unload_model_weights(model=None):
    """Unload the currently active model to RAM"""
    if model is None:
        model = model_data.sd_model
        
    if model is None:
        return "No model is currently loaded"
        
    print(f"Unloading model weights for {model.sd_checkpoint_info.title}")
    
    if hasattr(model, 'model_unload'):
        model.model_unload()
    elif hasattr(model, 'to') and hasattr(model, 'offload_device'):
        model.to(model.offload_device)
    else:
        model.to('cpu')
    
    model_management.soft_empty_cache(force=True)
    gc.collect()
    
    return f"Unloaded model {model.sd_checkpoint_info.title} to RAM"

def load_model_to_device(model=None):
    """Load a model from RAM to VRAM"""
    if model is None:
        model = model_data.sd_model
        
    if model is None:
        return "No model is currently loaded"
    
    print(f"Loading model weights for {model.sd_checkpoint_info.title} to device")
    
    if hasattr(model, 'model_load'):
        model.model_load()
    else:
        device = model_management.get_torch_device()
        model.to(device)
    
    return f"Loaded model {model.sd_checkpoint_info.title} to device"

def list_loaded_models():
    """Return a list of all currently loaded models"""
    if not model_data.loaded_sd_models:
        return "No models currently loaded"
    
    import psutil
    process = psutil.Process()
    total_ram = process.memory_info().rss / (1024 * 1024 * 1024)
    
    result = f"Currently loaded models (Total RAM: {total_ram:.2f} GB):\n"
    for i, model in enumerate(model_data.loaded_sd_models):
        active = " (active)" if model == model_data.sd_model else ""
        result += f"[{i}] {model.sd_checkpoint_info.title}{active}\n"
    
    return result

def unload_specific_model(model_index):
    """Unload a specific model by index"""
    try:
        model_index = int(model_index)
    except:
        return "Model index must be a number"
    
    if not model_data.loaded_sd_models:
        return "No models currently loaded"
    
    if model_index < 0 or model_index >= len(model_data.loaded_sd_models):
        return f"Invalid model index: {model_index}, valid range is 0-{len(model_data.loaded_sd_models)-1}"
    
    model_to_unload = model_data.loaded_sd_models[model_index]
    name = model_to_unload.sd_checkpoint_info.title
    
    # Check if we're unloading the active model
    is_active = model_to_unload == model_data.sd_model
    
    # If unloading active model, switch to another model first
    if is_active and len(model_data.loaded_sd_models) > 1:
        new_index = 0 if model_index != 0 else 1
        new_active_model = model_data.loaded_sd_models[new_index]
        print(f"Switching active model from {name} to {new_active_model.sd_checkpoint_info.title}")
        model_data.set_sd_model(new_active_model, already_loaded=True)
    
    # Remove from list
    model_data.loaded_sd_models.pop(model_index)
    
    # Unload model
    if hasattr(model_to_unload, 'model_unload'):
        print(f"Calling model_unload() for {name}")
        model_to_unload.model_unload()
    elif hasattr(model_to_unload, 'to') and hasattr(model_to_unload, 'offload_device'):
        print(f"Moving {name} to {model_to_unload.offload_device}")
        model_to_unload.to(model_to_unload.offload_device)
    else:
        print(f"Moving {name} to CPU")
        model_to_unload.to('cpu')
    
    # Force cleanup
    model_management.soft_empty_cache(force=True)
    gc.collect()
    
    status = f"Unloaded model: {name}"
    if is_active and len(model_data.loaded_sd_models) == 0:
        status += "\nWarning: No active model remaining"
    
    return status


def apply_token_merging(sd_model, token_merging_ratio):
    if token_merging_ratio <= 0:
        return

    print(f'token_merging_ratio = {token_merging_ratio}')

    from ldm_patched.contrib.external_tomesd import TomePatcher

    sd_model.forge_objects.unet = TomePatcher().patch(
        model=sd_model.forge_objects.unet,
        ratio=token_merging_ratio
    )

    return
