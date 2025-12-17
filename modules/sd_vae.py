import os
import collections
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes
from ldm_patched.modules import diffusers_convert

import glob
from copy import deepcopy


vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}


base_vae = None
loaded_vae_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()


def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None

    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')

    return sha256[0:10] if sha256 else None


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_vae_list():
    vae_dict.clear()

    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(vae_path, '**/*.ckpt'),
        os.path.join(vae_path, '**/*.pt'),
        os.path.join(vae_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    for vae_file in vae_dict.values():
        if os.path.basename(vae_file).startswith(checkpoint_path):
            return vae_file

    return None


@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.vae, self.source


def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file) -> VaeResolution:
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_from_setting()

    return res


def load_vae_dict(filename, map_location):
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    vae_dict_1 = _normalize_vae_state_dict(vae_dict_1, source=filename)
    return vae_dict_1


def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global vae_dict, base_vae, loaded_vae_file
    # save_settings = False

    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        if cache_enabled and vae_file in checkpoints_loaded:
            # use vae checkpoint cache
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            vae_dict_1 = _normalize_vae_state_dict(checkpoints_loaded[vae_file], source=f"cache:{vae_file}")
            _load_vae_dict(model, vae_dict_1)
        else:
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            print(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # cache newly loaded vae
                checkpoints_loaded[vae_file] = vae_dict_1.copy()

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    # Prefer loading into the Forge VAE wrapper if available, because decode/encode in Forge uses it directly.
    forge_vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    target_first_stage = getattr(forge_vae, "first_stage_model", None) if forge_vae is not None else None
    if target_first_stage is None:
        target_first_stage = model.first_stage_model

    # Capture optional Flux2 latent BN stats for the wrapper (Comfy-style weights sometimes store these as bn.* buffers).
    if forge_vae is not None and hasattr(forge_vae, "set_latent_bn_stats"):
        if "bn.running_mean" in vae_dict_1 and "bn.running_var" in vae_dict_1:
            forge_vae.set_latent_bn_stats(vae_dict_1["bn.running_mean"], vae_dict_1["bn.running_var"])
            print("VAE load: detected bn.* latent stats (ignored for compatibility).")
        else:
            # Prevent stale stats when switching VAEs.
            forge_vae.set_latent_bn_stats(None, None)

    state_dict_to_load = vae_dict_1
    if any(k.startswith("bn.") for k in vae_dict_1.keys()):
        # The underlying AutoencoderKL does not define a `bn` module; strip these keys before strict loading.
        state_dict_to_load = {k: v for k, v in vae_dict_1.items() if not k.startswith("bn.")}

    try:
        # strict=True ensures we don't silently ignore a VAE that didn't actually load.
        target_first_stage.load_state_dict(state_dict_to_load, strict=True)
        # Keep the canonical reference consistent.
        if forge_vae is not None:
            forge_vae.first_stage_model = target_first_stage
            model.first_stage_model = target_first_stage
            print(f"VAE load: loaded into existing Forge VAE wrapper ({_summarize_vae_first_stage(target_first_stage)})")
        else:
            print(f"VAE load: loaded into model.first_stage_model ({_summarize_vae_first_stage(target_first_stage)})")
    except RuntimeError as e:
        # If the current VAE architecture doesn't match, rebuild a VAE wrapper that matches the weights
        # (e.g. flux2/BN VAEs) and make sure Forge references are updated too.
        print(f"VAE load: state_dict mismatch, rebuilding VAE wrapper ({type(e).__name__}: {e})")

        from ldm_patched.modules import sd as ldm_sd

        device = getattr(forge_vae, "device", None)
        rebuilt = ldm_sd.VAE(sd=vae_dict_1, device=device)
        rebuilt.throw_exception_if_invalid()

        # Preserve packing behavior driven by the UNet model config (needed for flux2 packed latents).
        try:
            model_config = model.forge_objects.unet.model.model_config if hasattr(model, "forge_objects") else None
            if model_config is not None and hasattr(model_config, "packed_vae_latent_channels"):
                rebuilt.set_packed_latents(
                    model_config.packed_vae_latent_channels,
                    getattr(model_config, "packed_vae_spatial_factor", 2),
                )
        except Exception as pack_e:
            print(f"VAE load: warning, could not apply packed-latent settings ({pack_e})")

        if hasattr(model, "forge_objects") and getattr(model.forge_objects, "vae", None) is not None:
            model.forge_objects.vae = rebuilt
            if hasattr(model, "forge_objects_original") and getattr(model.forge_objects_original, "vae", None) is not None:
                model.forge_objects_original.vae = rebuilt
            if hasattr(model, "forge_objects_after_applying_lora") and getattr(model.forge_objects_after_applying_lora, "vae", None) is not None:
                model.forge_objects_after_applying_lora.vae = rebuilt

        model.first_stage_model = rebuilt.first_stage_model
        print("VAE load: rebuilt VAE wrapper is now active.")


def _summarize_vae_first_stage(first_stage_model) -> str:
    try:
        w = first_stage_model.decoder.conv_in.weight
        shape = tuple(w.shape)
        # The scalar read helps confirm that weights actually changed when swapping VAEs.
        scalar = float(w.reshape(-1)[0].item())
        return f"decoder.conv_in.weight={shape}, sample={scalar:.6g}"
    except Exception:
        return f"type={type(first_stage_model).__name__}"


def _looks_like_vae_state_dict(sd: dict) -> bool:
    if not sd:
        return False
    # Common LDM VAE keys
    if "decoder.conv_in.weight" in sd or "encoder.conv_in.weight" in sd:
        return True
    # Diffusers VAE keys (before conversion)
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in sd:
        return True
    # A few other common patterns
    return any(k.startswith("decoder.") or k.startswith("encoder.") for k in sd.keys())


def _strip_prefix(sd: dict, prefix: str) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _normalize_vae_state_dict(sd: dict, source: str = "unknown") -> dict:
    """
    Normalize external VAE checkpoints into the expected (unprefixed) LDM format.

    External VAEs (especially flux2-based ones) often store weights under prefixes like
    `first_stage_model.` or `vae.` which must be stripped before loading or format detection.
    """
    if not sd:
        return sd

    if _looks_like_vae_state_dict(sd):
        out = sd
    else:
        # Try common prefixes used by checkpoints and loaders.
        prefixes = [
            "first_stage_model.",
            "vae.",
            "model.first_stage_model.",
            "model.vae.",
            "module.first_stage_model.",
            "module.vae.",
        ]
        out = sd
        for p in prefixes:
            cand = _strip_prefix(sd, p)
            if _looks_like_vae_state_dict(cand):
                print(f"VAE load: stripped prefix {p!r} from {source}")
                out = cand
                break

    # Convert diffusers VAE format to LDM format after prefix normalization.
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in out:
        out = diffusers_convert.convert_vae_state_dict(out)

    return out


def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()
    else:
        vae_source = "from function argument"

    if loaded_vae_file == vae_file:
        return

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_vae(sd_model, vae_file, vae_source)

    sd_hijack.model_hijack.hijack(sd_model)

    script_callbacks.model_loaded_callback(sd_model)

    print("VAE weights loaded.")
    return sd_model
