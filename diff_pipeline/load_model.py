"""
diff_pipeline/load_model.py

Path-based hijack registry for routing model loading through a diffusers pipeline.

The hijack fires in sd_models.load_model() as soon as checkpoint_info is known —
BEFORE the state dict is read from disk.  This lets loaders use diffusers
from_single_file() or from_pretrained() directly on the checkpoint path instead
of converting already-loaded ldm weights.

Usage
-----
Register a (predicate, loader) pair from an extension or startup script:

    from diff_pipeline.load_model import register_path_hijack

    def _is_sdxl(checkpoint_info):
        arch = checkpoint_info.metadata.get("modelspec.architecture", "")
        return "xl" in arch.lower() or "sdxl" in checkpoint_info.name.lower()

    def _load_sdxl(checkpoint_info):
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_info.filename,
            torch_dtype=torch.float16,
        )
        return pipe          # Return the model; return None to fall through

    register_path_hijack(_is_sdxl, _load_sdxl)

If the loader returns None, the normal ldm loading pipeline is used.

Built-in helpers
----------------
  dummy_sdxl_hijack(checkpoint_info)  — ready-to-use SDXL from_single_file loader
  dummy_sd1_hijack(checkpoint_info)   — stub; replace body for SD 1.x
  dummy_sd3_hijack(checkpoint_info)   — stub; replace body for SD3

  is_sdxl_checkpoint(checkpoint_info) — predicate for SDXL detection
  is_sd1_checkpoint(checkpoint_info)  — predicate for SD 1.x detection
  is_sd3_checkpoint(checkpoint_info)  — predicate for SD3 detection

Legacy CLI path (--forge-diffusers-pipeline)
--------------------------------------------
If no path hijack claims the checkpoint, _legacy_sdxl_fallback() runs from
forge_loader.py after normal ldm loading, unchanged from before.
"""

from __future__ import annotations

import logging
import torch
from typing import Callable, Optional, Any

from diff_pipeline._cache import lru_cached

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware-aware dtype selection — bf16 > fp32 > fp16
# ---------------------------------------------------------------------------

def preferred_unet_dtype(device: "torch.device | None" = None) -> torch.dtype:
    """Return the best compute dtype for the UNet on *device*.

    Priority: bf16 > fp32 > fp16

    bf16 shares the same exponent range as fp32 so it avoids the NaN /
    overflow issues that plague fp16 in deep diffusion stacks.  fp32 is the
    safe fallback for hardware without native bf16 (e.g. Turing, RDNA2).
    fp16 is used only on devices that explicitly support it but lack bf16
    (e.g. MPS, Intel XPU with no bf16 fast-path, Volta on Windows).

    The decision is delegated to ``model_management`` so that device-specific
    quirks (MPS, AMD RDNA2, Nvidia 16-series blacklist, XPU …) are handled
    in one place and stay in sync with the rest of the forge backend.
    """
    from ldm_patched.modules.model_management import should_use_bf16, should_use_fp16

    if should_use_bf16(device):
        dtype = torch.bfloat16
    elif should_use_fp16(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    log.info(
        "DiffPipeline dtype selected: %s on %s  "
        "(bf16=native Ampere+/RDNA3 — no overflow risk; "
        "fp16=Turing/Volta/MPS — narrower exponent range, watch for NaN; "
        "fp32=safe fallback — higher VRAM use)",
        dtype, device,
    )
    return dtype


# ---------------------------------------------------------------------------
# Path-hijack registry
# ---------------------------------------------------------------------------

# List of (predicate, loader) pairs, evaluated in registration order.
# predicate: (checkpoint_info) -> bool
# loader:    (checkpoint_info) -> model_or_None
_PATH_HIJACK_REGISTRY: list[tuple[Callable, Callable]] = []


def register_path_hijack(predicate: Callable, loader: Callable) -> None:
    """Register a path-based diffusers hijack.

    Args:
        predicate: callable(checkpoint_info) -> bool
                   Return True if this loader should handle the checkpoint.
        loader:    callable(checkpoint_info) -> model or None
                   Build and return the model, or return None to fall through
                   to the next hijack / normal ldm loading.
    """
    import sys
    print(f"[DEBUG register_path_hijack] module id={id(sys.modules[__name__])}, adding predicate={predicate.__name__}, loader={loader.__name__}")
    _PATH_HIJACK_REGISTRY.append((predicate, loader))
    print(f"[DEBUG register_path_hijack] registry now has {len(_PATH_HIJACK_REGISTRY)} entries")


def unregister_path_hijack(loader: Callable) -> None:
    """Remove a previously registered path hijack by its loader function."""
    global _PATH_HIJACK_REGISTRY
    _PATH_HIJACK_REGISTRY = [(p, ldr) for p, ldr in _PATH_HIJACK_REGISTRY if ldr is not loader]


def maybe_apply_path_hijack(checkpoint_info) -> Optional[Any]:
    """Try each registered path hijack in order.

    Called from sd_models.load_model() BEFORE get_checkpoint_state_dict().

    Only runs when --forge-diffusers-pipeline is set.  If the flag is set but
    no hijack claims the checkpoint (or every loader fails), a message is
    printed on its own line so the user knows the hijack was attempted.

    Returns the loaded model if a hijack claimed it, or None to proceed with
    the normal ldm loading pipeline.
    """
    import sys
    from modules.shared import cmd_opts

    print(f"[DEBUG maybe_apply_path_hijack] module id={id(sys.modules[__name__])}, registry has {len(_PATH_HIJACK_REGISTRY)} entries: {[ldr.__name__ for _, ldr in _PATH_HIJACK_REGISTRY]}")

    if not getattr(cmd_opts, 'forge_diffusers_pipeline', False):
        # Flag not set — path hijack is inactive; use normal ldm loading.
        return None

    if not _PATH_HIJACK_REGISTRY:
        print(
            "\n[Diffusers path hijack] --forge-diffusers-pipeline is set but no path "
            "hijack is registered. Falling back to normal ldm loading.\n"
            "  Register one with: register_path_hijack(predicate, loader)\n"
        )
        return None

    claimed = False
    for predicate, loader in _PATH_HIJACK_REGISTRY:
        try:
            if not predicate(checkpoint_info):
                continue
        except Exception as e:
            print(f"[Diffusers path hijack] Predicate {predicate.__name__} error, skipping: {e}")
            continue

        claimed = True
        try:
            model = loader(checkpoint_info)
            if model is not None:
                print(f"[Diffusers path hijack] {loader.__name__} loaded {checkpoint_info.name}")
                return model
            else:
                print(
                    f"\n[Diffusers path hijack] {loader.__name__} returned None for "
                    f"{checkpoint_info.name}. Falling back to normal ldm loading.\n"
                )
        except Exception as e:
            print(
                f"\n[Diffusers path hijack] {loader.__name__} failed for "
                f"{checkpoint_info.name}:\n  {e}\n"
                f"Falling back to normal ldm loading.\n"
            )

    if not claimed:
        print(
            f"\n[Diffusers path hijack] --forge-diffusers-pipeline is set but no "
            f"registered hijack matched {checkpoint_info.name!r}. "
            f"Falling back to normal ldm loading.\n"
        )

    return None


# ---------------------------------------------------------------------------
# Safetensors header key reader (no weights loaded)
# ---------------------------------------------------------------------------

@lru_cached
def _read_safetensors_tensor_keys(filename: str) -> Optional[set]:
    """Read only tensor key names from a safetensors header — zero weight loading.

    The safetensors format begins with an 8-byte little-endian length followed
    by a JSON object whose top-level keys are tensor names (plus the special
    ``__metadata__`` key).  We parse that JSON and return the non-metadata keys
    as a set so callers can do fast membership tests without touching the actual
    weight data.

    Returns None if the file cannot be read or is not a safetensors file.
    """
    import json
    try:
        with open(filename, "rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                return None
            header_len = int.from_bytes(length_bytes, "little")
            header_json = f.read(header_len)
        obj = json.loads(header_json)
        return {k for k in obj if k != "__metadata__"}
    except Exception:
        return None


def _detect_model_type(checkpoint_info) -> str:
    """Return a canonical model-type string for *checkpoint_info*.

    Detection priority (most reliable → least reliable):

    1. **Diffusers-native** — ``__metadata__._class_name`` written by diffusers
       ``save_pretrained`` / training scripts.
    2. **modelspec** — ``__metadata__.modelspec.architecture`` used by civitai
       and many community tools (when the creator bothers to set it).
    3. **Tensor key inspection** — read tensor names from the safetensors header
       (no weights loaded) and apply the same landmark-key heuristics that
       reForge's ``detect_unet_config`` uses on the full state dict.
    4. **Filename heuristic** — last resort; intentionally conservative.

    Returns one of: ``"sdxl"``, ``"sd3"``, ``"flux"``, ``"sd1"``, ``"sd2"``,
    or ``""`` (unknown).
    """
    meta = checkpoint_info.metadata  # __metadata__ dict, may be empty

    # --- 1. Diffusers _class_name ----------------------------------------
    class_name = meta.get("_class_name", "")
    if class_name:
        cn = class_name.lower()
        if "xl" in cn:
            return "sdxl"
        if "stable_diffusion_3" in cn or "stablediffusion3" in cn:
            return "sd3"
        if "flux" in cn:
            return "flux"
        if "stable_diffusion_pipeline" in cn or cn == "stablediffusionpipeline":
            return "sd1"

    # --- 2. modelspec.architecture ----------------------------------------
    arch = meta.get("modelspec.architecture", "")
    if arch:
        al = arch.lower()
        if "stable-diffusion-xl" in al or "sdxl" in al:
            return "sdxl"
        if "stable-diffusion-3" in al:
            return "sd3"
        if "flux" in al:
            return "flux"
        if al.startswith("stable-diffusion-v1"):
            return "sd1"
        if al.startswith("stable-diffusion-v2"):
            return "sd2"

    # --- 3. Tensor key inspection (safetensors header only) ---------------
    if getattr(checkpoint_info, 'is_safetensors', False):
        keys = _read_safetensors_tensor_keys(checkpoint_info.filename)
        if keys is not None:
            # SD3 / MMDiT
            if any(k.startswith("model.diffusion_model.x_embedder.proj.weight") or
                   "joint_blocks.0.context_block.attn.qkv.weight" in k
                   for k in keys):
                return "sd3"
            # Flux
            if any("double_blocks.0.img_attn.norm.key_norm.scale" in k for k in keys):
                return "flux"
            # SDXL — has conditioner embedders; SD1/2 use cond_stage_model instead
            has_conditioner = any(k.startswith("conditioner.") for k in keys)
            has_cond_stage  = any(k.startswith("cond_stage_model.") for k in keys)
            has_diffusion   = any(k.startswith("model.diffusion_model.") for k in keys)
            if has_conditioner and has_diffusion:
                return "sdxl"
            if has_cond_stage and has_diffusion:
                # SD2 uses an OpenCLIP model with a `model` attribute
                has_sd2_clip = any(k.startswith("cond_stage_model.model.") for k in keys)
                return "sd2" if has_sd2_clip else "sd1"

    # --- 4. Filename heuristic (last resort) ------------------------------
    name_lower = checkpoint_info.name.lower()
    if "sdxl" in name_lower or "sd_xl" in name_lower or "stable-diffusion-xl" in name_lower or "-xl" in name_lower or "_xl" in name_lower:
        return "sdxl"
    if "sd3" in name_lower or "stable-diffusion-3" in name_lower:
        return "sd3"
    if "flux" in name_lower:
        return "flux"

    return ""


# ---------------------------------------------------------------------------
# Built-in predicates
# ---------------------------------------------------------------------------

def is_sdxl_checkpoint(checkpoint_info) -> bool:
    """Return True if checkpoint_info appears to be an SDXL model."""
    return _detect_model_type(checkpoint_info) == "sdxl"


def is_sd1_checkpoint(checkpoint_info) -> bool:
    """Return True if checkpoint_info appears to be an SD 1.x model."""
    return _detect_model_type(checkpoint_info) == "sd1"


def is_sd2_checkpoint(checkpoint_info) -> bool:
    """Return True if checkpoint_info appears to be an SD 2.x model."""
    return _detect_model_type(checkpoint_info) == "sd2"


def is_sd3_checkpoint(checkpoint_info) -> bool:
    """Return True if checkpoint_info appears to be an SD 3 model."""
    return _detect_model_type(checkpoint_info) == "sd3"


def is_flux_checkpoint(checkpoint_info) -> bool:
    """Return True if checkpoint_info appears to be a Flux model."""
    return _detect_model_type(checkpoint_info) == "flux"


# ---------------------------------------------------------------------------
# Built-in loaders (path-based, using from_single_file)
# ---------------------------------------------------------------------------

def dummy_sdxl_hijack(checkpoint_info) -> Any:
    """Load an SDXL checkpoint directly via diffusers from_single_file.

    Returns a DiffusersModelAdapter wrapping the StableDiffusionXLPipeline so
    that the webui's ldm-style interfaces (encode_first_stage, cond_stage_key,
    is_sdxl, etc.) are satisfied without further modification.

    Register with:
        from diff_pipeline.load_model import register_path_hijack
        from diff_pipeline.load_model import is_sdxl_checkpoint, dummy_sdxl_hijack
        register_path_hijack(is_sdxl_checkpoint, dummy_sdxl_hijack)
    """
    from diffusers import StableDiffusionXLPipeline
    from diff_pipeline.adapter import DiffusersModelAdapter, _pick_device, _pick_dtype

    device = _pick_device()
    dtype  = _pick_dtype(device)

    print(f"[diffusers path hijack] Loading SDXL via from_single_file "
          f"(device={device}, dtype={dtype}): {checkpoint_info.filename}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_info.filename,
        torch_dtype=dtype,
        use_safetensors=checkpoint_info.is_safetensors,
    )

    # --- V-prediction detection from safetensors sentinel keys ---------------
    # diffusers from_single_file() defaults to prediction_type="epsilon" when the
    # checkpoint has no embedded diffusers config.  V-pred SDXL checkpoints (e.g.
    # anime models trained with v-parameterization) signal this via empty sentinel
    # tensors at the top level of the state dict: "v_pred" and optionally "ztsnr".
    # Patch both UNet and scheduler configs here so _build_model_sampling_from_pipe
    # picks up V_PREDICTION before it builds model_sampling.
    if getattr(checkpoint_info, 'is_safetensors', False):
        _st_keys = _read_safetensors_tensor_keys(checkpoint_info.filename) or set()
        if "v_pred" in _st_keys:
            _unet_cfg   = getattr(pipe.unet,      "config", None)
            _sched_cfg  = getattr(getattr(pipe, "scheduler", None), "config", None)
            if _unet_cfg  is not None:
                _unet_cfg.prediction_type  = "v_prediction"
            if _sched_cfg is not None:
                _sched_cfg.prediction_type = "v_prediction"
            print("[diffusers path hijack] V-prediction detected from 'v_pred' key — "
                  "patched unet.config and scheduler.config")
            if "ztsnr" in _st_keys:
                if _sched_cfg is not None:
                    _sched_cfg.rescale_betas_zero_snr = True
                print("[diffusers path hijack] ZTSNR detected from 'ztsnr' key — "
                      "patched scheduler.config.rescale_betas_zero_snr=True")

    from modules.shared_cmd_options import cmd_opts
    _auto_offload = getattr(cmd_opts, 'forge_diffusers_auto_offload', False)
    _seq_offload  = getattr(cmd_opts, 'forge_diffusers_sequential_offload', False)

    if _auto_offload:
        # Move everything except UNet to device; auto-offload handles UNet block streaming.
        for attr in ("text_encoder", "text_encoder_2", "vae", "image_encoder"):
            sub = getattr(pipe, attr, None)
            if sub is not None:
                sub.to(device)
        from diff_pipeline.pipeline import apply_auto_offload_to_unet
        apply_auto_offload_to_unet(pipe.unet, device)
    elif _seq_offload:
        from accelerate import cpu_offload
        pipe.to(device)
        for child in pipe.unet.children():
            cpu_offload(child, execution_device=device, offload_buffers=True)
    else:
        pipe.to(device)

    # --- VAE decode precision upcast ----------------------------------------
    # The SDXL VAE decoder produces NaN in fp16 when intermediate activations
    # overflow the fp16 max (~65504).  We fix this once at load time so that
    # decode_first_stage() never needs to mutate model state at runtime — which
    # would break torch.compile / inductor CUDA-graph capture.
    #
    # Dtype selection (Tensor Core priority):
    #   bf16  — same exponent range as fp32, no overflow risk, natively
    #            accelerated by Tensor Cores on Ampere+ (preferred).
    #   fp32  — full range, uses tf32 on Ampere+ (also hits Tensor Cores).
    #            Fallback for devices without bf16 support or on CPU.
    if dtype in (torch.float16,):  # only upcast if we loaded in a reduced dtype
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            vae_decode_dtype = torch.bfloat16
        else:
            vae_decode_dtype = torch.float32
        pipe.vae.to(dtype=vae_decode_dtype)
        print(f"[diffusers path hijack] VAE upcasted to {vae_decode_dtype} "
              f"to avoid fp16 overflow during decode.")

    # --- Textual inversion embeddings ----------------------------------------
    # Load all .safetensors embeddings from the webui embeddings directory so
    # trigger words work in prompts without any extra user steps.
    try:
        from modules.shared_cmd_options import cmd_opts as _cmd_opts
        _emb_dir = getattr(_cmd_opts, 'embeddings_dir', None)
        if _emb_dir:
            from diff_pipeline.adapter import load_textual_inversion_embeddings
            load_textual_inversion_embeddings(pipe, _emb_dir)
    except Exception as _e:
        print(f"[diffusers path hijack] Could not load textual inversion embeddings: {_e}")

    return DiffusersModelAdapter(pipe, checkpoint_info, model_type="sdxl")


def dummy_sd1_hijack(checkpoint_info) -> Any:
    """Stub loader for SD 1.x checkpoints.

    Register with:
        from diff_pipeline.load_model import register_path_hijack
        from diff_pipeline.load_model import is_sd1_checkpoint, dummy_sd1_hijack
        register_path_hijack(is_sd1_checkpoint, dummy_sd1_hijack)

    Replace this body with real loading logic, e.g.:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_single_file(checkpoint_info.filename, ...)
        return pipe
    """
    raise NotImplementedError(
        "dummy_sd1_hijack: stub called. Replace this body with real diffusers "
        "loading logic for SD 1.x. checkpoint path: " + checkpoint_info.filename
    )


def dummy_sd3_hijack(checkpoint_info) -> Any:
    """Stub loader for SD3 checkpoints.

    Register with:
        from diff_pipeline.load_model import register_path_hijack
        from diff_pipeline.load_model import is_sd3_checkpoint, dummy_sd3_hijack
        register_path_hijack(is_sd3_checkpoint, dummy_sd3_hijack)

    Replace this body with real loading logic, e.g.:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_single_file(checkpoint_info.filename, ...)
        return pipe
    """
    raise NotImplementedError(
        "dummy_sd3_hijack: stub called. Replace this body with real diffusers "
        "loading logic for SD3. checkpoint path: " + checkpoint_info.filename
    )


# ---------------------------------------------------------------------------
# Legacy post-load path: --forge-diffusers-pipeline CLI flag
# (called from forge_loader.py after normal ldm loading, unchanged)
# ---------------------------------------------------------------------------

def _legacy_sdxl_fallback(sd_model, forge_objects) -> None:
    """Original forge SDXL diffusers-pipeline behaviour via --forge-diffusers-pipeline.

    Fires after normal ldm loading when sd_model.is_sdxl and the CLI flag are
    both set.  Not related to the path hijack system above.
    """
    from modules.shared import cmd_opts
    sd_model.diff_pipeline = None

    if not getattr(sd_model, 'is_sdxl', False):
        return
    if not getattr(cmd_opts, 'forge_diffusers_pipeline', False):
        return

    from diff_pipeline.pipeline import DiffPipeline
    from ldm_patched.modules.patcher_extension import WrappersMP
    from ldm_patched.modules.model_management import get_torch_device

    sd_model.diff_pipeline = DiffPipeline(forge_objects.unet, sd_model)
    _dp = sd_model.diff_pipeline

    # Cast HF UNet to the hardware-preferred dtype (bf16 > fp32 > fp16) at
    # load time so no runtime upcast is needed during inference.
    _infer_device = get_torch_device()
    _preferred = preferred_unet_dtype(_infer_device)
    _current = next(_dp._hf_unet.parameters()).dtype
    if _current != _preferred:
        _dp._hf_unet = _dp._hf_unet.to(dtype=_preferred)
        log.info(
            "DiffPipeline (legacy path): HF UNet recast %s → %s on %s "
            "(ldm loaded in a different dtype than the diffusers-preferred one — "
            "image output may differ from prior runs)",
            _current, _preferred, _infer_device,
        )
    else:
        log.info(
            "DiffPipeline (legacy path): HF UNet dtype %s on %s — no recast needed",
            _current, _infer_device,
        )

    def _diff_apply_model_wrapper(
        executor, x, t,
        c_concat=None, c_crossattn=None,
        control=None, transformer_options=None, **kwargs
    ):
        if transformer_options is None:
            transformer_options = {}
        return _dp.apply_model(
            x, t,
            c_concat=c_concat, c_crossattn=c_crossattn,
            control=control, transformer_options=transformer_options,
            **kwargs
        )

    forge_objects.unet.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL, "forge_diffusers", _diff_apply_model_wrapper
    )


def maybe_apply_diffusers_hijack(sd_model, forge_objects) -> None:
    """Entry point called from forge_loader.py for the legacy CLI path."""
    _legacy_sdxl_fallback(sd_model, forge_objects)
