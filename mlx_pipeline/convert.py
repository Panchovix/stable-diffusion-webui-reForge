"""
mlx_pipeline/convert.py — Weight conversion from ldm / HF UNet format to MLX.

Two-stage pipeline
------------------
1. ``ldm_sd_to_hf(ldm_state_dict, ldm_unet_cfg)``
   Applies the existing ``unet_to_diffusers()`` key mapping to produce an
   {hf_key: torch_tensor} dict — the same mapping used by DiffPipeline.

2. ``hf_sd_to_mlx(hf_sd, model)``
   Iterates over the HF state dict and converts each tensor to an MLX
   bfloat16 array, transposing Conv2d weights from PyTorch's OIHW to
   MLX's OHWI format.

The converted list of (key, mx.array) tuples is suitable for passing
directly to ``SDXLUNet.load_weights(pairs, strict=False)``.

Key format note
---------------
HF path → MLX module attribute path are identical (e.g.
``down_blocks.1.resnets.0.conv1.weight``), so no additional key
transformation is needed between stages 2 and the load call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch
    import mlx.core as mx
    from mlx_pipeline.unet import SDXLUNet

log = logging.getLogger(__name__)


# ── Stage 1: ldm state dict → HF key dict ────────────────────────────────────

def ldm_sd_to_hf(
    ldm_state_dict: Dict[str, "torch.Tensor"],
    ldm_unet_cfg: dict,
) -> Dict[str, "torch.Tensor"]:
    """Convert an ldm UNet state dict to HF-named tensors.

    Parameters
    ----------
    ldm_state_dict : dict from ``ldm_model.diffusion_model.state_dict()``
    ldm_unet_cfg   : merged unet_config dict (model-embedded + SDXL defaults)

    Returns
    -------
    {hf_key: torch.Tensor} mapping — keys match HF UNet2DConditionModel.
    Tensors are still on their original device / dtype (not yet converted).
    """
    from diff_pipeline.pipeline import _SDXL_LDM_UNET_CONFIG
    from ldm_patched.modules.utils import unet_to_diffusers

    # Merge provided config over the base SDXL defaults
    merged = dict(_SDXL_LDM_UNET_CONFIG)
    merged.update(ldm_unet_cfg)

    key_map = unet_to_diffusers(merged)  # {hf_key: ldm_rel_key}

    hf_sd: Dict[str, "torch.Tensor"] = {}
    missing_ldm: List[str] = []

    for hf_key, ldm_rel_key in key_map.items():
        if ldm_rel_key in ldm_state_dict:
            hf_sd[hf_key] = ldm_state_dict[ldm_rel_key]
        else:
            missing_ldm.append(hf_key)

    if missing_ldm:
        log.debug(
            "[MLX convert] %d HF keys had no matching ldm key (first 5: %s)",
            len(missing_ldm), missing_ldm[:5],
        )

    log.info(
        "[MLX convert] ldm→HF mapping: %d / %d HF keys resolved",
        len(hf_sd), len(key_map),
    )
    return hf_sd


# ── Stage 2: HF tensors → MLX bfloat16 arrays ────────────────────────────────

def _tensor_to_mlx_bf16(key: str, tensor: "torch.Tensor") -> "mx.array":
    """Convert a single PyTorch tensor to an MLX bfloat16 array.

    Conv2d weight tensors (identified by key ending in ".weight" on a Conv
    layer — heuristic: 4-D tensors whose key path contains "conv") are
    transposed from PyTorch's OIHW layout to MLX's OHWI layout.

    All other tensors (Linear weights, GroupNorm/LayerNorm gamma/beta, biases)
    are copied verbatim after casting.
    """
    import mlx.core as mx

    # Detach, CPU, float32 (numpy does not support bfloat16 directly)
    arr_np = tensor.detach().cpu().float().numpy()

    # Conv2d weight: [O, I, kH, kW] → [O, kH, kW, I]  (OIHW → OHWI)
    is_conv_weight = (
        arr_np.ndim == 4
        and key.endswith(".weight")
        and any(kw in key for kw in ("conv", "downsamplers", "upsamplers"))
    )
    if is_conv_weight:
        arr_np = np.transpose(arr_np, (0, 2, 3, 1))

    return mx.array(arr_np).astype(mx.bfloat16)


def hf_sd_to_mlx(
    hf_sd: Dict[str, "torch.Tensor"],
) -> List[Tuple[str, "mx.array"]]:
    """Convert HF-keyed state dict to a list of (key, mx.array_bf16) pairs.

    Suitable for passing to ``SDXLUNet.load_weights(pairs, strict=False)``.
    """
    pairs: List[Tuple[str, "mx.array"]] = []
    skipped = 0

    for key, tensor in hf_sd.items():
        try:
            pairs.append((key, _tensor_to_mlx_bf16(key, tensor)))
        except Exception as e:
            log.warning("[MLX convert] Skipping key %r: %s", key, e)
            skipped += 1

    log.info(
        "[MLX convert] Converted %d tensors to bfloat16 MLX arrays (%d skipped)",
        len(pairs), skipped,
    )
    return pairs


# ── Convenience: full pipeline (ldm model → MLX SDXLUNet) ────────────────────

def load_weights_from_ldm(
    mlx_unet: "SDXLUNet",
    ldm_model,
    report: bool = True,
) -> None:
    """One-shot: read ldm weights, convert to MLX bf16, load into mlx_unet.

    Parameters
    ----------
    mlx_unet  : freshly-constructed SDXLUNet instance (weights are placeholders)
    ldm_model : ldm_patched model object with .diffusion_model and .model_config
    report    : if True, print a one-line summary
    """
    import mlx.core as mx

    ldm_sd = ldm_model.diffusion_model.state_dict()
    ldm_unet_cfg = dict(getattr(ldm_model, "model_config", type("_", (), {"unet_config": {}})()).unet_config)

    # Stage 1: ldm → HF keys
    hf_sd = ldm_sd_to_hf(ldm_sd, ldm_unet_cfg)

    # Stage 2: HF tensors → MLX bfloat16
    pairs = hf_sd_to_mlx(hf_sd)

    # Load into module (strict=False allows missing up/down-sampler entries
    # that unet_to_diffusers() generates speculatively)
    mlx_unet.load_weights(pairs, strict=False)

    # Ensure ALL parameters are bfloat16 — load_weights only touches keys that
    # exist in `pairs`; any missed keys (e.g. optional bias terms not present
    # in the ldm checkpoint) remain at their default dtype (float32).
    # One sweep ensures uniform bfloat16 throughout and preserves MLX's
    # dtype-propagation guarantee (output dtype follows parameter dtype).
    import mlx.nn as _nn
    mlx_unet.apply(lambda x: x.astype(mx.bfloat16))

    # Force evaluation so Metal kernel compilation happens at load time
    # (avoids the first-step latency spike during sampling)
    mx.eval(mlx_unet.parameters())

    if report:
        # mlx_unet.parameters() returns a nested dict; flatten it first.
        import mlx.utils as mx_utils
        flat = mx_utils.tree_flatten(mlx_unet.parameters())  # [(path, mx.array), …]
        n_params = sum(v.size for _, v in flat if hasattr(v, "size"))
        print(
            f"[MLX Pipeline] Weights loaded: {len(pairs)} tensors → "
            f"{n_params / 1e9:.2f}B bfloat16 parameters on Metal"
        )
