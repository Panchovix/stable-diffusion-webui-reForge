"""
jax_pipeline/convert.py — Weight conversion from ldm / HF UNet format to JAX.

Two-stage pipeline
------------------
1. ``ldm_sd_to_hf(ldm_state_dict, ldm_unet_cfg)``
   Reuses the existing ``unet_to_diffusers()`` key mapping (the same one
   DiffPipeline and mlx_pipeline use) to produce an {hf_key: torch_tensor}
   dict.

2. ``hf_sd_to_jax(hf_sd, dtype)``
   Converts each tensor to a jax.numpy array, transposing Conv2d weights
   from PyTorch's OIHW layout to JAX's HWIO layout (the kernel shape
   expected by ``jax.lax.conv_general_dilated`` with
   dimension_numbers=('NHWC', 'HWIO', 'NHWC')).

The result is a flat ``{hf_key: jnp.ndarray}`` dict — a valid JAX pytree —
keyed by the exact HF dotted attribute path (e.g.
"down_blocks.1.resnets.0.conv1.weight"). ``jax_pipeline/unet.py`` looks up
keys directly by this path; there is no separate module tree to build.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

log = logging.getLogger(__name__)


# ── Stage 1: ldm state dict → HF key dict ────────────────────────────────────

def ldm_sd_to_hf(
    ldm_state_dict: Dict[str, "torch.Tensor"],
    ldm_unet_cfg: dict,
) -> Dict[str, "torch.Tensor"]:
    """Convert an ldm UNet state dict to HF-named tensors.

    Identical mapping logic to ``mlx_pipeline.convert.ldm_sd_to_hf`` and
    ``DiffPipeline`` — reuses ``ldm_patched.modules.utils.unet_to_diffusers``
    rather than hand-rolling key renaming.
    """
    from diff_pipeline.pipeline import _SDXL_LDM_UNET_CONFIG
    from ldm_patched.modules.utils import unet_to_diffusers

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
            "[JAX convert] %d HF keys had no matching ldm key (first 5: %s)",
            len(missing_ldm), missing_ldm[:5],
        )

    log.info(
        "[JAX convert] ldm→HF mapping: %d / %d HF keys resolved",
        len(hf_sd), len(key_map),
    )
    return hf_sd


# ── Stage 2: HF tensors → JAX arrays ─────────────────────────────────────────

def _is_conv_weight(key: str, ndim: int) -> bool:
    """Heuristic identifying a Conv2d weight tensor by key + rank.

    Same heuristic as mlx_pipeline.convert: 4-D tensor, key ends in
    ".weight", and the key path mentions "conv"/"downsamplers"/"upsamplers"
    (covers conv_in/conv1/conv2/conv_shortcut/downsamplers.0.conv/
    upsamplers.0.conv). Everything else (Linear weights, norm gamma/beta,
    biases) passes through unchanged.
    """
    return (
        ndim == 4
        and key.endswith(".weight")
        and any(kw in key for kw in ("conv", "downsamplers", "upsamplers"))
    )


def _tensor_to_jax(key: str, tensor: "torch.Tensor", dtype) -> "jnp.ndarray":
    """Convert a single PyTorch tensor to a JAX array.

    Conv2d weight tensors are transposed from PyTorch's OIHW layout to
    JAX's HWIO layout: [O,I,kH,kW] -> [kH,kW,I,O].
    """
    import jax.numpy as jnp

    arr_np = tensor.detach().cpu().float().numpy()

    if _is_conv_weight(key, arr_np.ndim):
        arr_np = np.transpose(arr_np, (2, 3, 1, 0))  # OIHW -> HWIO

    return jnp.asarray(arr_np, dtype=dtype)


def hf_sd_to_jax(
    hf_sd: Dict[str, "torch.Tensor"],
    dtype=None,
) -> Dict[str, "jnp.ndarray"]:
    """Convert an HF-keyed state dict to a flat {key: jnp.ndarray} dict.

    ``dtype`` defaults to ``jnp.bfloat16`` if not given.
    """
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.bfloat16

    params: Dict[str, "jnp.ndarray"] = {}
    skipped = 0

    for key, tensor in hf_sd.items():
        try:
            params[key] = _tensor_to_jax(key, tensor, dtype)
        except Exception as e:
            log.warning("[JAX convert] Skipping key %r: %s", key, e)
            skipped += 1

    log.info(
        "[JAX convert] Converted %d tensors to %s JAX arrays (%d skipped)",
        len(params), dtype, skipped,
    )
    return params


# ── Convenience: full pipeline (ldm model → JAX params dict) ─────────────────

def load_weights_from_ldm(
    ldm_model,
    dtype=None,
    report: bool = True,
) -> Dict[str, "jnp.ndarray"]:
    """One-shot: read ldm weights, convert to a flat JAX params dict.

    Parameters
    ----------
    ldm_model : ldm_patched model object with .diffusion_model and .model_config
    dtype     : target jnp dtype (default jnp.bfloat16)
    report    : if True, print a one-line summary

    Returns
    -------
    {hf_key: jnp.ndarray} — flat dict, a valid JAX pytree, ready to pass to
    ``jax_pipeline.unet.unet_forward(params, ...)``.
    """
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.bfloat16

    ldm_sd = ldm_model.diffusion_model.state_dict()
    ldm_unet_cfg = dict(getattr(ldm_model, "model_config", type("_", (), {"unet_config": {}})()).unet_config)

    hf_sd = ldm_sd_to_hf(ldm_sd, ldm_unet_cfg)
    params = hf_sd_to_jax(hf_sd, dtype=dtype)

    if report:
        n_params = sum(int(np.prod(v.shape)) for v in params.values())
        print(
            f"[JAX Pipeline] Weights loaded: {len(params)} tensors → "
            f"{n_params / 1e9:.2f}B {dtype} parameters"
        )

    return params
