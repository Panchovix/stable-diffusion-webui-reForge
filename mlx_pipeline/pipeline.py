"""
mlx_pipeline/pipeline.py — Torch ↔ MLX bridge for SDXL UNet inference.

MLXSDXLPipeline wraps the MLX UNet and presents the same
``apply_model(x, t, ...)`` interface as ``DiffPipeline`` so the forge
wrapper system and the rest of the webui are completely unaware of the
backend change.

Data-flow
---------
::

    apply_model(x, t, c_crossattn, y, ...)           ← forge sampler call
        │
        ├── sigma preconditioning          (model_sampling.calculate_input /
        │                                   model_sampling.timestep)
        ├── torch → MLX conversion         (detach, cpu, np, mx.bfloat16)
        │     ├── xc   [B,4,H,W]
        │     ├── ts   [B]
        │     ├── enc  [B,S,2048]
        │     └── added_cond_kwargs
        │
        ├── SDXLUNet.__call__()             (Metal bfloat16 forward pass)
        │
        ├── MLX → torch conversion         (np.array implicit eval, back to device)
        │
        └── sigma postconditioning         (model_sampling.calculate_denoised)

Conditioning resolution
-----------------------
Follows the same priority logic as DiffPipeline (observation #3 in memory):
  * text_embeds  : adm_text_embeds kwarg  →  y[:1280]  →  zeros fallback
  * time_ids     : adm_time_ids kwarg     →  derived from latent shape
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from mlx_pipeline.unet import SDXLUNet
    from modules_forge.unet_patcher import UnetPatcher

log = logging.getLogger(__name__)


# ── torch ↔ MLX conversion helpers ──────────────────────────────────────────

def _torch_to_mlx(t: torch.Tensor, dtype=None):
    """Convert a PyTorch tensor to an MLX bfloat16 array.

    Handles MPS, CUDA, and CPU tensors uniformly via numpy.
    bfloat16 numpy does not exist, so we cast to float32 first.
    """
    import mlx.core as mx
    arr = t.detach().float().cpu().numpy()
    a = mx.array(arr)
    if dtype is not None:
        a = a.astype(dtype)
    else:
        a = a.astype(mx.bfloat16)
    return a


def _mlx_to_torch(
    a,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an MLX array to a PyTorch tensor on *device* with *dtype*.

    np.array() implicitly evaluates the MLX compute graph — Metal handles
    scheduling; no explicit mx.eval() needed here.
    """
    import mlx.core as mx
    # bfloat16 MLX arrays are not numpy-compatible directly; cast to fp32 first.
    arr = np.array(a.astype(mx.float32))
    return torch.from_numpy(arr.copy()).to(device=device, dtype=dtype)


# ── main pipeline class ──────────────────────────────────────────────────────

class MLXSDXLPipeline:
    """SDXL UNet forward-pass engine backed by Apple MLX on Metal.

    Constructed once at model load time; ``apply_model()`` is called every
    sampling step by the forge wrapper registered in ``mlx_pipeline.__init__``.
    """

    def __init__(self, unet_patcher: "UnetPatcher", sd_model: Any):
        self.unet_patcher = unet_patcher
        self.sd_model = sd_model

        # Sigma preconditioning from the loaded model's model_sampling object.
        # Correct path: unet_patcher.model.model_sampling
        # (same as diff_pipeline/pipeline.py:637 and processing.py:1075)
        self.model_sampling = unet_patcher.model.model_sampling

        # Build and populate the MLX UNet
        log.info("[MLX Pipeline] Constructing SDXL MLX UNet …")
        from mlx_pipeline.unet import SDXLUNet
        from mlx_pipeline.convert import load_weights_from_ldm

        self._mlx_unet = SDXLUNet()
        load_weights_from_ldm(self._mlx_unet, unet_patcher.model, report=True)

        # LoRA / DoRA tracking: stores the ldm_patched patches_uuid (a UUID4
        # regenerated on every LoRA add/remove/strength change) that was in
        # effect when the MLX UNet weights were last loaded.
        # Initialized from the patcher NOW so the first generation does NOT
        # trigger a spurious reload when LoRA hasn't changed since model load.
        self._lora_sig: str = str(unet_patcher.patches_uuid)

        log.info("[MLX Pipeline] Ready — using bfloat16 on Metal")

    # ── forward pass ─────────────────────────────────────────────────────────

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[torch.Tensor] = None,
        c_crossattn: Optional[torch.Tensor] = None,
        control: Optional[Dict] = None,
        transformer_options: Optional[Dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Drop-in replacement for ``BaseModel.apply_model()``.

        Parameters match the ldm_patched calling convention; the return
        value is a *denoised* latent tensor in the same dtype/device as *x*.
        """
        import mlx.core as mx

        if transformer_options is None:
            transformer_options = {}

        # ── 0. LoRA / DoRA hot-reload (non-MLX-sampler fallback path) ────────
        # When a non-MLX sampler is used, apply_model IS called; the MLX
        # sampler hook skips this path.  ldm_patched's patches_uuid is a
        # UUID4 that is regenerated on every LoRA add/remove/strength change
        # and is far cheaper to compare than hashing the full patches dict.
        # patch_weight_to_device() has already merged LoRA into the PyTorch
        # diffusion_model weights before this wrapper is reached.
        try:
            from mlx_pipeline.lora import reload_mlx_unet_weights
            _cur_uuid = str(self.unet_patcher.patches_uuid)
            if _cur_uuid != self._lora_sig:
                reload_mlx_unet_weights(self._mlx_unet, self.unet_patcher)
                self._lora_sig = _cur_uuid
        except Exception as _lora_exc:
            log.warning("[MLX LoRA] LoRA reload skipped: %s", _lora_exc)

        sigma = t  # alias (t carries sigma values from k-diffusion)
        device = x.device
        out_dtype = x.dtype  # preserve caller's dtype on output

        # ── 1. Sigma preconditioning (matches ldm BaseModel / DiffPipeline) ──
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        # Integer/float timestep for UNet positional encoding
        timestep = self.model_sampling.timestep(sigma).float()

        # ── 2. Resolve SDXL conditioning ─────────────────────────────────────
        # text_embeds: 1280-dim pooled CLIP-G output
        adm_text_embeds = kwargs.get("adm_text_embeds", None)
        adm_time_ids    = kwargs.get("adm_time_ids", None)
        y               = kwargs.get("y", None)

        if adm_text_embeds is not None:
            text_embeds = adm_text_embeds
        elif y is not None:
            text_embeds = y[:, :1280]
        else:
            log.warning(
                "[MLX Pipeline] No pooled conditioning (adm_text_embeds/y absent) "
                "— using zero fallback. Quality will be degraded."
            )
            text_embeds = torch.zeros(
                x.shape[0], 1280, device=device, dtype=torch.float32
            )

        # time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        if adm_time_ids is not None:
            time_ids = adm_time_ids
        else:
            _h, _w = x.shape[2] * 8, x.shape[3] * 8
            time_ids = torch.tensor(
                [[_h, _w, 0, 0, _h, _w]],
                device=device, dtype=torch.float32,
            ).expand(text_embeds.shape[0], -1)

        # ── 3. Convert to MLX bfloat16 ────────────────────────────────────────
        mx_sample    = _torch_to_mlx(xc)                       # [B,4,H,W]
        mx_timestep  = _torch_to_mlx(timestep, dtype=mx.float32)  # [B]
        mx_enc_hs    = _torch_to_mlx(c_crossattn) if c_crossattn is not None \
                       else mx.zeros((x.shape[0], 1, 2048), dtype=mx.bfloat16)
        mx_text_emb  = _torch_to_mlx(text_embeds)              # [B, 1280]
        mx_time_ids  = _torch_to_mlx(time_ids, dtype=mx.float32)  # [B, 6]

        added_cond_kwargs = {
            "text_embeds": mx_text_emb,
            "time_ids":    mx_time_ids,
        }

        # ── 4. MLX forward pass ───────────────────────────────────────────────
        mx_output = self._mlx_unet(
            mx_sample,
            mx_timestep,
            mx_enc_hs,
            added_cond_kwargs,
        )  # [B, 4, H, W]  bfloat16

        # ── 5. Convert back to torch ──────────────────────────────────────────
        model_output = _mlx_to_torch(mx_output, device=device, dtype=torch.float32)

        # ── 6. Sigma postconditioning ─────────────────────────────────────────
        denoised = self.model_sampling.calculate_denoised(sigma, model_output, x)

        return denoised.to(dtype=out_dtype)
