"""
mlx_pipeline/mlx_denoiser.py — MLX-native CFG denoiser for SDXL.

Pre-converts conditioning tensors (cross-attention, pooled CLIP-G) to MLX
*once* at construction time.  Each denoising call stays entirely in MLX:

    Conditioning pre-built once
         ↓
    __call__(x, sigma)           ← all mx.array
         ├── EPS preconditioning  (MLX: x / √(σ²+1))
         ├── σ → timestep         (MLX: nearest-neighbour in log-σ table)
         ├── batch cond ‖ uncond  (single UNet forward, batch=2B)
         ├── CFG                  (MLX: uncond + scale*(cond-uncond))
         └── EPS postconditioning (MLX: x - out*σ)
         ↓
    denoised: mx.array            ← no torch round-trip

Conditioning extraction
-----------------------
Reads ``model_conds`` from the ldm cond-list format that
KDiffusionSampler stores in ``sampler_extra_args``:

    cond_list[0]['model_conds']['crossattn'].cond   → [1, S, 2048] torch
    cond_list[0]['model_conds']['pooled_output'].cond → [1, 1280+] torch
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    import mlx.core as mx
    from mlx_pipeline.unet import SDXLUNet

log = logging.getLogger(__name__)


# ── tensor helpers ─────────────────────────────────────────────────────────────

def _t2m(t: torch.Tensor, dtype=None) -> "mx.array":
    """torch → MLX (bfloat16 by default; float32 when dtype=mx.float32)."""
    import mlx.core as mx
    arr = t.detach().float().cpu().numpy()
    a = mx.array(arr)
    return a.astype(dtype if dtype is not None else mx.bfloat16)


def _m2t(a: "mx.array") -> torch.Tensor:
    """MLX → torch float32 CPU.

    np.array() triggers implicit MLX evaluation — no explicit eval needed.
    """
    import mlx.core as mx
    return torch.from_numpy(np.array(a.astype(mx.float32)).copy()).float()


# ── conditioning extraction ─────────────────────────────────────────────────────

def _get_raw_cond_dict(cond_obj: Any) -> Optional[Dict]:
    """
    Extract the raw conditioning dict from whatever format KDiffusionSampler
    stores in ``sampler_extra_args``.

    Two formats are encountered at the sampler level:

    * ``MulticondLearnedConditioning`` (cond):
        ``.batch[0][0].schedules[0].cond``
        → dict ``{'crossattn': Tensor[1,S,2048], 'pooled_output': Tensor[1,1280]}``

    * ``list[list[ScheduledPromptConditioning]]`` (uncond):
        ``[0][0].cond``
        → same dict format

    Both return a plain-tensor dict — NOT the ``model_conds`` / ``CONDRegular``
    wrapping used deeper inside ``sampling_function``.
    """
    try:
        from modules.prompt_parser import MulticondLearnedConditioning
        if isinstance(cond_obj, MulticondLearnedConditioning):
            raw = cond_obj.batch[0][0].schedules[0].cond
        elif isinstance(cond_obj, list) and cond_obj and cond_obj[0]:
            raw = cond_obj[0][0].cond
        else:
            return None
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _extract_cond_mx(
    cond_obj: Any,
    B: int,
    latent_hw: Tuple[int, int],
) -> Tuple["mx.array", "mx.array", "mx.array"]:
    """
    Extract encoder_hidden_states [B, S, 2048], text_embeds [B, 1280],
    and time_ids [B, 6] as MLX bfloat16 / float32 arrays from the raw
    conditioning object stored in ``sampler_extra_args``.

    Parameters
    ----------
    cond_obj   : ``MulticondLearnedConditioning`` or ``list[list[SPC]]``
    B          : batch size (number of samples)
    latent_hw  : (H, W) of the *latent* — used to build time_ids
    """
    import mlx.core as mx

    enc_hs_t = text_embeds_t = None

    raw = _get_raw_cond_dict(cond_obj)
    if raw is not None:
        ca = raw.get("crossattn",     None)   # Tensor [1, S, 2048]
        po = raw.get("pooled_output", None)   # Tensor [1, ≥1280]
        if ca is not None:
            enc_hs_t     = ca.detach().cpu().float()
        if po is not None:
            text_embeds_t = po.detach().cpu().float()[:, :1280]

    enc_hs = (
        _t2m(enc_hs_t.expand(B, -1, -1))
        if enc_hs_t is not None
        else mx.zeros((B, 1, 2048), dtype=mx.bfloat16)
    )
    text_embeds = (
        _t2m(text_embeds_t.expand(B, -1))
        if text_embeds_t is not None
        else mx.zeros((B, 1280), dtype=mx.bfloat16)
    )
    _h = latent_hw[0] * 8
    _w = latent_hw[1] * 8
    time_ids = mx.array(
        [[float(_h), float(_w), 0.0, 0.0, float(_h), float(_w)]] * B,
        dtype=mx.float32,
    )

    return enc_hs, text_embeds, time_ids


# ── MLX CFG denoiser ────────────────────────────────────────────────────────────

class MLXCFGDenoiser:
    """
    MLX-native CFG denoiser for SDXL.

    Conditioning (encoder hidden states + pooled CLIP-G) is pre-converted
    to MLX at construction time.  The denoising call runs entirely in MLX
    — no torch tensors are created inside ``__call__``.

    Parameters
    ----------
    mlx_unet      : SDXLUNet (loaded bfloat16 on Metal)
    model_sampling: ldm model_sampling object for sigma ↔ timestep
    extra_args    : k-diffusion sampler extra_args dict
                    (keys: ``cond``, ``uncond``, ``cond_scale``, …)
    latent_shape  : ``(B, 4, H, W)`` of the initial noisy latent
    """

    def __init__(
        self,
        mlx_unet: "SDXLUNet",
        model_sampling: Any,
        extra_args: Dict,
        latent_shape: Tuple[int, ...],
    ) -> None:
        import mlx.core as mx

        self._unet       = mlx_unet
        self._ms         = model_sampling
        self.cond_scale  = float(extra_args.get("cond_scale", 7.5))
        B                = latent_shape[0]
        latent_hw        = (latent_shape[2], latent_shape[3])   # (H, W)

        cond_list   = extra_args.get("cond",   [])
        uncond_list = extra_args.get("uncond", [])

        enc_c, te_c, ti = _extract_cond_mx(cond_list,   B, latent_hw)
        enc_u, te_u, _  = _extract_cond_mx(uncond_list, B, latent_hw)

        # Build batched conditioning: cond first, uncond second  → [2B, …]
        self._enc_b = mx.concatenate([enc_c, enc_u], axis=0)   # [2B, S, 2048]
        self._te_b  = mx.concatenate([te_c,  te_u],  axis=0)   # [2B, 1280]
        self._ti_b  = mx.concatenate([ti,    ti],    axis=0)   # [2B, 6]

        # Cache log-sigma table for fast timestep lookup (avoids torch round-trip)
        log_sigmas_np       = model_sampling.log_sigmas.detach().cpu().float().numpy()
        self._log_sigmas    = mx.array(log_sigmas_np)           # [N]

        mx.eval(self._enc_b, self._te_b, self._ti_b, self._log_sigmas)
        log.debug(
            "[MLX denoiser] Conditioning pre-built: enc=%s te=%s cond_scale=%.1f",
            self._enc_b.shape, self._te_b.shape, self.cond_scale,
        )

    # ── sigma ops (all in MLX) ────────────────────────────────────────────────

    def _timestep(self, sigma: "mx.array") -> "mx.array":
        """sigma [B] float32 → nearest discrete timestep [B] float32."""
        import mlx.core as mx
        log_s = mx.log(sigma)                                          # [B]
        diff  = mx.abs(log_s[:, None] - self._log_sigmas[None, :])    # [B, N]
        return mx.argmin(diff, axis=1).astype(mx.float32)              # [B]

    @staticmethod
    def _calc_input(sigma: "mx.array", x: "mx.array") -> "mx.array":
        """EPS preconditioning: xc = x / √(σ²+1)."""
        import mlx.core as mx
        s = sigma.reshape((-1, 1, 1, 1)).astype(mx.float32)
        return (x.astype(mx.float32) / mx.sqrt(s * s + 1.0)).astype(mx.bfloat16)

    @staticmethod
    def _calc_denoised(
        sigma: "mx.array", model_out: "mx.array", x: "mx.array"
    ) -> "mx.array":
        """EPS postconditioning: denoised = x - model_out * σ.

        Returns float32 to match the reference (model_base.py line 194:
        ``model_output = diffusion_model(...).float()``).  The sampler
        loop and ODE/SDE arithmetic then operate in full float32 precision.
        """
        import mlx.core as mx
        s = sigma.reshape((-1, 1, 1, 1)).astype(mx.float32)
        return x.astype(mx.float32) - model_out.astype(mx.float32) * s  # float32

    # ── forward ───────────────────────────────────────────────────────────────

    def __call__(
        self,
        x: "mx.array",
        sigma: "mx.array",
        **_kwargs: Any,       # absorb extra_args kwargs forwarded by sampler
    ) -> "mx.array":
        """
        x     : [B, 4, H, W]  bfloat16
        sigma : [B]            float32
        → denoised [B, 4, H, W] bfloat16
        """
        import mlx.core as mx

        B = x.shape[0]

        # Preconditioning (stays in MLX)
        xc = self._calc_input(sigma, x)           # [B, 4, H, W]
        ts = self._timestep(sigma)                # [B]

        # Tile for cond ‖ uncond  (2B)
        xc_b = mx.concatenate([xc, xc], axis=0)  # [2B, 4, H, W]
        ts_b = mx.concatenate([ts, ts], axis=0)  # [2B]

        added = {
            "text_embeds": self._te_b,            # [2B, 1280]
            "time_ids":    self._ti_b,            # [2B, 6]
        }

        # Single batched UNet forward — no redundant torch↔MLX conversions
        out = self._unet(xc_b, ts_b, self._enc_b, added)  # [2B, 4, H, W]  bfloat16

        # CFG: uncond + scale*(cond − uncond)
        #
        # MUST be computed in float32.  bfloat16 has only 7 mantissa bits
        # (~0.015 precision near value 2), so differences between cond and
        # uncond outputs are catastrophically quantised — small guidance
        # directions are rounded to zero, producing isolated bright spots
        # ("spots of lights") that become prominent at low CFG scales (≤5)
        # where the uncond term has a larger relative weight.
        out_c = out[:B].astype(mx.float32)   # cond   [B, 4, H, W] f32
        out_u = out[B:].astype(mx.float32)   # uncond [B, 4, H, W] f32
        cfg_out = out_u + self.cond_scale * (out_c - out_u)  # f32

        # Postconditioning (stays in MLX, _calc_denoised handles f32 input)
        return self._calc_denoised(sigma, cfg_out, x)
