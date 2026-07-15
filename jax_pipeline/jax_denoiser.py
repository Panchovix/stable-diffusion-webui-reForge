"""
jax_pipeline/jax_denoiser.py — JAX-native CFG denoiser for SDXL.

Mirrors mlx_pipeline/mlx_denoiser.py exactly — same conditioning
extraction, same batched-CFG data flow, same float32-for-CFG-math
rationale — but wraps jax_pipeline's pure-functional UNet
(``forward_fn(params, x, t, encoder_hidden_states, added_cond_kwargs)``,
see jax_pipeline.unet.unet_forward) instead of a stateful mlx.nn.Module,
so construction takes ``(forward_fn, params)`` rather than a single
callable module.

Pre-converts conditioning tensors (cross-attention, pooled CLIP-G) to JAX
*once* at construction time. Each denoising call stays entirely in JAX:

    Conditioning pre-built once
         v
    __call__(x, sigma)           <- all jnp.ndarray
         +-- EPS preconditioning  (x / sqrt(sigma^2+1))
         +-- sigma -> timestep    (nearest-neighbour in log-sigma table)
         +-- batch cond || uncond (single UNet forward, batch=2B)
         +-- CFG                  (uncond + scale*(cond-uncond), float32)
         +-- EPS postconditioning (x - out*sigma)
         v
    denoised: jnp.ndarray         <- no torch round-trip

Conditioning extraction
------------------------
Reads ``model_conds`` from the ldm cond-list format that
KDiffusionSampler stores in ``sampler_extra_args`` — this extraction
logic is framework-agnostic (pure torch-side dict/tensor plumbing); only
the final array conversion differs from mlx_pipeline's:

    cond_list[0]['model_conds']['crossattn'].cond     -> [1, S, 2048] torch
    cond_list[0]['model_conds']['pooled_output'].cond -> [1, 1280+] torch
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    import jax.numpy as jnp

log = logging.getLogger(__name__)


# -- conditioning extraction (framework-agnostic torch-side logic) --------------

def _get_raw_cond_dict(cond_obj: Any) -> Optional[Dict]:
    """Extract the raw conditioning dict from whatever format
    KDiffusionSampler stores in ``sampler_extra_args``.

    Two formats are encountered at the sampler level:

    * ``MulticondLearnedConditioning`` (cond):
        ``.batch[0][0].schedules[0].cond``
        -> dict ``{'crossattn': Tensor[1,S,2048], 'pooled_output': Tensor[1,1280]}``

    * ``list[list[ScheduledPromptConditioning]]`` (uncond):
        ``[0][0].cond``
        -> same dict format

    Both return a plain-tensor dict — NOT the ``model_conds`` / ``CONDRegular``
    wrapping used deeper inside ``sampling_function``. Identical to
    ``mlx_pipeline.mlx_denoiser._get_raw_cond_dict`` — this step never
    touches MLX or JAX, only torch/webui plumbing.
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


def _extract_cond_jax(
    cond_obj: Any,
    B: int,
    latent_hw: Tuple[int, int],
) -> Tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
    """Extract encoder_hidden_states [B, S, 2048], text_embeds [B, 1280],
    and time_ids [B, 6] as JAX bfloat16 / float32 arrays from the raw
    conditioning object stored in ``sampler_extra_args``.

    Parameters
    ----------
    cond_obj   : ``MulticondLearnedConditioning`` or ``list[list[SPC]]``
    B          : batch size (number of samples)
    latent_hw  : (H, W) of the *latent* — used to build time_ids
    """
    import jax.numpy as jnp

    enc_hs_t = text_embeds_t = None

    raw = _get_raw_cond_dict(cond_obj)
    if raw is not None:
        ca = raw.get("crossattn", None)      # Tensor [1, S, 2048]
        po = raw.get("pooled_output", None)  # Tensor [1, >=1280]
        if ca is not None:
            enc_hs_t = ca.detach().cpu().float()
        if po is not None:
            text_embeds_t = po.detach().cpu().float()[:, :1280]

    enc_hs = (
        jnp.asarray(enc_hs_t.expand(B, -1, -1).numpy(), dtype=jnp.bfloat16)
        if enc_hs_t is not None
        else jnp.zeros((B, 1, 2048), dtype=jnp.bfloat16)
    )
    text_embeds = (
        jnp.asarray(text_embeds_t.expand(B, -1).numpy(), dtype=jnp.bfloat16)
        if text_embeds_t is not None
        else jnp.zeros((B, 1280), dtype=jnp.bfloat16)
    )
    h_px = latent_hw[0] * 8
    w_px = latent_hw[1] * 8
    time_ids = jnp.asarray(
        [[float(h_px), float(w_px), 0.0, 0.0, float(h_px), float(w_px)]] * B,
        dtype=jnp.float32,
    )

    return enc_hs, text_embeds, time_ids


# -- JAX CFG denoiser ----------------------------------------------------------

class JAXCFGDenoiser:
    """JAX-native CFG denoiser for SDXL.

    Conditioning (encoder hidden states + pooled CLIP-G) is pre-converted
    to JAX at construction time. The denoising call runs entirely in JAX
    — no torch tensors are created inside ``__call__``.

    Parameters
    ----------
    forward_fn     : jax.jit-compiled ``unet_forward``-shaped callable,
                      ``forward_fn(params, x, t, encoder_hidden_states,
                      added_cond_kwargs) -> jnp.ndarray`` (see
                      ``jax_pipeline.host_offload.make_forward`` /
                      ``JAXSDXLPipeline._forward``)
    params         : the UNet's current params pytree (device-resident by
                      the time this is constructed — the sampler-loop hook
                      calls ``phase_manager.activate("unet")`` first)
    model_sampling : ldm model_sampling object for sigma <-> timestep
    extra_args     : k-diffusion sampler extra_args dict
                      (keys: ``cond``, ``uncond``, ``cond_scale``, ...)
    latent_shape   : ``(B, 4, H, W)`` of the initial noisy latent
    """

    def __init__(
        self,
        forward_fn,
        params: Dict[str, "jnp.ndarray"],
        model_sampling: Any,
        extra_args: Dict,
        latent_shape: Tuple[int, ...],
    ) -> None:
        import jax.numpy as jnp

        self._forward = forward_fn
        self._params = params
        self._ms = model_sampling
        self.cond_scale = float(extra_args.get("cond_scale", 7.5))
        B = latent_shape[0]
        latent_hw = (latent_shape[2], latent_shape[3])  # (H, W)

        cond_list = extra_args.get("cond", [])
        uncond_list = extra_args.get("uncond", [])

        enc_c, te_c, ti = _extract_cond_jax(cond_list, B, latent_hw)
        enc_u, te_u, _ = _extract_cond_jax(uncond_list, B, latent_hw)

        # Build batched conditioning: cond first, uncond second -> [2B, ...]
        self._enc_b = jnp.concatenate([enc_c, enc_u], axis=0)  # [2B, S, 2048]
        self._te_b = jnp.concatenate([te_c, te_u], axis=0)     # [2B, 1280]
        self._ti_b = jnp.concatenate([ti, ti], axis=0)         # [2B, 6]

        # Cache log-sigma table for fast timestep lookup (avoids torch round-trip)
        log_sigmas_np = model_sampling.log_sigmas.detach().cpu().float().numpy()
        self._log_sigmas = jnp.asarray(log_sigmas_np)  # [N]

        log.debug(
            "[JAX denoiser] Conditioning pre-built: enc=%s te=%s cond_scale=%.1f",
            self._enc_b.shape, self._te_b.shape, self.cond_scale,
        )

    # -- sigma ops (all in JAX) ------------------------------------------------

    def _timestep(self, sigma: "jnp.ndarray") -> "jnp.ndarray":
        """sigma [B] float32 -> nearest discrete timestep [B] float32."""
        import jax.numpy as jnp
        log_s = jnp.log(sigma)                                      # [B]
        diff = jnp.abs(log_s[:, None] - self._log_sigmas[None, :])  # [B, N]
        return jnp.argmin(diff, axis=1).astype(jnp.float32)          # [B]

    @staticmethod
    def _calc_input(sigma: "jnp.ndarray", x: "jnp.ndarray") -> "jnp.ndarray":
        """EPS preconditioning: xc = x / sqrt(sigma^2+1)."""
        import jax.numpy as jnp
        s = sigma.reshape((-1, 1, 1, 1)).astype(jnp.float32)
        return (x.astype(jnp.float32) / jnp.sqrt(s * s + 1.0)).astype(jnp.bfloat16)

    @staticmethod
    def _calc_denoised(sigma: "jnp.ndarray", model_out: "jnp.ndarray", x: "jnp.ndarray") -> "jnp.ndarray":
        """EPS postconditioning: denoised = x - model_out * sigma.

        Returns float32 to match the reference (model_base.py:
        ``model_output = diffusion_model(...).float()``). The sampler
        loop and ODE/SDE arithmetic then operate in full float32 precision.
        """
        import jax.numpy as jnp
        s = sigma.reshape((-1, 1, 1, 1)).astype(jnp.float32)
        return x.astype(jnp.float32) - model_out.astype(jnp.float32) * s  # float32

    # -- forward ---------------------------------------------------------------

    def __call__(
        self,
        x: "jnp.ndarray",
        sigma: "jnp.ndarray",
        **_kwargs: Any,  # absorb extra_args kwargs forwarded by sampler
    ) -> "jnp.ndarray":
        """
        x     : [B, 4, H, W]  bfloat16 (or float32 — cast happens in _calc_input)
        sigma : [B]           float32
        -> denoised [B, 4, H, W] float32
        """
        import jax.numpy as jnp

        B = x.shape[0]

        # Preconditioning (stays in JAX)
        xc = self._calc_input(sigma, x)  # [B, 4, H, W]
        ts = self._timestep(sigma)       # [B]

        # Tile for cond || uncond (2B)
        xc_b = jnp.concatenate([xc, xc], axis=0)  # [2B, 4, H, W]
        ts_b = jnp.concatenate([ts, ts], axis=0)  # [2B]

        added = {
            "text_embeds": self._te_b,  # [2B, 1280]
            "time_ids": self._ti_b,     # [2B, 6]
        }

        # Single batched UNet forward — no redundant torch<->JAX conversions
        out = self._forward(self._params, xc_b, ts_b, self._enc_b, added)  # [2B, 4, H, W]

        # CFG: uncond + scale*(cond - uncond)
        #
        # MUST be computed in float32 — see mlx_denoiser.py's identical
        # note: bfloat16's 7 mantissa bits quantise cond-uncond differences
        # catastrophically, producing isolated bright-spot artifacts that
        # are most visible at low CFG scales.
        out_c = out[:B].astype(jnp.float32)  # cond   [B, 4, H, W] f32
        out_u = out[B:].astype(jnp.float32)  # uncond [B, 4, H, W] f32
        cfg_out = out_u + self.cond_scale * (out_c - out_u)  # f32

        # Postconditioning (stays in JAX, _calc_denoised handles f32 input)
        return self._calc_denoised(sigma, cfg_out, x)
