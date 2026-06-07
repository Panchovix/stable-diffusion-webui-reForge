"""nodes_sure_wav_ag.py — SURE-AGWAV: Wavelet + Attention-Guided SURE node.

Patches any model to apply per-subband wavelet SURE correction weighted by
attention entropy after every CFG denoising step.  Compatible with all samplers.

Algorithm per step
------------------
1. Run model(x0_hat, σ) with attn1 entropy hooks → get sure_x0, capture U.
2. Residual in pixel space:  r = x0_hat − sure_x0
3. Wavelet decompose r and x0_hat into subbands via wavedec2.
4. For each subband k at scale l:
     Ũ_k = bilinear_resize(U, subband_size_l)    ← entropy at this scale
     W_k  = 1 + attn_weight · Ũ_k               ← spatial weight
     corrected_k = x0_k − α · approx_coeff · W_k · r_k
5. Optional FFT low-pass scale on approximation subband gradient (FreeU-style).
6. Reconstruct via waverec2.

Cost: one extra UNet forward per step (same as SURE-AG).
Requires: ptwt + pywt  (pip install pytorch-wavelets pywavelets)
"""

import logging

import torch
import ldm_patched.modules.samplers as _samplers

from ldm_patched.k_diffusion.sure_attention import (
    build_capture_model_options,
    _aggregate_entropy_map,
)
from ldm_patched.k_diffusion.sure_wav_ag import (
    _project_entropy_to_subband,
    _fft_lowpass_scale,
)

_logger = logging.getLogger("nodes_sure_wav_ag")


class SureWaveletAttentionGuidance:
    """SURE-AGWAV guidance node.

    Combines location-aware attention entropy with frequency-aware wavelet
    correction.  Degrades gracefully:
      - attn_weight=0        → plain wavelet SURE (no attention weighting)
      - pywt unavailable     → falls back to pixel-space SURE-AG
      - fft_scale=1.0        → no FFT modulation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "alpha": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 0.49, "step": 0.001,
                    "tooltip": "SURE step size. Auto-clamped to 1/(2*(1+attn_weight)).",
                }),
                "attn_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Entropy amplification per subband. 0 = wavelet SURE without attention weighting.",
                }),
                "attn_blocks": (["all", "middle", "mid+out", "input", "output"],),
                "approx_coeff": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Gradient scale (2.0 = SURE theory).",
                }),
                "wavelet": (["db4", "db2", "haar", "sym4", "sym8", "bior2.2"],
                            {"default": "db4"}),
                "wavelet_level": ("INT", {
                    "default": 3, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Wavelet decomposition depth.",
                }),
                "lp_frac": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Fraction of detail levels to correct (1.0=all, 0.0=approx only).",
                }),
                "fft_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "FFT low-pass scale on approx subband gradient. "
                               "1.0=disabled. <1 dampens high-freq noise; >1 boosts structure (FreeU-style).",
                }),
                "fft_threshold": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Radius (in latent pixels) of the FFT low-frequency window.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "guidance"

    def patch(
        self, model, alpha, attn_weight, attn_blocks, approx_coeff,
        wavelet, wavelet_level, lp_frac, fft_scale, fft_threshold,
    ):
        m = model.clone()

        alpha         = float(alpha)
        attn_weight   = float(attn_weight)
        approx_coeff  = float(approx_coeff)
        fft_scale     = float(fft_scale)
        fft_threshold = int(fft_threshold)

        # Lean §3: descent bound α < 1/(2(1+w))
        alpha_max = 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4
        alpha_eff = min(alpha, alpha_max)
        if alpha > alpha_max:
            _logger.warning(
                "[sure_wav_ag] alpha=%.4f exceeds safe bound %.4f for attn_weight=%.2f; clamped",
                alpha, alpha_max, attn_weight,
            )

        def post_cfg_function(args):
            x0_hat        = args["denoised"]
            sigma         = args["sigma"]
            model_inner   = args["model"]
            model_options = args["model_options"]
            cond          = args["cond"]

            if min(x0_hat.shape[2:]) <= 4:
                return x0_hat

            # ── Extra forward with attention capture ─────────────────────────
            attn_store = []
            capture_opts = build_capture_model_options(
                model_options, attn_store, attn_blocks
            )

            (sure_x0,) = _samplers.calc_cond_batch(
                model_inner, [cond], x0_hat, sigma, capture_opts
            )

            # ── Aggregate entropy map ─────────────────────────────────────────
            U = _aggregate_entropy_map(
                attn_store, x0_hat.shape, x0_hat.device, x0_hat.dtype
            )

            # ── Wavelet-space correction with per-subband entropy weighting ───
            try:
                import ptwt
                import pywt as _pywt
                _wav = _pywt.Wavelet(wavelet)

                residual = (x0_hat - sure_x0).detach()
                import math

                res_coeffs = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode="reflect")
                x0_coeffs  = ptwt.wavedec2(x0_hat.detach(), _wav, level=wavelet_level, mode="reflect")

                n_correct_detail = max(0, math.floor(lp_frac * wavelet_level))
                corrected_coeffs = []

                # Approximation subband
                r_a = res_coeffs[0]
                x_a = x0_coeffs[0]
                if U is not None and attn_weight > 0.0:
                    U_a = _project_entropy_to_subband(U, (r_a.shape[2], r_a.shape[3]))
                    W_a = 1.0 + attn_weight * U_a
                else:
                    W_a = 1.0
                grad_a = approx_coeff * r_a * W_a
                if abs(fft_scale - 1.0) > 1e-6 and r_a.shape[2] >= 2:
                    grad_a = _fft_lowpass_scale(grad_a, fft_threshold, fft_scale)
                corrected_coeffs.append(x_a - alpha_eff * grad_a)

                # Detail subbands
                for lvl_idx in range(1, len(res_coeffs)):
                    cH_r, cV_r, cD_r = res_coeffs[lvl_idx]
                    cH_x, cV_x, cD_x = x0_coeffs[lvl_idx]
                    if lvl_idx > n_correct_detail:
                        corrected_coeffs.append((cH_x, cV_x, cD_x))
                        continue
                    if U is not None and attn_weight > 0.0:
                        U_sb = _project_entropy_to_subband(
                            U, (cH_r.shape[2], cH_r.shape[3])
                        )
                        W_sb = 1.0 + attn_weight * U_sb
                    else:
                        W_sb = 1.0
                    step = alpha_eff * approx_coeff * W_sb
                    corrected_coeffs.append((
                        cH_x - step * cH_r,
                        cV_x - step * cV_r,
                        cD_x - step * cD_r,
                    ))

                corrected = ptwt.waverec2(corrected_coeffs, _wav).detach()
                corrected = corrected[..., :x0_hat.shape[2], :x0_hat.shape[3]]

            except ImportError:
                # Graceful fallback: pixel-space SURE-AG
                residual = x0_hat - sure_x0
                if U is not None and attn_weight > 0.0:
                    grad = approx_coeff * residual * (1.0 + attn_weight * U)
                else:
                    grad = approx_coeff * residual
                corrected = x0_hat - alpha_eff * grad

            _logger.debug(
                "[sure_wav_ag] sigma=%.4f  alpha=%.4f  entropy_rms=%.4f  "
                "attn_layers=%d  wavelet=%s  level=%d  fft_scale=%.2f",
                float(sigma.max()), alpha_eff,
                float(U.pow(2).mean().sqrt()) if U is not None else 0.0,
                len(attn_store), wavelet, wavelet_level, fft_scale,
            )

            return corrected

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SureWaveletAttentionGuidance": SureWaveletAttentionGuidance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SureWaveletAttentionGuidance": "SURE Wavelet Attention Guidance",
}
