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
     g_k  = approx_coeff · W_k · r_k             ← per-subband gradient
5. Accumulate Parseval dot products: r_dot_g = Σ⟨r_k,g_k⟩, g_sq = Σ‖g_k‖²
6. Adaptive alpha via _wavag_step_alpha (fixed/sigma/analytical/bo).
7. Apply: x0_k_corrected = x0_k − α · g_k
8. Optional FFT low-pass scale on approximation subband gradient (FreeU-style).
9. Reconstruct via waverec2.

Cost: one extra UNet forward per step (same as SURE-AG).
Requires: ptwt + pywt  (pip install pytorch-wavelets pywavelets)
"""

import logging
import math

import torch
import ldm_patched.modules.samplers as _samplers

from ldm_patched.k_diffusion.sure_attention import (
    build_capture_model_options,
    _aggregate_entropy_map,
)
from ldm_patched.k_diffusion.sure_wav_ag import (
    _project_entropy_to_subband,
    _fft_lowpass_scale,
    _make_step_state,
    _wavag_step_alpha,
)

_logger = logging.getLogger("nodes_sure_wav_ag")

_ALPHA_MODES = ["sigma", "analytical", "bo", "fixed"]


class SureWaveletAttentionGuidance:
    """SURE-AGWAV guidance node.

    Combines location-aware attention entropy with frequency-aware wavelet
    correction.  Degrades gracefully:
      - attn_weight=0        → plain wavelet SURE (no attention weighting)
      - pywt unavailable     → falls back to pixel-space SURE-AG
      - fft_scale=1.0        → no FFT modulation
      - alpha_mode="fixed"   → static alpha (original behaviour)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "alpha": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 0.49, "step": 0.001,
                    "tooltip": "SURE step size. Hard ceiling: 1/(2*(1+attn_weight)).",
                }),
                "alpha_mode": (_ALPHA_MODES, {
                    "tooltip": (
                        "sigma: scale by 1/(1+min(σ,1)) — EDM/Karras aware (recommended). "
                        "analytical: α*=⟨r,g⟩/‖g‖² closed-form, free via Parseval. "
                        "bo: Bayesian optimisation with cross-step warm-starting (needs optuna). "
                        "fixed: use alpha as-is."
                    ),
                }),
                "attn_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Per-subband entropy amplification. 0 = wavelet SURE only.",
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
                }),
                "lp_frac": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Fraction of detail levels to correct (1.0=all, 0.0=approx only).",
                }),
                "fft_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "FFT low-pass scale on approx gradient. 1.0=off.",
                }),
                "fft_threshold": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "guidance"

    def patch(
        self, model, alpha, alpha_mode, attn_weight, attn_blocks, approx_coeff,
        wavelet, wavelet_level, lp_frac, fft_scale, fft_threshold,
    ):
        m = model.clone()

        alpha         = float(alpha)
        attn_weight   = float(attn_weight)
        approx_coeff  = float(approx_coeff)
        fft_scale     = float(fft_scale)
        fft_threshold = int(fft_threshold)
        wavelet_level = int(wavelet_level)

        # Lean §3: hard ceiling α < 1/(2(1+w))
        alpha_max = 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4
        if alpha > alpha_max:
            _logger.warning(
                "[sure_wav_ag] alpha=%.4f exceeds safe bound %.4f for attn_weight=%.2f; clamped",
                alpha, alpha_max, attn_weight,
            )

        # _step_state persists for the lifetime of this closure = one generation.
        # Reset is triggered inside _wavag_step_alpha when σ jumps (new run).
        _step_state = _make_step_state(alpha)

        def post_cfg_function(args):
            x0_hat        = args["denoised"]
            sigma         = args["sigma"]
            model_inner   = args["model"]
            model_options = args["model_options"]
            cond          = args["cond"]

            sigma_t = float(sigma.max())

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

            # ── Wavelet-space correction ──────────────────────────────────────
            try:
                import ptwt
                import pywt as _pywt

                _wav = _pywt.Wavelet(wavelet)
                residual = (x0_hat - sure_x0).detach()

                res_coeffs = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode="reflect")
                x0_coeffs  = ptwt.wavedec2(x0_hat.detach(), _wav, level=wavelet_level, mode="reflect")
                n_correct_detail = max(0, math.floor(lp_frac * wavelet_level))

                # ── Pass 1: build grad_coeffs + Parseval accumulators ─────────
                r_dot_g   = 0.0
                g_sq      = 0.0
                grad_coeffs: list = []

                # Approximation subband
                r_a = res_coeffs[0]
                if U is not None and attn_weight > 0.0:
                    W_a = 1.0 + attn_weight * _project_entropy_to_subband(
                        U, (r_a.shape[2], r_a.shape[3])
                    )
                else:
                    W_a = 1.0
                g_a = approx_coeff * r_a * W_a
                if abs(fft_scale - 1.0) > 1e-6 and r_a.shape[2] >= 2:
                    g_a = _fft_lowpass_scale(g_a, fft_threshold, fft_scale)
                grad_coeffs.append(g_a)
                r_dot_g += float((r_a * g_a).sum())
                g_sq    += float((g_a * g_a).sum())

                # Detail subbands
                for lvl_idx in range(1, len(res_coeffs)):
                    cH_r, cV_r, cD_r = res_coeffs[lvl_idx]
                    if lvl_idx > n_correct_detail:
                        grad_coeffs.append((
                            torch.zeros_like(cH_r),
                            torch.zeros_like(cV_r),
                            torch.zeros_like(cD_r),
                        ))
                        continue
                    if U is not None and attn_weight > 0.0:
                        W_sb = 1.0 + attn_weight * _project_entropy_to_subband(
                            U, (cH_r.shape[2], cH_r.shape[3])
                        )
                    else:
                        W_sb = 1.0
                    c = approx_coeff * W_sb
                    g_H, g_V, g_D = c * cH_r, c * cV_r, c * cD_r
                    grad_coeffs.append((g_H, g_V, g_D))
                    r_dot_g += float((cH_r * g_H + cV_r * g_V + cD_r * g_D).sum())
                    g_sq    += float((g_H * g_H + g_V * g_V + g_D * g_D).sum())

                # ── Adaptive alpha ────────────────────────────────────────────
                alpha_eff = _wavag_step_alpha(
                    alpha, alpha_max, sigma_t,
                    r_dot_g, g_sq,
                    alpha_mode, _step_state,
                )

                # ── Pass 2: apply correction ──────────────────────────────────
                corrected_coeffs: list = [x0_coeffs[0] - alpha_eff * grad_coeffs[0]]
                for lvl_idx in range(1, len(x0_coeffs)):
                    cH_x, cV_x, cD_x = x0_coeffs[lvl_idx]
                    g_H, g_V, g_D     = grad_coeffs[lvl_idx]
                    corrected_coeffs.append((
                        cH_x - alpha_eff * g_H,
                        cV_x - alpha_eff * g_V,
                        cD_x - alpha_eff * g_D,
                    ))

                corrected = ptwt.waverec2(corrected_coeffs, _wav).detach()
                corrected = corrected[..., :x0_hat.shape[2], :x0_hat.shape[3]]

            except ImportError:
                # Graceful fallback: pixel-space SURE-AG
                residual = (x0_hat - sure_x0).detach()
                if U is not None and attn_weight > 0.0:
                    g = approx_coeff * residual * (1.0 + attn_weight * U)
                else:
                    g = approx_coeff * residual
                r_dot_g = float((residual * g).sum())
                g_sq    = float((g * g).sum())
                alpha_eff = _wavag_step_alpha(
                    alpha, alpha_max, sigma_t,
                    r_dot_g, g_sq,
                    alpha_mode, _step_state,
                )
                corrected = (x0_hat - alpha_eff * g).detach()

            _logger.debug(
                "[sure_wav_ag] sigma=%.5f  alpha_eff=%.5f  entropy_rms=%.4f  attn_layers=%d",
                sigma_t,
                _step_state["cur_alpha"],
                float(U.pow(2).mean().sqrt()) if U is not None else 0.0,
                len(attn_store),
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
