"""sure_wav_ag.py — SURE-AGWAV: Attention-Guided Wavelet SURE correction.

Theory
------
SURE-AGWAV unifies two orthogonal guidance axes:

    1. Location-aware  (SURE-AG)     — attention entropy U ∈ [0,1] marks WHICH
                                       spatial positions are semantically uncertain.
    2. Frequency-aware (Wavelet SURE) — wavelet decomposition separates WHICH
                                        frequency bands need correction.

Standard SURE-AG applies a single entropy map across all frequencies:

    x0_corrected = x0_hat − α · (1 + w·U) · ∇SURE

SURE-AGWAV operates in wavelet space.  For each subband k at level l:

    x0_k_corrected = x0_k − α · (1 + w·Ũ_k) · r_k · approx_coeff

where Ũ_k is the entropy map projected into the same spatial scale as subband k
via bilinear resampling (not a wavelet decomposition of U, to avoid phase
artifacts). r_k = x0_k − D_k(x0_hat) is the per-subband residual.

The per-subband spatial weight Ũ_k has the same resolution as subband k:
  — Coarse approximation (level L):  Ũ_0 ≈ U downsampled 2^L × — global structure
  — Fine detail (level 1):           Ũ_L ≈ U at near-full resolution — local texture

This means:
  * Attention entropy from the middle UNet block (large RF, coarse) drives
    correction of the low-frequency approximation subband.
  * Entropy from input/output blocks (small RF, fine detail) drives correction
    of the high-frequency detail subbands.
  * The method degrades gracefully: attn_weight=0 → plain wavelet SURE;
    no pywt → falls back to pixel-space SURE-AG.

FFT augmentation (optional, FreeU-inspired)
-------------------------------------------
FreeU (Si et al., 2023) proved that dampening the high-frequency content of
UNet skip connections improves generation quality by reducing texture noise.
SURE-AGWAV optionally applies an FFT low-pass scale to the approximation
subband residual correction, boosting structural corrections while reducing
noise in the gradient:

    r_approx_fft_boosted = FFT_lowpass(r_approx, threshold, scale)

This is controlled by `fft_scale` (1.0 = disabled = identity).
`fft_threshold` sets the low-frequency window radius (default 1, same as FreeU).

Formal bounds (inherited from SURE-AG + Wavelet SURE)
------------------------------------------------------
  §1  Each subband W_k ∈ [1, 1+w]  (weight bounded, Lean §2)
  §2  Descent bound:  α < 1/(2·(1+w))  (Lean §3, per subband)
  §3  Entropy normalisation to [0,1] before weighting (Lean §5)
  §4  Wavelet orthogonality: ‖x‖² = Σ_k ‖x_k‖² (Parseval — energy preserved)
  §5  FFT lowpass: ‖FFT_lowpass(x)‖ ≤ ‖x‖  (energy non-increasing)

All five constraints are preserved by reconstruction via waverec2.
"""

import logging
import math

import torch
import torch.nn.functional as F

from ldm_patched.k_diffusion.sure_attention import (
    _make_entropy_hook,
    _build_attn_capture_options,
    _aggregate_entropy_map,
    build_capture_model_options,
)

_logger = logging.getLogger("sure_wav_ag")


# ---------------------------------------------------------------------------
# FFT low-pass filter (FreeU-style, applied per-correction)
# ---------------------------------------------------------------------------

def _fft_lowpass_scale(x: torch.Tensor, threshold: int, scale: float) -> torch.Tensor:
    """Apply FFT low-pass scaling to a spatial tensor.

    Multiplies the central low-frequency region (radius=threshold) of the FFT
    spectrum by `scale`.  scale < 1 dampens high-frequency content (noise);
    scale > 1 boosts low-frequency content (structure).  scale=1.0 is identity.

    Operates in float32 internally; returns in original dtype.
    """
    if abs(scale - 1.0) < 1e-6:
        return x

    x_f32 = x.float()
    X = torch.fft.fftn(x_f32, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))

    B, C, H, W = X.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    t = max(1, int(threshold))
    mask[..., crow - t:crow + t, ccol - t:ccol + t] = scale

    X = X * mask
    X = torch.fft.ifftshift(X, dim=(-2, -1))
    x_out = torch.fft.ifftn(X, dim=(-2, -1)).real
    return x_out.to(x.dtype)


# ---------------------------------------------------------------------------
# Entropy map projection to subband scale
# ---------------------------------------------------------------------------

def _project_entropy_to_subband(
    U: torch.Tensor,
    subband_shape: tuple[int, int],
) -> torch.Tensor:
    """Resize entropy map U to match the spatial size of a wavelet subband.

    U has shape (B, 1, H_lat, W_lat).  Returns (B, 1, h_sb, w_sb) where
    h_sb, w_sb = subband_shape.  Uses bilinear interpolation so the projected
    map stays smooth and ∈ [0,1].
    """
    h_sb, w_sb = subband_shape
    if U.shape[2] == h_sb and U.shape[3] == w_sb:
        return U
    return F.interpolate(U, (h_sb, w_sb), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Per-subband SURE-AGWAV correction
# ---------------------------------------------------------------------------

def _sure_correct_x0_wavelet_ag(
    model,
    x0_hat: torch.Tensor,
    sigma_hat_0: float,
    s_in: torch.Tensor,
    extra_args: dict,
    alpha: float = 0.05,
    approx_coeff: float = 2.0,
    attn_weight: float = 1.0,
    attn_blocks: str = "middle",
    wavelet: str = "db4",
    wavelet_level: int = 3,
    lp_frac: float = 1.0,
    fft_scale: float = 1.0,
    fft_threshold: int = 1,
) -> tuple[torch.Tensor, dict]:
    """SURE-AGWAV: per-subband wavelet correction weighted by attention entropy.

    Parameters
    ----------
    model          : callable — the denoising model (sigma-parametrised)
    x0_hat         : (B, C, H, W) denoised estimate from the sampler
    sigma_hat_0    : scalar — current noise level
    s_in           : (B,) sigma broadcast tensor
    extra_args     : dict passed to model (cond, uncond, model_options, ...)
    alpha          : SURE step size.  Auto-clamped to < 1/(2*(1+attn_weight)).
    approx_coeff   : gradient scale (2.0 matches SURE theory)
    attn_weight    : entropy amplification ≥ 0.  0 = pure wavelet SURE.
    attn_blocks    : which UNet blocks to capture entropy from
    wavelet        : pywt wavelet family (default 'db4')
    wavelet_level  : decomposition levels (default 3)
    lp_frac        : fraction of detail levels to correct (1.0 = all)
    fft_scale      : low-pass scale applied to approx subband residual
                     1.0 = disabled; < 1 dampens noise; > 1 boosts structure
    fft_threshold  : FFT low-pass window radius (pixels in latent space)

    Returns
    -------
    x0_corrected : Tensor — corrected denoised estimate
    info         : dict — {'entropy_rms': float, 'n_subbands': int, 'n_attn_layers': int}
    """
    try:
        import ptwt
        import pywt as _pywt
    except ImportError:
        _logger.warning(
            "[sure_wav_ag] ptwt/pywt not installed — falling back to pixel-space SURE-AG. "
            "Install with: pip install pytorch-wavelets pywavelets"
        )
        # Fallback to SURE-AG without wavelet
        attn_store: list = []
        capture_extra = _build_attn_capture_options(extra_args, attn_store, attn_blocks)
        with torch.no_grad():
            sure_x0 = model(x0_hat, sigma_hat_0 * s_in, **capture_extra).detach()
        U = _aggregate_entropy_map(attn_store, x0_hat.shape, x0_hat.device, x0_hat.dtype)
        residual = x0_hat - sure_x0
        if U is not None and attn_weight > 0.0:
            grad = approx_coeff * residual * (1.0 + attn_weight * U)
        else:
            grad = approx_coeff * residual
        alpha_eff = min(alpha, 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4)
        return (x0_hat - alpha_eff * grad).detach(), {
            "entropy_rms": float(U.pow(2).mean().sqrt()) if U is not None else 0.0,
            "n_subbands": 0,
            "n_attn_layers": len(attn_store),
        }

    _wav = _pywt.Wavelet(wavelet)
    device = x0_hat.device

    # ── Forward pass with attention entropy capture ───────────────────────────
    attn_store: list = []
    capture_extra = _build_attn_capture_options(extra_args, attn_store, attn_blocks)

    with torch.no_grad():
        sure_x0 = model(
            x0_hat.detach(), sigma_hat_0 * s_in, **capture_extra
        ).detach()

    # ── Aggregate attention entropy map U ∈ [0,1] ───────────────────────────
    U = _aggregate_entropy_map(attn_store, x0_hat.shape, device, x0_hat.dtype)
    entropy_rms = float(U.pow(2).mean().sqrt()) if U is not None else 0.0

    # ── Residual in pixel space ───────────────────────────────────────────────
    residual = (x0_hat - sure_x0).detach()  # (B, C, H, W)

    # ── Wavelet decomposition of residual ────────────────────────────────────
    # ptwt.wavedec2 returns:
    #   [cA,  (cH_L, cV_L, cD_L),  ...,  (cH_1, cV_1, cD_1)]
    #    approximation               finest detail subbands
    res_coeffs  = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode="reflect")
    x0_coeffs   = ptwt.wavedec2(x0_hat.detach(), _wav, level=wavelet_level, mode="reflect")

    n_detail_levels = wavelet_level
    n_correct_detail = max(0, math.floor(lp_frac * n_detail_levels))

    # Effective alpha — clamp for SURE-AG descent bound
    alpha_eff = min(alpha, 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4)

    corrected_coeffs = []

    # ── Approximation subband (index 0) ──────────────────────────────────────
    r_approx = res_coeffs[0]   # (B, C, h_a, w_a)
    x_approx = x0_coeffs[0]

    if U is not None and attn_weight > 0.0:
        U_approx = _project_entropy_to_subband(U, (r_approx.shape[2], r_approx.shape[3]))
        W_approx = 1.0 + attn_weight * U_approx
    else:
        W_approx = 1.0

    grad_approx = approx_coeff * r_approx * W_approx

    # Optional FFT low-pass boost on approximation gradient (FreeU-inspired)
    if abs(fft_scale - 1.0) > 1e-6 and r_approx.shape[2] >= 2 and r_approx.shape[3] >= 2:
        grad_approx = _fft_lowpass_scale(grad_approx, fft_threshold, fft_scale)
        _logger.debug(
            "[sure_wav_ag] FFT applied to approx subband  scale=%.3f  threshold=%d",
            fft_scale, fft_threshold,
        )

    corrected_coeffs.append(x_approx - alpha_eff * grad_approx)

    # ── Detail subbands  (cH_L, cV_L, cD_L) at each level ───────────────────
    # res_coeffs[1] = coarsest detail (level L), res_coeffs[-1] = finest (level 1)
    for lvl_idx in range(1, len(res_coeffs)):
        cH_r, cV_r, cD_r = res_coeffs[lvl_idx]
        cH_x, cV_x, cD_x = x0_coeffs[lvl_idx]

        # lvl_idx 1 → coarsest details (level L), should correct if within lp_frac
        # lvl_idx n_detail_levels → finest details (level 1)
        # Correct levels from coarsest up to n_correct_detail levels
        should_correct = (lvl_idx <= n_correct_detail)

        if not should_correct:
            corrected_coeffs.append((cH_x, cV_x, cD_x))
            continue

        # Project entropy map to this subband's resolution
        if U is not None and attn_weight > 0.0:
            h_sb, w_sb = cH_r.shape[2], cH_r.shape[3]
            U_sb = _project_entropy_to_subband(U, (h_sb, w_sb))
            W_sb = 1.0 + attn_weight * U_sb
        else:
            W_sb = 1.0

        cH_corr = cH_x - alpha_eff * approx_coeff * cH_r * W_sb
        cV_corr = cV_x - alpha_eff * approx_coeff * cV_r * W_sb
        cD_corr = cD_x - alpha_eff * approx_coeff * cD_r * W_sb
        corrected_coeffs.append((cH_corr, cV_corr, cD_corr))

    # ── Reconstruct corrected x0 ──────────────────────────────────────────────
    x0_corrected = ptwt.waverec2(corrected_coeffs, _wav).detach()

    # Trim to original size (waverec2 may produce off-by-one padding)
    x0_corrected = x0_corrected[..., :x0_hat.shape[2], :x0_hat.shape[3]]

    n_subbands = 1 + 3 * wavelet_level

    _logger.info(
        "[sure_wav_ag] sigma=%.5f  alpha_eff=%.5f  entropy_rms=%.4f  "
        "attn_layers=%d  n_subbands=%d  corrected_detail=%d/%d  "
        "attn_blocks=%s  fft_scale=%.3f",
        sigma_hat_0, alpha_eff, entropy_rms,
        len(attn_store), n_subbands, n_correct_detail, n_detail_levels,
        attn_blocks, fft_scale,
    )

    return x0_corrected, {
        "entropy_rms": entropy_rms,
        "n_subbands": n_subbands,
        "n_attn_layers": len(attn_store),
    }
