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
via bilinear resampling.  r_k = x0_k − D_k(x0_hat) is the per-subband residual.

Adaptive alpha modes
--------------------
alpha_mode="fixed"
    Use alpha as-is, clamped by the SURE-AG descent bound α < 1/(2(1+w)).

alpha_mode="sigma"  (recommended default)
    _sure_effective_alpha: alpha / (1 + min(σ_t, 1))
    Prevents near-zero steps on EDM/Karras schedules (σ_t >> 1) while keeping
    effective_alpha ≥ alpha/2 throughout.  Lean: effectiveAlpha_lt_half.

alpha_mode="analytical"
    Closed-form minimiser of the quadratic SURE proxy: α* = ⟨r,g⟩/‖g‖²
    By Parseval: Σ_k ⟨r_k, g_k⟩ / Σ_k ‖g_k‖² — computed in wavelet space
    at zero extra cost.  When attn_weight > 0 the spatially-varying W makes g
    non-collinear with r, so α* is genuinely non-trivial.

alpha_mode="bo"
    Bayesian optimisation (optuna) with cross-step warm-starting.
    Falls back to analytical when optuna is absent.

Cross-step state (_step_state)
------------------------------
A mutable dict persisted across diffusion steps inside the post_cfg closure.
Keys: last_sigma, cur_alpha, prev_sure, prev_prev_sure, bo_study.
Reset automatically when σ_t > 1.5 × last σ_t (new generation / highres-fix).

FFT augmentation (optional, FreeU-inspired)
-------------------------------------------
Optional FFT low-pass scale on the approximation subband gradient.
fft_scale < 1 dampens high-freq noise; fft_scale > 1 boosts structure.
fft_scale=1.0 is identity (disabled).

Formal bounds
-------------
  §1  Each subband W_k ∈ [1, 1+w]  (Lean §2)
  §2  Descent bound:  α < 1/(2·(1+w))  (Lean §3)
  §3  Entropy normalisation to [0,1]  (Lean §5)
  §4  Parseval: ‖x‖² = Σ_k ‖x_k‖²
  §5  FFT lowpass: ‖FFT_lowpass(x)‖ ≤ ‖x‖  for scale ≤ 1
"""

import logging
import math

import torch
import torch.nn.functional as F

from ldm_patched.k_diffusion.sure_attention import (
    _build_attn_capture_options,
    _aggregate_entropy_map,
    build_capture_model_options,
)

_logger = logging.getLogger("sure_wav_ag")

# Set to True to enable verbose print() debug output (easier than setting log levels)
_DEBUG_PRINT = False


def _dbg(*args):
    """Print debug line when _DEBUG_PRINT is True."""
    if _DEBUG_PRINT:
        print("[SURE-AGWAV-DBG]", *args)


# ---------------------------------------------------------------------------
# FFT low-pass filter (FreeU-style)
# ---------------------------------------------------------------------------

def _fft_lowpass_scale(x: torch.Tensor, threshold: int, scale: float) -> torch.Tensor:
    """Apply FFT low-pass scaling. scale=1.0 is identity."""
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
    return torch.fft.ifftn(X, dim=(-2, -1)).real.to(x.dtype)


# ---------------------------------------------------------------------------
# Entropy map projection to subband scale
# ---------------------------------------------------------------------------

def _project_entropy_to_subband(
    U: torch.Tensor,
    subband_shape: tuple[int, int],
) -> torch.Tensor:
    """Resize entropy map U (B,1,H,W) to match a wavelet subband's spatial size."""
    h_sb, w_sb = subband_shape
    if U.shape[2] == h_sb and U.shape[3] == w_sb:
        return U
    return F.interpolate(U, (h_sb, w_sb), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Cross-step state management
# ---------------------------------------------------------------------------

def _make_step_state(alpha_base: float) -> dict:
    """Create a fresh cross-step state dict for adaptive alpha."""
    return {
        "last_sigma":     float("inf"),
        "cur_alpha":      alpha_base,
        "prev_sure":      None,
        "prev_prev_sure": None,
        "bo_study":       None,
    }


# ---------------------------------------------------------------------------
# Adaptive alpha computation (Parseval inner products, no extra UNet calls)
# ---------------------------------------------------------------------------

def _wavag_step_alpha(
    alpha_base: float,
    alpha_max: float,
    sigma_t: float,
    r_dot_g: float,
    g_sq: float,
    alpha_mode: str,
    step_state: dict,
) -> float:
    """Compute effective step size for this diffusion step.

    Parameters
    ----------
    alpha_base  : user-set alpha (upper bound before adaptation)
    alpha_max   : hard safety ceiling — 0.5/(1+attn_weight) − ε
    sigma_t     : current noise level (scalar float)
    r_dot_g     : Σ_k ⟨r_k, g_k⟩ in wavelet space (Parseval dot product)
    g_sq        : Σ_k ‖g_k‖² in wavelet space (Parseval norm²)
    alpha_mode  : "fixed" | "sigma" | "analytical" | "bo"
    step_state  : mutable dict (from _make_step_state); mutated in-place
    """
    # ── Generation reset: σ jumped up → new run or highres-fix ───────────────
    if sigma_t > step_state["last_sigma"] * 1.5:
        step_state.update(
            prev_sure=None, prev_prev_sure=None,
            cur_alpha=alpha_base, bo_study=None,
        )
        _dbg(f"step_state RESET  sigma_t={sigma_t:.4f} > last={step_state['last_sigma']:.4f}")
    step_state["last_sigma"] = sigma_t

    _dbg(f"_wavag_step_alpha  mode={alpha_mode}  alpha_base={alpha_base:.5f}"
         f"  alpha_max={alpha_max:.5f}  sigma_t={sigma_t:.4f}"
         f"  r_dot_g={r_dot_g:.4f}  g_sq={g_sq:.4f}")

    if alpha_mode == "fixed":
        eff = min(alpha_base, alpha_max)

    elif alpha_mode == "sigma":
        from ldm_patched.k_diffusion.sampling import _sure_effective_alpha
        sigma_scaled = _sure_effective_alpha(alpha_base, sigma_t, adam_active=False)
        eff = min(sigma_scaled, alpha_max)
        _dbg(f"  sigma mode: sigma_scaled={sigma_scaled:.5f} → eff={eff:.5f}")

    elif alpha_mode in ("analytical", "bo"):
        g_sq_safe = max(g_sq, 1e-12)
        alpha_analytical = float(max(1e-5, min(alpha_max, r_dot_g / g_sq_safe)))
        _dbg(f"  analytical α* = {r_dot_g:.4f}/{g_sq:.4f} = {alpha_analytical:.5f}")

        if alpha_mode == "analytical":
            eff = alpha_analytical

        else:  # "bo"
            lo = max(alpha_analytical * 0.01, 1e-5)
            hi = min(alpha_analytical * 20.0, alpha_max)
            if lo >= hi:
                hi = min(lo * 10.0, alpha_max)

            # Quadratic proxy (no UNet calls): -2a·⟨r,g⟩ + a²·‖g‖²
            def _proxy(a):
                return -2.0 * a * r_dot_g + a * a * g_sq

            try:
                import optuna as _optuna
                _optuna.logging.set_verbosity(_optuna.logging.WARNING)
                study = _optuna.create_study(
                    direction="minimize",
                    sampler=_optuna.samplers.TPESampler(
                        seed=42, n_startup_trials=3, multivariate=False,
                    ),
                )
                # Warm-start from previous step
                if step_state["bo_study"] is not None:
                    for t in step_state["bo_study"].trials:
                        if t.value is not None and t.value != float("inf"):
                            try:
                                study.add_trial(
                                    _optuna.trial.create_trial(
                                        params=t.params,
                                        distributions=t.distributions,
                                        value=t.value,
                                    )
                                )
                            except Exception:
                                pass

                study.enqueue_trial({"alpha": alpha_analytical})
                _no_improve = [0]
                _best_val   = [float("inf")]

                def _cb(study, trial):
                    val = study.best_value
                    if val < _best_val[0]:
                        _best_val[0] = val
                        _no_improve[0] = 0
                    else:
                        _no_improve[0] += 1
                    if _no_improve[0] >= 4:
                        study.stop()

                study.optimize(
                    lambda t: _proxy(t.suggest_float("alpha", lo, hi, log=True)),
                    n_trials=12, callbacks=[_cb], show_progress_bar=False,
                )
                eff = float(study.best_params["alpha"])
                step_state["bo_study"] = study
                _dbg(f"  BO best={eff:.5f}  trials={len(study.trials)}")

            except ImportError:
                _logger.debug("[sure_wav_ag_bo] optuna absent — analytical fallback")
                _dbg("  BO: optuna absent, using analytical")
                eff = alpha_analytical
                step_state["bo_study"] = None

    else:
        _logger.warning("[sure_wav_ag] unknown alpha_mode=%r; using 'fixed'", alpha_mode)
        eff = min(alpha_base, alpha_max)

    # Track SURE proxy for cross-step history
    sure_proxy = -2.0 * eff * r_dot_g + eff * eff * g_sq
    step_state["prev_prev_sure"] = step_state["prev_sure"]
    step_state["prev_sure"]      = float(sure_proxy)
    step_state["cur_alpha"]      = eff

    _dbg(f"  → eff_alpha={eff:.5f}  sure_proxy={sure_proxy:.4f}")
    return float(eff)


# ---------------------------------------------------------------------------
# Core SURE-AGWAV correction function
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
    alpha_mode: str = "sigma",
    step_state: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """SURE-AGWAV: per-subband wavelet correction weighted by attention entropy.

    Parameters
    ----------
    alpha_mode  : "fixed" | "sigma" | "analytical" | "bo"
    step_state  : persistent dict from _make_step_state(); pass the same dict
                  on every call within one generation to enable BO/history.
                  Pass None to auto-create (stateless, fine for "fixed"/"sigma").
    """
    alpha_max = 0.5 / (1.0 + max(attn_weight, 0.0)) - 1e-4
    if step_state is None:
        step_state = _make_step_state(alpha)

    _dbg(f"_sure_correct_x0_wavelet_ag  sigma={sigma_hat_0:.5f}"
         f"  alpha={alpha:.5f}  alpha_mode={alpha_mode}"
         f"  wavelet={wavelet}  level={wavelet_level}"
         f"  attn_blocks={attn_blocks}  attn_weight={attn_weight:.2f}"
         f"  lp_frac={lp_frac:.2f}  fft_scale={fft_scale:.3f}")

    try:
        import ptwt
        import pywt as _pywt
        _dbg("  ptwt/pywt available")
    except ImportError:
        _logger.warning(
            "[sure_wav_ag] ptwt/pywt not installed — falling back to pixel-space SURE-AG. "
            "Install with: pip install pytorch-wavelets pywavelets"
        )
        _dbg("  ptwt MISSING — pixel-space fallback")
        attn_store: list = []
        capture_extra = _build_attn_capture_options(extra_args, attn_store, attn_blocks)
        with torch.no_grad():
            sure_x0 = model(x0_hat, sigma_hat_0 * s_in, **capture_extra).detach()
        U = _aggregate_entropy_map(attn_store, x0_hat.shape, x0_hat.device, x0_hat.dtype)
        residual = (x0_hat - sure_x0).detach()
        if U is not None and attn_weight > 0.0:
            g = approx_coeff * residual * (1.0 + attn_weight * U)
        else:
            g = approx_coeff * residual
        r_dot_g = float((residual * g).sum())
        g_sq    = float((g * g).sum())
        alpha_eff = _wavag_step_alpha(
            alpha, alpha_max, sigma_hat_0, r_dot_g, g_sq, alpha_mode, step_state,
        )
        return (x0_hat - alpha_eff * g).detach(), {
            "entropy_rms":   float(U.pow(2).mean().sqrt()) if U is not None else 0.0,
            "n_subbands":    0,
            "n_attn_layers": len(attn_store),
            "alpha_eff":     alpha_eff,
        }

    _wav = _pywt.Wavelet(wavelet)
    device = x0_hat.device

    # ── Forward pass with attention entropy capture ───────────────────────────
    attn_store: list = []
    capture_extra = _build_attn_capture_options(extra_args, attn_store, attn_blocks)
    with torch.no_grad():
        sure_x0 = model(x0_hat.detach(), sigma_hat_0 * s_in, **capture_extra).detach()

    _dbg(f"  attn layers captured: {len(attn_store)}")

    # ── Aggregate attention entropy map U ∈ [0,1] ────────────────────────────
    U = _aggregate_entropy_map(attn_store, x0_hat.shape, device, x0_hat.dtype)
    entropy_rms = float(U.pow(2).mean().sqrt()) if U is not None else 0.0
    _dbg(f"  entropy_rms={entropy_rms:.4f}  U={'None' if U is None else str(tuple(U.shape))}")

    # ── Residual ──────────────────────────────────────────────────────────────
    residual = (x0_hat - sure_x0).detach()
    _dbg(f"  residual rms={float(residual.pow(2).mean().sqrt()):.5f}")

    # ── Wavelet decomposition ─────────────────────────────────────────────────
    res_coeffs = ptwt.wavedec2(residual, _wav, level=wavelet_level, mode="reflect")
    x0_coeffs  = ptwt.wavedec2(x0_hat.detach(), _wav, level=wavelet_level, mode="reflect")
    n_correct_detail = max(0, math.floor(lp_frac * wavelet_level))

    _dbg(f"  wavedec2: {len(res_coeffs)} bands  approx_shape={tuple(res_coeffs[0].shape)}"
         f"  n_correct_detail={n_correct_detail}/{wavelet_level}")

    # ── Pass 1: build grad_coeffs and accumulate Parseval dot products ────────
    # r_dot_g = Σ_k ⟨r_k, g_k⟩  →  used for α* = r_dot_g/g_sq
    # g_sq    = Σ_k ‖g_k‖²
    r_dot_g  = 0.0
    g_sq     = 0.0
    grad_coeffs: list = []  # same structure as x0_coeffs

    # Approximation subband
    r_a = res_coeffs[0]
    if U is not None and attn_weight > 0.0:
        W_a = 1.0 + attn_weight * _project_entropy_to_subband(U, (r_a.shape[2], r_a.shape[3]))
    else:
        W_a = 1.0
    g_a = approx_coeff * r_a * W_a
    if abs(fft_scale - 1.0) > 1e-6 and r_a.shape[2] >= 2:
        g_a = _fft_lowpass_scale(g_a, fft_threshold, fft_scale)
        _dbg(f"  FFT lowpass on approx: scale={fft_scale:.3f}  threshold={fft_threshold}")
    grad_coeffs.append(g_a)
    r_dot_g += float((r_a * g_a).sum())
    g_sq    += float((g_a * g_a).sum())

    # Detail subbands
    for lvl_idx in range(1, len(res_coeffs)):
        cH_r, cV_r, cD_r = res_coeffs[lvl_idx]
        if lvl_idx > n_correct_detail:
            # Pass-through: zero gradient
            grad_coeffs.append((
                torch.zeros_like(cH_r),
                torch.zeros_like(cV_r),
                torch.zeros_like(cD_r),
            ))
            _dbg(f"  L{wavelet_level - lvl_idx + 1} (idx {lvl_idx}): SKIPPED (lp_frac)")
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
        band_rdg = float((cH_r * g_H + cV_r * g_V + cD_r * g_D).sum())
        band_gsq = float((g_H * g_H + g_V * g_V + g_D * g_D).sum())
        r_dot_g += band_rdg
        g_sq    += band_gsq
        _dbg(f"  L{wavelet_level - lvl_idx + 1} (idx {lvl_idx}):"
             f"  shape={tuple(cH_r.shape)}"
             f"  r_rms={float((cH_r**2+cV_r**2+cD_r**2).mean().sqrt()):.5f}"
             f"  rdg={band_rdg:.4f}  gsq={band_gsq:.4f}")

    _dbg(f"  Parseval totals: r_dot_g={r_dot_g:.4f}  g_sq={g_sq:.4f}")

    # ── Adaptive alpha ────────────────────────────────────────────────────────
    alpha_eff = _wavag_step_alpha(
        alpha, alpha_max, sigma_hat_0, r_dot_g, g_sq, alpha_mode, step_state,
    )

    # ── Pass 2: apply correction ──────────────────────────────────────────────
    corrected_coeffs: list = [x0_coeffs[0] - alpha_eff * grad_coeffs[0]]
    for lvl_idx in range(1, len(x0_coeffs)):
        cH_x, cV_x, cD_x = x0_coeffs[lvl_idx]
        g_H, g_V, g_D     = grad_coeffs[lvl_idx]
        corrected_coeffs.append((
            cH_x - alpha_eff * g_H,
            cV_x - alpha_eff * g_V,
            cD_x - alpha_eff * g_D,
        ))

    # ── Reconstruct ──────────────────────────────────────────────────────────
    x0_corrected = ptwt.waverec2(corrected_coeffs, _wav).detach()
    x0_corrected = x0_corrected[..., :x0_hat.shape[2], :x0_hat.shape[3]]

    delta_rms = float((x0_corrected - x0_hat).pow(2).mean().sqrt())
    _dbg(f"  correction delta_rms={delta_rms:.5f}"
         f"  x0_corrected range=[{float(x0_corrected.min()):.3f}, {float(x0_corrected.max()):.3f}]")

    n_subbands = 1 + 3 * wavelet_level
    _logger.info(
        "[sure_wav_ag] sigma=%.5f  mode=%s  alpha_eff=%.5f  entropy_rms=%.4f  "
        "attn_layers=%d  n_subbands=%d  corrected=%d/%d  fft_scale=%.3f",
        sigma_hat_0, alpha_mode, alpha_eff, entropy_rms,
        len(attn_store), n_subbands, n_correct_detail, wavelet_level, fft_scale,
    )

    return x0_corrected, {
        "entropy_rms":   entropy_rms,
        "n_subbands":    n_subbands,
        "n_attn_layers": len(attn_store),
        "alpha_eff":     alpha_eff,
    }
