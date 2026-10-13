"""
CLPC composite error module.

Lean-backed error norms for the Closed-Loop Predictor-Corrector sampler:
  - ODE embedded-pair Richardson error
  - OT drift error (8h/δ² window constant for flow models; FlowModelCoord.lean)
  - Haar wavelet subband error — LL (structural) and HF (texture/noise)
    Subband independence proved in WaveletDomain.lean: subband_correction_independence.
    wav_hf is used as the Kalman observation-noise proxy R (KalmanMatrixDesign.lean).
  - SURE wavelet error (from alpha_eff)
  - CFG guidance-drift error (zero NFE via post_cfg hook)
  - G-score F(x) = P_cfg · P_sure · P_entropy · P_ot  (GaussianProcessODE.lean)
    Lyapunov function; proxy for Doob h-function (DoobSOC.lean, ProxySOCvsFull.lean).
  - KL proxy = −log F  (KLOptimality.lean: kl_proxy_positive)
    Convergence monitor; → 0 at target.

All GPU tensor ops are float32 (hardware constraint).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class CLPCWeights:
    ode: float = 1.0
    ot: float = 0.5
    sure: float = 0.3
    cfg: float = 0.2
    # Weight on conditional-space error (1 - P_token) in the composite sum —
    # informational/monitoring only (composite() feeds the progress display,
    # not step-size control: clpc_sampler._clpc_loop's `pid.propose_step` call
    # on this composite is explicitly monitoring-only there too). This is now
    # the ONLY place token/CLIP-derived error affects anything sampler-side —
    # clpc_sampler._kalman_blend deliberately excludes it from the actual
    # corrector-trust gain (K), since a live run showed folding it into that
    # continuous-uncertainty term pinned K near 1.0 for an entire generation.
    # Set from the UI's "token_kalman_weight" slider (clpc_sampler.py's
    # sample_clpc_ode/sde); default 0.0 here so this class stays fully
    # backward-compatible for any caller that constructs it directly.
    token: float = 0.0

    def normalise(self) -> "CLPCWeights":
        total = self.ode + self.ot + self.sure + self.cfg + self.token
        if total <= 0:
            return CLPCWeights()
        return CLPCWeights(
            ode=self.ode / total,
            ot=self.ot / total,
            sure=self.sure / total,
            cfg=self.cfg / total,
            token=self.token / total,
        )


@dataclasses.dataclass
class CLPCError:
    ode: float = 0.0
    ot: float = 0.0
    sure: float = 0.0
    cfg: float = 0.0
    # Haar wavelet subband errors (informational — not in composite weight sum).
    # Used by the Kalman blend in _clpc_loop (KalmanMatrixDesign.lean).
    wav_ll: float = 0.0   # LL subband: structural / low-freq innovation
    wav_hf: float = 0.0   # HF subbands (LH+HL+HH): texture / noise innovation
    # G-score and KL proxy (GaussianProcessODE.lean, KLOptimality.lean).
    gscore: float = 0.0   # F(x) = P_cfg·P_sure·P_entropy·P_ot·P_token ∈ (0, 1]
    kl_proxy: float = 0.0 # −log F; convergence monitor, → 0 at target
    # 5th G-score factor (TokenSubspaceGuidance.lean: gscore5_product_in_unit_interval).
    # 1.0 (neutral) when token guidance is disabled or has captured no diagnostics yet.
    token_score: float = 1.0  # P_token = vanish_score · leak_score · bias_score ∈ (0, 1]

    def composite(self, weights: CLPCWeights) -> float:
        return (
            weights.ode * self.ode
            + weights.ot * self.ot
            + weights.sure * self.sure
            + weights.cfg * self.cfg
            + weights.token * (1.0 - self.token_score)
        )

    def __repr__(self) -> str:
        return (
            f"CLPCError(ode={self.ode:.4f}, ot={self.ot:.4f}, "
            f"sure={self.sure:.4f}, cfg={self.cfg:.4f}, "
            f"wav_ll={self.wav_ll:.4f}, wav_hf={self.wav_hf:.4f}, "
            f"gscore={self.gscore:.4f}, kl={self.kl_proxy:.4f}, "
            f"token_score={self.token_score:.4f})"
        )


# ---------------------------------------------------------------------------
# ODE error  (Richardson / embedded pair)
# ---------------------------------------------------------------------------

def compute_ode_error(
    x_adams: torch.Tensor,
    x_euler: Optional[torch.Tensor],
    atol: float = 1e-4,   # kept for API compat; unused in relative-norm path
    rtol: float = 1e-3,
) -> float:
    """
    Embedded-pair ODE error: relative L2 norm of Adams-vs-Euler gap.

    Returns ‖x_adams − x_euler‖ / ‖x_adams‖ ∈ [0, ∞), naturally O(0.01–0.5)
    for diffusion latents regardless of noise level σ.

    Replaces the atol/rtol tolerance-scaled form, which blew up to ~50–250
    in early steps (|x| ≈ 10–20 makes scale=rtol·|x|≈0.02, so gap/scale≫1).
    """
    if x_euler is None:
        return 0.0
    diff = (x_adams - x_euler).float()
    ref = x_adams.float().norm().clamp(min=1e-8)
    return float(diff.norm() / ref)


# ---------------------------------------------------------------------------
# OT drift error
# ---------------------------------------------------------------------------

def compute_ot_error(
    denoised_t: torch.Tensor,
    denoised_prev: torch.Tensor,
    sigma_t: float,
    sigma_prev: float,
    h: float,
    is_flow_model: bool,
) -> float:
    """
    Measures denoised-estimate drift relative to the OT straight-trajectory
    prediction.

    For flow models the λ-window constant is 8h/δ² (FlowModelCoord.lean:
    coord_transfer_combined); σ acts as the δ lower-bound guard.
    """
    if denoised_prev is None:
        return 0.0

    diff = (denoised_t - denoised_prev).float()
    # Relative change in denoised estimate: dimensionless, O(0.01–0.3) per step.
    # The previous window-bound / h division blew up because h (log-SNR step) is
    # dimensionless while diff.norm() scales with |denoised|, which can be 10–20.
    ref = denoised_t.float().norm().clamp(min=1e-8)
    err = float(diff.norm() / ref)
    return float(min(err, 10.0))


# ---------------------------------------------------------------------------
# Haar wavelet subband error  (Priority 2)
# ---------------------------------------------------------------------------

def compute_haar_subband_errors(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[float, float]:
    """
    One-level 2D Haar wavelet decomposition of the innovation (a − b).
    Returns (ll_err, hf_err) normalised per-element.

    Subbands are orthogonal under Parseval (WaveletDomain.lean:
    wavelet_parseval, subband_correction_independence) — LL and HF errors
    are independent and can be used as separate signals.

    wav_hf is used as Kalman observation-noise proxy R:
      K = ode_err / (ode_err + wav_hf_err)
    (KalmanMatrixDesign.lean: prediction_optimal_linear_1d)

    All ops in float32 on input device — GPU fp32 constraint.

    2x2 Haar block with pixels [a, b; c, d]:
      LL = (a+b+c+d)/4   low-low  (structural / DC)
      LH = (a-b+c-d)/4   low-high (horizontal edges)
      HL = (a+b-c-d)/4   high-low (vertical edges)
      HH = (a-b-c+d)/4   high-high (diagonal / noise)
    """
    diff = (a - b).float()
    B, C, H, W = diff.shape

    # Pad to even spatial dimensions
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        diff = F.pad(diff, (0, pad_w, 0, pad_h))

    H2, W2 = diff.shape[2], diff.shape[3]
    # Reshape into 2×2 blocks: (B, C, H//2, 2, W//2, 2)
    d = diff.reshape(B, C, H2 // 2, 2, W2 // 2, 2)
    p = d[:, :, :, 0, :, 0]   # top-left
    q = d[:, :, :, 0, :, 1]   # top-right
    r = d[:, :, :, 1, :, 0]   # bottom-left
    s = d[:, :, :, 1, :, 1]   # bottom-right

    ll = (p + q + r + s) * 0.25
    lh = (p - q + r - s) * 0.25   # horizontal edges
    hl = (p + q - r - s) * 0.25   # vertical edges
    hh = (p - q - r + s) * 0.25   # diagonal

    n = (ll.numel() ** 0.5) + 1e-8
    ll_err = float(ll.norm()) / n
    hf_sq = lh.norm() ** 2 + hl.norm() ** 2 + hh.norm() ** 2
    hf_err = float(hf_sq.sqrt()) / n

    return ll_err, hf_err


# ---------------------------------------------------------------------------
# SURE error
# ---------------------------------------------------------------------------

def compute_sure_error(sure_info: Optional[dict]) -> float:
    """
    Extract SURE correction magnitude from the info dict returned by
    _sure_correct_x0_wavelet_ag().  alpha_eff ∈ [0,1]; treat high alpha_eff
    as large SURE error (the correction was large).
    """
    if sure_info is None:
        return 0.0
    alpha_eff = sure_info.get("alpha_eff", 0.0)
    return float(alpha_eff)


# ---------------------------------------------------------------------------
# CFG guidance-drift error
# ---------------------------------------------------------------------------

def compute_cfg_drift(cfg_drift_state: dict) -> float:
    """
    Read the last CFG drift value captured by the post_cfg hook.
    Returns e_cfg = 1 - cos(ĝ_t, ĝ_{t-1}) ∈ [0, 2], normalised to [0, 1].
    """
    raw = cfg_drift_state.get("last_cfg_drift", 0.0)
    return float(raw) / 2.0


# ---------------------------------------------------------------------------
# Spectral G-score components  (GaussianProcessODE.lean)
# ---------------------------------------------------------------------------

def compute_spectral_scores(denoised: torch.Tensor) -> Tuple[float, float]:
    """
    Single rfft2 pass → (entropy_score, ot_score), both ∈ (0, 1].

    **P_entropy** (information-restoration score):
        Spectral Shannon entropy H = −Σ p_k log p_k of the power spectrum.
        Lower entropy = more structured output = G_entropy condition met.
        Score = 1 − H/H_max  →  1 when fully structured, 0 when uniform (white noise).
        (GaussianProcessODE.lean: entropy_strictly_decreasing)

    **P_ot** (optimal-transport / low→high freq transfer score):
        Fraction of spectral energy in high spatial-frequency bands.
        As denoising progresses, energy concentrates in low freqs first, then
        spreads to edges/textures — G_ot condition: low-to-high transfer.
        Score = HF_energy / total_energy.

    All ops on GPU float32 (hardware constraint).
    Uses torch.fft.rfft2 (vectorized; returns complex64 from float32 input).
    """
    x = denoised.float()  # GPU float32 — never upcast to float64 here
    # rfft2: (B, C, H, W) float32 → (B, C, H, W//2+1) complex64
    # .abs().pow(2) → float32 power spectrum in one fused kernel
    power = torch.fft.rfft2(x).abs().pow(2)  # (B, C, H, Wh), float32
    _B, _C, H, Wh = power.shape

    # ── P_entropy ─────────────────────────────────────────────────────────
    # Normalise per (B, C) channel to a spectral probability distribution
    total = power.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    p = power / total                             # (B, C, H, Wh), sums to 1
    log_p = p.clamp(min=1e-10).log()             # vectorized log, float32
    entropy = -(p * log_p).sum(dim=(-2, -1))     # (B, C) — Shannon H per channel
    h_mean = entropy.mean().item()               # scalar; pull off GPU once
    h_max = math.log(float(H * Wh)) if H * Wh > 1 else 1.0
    entropy_score = 1.0 - min(max(h_mean / h_max, 0.0), 1.0)

    # ── P_ot ──────────────────────────────────────────────────────────────
    # Low band: DC and low-spatial-freq block (top-left quadrant of spectrum)
    # High band: edges / textures / diagonal noise — everything else
    H4 = max(H // 4, 1)
    Wh2 = max(Wh // 2, 1)
    total_energy = power.sum().clamp(min=1e-8)
    hf_energy = (
        power[:, :, H4:, :].sum()       # high vertical freqs
        + power[:, :, :H4, Wh2:].sum()  # high horizontal freqs (low vert)
    )
    ot_score = float((hf_energy / total_energy).clamp(0.0, 1.0))

    return entropy_score, ot_score


def compute_gscore(
    cfg_err: float,
    sure_err: float,
    entropy_score: float,
    ot_score: float,
    token_score: float = 1.0,
) -> float:
    """
    G-score F(x) = P_cfg · P_sure · P_entropy · P_ot · P_token ∈ (0, 1].

    Proved as Lyapunov function for the sampling ODE in GaussianProcessODE.lean
    (gscore_product_in_unit_interval, gscore_product_monotone, lyapunov_contraction),
    extended to 5 factors in TokenSubspaceGuidance.lean
    (gscore5_product_in_unit_interval, gscore5_product_monotone) — adding
    token_score changes only the Lyapunov constant, never the contraction
    rate (lyapunov_rate_invariant_under_token_term).

    Proxy for the Doob h-function; sufficiency proved in ProxySOCvsFull.lean:
    convergence rate = (1−r)ⁿ, identical to exact HJB solution.

    cfg_err, sure_err are existing error scalars; converted via exp(−err).
    entropy_score, ot_score come from compute_spectral_scores(), already ∈ (0,1].
    token_score comes from compute_token_guidance_score(); defaults to 1.0
    (neutral) so this call stays backward-compatible when token guidance is
    disabled — see anti-pattern guard in plans/03-clpc-token-conditional-guidance.md.
    """
    p_cfg = math.exp(-min(cfg_err * 4.0, 20.0))   # cfg_err ∈ [0,1] → scale up
    p_sure = math.exp(-min(sure_err * 4.0, 20.0))
    gscore = (
        p_cfg * p_sure * max(entropy_score, 1e-8) * max(ot_score, 1e-8)
        * max(token_score, 1e-8)
    )
    return max(gscore, 1e-8)


def compute_token_guidance_score(token_info: Optional[dict]) -> float:
    """
    P_token = vanish_score · intention_drift_score · bias_score ∈ (0, 1].

    `token_info` is the dict produced by
    sure_token_guidance.aggregate_token_guidance_info() for the current step
    (or None if the token-guidance extension is disabled / has captured no
    attn2 diagnostics yet this step — in which case P_token is neutral, 1.0,
    per TokenSubspaceGuidance.lean's `token_term_le_one_discounts_gscore`:
    a factor of 1 never changes the product).

    Uses `intention_drift_score` in place of the raw `leak_score` — it is
    the confidence-weighted refinement of the same underlying leak signal
    (Plan 04: weighted by how confident the rule-engine/Matrix-Tree
    "intention tree" was about each attribute-to-entity assignment), so
    using BOTH would double-count one signal rather than improve accuracy.
    When no intention tree was available this step,
    `aggregate_token_guidance_info` sets `intention_drift_score` equal to
    `leak_score` exactly, so this is a strict refinement, not a behavior
    change, in that fallback case.
    """
    if not token_info:
        return 1.0
    vanish = float(token_info.get("vanish_score", 1.0))
    drift = float(token_info.get("intention_drift_score", token_info.get("leak_score", 1.0)))
    bias = float(token_info.get("bias_score", 1.0))
    return max(vanish * drift * bias, 1e-8)


def compute_kl_proxy(gscore: float) -> float:
    """
    KL proxy = −log F(x)  (KLOptimality.lean: kl_proxy_positive).

    Strictly positive when F < 1; → 0 as F → 1 (sampler at target).
    Used as a convergence monitor — not fed back into the control law.
    (ProxySOCvsFull.lean confirms the control law is already sufficient.)
    """
    return -math.log(max(gscore, 1e-8))


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_clpc_error(
    x_adams: torch.Tensor,
    x_euler: Optional[torch.Tensor],
    x_corr: Optional[torch.Tensor],
    denoised_t: torch.Tensor,
    denoised_prev: Optional[torch.Tensor],
    sigma_t: float,
    sigma_prev: float,
    h: float,
    is_flow_model: bool,
    sure_info: Optional[dict],
    cfg_drift_state: dict,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    token_info: Optional[dict] = None,
) -> CLPCError:
    """
    x_corr: corrected state (None on first step).  Used for Haar subband error
    decomposition of the innovation (x_corr − x_adams).

    denoised_t: model x0 prediction at current step.  Used for spectral G-score
    components (P_entropy, P_ot) via a single rfft2 pass (float32, GPU).

    token_info: aggregated {"vanish_score","leak_score","bias_score"} dict from
    sure_token_guidance.aggregate_token_guidance_info(), or None when the
    token-guidance extension is disabled (→ neutral P_token = 1.0).
    """
    if x_corr is not None:
        wav_ll, wav_hf = compute_haar_subband_errors(x_corr, x_adams)
    else:
        wav_ll, wav_hf = 0.0, 0.0

    cfg_err = compute_cfg_drift(cfg_drift_state)
    sure_err = compute_sure_error(sure_info)

    # G-score: single FFT pass for P_entropy + P_ot, then product with
    # P_cfg · P_sure · P_token (GaussianProcessODE.lean, ProxySOCvsFull.lean,
    # TokenSubspaceGuidance.lean)
    entropy_score, ot_score = compute_spectral_scores(denoised_t)
    token_score = compute_token_guidance_score(token_info)
    gscore = compute_gscore(cfg_err, sure_err, entropy_score, ot_score, token_score)
    kl_proxy = compute_kl_proxy(gscore)

    if token_info is not None:
        print(f"[TokenSubspaceGuidance] build_clpc_error: token_info={token_info} "
              f"-> token_score={token_score:.4f}  gscore={gscore:.4f}")

    return CLPCError(
        ode=compute_ode_error(x_adams, x_euler, atol, rtol),
        ot=compute_ot_error(denoised_t, denoised_prev, sigma_t, sigma_prev, h, is_flow_model),
        sure=sure_err,
        cfg=cfg_err,
        wav_ll=wav_ll,
        wav_hf=wav_hf,
        gscore=gscore,
        kl_proxy=kl_proxy,
        token_score=token_score,
    )
