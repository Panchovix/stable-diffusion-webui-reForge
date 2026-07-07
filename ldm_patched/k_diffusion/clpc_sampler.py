"""
CLPC — Closed-Loop Predictor-Corrector adaptive sampler.

Two variants:
  clpc_ode  — deterministic (tau_t = 0)
  clpc_sde  — stochastic   (tau_t from get_tau_interval_func)

Both use:
  • Configurable Adams predictor (see PREDICTOR_* constants below)
  • Adams-Moulton corrector (implicit; lam_t in nodes — always bounded)
  • ER-SDE-style coordinate for flow models (er_lambda = σ/(1-σ) after log-SNR)
  • DC-Solver online calibration of corrector coefficients
  • PECE loop gated by sigma (pece_sigma_threshold=4.0, proved in PECEOrderGain.lean)
  • CFG guidance-drift capture via post_cfg hook (zero NFE)
  • PIDStepSizeController for adaptive step sizing
  • Composite error norm → monitor (Lean-backed weights)

Four Lean-proved enhancements (see lean_proofs_rfv/THEOREM_BUFFER.md):

  Priority 1 — Chebyshev node selection (use_chebyshev=True, default):
    When selecting history entries for the Adams predictor, prefer entries whose
    λ positions approximate Chebyshev nodes on [λ_oldest, λ_s].  Proved minimax-
    optimal in ChebyshevAdaptive.lean (chebyshev_monic_minimax, adams_vs_chebyshev).
    Eliminates Runge-phenomenon risk for order ≥ 2 without falling back to Euler.

  Priority 2 — Haar wavelet subband error (always active):
    One-level 2D Haar decomposition of the innovation (x_corr − x_pred) gives
    independent LL (structural) and HF (texture/noise) error signals.  Subband
    independence proved in WaveletDomain.lean (subband_correction_independence).
    HF error feeds the Kalman observation-noise proxy R.

  Priority 3 — Kalman-gain blend (use_kalman=True, default):
    After PECE correction, blend predictor and corrector outputs with gain
      K = ode_err / (ode_err + wav_hf_err)
    where ode_err is the embedded-pair ODE error (prediction uncertainty P) and
    wav_hf_err is the Haar HF subband error (observation noise R).
    State-transition matrix A = (σ_t/σ_s)·I proved strictly contractive in
    KalmanMatrixDesign.lean (sigma_scheduled_contractive).  MMSE optimality
    proved in KalmanFilter.lean (prediction_optimal_linear_1d).
    Token/CLIP-derived error (from sure_token_guidance, when active) is
    DELIBERATELY NOT part of this gain — see _kalman_blend's docstring. It's
    tracked as its own separate, un-smoothed channel (no persistent
    confidence carried between steps) that only feeds the informational
    G-score composite, since it can shift abruptly step-to-step in a way
    ode_err's continuous-uncertainty semantics don't.

  Priority 4 — Configurable maximum order (max_order=3, default; UniPC-style):
    Predictor/corrector order is capped at `min(len(history), max_order)` /
    `min(len(history)+1, max_order+1)` instead of the previously hardcoded
    3/4, mirroring UniPC's user-facing `order` parameter. Raising max_order
    is a valid corrector update at ANY order — the b-coefficients always sum
    to 1, proved in general (not just for the hand-verified n=2,3 cases) via
    Mathlib's Lagrange interpolation library in VariableOrderGain.lean
    (am_b_coeffs_sum_to_one_general) — and strictly reduces local truncation
    error as h → 0 (order_gain_ratio_tendsto_zero). But partition of unity
    only bounds the coefficients' SUM, not any individual one: past order 3,
    individual-coefficient boundedness is only established for Chebyshev-
    spaced nodes (chebyshev_monic_minimax, already order-general), so
    use_chebyshev is now also applied to the corrector's history selection
    when its order exceeds 3, not just the predictor's (see
    variable_order_needs_chebyshev_beyond_three). `lower_order_final=True`
    (default) additionally ramps order down near the end of the σ schedule,
    matching UniPC's own `lower_order_final`.

Predictor modes (lean_proofs_rfv/RFVProofs/AdamsStability.lean):
  "euler"           — order-1 always; b = 1-σ_t/σ_s ∈ (0,1); unconditionally safe.
  "implicit_adams"  — AM-style: lam_t added as rightmost node with denoised_s proxy;
                      integration inside node convex hull → bounded (default).
"""

from __future__ import annotations

import collections
import dataclasses
import math
import os
from typing import Optional, List, Callable

import torch
import torch.nn.functional as F

# Set CLPC_DEBUG=1 to print per-step error breakdown to stdout
_DEBUG = os.environ.get("CLPC_DEBUG", "0") == "1"

# Predictor mode constants — see AdamsStability.lean for formal proofs.
# "euler":           order-1 always; b = 1-σ_t/σ_s ∈ (0,1); no Runge possible.
# "implicit_adams":  AM-style; lam_t added as rightmost node with denoised_s proxy;
#                    integration [lam_s, lam_t] inside node convex hull → bounded.
PREDICTOR_EULER = "euler"
PREDICTOR_IMPLICIT_ADAMS = "implicit_adams"

import ldm_patched.modules.model_patcher
import ldm_patched.modules.model_sampling

from ldm_patched.k_diffusion import sa_solver as _sa
from ldm_patched.k_diffusion.clpc_error import (
    CLPCWeights, CLPCError, build_clpc_error,
)
from ldm_patched.k_diffusion.clpc_calibration import (
    CLPCCalibrationState, accumulate_calibration, get_corrector_coeffs,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _detect_flow_model(model) -> bool:
    try:
        ms = model.inner_model.model_patcher.get_model_object("model_sampling")
        return isinstance(ms, ldm_patched.modules.model_sampling.CONST)
    except Exception:
        return False


def _sigma_to_lambda(sigma: torch.Tensor, is_flow: bool) -> torch.Tensor:
    """
    EPS:  λ = -log(σ)          (half-log-SNR via log(α/σ), α≈1 for VP)
    Flow: λ = σ/(1-σ)          (er_lambda coordinate; monotone proved in FlowModelCoord.lean)

    Always returns a CPU float64 tensor — lambda values are only used in
    Vandermonde solves (CPU float64) and scalar arithmetic, never in GPU kernels.
    """
    if is_flow:
        sigma_c = sigma.double().cpu().clamp(1e-5, 1.0 - 1e-5)
        return sigma_c / (1.0 - sigma_c)
    else:
        return sigma.double().cpu().log().neg()


def _offset_sigma_for_flow(sigmas: torch.Tensor, is_flow: bool) -> torch.Tensor:
    if is_flow and sigmas[0] >= 1.0:
        sigmas = sigmas.clone()
        sigmas[0] = sigmas[0] - 1e-4
    return sigmas


# ── Priority 1: Chebyshev node selection ─────────────────────────────────────

def _chebyshev_node_positions(n: int, lam_a: float, lam_b: float) -> List[float]:
    """
    Chebyshev nodes of the first kind on [lam_a, lam_b]:
      t_j = (a+b)/2 + (b-a)/2 · cos(π(2j+1)/(2n))  for j = 0..n-1.

    Proved minimax-optimal in ChebyshevAdaptive.lean (chebyshev_monic_minimax):
    among all monic degree-n polynomials, Chebyshev spacing minimises ‖·‖∞.
    """
    if n <= 0:
        return []
    mid = (lam_a + lam_b) / 2.0
    half = (lam_b - lam_a) / 2.0
    return [mid + half * math.cos(math.pi * (2 * j + 1) / (2 * n)) for j in range(n)]


def _select_chebyshev_history(history: "CLPCHistory", n_nodes: int, lam_s: float) -> list:
    """
    Select n_nodes history entries whose λ values best approximate Chebyshev
    nodes on [λ_oldest, lam_s].

    Replaces history.last(n_nodes) which clusters all nodes near lam_s
    (equally-spaced → Runge risk, proved in RungeGuard.lean:
    lagrange_two_node_at_exterior_point_unbounded).  Chebyshev spacing keeps
    the interpolation bounded for any step-size ratio (adams_vs_chebyshev).
    """
    all_entries = history.last(len(history))
    if len(all_entries) <= n_nodes:
        return all_entries

    targets = _chebyshev_node_positions(n_nodes, all_entries[0].lam, lam_s)
    used: set = set()
    selected = []
    for target in targets:
        best_i, best_dist = -1, float("inf")
        for i, entry in enumerate(all_entries):
            if i in used:
                continue
            d = abs(entry.lam - target)
            if d < best_dist:
                best_dist, best_i = d, i
        if best_i >= 0:
            used.add(best_i)
            selected.append(all_entries[best_i])

    selected.sort(key=lambda e: e.lam)
    return selected


# ── Priority 3: Kalman-gain blend ─────────────────────────────────────────────

#: Weight on token-guidance badness (1 - P_token) inside the Kalman P term —
#: see TokenSubspaceGuidance.lean Part 7 (kalman_gain_with_token_term_valid,
#: token_term_increases_corrector_trust). Same order of magnitude as ode_err's
#: own scale (both are O(0.01-0.5) relative errors); tune here, not per-call.
TOKEN_KALMAN_WEIGHT = 0.5


def _kalman_blend(
    x_pred: torch.Tensor,
    x_corr: torch.Tensor,
    ode_err: float,
    wav_hf_err: float,
    token_err: float = 0.0,
) -> torch.Tensor:
    """
    Kalman-gain-weighted blend of predictor and corrector outputs.

    A = (σ_t/σ_s)·I — proved strictly contractive in KalmanMatrixDesign.lean:
    sigma_scheduled_contractive (‖A‖ = σ_t/σ_s < 1 for any standard schedule).

    K = P / (P + R)
      P = ode_err  — prediction uncertainty: embedded-pair ODE gap only.
      R = wav_hf_err — observation noise (Haar HF subband of innovation)

    MMSE optimality proved in KalmanFilter.lean: prediction_optimal_linear_1d.

    `token_err` (1 - P_token, conditional-space badness from
    sure_token_guidance) is measured and printed here for visibility, but is
    DELIBERATELY EXCLUDED from P/K. An earlier version folded it into P as
    `P = ode_err + TOKEN_KALMAN_WEIGHT * token_err` (TokenSubspaceGuidance.lean
    Part 7: kalman_gain_with_token_term_valid — still a mathematically valid
    formula: K stays in [0,1], adding token_err never decreases corrector
    trust). That was empirically wrong for this use, not just imperfect: a
    live run showed leak_score plateau well below 1.0 for an ENTIRE
    generation (token_err rarely reaches ~0), which kept P inflated and K
    pinned near 1.0 regardless of what the ODE numerics actually indicated —
    see plans/03-clpc-token-conditional-guidance.md's dated entry on this.
    The root issue: ode_err is a smoothly-evolving numerical-integration
    quantity the Kalman filter's continuous-uncertainty assumptions actually
    fit; token/CLIP-derived diagnostics (vanish/leak/intention-drift) can
    shift abruptly step-to-step as cross-attention patterns reorganize, with
    no reason to behave like a slowly-decaying confidence estimate. Rather
    than force it into a "confidence" role it doesn't have, `token_err` is
    now reported as its own independent, un-blended, un-smoothed channel —
    recomputed fresh every step straight from that step's diagnostics,
    with no persistent state carried across steps at all.
    """
    P = ode_err
    K = P / (P + wav_hf_err + 1e-8)
    K = min(max(K, 0.0), 1.0)
    if token_err > 1e-6:
        print(f"[TokenSubspaceGuidance] token/CLIP error channel (separate from Kalman blend): "
              f"token_err={token_err:.4f} (measured fresh this step, not fed into P/K) "
              f"| kalman: ode_err={ode_err:.4f} wav_hf_err={wav_hf_err:.4f} -> P={P:.4f} K={K:.4f}")
    return x_pred.float().lerp(x_corr.float(), K).to(dtype=x_pred.dtype)


# ── history ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CLPCHistoryEntry:
    x: torch.Tensor        # latent at this step
    denoised: torch.Tensor # model denoised output (x0 prediction)
    sigma: float
    lam: float             # lambda coordinate
    d: torch.Tensor        # score/velocity used (denoised for EPS-based Adams)
    error: CLPCError


class CLPCHistory:
    """Fixed-capacity ring buffer for predictor history."""

    def __init__(self, capacity: int = 5):
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, entry: CLPCHistoryEntry) -> None:
        self._buf.append(entry)

    def __len__(self) -> int:
        return len(self._buf)

    def last(self, n: int) -> List[CLPCHistoryEntry]:
        items = list(self._buf)
        return items[-n:] if n <= len(items) else items

    def latest(self) -> Optional[CLPCHistoryEntry]:
        return self._buf[-1] if self._buf else None


# ── Live model_options access (bypasses the disconnected extra_args copy) ─────

def _get_live_transformer_options(model) -> dict:
    """Read transformer_options directly off the ACTUAL ModelPatcher driving
    inference (forge_objects.unet.model_options).

    `model` here is `self.model_wrap_cfg` (A1111's CFGDenoiser) wrapping the
    real model, mirroring the exact attribute path
    `modules_forge/forge_sampler.py::forge_sample` and
    `modules/sd_samplers_cfg_denoiser.py::CFGDenoiser.forward` use to reach
    the live model_options for the actual per-step forward pass — the same
    object extensions like sure_token_guidance.py patch via
    `ModelPatcher.set_model_attn2_replace`. `extra_args["model_options"]`
    (what `_clpc_loop` builds locally for its own CFG-drift hook) is a
    SEPARATE dict starting from `{}` and never reaches that live object under
    the diff_pipeline forward path, so anything installed by another
    extension directly on the model (like the token-guidance diagnostic
    store) can only be found here, not via extra_args.
    """
    try:
        unet = model.inner_model.inner_model.forge_objects.unet
        return unet.model_options.get("transformer_options", {})
    except AttributeError:
        return {}


# ── CFG drift hook ────────────────────────────────────────────────────────────

def _install_cfg_drift_hook(model_options: dict) -> dict:
    """
    Installs a post_cfg hook that captures CFG guidance direction and computes
    cosine drift between consecutive steps.  Returns updated model_options.
    State is returned via the shared cfg_drift_state dict (mutated in-place).
    """
    cfg_drift_state: dict = {}

    def post_cfg_fn(args):
        cond_d = args.get("cond_denoised")
        uncond_d = args.get("uncond_denoised")
        if cond_d is None or uncond_d is None:
            return args["denoised"]
        g = (cond_d - uncond_d).float()
        g_norm = g / (g.norm() + 1e-8)
        prev = cfg_drift_state.get("prev_g_norm")
        if prev is not None:
            sim = F.cosine_similarity(
                g_norm.flatten().unsqueeze(0),
                prev.flatten().unsqueeze(0),
            ).item()
            cfg_drift_state["last_cfg_drift"] = 1.0 - sim
        cfg_drift_state["prev_g_norm"] = g_norm.detach()
        return args["denoised"]

    model_options = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_fn, disable_cfg1_optimization=True,
    )
    return model_options, cfg_drift_state


# ── Adams predictor ───────────────────────────────────────────────────────────

def _adams_predict(
    x: torch.Tensor,
    history: CLPCHistory,
    sigma_s: torch.Tensor,
    sigma_t: torch.Tensor,
    lam_s: torch.Tensor,
    lam_t: torch.Tensor,
    tau_t: float,
    cal_state: CLPCCalibrationState,
    is_flow: bool,
    predictor_mode: str = PREDICTOR_IMPLICIT_ADAMS,
    use_chebyshev: bool = True,
    max_order: int = 3,
) -> torch.Tensor:
    """
    SA-Solver predictor, two modes (formally classified in AdamsStability.lean):

    PREDICTOR_EULER ("euler")
        Always order-1.  b = 1 - σ_t/σ_s ∈ (0, 1) unconditionally.
        Proved in RungeGuard.lean: euler_b_coeff_in_unit_interval.
        Single Lagrange node → no polynomial extrapolation possible.

    PREDICTOR_IMPLICIT_ADAMS ("implicit_adams")  [default]
        AM-style predictor: lam_t is appended as the rightmost node and
        history.latest().d (denoised_s) is used as its proxy function value.
        Integration [lam_s, lam_t] now lies *inside* the node convex hull
        → Lagrange interpolation, not extrapolation → bounded coefficients.
        Proved in AdamsStability.lean: am_order2_nonneg_sum_bounded (order 2),
        am_order3_basis_sum_one (order 3, partition of unity).
        The MAX_H_RATIO guard from the explicit AB predictor is not needed here.

    Phase 4 constraint: all Vandermonde arithmetic on CPU float64.
    b_coeffs moved to x.device / x.dtype only for the final tensordot.
    """
    h = float(lam_t - lam_s)
    decay = math.exp(-(tau_t ** 2) * h)
    order = min(len(history), max_order)

    # ── order-0 guard (shouldn't reach here after seed-push restructure) ────
    if order == 0:
        denoised = history.latest().d if history.latest() else x
        lam_cpu = torch.tensor([float(lam_s)], dtype=torch.float64)
        b0 = _sa.compute_stochastic_adams_b_coeffs(
            sigma_t.double().cpu(), lam_cpu,
            lam_s.double().cpu(), lam_t.double().cpu(),
            tau_t, is_corrector_step=False,
        ).to(device=x.device, dtype=x.dtype)
        return float(sigma_t / sigma_s) * decay * x + b0[0] * denoised.to(dtype=x.dtype)

    # ── Euler predictor ──────────────────────────────────────────────────────
    if predictor_mode == PREDICTOR_EULER:
        # Single node at lam_s; L₀ ≡ 1; b = ∫exp(λ-lam_t)dλ ∈ (0,1).
        entries = history.last(1)
        lam_cpu = torch.tensor([e.lam for e in entries], dtype=torch.float64)
        b_coeffs = _sa.compute_stochastic_adams_b_coeffs(
            sigma_t.double().cpu(), lam_cpu,
            lam_s.double().cpu(), lam_t.double().cpu(),
            tau_t, is_corrector_step=False,
        )
        if _DEBUG:
            bv = [f"{float(v):.4f}" for v in b_coeffs]
            print(f"  [pred b={bv}  sum={float(b_coeffs.sum()):.4f}"
                  f"  lams={[f'{float(v):.3f}' for v in lam_cpu]}  mode=euler]")
        b_coeffs = b_coeffs.to(device=x.device, dtype=x.dtype)
        d_stack = torch.stack([e.d for e in entries], dim=1)
        pred_res = torch.tensordot(d_stack, b_coeffs, dims=([1], [0]))
        return float(sigma_t / sigma_s) * decay * x + pred_res

    # ── Implicit Adams predictor ─────────────────────────────────────────────
    # Append lam_t as rightmost node with denoised_s (from history.latest()) as
    # proxy.  The Lagrange polynomial now spans [history_lams..., lam_s, lam_t],
    # so the integration domain [lam_s, lam_t] lies inside the node support.
    # (AdamsStability.lean: am_order2_nonneg_sum_bounded, am_order3_basis_sum_one)
    #
    # When use_chebyshev=True, select history entries whose λ positions best
    # approximate Chebyshev nodes on [λ_oldest, lam_s] instead of naively
    # taking the most-recent N entries (which cluster near lam_s and reproduce
    # the equally-spaced Runge risk).  ChebyshevAdaptive.lean: adams_vs_chebyshev.
    # CPU float64 throughout — never on GPU.
    if use_chebyshev and order >= 2 and len(history) > order:
        entries = _select_chebyshev_history(history, order, float(lam_s))
    else:
        entries = history.last(order)
    latest = history.latest()  # entry at lam_s; latest.d = denoised_s

    lambdas_list = [e.lam for e in entries] + [float(lam_t)]
    all_d = [e.d for e in entries] + [latest.d]  # denoised_s as proxy at lam_t

    lambdas_cpu = torch.tensor(lambdas_list, dtype=torch.float64)
    d_stack = torch.stack(all_d, dim=1)

    b_coeffs = _sa.compute_stochastic_adams_b_coeffs(
        sigma_t.double().cpu(), lambdas_cpu,
        lam_s.double().cpu(), lam_t.double().cpu(),
        tau_t, is_corrector_step=False,
    )
    if _DEBUG:
        bv = [f"{float(v):.4f}" for v in b_coeffs]
        print(f"  [pred b={bv}  sum={float(b_coeffs.sum()):.4f}"
              f"  lams={[f'{float(v):.3f}' for v in lambdas_cpu]}  mode=implicit_adams]")
    b_coeffs = b_coeffs.to(device=x.device, dtype=x.dtype)
    pred_res = torch.tensordot(d_stack, b_coeffs, dims=([1], [0]))
    return float(sigma_t / sigma_s) * decay * x + pred_res


# ── Adams corrector ───────────────────────────────────────────────────────────

def _adams_correct(
    x_pred: torch.Tensor,
    denoised_pred: torch.Tensor,
    x_orig: torch.Tensor,
    history: CLPCHistory,
    sigma_prev: torch.Tensor,
    sigma_curr: torch.Tensor,
    lam_prev: torch.Tensor,
    lam_curr: torch.Tensor,
    tau_t: float,
    cal_state: CLPCCalibrationState,
    pece_sigma_threshold: float,
    max_order: int = 3,
    use_chebyshev: bool = True,
) -> torch.Tensor:
    """
    Adams-Moulton corrector.  PECE gated by sigma (proved in PECEOrderGain.lean:
    corrector order gain is only beneficial when h ≤ ε * 2C2/C3).
    """
    sigma_val = float(sigma_curr)
    if sigma_val > pece_sigma_threshold:
        # High-noise regime: corrector error amplification risk outweighs gain
        return x_pred

    order = min(len(history) + 1, max_order + 1)  # corrector can use one extra point
    entries = history.last(order - 1)

    # Same Runge-phenomenon guard as _adams_predict: when h_curr >> h_prev,
    # Lagrange oscillation produces large canceling coefficients in the corrector
    # too (step-28: b=['1.61','-3.46','2.33','0.37']).  Fall back to order-2
    # (trapezoid: one history point + current) so lam_curr stays within the
    # interpolation support of the retained nodes.
    MAX_H_RATIO = 2.5
    if len(entries) >= 2:
        h_curr = float(abs(lam_curr - lam_prev))
        h_prev_step = abs(entries[-1].lam - entries[-2].lam)
        if h_prev_step > 0 and h_curr > MAX_H_RATIO * h_prev_step:
            order = 2
            entries = entries[-1:]  # keep only the most recent history entry

    # Chebyshev node selection, extended to the corrector for order > 3.
    # am_b_coeffs_sum_to_one_general (VariableOrderGain.lean) proves the AM
    # b-coefficients sum to 1 at ANY order — but that only bounds their SUM,
    # not any individual coefficient. The one order-general INDIVIDUAL bound
    # available (chebyshev_monic_minimax, ChebyshevAdaptive.lean) requires
    # Chebyshev-spaced nodes; recency-spaced nodes have no such guarantee
    # beyond order 3 (variable_order_needs_chebyshev_beyond_three). Raising
    # max_order past 3 without this would reproduce, in the corrector, the
    # same equally-spaced Runge risk _select_chebyshev_history already
    # patches in the predictor.
    if use_chebyshev and order - 1 >= 3 and len(history) > order - 1:
        entries = _select_chebyshev_history(history, order - 1, float(lam_prev))

    # Append current predicted d
    d_curr = denoised_pred
    all_d = [e.d for e in entries] + [d_curr]
    lambdas_list = [e.lam for e in entries] + [float(lam_curr)]
    # CPU float64 — Vandermonde solve stays off GPU on FP32-only hardware
    lambdas_cpu = torch.tensor(lambdas_list, dtype=torch.float64)
    d_stack = torch.stack(all_d, dim=1)

    b_coeffs = _sa.compute_stochastic_adams_b_coeffs(
        sigma_curr.double().cpu(), lambdas_cpu,
        lam_prev.double().cpu(), lam_curr.double().cpu(),
        tau_t, is_corrector_step=True,
    )
    if _DEBUG:
        bv = [f"{float(v):.4f}" for v in b_coeffs]
        print(f"  [corr b={bv}  sum={float(b_coeffs.sum()):.4f}"
              f"  lams={[f'{float(v):.3f}' for v in lambdas_cpu]}]")
    # DC-Solver calibration disabled (see _adams_predict for rationale)
    # b_coeffs = get_corrector_coeffs(cal_state, b_coeffs)
    b_coeffs = b_coeffs.to(device=x_pred.device, dtype=x_pred.dtype)

    h = float(lam_curr - lam_prev)
    decay = math.exp(-(tau_t ** 2) * h)
    corr_res = torch.tensordot(d_stack, b_coeffs, dims=([1], [0]))
    x_corr = float(sigma_curr / sigma_prev) * decay * x_orig + corr_res
    if _DEBUG:
        print(f"  [|x_orig|={float(x_orig.norm()):.3f}"
              f"  |x_pred|={float(x_pred.norm()):.3f}"
              f"  |x_corr|={float(x_corr.norm()):.3f}"
              f"  |x_corr-x_pred|={float((x_corr-x_pred).norm()):.4f}]")
    return x_corr


# ── SDE noise injection ───────────────────────────────────────────────────────

def _sde_noise(
    sigma_next: torch.Tensor,
    sigma_curr: torch.Tensor,
    lam_curr: torch.Tensor,
    lam_next: torch.Tensor,
    tau_t: float,
    x_shape,
    device,
    dtype,
    s_noise: float,
    adaptive_noise_scale: float = 1.0,
) -> torch.Tensor:
    h = float(lam_next - lam_curr)
    expm1_val = math.expm1(2.0 * (tau_t ** 2) * abs(h))
    noise_var = max(expm1_val, 0.0)
    noise_scale = float(sigma_next) * math.sqrt(noise_var) * s_noise * adaptive_noise_scale
    if noise_scale < 1e-9:
        return torch.zeros(x_shape, device=device, dtype=dtype)
    return torch.randn(x_shape, device=device, dtype=dtype) * noise_scale


# ── core loop (shared ODE/SDE) ────────────────────────────────────────────────

def _clpc_loop(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict,
    callback,
    disable: bool,
    weights: CLPCWeights,
    # predictor
    predictor_mode: str,
    # Lean-proved enhancements
    use_chebyshev: bool,     # Priority 1: Chebyshev node selection
    use_kalman: bool,        # Priority 3: Kalman-gain blend
    # Priority 4: configurable maximum order (UniPC-style), Chebyshev-gated
    # for order > 3 — see VariableOrderGain.lean
    max_order: int,
    lower_order_final: bool,
    # step-size control
    atol: float,
    rtol: float,
    pid_pcoeff: float,
    pid_icoeff: float,
    pid_dcoeff: float,
    max_steps: int,
    # PECE
    pece_sigma_threshold: float,
    # SDE
    tau_eta: float,          # 0.0 → ODE
    s_noise: float,
    # adaptive noise
    adaptive_noise: bool,
    # Token-space conditional guidance corrector feedback (TokenSubspaceGuidance.lean Part 7)
    token_kalman_weight: float = TOKEN_KALMAN_WEIGHT,
) -> torch.Tensor:
    """Shared ODE/SDE loop."""
    from ldm_patched.k_diffusion.sampling import PIDStepSizeController
    from tqdm.auto import tqdm

    is_flow = _detect_flow_model(model)
    sigmas = _offset_sigma_for_flow(sigmas, is_flow)

    # Install CFG drift hook
    model_options = extra_args.get("model_options", {})
    model_options, cfg_drift_state = _install_cfg_drift_hook(model_options)
    extra_args = {**extra_args, "model_options": model_options}

    # Token-subspace guidance diagnostics (sure_token_guidance.py), if the
    # extension patched attn2 hooks onto this model. None when disabled —
    # build_clpc_error() then uses the neutral token_score=1.0 default.
    #
    # IMPORTANT: read this from the LIVE model (forge_objects.unet.model_options),
    # NOT from `model_options` above. `extra_args` (== sd_samplers_kdiffusion.py's
    # `sampler_extra_args`) never contains a "model_options" key in the first
    # place — the dict built two lines up starts from `{}` and only ever holds
    # the CFG-drift post_cfg hook CLPC installs on itself. The real per-step
    # forward pass (forge_sampler.forge_sample / sd_samplers_cfg_denoiser) reads
    # model_options straight off `forge_objects.unet` — the same object
    # sure_token_guidance.patch_model_with_token_guidance() patches — so that's
    # the only place attn2's token_guidance_store can actually be found.
    token_guidance_store = _get_live_transformer_options(model).get("token_guidance_store")
    if token_guidance_store is not None:
        print(f"[TokenSubspaceGuidance] clpc_sampler: ENABLED — found token_guidance_store on "
              f"this model (len={len(token_guidance_store)} entries so far). P_token feeds the "
              f"G-score every step via a SEPARATE, un-Kalman-blended channel weighted by "
              f"token_kalman_weight={token_kalman_weight} (informational/monitoring only — see "
              f"_kalman_blend's docstring for why this is deliberately NOT folded into the "
              f"corrector-trust gain).")
    else:
        print(f"[TokenSubspaceGuidance] clpc_sampler: INACTIVE — no token_guidance_store found on "
              f"this model's transformer_options. Enable the 'SURE Token Subspace Guidance' "
              f"extension (separate accordion) to activate the conditional-space G-score channel; "
              f"this run uses the neutral default (P_token=1.0, token_kalman_weight has no effect).")

    # Convert sigma schedule to lambda
    lambdas = _sigma_to_lambda(sigmas, is_flow)

    # Build tau_func
    if tau_eta > 0:
        tau_func = _sa.get_tau_interval_func(float(sigmas[0]), float(sigmas[-2]), eta=tau_eta)
    else:
        tau_func = lambda s: 0.0  # noqa: E731

    # Calibration state
    cal_state = CLPCCalibrationState()
    history = CLPCHistory(capacity=5)

    # PID controller — initial h = first lambda step
    h_init = float((lambdas[1] - lambdas[0]).abs())
    pid = PIDStepSizeController(
        h_init, pid_pcoeff, pid_icoeff, pid_dcoeff, order=3, accept_safety=0.81,
    )

    weights = weights.normalise()

    # Track adaptive noise scale (ratchet up if ODE error persistently > OT error)
    adaptive_noise_scale = 1.0
    ode_err_acc = 0.0
    ot_err_acc = 0.0
    acc_count = 0

    # G-score Lyapunov tracking (GaussianProcessODE.lean, ProxySOCvsFull.lean)
    # kl_proxy = -log F; should decrease monotonically toward 0 at target.
    prev_gscore = 0.0
    prev_kl_proxy = float("inf")

    step_count = 0
    accepted_count = 0
    n = len(sigmas) - 1
    i = 0  # index into sigmas schedule

    sigma_start = float(sigmas[0])
    sigma_end = float(sigmas[-1])
    sigma_range = max(sigma_start - sigma_end, 1e-8)

    def _sigma_pct(sigma_val: float) -> float:
        """Fraction of sigma range covered so far, in [0, 100]."""
        return min(max((sigma_start - sigma_val) / sigma_range * 100.0, 0.0), 100.0)

    # Single progress bar: sigma-range coverage in 0.1% units.
    # Stage label is embedded in the description so it updates live during
    # blocking model calls — no separate inner bar that appears frozen.
    _PCT_SCALE = 1000
    pbar_outer = tqdm(
        total=_PCT_SCALE,
        desc="CLPC  0.0%  σ={:.3f}  [init]".format(sigma_start),
        unit="%",
        unit_scale=1.0 / 10,   # display as x.x%
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]",
        disable=disable,
        position=0,
        leave=True,
    )

    pct_reported = 0  # tracks how many 0.1% units we've already added to outer bar

    def _advance_outer(sigma_val: float, stage: str = "") -> None:
        nonlocal pct_reported
        pct_units = int(_sigma_pct(sigma_val) * 10)  # 0–1000
        delta = max(pct_units - pct_reported, 0)
        if delta:
            pbar_outer.update(delta)
            pct_reported = pct_units
        pct_display = pct_units / 10
        stage_tag = f"  [{stage}]" if stage else ""
        pbar_outer.set_description(
            "CLPC {:5.1f}%  σ={:.3f}  steps={}/{}  E={:.3f}  F={:.3f}{}".format(
                pct_display, sigma_val, accepted_count, step_count,
                getattr(_advance_outer, "_last_err", 0.0),
                getattr(_advance_outer, "_last_gscore", 0.0),
                stage_tag,
            )
        )

    def _stage(name: str) -> None:
        # Update outer bar description immediately so the label is visible
        # during the blocking model call that follows — no inner bar needed.
        _advance_outer(float(sigmas[i]) if i < len(sigmas) else 0.0, stage=name)

    try:
        while i < n and step_count < max_steps:
            sigma_s = sigmas[i]
            sigma_t = sigmas[i + 1]
            lam_s = lambdas[i]
            lam_t = lambdas[i + 1]
            h = float((lam_t - lam_s).abs())

            if float(sigma_t) == 0:
                # Final denoising step
                _stage("Denoise")
                denoised = model(x, sigma_s * x.new_ones([x.shape[0]]), **extra_args)
                x = denoised
                _advance_outer(0.0)
                # WebUI callback: report i scaled to n so bar reaches 100%
                if callback is not None:
                    callback({"x": x, "i": n - 1, "sigma": sigma_s, "denoised": denoised})
                break

            tau_t = tau_func(float(sigma_s))

            # Order ramp-down near the end of the schedule (UniPC's
            # lower_order_final): with `remaining` ODE steps left (including
            # this one), a predictor/corrector order above `remaining` would
            # extrapolate past the end of the trajectory it's fitting.
            # remaining = n - i counts steps [i, n) still to take, INCLUDING
            # the current one — so remaining=1 on the last real step forces
            # order down to 1 (Euler/trapezoid), matching UniPC's own
            # `step_order = min(order, steps + 1 - step)`.
            remaining = n - i
            step_max_order = min(max_order, remaining) if lower_order_final else max_order

            # ── PREDICT ──────────────────────────────────────────────────────
            _stage("Predict")
            prev_entry = history.latest()
            decay = math.exp(-(tau_t ** 2) * h)

            if len(history) == 0:
                # Seed step: evaluate at current point and push BEFORE predicting so
                # _adams_predict can use the correct order-1 Adams coefficient.
                # (The previous bug used expm1(h_signed) = sigma_s/sigma_t - 1, which
                # diverges as sigma_t→0.  The correct order-1 coefficient is 1-sigma_t/sigma_s,
                # computed by compute_stochastic_adams_b_coeffs with a single history entry.)
                denoised_s = model(x, sigma_s * x.new_ones([x.shape[0]]), **extra_args)
                history.push(CLPCHistoryEntry(
                    x=x.detach(), denoised=denoised_s.detach(),
                    sigma=float(sigma_s), lam=float(lam_s),
                    d=denoised_s.detach(), error=CLPCError(),
                ))
                prev_entry = history.latest()
                x_pred = _adams_predict(
                    x, history, sigma_s, sigma_t, lam_s, lam_t, tau_t,
                    cal_state, is_flow, predictor_mode, use_chebyshev, step_max_order,
                )
                x_euler = None  # no prior embedded-pair baseline on seed step

            else:
                x_pred = _adams_predict(
                    x, history, sigma_s, sigma_t, lam_s, lam_t, tau_t,
                    cal_state, is_flow, predictor_mode, use_chebyshev, step_max_order,
                )
                # Euler baseline for embedded pair (zero extra NFE — reuses prev d).
                # Use SA-Solver order-1 b_coeff (CPU float64) — NOT expm1(h_signed).
                # expm1(log(sigma_s/sigma_t)) = sigma_s/sigma_t - 1 which diverges
                # as sigma_t→0.  The correct coefficient is 1 - sigma_t/sigma_s,
                # which is what compute_stochastic_adams_b_coeffs returns for order=1.
                euler_lam_cpu = torch.tensor([float(lam_s)], dtype=torch.float64)
                euler_b = _sa.compute_stochastic_adams_b_coeffs(
                    sigma_t.double().cpu(), euler_lam_cpu,
                    lam_s.double().cpu(), lam_t.double().cpu(),
                    tau_t, is_corrector_step=False,
                ).to(device=x.device, dtype=x.dtype)
                x_euler = float(sigma_t / sigma_s) * decay * x + euler_b[0] * prev_entry.d.to(dtype=x.dtype)

            # ── EVALUATE + CORRECT (PECE) ─────────────────────────────────
            # SDE noise is NOT added here. Adding it before the model call
            # creates a sigma mismatch: x_pred + noise has effective noise level
            # sqrt(σ_t² + σ_s² − σ_t²) = σ_s, but the model is labelled σ_t.
            # The degraded denoised_pred propagates into the corrector via corr_res.
            # Noise is injected after the Kalman blend instead (see below).
            _stage("Correct")
            denoised_pred = model(x_pred, sigma_t * x_pred.new_ones([x_pred.shape[0]]), **extra_args)
            x_corr = _adams_correct(
                x_pred, denoised_pred, x, history,
                sigma_s, sigma_t, lam_s, lam_t,
                tau_t, cal_state, pece_sigma_threshold,
                step_max_order, use_chebyshev,
            )

            # PECE final E: re-evaluate at x_corr so Adams history stays consistent
            # with the corrected trajectory.  Without this, the next predictor uses
            # f(x_pred) while the trajectory advanced through x_corr — the Jacobian
            # of the denoiser is channel-specific so the mismatch shows as color noise.
            # Proved necessary in EmbeddedPairError.lean (history_inconsistency_accumulates).
            # Only fires when the corrector actually changed x (sigma ≤ threshold).
            pece_active = float(sigma_t) <= pece_sigma_threshold and len(history) > 0
            if pece_active:
                denoised_for_history = model(
                    x_corr, sigma_t * x_corr.new_ones([x_corr.shape[0]]), **extra_args
                )
            else:
                denoised_for_history = denoised_pred

            # ── COMPUTE ERROR ─────────────────────────────────────────────
            _stage("Error  ")
            denoised_prev = prev_entry.denoised if prev_entry else None
            sigma_prev_val = prev_entry.sigma if prev_entry else float(sigma_s)

            token_info = None
            if token_guidance_store is not None:
                from ldm_patched.k_diffusion.sure_token_guidance import aggregate_token_guidance_info
                token_info = aggregate_token_guidance_info(token_guidance_store)
                token_guidance_store.clear()  # reset for the next step's accumulation

            err = build_clpc_error(
                x_adams=x_pred,
                x_euler=x_euler,
                x_corr=x_corr,          # Priority 2: Haar subband error of innovation
                denoised_t=denoised_for_history,
                denoised_prev=denoised_prev,
                sigma_t=float(sigma_t),
                sigma_prev=sigma_prev_val,
                h=h,
                is_flow_model=is_flow,
                sure_info=None,
                cfg_drift_state=cfg_drift_state,
                atol=atol,
                rtol=rtol,
                token_info=token_info,
            )
            composite = err.composite(weights)
            _advance_outer._last_err = composite       # type: ignore[attr-defined]
            _advance_outer._last_gscore = err.gscore   # type: ignore[attr-defined]

            # ── KALMAN BLEND (Priority 3) ─────────────────────────────────
            # K = ode_err / (ode_err + wav_hf_err). When prediction is
            # numerically uncertain (large ODE gap) → trust corrector. When
            # the correction is high-frequency noise (large wav_hf) → trust
            # predictor. A=(σ_t/σ_s)·I proved contractive; two-term K proved
            # MMSE-optimal in KalmanMatrixDesign.lean + KalmanFilter.lean.
            #
            # token_err (1 - P_token) is DELIBERATELY NOT part of this blend
            # — see _kalman_blend's docstring for why folding it into P
            # caused K to stay pinned near 1.0 for an entire live run. It's
            # measured fresh every step and passed through only for the
            # print statement (a separate, un-smoothed diagnostic channel);
            # `token_kalman_weight` now scales its (also purely
            # informational) contribution to the composite G-score display
            # via `weights.token`, not the corrector-trust gain.
            token_err = max(0.0, 1.0 - err.token_score)
            if use_kalman:
                x_out = _kalman_blend(x_pred, x_corr, err.ode, err.wav_hf, token_err)
            else:
                x_out = x_corr

            # ── SDE NOISE (after correction) ──────────────────────────────
            # Injected here so the model always sees clean x_pred at the correct
            # σ_t label. The noisy x_out is stored in history and becomes the
            # next step's starting x, maintaining the stochastic trajectory.
            #
            # Skip the penultimate step (the one just before sigma_t = 0).
            # At that step h = log(σ_{n-2}/σ_{n-1}) is large, making
            # noise_scale = σ_t * sqrt(expm1(2h)) ≈ σ_{n-2} ≫ σ_{n-1}.
            # Adding it means the final model call sees ~10× its sigma label
            # in actual noise, which can't be removed → grain.
            _is_penultimate = (i + 2 < len(sigmas) and float(sigmas[i + 2]) == 0.0)
            if tau_t > 0 and not _is_penultimate:
                x_out = x_out + _sde_noise(
                    sigma_t, sigma_s, lam_s, lam_t, tau_t,
                    x_out.shape, x_out.device, x_out.dtype,
                    s_noise, adaptive_noise_scale,
                )

            # ── PID MONITOR + ACCEPT ─────────────────────────────────────
            _stage("PID    ")
            pid.propose_step(composite)   # monitoring only

            if history.latest() is not None:
                # Calibration accumulation kept for future redesign; blend disabled
                accumulate_calibration(
                    cal_state,
                    lambdas=torch.tensor(
                        [e.lam for e in history.last(4)],
                        dtype=torch.float64,  # CPU (no device= arg)
                    ),
                    lambda_s=lam_s,
                    lambda_t=lam_t,
                    update_target=x_corr - x,
                )

            if _DEBUG:
                x_euler_norm = float(x_euler.norm()) if x_euler is not None else float("nan")
                x_pred_norm = float(x_pred.norm())
                euler_gap = float((x_pred - x_euler).norm()) if x_euler is not None else float("nan")
                K_val = err.ode / (err.ode + err.wav_hf + 1e-8) if use_kalman else 1.0
                kl_delta = err.kl_proxy - prev_kl_proxy
                print(
                    f"[CLPC] step={i:3d}  σ {float(sigma_s):.4f}→{float(sigma_t):.4f}"
                    f"  h={h:.3f}  ord={min(len(history),3)}"
                    f"  ode={err.ode:.3f}  ot={err.ot:.3f}  cfg={err.cfg:.3f}"
                    f"  wav_ll={err.wav_ll:.3f}  wav_hf={err.wav_hf:.3f}  K={K_val:.3f}"
                    f"  E={composite:.3f}"
                    f"  F={err.gscore:.4f}  KL={err.kl_proxy:.3f}(Δ{kl_delta:+.3f})"
                    f"  |x_pred|={x_pred_norm:.3f}  |x_euler|={x_euler_norm:.3f}"
                    f"  gap={euler_gap:.4f}"
                    f"  pece={pece_active}"
                    f"  cal_blend={cal_state.blend:.2f}"
                )

            if adaptive_noise and tau_t > 0:
                ode_err_acc += err.ode
                ot_err_acc += err.ot
                acc_count += 1
                if acc_count >= 5:
                    ratio = ode_err_acc / (ot_err_acc + 1e-8)
                    if ratio > 3.0:
                        adaptive_noise_scale = min(adaptive_noise_scale * 1.1, 2.0)
                    elif ratio < 0.5:
                        adaptive_noise_scale = max(adaptive_noise_scale * 0.95, 0.5)
                    ode_err_acc = ot_err_acc = acc_count = 0

            history.push(CLPCHistoryEntry(
                x=x_out.detach(), denoised=denoised_for_history.detach(),
                sigma=float(sigma_t), lam=float(lam_t),
                d=denoised_for_history.detach(),
                error=err,
            ))

            # Update Lyapunov tracking — kl_proxy should decrease toward 0
            prev_gscore = err.gscore
            prev_kl_proxy = err.kl_proxy

            x = x_out
            accepted_count += 1
            step_count += 1
            i += 1

            _advance_outer(float(sigma_t))

            if callback is not None:
                webui_i = int(_sigma_pct(float(sigma_t)) / 100.0 * n)
                callback({"x": x, "i": webui_i, "sigma": sigma_s,
                          "denoised": denoised_pred, "error": err})

    finally:
        # Fill outer bar to 100% if we finished normally
        delta = _PCT_SCALE - pct_reported
        if delta > 0:
            pbar_outer.update(delta)
        pbar_outer.close()

    return x


# ── public entry points ───────────────────────────────────────────────────────

@torch.no_grad()
def sample_clpc_ode(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[dict] = None,
    callback=None,
    disable: bool = False,
    # predictor mode
    predictor: str = PREDICTOR_IMPLICIT_ADAMS,
    # Lean-proved enhancements
    use_chebyshev: bool = True,   # Priority 1: Chebyshev node selection
    use_kalman: bool = True,      # Priority 3: Kalman-gain blend
    # Priority 4: configurable maximum order (UniPC-style; VariableOrderGain.lean)
    max_order: int = 3,
    lower_order_final: bool = True,
    # error weights
    w_ode: float = 1.0,
    w_ot: float = 0.5,
    w_sure: float = 0.0,
    w_cfg: float = 0.3,
    # adaptive stepping
    atol: float = 1e-4,
    rtol: float = 1e-3,
    pid_pcoeff: float = 0.4,
    pid_icoeff: float = 0.3,
    pid_dcoeff: float = 0.0,
    max_steps: int = 1000,
    # PECE
    pece_sigma_threshold: float = 4.0,
    # Token-space conditional guidance corrector feedback (TokenSubspaceGuidance.lean Part 7).
    # Only has an effect if the "SURE Token Subspace Guidance" extension has also
    # patched attn2 hooks onto this model — see the ENABLED/INACTIVE log line
    # _clpc_loop prints at the start of every run.
    token_kalman_weight: float = TOKEN_KALMAN_WEIGHT,
) -> torch.Tensor:
    """CLPC deterministic (ODE) sampler.

    predictor:     "euler" or "implicit_adams" (AdamsStability.lean).
    use_chebyshev: Chebyshev node selection for history points (ChebyshevAdaptive.lean).
                   REQUIRED for soundness when max_order > 3 — the only
                   order-general individual-coefficient bound available
                   (chebyshev_monic_minimax) needs Chebyshev spacing;
                   partition of unity alone (am_b_coeffs_sum_to_one_general)
                   bounds only the coefficient SUM (VariableOrderGain.lean).
    use_kalman:    Kalman-gain blend of pred/corr outputs (KalmanMatrixDesign.lean).
    max_order:     Maximum predictor order (corrector gets one extra point,
                   i.e. max_order+1 nodes), analogous to UniPC's `order`.
                   order_gain_ratio_tendsto_zero (VariableOrderGain.lean)
                   proves each +1 strictly reduces local truncation error as
                   h → 0; am_b_coeffs_sum_to_one_general proves the corrector
                   stays a consistent update at any order.
    lower_order_final: Ramp order down as the schedule nears its last step
                   (UniPC's `lower_order_final`), so a high order is never
                   asked to extrapolate past the trajectory's end.
    """
    return _clpc_loop(
        model=model, x=x, sigmas=sigmas,
        extra_args=extra_args or {},
        callback=callback, disable=disable,
        weights=CLPCWeights(ode=w_ode, ot=w_ot, sure=w_sure, cfg=w_cfg, token=token_kalman_weight),
        predictor_mode=predictor,
        use_chebyshev=use_chebyshev,
        use_kalman=use_kalman,
        max_order=max_order,
        lower_order_final=lower_order_final,
        atol=atol, rtol=rtol,
        pid_pcoeff=pid_pcoeff, pid_icoeff=pid_icoeff, pid_dcoeff=pid_dcoeff,
        max_steps=max_steps,
        pece_sigma_threshold=pece_sigma_threshold,
        tau_eta=0.0, s_noise=1.0,
        adaptive_noise=False,
        token_kalman_weight=token_kalman_weight,
    )


@torch.no_grad()
def sample_clpc_sde(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[dict] = None,
    callback=None,
    disable: bool = False,
    # predictor mode
    predictor: str = PREDICTOR_IMPLICIT_ADAMS,
    # Lean-proved enhancements
    use_chebyshev: bool = True,   # Priority 1: Chebyshev node selection
    use_kalman: bool = True,      # Priority 3: Kalman-gain blend
    # Priority 4: configurable maximum order (UniPC-style; VariableOrderGain.lean)
    max_order: int = 3,
    lower_order_final: bool = True,
    # error weights
    w_ode: float = 1.0,
    w_ot: float = 0.5,
    w_sure: float = 0.0,
    w_cfg: float = 0.3,
    # adaptive stepping
    atol: float = 1e-4,
    rtol: float = 1e-3,
    pid_pcoeff: float = 0.4,
    pid_icoeff: float = 0.3,
    pid_dcoeff: float = 0.0,
    max_steps: int = 1000,
    # PECE
    pece_sigma_threshold: float = 4.0,
    # SDE
    tau_eta: float = 1.0,
    s_noise: float = 1.0,
    adaptive_noise: bool = True,
    # accepted but unused — k-diffusion runner passes this to all SDE samplers
    noise_sampler=None,
    # Token-space conditional guidance corrector feedback (TokenSubspaceGuidance.lean Part 7).
    # Only has an effect if the "SURE Token Subspace Guidance" extension has also
    # patched attn2 hooks onto this model — see the ENABLED/INACTIVE log line
    # _clpc_loop prints at the start of every run.
    token_kalman_weight: float = TOKEN_KALMAN_WEIGHT,
) -> torch.Tensor:
    """CLPC stochastic (SDE) sampler with optional adaptive noise.

    predictor:     "euler" or "implicit_adams" (AdamsStability.lean).
    use_chebyshev: Chebyshev node selection for history points (ChebyshevAdaptive.lean).
                   REQUIRED for soundness when max_order > 3 — see
                   VariableOrderGain.lean (sample_clpc_ode's docstring has
                   the full citation).
    use_kalman:    Kalman-gain blend of pred/corr outputs (KalmanMatrixDesign.lean).
    max_order:     Maximum predictor order (corrector gets max_order+1 nodes),
                   analogous to UniPC's `order` (VariableOrderGain.lean).
    lower_order_final: Ramp order down near the schedule's last step (UniPC's
                   `lower_order_final`).
    """
    return _clpc_loop(
        model=model, x=x, sigmas=sigmas,
        extra_args=extra_args or {},
        callback=callback, disable=disable,
        weights=CLPCWeights(ode=w_ode, ot=w_ot, sure=w_sure, cfg=w_cfg, token=token_kalman_weight),
        predictor_mode=predictor,
        use_chebyshev=use_chebyshev,
        use_kalman=use_kalman,
        max_order=max_order,
        lower_order_final=lower_order_final,
        atol=atol, rtol=rtol,
        pid_pcoeff=pid_pcoeff, pid_icoeff=pid_icoeff, pid_dcoeff=pid_dcoeff,
        max_steps=max_steps,
        pece_sigma_threshold=pece_sigma_threshold,
        tau_eta=tau_eta, s_noise=s_noise,
        adaptive_noise=adaptive_noise,
        token_kalman_weight=token_kalman_weight,
    )
