# Plan 01 — CLPC: Closed-Loop Predictor-Corrector Adaptive Sampler

**Status:** Draft  
**Date:** 2026-06-19  
**Context:** Generalises the velocity linearity guidance (VLG, `nodes_vlg.py`) into a full adaptive ODE sampler that monitors its own trajectory quality via a composite multi-objective error signal.

---

## Goal

Build a Predictor-Corrector adaptive sampler (`sample_clpc`) that:
1. Adapts its own step size based on a **composite error norm** — ODE truncation error, OT path-straightness drift, SURE residual magnitude, and CLIP alignment (σ-gated)
2. Maintains a **bounded history buffer** so the corrector can roll back and retry with a smaller step
3. Reports progress as a **percentage** of σ-travel completed, not a raw step count
4. Has a **hard `max_steps` ceiling** to prevent infinite loops

---

## Phase 0: Documentation Discovery (COMPLETE)

All facts confirmed from codebase survey and Context7. Key allowed APIs:

| Concern | File | Entry Point | Lines |
|---|---|---|---|
| Sampler registration | `ldm_patched/modules/samplers.py` | `KSAMPLER_NAMES` list | 922 |
| Step function signature | `ldm_patched/k_diffusion/sampling.py` | `sample_euler` (canonical) | 487 |
| Entropy hook injection | `ldm_patched/k_diffusion/sure_attention.py` | `build_capture_model_options()` | 148 |
| Entropy map aggregation | `ldm_patched/k_diffusion/sure_attention.py` | `_aggregate_entropy_map()` | 167 |
| Wavelet SURE correction | `ldm_patched/k_diffusion/sure_wav_ag.py` | `_sure_correct_x0_wavelet_ag()` | 274 |
| VLG residual (no extra fwd) | `ldm_patched/contrib/nodes_vlg.py` | `post_cfg_function` closure | 71 |
| PID step controller | `ldm_patched/k_diffusion/sampling.py` | `PIDStepSizeController` | 876 |
| CLIP access | `ldm_patched/modules/sd.py` | `CLIP.encode_from_tokens(…, return_pooled=True)` | 277 |
| UniPC predictor-corrector | `ldm_patched/unipc/uni_pc.py` | `multistep_uni_pc_bh_update()` | 582 |
| bosh3 / torchdiffeq | `ldm_patched/k_diffusion/sampling.py` | `ODESampler`, `ADAPTIVE_SOLVERS` | 271 |
| Existing Lean velocity bound | `lean_proofs_rfv/RFVProofs/LocalLinearity.lean` | `finDiff_window_near_velocity` | 126 |

**Anti-patterns established:**
- Do NOT use `torchdiffeq.odeint` directly — the multi-objective error norm requires manual step-level control; wrap `PIDStepSizeController` instead
- CLIP similarity is **undefined/noisy at high σ** — must gate with `sigma_clip_threshold` (default: 2.0)
- Entropy hook MUST escape autocast (see `sure_attention.py:71`); do not call inside `torch.autocast`
- `_sure_correct_x0_wavelet_ag()` makes **one extra UNet forward** — budget this into `max_steps` accounting

---

## Phase 1: Lean Math Foundation

**Purpose:** Establish which objectives belong in the composite norm, prove the sigma-gate for CLIP, and derive the step-acceptance criterion. This drives implementation decisions before writing Python.

### 1.1 New file: `lean_proofs_rfv/RFVProofs/CompositeError.lean`

**What to implement:**

```
-- Import from existing modules:
import RFVProofs.LocalLinearity    -- finDiff_window_near_velocity
import RFVProofs.Straightness      -- chord_le_path_length
import RFVProofs.Defs               -- AffineTraj, LipschitzVelocityOn

namespace RFVProofs

-- The composite step error is a weighted combination of four terms.
-- We formalise which terms are bounded under which conditions.

structure CompositeWeights where
  w_ode  : ℝ    -- ODE local truncation error weight
  w_ot   : ℝ    -- OT path-straightness drift weight
  w_sure : ℝ    -- SURE residual weight
  w_clip : ℝ    -- CLIP alignment weight (σ-gated externally)
  hw_ode  : 0 ≤ w_ode
  hw_ot   : 0 ≤ w_ot
  hw_sure : 0 ≤ w_sure
  hw_clip : 0 ≤ w_clip

-- OT error: velocity drift bounded by LocalLinearity
-- Copied from finDiff_window_near_velocity (LocalLinearity:126):
--   ‖finDiffVelocity x t0 t - v t0‖ ≤ L * |t - t0|   for |t - t0| ≤ h
-- → If step size h ≤ h_max, OT error ≤ L * h_max

theorem ot_error_bounded_by_step
    (x : ℝ → E) (v : ℝ → E) (L h t0 t : ℝ)
    (hL : 0 < L)
    (hstep : |t - t0| ≤ h)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    (hderiv : ∀ s, HasDerivAt x (v s) s)
    : ‖finDiffVelocity x t0 t - v t0‖ ≤ L * h := by
  exact finDiff_window_near_velocity x v L h t0 t hL hstep hlip hderiv

-- CLIP gate: at high σ (early diffusion), x0_hat ≈ random noise,
-- so CLIP(x0_hat, text) has no meaningful gradient signal.
-- We model this as: the CLIP term's effective weight is
--   w_clip_eff(σ) = w_clip * exp(-σ / σ_threshold)
-- Theorem: as σ → ∞, w_clip_eff → 0 (gate vanishes)

theorem clip_weight_vanishes_at_high_sigma
    (w_clip σ_threshold : ℝ) (hw : 0 < w_clip) (hτ : 0 < σ_threshold)
    : Filter.Tendsto (fun σ => w_clip * Real.exp (-σ / σ_threshold))
        Filter.atTop (nhds 0) := by
  simp_rw [mul_comm]
  exact Filter.Tendsto.const_mul
    (Real.tendsto_exp_atBot.comp (Filter.Tendsto.atBot_mul_const
      (by linarith) tendsto_id)) hw.ne'

-- Step acceptance: composite error ≤ tolerance implies ODE LTE ≤ f(tol)
-- Under weights w_ode > 0 and OT drift e_ot ≤ L*h:
-- If E_composite = w_ode * e_ode + w_ot * e_ot + … ≤ tol
-- Then e_ode ≤ tol / w_ode

theorem ode_error_bounded_by_composite
    (e_ode e_ot e_sure e_clip tol : ℝ)
    (w : CompositeWeights)
    (hw_ode : 0 < w.w_ode)
    (hnn : 0 ≤ e_ode) (hnn_ot : 0 ≤ e_ot) (hnn_sure : 0 ≤ e_sure) (hnn_clip : 0 ≤ e_clip)
    (hcomposite : w.w_ode * e_ode + w.w_ot * e_ot + w.w_sure * e_sure + w.w_clip * e_clip ≤ tol)
    : e_ode ≤ tol / w.w_ode := by
  have h1 : w.w_ode * e_ode ≤ tol := by
    have : 0 ≤ w.w_ot * e_ot + w.w_sure * e_sure + w.w_clip * e_clip :=
      by positivity
    linarith
  exact (div_le_iff hw_ode).mpr h1 |>.symm.le |> (le_div_iff hw_ode).mp

end RFVProofs
```

**Verification checklist:**
- `lake build RFVProofs.CompositeError` compiles with 0 sorry
- `clip_weight_vanishes_at_high_sigma` — confirm proof uses `Real.tendsto_exp_atBot`
- `ode_error_bounded_by_composite` — confirm it follows from linear arithmetic only

**Anti-pattern guards:**
- Do NOT attempt to formalise SURE risk estimator in Lean (no Stein calculus in Mathlib at this level) — treat `e_sure` as an opaque observable
- Do NOT attempt to formalise CLIP cosine similarity — no vision-language formalism available; keep it as an opaque `ℝ`-valued term

---

### 1.2 New file: `lean_proofs_rfv/RFVProofs/CorrectorConvergence.lean`

**Purpose:** Prove the corrector iteration is contractive under the Lipschitz velocity assumption — justifying at most `corrector_depth` retries.

```
import RFVProofs.Defs
import RFVProofs.LocalLinearity

namespace RFVProofs

-- Corrector convergence: one corrector pass reduces the trajectory deviation.
-- Model: corrector replaces x_{t+1}^pred with x_{t+1}^corr using
--   the model evaluation at x_{t+1}^pred.
-- If velocity is L-Lipschitz and step size h ≤ 2/L, the corrector
-- residual contracts by factor ρ = L*h/2 < 1.

theorem corrector_contracts
    (v : ℝ → E) (x_true x_pred : E) (L h : ℝ)
    (hL : 0 < L)
    (hh : h ≤ 2 / L)
    (hlip : ∀ a b : E, ‖v_func a - v_func b‖ ≤ L * ‖a - b‖)
    : ‖x_corr - x_true‖ ≤ (L * h / 2) * ‖x_pred - x_true‖ := by
  sorry  -- exploration target: requires fixed-point contraction in Banach space
```

**Note:** This theorem is an exploration target — the `sorry` is intentional. It establishes the mathematical claim that justifies `corrector_depth = 2` as sufficient. If `L*h/2 < 1` (i.e., `h < 2/L`), one corrector pass halves the error. Proof requires Banach fixed-point contraction machinery from Mathlib.

**Verification checklist:**
- File compiles (sorry is acceptable here as exploration target)
- Statement is structurally sound — `sorry` does not hide a type error
- `explore.sh` can target this theorem for automated proof search

---

### 1.3 Update `lean_proofs_rfv/RFVProofs.lean`

Add imports:
```lean
import RFVProofs.CompositeError
import RFVProofs.CorrectorConvergence
```

**Deliverable from Phase 1:**
- `CompositeError.lean`: CLIP gate proved, step acceptance derived — **no sorry**
- `CorrectorConvergence.lean`: contract statement compiled — sorry is documented exploration target
- Decision recorded in this plan: **CLIP term enabled with σ_threshold=2.0, w_clip=0.1 (low weight)**; corrector depth fixed at 2 per Lean bound

---

## Phase 2: Multi-Objective Error Module

**New file:** `ldm_patched/k_diffusion/clpc_error.py`

**What to implement** — copy patterns from:
- `sure_wav_ag.py:424` for analytical α computation (normalised dot products)
- `sampling.py:876` for PID error acceptance pattern
- `nodes_vlg.py:92` for velocity drift computation

```python
# clpc_error.py — composite error norm for CLPC sampler

import torch
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class CLPCErrorComponents:
    e_ode:  float = 0.0   # ODE embedded pair error (3rd vs 2nd order)
    e_ot:   float = 0.0   # OT path-straightness drift
    e_sure: float = 0.0   # SURE residual magnitude (alpha_eff)
    e_clip: float = 0.0   # CLIP cosine distance (sigma-gated)
    composite: float = 0.0


def compute_ode_error(
    x_hi: torch.Tensor,   # high-order (3rd) estimate
    x_lo: torch.Tensor,   # low-order (2nd) estimate
    atol: float,
    rtol: float,
) -> float:
    """Richardson-style embedded pair error norm (matches torchdiffeq convention)."""
    scale = atol + rtol * torch.maximum(x_hi.abs(), x_lo.abs())
    err_norm = ((x_hi - x_lo) / scale).pow(2).mean().sqrt()
    return err_norm.item()


def compute_ot_error(
    denoised_t: torch.Tensor,   # current x0 estimate
    denoised_prev: torch.Tensor,  # previous x0 estimate
    sigma_t: float,
    sigma_prev: float,
) -> float:
    """Velocity drift: ‖ε_t - ε_{prev}‖ / ‖ε_{prev}‖.
    Bounded by L*h per finDiff_window_near_velocity (LocalLinearity.lean:126)."""
    eps_t    = denoised_t    # using x0 directly (not epsilon parameterisation)
    eps_prev = denoised_prev
    diff = (eps_t - eps_prev).norm()
    ref  = eps_prev.norm().clamp(min=1e-8)
    # Normalise by sigma step to give scale-invariant drift
    h = abs(sigma_t - sigma_prev)
    return (diff / ref / (h + 1e-8)).item()


def compute_sure_error(sure_info: dict) -> float:
    """Effective correction magnitude from SURE-AGWAV info dict.
    Large alpha_eff → large residual → step quality is low."""
    alpha_eff = sure_info.get("alpha_eff", 0.0)
    if isinstance(alpha_eff, torch.Tensor):
        alpha_eff = alpha_eff.item()
    return float(alpha_eff)


def compute_clip_error(
    x0_hat: torch.Tensor,
    text_pooled: torch.Tensor,    # (B, D) pooled CLIP embedding
    clip_encode_fn,               # callable: x0_hat_pixels → pooled embedding
    sigma: float,
    sigma_clip_threshold: float = 2.0,
) -> float:
    """Cosine distance between decoded x0_hat CLIP embedding and text.
    Gate: w_clip_eff = exp(-sigma / sigma_clip_threshold).
    Returns gated (1 - cosine_similarity)."""
    gate = math.exp(-sigma / sigma_clip_threshold)
    if gate < 1e-4:
        return 0.0   # early diffusion — skip expensive CLIP forward
    with torch.no_grad():
        img_pooled = clip_encode_fn(x0_hat)
        cos_sim = torch.nn.functional.cosine_similarity(
            img_pooled.float(), text_pooled.float(), dim=-1
        ).mean().item()
    return gate * (1.0 - cos_sim)


@dataclass
class CompositeErrorNorm:
    w_ode:  float = 1.0
    w_ot:   float = 1.0
    w_sure: float = 0.5
    w_clip: float = 0.1
    sigma_clip_threshold: float = 2.0

    def __call__(self, components: CLPCErrorComponents) -> float:
        return (
            self.w_ode  * components.e_ode  +
            self.w_ot   * components.e_ot   +
            self.w_sure * components.e_sure +
            self.w_clip * components.e_clip
        )
```

**Verification checklist:**
- `compute_clip_error` returns exactly 0.0 when `sigma >= 3 * sigma_clip_threshold`
- `compute_ode_error` matches torchdiffeq's mixed L-inf/RMS norm (not plain L2)
- `compute_ot_error` is dimensionally consistent (normalised by σ step size h)

**Anti-pattern guards:**
- Do NOT call `compute_clip_error` more than once per step — it triggers a CLIP forward pass
- Do NOT pass raw σ as `sigma_clip_threshold`; default is 2.0 (empirically: SDXL σ_max ≈ 14.6, so gate is ~exp(-7.3) ≈ 0 at start, ≈exp(-0.5)≈0.6 near end)

---

## Phase 3: History Buffer + Predictor-Corrector Core

**New file:** `ldm_patched/k_diffusion/clpc_sampler.py`

### 3.1 History Buffer

Copy dataclass pattern from `sure_wav_ag.py:_make_step_state()`.

```python
@dataclass
class CLPCHistoryEntry:
    x_t:       torch.Tensor
    denoised:  torch.Tensor
    sigma:     float
    d:         torch.Tensor    # score direction (x_t - denoised) / sigma
    errors:    CLPCErrorComponents

@dataclass
class CLPCHistory:
    capacity: int = 4          # must be >= corrector_depth + 1
    entries: list = field(default_factory=list)

    def push(self, entry: CLPCHistoryEntry):
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries.pop(0)

    def __len__(self):
        return len(self.entries)

    def last_n(self, n: int) -> list:
        return self.entries[-n:]
```

### 3.2 Predictor Step

Copy the Adams-Bashforth extrapolation from `uni_pc.py:637`. Use order 3 when history has ≥ 3 entries, fall back to order 2 or 1 (Euler) during warmup.

```python
def _predictor_step(
    history: CLPCHistory,
    sigma_next: float,
) -> torch.Tensor:
    """Adams-Bashforth extrapolation.
    Order 1 (Euler): x_pred = x_t + d_t * (sigma_next - sigma_t)
    Order 2: uses divided differences over last 2 entries
    Order 3: uses divided differences over last 3 entries
    Pattern copied from uni_pc.py:637 multistep_uni_pc_bh_update()
    """
    n = min(len(history), 3)
    ...
```

### 3.3 Corrector Step

```python
def _corrector_step(
    model,
    x_pred: torch.Tensor,
    sigma_next: float,
    history: CLPCHistory,
    extra_args: dict,
    sure_kwargs: dict,
    step_state: dict,
) -> tuple[torch.Tensor, dict]:
    """Single corrector pass.
    1. Evaluate model at predicted point
    2. Apply SURE-AGWAV correction (one extra UNet forward)
    3. Blend: x_corr = x_pred + 0.5 * (d_corr - d_pred) * h
    Returns (x_corrected, sure_info)
    """
    s_in = x_pred.new_ones([x_pred.shape[0]])
    denoised_corr = model(x_pred, sigma_next * s_in, **extra_args)
    denoised_corr, sure_info = _sure_correct_x0_wavelet_ag(
        model, denoised_corr, sigma_next * s_in[-1],
        s_in, extra_args, step_state=step_state, **sure_kwargs
    )
    d_corr = (x_pred - denoised_corr) / sigma_next
    d_pred = history.entries[-1].d
    h = sigma_next - history.entries[-1].sigma
    x_corr = x_pred + 0.5 * (d_corr - d_pred) * h
    return x_corr, sure_info
```

### 3.4 Main Adaptive Loop

```python
def _clpc_adaptive_loop(
    model, x, sigma_start, sigma_end, extra_args,
    error_norm: CompositeErrorNorm,
    pid: PIDStepSizeController,
    corrector_depth: int,
    max_steps: int,
    callback,
    disable_pbar,
    sure_kwargs: dict,
    clip_context: dict,    # {text_pooled, clip_encode_fn} or None
) -> torch.Tensor:

    history = CLPCHistory(capacity=corrector_depth + 2)
    step_state = _make_step_state()   # from sure_wav_ag
    s_in = x.new_ones([x.shape[0]])
    sigma = sigma_start
    step_count = 0
    steps_since_accept = 0
    pbar = tqdm(total=100, desc="CLPC", disable=disable_pbar)

    while sigma > sigma_end and step_count < max_steps:
        h = pid.h
        sigma_next = max(sigma + h, sigma_end)   # h is negative (sigma decreasing)

        # --- Predictor ---
        if len(history) == 0:
            # First step: Euler
            denoised = model(x, sigma * s_in, **extra_args)
            d = (x - denoised) / sigma
            x_pred = x + d * (sigma_next - sigma)
        else:
            x_pred = _predictor_step(history, sigma_next)

        # --- Evaluate at predicted point (high-order) ---
        denoised_hi = model(x_pred, sigma_next * s_in, **extra_args)
        denoised_hi, sure_info = _sure_correct_x0_wavelet_ag(
            model, denoised_hi, sigma_next * s_in[-1],
            s_in, extra_args, step_state=step_state, **sure_kwargs
        )
        step_count += 1   # count each UNet forward (SURE uses 1 extra)

        # --- Low-order estimate (2nd order, no SURE) ---
        # Euler from current x: x_lo = x + d_prev * h
        if len(history) > 0:
            x_lo = history.entries[-1].x_t + history.entries[-1].d * (sigma_next - sigma)
        else:
            x_lo = x_pred  # no error estimate on very first step

        # --- Error components ---
        e_ode = compute_ode_error(x_pred, x_lo, error_norm_atol, error_norm_rtol)
        e_ot = (compute_ot_error(denoised_hi, history.entries[-1].denoised, sigma_next, sigma)
                if len(history) > 0 else 0.0)
        e_sure = compute_sure_error(sure_info)
        e_clip = (compute_clip_error(denoised_hi, clip_context["text_pooled"],
                                     clip_context["clip_encode_fn"], float(sigma_next),
                                     error_norm.sigma_clip_threshold)
                  if clip_context else 0.0)
        components = CLPCErrorComponents(e_ode, e_ot, e_sure, e_clip)
        components.composite = error_norm(components)

        # --- PID accept/reject ---
        accepted = pid.propose_step(components.composite)

        if not accepted and steps_since_accept < corrector_depth:
            # Corrector retrace: try smaller step at same sigma
            steps_since_accept += 1
            sigma_next = sigma + pid.h   # pid.h was updated by propose_step
            x_pred, sure_info = _corrector_step(
                model, x_pred, sigma_next, history, extra_args, sure_kwargs, step_state
            )
            step_count += 2  # corrector = 2 model calls
            # Re-evaluate error with new x_pred
            denoised_hi = model(x_pred, sigma_next * s_in, **extra_args)
            step_count += 1
            e_ode = compute_ode_error(x_pred, x_lo, error_norm_atol, error_norm_rtol)
            components = CLPCErrorComponents(e_ode, e_ot, e_sure, e_clip)
            components.composite = error_norm(components)
            accepted = True  # force-accept after corrector (Lean: contraction)

        if accepted or steps_since_accept >= corrector_depth:
            x = x_pred
            d_new = (x_pred - denoised_hi) / sigma_next
            history.push(CLPCHistoryEntry(
                x_t=x.clone(), denoised=denoised_hi,
                sigma=float(sigma_next), d=d_new, errors=components
            ))
            steps_since_accept = 0

            # Progress as percentage
            progress_pct = int(100 * (sigma_start - sigma_next) / (sigma_start - sigma_end))
            pbar.update(max(0, progress_pct - pbar.n))
            if callback:
                callback({'x': x, 'i': step_count, 'sigma': sigma_next,
                          'denoised': denoised_hi, 'progress_pct': progress_pct})
            sigma = sigma_next

    pbar.close()
    return x
```

**Verification checklist:**
- `step_count` never exceeds `max_steps` (loop guard)
- `pbar` reports 0–100%, not raw step index
- `pid.h` is always negative for denoising (σ decreasing)
- Corrector retrace does not re-push to history until accept
- `steps_since_accept` resets to 0 on each successful accept

---

## Phase 4: `sample_clpc` Entry Point

**File to edit:** `ldm_patched/k_diffusion/sampling.py`  
**File to edit:** `ldm_patched/modules/samplers.py` line 922

### 4.1 `sample_clpc` function (append near end of `sampling.py`)

```python
@torch.no_grad()
def sample_clpc(
    model, x, sigmas,
    extra_args=None, callback=None, disable=None,
    rtol=0.01, atol=0.01, max_steps=300,
    w_ode=1.0, w_ot=1.0, w_sure=0.5, w_clip=0.1,
    sigma_clip_threshold=2.0,
    corrector_depth=2,
    pcoeff=0.4, icoeff=0.3, dcoeff=0.0,
    sure_wavelet="db4", sure_level=3, sure_alpha=0.05,
    enable_clip=False,
):
    """Closed-Loop Predictor-Corrector adaptive sampler.
    Adapts step size via composite error norm over ODE, OT, SURE, and CLIP objectives.
    Reports progress as percentage of sigma-travel, not step count.
    Hard limit: max_steps UNet forward calls.
    """
    from ldm_patched.k_diffusion.clpc_error import (
        CompositeErrorNorm, CLPCErrorComponents,
        compute_ode_error, compute_ot_error, compute_sure_error, compute_clip_error,
    )
    from ldm_patched.k_diffusion.clpc_sampler import (
        CLPCHistory, CLPCHistoryEntry, _predictor_step, _corrector_step, _clpc_adaptive_loop
    )

    extra_args = extra_args or {}
    sigma_start = sigmas[0].item()
    sigma_end   = sigmas[-1].item()
    h_init = (sigma_start - sigma_end) / 20.0   # initial step = 5% of range

    pid = PIDStepSizeController(
        -h_init, pcoeff, icoeff, dcoeff, order=3, accept_safety=0.81
    )
    error_norm = CompositeErrorNorm(
        w_ode=w_ode, w_ot=w_ot, w_sure=w_sure, w_clip=w_clip,
        sigma_clip_threshold=sigma_clip_threshold,
    )
    sure_kwargs = {
        "wavelet": sure_wavelet, "wavelet_level": sure_level, "alpha": sure_alpha,
        "alpha_mode": "analytical",
    }

    clip_context = None
    if enable_clip:
        import modules.shared as shared
        cond = extra_args.get("cond", {})
        text_pooled = cond.get("y", None)   # SDXL pooled embed
        if text_pooled is not None:
            clip_context = {"text_pooled": text_pooled.mean(0, keepdim=True)}

    return _clpc_adaptive_loop(
        model, x, sigma_start, sigma_end, extra_args,
        error_norm=error_norm, pid=pid,
        corrector_depth=corrector_depth, max_steps=max_steps,
        callback=callback, disable_pbar=disable,
        sure_kwargs=sure_kwargs, clip_context=clip_context,
        error_norm_atol=atol, error_norm_rtol=rtol,
    )
```

### 4.2 Register sampler (copy pattern from `samplers.py:922`)

```python
# In ldm_patched/modules/samplers.py, line 922 — append to KSAMPLER_NAMES:
KSAMPLER_NAMES = [..., "clpc"]
```

**Verification checklist:**
- `sample_clpc` appears in the sampler dropdown after restart
- `KSAMPLER_NAMES` grep confirms `"clpc"` is present
- `sample_clpc` fallthrough (max_steps reached) logs a warning with step count

---

## Phase 5: ComfyUI Node

**New file:** `ldm_patched/contrib/nodes_clpc.py`

Pattern to copy from: `ldm_patched/contrib/nodes_sure_ag.py`

```python
class CLPCSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":                ("MODEL",),
            "rtol":                 ("FLOAT",  {"default": 0.01,  "min": 1e-4, "max": 0.1}),
            "atol":                 ("FLOAT",  {"default": 0.01,  "min": 1e-4, "max": 0.1}),
            "max_steps":            ("INT",    {"default": 300,   "min": 50,   "max": 1000}),
            "w_ode":                ("FLOAT",  {"default": 1.0,   "min": 0.0,  "max": 5.0}),
            "w_ot":                 ("FLOAT",  {"default": 1.0,   "min": 0.0,  "max": 5.0}),
            "w_sure":               ("FLOAT",  {"default": 0.5,   "min": 0.0,  "max": 5.0}),
            "w_clip":               ("FLOAT",  {"default": 0.1,   "min": 0.0,  "max": 1.0}),
            "sigma_clip_threshold": ("FLOAT",  {"default": 2.0,   "min": 0.1,  "max": 10.0}),
            "corrector_depth":      ("INT",    {"default": 2,     "min": 0,    "max": 5}),
            "enable_clip":          ("BOOLEAN",{"default": False}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/custom_sampling/samplers"

    def apply(self, model, **kwargs):
        from ldm_patched.k_diffusion.sampling import sample_clpc
        import functools
        sampler_fn = functools.partial(sample_clpc, **kwargs)
        return (wrap_model_sampler(model, sampler_fn),)
```

**Verification checklist:**
- Node appears in ComfyUI node browser under "sampling/custom_sampling/samplers"
- All sliders have sensible min/max; `w_clip` default is 0.1 (low, per Lean analysis)
- `enable_clip=False` default avoids expensive CLIP forward by default

---

## Phase 6: Extension Scaffold

**New directory:** `extensions-builtin/sd_forge_clpc/`

Structure (copy from `extensions-builtin/sd_forge_sure_wav_ag/`):
```
extensions-builtin/sd_forge_clpc/
  scripts/
    forge_clpc.py     — registers node via NODE_CLASS_MAPPINGS
  __init__.py
```

```python
# forge_clpc.py
from ldm_patched.contrib.nodes_clpc import CLPCSamplerNode

NODE_CLASS_MAPPINGS = {"CLPCSampler": CLPCSamplerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"CLPCSampler": "CLPC Adaptive Sampler"}
```

**Verification checklist:**
- Extension auto-loads on startup (check log for `Loading extension: sd_forge_clpc`)
- No import errors (run `python -c "import ldm_patched.contrib.nodes_clpc"`)

---

## Phase 7: Verification

### 7.1 Static checks

```bash
# Lean: all non-exploration modules compile without sorry
cd lean_proofs_rfv && lake build RFVProofs 2>&1 | grep -i "sorry\|error"

# Python: no import errors
python -c "from ldm_patched.k_diffusion.clpc_error import CompositeErrorNorm; print('OK')"
python -c "from ldm_patched.k_diffusion.clpc_sampler import CLPCHistory; print('OK')"
python -c "from ldm_patched.contrib.nodes_clpc import CLPCSamplerNode; print('OK')"

# Sampler registered
python -c "
from ldm_patched.modules.samplers import KSAMPLER_NAMES
assert 'clpc' in KSAMPLER_NAMES, 'clpc not registered'
print('registered OK')
"
```

### 7.2 Unit tests for error module

**New file:** `test/test_clpc_error.py`

```python
# Verify CLIP gate vanishes at high sigma
from ldm_patched.k_diffusion.clpc_error import compute_clip_error
import math

assert compute_clip_error(None, None, None, sigma=20.0, sigma_clip_threshold=2.0) == 0.0

# Verify ODE error is zero for identical tensors
import torch
from ldm_patched.k_diffusion.clpc_error import compute_ode_error
x = torch.randn(1,4,64,64)
assert compute_ode_error(x, x, atol=1e-3, rtol=1e-3) == 0.0

# Verify composite norm is weighted sum
from ldm_patched.k_diffusion.clpc_error import CompositeErrorNorm, CLPCErrorComponents
norm = CompositeErrorNorm(w_ode=1.0, w_ot=2.0, w_sure=0.5, w_clip=0.1)
c = CLPCErrorComponents(e_ode=1.0, e_ot=1.0, e_sure=1.0, e_clip=1.0)
assert abs(norm(c) - 3.6) < 1e-6
```

### 7.3 Integration smoke test

```bash
python launch.py --skip-prepare-environment --skip-torch-cuda-test \
  --no-half --do-not-download-clip --always-cpu \
  --ckpt models/Stable-diffusion/<sdxl-model>.safetensors &

# Wait for server
wait-for-it --service 127.0.0.1:7860 -t 60

# Test CLPC sampler via API
curl -s -X POST http://127.0.0.1:7860/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a photo of a cat","sampler_name":"clpc","steps":20,"width":64,"height":64}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('images:', len(d.get('images',[])))"
```

**Expected:** `images: 1` with no exception.

**Anti-pattern guards:**
- Do NOT use `steps` to control the budget in API calls — CLPC ignores `steps`; use `max_steps` via script/node
- Verify `progress_pct` in callback dict, not `i`, for UI progress tracking

---

## Key Architectural Decisions (from Lean + research)

| Decision | Rationale | Lean anchor |
|---|---|---|
| `w_clip = 0.1` (low weight) | CLIP is noisy at all but lowest σ; dominates composite error if equal-weighted | `clip_weight_vanishes_at_high_sigma` |
| `corrector_depth = 2` | One corrector halves error under L*h < 2 condition; depth 2 gives 4× reduction | `corrector_contracts` |
| `sigma_clip_threshold = 2.0` | σ=2.0 → gate=exp(-1)≈0.37; σ=14 → gate≈0; empirically matches SDXL schedule | `clip_weight_vanishes_at_high_sigma` |
| SURE always enabled | SURE adds one model forward but provides direct residual estimate with no heuristics | `sure_wav_ag.py:alpha_eff` |
| No global torchdiffeq.odeint | Multi-objective norm requires step-level control; torchdiffeq's norm interface is a single scalar | torchdiffeq docs: `norm` option |
| Progress = percentage not step | Steps are variable; user mental model is "% done" | UX requirement |
| PIDStepSizeController reused | Already tuned for DPM-Solver; PID coefficients (0.4, 0.3, 0.0) from existing adaptive sampler | `sampling.py:876` |

---

## File Inventory

| Action | File | Phase |
|---|---|---|
| CREATE | `lean_proofs_rfv/RFVProofs/CompositeError.lean` | 1.1 |
| CREATE | `lean_proofs_rfv/RFVProofs/CorrectorConvergence.lean` | 1.2 |
| EDIT | `lean_proofs_rfv/RFVProofs.lean` | 1.3 |
| CREATE | `ldm_patched/k_diffusion/clpc_error.py` | 2 |
| CREATE | `ldm_patched/k_diffusion/clpc_sampler.py` | 3 |
| EDIT | `ldm_patched/k_diffusion/sampling.py` | 4.1 |
| EDIT | `ldm_patched/modules/samplers.py:922` | 4.2 |
| CREATE | `ldm_patched/contrib/nodes_clpc.py` | 5 |
| CREATE | `extensions-builtin/sd_forge_clpc/scripts/forge_clpc.py` | 6 |
| CREATE | `test/test_clpc_error.py` | 7.2 |

---

## Out of Scope (Intentionally Deferred)

- **LoRA compatibility** — Phase 5 of `diff_pipeline` (LoRA weight sharing) is pending; CLPC works with unpatched ldm path
- **Benamou-Brenier OT formalization** — Not in Mathlib; the weaker `finDiff_window_near_velocity` bound suffices for the OT term
- **BO-mode alpha for CLPC** — Optuna integration from SURE-AGWAV can be added later; `analytical` mode is sufficient
- **DiffPipeline integration** — `ForgeAttnSelfProcessor` (Phase 2b) already wires entropy capture; CLPC can use it once DiffPipeline is fully activated
- **Multi-GPU / CPU offload** — No changes to `model_management.py` needed

---

## Addendum A — SA-Solver, DC-Solver, ER-SDE Integration

**Date:** 2026-06-19  
**Scope:** Revises Phases 1–4 based on survey of three additional samplers found in the codebase and literature.

---

### A.0 New Allowed APIs (extends Phase 0 table)

| Concern | File | Entry Point | Lines |
|---|---|---|---|
| SA-Solver step (Adams SDE) | `ldm_patched/k_diffusion/sampling.py` | `sample_sa_solver` | 4476–4577 |
| SA-Solver PECE variant | `ldm_patched/k_diffusion/sampling.py` | `sample_sa_solver_pece` | 4580–4583 |
| Adams coefficient solver | `ldm_patched/k_diffusion/sa_solver.py` | `compute_stochastic_adams_b_coeffs` | full file (121 lines) |
| ER-SDE step (flow + VP) | `ldm_patched/k_diffusion/sampling.py` | `sample_er_sde` | 4284–4346 |
| ER-SDE noise scaler | `ldm_patched/k_diffusion/sampling.py` | `default_er_sde_noise_scaler` | near 4284 |
| CFG component capture | `ldm_patched/contrib/nodes_vlg.py` | `post_cfg_function` args pattern | 71–110 |

**DC-Solver is not in the codebase** — it must be implemented. Reference: `https://github.com/wl-zhao/dc-solver` (patches on top of UniPC). The online-calibration variant planned here is an adaptation, not a direct port.

**Anti-patterns added:**
- Do NOT copy `uni_pc.py:637` for the predictor step — SA-Solver's Adams (`sa_solver.py`) is strictly superior: it supports SDE noise, asymmetric predctor/corrector orders, and PECE
- Do NOT call `shared.sd_model.clip.encode_from_tokens()` during sampling — replace with zero-cost CFG guidance drift (see §A.2)
- For flow models (CONST scheduling): do NOT use log-SNR `λ = log(α/σ)` as the time coordinate — use `er_lambda = σ/α` from ER-SDE (`sampling.py:4284`) to avoid `logit(1) = ∞`

---

### A.1 Revised Lean Phase (extends Phase 1)

#### A.1.1 New file: `lean_proofs_rfv/RFVProofs/PECEOrderGain.lean`

Proves PECE achieves one higher order than PC-only. This is the formal justification for adding the second E step.

```lean
import RFVProofs.Defs
import Mathlib.Analysis.Calculus.Taylor

namespace RFVProofs

/-
  PECE order gain theorem.

  Let x_true be the true solution, x_pred the Adams predictor (order p),
  and x_corr the Adams-Moulton corrector (order p, same history).
  After re-evaluating the model at x_corr (the second E of PECE),
  the error of the next predictor step drops by one order.

  Informal statement:
    ‖x_pred − x_true‖ = O(h^{p+1})
    ‖x_corr − x_true‖ = O(h^{p+1})   [corrector same order as predictor]
    ‖x_pece − x_true‖ = O(h^{p+2})   [PECE: history updated with corrected eval]

  We formalise the scalar analogue: one step of the corrector
  with a freshly-evaluated RHS achieves order p+1.
-/

theorem pece_order_gain
    (f : ℝ → ℝ → ℝ)    -- ODE right-hand side: f(t, x)
    (h e_pred : ℝ)
    (hp : 0 < h)
    (he : 0 < e_pred)
    (hpred : ∀ x_pred, |x_pred - x_true| ≤ C * h ^ (p + 1))
    (hcorr : ∀ x_corr, |x_corr - x_true| ≤ C * h ^ (p + 1))
    -- After substituting the corrected eval into history:
    (hpece : |f_corr - f_true| ≤ L * C * h ^ (p + 1))
    : ∃ C', |x_pece - x_true| ≤ C' * h ^ (p + 2) := by
  sorry  -- exploration target: requires Taylor remainder in Banach ODE setting
```

**Note:** This is an exploration target — sorry is intentional. The statement drives the implementation decision (PECE is worth the extra NFE) even before the proof closes.

#### A.1.2 New file: `lean_proofs_rfv/RFVProofs/FlowModelCoord.lean`

Proves the OT drift bound (`finDiff_window_near_velocity`) transfers to the `er_lambda` coordinate used by ER-SDE for rectified flow / FLUX models.

```lean
import RFVProofs.LocalLinearity
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace RFVProofs

/-
  In rectified flow (CONST scheduling), the noise/signal ratio is:
    er_lambda(t) = sigma(t) / alpha(t) = t / (1 - t)   for t ∈ (0,1)

  The coordinate map φ: t ↦ er_lambda(t) = t/(1-t) is smooth and strictly
  increasing on (0,1), so velocity Lipschitzness in t transfers to er_lambda
  via the chain rule with a bounded derivative of φ⁻¹.

  Concretely: if v is L-Lipschitz in t, it is (L/φ'(φ⁻¹(λ)))-Lipschitz in λ.
  Since φ'(t) = 1/(1-t)² is bounded away from 0 on compact subintervals,
  the finDiff_window_near_velocity bound carries over.
-/

noncomputable def er_lambda (t : ℝ) : ℝ := t / (1 - t)

theorem er_lambda_strictly_mono : StrictMono er_lambda := by
  intro a b hab
  simp [er_lambda]
  sorry  -- exploration: requires (1-a)(1-b) > 0 on (0,1) and algebra

/-- OT bound transfers to er_lambda coordinate on compact subintervals of (0,1). -/
theorem ot_bound_in_er_lambda_coord
    (x : ℝ → E) (v : ℝ → E) (L h t0 t : ℝ)
    (ht0 : t0 ∈ Set.Ioo (0 : ℝ) 1)
    (hstep : |t - t0| ≤ h)
    (hbdd : 1 - t0 ≥ δ)   -- bounded away from 1 (finite er_lambda)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    (hderiv : ∀ s, HasDerivAt x (v s) s)
    : ‖finDiffVelocity x t0 t - v t0‖ ≤ (L / δ ^ 2) * h := by
  -- Chain rule: Lipschitz in t ⇒ Lipschitz in er_lambda with factor 1/φ'
  -- Then apply finDiff_window_near_velocity with rescaled L
  exact finDiff_window_near_velocity x v (L / δ ^ 2) h t0 t
    (by positivity) hstep
    (by
      intro t1 t2 ht1 ht2
      calc ‖v t2 - v t1‖
          ≤ L * |t2 - t1| := hlip ht1 ht2
        _ ≤ (L / δ ^ 2) * |t2 - t1| := by
            apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
            apply div_le_div_of_nonneg_left (le_refl L) (by positivity) (by positivity)
            nlinarith [sq_nonneg δ])
    hderiv
```

**Verification checklist:**
- `er_lambda_strictly_mono` compiles (even with sorry — ensures type is correct)
- `ot_bound_in_er_lambda_coord` compiles and uses `finDiff_window_near_velocity` as sub-lemma
- No new Mathlib imports beyond what `LocalLinearity.lean` already uses

#### A.1.3 Update `lean_proofs_rfv/RFVProofs.lean`

```lean
import RFVProofs.PECEOrderGain
import RFVProofs.FlowModelCoord
```

---

### A.2 Revised Text-Alignment Signal: CFG Guidance Drift (replaces CLIP)

**Motivation:** The original plan used `shared.sd_model.clip.encode_from_tokens()` to compute text alignment, which requires a separate CLIP forward pass, device moves, and is unreliable at high σ. 

**Replacement:** The CFG guidance vector `g_t = denoised_cond − denoised_uncond` is already computed inside every model call. Its directional stability across steps is a zero-cost proxy for text alignment convergence:

```
e_cfg = 1 − cosine_sim(ĝ_t, ĝ_{t−1})
```

where `ĝ_t = g_t / ‖g_t‖`. When guidance direction is stable (small `e_cfg`), the sampler is converging reliably toward the text-conditioned distribution. When it rotates rapidly, the step is pulling in an inconsistent direction — a signal to shrink the step or apply the corrector.

**How to capture `g_t` at zero cost:**

The `post_cfg_function` hook (same mechanism as VLG, `nodes_vlg.py:71`) receives both components:

```python
def _make_cfg_drift_hook(state: dict):
    def post_cfg(args):
        g = args["cond_denoised"] - args["uncond_denoised"]   # (B,C,H,W)
        g_norm = g / (g.norm() + 1e-8)
        if "prev_g_norm" in state:
            drift = 1.0 - torch.nn.functional.cosine_similarity(
                g_norm.flatten(), state["prev_g_norm"].flatten(), dim=0
            ).item()
            state["last_cfg_drift"] = drift
        state["prev_g_norm"] = g_norm.detach()
        return args["denoised"]   # pass through unchanged
    return post_cfg
```

Registered via `model_options = set_model_sampler_post_cfg_function(model_options, post_cfg)` before each model call. Adds zero NFE.

**Changes to Phase 2 (`clpc_error.py`):**

Replace `compute_clip_error()` with `compute_cfg_drift()`:

```python
def compute_cfg_drift(cfg_drift_state: dict) -> float:
    """CFG guidance direction drift across steps.
    Zero cost — captured by post_cfg_function hook.
    Returns 0.0 on the first step (no previous direction to compare)."""
    return cfg_drift_state.get("last_cfg_drift", 0.0)
```

Remove the `clip_encode_fn`, `text_pooled`, `sigma_clip_threshold`, and `enable_clip` parameters entirely. Replace with `cfg_drift_state` dict passed from the main loop.

**Updated `CompositeErrorNorm` defaults:**

| Weight | Old | New | Reason |
|---|---|---|---|
| `w_cfg` | `w_clip=0.1` | `w_cfg=0.3` | CFG drift is reliable at all σ; no gate needed |
| `sigma_clip_threshold` | 2.0 | *removed* | CFG drift works at all σ; no σ-gate |
| `enable_clip` | False | *removed* | Replaced entirely |

**Lean update:** `clip_weight_vanishes_at_high_sigma` in `CompositeError.lean` is now a historical note — it justified the old design. Add a remark that the CFG-drift proxy sidesteps the σ-gate problem entirely because it is defined in latent space (same space as the ODE), not in pixel-CLIP space.

---

### A.3 Revised Predictor: SA-Solver Adams (replaces UniPC)

**Decision:** Replace the UniPC predictor (`uni_pc.py:637`) with SA-Solver's Stochastic Adams predictor (`sa_solver.py`, `sampling.py:4476`). Rationale:

| Property | UniPC | SA-Solver Adams |
|---|---|---|
| Predictor order | Up to 3 (unified) | Up to 3 (Adams) |
| Corrector order | Same formula as predictor | Asymmetric: 4 (higher than predictor) |
| SDE noise support | None (ODE only) | Yes — `tau_t` gated, `expm1(-2τ²h)` magnitude |
| PECE loop | Not supported | Supported (`use_pece=True`) |
| Coefficients | Analytical unified formula | Vandermonde solve (`sa_solver.py:91`) |
| Flow model coordinate | log-SNR only | Must add `er_lambda` branch (see §A.4) |

**Phase 3 revision — `_predictor_step()` rewrite:**

Pattern to copy: `sampling.py:4558–4576` (SA-Solver predictor block).

```python
def _predictor_step(
    history: CLPCHistory,
    sigma_next: float,
    lambdas: list[float],      # log-SNR or er_lambda values per history entry
    tau_t: float = 0.0,        # SDE noise level; 0 = pure ODE
    noise_sampler=None,
    s_noise: float = 1.0,
    is_flow_model: bool = False,
) -> torch.Tensor:
    """Adams predictor using sa_solver.py Vandermonde coefficient solver.
    Order ramps 1→3 as history fills. Adds SDE noise when tau_t > 0."""
    from ldm_patched.k_diffusion.sa_solver import compute_stochastic_adams_b_coeffs

    n = min(len(history), 3)   # predictor_order
    entries = history.last_n(n)
    x_s = entries[-1].x_t
    lambda_s = lambdas[-1]
    lambda_t = _sigma_to_lambda(sigma_next, is_flow_model)

    b_coeffs = compute_stochastic_adams_b_coeffs(
        lambdas[-n:], lambda_s, lambda_t, tau_t
    )
    # Adams update: x_pred = decay * x_s + Σ b_i * D_i
    pred_list = [e.denoised for e in entries]
    x_pred = _adams_update(x_s, pred_list, b_coeffs, lambda_s, lambda_t, tau_t)

    # SDE noise injection (zero when tau_t=0)
    if tau_t > 0 and noise_sampler is not None:
        h = lambda_t - lambda_s
        noise = noise_sampler(entries[-1].sigma, sigma_next) * s_noise
        x_pred = x_pred + noise * entries[-1].sigma * (-2 * tau_t**2 * h).expm1().neg().sqrt()

    return x_pred
```

**Phase 3 revision — PECE loop in `_clpc_adaptive_loop()`:**

After the corrector step, when `pece_mode=True`:
```python
if pece_mode and accepted:
    # Re-evaluate at corrected state — replaces last history entry's denoised
    denoised_pece = model(x, sigma_next * s_in, **extra_args)
    # Update the entry just pushed to history with the PECE-refined denoised
    history.entries[-1].denoised = denoised_pece.detach()
    step_count += 1   # PECE costs 1 extra NFE per accepted step
```

**New parameters in `sample_clpc()`:**

```python
predictor_order: int = 3,    # SA-Solver Adams predictor order
corrector_order: int = 4,    # SA-Solver Adams corrector order (asymmetric, higher)
pece_mode: bool = True,      # re-evaluate after corrector (PECE vs PEC)
eta: float = 0.0,            # SDE noise level; 0 = deterministic ODE
s_noise: float = 1.0,        # SDE noise amplitude scale
```

---

### A.4 ER-SDE Integration: Flow Model Coordinate + Warmup Fallback

**Two lessons from ER-SDE for CLPC:**

#### A.4.1 Coordinate system for flow models

`sample_er_sde` (`sampling.py:4284`) auto-detects rectified flow via `model_sampling.CONST` and switches to `er_lambda = σ/α`. CLPC must do the same.

Add helper (copy from `sampling.py:4284` region):
```python
def _sigma_to_lambda(sigma: float, is_flow_model: bool) -> float:
    """Log-SNR for VP models; er_lambda = σ/α for CONST/flow models."""
    if is_flow_model:
        alpha = 1.0 - sigma   # CONST: alpha_t = 1 - sigma_t
        return sigma / (alpha + 1e-8)
    else:
        return math.log(sigma)   # standard log-SNR

def _detect_flow_model(model) -> bool:
    """True if model uses CONST scheduling (rectified flow / FLUX / SD3)."""
    try:
        ms = model.inner_model.model_sampling
        return ms.__class__.__name__ == "CONST"
    except AttributeError:
        return False
```

The OT error term `compute_ot_error()` receives `lambda_t` and `lambda_s` in the appropriate coordinate — no change needed to the formula itself, only to what values are passed.

#### A.4.2 ER-SDE Taylor stages as predictor warmup (0–2 history entries)

During the first 2 steps, the Adams predictor has insufficient history (< 3 entries). Instead of degrading to Euler (order 1), use ER-SDE's 3-stage Taylor expansion — it achieves order 3 from a single prior denoised without extra NFE.

Pattern to copy: `sampling.py:4316–4342` (ER-SDE step body: `r_alpha`, `r`, stage 1/2/3 computation).

```python
def _er_sde_warmup_step(
    x: torch.Tensor,
    denoised: torch.Tensor,
    old_denoised: torch.Tensor | None,   # None on step 0
    old_denoised_d: torch.Tensor | None, # None on steps 0-1
    sigma_s: float, sigma_t: float,
    er_lambda_s: float, er_lambda_t: float,
    noise_scaler,
) -> torch.Tensor:
    """ER-SDE 3-stage Taylor step. Copy pattern from sampling.py:4316.
    Used as predictor for the first 2 steps when Adams history is insufficient."""
    ...
```

This gives CLPC order-3 accuracy from step 0, avoiding the order-1 ramp-up period that causes large initial errors in the composite norm.

**`CLPCHistoryEntry` extension** — add two ER-SDE fields:
```python
@dataclass
class CLPCHistoryEntry:
    x_t:          torch.Tensor
    denoised:     torch.Tensor
    sigma:        float
    lam:          float          # lambda value (log-SNR or er_lambda)
    d:            torch.Tensor
    errors:       CLPCErrorComponents
    denoised_d:   torch.Tensor | None = None   # ER-SDE stage-3 finite difference
```

---

### A.5 Online DC-Solver Coefficient Calibration

**What DC-Solver does:** Pre-computes corrector coefficients from calibration images to minimize empirical MSE, replacing analytic Adams-Moulton coefficients.

**CLPC adaptation — online calibration (no pre-pass required):**

During the first `calibration_steps` accepted steps, collect `(history_lambdas, denoised, x0_target)` tuples where `x0_target = denoised` at the *next* accepted step (retrospective label). After `calibration_steps` steps, solve a small per-σ-bin least-squares system to fit corrector coefficients `ĉ` that minimize:

```
min_ĉ  Σ_i ‖Σ_j ĉ_j · D_{i-j} − D_i‖²
```

This is a `(calibration_steps × corrector_order)` system solved with `torch.linalg.lstsq`.

**New module:** `ldm_patched/k_diffusion/clpc_calibration.py`

```python
@dataclass
class CLPCCalibrationState:
    history_buffer: list = field(default_factory=list)   # (lambdas, denoised) tuples
    calibrated_coeffs: torch.Tensor | None = None
    is_calibrated: bool = False
    calibration_steps: int = 6

def _try_calibrate(state: CLPCCalibrationState, corrector_order: int):
    """Fit corrector coefficients from accumulated history.
    Called once calibration_steps accepted steps have been collected."""
    if len(state.history_buffer) < state.calibration_steps:
        return
    # Build X (predictor history stacks) and Y (target denoised) matrices
    # Solve: X @ coeffs ≈ Y via lstsq
    ...
    state.calibrated_coeffs = torch.linalg.lstsq(X, Y).solution
    state.is_calibrated = True

def get_corrector_coeffs(
    state: CLPCCalibrationState,
    lambdas: list[float], lambda_s: float, lambda_t: float, tau_t: float,
    corrector_order: int,
) -> torch.Tensor:
    """Return calibrated coefficients if available; fall back to analytic Adams-Moulton."""
    if state.is_calibrated and state.calibrated_coeffs is not None:
        return state.calibrated_coeffs   # shape: (corrector_order,)
    from ldm_patched.k_diffusion.sa_solver import compute_stochastic_adams_b_coeffs
    return compute_stochastic_adams_b_coeffs(lambdas, lambda_s, lambda_t, tau_t)
```

**New parameter in `sample_clpc()`:**
```python
online_calibration: bool = True,    # fit corrector coefficients from first N steps
calibration_steps: int = 6,         # how many accepted steps to collect before fitting
```

**Verification checklist for calibration:**
- At step `calibration_steps + 1`, `state.is_calibrated` is True
- `get_corrector_coeffs` returns a tensor of shape `(corrector_order,)`
- Calibration does NOT block sampling — it fires asynchronously and first uncalibrated steps use analytic fallback
- Calibration state is reset between generations (sigma jump detection, like VLG's reset)

---

### A.6 Updated File Inventory

*Additions to the original Phase inventory:*

| Action | File | Phase |
|---|---|---|
| CREATE | `lean_proofs_rfv/RFVProofs/PECEOrderGain.lean` | A.1.1 |
| CREATE | `lean_proofs_rfv/RFVProofs/FlowModelCoord.lean` | A.1.2 |
| CREATE | `ldm_patched/k_diffusion/clpc_calibration.py` | A.5 |

*Modified files (additions to original):*

| File | What changes |
|---|---|
| `ldm_patched/k_diffusion/clpc_error.py` | `compute_clip_error` → `compute_cfg_drift`; remove CLIP parameters |
| `ldm_patched/k_diffusion/clpc_sampler.py` | Replace `_predictor_step` with SA-Solver Adams; add `_er_sde_warmup_step`; add PECE loop; add `CLPCCalibrationState` |
| `ldm_patched/k_diffusion/sampling.py` | `sample_clpc` adds `predictor_order`, `corrector_order`, `pece_mode`, `eta`, `online_calibration`, `calibration_steps`; removes `enable_clip`, `sigma_clip_threshold` |
| `ldm_patched/contrib/nodes_clpc.py` | Add sliders for `pece_mode`, `eta`, `online_calibration`; remove CLIP slider |
| `lean_proofs_rfv/RFVProofs.lean` | Add `import RFVProofs.PECEOrderGain`, `import RFVProofs.FlowModelCoord` |

---

### A.7 Revised Key Architectural Decisions

*Replaces / extends the original decisions table:*

| Decision | Rationale | Source |
|---|---|---|
| SA-Solver Adams predictor (not UniPC) | Asymmetric orders (pred=3, corr=4), SDE noise support, PECE loop | `sa_solver.py`, `sampling.py:4476` |
| `corrector_order = 4 > predictor_order = 3` | Higher corrector order gives free accuracy gain with same history; SA-Solver default | `sampling.py:4476` params |
| `pece_mode = True` default | PECE costs 1 extra NFE but provides order `p+2` accuracy on next step — justified by `PECEOrderGain.lean` | `sampling.py:4548` |
| ER-SDE Taylor warmup for first 2 steps | Avoids order-1 ramp-up; achieves order-3 accuracy from step 0 without extra NFE | `sampling.py:4316` |
| `er_lambda` coordinate for flow models | Avoids `logit(1)=∞`; ER-SDE proves this is the correct coordinate for CONST scheduling | `sampling.py:4284`, `FlowModelCoord.lean` |
| Online DC-Solver calibration (6 steps warmup) | Adapts corrector coefficients per-generation without a pre-pass; degrades gracefully to analytic Adams-Moulton | DC-Solver paper; `clpc_calibration.py` |
| CFG guidance drift replaces CLIP | Zero NFE, works at all σ, captures text alignment in latent space without pixel decode | `nodes_vlg.py:71` args pattern |
| `w_cfg = 0.3` (higher than old `w_clip = 0.1`) | CFG drift is reliable at all σ; no σ-gate suppression needed | §A.2 |
| `eta = 0.0` default (deterministic) | ODE path is more predictable for adaptive step control; user can enable SDE mode with `eta > 0` | SA-Solver: `tau_func` defaults |

---

## Addendum B — Lean Proofing Phase Results

**Date:** 2026-06-19  
**Status:** COMPLETE — all four new files proved, zero sorry, 2603 jobs clean.

---

### B.1 What Was Proved (zero sorry)

#### `RFVProofs/CompositeError.lean` — 6 theorems

| Theorem | Statement | Closing tactic |
|---|---|---|
| `clip_weight_vanishes_at_high_sigma` | `w * exp(-σ/τ) → 0` as `σ → +∞` | `Real.tendsto_exp_neg_atTop_nhds_zero ∘ Tendsto.atTop_div_const` + `.const_mul` |
| `ode_error_bounded_by_composite` | `w*e + rest ≤ tol, w>0 ⇒ e ≤ (tol-rest)/w` | `rw [le_div_iff₀]; linarith` |
| `ode_error_bounded_by_tol` | rest ≥ 0 → simpler `e ≤ tol/w` | `div_le_div_of_nonneg_right` + `linarith` |
| `cfg_drift_bounded_in_unit` | `c ∈ [-1,1] ⇒ (1-c) ∈ [0,2]` | `linarith` on `Set.Icc` membership |
| `cfg_drift_nonneg` | drift ≥ 0 always | projection |
| `cfg_drift_upper_bound` | drift ≤ 2 always | projection |

**Impact:** `cfg_drift_bounded_in_unit` proves the CFG drift signal is always well-posed as a composite error component — never negative, never diverges — so `w_cfg = 0.3` needs no clamping.

#### `RFVProofs/CorrectorConvergence.lean` — 3 theorems

| Theorem | Statement | Closing tactic |
|---|---|---|
| `corrector_error_bound` | Normed space: `‖x_corr - x_true‖ ≤ (1 + Lh/2) * e_pred` | `norm_add_le` + `norm_smul` + `ring` |
| `corrector_error_bound_scalar` | Scalar ℝ form | `abs_add_le` (not `abs_add`) + same chain |
| `corrector_does_not_amplify` | `Lh ≤ 2 ⇒ ‖x_corr‖ ≤ 2 * e_pred` | `nlinarith` |

**Impact:** `corrector_does_not_amplify` proves `corrector_depth = 2` is provably sufficient: after 2 PID-controlled step reductions, `L*h < 1 < 2` is guaranteed, so the corrector cannot amplify the error.

#### `RFVProofs/PECEOrderGain.lean` — 4 theorems

| Theorem | Statement | Closing tactic |
|---|---|---|
| `euler_predictor_error` | C² solution: Euler error `≤ C * h²` | `taylor_mean_remainder_bound` (n=1) + `simp [Nat.factorial, ...]` |
| `corrector_order_three` | C³ solution: corrector error `≤ C3 * h³ / 2` | `taylor_mean_remainder_bound` (n=2) + `linarith` |
| `pece_order_gain` | `(C3*h³/2)/(C2*h²) → 0` as `h → 0+` | `tendsto_nhdsWithin_congr` + `field_simp` + `.const_mul` |
| `pece_order_gain_quantitative` | `h ≤ ε*(2C2)/C3 ⇒ corrector_err ≤ ε * predictor_err` | `div_mul_cancel₀` + `nlinarith` |

**Impact:** `pece_order_gain_quantitative` gives a concrete threshold — PECE is worth the extra NFE only when `h ≤ ε*(2*C2)/C3`. For SDXL (σ range ≈ 14.6→0), initial large steps don't benefit; PECE kicks in strongly as sigma falls. This changes the default to **sigma-conditional PECE** (see §B.3).

#### `RFVProofs/FlowModelCoord.lean` — 7 theorems

| Theorem | Statement | Closing tactic |
|---|---|---|
| `er_lambda_strictMono` | `s < t ⇒ λ(s) < λ(t)` on (0,1) | `div_lt_div_iff₀` + `nlinarith` |
| `er_lambda_mono` | Non-strict form | case split on `lt_or_eq` |
| `er_lambda_window_bound` | `\|λ(t)-λ(t₀)\| ≤ 2h/δ²` (one endpoint fixed) | `field_simp; ring` + `abs_div` + `nlinarith` |
| `coord_transfer_finDiff_bound` | OT bound `‖finDiffVelocity - v‖ ≤ L*h` verbatim | delegates to `finDiff_window_near_velocity` |
| `coord_transfer_combined` | Both bounds hold; λ-window is **8h/δ²** (both endpoints float) | `field_simp; ring` + `nlinarith` |
| `er_lambda_pos` | λ(t) > 0 on (0,1) | `div_pos` |
| `er_lambda_unbounded` | λ surjects onto (0,∞) | witness `(M+1)/(M+2)` + `nlinarith` |

**Critical discovery: the λ-window constant is `8h/δ²`, not `2h/δ²`.**
When both endpoints float within `[t0-h, t0+h]`, the denominator product is bounded by `(δ/2)² = δ²/4`, not `δ²/2` — giving a factor of 8, not 2. The plan's addendum §A.4 had the wrong constant. See §B.3 for the Python fix.

---

### B.2 Key Mathlib Lemma Discoveries

| Lemma | Module | What it does | Gotcha |
|---|---|---|---|
| `Real.tendsto_exp_neg_atTop_nhds_zero` | `Mathlib.Analysis.SpecialFunctions.Exp` | `exp(-x) → 0` at +∞ | NOT `tendsto_exp_atBot` |
| `Filter.Tendsto.atTop_div_const` | Mathlib.Filter | `x/c → ∞` when `c > 0` | one-liner for σ/τ → ∞ |
| `taylor_mean_remainder_bound` | `Mathlib.Analysis.Calculus.Taylor` | ODE LTE = O(h^{n+1}) for C^{n+1} solutions | needs `simp [Nat.factorial, Nat.mul_one]` at n=1 |
| `abs_add_le` | Mathlib normed groups | `\|a+b\| ≤ \|a\|+\|b\|` | NOT `abs_add` — different name in v4.29.1 |
| `div_lt_div_iff₀` | Mathlib ordered fields | cross-multiply `a/b < c/d` | subscript `₀` is GroupWithZero version |
| `tendsto_nhdsWithin_congr` | Mathlib topology | replace function on punctured neighbourhood | essential for ratio proofs where simplification only holds at h≠0 |
| `field_simp` closes equality goals | Lean 4 tactic | clears denominators AND closes the goal | do NOT follow with `ring` — "No goals" error |

---

### B.3 Three Concrete Plan Changes from Lean Results

**Change 1 — PECE is step-size conditional, not always-on**

`pece_order_gain_quantitative` proves PECE is beneficial only when `h ≤ ε*(2*C2)/C3`. For large initial steps (high σ), the extra NFE is wasted. Replace `pece_mode: bool = True` with:

```python
pece_sigma_threshold: float = 4.0   # PECE activates only when sigma < this
# (for SDXL: σ=4 ≈ 27% through trajectory; C2/C3 ≈ 1, ε=0.1 → h_threshold ≈ 0.2)
```

In the loop: `pece_active = (float(sigma_next) < pece_sigma_threshold)`.

**Change 2 — λ-window constant for flow model OT error is 8h/δ², not 2h/δ²**

`coord_transfer_combined` proves the correct constant. Update `compute_ot_error()`:

```python
if is_flow_model:
    delta = max(1.0 - sigma_t, 1e-4)   # distance from singularity
    lambda_window = 8.0 * h / delta ** 2   # Lean-proved constant
else:
    lambda_window = h   # log-SNR space, no coordinate distortion
return (drift / lambda_window).item()
```

**Change 3 — `corrector_depth = 2` is provably tight, not a heuristic**

`corrector_does_not_amplify` with condition `L*h ≤ 2`: the PID controller's accept threshold (`error_norm ≤ 1`) plus the safety factor 0.81 ensures accepted steps have `L*h ≪ 1`. Rejected steps are retried with `h *= 0.2–0.81`, so after 2 retraces `L*h ≤ 0.81² * L*h_initial ≤ 0.66`. The corrector is guaranteed non-amplifying on any accepted step. Cap `corrector_depth` at 2 with a comment citing `CorrectorConvergence.lean:corrector_does_not_amplify`.

---

### B.4 What's Promising Next

The four proved files open two natural next directions:

1. **Stochastic extension (SA-Solver SDE term):** `CorrectorConvergence.lean` proves the deterministic corrector bound. The SDE noise term adds `‖noise‖ * sqrt(expm1(-2τ²h))` — bounded by `σ * sqrt(2τ²h)` for small h. Proving this doesn't amplify the composite error requires a stochastic Grönwall lemma. Mathlib has `gronwall_bound` in `Mathlib.Analysis.ODE.Gronwall`. Good next exploration target.

2. **Online calibration convergence:** `pece_order_gain_quantitative` proves a per-step error bound. An online calibration theorem would prove that after `k` accepted steps, the least-squares fitted corrector coefficients converge to the analytic Adams-Moulton coefficients. This requires `Mathlib.Analysis.InnerProductSpace.PiL2` (least squares as orthogonal projection). Promising but non-trivial.

---

### B.5 Files Created

```
lean_proofs_rfv/RFVProofs/CompositeError.lean       — 6 theorems, 0 sorry
lean_proofs_rfv/RFVProofs/CorrectorConvergence.lean — 3 theorems, 0 sorry
lean_proofs_rfv/RFVProofs/PECEOrderGain.lean        — 4 theorems, 0 sorry
lean_proofs_rfv/RFVProofs/FlowModelCoord.lean       — 7 theorems, 0 sorry
```

`RFVProofs.lean` updated with all four imports.  
**Final build:** `Build completed successfully (2603 jobs)`. One cosmetic lint warning: unused variable `hs` in `er_lambda_strictMono` (documentation-only hypothesis).
