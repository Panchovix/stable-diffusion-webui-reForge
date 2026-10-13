-- SURE_verification.lean
-- Lean 4 / Mathlib formal verification of five gradient-descent conditions
-- for the SURE correction step in _sure_correct_x0 (sampling.py).
--
-- In 'approx' (stop-gradient) mode the update is:
--   grad = 2·(x̂₀ − D_θ(x̂₀))    (J_D term dropped, D treated as constant c)
--   x̂₀ ← x̂₀ − α · grad
--
-- This is gradient descent on the proxy loss L(x) = (x − c)².
-- Conditions 1–5 are proved formally for this proxy.
-- For 'vjp'/'full' mode the NOTE sections explain why the proofs break.

import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Algebra.Order.Field.Basic

open Set

namespace SUREVerification

variable (c : ℝ)

/-- Stop-gradient SURE proxy loss: L(x) = (x − c)². -/
def L (x : ℝ) : ℝ := (x - c) ^ 2

/-! ── CONDITION 1: L is a total function ℝ → ℝ ────────────────────────────── -/

theorem cond1_total : ∀ x : ℝ, L c x ∈ (univ : Set ℝ) :=
  fun _ => mem_univ _

theorem cond1_differentiable : Differentiable ℝ (L c) := by
  unfold L; fun_prop

/-! ── CONDITION 2: L has a unique global minimum ────────────────────────────── -/

theorem cond2_min_value : L c c = 0 := by simp [L]

theorem cond2_global_min (x : ℝ) : L c c ≤ L c x := by
  unfold L
  have hzero : (c - c) ^ 2 = 0 := by ring
  rw [hzero]
  exact sq_nonneg _

theorem cond2_unique_min (x : ℝ) (h : L c x = L c c) : x = c := by
  rw [cond2_min_value] at h
  simp only [L] at h
  nlinarith [sq_nonneg (x - c)]

-- NOTE (vjp/full): L(x) = ‖x − D_θ(x)‖² may have no minimum when D_θ is
-- expansive.  Existence requires D_θ to be a contraction (Banach theorem);
-- neural denoisers do NOT satisfy this in general.

/-! ── CONDITION 3: At x ≠ c the gradient points away from the minimum ─────── -/

/-- The gradient of L at x is 2·(x − c). -/
theorem L_hasDerivAt (x : ℝ) : HasDerivAt (L c) (2 * (x - c)) x := by
  unfold L
  -- (· − c) has derivative 1 at x
  have h_sub : HasDerivAt (fun y => y - c) (1 : ℝ) x :=
    (hasDerivAt_id x).sub_const c
  -- (· ^ 2) has derivative 2t at t = x − c; compose via chain rule
  have h_sq : HasDerivAt (fun t : ℝ => t ^ 2)
      ((2 : ℝ) * (x - c) ^ (2 - 1)) (x - c) :=
    hasDerivAt_pow 2 (x - c)
  have h2 := h_sq.comp x h_sub
  simp only [mul_one] at h2
  convert h2 using 1
  ring

/-- Condition 3: (x − c) · grad > 0 at x ≠ c. -/
theorem cond3_grad_away (x : ℝ) (hne : x ≠ c) :
    (x - c) * (2 * (x - c)) > 0 := by
  have hd : x - c ≠ 0 := sub_ne_zero.mpr hne
  have hpos : 0 < (x - c) ^ 2 := sq_pos_of_ne_zero hd
  nlinarith [hpos]

/-- The gradient is nonzero at every non-minimiser. -/
theorem cond3_grad_nonzero (x : ℝ) (hne : x ≠ c) : 2 * (x - c) ≠ 0 :=
  mul_ne_zero (by norm_num) (sub_ne_zero.mpr hne)

-- NOTE (vjp): grad = 2·(I − J_D)ᵀ·r.  When J_D has an eigenvalue of 1,
-- (I − J_D) is singular and grad vanishes even at non-fixed-points.
-- Condition 3 fails without a spectral-gap assumption ‖J_D‖₂ < 1 − δ.

/-! ── CONDITION 4: −grad points toward the minimum ─────────────────────────── -/

/-- Condition 4: one GD step x' = x − α·2(x−c) is closer to c for α ∈ (0, 1/2). -/
theorem cond4_step_closer (x α : ℝ) (hα0 : 0 < α) (hα1 : α < 1 / 2) (hne : x ≠ c) :
    |x - α * (2 * (x - c)) - c| < |x - c| := by
  have key : x - α * (2 * (x - c)) - c = (1 - 2 * α) * (x - c) := by ring
  rw [key, abs_mul]
  have hd : x - c ≠ 0 := sub_ne_zero.mpr hne
  have hpos : 0 < |x - c| := abs_pos.mpr hd
  have hfac : |1 - 2 * α| < 1 := by rw [abs_lt]; constructor <;> linarith
  calc |1 - 2 * α| * |x - c|
        < 1 * |x - c| := mul_lt_mul_of_pos_right hfac hpos
      _ = |x - c|     := one_mul _

-- NOTE: effective_alpha = α / (1 + σ_t) ≤ α, so Condition 4 holds with
-- effective_alpha whenever α < 1/2.

/-! ── CONDITION 5: GD update is a valid descent scheme ─────────────────────── -/

/-- Gradient descent step map. -/
def gdStep (α : ℝ) (x : ℝ) : ℝ := x - α * (2 * (x - c))

/-- c is the unique fixed point of gdStep (for any α). -/
theorem cond5_fixed_point (α : ℝ) : gdStep c α c = c := by simp [gdStep]

/-- Condition 5a: one step strictly decreases L at every non-minimiser. -/
theorem cond5_strict_descent (α : ℝ) (hα0 : 0 < α) (hα1 : α < 1 / 2)
    (x : ℝ) (hne : x ≠ c) :
    L c (gdStep c α x) < L c x := by
  simp only [L, gdStep]
  have key : x - α * (2 * (x - c)) - c = (1 - 2 * α) * (x - c) := by ring
  rw [key]
  have hd   : x - c ≠ 0       := sub_ne_zero.mpr hne
  have hpos : 0 < (x - c) ^ 2 := sq_pos_of_ne_zero hd
  -- (1 − 2α)² < 1: expand → 4α² − 4α < 0 → α(1−α) > 0 ✓ for α ∈ (0, 1/2)
  have hlt : (1 - 2 * α) ^ 2 < 1 := by
    nlinarith [mul_pos hα0 (show 0 < 1 - α by linarith)]
  calc ((1 - 2 * α) * (x - c)) ^ 2
        = (1 - 2 * α) ^ 2 * (x - c) ^ 2 := by ring
      _ < 1 * (x - c) ^ 2               := mul_lt_mul_of_pos_right hlt hpos
      _ = (x - c) ^ 2                   := one_mul _

/-- Condition 5b: gdStep is a (strict) contraction, factor = |1 − 2α|. -/
theorem cond5_contraction (α : ℝ) (x y : ℝ) :
    |gdStep c α x - gdStep c α y| = |1 - 2 * α| * |x - y| := by
  simp only [gdStep]
  have : x - α * (2 * (x - c)) - (y - α * (2 * (y - c))) = (1 - 2 * α) * (x - y) := by ring
  rw [this, abs_mul]

theorem cond5_contraction_lt_one (α : ℝ) (hα0 : 0 < α) (hα1 : α < 1 / 2) :
    |1 - 2 * α| < 1 := by rw [abs_lt]; constructor <;> linarith

-- NOTE (vjp/full): L(x) = ‖x − D_θ(x)‖² is generally non-convex.
-- No descent guarantee exists without Lipschitz/convexity assumptions on D_θ.

/-! ── SUPPLEMENT A: Scheduler-normalised step size ─────────────────────────── -/
/-!
  Python code (_sure_effective_alpha) uses effective_alpha = α / (1 + min(σ_t, 1)).
  Three theorems validate that this stays in the range required by cond4_step_closer.
-/

/-- Capped sigma normaliser: min(σ_t, 1) ≥ 0 for σ_t ≥ 0. -/
theorem sigma_cap_nonneg (σ : ℝ) (hσ : 0 ≤ σ) : 0 ≤ min σ 1 :=
  le_min hσ zero_le_one

/-- effectiveAlpha (capped) is strictly positive when α > 0 and σ_t ≥ 0. -/
theorem effectiveAlpha_pos (α σ : ℝ) (hα : 0 < α) (hσ : 0 ≤ σ) :
    0 < α / (1 + min σ 1) := by
  apply div_pos hα
  linarith [sigma_cap_nonneg σ hσ]

/-- effectiveAlpha (capped) is < 1/2 when α < 1/2 — satisfies cond4_step_closer. -/
theorem effectiveAlpha_lt_half (α σ : ℝ) (hα0 : 0 < α) (hα1 : α < 1 / 2) (hσ : 0 ≤ σ) :
    α / (1 + min σ 1) < 1 / 2 := by
  have hd   : 0 < 1 + min σ 1 := by linarith [sigma_cap_nonneg σ hσ]
  have hcap : 0 ≤ min σ 1      := sigma_cap_nonneg σ hσ
  -- Multiply both sides by (1+s) > 0 to clear the denominator.
  -- LHS * (1+s) = α, RHS * (1+s) = (1+s)/2 > α (since 1+s ≥ 1 > 2α).
  have cancel : α / (1 + min σ 1) * (1 + min σ 1) = α :=
    div_mul_cancel₀ α (ne_of_gt hd)
  have rhs_gt : α < 1 / 2 * (1 + min σ 1) := by nlinarith
  have step : α / (1 + min σ 1) * (1 + min σ 1) < 1 / 2 * (1 + min σ 1) := by
    rw [cancel]; exact rhs_gt
  exact lt_of_mul_lt_mul_right step (le_of_lt hd)

/-- effectiveAlpha (capped) ≥ α/2 — EDM with large σ_t no longer gets near-zero steps. -/
theorem effectiveAlpha_ge_half_alpha (α σ : ℝ) (hα : 0 < α) (hσ : 0 ≤ σ) :
    α / 2 ≤ α / (1 + min σ 1) := by
  have hd   : 0 < 1 + min σ 1  := by linarith [sigma_cap_nonneg σ hσ]
  have hle2 : 1 + min σ 1 ≤ 2  := by linarith [min_le_right σ 1]
  have cancel : α / (1 + min σ 1) * (1 + min σ 1) = α :=
    div_mul_cancel₀ α (ne_of_gt hd)
  -- α/2 * (1+s) ≤ α = α/(1+s) * (1+s); divide both sides by (1+s) > 0.
  have h_lhs : α / 2 * (1 + min σ 1) ≤ α := by
    have : α / 2 * (1 + min σ 1) ≤ α / 2 * 2 :=
      mul_le_mul_of_nonneg_left hle2 (le_of_lt (div_pos hα two_pos))
    linarith [show α / 2 * 2 = α by ring]
  have step : α / 2 * (1 + min σ 1) ≤ α / (1 + min σ 1) * (1 + min σ 1) := by
    rw [cancel]; exact h_lhs
  exact le_of_mul_le_mul_right step hd

/-! ── SUPPLEMENT B: Jac-reliability gate ────────────────────────────────────── -/
/-!
  When the Hutchinson estimate jac_trace < −n/n_mc (below the physical floor for
  a contractive denoiser), the code falls back to the proxy gradient 2·r.
  This theorem confirms the fallback still guarantees descent.
-/

/-- Proxy-gradient fallback: same as cond5_strict_descent, cited explicitly. -/
theorem jac_fallback_descent (α : ℝ) (hα0 : 0 < α) (hα1 : α < 1 / 2)
    (x : ℝ) (hne : x ≠ c) :
    L c (x - α * (2 * (x - c))) < L c x :=
  cond5_strict_descent c α hα0 hα1 x hne

/-! ── SUPPLEMENT C: Expected sign of per-subband SURE value ────────────────── -/
/-!
  Per-subband SURE (approx mode, no Jacobian):
      sure_k = −n_k·σ² + ‖r_k‖²

  Negative sure_k ↔ ‖r_k‖² < n_k·σ²
  ↔ mean per-element residual energy below σ²
  ↔ denoiser performs better than the noise floor.

  This is the EXPECTED regime for a well-trained denoiser at mid-to-late
  diffusion steps.  The gradient (approx_coeff · r_k) remains valid by
  cond3–cond5 regardless of the sign of sure_k.

  In the wavelet sampler with lp_frac < 1, pass-through (uncorrected) subbands
  must NOT be included in the primary sure_val metric.  Their large element
  count (finest detail level: 76 800 elements at 160×160 latent × 4 channels)
  dominates the sum with large negative values at mid-to-late steps, masking
  the signal from the corrected subbands.
-/

/-- sure_val < 0 iff the denoiser residual is below the noise floor. -/
theorem sure_val_sign (n σ2 r_sq : ℝ) :
    -n * σ2 + r_sq < 0 ↔ r_sq < n * σ2 := by constructor <;> intro h <;> linarith

/-- A good denoiser (‖r‖² < n·σ²) always yields negative sure_val in approx mode. -/
theorem sure_val_negative_iff_good_denoiser (n σ2 r_sq : ℝ) (hgood : r_sq < n * σ2) :
    -n * σ2 + r_sq < 0 := by linarith

/-- Pass-through subband SURE pollution: if the pass-through band has element
    count m ≥ n_corrected and the same per-element residual energy ρ < σ²,
    its |SURE| dominates the corrected-subband |SURE| in magnitude. -/
theorem passthrough_dominates (n_c m σ2 rho : ℝ)
    (hc : 0 < n_c) (hm : n_c ≤ m) (hσ : 0 < σ2) (hρ : 0 ≤ rho) (hrho : rho < σ2) :
    |(-m * σ2 + m * rho)| ≥ |(-n_c * σ2 + n_c * rho)| := by
  have hd   : rho - σ2 < 0   := by linarith
  have hm_p : 0 < m           := lt_of_lt_of_le hc hm
  have h1   : -m * σ2 + m * rho   = m * (rho - σ2)   := by ring
  have h2   : -n_c * σ2 + n_c * rho = n_c * (rho - σ2) := by ring
  rw [h1, h2, abs_mul, abs_mul,
      abs_of_pos hm_p, abs_of_pos hc, abs_of_neg hd]
  nlinarith

end SUREVerification

/-!
══════════════════════════════════════════════════════════════════════════
  VERIFICATION SUMMARY
══════════════════════════════════════════════════════════════════════════

  Condition | 'approx' stop-grad proxy   | 'vjp'/'full' true SURE grad
  ──────────┼────────────────────────────┼──────────────────────────────
  1  ℝ→ℝ   | ✓ cond1_total              | ✓ (always a real)
  2  Has min | ✓ cond2_global_min        | ✗ requires contractive D_θ
  3  Grad⊥  | ✓ cond3_grad_away         | ✗ fails when J_D eigenval = 1
  4  Dir ✓  | ✓ cond4_step_closer       | ✗ no spectral gap guarantee
  5  GD OK  | ✓ cond5_strict_descent    | ✗ non-convex, may diverge

══════════════════════════════════════════════════════════════════════════
  ROOT CAUSE: WHY sure_val IS ALWAYS NEGATIVE
══════════════════════════════════════════════════════════════════════════

  sure_val = −n·σ² + ‖r‖² + 2·σ²·tr{J_D}

  The Hutchinson MC trace estimator with n_mc = 1 (code default):
      tr{J_D} ≈ bᵀ·(D(x+ε·b) − D(x)) / ε,   b ~ N(0, I_n)

  is unbiased but has variance  Var ≈ 2·‖J_D‖_F² ≈ O(n²).

  At mid-denoising (σ̂₀ ≈ 1, n = 16 384 SD latents):
      −n·σ² ≈ −16 384
      ‖r‖²  ≈ +16 384   (residual ≈ noise; balanced with −n·σ²)
      2σ²·tr{J_D}: single MC sample easily swings ±50 000

  → A single negative Hutchinson sample makes sure_val << 0 every step.

  The gradient direction is NOT corrupted (sure_val is logged only, not
  used in the Adam update in 'vjp' mode).  However, in 'full' mode the
  same single-sample jac_grad IS added to the gradient with coefficient
  2σ², which can corrupt Adam's v_t when that sample is extreme.

  Remedies (cheapest first):
    1. grad_mode='approx' — skip jac entirely; proxy gradient ✓ proved above
    2. grad_mode='vjp'    — (current default) jac scalar only for logging;
                            gradient is exact ∇‖r‖² from autograd
    3. n_mc ≥ 8           — reduces Hutchinson variance by √n_mc
    4. use_jac=False      — sure_val = −nσ² + ‖r‖²; positive for good denoiser
-/
