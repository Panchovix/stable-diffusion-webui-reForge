-- RFVProofs/CompositeError.lean
-- Three results about the composite error budget for the CLPC sampler:
-- (1) clip_weight_vanishes_at_high_sigma: exponential scheduling weight → 0
-- (2) ode_error_bounded_by_composite: weighted sum bound → per-term bound
-- (3) cfg_drift_bounded_in_unit: CFG cosine drift ∈ [0, 2]

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp

open Real Filter Topology

namespace RFVProofs

/-! ## (1) Exponential clip-weight vanishes as σ → ∞ -/

/-- The product `w * exp(-σ/τ)` tends to 0 as `σ → +∞`, for any fixed
    `w : ℝ` and `τ > 0`. This formalizes the clip-weight schedule: at high
    noise levels the exponential scheduling term drives the composite weight
    to zero, so large-σ steps no longer contribute to the error budget.

    Proof path: `σ/τ → +∞` (via `Filter.Tendsto.atTop_div_const`), then
    `exp(-·)` composed with that goes to 0 (via
    `Real.tendsto_exp_neg_atTop_nhds_zero`), then multiply by constant `w`
    (via `Tendsto.const_mul`). -/
theorem clip_weight_vanishes_at_high_sigma (w τ : ℝ) (hτ : 0 < τ) :
    Tendsto (fun σ => w * Real.exp (-σ / τ)) atTop (nhds 0) := by
  -- Step 1: σ/τ → +∞
  have h1 : Tendsto (fun σ : ℝ => σ / τ) atTop atTop :=
    Filter.Tendsto.atTop_div_const hτ tendsto_id
  -- Step 2: exp(-x) → 0 at +∞
  have h2 : Tendsto (fun x : ℝ => Real.exp (-x)) atTop (nhds 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero
  -- Step 3: exp(-σ/τ) → 0 by composition
  have h3 : (fun σ : ℝ => Real.exp (-σ / τ)) = (fun σ => Real.exp (-(σ / τ))) := by
    ext σ; ring_nf
  have hexp : Tendsto (fun σ : ℝ => Real.exp (-σ / τ)) atTop (nhds 0) := by
    rw [h3]; exact h2.comp h1
  -- Step 4: multiply by constant w; w * 0 = 0
  have hmul := hexp.const_mul w
  simp at hmul
  exact hmul

/-! ## (2) ODE error bounded by composite weighted sum -/

/-- If the composite error budget `w_ode * e_ode + rest ≤ tol` with `w_ode > 0`,
    then the ODE component satisfies `e_ode ≤ (tol - rest) / w_ode`.
    This is the per-component extraction from a weighted-sum error budget:
    knowing the full budget dominates any single weighted term lets us
    read off the individual error tolerance.

    Proof: rearrange `w_ode * e_ode ≤ tol - rest` then divide; `linarith`. -/
theorem ode_error_bounded_by_composite
    (w_ode e_ode tol rest : ℝ)
    (hw : 0 < w_ode)
    (hbound : w_ode * e_ode + rest ≤ tol) :
    e_ode ≤ (tol - rest) / w_ode := by
  rw [le_div_iff₀ hw]
  linarith

/-- Corollary: if rest ≥ 0, the bound simplifies to tol / w_ode.
    This covers the standard case where all other error terms are non-negative. -/
theorem ode_error_bounded_by_tol
    (w_ode e_ode tol rest : ℝ)
    (hw : 0 < w_ode)
    (hrest : 0 ≤ rest)
    (hbound : w_ode * e_ode + rest ≤ tol) :
    e_ode ≤ tol / w_ode := by
  have h := ode_error_bounded_by_composite w_ode e_ode tol rest hw hbound
  have : (tol - rest) / w_ode ≤ tol / w_ode := by
    apply div_le_div_of_nonneg_right _ (le_of_lt hw)
    linarith
  linarith

/-! ## (3) CFG cosine drift bounded in [0, 2] -/

/-- The cosine drift `1 - c` lies in `[0, 2]` whenever the cosine similarity
    `c ∈ [-1, 1]`. In the CLPC context, `c = cos_sim(g_t, g_{t-1})` where
    `g_t` is the CFG gradient at step `t`; the drift measures how much the
    guidance direction rotates between consecutive steps.
    - Lower bound 0: achieved at `c = 1` (perfectly aligned guidance, no drift)
    - Upper bound 2: achieved at `c = -1` (gradient reversal, maximum drift) -/
theorem cfg_drift_bounded_in_unit (c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    1 - c ∈ Set.Icc (0 : ℝ) 2 := by
  obtain ⟨hlo, hhi⟩ := hc
  constructor <;> [linarith; linarith]

/-- The drift is non-negative: cosine similarity ≤ 1 means guidance cannot
    "more than align", so `1 - c ≥ 0`. -/
theorem cfg_drift_nonneg (c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    0 ≤ 1 - c := (cfg_drift_bounded_in_unit c hc).1

/-- The drift is at most 2: even at full reversal the drift is bounded. -/
theorem cfg_drift_upper_bound (c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    1 - c ≤ 2 := (cfg_drift_bounded_in_unit c hc).2

end RFVProofs
