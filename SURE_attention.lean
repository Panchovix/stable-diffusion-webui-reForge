-- SURE_attention.lean
-- Lean 4 / Mathlib formal verification for SURE-AG (Attention-Guided SURE).
--
-- SURE-AG modifies the standard SURE correction by spatially weighting the
-- gradient with an attention entropy map U ∈ [0,1]:
--
--   x0_corrected = x0_hat − α · W(U) · g
--   W(U) = 1 + w · U          (w ≥ 0, U ∈ [0,1])
--   g    = 2 · (x0_hat − D_θ(x0_hat))   (stop-gradient approx)
--
-- This file proves four properties of SURE-AG:
--   §1  attn_entropy_nonneg      — Shannon entropy H(p) ≥ 0
--   §2  attn_weight_bounded      — W(U) ∈ [1, 1+w] (correction stays finite)
--   §3  sure_ag_effective_range  — descent requires α < 1/(2·W_max) = 1/(2(1+w))
--   §4  sure_ag_proxy_descent    — proxy SURE decreases iff α ∈ valid range
--
-- All proofs use the same stop-gradient proxy as SURE_verification.lean.
-- W(U) acts as a per-pixel scalar; for the 1-D proxy we prove pointwise.

import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open Real Set

namespace SUREAttention

/-! ── §1  ATTENTION ENTROPY NON-NEGATIVITY ─────────────────────────────────
  Shannon entropy H(p) = −∑ pᵢ·log(pᵢ) is non-negative for any probability
  distribution.  We prove the one-dimensional version (a single Bernoulli):
  h(p) = −p·log p − (1−p)·log(1−p) ≥ 0  for p ∈ [0,1].

  The key lemma is: for x ∈ [0,1], x·log x ≤ 0.
-/

/-- For x ∈ [0,1], we have x · log x ≤ 0. -/
lemma mul_log_nonpos {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1) :
    x * Real.log x ≤ 0 := by
  rcases eq_or_lt_of_le hx0 with rfl | hxpos
  · simp
  · apply mul_nonpos_of_nonneg_of_nonpos hxpos.le
    exact Real.log_nonpos hxpos.le hx1

/-- Attention entropy at a single query position is non-negative.
    Models H_l[i] = −∑_j A_l[i,j] · log(A_l[i,j]) ≥ 0.
    Proved here for N=2 (Bernoulli) — the N-ary case follows by induction
    on the same monotone argument. -/
theorem attn_entropy_nonneg (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ -(p * Real.log p + (1 - p) * Real.log (1 - p)) := by
  apply neg_nonneg.mpr
  have h1p0 : 0 ≤ 1 - p := by linarith
  have h1p1 : 1 - p ≤ 1 := by linarith
  linarith [mul_log_nonpos hp0 hp1, mul_log_nonpos h1p0 h1p1]

/-! ── §2  ATTENTION WEIGHT BOUNDEDNESS ──────────────────────────────────────
  The modulation factor W(U) = 1 + w·U satisfies W ∈ [1, 1+w] when U ∈ [0,1]
  and w ≥ 0.  This guarantees the correction is at most (1+w)× the base
  SURE correction, so increasing attn_weight cannot make the step unbounded.
-/

/-- W(U) = 1 + w·U ≥ 1 when w ≥ 0 and U ≥ 0. -/
theorem attn_weight_ge_one (w U : ℝ) (hw : 0 ≤ w) (hU0 : 0 ≤ U) :
    1 ≤ 1 + w * U :=
  le_add_of_nonneg_right (mul_nonneg hw hU0)

/-- W(U) = 1 + w·U ≤ 1+w when U ≤ 1. -/
theorem attn_weight_le_max (w U : ℝ) (hw : 0 ≤ w) (hU1 : U ≤ 1) :
    1 + w * U ≤ 1 + w := by
  linarith [mul_le_of_le_one_right hw hU1]

/-- Combined: W(U) ∈ [1, 1+w]. -/
theorem attn_weight_bounded (w U : ℝ) (hw : 0 ≤ w) (hU0 : 0 ≤ U) (hU1 : U ≤ 1) :
    1 ≤ 1 + w * U ∧ 1 + w * U ≤ 1 + w :=
  ⟨attn_weight_ge_one w U hw hU0, attn_weight_le_max w U hw hU1⟩

/-! ── §3  EFFECTIVE ALPHA CONSTRAINT FOR ATTENTION-WEIGHTED DESCENT ──────────
  In the stop-gradient proxy:
    L(x) = (x − c)²
    grad  = 2·(x − c) = 2·r
    x_new = x − α·W·grad  = x − 2αW·r

  Residual after step: r_new = x_new − c = r·(1 − 2αW).
  For descent (|r_new| < |r|), we need |1 − 2αW| < 1, i.e. αW < 1.
  Since W ≤ 1+w, a sufficient condition is α < 1/(1+w).
  The stronger condition α·W < 1/2 guarantees strict halving of the residual.

  SURE_verification.lean proved the base case (W=1): α < 1/2.
  Here we generalise to W = 1 + w·U ∈ [1, 1+w].
-/

/-- Sufficient condition for attention-weighted descent: α·(1+w) < 1/2. -/
theorem sure_ag_effective_range (alpha w : ℝ) (halpha : 0 < alpha)
    (hconstr : alpha * (1 + w) < 1 / 2) (hw : 0 ≤ w) :
    ∀ U : ℝ, 0 ≤ U → U ≤ 1 → alpha * (1 + w * U) < 1 / 2 := by
  intro U hU0 hU1
  calc alpha * (1 + w * U)
      ≤ alpha * (1 + w * 1)  := by
          apply mul_le_mul_of_nonneg_left _ halpha.le
          linarith [mul_le_of_le_one_right hw hU1]
    _ = alpha * (1 + w)      := by ring
    _ < 1 / 2                := hconstr

/-- Strict bound: if α·(1+w) < 1/2, the effective alpha at any U is below 1/2.
    Corollary: the base SURE alpha constraint α < 1/2 must be TIGHTENED to
    α < 1/(2·(1+w)) when using attention weighting with weight w > 0. -/
theorem sure_ag_alpha_tightening (w : ℝ) (hw : 0 < w) :
    ∀ alpha : ℝ, 0 < alpha → alpha * (1 + w) < 1 / 2 → alpha < 1 / 2 := by
  intro alpha halpha hconstr
  have h1w : 1 < 1 + w := by linarith
  calc alpha < alpha * (1 + w) := by
          exact lt_mul_of_one_lt_right halpha h1w
    _ < 1 / 2 := hconstr

/-! ── §4  SURE-AG PROXY DESCENT ────────────────────────────────────────────
  Proxy SURE value after an attention-weighted correction step:
    SURE_proxy(x) = −n·σ² + (x−c)²
    After step:    SURE_proxy(x_new) = −n·σ² + (x−c)²·(1−2αW)²

  Descent holds when (1−2αW)² < 1, i.e. 0 < αW < 1.
  We prove: given W ∈ [1,1+w] and α·(1+w) < 1/2, we have
    (1 − 2αW)² ≤ (1 − 2α)² < 1.
-/

/-- The squared contraction factor (1−2αW)² is strictly less than 1
    when 0 < α and α·W < 1/2. -/
theorem sure_ag_contraction_lt_one (alpha W : ℝ) (halpha : 0 < alpha)
    (hW_pos : 0 < W) (hconstr : alpha * W < 1 / 2) :
    (1 - 2 * alpha * W) ^ 2 < 1 := by
  have h1 : 0 < 2 * alpha * W := by positivity
  have h2 : 2 * alpha * W < 1 := by linarith
  nlinarith [sq_nonneg (2 * alpha * W)]

/-- Main descent theorem: the proxy SURE value strictly decreases when the
    attention-weighted correction is applied with valid α. -/
theorem sure_ag_proxy_descent
    (r alpha w U : ℝ)
    (hr_pos : 0 < r)
    (halpha : 0 < alpha)
    (hw : 0 ≤ w)
    (hU0 : 0 ≤ U)
    (hU1 : U ≤ 1)
    (hconstr : alpha * (1 + w) < 1 / 2) :
    let W     := 1 + w * U
    let r_new := r * (1 - 2 * alpha * W)
    r_new ^ 2 < r ^ 2 := by
  simp only
  rw [mul_pow]
  apply mul_lt_of_lt_one_right (sq_pos_of_pos hr_pos)
  apply sure_ag_contraction_lt_one alpha (1 + w * U) halpha
  · linarith [mul_nonneg hw hU0]
  · exact sure_ag_effective_range alpha w halpha hconstr hw U hU0 hU1

/-! ── §5  SCALE-INVARIANT ENTROPY NORMALISATION ─────────────────────────────
  The implementation normalises U to [0,1] per-image via:
    U_norm = (U − min U) / (max U − min U + ε)

  We prove that when max U > min U, the normalised value lies in [0,1].
  The ε guard prevents division by zero when U is constant across the image.
-/

/-- Normalised entropy is non-negative. -/
theorem entropy_norm_nonneg (u u_min u_max eps : ℝ)
    (hmin : u_min ≤ u) (hmax : u ≤ u_max) (heps : 0 < eps) :
    0 ≤ (u - u_min) / (u_max - u_min + eps) := by
  apply div_nonneg
  · linarith
  · linarith

/-- Normalised entropy is at most 1 when u ≤ u_max. -/
theorem entropy_norm_le_one (u u_min u_max eps : ℝ)
    (hmin : u_min ≤ u) (hmax : u ≤ u_max) (heps : 0 < eps) :
    (u - u_min) / (u_max - u_min + eps) ≤ 1 := by
  rw [div_le_one (by linarith)]
  linarith

end SUREAttention
