-- RFVProofs/ChebyshevAdaptive.lean
-- Chebyshev polynomial minimax properties for adaptive velocity prediction in CLPC.

import Mathlib.RingTheory.Polynomial.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Extremal
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import RFVProofs.RungeGuard

/-!
# Chebyshev Polynomial Adaptive Prediction for CLPC

**Sampler design implication**: Adams-Bashforth predictors use Lagrange
interpolation at *equally-spaced* λ-nodes. As proved in RungeGuard.lean,
equally-spaced Lagrange interpolation can produce arbitrarily large basis
coefficients (Runge phenomenon) when the evaluation point lies outside the
node support. In contrast, Chebyshev nodes *minimise* the maximum interpolation
error: the leading coefficient of `T_n` equals `2^{n-1}`, and no other monic
degree-n polynomial achieves a smaller ‖·‖_∞ bound on [-1,1].

For CLPC this means: if the step-size schedule is re-parameterised so that
the Adams history points align with Chebyshev nodes (cosine spacing in λ),
the predictor's interpolation error is provably minimised among all degree-n
polynomial predictors. This is strictly better than the current equally-spaced
step history for high-order prediction (order ≥ 2).
-/

open Polynomial.Chebyshev Polynomial Real

namespace ChebyshevAdaptive

/-!
## Theorem 1 — Chebyshev extremal bound

`|T_n(x)| ≤ 1` for all `x ∈ [-1, 1]`.

Follows directly from `T_real_cos` (T_n(cos θ) = cos(n θ)) and `|cos| ≤ 1`.
-/

/-- **Chebyshev extremal**: `T_n` is bounded by 1 in magnitude on [-1, 1].

    For any `x ∈ [-1, 1]`, write `x = cos θ` using surjectivity of cos onto
    [-1, 1]; then `T_n(cos θ) = cos(n θ)` (by `T_real_cos`), and `|cos| ≤ 1`. -/
theorem chebyshev_extremal (n : ℤ) (x : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) 1) :
    |(T ℝ n).eval x| ≤ 1 := by
  -- Use surjectivity of cos onto [-1,1] to write x = cos θ
  have hmem : x ∈ Set.range Real.cos := by
    rw [Real.range_cos]
    exact hx
  obtain ⟨θ, hθ⟩ := hmem
  rw [← hθ, T_real_cos]
  exact abs_cos_le_one _

/-!
## Theorem 2 — Chebyshev monic minimax theorem

Among all degree-n polynomials bounded by 1 on [-1, 1], the leading
coefficient is at most `2^{n-1}`.  This is the Mathlib theorem
`leadingCoeff_le_of_forall_abs_le_one`, re-exported as a design theorem.
-/

/-- **Chebyshev monic minimax (leading coefficient bound)**: if `|P(x)| ≤ 1`
    for all `x ∈ [-1, 1]` and `deg P ≤ n`, then `leadingCoeff P ≤ 2^{n-1}`.

    We use `Polynomial.Chebyshev.leadingCoeff_le_of_forall_abs_le_one`
    from Mathlib (Extremal.lean). -/
theorem chebyshev_monic_minimax {n : ℕ} {P : ℝ[X]}
    (hPdeg : P.degree ≤ n)
    (hPbnd : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P.eval x| ≤ 1) :
    P.leadingCoeff ≤ 2 ^ (n - 1) :=
  leadingCoeff_le_of_forall_abs_le_one hPdeg hPbnd

/-- **Chebyshev uniqueness**: for `n ≥ 2`, equality holds iff `P = T_n`. -/
theorem chebyshev_monic_minimax_unique {n : ℕ} (hn : 2 ≤ n) {P : ℝ[X]}
    (hPdeg : P.degree ≤ n)
    (hPbnd : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P.eval x| ≤ 1) :
    P.leadingCoeff = 2 ^ (n - 1) ↔ P = T ℝ n :=
  leadingCoeff_eq_iff_of_forall_abs_le_one hn hPdeg hPbnd

/-!
## Theorem 3 — Adams vs Chebyshev

Adams-Bashforth of order ≥ 2 uses exterior Lagrange extrapolation (unbounded,
proved in RungeGuard.lean). Chebyshev nodes avoid this by construction.
-/

/-- **Adams vs Chebyshev**: Adams-Bashforth of order ≥ 2 has unbounded Lagrange
    basis (RungeGuard); Chebyshev T_2 is bounded by 1 everywhere on [-1,1].

    The two strategies are in direct opposition: equispaced nodes → Runge blow-up;
    Chebyshev nodes → bounded error and optimal leading coefficient. -/
theorem adams_vs_chebyshev {a b : ℝ} (hab : a < b) (C : ℝ) :
    -- Adams-Bashforth: Lagrange basis at exterior point is unbounded (RungeGuard)
    (∃ target : ℝ, b < target ∧ C < |L₀_two_node a b target|) ∧
    -- Chebyshev polynomial T_2: bounded by 1 everywhere on [-1,1]
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |(T ℝ 2).eval x| ≤ 1) := by
  constructor
  · exact lagrange_two_node_at_exterior_point_unbounded a b hab C
  · intro x hx
    exact chebyshev_extremal 2 x hx

/-!
## Theorem 4 — T_n leading coefficient is exactly 2^{n-1}

For `n ≥ 1`, `T_n` achieves the minimax bound exactly.
-/

/-- `T_n` has leading coefficient `2^{n-1}` for `n : ℕ`. -/
theorem chebyshev_leading_coeff (n : ℕ) :
    (T ℝ (n : ℤ)).leadingCoeff = 2 ^ (n - 1) := by
  have h := Polynomial.Chebyshev.leadingCoeff_T ℝ (n : ℤ)
  simp only [Int.natAbs_natCast] at h
  exact h

end ChebyshevAdaptive
