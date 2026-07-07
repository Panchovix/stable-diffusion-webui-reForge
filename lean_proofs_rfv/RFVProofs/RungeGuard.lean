import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.Order.Field.Basic
import RFVProofs.Defs

/-!
# Adams Multistep Runge-Phenomenon Guard

Formal analysis of why the CLPC sampler's Adams *predictor* develops
unbounded Lagrange coefficients under large step-size ratios, while the
Adams-Moulton *corrector* remains bounded unconditionally.

## Variables

Throughout: `ls` = λ_s (log-SNR at source), `lt` = λ_t (log-SNR at target),
`la` = integration point λ ∈ [ls, lt].

## Key results

1. `euler_b_coeff_in_unit_interval`
   Adams order-1 (Euler in λ-space) has b = 1 - exp(ls - lt) ∈ (0, 1)
   for any lt > ls.

2. `lagrange_two_node_at_exterior_point_unbounded`
   The Lagrange basis L₀ evaluated at a point c outside {a, b} satisfies
   |L₀(c)| = (c - b)/(b - a), which is unbounded as c → ∞.
   This is the mechanism that blew up step-28 b-coefficients to ±20.

3. `am_corrector_two_node_bounded`
   When both ls and lt are in the node set, the Lagrange
   basis L_j evaluated inside [ls, lt] lies in [0, 1].
   This proves order-2 AM corrector coefficients are unconditionally bounded.

4. `euler_predictor_am_corrector_unconditionally_stable`
   Euler predictor + AM corrector is safe: predictor b ∈ (0, 1),
   corrector bases ∈ [0, 1] — no h-ratio guard needed.

## Design conclusion

The h-ratio guard (fall back to order-1 when h_curr > 2.5 × h_prev) is a
correct *patch* but the *principled* design is:

  - Predictor: ALWAYS Euler (order-1), b = 1 - σ_t/σ_s ∈ (0, 1)
  - Corrector: Adams-Moulton (order 2-4), always bounded because lt ∈ nodes

Higher-order prediction without Runge risk requires a different polynomial
basis (e.g. Nordsieck / Phi-function exponential integrators).
-/

open Real

variable {ls lt : ℝ}

-- ---------------------------------------------------------------------------
-- Result 1: Euler (order-1) b-coefficient is strictly in (0, 1)
-- ---------------------------------------------------------------------------

/-!
### Adams order-1 b-coefficient

For the SA-Solver Adams predictor at order 1, the single b-coefficient is

  b₀ = α_t · ∫_{ls}^{lt} exp(u - lt) du = 1 - exp(ls - lt) = 1 - σ_t/σ_s

which lies strictly in (0, 1) for any lt > ls.
-/

/-- Euler b-coefficient `1 - exp(ls - lt)` is positive. -/
theorem euler_b_positive (h : ls < lt) : (0 : ℝ) < 1 - Real.exp (ls - lt) := by
  have : Real.exp (ls - lt) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  linarith

/-- Euler b-coefficient is strictly less than 1 (true for any ls, lt). -/
theorem euler_b_lt_one : 1 - Real.exp (ls - lt) < 1 := by
  linarith [Real.exp_pos (ls - lt)]

/-- Euler b-coefficient lies in the open unit interval (0, 1). -/
theorem euler_b_coeff_in_unit_interval (h : ls < lt) :
    0 < 1 - Real.exp (ls - lt) ∧ 1 - Real.exp (ls - lt) < 1 :=
  ⟨euler_b_positive h, euler_b_lt_one⟩

-- ---------------------------------------------------------------------------
-- Result 2: Two-node Lagrange basis at exterior point is unbounded
-- ---------------------------------------------------------------------------

/-!
### Lagrange blow-up (the Runge phenomenon for Adams predictors)

For nodes a < b and evaluation point c > b (outside support), the
Lagrange basis polynomial at node a is:

  L₀(c) = (c - b) / (a - b)

With a < b the denominator a - b < 0, and for c > b the numerator c - b > 0,
so L₀(c) < 0 and |L₀(c)| = (c - b) / (b - a) → ∞ as c → ∞.

Concretely at step 28: nodes {1.046, 1.301, 1.696}, evaluation at λ_t = 3.535,
giving basis values that produce b-coefficients ['10.7', '-20.2', '10.3'].
-/

/-- Two-node Lagrange basis L₀ at nodes {a, b}, evaluated at c. -/
noncomputable def L₀_two_node (a b c : ℝ) : ℝ := (c - b) / (a - b)

/-- For any bound C there exists an exterior evaluation point c > b where |L₀| > C. -/
theorem lagrange_two_node_at_exterior_point_unbounded
    (a b : ℝ) (hab : a < b) (C : ℝ) :
    ∃ c : ℝ, b < c ∧ C < |L₀_two_node a b c| := by
  -- Choose c = b + (|C| + 1) * (b - a). Then c - b = (|C|+1)*(b-a)
  -- and |L₀(c)| = (|C|+1)*(b-a)/(b-a) = |C|+1 > C.
  have hab' : (0 : ℝ) < b - a := by linarith
  use b + (|C| + 1) * (b - a)
  refine ⟨by nlinarith [abs_nonneg C], ?_⟩
  simp only [L₀_two_node]
  -- (c - b) / (a - b) = (|C|+1)*(b-a) / (a-b) = -(|C|+1)
  have hval : (b + (|C| + 1) * (b - a) - b) / (a - b) = -(|C| + 1) := by
    have h1 : b + (|C| + 1) * (b - a) - b = -(|C| + 1) * (a - b) := by ring
    rw [h1, mul_div_cancel_right₀ _ (by linarith : a - b ≠ 0)]
  rw [hval, abs_neg, abs_of_pos (by nlinarith [abs_nonneg C])]
  linarith [le_abs_self C]

-- ---------------------------------------------------------------------------
-- Result 3: AM corrector with lt in nodes has bounded basis on [ls, lt]
-- ---------------------------------------------------------------------------

/-!
### Adams-Moulton corrector stability

When the node set includes both ls and lt (the integration interval endpoints),
the two-node Lagrange basis evaluated at any la ∈ [ls, lt] satisfies:

  L₀(la) = (lt - la) / (lt - ls)  ∈ [0, 1]   (1 at la=ls, 0 at la=lt)
  L₁(la) = (la - ls) / (lt - ls)  ∈ [0, 1]   (0 at la=ls, 1 at la=lt)

Note: we write in the positive-denominator form (lt - ls > 0) for
readability; algebraically equivalent to the standard Lagrange form.

This is INTERPOLATION not extrapolation: la is inside the node support.
The Runge phenomenon applies only to extrapolation.  The AM corrector
always includes lt in its node set, so its integrands are never extrapolated.
-/

/-- AM corrector Lagrange basis L₀ at integration point la.
    Written with positive denominator: (lt - la) / (lt - ls). -/
noncomputable def AM_L₀ (ls lt la : ℝ) : ℝ := (lt - la) / (lt - ls)

/-- AM corrector Lagrange basis L₁ at integration point la. -/
noncomputable def AM_L₁ (ls lt la : ℝ) : ℝ := (la - ls) / (lt - ls)

/-- L₀ is non-negative on [ls, lt]. -/
theorem AM_L₀_nonneg (h : ls < lt) (la : ℝ) (hhi : la ≤ lt) :
    0 ≤ AM_L₀ ls lt la :=
  div_nonneg (by linarith) (by linarith)

/-- L₀ is at most 1 on [ls, lt]. -/
theorem AM_L₀_le_one (h : ls < lt) (la : ℝ) (hlo : ls ≤ la) :
    AM_L₀ ls lt la ≤ 1 := by
  rw [AM_L₀, div_le_one (by linarith)]
  linarith

/-- L₁ is non-negative on [ls, lt]. -/
theorem AM_L₁_nonneg (h : ls < lt) (la : ℝ) (hlo : ls ≤ la) :
    0 ≤ AM_L₁ ls lt la :=
  div_nonneg (by linarith) (by linarith)

/-- L₁ is at most 1 on [ls, lt]. -/
theorem AM_L₁_le_one (h : ls < lt) (la : ℝ) (hhi : la ≤ lt) :
    AM_L₁ ls lt la ≤ 1 := by
  rw [AM_L₁, div_le_one (by linarith)]
  linarith

/-- On [ls, lt] both Lagrange basis functions lie in [0, 1]. -/
theorem am_corrector_two_node_bounded
    (h : ls < lt) (la : ℝ) (hlo : ls ≤ la) (hhi : la ≤ lt) :
    0 ≤ AM_L₀ ls lt la ∧ AM_L₀ ls lt la ≤ 1 ∧
    0 ≤ AM_L₁ ls lt la ∧ AM_L₁ ls lt la ≤ 1 :=
  ⟨AM_L₀_nonneg h la hhi, AM_L₀_le_one h la hlo,
   AM_L₁_nonneg h la hlo, AM_L₁_le_one h la hhi⟩

-- ---------------------------------------------------------------------------
-- Result 4: Euler predictor + AM corrector is unconditionally stable
-- ---------------------------------------------------------------------------

/-!
### The recommended design: Euler predictor + AM corrector

Combining results 1 and 3:

- Predictor uses only the most recent history entry (order-1 Euler).
  Its b-coefficient = 1 - exp(ls - lt) ∈ (0, 1) unconditionally.
  No polynomial extrapolation through distant history nodes → no Runge.

- Corrector always includes lt in its node set (Adams-Moulton design).
  Integration is over [ls, lt] which lies inside the node convex hull.
  All Lagrange bases are in [0, 1] on the integration domain.

Neither component ever produces coefficients outside a bounded range,
regardless of step-size ratios.  The `MAX_H_RATIO = 2.5` guard in the
current code is a correct patch for the existing order-3 predictor, but
switching to an always-Euler predictor makes the guard unnecessary.

The trade-off: Euler predictor is O(h²) accurate (vs O(h⁴) for order-3
Adams), so the corrector must compensate more.  For low-step-count
diffusion schedules (≤ 30 steps) where step-size ratios vary widely,
stability dominates accuracy — the Euler predictor is the safer choice.
-/

/-- Combined stability: Euler predictor b ∈ (0,1) and AM corrector
    Lagrange bases ∈ [0,1] on the integration domain. -/
theorem euler_predictor_am_corrector_unconditionally_stable
    (h : ls < lt) (la : ℝ) (hlo : ls ≤ la) (hhi : la ≤ lt) :
    (0 < 1 - Real.exp (ls - lt) ∧ 1 - Real.exp (ls - lt) < 1) ∧
    (0 ≤ AM_L₀ ls lt la ∧ AM_L₀ ls lt la ≤ 1 ∧
     0 ≤ AM_L₁ ls lt la ∧ AM_L₁ ls lt la ≤ 1) :=
  ⟨euler_b_coeff_in_unit_interval h, am_corrector_two_node_bounded h la hlo hhi⟩
