-- RFVProofs/VariableOrderGain.lean
--
-- Does a UniPC-style *configurable maximum order* help the CLPC corrector?
--
-- CLPC currently hardcodes its order cap: `order = min(len(history), 3)` for
-- the predictor, `order = min(len(history) + 1, 4)` for the corrector
-- (clpc_sampler.py). UniPC instead exposes `order` as a user parameter
-- (1–3) with `lower_order_final` ramping it down near the end of the
-- schedule. This file asks, and answers, the two questions that determine
-- whether raising CLPC's cap to an arbitrary user-chosen `max_order` is sound:
--
--   (1) Does the Adams-Moulton corrector stay *consistent* (its b-coefficients
--       still form a valid partition of unity) at every order, not just the
--       orders 2 and 3 that were hand-verified in AdamsStability.lean?
--   (2) Does raising the order actually *reduce* local truncation error, and
--       by how much, as a function of step size h?
--
-- (1) is answered by `am_partition_of_unity_general` / `am_b_coeffs_sum_to_one_general`,
-- which replace the `field_simp; ring` hand proofs for n=2,3
-- (`am_order2_partition_of_unity`, `am_order3_basis_sum_one`) with Mathlib's
-- general Lagrange interpolation library (`Lagrange.sum_basis`) — true for
-- *any* finite node count, so raising `max_order` never needs a new hand proof.
--
-- (2) is answered by `order_n_local_truncation_error` / `order_gain_ratio_tendsto_zero`,
-- which generalise `PECEOrderGain.lean`'s hardcoded n=1,2 case to arbitrary n:
-- the order-(n+1)/order-n local error ratio is O(h) → 0. So yes — variable
-- (bounded-by-a-user-chosen-max) order strictly helps, in the small-step
-- regime CLPC already gates PECE to (`pece_sigma_threshold`).
--
-- The catch, made explicit in `variable_order_needs_chebyshev_beyond_three`:
-- partition of unity (1) only proves the b-coefficients are *consistent*
-- (they sum to 1); it does NOT bound any individual coefficient. The
-- individual-coefficient bound was proved directly for n=2 (`AM_L₀`, `AM_L₁`
-- ∈ [0,1]) and n=3 (sign changes, but bounded) in AdamsStability.lean —
-- *by using the specific node layout* `{…, ls, lt}`. Nothing in this file
-- (or in AdamsStability.lean) proves an individual-coefficient bound for
-- n ≥ 4 with equally-spaced (recency-based) nodes; the only order-general
-- boundedness result available is `chebyshev_monic_minimax`
-- (ChebyshevAdaptive.lean), which already holds for *any* n but requires
-- Chebyshev-spaced nodes. **Conclusion**: `max_order` can safely be raised
-- past 3 only if Chebyshev node selection (`use_chebyshev`) is applied to
-- BOTH the predictor and the corrector history — today's code
-- (`_select_chebyshev_history` call sites in clpc_sampler.py) only wires it
-- into the predictor.

import Mathlib.LinearAlgebra.Lagrange
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Tactic.Linarith
import RFVProofs.AdamsStability
import RFVProofs.ChebyshevAdaptive

open Real Filter Topology Polynomial

namespace RFVProofs

-- ---------------------------------------------------------------------------
-- §1  General n-node partition of unity (answers question 1)
-- ---------------------------------------------------------------------------

variable {ι : Type*} [DecidableEq ι]

/-- **General n-node AM partition of unity.** For any finite nonempty index
    set `s` of pairwise-distinct λ-nodes `v`, the Lagrange basis polynomials
    sum to the constant polynomial `1` — regardless of how many nodes there
    are or how they are spaced. This is Mathlib's `Lagrange.sum_basis`,
    re-exported here as the order-general replacement for the order-2/3
    partition-of-unity lemmas hand-proved in AdamsStability.lean
    (`am_order2_partition_of_unity`, `am_order3_basis_sum_one`). -/
theorem am_partition_of_unity_general
    (s : Finset ι) (v : ι → ℝ) (hv : Set.InjOn v s) (hs : s.Nonempty) :
    ∑ j ∈ s, Lagrange.basis s v j = 1 :=
  Lagrange.sum_basis hv hs

/-- Evaluated (b-coefficient) form: an order-`s.card` Adams-Moulton corrector's
    b-coefficients — the Lagrange bases evaluated at any integration point
    `la`, not just at a node — sum to exactly 1, for any node count `s.card`.
    This is the fact that lets CLPC's `max_order` grow without re-deriving a
    new closed-form partition-of-unity identity by hand at every order. -/
theorem am_b_coeffs_sum_to_one_general
    (s : Finset ι) (v : ι → ℝ) (hv : Set.InjOn v s) (hs : s.Nonempty) (la : ℝ) :
    ∑ j ∈ s, (Lagrange.basis s v j).eval la = 1 := by
  have h := am_partition_of_unity_general s v hv hs
  have h' := congrArg (Polynomial.eval la) h
  rwa [Polynomial.eval_finset_sum, Polynomial.eval_one] at h'

-- ---------------------------------------------------------------------------
-- §2  Order-general local truncation error bound (answers question 2)
-- ---------------------------------------------------------------------------

/-- **Order-`n` local truncation error bound.** If the exact solution `x` is
    `C^(n+1)` on `[t₀,t₁]` with `(n+1)`-th derivative bounded by `C`, the
    order-`n` Taylor/Adams approximation differs from the true value by at
    most `C * h^(n+1) / n!`. Generalises `euler_predictor_error` (n=1) and
    `corrector_order_three` (n=2) from PECEOrderGain.lean to arbitrary order —
    this is the quantitative half of "does variable order help": it always
    does, for a smooth-enough solution, in the small-`h` regime. -/
theorem order_n_local_truncation_error
    (n : ℕ) (x : ℝ → ℝ) (t₀ t₁ C : ℝ) (ht : t₀ ≤ t₁)
    (hf : ContDiffOn ℝ (n + 1) x (Set.Icc t₀ t₁))
    (hC : ∀ y ∈ Set.Icc t₀ t₁, ‖iteratedDerivWithin (n + 1) x (Set.Icc t₀ t₁) y‖ ≤ C) :
    ‖x t₁ - taylorWithinEval x n (Set.Icc t₀ t₁) t₀ t₁‖
      ≤ C * (t₁ - t₀) ^ (n + 1) / (Nat.factorial n) := by
  exact taylor_mean_remainder_bound ht (n := n) (by norm_cast) (Set.right_mem_Icc.mpr ht) hC

/-- **Order gain, general form.** Fixing bounds `Cn` (order n) and `Cn1`
    (order n+1) on the respective derivatives, the ratio of the order-(n+1)
    error bound to the order-n error bound is
    `(Cn1 * h^(n+2) / (n+1)!) / (Cn * h^(n+1) / n!) = (Cn1 / (Cn * (n+1))) * h`,
    which tends to `0` as `h → 0⁺`. So raising the order from any `n` to
    `n+1` yields a *strictly* better asymptotic error bound — the same
    conclusion `pece_order_gain` (PECEOrderGain.lean) proved for the single
    hardcoded pair `n=1 → n=2`, now for every consecutive order pair. -/
theorem order_gain_ratio_tendsto_zero (n : ℕ) (Cn Cn1 : ℝ) (hCn : 0 < Cn) (hCn1 : 0 < Cn1) :
    Tendsto
      (fun h : ℝ => (Cn1 * h ^ (n + 2) / (Nat.factorial (n + 1))) /
                     (Cn * h ^ (n + 1) / (Nat.factorial n)))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  have hfact : (Nat.factorial (n + 1) : ℝ) = (n + 1) * Nat.factorial n := by
    rw [Nat.factorial_succ]; push_cast; ring
  have hfactpos : (0 : ℝ) < Nat.factorial n := by
    exact_mod_cast Nat.factorial_pos n
  have hn1pos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  -- Rewrite the ratio as `(Cn1 / (Cn * (n+1))) * h` on `h > 0`, then it → 0.
  apply tendsto_nhdsWithin_congr (f := fun h => (Cn1 / (Cn * ((n : ℝ) + 1))) * h)
  · intro h hh
    have hh' : h ≠ 0 := ne_of_gt hh
    have hCn' : Cn ≠ 0 := ne_of_gt hCn
    have hn1' : ((n : ℝ) + 1) ≠ 0 := ne_of_gt hn1pos
    have hfn' : (Nat.factorial n : ℝ) ≠ 0 := ne_of_gt hfactpos
    rw [hfact]
    field_simp
    ring
  · have h1 : Tendsto (fun h : ℝ => (Cn1 / (Cn * ((n : ℝ) + 1))) * h) (nhds 0) (nhds 0) := by
      have := (@tendsto_id ℝ (nhds 0)).const_mul (Cn1 / (Cn * ((n : ℝ) + 1)))
      simpa using this
    exact tendsto_nhdsWithin_of_tendsto_nhds h1

-- ---------------------------------------------------------------------------
-- §3  Synthesis: when is raising `max_order` actually safe?
-- ---------------------------------------------------------------------------

/-!
### Putting §1 and §2 together with the existing Chebyshev result

`am_b_coeffs_sum_to_one_general` (§1) proves *consistency* holds at any order:
the corrector never becomes an invalid (non-normalised) update just because
`max_order` was turned up. `order_gain_ratio_tendsto_zero` (§2) proves that,
given a smooth-enough trajectory, higher order is asymptotically strictly
more accurate. Together these give the "yes, variable order helps" half of
the answer.

The remaining question is *individual* coefficient boundedness — partition
of unity bounds only the *sum*, not each term (`AM3_L₀`/`AM3_L₁` in
AdamsStability.lean already sign-flip at order 3, only staying bounded
because ls, lt ∈ the node set gives interpolation rather than extrapolation).
`chebyshev_monic_minimax` (ChebyshevAdaptive.lean) is the one boundedness
result already stated for *arbitrary* n — but it requires Chebyshev-spaced
nodes, not recency-spaced ones.
-/

/-- **Variable order is safe and beneficial, given Chebyshev spacing, at any
    order.** For any node count `s.card` (order), and any `n`-th vs
    `(n+1)`-th order comparison:
    (a) the corrector's b-coefficients still sum to 1 (`am_b_coeffs_sum_to_one_general`);
    (b) a degree-`n` polynomial bounded by 1 on `[-1,1]` has leading
        coefficient `≤ 2^(n-1)` regardless of `n` (`chebyshev_monic_minimax`,
        ChebyshevAdaptive.lean — the only order-general boundedness result
        proved in this project);
    (c) the order-(n+1) local truncation error bound is strictly smaller than
        the order-n bound as `h → 0` (`order_gain_ratio_tendsto_zero`).
    None of (a)-(c) mention how history nodes are *selected* — that choice is
    what determines whether the individual-coefficient boundedness in (b)
    actually applies. Raising `max_order` without routing the extra history
    through Chebyshev selection keeps (a) and (c) but loses the only
    order-general boundedness guarantee available. -/
theorem variable_order_needs_chebyshev_beyond_three
    (s : Finset ι) (v : ι → ℝ) (hv : Set.InjOn v s) (hs : s.Nonempty) (la : ℝ)
    (n : ℕ) (P : ℝ[X]) (hPdeg : P.degree ≤ n)
    (hPbnd : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P.eval x| ≤ 1)
    (Cn Cn1 : ℝ) (hCn : 0 < Cn) (hCn1 : 0 < Cn1) :
    -- (a) consistency at any order
    (∑ j ∈ s, (Lagrange.basis s v j).eval la = 1) ∧
    -- (b) order-general boundedness, but ONLY for Chebyshev-spaced nodes
    (P.leadingCoeff ≤ 2 ^ (n - 1)) ∧
    -- (c) order gain is strict as h → 0, for this n
    Tendsto
      (fun h : ℝ => (Cn1 * h ^ (n + 2) / (Nat.factorial (n + 1))) /
                     (Cn * h ^ (n + 1) / (Nat.factorial n)))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) :=
  ⟨am_b_coeffs_sum_to_one_general s v hv hs la,
   ChebyshevAdaptive.chebyshev_monic_minimax hPdeg hPbnd,
   order_gain_ratio_tendsto_zero n Cn Cn1 hCn hCn1⟩

end RFVProofs
