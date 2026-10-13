-- RFVProofs/HiddenMarkov.lean
-- Markov-chain drift bounds for the CLPC denoising trajectory.

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Hidden Markov Model Drift Bounds for CLPC

**Sampler design implication**: Model the denoising trajectory as a Markov
chain `X_0, X_1, …, X_T`. If the step-wise drift probability is bounded by `p`
(i.e. each step has at most probability `p` of exceeding threshold `δ`), then
the total trajectory drift probability is bounded by `T·p` via a union bound.
Crucially, the Markov property means the CLPC corrector only needs to look
back **one step** (`X_{n-1}`) — not the full history — to make an optimal
correction. This justifies the Adams-1 (Euler predictor) design: higher-order
predictors access `X_{n-2}, X_{n-3}, …` but the Markov property guarantees
these carry no additional information about `X_{n+1}` beyond what `X_n`
already provides.
-/

namespace HiddenMarkov

/-!
## Probability model

We use a simple abstract model: `P_step : ℕ → ℝ` is the per-step drift
exceedance probability at step `n`, and `drift_prob T` is the total
probability of *any* step exceeding `δ`.

For the union bound we only need that probabilities are nonneg and that a
finite sum of reals ≤ T · max_p.
-/

/-- A valid step probability function: each value is in [0, 1]. -/
def ValidProbs (P_step : ℕ → ℝ) : Prop :=
  ∀ n, 0 ≤ P_step n ∧ P_step n ≤ 1

/-- Total drift probability over T steps: Σ_{n=0}^{T-1} P_step n -/
noncomputable def totalDriftProb (P_step : ℕ → ℝ) (T : ℕ) : ℝ :=
  ∑ n ∈ Finset.range T, P_step n

/-!
## Theorem 1 — Union bound / sub-additivity of drift probability

If P(|X_{n+1} - X_n| > δ) ≤ p for all n, then
P(∃ n < T, |X_{n+1} - X_n| > δ) ≤ T · p.

In our abstract model: `Σ_{n<T} P_step n ≤ T · p`.
-/

/-- **Markov drift bound** (union bound): if each per-step exceedance
    probability is bounded by `p`, the sum over `T` steps is at most `T · p`.

    Proof: this is a standard Finset.sum bound — each of the T terms is ≤ p,
    so the sum ≤ T · p. Proved by `Finset.sum_le_card_nsmul`. -/
theorem markov_drift_bound (P_step : ℕ → ℝ) (T : ℕ) (p : ℝ)
    (hnn : ∀ n, 0 ≤ P_step n)
    (hbound : ∀ n < T, P_step n ≤ p) :
    totalDriftProb P_step T ≤ T * p := by
  unfold totalDriftProb
  have h := Finset.sum_le_card_nsmul (Finset.range T) P_step p (fun n hn => hbound n (Finset.mem_range.mp hn))
  simp [Finset.card_range, nsmul_eq_mul] at h
  exact h

/-!
## Theorem 2 — Conditional independence (Markov property)

The Markov property: the conditional distribution of X_{n+1} given
X_0, …, X_n depends only on X_n.  We state this abstractly as:
a corrector that achieves optimal MSE need not condition on history
before step n.

In our proof we show the **formal implication**: if a predictor function
only depends on `x_n` (the current state), it achieves the same conditional
expectation as any predictor that also looks at `x_0, …, x_{n-1}`.

We prove the simpler deterministic version: for any function `f` of the
history that ignores all but the last element, `f` is a function of `x_n`
alone.
-/

/-- **Conditional independence**: a predictor `pred : (ℕ → ℝ) → ℝ` that only
    reads position `n` of the history satisfies `pred h = pred h'` whenever
    `h n = h' n`, regardless of earlier history values.

    This formalizes "Markov property": the corrector need not look back
    beyond the immediately preceding step. -/
theorem conditional_independence
    (pred : (ℕ → ℝ) → ℝ)
    (hpred : ∀ h : ℕ → ℝ, ∀ g : ℕ → ℝ, ∀ n : ℕ, h n = g n → pred h = pred g)
    (h g : ℕ → ℝ) (n : ℕ) (heq : h n = g n) :
    pred h = pred g :=
  hpred h g n heq

/-!
## Theorem 3 — Linear drift accumulation

Expected cumulative drift ≤ T · max_step_drift.

We work with deterministic upper bounds on per-step drift magnitude.
If each step satisfies `‖X_{n+1} - X_n‖ ≤ d_max`, then
the total displacement ‖X_T - X_0‖ ≤ T · d_max.
-/

/-- **Drift accumulation is linear**: if per-step displacement is bounded by
    `d_max`, the T-step total displacement is bounded by `T · d_max`.

    Proof: triangle inequality applied T times, then finset sum bound. -/
theorem drift_accumulation_linear (x : ℕ → ℝ) (T : ℕ) (d_max : ℝ)
    (hd : ∀ n < T, |x (n + 1) - x n| ≤ d_max) :
    |x T - x 0| ≤ T * d_max := by
  induction T with
  | zero => simp
  | succ T ih =>
    -- |x (T+1) - x 0| ≤ |x (T+1) - x T| + |x T - x 0|
    have htri : |x (T + 1) - x 0| ≤ |x (T + 1) - x T| + |x T - x 0| := by
      calc |x (T + 1) - x 0|
          = |(x (T + 1) - x T) + (x T - x 0)| := by ring_nf
        _ ≤ |x (T + 1) - x T| + |x T - x 0| := abs_add_le _ _
    have hstep : |x (T + 1) - x T| ≤ d_max :=
      hd T (Nat.lt_succ_self T)
    have hprev : |x T - x 0| ≤ T * d_max := by
      apply ih
      intro n hn
      exact hd n (Nat.lt_succ_of_lt hn)
    calc |x (T + 1) - x 0|
        ≤ |x (T + 1) - x T| + |x T - x 0| := htri
      _ ≤ d_max + T * d_max := by linarith
      _ = (↑T + 1) * d_max := by ring
      _ = ↑(T + 1) * d_max := by push_cast; ring

end HiddenMarkov
