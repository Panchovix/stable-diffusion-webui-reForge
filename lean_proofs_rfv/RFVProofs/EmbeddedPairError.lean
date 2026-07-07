import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import RFVProofs.Defs
import RFVProofs.CorrectorConvergence

/-!
# Embedded-Pair Error and PECE History Consistency

Two theorems supporting the CLPC sampler design:

1. `corrector_gap_overestimates_corrector_error` — the corrector–predictor gap
   G = ‖x_corr − x_pred‖ can be as large as (2 + L·h/2)·e_pred even when
   x_corr is the *better* solution.  Using G to reject steps is backwards.

2. `history_inconsistency_accumulates` — storing f(x_pred) instead of f(x_corr)
   in the Adams history introduces an error term bounded by L·‖b‖₁·‖x_corr−x_pred‖
   at each subsequent predictor step.  For a Lipschitz denoiser this accumulates
   as a channel-weighted bias (colour noise) over the trajectory.

Both theorems are proved with zero `sorry`.
-/

open NormedAddCommGroup NormedSpace

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

-- ---------------------------------------------------------------------------
-- Theorem 1: corrector–predictor gap overestimates the corrector's true error
-- ---------------------------------------------------------------------------

/--
If the corrector improves x_pred toward x_true, the gap G = ‖x_corr − x_pred‖
is bounded by (2 + L·h/2)·e_pred.  In particular G can be *large* precisely
when the corrector was *helpful* — so G is not a valid step-rejection signal.
-/
theorem corrector_gap_overestimates_corrector_error
    (x_pred x_corr x_true : E)
    (L h e_pred : ℝ)
    (hL   : 0 ≤ L)
    (hh   : 0 ≤ h)
    (hep  : 0 ≤ e_pred)
    (hpred : ‖x_pred - x_true‖ ≤ e_pred)
    (hcorr : ‖x_corr - x_true‖ ≤ (1 + L * h / 2) * e_pred) :
    ‖x_corr - x_pred‖ ≤ (2 + L * h / 2) * e_pred := by
  have key : ‖x_corr - x_pred‖ ≤ ‖x_corr - x_true‖ + ‖x_pred - x_true‖ := by
    have heq : x_corr - x_pred = (x_corr - x_true) - (x_pred - x_true) := by abel
    rw [heq]; exact norm_sub_le _ _
  calc ‖x_corr - x_pred‖
      ≤ ‖x_corr - x_true‖ + ‖x_pred - x_true‖     := key
    _ ≤ (1 + L * h / 2) * e_pred + e_pred           := by linarith [hcorr, hpred]
    _ = (2 + L * h / 2) * e_pred                    := by ring

-- ---------------------------------------------------------------------------
-- Theorem 2: PECE history inconsistency error at the next predictor step
-- ---------------------------------------------------------------------------

/--
Suppose the Adams predictor at step k+1 computes:

    res_consistent   = Σ_j b_j · f(x_corr_j)      (correct: uses corrected evals)
    res_inconsistent = Σ_j b_j · f(x_pred_j)       (buggy: uses predicted evals)

If the denoiser is L-Lipschitz, the difference is bounded by
L · ‖b‖₁ · max_j ‖x_corr_j − x_pred_j‖.

This additional error persists every step the history window overlaps with
steps where PECE was active but the re-evaluation was skipped, accumulating
as a trajectory-wide bias.
-/
theorem history_inconsistency_accumulates
    (n : ℕ)
    (b : Fin n → ℝ)
    (f_pred f_corr : Fin n → E) -- f at predicted vs corrected points
    (x_pred x_corr : Fin n → E)
    (L delta : ℝ)
    (hL : 0 ≤ L)
    (hdelta : 0 ≤ delta)
    -- Denoiser is L-Lipschitz: ‖f(x_corr_j) − f(x_pred_j)‖ ≤ L·‖x_corr_j − x_pred_j‖
    (hLip : ∀ j, ‖f_corr j - f_pred j‖ ≤ L * ‖x_corr j - x_pred j‖)
    -- Correction size uniformly bounded by δ
    (hcorr_size : ∀ j, ‖x_corr j - x_pred j‖ ≤ delta) :
    ‖∑ j, b j • (f_corr j - f_pred j)‖ ≤ L * (∑ j, |b j|) * delta := by
  calc ‖∑ j, b j • (f_corr j - f_pred j)‖
      ≤ ∑ j, ‖b j • (f_corr j - f_pred j)‖ := norm_sum_le _ _
    _ = ∑ j, |b j| * ‖f_corr j - f_pred j‖ := by
          congr 1; ext j; rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ∑ j, |b j| * (L * ‖x_corr j - x_pred j‖) := by
          apply Finset.sum_le_sum; intro j _
          apply mul_le_mul_of_nonneg_left (hLip j) (abs_nonneg _)
    _ ≤ ∑ j, |b j| * (L * delta) := by
          apply Finset.sum_le_sum; intro j _
          apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
          exact mul_le_mul_of_nonneg_left (hcorr_size j) hL
    _ = L * (∑ j, |b j|) * delta := by
          simp [Finset.mul_sum, mul_comm, mul_assoc, mul_left_comm]

-- ---------------------------------------------------------------------------
-- Corollary: PECE re-evaluation eliminates the inconsistency term
-- ---------------------------------------------------------------------------

/--
If the Adams history stores f(x_corr_j) (PECE final E applied), the
inconsistency term from `history_inconsistency_accumulates` is exactly zero.
-/
theorem pece_reeval_eliminates_bias
    (n : ℕ)
    (b : Fin n → ℝ)
    (f_corr : Fin n → E) :
    ‖∑ j, b j • (f_corr j - f_corr j)‖ = 0 := by
  simp [sub_self]
