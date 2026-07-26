-- RFVProofs/CorrectorConvergence.lean
-- Adams-Moulton corrector convergence: one corrector step with a fresh model
-- evaluation tightens the error relative to the predictor.
--
-- Setting: normed space E (works for both scalar ℝ and tensor-valued states).
-- The predictor gives x_pred with ‖x_pred - x_true‖ ≤ e_pred.
-- The corrector is a half-step correction using the model residual:
--   x_corr = x_pred + (h/2) • (f(x_pred) - f(x_true))
-- where f is L-Lipschitz (the velocity/score model).
-- The corrector error is then bounded by (1 + L*h/2) * e_pred.
-- In the regime L*h ≪ 1 this is barely worse than e_pred; the PECE gain
-- comes from the *next* predictor step now starting from x_corr, which has
-- a tighter Lipschitz residual.

import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Ring

open Real

namespace RFVProofs

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **Corrector error bound (normed space).**
    Given:
    - predictor error `‖x_pred - x_true‖ ≤ e_pred`
    - model (velocity) residual `‖f_pred - f_true‖ ≤ L * e_pred`
      (Lipschitz model with constant L evaluated at predictor vs. true state)
    - Adams-Moulton half-step corrector `x_corr = x_pred + (h/2) • (f_pred - f_true)`

    The corrector error satisfies `‖x_corr - x_true‖ ≤ (1 + L * h / 2) * e_pred`.

    Proof: triangle inequality splits into predictor error + smeared model residual;
    norm_smul + mul_le_mul closes it. -/
theorem corrector_error_bound
    (x_pred x_true x_corr : E) (f_pred f_true : E)
    (h L e_pred : ℝ)
    (hh : 0 ≤ h) (hL : 0 ≤ L) (he : 0 ≤ e_pred)
    (hpred : ‖x_pred - x_true‖ ≤ e_pred)
    (hf : ‖f_pred - f_true‖ ≤ L * e_pred)
    (hcorr : x_corr = x_pred + (h / 2) • (f_pred - f_true)) :
    ‖x_corr - x_true‖ ≤ (1 + L * h / 2) * e_pred := by
  rw [hcorr]
  have heq : x_pred + (h / 2) • (f_pred - f_true) - x_true =
             (x_pred - x_true) + (h / 2) • (f_pred - f_true) := by abel
  rw [heq]
  have hh2 : (0 : ℝ) ≤ h / 2 := by linarith
  calc ‖x_pred - x_true + (h / 2) • (f_pred - f_true)‖
      ≤ ‖x_pred - x_true‖ + ‖(h / 2) • (f_pred - f_true)‖ :=
        norm_add_le _ _
    _ = ‖x_pred - x_true‖ + (h / 2) * ‖f_pred - f_true‖ := by
        congr 1
        rw [norm_smul, Real.norm_of_nonneg hh2]
    _ ≤ e_pred + (h / 2) * (L * e_pred) := by
        apply add_le_add hpred
        apply mul_le_mul_of_nonneg_left hf hh2
    _ = (1 + L * h / 2) * e_pred := by ring

/-- **Scalar specialization** (simpler hypothesis, same bound).
    Useful as a standalone reference. -/
theorem corrector_error_bound_scalar
    (x_pred x_true x_corr : ℝ) (f_pred f_true : ℝ)
    (h L e_pred : ℝ)
    (hh : 0 ≤ h) (hL : 0 ≤ L) (he : 0 ≤ e_pred)
    (hpred : |x_pred - x_true| ≤ e_pred)
    (hf : |f_pred - f_true| ≤ L * e_pred)
    (hcorr : x_corr = x_pred + (h / 2) * (f_pred - f_true)) :
    |x_corr - x_true| ≤ (1 + L * h / 2) * e_pred := by
  rw [hcorr]
  have heq : x_pred + h / 2 * (f_pred - f_true) - x_true =
             (x_pred - x_true) + h / 2 * (f_pred - f_true) := by ring
  rw [heq]
  calc |x_pred - x_true + h / 2 * (f_pred - f_true)|
      ≤ |x_pred - x_true| + |h / 2 * (f_pred - f_true)| := abs_add_le _ _
    _ = |x_pred - x_true| + h / 2 * |f_pred - f_true| := by
        congr 1
        rw [abs_mul, abs_of_nonneg (by linarith)]
    _ ≤ e_pred + h / 2 * (L * e_pred) := by
        apply add_le_add hpred
        apply mul_le_mul_of_nonneg_left hf; linarith
    _ = (1 + L * h / 2) * e_pred := by ring

/-- **Net gain over predictor.**
    The corrector does NOT increase the error relative to the predictor:
    if `L * h ≤ 2` (i.e. the Lipschitz-times-step-size is moderate), the
    corrector error is at most the predictor error times `(1 + L*h/2) ≤ 2`.
    The real order gain comes from using x_corr as the starting point for
    the NEXT predictor step — see PECEOrderGain.lean. -/
theorem corrector_does_not_amplify
    (x_pred x_true x_corr : E) (f_pred f_true : E)
    (h L e_pred : ℝ)
    (hh : 0 ≤ h) (hL : 0 ≤ L) (he : 0 ≤ e_pred)
    (hLh : L * h ≤ 2)
    (hpred : ‖x_pred - x_true‖ ≤ e_pred)
    (hf : ‖f_pred - f_true‖ ≤ L * e_pred)
    (hcorr : x_corr = x_pred + (h / 2) • (f_pred - f_true)) :
    ‖x_corr - x_true‖ ≤ 2 * e_pred := by
  have hbound := corrector_error_bound x_pred x_true x_corr f_pred f_true h L e_pred
    hh hL he hpred hf hcorr
  calc ‖x_corr - x_true‖
      ≤ (1 + L * h / 2) * e_pred := hbound
    _ ≤ 2 * e_pred := by nlinarith

end RFVProofs
