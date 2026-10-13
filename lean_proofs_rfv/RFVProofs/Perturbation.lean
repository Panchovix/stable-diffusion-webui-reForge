-- RFVProofs/Perturbation.lean
-- Quantitative version of the diagnostic in Stability.lean: real samplers
-- are never *exactly* affine, only affine up to some bounded residual
-- (discretisation error, the model's departure from the rectified-flow /
-- constant-velocity ideal, etc). This bounds how much the observed
-- finite-difference velocity can drift as a function of that residual —
-- i.e. how "almost stable" the observed field is allowed to be before it
-- signals a real problem.
--
-- `IsAffineUpTo` lives in Defs.lean (not here) so Exploration.lean can use
-- it without also pulling this file's theorem name into scope.

import RFVProofs.Defs
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Positivity

namespace RFVProofs

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- On an almost-affine trajectory, any two-point finite-difference velocity
    estimate is within `2δ / |t2 - t1|` of the true constant velocity `c`.
    This is the formal version of "the observed velocity-field history is
    *almost* stable, with drift controlled by how far the path departs from
    the constant-velocity ideal" — the threshold an actual sampling-time
    diagnostic would compare against.

    Proof: `finDiffVelocity x t1 t2 - c = (t2-t1)⁻¹ • (x t2 - x t1 -
    (t2-t1)•c)` (pure algebra — `simp [smul_sub, smul_smul,
    inv_mul_cancel₀]`), and the inner term is `(x t2 - tangent t2) -
    (x t1 - tangent t1)`, bounded by `δ + δ` via the triangle inequality and
    `hx`. Take norms and divide. (Note `hδ : 0 ≤ δ` ends up unused — it's
    implied by `hx` + nonnegativity of norms, but kept in the signature
    since it's the natural hypothesis to state.) -/
theorem finDiff_near_const_of_affine_up_to
    (x : ℝ → E) (x0 c : E) (δ : ℝ) (hδ : 0 ≤ δ)
    (hx : IsAffineUpTo x x0 c δ) {t1 t2 : ℝ} (h : t1 ≠ t2) :
    ‖finDiffVelocity x t1 t2 - c‖ ≤ 2 * δ / |t2 - t1| := by
  have hne : t2 - t1 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have key : finDiffVelocity x t1 t2 - c
      = (t2 - t1)⁻¹ • (x t2 - x t1 - (t2 - t1) • c) := by
    unfold finDiffVelocity
    simp only [smul_sub, smul_smul, inv_mul_cancel₀ hne, one_smul]
  have hbound : ‖x t2 - x t1 - (t2 - t1) • c‖ ≤ 2 * δ := by
    have heq2 : x t2 - x t1 - (t2 - t1) • c
        = (x t2 - (x0 + t2 • c)) - (x t1 - (x0 + t1 • c)) := by
      have hsub : (t2 - t1) • c = t2 • c - t1 • c := by rw [sub_smul]
      rw [hsub]; abel
    rw [heq2]
    calc ‖(x t2 - (x0 + t2 • c)) - (x t1 - (x0 + t1 • c))‖
        ≤ ‖x t2 - (x0 + t2 • c)‖ + ‖x t1 - (x0 + t1 • c)‖ := norm_sub_le _ _
      _ ≤ δ + δ := add_le_add (hx t2) (hx t1)
      _ = 2 * δ := by ring
  rw [key, norm_smul, norm_inv, Real.norm_eq_abs]
  calc |t2 - t1|⁻¹ * ‖x t2 - x t1 - (t2 - t1) • c‖
      ≤ |t2 - t1|⁻¹ * (2 * δ) := by
        apply mul_le_mul_of_nonneg_left hbound
        positivity
    _ = 2 * δ / |t2 - t1| := by ring

end RFVProofs
