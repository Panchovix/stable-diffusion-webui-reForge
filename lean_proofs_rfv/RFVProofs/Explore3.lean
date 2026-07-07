-- RFVProofs/Explore3.lean — target: finDiff_near_const_of_affine_up_to
-- (Perturbation.lean). See Explore1.lean for why hand-written attempt
-- comes first and `exact?`/`apply?` are last-resort with a heartbeat cap.

import RFVProofs.Defs
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith

open RFVProofs

namespace RFVProofs.Exploration

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

set_option maxHeartbeats 1000000 in
example
    (x : ℝ → E) (x0 c : E) (δ : ℝ) (hδ : 0 ≤ δ)
    (hx : IsAffineUpTo x x0 c δ) {t1 t2 : ℝ} (h : t1 ≠ t2) :
    ‖finDiffVelocity x t1 t2 - c‖ ≤ 2 * δ / |t2 - t1| := by
  -- NOTE: the inner fallback here must NOT be `sorry` — an inner `| sorry`
  -- would make this whole branch always "succeed" trivially, so `first`
  -- would never even try exact?/apply?/aesop below. Only the outermost
  -- fallback gets to be `sorry`.
  first
  | (have hne : t2 - t1 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
     have key : x t2 - x t1 - (t2 - t1) • c
              = (x t2 - (x0 + t2 • c)) - (x t1 - (x0 + t1 • c)) := by
       have hsub : (t2 - t1) • c = t2 • c - t1 • c := by rw [sub_smul]
       rw [hsub]; abel
     unfold finDiffVelocity
     have hbound : ‖x t2 - x t1 - (t2 - t1) • c‖ ≤ 2 * δ := by
       rw [key]
       calc ‖(x t2 - (x0 + t2 • c)) - (x t1 - (x0 + t1 • c))‖
           ≤ ‖x t2 - (x0 + t2 • c)‖ + ‖x t1 - (x0 + t1 • c)‖ := norm_sub_le _ _
         _ ≤ δ + δ := add_le_add (hx t2) (hx t1)
         _ = 2 * δ := by ring
     nlinarith [norm_smul (t2 - t1)⁻¹ (x t2 - x t1 - (t2 - t1) • c),
                abs_inv (t2 - t1), hbound]
     done)
  | exact?
  | apply?
  | aesop
  | sorry

end RFVProofs.Exploration
