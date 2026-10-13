-- RFVProofs/Explore1.lean — target: chord_le_path_length (Straightness.lean)
-- Split into its own file (see explore.sh) so a runaway `exact?` search on
-- this goal can't starve the other two targets. The hand-written,
-- domain-specific attempt goes FIRST: a bare `‖_‖ ≤ _` goal is so generic
-- that `exact?`/`apply?` enumerate huge unrelated swaths of Mathlib's order
-- theory (lists, arrays, `Std.PRange`, ...) before ever reaching anything
-- about integrals — that can run for a very long time, so they're kept as
-- a last resort with a heartbeat cap.

import RFVProofs.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.Abel

open MeasureTheory RFVProofs

namespace RFVProofs.Exploration

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

set_option maxHeartbeats 1000000 in
example
    (x v : ℝ → E) (a b : ℝ) (hab : a ≤ b)
    (hderiv : ∀ t ∈ Set.Icc a b, HasDerivAt x (v t) t)
    (hint : IntervalIntegrable v MeasureTheory.volume a b) :
    ‖x b - x a‖ ≤ ∫ t in a..b, ‖v t‖ := by
  -- `exact` always either fully closes the goal or hard-fails, so this
  -- branch doesn't strictly need a trailing `done` like Explore2/3 do —
  -- kept for consistency with the other exploration files.
  first
  | (have h := intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint
     first
     | (rw [← h]; exact intervalIntegral.norm_integral_le_abs_integral_norm)
     | (rw [← h]; exact intervalIntegral.norm_integral_le_integral_norm hab)
     done)
  | exact?
  | apply?
  | aesop
  | sorry

end RFVProofs.Exploration
