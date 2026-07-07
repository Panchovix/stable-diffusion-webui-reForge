-- RFVProofs/Explore2.lean — target: straightness_eq_one_of_const_velocity
-- (Straightness.lean). See Explore1.lean for why hand-written attempt
-- comes first and `exact?`/`apply?` are last-resort with a heartbeat cap.

import RFVProofs.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

open MeasureTheory RFVProofs

namespace RFVProofs.Exploration

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

set_option maxHeartbeats 1000000 in
example
    (x0 c : E) (a b : ℝ) (hab : a < b) (hc : c ≠ 0) :
    straightnessRatio (fun t => x0 + t • c) (fun _ => c) a b = 1 := by
  -- NOTE: `done` at the end of the hand-written branch is load-bearing.
  -- `ring_nf`/`norm_num`/`simp`-family tactics can "succeed" by making
  -- partial progress without fully closing the goal, which `first` treats
  -- as a successful branch (it only checks for a thrown exception, not
  -- goal closure) — silently skipping every later alternative including
  -- `exact?` and the final `sorry`. `done` forces a hard failure here if
  -- any goal remains, so `first` actually falls through.
  first
  | (unfold straightnessRatio
     simp only [intervalIntegral.integral_const, norm_smul]
     field_simp
     ring_nf
     first
     | norm_num [abs_of_pos (sub_pos.mpr hab), norm_pos_iff.mpr hc]
     | nlinarith [abs_of_pos (sub_pos.mpr hab)]
     done)
  | exact?
  | apply?
  | aesop
  | sorry

end RFVProofs.Exploration
