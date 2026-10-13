-- RFVProofs/Explore5.lean — target: finDiff_window_near_velocity
-- (LocalLinearity.lean). See Explore1.lean for why exact?/apply?/aesop are
-- last-resort with a heartbeat cap; this target has no hand-written attempt
-- yet (genuinely new — left to the search first, per the user's request to
-- "see where Lean leads us" before hand-proving).

import RFVProofs.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith

open MeasureTheory

namespace RFVProofs.Exploration

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

set_option maxHeartbeats 1000000 in
example
    (x v : ℝ → E) (t0 h L : ℝ) (hL : 0 ≤ L) (hh : 0 < h)
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint1 : ∀ t ∈ Set.Icc (t0 - h) (t0 + h),
      IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    {t1 t2 : ℝ} (ht1 : t1 ∈ Set.Icc (t0 - h) (t0 + h))
    (ht2 : t2 ∈ Set.Icc (t0 - h) (t0 + h)) (hne : t1 ≠ t2) :
    ‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h := by
  first
  | exact?
  | apply?
  | aesop
  | sorry

end RFVProofs.Exploration
