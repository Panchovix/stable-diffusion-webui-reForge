-- RFVProofs/Explore4.lean — target: affine_dev_le_of_lipschitz_velocity
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
    (x v : ℝ → E) (t0 t h L : ℝ) (hL : 0 ≤ L) (hh : 0 ≤ h)
    (ht : t ∈ Set.Icc (t0 - h) (t0 + h))
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L) :
    ‖x t - (x t0 + (t - t0) • v t0)‖ ≤ L * (t - t0) ^ 2 / 2 := by
  -- Strategy: g s := x s - x t0 - (s - t0) • v t0 has g t0 = 0 and
  -- HasDerivAt g (v s - v t0) s, so by the same FTC-norm bound as
  -- chord_le_path_length, ‖g t‖ ≤ ∫ in t0..t, ‖v s - v t0‖ ≤ ∫ L*|s-t0| ds
  -- (using hlip) = L*(t-t0)^2/2. Left to exact?/apply?/aesop whether
  -- Mathlib already has the composed Taylor-remainder-from-Lipschitz-deriv
  -- lemma directly.
  first
  | exact?
  | apply?
  | aesop
  | sorry

end RFVProofs.Exploration
