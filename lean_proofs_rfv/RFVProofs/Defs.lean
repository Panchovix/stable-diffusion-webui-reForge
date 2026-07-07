-- RFVProofs/Defs.lean
-- Pure definitions, no theorem statements. Deliberately separated out so
-- that Exploration.lean can import *only* this file: if the open theorems
-- in Straightness.lean / Perturbation.lean were in scope, `exact?`/`apply?`
-- would trivially "solve" each goal by citing that very theorem's own
-- (still-`sorry`'d) declaration — a vacuous, sorryAx-laden non-proof. By
-- keeping definitions and theorem statements in different files, genuine
-- search has nothing circular to find.

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

namespace RFVProofs

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- An affine (constant-velocity) trajectory through `E`: `x t = x0 + t • c`.
    `c` is the velocity field a rectified-flow / VP-EPS model is trained to
    predict; Rectified Flow's training objective pushes real trajectories
    toward this form. -/
def AffineTraj (x0 c : E) (t : ℝ) : E := x0 + t • c

/-- A trajectory is `IsAffine` if it has this form for *some* `x0`, `c`. -/
def IsAffine (x : ℝ → E) : Prop := ∃ x0 c : E, ∀ t, x t = x0 + t • c

/-- Two-point finite-difference velocity estimate computed from sampling
    history: `(x t2 - x t1) / (t2 - t1)`, vector-valued. -/
noncomputable def finDiffVelocity (x : ℝ → E) (t1 t2 : ℝ) : E :=
  (t2 - t1)⁻¹ • (x t2 - x t1)

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Straightness ratio of a trajectory `x` with velocity field `v = x'` over
    `[a, b]`: total path length divided by chord length. -/
noncomputable def straightnessRatio (x v : ℝ → F) (a b : ℝ) : ℝ :=
  (∫ t in a..b, ‖v t‖) / ‖x b - x a‖

/-- `x` is affine up to a bounded residual `δ`: `x t = x0 + t • c + r t` with
    `‖r t‖ ≤ δ` for every `t`, i.e. `‖x t - (x0 + t • c)‖ ≤ δ`. Models a
    sampler whose path is *almost* the rectified-flow straight line but for
    a bounded deviation `δ`. -/
def IsAffineUpTo (x : ℝ → F) (x0 c : F) (δ : ℝ) : Prop :=
  ∀ t, ‖x t - (x0 + t • c)‖ ≤ δ

/-- The velocity field `v` is `L`-Lipschitz *on a window* `s` (not globally):
    multi-step samplers like DPM++ 2M only need the trajectory to be
    well-approximated by a low-order polynomial over the few steps in their
    history buffer, not globally straight — a real EPS/VP trajectory is
    typically curved overall but locally near-linear once the step size is
    small enough. `L` is the local curvature bound: how fast the velocity
    itself is allowed to change inside the window. `L = 0` recovers exactly
    affine (`IsAffine`) on that window. -/
def LipschitzVelocityOn (v : ℝ → F) (s : Set ℝ) (L : ℝ) : Prop :=
  ∀ t1 ∈ s, ∀ t2 ∈ s, ‖v t2 - v t1‖ ≤ L * |t2 - t1|

end RFVProofs
