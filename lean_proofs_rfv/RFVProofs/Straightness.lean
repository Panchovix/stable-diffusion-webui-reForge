-- RFVProofs/Straightness.lean
-- Literature-grounded anchor: the "straightness ratio" of a rectified-flow
-- trajectory (path length / chord length) is always ≥ 1, with equality
-- exactly for constant-velocity (affine) trajectories. Empirically, high-
-- dimensional flow-matching trajectories concentrate around ratio ≈ 1.02
-- (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html), i.e. very
-- close to the affine ideal formalized in Stability.lean.
--
-- These statements need real interval-integral / FTC machinery whose exact
-- Mathlib v4.29.1 lemma names are not hand-verified here — they are the
-- intended targets for explore.sh's `exact?` / `aesop` / LeanSearchClient
-- sweep, left as `sorry`. `straightnessRatio` itself lives in Defs.lean
-- (not here) so Exploration.lean can use the definition without also
-- pulling these `sorry`'d theorem names into scope.

-- NOTE: as of this Mathlib rev, the old flat files
-- `Mathlib.MeasureTheory.Integral.IntervalIntegral` and
-- `Mathlib.Analysis.Calculus.FundThmCalculus` no longer exist — both were
-- split into folders. Import the actual submodules instead.
import RFVProofs.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FieldSimp

open MeasureTheory

namespace RFVProofs

-- `chord_le_path_length` needs `E` complete (Bochner-integrability of the
-- derivative against a `CompleteSpace` codomain is what
-- `intervalIntegral.integral_eq_sub_of_hasDerivAt` actually requires —
-- `exact?`/`apply?` never surfaced this because the *un-completed* version
-- of the goal has no instance to find a lemma against in the first place).
-- Any finite-dimensional space (e.g. the latent space SDXL samples in) is
-- automatically complete, so this is not a real restriction in practice.
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- Path length dominates chord length: `‖x b - x a‖ ≤ ∫ t in a..b, ‖v t‖`.
    Proof: FTC-2 (`intervalIntegral.integral_eq_sub_of_hasDerivAt`) gives
    `∫ v = x b - x a`, then bound the norm of that integral by the integral
    of the norm (`intervalIntegral.norm_integral_le_integral_norm`). Neither
    `exact?` nor `apply?` found this two-lemma composition on their own —
    each step alone is a single Mathlib lemma, but chaining them wasn't in
    their search depth. -/
theorem chord_le_path_length
    (x v : ℝ → E) (a b : ℝ) (hab : a ≤ b)
    (hderiv : ∀ t ∈ Set.Icc a b, HasDerivAt x (v t) t)
    (hint : IntervalIntegrable v MeasureTheory.volume a b) :
    ‖x b - x a‖ ≤ ∫ t in a..b, ‖v t‖ := by
  have hderiv' : ∀ t ∈ Set.uIcc a b, HasDerivAt x (v t) t := by
    rw [Set.uIcc_of_le hab]
    exact hderiv
  have heq : ∫ t in a..b, v t = x b - x a :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv' hint
  rw [← heq]
  exact intervalIntegral.norm_integral_le_integral_norm hab

-- Equality case: a constant velocity field `c` over a nondegenerate
-- interval gives straightness ratio exactly 1 — the rectified-flow ideal.
-- Proof: the integral of the constant `‖c‖` is `(b - a) * ‖c‖`
-- (`intervalIntegral.integral_const`); the chord is `(b - a) • c`
-- (`sub_smul` + `abel`); the ratio then cancels by `field_simp`.
omit [CompleteSpace E] in
theorem straightness_eq_one_of_const_velocity
    (x0 c : E) (a b : ℝ) (hab : a < b) (hc : c ≠ 0) :
    straightnessRatio (fun t => x0 + t • c) (fun _ => c) a b = 1 := by
  have hsub : (x0 + b • c) - (x0 + a • c) = (b - a) • c := by
    rw [sub_smul]; abel
  unfold straightnessRatio
  simp only [intervalIntegral.integral_const, hsub, norm_smul, Real.norm_eq_abs,
    smul_eq_mul]
  rw [abs_of_pos (sub_pos.mpr hab)]
  have hcne : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  have hbane : (b - a) ≠ 0 := sub_ne_zero.mpr (ne_of_gt hab)
  field_simp

end RFVProofs
