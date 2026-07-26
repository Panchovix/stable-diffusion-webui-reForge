-- RFVProofs/Stability.lean
-- Core, fully proved result: on an affine (constant-velocity) trajectory,
-- the two-point finite-difference velocity estimate is exactly constant —
-- the sampling history's observed "velocity field" never drifts. The
-- contrapositive gives the diagnostic the user is after: if two
-- finite-difference estimates taken from a real sampling run disagree, the
-- trajectory provably is NOT affine (i.e. sampling deviated from the
-- rectified-flow / constant-velocity assumption on that span).
--
-- Pure vector-space algebra — no calculus or Mathlib analysis machinery
-- needed, so this file is hand-proved (no `sorry`). Straightness.lean and
-- Perturbation.lean build the calculus-flavoured, literature-grounded
-- statements on top of Defs.lean and are the actual `explore.sh` targets.

import RFVProofs.Defs
import Mathlib.Tactic.Abel

namespace RFVProofs

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- On an affine trajectory, the finite-difference estimate recovers the
    true velocity `c` exactly, regardless of which two (distinct) sample
    times are used. This is the formal version of "linear trajectory has
    constant slope `d(x_hat)/d(x0)`". -/
theorem finDiff_affine (x0 c : E) {t1 t2 : ℝ} (h : t1 ≠ t2) :
    finDiffVelocity (AffineTraj x0 c) t1 t2 = c := by
  have hne : t2 - t1 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hsub : AffineTraj x0 c t2 - AffineTraj x0 c t1 = (t2 - t1) • c := by
    unfold AffineTraj
    rw [sub_smul]
    abel
  unfold finDiffVelocity
  rw [hsub, smul_smul, inv_mul_cancel₀ hne, one_smul]

/-- Stability across multiple steps: any two finite-difference estimates
    taken from an affine trajectory's sampling history agree — the observed
    velocity field is constant across the whole run, not just locally. -/
theorem finDiff_stable (x0 c : E) {t1 t2 t3 t4 : ℝ}
    (h12 : t1 ≠ t2) (h34 : t3 ≠ t4) :
    finDiffVelocity (AffineTraj x0 c) t1 t2 = finDiffVelocity (AffineTraj x0 c) t3 t4 := by
  rw [finDiff_affine x0 c h12, finDiff_affine x0 c h34]

/-- Same statement, generic over `IsAffine` rather than the literal
    `AffineTraj` constructor. -/
theorem stable_of_affine (x : ℝ → E) (hx : IsAffine x) {t1 t2 t3 t4 : ℝ}
    (h12 : t1 ≠ t2) (h34 : t3 ≠ t4) :
    finDiffVelocity x t1 t2 = finDiffVelocity x t3 t4 := by
  obtain ⟨x0, c, hc⟩ := hx
  have hxeq : x = AffineTraj x0 c := funext hc
  rw [hxeq]
  exact finDiff_stable x0 c h12 h34

/-- **The diagnostic.** If two finite-difference velocity estimates taken
    from a real sampling history *disagree*, the trajectory cannot be
    affine on that span — i.e. the run departed from the rectified-flow /
    constant-velocity assumption between those sample points. This is the
    formal counterpart of "observe the velocity field history and determine
    if the sampling is right." -/
theorem not_affine_of_unstable (x : ℝ → E) {t1 t2 t3 t4 : ℝ}
    (h12 : t1 ≠ t2) (h34 : t3 ≠ t4)
    (hneq : finDiffVelocity x t1 t2 ≠ finDiffVelocity x t3 t4) :
    ¬ IsAffine x :=
  fun hx => hneq (stable_of_affine x hx h12 h34)

end RFVProofs
