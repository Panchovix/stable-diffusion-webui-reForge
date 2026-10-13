import RFVProofs.Defs
import RFVProofs.Straightness
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Tactic.Abel

open MeasureTheory intervalIntegral RFVProofs

namespace RFVProofs.Test

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

theorem affine_dev_le_of_lipschitz_velocity
    (x v : ℝ → E) (t0 t h L : ℝ) (hL : 0 ≤ L) (hh : 0 ≤ h)
    (ht : t ∈ Set.Icc (t0 - h) (t0 + h))
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L) :
    ‖x t - (x t0 + (t - t0) • v t0)‖ ≤ L * (t - t0) ^ 2 / 2 := by
  set g : ℝ → E := fun s => (x s - x t0) - (s - t0) • v t0 with hg_def
  have hwin_t0 : t0 ∈ Set.Icc (t0 - h) (t0 + h) := by constructor <;> linarith
  have hderiv_g : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt g (v s - v t0) s := by
    intro s hs
    have h1 : HasDerivAt (fun u => x u - x t0) (v s) s :=
      (hderiv s hs).sub_const (x t0)
    have h2 : HasDerivAt (fun u : ℝ => (u - t0) • v t0) (v t0) s := by
      have := ((hasDerivAt_id s).sub_const t0).smul_const (v t0)
      simpa using this
    simpa [hg_def] using h1.sub h2
  have hgt0 : g t0 = 0 := by simp [hg_def]
  have hgoal_eq : g t = x t - (x t0 + (t - t0) • v t0) := by
    simp [hg_def]; abel
  rw [← hgoal_eq]
  rcases le_total t0 t with hor | hor
  · -- case t0 ≤ t
    have hsub : Set.Icc t0 t ⊆ Set.Icc (t0 - h) (t0 + h) := by
      apply Set.Icc_subset_Icc
      · linarith [ht.1]
      · linarith [ht.2]
    have hderiv_g' : ∀ s ∈ Set.Icc t0 t, HasDerivAt g (v s - v t0) s :=
      fun s hs => hderiv_g s (hsub hs)
    have hbound : ‖g t - g t0‖ ≤ ∫ s in t0..t, ‖v s - v t0‖ :=
      chord_le_path_length g (fun s => v s - v t0) t0 t hor hderiv_g' hint
    rw [hgt0, sub_zero] at hbound
    refine hbound.trans ?_
    have hintegrand : ∀ s ∈ Set.Icc t0 t, ‖v s - v t0‖ ≤ L * (s - t0) := by
      intro s hs
      have := hlip t0 hwin_t0 s (hsub hs)
      rwa [abs_of_nonneg (by linarith [hs.1] : s - t0 ≥ 0)] at this
    have hL_int : IntervalIntegrable (fun s => L * (s - t0)) MeasureTheory.volume t0 t := by
      apply Continuous.intervalIntegrable
      fun_prop
    have hmono : (∫ s in t0..t, ‖v s - v t0‖) ≤ ∫ s in t0..t, L * (s - t0) := by
      apply intervalIntegral.integral_mono_on hor hint.norm hL_int hintegrand
    refine hmono.trans ?_
    have : (∫ s in t0..t, L * (s - t0)) = L * ((t - t0) ^ 2 / 2) := by
      rw [intervalIntegral.integral_const_mul]
      congr 1
      have h1 : (∫ s in t0..t, (s - t0)) = ∫ s in (t0 - t0)..(t - t0), s :=
        integral_comp_sub_right (fun x => x) t0
      rw [h1, sub_self, integral_id]
      ring
    rw [this]
    exact le_of_eq (by ring)
  · -- case t ≤ t0
    have hsub : Set.Icc t t0 ⊆ Set.Icc (t0 - h) (t0 + h) := by
      apply Set.Icc_subset_Icc
      · linarith [ht.1]
      · linarith [ht.2]
    have hderiv_g' : ∀ s ∈ Set.Icc t t0, HasDerivAt g (v s - v t0) s :=
      fun s hs => hderiv_g s (hsub hs)
    have hint' : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t t0 := hint.symm
    have hbound : ‖g t0 - g t‖ ≤ ∫ s in t..t0, ‖v s - v t0‖ :=
      chord_le_path_length g (fun s => v s - v t0) t t0 hor hderiv_g' hint'
    rw [hgt0, zero_sub, norm_neg] at hbound
    refine hbound.trans ?_
    have hintegrand : ∀ s ∈ Set.Icc t t0, ‖v s - v t0‖ ≤ L * (t0 - s) := by
      intro s hs
      have := hlip t0 hwin_t0 s (hsub hs)
      rwa [abs_of_nonpos (by linarith [hs.2] : s - t0 ≤ 0), neg_sub] at this
    have hL_int : IntervalIntegrable (fun s => L * (t0 - s)) MeasureTheory.volume t t0 := by
      apply Continuous.intervalIntegrable
      fun_prop
    have hmono : (∫ s in t..t0, ‖v s - v t0‖) ≤ ∫ s in t..t0, L * (t0 - s) := by
      apply intervalIntegral.integral_mono_on hor hint'.norm hL_int hintegrand
    refine hmono.trans ?_
    have : (∫ s in t..t0, L * (t0 - s)) = L * ((t - t0) ^ 2 / 2) := by
      rw [intervalIntegral.integral_const_mul]
      congr 1
      have h1 : (∫ s in t..t0, (t0 - s)) = -(∫ s in t..t0, (s - t0)) := by
        rw [← intervalIntegral.integral_neg]; congr 1; funext s; ring
      rw [h1]
      have h2 : (∫ s in t..t0, (s - t0)) = ∫ s in (t - t0)..(t0 - t0), s :=
        integral_comp_sub_right (fun x => x) t0
      rw [h2, sub_self, integral_id]
      ring
    rw [this]
    exact le_of_eq (by ring)

end RFVProofs.Test
