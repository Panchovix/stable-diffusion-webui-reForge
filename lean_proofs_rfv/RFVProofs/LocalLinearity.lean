-- RFVProofs/LocalLinearity.lean
import RFVProofs.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Positivity

open MeasureTheory

namespace RFVProofs

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

-- Helper: ∫ s in a..b, (s - a) = (b - a)^2 / 2
-- Antiderivative F(u) = (u - a)^2 / 2; built from HasDerivAt.pow 2.
private lemma integral_sub_left (a b : ℝ) :
    ∫ s in a..b, (s - a) = (b - a)^2 / 2 := by
  have hderiv : ∀ s ∈ Set.uIcc a b, HasDerivAt (fun u => (u - a)^2 / 2) (s - a) s := by
    intro s _
    have hid : HasDerivAt (fun u : ℝ => u) 1 s := hasDerivAt_id s
    have hca : HasDerivAt (fun _ : ℝ => a) 0 s := hasDerivAt_const s a
    -- d/du (u - a) = 1
    have h1 : HasDerivAt (fun u : ℝ => u - a) 1 s := by
      have h := HasDerivAt.sub hid hca; convert h using 1; simp
    -- d/du (u - a)^2 = 2 * (s - a)
    have h2 : HasDerivAt (fun u : ℝ => (u - a)^2) (2 * (s - a)) s := by
      simp only [sq]
      have h := HasDerivAt.mul h1 h1; convert h using 1; ring
    -- d/du (u - a)^2 / 2 = (s - a)
    have h := HasDerivAt.div_const h2 2
    convert h using 1; ring
  have hint : IntervalIntegrable (fun s => s - a) MeasureTheory.volume a b :=
    (continuous_id.sub continuous_const).intervalIntegrable a b
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint
  simp [sub_self] at hftc
  linarith

-- Helper: ∫ s in a..b, (b - s) = (b - a)^2 / 2
-- Proof: (b - s) = (b - a) - (s - a), split and use integral_sub_left.
private lemma integral_right_sub (a b : ℝ) :
    ∫ s in a..b, (b - s) = (b - a)^2 / 2 := by
  have hint1 : IntervalIntegrable (fun _ : ℝ => b - a) MeasureTheory.volume a b :=
    continuous_const.intervalIntegrable a b
  have hint2 : IntervalIntegrable (fun s : ℝ => s - a) MeasureTheory.volume a b :=
    (continuous_id.sub continuous_const).intervalIntegrable a b
  rw [show (fun s : ℝ => b - s) = (fun s => (b - a) - (s - a)) from by ext s; ring,
      intervalIntegral.integral_sub hint1 hint2,
      intervalIntegral.integral_const, integral_sub_left, smul_eq_mul]
  ring

-- Helper: uIcc membership from window bounds
private lemma uIcc_mem_of_bounds {t0 t s h : ℝ} (ht_lo : t0 - h ≤ t) (ht_hi : t ≤ t0 + h)
    (hs : s ∈ Set.uIcc t0 t) : s ∈ Set.Icc (t0 - h) (t0 + h) := by
  rcases Set.mem_uIcc.mp hs with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact ⟨by linarith, by linarith⟩
  · exact ⟨by linarith, by linarith⟩

/-- Windowed local-linearity bound: on a window `[t0 - h, t0 + h]` where the
    velocity field is `L`-Lipschitz, the trajectory deviates from its
    tangent line at `t0` by at most `L * (t - t0)^2 / 2`. -/
theorem affine_dev_le_of_lipschitz_velocity
    (x v : ℝ → E) (t0 t h L : ℝ) (hL : 0 ≤ L) (hh : 0 ≤ h)
    (ht : t ∈ Set.Icc (t0 - h) (t0 + h))
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L) :
    ‖x t - (x t0 + (t - t0) • v t0)‖ ≤ L * (t - t0) ^ 2 / 2 := by
  have ht0 : t0 ∈ Set.Icc (t0 - h) (t0 + h) := ⟨by linarith, by linarith⟩
  obtain ⟨ht_lo, ht_hi⟩ := ht
  have hderiv' : ∀ s ∈ Set.uIcc t0 t, HasDerivAt x (v s) s := fun s hs =>
    hderiv s (uIcc_mem_of_bounds ht_lo ht_hi hs)
  have hint_const : IntervalIntegrable (fun _ : ℝ => v t0) MeasureTheory.volume t0 t :=
    continuous_const.intervalIntegrable t0 t
  have hint_v : IntervalIntegrable v MeasureTheory.volume t0 t := by
    have h1 := hint.add hint_const; simp_rw [sub_add_cancel] at h1; exact h1
  have hftc : ∫ s in t0..t, v s = x t - x t0 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv' hint_v
  have heq : x t - (x t0 + (t - t0) • v t0) = ∫ s in t0..t, (v s - v t0) := by
    have hisub := intervalIntegral.integral_sub hint_v hint_const
    rw [intervalIntegral.integral_const, hftc] at hisub
    rw [hisub]; abel
  rw [heq]
  rcases le_total t0 t with h_le | h_le
  · -- Forward: t0 ≤ t
    have hint_Lst : IntervalIntegrable (fun s => L * (s - t0)) MeasureTheory.volume t0 t :=
      ((continuous_id.sub continuous_const).intervalIntegrable t0 t).const_mul L
    calc ‖∫ s in t0..t, (v s - v t0)‖
        ≤ ∫ s in t0..t, ‖v s - v t0‖ :=
          intervalIntegral.norm_integral_le_integral_norm h_le
      _ ≤ ∫ s in t0..t, L * (s - t0) := by
          apply intervalIntegral.integral_mono_on h_le hint.norm hint_Lst
          intro s hs
          have hs_win : s ∈ Set.Icc (t0 - h) (t0 + h) :=
            ⟨by linarith [hs.1], by linarith [hs.2, ht_hi]⟩
          have hlips := hlip t0 ht0 s hs_win
          -- |s - t0| = s - t0 since s ≥ t0
          rwa [abs_of_nonneg (sub_nonneg.mpr hs.1)] at hlips
      _ = L * (t - t0) ^ 2 / 2 := by
          rw [intervalIntegral.integral_const_mul, integral_sub_left]; ring
  · -- Backward: t ≤ t0, flip integration direction
    have h_le' : t ≤ t0 := h_le
    rw [intervalIntegral.integral_symm, norm_neg]
    have hint_rev : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t t0 :=
      hint.symm
    have hint_Lt0s : IntervalIntegrable (fun s => L * (t0 - s)) MeasureTheory.volume t t0 :=
      ((continuous_const.sub continuous_id).intervalIntegrable t t0).const_mul L
    calc ‖∫ s in t..t0, (v s - v t0)‖
        ≤ ∫ s in t..t0, ‖v s - v t0‖ :=
          intervalIntegral.norm_integral_le_integral_norm h_le'
      _ ≤ ∫ s in t..t0, L * (t0 - s) := by
          apply intervalIntegral.integral_mono_on h_le' hint_rev.norm hint_Lt0s
          intro s hs
          have hs_win : s ∈ Set.Icc (t0 - h) (t0 + h) :=
            ⟨by linarith [hs.1, ht_lo], by linarith [hs.2]⟩
          have hlips := hlip t0 ht0 s hs_win
          -- |s - t0| = -(s - t0) = t0 - s since s ≤ t0
          rw [abs_of_nonpos (sub_nonpos.mpr hs.2), neg_sub] at hlips
          exact hlips
      _ = L * (t - t0) ^ 2 / 2 := by
          rw [intervalIntegral.integral_const_mul, integral_right_sub]; ring

/-- Corollary: the finite-difference velocity estimate over any two points
    `t1 t2` inside the window is within `L * h` of the true instantaneous
    velocity `v t0` at the window center. -/
theorem finDiff_window_near_velocity
    (x v : ℝ → E) (t0 h L : ℝ) (hL : 0 ≤ L) (hh : 0 < h)
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint1 : ∀ t ∈ Set.Icc (t0 - h) (t0 + h),
      IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    {t1 t2 : ℝ} (ht1 : t1 ∈ Set.Icc (t0 - h) (t0 + h))
    (ht2 : t2 ∈ Set.Icc (t0 - h) (t0 + h)) (hne : t1 ≠ t2) :
    ‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h := by
  have hne' : t2 - t1 ≠ 0 := sub_ne_zero.mpr (Ne.symm hne)
  have key : finDiffVelocity x t1 t2 - v t0
      = (t2 - t1)⁻¹ • (x t2 - x t1 - (t2 - t1) • v t0) := by
    unfold finDiffVelocity
    simp only [smul_sub, smul_smul, inv_mul_cancel₀ hne', one_smul]
  rw [key, norm_smul, norm_inv, Real.norm_eq_abs]
  have hderiv12 : ∀ s ∈ Set.uIcc t1 t2, HasDerivAt x (v s) s := fun s hs => by
    apply hderiv
    rcases Set.mem_uIcc.mp hs with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact ⟨by linarith [ht1.1], by linarith [ht2.2]⟩
    · exact ⟨by linarith [ht2.1], by linarith [ht1.2]⟩
  have hint12 : IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t1 t2 :=
    (hint1 t1 ht1).symm.trans (hint1 t2 ht2)
  have hint_const12 : IntervalIntegrable (fun _ : ℝ => v t0) MeasureTheory.volume t1 t2 :=
    continuous_const.intervalIntegrable t1 t2
  have hint_v12 : IntervalIntegrable v MeasureTheory.volume t1 t2 := by
    have h1 := hint12.add hint_const12; simp_rw [sub_add_cancel] at h1; exact h1
  have hftc12 : ∫ s in t1..t2, v s = x t2 - x t1 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv12 hint_v12
  have heq12 : x t2 - x t1 - (t2 - t1) • v t0 = ∫ s in t1..t2, (v s - v t0) := by
    have hisub := intervalIntegral.integral_sub hint_v12 hint_const12
    rw [intervalIntegral.integral_const, hftc12] at hisub
    exact hisub.symm
  have hbound : ‖∫ s in t1..t2, (v s - v t0)‖ ≤ L * h * |t2 - t1| := by
    rcases le_total t1 t2 with h_le | h_ge
    · rw [abs_of_nonneg (by linarith)]
      have hint_Lh : IntervalIntegrable (fun _ : ℝ => L * h) MeasureTheory.volume t1 t2 :=
        continuous_const.intervalIntegrable t1 t2
      calc ‖∫ s in t1..t2, (v s - v t0)‖
          ≤ ∫ s in t1..t2, ‖v s - v t0‖ :=
            intervalIntegral.norm_integral_le_integral_norm h_le
        _ ≤ ∫ s in t1..t2, L * h := by
            apply intervalIntegral.integral_mono_on h_le hint12.norm hint_Lh
            intro s hs
            have hs_win : s ∈ Set.Icc (t0 - h) (t0 + h) :=
              ⟨by linarith [hs.1, ht1.1], by linarith [hs.2, ht2.2]⟩
            calc ‖v s - v t0‖
                ≤ L * |s - t0| := hlip t0 (by constructor <;> linarith) s hs_win
              _ ≤ L * h := mul_le_mul_of_nonneg_left
                    (by rw [abs_le]; constructor <;> [linarith [hs_win.1]; linarith [hs_win.2]])
                    hL
        _ = L * h * (t2 - t1) := by
            rw [intervalIntegral.integral_const, smul_eq_mul]; ring
    · have h_lt : t2 < t1 := lt_of_le_of_ne h_ge (Ne.symm hne)
      rw [abs_of_neg (by linarith), intervalIntegral.integral_symm, norm_neg]
      have h_le' : t2 ≤ t1 := h_ge
      have hint_Lh : IntervalIntegrable (fun _ : ℝ => L * h) MeasureTheory.volume t2 t1 :=
        continuous_const.intervalIntegrable t2 t1
      calc ‖∫ s in t2..t1, (v s - v t0)‖
          ≤ ∫ s in t2..t1, ‖v s - v t0‖ :=
            intervalIntegral.norm_integral_le_integral_norm h_le'
        _ ≤ ∫ s in t2..t1, L * h := by
            apply intervalIntegral.integral_mono_on h_le' hint12.symm.norm hint_Lh
            intro s hs
            have hs_win : s ∈ Set.Icc (t0 - h) (t0 + h) :=
              ⟨by linarith [hs.1, ht2.1], by linarith [hs.2, ht1.2]⟩
            calc ‖v s - v t0‖
                ≤ L * |s - t0| := hlip t0 (by constructor <;> linarith) s hs_win
              _ ≤ L * h := mul_le_mul_of_nonneg_left
                    (by rw [abs_le]; constructor <;> [linarith [hs_win.1]; linarith [hs_win.2]])
                    hL
        _ = L * h * -(t2 - t1) := by
            rw [intervalIntegral.integral_const, smul_eq_mul]; ring
  rw [heq12]
  calc |t2 - t1|⁻¹ * ‖∫ s in t1..t2, (v s - v t0)‖
      ≤ |t2 - t1|⁻¹ * (L * h * |t2 - t1|) :=
        mul_le_mul_of_nonneg_left hbound (by positivity)
    _ = L * h := by field_simp [abs_ne_zero.mpr hne']

end RFVProofs
