-- RFVProofs/FlowModelCoord.lean
-- Two results about the flow-model coordinate λ(t) = t/(1-t) on (0,1):
-- (1) er_lambda_strictMono: λ is strictly monotone on (0,1)
-- (2) er_lambda_window_bound: the λ-distance between nearby t's is ≤ 2h/δ²
-- (3) coord_transfer_finDiff_bound: the LocalLinearity OT bound transfers
--     to λ-coordinates with the finite-difference error at most L*h in t-space

import RFVProofs.LocalLinearity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum

open RFVProofs Real

namespace RFVProofs

/-! ## (1) Strict monotonicity of λ(t) = t/(1-t) on (0,1) -/

/-- `λ(t) = t/(1-t)` is strictly monotone on (0,1): `s < t` implies `λ(s) < λ(t)`.

    Proof: `div_lt_div_iff₀` rewrites the goal to `s*(1-t) < t*(1-s)`,
    i.e. `s - st < t - ts`, i.e. `s < t`. `nlinarith` closes it.

    Sampler significance: as the diffusion process runs forward (increasing noise),
    `λ(t)` gives the log signal-to-noise ratio in reverse (decreasing SNR). The
    strict monotonicity means the λ-parametrization is a valid change of variables
    and the sampler's time axis is consistently ordered. -/
theorem er_lambda_strictMono (s t : ℝ) (hs : 0 < s) (hst : s < t) (ht : t < 1) :
    s / (1 - s) < t / (1 - t) := by
  have h1s : 0 < 1 - s := by linarith
  have h1t : 0 < 1 - t := by linarith
  rw [div_lt_div_iff₀ h1s h1t]
  nlinarith

/-- Monotone form: `s ≤ t` implies `λ(s) ≤ λ(t)` (non-strict version).
    Follows by cases: equality gives reflexivity, strict inequality gives
    `er_lambda_strictMono`. -/
theorem er_lambda_mono (s t : ℝ) (hs : 0 < s) (hst : s ≤ t) (ht : t < 1) :
    s / (1 - s) ≤ t / (1 - t) := by
  rcases hst.lt_or_eq with hlt | rfl
  · exact le_of_lt (er_lambda_strictMono s t hs hlt ht)
  · exact le_refl _

/-! ## (2) λ-window distance bound -/

/-- **λ-coordinate window bound.**
    When `t` and `t₀` are within `h` of each other and `1 - t₀ ≥ δ > 0`
    (i.e., we are at least `δ` away from the singularity at `t=1`) with
    `h ≤ δ/2`, the λ-distance satisfies `|λ(t) - λ(t₀)| ≤ 2h/δ²`.

    Algebraically: `λ(t) - λ(t₀) = (t-t₀)/((1-t)(1-t₀))`.
    - `1-t₀ ≥ δ` by hypothesis
    - `1-t ≥ δ/2` because `t ≤ t₀ + h ≤ (1-δ) + δ/2 = 1 - δ/2`
    - Product: `(1-t)(1-t₀) ≥ (δ/2)·δ = δ²/2`
    - Ratio: `|t-t₀|/((1-t)(1-t₀)) ≤ h/(δ²/2) = 2h/δ²`

    This bound shows the λ-coordinate does NOT blow up the window as long as
    we stay at least `δ` from the boundary `t=1`. -/
theorem er_lambda_window_bound (t t0 h δ : ℝ)
    (hδ : 0 < δ) (hh : 0 < h) (hh_small : h ≤ δ / 2)
    (ht0 : 1 - t0 ≥ δ) (ht : |t - t0| ≤ h) :
    |t / (1 - t) - t0 / (1 - t0)| ≤ 2 * h / δ ^ 2 := by
  have h1t0 : 0 < 1 - t0 := by linarith
  -- t stays below 1
  have ht_bound : t < 1 := by
    have htu : t ≤ t0 + h := by linarith [(abs_le.mp ht).2]
    linarith
  have h1t : 0 < 1 - t := by linarith
  -- 1-t ≥ δ/2 (since t ≤ t0 + h ≤ (1-δ) + δ/2 = 1-δ/2)
  have hδt : δ / 2 ≤ 1 - t := by
    have htu : t ≤ t0 + h := by linarith [(abs_le.mp ht).2]
    linarith
  -- Algebraic identity
  have heq : t / (1 - t) - t0 / (1 - t0) = (t - t0) / ((1 - t) * (1 - t0)) := by
    field_simp; ring
  rw [heq, abs_div, abs_of_pos (mul_pos h1t h1t0)]
  -- Cross-multiply: |t-t₀| * δ² ≤ 2h * (1-t)(1-t₀)
  rw [div_le_div_iff₀ (mul_pos h1t h1t0) (by positivity)]
  -- Lower bound the denominator
  have hprod : (1 - t) * (1 - t0) ≥ δ ^ 2 / 2 := by nlinarith
  nlinarith [abs_nonneg (t - t0)]

/-! ## (3) Coordinate transfer of the LocalLinearity OT bound -/

/-- **LocalLinearity OT bound in λ-coordinates (Part A).**
    The finite-difference velocity error `‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h`
    is a direct consequence of `finDiff_window_near_velocity` from LocalLinearity.lean.
    This is independent of the λ-reparametrization: it holds in t-space.

    The connection to λ-coordinates is qualitative: the bound `L * h` in t-space
    corresponds to an effective window of size `2h/δ²` in λ-space
    (by `er_lambda_window_bound`), so the "per-λ-unit" error is
    `(L * h) / (2h/δ²) = L * δ² / 2` — finite and controlled as long as δ > 0. -/
theorem coord_transfer_finDiff_bound
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (x v : ℝ → E) (t0 h L : ℝ)
    (hL : 0 ≤ L) (hh : 0 < h)
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint1 : ∀ t ∈ Set.Icc (t0 - h) (t0 + h),
      IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    {t1 t2 : ℝ} (ht1 : t1 ∈ Set.Icc (t0 - h) (t0 + h))
    (ht2 : t2 ∈ Set.Icc (t0 - h) (t0 + h)) (hne : t1 ≠ t2) :
    ‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h :=
  finDiff_window_near_velocity x v t0 h L hL hh hderiv hint1 hlip ht1 ht2 hne

/-- **Combined λ-coordinate transfer.**
    Both the finite-difference error and the λ-window size are bounded:
    - `‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h` (from LocalLinearity)
    - `|λ(t2) - λ(t1)| ≤ 2h/δ²` when `1-t0 ≥ δ > 0` and `h ≤ δ/2`

    The second bound requires `t1` and `t2` to both be in `[t0-h, t0+h]`
    and uses `1 - t1 ≥ δ/2` (from `h ≤ δ/2` and `1-t0 ≥ δ`). -/
theorem coord_transfer_combined
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (x v : ℝ → E) (t0 h L δ : ℝ)
    (hL : 0 ≤ L) (hh : 0 < h) (hδ : 0 < δ) (hh_small : h ≤ δ / 2)
    (ht0_away : 1 - t0 ≥ δ)
    (hderiv : ∀ s ∈ Set.Icc (t0 - h) (t0 + h), HasDerivAt x (v s) s)
    (hint1 : ∀ t ∈ Set.Icc (t0 - h) (t0 + h),
      IntervalIntegrable (fun s => v s - v t0) MeasureTheory.volume t0 t)
    (hlip : LipschitzVelocityOn v (Set.Icc (t0 - h) (t0 + h)) L)
    {t1 t2 : ℝ} (ht1 : t1 ∈ Set.Icc (t0 - h) (t0 + h))
    (ht2 : t2 ∈ Set.Icc (t0 - h) (t0 + h)) (hne : t1 ≠ t2) :
    -- A: finite-difference error in t-space
    ‖finDiffVelocity x t1 t2 - v t0‖ ≤ L * h ∧
    -- B: λ-window size bound (direct proof: both in [t0-h,t0+h], 1-t0 ≥ δ ≥ 2h)
    -- We get 1-t1 ≥ δ/2, 1-t2 ≥ δ/2, |t2-t1| ≤ 2h
    -- So |λ(t2) - λ(t1)| = |t2-t1|/((1-t2)(1-t1)) ≤ 2h/(δ/2)² = 8h/δ²
    |t2 / (1 - t2) - t1 / (1 - t1)| ≤ 8 * h / δ ^ 2 := by
  constructor
  · exact coord_transfer_finDiff_bound x v t0 h L hL hh hderiv hint1 hlip ht1 ht2 hne
  · rw [Set.mem_Icc] at ht1 ht2
    have hδt1 : δ / 2 ≤ 1 - t1 := by linarith [ht1.2]
    have hδt2 : δ / 2 ≤ 1 - t2 := by linarith [ht2.2]
    have h1t1 : 0 < 1 - t1 := by linarith
    have h1t2 : 0 < 1 - t2 := by linarith
    have heq : t2 / (1 - t2) - t1 / (1 - t1) = (t2 - t1) / ((1 - t2) * (1 - t1)) := by
      field_simp; ring
    rw [heq, abs_div, abs_of_pos (mul_pos h1t2 h1t1)]
    rw [div_le_div_iff₀ (mul_pos h1t2 h1t1) (by positivity)]
    have habs : |t2 - t1| ≤ 2 * h := by
      rw [abs_le]; constructor <;> linarith [ht1.1, ht1.2, ht2.1, ht2.2]
    have hprod : (1 - t2) * (1 - t1) ≥ δ ^ 2 / 4 := by nlinarith
    nlinarith [abs_nonneg (t2 - t1)]

/-! ## Bonus: the λ-map maps (0,1) onto (0,∞) -/

/-- `λ(t) = t/(1-t)` maps `(0,1)` into `(0,∞)`. -/
theorem er_lambda_pos (t : ℝ) (ht0 : 0 < t) (ht1 : t < 1) :
    0 < t / (1 - t) :=
  div_pos ht0 (by linarith)

/-- `λ` is unbounded above: for any `M`, there exists `t ∈ (0,1)` with `λ(t) > M`. -/
theorem er_lambda_unbounded (M : ℝ) : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ M < t / (1 - t) := by
  obtain hM | hM := lt_or_ge 0 M
  · -- M > 0: witness t = (M+1)/(M+2), giving λ(t) = M+1 > M
    use (M + 1) / (M + 2)
    have hM2 : 0 < M + 2 := by linarith
    refine ⟨by positivity, ?_, ?_⟩
    · rw [div_lt_one hM2]; linarith
    · have h1 : 1 - (M + 1) / (M + 2) = 1 / (M + 2) := by field_simp; ring
      rw [h1]; field_simp; nlinarith
  · -- M ≤ 0: t = 1/2 gives λ(1/2) = 1 > 0 ≥ M
    exact ⟨1/2, by norm_num, by norm_num, by norm_num; linarith⟩

end RFVProofs
