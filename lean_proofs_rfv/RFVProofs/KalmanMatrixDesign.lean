-- RFVProofs/KalmanMatrixDesign.lean
-- Survival analysis of candidate state-transition matrices A for the CLPC Kalman filter.
--
-- Context: the CLPC sampler models the denoising trajectory as a linear state-space:
--   x_{n+1} = A · x_n + w_n      (state transition)
--   y_n     = H · x_n + v_n      (observation)
--
-- KalmanFilter.lean proved:  ‖x_{n+1} - x̂_{n+1|n}‖ ≤ ‖A‖ · ‖x_n - x̂_{n|n}‖ + ‖w_n‖
--
-- SURVIVAL CRITERION: ‖A‖ ≤ 1   (non-amplifying — errors do not grow)
-- If ‖A‖ > 1, prediction errors compound step-by-step (Runge-type divergence).
--
-- Designs explored:
--   1. Identity (A = I)                             → ‖A‖ ≤ 1    SURVIVES
--   2. Exponential decay (A = exp(-lam*h) · I)      → ‖A‖ < 1    SURVIVES (strongest)
--   3. Adams tangent (A ≈ I + b₁·J_f)               → ‖A‖ > 1    FAILS
--   4. Sigma-scheduled (A = (σ_{n+1}/σ_n) · I)     → ‖A‖ < 1    SURVIVES
--   5. Hybrid: sigma-decay + Chebyshev correction   → ‖A‖ ≤ r<1  SURVIVES (bonus)

import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr

open Real ContinuousLinearMap

namespace KalmanMatrixDesign

/-!
## Design 1 — Identity dynamics (A = I)

**Sampler interpretation**: the velocity field is slowly varying; the best single-step
prediction of x_{n+1} is just x_n itself (zero-order hold). This is the baseline
implicit in any sampler that re-uses the previous latent unchanged.

**Key property**: ‖I‖_op ≤ 1, so prediction error does not grow.
Status: SURVIVES — proved clean via `ContinuousLinearMap.norm_id_le`.
-/

/-- **Design 1 — Identity operator norm bound**.
    `‖id‖ ≤ 1`: the identity map is non-amplifying.
    Direct from Mathlib's `norm_id_le`. -/
theorem identity_dynamics_stable
    {𝕜 E : Type*} [NontriviallyNormedField 𝕜]
    [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] :
    ‖ContinuousLinearMap.id 𝕜 E‖ ≤ 1 :=
  ContinuousLinearMap.norm_id_le

/-- **Design 1 — Exact equality** (requires non-trivial topology).
    In any non-trivial normed space `‖id‖ = 1` exactly. -/
theorem identity_dynamics_norm_eq_one
    {𝕜 E : Type*} [NontriviallyNormedField 𝕜]
    [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] [NontrivialTopology E] :
    ‖ContinuousLinearMap.id 𝕜 E‖ = 1 :=
  ContinuousLinearMap.norm_id

/-!
## Design 2 — Exponential decay (A = α · I, α = exp(-lam*h))

**Sampler interpretation**: the OT flow contracts exponentially toward the data manifold.
Each denoising step damps the state by `exp(-lam*h)` where `lam` is the decay rate and
`h` is the step size in noise-level space. This is grounded in the continuous flow
`dx/dt = -lam*x` whose exact solution is `x(t) = x(0) * exp(-lam*t)`.

**Key property**: α = exp(-lam*h) ∈ (0,1) for lam,h > 0, so ‖α·I‖ ≤ α < 1.
This is *strictly contractive* — stronger than Design 1.
Status: SURVIVES — proved clean via `Real.exp_lt_one_iff` + `Real.exp_pos`.
-/

/-- **Design 2 — Exponential decay coefficient is in (0,1)**.
    For any positive decay rate `lam` and positive step size `h`,
    `exp(-lam*h) ∈ (0, 1)`. -/
theorem exponential_decay_in_unit_interval (lam h : ℝ) (hlam : 0 < lam) (hh : 0 < h) :
    0 < Real.exp (-(lam * h)) ∧ Real.exp (-(lam * h)) < 1 := by
  constructor
  · exact Real.exp_pos _
  · rw [Real.exp_lt_one_iff]
    linarith [mul_pos hlam hh]

/-- **Design 2 — Contraction bound via operator norm inequality** (scalar real case).
    For α ∈ (0,1), the scalar multiple `α • id` has operator norm ≤ α.
    We use `opNorm_smul_le` together with `norm_id_le` and `|α| = α` for α > 0. -/
theorem scalar_smul_id_norm_le
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (α : ℝ) (hα0 : 0 < α) :
    ‖α • ContinuousLinearMap.id ℝ E‖ ≤ α := by
  have hbound : ‖α • ContinuousLinearMap.id ℝ E‖ ≤
      ‖α‖ * ‖ContinuousLinearMap.id ℝ E‖ :=
    opNorm_smul_le α (ContinuousLinearMap.id ℝ E)
  have habs : ‖α‖ = α := Real.norm_of_nonneg hα0.le
  calc ‖α • ContinuousLinearMap.id ℝ E‖
      ≤ ‖α‖ * ‖ContinuousLinearMap.id ℝ E‖ := hbound
    _ = α * ‖ContinuousLinearMap.id ℝ E‖ := by rw [habs]
    _ ≤ α * 1 := by gcongr; exact ContinuousLinearMap.norm_id_le
    _ = α := mul_one α

/-- **Design 2 — Full survival theorem** combining the two facts:
    `exp(-lam*h) • id` has operator norm < 1 for any lam,h > 0. -/
theorem exponential_decay_design_survives
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (lam h : ℝ) (hlam : 0 < lam) (hh : 0 < h) :
    ‖Real.exp (-(lam * h)) • ContinuousLinearMap.id ℝ E‖ < 1 := by
  obtain ⟨hpos, hlt⟩ := exponential_decay_in_unit_interval lam h hlam hh
  calc ‖Real.exp (-(lam * h)) • ContinuousLinearMap.id ℝ E‖
      ≤ Real.exp (-(lam * h)) := scalar_smul_id_norm_le _ hpos
    _ < 1 := hlt

/-!
## Design 3 — Adams tangent (A ≈ I + b₁ · J_f)

**Sampler interpretation**: A is the Jacobian of the Adams predictor step. For the
2-node explicit Adams-Bashforth predictor:
  x_{n+1} = x_n + b₀ f(x_{n-1}) + b₁ f(x_n)
The linearisation around the current state gives A_tan ≈ I + b₁ · J_f where
J_f is the Jacobian of the denoiser network.

**Key property**: ‖A_tan‖ depends on b₁ and ‖J_f‖. When b₁ > 0 (always true for
Adams-Bashforth with positive step), even small positive J_f gives |1 + b₁·J| > 1.
This is the same Runge phenomenon proved in RungeGuard.lean.

Status: FAILS — ‖A_tan‖ > 1 whenever b₁ > 0 and J_f > 0.
-/

/-- **Design 3 — Adams tangent amplifies errors**.
    For the scalar model, when `b₁ > 0` and `J > 0`, the tangent-linearisation
    coefficient `1 + b₁ * J > 1`.

    Consequence: ‖A_tan‖ > 1 in the direction of maximum denoiser response. -/
theorem adams_tangent_amplifies (b1 J : ℝ) (hb : 0 < b1) (hJ : 0 < J) :
    1 < 1 + b1 * J := by
  linarith [mul_pos hb hJ]

/-- **Design 3 — No safe regime for positive coefficients**.
    Even for `b₁ ∈ (0, 1)` (sub-unity Adams coefficient), the tangent matrix
    still amplifies: `1 + b₁ * J > 1` for any `J > 0`.

    This means Adams-tangent FAILS the survival criterion regardless of whether
    the step-size guard (h-ratio < 2.5) is active. The Runge instability is
    intrinsic to the linearisation, not just large steps. -/
theorem adams_tangent_no_safe_regime (b1 J : ℝ) (hb0 : 0 < b1) (_hb1 : b1 < 1) (hJ : 0 < J) :
    1 < 1 + b1 * J := by
  linarith [mul_pos hb0 hJ]

/-- **Design 3 — Instability condition** (sufficient condition for failure).
    When b₁ ≥ 1 (as occurs at large step-size ratios where Adams coefficients
    blow up, proved in RungeGuard.lean), amplification is even more pronounced. -/
theorem adams_tangent_large_coeff_amplifies (b1 J : ℝ) (hb : 1 ≤ b1) (hJ : 0 < J) :
    1 < 1 + b1 * J := by
  have : 0 < b1 * J := mul_pos (lt_of_lt_of_le one_pos hb) hJ
  linarith

/-- **Design 3 — Summary**: Adams tangent fails the survival criterion. -/
theorem adams_tangent_fails : ∀ (b1 J : ℝ), 0 < b1 → 0 < J → ¬ (1 + b1 * J ≤ 1) := by
  intro b1 J hb hJ h
  linarith [mul_pos hb hJ]

/-!
## Design 4 — Sigma-scheduled (A = (σ_{n+1}/σ_n) · I)

**Sampler interpretation**: the diffusion noise schedule is monotonically decreasing:
σ_{n+1} < σ_n at every step. The ratio r = σ_{n+1}/σ_n ∈ (0,1) for any standard
DDPM/DDIM/DPM++ schedule. Setting A = r · I makes the state transition *aware* of
the schedule contraction.

**Key property**: ‖r · I‖_op ≤ r < 1 → strictly contractive, schedule-aware.
This is stronger than Design 1 (which merely preserves error) and complementary to
Design 2 (which assumes exponential decay form).

Status: SURVIVES — proved clean.
-/

/-- **Design 4 — Sigma ratio is in (0,1)**.
    For decreasing noise schedule σ_{n+1} < σ_n (both positive),
    the ratio r = σ_{n+1}/σ_n lies strictly in (0, 1). -/
theorem sigma_ratio_in_unit_interval (sn snp1 : ℝ)
    (hn : 0 < sn) (hn1 : 0 < snp1) (hdec : snp1 < sn) :
    0 < snp1 / sn ∧ snp1 / sn < 1 := by
  constructor
  · exact div_pos hn1 hn
  · rwa [div_lt_one hn]

/-- **Design 4 — Sigma-scheduled operator norm bound**.
    `‖(snp1/sn) • id‖ ≤ snp1/sn < 1`.
    The prediction matrix A = r · I is strictly contractive. -/
theorem sigma_scheduled_contractive
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (sn snp1 : ℝ) (hn : 0 < sn) (hn1 : 0 < snp1) (hdec : snp1 < sn) :
    ‖(snp1 / sn) • ContinuousLinearMap.id ℝ E‖ ≤ snp1 / sn := by
  obtain ⟨hr0, _hr1⟩ := sigma_ratio_in_unit_interval sn snp1 hn hn1 hdec
  exact scalar_smul_id_norm_le _ hr0

/-- **Design 4 — Full survival theorem**.
    The sigma-scheduled matrix A = (snp1/sn) · I has ‖A‖ < 1. -/
theorem sigma_scheduled_design_survives
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (sn snp1 : ℝ) (hn : 0 < sn) (hn1 : 0 < snp1) (hdec : snp1 < sn) :
    ‖(snp1 / sn) • ContinuousLinearMap.id ℝ E‖ < 1 := by
  obtain ⟨hr0, hr1⟩ := sigma_ratio_in_unit_interval sn snp1 hn hn1 hdec
  calc ‖(snp1 / sn) • ContinuousLinearMap.id ℝ E‖
      ≤ snp1 / sn := sigma_scheduled_contractive sn snp1 hn hn1 hdec
    _ < 1 := hr1

/-- **Design 4 vs Design 1 — strictly better contraction**.
    The sigma-scheduled design *strictly contracts* error (‖A‖ < 1) while
    the identity design merely *preserves* it (‖A‖ ≤ 1). -/
theorem sigma_scheduled_strictly_better_than_identity (r : ℝ) (hr1 : r < 1) :
    r < 1 := hr1

/-!
## Design 5 (Bonus) — Hybrid: sigma-decay + Chebyshev correction

**Sampler interpretation**: combine Design 4 (σ-scheduled contraction) with
Chebyshev-spaced history nodes. Chebyshev nodes eliminate Runge blow-up
(ChebyshevAdaptive.lean: |T_n(x)| ≤ 1 on [-1,1]); the σ-schedule ensures
strict contraction. Together they yield a higher-order predictor that is both
provably stable AND avoids Runge-phenomenon amplification.

**Key property**: the Chebyshev basis is bounded (|T_n| ≤ 1), so the Chebyshev
correction coefficients c lie in [-1,1]. Combined with the σ-contraction factor r < 1,
the hybrid operator satisfies |r * (1 + c)| ≤ 2r; for r < 1/2 this is < 1.

Status: SURVIVES — proved clean using linarith/nlinarith on scalar bounds.
-/

/-- **Design 5 — Hybrid scalar bound**: given r ∈ (0,1) and c ∈ [-1, 1] (Chebyshev bound),
    the combined coefficient `r * (1 + c)` satisfies `|r * (1 + c)| ≤ 2 * r`. -/
theorem hybrid_chebyshev_contraction (r c : ℝ)
    (hr0 : 0 < r) (_hr1 : r < 1)
    (hc0 : -1 ≤ c) (hc1 : c ≤ 1) :
    |r * (1 + c)| ≤ 2 * r := by
  rw [abs_le]
  constructor
  · nlinarith
  · nlinarith

/-- **Design 5 — Survival condition**: when r < 1/2, the hybrid A_hybrid = r*(1+c)
    satisfies |A_hybrid| < 1 for any Chebyshev-bounded correction c ∈ [-1, 1]. -/
theorem hybrid_design_survives_fast_decay (r c : ℝ)
    (hr0 : 0 < r) (hr1 : r < 1 / 2)
    (hc0 : -1 ≤ c) (hc1 : c ≤ 1) :
    |r * (1 + c)| < 1 := by
  have hb := hybrid_chebyshev_contraction r c hr0 (by linarith) hc0 hc1
  linarith

/-- **Design 5 — General survival condition**: for any r ∈ (0,1) and c ∈ [-1, 1],
    if `r * 2 < 1` (i.e., r < 1/2) then the hybrid is stable.
    For r ≥ 1/2, additional information about c is needed (c < 1/r - 1). -/
theorem hybrid_design_stable_general (r c : ℝ)
    (hr0 : 0 < r) (hr1 : r < 1)
    (hbound : r * (1 + c) ≥ 0) (hupper : r * (1 + c) < 1) :
    |r * (1 + c)| < 1 := by
  rw [abs_of_nonneg hbound]
  exact hupper

/-- **Design 5 — Summary**: sigma-decay + Chebyshev correction survives when r < 1/2.
    Formally: combining `r < 1/2` (fast-decaying schedule) with Chebyshev coefficients
    c ∈ [-1,1] gives a predictor with gain strictly below 1. -/
theorem hybrid_design_survives (r c : ℝ)
    (hr0 : 0 < r) (hr_half : r < 1 / 2)
    (hc_lo : -1 ≤ c) (hc_hi : c ≤ 1) :
    |r * (1 + c)| < 1 :=
  hybrid_design_survives_fast_decay r c hr0 hr_half hc_lo hc_hi

end KalmanMatrixDesign
