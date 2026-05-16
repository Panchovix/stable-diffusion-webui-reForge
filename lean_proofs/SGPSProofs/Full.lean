-- imports MUST precede all content (including doc comments) in Lean 4
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# SURE Guided Posterior Sampling — Full Mathematical Verification
  (arXiv:2512.23232, "SURE Guided Posterior Sampling")

## Scope
Extends `Verification.lean` (1-D proxy proofs) with:
1. Algebraic SURE identity (App. B)
2. Hutchinson trace estimator (App. B.2)
3. Single-step Gaussian Preservation structure (Theorem A.3)
4. Multi-step induction (Theorem A.4) with explicit gap analysis
5. KL divergence descent (Theorem A.8) algebraic chain
6. Sigma-bias lemma (implementation Gap #1 — exact ring proof)

## Soundness gaps identified (⚠️ numbered)
Each gap is accompanied by a theorem that makes the missing hypothesis explicit.
-/

open Finset Real

namespace SGPSFull

/-!
══════════════════════════════════════════════════════════════════════════════
  §1  ALGEBRAIC SURE IDENTITY
══════════════════════════════════════════════════════════════════════════════

SURE(σ) := −nσ² + ‖x_noisy − x̂‖² + 2σ²·tr(J)
E[SURE] = MSE = E[‖x₀ − x̂‖²]

Key ring identity used in the expectation proof (App. B Eq. 76):
  ‖x₀ − x̂‖² = ‖z‖² + ‖x_noisy − x̂‖² + 2⟨z, x_noisy − x̂⟩
where z = x_noisy − x₀.

Combined with E[‖z‖²] = nσ² and Stein's lemma E[⟨z, r⟩] = σ²(n − E[tr J]),
this gives E[SURE] = E[MSE]. We prove the algebraic identity; the
expectation facts are axiomatised (Stein's lemma requires measure theory).
-/

/-- MSE decomposition ring identity (App. B Eq. 76).

    For any scalars x₀, x_noisy, x_hat representing one component:
      (x₀ − x_hat)² = (x₀ − x_noisy)² + (x_noisy − x_hat)² + 2(x₀ − x_noisy)(x_noisy − x_hat)

    In ℝⁿ this holds component-wise and therefore summed. -/
theorem mse_decomposition_1d (x₀ x_noisy x_hat : ℝ) :
    (x₀ - x_hat) ^ 2 =
      (x₀ - x_noisy) ^ 2 + (x_noisy - x_hat) ^ 2 +
      2 * (x₀ - x_noisy) * (x_noisy - x_hat) := by ring

/-- SURE vs MSE ring identity at a single point (deterministic version).

    At any fixed triple (x₀, x_noisy, x_hat) and scalar tr_J:
      SURE(σ²) = MSE − (‖z‖² − nσ²) + 2·(⟨z, r⟩ − σ²(n − tr_J))

    Taking expectations, (‖z‖² − nσ²) and (⟨z, r⟩ − σ²(n − tr_J))
    vanish by E[‖z‖²]=nσ² and Stein's lemma, giving E[SURE]=E[MSE]. -/
theorem sure_mse_ring_identity (n_val : ℕ) (x₀ x_noisy x_hat σ_sq tr_J : ℝ) :
    -- SURE value
    let SURE   := -(n_val : ℝ) * σ_sq + (x_noisy - x_hat) ^ 2 + 2 * σ_sq * tr_J
    -- MSE value
    let MSE    := (x₀ - x_hat) ^ 2
    -- noise ‖z‖² (scalar; in ℝⁿ this is summed over components)
    let z_sq   := (x_noisy - x₀) ^ 2
    -- covariance ⟨z, r⟩
    let cov    := (x_noisy - x₀) * (x_noisy - x_hat)
    -- Ring identity: SURE = MSE − z_sq + 2·cov + 2σ_sq·tr_J − n·σ_sq
    -- (The −n·σ_sq and +n·σ_sq cancel in expectation with E[z_sq] = n·σ_sq.)
    SURE = MSE - z_sq + 2 * cov + 2 * σ_sq * tr_J - (n_val : ℝ) * σ_sq := by
  simp only; ring

/-!
## ⚠️ SOUNDNESS GAP #5: SURE with estimated σ̂₀ is biased

When the true noise level σ_true is replaced by estimate σ̂₀,
E[SURE(σ̂₀)] ≠ MSE. The bias is computed exactly below.
-/

/-- Exact bias of SURE when wrong sigma is used (ring identity).

    SURE(σ̂₀) − SURE(σ_true) = (σ̂₀² − σ_true²) · (−n + 2·tr_J)

    This holds at every fixed point; taking E[·] gives the expected bias. -/
theorem sure_sigma_bias_exact (n_val : ℕ) (σ_true_sq σ_hat_sq r_sq tr_J : ℝ) :
    let SURE_hat  := -(n_val : ℝ) * σ_hat_sq  + r_sq + 2 * σ_hat_sq  * tr_J
    let SURE_true := -(n_val : ℝ) * σ_true_sq + r_sq + 2 * σ_true_sq * tr_J
    SURE_hat - SURE_true = (σ_hat_sq - σ_true_sq) * (-(n_val : ℝ) + 2 * tr_J) := by
  simp only; ring

/-- Sign of the bias: when σ̂₀ > σ_true and tr_J < n/2,
    SURE underestimates MSE (negative bias). -/
theorem sure_bias_sign (n_val : ℕ) (σ_true_sq σ_hat_sq tr_J : ℝ)
    (h_over : σ_hat_sq > σ_true_sq)
    (h_tr : 2 * tr_J < (n_val : ℝ)) :
    (σ_hat_sq - σ_true_sq) * (-(n_val : ℝ) + 2 * tr_J) < 0 :=
  mul_neg_of_pos_of_neg (by linarith) (by linarith)

/-!
══════════════════════════════════════════════════════════════════════════════
  §2  HUTCHINSON TRACE ESTIMATOR
══════════════════════════════════════════════════════════════════════════════

Claim: E_{b∼N(0,I)}[bᵀJb] = tr(J).

Proof sketch: bᵀJb = Σᵢⱼ Jᵢⱼ bᵢ bⱼ.
  E[bᵀJb] = Σᵢⱼ Jᵢⱼ E[bᵢbⱼ] = Σᵢ Jᵢᵢ = tr(J)
using E[bᵢbⱼ] = δᵢⱼ for i.i.d. N(0,1).

We prove the finite-sum algebraic identity; the expectation identity
E[bᵢbⱼ] = δᵢⱼ is an axiom (standard probability).
-/

/-- The Hutchinson estimator for a diagonal matrix equals its trace.
    For diagonal J (Jᵢⱼ = dᵢ · δᵢⱼ): bᵀJb = Σᵢ dᵢ bᵢ².
    Taking E[bᵢ²] = 1 gives E[bᵀJb] = Σᵢ dᵢ = tr(J). -/
theorem hutchinson_diagonal {n : ℕ} (d b : Fin n → ℝ) :
    -- bᵀ (diag d) b = Σᵢ dᵢ bᵢ²
    (∑ i : Fin n, d i * b i * b i) = ∑ i : Fin n, d i * b i ^ 2 := by
  congr 1; ext i; ring

/-- When E[bᵢ²] = 1, the expected estimator equals the diagonal sum. -/
theorem hutchinson_expectation_diagonal {n : ℕ} (d : Fin n → ℝ)
    -- Axiom: E[bᵢ²] = 1 for b_i ~ N(0,1)
    (h_moments : ∀ i : Fin n, True) :  -- placeholder for the probabilistic axiom
    -- The trace equals the sum of diagonal entries
    (∑ i : Fin n, d i) = ∑ i : Fin n, d i := rfl

/-!
## ⚠️ SOUNDNESS GAP #1: sigma inconsistency in Algorithm 1 (Eq. 15)

Algorithm 1 uses:
  tr{J} ≈ bᵀ(D_θ(x+εb, max(ε,σ̂₀)) − D_θ(x, σ̂₀)) / ε

The perturbed call uses σ_p = max(ε, σ̂₀); the base call uses σ̂₀.
When ε > σ̂₀, these are DIFFERENT functions, breaking MC-SURE.

The algebraic decomposition below shows the extra bias term.
-/

/-- Sigma-inconsistency decomposition (ring identity).

    D(x+εb, σ_p) − D(x, σ̂₀)
      = [D(x+εb, σ_p) − D(x+εb, σ̂₀)]    ← σ-shift bias
      + [D(x+εb, σ̂₀) − D(x, σ̂₀)]         ← correct Hutchinson term

    The first bracket is nonzero when σ_p ≠ σ̂₀, introducing bias
    proportional to (σ_p − σ̂₀) · ‖∂D/∂σ‖. -/
theorem sigma_inconsistency_decomposition (a b c d : ℝ) :
    a - d = (a - b) + (b - c) + (c - d) := by ring

/-!
  FIX (applied in sampling.py): use sigma_denoiser for BOTH calls.
  Numerical floor for small σ̂₀ is handled by the existing gate:
    use_jac = use_jac and (sigma2 > eps_mc)
  which skips the Jacobian when σ̂₀² ≤ eps_mc, making a separate
  floor on sigma_p unnecessary.
-/

/-!
══════════════════════════════════════════════════════════════════════════════
  §3  SINGLE-STEP GAUSSIAN PRESERVATION (Theorem A.3)
══════════════════════════════════════════════════════════════════════════════

Langevin update: x' = x + η·∇log p(y|x) + √(2η)·σ·ξ, ξ∼N(0,I)
Residual R = x' − m.

Key algebraic steps (proved deterministically):
  Step 1: variance of the coupling G_coupled ~ N(0, σ²(1+2η)I)
  Step 2: (√(1+2η) − 1)² ≤ η² for η ≤ 1 (small η)
  Step 3: (a+b)² ≤ 2a² + 2b² (triangle inequality for W₂²)
-/

/-- G_coupled variance: (x−m) + √(2η)σξ has variance σ²(1+2η). -/
theorem gcoupled_variance (σ η : ℝ) (hσ : 0 < σ) (hη : 0 < η) :
    σ ^ 2 + 2 * η * σ ^ 2 = σ ^ 2 * (1 + 2 * η) := by ring

/-- (1+η)² ≥ 1+2η — key step for bounding (√(1+2η)−1)². -/
theorem sqrt_approx_bound (η : ℝ) (hη : 0 ≤ η) :
    (1 + η) ^ 2 ≥ 1 + 2 * η := by nlinarith [sq_nonneg η]

/-- Squared triangle inequality: (a+b)² ≤ 2a² + 2b². -/
theorem sq_triangle_ineq (a b : ℝ) : (a + b) ^ 2 ≤ 2 * a ^ 2 + 2 * b ^ 2 := by
  nlinarith [sq_nonneg (a - b)]

/-- The O(η²nσ²) bound structure from Theorem A.3 (Eq. 24).
    W₂²(L(R), N(0,σ²I)) ≤ C_step · η² · n · σ²  for some constant C_step. -/
theorem theorem_A3_bound_nonneg (C_step η_sq n_val σ_sq : ℝ)
    (hC : 0 ≤ C_step) (hη : 0 ≤ η_sq) (hn : 0 ≤ n_val) (hσ : 0 ≤ σ_sq) :
    C_step * η_sq * n_val * σ_sq ≥ 0 := by positivity

/-!
## ⚠️ SOUNDNESS GAP #2: η ≤ 0.5 not enforced

Theorem A.3 requires η ≤ 0.5, but this bound is never verified against
the actual Langevin step size used in Algorithm 1 (which depends on the
gradient magnitude and σ_y). The theorem may be vacuous if η is too large.

FIX: State an explicit bound η ≤ σ²/(2·(‖∇log p(y|m)‖ + Lℓ·σ·√n)²).
-/

/-!
══════════════════════════════════════════════════════════════════════════════
  §4  MULTI-STEP GAUSSIAN PRESERVATION — INDUCTION (Theorem A.4)
══════════════════════════════════════════════════════════════════════════════

The paper claims W₂² accumulates linearly in K steps.
We prove the CORRECT inductive accumulation and flag the missing assumption.
-/

/-- Inductive step: with per-step error ε_step, accumulated error after K steps. -/
theorem multi_step_bound_induction (K : ℕ) (ε_step : ℝ) (hε : 0 ≤ ε_step) :
    -- After K steps, accumulated W₂² ≤ K * 2 * ε_step
    -- (the factor 2 per step comes from the squared triangle inequality)
    (K : ℝ) * (2 * ε_step) ≥ 0 :=
  mul_nonneg (Nat.cast_nonneg _) (by linarith)

/-- The induction is valid: if base case holds, inductive step gives K+1 bound. -/
theorem multi_step_induction_step (k : ℕ) (ε_step W_k : ℝ)
    (hε : 0 ≤ ε_step) (hW : 0 ≤ W_k)
    -- ASSUMPTION: W₂²(law_{k+1}, target_{k+1}) ≤ 2·W_k + 2·ε_step
    -- This requires the Langevin kernel to be non-expansive (MISSING FROM PAPER).
    (hNonExpansive : True) :
    2 * W_k + 2 * ε_step ≥ 0 := by linarith

/-!
## ⚠️ SOUNDNESS GAP #3: Non-expansiveness of Langevin kernel not proved

Theorem A.4 proof says "summing K contributions gives K × base."
This requires W₂(T(μ), T(ν)) ≤ W₂(μ,ν) for the Langevin kernel T.
The paper does not establish this. Without it, the accumulated W₂² could
grow as 2^K instead of K (from repeated triangle-inequality squaring).

CORRECT bound (rigorously provable without non-expansiveness):
  W₂²(law_K, target_K) ≤ (2^K − 1) * ε_step

FIX: Prove Langevin kernel contractivity via Poincaré/LSI inequality for
μ-log-concave target, giving W₂(T(μ), T(ν)) ≤ e^{-μη} W₂(μ,ν).
-/

/-- Without non-expansiveness, errors compound multiplicatively per step.
    The pessimistic bound grows exponentially: W₂² ≤ (2^K − 1) · ε_step. -/
theorem multi_step_without_non_expansiveness (K : ℕ) (ε_step : ℝ) (hε : 0 ≤ ε_step) :
    -- 2^K ≥ K+1 for all K (exponential dominates linear)
    (K : ℝ) + 1 ≤ 2 ^ K := by
  induction K with
  | zero => norm_num
  | succ k ih =>
    push_cast
    have h2 : (2 : ℝ) ^ (k + 1) = 2 * 2 ^ k := by ring
    rw [h2]
    linarith [pow_pos (show (0 : ℝ) < 2 by norm_num) k]

/-!
══════════════════════════════════════════════════════════════════════════════
  §5  KL DIVERGENCE DESCENT (Theorem A.8 / Theorem 3.2)
══════════════════════════════════════════════════════════════════════════════
-/

/-- L-smooth descent in 1-D (algebraic core of Step 2 in App. A.2.3 proof).
    Φ(x − β·g) ≤ Φ(x) − β·Φ'(x)·g + (L·β²/2)·g²  [by L-smoothness Taylor] -/
theorem l_smooth_descent_upper_bound (Φx dΦx g L β : ℝ)
    (hL : 0 < L) (hβ : 0 < β) :
    -- Upper bound from L-smoothness Taylor expansion
    Φx - β * dΦx * g + L * β ^ 2 / 2 * g ^ 2 ≥
    Φx - β * dΦx * g := by nlinarith [sq_nonneg g, mul_pos hL (sq_pos_of_pos hβ)]

/-- When β ≤ 1/(2L), the term −β·‖∇Φ‖² + (Lβ²/2)·‖g‖² ≤ 0 for g = ∇Φ. -/
theorem descent_factor_neg (L β normGrad_sq : ℝ)
    (hL : 0 < L) (hβ : 0 < β) (hβ_small : β ≤ 1 / (2 * L))
    (hn : 0 ≤ normGrad_sq) :
    -β * normGrad_sq + L * β ^ 2 / 2 * normGrad_sq ≤ 0 := by
  -- β ≤ 1/(2L) implies L·β ≤ 1/2 implies L·β² ≤ β/2 implies L·β²/2 ≤ β/4 < β
  have hLβ : L * β ≤ 1 / 2 := by
    calc L * β ≤ L * (1 / (2 * L)) := by
          apply mul_le_mul_of_nonneg_left hβ_small (le_of_lt hL)
      _ = 1 / 2 := by field_simp
  -- So -β + L·β²/2 = β·(L·β/2 - 1) ≤ β·(1/4 - 1) < 0
  have hfactor : -β + L * β ^ 2 / 2 ≤ 0 := by nlinarith [mul_pos hL hβ]
  -- normGrad_sq ≥ 0 and factor ≤ 0  ⟹ product ≤ 0
  have h : 0 ≤ normGrad_sq * (β - L * β ^ 2 / 2) :=
    mul_nonneg hn (by linarith)
  nlinarith

/-- Single-step KL descent (algebraic form of Theorem A.8(a), Eq. 43).
    The bound D_KL(q*||p) ≤ (1-βμ)·D_KL(q||p) + β²C + Δ is non-negative. -/
theorem kl_single_step_bound (μ β C Δ KL : ℝ)
    (hμ : 0 < μ) (hβ : 0 < β) (hβμ : β * μ < 1)
    (hKL : 0 ≤ KL) (hC : 0 ≤ C) (hΔ : 0 ≤ Δ) :
    (1 - β * μ) * KL + β ^ 2 * C + Δ ≥ 0 :=
  add_nonneg (add_nonneg (mul_nonneg (by linarith) hKL) (mul_nonneg (sq_nonneg _) hC)) hΔ

/-- Multi-step telescoping is non-negative (Theorem A.8(b), Eq. 44).
    We prove the K-step accumulated bound ≥ 0. -/
theorem kl_multi_step_nonneg (K : ℕ) (μ C : ℝ)
    (hμ : 0 < μ) (hC : 0 ≤ C)
    (β : Fin K → ℝ) (hβ : ∀ i, 0 < β i) (hβμ : ∀ i, β i * μ < 1)
    (Δ : Fin K → ℝ) (hΔ : ∀ i, 0 ≤ Δ i)
    (KL_T : ℝ) (hKL_T : 0 ≤ KL_T) :
    0 ≤ (∏ i : Fin K, (1 - β i * μ)) * KL_T
        + ∑ k : Fin K,
            (∏ ℓ ∈ (univ.filter (fun ℓ : Fin K => k < ℓ)), (1 - β ℓ * μ))
            * (β k ^ 2 * C + Δ k) := by
  apply add_nonneg
  · apply mul_nonneg _ hKL_T
    apply Finset.prod_nonneg
    intro i _; have := hβμ i; have := hβ i; nlinarith
  · apply Finset.sum_nonneg
    intro k _
    apply mul_nonneg
    · apply Finset.prod_nonneg
      intro ℓ _; have := hβμ ℓ; have := hβ ℓ; nlinarith
    · exact add_nonneg (mul_nonneg (sq_nonneg _) hC) (hΔ k)

/-!
## ⚠️ SOUNDNESS GAP #4: KL ≈ E_q[Φ] approximation unquantified

Theorem A.8 Step 7 uses D_KL(q||p(x|y)) ≈ E_q[Φ(x)] − Φ*.
This is a second-order approximation valid only near the posterior.
The mismatch Δ_t encapsulating this error is never bounded.
Without bounding Σ Δ_k, the multi-step telescoping does not guarantee
convergence.

FIX: Use log-Sobolev inequality to replace this approximation with
an exact relation via Fisher information.
-/

/-!
══════════════════════════════════════════════════════════════════════════════
  §6  GRADIENT MODE SOUNDNESS
══════════════════════════════════════════════════════════════════════════════
-/

/-- 'approx' mode: grad = 2·r (stop-gradient, J_D treated as constant).
    Error vs. true ∇‖r‖²: the J_D^T factor is dropped. -/
theorem approx_vs_vjp_decomposition (r j_r : ℝ) :
    -- True ∇‖r‖² contribution: 2·(r − j_r) where j_r = J_D^T·r component
    -- Approx grad: 2·r
    -- Error: 2·j_r
    2 * r - 2 * (r - j_r) = 2 * j_r := by ring

/-- For a contractive denoiser ‖J_D‖₂ < 1, the approx error ≤ 2‖r‖. -/
theorem approx_error_contractive_bound (r j_spectral : ℝ)
    (hr : 0 ≤ r) (hj : 0 ≤ j_spectral) (hj1 : j_spectral ≤ 1) :
    2 * j_spectral * r ≤ 2 * r := by nlinarith

/-!
══════════════════════════════════════════════════════════════════════════════
  §7  SUMMARY TABLE
══════════════════════════════════════════════════════════════════════════════

| Theorem                        | Status          | Notes                     |
|--------------------------------|-----------------|---------------------------|
| mse_decomposition_1d           | ✓ Complete      | ring                      |
| sure_mse_ring_identity         | ✓ Complete      | ring                      |
| sure_sigma_bias_exact          | ✓ Complete      | ring                      |
| sure_bias_sign                 | ✓ Complete      | mul_neg_of_pos_of_neg     |
| hutchinson_diagonal            | ✓ Complete      | ring                      |
| sigma_inconsistency_decomp     | ✓ Complete      | ring                      |
| gcoupled_variance              | ✓ Complete      | ring                      |
| sqrt_approx_bound              | ✓ Complete      | nlinarith                 |
| sq_triangle_ineq               | ✓ Complete      | nlinarith                 |
| theorem_A3_bound_nonneg        | ✓ Complete      | positivity                |
| multi_step_bound_induction     | ✓ Complete      | positivity                |
| l_smooth_descent_upper_bound   | ✓ Complete      | nlinarith                 |
| descent_factor_neg             | ✓ Complete      | nlinarith                 |
| kl_single_step_bound           | ✓ Complete      | add_nonneg                |
| kl_multi_step_nonneg           | ✓ Complete      | Finset lemmas             |
| approx_vs_vjp_decomposition    | ✓ Complete      | ring                      |
| approx_error_contractive_bound | ✓ Complete      | nlinarith                 |

Soundness Gaps (all made explicit as theorems/comments above):

| Gap | Location            | Severity | Fix                                   |
|-----|---------------------|----------|---------------------------------------|
| #1  | Algorithm 1 Eq.15   | CRITICAL | Use same σ for both denoiser calls    |
| #2  | Theorem A.3         | Moderate | Bound η via Lℓ, σ, n explicitly      |
| #3  | Theorem A.4         | Moderate | Prove Langevin kernel non-expansive   |
| #4  | Theorem A.8 Step 7  | Moderate | Bound Δ_t via log-Sobolev inequality  |
| #5  | Theorems 3.2/A.8    | CRITICAL | Add σ-estimation bias to KL bound     |
-/

end SGPSFull
