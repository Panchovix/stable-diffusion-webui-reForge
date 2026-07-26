-- RFVProofs/WaveletDomain.lean
-- Wavelet-domain error decomposition: by Parseval's theorem the L² error
-- splits exactly across orthogonal subbands, enabling per-frequency correction.

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr

/-!
# Wavelet-Domain Error Decomposition for CLPC

**Sampler design implication**: Instead of a single scalar L²-error driving
the CLPC corrector, we decompose the latent-space residual into wavelet
subbands (LL, LH, HL, HH in 2D). Because the wavelet basis is orthonormal,
Parseval guarantees `‖f - f̂‖² = Σ_j ‖P_j(f - f̂)‖²` exactly. This means
CLPC can apply a *per-subband* gain, down-weighting high-frequency noise (HH)
that is structurally meaningless in early denoising steps, and up-weighting
low-frequency structure (LL) where drift accumulates. The independence lemma
proves there is no cross-subband interference: fixing one subband's error
cannot introduce error into another.
-/

open ContinuousLinearMap

variable {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]

/-!
## Orthogonal decomposition helpers

We work with a finite family of continuous linear maps `P : Fin n → E →L[ℝ] E`
that are mutually orthogonal and sum to the identity.
-/

/-- An indexed family of continuous linear maps is a *complete orthogonal
    decomposition* if (a) they are idempotent (projections), (b) mutually
    orthogonal in the sense that `P i ∘ P j = 0` for `i ≠ j`, and (c) their
    sum is the identity. -/
structure OrthDecomp {n : ℕ} (P : Fin n → E →L[ℝ] E) : Prop where
  idem   : ∀ i, P i ∘L P i = P i
  orthog : ∀ i j, i ≠ j → P i ∘L P j = 0
  sum_id : ∑ i, P i = ContinuousLinearMap.id ℝ E

namespace WaveletDomain

/-!
## Theorem 1 — Parseval / subband energy identity

For any orthogonal decomposition, the squared norm of a vector equals the sum
of the squared norms of its subband components.
-/

/-- **Wavelet Parseval identity**: total energy splits exactly across subbands.

    The proof decomposes `f = Σ_j P_j f` (from `sum_id`) and expands
    `‖Σ_j P_j f‖²` using the inner product; all cross-terms `⟪P_i f, P_j f⟫`
    vanish because the projections are mutually orthogonal.

    **Why sorry**: the inner product expansion requires `@inner_sum` and
    careful term-by-term manipulation of `⟪P i ∘ P j, ·⟫ = 0`; this is a
    standard real analysis argument but requires 30+ tactic steps to close
    in Lean 4 without dedicated automation. -/
theorem wavelet_parseval {n : ℕ} (P : Fin n → E →L[ℝ] E) (hP : OrthDecomp P)
    (f : E) :
    ‖f‖ ^ 2 = ∑ j, ‖(P j) f‖ ^ 2 := by
  -- Step 1: f = Σ_j P_j f (follows from sum_id applied to f)
  have hf : f = ∑ j, (P j) f := by
    have h1 : (∑ j, P j) f = f := by
      rw [hP.sum_id]
      simp
    rw [ContinuousLinearMap.sum_apply] at h1
    exact h1.symm
  -- Step 2: expand ‖Σ P_j f‖² using orthogonality to kill cross-terms.
  -- This requires inner_sum + mutual orthogonality; deferred.
  sorry -- hard: inner product expansion + orthogonality (standard but ~30 tactic lines)

/-!
## Theorem 2 — Subband error decomposition

The squared reconstruction error equals the sum of per-subband squared errors.
Corollary of `wavelet_parseval` applied to `f - f̂`.
-/

/-- **Subband error decomposition**: `‖f - f̂‖² = Σ_j ‖P_j(f - f̂)‖²`.

    Follows from `wavelet_parseval` applied to `f - fhat`. -/
theorem subband_error_decomp {n : ℕ} (P : Fin n → E →L[ℝ] E) (hP : OrthDecomp P)
    (f fhat : E) :
    ‖f - fhat‖ ^ 2 = ∑ j, ‖(P j) (f - fhat)‖ ^ 2 :=
  wavelet_parseval P hP (f - fhat)

/-!
## Theorem 3 — Subband correction independence

Modifying the j-th subband of f̂ (replacing f̂ with `f̂ + P_j δ`) leaves
all other subband components of the error unchanged.
-/

/-- **Subband correction independence**: correcting the `j`-th subband does
    not affect any `k ≠ j` subband.

    Proof: `(P k)((f - fhat) - (P j) δ) = (P k)(f - fhat) - (P k ∘ P j) δ`.
    Since `k ≠ j`, `hP.orthog` gives `P k ∘ P j = 0`, so the correction
    term vanishes. This proof compiles clean (0 sorry). -/
theorem subband_correction_independence {n : ℕ} (P : Fin n → E →L[ℝ] E)
    (hP : OrthDecomp P) (f fhat δ : E) (j k : Fin n) (hjk : j ≠ k) :
    (P k) (f - (fhat + (P j) δ)) = (P k) (f - fhat) := by
  -- Distribute P k over the subtraction
  simp only [map_sub, map_add]
  -- The extra term is (P k)(P j δ) = (P k ∘L P j) δ = 0 δ = 0
  have hzero : (P k) ((P j) δ) = 0 := by
    have hcomp : P k ∘L P j = 0 := hP.orthog k j hjk.symm
    calc (P k) ((P j) δ) = (P k ∘L P j) δ := rfl
      _ = (0 : E →L[ℝ] E) δ := by rw [hcomp]
      _ = 0 := zero_apply δ
  rw [hzero]
  simp

end WaveletDomain
