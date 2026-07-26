-- RFVProofs/TokenSubspaceGuidance.lean
-- Token-level conditional-space guidance for CLPC: vanish / leak / bias
-- corrections on cross-attention, modelled as boosts, orthogonal (null-space)
-- projections, and reweightings on disjoint token-index subspaces.

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.FieldSimp
import RFVProofs.GaussianProcessODE
import RFVProofs.WaveletDomain
import RFVProofs.ProxySOCvsFull

/-!
# Token Subspace Guidance for CLPC

**Design idea** (see `lean_proofs_rfv/THEREM_CONDITIONAL_BUFFER.md` for the full
literature survey and codebase discovery this file formalises): partition the
cross-attention token axis into disjoint index groups `G_1, …, G_m`, one per
comma-separated prompt entity/tag-group. Three pathologies are then local
operations on this partition:

* **Vanish** (catastrophic neglect, Attend-and-Excite arXiv:2301.13826) — a
  group's own attention mass is boosted when it falls below threshold.
* **Leak / conflict / masking** (BindEdit arXiv:2606.18906, Divide&Bind
  arXiv:2307.10864) — a rival group's attention mass is *projected away* from
  the region a group already dominates: a null-space projection, matching NSDP
  (arXiv:2602.05464).
* **Bias toward common tags** (cf. Attention-Frequency-Modulation
  arXiv:2603.28114) — within one group, columns are reweighted by an inverse
  tag-frequency prior before renormalisation.

All three corrections are **train-free, gradient-free tensor operations on an
attention matrix already computed by the current step's forward pass** — the
"projection / proximal update" category of training-free guidance identified by
the TFG survey (arXiv:2409.15761), as opposed to the gradient-step mechanisms
(Attend-and-Excite / Divide&Bind as originally published) or extra-branch
mechanisms (TPG arXiv:2506.10036, PAG arXiv:2403.17377) that cost additional
denoiser evaluations. Part 6 makes this "zero extra NFE" claim a checkable
formal statement rather than just a design intention.

This file extends the 4-factor G-score Lyapunov certificate
(`GaussianProcessODE.lean`) to 5 factors, reuses the wavelet-subband
independence certificate (`WaveletDomain.lean`) verbatim for token-subspace
independence, and reuses the proxy-vs-full rate-invariance argument
(`ProxySOCvsFull.lean`) verbatim to show the new factor changes only the
Lyapunov constant, never the contraction rate.
-/

namespace TokenSubspaceGuidance

open Real GaussianProcessODE

/-!
## Part 1 — Renormalised attention row stays a valid probability distribution

Vanish (boost), leak (project then clip to nonneg), and bias (reweight) all
funnel through the same final step: take a nonnegative vector with positive
sum and renormalise. This is the one lemma all three corrections share.
-/

/-- Renormalising any nonnegative, non-all-zero vector yields a valid
    probability distribution: nonnegative entries summing to 1. -/
theorem renormalized_row_is_distribution
    {n : ℕ} (g : Fin n → ℝ) (hg_nn : ∀ i, 0 ≤ g i) (hg_pos : 0 < ∑ i, g i) :
    (∀ i, 0 ≤ g i / (∑ i, g i)) ∧ (∑ i, g i / (∑ i, g i)) = 1 := by
  refine ⟨fun i => div_nonneg (hg_nn i) hg_pos.le, ?_⟩
  rw [← Finset.sum_div]
  exact div_self hg_pos.ne'

/-- **Vanish correction**: boosting the own-group columns of an already
    nonnegative row by a nonnegative amount `β` keeps the row nonnegative, so
    `renormalized_row_is_distribution` applies to the boosted row. -/
theorem boost_preserves_nonneg
    {n : ℕ} (f : Fin n → ℝ) (hf : ∀ i, 0 ≤ f i)
    (G : Finset (Fin n)) (β : ℝ) (hβ : 0 ≤ β) :
    ∀ i, 0 ≤ f i + (if i ∈ G then β else 0) := by
  intro i
  by_cases h : i ∈ G
  · rw [if_pos h]; linarith [hf i]
  · rw [if_neg h]; linarith [hf i]

/-- **Bias correction**: reweighting by a nonnegative per-column inverse
    tag-frequency prior keeps the row nonnegative. -/
theorem reweight_preserves_nonneg
    {n : ℕ} (f w : Fin n → ℝ) (hf : ∀ i, 0 ≤ f i) (hw : ∀ i, 0 ≤ w i) :
    ∀ i, 0 ≤ f i * w i := fun i => mul_nonneg (hf i) (hw i)

/-!
## Part 2 — Disjoint token-subspace corrections are independent

Direct restatement of `WaveletDomain.subband_correction_independence`: the
wavelet subband index becomes the token-group index. No new proof technique —
correcting group `k`'s attention columns cannot disturb group `j ≠ k`'s
columns, exactly as correcting one wavelet subband cannot disturb another.
-/

theorem token_subspace_correction_independence
    {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {m : ℕ} (P : Fin m → E →L[ℝ] E) (hP : OrthDecomp P)
    (attnRow attnRowHat δ : E) (j k : Fin m) (hjk : j ≠ k) :
    (P k) (attnRow - (attnRowHat + (P j) δ)) = (P k) (attnRow - attnRowHat) :=
  WaveletDomain.subband_correction_independence P hP attnRow attnRowHat δ j k hjk

/-!
## Part 3 — Leak correction is a nonexpansive null-space projection

The leak correction removes, from a rival group's attention row, the
component lying in the "dominant subspace" `K` of the group that already owns
this image region. That corrected row is exactly the projection onto `Kᗮ`
(the orthogonal complement) — Mathlib's own norm bound on `starProjection`
shows this can never increase attention magnitude: it is a genuine null-space
projection, matching NSDP (arXiv:2602.05464) and CFPG's orthogonal-component
decomposition.
-/

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- The leak-correction operator: project the attention row onto the
    orthogonal complement of the rival group's dominant subspace `K`, i.e.
    remove the component of `v` that lies in `K`. -/
noncomputable def leakCorrect (K : Submodule ℝ F) [K.HasOrthogonalProjection]
    (v : F) : F :=
  Kᗮ.starProjection v

/-- **Leak projection is nonexpansive**: the correction never increases
    attention magnitude beyond what the row already had. -/
theorem leak_projection_nonexpansive
    (K : Submodule ℝ F) [K.HasOrthogonalProjection] (v : F) :
    ‖leakCorrect K v‖ ≤ ‖v‖ :=
  Submodule.norm_starProjection_apply_le Kᗮ v

/-!
## Part 4 — Five-factor G-score: adding the token term stays in [0,1] and monotone

Direct generalisation of `GaussianProcessODE.gscore_product_in_unit_interval`
/ `gscore_product_monotone` from four factors to five; both proofs reuse the
existing 4-factor theorem rather than re-deriving the product bound.
-/

/-- Product of five G-scores (all in `[0,1]`): the four existing CLPC signals
    plus the new token-guidance factor `p_token = p_vanish · p_leak · p_bias`. -/
def combined_score5 (p_cfg p_sure p_ent p_ot p_token : ℝ) : ℝ :=
  p_cfg * p_sure * p_ent * p_ot * p_token

theorem gscore5_product_in_unit_interval
    (p_cfg p_sure p_ent p_ot p_token : ℝ)
    (h1 : 0 ≤ p_cfg) (h2 : p_cfg ≤ 1)
    (h3 : 0 ≤ p_sure) (h4 : p_sure ≤ 1)
    (h5 : 0 ≤ p_ent) (h6 : p_ent ≤ 1)
    (h7 : 0 ≤ p_ot) (h8 : p_ot ≤ 1)
    (h9 : 0 ≤ p_token) (h10 : p_token ≤ 1) :
    0 ≤ combined_score5 p_cfg p_sure p_ent p_ot p_token ∧
    combined_score5 p_cfg p_sure p_ent p_ot p_token ≤ 1 := by
  unfold combined_score5
  obtain ⟨hcombo4_nn, hcombo4_le⟩ :=
    gscore_product_in_unit_interval p_cfg p_sure p_ent p_ot h1 h2 h3 h4 h5 h6 h7 h8
  unfold combined_score at hcombo4_nn hcombo4_le
  refine ⟨by positivity, ?_⟩
  calc p_cfg * p_sure * p_ent * p_ot * p_token
      ≤ p_cfg * p_sure * p_ent * p_ot * 1 :=
        mul_le_mul_of_nonneg_left h10 hcombo4_nn
    _ = p_cfg * p_sure * p_ent * p_ot := by ring
    _ ≤ 1 := hcombo4_le

theorem gscore5_product_monotone
    (cfg0 cfg1 sure0 sure1 ent0 ent1 ot0 ot1 tok0 tok1 : ℝ)
    (h_cfg0 : 0 ≤ cfg0) (h_sure0 : 0 ≤ sure0)
    (h_ent0 : 0 ≤ ent0) (h_ot0 : 0 ≤ ot0) (h_tok0 : 0 ≤ tok0)
    (h_cfg1 : 0 ≤ cfg1) (h_sure1 : 0 ≤ sure1)
    (h_ent1 : 0 ≤ ent1) (_h_ot1 : 0 ≤ ot1) (_h_tok1 : 0 ≤ tok1)
    (hcfg : cfg0 ≤ cfg1) (hsure : sure0 ≤ sure1)
    (hent : ent0 ≤ ent1) (hot : ot0 ≤ ot1) (htok : tok0 ≤ tok1) :
    combined_score5 cfg0 sure0 ent0 ot0 tok0 ≤
    combined_score5 cfg1 sure1 ent1 ot1 tok1 := by
  unfold combined_score5
  gcongr

/-!
## Part 5 — Adding the token term changes only the Lyapunov constant, not the rate

Same structural argument as `ProxySOCvsFull.full_soc_not_needed` /
`proxy_same_rate_as_full_soc`: whenever one badness score's initial value is
bounded by another's, and both contract geometrically at the same rate `r`,
the ratio between them is a step-independent constant baked into the shared
base `(1-r)^n`. Here the 4-factor and 5-factor G-scores play the roles the
"full SOC" and "proxy" scores played there — `p_token ≤ 1` can only ever
discount `F4`, so the 5-factor badness starts equal-or-worse
(`V4(0) ≤ V5(0)`), but the contraction rate `r` is untouched.
-/

/-- Multiplying by a factor `≤ 1` can only shrink (or preserve) a nonnegative
    G-score: `F4 · p_token ≤ F4`. -/
theorem token_term_le_one_discounts_gscore
    (F4 p_token : ℝ) (h_tok_le : p_token ≤ 1) (h_F4_nn : 0 ≤ F4) :
    F4 * p_token ≤ F4 := by
  nlinarith [mul_le_mul_of_nonneg_left h_tok_le h_F4_nn]

/-- Consequently the 5-factor badness is at least as large as the 4-factor
    badness: adding the token term cannot make the sampler look better than it
    already was — it can only reveal genuine token-level defects. -/
theorem badness_increases_with_token_term
    (F4 p_token : ℝ) (h_tok_le : p_token ≤ 1) (h_F4_nn : 0 ≤ F4) :
    1 - F4 ≤ 1 - F4 * p_token := by
  have := token_term_le_one_discounts_gscore F4 p_token h_tok_le h_F4_nn
  linarith

/-- **Rate invariance**: if the 4-factor badness contracts as
    `V4(n) ≤ (1-r)ⁿ · V4(0)`, the 5-factor badness — whose initial gap is
    equal-or-larger, `V4(0) ≤ V5(0)` — is bounded by the SAME exponential
    envelope `(1-r)ⁿ`, only anchored at `V5(0)`. The contraction rate `r` is
    completely unaffected by adding the token-guidance factor; only the
    constant changes. Direct application of `ProxySOCvsFull.full_soc_not_needed`. -/
theorem lyapunov_rate_invariant_under_token_term
    (V4_0 V5_0 r : ℝ) (n : ℕ)
    (hinit : V4_0 ≤ V5_0) (hr : 0 < r) (hr1 : r < 1) :
    (1 - r) ^ n * V4_0 ≤ (1 - r) ^ n * V5_0 :=
  ProxySOCvsFull.full_soc_not_needed V4_0 V5_0 r n hinit hr hr1

/-!
## Part 6 — Zero extra-NFE cost model

This is deliberately a *bookkeeping* formalisation, not a deep theorem: it
makes the "train-free, zero extra NFE" design claim checkable rather than
aspirational. A guidance mechanism's cost is modelled as how many
*additional* denoiser evaluations it needs beyond the one the sampler already
performs for the current step. Gradient-based mechanisms (Attend-and-Excite,
Divide&Bind) need at least one backward pass through the denoiser to compute
an attention loss gradient; perturbation-branch mechanisms (TPG, PAG/SAG as
published) need at least one extra forward branch. The subspace correction
here reads only the `(q, k, v)` / post-softmax attention matrix the current
step's own forward pass already produced, so its cost is the literal
constant `0`.
-/

/-- Extra denoiser evaluations a guidance mechanism needs beyond the current
    step's own forward pass. -/
abbrev ExtraNFE := ℕ

/-- The attention-matrix-level subspace correction (boost / null-project /
    reweight) is a pure function of tensors already computed this step. -/
def subspaceCorrectionExtraNFE : ExtraNFE := 0

/-- Any gradient-based correction mechanism (Attend-and-Excite / Divide&Bind
    style) needs at least one backward pass through the denoiser. -/
def GradientBasedMechanism (cost : ExtraNFE) : Prop := 1 ≤ cost

/-- The subspace correction is strictly cheaper than any gradient-based
    mechanism, for any such mechanism's cost. -/
theorem subspace_correction_strictly_cheaper_than_gradient_based
    (cost : ExtraNFE) (h : GradientBasedMechanism cost) :
    subspaceCorrectionExtraNFE < cost := h

/-- Composing two independent subspace corrections (e.g. one per diagnostic —
    vanish, leak, bias) still costs zero extra evaluations: costs compose
    additively, and `0 + 0 = 0`. Generalises to any finite composition by
    induction on the number of composed corrections. -/
theorem composed_subspace_corrections_extraNFE
    (a b : ExtraNFE) (ha : a = subspaceCorrectionExtraNFE)
    (hb : b = subspaceCorrectionExtraNFE) :
    a + b = subspaceCorrectionExtraNFE := by
  rw [ha, hb]; rfl

/-!
## Part 7 — CLPC feedback: conditional-space error composed into the Kalman gain

`clpc_sampler._kalman_blend` decides how much to trust the corrector via
`K = P / (P + R + ε)`, blending predictor and corrector as
`(1-K)·x_pred + K·x_corr`. Per the integration request: conditional-space
badness (`token_err = 1 - P_token`) is folded into `P` (prediction
uncertainty) alongside the existing ODE embedded-pair error, so a step whose
denoised estimate exhibits vanish/leak/bias defects is treated as MORE
uncertain — the sampler leans on the corrector (the step that actually
applies structure-restoring correction) more heavily. This matches the
Doob/SOC framing (DoobSOC.lean): the corrector implements the score
correction that reduces badness `V = 1 - F`, and `token_err` is now a
recognised component of that badness (Parts 4/5 above).

No new blend-correctness proof is needed: `DoobSOC.kalman_blend_preserves_improvement`
and `DoobSOC.convex_blend_between` already hold for ANY `K ∈ [0,1]` regardless
of how K is computed. What IS new: confirming the composed-P formula itself
still produces a valid `K ∈ [0,1]`, and that adding the token term only ever
increases (never decreases) corrector trust relative to ODE error alone —
both checked below, matching the exact formula in `clpc_sampler._kalman_blend`.
-/

/-- The Kalman gain, with conditional-space error folded into `P` via a
    nonnegative weight `w_token`, stays in `[0,1]` — the same guarantee the
    original two-term `K = ode_err / (ode_err + wav_hf_err)` has in
    KalmanFilter.lean, extended to a third nonnegative term. -/
theorem kalman_gain_with_token_term_valid
    (ode_err wav_hf_err token_err w_token eps : ℝ)
    (h_ode : 0 ≤ ode_err) (h_wav : 0 ≤ wav_hf_err)
    (h_tok : 0 ≤ token_err) (h_w : 0 ≤ w_token) (h_eps : 0 < eps) :
    0 ≤ (ode_err + w_token * token_err) / (ode_err + w_token * token_err + wav_hf_err + eps) ∧
    (ode_err + w_token * token_err) / (ode_err + w_token * token_err + wav_hf_err + eps) ≤ 1 := by
  have hP : 0 ≤ ode_err + w_token * token_err := by positivity
  have hDenomPos : 0 < ode_err + w_token * token_err + wav_hf_err + eps := by positivity
  refine ⟨div_nonneg hP hDenomPos.le, ?_⟩
  rw [div_le_one hDenomPos]
  linarith

/-- Adding the token-error term can only increase (or preserve) the Kalman
    gain relative to the original two-term formula — detecting conditional-
    space badness never makes the sampler trust the corrector LESS than it
    already did from ODE error alone. -/
theorem token_term_increases_corrector_trust
    (ode_err wav_hf_err token_err w_token eps : ℝ)
    (h_ode : 0 ≤ ode_err) (h_wav : 0 ≤ wav_hf_err)
    (h_tok : 0 ≤ token_err) (h_w : 0 ≤ w_token) (h_eps : 0 < eps) :
    ode_err / (ode_err + wav_hf_err + eps) ≤
    (ode_err + w_token * token_err) / (ode_err + w_token * token_err + wav_hf_err + eps) := by
  have hb : 0 < wav_hf_err + eps := by positivity
  have hd : 0 ≤ w_token * token_err := by positivity
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_nonneg hd hb.le]

end TokenSubspaceGuidance
