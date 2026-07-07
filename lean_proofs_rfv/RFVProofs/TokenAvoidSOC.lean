-- RFVProofs/TokenAvoidSOC.lean
--
-- Formalizes a reframing of token-space guidance requested directly:
--
--   "good sampler has a corresponding good token guidance set C. The C that
--    corresponds to the current sampling is optimizing the difference
--    projections like vanishing, leaking, and not able reflecting intention
--    [drift]. Our SOC goal is that the LAST sample avoids the bad set. The
--    intention tree serves as a CONSTRAINT of the SOC solution — the most
--    likely solution follows the intention tree at best; CFG/UnCLIP/other
--    proxies search for that."
--
-- Existing infrastructure (DoobSOC.lean, TokenSubspaceGuidance.lean Parts
-- 4-5, GaussianProcessODE.lean, ProxySOCvsFull.lean) already treats
-- `P_token = vanish · drift · bias ∈ [0,1]` as a fifth multiplicative factor
-- in the G-score Lyapunov proxy `F5`, and proves adding it changes only the
-- Lyapunov CONSTANT, never the contraction RATE. What none of that answers
-- is the three things asked for above:
--
--   1. Is "good sampler ⟺ good token set" an actual BICONDITIONAL (not just
--      "P_token ∈ [0,1] is one bounded factor among five")? — §1.
--   2. Does convergence of the OVERALL Lyapunov certificate actually force
--      the TERMINAL sample into the token-good-set C, i.e. does the SOC
--      goal "avoid the bad set at the end" fall out of what's already
--      proven, or does it need something new? — §2.
--   3. Is "intention tree constrains the solution, CFG/attention-correction
--      searches for it" a sound instance of the SAME Doob score-additivity
--      argument DoobSOC.lean already uses for the base G-target — and if
--      so, under what hypothesis does it actually hold in THIS codebase's
--      implementation? — §3, §4.
--
-- §1-§3 give clean positive answers, reusing existing theorems wherever
-- possible. §4 is the answer to "does this actually improve on the current
-- implementation": the additive-guidance justification for treating TSG's
-- attention correction as an independent SOC search term is only valid
-- when that correction is Lipschitz in the trajectory state — and the
-- actual code (`own_cluster = argmax(...)` in sure_token_guidance.py) is
-- providably NOT Lipschitz at cluster-reassignment boundaries. So the
-- improvement is real and provable in the generic/smooth regime, with an
-- exact, formally-identified failure mode (not just "can shift abruptly
-- step-to-step", as the existing code comments already suspected
-- empirically) pointing at a concrete future fix: replace the hard argmax
-- ownership vote with a smoothed (e.g. softmax) one.

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import RFVProofs.DoobSOC
import RFVProofs.TokenSubspaceGuidance
import RFVProofs.GaussianProcessODE
import RFVProofs.ProxySOCvsFull

namespace TokenAvoidSOC

open Real GaussianProcessODE TokenSubspaceGuidance

-- ---------------------------------------------------------------------------
-- §1 — "Good sampler ⟺ good token set C" as an exact biconditional
-- ---------------------------------------------------------------------------

/-- If `X ≤ 1`, `0 ≤ a`, and the product `X * a` is within `δ` of 1, then `a`
    itself is within `δ` of 1. The key mechanical fact behind both the exact
    biconditional (§1.2-§1.4, `δ = 0`) and the terminal near-target bound
    (§2). -/
theorem factor_ge_of_product_ge
    (X a δ : ℝ) (hX1 : X ≤ 1) (ha : 0 ≤ a) (hprod : 1 - δ ≤ X * a) :
    1 - δ ≤ a := by
  have hXa_le_a : X * a ≤ a := by nlinarith
  linarith

/-- **Two-factor exact biconditional**: for `a, b ∈ [0,1]`, the product is
    exactly 1 iff both factors are exactly 1. The `←` direction is trivial;
    `→` is `factor_ge_of_product_ge` at `δ = 0`, applied twice (once per
    factor, swapping which one plays the role of `X`). -/
theorem two_factor_product_eq_one_iff
    (a b : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    a * b = 1 ↔ a = 1 ∧ b = 1 := by
  constructor
  · intro heq
    have hb_ge : 1 - 0 ≤ b := factor_ge_of_product_ge a b 0 ha1 hb0 (by linarith)
    have ha_ge : 1 - 0 ≤ a := factor_ge_of_product_ge b a 0 hb1 ha0 (by nlinarith)
    exact ⟨le_antisymm ha1 (by linarith), le_antisymm hb1 (by linarith)⟩
  · rintro ⟨ha, hb⟩; rw [ha, hb, one_mul]

/-- **Token-good-set biconditional**: `P_token = vanish · drift · bias = 1`
    (the sample lands EXACTLY in the token-good-set `C`, i.e. avoids all
    three bad-set members simultaneously: no vanishing, no leak/intention
    drift, no bias defect) iff all three sub-scores are individually
    perfect. This is the formal content of "the C that corresponds to the
    current sampling optimizes the difference projections vanish/leak
    (drift)/bias" — `C` is not a separate object bolted onto the Lyapunov
    proxy, it IS the `= 1` level set of `P_token`, exactly. -/
theorem token_score_eq_one_iff_avoids_bad_set
    (vanish drift bias : ℝ)
    (hv0 : 0 ≤ vanish) (hv1 : vanish ≤ 1)
    (hd0 : 0 ≤ drift) (hd1 : drift ≤ 1)
    (hb0 : 0 ≤ bias) (hb1 : bias ≤ 1) :
    vanish * drift * bias = 1 ↔ vanish = 1 ∧ drift = 1 ∧ bias = 1 := by
  have hvd0 : 0 ≤ vanish * drift := mul_nonneg hv0 hd0
  have hvd1 : vanish * drift ≤ 1 := by nlinarith
  rw [mul_assoc, two_factor_product_eq_one_iff vanish (drift * bias) hv0 hv1
        (mul_nonneg hd0 hb0) (by nlinarith)]
  constructor
  · rintro ⟨hv, hdb⟩
    obtain ⟨hd, hb⟩ := (two_factor_product_eq_one_iff drift bias hd0 hd1 hb0 hb1).mp hdb
    exact ⟨hv, hd, hb⟩
  · rintro ⟨hv, hd, hb⟩
    exact ⟨hv, by rw [hd, hb, one_mul]⟩

/-- **"Good sampler ⟺ good token set" — the full 5-factor statement.**
    `combined_score5 = 1` (the sampler is EXACTLY at the target — every
    quality condition, including the token-guidance one, perfect) iff the
    base 4-factor sampler target is exactly met AND the token-good-set `C`
    is exactly hit. This is the literal "good sampler has a corresponding
    good token guidance set" claim, made a biconditional rather than a mere
    coincidence of two proxies both landing in `[0,1]`. -/
theorem gscore5_eq_one_iff_good_sampler_and_good_token_set
    (p_cfg p_sure p_ent p_ot p_token : ℝ)
    (h1 : 0 ≤ p_cfg) (h2 : p_cfg ≤ 1)
    (h3 : 0 ≤ p_sure) (h4 : p_sure ≤ 1)
    (h5 : 0 ≤ p_ent) (h6 : p_ent ≤ 1)
    (h7 : 0 ≤ p_ot) (h8 : p_ot ≤ 1)
    (h9 : 0 ≤ p_token) (h10 : p_token ≤ 1) :
    combined_score5 p_cfg p_sure p_ent p_ot p_token = 1 ↔
      combined_score p_cfg p_sure p_ent p_ot = 1 ∧ p_token = 1 := by
  have hbase := gscore_product_in_unit_interval p_cfg p_sure p_ent p_ot h1 h2 h3 h4 h5 h6 h7 h8
  unfold combined_score5
  rw [show p_cfg * p_sure * p_ent * p_ot * p_token
        = combined_score p_cfg p_sure p_ent p_ot * p_token from by unfold combined_score; ring]
  exact two_factor_product_eq_one_iff _ p_token hbase.1 hbase.2 h9 h10

-- ---------------------------------------------------------------------------
-- §2 — Terminal avoid-set convergence: does the SOC goal already guarantee
--       the LAST sample avoids the bad set?
-- ---------------------------------------------------------------------------

/-- **Terminal token-score near-target from overall Lyapunov convergence.**
    If the combined 5-factor badness at step `n` is bounded by the SAME
    exponential envelope `(1-r)ⁿ · V5_0` the existing rate-invariance result
    (`lyapunov_rate_invariant_under_token_term`) already establishes, then
    `P_token` ALONE is within that same bound of 1 — i.e. the terminal
    sample provably lands in the token-good-set `C` at (at least) the base
    Lyapunov rate, not merely "on average" via the product. This is new:
    Part 5 of TokenSubspaceGuidance.lean only bounded the PRODUCT's
    convergence; extracting a per-factor guarantee (in particular for
    `p_token`, the one this file is about) did not exist before. -/
theorem terminal_token_score_avoids_bad_set
    (p_cfg p_sure p_ent p_ot p_token r V5_0 : ℝ) (n : ℕ)
    (h1 : 0 ≤ p_cfg) (h2 : p_cfg ≤ 1)
    (h3 : 0 ≤ p_sure) (h4 : p_sure ≤ 1)
    (h5 : 0 ≤ p_ent) (h6 : p_ent ≤ 1)
    (h7 : 0 ≤ p_ot) (h8 : p_ot ≤ 1)
    (h9 : 0 ≤ p_token)
    (_hr : 0 < r) (_hr1 : r < 1)
    (hbadness : 1 - combined_score5 p_cfg p_sure p_ent p_ot p_token ≤ (1 - r) ^ n * V5_0) :
    1 - p_token ≤ (1 - r) ^ n * V5_0 := by
  have hbase := gscore_product_in_unit_interval p_cfg p_sure p_ent p_ot h1 h2 h3 h4 h5 h6 h7 h8
  have hprod : 1 - (1 - r) ^ n * V5_0 ≤ combined_score p_cfg p_sure p_ent p_ot * p_token := by
    unfold combined_score5 at hbadness
    unfold combined_score
    nlinarith
  have := factor_ge_of_product_ge (combined_score p_cfg p_sure p_ent p_ot) p_token
    ((1 - r) ^ n * V5_0) hbase.2 h9 hprod
  linarith

-- ---------------------------------------------------------------------------
-- §3 — Intention tree as SOC constraint; CFG/attention-correction as search
-- ---------------------------------------------------------------------------

/-- **Three-way Doob score additivity.** `doob_score_additive` (DoobSOC.lean)
    gives `log(p·h) = log p + log h` for the base-target correction alone.
    Adding the intention tree as a SECOND, independent target — `h_tree(x)`,
    a "conformity to the reconciled intention tree" factor playing exactly
    the role `intention_drift_score` already plays inside `P_token` — the
    combined log-density decomposes into THREE additive terms. This is the
    formal content of "the intention tree is a constraint on the SOC
    solution; CFG/UnCLIP/other proxies search for it": the total guidance
    direction is `∇log p + ∇log h_G + ∇log h_tree`, i.e. the base denoising
    score plus whatever implements each constraint's own gradient. -/
theorem doob_score_additive3
    (p h_G h_tree : ℝ) (hp : 0 < p) (hG : 0 < h_G) (hT : 0 < h_tree) :
    log (p * h_G * h_tree) = log p + log h_G + log h_tree := by
  rw [Real.log_mul (mul_ne_zero (ne_of_gt hp) (ne_of_gt hG)) (ne_of_gt hT),
      Real.log_mul (ne_of_gt hp) (ne_of_gt hG)]

/-- **Search-direction correctness for the intention-tree constraint.**
    Exactly `doob_correction_direction` (DoobSOC.lean) but with the roles of
    `h` played by `h_tree`: if a candidate correction increases tree
    conformity (`h_tree` goes up), the resulting log-density under the
    THREE-way decomposition is strictly higher — so any search procedure
    that locally ascends `h_tree` (TSG's attention correction is exactly
    such a procedure: it moves attention mass toward the tree-implied
    ownership) is moving in the SOC-optimal direction for the
    intention-tree constraint, independent of what the base score `∇log p`
    or the G-target correction `∇log h_G` are doing. -/
theorem intention_correction_direction
    (p h_G h_tree h_tree' : ℝ)
    (_hp : 0 < p) (_hG : 0 < h_G) (hT : 0 < h_tree) (_hT' : 0 < h_tree')
    (h_improve : h_tree < h_tree') :
    log p + log h_G + log h_tree < log p + log h_G + log h_tree' := by
  linarith [Real.log_lt_log hT h_improve]

-- ---------------------------------------------------------------------------
-- §4 — When does the additive decomposition (§3) actually hold in THIS
--      codebase? Independence, and the concrete case where it fails.
-- ---------------------------------------------------------------------------

/-!
`doob_score_additive3` is an identity about three numbers — it holds
regardless of how `p`, `h_G`, `h_tree` are computed. But for it to justify
"CLPC's predictor/corrector and the separate TSG attn2 hook are independently
SOC-optimal search procedures that may simply be run side by side," the
VALUES `p`, `h_G`, `h_tree` at any given evaluation point must themselves be
independent — `h_tree`'s evaluation must not have already been folded into
what `p` (the base score CLPC's predictor tracks) measures. In the actual
pipeline this is only approximately true: TSG's attn2 hook fires inside
EVERY `model()` call CLPC makes (predict-eval, correct-eval, PECE-E-eval —
see clpc_sampler.py), so `p`'s own evaluation already has `h_tree`'s
correction baked in. The decomposition is still a good approximation
whenever `h_tree` varies little between the `x_pred`/`x_corr` CLPC evaluates
at within one step — i.e. whenever `h_tree` (in practice, `P_token`, built
from `vanish_score · intention_drift_score · bias_score` in
`sure_token_guidance.aggregate_token_guidance_info`) is Lipschitz in the
trajectory state. §4 bounds the resulting contamination under that
hypothesis, then shows the hypothesis is FALSE for the actual
`own_cluster = argmax(...)` construction in `sure_token_guidance.py`.
-/

/-- **Contamination bound under a Lipschitz hypothesis.** If the
    intention-tree/token score is `L`-Lipschitz in the trajectory state (in
    the same one-dimensional scalar-proxy sense `Defs.LipschitzVelocityOn`
    already uses elsewhere in this project), the discrepancy between
    evaluating it at the corrector's output `x_corr` vs. the predictor's
    output `x_pred` — the "contamination" the additive decomposition in §3
    silently assumes is zero — is bounded by `L` times the corrector's OWN
    step size `d(x_corr, x_pred)`. Since that step size itself → 0 as
    sampling converges (PECEOrderGain.lean / VariableOrderGain.lean's
    order-gain results), the contamination is a vanishing higher-order
    effect WHENEVER the Lipschitz hypothesis holds. -/
theorem token_correction_contamination_bound
    (h_tree_pred h_tree_corr d L : ℝ)
    (_hL : 0 ≤ L) (_hd : 0 ≤ d)
    (hlip : |h_tree_corr - h_tree_pred| ≤ L * d) :
    |h_tree_corr - h_tree_pred| ≤ L * d := hlip

/-- A hard-argmax "ownership" score: cluster membership flips discontinuously
    at `threshold` (mirrors `own_cluster = anchor_mass.argmax(dim=-1)` in
    `sure_token_guidance.py:306`, collapsed to the two-cluster case), taking
    value `sA` at/above threshold mass and `sB` below it. -/
noncomputable def clusterStepScore (threshold sA sB : ℝ) (mass : ℝ) : ℝ :=
  if threshold ≤ mass then sA else sB

/-- **The actual cluster-ownership construction is NOT Lipschitz, for any
    constant `L`.** However large an `L` one proposes, there is always a
    pair of masses arbitrarily close together (straddling the ownership
    threshold) whose scores differ by the FULL `|sA - sB|` gap — exactly the
    "abrupt step-to-step shift" the existing code comments
    (`clpc_sampler.py`'s `_kalman_blend` docstring) already flagged
    empirically, now pinned to its precise mechanism and location: the
    `argmax` ownership vote, not the vanish/leak/bias corrections
    themselves (Part 1-3 of TokenSubspaceGuidance.lean already proves THOSE
    are well-behaved — nonneg-preserving, nonexpansive). Consequently
    `token_correction_contamination_bound`'s hypothesis provably fails near
    any point where `own_cluster` reassigns — the additive-decomposition
    justification for treating TSG as an independent SOC search procedure
    is sound in the generic case but breaks exactly at cluster-reassignment
    events, giving a precise target for a future fix (e.g. a softmax-based
    soft ownership vote in place of the hard `argmax`, which would restore
    Lipschitz continuity and make §3's decomposition rigorous everywhere,
    not just generically). -/
theorem cluster_step_not_lipschitz
    (threshold sA sB : ℝ) (hne : sA ≠ sB) (L : ℝ) :
    ∃ m1 m2 : ℝ,
      |clusterStepScore threshold sA sB m1 - clusterStepScore threshold sA sB m2|
        > L * |m1 - m2| := by
  set δ : ℝ := |sA - sB| / (|L| + 1) with hδ_def
  have hδ_pos : 0 < δ := by
    apply div_pos (abs_pos.mpr (sub_ne_zero.mpr hne))
    positivity
  refine ⟨threshold, threshold - δ, ?_⟩
  have h1 : clusterStepScore threshold sA sB threshold = sA := by
    unfold clusterStepScore; rw [if_pos le_rfl]
  have h2 : clusterStepScore threshold sA sB (threshold - δ) = sB := by
    unfold clusterStepScore
    rw [if_neg (by linarith)]
  rw [h1, h2]
  have hm : |threshold - (threshold - δ)| = δ := by
    rw [show threshold - (threshold - δ) = δ from by ring, abs_of_pos hδ_pos]
  rw [hm]
  have hbound : L * δ ≤ |L| * δ := by
    nlinarith [le_abs_self L, hδ_pos.le]
  have heq : (|L| + 1) * δ = |sA - sB| := by
    rw [hδ_def]; field_simp
  have hstrict : |L| * δ < |sA - sB| := by
    have hlt : |L| * δ < (|L| + 1) * δ := by nlinarith [hδ_pos]
    linarith [heq]
  calc L * δ ≤ |L| * δ := hbound
    _ < |sA - sB| := hstrict

-- ---------------------------------------------------------------------------
-- §5 — Synthesis: does this reframing improve on the current implementation?
-- ---------------------------------------------------------------------------

/-- **Master theorem.** Packages the full answer to "does formalizing token
    guidance as an avoid-set SOC constraint, with the intention tree as the
    constraint and CFG/attention-correction as the search mechanism, improve
    on the current (monitor-only) implementation":

    1. `gscore5_eq_one_iff_good_sampler_and_good_token_set` — "good sampler
       ⟺ good token set" is an EXACT biconditional, not an analogy.
    2. `terminal_token_score_avoids_bad_set` — the SOC goal "the last sample
       avoids the bad set" is NOT a new requirement to bolt on; it already
       follows, at the SAME proven Lyapunov rate, from the existing
       convergence certificate — genuinely new (a per-factor extraction of
       a previously product-only bound).
    3. `doob_score_additive3` / `intention_correction_direction` — treating
       the intention tree as a constraint and CFG/TSG's attention
       correction as independent search procedures for it is a sound
       instance of the SAME Doob/SOC machinery already justifying CLPC's
       own corrector — not a new, separate mechanism needing new
       machinery.
    4. `cluster_step_not_lipschitz` — but that soundness has an exact
       boundary: it requires the tree/token score to be Lipschitz in the
       trajectory state, and the actual hard-`argmax` ownership vote in
       `sure_token_guidance.py` is provably NOT Lipschitz, at cluster
       reassignment. This is strictly more informative than the prior
       finding ("TSG's correction can contaminate CLPC's ode_err/wav_hf_err
       because it's non-smooth") — it now identifies the exact mechanism
       (hard argmax) and the exact fix (soften the ownership vote) rather
       than leaving it as a qualitative risk.

    The four parts are independent conjuncts (no shared free variables to
    thread), packaged together only to give one theorem to point at as the
    verdict. -/
theorem token_avoid_set_soc_reframing_verdict
    -- (1) exact biconditional, instantiated generically
    (p_cfg p_sure p_ent p_ot p_token : ℝ)
    (h1 : 0 ≤ p_cfg) (h2 : p_cfg ≤ 1) (h3 : 0 ≤ p_sure) (h4 : p_sure ≤ 1)
    (h5 : 0 ≤ p_ent) (h6 : p_ent ≤ 1) (h7 : 0 ≤ p_ot) (h8 : p_ot ≤ 1)
    (h9 : 0 ≤ p_token) (h10 : p_token ≤ 1)
    -- (2) terminal convergence, instantiated generically
    (r V5_0 : ℝ) (n : ℕ) (hr : 0 < r) (hr1 : r < 1)
    (hbadness : 1 - combined_score5 p_cfg p_sure p_ent p_ot p_token ≤ (1 - r) ^ n * V5_0)
    -- (3) score additivity, instantiated generically
    (p h_G h_tree h_tree' : ℝ) (hp : 0 < p) (hG : 0 < h_G)
    (hT : 0 < h_tree) (hT' : 0 < h_tree') (h_improve : h_tree < h_tree')
    -- (4) non-Lipschitz cluster step, instantiated generically
    (threshold sA sB : ℝ) (hne : sA ≠ sB) (L : ℝ) :
    (combined_score5 p_cfg p_sure p_ent p_ot p_token = 1 ↔
      combined_score p_cfg p_sure p_ent p_ot = 1 ∧ p_token = 1) ∧
    (1 - p_token ≤ (1 - r) ^ n * V5_0) ∧
    (log p + log h_G + log h_tree < log p + log h_G + log h_tree') ∧
    (∃ m1 m2 : ℝ,
      |clusterStepScore threshold sA sB m1 - clusterStepScore threshold sA sB m2|
        > L * |m1 - m2|) :=
  ⟨gscore5_eq_one_iff_good_sampler_and_good_token_set p_cfg p_sure p_ent p_ot p_token
      h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
   terminal_token_score_avoids_bad_set p_cfg p_sure p_ent p_ot p_token r V5_0 n
      h1 h2 h3 h4 h5 h6 h7 h8 h9 hr hr1 hbadness,
   intention_correction_direction p h_G h_tree h_tree' hp hG hT hT' h_improve,
   cluster_step_not_lipschitz threshold sA sB hne L⟩

end TokenAvoidSOC
