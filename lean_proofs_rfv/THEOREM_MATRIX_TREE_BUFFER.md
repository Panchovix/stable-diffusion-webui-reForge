# THEOREM_MATRIX_TREE_BUFFER.md
# Research Buffer — Prompt Dependency Structure via Rule Engine + Matrix-Tree Theorem

**Date:** 2026-07-06
**Status:** Design/research VERIFIED (§0) and Lean Phase 4 COMPLETE (§6) — 8 theorems, 0 sorries.
Python implementation (Phases 1-3, 5) not started — see `plans/04-matrix-tree-prompt-structure.md`.

---

## 6. Lean results (2026-07-06) — `RFVProofs/MatrixTreeSoftOwnership.lean`

**Source paper, read directly this session (not from memory)**: user supplied
`lean_proofs_rfv/D07-1015.pdf` — Koo, Globerson, Carreras & Collins, EMNLP-CoNLL 2007. Read pages
1-5 (full algorithm section). Confirmed exact mechanism: complete directed graph over tokens,
weighted Laplacian `L(θ)`, Theorem 1 (Tutte 1984) — cofactor `L^(m,m)` = total weight of spanning
trees rooted at `m`, cofactors sign-consistent across rows/columns (Eq. 6); Proposition 1 — partition
function `Z(θ) = |L̂(θ)|` is a single determinant (row 1 replaced by root scores), proved by row
expansion; §3.2 — every edge marginal `μ_{h,m} = ∂log Z/∂θ_{h,m}` obtained from ONE matrix inversion
via Jacobi's formula `∂log|X|/∂X = (X⁻¹)ᵀ`. **Closed-form, O(n³), zero iteration, zero gradient
descent** — directly supports the "no unnecessary backpropagation" and "one pre-sampling pass"
constraints.

**Build:** `lake build` → 2634 jobs, 0 errors. New file: 0 sorries.

| # | Theorem | sorry? | Note |
|---|---------|--------|------|
| 1 | `marginal_sums_to_one` | 0 | Marginals summed over candidate parents = 1, via `Finset.sum_fiberwise` — the combinatorial "exactly one parent per tree" fact, NOT Tutte's determinant algebra (cited, not reproved) |
| 2 | `marginal_is_valid_distribution` | 0 | Corollary: marginals are already a valid probability distribution, no renormalization needed before use as soft-ownership weight |
| 3 | `soft_leak_preserves_nonneg` | 0 | Generalized `sim*(1-leak_strength*(1-w))` stays nonneg for `leak_strength,w ∈ [0,1]` |
| 4 | `soft_leak_matches_hard_at_w_one` | 0 | At `w=1`: formula = identity (own cluster, no attenuation) |
| 5 | `soft_leak_matches_hard_at_w_zero` | 0 | At `w=0`: formula = `sim*(1-leak_strength)`, EXACTLY today's shipped boolean formula — the "migration changes nothing for already-classified tags" certificate |
| 6 | `soft_corrected_row_is_distribution` | 0 | Corrected row still renormalizes to a valid distribution — direct reuse of `TokenSubspaceGuidance.renormalized_row_is_distribution` (Part 1), no new renormalization proof needed |
| 7 | `matrix_tree_cost_step_invariant` | 0 | Preprocessing cost is `0` regardless of `n_steps` — "one pre-sampling pass, not per-step" |
| 8 | `matrix_tree_strictly_cheaper_than_gradient_based` | 0 | Reuses `TokenSubspaceGuidance`'s Part 6 `GradientBasedMechanism`/defeq trick directly (`:= h`) — "punishment towards unnecessary backprop" |

**Key implementation notes for the Python phases:**
- `Finset.sum_fiberwise` (Mathlib `Algebra.BigOperators.Group.Finset.Basic`) was the exact lemma
  needed for Part 8 — found via `lean_loogle`, no manual fiber-sum derivation required.
- Theorems 3-6 give a **checkable migration contract**: when Phase 3's Python code sets `w=0` or
  `w=1` for hard rule-engine-classified tags (Layers 1-2), the output must be bit-identical to
  today's shipped boolean formula — theorems 4-5 are the formal version of that requirement, phrased
  as plain `ring`-provable equalities directly against the real correction formula.
- Explicitly did NOT attempt to formalize: Tutte's Matrix-Tree Theorem itself (Theorem 1 / Prop 1 of
  D07-1015), GPU/CPU vectorization, FP32 numerical stability, or VRAM/RAM residency — none of these
  are Lean-provable properties over ℝ; documented as out of scope in the file's docstring rather than
  silently ignored.
**Companion docs:** `THEREM_CONDITIONAL_BUFFER.md` (token-subspace guidance this extends),
`plans/03-clpc-token-conditional-guidance.md` (the shipped vanish/leak/bias feature this replaces
the clustering heuristic inside).

---

## 0. VERIFIED CORRECTION (2026-07-06, second pass) — read this before §1-5 below

The user supplied the actual PDF (`lean_proofs_rfv/1702.00887v3.pdf`) for what they believed was the
"2017 Matrix-Tree work." **Full read of all 21 pages (main body + both appendices + reference list)
confirms this paper does NOT use the Matrix-Tree Theorem anywhere.** Kim, Denton, Hoang & Rush,
"Structured Attention Networks" (ICLR 2017) computes dependency-tree marginals via **Eisner's
inside-outside algorithm** (Section 3.2, Eq. 4; full forward/backward pseudocode in Appendix B,
pages 20-21; citing Eisner 1996 and Baker 1979) — a chart-parsing dynamic program that requires
**projective** trees (no crossing dependencies) and runs in O(n³). Neither "Matrix-Tree" nor
"Kirchhoff" appear anywhere in the paper's text or its ~50 references.

**The correct citation for the technique originally described (complete graph, any-token-to-any-token
edges, graph Laplacian, Kirchhoff's theorem for spanning-tree marginals) is a different paper,
now CONFIRMED via live WebSearch:**

> **Terry Koo, Amir Globerson, Xavier Carreras, Michael Collins. "Structured Prediction Models via
> the Matrix-Tree Theorem." EMNLP-CoNLL 2007, pp. 141–150.** ([ACL Anthology D07-1015](https://aclanthology.org/D07-1015/))

This paper is specifically about **non-projective** dependency structures (equivalently, directed
spanning trees over a complete graph) and shows partition functions + edge marginals are computable
via Kirchhoff's Matrix-Tree Theorem applied to the graph Laplacian — exactly the mechanism described
in this project's original design (§2 Layer 3 below).

**Also confirmed via WebSearch**: non-projective parsing is the established better fit for
free-word-order / non-adjacent-relation structures (research on Czech/Dutch and free-word-order
languages generally; MST-style non-projective parsers shown more accurate and more tolerant of
free word order than projective parsers in comparative studies). This matters directly: a
comma-separated tag list is a **free-word-order structure** — a late attribute like "standing
together" can relate to two entities introduced much earlier, which Eisner's projective chart parser
would structurally struggle to represent (it requires nested, non-crossing spans), while the
Matrix-Tree/complete-graph approach has no such constraint.

**Recommendation (evidence-based): use genuine Matrix-Tree Theorem (Koo et al. 2007), NOT Eisner's
inside-outside (Kim et al. 2017's actual method), for Phase 3.** Kim et al. 2017 remains useful as a
*secondary* citation for the general framing ("parsing as a differentiable neural network layer,
trained end-to-end via marginals") but its concrete algorithm is the wrong one for this project's
non-projective, complete-graph design.

**Other citations also now CONFIRMED via live WebSearch (correcting §1-5's original PLAUSIBLE
framing):**
- **Danbooru's actual tag taxonomy is 5 categories: artist, character, copyright, general, meta** —
  there is NO separate "count tag" category. `1girl`/`1boy` are ordinary **general** tags ("general
  tags describe objects, actions, and attributes, e.g. 1girl, blue_hair, sitting"). Confirmed meta-tag
  examples: `translated`, `copyright_request`, `duplicate`, `image_sample`, `bad_id`. Phase 1's rule
  engine should therefore be framed as "a curated pattern this project uses to detect entity/anchor
  tags within the general category," not as "Danbooru's count-tag category" (that category doesn't
  exist).
- **LoRA trigger words**: Civitai's metadata schema really does use a `trainedWords` array field
  (confirmed via multiple independent sources: civitai.com articles, ComfyUI-Lora-Manager's own
  schema docs, multiple community trigger-word-fetching extensions). This confirms Phase 1's LoRA
  design is sound, but Phase 0 of the plan still requires checking THIS repo's actual local LoRA
  metadata reading code before assuming the schema applies verbatim here.

The rest of this document (§1-5) is left as originally written for design-rationale context, but
**§2 Layer 3 and §5's citation list should be read through the corrections above** — the mechanism
described was always right, only the paper attribution needed fixing.

---

---

## 1. Problem statement

The token-subspace guidance shipped in Plan 03 (`sure_token_guidance.py`) clusters attribute tags to
entities using a single flat heuristic: `_is_anchor_group()` (8 hardcoded English substrings:
girl/boy/woman/man/person/child/male/female) and `_build_clusters()` (assign every tag to its nearest
*preceding* anchor tag, or "unclustered" if none precedes it). This works for the canonical
`"1girl, brown hair, 1boy, blue hair"` test case (verified working across this session's iterations)
but has no way to:

- Distinguish a **meta/technical tag** ("highres", "masterpiece", "translated") from a **content tag**
  — meta tags currently just fall into "unclustered" and get silently skipped, which is accidentally
  correct but not principled.
- Handle a **LoRA's trigger/activation keyword** — no LoRA-awareness exists at all today.
- Resolve an **unknown token** (a tag not in any hardcoded list, a typo, a non-English tag, or free
  natural-language text like "a boy with brown hair") — today anything not substring-matching the
  8-keyword list simply can never become an anchor, so a prompt with no literal "girl"/"boy"/etc.
  substring gets zero entity clustering at all.
- Express **soft/uncertain** ownership — today ownership is a hard 0/1 assignment per query row
  (`own_cluster` argmax), with no notion of "70% confident this belongs to entity A."

## 2. Proposed 3-layer architecture

### Layer 1 — Rule engine (deterministic, offline dictionary lookups)

Classify every prompt segment (from `prompt_parser._split_top_level_commas`) into one of three
categories using **static, hand-maintained tables** (not a live API call — Danbooru's tag database
has millions of entries; a full mirror is out of scope):

1. **Anchor/entity tags** — Danbooru's "tag count" convention: `\d*(girl|boy|other|futanari)s?\b`,
   `solo`, `solo_focus`, `multiple_girls`, `multiple_boys`, `multiple_others`. This *generalizes*
   the current `_ANCHOR_KEYWORDS` substring list into an actual pattern grounded in the Danbooru
   wiki's documented tag-count convention (PLAUSIBLE — Danbooru does document a "general" tag
   subcategory for subject count/gender; exact wiki page name and category taxonomy needs the
   live-search follow-up in §5).
2. **Meta tags** — Danbooru's documented "meta" tag category (technical/non-content tags):
   `highres`, `absurdres`, `lowres`, `translated`, `commentary`, `commentary_request`, `bad_id`,
   `md5_mismatch`, etc. These should be **excluded entirely** from the entity graph — not
   "unclustered-and-skipped" (today's accidental behavior) but explicitly never a candidate for
   vanish/leak/bias correction at all, since they don't describe visual content.
3. **LoRA trigger/activation keywords** — parse `<lora:name:weight>` syntax already present in the
   prompt; look up `name`'s sidecar metadata (Forge/A1111 convention: `models/Lora/name.json` /
   `name.civitai.info`, commonly containing an `"activation text"` or `"trainedWords"` field per
   civitai's metadata schema — PLAUSIBLE, needs on-disk format verification in Phase 1, not assumed).
   Classify each trigger word: if it pattern-matches the anchor regex above, treat as entity-modifying;
   otherwise treat it as a **global concept tag** (style/effect that applies to the whole image, not
   a specific entity) — same treatment as a pre-anchor global tag in Layer 2.

### Layer 2 — Place-aware precedence engine (tie-breaking convention)

Generalizes the already-shipped `_build_clusters()` nearest-preceding-anchor rule with an explicit
**3-way partition** instead of today's 2-way (clustered / cluster=-1-and-ignored):

- **Global** — tags appearing *before* the first anchor tag (Danbooru convention: quality/style tags
  conventionally lead a tag list, e.g. `"masterpiece, best quality, 1girl, ..."`). These describe the
  whole image, not one entity — they should be *protected everywhere* (never leak-attenuated in any
  cluster), unlike today's "unclustered → skip" treatment which merely ignores them.
- **Entity-bound** — tags after an anchor, bound to the nearest preceding anchor (unchanged from
  today's shipped behavior — this convention is exactly what the user specified: `"1boy, brown hair,
  1girl, blue hair"` binds brown-hair→boy, blue-hair→girl by proximity, matching the natural reading
  order convention already implemented and verified working in Plan 03's real-world tests).
- **Meta** — excluded entirely (Layer 1's classification), never enters the ownership graph.

### Layer 3 — Matrix-Tree engine for unknown tokens

For every tag Layer 1 could **not** classify (not a known anchor pattern, not a known meta tag, not
a LoRA trigger) — including free natural-language phrases — build a graph-based soft clustering
instead of leaving it "unclustered":

1. **Complete graph**: treat every remaining (anchor ∪ unknown) group as a graph node, initially
   fully connected.
2. **CLIP affinity weights**: weight each edge by cosine similarity between the two groups' mean-pooled
   CLIP token embeddings — reusing the *same* per-token embeddings already computed for conditioning
   (from `encode_with_transformers`), **not** an extra CLIP forward pass, to preserve the zero/near-
   zero-extra-cost property established in Plan 03.
3. **Matrix-Tree Theorem (Kirchhoff, 1847)**: build the weighted graph Laplacian `L = D - W`. The
   Matrix-Tree Theorem states any cofactor of `L` equals the total weight of all spanning trees
   (the "partition function" over trees). **Kim, Denton, Hoang & Rush, "Structured Attention
   Networks" (ICLR 2017)** — PLAUSIBLE citation, needs live re-verification — used exactly this
   mechanism to make non-projective dependency-tree attention **differentiable**: the marginal
   probability that a specific edge `(i,j)` appears in a random spanning tree (drawn according to
   the Laplacian-weighted distribution) is computable in closed form from `L⁻¹` (via the matrix
   determinant lemma / adjugate), *without enumerating all spanning trees* (which is
   `#trees = n^(n-2)` for a complete graph by Cayley's formula — intractable to enumerate directly
   even for a dozen tags). This gives a genuinely **soft** ownership weight
   `w(unknown_tag → entity)` per unknown-tag/entity pair, rather than a hard argmax.
4. **Validation via round-trip check**: for a curated set of natural-language / Danbooru-tag paraphrase
   pairs (e.g. `"a boy with brown hair"` vs `"1boy, brown hair"`), verify Layer 3's soft clustering of
   the natural-language version assigns most of its marginal probability mass to the same
   entity-cluster that Layer 1+2's deterministic rules assign for the tag-equivalent version. This is
   a genuine regression-test opportunity, not just a one-off manual spot check.

### Final integration

The 3-layer output — a mix of hard cluster assignments (Layers 1-2) and soft marginal weights
(Layer 3) — replaces `_build_clusters()`'s single hard-assignment output. The existing correction
math (`_apply_token_subspace_corrections`, `TokenSubspaceGuidance.lean` Parts 1/3) needs generalizing
from boolean `not_own_cluster` masks to continuous `[0,1]` ownership weights (§4 below) to consume
this richer signal without discarding the soft-clustering information back down to a hard argmax.

---

## 3. Why this doesn't need extra NFE (consistent with the original design constraint)

CLIP text-embedding lookups needed for Layer 3's affinity graph are already computed once per prompt
by `get_learned_conditioning()` regardless of whether this feature exists — the encoder runs once per
generation, not once per denoising step. The Matrix-Tree computation itself (Laplacian cofactor +
adjugate) operates on an `n×n` matrix where `n` = number of prompt segments (typically single digits
to low tens), a negligible CPU/GPU cost compared to a single UNet forward pass, let alone the 20-50
forward passes in a full generation. This is a **one-time, prompt-level preprocessing cost**, exactly
like Plan 03's `get_token_subspaces()` call — not a per-step cost, and should be formalized as such
(§4, cost-model extension) to keep the "extra NFE as small as possible" property explicit and
checkable rather than assumed.

---

## 4. Lean formalization scope (what's tractable vs. not)

**Confirmed via `lean_leansearch`/`lean_loogle` this session**: Mathlib has
`SimpleGraph.lapMatrix` (`Mathlib.Combinatorics.SimpleGraph.LapMatrix`) with a proved
reachability-kernel characterization (`lapMatrix_mulVec_eq_zero_iff_forall_reachable`), but **no**
formalization of the Matrix-Tree/Kirchhoff theorem itself (the determinant-cofactor-equals-spanning-
tree-count result) was found. Treat this as **unformalized in the current Mathlib version** —
consistent with this project's existing "we are ahead of Mathlib" entries in
`THEOREM_BUFFER.md`'s Cross-Reference Audit for Doob h-transform/Feynman-Kac. Do not attempt to
reprove the full Matrix-Tree Theorem from scratch — it requires the Cauchy-Binet formula and general
all-minors matrix-tree generalization, a multi-hundred-line undertaking disproportionate to this
plan's scope.

**What IS tractable and valuable** (cite Kirchhoff's theorem as an external, trusted result — same
pattern as citing the continuous-time Doob h-transform in `ProxySOCvsFull.lean` without reproving
SDE theory):

1. **Soft-ownership generalization of Part 1/3**: today's `boost_preserves_nonneg`,
   `reweight_preserves_nonneg`, and the leak correction in `_apply_token_subspace_corrections` all
   operate on a *boolean* `not_own_cluster` mask. Generalize to a continuous ownership weight
   `w ∈ [0,1]` (Matrix-Tree marginal probability) and prove the corrected row is still nonnegative
   and renormalizes to a valid distribution — direct generalization of existing lemmas, same proof
   technique (`positivity`/`linarith`), not a new technique.
2. **3-way partition well-definedness**: every group maps to exactly one of {global, entity-bound,
   meta} (Layers 1-2) — a small, direct case-split proof.
2. **Cost-model extension** (Part 6 style): the Matrix-Tree computation is a *one-time* cost
   (§3) — extend `subspaceCorrectionExtraNFE`'s bookkeeping with a `matrixTreePreprocessingExtraNFE`
   constant, prove it's `0` (reads only already-computed CLIP embeddings) and that this cost does
   NOT scale with the number of diffusion steps (a `∀ n_steps, cost = matrixTreePreprocessingExtraNFE`
   invariance statement — trivial but makes the "one-time not per-step" property checkable).
3. Do **not** attempt: proving the Matrix-Tree Theorem itself, proving CLIP embedding cosine
   similarity has any particular semantic property (that's an empirical ML claim, not a theorem), or
   proving the round-trip validation check (§2 Layer 3.4) always succeeds (it's a stochastic/empirical
   property to be measured, not proved).

---

## 5. WebSearch verification status (updated 2026-07-06, second pass)

| Query | Status | Result |
|---|---|---|
| Kim et al. 2017 mechanism | ✅ RESOLVED (by direct PDF read, not search) | Uses Eisner's inside-outside (projective), NOT Matrix-Tree. See §0. |
| Genuine Matrix-Tree Theorem paper | ✅ CONFIRMED via WebSearch | Koo, Globerson, Carreras, Collins, EMNLP-CoNLL 2007, [ACL Anthology D07-1015](https://aclanthology.org/D07-1015/). See §0. |
| Projective vs. non-projective fit for free-word-order structure | ✅ CONFIRMED via WebSearch | Non-projective parsing established as the better fit for free-word-order / non-adjacent relations — supports using Matrix-Tree over Eisner for prompt tag lists. See §0. |
| Danbooru tag taxonomy | ✅ CONFIRMED via WebSearch | 5 categories: artist/character/copyright/general/meta. NO separate "count tag" category — `1girl`/`1boy` are ordinary `general` tags. Confirmed meta examples: `translated`, `copyright_request`, `duplicate`, `image_sample`, `bad_id`. See §0. |
| LoRA trigger-word metadata schema | ✅ CONFIRMED via codebase read (authoritative — overrides the generic web schema) | This repo does NOT use Civitai's `trainedWords` array. It reads `<lora_basename>.json` sidecar files via `modules/ui_extra_networks_user_metadata.py:write_user_metadata` (writes to `basename + '.json'`), with the trigger words stored as `user_metadata["activation text"]` — a single **comma-separated string** (see `extensions-builtin/Lora/ui_extra_networks_lora.py:44` and `ui_edit_user_metadata.py:63,186` which does `re.split(re_comma, activation_text)` to get individual words). Phase 1 must read this exact field/format, not Civitai's array schema. |
| Chu-Liu/Edmonds algorithm (deterministic tree extraction) | ⬜ NOT YET NEEDED | Only relevant if Phase 3 requires a hard/deterministic tree rather than soft marginals; defer until Phase 3 design confirms this is needed. |
