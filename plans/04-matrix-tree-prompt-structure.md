# Plan 04: Matrix-Tree Prompt Dependency Structure for Token Subspace Guidance

## Status: ALL PHASES (0, 1, 2+3, 2.5, 4, 5) IMPLEMENTED — needs a live WebUI/model test to confirm
## the parts that could not be exercised in this sandbox (no torch/CLIP/live model available)

Phase 4 result: `lean_proofs_rfv/RFVProofs/MatrixTreeSoftOwnership.lean`, 8 theorems, 0 sorries, full
project builds clean (2634 jobs). Built from the actual Koo et al. 2007 paper (user-supplied
`D07-1015.pdf`, read directly, not from memory). See `THEOREM_MATRIX_TREE_BUFFER.md §6` for the
theorem-by-theorem breakdown. Key outcome: the soft-ownership generalization (continuous weight
`w ∈ [0,1]`) is now PROVED identical to today's shipped boolean leak formula at the degenerate
weights `w ∈ {0,1}` — Phases 2-3/5's Python migration has a checkable correctness contract to match
against, not just a design intention.

Phase 1 result: `modules/rule_engine/` package (regex-to-DFA compiler, YAML schema, token-table
output, LoRA trigger extraction) + `configs/prompt_rules/` presets (danbooru/pony/illustrious). See
the Phase 1 section below for the full breakdown and what's been tested vs. left unverified.

Phases 2+3 (merged into one semantic-analyzer phase, `modules/rule_engine/{matrix_tree,
clip_vector_db, semantic_analyzer}.py`) are also done — see the Phase 2+3 section below for the
full breakdown, including what's verified (Matrix-Tree algorithm correctness vs. brute force,
3-situation pipeline mechanics) vs. NOT verified (real CLIP embeddings, weight tuning, per-segment
embedding extraction from the live conditioning tensor). Phase 5 (sampler/WebUI wiring — including
that embedding-extraction step) not started yet.

## Goal

Replace `sure_token_guidance.py`'s flat 8-keyword substring heuristic (`_is_anchor_group`,
`_build_clusters`) with a 3-layer prompt-structure engine that can:

1. Recognize **known** entity-indicating general tags and LoRA trigger words via a **rule engine**
   (deterministic, offline).
2. Apply the **precedence convention** (pre-anchor tags are global; post-anchor tags bind to the
   nearest preceding anchor) as an explicit **3-way partition** (global / entity-bound / meta),
   generalizing today's 2-way (clustered / ignored) split.
3. Resolve **unknown tokens** (typos, non-English tags, free natural-language phrasing like "a boy
   with brown hair") via the **Matrix-Tree Theorem** (Kirchhoff, 1847; Koo, Globerson, Carreras &
   Collins, EMNLP-CoNLL 2007) computation over a CLIP-affinity graph, giving **soft** (probabilistic)
   ownership instead of the current hard argmax.

Companion research doc: `lean_proofs_rfv/THEOREM_MATRIX_TREE_BUFFER.md` — **read §0 first**. The
paper the user originally supplied (`1702.00887v3.pdf`, Kim et al. 2017 "Structured Attention
Networks") was fully read and does NOT use the Matrix-Tree Theorem — it uses Eisner's projective
inside-outside algorithm. The correct, now-CONFIRMED citation for the actual technique this plan
uses is **Koo, Globerson, Carreras & Collins, "Structured Prediction Models via the Matrix-Tree
Theorem," EMNLP-CoNLL 2007** ([ACL Anthology D07-1015](https://aclanthology.org/D07-1015/)) — see
buffer §0 for the full evidence trail, including WebSearch-confirmed research that non-projective
parsing (what Matrix-Tree gives you) is the better fit for free-word-order structures like
comma-separated tag lists, vs. Eisner's projective constraint.

**Hard constraints carried over from Plan 03**: zero/near-zero extra NFE (Matrix-Tree computation
must be a one-time, prompt-level preprocessing cost, not per-diffusion-step), Lean-first validation
of any new correction-math generalization before touching sampler code.

---

## Phase 0: Documentation Discovery — COMPLETE, all findings verified

All four Phase 0 checks are resolved (buffer doc §0/§5 has the full evidence trail):

1. ✅ **Algorithm**: use Koo et al. 2007's Matrix-Tree Theorem (non-projective), NOT Kim et al. 2017's
   Eisner inside-outside (projective) — confirmed by full PDF read + WebSearch corroboration that
   non-projective structure fits free-word-order tag lists better.
2. ✅ **Danbooru taxonomy**: 5 categories (artist/character/copyright/general/meta), confirmed via
   WebSearch. There is NO "tag count" category — `1girl`/`1boy` are ordinary `general` tags. Phase 1's
   anchor-detection regex is **this project's own convention**, not a cited Danbooru category.
   Confirmed meta-tag examples: `translated`, `copyright_request`, `duplicate`, `image_sample`,
   `bad_id`.
3. ✅ **LoRA metadata — codebase-confirmed** (authoritative, overrides generic web schema): this repo
   reads `<lora_basename>.json` sidecar files (`modules/ui_extra_networks_user_metadata.py:
   write_user_metadata`, writes `basename + '.json'`), with trigger words in
   `user_metadata["activation text"]` — a **comma-separated string**, split via `re.split(re_comma,
   activation_text)` (`extensions-builtin/Lora/ui_edit_user_metadata.py:186`). This is NOT Civitai's
   `trainedWords` array — Phase 1 must read this exact field/format.
4. ⬜ **Chu-Liu/Edmonds**: deferred — not needed unless Phase 3 requires a deterministic (not
   marginal-probability) tree; revisit if that need arises during Phase 3 implementation.

### Allowed APIs (confirmed this session, safe to build on without re-verification)

| API | File:line | Confirmed use |
|---|---|---|
| `_split_top_level_commas`, `get_token_subspaces` | `modules/prompt_parser.py:517,541` | Existing comma-group + token-range extraction Phase 1/2 build on top of, not replace. |
| `_is_anchor_group`, `_build_clusters` | `ldm_patched/k_diffusion/sure_token_guidance.py:111,116` | The exact functions Phase 1/2 replace. |
| `_apply_token_subspace_corrections` | `ldm_patched/k_diffusion/sure_token_guidance.py:134` | The correction function whose `own_cluster`/`cluster_of`/`confident` boolean logic Phase 4 generalizes to continuous weights. |
| `TokenSubspaceGuidance.lean` Parts 1/3 (`renormalized_row_is_distribution`, `boost_preserves_nonneg`, `leak_projection_nonexpansive`) | `lean_proofs_rfv/RFVProofs/TokenSubspaceGuidance.lean:69,79,133` | Theorems Phase 5's Lean work generalizes from boolean to `[0,1]`-weighted masks. |
| `SimpleGraph.lapMatrix` | Mathlib `Combinatorics.SimpleGraph.LapMatrix` | Confirmed present via `lean_leansearch`/`lean_loogle` this session; use for any graph-Laplacian modeling in Lean. Do NOT assume the full Matrix-Tree/Kirchhoff determinant-count theorem is also present — it was not found. |

### Anti-patterns to avoid

- **Do not** hand-roll a Danbooru tag database mirror — use a small curated static table (a few dozen
  meta tags, a regex for count/entity tags), not an attempt at comprehensive coverage.
- **Do not** assume LoRA metadata format without checking this repo's actual LoRA-loading code first
  (§0.3) — civitai's schema may not match what Forge/reForge actually reads.
- **Do not** add an extra CLIP forward pass for Layer 3's affinity graph — reuse the per-token
  embeddings already produced by the conditioning encode step (verify the exact tensor is accessible;
  if not, this is a real blocker to flag back to the user, not something to work around with a second
  encode call).
- **Do not** attempt to reprove the Matrix-Tree Theorem in Lean — cite it as an external result
  (see buffer doc §4).

---

## Phase 1: Rule Engine — DONE (implemented as a compiler-toolchain pipeline)

Implemented as a new package, `modules/rule_engine/`, going considerably beyond the original
minimal `classify_tag()` sketch per an expanded design request during implementation: a
YAML-configurable rule engine built as a proper regex-to-DFA compiler, a richer 7-class tag
taxonomy (classes can overlap), and 3 composable presets.

- **`regex_dfa.py`** — a real regex compiler: recursive-descent parser → Thompson-construction NFA →
  subset-construction DFA, supporting literals, `.`, `[abc]`/`[^abc]`/ranges, Perl shorthand classes
  `\d`/`\w`/`\s` (+ negations), alternation `|`, grouping `(...)`, postfix `*`/`+`/`?`, and glob
  convenience (a leading bare `*`/`?` with no atom means `.*`/`.`, so `"*boy"` works as specified).
  Multiple patterns compile into ONE merged DFA (lexer-generator style), so classifying a tag is a
  single linear-time walk — **no backtracking is possible by construction**. Verified against the
  classic ReDoS pattern `(a*)*b` on 5000 characters: 0.5ms (a backtracking engine like Python's `re`
  would hang on this input).
- **`schema.py`** — YAML loading (pattern → list of classes), multi-file composition (union classes
  on duplicate patterns across files), and the 7-class taxonomy: `META, OBJECT, ADJ, VERB, GLOBAL,
  CONCEPT, LORAACT` (classes are NOT mutually exclusive — e.g. `"comic"` is both `GLOBAL` and
  `CONCEPT`). Fixed a real bug found during testing: tags are normalized (whitespace → underscore,
  lowercased) before matching, since prompts are typed with spaces (`"brown hair"`) but Danbooru's
  canonical tag form uses underscores (`"brown_hair"`) — without this, every space-separated
  attribute tag would silently fall through to "unknown."
- **`token_table.py`** — the compiler-pipeline handoff artifact: `build_token_table()` converts
  `prompt_parser.get_token_subspaces()`'s groups into `TokenTableEntry` records carrying both
  classification and token-index range, with `needs_clip=True` marking exactly which segments Phase
  3's Matrix-Tree engine must send to CLIP.
- **`lora_triggers.py`** — `<lora:name:weight>` extraction (regex-tested, verified) and sidecar
  `<basename>.json` → `user_metadata["activation text"]` lookup via `networks.available_networks`
  (this repo's actual, codebase-confirmed convention, NOT Civitai's `trainedWords` schema). **Caveat**:
  the `networks.available_networks` lookup itself is untested against a real LoRA file — no
  live WebUI/torch available in this sandbox; verify on first real use.
- **Presets** (`configs/prompt_rules/`): `danbooru.yaml` (base — entity/meta/adj/verb/global/concept
  patterns), `pony.yaml` and `illustrious.yaml` (overlays — load together with `danbooru.yaml`, add
  only their checkpoint-family-specific conventions like Pony's `score_9`/`source_pony`/`rating_*`
  ladder). Composition tested: `danbooru.yaml + pony.yaml` correctly unions to classify both bases.

### Verification performed

- [x] End-to-end test against this session's exact running example: `"1girl, brown hair, 1boy, blue
      hair"` → `1girl`/`1boy` = `OBJECT`, `brown hair`/`blue hair` = `ADJ`, 0 unknown.
  - [x] Quality + unknown natural-language test: `"masterpiece, best quality, 1girl, a boy with brown
      hair, comic"` → `masterpiece`=`META`, `1girl`=`OBJECT`, `comic`=`{GLOBAL,CONCEPT}`, and
      `"a boy with brown hair"` correctly falls through as unknown (→ Phase 3's job).
  - [x] Preset composition test: `danbooru.yaml + pony.yaml` on Pony-style tags (`score_9`,
        `source_pony`, `anthro`) — all classify correctly, 0 unknown.
  - [x] DFA correctness test suite (17 cases: wildcards, shorthand classes, alternation, character
        ranges, negated classes, empty string) — all pass.
  - [x] ReDoS stress test (ensures the "no weird backward situation" constraint actually holds).

### What to implement (superseded by the above — kept for history)

New module, e.g. `modules/prompt_tag_rules.py`:
- `classify_tag(text: str) -> Literal["anchor", "meta", "unknown"]` — regex-match against a curated
  anchor pattern (generalizing `_ANCHOR_KEYWORDS`: `\d*(girl|boy|other|futanari)s?\b`, `solo`,
  `multiple_(girls|boys|others)`) and a curated meta-tag set (confirmed examples from Phase 0.2:
  `translated`, `copyright_request`, `duplicate`, `image_sample`, `bad_id`, plus this project's own
  additions like `highres`/`absurdres`/`masterpiece`-adjacent quality tags if desired). Note: these
  are **this project's own curated patterns**, not a literal Danbooru "count tag" category — Danbooru
  has no such category (§0 correction); `1girl`/`1boy` are ordinary `general` tags we specifically
  choose to treat as entity anchors. Falls through to `"unknown"` for anything else — Layer 3's job.
- `extract_lora_triggers(prompt: str) -> list[tuple[str, str]]` — parse `<lora:name:weight>` syntax
  (reuse whatever regex this codebase's own LoRA-loading code already uses, do not write a second
  one), read each LoRA's sidecar `<basename>.json` via the SAME mechanism as
  `modules/ui_extra_networks_user_metadata.py`'s `get_user_metadata`/`write_user_metadata`, extract
  `user_metadata["activation text"]` and split on `re_comma` (confirmed format, Phase 0.3 — a
  comma-separated string, NOT Civitai's `trainedWords` array), classify each resulting word via
  `classify_tag` (falls back to "global concept" if not anchor-matching, per the buffer doc's
  Layer 1.3 design).

### Documentation references

- `THEOREM_MATRIX_TREE_BUFFER.md §2 Layer 1` for the design rationale, `§0` for the corrected
  Danbooru-taxonomy and LoRA-metadata findings.
- `modules/ui_extra_networks_user_metadata.py` (`get_user_metadata`/`write_user_metadata`,
  `metadata_path = basename + '.json'`) and `extensions-builtin/Lora/ui_edit_user_metadata.py:186`
  (`re.split(re_comma, activation_text)`) — copy these exact read patterns, don't reinvent.

### Verification checklist

- [ ] Unit test `classify_tag` against the current test prompt's groups (`1girl`→anchor,
      `brown hair`→unknown, `1boy`→anchor, `blue hair`→unknown) plus a handful of meta tags
      (`highres`→meta) and edge cases (`multiple_girls`→anchor).
- [ ] If this codebase has ANY existing LoRA prompt loaded during testing, confirm
      `extract_lora_triggers` returns a sane result against a real `<lora:...>` tag — do not ship
      Phase 1 with only synthetic/no-LoRA testing.

---

## Phases 2+3 (MERGED): Semantic analyzer — DONE

**Architecture change from the original plan, directed during implementation**: Phases 2
(place-aware precedence) and 3 (Matrix-Tree for unknown tokens only) are merged into ONE semantic
analyzer that runs the Matrix-Tree computation over **every** prompt segment (not just unknowns),
using CLIP-embedding similarity as an unsupervised nearest-neighbor classifier, and folds the
"nearest preceding anchor" precedence convention into the Matrix-Tree edge potentials as a soft
prior rather than a separate hard tie-breaking rule. `_build_clusters()`'s hard 2-way (later 3-way)
partition is superseded, not extended — this is a deeper redesign than originally scoped.

### What was implemented (`modules/rule_engine/`)

- **`matrix_tree.py`** — Koo et al. 2007 §3.1-3.2's exact algorithm, implemented for the first time
  this session (Lean only proved properties ABOUT it, not the algorithm itself): build the
  edge-weight matrix `A` and root-selection vector `r` from potentials `theta`, the Laplacian `L`,
  `L̂` (row 1 replaced by root scores), partition function `Z = det(L̂)`, and every marginal
  `μ_{h,m}` from ONE matrix inversion via the paper's Kronecker-delta formula. Single-root setting
  only (§3.3's multi-root case not implemented — not needed here).
  - **Verified against independent brute-force enumeration** (every valid parent assignment,
    filtered to single-root trees, summed directly) for n=2,3,4 random potential matrices: partition
    function and all marginals match to `1e-6`, and the marginal-sums-to-1 property
    (`MatrixTreeSoftOwnership.lean`'s `marginal_sums_to_one`) holds empirically in every case.
- **`clip_vector_db.py`** — a small in-memory nearest-neighbor cache (cosine similarity over
  L2-normalized embeddings), backend-agnostic (plain numpy in/out; caller supplies embeddings from
  whichever text encoder is active).
- **`semantic_analyzer.py`** — `analyze(token_table, embeddings, vector_db=None)`: for each segment,
  resolves one of the three specified situations —
  1. **Exact rule-engine match** (Phase 1 already classified it) → probability 1.0 for each assigned
     class, AND seeds the vector database as a labeled example.
  2. **Near-match** (cosine similarity ≥ `NEAR_MATCH_THRESHOLD=0.85` to a seeded example) → inherits a
     similarity-weighted soft k-NN blend of the neighbors' classes.
  3. **Novel/unknown** (neither) → no class prior; relies entirely on the Matrix-Tree structure.

  Output is a **per-class probability** (independent `[0,1]` membership, not a softmax categorical —
  classes legitimately overlap, e.g. `comic` = `{GLOBAL, CONCEPT}`), matching the "value expressed as
  probability of class" requirement. Then builds Matrix-Tree potentials combining CLIP cosine
  similarity (semantic relatedness, computed over ALL segment pairs, not just unknown-to-anchor) with
  a precedence bias term (earlier, OBJECT-like segments get a root/parent-preference boost) and an
  OBJECT-probability-scaled root score, and calls `matrix_tree.compute_tree_marginals` — "the tree
  that represents what we know now."

### Verification performed

- [x] Matrix-Tree algorithm: brute-force cross-check, n=2/3/4, exact match (see above).
- [x] End-to-end pipeline test with constructed synthetic embeddings exercising all three
      situations on one prompt (`1girl`/`brown hair`/`1boy` exact-matched by the rule engine;
      `chestnut hair` engineered to be embedding-close to `brown hair` → correctly triggers
      situation 2 and inherits `ADJ`; `xyzzyx_made_up_thing` engineered as dissimilar noise →
      correctly triggers situation 3). All assertions passed.
- [x] Confirmed the precedence-bias term has a real, visible, tunable effect: in the test, `1girl`
      (an earlier OBJECT) gets non-trivial probability (0.27) as `brown hair`'s parent even though a
      near-duplicate tag has higher raw cosine similarity (0.56) — the two terms genuinely compete
      rather than one trivially dominating.

### NOT verified (no live WebUI/torch/CLIP available in this sandbox)

- Real CLIP embeddings were never used — only hand-constructed synthetic vectors. The MECHANICS
  (3-situation branching, vector DB growth, Matrix-Tree wiring) are verified; whether
  `NEAR_MATCH_THRESHOLD=0.85` and the `W_ROOT_OBJECT`/`W_EDGE_SIMILARITY`/`W_EDGE_PRECEDENCE` weights
  (currently `4.0`/`4.0`/`2.0`, chosen for a visible, balanced effect in the synthetic test, not
  tuned against real data) behave sensibly on real CLIP-embedded Danbooru tags is unverified — this
  must be checked and likely re-tuned once wired to the real text encoder (Phase 5).
- Per-segment embedding EXTRACTION (mean-pooling each token table entry's own `[start,end)` range out
  of the already-computed conditioning embedding tensor) is not yet implemented — `semantic_analyzer
  .analyze()` currently takes `embeddings` as a plain argument the caller must supply. Phase 5 must
  implement the actual slicing-from-conditioning-tensor step and confirm it adds zero extra
  text-encoder forward passes (the original Phase 3 zero-cost verification checklist item, not yet
  exercised against a real model).
- The originally-planned paraphrase-pair regression test (`"a boy with brown hair"` vs `"1boy, brown
  hair"` clustering the same way) was not run — it needs real CLIP embeddings to be meaningful; a
  synthetic-embedding version would only test the mechanics already covered above.

---

## Phase 2.5 (NEW): Refine engine — the final pass — DONE

**Added during implementation, directed by the user**: a final reconciliation pass that combines
the rule engine's hard facts (Phase 1, ground truth) with the semantic analyzer's soft structural
inference (Phase 2+3, per-chunk Matrix-Tree) into one "intention matrix tree" ready for Phase 5, plus
multi-chunk masking so long prompts remain fully debuggable rather than silently truncated to the
first 75 tokens.

### What was implemented

- **`chunk_mask.py`** — computes which logical 75-token CLIP chunk each segment falls into
  (`chunk_index = start // 75`) and groups segments by chunk. Documented caveat: this is a debug-grade
  approximation of `tokenize_line`'s real chunking (which prefers to break at a comma, potentially
  shifting the boundary by a token or two) — good enough for visibility, not a substitute for fixing
  `get_token_subspaces` itself to track exact chunk boundaries (still a known gap).
- **`semantic_analyzer.analyze_multi_chunk()`** — runs one independent Matrix-Tree per chunk (chunks
  are independent attention contexts in the real model — there is no cross-chunk edge to compute),
  while sharing ONE vector database across all chunks, so a tag exact-matched in chunk 0 can still
  near-match an unclassified tag in chunk 1.
- **`refine_engine.py`** — `refine(token_table, semantic_results)`:
  - **Location-aware tie-breaking**: when a segment's top parent candidates are within
    `TIE_EPSILON=0.05` of each other, break the tie by PROMPT POSITION — prefer the nearest
    *preceding* candidate (this project's established convention), falling back to nearest by
    absolute distance if none precede.
  - **Reconsideration**: `META`-classified segments are forced to `ROOT` and marked `excluded`
    regardless of what the tree computed (they're non-region-specific by definition). `OBJECT`- and
    `ADJ`-classified segments have their candidate parents restricted to `{ROOT} ∪ OBJECT segments}`
    — a real bug this caught during testing: two semantically-similar attribute tags (e.g. two
    hair-color tags) were mutually attracting as each other's "parent" before this restriction was
    added, which is structurally wrong (an attribute should bind to an entity, never to another
    attribute). `VERB` currently gets no special restriction (a known limitation — the user's own
    class description notes VERB may legitimately tie multiple objects together, which needs a
    multi-parent representation not implemented here).
  - Segments with no class at all (situation 3) get one inferred from their chosen parent: attached
    to an `OBJECT` → `ADJ`; attached to `ROOT` → `GLOBAL` — closing the loop so every segment ends up
    classified by the end of the pipeline.

### Verification performed

- [x] Re-ran the whole session's running example (`"masterpiece, 1girl, brown hair, 1boy, blue
      hair"`) end-to-end: `masterpiece` → forced `ROOT`+excluded; `brown hair` → `1girl`; `blue hair`
      → `1boy` — the exact correct final structure, resolved purely from reconsideration + the
      nearest-preceding tie-break (no manual correction needed).
- [x] Multi-chunk masking: constructed a synthetic long prompt with segments spanning chunk 0
      (`1girl, brown hair`) and chunk 1 (`1boy, blue hair`) — confirmed two independent 2-node
      Matrix-Trees were built (no cross-chunk contamination), and `refine()` produced one unified,
      correctly-ordered cross-chunk view.
- [x] Location-aware tie-breaking in isolation: constructed embeddings where an attribute tag is
      *exactly* equally similar to two candidate entities (a genuine tie, not just close) — confirmed
      the nearest PRECEDING candidate (by position) wins, not an arbitrary one.

### NOT verified

- Same real-CLIP caveats as Phase 2+3 above — all tests use hand-constructed synthetic embeddings.
- `chunk_mask.py`'s boundary approximation vs. `tokenize_line`'s real comma-preferring chunk breaks —
  not cross-checked against the actual tokenizer.

---

## Phase 4: Lean formalization (soft-ownership generalization) — DONE

Implemented as a new file, `RFVProofs/MatrixTreeSoftOwnership.lean` (imports
`TokenSubspaceGuidance.lean`, registered in `RFVProofs.lean`), rather than a new Part inside
`TokenSubspaceGuidance.lean` — the marginal-distribution theorems (Part 8) needed their own
abstract `Tree`/`Candidate` type variables, warranting a separate file per the plan's original
"decide based on size" note.

### What was proved (8 theorems, 0 sorries, `lake build` 2634 jobs clean)

1. ✅ **Soft-ownership row validity**: `soft_corrected_row_is_distribution` — direct reuse of
   `renormalized_row_is_distribution` (Part 1), not a re-derivation.
2. ✅ **Soft leak attenuation validity**: `soft_leak_preserves_nonneg` for continuous
   `leak_strength, w ∈ [0,1]`, PLUS two theorems not originally scoped but added because they were
   cheap and valuable: `soft_leak_matches_hard_at_w_one`/`_at_w_zero` — proves the new continuous
   formula is EXACTLY today's shipped boolean formula at the degenerate weights, a checkable
   migration-safety contract for Phases 1-3/5.
3. **3-way partition well-definedness**: NOT done as a separate theorem — superseded by the
   `soft_leak_matches_hard_at_w_zero`/`_at_w_one` pair, which directly certifies the two hard cases
   (own/rival) the 3-way partition reduces to; "meta" exclusion doesn't need a correction-math proof
   since excluded groups simply never enter the formula at all (a Python-side gating decision, not a
   Lean-provable property).
4. ✅ **Cost-model extension**: `matrix_tree_cost_step_invariant` (cost independent of `n_steps`) +
   `matrix_tree_strictly_cheaper_than_gradient_based` (reuses Part 6's `GradientBasedMechanism` defeq
   trick directly) + `matrix_tree_plus_subspace_correction_extraNFE` (combined pipeline still `0`).
5. ✅ **Did NOT attempt**: Tutte's Matrix-Tree Theorem itself — cited (Koo et al. 2007 §3, read
   directly from the user-supplied PDF) rather than reproved. Instead formalized the simpler,
   fully-tractable combinatorial consequence that actually matters for the correction math:
   `marginal_sums_to_one`/`marginal_is_valid_distribution` (Part 8) — marginals form a partition of
   unity over candidate parents purely because every spanning tree gives each node exactly one
   parent, proved via Mathlib's `Finset.sum_fiberwise`, no determinant algebra needed.

See `THEOREM_MATRIX_TREE_BUFFER.md §6` for the full theorem table and implementation notes.

---

## Phase 5: Integration + WebUI wiring — DONE

### What was implemented

- **`modules/rule_engine/embedding_extraction.py`** — reads the FINAL, style-expanded prompt
  (`p.all_prompts`) and the model's ACTUAL already-computed conditioning tensor (`p.c`, navigated as
  `p.c.batch[0][0].schedules[-1].cond`), falling back to a same-text re-encode only if that structure
  doesn't match. This was a direction correction mid-implementation: the first draft would have
  re-encoded `p.prompt` (the raw, pre-style prompt) independently, which could diverge from what the
  model actually sees whenever style presets or other prompt processing are in play — fixed to read
  the FINAL artifacts instead, per the explicit requirement.
  `slice_segment_embedding`/`extract_all_segment_embeddings` mean-pool each segment's own
  `[chunk_index, local_start, local_end)` range directly out of that tensor (multi-chunk aware, using
  Phase 2.5's `chunk_mask.py`) — **verified correct** with a mock tensor exercising both the BOS-offset
  and cross-chunk-boundary math (a 2-chunk synthetic (1,154,4) tensor, exact value assertions passed).
- **`modules/rule_engine/pipeline.py`** — `run_pipeline(model, p, ruleset)` orchestrates the full
  chain: final prompt → `get_token_subspaces` → `build_token_table` → `compute_chunk_mask` →
  `get_prompt_embeddings_for_pipeline` → `analyze_multi_chunk` → `refine` → `FinalTokenInfo` list.
- **`sure_token_guidance.py`** — added `_build_clusters_from_intention_tree(ranges, final_tokens)`,
  which replaces the flat 8-keyword heuristic's `cluster_of` computation with the intention tree's
  `final_parent`/`final_classes`/`excluded` fields (verified correct against the exact running
  example: `masterpiece`→excluded, `1girl`/`1boy`→their own clusters, `brown hair`→1girl's cluster,
  `blue hair`→1boy's cluster). Threaded `final_tokens` as an optional parameter through
  `_apply_token_subspace_corrections` → `_make_token_guidance_hook` → `patch_model_with_token_guidance`
  — when omitted, everything falls back to Plan 03's keyword heuristic unchanged (no breaking change).
  Crucially, only the **prompt-side cluster assignment** (which tag belongs to which entity) came
  from the intention tree — the **spatial** per-query-row ownership decision (which image region is
  currently which entity, via live attention-mass argmax + confidence gate) is untouched, since that's
  a genuinely different piece of information the text-side tree cannot supply.
- **Intention-drift scoring** (the explicit ask: "more accurately decide what got loss, wrong,
  drift"): a new `intention_drift_score` diagnostic, computed as the per-group observed `leak_frac`
  weighted by the intention tree's own `parent_confidence` for that group — a confidently-asserted-
  but-violated binding counts as more drift than an uncertain one. Falls back to being numerically
  identical to `leak_score` when no intention tree is available (verified this is a strict
  refinement, not a behavior change, in that case). `clpc_error.compute_token_guidance_score` now
  uses `intention_drift_score` in place of the raw `leak_score` in the G-score product (not
  multiplied alongside it, to avoid double-counting the same underlying signal).
- **WebUI**: `sd_forge_sure_token_ag`'s accordion gained "Use intention tree" (default on, with
  automatic, exception-safe fallback to the keyword heuristic) and a "Tag rule presets" multi-select
  (danbooru/pony/illustrious, loaded together via `load_rule_yaml`). Fixed a real path bug caught
  during implementation: `scripts.basedir()` returns the EXTENSION's own root directory, not the
  `scripts/` subfolder the script file lives in — the preset-directory path had one `..` too many;
  verified the corrected path resolves to the real `configs/prompt_rules/` directory.

### Verification performed

- [x] Embedding slicing math (BOS offset + cross-chunk column indexing + mean-pooling): exact match
      against hand-computed expected values, via a mock tensor object mimicking torch's API surface.
- [x] `_build_clusters_from_intention_tree`: exact match against the session's running example.
- [x] Full pipeline + all touched files: `py_compile` clean.
- [x] `scripts.basedir()`-relative preset path: resolved and confirmed to list the 3 real preset files.

### NOT verified (no live WebUI/torch/CLIP in this sandbox — same standing caveat as Phases 2+3/2.5)

- `p.c.batch[0][0].schedules[-1].cond` navigation — matches this session's reading of
  `prompt_parser.py`'s conditioning classes, never exercised against a real `StableDiffusionProcessing`
  object. If this navigation is wrong, `embedding_extraction.py` will print a clear diagnostic and
  fall back to re-encoding the correct final prompt text rather than silently misbehaving — but the
  "zero extra encoder cost" property only holds on the primary (untested) path.
- The whole hook-side change (`_apply_token_subspace_corrections`'s new `final_tokens` branch) has
  not been run against real tensors — only the pure-Python translation function was unit-tested.
- SDXL's dual CLIP-L/CLIP-G encoders: no SDXL-specific handling was added in this phase; the existing
  caveat from `get_token_subspaces` applies unchanged.
- The `intention_drift_score` weighting scheme (confidence × leak_frac) is a reasonable, documented
  design choice, not empirically tuned against real generations.

### First real live-WebUI test (2026-07-06) — one real bug found and fixed

The user ran the first real test against a live SDXL-family model with a 13-segment prompt. Result:
the rule engine, chunk masking, and fallback safety net all worked exactly as designed — `p.c`
navigation reached the schedule entry correctly, but crashed on `.dim()` with
`AttributeError: 'dict' object has no attribute 'dim'`, and the pipeline's own try/except caught it
cleanly and fell back to the Plan 03 keyword heuristic without crashing generation (confirming the
safety-net design worked as intended).

**Root cause**: for this model family, `p.c`'s schedule entry `.cond` is not a bare tensor — it's a
dict `{"crossattn": tensor, "vector": pooled_tensor}` (SDXL needs both the cross-attention sequence
and the pooled vector). This exact shape was already visible in this session's own earlier reading of
`modules_forge/forge_sampler.py::cond_from_a1111_to_patched_ldm`, but `embedding_extraction.py` hadn't
accounted for it.

**Fix**: added `_unwrap_cond()` (handles both the dict form and a `(cond, pooled)` tuple form some
`get_learned_conditioning` call paths return), used on both the primary (`p.c`-read) and fallback
(re-encode) paths. Verified against a mock reproducing the exact crash shape (a dict with
`crossattn`/`vector` keys) — unwrap + slice + pool now succeeds end-to-end.

**Also improved per user request**: debug/info prints across `embedding_extraction.py` and the
extension script now explicitly confirm which conditioning form was detected (dict/tuple/plain),
print the resulting tensor shape (so it can be visually cross-checked against the attn2 hook's own
reported `Nk`), and the pipeline failure handler now prints a full traceback plus an explicit
SUCCEEDED/FAILED line, rather than only a one-line exception summary.

**Not yet re-tested**: the fix above hasn't been re-run against the live model yet — the next test run
should show `[RuleEngine] conditioning is a dict (keys=['crossattn', 'vector']...) — unwrapped to the
crossattn tensor OK` instead of the crash, followed by the intention-tree pipeline actually completing
(`SUCCEEDED: N segment(s) reconciled`) instead of falling back.

**Also noted, not a code bug**: the log showed `cfg=TokenGuidanceConfig(tau_vanish=0.05, ...)` — the
OLD pre-Plan-03-calibration-fix default. This is stale Gradio UI state from before that default was
changed to `0.4`; a browser refresh/reload of the WebUI page (not just re-running generation) picks up
the new default.

### Second real live-WebUI test (2026-07-07) — genuine architectural gap found and fixed

With the conditioning-unwrap fix in place, the pipeline ran to completion, but surfaced a real
structural gap: the prompt `"brown hair, one 18 years old girl, biker clothes, masterpiece, ..."` has
**no bare Danbooru-style entity tag anywhere** — "girl" only appears embedded inside the natural-
language phrase `"one 18 years old girl"`, which doesn't match any rule-engine pattern as a whole
segment, and since no OTHER tag in the prompt seeded the vector database with a "girl"/"1girl"
example, the CLIP near-match layer had nothing to compare it against either. Result: `object_indices`
came back completely empty, so the ENTIRE "attributes bind to entities" mechanism — which the whole
Phase 2.5 reconsideration design depends on — had no anchor to work with at all, and every segment's
final parent was essentially noise (`"brown hair"` landed on `"biker clothes"` with `p=0.115`, barely
above the uniform 1/13≈0.077 baseline).

**Fix**: added a last-resort fallback in `refine_engine.py` (`_fallback_anchor_scan`): for any segment
that BOTH the rule engine (Phase 1, whole-segment pattern match) AND the semantic analyzer (Phase 2+3,
CLIP near-match) left completely unclassified, scan its raw text for a small set of known
entity-indicating whole words (girl/boy/woman/man/person/child/male/female + plurals — the same set
Plan 03's original keyword heuristic used, now repurposed as a safety net rather than the primary
mechanism) and promote to `OBJECT` if found. This runs BEFORE `object_indices` is computed, so the
existing (already-correct) reconsideration/tie-breaking logic gets a real anchor to work with.

**Verified**: reconstructed the exact 13-segment scenario with embeddings matched to the real log's
actual behavior (all 6 unclassified segments genuinely dissimilar to everything, i.e. true situation-3
"novel/unknown", not an artifact of a badly-constructed test) — confirmed `"one 18 years old girl"` is
now promoted to `OBJECT`, and `"brown hair"` (already correctly `ADJ` via the rule engine) now binds to
her as its parent instead of an arbitrary quality tag.

**Known remaining limitation, by design**: this only closes the OBJECT-recognition gap specifically
(the single most architecturally important one, since every other reconsideration rule depends on a
real anchor existing) — it does not generally solve every unclassified attribute. In the test,
`"biker clothes"` (containing no fallback-anchor word) still lands on a fairly arbitrary parent with
these synthetic, orthogonal-random embeddings; real CLIP embeddings should do meaningfully better
here since "biker clothes" has genuine semantic content the random test vectors don't capture — this
remains to be confirmed against a live model.

### Generalization to a proper sub-word mechanism (2026-07-07) — META_TREE decomposition

The fallback-anchor scan above closed the OBJECT gap, but only via a hardcoded 14-word list and only
for OBJECT — every other class (ADJ, VERB, GLOBAL, CONCEPT, LORAACT, META) had no equivalent, and any
prompt using natural language rather than bare Danbooru tags for those classes would fall through to
pure structural guessing. Per request, generalized this into a proper Phase 1 mechanism instead of a
Phase-2.5 patch:

- `modules/prompt_parser.py::get_token_subspaces()` now also splits each group's bare text into
  sub-words (`_split_into_subwords`, splitting on space AND underscore — the same identifier-boundary
  convention a real lexer uses) and computes an approximate token-index sub-range for each word via
  proportional distribution of the group's real token count across its words (`cum_tokens += n *
  (weight/total_weight); local_end = round(cum_tokens)`). This had to be proportional rather than
  independently-tokenizing each word and concatenating: a self-authored mock-tokenizer test showed
  independently tokenizing "pale" + "skin" can sum to MORE tokens than tokenizing "pale_skin" as one
  string (BPE merges differently), which would silently overflow the parent group's own token range if
  summed naively. The proportional-distribution fix guarantees sub-ranges always stay within
  `[offset, offset+n)`.
- `modules/rule_engine/token_table.py` adds `_decompose_and_classify()`: when a segment's whole text
  matches no rule-engine pattern, split it into sub-words and classify EACH one against the SAME
  compiled ruleset (all 7 classes, whatever patterns are actually loaded — not a hardcoded word list),
  union the non-empty results back onto the parent segment. The per-word breakdown is preserved (not
  discarded) as a new `MetaTreeNode` — a composite node pointing at the classified sub-tree, so anything
  downstream that wants to know WHICH word contributed WHICH class still can. `refine_engine.py`'s
  `_fallback_anchor_scan` is kept as a documented second-level-only safety net (for the edge case of a
  custom/minimal ruleset with no OBJECT pattern loaded at all) — the gate gets checked but rarely fires
  now since Phase 1 handles the common case earlier and more generally.

**Verified** against the real `danbooru.yaml` preset (not a mock ruleset) with 5 segments:
`"one 18 years old girl"` → correctly decomposes to `OBJECT` via "girl" and prints a META_TREE
breakdown (`[('girl', ['OBJECT'])]`); `"pale_skin"` and `"biker clothes"` correctly stay unclassified
at Phase 1 (this preset has no ADJ pattern for "pale"/"skin"/"biker"/"clothes" — a content-coverage
gap in the YAML, not a mechanism bug); `"very aesthetic looking man"` also correctly stays
unclassified at Phase 1 — initially flagged as a suspected bug, but confirmed to be correct: Danbooru's
actual tagging convention uses `1boy`/`1girl` regardless of apparent age rather than bare "man"/"woman",
so `danbooru.yaml`'s OBJECT pattern intentionally doesn't cover those words. Ran the full pipeline
(Phase 1 through refine engine) on the same 5 segments and confirmed the SECOND layer — refine_engine's
`_fallback_anchor_scan`, which does cover man/woman/person/child/male/female — correctly rescues
`"very aesthetic looking man"` to `OBJECT` at Phase 2.5 when Phase 1's ruleset doesn't cover it,
confirming the two-layer design (general mechanism first, small hardcoded safety net second) works
end-to-end with no regressions.

### Third real live-WebUI test (2026-07-07) — location-aware tie-breaker wasn't actually location-primary

Live log: `1girl, brown hair, 1boy, blue hair, masterpiece, ...` (plus quality/style tags) — `"blue
hair"` (an ADJ, immediately after `"1boy"` in the prompt) ended up bound to `"1girl"` (three segments
back) instead, at `p=0.572` vs `1boy`'s `p=0.246`. The "location-aware tie-breaker" as originally built
only consulted position when the top two Matrix-Tree candidates were within `TIE_EPSILON=0.05` of each
other — a 0.326 gap is nowhere near a tie, so position never got a vote here at all, and raw
CLIP-similarity noise (an SDXL text embedding doesn't encode word ORDER, only word IDENTITY/meaning)
picked the parent instead.

**Root cause, one level deeper**: `semantic_analyzer.py`'s Matrix-Tree edge potential already had a
"precedence bonus" term (`W_EDGE_PRECEDENCE * object_prob(h)` for any `h < m`), but it's applied FLAT to
every preceding OBJECT candidate regardless of distance — `"1girl"` (3 segments back) and `"1boy"` (1
segment back) got the exact same structural bonus. Nothing in the system actually encoded "adjacent
beats merely-earlier"; only semantic similarity discriminated between multiple valid preceding
entities, and that signal has no reason to respect prompt order.

**Fix**: in `refine_engine.py`, for the OBJECT and ADJ reconsideration branches specifically, replaced
the tie-window-based `_break_tie_by_position` call with a new `_nearest_preceding_or_best`: position is
now PRIMARY, not a last-resort tie-breaker, among the class-restricted candidate set. The nearest REAL
(non-ROOT) preceding candidate wins outright whenever one exists — this is the load-bearing convention
these tag prompts are actually written by (`1girl, ..., 1boy, blue hair` unambiguously means the boy's
hair to a human reader), so a same-order-of-magnitude Matrix-Tree marginal favoring a farther candidate
is still just embedding noise next to it. Falls back to the tree's raw top-marginal candidate only when
NOTHING in the restricted set precedes the child at all (a genuine forward reference, e.g. an attribute
written before any entity appears, where position gives no guidance). Left the generic (situation-3,
still-unclassified) branch's near-tie-only breaking untouched — those segments have no explicit
OBJECT/ADJ class yet when this runs, so the strong "attribute binds nearest entity" convention doesn't
apply to them the same way; the Matrix-Tree Phase-2/3 precedence-bonus term in `semantic_analyzer.py`
was left as-is too (a global distance-decay there would be a larger, separate change affecting every
edge in the tree, not just OBJECT/ADJ ownership).

**Verified**: reconstructed the exact failing shape (`1girl, brown hair, 1boy, blue hair, masterpiece,
...`) with embeddings deliberately biased so `"blue hair"` is semantically closer to `"1girl"` than to
`"1boy"` (reproducing the reported `p=0.851` vs `p=0.143`-style noise) — confirmed `"blue hair"` now
binds to `"1boy"` regardless. Re-ran the full 14-segment scenario from the actual log (all three
presets loaded) and confirmed the same result at scale. Re-ran the pre-existing running-example and
forward-reference (attribute before any entity) regressions — both still pass, confirming this is a
strictly narrower, more correct rule rather than a behavior change for cases that were already right.

### Fourth real live-WebUI test (2026-07-07) — GLOBAL/CONCEPT had no reconsideration branch at all

Same log as the tie-breaker fix above also showed `"depth of field"` (rule-engine classified `GLOBAL`,
exact match, high confidence) ending up bound to `"1boy"` at `p=0.277`, as if it were an attribute of
the boy specifically. Unlike the tie-breaker bug, this wasn't a matter of the right rule picking the
wrong candidate — `refine_chunk`'s `if/elif` chain simply had no branch for GLOBAL/CONCEPT at all, so
any such segment fell straight through to the generic (situation-3-style) tie-break, which has no
concept of "whole-image, don't bind to any entity."

**Fix**: added a dedicated sentinel `GLOBAL_NODE = -1` (depth -1, distinct from `ROOT`/depth 0) in
`refine_engine.py`. Any segment carrying `GLOBAL` or `CONCEPT` (schema.py: "whole-image effect" /
"whole-image style/theme") is now force-attached to `GLOBAL_NODE` via an explicit reconsideration
branch — visually active (unlike META, it is NOT `excluded` from attention correction), just never
entity-owned. `_build_clusters_from_intention_tree` in `sure_token_guidance.py` was updated from
`final_parent == 0` to `final_parent <= 0` so both ROOT and the new GLOBAL_NODE map to the pre-existing
"cluster -1 = unclustered/global" bucket that module already used — no new correction-core logic
needed, since -1-as-unclustered was already a first-class concept there.

**Also added, per request ("see if CLIP can indicate what is possible is GLOBAL" / "node promotion")**:
a second GLOBAL safety net for segments that stay fully unclassified through both the rule engine and
the strict CLIP near-match (situation 3). `semantic_analyzer.py` now computes, for every segment
regardless of situation, a `weak_global_hint` — the best cosine similarity to any GLOBAL/CONCEPT vector-
DB example seen so far, deliberately IGNORING the strict `NEAR_MATCH_THRESHOLD` (0.85) used for real
classification. In `refine_engine.py`, a segment is promoted (node-promoted) straight to `GLOBAL_NODE`
if it BOTH trails every OBJECT/ADJ-classified segment in its chunk (the Danbooru convention of
subject-then-attributes-then-quality/style-tags-last) AND clears a softer `GLOBAL_HINT_THRESHOLD` (0.5)
on that weak hint — the same "node promotion" concept as the pre-existing OBJECT fallback scan, but
driven by an actual CLIP similarity signal instead of a hardcoded word list.

**Verified**: reconstructed the exact failing case (`"depth of field"` in the real 14-segment log) —
confirmed it now resolves to `final_parent=GLOBAL_NODE`, `final_parent_text="GLOBAL"`, `excluded=False`,
and that `_build_clusters_from_intention_tree` maps it to cluster `-1` as expected. Separately
constructed a case with a genuinely novel trailing tag (`"cinematic vibe"`) whose embedding was
deliberately biased to a 0.65 cosine similarity to a known GLOBAL exemplar (`"depth of field"`) —
below the strict 0.85 near-match bar but above the 0.5 safety-net floor — and confirmed it gets
node-promoted to `GLOBAL_NODE` too. Re-ran all three prior regressions (running example, forward
reference, location-aware tie-breaker) — all still pass unchanged, confirming this is additive and
doesn't disturb OBJECT/ADJ/META handling.

### Fifth real live-WebUI test (2026-07-07) — OBJECT-OBJECT false parenting + PRIMARY_OBJ + weak links

Same log family: `"1boy"` (an independent subject) ended up parented by `"1girl"` at `p=0.465`, exactly
the kind of false entity-to-entity chaining the tie-breaker fix was never meant to cause, but did:
`_nearest_preceding_or_best`, when applied to the OBJECT reconsideration branch, treats ANY real
preceding candidate as automatically preferable to ROOT — correct for an ATTRIBUTE that should bind to
its nearest entity, wrong for two CO-EQUAL subjects, which should never chain into each other. Root
cause one level deeper: `matrix_tree.py`'s single-root formulation permits only ONE node to attach
directly to ROOT (its own docstring: "the virtual root-symbol has EXACTLY ONE child"), so with two
independent subjects the tree is structurally forced to make one of them "depend" on the other just to
produce SOME valid spanning tree — there is no way for the raw Matrix-Tree marginals alone to express
"these are peers." User-reported downstream symptom: this false parenting visibly mixed the two
characters' hair-color attributes.

**Fix — new `PRIMARY_OBJ` tag class**: added to `schema.py`'s `TAG_CLASSES` (now 8 classes) as an
explicit subset of `OBJECT` — a main character/subject that always independently appears in the image
and must never be subordinate to another entity, distinct from a plain `OBJECT` tag (e.g. "cat",
"sword") which CAN belong to one. `danbooru.yaml`'s character-count patterns
(`girl|boy|other|futanari`, `multiple_(girls|boys|others)`) now carry both `[OBJECT, PRIMARY_OBJ]`;
`"solo"`/`"solo_focus"` stay `OBJECT`-only (compositional signals, not a distinct character). In
`refine_engine.py`, a new reconsideration branch checked BEFORE the plain OBJECT branch: any
`PRIMARY_OBJ` segment is forced straight to `ROOT`, never routed through `_nearest_preceding_or_best`
at all — this is what actually breaks the false-chaining, since that function's whole premise (nearest
real candidate beats ROOT) is exactly backwards for peer subjects. The plain OBJECT branch (now only
reached for OBJECT-without-PRIMARY_OBJ, i.e. genuinely secondary/subordinate objects) was narrowed from
`{ROOT} ∪ other OBJECTs` to `{ROOT} ∪ PRIMARY_OBJ segments` — a subordinate object binds to a main
character or ROOT, never to another subordinate object — falling back to the full OBJECT set for a
minimal/custom ruleset that doesn't define `PRIMARY_OBJ` at all (backward compatible).

**Also added, per request ("node may have non-looping shared parent like B-Tree using weak link to
indicate some description is for group of OBJECTs")**: `FinalTokenInfo.weak_links` — a NEW
`WEAK_LINK_MARGIN` (0.15) pass over the OBJECT/ADJ branches' restricted candidate pool records any OTHER
PRIMARY_OBJ/OBJECT index whose marginal came within that margin of the chosen parent's, WITHOUT
changing `final_parent` itself. This is deliberately non-structural — the Matrix-Tree math still
requires a strict single parent per node (no cycles, no multi-parent), so `weak_links` is pure auxiliary
metadata layered on top, like a B-tree node's non-structural cross-references, surfacing cases (e.g. an
attribute like "matching outfits" that's nearly equally likely to belong to either of two characters)
the strict tree structure can't represent directly. Threaded through
`sure_token_guidance.py::_build_clusters_from_intention_tree` (now returns a third `weak_cluster_links`
list, translated into 0-indexed cluster space) and consumed in the leak-attenuation loop: a rival
cluster that's a documented weak link for a group is attenuated at HALF `leak_strength` rather than
full, instead of being fully suppressed the instant that rival dominates a query position — the
description plausibly belongs there too, so full suppression would fight the very ambiguity the
Matrix-Tree itself couldn't resolve.

**Verified**: reconstructed the exact failing 14-segment log scenario — confirmed both `"1girl"` and
`"1boy"` now independently resolve to `final_parent=ROOT` (no more false chaining), while
`"brown hair"`/`"blue hair"` still correctly bind to their respective nearest entity (no attribute
mixing). Constructed a secondary-object case (`"1girl", "solo", "1boy"`) and confirmed `"solo"`
(OBJECT-only, no PRIMARY_OBJ) binds to the nearest PRIMARY_OBJ (`"1girl"`) rather than chaining to the
OTHER secondary object or peer-chaining to `"1boy"`. Re-ran the four prior regressions (running example,
forward reference, location-aware tie-breaker, GLOBAL node) — all still pass. The weak-link
leak-attenuation math itself (`sure_token_guidance.py`) could not be executed in this sandbox (no torch
available here — confirmed via `import torch` failing outright, not just the module-level stub trick
used for the pure-numpy rule-engine tests elsewhere in this plan); its boolean-masking LOGIC was instead
verified via a numpy stand-in mirroring the exact same tensor shapes/operations (own-cluster comparison,
weak-rival masking, `where`-based blending) — confirmed own-cluster positions stay untouched,
weak-linked rivals get exactly half attenuation, and non-weak rivals get full attenuation, but this
specific piece remains **unverified against a live model/real torch** and should be watched on the next
actual WebUI run.

### Reverted: weak-link leak-attenuation softening (2026-07-07) — confirmed harmful on first live run

The very next live-WebUI run (a real two-character prompt, 26 CLPC steps) showed exactly the risk the
"unverified against a live model" caveat above was warning about: `vanish_score` converged cleanly
(0.985 → 1.0) and `token_score` improved monotonically (0.956 → 0.971) over the whole run — token
guidance itself was healthy — but `leak_score` stayed flat at ~0.83 for all 26 steps, never converging
upward the way every other per-step metric did. Root cause: `WEAK_LINK_MARGIN` (0.15) is far too
generous against REAL CLIP embeddings — the synthetic random-vector tests used to verify this feature
happened not to exercise this failure mode, but with real embeddings, two characters described by
fairly generic attribute text (e.g. "brown hair" vs "blue hair") routinely land within 0.15 of each
other on ordinary embedding noise alone, not genuine shared-description ambiguity like "matching
outfits". That meant the leak-attenuation loop was applying half-strength correction far more broadly
than intended — across most ADJ segments in a 2+ character prompt, not just the rare genuinely-shared
case — leaving real attention leak under-corrected for the entire run.

**Fix**: reverted the leak-attenuation loop in `sure_token_guidance.py` to always apply full-strength
attenuation, regardless of `weak_cluster_links`. `FinalTokenInfo.weak_links` and the threaded
`weak_cluster_links` return value are UNCHANGED and still computed/available (confirmed harmless —
pure Python, no torch cost, and the underlying "which candidates are near-equally plausible" signal
is still useful diagnostic information) — only the consumption inside the hot correction path was
pulled back out. A future attempt at actually softening leak-attenuation for genuine group-shared
descriptions should use a much tighter margin than 0.15 (informed by real CLIP similarity distributions,
not synthetic random vectors) and be verified against a live model before being trusted, exactly as the
PRIMARY_OBJ and GLOBAL_NODE fixes above were confirmed safe (in those cases, mechanically: PRIMARY_OBJ
only touches a diagnostic confidence weight since entities already get their own cluster regardless of
`final_parent`, and GLOBAL_NODE only affects segments already excluded from entity-region correction).

**Lesson for this project's own verification discipline**: this is the first case this session where a
numpy-stand-in-verified piece of tensor logic (correct in isolation) still caused a real regression once
exercised with real embeddings rather than synthetic ones — the boolean-masking arithmetic was right,
but the THRESHOLD calibration it depended on wasn't, and no amount of synthetic-vector testing could
have caught that. Worth remembering before reaching for the same "verify via numpy stand-in, mark as
unverified against live model" pattern again for anything threshold-sensitive.

---

## Final Phase: Verification

1. `lake build` clean (Phase 4) — done.
2. Rule-engine unit tests pass (Phase 1) — done.
3. Semantic analyzer / refine engine / Phase 5 mechanics tests pass (Phases 2+3, 2.5, 5) — done, all
   on synthetic/mock data.
4. **Remaining, requires a live WebUI + real model**: manual test with both the exact running-example
   tag prompt and a natural-language prompt, watching for `[RuleEngine]`/`[SemanticAnalyzer]`/
   `[RefineEngine]`/`[TokenSubspaceGuidance]` console output to confirm: (a) the final prompt (not raw)
   is what gets analyzed, (b) `p.c` navigation succeeds or the documented fallback fires cleanly, (c)
   the intention-tree-derived clusters look correct, (d) `intention_drift_score` behaves sensibly
   across steps.
5. Update `THEOREM_MATRIX_TREE_BUFFER.md`/this plan once that live test has run.
