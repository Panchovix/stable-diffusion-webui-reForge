# Plan 03: CLPC Token-Level Conditional-Space Guidance

## Status: Phases 1-5 implemented, UNTESTED against a real model (no torch in this sandbox)

- **Phase 1 (Lean)**: `lean_proofs_rfv/RFVProofs/TokenSubspaceGuidance.lean`, 10 theorems, 0 sorries,
  full project builds clean (2633 jobs). See `THEREM_CONDITIONAL_BUFFER.md §0`.
- **Phase 2**: `modules/prompt_parser.get_token_subspaces()` + `_split_top_level_commas()`.
- **Phase 3**: `ldm_patched/k_diffusion/sure_token_guidance.py` — persistent attn2 hook, vanish/leak/bias
  corrections. Mid-implementation, a synthetic numpy smoke test (see chat transcript / commit history)
  caught a real bug: naive per-group argmax "ownership" fails on adversarial leak cases (a leaked
  attribute can out-mass the rightful entity token). Fixed via anchor-keyword clustering
  (`_ANCHOR_KEYWORDS`, `_build_clusters`) — attribute groups inherit their nearest preceding
  entity-noun group's cluster for leak-correction purposes, verified against a synthetic
  "1girl, brown hair, 1boy, blue hair" case.
- **Phase 4**: `clpc_error.py` — `CLPCError.token_score`, `compute_gscore(..., token_score=1.0)`,
  `compute_token_guidance_score()`, `build_clpc_error(..., token_info=None)`. `clpc_sampler.py` reads
  an optional `token_guidance_store` from `model_options["transformer_options"]` and aggregates it
  once per step.
- **Phase 5**: `extensions-builtin/sd_forge_sure_token_ag/scripts/forge_sure_token_ag.py` — new WebUI
  accordion, disabled by default, mirroring `forge_sure_ag.py`'s UI conventions.

**NOT YET DONE**: no real-model/WebUI run (this sandbox has no `torch`; only `py_compile` and a
numpy-based logic mock of the correction math were run — see below). Treat this as a debuggable
first pass, not a verified-working feature, until run once in the actual WebUI.

### How to check it's working (debug prints to watch for)

- `[TokenSubspaceGuidance] prompt->groups: ...` — printed once per `process_before_every_sampling`
  call (Phase 2). Confirms the comma-groups and token-index ranges look right for your prompt.
- `[TokenSubspaceGuidance] installing attn2 hooks on blocks=...` — printed once when the extension
  patches the model (Phase 3 install).
- `[TokenSubspaceGuidance:<block><id>] groups=... clusters=... peak_mass=... vanishing=... leak_frac=...
  vanish_score=... leak_score=...` — printed on **every** attn2 hook call for the blocks you selected
  (default UI setting: `middle` only, to keep this to a handful of lines per step). `clusters` shows
  which attribute group got attached to which entity-anchor group — this is the line to check first
  for the "1girl, brown hair, 1boy, blue hair" case.
- `[TokenSubspaceGuidance:<block><id>] SKIP: Nk=... != expected ...` — means the prompt didn't fit the
  single-chunk assumption (Phase 2/3 limitation); guidance silently no-ops for that layer, by design.
- `[TokenSubspaceGuidance] step aggregate over N attn2 calls: {...}` — once per CLPC sampler step
  (Phase 4), only if using the CLPC sampler specifically.
- `[TokenSubspaceGuidance] build_clpc_error: token_info=... -> token_score=... gscore=...` — once per
  CLPC sampler step, confirms the 5th G-score factor is actually being folded in.

### Known limitations (by design, documented in code)

1. Correction applies to every batch row the attn2 layer sees, including CFG uncond/negative-prompt
   rows (no cheap way to know which rows are cond from inside a persistent per-layer hook).
2. Only single-chunk prompts (≤75 real tokens) are corrected; longer prompts print a warning and
   are skipped (`SKIP: Nk=...` line above). **Update (first real-world test, 2026-07-06)**: the
   original check required `Nk == chunk_length + 2` exactly and incorrectly skipped whenever Forge/
   A1111 padded the positive prompt's encoding to match a longer negative prompt's chunk count
   (observed `Nk=154` = 2×77 chunks with a short positive test prompt). Fixed: `Nk` is now accepted
   as any multiple of the 77-token chunk width, and correction is applied to the first chunk only
   (where the analyzed prompt's real content lives) — verified against the exact `Nk=154` shape from
   the bug report via a numpy mock. Prompts that *themselves* exceed 75 tokens are still skipped.
3. SDXL's dual CLIP-L/CLIP-G encoders may tokenize the same text to different lengths; only the
   primary tokenizer's offsets are used.
4. The tag-frequency table (`_EXAMPLE_TAG_FREQUENCY`) is a tiny illustrative placeholder, not a real
   corpus — bias correction is only demonstrably active for the ~8 tags in that table today.
5b. **Calibration bug (2026-07-06, 2nd real-world test)**: `tau_vanish` was a flat absolute mass
    constant (0.05); a 1-token group in a 154-token chunk-padded row has uniform baseline mass of
    only ~0.006, so 0.05 was actually a HIGH bar — the vanish-boost fired on almost every group
    almost every step (confirmed against the actual bug-report log: peak_mass=[0.0106, 0.0556,
    0.0273, 0.0205], all flagged "vanishing" against tau=0.05). This likely explains why the second
    test's image got worse (new "dashed hair" artifact) rather than better — noise was being
    injected into every group, not just genuinely neglected ones. Fixed: `tau_vanish` is now a
    multiplier of each group's fair share (`width/Nk`), default 0.4; re-checked against the same log
    data and none of the 4 groups would now trigger boost (correct — none of the tags are actually
    missing, this prompt is a pure leak/binding case). Also fixed a related diagnostic-only bug: the
    printed `vanishing=[...]` list used "any single attention head below threshold", which with 8+
    heads flags nearly everything regardless of the aggregate; it now uses the mean-across-heads
    peak instead.
5c. **Calibration bug (2026-07-06, 3rd real-world test)**: with `tau_vanish` fixed, `attn_blocks=all`
    correctly engaged many more layers (input4/5/7/8, middle0, output0-5), but `leak_frac` for
    "brown hair" stayed pinned around 0.5 across the ENTIRE run instead of trending down — the girl
    still rendered with mixed hair. Root cause: `own_cluster` forced EVERY query position (including
    background/ambiguous pixels where neither "1girl" nor "1boy" is really present) into whichever
    entity anchor happened to have marginally more mass, diluting the correction across roughly half
    the image that isn't either entity's actual region. Fixed: added a confidence gate
    (`leak_min_confidence`, default 0.3× the winning anchor's fair share) — a row only counts as
    confidently owned by an entity if that entity's own anchor token clears this bar; otherwise the
    row is left uncorrected (same as "unclustered"). Verified via a numpy mock with synthetic
    girl/boy/background rows: background rows are now correctly excluded (`confident=False`) while
    genuine entity rows are still corrected (leak mass 0.156→0.091 in the mock boy row). New debug
    field `confident_frac` shows what fraction of each layer's rows were confidently assigned — watch
    this on the next run; if it's still very high (most rows claimed by an entity), `leak_min_confidence`
    needs to go higher.
5d. **Result (2026-07-06, 4th real-world test)**: with the confidence gate in place, the same test
    prompt now renders with only a small, cleanly-bounded patch of the girl's hair off-color, instead
    of the diffuse full-image color mixing / identity swap seen in earlier runs. Diagnostics are
    healthy: `vanish_score=1.0` consistently (correctly inert — nothing is actually missing),
    `confident_frac` varies sensibly by layer/resolution (~0.55–0.99, no longer pinned near 1.0 or
    stuck at a single value), `leak_frac` for "brown hair" trending down in most layers vs. prior
    runs. Treated as a working baseline — some residual leak is expected and consistent with the
    literature this is based on (Attend-and-Excite/Divide&Bind/BindEdit all report *improved*, not
    perfect, attribute binding). Further tuning (higher `leak_strength`, adjusting
    `leak_min_confidence`) is optional polish from here, not bug-fixing.
5e. **CLPC corrector-feedback integration (2026-07-06, approved by user)**: conditional-space error
    (`token_err = 1 - P_token`) is now folded into the CLPC Kalman-gain's `P` (prediction
    uncertainty) term alongside the existing ODE embedded-pair error: `P = ode_err +
    TOKEN_KALMAN_WEIGHT * token_err` (weight 0.5, `clpc_sampler.py`). A step whose denoised estimate
    shows vanish/leak/bias defects is now treated as more uncertain, so the sampler leans on the
    corrector more heavily — literally "modeling the conditional-space error and guiding the sampler
    toward it," per the user's request. Formalized in `TokenSubspaceGuidance.lean` Part 7:
    `kalman_gain_with_token_term_valid` (K stays in [0,1]) and
    `token_term_increases_corrector_trust` (adding token_err never decreases corrector trust) — both
    0 sorry, full project builds clean (2633 jobs). No new blend-correctness proof was needed since
    `DoobSOC.kalman_blend_preserves_improvement`/`convex_blend_between` already hold for any K∈[0,1].
    Also added `CLPCWeights.token` (default 0.0, backward-compatible) so `composite()` can report
    conditional-space error in the progress display if a caller opts in. Numerically re-verified
    against the exact Lean guarantees (K∈[0,1], monotone in token_err) including the edge case of
    pure token badness with zero ODE/wavelet error (K correctly jumps 0.0→1.0). NOT yet run against
    the real WebUI — same caveat as the rest of this plan.
5f. **Wiring gap found (2026-07-06)**: the user correctly noticed the dedicated "CLPC Sampler
    Settings" extension (`extensions-builtin/sd_forge_clpc/scripts/forge_clpc.py`) had NO option or
    log referencing token-space guidance at all — it's a totally separate script from "SURE Token
    Subspace Guidance". Two real gaps: (1) `TOKEN_KALMAN_WEIGHT` was a hardcoded module constant in
    `clpc_sampler.py`, not exposed anywhere in the UI; (2) the one-time `token_guidance_store` log
    line only printed in the FOUND case — if the other extension was never enabled, `_clpc_loop`
    printed nothing at all about token guidance, making it look totally unwired even when working as
    designed. Fixed: `token_kalman_weight` is now a real parameter threaded through
    `sample_clpc_ode`/`sample_clpc_sde`/`_clpc_loop`/`_kalman_blend`, exposed as a new slider in the
    CLPC Sampler Settings accordion ("Token-space corrector feedback weight", default 0.5, explicitly
    labeled as requiring the separate extension). `_clpc_loop` now prints an unconditional
    `ENABLED`/`INACTIVE` status line every run regardless of whether the other extension is on, so the
    absence is as visible as the presence.
5g. **Real wiring bug found (2026-07-06)**: log showed the token-guidance extension successfully
    printing `installing attn2 hooks...` (patching succeeded), yet `_clpc_loop` immediately printed
    `INACTIVE`. Root cause traced through `modules/sd_samplers_kdiffusion.py` →
    `modules_forge/forge_sampler.py::forge_sample`: `_clpc_loop`'s local `model_options` variable is
    built from `extra_args.get("model_options", {})`, but `extra_args` (== A1111's
    `sampler_extra_args`) NEVER contains a `"model_options"` key at all — it only has
    `cond`/`uncond`/`image_cond`/`cond_scale`/`s_min_uncond`. `_install_cfg_drift_hook` was building
    its post_cfg hook into a dict that starts from `{}`, completely disconnected from the real model.
    The actual per-step forward pass (`forge_sample`) reads `model_options` straight off
    `self.inner_model.inner_model.forge_objects.unet.model_options` — the SAME object
    `sure_token_guidance.patch_model_with_token_guidance()` patches (which is why attn2 hooks
    correctly fired even though `_clpc_loop`'s own lookup couldn't see them). Fixed: added
    `_get_live_transformer_options(model)` which reads `model.inner_model.inner_model.forge_objects
    .unet.model_options` directly (mirroring `forge_sample`'s own attribute path with a safe
    `except AttributeError` fallback), and `_clpc_loop` now uses this for the `token_guidance_store`
    lookup instead of the disconnected `extra_args` copy. Not yet re-tested against the real WebUI —
    needs another run to confirm the ENABLED line now appears when both extensions are on.
5h. **Kalman/token-error decoupling (2026-07-07)**: two consecutive real live-WebUI runs (26 and 15
    CLPC steps) both showed the same pathology after 5e's three-term Kalman gain landed: `vanish_score`
    converged cleanly toward 1.0 and `token_score` improved monotonically over each run, but
    `leak_score` stayed essentially FLAT for the entire generation (~0.83 in one run, ~0.85 in another
    — never trending toward 1.0 the way every other per-step metric did), and the overall `gscore`
    plateaued rather than continuing to climb. Root cause: `P = ode_err + TOKEN_KALMAN_WEIGHT *
    token_err` mixes two signals with fundamentally different temporal behavior into one "prediction
    uncertainty" term. `ode_err` is a smoothly-evolving numerical-integration quantity — the Kalman
    filter's continuous-uncertainty assumptions genuinely fit it, and it decays toward 0 as sampling
    converges. `token_err` (1 - P_token, driven by vanish/leak/intention-drift diagnostics over
    discrete cross-attention patterns) has no such guarantee — real leak/vanish rarely reach exactly
    "perfect" for a whole generation, so `token_err` stayed small-but-nonzero (~0.02-0.05) for nearly
    every step, keeping `P` inflated and `K` pinned near 1.0 (full corrector trust) for most of both
    runs regardless of what the ODE numerics actually indicated. The user's framing: token/CLIP error
    should be "measured as its own [channel], no confidence on it since it may change suddenly" —
    exactly the mismatch above (Kalman's `P` implicitly assumes a decaying-uncertainty/confidence
    semantics that a discrete, possibly-abruptly-shifting diagnostic doesn't have).

    **Fix**: `_kalman_blend` (`clpc_sampler.py`) reverted to the plain two-term gain `K = ode_err /
    (ode_err + wav_hf_err)` — `token_err` no longer participates in `P`/`K` at all. It's still measured
    fresh every step (no persistent state carried between steps, matching "no confidence on it") and
    printed for visibility, but purely as a separate, un-blended, un-smoothed channel. The formerly
    dead `CLPCWeights.token` field (previously always 0.0 in practice, since neither
    `sample_clpc_ode`/`sample_clpc_sde` ever set it) is now wired to the existing
    `token_kalman_weight` UI slider instead, so that slider continues to have a real, honestly-labeled
    effect: weighting `(1 - P_token)`'s contribution to the composite `E`/G-score progress display —
    purely informational/monitoring (the `pid.propose_step(composite)` call downstream is explicitly
    monitoring-only, confirmed by its own inline comment, not step-size control), never the corrector-
    trust gain. UI slider relabeled "Token-space error weight (G-score, monitoring only)" with updated
    info text. The `TokenSubspaceGuidance.lean` Part 7 proof (`kalman_gain_with_token_term_valid`,
    `token_term_increases_corrector_trust`) is NOT invalidated by this — it proved the three-term
    formula was mathematically VALID (K stays in [0,1], never decreases trust), not that it was the
    empirically correct design choice for this signal; this is a case where a formally-verified-valid
    formula was still the wrong engineering decision once tested against a real model, and the fix is
    an implementation-level reversion, not a proof retraction.

    **Verified**: pure-Python re-derivation of both the new Kalman gain (confirmed `K` is now
    completely independent of `token_err` — identical output whether `token_err` is 0 or large, since
    it's no longer a parameter to the formula at all) and the new composite wiring (confirmed a
    near-zero `ode_err` with a realistic non-perfect `token_score=0.97` contributes only a small,
    proportionate ~0.01 to the composite `E`, not a dominant term) — both via `clpc_error.py`'s actual
    `CLPCWeights`/`CLPCError` dataclasses (no torch tensor ops involved in `composite()`, so this
    piece runs directly, unlike the attn2-hook tensor math elsewhere in this project that needs a real
    model to exercise). Not yet re-run against the live WebUI with this specific change — the next run
    should show `leak_score` (or at least `gscore`) trending upward for the whole generation instead of
    plateauing, and the kalman-blend debug line now reading "token/CLIP error channel (separate from
    Kalman blend)" instead of "kalman blend: ... token_err=... (weight=...)".
6. Entity "anchor" detection is an 8-keyword English substring heuristic
   (`girl/boy/woman/man/person/child/male/female`); prompts describing entities without one of these
   words fall back to raw per-group argmax ownership (the less robust heuristic the smoke test showed
   fails under heavy leakage).

## Companion research doc

`lean_proofs_rfv/THEREM_CONDITIONAL_BUFFER.md` — literature survey, codebase findings, and the
subspace-projection design synthesis this plan implements. Read it before starting any phase; it is
not repeated in full here.

---

## Goal

Add a 5th, token-level term `p_token = p_vanish · p_leak · p_bias` to the CLPC G-score, fixing three
attention pathologies:

1. **Vanish** — a prompt token gets near-zero attention anywhere (catastrophic neglect).
2. **Leak / conflict / masking** — one entity's tag attention bleeds into another entity's spatial
   region (`"1girl, brown hair, 1boy, blue hair"` → boy also renders brown hair).
3. **Bias** — a statistically common tag crowds out a co-occurring rarer tag within the same entity's
   tag group.

**Hard constraint: extra NFE ≈ 0.** Every correction must be a tensor operation on an attention matrix
already computed this step — no extra denoiser forward pass, no backprop-through-UNet gradient step
(ruling out the Attend-and-Excite / Divide&Bind optimization mechanisms, whose *diagnostics* we still
reuse — see buffer §3.1–3.2). **Order constraint: prove in Lean before touching the sampler.**

---

## Phase 0: Allowed APIs (confirmed by code reading, see buffer §2)

| API | File | Confirmed use |
|---|---|---|
| `patches_replace["attn1"][block][id]` hook, signature `(q, k, v, extra_options, mask=None) → out` | `ldm_patched/k_diffusion/sure_attention.py:54-146` | Existing SURE-AG entropy capture pattern to mirror — **but attn1 is self-attention (spatial↔spatial)**. Token guidance needs the analogous `attn2` (cross-attention, spatial↔text) slot. |
| `attention_basic_with_sim(...)` | `ldm_patched/contrib/nodes_sag.py` | Returns full post-softmax `sim` in fp32 alongside `out`; already proven safe under autocast (see git history: `af3f5f36`, `178dbf1f`, `79c1cd4e` — fp32 escape + operator-precedence fixes already landed for this exact call). |
| `set_model_options_patch_replace(...)` | `ldm_patched/modules/model_patcher.py` | Installs hooks; used identically for attn1 today. |
| `CLPCError` dataclass + `build_clpc_error()` | `ldm_patched/k_diffusion/clpc_error.py:54-69, 317-362` | Insertion point for `p_token` field. |
| `compute_gscore(cfg_err, sure_err, entropy_score, ot_score)` | `ldm_patched/k_diffusion/clpc_error.py:278-299` | Extend to 5 args. |
| `RFVProofs/GaussianProcessODE.lean` (`gscore_product_in_unit_interval`, `gscore_product_monotone`, `lyapunov_contraction`) | `lean_proofs_rfv/RFVProofs/GaussianProcessODE.lean` | Generalize from 4-factor to 5-factor product — same tactics (`positivity`, `linarith`, `gcongr`). |
| `RFVProofs/WaveletDomain.lean` (`subband_correction_independence`) | `lean_proofs_rfv/RFVProofs/WaveletDomain.lean` | Template for the disjoint-token-subspace independence theorem (subband → token-index partition). |
| Mathlib Hilbert-space projection (`starProjection_minimal` family, `Mathlib.Analysis.InnerProductSpace.Projection`) | flagged as unused-but-available in `THEOREM_BUFFER.md`'s Cross-Reference Audit | Nonexpansive-projection theorem for the leak correction. |

### Anti-patterns to avoid

- **Do not** hook `attn1` for token diagnostics — it is self-attention; its columns are spatial
  positions, not text tokens. Verify `patches_replace["attn2"]` exists for the target UNet block
  (`grep -n "attn2" ldm_patched/modules/model_patcher.py ldm_patched/ldm/modules/attention.py`) before
  building Phase 2 — do not assume the key name without confirming it in this codebase.
- **Do not** implement vanish/leak correction as a gradient step (Attend-and-Excite/Divide&Bind
  mechanism) — that costs extra backward passes. Only their *diagnostic definitions* (max-attention
  excitation score, cross-entity overlap) are reused.
- **Do not** invent a new prompt-parsing grammar. Comma-segment tokenization (Phase 1) must reuse the
  **same CLIP tokenizer instance** already used by `get_learned_conditioning()` — a second tokenizer
  or a hand-rolled BPE split will desync token indices from the real attention columns.
- **Do not** skip the Lean phase to save time — the user explicitly requires Lean-first validation
  before sampler code changes.

---

## Phase 1: Lean formalization (`lean_proofs_rfv/RFVProofs/TokenSubspaceGuidance.lean`)

### What to prove (copy proof *style* from `GaussianProcessODE.lean` / `WaveletDomain.lean`, don't invent new tactics)

1. `subspace_row_renormalized_is_distribution` — after boost (vanish) or null-projection (leak) or
   reweight (bias) and renormalization, the row is still nonnegative and sums to 1. Model the
   attention row abstractly as `f : Fin n → ℝ` with `∀ i, 0 ≤ f i` and `∑ f = 1`; corrected row is
   `f' i = f i / ∑ f` after a nonnegative perturbation — mirror the proof shape of
   `KalmanFilter.lean`'s `innovation_shrinks_scalar`.
2. `token_subspace_correction_independence` — direct restatement of
   `WaveletDomain.lean:subband_correction_independence` with the wavelet-subband index replaced by the
   token-group index `k : Fin m` over a partition `G : Fin m → Finset (Fin n)` with pairwise-disjoint
   `G k`. Correcting group `G_k`'s columns must not require touching `G_j`'s columns for `j ≠ k`.
3. `leak_projection_nonexpansive` — `‖proj_{S}ᗮ v‖ ≤ ‖v‖` for the leak correction, sourced from
   Mathlib's orthogonal-projection API (this is the sorry the Cross-Reference Audit in
   `THEOREM_BUFFER.md` already flagged as fixable — use it here instead of re-deriving from scratch).
4. `gscore5_product_in_unit_interval` and `gscore5_product_monotone` — generalize
   `GaussianProcessODE.lean`'s 4-factor theorems to
   `combined_score5 p_cfg p_sure p_ent p_ot p_token = p_cfg*p_sure*p_ent*p_ot*p_token`, same proof
   technique (chain of `mul_le_mul_of_nonneg_left` + `linarith`, or `gcongr`).
5. `lyapunov_rate_invariant_under_token_term` — adding the 5th factor changes only the Lyapunov
   constant, not the contraction rate `r` — same argument shape as `ProxySOCvsFull.lean`'s
   `proxy_same_rate_as_full_soc` (swap "proxy vs full SOC" for "4-factor vs 5-factor G-score").

### File registration

- Add `import RFVProofs.TokenSubspaceGuidance` to `lean_proofs_rfv/RFVProofs.lean` (after the
  `ProxySOCvsFull` import line) — no `lakefile.toml` changes needed (auto-discovered, per
  `THEOREM_BUFFER.md` conventions confirmed in Phase 0 discovery).
- `import RFVProofs.GaussianProcessODE` and `import RFVProofs.WaveletDomain` at the top, matching
  `DoobSOC.lean`'s cross-file-import style (`open GaussianProcessODE WaveletDomain`).

### Verification checklist

- [x] `lake build` from `lean_proofs_rfv/` completes with 0 errors (2633 jobs; pre-existing sorries
      in `WaveletDomain.lean`/`KLOptimality.lean`/`Straightness.lean`/`Perturbation.lean`/
      `LocalLinearity.lean` are unrelated and untouched).
- [x] `grep -n "sorry" RFVProofs/TokenSubspaceGuidance.lean` → 0 matches. The nonexpansive-projection
      theorem needed no sorry — `Submodule.norm_starProjection_apply_le` (found via `lean_leansearch`)
      matched exactly, closing the gap `THEOREM_BUFFER.md`'s Cross-Reference Audit had flagged.
- [x] `THEREM_CONDITIONAL_BUFFER.md §0` updated with the actual theorem list + sorry count (0/10).

**Note:** the `mcp__lean-lsp__*` tools could not reach `lake` (not on the MCP server's PATH); all
build/iteration in this session used `lake` directly via Bash with
`export PATH="$HOME/.elan/bin:$PATH"`. Future sessions should either fix the MCP server's PATH or
keep using this workaround.

---

## Phase 2: Token→subspace map (WebUI side, `modules/prompt_parser.py`)

### What to implement

A new function, e.g. `get_token_subspaces(prompt: str) -> list[tuple[int, int]]`, that:
1. Splits `prompt` on top-level commas (respecting existing `(...)`/`[...]` weight-syntax nesting —
   reuse whatever bracket-depth tracking `re_attention` already does, don't write a second parser).
2. Re-tokenizes each segment with the **same tokenizer object** `get_learned_conditioning` uses
   (locate it via the model's `FrozenCLIPEmbedder`/text-encoder wrapper already referenced in
   `prompt_parser.py`), and accumulates offsets to produce disjoint index ranges
   `[(start_0, end_0), (start_1, end_1), ...]` into the final token sequence (including the BOS
   offset).
3. Returns this alongside the existing `ScheduledPromptConditioning`/`MulticondLearnedConditioning`
   objects — do not replace them, append to the data the sampler already receives.

### Documentation references

- `modules/prompt_parser.py:211-269` (`get_multicond_prompt_list`) for the comma/AND-splitting pattern
  to follow structurally.
- `modules/prompt_parser.py:366-380` (`re_attention`) for bracket-depth-aware tokenization to copy.

### Verification checklist

- [ ] For prompt `"1girl, brown hair, 1boy, blue hair"`, print the returned ranges and confirm they
      partition `{0, …, N_tokens-1}` with no gaps/overlaps (modulo BOS/EOS/padding tokens, which
      should be excluded from all groups, not silently assigned to group 0).
- [ ] Confirm token count matches what the actual CLIP encoder produces for the same string (compare
      against `model.cond_stage_model.tokenize([prompt])` or equivalent already used elsewhere in this
      codebase).

---

## Phase 3: Cross-attention hook — diagnostics + correction (new file, e.g. `ldm_patched/k_diffusion/sure_token_guidance.py`)

### What to implement

Mirror `sure_attention.py`'s hook-construction pattern (`_make_entropy_hook`,
`_build_attn_capture_options`) but:
- Target `patches_replace["attn2"]` (cross-attention), not `attn1` — **verify this key exists** per
  the Phase 0 anti-pattern guard before writing the hook.
- Inside the hook, after computing post-softmax `sim` (same `attention_basic_with_sim` call as
  `sure_attention.py` uses, fp32, autocast-escaped per the already-landed fix in commit `178dbf1f`):
  1. Slice `sim` columns by the `Phase 2` token ranges.
  2. Compute the three diagnostics: per-group max attention mass (vanish score), cross-group overlap
     via the spatial support of each group's max-attention rows (leak score), within-group tag-
     frequency-weighted variance (bias score) — using the design in
     `THEREM_CONDITIONAL_BUFFER.md §4`.
  3. Apply the corrections (boost / null-project / reweight) directly to `sim`, renormalize each row,
     and compute `out = corrected_sim @ v` instead of passing through the uncorrected output —
     this is the one behavioral difference from the observe-only entropy hook.
  4. Store the three scalar diagnostic scores (pre-correction, for the G-score in Phase 5) in the same
     `store: list` side-channel pattern `sure_attention.py` uses.

### Documentation references

- `ldm_patched/k_diffusion/sure_attention.py:54-146` — hook shape, autocast/fp32 handling, store
  side-channel pattern.
- `THEREM_CONDITIONAL_BUFFER.md §4` — exact correction formulas (boost/null-project/reweight).
- Tag-frequency table: source from an existing static resource if one exists in-repo (`grep -ril
  "tag.*freq\|danbooru" --include=*.py --include=*.json .` before building a new one — don't hardcode
  a frequency table without checking for an existing tagger/autocomplete dataset already shipped with
  this WebUI, e.g. under `tags/` or the tag-autocomplete extension).

### Verification checklist

- [ ] Unit-test the hook in isolation (no full sampler run) with a synthetic `sim` tensor and known
      token ranges — confirm rows still sum to 1 after correction, confirm the three scores are in
      `[0,1]`.
- [ ] Confirm no extra forward pass: profile NFE count before/after (should be identical — same
      `sample_clpc_ode`/`sample_clpc_sde` step count as without this feature).

---

## Phase 4: G-score integration (`ldm_patched/k_diffusion/clpc_error.py`, `clpc_sampler.py`)

### What to implement

- Add `p_token` field to `CLPCError` (near existing fields, `~L54`).
- Extend `compute_gscore` (`~L278-299`) to `compute_gscore(cfg_err, sure_err, entropy_score, ot_score,
  token_score)` returning the 5-factor product, matching the Lean `combined_score5` proved in Phase 1
  exactly (same factor order, same clamp/`exp(-min(err*4,20))` treatment if `token_score` is expressed
  as an error rather than a already-in-[0,1] score — decide which based on what Phase 3 actually
  returns and keep it consistent with the Lean statement, not ad hoc).
- Extend `build_clpc_error()` (`~L317-362`) to accept the Phase 3 diagnostic scores and populate the
  new field — this must come from the *already-computed* hook output, not trigger a new computation.

### Verification checklist

- [ ] `compute_gscore` output stays in `(0,1]` for the new 5-argument call — add a quick assertion or
      test mirroring the Lean `gscore5_product_in_unit_interval` statement numerically.
- [ ] Existing 4-term callers (if any code path calls `compute_gscore` without token guidance enabled)
      must still work — default `token_score=1.0` (neutral factor) when the feature is off, not a
      breaking signature change.

---

## Phase 5: WebUI wiring (extension, mirroring `sd_forge_sure_ag`)

### What to implement

- New extension folder `extensions-builtin/sd_forge_sure_token_ag/scripts/forge_sure_token_ag.py`
  (copy the registration/UI pattern from `extensions-builtin/sd_forge_sure_ag/scripts/forge_sure_ag.py`
  — sliders for `τ_vanish`, boost `β`, leak-projection strength, bias-reweight strength; a checkbox to
  enable/disable, defaulting to **off** until Phase 1–4 verification is complete).
- Wire the extension's UI values through to `build_clpc_error()`/the Phase 3 hook via the same
  `model_options` plumbing `forge_sure_ag.py` already demonstrates.

### Verification checklist (manual, in-browser)

- [ ] Golden path: prompt `"1girl, brown hair, 1boy, blue hair"`, feature OFF vs ON, screenshot both —
      confirm ON reduces visible hair-color bleed onto the boy without the feature crashing or
      changing step count.
- [ ] Regression: run an existing SURE-AG/AGWAV prompt with the new feature OFF — confirm bit-identical
      output to pre-change baseline (the new hook must be fully inert when disabled).
- [ ] Vanish case: a prompt with 3+ entities where one commonly gets dropped (e.g. crowded prompt) —
      confirm the previously-missing entity's tag now shows measurable attention mass increase in a
      debug overlay/log.

---

## Final Phase: Verification

1. `cd lean_proofs_rfv && lake build` — 0 new errors, 0 new sorries beyond the 3 pre-existing ones.
2. `grep -rn "attn1" ldm_patched/k_diffusion/sure_token_guidance.py` — must return nothing (anti-
   pattern guard from Phase 0 — confirms the new hook targets `attn2`, not a copy-paste of `attn1`).
3. Confirm zero extra NFE: compare total denoiser call count for a fixed step count, feature on vs off
   (should be identical — log from `clpc_sampler.py`'s existing step counter).
4. Manual WebUI test per Phase 5's checklist.
5. Update `THEREM_CONDITIONAL_BUFFER.md` §1 status line to "implemented" and append final Lean
   theorem/sorry counts, following the dated-entry convention `THEOREM_BUFFER.md` already uses.
