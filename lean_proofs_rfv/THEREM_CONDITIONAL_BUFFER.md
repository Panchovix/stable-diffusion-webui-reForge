# THEREM_CONDITIONAL_BUFFER.md
# Research Buffer — Token-Level Conditional-Space Guidance for CLPC

**Date:** 2026-07-06
**Status:** Phase 1 (Lean) complete — `RFVProofs/TokenSubspaceGuidance.lean` builds clean, 0 sorries. Phase 2+ (WebUI/attention hook/sampler wiring) not started; see `plans/03-clpc-token-conditional-guidance.md`.
**Companion doc:** `lean_proofs_rfv/THEOREM_BUFFER.md` (prior CLPC sampler-decomposition + G-score work — this buffer extends that G-score with a 5th, token-level term).

## 0. Phase 1 Lean results (2026-07-06)

**File:** `RFVProofs/TokenSubspaceGuidance.lean` — registered in `RFVProofs.lean` after `ProxySOCvsFull`.
**Build:** `lake build` → 2633 jobs, 0 errors. New file: 0 sorries (pre-existing sorries in `WaveletDomain.lean`, `KLOptimality.lean` ×2, `Straightness.lean` ×2, `Perturbation.lean`, `LocalLinearity.lean` are unaffected/unrelated).

| # | Theorem | sorry? | Note |
|---|---------|--------|------|
| 1 | `renormalized_row_is_distribution` | 0 | nonneg row + positive sum → renormalized row is a valid distribution |
| 2 | `boost_preserves_nonneg` | 0 | vanish correction keeps row nonnegative (`if_pos`/`if_neg` case split) |
| 3 | `reweight_preserves_nonneg` | 0 | bias correction keeps row nonnegative (`mul_nonneg`) |
| 4 | `token_subspace_correction_independence` | 0 | direct reuse of `WaveletDomain.subband_correction_independence` |
| 5 | `leakCorrect` / `leak_projection_nonexpansive` | 0 | leak correction = `Kᗮ.starProjection v`; nonexpansive via Mathlib's `Submodule.norm_starProjection_apply_le` (found via `lean_leansearch`, no re-derivation needed) |
| 6 | `gscore5_product_in_unit_interval` | 0 | 5-factor product ∈ [0,1], reuses `GaussianProcessODE.gscore_product_in_unit_interval` rather than re-proving from scratch |
| 7 | `gscore5_product_monotone` | 0 | `gcongr` closes the 5-factor monotonicity directly, same as the 4-factor original |
| 8 | `token_term_le_one_discounts_gscore` / `badness_increases_with_token_term` | 0 | `p_token ≤ 1` can only discount F4, never inflate it |
| 9 | `lyapunov_rate_invariant_under_token_term` | 0 | direct reuse of `ProxySOCvsFull.full_soc_not_needed` — proves the 5th factor changes only the Lyapunov constant, not the contraction rate `r` |
| 10 | `subspace_correction_strictly_cheaper_than_gradient_based` / `composed_subspace_corrections_extraNFE` | 0 | zero-extra-NFE cost model: attention-matrix corrections cost `0`, any gradient-based mechanism costs `≥ 1`, composition stays at `0` |

**Key implementation notes for future phases:**
- `Submodule.norm_starProjection_apply_le` (Mathlib, `Analysis.InnerProductSpace.Projection.Basic`) was the exact lemma the prior `THEOREM_BUFFER.md` Cross-Reference Audit had flagged as "available but unused" — confirmed and used here with zero new axioms.
- Theorems 6, 7, 9 are *thin reuse wrappers* around existing 4-factor/proxy-vs-full theorems, not new proof techniques — exactly the low-effort generalization the design synthesis (§4 below) predicted.
- Theorem 10's cost model is explicitly bookkeeping/definitional (documented as such in the file), not a physical derivation — it makes the "train-free, zero extra NFE" claim checkable rather than aspirational, per the TFG design-space categorization (arXiv:2409.15761) found in the literature search.
- `Submodule.starProjection_add_starProjection_orthogonal` confirms Mathlib auto-derives a `Kᗮ.HasOrthogonalProjection` instance from `[K.HasOrthogonalProjection]`, so the leak-correction definition needed no extra instance plumbing.

---

## 1. Problem statement

The CLPC sampler currently guides on four *global scalar* error signals (CFG drift, SURE, spectral
entropy, OT gap → the G-score product `F(x) = P_cfg · P_sure · P_entropy · P_ot`). None of these see
*which token* is misbehaving. Three token-level pathologies are unaddressed:

1. **Vanish** — a prompt token receives near-zero attention mass anywhere in the image (the concept
   is dropped). Classic "catastrophic neglect."
2. **Conflict / leaking / masking** — a token's attention mass bleeds into image regions that belong
   to a *different* entity, e.g. `"1girl, brown hair, 1boy, blue hair"` → the boy's region also
   attends to "brown hair", so he renders with brown hair too.
3. **Bias toward common tags** — within one entity's own tag group, a statistically frequent tag
   absorbs disproportionate attention share relative to co-occurring rarer tags in the same group,
   crowding them out even when both are meant to co-apply.

The user's proposed unification: treat the token axis of the (already-computed) cross-attention
matrix as partitioned into **disjoint subspaces**, one per comma-separated entity/tag-group, and
solve each pathology as an operation *within or between* those subspaces (boost own-subspace mass →
vanish; project out cross-subspace overlap → leak; reweight within-subspace mass by inverse tag
frequency → bias). This buffer confirms that framing has direct precedent in the 2025–2026 literature
and is compatible with the existing Lean G-score machinery.

---

## 2. Codebase findings (Phase 0 discovery, via subagent code reading)

### 2.1 Attention hook — reusable, zero extra NFE

- [ldm_patched/k_diffusion/sure_attention.py](../ldm_patched/k_diffusion/sure_attention.py) already
  installs a `patches_replace["attn1"][block][id]` hook (`_make_entropy_hook`, ~L54-92) that receives
  `(q, k, v, extra_options, mask)`, recomputes the **full post-softmax attention matrix** in fp32 via
  `attention_basic_with_sim` (from `nodes_sag.py`), and returns the unmodified output. This capture
  happens on the *existing* forward pass — no extra model evaluation.
- `_build_attn_capture_options` (~L95-146) is the injection point; `_tokens_to_spatial` (~L259) maps
  attention query rows back to a `(h_lat, w_lat)` image grid, but only *spatially* — there is currently
  **no token→entity mapping**.
- **Implication:** vanish/leak/bias diagnostics can piggyback on this exact hook. The correction is a
  tensor op on an attention matrix that is already resident in memory — this is the key constraint
  the user set ("extra NFE as small as possible") and it is achievable exactly, not approximately.

### 2.2 Prompt parsing — the token→entity map does not exist yet (real gap)

- [modules/prompt_parser.py](../modules/prompt_parser.py) supports `AND`-separated composable prompts
  (`get_multicond_prompt_list`, ~L211-269) and `[a:b:step]` schedules, but both operate at the
  **embedding level** — by the time `get_learned_conditioning()` runs, individual CLIP token
  boundaries per comma-segment are gone.
- `re_attention` (~L366-380) parses `(word:weight)` syntax at the text level but doesn't retain a
  token-index span either.
- **Conclusion:** a new, small utility must re-tokenize each comma-segment with the same CLIP
  tokenizer used for the prompt and compute cumulative offsets to recover token-index ranges
  `G_1, …, G_m` (disjoint subsets of `{0, …, N_tokens-1}`). This is the standard technique used by
  regional-prompting / attention-couple extensions elsewhere; reForge has no such map today.

### 2.3 G-score — clean multiplicative product, one clear insertion point

- [ldm_patched/k_diffusion/clpc_error.py](../ldm_patched/k_diffusion/clpc_error.py) `compute_gscore`
  (~L278-299): `gscore = p_cfg · p_sure · entropy_score · ot_score ∈ (0,1]`. `CLPCError.composite()`
  (~L63-69) is a weighted sum of the four raw errors, evaluated every step from tensors already
  computed by the predictor/corrector (`x_adams`, `x_euler`, `x_corr`, `denoised_t`) — no extra NFE
  today, and none needed for a 5th term either.
- `build_clpc_error()` (~L317-362) is the natural place to add a `token_guidance_info` argument and a
  `p_token` field, computed as a scalar reduction over the attention-hook diagnostics captured in
  §2.1 (cheap: a handful of `max`/`sum`/`einsum` reductions, not a matmul-scale cost).
- Prior art already in this repo: `gscore_product_in_unit_interval` and `gscore_product_monotone`
  in `RFVProofs/GaussianProcessODE.lean` are stated generically enough that extending the product to
  5 factors is a direct, low-effort generalization, not a new proof technique.

### 2.4 Adjacent prior art already in-repo (reuse, don't reinvent)

- `extensions-builtin/sd_forge_ipadapter/lib_ipadapter/CrossAttentionPatch.py` `Attn2Replace` shows the
  same `(out, q, k, v, extra_options)` callback shape, but patches *after* attention — not suitable for
  the projection-based correction here, which must act on the attention matrix *before* the value
  aggregation (`softmax(QKᵀ)·V`).
  Composable-diffusion `AND` weighting and Tiled/MultiDiffusion regional generation both exist in-repo
  but neither does token-level attention masking. Confirmed genuine gap.

---

## 3. Literature survey (WebSearch, July 2026)

### 3.1 Vanish / catastrophic neglect

- **Attend-and-Excite** (Chefer et al., SIGGRAPH 2023, [arXiv:2301.13826](https://ar5iv.labs.arxiv.org/html/2301.13826)) —
  names "catastrophic neglect" precisely: a subject token gets negligible attention anywhere in the
  image. Fix is a per-step **latent optimization** (gradient ascent on max-attention-value loss) —
  **costs extra backprop-through-UNet per step**, i.e. NOT zero-NFE. Useful for the diagnostic
  definition (per-token max-attention excitation score) but the *optimization* mechanism is the wrong
  cost class for this project's NFE constraint — use the diagnostic, not the gradient-step fix.
- **Token Perturbation Guidance (TPG)** ([arXiv:2506.10036](https://arxiv.org/abs/2506.10036)) —
  training-free, CFG-style guidance via token shuffling; conceptually close to "perturb-and-compare"
  but implemented as **an extra forward branch** (extra NFE) unless folded into an existing CFG batch.
  Confirms perturbation-based vanish-guidance is a known technique class, but the reForge
  implementation should instead use the **already-captured** attention matrix (§2.1) rather than a new
  perturbed branch, to stay at zero extra NFE.

### 3.2 Conflict / leaking / masking (the "1girl…1boy…" problem)

- **BindEdit** ([arXiv:2606.18906](https://arxiv.org/html/2606.18906v1), 2026) — names two leakage modes
  directly analogous to the target bug: *Edit-Token Leakage* (ambiguous token-region alignment) and
  *Source Dominance Leakage* (one entity's tokens overwhelm another's attention). Fix: "Attention
  Binding Guidance" — semantic-level binding via cross-attention + instance-level isolation via
  self-attention. This is architecturally the closest match to the requested feature.
- **ALE-Edit** ([arXiv:2412.04715](https://arxiv.org/pdf/2412.04715)) — Object-Restricted Embeddings
  (ORE) to localize per-object attributes in the text embedding, and Region-Guided Blending for
  Cross-Attention Masking (RGB-CAM) to align attention with the target region. RGB-CAM is directly a
  "mask the attention matrix by region/entity" operation — same cost class as what's proposed here.
- **Divide & Bind** ([arXiv:2307.10864](https://arxiv.org/abs/2307.10864)) — Jensen-Shannon-divergence
  loss forcing an attribute's attention map to overlap its own noun and stay **disjoint** from other
  entities' maps. This is the leakage-suppression objective stated as a *distributional* disjointness
  constraint — mathematically the continuous analogue of the discrete subspace-orthogonality framing
  the user proposed. NOTE: original implementation optimizes this via a gradient loss (extra
  backprop), so again: reuse the *objective* (disjointness / low cross-entity overlap), implement the
  *mechanism* as a direct projection on the captured attention matrix, not a gradient step.
- **Focused Cross-Attention (FCA)** ([arXiv:2404.13766](https://arxiv.org/pdf/2404.13766)) — uses
  syntactic parsing to *restrict* an attribute token's attention to the region of its governing object
  noun. Directly supports building `G_1, …, G_m` from comma/AND syntax (§2.2) rather than requiring a
  full dependency parser — comma-separated danbooru-style tag lists are an easier case than free-form
  sentences.

### 3.3 Bias toward common tags

- **Rare-to-Frequent (R2F)** ([OpenReview](https://openreview.net/forum?id=BgxsmpVoOX)) — LLM-guided
  rare-concept composition; effective but requires an LLM in the loop, too heavy for an inference-time
  per-step guidance term.
- **RAIGen** ([arXiv:2602.06806](https://arxiv.org/html/2602.06806)) — rare-attribute identification;
  useful for offline tag-frequency table construction, not a runtime mechanism.
- **Attention Frequency Modulation** ([arXiv:2603.28114](https://arxiv.org/pdf/2603.28114)) —
  training-free **spectral** modulation of cross-attention, i.e. a zero-extra-NFE reweighting of
  existing attention by frequency-domain statistics. Directly analogous to the existing SURE-AGWAV
  wavelet subband weighting already in this repo (`ldm_patched/k_diffusion/sure_wav_ag.py`); confirms
  that a within-group inverse-frequency reweighting of attention columns (no perturbation branch, no
  backprop) is a validated technique class for tag-frequency bias correction.

### 3.4 Subspace / projection formalization (directly supports the user's framing)

- **Refine and Purify — Orthogonal Basis Optimization with Null-Space Denoising (NSDP)**
  ([arXiv:2602.05464](https://arxiv.org/abs/2602.05464)) — constructs orthogonal semantic bases via
  SVD and suppresses "semantic leakage" by **projecting embeddings onto the null space of irrelevant
  subspaces**. This is the closest literature match to "solve leakage via subspace projection" —
  vocabulary and mechanism both transfer directly: define per-entity subspace bases from the token
  partition `G_k`, and null-space-project one entity's attention row away from another entity's
  region-defining subspace.
- **Classifier-Free Projection Guidance (CFPG)** — disentangles guidance into parallel/orthogonal
  components relative to the diffusion text-embedding geometry. Same subspace-decomposition idea
  applied to *guidance vectors* rather than attention maps — useful as the Lean-level abstraction
  (guidance correction = orthogonal projection operator, proven nonexpansive) that both the embedding-
  level and attention-level implementations can share.
- **Compositional Visual Generation with Composable Diffusion Models** (Liu et al., ECCV 2022) — the
  foundational statement that conditioning subspaces can be treated as independent, composable energy
  terms. **Cost caveat:** the original method literally re-runs the denoiser once per component
  (N extra NFEs for N entities). The reForge cross-attention already has all entities' keys/values in
  a single forward pass (they're concatenated into one text-embedding sequence), so the same
  independence property can be exploited **without** the N-way NFE cost the original paper pays —
  this is the concrete reason the "subspace projection on the existing attention matrix" design beats
  literal composable diffusion for this use case.

### 3.5 Cost-class summary (why "zero extra NFE" is achievable, not just aspirational)

| Technique class | Mechanism | Extra NFE? |
|---|---|---|
| Attend-and-Excite, Divide&Bind (as published) | gradient step on attention loss | Yes — 1+ backprop/step |
| TPG, PAG/SAG (as published) | extra perturbed forward branch | Yes — usually 1 extra branch |
| R2F | LLM-in-the-loop planning | Yes — LLM calls |
| **RGB-CAM (ALE-Edit), NSDP, Attention-Frequency-Modulation, this repo's SURE-AG entropy hook** | direct algebraic op (mask/project/reweight) on an attention matrix already computed this step | **No** |

The plan (§ next document) adopts the last row's cost class exclusively: every correction is a
deterministic tensor operation (renormalizing softmax slice, orthogonal projection, inverse-frequency
reweight) applied to the attention matrix the SURE-AG hook already captures.

---

## 4. Design synthesis — token subspace guidance

Given token-index partition `G_1, …, G_m` (from §2.2) and captured attention row `A[i, :]` for image
query position `i` (from §2.1):

1. **Vanish correction** — let `own(i)` be the group whose subject-noun token has max attention at
   row `i`. If `Σ_{j∈G_own(i)} A[i,j] < τ_vanish`, boost: `A'[i,j] = A[i,j] + β·𝟙[j∈G_own(i)]`, then
   renormalize the row. This is a projection **onto** the own-subspace.
2. **Leak correction** — for each *other* group `G_k ≠ own(i)`, project the attention mass in `G_k`
   away from the component that overlaps `own(i)`'s spatial support: `A'[i, G_k] = A[i,G_k] − proj(A[i,G_k] onto own(i)-support)`, then renormalize. This is a **null-space projection**, matching NSDP.
3. **Bias correction** — within one group `G_k`, reweight columns by a precomputed inverse tag-
   frequency prior before renormalizing, so a common tag's raw attention share doesn't crowd out a
   co-occurring rare tag in the same group. This is a **within-subspace reweighting**, matching
   Attention-Frequency-Modulation's mechanism (applied per-tag rather than per-frequency-band).

All three are row-wise operations on an already-in-memory `[N_query, N_key]` matrix — O(N_query·N_key)
per step, negligible next to one UNet forward pass, and require **zero additional denoiser
evaluations**.

### Lean formalization targets (for `plans/03-clpc-token-conditional-guidance.md` Phase 1)

1. Renormalized row after boost/projection/reweight is still a valid probability distribution
   (nonneg, sums to 1) — parallels `KalmanFilter.lean`'s "innovation shrinks" style bound.
2. Disjoint-subspace corrections don't interfere with each other — direct restatement of
   `WaveletDomain.lean`'s `subband_correction_independence` with token-index partition subbing for
   frequency subband.
3. Orthogonal (null-space) projection is nonexpansive — `‖proj(v)‖ ≤ ‖v‖` — available directly from
   Mathlib's Hilbert-space projection API, already flagged as an *unused* available lemma family in
   `THEOREM_BUFFER.md §"Cross-Reference Audit"` ("MMSE / orthogonal projection... Mathlib has the
   underpinning; sorry can be fixed").
4. 5-factor G-score product generalization of `gscore_product_in_unit_interval` /
   `gscore_product_monotone` (`GaussianProcessODE.lean`) — adding `p_token = p_vanish · p_leak · p_bias`
   preserves the `[0,1]` bound and the monotone-improvement property, so the existing Lyapunov
   contraction certificate (`lyapunov_contraction`) carries over unchanged in *rate*, only the
   constant changes (same argument pattern as `ProxySOCvsFull.lean`).

---

## 5. Source bibliography

### Vanish / catastrophic neglect
- [Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models (SIGGRAPH 2023)](https://ar5iv.labs.arxiv.org/html/2301.13826)
- [Token Perturbation Guidance for Diffusion Models (2025)](https://arxiv.org/abs/2506.10036)
- [Disrupting Diffusion: Token-Level Attention Erasure Attack (2024)](https://arxiv.org/pdf/2405.20584)
- [PainterNet: Adaptive Image Inpainting with Actual-Token Attention (2024)](https://arxiv.org/pdf/2412.01223)

### Conflict / leaking / attribute binding
- [BindEdit: Taming Attention Leakage for Precise Multi-Object Image Editing (2026)](https://arxiv.org/html/2606.18906v1)
- [Addressing Attribute Leakages in Diffusion-based Image Editing without Training (2024)](https://arxiv.org/html/2412.04715v2)
- [Object-Attribute Binding in Text-to-Image Generation: Evaluation and Control (2024)](https://arxiv.org/pdf/2404.13766)
- [Divide & Bind Your Attention for Improved Generative Semantic Nursing (2023)](https://arxiv.org/abs/2307.10864)
- [Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment (NeurIPS 2023)](https://arxiv.org/html/2306.08877v3)
- [Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis (2022)](https://arxiv.org/pdf/2212.05032)
- [Diffusion Self-Guidance for Controllable Image Generation (2023)](https://arxiv.org/pdf/2306.00986)
- [LocInv: Localization-aware Inversion for Text-Guided Image Editing (2024)](https://arxiv.org/pdf/2405.01496)

### Common-tag / rare-concept bias
- [Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance](https://openreview.net/forum?id=BgxsmpVoOX)
- [RAIGen: Rare Attribute Identification in Text-to-Image Generative Models (2026)](https://arxiv.org/html/2602.06806)
- [Generating images of rare concepts using pre-trained diffusion models (2023)](https://arxiv.org/html/2304.14530v3)
- [Attention Frequency Modulation: Training-Free Spectral Modulation of Diffusion Cross-Attention (2026)](https://arxiv.org/pdf/2603.28114)
- [Debiasing Text-to-Image Diffusion Models (2024)](https://arxiv.org/html/2402.14577v1)

### Subspace / null-space / projection guidance
- [Refine and Purify: Orthogonal Basis Optimization with Null-Space Denoising for Conditional Representation Learning (2026)](https://arxiv.org/abs/2602.05464)
- [Compositional Visual Generation with Composable Diffusion Models (ECCV 2022)](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)
- [How Diffusion Models Learn to Factorize and Compose (NeurIPS 2024)](https://arxiv.org/html/2408.13256v1)
- [Compositional Image Decomposition with Diffusion Models (2024)](https://arxiv.org/html/2406.19298v1)

### Zero/low-extra-NFE guidance mechanisms (cost-class precedent)
- [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (2024)](https://arxiv.org/abs/2403.17377)
- [Improving Sample Quality of Diffusion Models Using Self-Attention Guidance (2022)](https://arxiv.org/pdf/2210.00939)
- [Gradient-Free Classifier Guidance for Diffusion Model Sampling (2024)](https://arxiv.org/pdf/2411.15393)

### Codebase (this repo, read directly, not web)
- `ldm_patched/k_diffusion/sure_attention.py` — attention capture hook (reused, not rewritten)
- `ldm_patched/k_diffusion/sure_wav_ag.py` — precedent for per-subband/frequency reweighting mechanism
- `modules/prompt_parser.py` — prompt structure; confirmed token→entity map gap
- `ldm_patched/k_diffusion/clpc_error.py`, `ldm_patched/k_diffusion/clpc_sampler.py` — G-score + CLPC loop
- `extensions-builtin/sd_forge_ipadapter/lib_ipadapter/CrossAttentionPatch.py` — adjacent hook pattern (post-attention, not reused directly)
- `lean_proofs_rfv/THEOREM_BUFFER.md` — prior G-score/Doob-SOC/wavelet Lean work this buffer extends
- `lean_proofs_rfv/RFVProofs/GaussianProcessODE.lean`, `DoobSOC.lean`, `WaveletDomain.lean`,
  `KalmanFilter.lean` — Lean style/conventions and reusable lemmas (product-in-unit-interval,
  subband independence, projection/MMSE groundwork)
