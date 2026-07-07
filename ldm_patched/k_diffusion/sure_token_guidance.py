"""SURE Token Subspace Guidance: token-level conditional-space corrections.

Formal backing: lean_proofs_rfv/RFVProofs/TokenSubspaceGuidance.lean (0 sorry).
Design + literature: lean_proofs_rfv/THEREM_CONDITIONAL_BUFFER.md.
Plan: plans/03-clpc-token-conditional-guidance.md.

Fixes three cross-attention pathologies by treating the token axis as a
partition into disjoint per-entity subspaces `G_1, ..., G_m` (one per
comma-separated prompt group, see modules.prompt_parser.get_token_subspaces):

  * Vanish  — a group's attention mass is boosted when its peak-anywhere
    mass falls below `tau_vanish` (catastrophic-neglect diagnostic, cf.
    Attend-and-Excite arXiv:2301.13826).
  * Leak    — a rival group's mass is attenuated wherever another group
    already dominates the query position (cf. BindEdit arXiv:2606.18906,
    RGB-CAM/ALE-Edit arXiv:2412.04715).
  * Bias    — groups are boosted inversely to a static tag-frequency prior,
    so a common co-occurring tag doesn't crowd out a rarer one.

**Zero extra NFE**: unlike SURE-AG (nodes_sure_ag.py), which replays a whole
extra UNet forward pass, this module is installed as a *persistent* attn2
replacement via `ModelPatcher.set_model_attn2_replace` — it corrects the
attention matrix the sampler's own single forward pass already computes.
See TokenSubspaceGuidance.lean Part 6 for the formal zero-extra-NFE cost
model this design satisfies.

KNOWN LIMITATION (see prompt_parser.get_token_subspaces docstring): the
correction is applied uniformly to every batch row the attn2 layer sees,
including any CFG uncond/negative-prompt rows — there is no cheap way for a
persistent per-layer hook to know which batch rows are the positive prompt
without extra plumbing from the sampler. For typical positive/negative
prompt pairs this is low-risk (negative prompts rarely have the kind of
entity structure this guidance targets) but is not a formal guarantee.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ldm_patched.k_diffusion.sure_attention import _tokens_to_spatial

_logger = logging.getLogger("sure_token_guidance")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TokenGuidanceConfig:
    # tau_vanish is a MULTIPLIER of each group's "fair share" of the row
    # (group_width / Nk), NOT a raw attention-mass value — a 1-token group's
    # fair share is naturally far smaller than a 3-token group's, and both
    # shrink as the context gets longer (e.g. 154-token chunk-padded rows).
    # A group is "vanishing" if its peak-anywhere mass is below
    # tau_vanish * fair_share. Calibration note: an earlier version used a
    # flat absolute 0.05 here, which (for a 1-token group in a 154-token row,
    # fair_share≈0.006) was actually a HIGH bar that fired on almost every
    # group almost every step in real testing — see
    # plans/03-clpc-token-conditional-guidance.md's calibration log.
    tau_vanish: float = 0.4
    beta_vanish: float = 0.15      # additive boost strength for vanishing groups
    leak_strength: float = 0.5     # fraction of a rival group's mass removed outside its own region
    bias_strength: float = 0.05    # additive boost strength from the inverse-frequency prior
    # Multiplier of the winning anchor's fair share (width/Nk) its own mass must
    # clear before a query row counts as "confidently" belonging to that entity.
    # Rows that don't clear this bar (background / ambiguous / no entity really
    # present) are left uncorrected instead of being forced into a cluster —
    # see the confidence-gate note above _apply_token_subspace_corrections.
    leak_min_confidence: float = 0.3
    # Softmax temperature for ownership assignment (replaces a hard argmax +
    # leak_min_confidence threshold — see cluster_step_not_lipschitz,
    # lean_proofs_rfv/RFVProofs/TokenAvoidSOC.lean: the hard version is
    # provably discontinuous at ownership ties and at the confidence floor,
    # so leak attenuation for a rival group could flip between 0 and
    # leak_strength for an arbitrarily small mass change). Lower = closer to
    # the old hard argmax (sharper, more discontinuous); higher = softer,
    # more gradual ownership near ties/the confidence floor. Same
    # illustrative-default caveat as the other constants here — not yet
    # calibrated against a live model, see plans/03-clpc-token-conditional-
    # guidance.md's calibration log for how tau_vanish/leak_strength were.
    leak_ownership_temperature: float = 0.5
    debug: bool = True             # print per-call diagnostics


# Placeholder inverse-tag-frequency table (NOT a real dataset — a small
# illustrative example so this feature is testable end to end; a production
# deployment should source this from e.g. the tag-autocomplete extension's
# Danbooru frequency CSVs, per plans/03-clpc-token-conditional-guidance.md
# Phase 3's documentation-reference note).
_EXAMPLE_TAG_FREQUENCY = {
    "1girl": 0.98, "1boy": 0.9, "solo": 0.85, "2girls": 0.6, "2boys": 0.5,
    "masterpiece": 0.95, "best quality": 0.95, "highres": 0.9,
}


def _lookup_tag_frequency(text: str, table: dict) -> float:
    """Frequency in [0,1] for a comma-segment's bare text (1.0 = ubiquitous,
    0.0 = never seen). Unknown tags default to 0.5 (neutral — neither
    boosted nor suppressed) since the example table is illustrative, not a
    real corpus."""
    key = text.strip().lower()
    return float(table.get(key, 0.5))


# Entity-anchor keywords: comma-groups containing one of these are treated as
# "owning" a spatial region (rather than being an attribute that leaks). This
# is what makes leak correction robust to *already-leaked* attention: an
# adversarial case where a leaked attribute (e.g. "brown hair" bleeding onto
# the boy) has MORE raw attention mass in that region than the boy's own
# "1boy" token would break naive per-group argmax ownership (it would pick
# "brown hair", the very thing that's leaking, as the "owner"). Restricting
# ownership votes to anchor groups only, and clustering trailing attribute
# groups to their nearest preceding anchor (the standard booru-tag reading
# order convention: "entity, attr, attr, entity2, attr, attr"), fixes this —
# verified against a synthetic "1girl, brown hair, 1boy, blue hair" leak case
# during implementation (see plans/03-clpc-token-conditional-guidance.md).
_ANCHOR_KEYWORDS = ("girl", "boy", "woman", "man", "person", "child", "male", "female")


def _is_anchor_group(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in _ANCHOR_KEYWORDS)


def _build_clusters(ranges: list[tuple[int, int, str]]) -> list[int]:
    """cluster_of[gi] = index of the nearest preceding anchor group (an
    anchor's own cluster id is its own index). Groups before any anchor get
    cluster -1 (unclustered — excluded from leak correction, since there's
    no entity yet to attribute them to).

    This is the Plan 03 keyword-heuristic fallback, used only when the
    Plan 04 rule-engine/Matrix-Tree "intention tree" (see
    `_build_clusters_from_intention_tree` below) is unavailable — e.g. the
    rule engine failed, or the caller didn't wire it up. When available, the
    intention tree is strictly more accurate (handles unknown tokens,
    natural-language phrasing, and semantic disambiguation the 8-keyword
    substring match cannot), so it always takes priority — see
    `_apply_token_subspace_corrections`'s `final_tokens` parameter.
    """
    cluster_of = []
    current = -1
    for gi, (_s, _e, text) in enumerate(ranges):
        if _is_anchor_group(text):
            current = gi
        cluster_of.append(current)
    return cluster_of


def _build_clusters_from_intention_tree(ranges: list, final_tokens: list) -> tuple:
    """Derive (cluster_of, confidence) from the Plan 04 rule-engine/Matrix-Tree
    pipeline's reconciled output (`modules.rule_engine.refine_engine.refine`)
    instead of the flat keyword heuristic.

    `final_tokens[gi]` is a `FinalTokenInfo` (see refine_engine.py) whose
    `final_parent` is one of {GLOBAL_NODE(-1), ROOT(0), an OBJECT segment's
    1-indexed position} (the refine engine's own reconsideration rules
    guarantee this), so the translation to this module's `cluster_of`
    convention (-1 = unclustered/global/meta, otherwise the index of the
    owning entity group) is direct:
      - `excluded` (META)                  -> cluster -1
      - "OBJECT" in final_classes           -> its own cluster (an entity
                                                defines its own region)
      - final_parent == ROOT or < 0         -> cluster -1 (global/unclaimed —
                                                the latter covers refine_engine's
                                                GLOBAL_NODE, a GLOBAL/CONCEPT
                                                whole-image tag reconsidered
                                                away from any specific entity)
      - otherwise                           -> cluster = final_parent - 1
                                               (the owning entity's group index)

    `confidence[gi]` = the refine engine's own `parent_confidence` (the
    Matrix-Tree marginal probability of the chosen assignment) — used to
    weight how much a violation of this assignment should count as
    "intention drift" (see the diagnostics section of
    `_apply_token_subspace_corrections`): a confidently-wrong observation is
    more informative than an uncertain one.

    Also returns `weak_cluster_links[gi]`: `ft.weak_links` (see
    refine_engine.py's `FinalTokenInfo` — other OBJECT/PRIMARY_OBJ segments
    this description plausibly ALSO belongs to, e.g. "matching outfits" with
    two near-equally-likely owners) translated from `final_parent`'s
    1-indexed space into this module's 0-indexed `cluster_of`/`gi` space,
    for the leak-attenuation loop below to treat as non-rival groups too.
    """
    cluster_of = []
    confidence = []
    weak_cluster_links = []
    for gi in range(len(ranges)):
        ft = final_tokens[gi]
        if ft.excluded:
            cluster_of.append(-1)
        elif "OBJECT" in ft.final_classes:
            cluster_of.append(gi)
        elif ft.final_parent <= 0:  # ROOT(0) or GLOBAL_NODE(-1, or any future negative sentinel)
            cluster_of.append(-1)
        else:
            cluster_of.append(ft.final_parent - 1)
        confidence.append(float(ft.parent_confidence))
        weak_cluster_links.append(frozenset(wl - 1 for wl in getattr(ft, "weak_links", ()) if wl > 0))
    return cluster_of, confidence, weak_cluster_links


# ---------------------------------------------------------------------------
# Correction core
# ---------------------------------------------------------------------------

def _apply_token_subspace_corrections(
    sim: torch.Tensor,
    groups: list[dict],
    n_chunks: int,
    chunk_length: int,
    cfg: TokenGuidanceConfig,
    diag_store: Optional[list],
    layer_tag: str,
    final_tokens: Optional[list] = None,
    spatial_store: Optional[list] = None,
    heads: int = 1,
    orig_shape: Optional[list] = None,
) -> torch.Tensor:
    """sim: (BH, Nq, Nk) post-softmax attention, each row already sums to 1.

    final_tokens: the Plan 04 rule-engine/Matrix-Tree pipeline's reconciled
    output (list[FinalTokenInfo], one per `groups` entry, same order — see
    `modules.rule_engine.refine_engine.refine`). When provided, clustering
    and per-group confidence come from the "intention tree" instead of the
    flat 8-keyword heuristic (`_build_clusters_from_intention_tree` instead
    of `_build_clusters`) — this is what lets the correction understand
    unknown tokens, natural-language phrasing, and semantically-disambiguated
    attribute binding the keyword heuristic cannot. Falls back to the
    keyword heuristic (uniform confidence 1.0) if omitted, e.g. when the
    rule-engine extension isn't enabled or failed for this generation.

    spatial_store, heads, orig_shape: when `spatial_store` is provided, the
    per-query-position LEAK intensity (the only one of the three corrections
    that varies across query positions — vanish and bias are applied
    uniformly across the whole row, see the vanish/bias loops below) is
    appended as a raw `(leak_intensity, heads, orig_shape)` tuple, mirroring
    `sure_attention._make_entropy_hook`'s capture convention exactly so
    `aggregate_token_spatial_map` below can reuse the same
    tokens-to-spatial-grid + bilinear-upsample recipe
    (`sure_attention._aggregate_entropy_map`) to bring it into the SAME
    pixel/latent grid CLPC's own Haar wavelet error already operates on —
    this is what lets CLPC treat token guidance as a genuine per-pixel
    signal instead of a single scalar averaged over the whole image.

    Returns the corrected (renormalized) sim, or `sim` unchanged if the
    groups don't line up with this layer's actual token axis (safe no-op
    fallback — never miscorrects silently).
    """
    if not groups or n_chunks != 1:
        return sim

    Nk = sim.shape[-1]
    chunk_width = chunk_length + 2  # BOS + chunk_length content/pad slots + trailing EOS slot (e.g. 77)
    # Nk is often a MULTIPLE of chunk_width, not exactly one chunk: Forge/A1111
    # pad the shorter of cond/uncond to match the other's chunk count (e.g. a
    # long negative prompt needing 2 chunks forces the positive prompt's
    # encoding to also span 2 chunks = 154 tokens, even though its own content
    # fits in the first). Our analyzed prompt's real content always lives in
    # the first chunk, so correct that chunk only and leave any padding chunks
    # beyond it untouched (they still count toward the row-sum renormalization
    # below, so the output stays a valid distribution over the FULL row).
    if chunk_width <= 0 or Nk % chunk_width != 0:
        if cfg.debug:
            print(f"[TokenSubspaceGuidance:{layer_tag}] SKIP: Nk={Nk} is not a multiple of "
                  f"chunk_width={chunk_width} (chunk_length={chunk_length}); attention passed "
                  f"through unmodified for this layer.")
        return sim

    n_actual_chunks = Nk // chunk_width
    if n_actual_chunks > 1 and cfg.debug:
        print(f"[TokenSubspaceGuidance:{layer_tag}] Nk={Nk} spans {n_actual_chunks} chunks "
              f"(likely padded to match a longer negative prompt) — correcting only the first "
              f"{chunk_width}-token chunk, where this prompt's real content lives.")

    bos = 1  # shift past the leading BOS column of the first chunk
    if final_tokens is not None and len(final_tokens) == len(groups):
        kept = [(g, ft) for g, ft in zip(groups, final_tokens) if g["end"] + bos <= chunk_width]
        ranges = [(g["start"] + bos, g["end"] + bos, g["text"]) for g, _ft in kept]
        final_tokens_in_range = [ft for _g, ft in kept]
    else:
        ranges = [(g["start"] + bos, g["end"] + bos, g["text"]) for g in groups
                  if g["end"] + bos <= chunk_width]
        final_tokens_in_range = None
    if not ranges:
        return sim

    group_mass = torch.stack([sim[..., s:e].sum(-1) for (s, e, _t) in ranges], dim=-1)  # (BH, Nq, G)
    peak_mass = group_mass.max(dim=1).values          # (BH, G) — vanish diagnostic

    # fair_share[g] = what group g would get under pure uniform attention. Used
    # to make both the vanish threshold and the leak-ownership confidence gate
    # below scale sensibly with group width / context length instead of being
    # raw magic constants (see calibration notes in
    # plans/03-clpc-token-conditional-guidance.md).
    fair_share = torch.tensor(
        [(e - s) / Nk for (s, e, _t) in ranges], device=sim.device, dtype=sim.dtype,
    )  # (G,)

    # Ownership is decided by ANCHOR groups only (see _build_clusters docstring
    # for why: letting an already-leaked attribute vote for its own ownership
    # is circular and breaks exactly on the adversarial case this feature
    # targets). Non-anchor groups inherit their nearest preceding anchor's
    # cluster and are corrected relative to that cluster's ownership.
    #
    # SOFT OWNERSHIP: a hard argmax-over-anchors plus a hard confidence
    # threshold (the previous design) is a genuine step function of the
    # attention masses — cluster_step_not_lipschitz
    # (lean_proofs_rfv/RFVProofs/TokenAvoidSOC.lean) proves such a construction
    # is NOT Lipschitz for any constant: an arbitrarily small mass change
    # near an ownership tie, or near the confidence floor, flips a rival
    # group's leak attenuation between 0 and full `leak_strength`. Replaced
    # with a temperature-scaled softmax over anchor masses (normalized by
    # each anchor's fair share, same motivation the old confidence gate
    # had) PLUS a "background / no confident owner" option competing in the
    # SAME softmax, whose logit is `leak_min_confidence` — the identical
    # value the old hard gate compared against, now a genuine alternative
    # instead of a discontinuous cutoff. This reproduces both things the
    # old design got right (background/ambiguous rows stay largely
    # uncorrected; a clear single-anchor winner gets full ownership) while
    # making the transition between them continuous — real-world testing
    # (plans/03-clpc-token-conditional-guidance.md) is what motivated the
    # original hard gate's existence and threshold value; that calibration
    # carries over to `leak_min_confidence` unchanged, only the sharpness of
    # the transition around it is now a separate, tunable knob
    # (`leak_ownership_temperature`).
    #
    # Plan 04: prefer the rule-engine/Matrix-Tree "intention tree" over the
    # flat keyword heuristic whenever it's available — see
    # _build_clusters_from_intention_tree's docstring for why this is
    # strictly more accurate. `intention_confidence` (None when falling back
    # to the heuristic) feeds the intention-drift diagnostic below.
    if final_tokens_in_range is not None:
        cluster_of, intention_confidence, weak_cluster_links = _build_clusters_from_intention_tree(
            ranges, final_tokens_in_range)
    else:
        cluster_of = _build_clusters(ranges)
        intention_confidence = None
        weak_cluster_links = [frozenset()] * len(ranges)  # flat heuristic has no weak-link concept
    anchor_gis = [gi for gi, c in enumerate(cluster_of) if c == gi]
    if anchor_gis:
        anchor_mass = group_mass[..., anchor_gis]              # (BH, Nq, A)
        anchor_gis_t = torch.tensor(anchor_gis, device=sim.device)
        anchor_fair_shares = fair_share[anchor_gis_t]                                    # (A,)
        anchor_ratio = anchor_mass / anchor_fair_shares.clamp(min=1e-8)                  # (BH, Nq, A)
        null_logit = torch.full_like(anchor_ratio[..., :1], cfg.leak_min_confidence)     # (BH, Nq, 1)
        temperature = max(cfg.leak_ownership_temperature, 1e-4)
        logits = torch.cat([anchor_ratio, null_logit], dim=-1) / temperature             # (BH, Nq, A+1)
        probs = logits.softmax(dim=-1)                                                   # (BH, Nq, A+1)
        anchor_probs = probs[..., :-1]                                                   # (BH, Nq, A) real-anchor ownership weight; remainder is "no confident owner"
        cluster_to_anchor_idx = {c: i for i, c in enumerate(anchor_gis)}
        own_anchor = anchor_probs.argmax(dim=-1)                                         # diagnostics only
        own_prob = anchor_probs.gather(-1, own_anchor.unsqueeze(-1)).squeeze(-1)         # (BH, Nq) continuous confidence
    else:
        anchor_probs = None
        cluster_to_anchor_idx = {}
        own_prob = torch.zeros(group_mass.shape[:2], device=sim.device, dtype=sim.dtype)  # no anchors: no ownership signal

    def _not_own_weight(c: int) -> torch.Tensor:
        """Continuous 'this query position confidently belongs to a DIFFERENT
        cluster than c' weight in [0,1] — the smooth replacement for the old
        hard `(own_cluster != c) & confident` boolean. Zero both when the
        position confidently belongs to c itself AND when it's background /
        no confident owner (matches the old gate's "leave it alone" cases);
        only rises toward 1 when a DIFFERENT specific anchor wins."""
        if anchor_probs is None or c not in cluster_to_anchor_idx:
            return torch.zeros(group_mass.shape[:2], device=sim.device, dtype=sim.dtype)
        idx_c = cluster_to_anchor_idx[c]
        return anchor_probs.sum(dim=-1) - anchor_probs[..., idx_c]

    corrected = sim.clone()

    # --- Vanish: boost a group's columns everywhere when it never peaks above tau ---
    # tau is RELATIVE to each group's fair share, not an absolute constant: a
    # 1-token group naturally has far lower raw mass than a 3-token group even
    # when neither is actually neglected, and fair share itself shrinks as Nk
    # grows (e.g. 154-token rows from chunk padding). A fixed absolute
    # threshold here (an earlier version of this file used 0.05 flat) mis-fired
    # on almost every group almost every step — see the calibration note in
    # plans/03-clpc-token-conditional-guidance.md.
    tau_eff = cfg.tau_vanish * fair_share  # (G,) — cfg.tau_vanish is now a multiplier, not a raw mass
    deficit = (tau_eff.unsqueeze(0) - peak_mass).clamp(min=0.0)  # (BH, G)
    for gi, (s, e, _text) in enumerate(ranges):
        width = e - s
        boost = (cfg.beta_vanish * deficit[:, gi] / width).view(-1, 1, 1)
        corrected[..., s:e] = corrected[..., s:e] + boost

    # --- Bias: static inverse-tag-frequency boost (common tags get less) ---
    for gi, (s, e, text) in enumerate(ranges):
        freq = _lookup_tag_frequency(text, _EXAMPLE_TAG_FREQUENCY)
        width = e - s
        boost = cfg.bias_strength * (1.0 - freq) / width
        corrected[..., s:e] = corrected[..., s:e] + boost

    # --- Leak: attenuate a group's mass wherever a DIFFERENT, CONFIDENTLY-owned
    #     cluster (entity) dominates; groups with no preceding anchor
    #     (cluster_of[gi] == -1) are left alone — there's no entity to
    #     attribute their leak to, same as low-confidence/background rows.
    #
    #     NOTE: `weak_cluster_links` (refine_engine.py's non-structural
    #     "shared parent" concept) is intentionally NOT consumed here.  An
    #     earlier version of this loop halved leak_strength for a group's
    #     documented weak-linked rivals, but WEAK_LINK_MARGIN (0.15) proved
    #     far too generous against real CLIP embeddings — two characters
    #     described by fairly generic attribute text (e.g. "brown hair" vs
    #     "blue hair") routinely land within 0.15 of each other on ordinary
    #     embedding noise, not genuine shared-description ambiguity, so this
    #     was softening leak-attenuation far more broadly than intended. A
    #     live run showed `leak_score` stuck flat (~0.83) for an entire
    #     26-step generation instead of converging the way `vanish_score`
    #     did, consistent with real leak going under-corrected across most
    #     of the prompt. Reverted to always-full-strength attenuation;
    #     `weak_links`/`weak_cluster_links` remain available as pure
    #     diagnostic data (see refine_engine.py) for a future, more
    #     carefully calibrated and live-model-verified attempt at actually
    #     using them here.
    # leak_intensity[bh, q] = actual attention MASS removed at this query
    # position by the leak correction (L1 change of the row, leak-only —
    # snapshotted before the loop so the vanish/bias boosts above, which are
    # uniform across every row and so carry no spatial information, don't
    # contaminate it). This is a MAGNITUDE, not a confidence: a rival
    # group's own-cluster confidence can be high (`_not_own_weight` ≈ 1) at
    # a position where that rival barely has any mass to begin with, in
    # which case little mass actually moves — using confidence alone here
    # would report the same "intensity" at that position as at a position
    # where a rival dominates with real mass. Magnitude is the right
    # quantity to compare against CLPC's own Haar innovation magnitude,
    # which is also a perturbation-magnitude concept, not a probability.
    pre_leak = corrected.clone()
    for gi, (s, e, _text) in enumerate(ranges):
        c = cluster_of[gi]
        if c == -1:
            continue
        weight = _not_own_weight(c).unsqueeze(-1)  # (BH, Nq, 1), in [0,1]
        corrected[..., s:e] = corrected[..., s:e] * (1.0 - cfg.leak_strength * weight)
    leak_intensity = (corrected - pre_leak).abs().sum(-1)  # (BH, Nq)

    row_sum = corrected.sum(-1, keepdim=True).clamp(min=1e-8)
    corrected = corrected / row_sum

    if spatial_store is not None:
        spatial_store.append((leak_intensity.detach(), heads, orig_shape))

    # --- Diagnostics ---
    with torch.no_grad():
        peak_mean = peak_mass.mean(dim=0)  # (G,) — aggregate across batch*heads
        # IMPORTANT: use the AGGREGATE (mean-across-heads) deficit here, not the
        # per-head `deficit` tensor used for the actual correction above. Basing
        # this on "any single head is below tau" (an earlier version of this
        # file did) flags almost every group almost every step with 8+ heads —
        # some head individually specializing away from a token doesn't mean
        # the token is neglected in the rendered image.
        deficit_mean = (tau_eff - peak_mean).clamp(min=0.0)  # (G,)
        vanishing = [ranges[i][2] for i in range(len(ranges)) if float(deficit_mean[i]) > 0]
        confident_frac = float(own_prob.mean())
        # leak_frac[g] = weighted fraction of rival-owned rows that still hold
        # >1% of g's mass, weighted continuously by _not_own_weight(c) instead
        # of a hard confident-and-not-own boolean (see the SOFT OWNERSHIP note
        # above _apply_token_subspace_corrections's ownership block). Rows
        # that are background/ambiguous or confidently g's own contribute
        # ~0 weight either way, same exclusion the old hard gate achieved,
        # now graded rather than a step at the confidence floor.
        leak_frac = []
        for gi, (s, e, _t) in enumerate(ranges):
            c = cluster_of[gi]
            if c == -1:
                leak_frac.append(0.0)
                continue
            not_own_weight = _not_own_weight(c)
            rival_mass = group_mass[..., gi]
            weight_sum = not_own_weight.sum().clamp(min=1e-6)
            leaked = (not_own_weight * (rival_mass > 0.01).float()).sum() / weight_sum
            leak_frac.append(float(leaked))

        vanish_score = float((1.0 - (deficit_mean > 0).float().mean()).clamp(0.0, 1.0))
        leak_score = float((1.0 - sum(leak_frac) / max(len(leak_frac), 1))) if leak_frac else 1.0
        leak_score = max(0.0, min(1.0, leak_score))
        bias_score = 1.0  # static prior always "satisfied" by construction; informational only

        # --- Intention drift: how much the OBSERVED attention (leak_frac)
        # diverges from what the intention tree said SHOULD happen, weighted
        # by how CONFIDENT the tree was about each assignment. A confidently
        # -wrong observation (high intention_confidence, high leak_frac) is
        # more meaningful drift than an uncertain one (low confidence, same
        # leak_frac) — this is what "more accurately decide what got loss,
        # wrong, drift" means concretely: not a flat average of leak_frac,
        # but one weighted by how sure the prompt's own structure was.
        # Falls back to unweighted leak_frac (confidence=1 uniformly) when
        # no intention tree was supplied, so `leak_score`/`intention_drift`
        # coincide in that case rather than silently diverging.
        if intention_confidence is not None and leak_frac:
            drift_terms = [intention_confidence[gi] * leak_frac[gi] for gi in range(len(leak_frac))]
        else:
            drift_terms = leak_frac
        intention_drift_score = 1.0 - (sum(drift_terms) / max(len(drift_terms), 1)) if drift_terms else 1.0
        intention_drift_score = max(0.0, min(1.0, intention_drift_score))

        if cfg.debug:
            clusters_readable = {
                ranges[gi][2]: (ranges[c][2] if c != -1 else "(unclustered)")
                for gi, c in enumerate(cluster_of)
            }
            print(f"[TokenSubspaceGuidance:{layer_tag}] groups={[t for _, _, t in ranges]} "
                  f"clusters={clusters_readable} "
                  f"peak_mass={[round(float(x), 4) for x in peak_mean.tolist()]} "
                  f"vanishing={vanishing} leak_frac={[round(x, 3) for x in leak_frac]} "
                  f"confident_frac={confident_frac:.3f} "
                  f"vanish_score={vanish_score:.3f} leak_score={leak_score:.3f} "
                  f"intention_drift_score={intention_drift_score:.3f}"
                  + (f" (confidence-weighted: {[round(c,3) for c in intention_confidence]})"
                     if intention_confidence is not None else " (no intention tree — unweighted)"))

        if diag_store is not None:
            diag_store.append({
                "vanish_score": vanish_score,
                "leak_score": leak_score,
                "bias_score": bias_score,
                "intention_drift_score": intention_drift_score,
            })

    return corrected


def _make_token_guidance_hook(groups_info: dict, cfg: TokenGuidanceConfig,
                               diag_store: list, layer_tag: str,
                               final_tokens: Optional[list] = None,
                               spatial_store: Optional[list] = None):
    """Build an attn2-replacement hook. Signature matches the
    patches_replace["attn2"][block] interface used throughout this codebase:
        hook(q, k, v, extra_options, mask=None) -> out
    (same convention as sure_attention.py's attn1 entropy hook.)

    final_tokens: Plan 04's reconciled intention-tree output (see
    `_apply_token_subspace_corrections`'s docstring); None falls back to the
    keyword heuristic.

    spatial_store: forwarded to `_apply_token_subspace_corrections` — see its
    docstring. `extra_options["original_shape"]` (the full UNet latent
    [B,C,H,W]) is populated identically for attn1 AND attn2 hooks by
    `BasicTransformerBlock.forward` (ldm_patched/ldm/modules/attention.py),
    the same field `sure_attention.py`'s entropy hook already reads.
    """
    groups = groups_info.get("groups", [])
    n_chunks = groups_info.get("n_chunks", 0)
    chunk_length = groups_info.get("chunk_length", 75)

    def hook(q, k, v, extra_options, mask=None):
        heads = extra_options["n_heads"]
        orig_shape = extra_options.get("original_shape")
        orig_dtype = q.dtype
        b, _n_q, dim_head_full = q.shape
        dim_head = dim_head_full // heads
        scale = dim_head ** -0.5

        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            def _split_heads(t):
                return (
                    t.float().unsqueeze(3)
                    .reshape(b, -1, heads, dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * heads, -1, dim_head)
                    .contiguous()
                )

            qh, kh, vh = _split_heads(q), _split_heads(k), _split_heads(v)
            sim = torch.einsum("b i d, b j d -> b i j", qh, kh) * scale

            if mask is not None:
                max_neg_value = -torch.finfo(sim.dtype).max
                mask_r = mask.reshape(mask.shape[0], -1)
                mask_r = mask_r.unsqueeze(1).repeat(heads, 1, 1)
                sim.masked_fill_(~mask_r, max_neg_value)

            sim = sim.softmax(dim=-1)  # (BH, Nq, Nk)

            corrected = _apply_token_subspace_corrections(
                sim, groups, n_chunks, chunk_length, cfg, diag_store, layer_tag,
                final_tokens=final_tokens,
                spatial_store=spatial_store, heads=heads, orig_shape=orig_shape,
            )

            out = torch.einsum("b i j, b j d -> b i d", corrected, vh)
            out = (
                out.reshape(b, heads, -1, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, -1, heads * dim_head)
            )

        return out.to(orig_dtype)

    return hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALL_BLOCKS = {"input": range(12), "middle": range(1), "output": range(12)}


def patch_model_with_token_guidance(unet, groups_info: dict,
                                     cfg: TokenGuidanceConfig,
                                     block_types: str = "middle",
                                     final_tokens: Optional[list] = None):
    """Install persistent attn2 token-subspace-guidance hooks on `unet`.

    final_tokens: Plan 04's reconciled intention-tree output (see
    `modules.rule_engine.pipeline.run_pipeline`), one `FinalTokenInfo` per
    entry in `groups_info["groups"]`, same order. When provided, clustering
    is driven by the rule-engine/Matrix-Tree pipeline instead of the flat
    8-keyword heuristic — see `_build_clusters_from_intention_tree`. None
    falls back to the Plan 03 heuristic (e.g. rule-engine extension
    disabled, or pipeline failed for this generation).

    Returns (patched_unet, diag_store) — `diag_store` accumulates plain-float
    diagnostic dicts across every hook call; callers (e.g. the CLPC sampler)
    should read + clear it once per sampling step via
    `aggregate_token_guidance_info` below. A second, raw-tensor store
    (`token_guidance_spatial_store` in `unet.model_options["transformer_options"]`)
    accumulates per-position leak-intensity captures for
    `aggregate_token_spatial_map` to reduce into a pixel/latent-space map —
    kept separate from `diag_store` since it holds un-reduced tensors, not
    plain floats.

    Clones `unet` first (same convention as SureAttentionGuidance.patch()) so
    the base model object isn't mutated across generations/UI runs.
    """
    unet = unet.clone()
    diag_store: list = []
    spatial_store: list = []

    if final_tokens is not None:
        print(f"[TokenSubspaceGuidance] using Plan 04 intention tree "
              f"({len(final_tokens)} segment(s)) for clustering — "
              f"NOT the flat keyword heuristic.")
    else:
        print("[TokenSubspaceGuidance] no intention tree supplied — "
              "falling back to the Plan 03 keyword heuristic.")

    if block_types == "all":
        active = _ALL_BLOCKS
    elif block_types in _ALL_BLOCKS:
        active = {block_types: _ALL_BLOCKS[block_types]}
    elif block_types == "mid+out":
        active = {"middle": _ALL_BLOCKS["middle"], "output": _ALL_BLOCKS["output"]}
    else:
        _logger.warning("sure_token_guidance: unknown attn_blocks=%r; using 'middle'", block_types)
        active = {"middle": _ALL_BLOCKS["middle"]}

    n_groups = len(groups_info.get("groups", []))
    print(f"[TokenSubspaceGuidance] installing attn2 hooks on blocks={list(active.keys())} "
          f"groups={n_groups} cfg={cfg}")

    for block_name, ids in active.items():
        for block_id in ids:
            layer_tag = f"{block_name}{block_id}"
            hook_fn = _make_token_guidance_hook(groups_info, cfg, diag_store, layer_tag,
                                                 final_tokens=final_tokens,
                                                 spatial_store=spatial_store)
            unet.set_model_attn2_replace(hook_fn, block_name, block_id)

    to = unet.model_options.setdefault("transformer_options", {})
    to["token_guidance_store"] = diag_store
    to["token_guidance_spatial_store"] = spatial_store

    return unet, diag_store


def aggregate_token_guidance_info(store: Optional[list]) -> Optional[dict]:
    """Average all diagnostic dicts captured since the last clear into one
    {"vanish_score", "leak_score", "bias_score"} dict, all in [0,1].

    Returns None if the store is empty/None (guidance disabled or no attn2
    layers fired yet this step) — callers should treat that as "neutral"
    (see clpc_error.compute_token_guidance_score).
    """
    if not store:
        return None
    n = len(store)
    agg = {
        "vanish_score": sum(d["vanish_score"] for d in store) / n,
        "leak_score": sum(d["leak_score"] for d in store) / n,
        "bias_score": sum(d["bias_score"] for d in store) / n,
        # Confidence-weighted drift from the Plan 04 intention tree (falls back
        # to unweighted leak_frac, same value as leak_score, when no tree was
        # available this run — see _apply_token_subspace_corrections).
        "intention_drift_score": sum(d.get("intention_drift_score", d["leak_score"]) for d in store) / n,
    }
    print(f"[TokenSubspaceGuidance] step aggregate over {n} attn2 calls: {agg}")
    return agg


def aggregate_token_spatial_map(store: Optional[list], latent_shape: tuple,
                                 device, dtype) -> Optional[torch.Tensor]:
    """Convert captured `(leak_intensity, heads, orig_shape)` entries (see
    `_apply_token_subspace_corrections`) into a single spatial map of shape
    `(B, 1, H_lat, W_lat)` in `[0,1]`, matching `latent_shape` — the SAME
    pixel/latent grid CLPC's Haar wavelet decomposition of `x_corr - x_pred`
    already operates on (`clpc_error.compute_haar_subband_errors`).

    Mirrors `sure_attention._aggregate_entropy_map`'s tokens-to-spatial-grid
    + bilinear-upsample recipe exactly (each attn2 layer's query axis is a
    flattened, block-downsampled version of the same latent grid; different
    blocks run at different downsample factors, hence the per-layer resize
    before averaging). Unlike entropy (query-position UNCERTAINTY, defined
    everywhere), leak intensity is 0 almost everywhere by construction (most
    query positions have no rival cluster to attenuate against) — a
    per-layer MAX rather than mean would let one hot layer dominate the
    average disproportionately, so layers are averaged the same way
    `_aggregate_entropy_map` averages entropy, keeping the two spatial-map
    aggregation conventions consistent for anyone reading both.

    Returns None if the store is empty (guidance disabled, or no attn2 calls
    captured a spatial signal yet this step).
    """
    if not store:
        return None

    B, _, H_lat, W_lat = latent_shape
    maps = []

    for leak_intensity, heads, orig_shape in store:
        BH, N = leak_intensity.shape
        if heads <= 0 or BH < heads:
            continue

        b_eff = BH // heads
        if b_eff > B:
            b_use = B
            leak_intensity = leak_intensity[heads * (b_eff - B):]
        else:
            b_use = b_eff

        # Average over attention heads → (b_use, N)
        intensity_b = leak_intensity.reshape(b_use, heads, N).mean(dim=1)

        if orig_shape is not None:
            H_src, W_src = int(orig_shape[2]), int(orig_shape[3])
        else:
            H_src, W_src = H_lat, W_lat

        h_l, w_l = _tokens_to_spatial(N, H_src, W_src)
        if h_l * w_l != N:
            continue

        intensity_spatial = intensity_b.float().reshape(b_use, 1, h_l, w_l)

        intensity_up = F.interpolate(intensity_spatial, (H_lat, W_lat),
                                      mode="bilinear", align_corners=False)

        if b_use < B:
            pad = intensity_up[-1:].expand(B - b_use, -1, -1, -1)
            intensity_up = torch.cat([intensity_up, pad], dim=0)

        maps.append(intensity_up.to(device=device))

    if not maps:
        return None

    M = torch.stack(maps, dim=0).mean(dim=0)
    M = M.clamp(0.0, 1.0).to(dtype=dtype)

    if M.isnan().any() or M.isinf().any():
        _logger.warning("[TokenSubspaceGuidance] NaN/Inf in spatial leak map; skipping this step")
        return None

    return M
