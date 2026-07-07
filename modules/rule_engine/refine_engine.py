"""Refine engine — the final pass of the rule-engine/semantic-analyzer
pipeline (Plan 04). Combines the rule engine's hard facts (Phase 1, ground
truth where it exists) with the semantic analyzer's soft structural
inference (Phase 2+3, a Matrix-Tree per logical chunk) into one reconciled
"intention matrix tree": for every prompt segment, a final parent choice and
a final class-probability distribution, ready for Phase 5's attention
correction to consume.

Two kinds of post-processing, as specified:

1. **Location-aware tie-breaking** — when a segment's top parent-marginal
   candidates are within `tie_epsilon` of each other (the tree genuinely
   can't distinguish them), break the tie using PROMPT POSITION: prefer the
   nearest PRECEDING candidate (this project's established "earlier anchor
   binds later attributes" convention), falling back to nearest by absolute
   distance if no tied candidate precedes the child.

2. **Reconsideration** — situations where the rule engine's hard, exact
   classification (ground truth) and the Matrix-Tree's purely structural
   inference would disagree, resolved in the rule engine's favor:
     - A `PRIMARY_OBJ`-classified segment (a main character/subject that
       always independently appears in the image, e.g. "1girl"/"1boy" — see
       schema.py) is FORCED to ROOT, full stop, never to another entity.
       This exists because `matrix_tree.py`'s single-root formulation only
       allows ONE node to attach directly to ROOT, so two co-equal subjects
       ("1girl", "1boy") would otherwise get structurally forced into a
       false parent-child relationship purely because the tree needs SOME
       single spanning structure — a real live-WebUI log showed exactly
       this: `"1boy"` (its own independent subject) ending up parented by
       `"1girl"` at p=0.465, which then visibly mixed the two characters'
       hair-color attributes downstream.
     - An `OBJECT`-classified segment WITHOUT `PRIMARY_OBJ` (a secondary/
       subordinate object, e.g. "cat", "sword" — something a main character
       can be described as carrying/wearing/near) structurally shouldn't be
       "owned" by an arbitrary attribute/verb, OR by another secondary
       object — its final parent is restricted to {ROOT, PRIMARY_OBJ
       segments} (falling back to the full OBJECT set if a minimal/custom
       ruleset defines no PRIMARY_OBJ at all).
     - A `META`-classified segment is explicitly non-region-specific by
       definition (7-class taxonomy) — its final parent is forced to ROOT
       regardless of what the tree computed, and it is marked excluded from
       downstream attention correction entirely.
     - A `GLOBAL`- or `CONCEPT`-classified segment (whole-image effect or
       style/theme — see schema.py's TAG_CLASSES) is, by definition, not
       owned by any specific entity either, but unlike META it IS visual
       and still needs attention correction — its final parent is forced
       to a dedicated `GLOBAL_NODE` sentinel (depth -1, distinct from
       ROOT/depth 0) rather than left for the tree to structurally guess at
       an arbitrary entity, which is exactly the bug this closes (a real
       live-WebUI log showed `"depth of field"`, correctly rule-engine
       classified GLOBAL, ending up bound to `"1boy"` as if it were his
       attribute, simply because GLOBAL had no reconsideration branch of
       its own and fell through to the generic tie-break).
     - A segment with NO class at all (situation 3, structural information
       only) gets its final class INFERRED from whichever parent it ended
       up with: attached to an OBJECT -> ADJ (an unlabelled attribute of
       that entity); attached to ROOT or GLOBAL_NODE -> GLOBAL (an
       unclaimed, whole-image descriptor). This is what closes the loop and
       gives every segment SOME classification by the end of the pipeline.

Also, a GLOBAL **node-promotion safety net**: a segment that stays fully
unclassified through rule engine + CLIP near-match, but (a) trails ALL
OBJECT/ADJ-classified content in its chunk (the Danbooru convention of
subject-then-attributes-then-quality/style tags), and (b) weakly resembles
a known GLOBAL/CONCEPT example via CLIP even below the strict near-match
threshold (`semantic_analyzer.SemanticAnalysisResult.weak_global_hint`), is
promoted straight to `GLOBAL_NODE` — the same "node promotion" concept as
the OBJECT fallback below, but driven by an actual CLIP similarity signal
rather than a hardcoded word list.

Also, **weak links** (a non-looping, B-tree-like "shared parent" concept):
the Matrix-Tree math itself requires every node to have exactly ONE
structural parent (matrix_tree.py's single-root formulation), so it has no
way to express "this description applies to a GROUP of entities" directly.
Rather than compromise that structure, an ADJ/OBJECT segment's `weak_links`
field records any OTHER PRIMARY_OBJ/OBJECT candidate whose restricted
Matrix-Tree marginal is within `WEAK_LINK_MARGIN` of the chosen parent's —
i.e. "almost as plausible an owner as the one actually picked". This is
purely auxiliary/informational: it never changes `final_parent` or risks a
cycle, it just surfaces cases like "matching outfits" where two characters
are both nearly-equally likely owners, for a consumer that wants to apply a
lighter-touch correction to the group rather than only the single winner.
"""

from __future__ import annotations

import dataclasses
import re

from modules.rule_engine.chunk_mask import compute_chunk_mask, group_by_chunk

TIE_EPSILON = 0.05   # marginal-probability gap below which two candidates count as "tied"
ROOT = 0             # sentinel parent id for the virtual root
GLOBAL_NODE = -1     # sentinel parent id for "whole-image, no specific owner" (depth -1)
GLOBAL_HINT_THRESHOLD = 0.5   # weak CLIP-similarity floor for GLOBAL node-promotion (softer
                               # than semantic_analyzer.NEAR_MATCH_THRESHOLD's 0.85 — this is
                               # a safety net, not a primary classification)
WEAK_LINK_MARGIN = 0.15       # marginal-probability gap (within the restricted OBJECT/
                               # PRIMARY_OBJ candidate pool) below which an alternate
                               # candidate counts as a "weak link" -- see module docstring

# SECOND-level OBJECT safety net (kept for defense-in-depth, no longer the
# primary mechanism). The PRIMARY fix for natural-language phrases like "one
# 18 years old girl" is now `modules.rule_engine.token_table`'s sub-word
# decomposition (Phase 1, generalized to all 7 classes via the actual
# compiled ruleset — "girl" alone matches the existing OBJECT pattern even
# though the whole phrase doesn't). That mechanism runs BEFORE the semantic
# analyzer even sees the segment, so by the time this code runs,
# `class_probs[i]` is already `{"OBJECT": 1.0}` in the common case and this
# scan never fires (the `if not class_probs[i]` gate below is what prevents
# double-application). This hardcoded scan only still matters for the edge
# case of a custom/minimal ruleset with no OBJECT-matching pattern loaded at
# all (e.g. someone deselects the danbooru.yaml base preset) — cheap enough
# to keep as a last-resort layer regardless.
_FALLBACK_ANCHOR_WORDS = frozenset({"girl", "girls", "boy", "boys", "woman", "women",
                                     "man", "men", "person", "people", "child", "children",
                                     "male", "female"})


def _fallback_anchor_scan(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return any(w in _FALLBACK_ANCHOR_WORDS for w in words)


@dataclasses.dataclass
class FinalTokenInfo:
    text: str
    start: int
    end: int
    chunk_index: int
    final_classes: frozenset
    final_parent: int            # index into the chunk's own segment list, or ROOT (0... see note)
    final_parent_text: str       # "ROOT" or the parent segment's text, for display
    parent_confidence: float
    excluded: bool                # True for META (and anything else ruled out of correction)
    reconsidered: bool             # True if the rule engine overrode the raw tree's top choice
    weak_links: frozenset = dataclasses.field(default_factory=frozenset)
    # weak_links: OTHER OBJECT/PRIMARY_OBJ segment indices (1-indexed, same
    # space as final_parent) whose Matrix-Tree marginal for this segment is
    # nearly as high as the CHOSEN parent's -- i.e. this description
    # plausibly applies to a GROUP of entities, not just the one it's
    # structurally attached to (e.g. "matching outfits" with two characters
    # both about equally likely parents). This is deliberately NOT a second
    # structural parent: `final_parent` stays the single, strict-tree
    # attachment the Matrix-Tree math requires (its single-root/single-parent
    # formulation -- see matrix_tree.py -- has no notion of shared
    # ownership), and `weak_links` is purely auxiliary, informational
    # metadata layered on top, like a B-tree node's non-structural
    # cross-references: it can never introduce a cycle or a second
    # "official" parent, only additional entities a consumer MAY also want
    # to apply lighter-touch correction to.


def _tied_candidates(dist: dict, tie_epsilon: float) -> list:
    best_p = max(dist.values())
    return [h for h, p in dist.items() if best_p - p <= tie_epsilon]


def _break_tie_by_position(candidates: list, child_local_index: int, chunk_entries: list) -> int:
    """chunk_entries: the chunk's own ChunkedEntry list (1-indexed candidate h
    maps to chunk_entries[h-1]; h=0 is ROOT). Prefers the nearest PRECEDING
    candidate by prompt position; falls back to nearest by absolute distance
    among the tied set if none of them precede the child."""
    if ROOT in candidates and len(candidates) == 1:
        return ROOT

    def position_of(h: int) -> int:
        return -1 if h == ROOT else chunk_entries[h - 1].local_start

    child_pos = chunk_entries[child_local_index].local_start
    preceding = [h for h in candidates if position_of(h) < child_pos]
    if preceding:
        return max(preceding, key=position_of)  # nearest preceding = largest position < child_pos
    return min(candidates, key=lambda h: abs(position_of(h) - child_pos))


def _restrict_to(dist: dict, allowed: set) -> dict:
    restricted = {h: p for h, p in dist.items() if h in allowed}
    return restricted if restricted else {ROOT: dist.get(ROOT, 1.0)}


def _nearest_preceding_or_best(dist: dict, child_local_index: int, chunk_entries: list) -> int:
    """Position-PRIMARY selection among the (already class-restricted)
    candidates in `dist`, for OBJECT/ADJ reconsideration specifically.

    The "earlier entity anchors its later attributes" convention isn't a
    tie-breaker of last resort here — it's the load-bearing rule these tag
    prompts are written by (`1girl, ..., 1boy, blue hair` means blue hair is
    the boy's). A same-order-of-magnitude-but-not-quite-tied Matrix-Tree
    marginal (e.g. 0.57 vs 0.25 — nowhere near TIE_EPSILON) is still just
    CLIP-similarity noise next to that convention, since theta's own
    precedence bonus (`semantic_analyzer.W_EDGE_PRECEDENCE`) is flat across
    ALL preceding OBJECT candidates regardless of distance — it doesn't
    itself discriminate "adjacent" from "three tags back". So: the nearest
    REAL (non-ROOT) preceding candidate wins outright whenever one exists.
    Only falls back to the raw top-marginal candidate when nothing in
    `dist` actually precedes the child (a genuine forward reference, e.g.
    an attribute written before any entity appears — position gives no
    guidance there, so the tree's structural guess is all there is)."""
    def position_of(h: int) -> int:
        return -1 if h == ROOT else chunk_entries[h - 1].local_start

    child_pos = chunk_entries[child_local_index].local_start
    preceding_real = [h for h in dist if h != ROOT and position_of(h) < child_pos]
    if preceding_real:
        return max(preceding_real, key=position_of)
    return max(dist, key=dist.get)


def _weak_links_from(restricted: dict, chosen: int, margin: float) -> frozenset:
    """See module docstring's "weak links" section. `restricted` is the same
    class-restricted candidate pool the primary decision was made from;
    `chosen` is that decision. Returns every OTHER real (non-ROOT) candidate
    whose marginal is within `margin` of `chosen`'s -- entities this
    description plausibly ALSO applies to, surfaced without altering the
    single structural parent."""
    chosen_p = restricted.get(chosen, 0.0)
    return frozenset(h for h, p in restricted.items()
                      if h != chosen and h != ROOT and chosen_p - p <= margin)


def refine_chunk(chunk_entries: list, semantic_result) -> list:
    """chunk_entries: list[ChunkedEntry] for ONE chunk (see chunk_mask.py).
    semantic_result: the SemanticAnalysisResult for that same chunk."""
    n = len(chunk_entries)
    tree = semantic_result.tree
    class_probs = [dict(p) for p in semantic_result.class_probabilities]  # local mutable copy

    # Second-level fallback anchor scan (see the comment above
    # _FALLBACK_ANCHOR_WORDS for why this is no longer the primary
    # mechanism): only for segments that are STILL unclassified after Phase
    # 1's rule engine (including its own sub-word/META_TREE decomposition)
    # AND Phase 2+3's CLIP near-match — never overrides an existing
    # classification.
    for i in range(n):
        if not class_probs[i] and _fallback_anchor_scan(chunk_entries[i].entry.text):
            class_probs[i]["OBJECT"] = 1.0
            print(f"[RefineEngine] {chunk_entries[i].entry.text!r} matched no rule/near-match, "
                  f"but contains a known entity word -- promoted to OBJECT as a last-resort "
                  f"safety net (reconsideration).")

    object_indices = {i + 1 for i in range(n) if class_probs[i].get("OBJECT", 0.0) >= 0.5}
    primary_indices = {i + 1 for i in range(n) if class_probs[i].get("PRIMARY_OBJ", 0.0) >= 0.5}

    # GLOBAL node-promotion safety net: a segment that's STILL unclassified
    # (no rule-engine match, no OBJECT rescue above, no CLIP near-match) is
    # promoted straight to GLOBAL_NODE if it (a) trails every OBJECT/ADJ-
    # classified segment in this chunk -- the Danbooru convention of
    # subject-then-attributes-then-quality/style tags -- and (b) weakly
    # resembles a known GLOBAL/CONCEPT example via CLIP, even below the
    # strict near-match threshold. "Last content position" is computed from
    # EXPLICIT (Phase 1/2) classification only, not situation-3 inference
    # below, to avoid a chicken-and-egg ordering problem.
    content_positions = [chunk_entries[i].local_start for i in range(n)
                          if class_probs[i].get("OBJECT", 0.0) >= 0.5
                          or class_probs[i].get("ADJ", 0.0) >= 0.5]
    last_content_pos = max(content_positions, default=-1)
    weak_hints = getattr(semantic_result, "weak_global_hint", None) or [0.0] * n
    for i in range(n):
        if class_probs[i]:
            continue
        if chunk_entries[i].local_start <= last_content_pos:
            continue
        hint = weak_hints[i] if i < len(weak_hints) else 0.0
        if hint >= GLOBAL_HINT_THRESHOLD:
            class_probs[i]["GLOBAL"] = 1.0
            print(f"[RefineEngine] {chunk_entries[i].entry.text!r} matched no rule/near-match, "
                  f"but trails all entity/attribute content (pos={chunk_entries[i].local_start} > "
                  f"{last_content_pos}) and weakly resembles a known GLOBAL/CONCEPT tag via CLIP "
                  f"(hint={hint:.3f} >= {GLOBAL_HINT_THRESHOLD}) -- promoted to GLOBAL_NODE "
                  f"(depth -1) as a last-resort safety net (node promotion).")

    finals = []
    for m in range(1, n + 1):
        entry = chunk_entries[m - 1].entry
        classes = frozenset(c for c, p in class_probs[m - 1].items() if p >= 0.5)
        dist = tree.parent_distribution(m)
        raw_best = max(dist, key=dist.get)
        reconsidered = False
        excluded = False
        weak_links: frozenset = frozenset()

        if "META" in classes:
            # Reconsideration: META is globally-referenced, never region-specific
            # (7-class taxonomy) -- force to ROOT regardless of tree structure,
            # and exclude from downstream attention correction entirely.
            chosen, excluded = ROOT, True
            reconsidered = (raw_best != ROOT)
        elif "PRIMARY_OBJ" in classes:
            # Reconsideration: a main character/subject is ALWAYS its own
            # independent top-level entity -- FORCE to ROOT, full stop, never
            # to another entity (even another PRIMARY_OBJ). Deliberately does
            # NOT go through _nearest_preceding_or_best: that function treats
            # ANY real preceding candidate as automatically better than ROOT,
            # which is exactly what caused "1boy" to get bound to "1girl" (a
            # real live-WebUI log) -- correct for an ATTRIBUTE binding to its
            # nearest entity, but wrong for two co-equal subjects, which
            # should never chain into each other regardless of the Matrix-
            # Tree's single-root structural bias toward picking ONE of them.
            chosen = ROOT
            reconsidered = (raw_best != ROOT)
        elif "OBJECT" in classes:
            # Reconsideration: a secondary/subordinate object (OBJECT
            # without PRIMARY_OBJ, e.g. "cat", "sword") belongs to a PRIMARY
            # entity, or ROOT if none precedes -- never to another secondary
            # object. Falls back to the full OBJECT set if this ruleset
            # defines no PRIMARY_OBJ distinction at all (backward-
            # compatible with minimal/custom rulesets). Position is PRIMARY
            # among the restricted candidates (see _nearest_preceding_or_best)
            # -- the nearest preceding owner wins outright, not just on
            # near-ties.
            owner_pool = primary_indices if primary_indices else object_indices
            allowed = {ROOT} | (owner_pool - {m})
            restricted = _restrict_to(dist, allowed)
            chosen = _nearest_preceding_or_best(restricted, m - 1, chunk_entries)
            reconsidered = (chosen != raw_best)
            weak_links = _weak_links_from(restricted, chosen, WEAK_LINK_MARGIN)
        elif "ADJ" in classes:
            # Reconsideration: an attribute describes an ENTITY (or is a
            # genuinely global attribute with no owner) -- it should never
            # bind to another attribute segment, even if two attributes
            # happen to be semantically close (e.g. two hair-color tags).
            # Position is PRIMARY here too, for the same reason as OBJECT.
            allowed = {ROOT} | object_indices
            restricted = _restrict_to(dist, allowed)
            chosen = _nearest_preceding_or_best(restricted, m - 1, chunk_entries)
            reconsidered = (chosen != raw_best)
            weak_links = _weak_links_from(restricted, chosen, WEAK_LINK_MARGIN)
            if weak_links:
                shared_with = [chunk_entries[h - 1].entry.text for h in sorted(weak_links)]
                chosen_label = "ROOT" if chosen == ROOT else chunk_entries[chosen - 1].entry.text
                print(f"[RefineEngine] {entry.text!r} weakly linked to {shared_with} in addition "
                      f"to its chosen parent {chosen_label!r} -- description plausibly shared "
                      f"across a group of entities (non-structural).")
        elif "GLOBAL" in classes or "CONCEPT" in classes:
            # Reconsideration: whole-image effect/style (schema.py's
            # TAG_CLASSES) is never owned by a specific entity -- unlike
            # META it's still visual and needs attention correction, so it
            # goes to the dedicated GLOBAL_NODE (depth -1) rather than being
            # excluded, and rather than being left for the tree to
            # structurally guess an arbitrary entity parent (the exact bug
            # this branch closes -- see module docstring).
            chosen = GLOBAL_NODE
            reconsidered = (raw_best != ROOT)
        else:
            tied = _tied_candidates(dist, TIE_EPSILON)
            chosen = (tied[0] if len(tied) == 1 else
                      _break_tie_by_position(tied, m - 1, chunk_entries))
            reconsidered = (chosen != raw_best)

        if not classes:
            # Situation-3 inference: no rule/near-match class at all -- infer
            # from the chosen parent, closing the loop so every segment ends
            # up with SOME classification.
            if chosen in (ROOT, GLOBAL_NODE):
                classes = frozenset({"GLOBAL"})
            elif chosen in object_indices:
                classes = frozenset({"ADJ"})
            print(f"[RefineEngine] {entry.text!r} had no class prior -- inferred "
                  f"{sorted(classes) if classes else '(still none)'} from its chosen parent")

        if chosen == GLOBAL_NODE:
            parent_text = "GLOBAL"
        elif chosen == ROOT:
            parent_text = "ROOT"
        else:
            parent_text = chunk_entries[chosen - 1].entry.text
        finals.append(FinalTokenInfo(
            text=entry.text, start=entry.start, end=entry.end,
            chunk_index=chunk_entries[m - 1].chunk_index,
            final_classes=classes, final_parent=chosen, final_parent_text=parent_text,
            parent_confidence=float(dist.get(chosen, 0.0)),
            excluded=excluded, reconsidered=reconsidered, weak_links=weak_links,
        ))

    return finals


def refine(token_table: list, semantic_results: dict, chunk_length: int = 75) -> list:
    """token_table: Phase 1's full list[TokenTableEntry]. semantic_results:
    {chunk_index: SemanticAnalysisResult}, as returned by
    `semantic_analyzer.analyze_multi_chunk`. Returns list[FinalTokenInfo] in
    original prompt order, spanning all chunks."""
    chunked = compute_chunk_mask(token_table, chunk_length=chunk_length)
    groups = group_by_chunk(chunked)

    all_finals: dict = {}
    for chunk_index, entries in groups.items():
        result = semantic_results.get(chunk_index)
        if result is None:
            print(f"[RefineEngine] WARNING: no semantic-analysis result for chunk "
                  f"{chunk_index}; skipping ({len(entries)} segment(s) left unrefined)")
            continue
        chunk_finals = refine_chunk(entries, result)
        for ce, final in zip(entries, chunk_finals):
            all_finals[ce.index] = final

    ordered = [all_finals[i] for i in range(len(token_table)) if i in all_finals]

    print(f"\n[RefineEngine] === Final intention matrix tree "
          f"({len(ordered)} segment(s) across {len(groups)} chunk(s)) ===")
    for f in ordered:
        flags = []
        if f.excluded:
            flags.append("EXCLUDED")
        if f.reconsidered:
            flags.append("RECONSIDERED")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"[RefineEngine]   chunk={f.chunk_index}  {f.text!r:25} "
              f"classes={sorted(f.final_classes) if f.final_classes else '(none)'}  "
              f"parent={f.final_parent_text!r} (p={f.parent_confidence:.3f}){flag_str}")

    return ordered
