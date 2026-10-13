"""Semantic analyzer (Plan 04 Phase 2) — the compiler-pipeline stage between
the rule engine's token table (Phase 1, lexer) and the sampler's attention
correction (Phase 5): turns each token table entry's hard classification (or
lack of one) into a per-class PROBABILITY vector, using CLIP-embedding
similarity as an unsupervised nearest-neighbor classifier, then builds the
Matrix-Tree potentials and computes the resulting parent marginals — "the
tree that represents what we know now."

Three situations per token (as specified), all folded into ONE
`class_probabilities` output (independent per-class membership probability
in [0,1] — NOT a softmax-normalized categorical, since classes legitimately
overlap, e.g. "comic" is both GLOBAL and CONCEPT):

  1. Exact rule-engine match (Phase 1 already assigned classes) — probability
     1.0 for each assigned class. This token ALSO seeds the vector database
     as a labeled example for situation 2.
  2. No exact match, but cosine-similar (>= `near_match_threshold`) to an
     already-seen exact match — inherit a similarity-weighted blend of the
     nearest neighbors' classes (soft k-NN vote).
  3. Neither — genuinely unknown/novel. No class prior; relies entirely on
     the Matrix-Tree structure (semantic + precedence potentials) below.

The Matrix-Tree edge potential `theta[h, m]` (h = candidate parent, m =
child, both 1-indexed prompt segments, 0 = virtual root) combines:
  - CLIP cosine similarity between segments h and m (semantic relatedness —
    "ask CLIP about all prompts classification", not just unknown-to-anchor).
  - A precedence bias: only when h comes BEFORE m in the prompt, scaled by
    how OBJECT-like h is — the "earlier entity anchors its later attributes"
    convention, now a soft PRIOR inside the probabilistic graph rather than
    a separate hard tie-breaking rule.
Root-selection score `theta[0, m]` is scaled by how OBJECT-like segment m
is — entities are the natural roots of a dependency structure.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from modules.rule_engine.clip_vector_db import VectorDB
from modules.rule_engine.matrix_tree import TreeMarginals, compute_tree_marginals
from modules.rule_engine.schema import TAG_CLASSES
from modules.rule_engine.token_table import TokenTableEntry

# Tunable weights — see module docstring for what each term represents.
NEAR_MATCH_THRESHOLD = 0.85   # cosine similarity considered "resembles a known tag"
NEAR_MATCH_TOP_K = 5
W_ROOT_OBJECT = 4.0           # root-score scaling by OBJECT-class probability
W_EDGE_SIMILARITY = 4.0       # edge-score scaling by CLIP cosine similarity
W_EDGE_PRECEDENCE = 2.0       # edge-score bonus for earlier, OBJECT-like candidates

# Classes meaning "whole-image effect/style, not owned by any specific
# entity" (see schema.py's TAG_CLASSES docstring) -- used by the weak-hint
# scan below, feeding the refine engine's GLOBAL node-promotion safety net.
_WHOLE_IMAGE_CLASSES = frozenset({"GLOBAL", "CONCEPT"})


@dataclasses.dataclass
class SemanticAnalysisResult:
    class_probabilities: list          # list[dict[str, float]], one per token-table entry
    situations: list                    # list[int] in {1,2,3}, one per entry (debug/audit)
    tree: TreeMarginals                 # the constructed Matrix-Tree marginals
    vector_db: VectorDB                  # the (now-populated) vector database, for reuse
    weak_global_hint: list = dataclasses.field(default_factory=list)
    # weak_global_hint[i] = best cosine similarity, IGNORING NEAR_MATCH_THRESHOLD,
    # between entry i's embedding and any GLOBAL/CONCEPT-classified vector-DB
    # example seen so far (0.0 if the DB has no such example yet). This is
    # deliberately a much softer signal than the strict near-match
    # classifier above -- "ask CLIP about all prompts classification" (Plan
    # 04's original spec) covers not just situations 1-3's own class but
    # also this auxiliary "does this look even a little bit like a
    # whole-image tag" hint, which the refine engine's GLOBAL safety net
    # uses for otherwise-unclassified trailing segments (see refine_engine.py).


def _classify_via_nearest_neighbors(embedding: np.ndarray, vector_db: VectorDB) -> tuple:
    """Returns (class_probs: dict, situation: int, matched_examples: list[str])."""
    neighbors = vector_db.query(embedding, top_k=NEAR_MATCH_TOP_K)
    if not neighbors or neighbors[0][1] < NEAR_MATCH_THRESHOLD:
        return {}, 3, []

    # Soft k-NN vote: weight each neighbor's classes by its similarity, only
    # counting neighbors that individually clear the threshold.
    close = [(entry, sim) for entry, sim in neighbors if sim >= NEAR_MATCH_THRESHOLD]
    weight_sum = sum(sim for _, sim in close)
    class_probs: dict = {}
    for entry, sim in close:
        w = sim / weight_sum
        for cls in entry.classes:
            class_probs[cls] = class_probs.get(cls, 0.0) + w
    # Clip to [0,1] (a class could theoretically accumulate >1 if it appears
    # in every close neighbor with weights summing past 1 due to float error).
    class_probs = {c: min(1.0, p) for c, p in class_probs.items()}
    return class_probs, 2, [entry.text for entry, _ in close]


def _weak_global_hint(embedding: np.ndarray, vector_db: VectorDB) -> float:
    """Best cosine similarity to any GLOBAL/CONCEPT-classified example
    currently in `vector_db`, regardless of `NEAR_MATCH_THRESHOLD` -- a soft
    "does CLIP think this could plausibly be whole-image?" signal, not a
    classification. 0.0 if the DB has no such example yet."""
    if len(vector_db) == 0:
        return 0.0
    best = 0.0
    for entry, sim in vector_db.query(embedding, top_k=len(vector_db)):
        if entry.classes & _WHOLE_IMAGE_CLASSES:
            best = max(best, sim)
    return best


def analyze(
    token_table: list,
    embeddings: list,
    vector_db: VectorDB = None,
) -> SemanticAnalysisResult:
    """token_table: list[TokenTableEntry] (Phase 1 output, in prompt order).
    embeddings: list of per-entry embedding vectors (same length/order as
    token_table) — mean-pooled CLIP token embeddings for each entry's own
    `[start, end)` range, sliced from the SAME embedding tensor already
    computed for conditioning (zero extra text-encoder cost beyond what
    generation already pays; see `THEOREM_MATRIX_TREE_BUFFER.md` §3).
    vector_db: an existing database to extend (e.g. persisted across a
    session so common tags get progressively better cached); a fresh one is
    created if omitted.
    """
    if len(token_table) != len(embeddings):
        raise ValueError(
            f"token_table ({len(token_table)}) and embeddings ({len(embeddings)}) must be the same length"
        )
    n = len(token_table)
    if vector_db is None:
        vector_db = VectorDB()

    class_probabilities: list = []
    situations: list = []
    weak_global_hint: list = []

    for entry, emb in zip(token_table, embeddings):
        # Computed against the vector DB's state BEFORE this entry's own
        # (possible) addition below -- same causal ordering as the primary
        # near-match classifier, so an entry never trivially "resembles
        # itself".
        weak_global_hint.append(_weak_global_hint(emb, vector_db))

        if entry.classes:
            # Situation 1: exact rule-engine match.
            class_probabilities.append({c: 1.0 for c in entry.classes})
            situations.append(1)
            vector_db.add(entry.text, emb, entry.classes)
        else:
            probs, situation, matched = _classify_via_nearest_neighbors(emb, vector_db)
            class_probabilities.append(probs)
            situations.append(situation)
            if situation == 2:
                print(f"[SemanticAnalyzer] {entry.text!r} resembles known tag(s) "
                      f"{matched} -> class_probs={ {k: round(v, 3) for k, v in probs.items()} }")
            else:
                print(f"[SemanticAnalyzer] {entry.text!r} is novel/unknown — "
                      f"no class prior, relying on Matrix-Tree structure alone")

    # --- Build Matrix-Tree potentials theta[0..n, 0..n] ---
    theta = np.zeros((n + 1, n + 1), dtype=np.float64)
    norm_embs = [np.asarray(e, dtype=np.float64) for e in embeddings]
    norm_embs = [e / (np.linalg.norm(e) + 1e-12) for e in norm_embs]

    def object_prob(i: int) -> float:
        return class_probabilities[i].get("OBJECT", 0.0)

    for m in range(1, n + 1):
        theta[0, m] = W_ROOT_OBJECT * object_prob(m - 1)
        for h in range(1, n + 1):
            if h == m:
                continue
            sim = float(np.dot(norm_embs[h - 1], norm_embs[m - 1]))
            precedence = W_EDGE_PRECEDENCE * object_prob(h - 1) if h < m else 0.0
            theta[h, m] = W_EDGE_SIMILARITY * sim + precedence

    tree = compute_tree_marginals(theta)

    print(f"[SemanticAnalyzer] built Matrix-Tree over {n} segment(s): Z={tree.Z:.4g}")
    for m in range(1, n + 1):
        dist = tree.parent_distribution(m)
        best_h = max(dist, key=dist.get)
        best_label = "ROOT" if best_h == 0 else token_table[best_h - 1].text
        print(f"[SemanticAnalyzer]   parent({token_table[m-1].text!r}) most likely = "
              f"{best_label!r} (p={dist[best_h]:.3f})  full={ {('ROOT' if k==0 else token_table[k-1].text): round(v,3) for k,v in dist.items()} }")

    return SemanticAnalysisResult(
        class_probabilities=class_probabilities,
        situations=situations,
        tree=tree,
        vector_db=vector_db,
        weak_global_hint=weak_global_hint,
    )


def analyze_multi_chunk(token_table: list, embeddings: list, chunk_length: int = 75,
                         vector_db: VectorDB = None) -> dict:
    """Runs `analyze()` independently PER CHUNK (chunks are independent
    attention contexts in the real model — there is no cross-chunk edge to
    compute a Matrix-Tree over), while sharing ONE vector database across all
    chunks (a tag exact-matched in an earlier chunk can still near-match an
    unclassified tag in a later one). Returns {chunk_index:
    SemanticAnalysisResult}, keyed exactly like `chunk_mask.group_by_chunk`.
    """
    from modules.rule_engine.chunk_mask import compute_chunk_mask, group_by_chunk, print_chunk_mask_summary

    chunked = compute_chunk_mask(token_table, chunk_length=chunk_length)
    groups = group_by_chunk(chunked)
    print_chunk_mask_summary(chunked)

    if vector_db is None:
        vector_db = VectorDB()

    results = {}
    for chunk_index in sorted(groups):
        entries = [ce.entry for ce in groups[chunk_index]]
        idxs = [ce.index for ce in groups[chunk_index]]
        chunk_embeddings = [embeddings[i] for i in idxs]
        print(f"[SemanticAnalyzer] --- analyzing chunk {chunk_index} "
              f"({len(entries)} segment(s)) ---")
        results[chunk_index] = analyze(entries, chunk_embeddings, vector_db=vector_db)

    return results
