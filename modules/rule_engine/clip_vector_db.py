"""A small in-memory "vector database" of CLIP-embedded keywords, used for
nearest-neighbor tag classification: exact rule-engine matches (Phase 1)
populate it as ground truth, and unclassified tokens query it to find "this
resembles a known tag" matches via cosine similarity — the unsupervised
nearest-neighbor classification step in the semantic analyzer (Phase 2).

Deliberately backend-agnostic (plain `numpy` arrays in, plain floats out):
the caller is responsible for producing embeddings from whichever text
encoder is active (SDXL's dual CLIP-L/CLIP-G or a single Transformer text
encoder) and handing them here as already-detached CPU arrays — this module
does no GPU/torch work itself, keeping the actual model's VRAM untouched by
this bookkeeping step (this is a handful of small vectors, cheap on CPU).
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class VectorEntry:
    text: str
    embedding: np.ndarray   # L2-normalized
    classes: frozenset


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


class VectorDB:
    """Grows over the course of classifying one prompt (or persists across a
    session if the caller keeps the same instance): every exact rule-engine
    match adds its embedding as a labeled example; every near-match query
    just reads, never writes (only ground truth grows the cache)."""

    def __init__(self):
        self._entries: list = []

    def add(self, text: str, embedding: np.ndarray, classes: frozenset) -> None:
        self._entries.append(VectorEntry(text=text, embedding=_normalize(np.asarray(embedding, dtype=np.float64)), classes=classes))

    def __len__(self) -> int:
        return len(self._entries)

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list:
        """Returns up to `top_k` (VectorEntry, cosine_similarity) pairs,
        sorted by similarity descending. Empty list if the DB has no
        entries yet."""
        if not self._entries:
            return []
        q = _normalize(np.asarray(embedding, dtype=np.float64))
        scored = [(entry, float(np.dot(q, entry.embedding))) for entry in self._entries]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
