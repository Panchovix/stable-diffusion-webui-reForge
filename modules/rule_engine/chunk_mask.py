"""Chunk masking for long prompts.

`modules.prompt_parser.get_token_subspaces()` computes each segment's token
range as one continuous count of real content tokens across the whole
prompt (documented limitation: it assumes everything fits in one
75-token CLIP chunk). In reality, CLIP's own chunking
(`modules/sd_hijack_clip.py::tokenize_line`) breaks the prompt into
`chunk_length`-token (75, i.e. 77 with BOS/EOS) windows, each becoming its
own independent attention context — `sure_token_guidance.py` already has to
special-case this (`Nk` a multiple of the chunk width) and, until now, only
ever corrected the FIRST chunk, silently ignoring the rest.

This module gives the rule-engine pipeline VISIBILITY into which logical
chunk each segment falls into (a "mask"), so a long, multi-chunk prompt's
full structure is still debuggable end-to-end, and so the semantic analyzer
can run one independent Matrix-Tree per chunk (chunks are independent
attention contexts in the real model — there is no cross-chunk edge to
compute) while still sharing one CLIP vector-database cache across all of
them (a tag learned in chunk 0 can still near-match an unclassified tag in
chunk 1).

CAVEAT: the exact chunk boundary CLIP produces can shift by a token or two
around comma breaks (`tokenize_line` prefers to break chunks at a comma
rather than mid-segment) — this `start // chunk_length` computation is a
debug-grade approximation of the real boundary, not a byte-exact
reproduction of `tokenize_line`'s own chunking decision. Good enough for
visibility/masking; not a substitute for fixing `get_token_subspaces` itself
to track real chunk boundaries, which remains a known gap (see Plan 04
Phase 5 TODO).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ChunkedEntry:
    index: int          # position in the original token_table
    entry: object        # the TokenTableEntry itself
    chunk_index: int     # which logical 75-token chunk this segment falls in
    local_start: int      # start offset WITHIN that chunk (0-based, real-token space)
    local_end: int


def compute_chunk_mask(token_table: list, chunk_length: int = 75) -> list:
    """Returns one ChunkedEntry per token_table entry, in original order."""
    out = []
    for i, entry in enumerate(token_table):
        chunk_index = entry.start // chunk_length
        local_start = entry.start % chunk_length
        local_end = local_start + (entry.end - entry.start)
        out.append(ChunkedEntry(
            index=i, entry=entry, chunk_index=chunk_index,
            local_start=local_start, local_end=local_end,
        ))
    return out


def group_by_chunk(chunked_entries: list) -> dict:
    """chunk_index -> list[ChunkedEntry], in original order within each chunk."""
    groups: dict = {}
    for ce in chunked_entries:
        groups.setdefault(ce.chunk_index, []).append(ce)
    return groups


def print_chunk_mask_summary(chunked_entries: list) -> None:
    groups = group_by_chunk(chunked_entries)
    n_chunks = len(groups)
    if n_chunks > 1:
        print(f"[RuleEngine] MASKING: prompt spans {n_chunks} logical chunks "
              f"(chunk_length=75 real tokens each) — each chunk is an independent "
              f"attention context; showing the full cross-chunk picture below.")
    for chunk_index in sorted(groups):
        texts = [ce.entry.text for ce in groups[chunk_index]]
        print(f"[RuleEngine] MASKING: chunk {chunk_index} contains {len(texts)} "
              f"segment(s): {texts}")
