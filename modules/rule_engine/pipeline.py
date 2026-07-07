"""Top-level orchestration: prompt -> rule engine -> semantic analyzer ->
refine engine -> the "intention matrix tree" that Phase 5 wires into the
sampler's attention correction.

Reads the prompt and conditioning the model will ACTUALLY use (final,
style-expanded prompt text via `p.all_prompts`; the real conditioning tensor
via `p.c`, falling back to a same-text re-encode only if that's not
available) — never a re-derivation from the raw, pre-style `p.prompt` — see
`embedding_extraction.py`'s module docstring for the full rationale.
"""

from __future__ import annotations

from modules.prompt_parser import get_token_subspaces
from modules.rule_engine.chunk_mask import compute_chunk_mask
from modules.rule_engine.embedding_extraction import get_final_prompt_text, get_prompt_embeddings_for_pipeline
from modules.rule_engine.refine_engine import refine
from modules.rule_engine.schema import CompiledRuleSet
from modules.rule_engine.semantic_analyzer import analyze_multi_chunk
from modules.rule_engine.token_table import build_token_table


def run_pipeline(model, p, ruleset: CompiledRuleSet, chunk_length: int = 75,
                  batch_index: int = 0, vector_db=None):
    """Returns (final_tokens: list[FinalTokenInfo], groups_info: dict) —
    `groups_info` is `get_token_subspaces`'s own return value (needed by the
    attn2 hook for chunk-count/Nk validation, same as Plan 03), and
    `final_tokens` is the reconciled intention-tree output ready for
    `sure_token_guidance.py` to consume in place of its keyword-heuristic
    clustering.

    Returns (None, groups_info) if the prompt has no groups at all (nothing
    to analyze) — callers should treat that as "guidance no-ops this run",
    matching the existing Plan 03 convention.
    """
    final_prompt = get_final_prompt_text(p, batch_index=batch_index)
    groups_info = get_token_subspaces(model, final_prompt)
    if not groups_info["groups"]:
        return None, groups_info

    token_table = build_token_table(groups_info["groups"], ruleset)
    chunked_entries = compute_chunk_mask(token_table, chunk_length=chunk_length)

    embeddings = get_prompt_embeddings_for_pipeline(
        model, p, chunked_entries, chunk_width=chunk_length + 2, batch_index=batch_index,
    )

    semantic_results = analyze_multi_chunk(
        token_table, embeddings, chunk_length=chunk_length, vector_db=vector_db,
    )
    final_tokens = refine(token_table, semantic_results, chunk_length=chunk_length)

    return final_tokens, groups_info
