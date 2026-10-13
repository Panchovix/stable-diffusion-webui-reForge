"""Read the FINAL prompt text and the FINAL conditioning tensor the model
actually consumes — not a re-derivation from `p.prompt` (the raw user
input), which may still contain unexpanded style presets, and not a second,
independent encode call that risks diverging from what generation actually
uses.

`modules/processing.py` applies style presets into `p.all_prompts`
(`StableDiffusionProcessing.init`: `self.all_prompts =
[shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in
self.all_prompts]`), and `p.setup_conds()` computes the real conditioning
tensor into `p.c` (a `MulticondLearnedConditioning`, via
`prompt_parser.get_multicond_learned_conditioning`) BEFORE
`process_before_every_sampling` hooks run — so both are already sitting on
`p` by the time an extension script can read them.

Primary path: read `p.c` directly (this IS the tensor cross-attention will
use — zero extra encoder cost, and exactly "what the model sees"). Fallback:
if `p.c`'s structure doesn't match what's expected (untested against a live
pipeline this session — see CAVEAT), re-encode the SAME final prompt text
via `model.get_learned_conditioning` — still correct (same input text,
deterministic encoder) but not literally the same tensor object, and costs
one extra (cheap, one-time, non-NFE) text-encoder pass.

CAVEAT (unverified — no live WebUI/torch available in this sandbox): the
exact `p.c.batch[0][0].schedules[-1].cond` navigation below matches this
session's prior reading of `modules/prompt_parser.py`'s
`MulticondLearnedConditioning`/`ComposableScheduledPromptConditioning`/
`ScheduledPromptConditioning` structure, but has not been exercised against
a real `StableDiffusionProcessing` object. SDXL's dual CLIP-L/CLIP-G
encoders are an additional, separately-flagged risk (see
`modules/prompt_parser.py::get_token_subspaces`'s own SDXL caveat) — this
module does not attempt anything SDXL-specific beyond what already applies
there.
"""

from __future__ import annotations

import numpy as np


def get_final_prompt_text(p, batch_index: int = 0) -> str:
    """The prompt text AFTER style-preset expansion — what the model actually
    encodes, not `p.prompt` (the raw user input before styles are applied)."""
    all_prompts = getattr(p, "all_prompts", None)
    if all_prompts:
        return all_prompts[batch_index if batch_index < len(all_prompts) else 0]
    print("[RuleEngine] WARNING: p.all_prompts not found; falling back to p.prompt "
          "(style presets may not be reflected)")
    return getattr(p, "prompt", "")


def _unwrap_cond(cond):
    """`cond` (from p.c's schedule entry, OR the return of
    model.get_learned_conditioning) is not always a bare tensor. Confirmed
    this session, via a real live-WebUI crash (`AttributeError: 'dict'
    object has no attribute 'dim'`) on an SDXL-family model: SDXL-style
    conditioning is a dict `{"crossattn": tensor, "vector": pooled_tensor}`
    (see `modules_forge/forge_sampler.py::cond_from_a1111_to_patched_ldm`,
    which does exactly this unwrap on the ComfyUI/Forge side). Some
    `get_learned_conditioning` call paths instead return a `(cond, pooled)`
    tuple. Unwrap both forms down to the plain (B, T, C) cross-attention
    tensor `_apply`/`slice_segment_embedding` actually needs."""
    if isinstance(cond, tuple):
        print(f"[RuleEngine] conditioning is a (cond, pooled) tuple — unwrapping to cond only.")
        cond = cond[0]
    if isinstance(cond, dict):
        available = list(cond.keys())
        cond = cond.get("crossattn", cond.get("cross_attn"))
        print(f"[RuleEngine] conditioning is a dict (keys={available}, likely SDXL-style: "
              f"crossattn+pooled vector) — unwrapped to the crossattn tensor "
              f"{'OK' if cond is not None else 'FAILED, no crossattn/cross_attn key found'}.")
    return cond


def get_final_conditioning_tensor(p, batch_index: int = 0):
    """Try to read the ALREADY-COMPUTED conditioning tensor straight off `p.c`
    (a MulticondLearnedConditioning) — this is literally the tensor the
    cross-attention layers will consume. Returns None (not an exception) if
    the expected structure isn't there, so the caller can fall back to
    re-encoding instead of crashing the whole generation."""
    c = getattr(p, "c", None)
    if c is None:
        print("[RuleEngine] p.c not yet populated (setup_conds() hasn't run?) "
              "— will need to re-encode instead of reading the live tensor.")
        return None
    try:
        composable_segment = c.batch[batch_index][0]   # first "AND"-segment
        cond = composable_segment.schedules[-1].cond     # last [a:b:step] schedule entry
        cond = _unwrap_cond(cond)
        if cond is None:
            print(f"[RuleEngine] p.c's schedule entry did not contain a usable "
                  f"crossattn tensor (dict without 'crossattn'/'cross_attn' key?) "
                  f"— falling back to re-encoding.")
            return None
        print(f"[RuleEngine] read the LIVE conditioning tensor from p.c directly "
              f"(zero extra encoder cost): shape={tuple(cond.shape)} — compare this "
              f"tensor's sequence length against the Nk the attn2 hook reports below.")
        return cond
    except (AttributeError, IndexError, TypeError) as e:
        print(f"[RuleEngine] Could not navigate p.c's structure as expected "
              f"(p.c.batch[{batch_index}][0].schedules[-1].cond): {e}. "
              f"Falling back to re-encoding the final prompt text directly.")
        return None


def slice_segment_embedding(cond_tensor, chunk_index: int, local_start: int, local_end: int,
                             chunk_width: int = 77):
    """Mean-pool the rows of `cond_tensor` (shape (T, C) or (1, T, C)) that
    belong to one segment's own token range WITHIN its own chunk — `+1` for
    the leading BOS column of that chunk, matching the convention already
    established in `sure_token_guidance.py`. Returns a plain (C,) numpy array
    on CPU."""
    if cond_tensor.dim() == 3:
        cond_tensor = cond_tensor[0]  # (T, C)
    bos = 1
    col_start = chunk_index * chunk_width + bos + local_start
    col_end = chunk_index * chunk_width + bos + local_end
    segment_rows = cond_tensor[col_start:col_end]
    pooled = segment_rows.float().mean(dim=0)
    return pooled.detach().cpu().numpy()


def extract_all_segment_embeddings(cond_tensor, chunked_entries: list, chunk_width: int = 77) -> list:
    """chunked_entries: list[ChunkedEntry] from `chunk_mask.compute_chunk_mask`,
    in original token_table order. Returns one (C,) numpy embedding per
    entry, in the same order."""
    return [
        slice_segment_embedding(cond_tensor, ce.chunk_index, ce.local_start, ce.local_end, chunk_width)
        for ce in chunked_entries
    ]


def get_prompt_embeddings_for_pipeline(model, p, chunked_entries: list, chunk_width: int = 77,
                                        batch_index: int = 0) -> list:
    """Top-level entry point: get the model's ACTUAL conditioning tensor if
    possible (zero extra cost), else fall back to re-encoding the correct
    FINAL (style-expanded) prompt text (`get_final_prompt_text`) — never the
    raw `p.prompt`. Returns one (C,) numpy embedding per `chunked_entries`
    item, in order."""
    cond_tensor = get_final_conditioning_tensor(p, batch_index=batch_index)
    if cond_tensor is None:
        final_text = get_final_prompt_text(p, batch_index=batch_index)
        print(f"[RuleEngine] FALLBACK PATH: re-encoding final prompt text for embedding "
              f"extraction (costs one extra, one-time, non-NFE text-encoder pass): {final_text!r}")
        cond_tensor = _unwrap_cond(model.get_learned_conditioning([final_text]))
        print(f"[RuleEngine] fallback re-encode produced shape={tuple(cond_tensor.shape)}")

    return extract_all_segment_embeddings(cond_tensor, chunked_entries, chunk_width=chunk_width)
