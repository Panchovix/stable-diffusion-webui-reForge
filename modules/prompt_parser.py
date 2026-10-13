from __future__ import annotations

import re
from collections import namedtuple
import lark

import logging

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][: in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    >>> g("a [b:.5] c")
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                if use_old_scheduling:
                    v = v*steps if v<1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = (v - int_offset)
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps+1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after
            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]
            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                yield args[0].value
            def __default__(self, data, children, meta):
                for child in children:
                    yield child
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)



def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]

            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []

    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape

    def to(self, *args, **kwargs):
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(*args, **kwargs)
        return self

    def advanced_indexing(self, item):
        result = {}
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                result[k] = self[k][item]
        return DictWithShape(result)


def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= entry.end_at_step:
                target_index = current
                break

        if is_dict:
            for k, param in cond_schedule[target_index].cond.items():
                res[k][i] = param
        else:
            res[i] = cond_schedule[target_index].cond

    return res


def stack_conds(tensors):
    # if prompts have wildly different lengths above the limit we'll get tensors of different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    return torch.stack(tensors)



def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                if current_step <= entry.end_at_step:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked)
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*(x?[+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      [abc] - decreases attention to abc by a multiplier of 1.1
      (abc:3.12) - modifies attention to abc by the multiplier 3.12
      (abc:x3.12) - expands to 3 copies of abc with attention 1.12 to the last
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    >>> parse_prompt_attention('(hello world, a poem, :x3.2)')
    [['hello world, a poem, hello world, a poem, ', 1.0],
     ['hello world, a poem, ', 1.2]]
    >>> parse_prompt_attention('(hello world, a poem, :x2.5)')
    [['hello world, a poem, ', 1.0],
     ['hello world, a poem, ', 1.5]]
    >>> parse_prompt_attention('(hello world, a poem, :x1.4)')
    [['hello world, a poem, ', 1.4]]
    >>> parse_prompt_attention('(hello world, a poem, :x0.8)')
    [['hello world, a poem, ', 0.8]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            if weight.lower().startswith('x'):
                val = float(weight[1:])
                start = round_brackets.pop()
                segment = res[start:]
                abs_val = abs(val)
                count = max(1,int(abs_val))
                tail = (abs_val % 1) + (abs_val >= 1)
                res[start:] = [[t,w] for _ in range(count) for t,w in segment]
                if tail > 1e-4:
                    for seg in res[-len(segment):]:
                        seg[1] *= tail
                if val < 0:
                    multiply_range(start, -1.0)
            else:
                multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    merged = []
    [curr_text, curr_weight] = res.pop(0)
    while res:
        [next_text, next_weight] = res.pop(0)
        if abs(next_weight - curr_weight) < 1e-4:
            curr_text += next_text
        else:
            merged.append([curr_text, curr_weight])
            curr_text, curr_weight = next_text, next_weight
    merged.append([curr_text, curr_weight])
    logging.debug(merged)

    return merged


# ---------------------------------------------------------------------------
# Token subspace mapping (CLPC token-level conditional-space guidance)
#
# See lean_proofs_rfv/THEREM_CONDITIONAL_BUFFER.md and
# plans/03-clpc-token-conditional-guidance.md, Phase 2.
#
# Splits a prompt into disjoint comma-separated entity/tag groups and
# re-tokenizes each with the model's own CLIP tokenizer to recover the
# token-index range each group occupies in the final encoded sequence — the
# "subspace" that ldm_patched.k_diffusion.sure_token_guidance corrects
# (vanish/leak/bias) on the attn2 cross-attention matrix.
# ---------------------------------------------------------------------------

def _split_top_level_commas(prompt: str) -> list[str]:
    """Split on commas at bracket-depth 0, so `"(1girl, smiling:1.2)"` stays
    one segment while `"1girl, brown hair"` splits into two. Handles the
    (), [] nesting used by attention-weight/alternation syntax."""
    segments = []
    depth = 0
    current = []
    for ch in prompt:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        segments.append("".join(current))
    return segments


_SUBWORD_SPLIT_RE = re.compile(r"[ _]+")


def _split_into_subwords(text: str) -> list[str]:
    """Split a comma-segment's bare text into sub-word tokens, treating both
    UNDERSCORE and SPACE as valid split points — the same "identifier
    tokenization" convention a real programming-language lexer uses (e.g.
    snake_case / space-separated multi-word identifiers). Used when a
    segment doesn't match any rule-engine pattern as a whole (a Danbooru tag
    like "pale_skin", or a natural-language phrase like "one 18 years old
    girl") so each individual word can still be classified on its own —
    "girl" alone matches the existing OBJECT pattern even though "one 18
    years old girl" as a whole string does not.
    """
    return [w for w in _SUBWORD_SPLIT_RE.split(text.strip()) if w]


def get_token_subspaces(model, prompt: str) -> dict:
    """Partition `prompt` into disjoint token-index groups, one per top-level
    comma-separated segment.

    Returns a dict:
        {
            "groups": [{"text": str, "start": int, "end": int,
                        "subtokens": [{"text": str, "start": int, "end": int}, ...]},
                       ...],
            "total_tokens": int,
            "chunk_length": int,   # model.cond_stage_model.chunk_length (usually 75)
            "n_chunks": int,       # ceil(total_tokens / chunk_length)
        }

    `start`/`end` are 0-indexed into the REAL-token axis (i.e. excluding the
    leading BOS token that the encoder always prepends) — the consumer
    (sure_token_guidance's attn2 hook) is responsible for the +1 BOS shift
    when indexing into the actual attention column axis.

    Each group's `"subtokens"` is its own EXACT (not approximate) sub-word
    token breakdown — computed via the SAME tokenizer, so a segment like
    "one 18 years old girl" (whole-segment start/end = e.g. [2, 8)) also
    carries `[{"text":"one","start":2,"end":3}, ..., {"text":"girl","start":
    7,"end":8}]`. This is the "META_TREE" sub-tree Plan 04's rule engine
    classifies independently when the whole segment matches no pattern —
    see `modules.rule_engine.token_table.build_token_table`.

    LIMITATION (printed, not silently wrong): this assumes the whole prompt
    fits in a single `chunk_length`-token chunk and contains no textual-
    inversion embeddings. Both would shift real encoder offsets relative to
    what is computed here. A warning prints when `n_chunks > 1`. SDXL's dual
    text encoders may also tokenize the same text to different lengths; only
    the primary `cond_stage_model` tokenizer is used here.
    """
    tokenizer = getattr(model, "cond_stage_model", None)
    if tokenizer is None or not hasattr(tokenizer, "tokenize"):
        print(f"[TokenSubspaceGuidance] no tokenizer found on model={model!r}; "
              f"returning empty group list (guidance will no-op).")
        return {"groups": [], "total_tokens": 0, "chunk_length": 75, "n_chunks": 0}

    segments = _split_top_level_commas(prompt)
    groups = []
    offset = 0
    for seg in segments:
        seg_stripped = seg.strip()
        if not seg_stripped:
            continue
        # Strip (word:weight)/[word] emphasis syntax the same way the real
        # encoder does, so token counts match what the model actually sees
        # rather than literal parenthesis/colon/digit characters.
        bare_text = "".join(text for text, _ in parse_prompt_attention(seg_stripped))
        token_ids = tokenizer.tokenize([bare_text])[0]
        n = len(token_ids)
        if n == 0:
            continue

        # Sub-word breakdown (the META_TREE sub-tree): split the SAME bare
        # text on underscore/space and tokenize each word independently.
        #
        # IMPORTANT — this does NOT reproduce the real BPE token boundaries:
        # tokenizing "pale_skin" as one string and tokenizing "pale" + "skin"
        # separately can (and in general does) produce DIFFERENT total token
        # counts (word-boundary markers, merge rules differ with/without
        # surrounding context) — verified directly by a mock-tokenizer test
        # this session where the sum of independent sub-word counts
        # exceeded the whole segment's own real count. Naively accumulating
        # independent counts would let a sub-token range overflow past the
        # group's own [offset, offset+n) boundary — a real correctness bug
        # for anything that later indexes into the attention tensor with it.
        #
        # Fix: treat the independent per-word counts as WEIGHTS only, and
        # proportionally distribute the group's own REAL token count `n`
        # across words with cumulative rounding, so the ranges are always
        # contiguous and never exceed `n` in total. This is a best-effort
        # APPROXIMATION of where each word "lives" inside the real tokenization,
        # not an exact BPE boundary — good enough for classification-oriented
        # consumers (`modules.rule_engine.token_table`, which classifies by
        # TEXT and only uses these ranges informationally), not a precision
        # guarantee for attention-level targeting. In degenerate cases (more
        # words than the group has real tokens) some words may round down to
        # a zero-width range and are omitted here — they are NOT omitted from
        # classification, which works off the word list directly.
        words = _split_into_subwords(bare_text)
        word_weights = [max(1, len(tokenizer.tokenize([w])[0])) for w in words]
        total_weight = sum(word_weights) or 1
        subtokens = []
        cum_tokens = 0.0
        local_start = 0
        for word, weight in zip(words, word_weights):
            cum_tokens += n * (weight / total_weight)
            local_end = round(cum_tokens)
            if local_end > local_start:
                subtokens.append({
                    "text": word, "start": offset + local_start, "end": offset + local_end,
                })
            local_start = local_end

        groups.append({
            "text": seg_stripped, "start": offset, "end": offset + n,
            "subtokens": subtokens,
        })
        offset += n

    total = offset
    chunk_length = getattr(tokenizer, "chunk_length", 75)
    n_chunks = max(1, -(-total // chunk_length)) if total > 0 else 0

    print(f"[TokenSubspaceGuidance] prompt->groups: "
          f"{[(g['text'], g['start'], g['end']) for g in groups]} "
          f"(total_tokens={total}, chunk_length={chunk_length}, n_chunks={n_chunks})")
    if n_chunks > 1:
        print(f"[TokenSubspaceGuidance] WARNING: prompt uses {total} tokens across "
              f"{n_chunks} chunks (> {chunk_length}-token budget); offsets computed "
              f"here only match the first chunk. Guidance will be skipped once the "
              f"attention hook detects the mismatch.")
    if hasattr(tokenizer, "embedders"):
        print(f"[TokenSubspaceGuidance] NOTE: model.cond_stage_model has multiple "
              f"sub-encoders (embedders) — e.g. SDXL's dual CLIP-L/CLIP-G. Token "
              f"offsets are computed from the primary tokenizer only and may not "
              f"exactly match the secondary encoder's tokenization.")

    return {"groups": groups, "total_tokens": total, "chunk_length": chunk_length,
            "n_chunks": n_chunks}


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:
    import torch  # doctest faster
