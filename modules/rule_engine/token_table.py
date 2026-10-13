"""Rule-engine "token table" — the compiler-pipeline handoff artifact between
the rule engine (lexer stage) and the precedence/Matrix-Tree engines
(parser stages, Plan 04 Phases 2-3).

Mirrors a real compiler's token stream: each prompt segment (from
`modules.prompt_parser.get_token_subspaces`) becomes one `TokenTableEntry`
carrying its classified tag classes (empty = unclassified) plus its token
index range, so downstream stages know both WHAT a segment is (its classes)
and WHERE it lives in the actual attention column axis, without needing to
re-parse anything.

**Sub-word decomposition (META_TREE)**: a single comma-segment can be a
natural-language phrase rather than a bare Danbooru tag — "one 18 years old
girl" matches no whole-segment rule pattern, but the word "girl" alone
matches the existing OBJECT pattern. When a segment's WHOLE text matches no
pattern, this module splits it into sub-word tokens (underscore and space
are both valid split points, matching the same identifier-tokenization
convention a real programming-language lexer uses) and classifies EACH word
independently against the SAME compiled ruleset — extended to all 7 classes,
not a hardcoded keyword list — aggregating whatever classes are found back
onto the parent segment. The per-word breakdown itself is preserved as a
`MetaTreeNode`: a composite node that POINTS TO the sub-tree of classified
words a segment decomposes into, rather than discarding that structure once
the aggregate class set is computed.
"""

from __future__ import annotations

import dataclasses

from modules.prompt_parser import _split_into_subwords
from modules.rule_engine.schema import CompiledRuleSet


@dataclasses.dataclass
class SubTokenClassification:
    text: str          # one word of a decomposed segment, e.g. "girl"
    start: int         # approximate token-index range within the parent
    end: int           # segment (see prompt_parser.get_token_subspaces's
                        # "subtokens" docstring for why this is an
                        # approximation, not an exact BPE boundary)
    classes: frozenset  # this word's own classification (may be empty)


@dataclasses.dataclass
class MetaTreeNode:
    """A composite node: represents a comma-segment whose own text matched
    no rule-engine pattern as a whole, but which decomposes into classified
    sub-word tokens. The parent `TokenTableEntry.classes` is the UNION of
    every non-empty sub-token's classes found here — this node preserves the
    breakdown itself for anything downstream that wants to know WHICH word
    contributed WHICH class (e.g. a future sub-word-level attention
    correction), rather than only exposing the flattened aggregate."""
    subtokens: list  # list[SubTokenClassification]


@dataclasses.dataclass
class TokenTableEntry:
    text: str                 # raw comma-segment text, e.g. "brown hair"
    start: int                # token-index start (see prompt_parser.get_token_subspaces)
    end: int                   # token-index end (exclusive)
    classes: frozenset         # subset of schema.TAG_CLASSES; empty = unclassified
    needs_clip: bool           # True iff `classes` is empty — Matrix-Tree engine's job
    meta_tree: object = None    # Optional[MetaTreeNode] — set iff sub-word decomposition
                                 # found at least one classified word (see module docstring)

    @property
    def is_object(self) -> bool:
        return "OBJECT" in self.classes

    @property
    def is_meta(self) -> bool:
        return "META" in self.classes


def _decompose_and_classify(text: str, start: int, end: int, subtoken_ranges: dict,
                             ruleset: CompiledRuleSet):
    """Try sub-word classification for a segment whose whole text matched no
    pattern. Returns (aggregate_classes: frozenset, meta_tree: Optional[MetaTreeNode]).
    `subtoken_ranges`: {word_text: (start, end)} from
    `prompt_parser.get_token_subspaces`'s per-group "subtokens" list — a word
    missing from this dict (can happen in the rare case where the group has
    fewer real tokens than words to distribute, see that function's
    docstring) falls back to the WHOLE segment's own range rather than being
    dropped from classification entirely.
    """
    words = _split_into_subwords(text)
    if not words:
        return frozenset(), None

    subtoken_classifications = []
    union_classes: set = set()
    for word in words:
        word_classes = ruleset.classify(word)
        if word_classes:
            union_classes |= word_classes
        w_start, w_end = subtoken_ranges.get(word, (start, end))
        subtoken_classifications.append(
            SubTokenClassification(text=word, start=w_start, end=w_end, classes=word_classes)
        )

    if not union_classes:
        return frozenset(), None

    return frozenset(union_classes), MetaTreeNode(subtokens=subtoken_classifications)


def build_token_table(groups: list, ruleset: CompiledRuleSet) -> list:
    """groups: the `groups` list from `get_token_subspaces()`'s return dict
    (each a {"text","start","end","subtokens"} mapping — "subtokens" is
    optional/may be absent for backward compatibility with older callers).
    Returns a list[TokenTableEntry] in original prompt order."""
    table = []
    for g in groups:
        classes = ruleset.classify(g["text"])
        meta_tree = None

        if not classes:
            subtoken_ranges = {st["text"]: (st["start"], st["end"]) for st in g.get("subtokens", [])}
            classes, meta_tree = _decompose_and_classify(
                g["text"], g["start"], g["end"], subtoken_ranges, ruleset)
            if meta_tree is not None:
                breakdown = [(s.text, sorted(s.classes)) for s in meta_tree.subtokens if s.classes]
                print(f"[RuleEngine] {g['text']!r} matched no whole-segment pattern; "
                      f"decomposed into sub-words {[s.text for s in meta_tree.subtokens]} "
                      f"-> aggregate classes {sorted(classes)} (META_TREE: {breakdown})")

        table.append(TokenTableEntry(
            text=g["text"], start=g["start"], end=g["end"],
            classes=classes, needs_clip=(len(classes) == 0),
            meta_tree=meta_tree,
        ))

    unknown = [e.text for e in table if e.needs_clip]
    print(f"[RuleEngine] token table: {len(table)} segment(s), "
          f"{sum(1 for e in table if not e.needs_clip)} classified, "
          f"{len(unknown)} unknown (-> Matrix-Tree engine): {unknown}")
    for e in table:
        via = " (via META_TREE decomposition)" if e.meta_tree is not None else ""
        print(f"[RuleEngine]   {e.text!r:25} range=[{e.start},{e.end}) "
              f"classes={sorted(e.classes) if e.classes else '(unknown)'}{via}")

    return table
