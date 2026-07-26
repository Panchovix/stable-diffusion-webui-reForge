"""YAML-configurable rule schema for the prompt rule engine.

Rule files map a PATTERN (this engine's regex/glob subset — see
`regex_dfa.py`) to a list of CLASSES describing what that tag/keyword means:

    "\\d*(girl|boy|other)s?": [OBJECT]
    "*_hair":                 [ADJ]
    "highres":                [META]

This mirrors a compiler's lexer-generator input (token pattern -> token
class), and multiple rule files can be composed (e.g. a base Danbooru
ruleset plus a Pony- or Illustrious-specific overlay) — patterns are merged
by union of their class lists if the same literal pattern string appears in
more than one file.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Optional

from modules.rule_engine import regex_dfa

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_tag_text(text: str) -> str:
    """Danbooru's canonical tag form uses underscores ("brown_hair"), but
    prompts are typically typed with spaces ("brown hair") — these must
    match the same rule. Collapse any run of whitespace to a single
    underscore (matching how tag-autocomplete tools normalize input against
    a Danbooru-style tag database) before lowercasing/trimming. Rule
    patterns should therefore always be written in underscore form."""
    return _WHITESPACE_RUN.sub("_", text.strip()).lower()


# The eight tag classes. Classes are NOT mutually exclusive — a single tag
# can carry more than one (e.g. "comic" is both GLOBAL and CONCEPT; "1girl"
# is both OBJECT and PRIMARY_OBJ).
TAG_CLASSES = frozenset({
    "META",         # quality/technical tags; global reference, no image-region effect
    "OBJECT",       # entity/subject tags ("1girl", "1boy", "cat", "sword")
    "PRIMARY_OBJ",  # a subset of OBJECT: a main character/subject that always
                    # independently appears in the image and is NEVER subordinate to
                    # another entity ("1girl", "1boy", "multiple_girls") -- distinct
                    # from a plain OBJECT tag ("cat", "sword") which CAN belong to a
                    # PRIMARY_OBJ (e.g. the sword the girl holds). This distinction
                    # exists because the Matrix-Tree's single-root formulation
                    # (matrix_tree.py) allows only ONE node to attach directly to
                    # ROOT, so without it two co-equal subjects ("1girl", "1boy")
                    # would otherwise get structurally forced into a false
                    # parent-child relationship with each other purely because the
                    # tree needs SOME single-root spanning structure — see
                    # refine_engine.py's PRIMARY_OBJ reconsideration branch.
    "ADJ",          # attribute tags describing a specific OBJECT's condition
    "VERB",         # action tags; may tie multiple/specific OBJECTs to a region
    "GLOBAL",       # whole-image effect (composition, lighting, setting)
    "CONCEPT",      # whole-image style/theme (may overlay with GLOBAL)
    "LORAACT",      # LoRA activation/trigger keyword
})


@dataclasses.dataclass
class CompiledRuleSet:
    dfa: regex_dfa.DFA
    alphabet_specific: frozenset
    rule_classes: dict          # rule_id -> frozenset[str] (subset of TAG_CLASSES)
    rule_patterns: dict         # rule_id -> original pattern string (debug/display)

    def classify(self, text: str) -> frozenset:
        """Classify a single tag/segment's text. Returns the union of classes
        from every pattern that matches (full match against the normalized
        text — see `_normalize_tag_text`). An empty result means "unknown" —
        this tag falls through to the Matrix-Tree engine (Plan 04 Phase 3)."""
        matched_rule_ids = regex_dfa.dfa_match(
            self.dfa, self.alphabet_specific, _normalize_tag_text(text))
        classes: set = set()
        for rule_id in matched_rule_ids:
            classes |= self.rule_classes[rule_id]
        return frozenset(classes)

    def matched_patterns(self, text: str) -> list:
        """Debug helper: which raw pattern strings matched this text."""
        matched_rule_ids = regex_dfa.dfa_match(
            self.dfa, self.alphabet_specific, _normalize_tag_text(text))
        return [self.rule_patterns[rid] for rid in matched_rule_ids]


class RuleSchemaError(ValueError):
    pass


def compile_ruleset(patterns: dict) -> CompiledRuleSet:
    """patterns: {pattern_string: list[str] of TAG_CLASSES}. Duplicate pattern
    strings across merged files should already be unioned by the caller
    (`load_rule_yaml`) before reaching here."""
    rule_classes: dict = {}
    rule_patterns: dict = {}
    dfa_patterns: dict = {}
    for rule_id, (pattern, classes) in enumerate(patterns.items()):
        bad = set(classes) - TAG_CLASSES
        if bad:
            raise RuleSchemaError(
                f"Unknown class(es) {sorted(bad)} for pattern {pattern!r}; "
                f"valid classes are {sorted(TAG_CLASSES)}"
            )
        rule_classes[rule_id] = frozenset(classes)
        rule_patterns[rule_id] = pattern
        dfa_patterns[rule_id] = pattern

    try:
        dfa, alphabet_specific = regex_dfa.compile_patterns(dfa_patterns)
    except regex_dfa.ParseError as e:
        raise RuleSchemaError(f"Failed to compile rule pattern: {e}") from e

    return CompiledRuleSet(
        dfa=dfa, alphabet_specific=alphabet_specific,
        rule_classes=rule_classes, rule_patterns=rule_patterns,
    )


def load_rule_yaml(*paths: str) -> CompiledRuleSet:
    """Load and merge one or more YAML rule files into a single compiled
    ruleset. If the same pattern string appears in multiple files, their
    class lists are unioned (later files add classes, never remove them)."""
    import yaml

    merged: dict = {}
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Rule file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise RuleSchemaError(f"Rule file {path!r} must be a YAML mapping of pattern -> classes")
        for pattern, classes in data.items():
            if isinstance(classes, str):
                classes = [classes]
            existing = merged.get(pattern, [])
            merged[pattern] = list(dict.fromkeys(existing + list(classes)))  # union, order-stable

    print(f"[RuleEngine] loaded {len(merged)} pattern(s) from {len(paths)} file(s): "
          f"{[os.path.basename(p) for p in paths]}")
    return compile_ruleset(merged)
