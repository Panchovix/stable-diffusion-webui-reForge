"""Compiler-toolchain regex -> NFA -> DFA engine for the prompt rule engine.

Supports a practical regular-expression subset — literals, `.` (any char),
character classes `[abc]`/`[^abc]` (with ranges `[a-z]`), alternation `|`,
grouping `(...)`, escapes `\\x`, and postfix quantifiers `*`/`+`/`?` — compiled
via Thompson's construction (regex -> NFA) then subset/powerset construction
(NFA -> DFA).

Multiple named patterns are merged into ONE combined DFA (exactly how a lexer
generator like flex/re2c merges many token rules into a single automaton), so
classifying a tag against dozens of rules is a single linear-time DFA walk —
by construction there is no backtracking search and therefore no
catastrophic-backtracking failure mode, regardless of pattern count, pattern
complexity, or tag length. This is the "try all DFA implementation so we
won't fall into some weird backward situation" requirement.

Glob convenience: a pattern beginning with a bare `*` or `?` with no
preceding atom (as in the example `"*boy"`, matching `1boy`/`2boy`/etc.) is
treated as shorthand for a leading `.*`/`.` — i.e. `*boy` parses exactly as
`.*boy` would.

Matching is always a FULL match (anchored at both ends) against the whole
tag text, never a substring search — a pattern like `"boy"` matches only the
literal tag `"boy"`, not `"1boy"` (use `"*boy"` / `.*boy` for that).
"""

from __future__ import annotations

import dataclasses
import string
from typing import Optional

# Sentinel alphabet symbol standing for "any character not seen in any
# pattern and not in the base printable-ASCII alphabet" — lets `.` / negated
# character classes correctly match arbitrary Unicode tag text without
# needing an unbounded alphabet.
OTHER = "\x00__OTHER__"


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Epsilon:
    pass


@dataclasses.dataclass(frozen=True)
class Lit:
    ch: str


@dataclasses.dataclass(frozen=True)
class AnyChar:
    pass


@dataclasses.dataclass(frozen=True)
class CharClass:
    chars: frozenset
    negate: bool


_DIGIT_CHARS = frozenset(string.digits)
_WORD_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_SPACE_CHARS = frozenset(" \t\n\r\f\v")

# Perl-style shorthand character classes: \d \D \w \W \s \S. Recognized both
# as a standalone atom (`_parse_atom`) and inside a `[...]` bracket
# expression (`_parse_charclass`, positive membership only there — a
# negated shorthand like `[\D]` inside brackets is not supported, use
# `[^0-9]` directly instead).
_SHORTHAND_CLASSES = {
    "d": CharClass(_DIGIT_CHARS, False),
    "D": CharClass(_DIGIT_CHARS, True),
    "w": CharClass(_WORD_CHARS, False),
    "W": CharClass(_WORD_CHARS, True),
    "s": CharClass(_SPACE_CHARS, False),
    "S": CharClass(_SPACE_CHARS, True),
}


@dataclasses.dataclass(frozen=True)
class Concat:
    parts: tuple


@dataclasses.dataclass(frozen=True)
class Alt:
    options: tuple


@dataclasses.dataclass(frozen=True)
class Star:
    inner: object


@dataclasses.dataclass(frozen=True)
class Plus:
    inner: object


@dataclasses.dataclass(frozen=True)
class Opt:
    inner: object


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, pattern: str):
        self.s = pattern
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.s[self.i] if self.i < len(self.s) else None

    def advance(self) -> str:
        ch = self.s[self.i]
        self.i += 1
        return ch

    def parse(self):
        node = self._parse_alt()
        if self.i != len(self.s):
            raise ParseError(f"Unexpected character at position {self.i} in pattern {self.s!r}")
        return node

    def _parse_alt(self):
        options = [self._parse_concat()]
        while self.peek() == "|":
            self.advance()
            options.append(self._parse_concat())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def _parse_concat(self):
        parts = []
        while self.peek() is not None and self.peek() not in ("|", ")"):
            parts.append(self._parse_repeat())
        if not parts:
            return Epsilon()
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def _parse_repeat(self):
        atom = self._parse_atom()
        while self.peek() in ("*", "+", "?"):
            op = self.advance()
            if op == "*":
                atom = Star(atom)
            elif op == "+":
                atom = Plus(atom)
            else:
                atom = Opt(atom)
        return atom

    def _parse_atom(self):
        ch = self.peek()
        if ch is None:
            raise ParseError("Unexpected end of pattern")
        # Glob convenience: a leading quantifier with no atom yet means
        # "wrap an implicit any-char" (e.g. "*boy" == ".*boy").
        if ch in ("*", "+", "?"):
            self.advance()
            if ch == "*":
                return Star(AnyChar())
            if ch == "+":
                return Plus(AnyChar())
            return Opt(AnyChar())
        if ch == "(":
            self.advance()
            node = self._parse_alt()
            if self.peek() != ")":
                raise ParseError(f"Expected ')' in pattern {self.s!r}")
            self.advance()
            return node
        if ch == "[":
            return self._parse_charclass()
        if ch == ".":
            self.advance()
            return AnyChar()
        if ch == "\\":
            self.advance()
            if self.peek() is None:
                raise ParseError(f"Dangling escape in pattern {self.s!r}")
            esc = self.advance()
            shorthand = _SHORTHAND_CLASSES.get(esc)
            if shorthand is not None:
                return shorthand
            return Lit(esc)
        self.advance()
        return Lit(ch)

    def _parse_charclass(self):
        self.advance()  # consume '['
        negate = False
        if self.peek() == "^":
            negate = True
            self.advance()
        chars: set = set()
        first = True
        while self.peek() is not None and (self.peek() != "]" or first):
            first = False
            c1 = self.advance()
            if c1 == "\\":
                if self.peek() is None:
                    raise ParseError(f"Dangling escape in character class {self.s!r}")
                esc = self.advance()
                shorthand = _SHORTHAND_CLASSES.get(esc)
                if shorthand is not None:
                    if shorthand.negate:
                        raise ParseError(
                            f"Negated shorthand \\{esc} is not supported inside [...] "
                            f"in pattern {self.s!r} — use an explicit negated range instead"
                        )
                    chars |= shorthand.chars
                    continue
                c1 = esc
            if self.peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.advance()  # consume '-'
                c2 = self.advance()
                if c2 == "\\":
                    c2 = self.advance()
                for code in range(ord(c1), ord(c2) + 1):
                    chars.add(chr(code))
            else:
                chars.add(c1)
        if self.peek() != "]":
            raise ParseError(f"Unterminated character class in pattern {self.s!r}")
        self.advance()  # consume ']'
        return CharClass(frozenset(chars), negate)


def parse_pattern(pattern: str):
    return _Parser(pattern).parse()


def literal_chars_in(node) -> set:
    """Collect every literal character mentioned anywhere in the AST, so the
    working alphabet can be sized correctly before Thompson construction."""
    if isinstance(node, Lit):
        return {node.ch}
    if isinstance(node, CharClass):
        return set(node.chars)
    if isinstance(node, (Epsilon, AnyChar)):
        return set()
    if isinstance(node, Concat):
        out = set()
        for p in node.parts:
            out |= literal_chars_in(p)
        return out
    if isinstance(node, Alt):
        out = set()
        for o in node.options:
            out |= literal_chars_in(o)
        return out
    if isinstance(node, (Star, Plus, Opt)):
        return literal_chars_in(node.inner)
    raise TypeError(f"unknown AST node {node!r}")


# ---------------------------------------------------------------------------
# NFA (Thompson construction)
# ---------------------------------------------------------------------------

class NFA:
    def __init__(self):
        self.transitions: dict = {}  # state -> list[(symbols: frozenset|None, dst)]
        self.n_states = 0

    def new_state(self) -> int:
        s = self.n_states
        self.n_states += 1
        self.transitions[s] = []
        return s

    def add_edge(self, src: int, symbols, dst: int) -> None:
        self.transitions[src].append((symbols, dst))


def _thompson(node, nfa: NFA, alphabet: frozenset) -> tuple:
    """Returns (start_state, accept_state) for this AST fragment."""
    if isinstance(node, Epsilon):
        s, a = nfa.new_state(), nfa.new_state()
        nfa.add_edge(s, None, a)
        return s, a
    if isinstance(node, Lit):
        s, a = nfa.new_state(), nfa.new_state()
        nfa.add_edge(s, frozenset({node.ch}), a)
        return s, a
    if isinstance(node, AnyChar):
        s, a = nfa.new_state(), nfa.new_state()
        nfa.add_edge(s, alphabet, a)
        return s, a
    if isinstance(node, CharClass):
        s, a = nfa.new_state(), nfa.new_state()
        syms = (alphabet - node.chars) if node.negate else frozenset(node.chars)
        nfa.add_edge(s, syms, a)
        return s, a
    if isinstance(node, Concat):
        frags = [_thompson(p, nfa, alphabet) for p in node.parts]
        for (_, a1), (s2, _) in zip(frags, frags[1:]):
            nfa.add_edge(a1, None, s2)
        return frags[0][0], frags[-1][1]
    if isinstance(node, Alt):
        s, a = nfa.new_state(), nfa.new_state()
        for opt in node.options:
            os_, oa = _thompson(opt, nfa, alphabet)
            nfa.add_edge(s, None, os_)
            nfa.add_edge(oa, None, a)
        return s, a
    if isinstance(node, Star):
        s, a = nfa.new_state(), nfa.new_state()
        is_, ia = _thompson(node.inner, nfa, alphabet)
        nfa.add_edge(s, None, is_)
        nfa.add_edge(s, None, a)
        nfa.add_edge(ia, None, is_)
        nfa.add_edge(ia, None, a)
        return s, a
    if isinstance(node, Plus):
        s, a = nfa.new_state(), nfa.new_state()
        is_, ia = _thompson(node.inner, nfa, alphabet)
        nfa.add_edge(s, None, is_)
        nfa.add_edge(ia, None, is_)
        nfa.add_edge(ia, None, a)
        return s, a
    if isinstance(node, Opt):
        s, a = nfa.new_state(), nfa.new_state()
        is_, ia = _thompson(node.inner, nfa, alphabet)
        nfa.add_edge(s, None, is_)
        nfa.add_edge(s, None, a)
        nfa.add_edge(ia, None, a)
        return s, a
    raise TypeError(f"unknown AST node {node!r}")


# ---------------------------------------------------------------------------
# DFA (subset/powerset construction)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DFA:
    start: int = 0
    transitions: dict = dataclasses.field(default_factory=dict)  # state -> {symbol: state}
    accepts: dict = dataclasses.field(default_factory=dict)      # state -> frozenset[rule_id]
    n_states: int = 0


def _epsilon_closure(nfa: NFA, states: frozenset) -> frozenset:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for symbols, dst in nfa.transitions[s]:
            if symbols is None and dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return frozenset(closure)


def _move(nfa: NFA, states: frozenset, symbol: str) -> frozenset:
    result = set()
    for s in states:
        for symbols, dst in nfa.transitions[s]:
            if symbols is not None and symbol in symbols:
                result.add(dst)
    return frozenset(result)


def subset_construction(nfa: NFA, alphabet: frozenset, nfa_start: int,
                         accept_rule_of: dict) -> DFA:
    """accept_rule_of: nfa accept-state -> rule id (supports merging many
    patterns into one DFA; a DFA state's `accepts` set can contain multiple
    rule ids when several patterns simultaneously match the same text)."""
    dfa = DFA()
    state_map: dict = {}

    def get_or_create(nfa_set: frozenset) -> int:
        if nfa_set in state_map:
            return state_map[nfa_set]
        idx = dfa.n_states
        dfa.n_states += 1
        state_map[nfa_set] = idx
        dfa.transitions[idx] = {}
        dfa.accepts[idx] = frozenset(
            accept_rule_of[s] for s in nfa_set if s in accept_rule_of
        )
        return idx

    start_set = _epsilon_closure(nfa, frozenset({nfa_start}))
    dfa.start = get_or_create(start_set)
    worklist = [start_set]
    seen = {start_set}
    while worklist:
        cur = worklist.pop()
        cur_idx = state_map[cur]
        for sym in alphabet:
            nxt = _epsilon_closure(nfa, _move(nfa, cur, sym))
            if not nxt:
                continue  # dead state — omit; missing transition = implicit reject
            if nxt not in seen:
                seen.add(nxt)
                worklist.append(nxt)
            dfa.transitions[cur_idx][sym] = get_or_create(nxt)
    return dfa


def dfa_match(dfa: DFA, alphabet_specific: frozenset, text: str) -> frozenset:
    """Run the DFA on the FULL string `text` (anchored start+end). Returns the
    set of rule ids accepting at the final state, or an empty frozenset if
    `text` is rejected (no transition exists) or ends in a non-accepting
    state. O(len(text)) with exactly one transition lookup per character —
    no backtracking is possible by construction."""
    state = dfa.start
    for ch in text:
        sym = ch if ch in alphabet_specific else OTHER
        nxt = dfa.transitions.get(state, {}).get(sym)
        if nxt is None:
            return frozenset()
        state = nxt
    return dfa.accepts.get(state, frozenset())


# ---------------------------------------------------------------------------
# Multi-pattern compilation entry point
# ---------------------------------------------------------------------------

def compile_patterns(patterns: dict) -> tuple:
    """patterns: {rule_id: pattern_string}. Returns (dfa, alphabet_specific).

    Builds ONE merged NFA (a super-start state epsilon-connected to every
    pattern's own Thompson fragment, each pattern's accept state tagged with
    its rule id) and subset-constructs it into ONE DFA — the lexer-generator
    pattern of merging many token rules into a single automaton.
    """
    literal_chars: set = set()
    asts = {}
    for rule_id, pattern in patterns.items():
        ast = parse_pattern(pattern)
        asts[rule_id] = ast
        literal_chars |= literal_chars_in(ast)

    alphabet_specific = frozenset(literal_chars | set(string.printable))
    alphabet = alphabet_specific | {OTHER}

    nfa = NFA()
    super_start = nfa.new_state()
    accept_rule_of: dict = {}
    for rule_id, ast in asts.items():
        s, a = _thompson(ast, nfa, alphabet)
        nfa.add_edge(super_start, None, s)
        accept_rule_of[a] = rule_id

    dfa = subset_construction(nfa, alphabet, super_start, accept_rule_of)
    return dfa, alphabet_specific
