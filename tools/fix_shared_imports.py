"""
LibCST codemod: fix_shared_imports

Replaces direct imports of `opts` and `state` from `modules.shared` with
dynamic attribute access via `shared.opts` / `shared.state`.

These names are set to None at module-definition time and are only assigned
during `shared_init.initialize()`.  Importing them at module load time captures
the initial None and causes AttributeError at runtime.

Usage (dry-run, show diffs):
    python tools/fix_shared_imports.py --dry-run

Usage (apply fixes in-place):
    python tools/fix_shared_imports.py

Optionally restrict to specific files or directories:
    python tools/fix_shared_imports.py --dry-run modules/sd_samplers_kdiffusion.py
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Sequence, Set, Union

import libcst as cst

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAZY_NAMES: Set[str] = {"opts", "state"}
SHARED_MODULE = "modules.shared"
SHARED_ALIAS = "shared"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dotted(node: Union[cst.Name, cst.Attribute]) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted(node.value)}.{node.attr.value}"
    return ""


def _alias_import_name(alias: cst.ImportAlias) -> str:
    return _dotted(alias.name)  # type: ignore[arg-type]


def _is_from_shared(node: cst.ImportFrom) -> bool:
    return node.module is not None and _dotted(node.module) == SHARED_MODULE  # type: ignore[arg-type]


def _make_shared_alias_line() -> cst.SimpleStatementLine:
    """Return an AST line for `import modules.shared as shared`."""
    return cst.SimpleStatementLine(
        body=[
            cst.Import(
                names=[
                    cst.ImportAlias(
                        name=cst.Attribute(
                            value=cst.Name("modules"),
                            attr=cst.Name("shared"),
                        ),
                        asname=cst.AsName(
                            whitespace_before_as=cst.SimpleWhitespace(" "),
                            whitespace_after_as=cst.SimpleWhitespace(" "),
                            name=cst.Name(SHARED_ALIAS),
                        ),
                    )
                ]
            )
        ],
        leading_lines=[],
    )


# ---------------------------------------------------------------------------
# Pass 1 – read-only CSTVisitor: collect facts about the module
# ---------------------------------------------------------------------------

class _GatherInfo(cst.CSTVisitor):
    """
    Gathers two facts before the transformer runs:
    - which LAZY_NAMES are imported from modules.shared
    - whether `import modules.shared as shared` already exists
    """

    def __init__(self) -> None:
        self.lazy_names_found: Set[str] = set()
        self.has_shared_alias: bool = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if not _is_from_shared(node):
            return
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            name = _alias_import_name(alias)
            if name in LAZY_NAMES:
                self.lazy_names_found.add(name)

    def visit_Import(self, node: cst.Import) -> None:
        if not isinstance(node.names, (list, tuple)):
            return
        for alias in node.names:
            if (
                isinstance(alias, cst.ImportAlias)
                and isinstance(alias.name, (cst.Name, cst.Attribute))
                and _dotted(alias.name) == SHARED_MODULE  # type: ignore[arg-type]
                and isinstance(alias.asname, cst.AsName)
                and isinstance(alias.asname.name, cst.Name)
                and alias.asname.name.value == SHARED_ALIAS
            ):
                self.has_shared_alias = True


# ---------------------------------------------------------------------------
# Pass 2 – CSTTransformer: rewrite the module
# ---------------------------------------------------------------------------

class _FixSharedImports(cst.CSTTransformer):
    """
    Receives pre-collected facts from _GatherInfo so every decision is
    already known before any leave_* method runs.

    Changes made:
    1. Removes `opts`/`state` from `from modules.shared import …`
       (drops entire line if no names remain)
    2. Injects `import modules.shared as shared` right after that line
       when one does not already exist
    3. Rewrites bare `opts.X` → `shared.opts.X` (same for `state`)
    """

    def __init__(self, names_to_fix: Set[str], has_shared_alias: bool) -> None:
        self._names_to_fix = names_to_fix
        self._inject = not has_shared_alias   # True iff we must add the import
        self._injected = False                # prevent injecting more than once

    # ------------------------------------------------------------------
    # 1 + 2: strip lazy names; inject alias import after the same line
    # ------------------------------------------------------------------

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> Union[cst.ImportFrom, cst.RemovalSentinel]:
        if not _is_from_shared(updated_node):
            return updated_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node

        keep = [
            a for a in updated_node.names
            if _alias_import_name(a) not in self._names_to_fix
        ]
        if len(keep) == len(updated_node.names):
            return updated_node  # nothing removed

        if not keep:
            # All names were lazy – remove the entire ImportFrom statement.
            # The enclosing SimpleStatementLine will become empty and be
            # handled in leave_SimpleStatementLine.
            return cst.RemoveFromParent()  # type: ignore[return-value]

        # Strip trailing comma from last kept name
        fixed = [
            a if i < len(keep) - 1 else a.with_changes(comma=cst.MaybeSentinel.DEFAULT)
            for i, a in enumerate(keep)
        ]
        return updated_node.with_changes(names=fixed)

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> Union[
        cst.SimpleStatementLine,
        cst.FlattenSentinel[cst.SimpleStatementLine],
        cst.RemovalSentinel,
    ]:
        # Only act on the line that originally contained the from-shared import.
        if not self._inject or self._injected:
            return updated_node

        had_from_shared = any(
            isinstance(stmt, cst.ImportFrom) and _is_from_shared(stmt)
            for stmt in original_node.body
        )
        if not had_from_shared:
            return updated_node

        self._injected = True

        # If leave_ImportFrom removed all names the body is now empty,
        # which means the SimpleStatementLine itself would be removed by
        # LibCST; we replace it with just the alias import.
        # Otherwise, keep the (modified) line and append the alias import.
        if not updated_node.body:
            return _make_shared_alias_line()

        return cst.FlattenSentinel([updated_node, _make_shared_alias_line()])

    # ------------------------------------------------------------------
    # 3: rewrite opts.X → shared.opts.X, state.X → shared.state.X
    # ------------------------------------------------------------------

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        # Only rewrite when the direct value is a bare Name like `opts` or `state`.
        # Already-qualified `shared.opts.X` has value=Attribute, not Name, so
        # this guard prevents double-wrapping.
        if not isinstance(updated_node.value, cst.Name):
            return updated_node
        name = updated_node.value.value
        if name not in self._names_to_fix:
            return updated_node
        # opts.attr  →  shared.opts.attr
        return updated_node.with_changes(
            value=cst.Attribute(
                value=cst.Name(SHARED_ALIAS),
                attr=cst.Name(name),
            )
        )


# ---------------------------------------------------------------------------
# File-level driver
# ---------------------------------------------------------------------------

def _quick_check(source: str) -> bool:
    """Cheap text-level gate; skips files that clearly don't need fixing."""
    if f"from {SHARED_MODULE} import" not in source:
        return False
    return any(
        f" {n}" in source or f",{n}" in source or f", {n}" in source
        for n in LAZY_NAMES
    )


def fix_file(path: Path, *, dry_run: bool = False) -> bool:
    """Returns True if the file was (or, in dry-run mode, would be) changed."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    if not _quick_check(source):
        return False

    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        print(f"  PARSE ERROR {path}: {exc}", file=sys.stderr)
        return False

    # Pass 1: gather facts
    info = _GatherInfo()
    tree.visit(info)

    if not info.lazy_names_found:
        return False

    # Pass 2: transform
    transformer = _FixSharedImports(info.lazy_names_found, info.has_shared_alias)
    new_tree = tree.visit(transformer)
    new_source = new_tree.code

    if new_source == source:
        return False

    if dry_run:
        diff = difflib.unified_diff(
            source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path) + "  [fixed]",
            n=2,
        )
        sys.stdout.writelines(diff)
        return True

    path.write_text(new_source, encoding="utf-8")
    print(f"  fixed: {path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect(roots: Sequence[str]) -> list[Path]:
    project = Path(__file__).resolve().parent.parent
    if roots:
        result: list[Path] = []
        for r in roots:
            p = Path(r)
            result.extend(p.rglob("*.py") if p.is_dir() else [p])
        return result
    return sorted(project.rglob("*.py"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="files/dirs to process (default: whole project)")
    ap.add_argument("--dry-run", action="store_true", help="print diffs without writing")
    args = ap.parse_args()

    files = _collect(args.files)
    changed = sum(fix_file(f, dry_run=args.dry_run) for f in files)
    verb = "Would change" if args.dry_run else "Changed"
    print(f"\n{verb} {changed} file(s).")


if __name__ == "__main__":
    main()
