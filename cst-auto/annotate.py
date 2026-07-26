"""
annotate.py — libcst-based automatic type annotation inference.

Infers missing type annotations from default parameter values and uniform
return statements, then writes them back into the source file.

Usage:
    python annotate.py [options] file_or_glob [...]

Options:
    --dry-run           Print diff but do not write files
    --check             Exit 1 if any file would be changed (CI mode)
    --modern            Use X | None, list[Any] instead of Optional[X], List[Any]
    --add-future        Inject `from __future__ import annotations` if absent
    --no-return         Skip return type inference
    --no-params         Skip parameter annotation
    --diff              Always print unified diff (even when writing)
    --summary           Print per-file change count summary
    --trace-imports     Follow imports to source files and verify class definitions
    --search-path PATH  Additional root directory to search for source files
                        (may be given multiple times; cwd is always searched)
"""

from __future__ import annotations

import argparse
import difflib
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParamChange:
    func_name: str
    param_name: str
    inferred_type: str
    import_origin: Optional[str] = None   # e.g. "torch.nn" for nn.GELU


@dataclass
class ReturnChange:
    func_name: str
    inferred_type: str


Change = Union[ParamChange, ReturnChange]


# ---------------------------------------------------------------------------
# Import tracing
# ---------------------------------------------------------------------------

@dataclass
class ImportRecord:
    """One imported name as it appears in the current file."""
    module: str          # dotted module path, e.g. "torch.nn"
    original_name: str   # name exported by that module, e.g. "GELU"
    local_name: str      # name used in this file, e.g. "nn" or "GELU"
    is_module: bool      # True for `import X` / `import X as Y`


def _node_to_dotted(node: cst.BaseExpression) -> str:
    """Recursively flatten an Attribute chain or Name to a dotted string."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        left = _node_to_dotted(node.value)
        return f"{left}.{node.attr.value}"
    return ""


class ImportTracer:
    """
    Builds a map of every name visible in a module's scope (from its imports)
    and optionally follows those imports to their source files to verify that
    a name is actually a class definition.
    """

    def __init__(
        self,
        tree: cst.Module,
        search_paths: Optional[list[Path]] = None,
        trace_enabled: bool = False,
    ) -> None:
        # local_name → ImportRecord
        self.records: dict[str, ImportRecord] = {}
        self._search_paths: list[Path] = search_paths or [Path(".")]
        self._trace_enabled = trace_enabled
        # Cache: fully-qualified name → True/False/None (None = unknown)
        self._class_cache: dict[str, Optional[bool]] = {}
        self._extract_imports(tree)

    # -- Import extraction --

    def _extract_imports(self, tree: cst.Module) -> None:
        for stmt in tree.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for item in stmt.body:
                if isinstance(item, cst.Import):
                    self._process_import(item)
                elif isinstance(item, cst.ImportFrom):
                    self._process_import_from(item)

    def _process_import(self, node: cst.Import) -> None:
        """Handle `import X`, `import X.Y`, `import X as Z`."""
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            module = _node_to_dotted(alias.name)
            if not module:
                continue
            if alias.asname:
                asname_node = alias.asname.name
                local = asname_node.value if isinstance(asname_node, cst.Name) else module
            else:
                # `import X.Y` → local name is just `X`
                local = module.split(".")[0]
            self.records[local] = ImportRecord(
                module=module,
                original_name=module,
                local_name=local,
                is_module=True,
            )

    def _process_import_from(self, node: cst.ImportFrom) -> None:
        """Handle `from X import Y`, `from X import Y as Z`."""
        if isinstance(node.names, cst.ImportStar):
            return
        if node.module is None:
            return
        module = _node_to_dotted(node.module)
        # Adjust for relative imports (leading dots)
        if node.relative:
            dots = "." * len(node.relative)
            module = dots + module if module else dots
        for alias in node.names:
            if not isinstance(alias.name, cst.Name):
                continue
            original = alias.name.value
            if alias.asname:
                asname_node = alias.asname.name
                local = asname_node.value if isinstance(asname_node, cst.Name) else original
            else:
                local = original
            self.records[local] = ImportRecord(
                module=module,
                original_name=original,
                local_name=local,
                is_module=False,
            )

    # -- Lookup helpers --

    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled

    def is_known(self, name: str) -> bool:
        """Return True if `name` was imported into this file."""
        return name in self.records

    def get_record(self, name: str) -> Optional[ImportRecord]:
        return self.records.get(name)

    def origin_of(self, local_name: str) -> Optional[str]:
        """Return the fully-qualified source module for a local name."""
        rec = self.records.get(local_name)
        if rec is None:
            return None
        if rec.is_module:
            return rec.module
        return rec.module

    # -- Deep tracing: resolve whether a name is a class --

    def is_class(self, dotted_name: str) -> Optional[bool]:
        """
        Try to confirm that `dotted_name` (as used in this file, e.g. "nn.GELU"
        or "SomeClass") refers to a class definition.

        Returns True  → confirmed class
                False → confirmed non-class
                None  → could not determine (file not found, parse error, etc.)

        Only does filesystem work when `trace_enabled=True`.
        """
        if not self._trace_enabled:
            return None
        if dotted_name in self._class_cache:
            return self._class_cache[dotted_name]

        result = self._resolve_class(dotted_name)
        self._class_cache[dotted_name] = result
        return result

    def _resolve_class(self, dotted_name: str) -> Optional[bool]:
        parts = dotted_name.split(".", 1)
        root = parts[0]
        rest = parts[1] if len(parts) > 1 else None

        rec = self.records.get(root)
        if rec is None:
            return None  # not imported — can't trace

        if rec.is_module:
            # `import torch.nn as nn` then `nn.GELU`
            # full module to search: rec.module + "." + rest (if any)
            if rest:
                target_module = rec.module + "." + rest.rsplit(".", 1)[0] if "." in rest else rec.module
                class_name = rest.rsplit(".", 1)[-1]
            else:
                # The name itself is a module — not a class in the usual sense
                return None
        else:
            # `from torch.nn import GELU` then `GELU`
            target_module = rec.module
            class_name = rec.original_name if rest is None else rest

        module_file = self._find_module_file(target_module)
        if module_file is None:
            return None

        try:
            src = module_file.read_text(encoding="utf-8")
            mod_tree = cst.parse_module(src)
        except Exception:
            return None

        for stmt in mod_tree.body:
            if isinstance(stmt, cst.ClassDef) and stmt.name.value == class_name:
                return True
        return False

    def _find_module_file(self, module: str) -> Optional[Path]:
        """Convert dotted module name to a .py file path."""
        if module.startswith("."):
            return None  # relative imports — skip
        parts = module.split(".")
        for base in self._search_paths:
            # e.g. modules/processing.py
            as_file = base.joinpath(*parts).with_suffix(".py")
            if as_file.is_file():
                return as_file
            # e.g. modules/processing/__init__.py
            as_pkg = base.joinpath(*parts, "__init__.py")
            if as_pkg.is_file():
                return as_pkg
        return None


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------

def _is_bytes_string(node: cst.SimpleString) -> bool:
    return node.value.startswith(("b'", 'b"', "B'", 'B"'))


def _looks_like_class(name: str, *, imported: bool = False) -> bool:
    """
    Heuristic: a name is likely a class if it starts with an uppercase letter.

    Without import confirmation we additionally reject ALL_CAPS names (e.g.
    ``DEFAULT``, ``MAX_SIZE``) since those are usually constants, not classes.
    When `imported=True` (the name is confirmed to come from an import statement)
    we allow acronym class names like ``GELU``, ``ReLU``, ``RNN``, ``BEiT``.
    """
    if not name:
        return False
    if not name[0].isupper():
        return False
    # ALL_CAPS without import context → assume constant, not class
    if name.isupper() and not imported:
        return False
    return True


def _build_dotted_name(node: cst.BaseExpression) -> Optional[str]:
    """
    Extract a dotted name string from a Name or Attribute chain.
    Returns None if the expression is something more complex (subscript, call, etc.).
    """
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        left = _build_dotted_name(node.value)
        if left is not None:
            return f"{left}.{node.attr.value}"
    return None


# Builtins already handled with their generic typing equivalents
_HANDLED_BUILTINS = frozenset({"set", "frozenset", "dict", "list", "tuple"})


def _infer_class_from_call(
    func: cst.BaseExpression,
    import_tracer: Optional[ImportTracer],
) -> Optional[str]:
    """
    Given the `func` part of a Call node, return a type string if the call
    looks like a class instantiation (e.g. `SomeClass(...)` or `nn.Module(...)`).

    Strategy:
    1. Build the dotted name of the callee.
    2. Check if the root name is known via import_tracer OR looks like a class
       (PascalCase heuristic).
    3. With --trace-imports, confirm via source file inspection.
    """
    dotted = _build_dotted_name(func)
    if dotted is None:
        return None

    root = dotted.split(".")[0]
    last = dotted.rsplit(".", 1)[-1]

    # Skip builtins that were already handled with generic type forms
    if dotted in _HANDLED_BUILTINS or root in _HANDLED_BUILTINS:
        return None

    # Deep-trace check: if enabled and we can confirm it's NOT a class, skip
    if import_tracer is not None and import_tracer.trace_enabled:
        verdict = import_tracer.is_class(dotted)
        if verdict is False:
            return None
        if verdict is True:
            return dotted

    # Heuristic: uppercase-starting last component → likely a class.
    # Pass imported=True when the root is a known import so acronym names
    # like GELU / RNN are not filtered out by the ALL_CAPS guard.
    root_imported = import_tracer is not None and import_tracer.is_known(root)
    if _looks_like_class(last, imported=root_imported):
        return dotted

    return None


def _infer_class_from_name(
    name: str,
    import_tracer: Optional[ImportTracer],
) -> Optional[str]:
    """
    A bare Name used as a default (e.g. `act_layer=nn.GELU` is Attribute, but
    `callback=MyClass` is Name).  Returns the class name if it's a class reference
    (not an instance), None otherwise.

    We require either trace confirmation OR (PascalCase AND known import).
    """
    is_imported = import_tracer is not None and import_tracer.is_known(name)
    if not _looks_like_class(name, imported=is_imported):
        return None

    if import_tracer is not None and not is_imported:
        return None  # import_tracer present but name not found → skip

    if import_tracer.trace_enabled:
        verdict = import_tracer.is_class(name)
        if verdict is False:
            return None
        # True or None → allow with annotation

    return name


def _infer_class_from_attr(
    node: cst.Attribute,
    import_tracer: Optional[ImportTracer],
) -> Optional[str]:
    """
    A dotted attribute used as a default, e.g. `act_layer=nn.GELU`.
    Returns the dotted name if it looks like a class reference.
    """
    dotted = _build_dotted_name(node)
    if dotted is None:
        return None

    last = dotted.rsplit(".", 1)[-1]
    root = dotted.split(".")[0]

    if import_tracer is not None and import_tracer.trace_enabled:
        verdict = import_tracer.is_class(dotted)
        if verdict is False:
            return None
        if verdict is True:
            return dotted

    root_imported = import_tracer is not None and import_tracer.is_known(root)
    if _looks_like_class(last, imported=root_imported):
        # Only emit if root is a known import (avoid random attribute chains)
        if import_tracer is None or root_imported:
            return dotted

    return None


def infer_type_from_default(
    node: cst.BaseExpression,
    modern: bool,
    import_tracer: Optional[ImportTracer] = None,
) -> tuple[Optional[str], set[str]]:
    """
    Return (type_string, needed_typing_imports) or (None, set()) if unknown.

    Inference sources (in priority order):
      1. Literal values  → int / float / str / bytes / bool / complex
      2. None literal    → Optional[Any] / Any | None
      3. Collection literals → List[Any] / Dict[…] / Tuple[…]
      4. set()/frozenset() calls
      5. Class instantiation calls: SomeClass(…) or module.Cls(…)  ← NEW
      6. Class references (bare): SomeClass or module.Cls          ← NEW
    """

    # ---- Primitives --------------------------------------------------------

    if isinstance(node, cst.Integer):
        return "int", set()

    if isinstance(node, cst.Float):
        return "float", set()

    if isinstance(node, cst.Imaginary):
        return "complex", set()

    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        inner_type, imports = infer_type_from_default(node.expression, modern, import_tracer)
        if inner_type in ("int", "float"):
            return inner_type, imports
        return None, set()

    if isinstance(node, cst.SimpleString):
        if _is_bytes_string(node):
            return "bytes", set()
        return "str", set()

    if isinstance(node, (cst.ConcatenatedString, cst.FormattedString)):
        return "str", set()

    # ---- None / bool -------------------------------------------------------

    if isinstance(node, cst.Name):
        if node.value in ("True", "False"):
            return "bool", set()
        if node.value == "None":
            if modern:
                return "Any | None", {"Any"}
            return "Optional[Any]", {"Optional", "Any"}

    # ---- Collections -------------------------------------------------------

    if isinstance(node, cst.List):
        return ("list[Any]", {"Any"}) if modern else ("List[Any]", {"List", "Any"})

    if isinstance(node, cst.Dict):
        return ("dict[str, Any]", {"Any"}) if modern else ("Dict[str, Any]", {"Dict", "Any"})

    if isinstance(node, cst.Tuple):
        return ("tuple[Any, ...]", {"Any"}) if modern else ("Tuple[Any, ...]", {"Tuple", "Any"})

    # ---- set() / frozenset() -----------------------------------------------

    if isinstance(node, cst.Call) and isinstance(node.func, cst.Name):
        if node.func.value == "set" and len(node.args) == 0:
            return ("set[Any]", {"Any"}) if modern else ("Set[Any]", {"Set", "Any"})
        if node.func.value == "frozenset" and len(node.args) == 0:
            return ("frozenset[Any]", {"Any"}) if modern else ("FrozenSet[Any]", {"FrozenSet", "Any"})

    # ---- Class instantiation: SomeClass(…) / module.Cls(…)  ---------------
    #
    # Default is a *call* → inferred type is the class itself (an instance).
    # No extra typing imports needed since the class is already imported
    # (it's used as a default value in the same file).

    if isinstance(node, cst.Call):
        class_name = _infer_class_from_call(node.func, import_tracer)
        if class_name is not None:
            return class_name, set()

    # ---- Class reference: SomeClass / module.Cls  --------------------------
    #
    # Default is the class object itself (not an instance), e.g. act_layer=nn.GELU.
    # Annotated as Type[SomeClass] (legacy) or type[SomeClass] (modern/3.9+).

    if isinstance(node, cst.Name) and node.value not in ("True", "False", "None"):
        class_name = _infer_class_from_name(node.value, import_tracer)
        if class_name is not None:
            if modern:
                return f"type[{class_name}]", set()
            return f"Type[{class_name}]", {"Type"}

    if isinstance(node, cst.Attribute):
        class_name = _infer_class_from_attr(node, import_tracer)
        if class_name is not None:
            if modern:
                return f"type[{class_name}]", set()
            return f"Type[{class_name}]", {"Type"}

    return None, set()


# ---------------------------------------------------------------------------
# Return type inference
# ---------------------------------------------------------------------------

def _infer_return_value_type(value: Optional[cst.BaseExpression]) -> Optional[str]:
    """Map a single return value node to a primitive type string."""
    if value is None:
        return "None"
    if isinstance(value, cst.Integer):
        return "int"
    if isinstance(value, cst.Float):
        return "float"
    if isinstance(value, cst.Imaginary):
        return "complex"
    if isinstance(value, cst.SimpleString):
        return "bytes" if _is_bytes_string(value) else "str"
    if isinstance(value, (cst.ConcatenatedString, cst.FormattedString)):
        return "str"
    if isinstance(value, cst.Name):
        if value.value in ("True", "False"):
            return "bool"
        if value.value == "None":
            return "None"
    return None  # unknown — prevents inference for this function


def infer_return_type(types: list[Optional[str]]) -> Optional[str]:
    """
    Return a single type string if all collected return types are uniform
    and known, else None.
    """
    if not types:
        return None
    if any(t is None for t in types):
        return None
    unique = set(types)
    if len(unique) == 1:
        return unique.pop()
    return None


# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------

def has_future_annotations(tree: cst.Module) -> bool:
    """Return True if the module has `from __future__ import annotations`."""
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.ImportFrom):
                continue
            if item.module is None:
                continue
            # flat: from __future__ import annotations
            if isinstance(item.module, cst.Name) and item.module.value == "__future__":
                if not isinstance(item.names, cst.ImportStar):
                    for alias in item.names:
                        if isinstance(alias.name, cst.Name) and alias.name.value == "annotations":
                            return True
            # attribute: from __future__ import annotations (rare but handle it)
            if (
                isinstance(item.module, cst.Attribute)
                and isinstance(item.module.value, cst.Name)
                and item.module.value.value == "__future__"
                and item.module.attr.value == "annotations"  # type: ignore[attr-defined]
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TypeAnnotationTransformer(cst.CSTTransformer):
    def __init__(
        self,
        modern: bool = False,
        skip_params: bool = False,
        skip_returns: bool = False,
        import_tracer: Optional[ImportTracer] = None,
    ) -> None:
        super().__init__()
        self.modern = modern
        self.skip_params = skip_params
        self.skip_returns = skip_returns
        self.import_tracer = import_tracer

        self._func_name_stack: list[str] = []
        self._lambda_depth: int = 0
        self._return_type_stack: list[list[Optional[str]]] = []

        self.needed_typing_imports: set[str] = set()
        self.changes: list[Change] = []

    # -- Lambda depth tracking --

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self._lambda_depth += 1
        return True

    def leave_Lambda(
        self, original_node: cst.Lambda, updated_node: cst.Lambda
    ) -> cst.Lambda:
        self._lambda_depth -= 1
        return updated_node

    # -- Function scope tracking --

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._func_name_stack.append(node.name.value)
        self._return_type_stack.append([])
        return True

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        func_name = self._func_name_stack.pop()
        returns_collected = self._return_type_stack.pop()

        if not self.skip_returns and updated_node.returns is None:
            inferred = infer_return_type(returns_collected)
            if inferred is not None:
                ret_ann = cst.Annotation(annotation=cst.parse_expression(inferred))
                updated_node = updated_node.with_changes(returns=ret_ann)
                self.changes.append(ReturnChange(func_name, inferred))

        return updated_node

    # -- Return value collection --

    def visit_Return(self, node: cst.Return) -> bool:
        if self._return_type_stack:
            t = _infer_return_value_type(node.value)
            self._return_type_stack[-1].append(t)
        return True

    # -- Parameter annotation --

    def leave_Param(
        self,
        original_node: cst.Param,
        updated_node: cst.Param,
    ) -> cst.Param:
        if self.skip_params:
            return updated_node

        if updated_node.annotation is not None:
            return updated_node

        if self._lambda_depth > 0:
            return updated_node

        if isinstance(updated_node.name, cst.Name):
            if updated_node.name.value in ("self", "cls"):
                return updated_node

        if updated_node.star:
            return updated_node

        if updated_node.default is None:
            return updated_node

        type_str, needed = infer_type_from_default(
            updated_node.default, self.modern, self.import_tracer
        )
        if type_str is None:
            return updated_node

        self.needed_typing_imports.update(needed)

        ann = cst.Annotation(annotation=cst.parse_expression(type_str))
        # PEP 8: spaces around = when annotation is present
        new_equal = cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(" "),
            whitespace_after=cst.SimpleWhitespace(" "),
        )

        func_name = self._func_name_stack[-1] if self._func_name_stack else "<module>"
        param_name = updated_node.name.value if isinstance(updated_node.name, cst.Name) else "?"

        # Record import origin for reporting
        origin: Optional[str] = None
        if self.import_tracer is not None:
            root = type_str.lstrip("type[").lstrip("Type[").split(".")[0].split("[")[0].rstrip("]")
            origin = self.import_tracer.origin_of(root)

        self.changes.append(ParamChange(func_name, param_name, type_str, import_origin=origin))

        return updated_node.with_changes(annotation=ann, equal=new_equal)


# ---------------------------------------------------------------------------
# Import injection
# ---------------------------------------------------------------------------

def inject_typing_imports(tree: cst.Module, needed: set[str]) -> cst.Module:
    """Add `from typing import <name>` for each needed name, idempotently."""
    if not needed:
        return tree
    context = CodemodContext()
    for name in sorted(needed):
        AddImportsVisitor.add_needed_import(context, "typing", name)
    wrapper = cst.metadata.MetadataWrapper(tree)
    return wrapper.visit(AddImportsVisitor(context))


def inject_future_annotations(tree: cst.Module) -> cst.Module:
    """Prepend `from __future__ import annotations` to the module."""
    stmt = cst.parse_statement("from __future__ import annotations\n")
    return tree.with_changes(body=[stmt, *tree.body])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_diff(original: str, modified: str, filepath: str) -> str:
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=filepath,
            tofile=f"{filepath} (annotated)",
        )
    )
    return "".join(lines)


def format_summary(filepath: str, changes: list[Change]) -> str:
    param_changes = [c for c in changes if isinstance(c, ParamChange)]
    return_changes = [c for c in changes if isinstance(c, ReturnChange)]
    lines = [f"[{filepath}] {len(changes)} annotation(s) added:"]
    if param_changes:
        lines.append(f"  params: {len(param_changes)}")
        for c in param_changes:
            origin_tag = f"  # from {c.import_origin}" if c.import_origin else ""
            lines.append(f"    {c.func_name}({c.param_name}: {c.inferred_type}){origin_tag}")
    if return_changes:
        lines.append(f"  returns: {len(return_changes)}")
        for c in return_changes:
            lines.append(f"    {c.func_name}() -> {c.inferred_type}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    modern: bool,
    skip_params: bool,
    skip_returns: bool,
    add_future: bool,
    trace_imports: bool = False,
    search_paths: Optional[list[Path]] = None,
) -> tuple[str, list[Change]]:
    """Parse, transform, and return (new_source, changes)."""
    original_src = path.read_text(encoding="utf-8")

    try:
        tree = cst.parse_module(original_src)
    except cst.ParserSyntaxError as e:
        print(f"WARNING: skipping {path} — parse error: {e}", file=sys.stderr)
        return original_src, []

    file_has_future = has_future_annotations(tree)
    use_modern = modern or file_has_future

    # Build search paths: always include the file's own parent (for local modules)
    effective_search = list(search_paths or [])
    if path.parent not in effective_search:
        effective_search.insert(0, path.parent)
    # Also include cwd
    cwd = Path(".")
    if cwd.resolve() not in [p.resolve() for p in effective_search]:
        effective_search.append(cwd)

    import_tracer = ImportTracer(
        tree,
        search_paths=effective_search,
        trace_enabled=trace_imports,
    )

    transformer = TypeAnnotationTransformer(
        modern=use_modern,
        skip_params=skip_params,
        skip_returns=skip_returns,
        import_tracer=import_tracer,
    )
    new_tree = tree.visit(transformer)

    if transformer.needed_typing_imports:
        new_tree = inject_typing_imports(new_tree, transformer.needed_typing_imports)

    if add_future and not file_has_future:
        new_tree = inject_future_annotations(new_tree)

    return new_tree.code, transformer.changes


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = glob.glob(pattern, recursive=True)
        if matched:
            paths.extend(Path(p) for p in matched if p.endswith(".py"))
        else:
            p = Path(pattern)
            if p.is_file() and p.suffix == ".py":
                paths.append(p)
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-annotate Python files using libcst type inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="File paths or glob patterns")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diff but do not write files")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if any file would be changed (CI mode)")
    parser.add_argument("--modern", action="store_true",
                        help="Use X | None / list[Any] instead of Optional[X] / List[Any]")
    parser.add_argument("--add-future", action="store_true",
                        help="Inject `from __future__ import annotations` if absent")
    parser.add_argument("--no-return", action="store_true",
                        help="Skip return type inference")
    parser.add_argument("--no-params", action="store_true",
                        help="Skip parameter annotation")
    parser.add_argument("--diff", action="store_true",
                        help="Always print unified diff (even when writing)")
    parser.add_argument("--summary", action="store_true",
                        help="Print per-file change summary")
    parser.add_argument(
        "--trace-imports", action="store_true",
        help=(
            "Follow imports to source files to verify class definitions. "
            "More accurate than the PascalCase heuristic alone."
        ),
    )
    parser.add_argument(
        "--search-path", action="append", dest="search_paths",
        metavar="PATH", default=[],
        help=(
            "Root directory to search when resolving imports (may be repeated). "
            "The file's own directory and cwd are always searched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_globs(args.files)

    if not paths:
        print("No Python files matched.", file=sys.stderr)
        sys.exit(1)

    search_paths = [Path(p) for p in args.search_paths]
    any_changed = False

    for path in paths:
        original_src = path.read_text(encoding="utf-8")

        new_src, changes = process_file(
            path,
            modern=args.modern,
            skip_params=args.no_params,
            skip_returns=args.no_return,
            add_future=args.add_future,
            trace_imports=args.trace_imports,
            search_paths=search_paths,
        )

        if new_src == original_src:
            continue

        any_changed = True

        if args.diff or args.dry_run:
            print(format_diff(original_src, new_src, str(path)), end="")

        if args.summary:
            print(format_summary(str(path), changes))

        if not args.dry_run and not args.check:
            path.write_text(new_src, encoding="utf-8")
            if not args.summary:
                print(f"Annotated: {path} ({len(changes)} change(s))")

    if args.check and any_changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
