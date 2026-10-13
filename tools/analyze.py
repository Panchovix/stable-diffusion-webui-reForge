#!/usr/bin/env python3
"""
Code analysis helper for reForge.

Usage:
  python tools/analyze.py symbols   <file.py>           # list classes/functions
  python tools/analyze.py calls     <file.py> [func]    # show calls made by (optional) function
  python tools/analyze.py callers   <dir_or_file> <name># find who calls <name>
  python tools/analyze.py imports   <file.py>           # list imports
  python tools/analyze.py graph     <file.py>           # dot call graph (stdout)
  python tools/analyze.py flow      <file.py> <func>    # trace call tree inside file
"""

import ast
import sys
import os
from pathlib import Path


def load(path: str) -> ast.Module:
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    return ast.parse(src, filename=path)


# ── symbols ─────────────────────────────────────────────────────────────────

def cmd_symbols(path: str):
    tree = load(path)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            print(f"class {node.name}  (line {node.lineno})")
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    print(f"    def {child.name}  (line {child.lineno})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"def {node.name}  (line {node.lineno})")


# ── imports ──────────────────────────────────────────────────────────────────

def cmd_imports(path: str):
    tree = load(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                print(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
            print(f"from {mod} import {names}")


# ── calls made by a function (or the whole file) ─────────────────────────────

def _collect_calls(node: ast.AST) -> list[str]:
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                calls.append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                calls.append(f"{_attr_chain(n.func)}")
    return calls


def _attr_chain(node):
    if isinstance(node, ast.Attribute):
        return f"{_attr_chain(node.value)}.{node.attr}"
    elif isinstance(node, ast.Name):
        return node.id
    return "?"


def cmd_calls(path: str, func_name: str | None = None):
    tree = load(path)
    if func_name:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                print(f"Calls made inside  {func_name}():")
                for c in sorted(set(_collect_calls(node))):
                    print(f"  {c}()")
                return
        print(f"Function '{func_name}' not found in {path}")
    else:
        print(f"All calls in {path}:")
        for c in sorted(set(_collect_calls(tree))):
            print(f"  {c}()")


# ── callers: grep-like search across files ───────────────────────────────────

def cmd_callers(root: str, name: str):
    paths = Path(root).rglob("*.py") if Path(root).is_dir() else [Path(root)]
    for p in paths:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for call in ast.walk(node):
                    if isinstance(call, ast.Call):
                        called = ""
                        if isinstance(call.func, ast.Name):
                            called = call.func.id
                        elif isinstance(call.func, ast.Attribute):
                            called = call.func.attr
                        if called == name:
                            print(f"{p}:{call.lineno}  in  {node.name}()")


# ── call flow (depth-limited DFS within one file) ────────────────────────────

def _build_call_map(tree: ast.Module) -> dict[str, list[str]]:
    cmap: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cmap[node.name] = list(set(_collect_calls(node)))
    return cmap


def cmd_flow(path: str, func_name: str, depth: int = 5):
    tree = load(path)
    cmap = _build_call_map(tree)
    defined = set(cmap.keys())

    def show(name, level, visited):
        prefix = "  " * level
        print(f"{prefix}{name}()")
        if level >= depth or name in visited:
            return
        visited = visited | {name}
        for callee in cmap.get(name, []):
            if callee in defined:
                show(callee, level + 1, visited)

    show(func_name, 0, set())


# ── dot graph via pyan3 ──────────────────────────────────────────────────────

def cmd_graph(path: str):
    pyan3 = "/var/data/python/bin/pyan3"
    if not Path(pyan3).exists():
        pyan3 = "pyan3"
    os.execlp(pyan3, pyan3, path, "--dot", "--no-defines")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd, *rest = args
    if cmd == "symbols" and rest:
        cmd_symbols(rest[0])
    elif cmd == "imports" and rest:
        cmd_imports(rest[0])
    elif cmd == "calls" and rest:
        cmd_calls(rest[0], rest[1] if len(rest) > 1 else None)
    elif cmd == "callers" and rest:
        cmd_callers(rest[0], rest[1] if len(rest) > 1 else "")
    elif cmd == "flow" and len(rest) >= 2:
        cmd_flow(rest[0], rest[1], int(rest[2]) if len(rest) > 2 else 5)
    elif cmd == "graph" and rest:
        cmd_graph(rest[0])
    else:
        print(__doc__)
        sys.exit(1)
