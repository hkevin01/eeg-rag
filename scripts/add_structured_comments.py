#!/usr/bin/env python3
"""
Inject structured safety-critical comment blocks above every class and function
definition in the eeg_rag source tree.

The comment format is a line-prefixed structured block following the fields:
  ID, Requirement, Purpose, Rationale, Inputs, Outputs, Preconditions,
  Postconditions, Assumptions, Side Effects, Failure Modes, Error Handling,
  Constraints, Verification, References.

The script is idempotent: it will NOT insert a second block if one already
exists immediately above the definition (identified by the separator line).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 75


def _first_docstring_line(node: ast.AST) -> Optional[str]:
    """Return the first line of the node's docstring, or None."""
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr):
        val = body[0].value
        text: Optional[str] = None
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            text = val.value
        if text:
            return text.strip().split("\n")[0].strip().rstrip(".")
    return None


def _unparse(node: ast.AST) -> str:
    """Safely unparse a node to a string."""
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            pass
    return "<expr>"


def _make_id(rel_path: str, class_name: Optional[str], name: str) -> str:
    """Build a dotted unique identifier for the code block."""
    parts = Path(rel_path).with_suffix("").parts
    try:
        idx = list(parts).index("eeg_rag")
        parts = parts[idx:]
    except ValueError:
        parts = parts[-3:]
    chain = list(parts)
    if class_name:
        chain.append(class_name)
    chain.append(name)
    return ".".join(chain)


def _args_summary(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Summarise parameters as 'name: type (default=val)' entries."""
    args = node.args
    parts: List[str] = []
    n_defaults = len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg in ("self", "cls"):
            continue
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        default_idx = i - (len(args.args) - n_defaults)
        if default_idx >= 0:
            try:
                dflt = _unparse(args.defaults[default_idx])
                parts.append(f"{arg.arg}{ann} (default={dflt})")
            except Exception:
                parts.append(f"{arg.arg}{ann}")
        else:
            parts.append(f"{arg.arg}{ann}")

    for arg in args.kwonlyargs:
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")

    if args.vararg:
        ann = f": {_unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")

    if args.kwarg:
        ann = f": {_unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    return "; ".join(parts) if parts else "None"


def _return_summary(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if node.returns:
        return _unparse(node.returns)
    return "Implicitly None or see body"


def _word_wrap(text: str, width: int = 68) -> str:
    """Wrap text to fit within *width* characters (for the comment value)."""
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for w in words:
        if sum(len(x) + 1 for x in current) + len(w) > width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comment-block generator
# ---------------------------------------------------------------------------

def build_comment_block(
    node: ast.AST,
    rel_path: str,
    class_name: Optional[str],
    indent: str,
) -> str:
    """Return the full structured comment block string (without trailing newline)."""

    is_class = isinstance(node, ast.ClassDef)
    name: str = node.name  # type: ignore[attr-defined]
    ds = _first_docstring_line(node)
    node_id = _make_id(rel_path, class_name, name)

    # --- field values -------------------------------------------------------
    if is_class:
        requirement = (
            f"`{name}` class shall be instantiable and expose the documented interface"
        )
        purpose = ds or f"Encapsulates {name.replace('_', ' ')} state and behaviour"
        rationale = (
            "Object-oriented encapsulation isolates state and enforces invariants"
        )
        inputs_str = "Constructor arguments — see __init__ signature"
        outputs_str = "N/A (class definition)"
        precond = "All imported dependencies must be available at import time"
        postcond = "Instance attributes initialised as documented; invariants hold"
        assumptions = "Python runtime ≥ 3.9; package dependencies installed"
        side_effects = (
            "May allocate heap memory; __init__ may open connections or load models"
        )
        fail_modes = (
            "ImportError if dependency missing; TypeError for invalid constructor args"
        )
        err_handling = "Constructor raises on invalid args; see __init__ body"
        constraints = "Thread-safety not guaranteed unless explicitly documented"
        verification = (
            f"Instantiate {name} with valid args; assert attribute types and values"
        )
    else:
        fn = node  # type: ignore[assignment]
        is_async = isinstance(fn, ast.AsyncFunctionDef)
        requirement = (
            f"`{name}` shall {(ds[0].lower() + ds[1:]) if ds else 'execute as specified'}"
        )
        purpose = ds or name.replace("_", " ").capitalize()
        rationale = (
            "Implements domain-specific logic per system design; see referenced specs"
        )
        inputs_str = _args_summary(fn)
        outputs_str = _return_summary(fn)
        precond = (
            "Owning object properly initialised (if method); "
            "inputs within documented valid ranges"
        )
        postcond = "Return value satisfies documented output type and range"
        assumptions = "Python runtime ≥ 3.9; inputs are well-typed at call site"
        side_effects = "May update instance state or perform I/O; see body"
        fail_modes = (
            "Invalid inputs raise ValueError/TypeError; "
            "I/O failures raise OSError or subclass"
        )
        err_handling = (
            "Validates critical inputs at boundary; propagates unexpected exceptions"
        )
        constraints = (
            "Must be awaited (async)" if is_async else "Synchronous — must not block event loop"
        )
        verification = (
            f"Unit test with representative, boundary, and invalid inputs; "
            f"assert return satisfies postcondition"
        )

    references = "EEG-RAG system design specification; see module docstring"

    # --- build lines --------------------------------------------------------
    pad = f"{indent}# "
    sep_line = f"{indent}# {SEPARATOR}"
    field_width = 13  # aligns the colon column

    def field(label: str, value: str) -> str:
        return f"{pad}{label:<{field_width}}: {value}"

    lines = [
        sep_line,
        field("ID", node_id),
        field("Requirement", requirement),
        field("Purpose", purpose),
        field("Rationale", rationale),
        field("Inputs", inputs_str),
        field("Outputs", outputs_str),
        field("Precond.", precond),
        field("Postcond.", postcond),
        field("Assumptions", assumptions),
        field("Side Effects", side_effects),
        field("Fail Modes", fail_modes),
        field("Err Handling", err_handling),
        field("Constraints", constraints),
        field("Verification", verification),
        field("References", references),
        sep_line,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

_BLOCK_SENTINEL = re.compile(r"^\s*#\s*-{10,}\s*$")  # matches separator line


def _already_has_block(lines: List[str], def_line_idx: int) -> bool:
    """
    Return True if there is already a structured comment block
    immediately above the definition (accounting for decorators).
    """
    # Walk upward past decorator lines and blank lines
    idx = def_line_idx - 1
    while idx >= 0:
        stripped = lines[idx].strip()
        if stripped.startswith("@"):
            idx -= 1
            continue
        if stripped == "":
            idx -= 1
            continue
        # First non-blank, non-decorator line above
        return bool(_BLOCK_SENTINEL.match(lines[idx]))
    return False


def _insertion_line_idx(lines: List[str], def_line_idx: int) -> int:
    """
    Return the index at which the comment block should be inserted.
    That is just before any decorators that belong to this definition.
    """
    idx = def_line_idx - 1
    while idx >= 0 and lines[idx].strip().startswith("@"):
        idx -= 1
    return idx + 1  # insert BEFORE this decorator (or the def itself)


def process_file(filepath: Path, src_root: Path) -> bool:
    """
    Process one Python file, adding structured comment blocks.
    Returns True if the file was modified.
    """
    src = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(filepath))
    except SyntaxError as exc:
        print(f"  SKIP (parse error): {filepath.relative_to(src_root)}: {exc}")
        return False

    lines = src.splitlines(keepends=True)
    rel_path = str(filepath.relative_to(src_root))

    # Collect (def_line_1based, node, current_class_name) tuples.
    # We use a simple visitor that tracks the enclosing class name.
    insertions: List[Tuple[int, ast.AST, Optional[str]]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._class_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            insertions.append((node.lineno, node, None))
            self._class_stack.append(node.name)
            self.generic_visit(node)
            self._class_stack.pop()

        def _visit_func(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            class_ctx = self._class_stack[-1] if self._class_stack else None
            insertions.append((node.lineno, node, class_ctx))
            # Do NOT recurse into nested functions via generic_visit here —
            # we handle them through the generic walk below anyway.
            old = list(self._class_stack)
            self.generic_visit(node)
            self._class_stack[:] = old

        visit_FunctionDef = _visit_func  # type: ignore[assignment]
        visit_AsyncFunctionDef = _visit_func  # type: ignore[assignment]

    _Visitor().visit(tree)

    if not insertions:
        return False

    # Sort descending by line number so insertions don't shift earlier lines.
    insertions.sort(key=lambda t: t[0], reverse=True)

    modified = False
    for lineno, node, class_name in insertions:
        def_idx = lineno - 1  # 0-based
        if _already_has_block(lines, def_idx):
            continue

        # Determine indentation from the definition line
        def_line = lines[def_idx]
        indent = len(def_line) - len(def_line.lstrip())
        indent_str = " " * indent

        block = build_comment_block(node, rel_path, class_name, indent_str)
        block_lines = [ln + "\n" for ln in block.split("\n")]

        insert_at = _insertion_line_idx(lines, def_idx)
        lines[insert_at:insert_at] = block_lines
        modified = True

    if modified:
        filepath.write_text("".join(lines), encoding="utf-8")

    return modified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    src_root = Path(__file__).parent.parent / "src" / "eeg_rag"
    if not src_root.exists():
        print(f"Source root not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    py_files = sorted(
        p
        for p in src_root.rglob("*.py")
        if "__pycache__" not in str(p)
    )

    print(f"Processing {len(py_files)} files under {src_root} …")
    changed = 0
    skipped = 0

    for fp in py_files:
        was_modified = process_file(fp, src_root)
        rel = fp.relative_to(src_root)
        if was_modified:
            print(f"  ✓  {rel}")
            changed += 1
        else:
            skipped += 1

    print(f"\nDone — {changed} files updated, {skipped} files unchanged/skipped.")


if __name__ == "__main__":
    main()
