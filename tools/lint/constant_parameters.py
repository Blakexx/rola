#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""NO DEAD PARAMETERS: a knob that only ever takes one value is a constant. Gated by `tools/lint/ratchet.py`.

The ruling this mechanizes (Blake, 2026-08-29) came from a real one: a stage shipped a
parameter documented as "stays None", which is a parameter in name only -- it reads as a
choice a caller has, it is carried through every signature and every call site that
touches it, and nothing in the tree can move it. That is the class vulture does not
catch: the name IS used, everywhere, always with the same value.

WHAT IS READ. Two shapes, both in Python, both by AST rather than by text:

  * A FUNCTION PARAMETER with a default, whose every call site either omits it or passes
    the SAME literal. One distinct value across the whole tree means the default is the
    only value, and the parameter is that default.
  * A DATACLASS FIELD with a default, constructed the same way everywhere.

A parameter with no default is not read: it is the caller's to supply, and a tree that
happens to supply one value today is not the same claim.

THE MARKER. A knob that is deliberately reserved -- a shape the design will use and the
tree does not exercise yet -- says so on its own line:

    def carry(routes, *, warps_per_cta=8):   #: reserved: the 2x4 launch, once the
                                             #: two-CTA arm is built

`reserved:` anywhere in a comment on the parameter's line, or in the enclosing
definition's docstring naming the parameter, is the declaration. It carries a REASON,
which is the whole point: "reserved" without one is indistinguishable from dead.

A new finding fails the commit; the findings no human has ruled on yet are the ratchet's baseline. Two blind
spots make every finding a question rather than a proof, and both are one-directional --
they over-report, never under-report:

  * A CALLER OUTSIDE THIS TREE (a notebook, a downstream package, a command line that
    threads a flag through) cannot be seen at all.
  * A POSITIONAL call site is not read. The rule matches keyword arguments, because that
    is what a knob is passed as; a caller that supplies the same parameter positionally
    looks, to this scan, like a caller that did not supply it.

    python tools/lint/constant_parameters.py
    python tools/lint/constant_parameters.py --test-fixtures
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOTS = ("rola", "tools", "measure")

#: The declaration that a knob is deliberately unexercised, with its reason beside it.
MARKER = "reserved:"


def python_files(roots=None) -> list[Path]:
    out: list[Path] = []
    for name in roots or ROOTS:
        base = ROOT / name
        if not base.is_dir():
            continue
        out += [p for p in sorted(base.rglob("*.py"))
                if FIXTURES not in p.parents and "__pycache__" not in p.parts]
    return out


#: The literal kinds a value can be COMPARED as: hashable, so two call sites passing the
#: same thing collapse to one entry. A list or a dict default is not read -- a mutable
#: default is its own defect and a different rule's.
_COMPARABLE = (int, float, complex, str, bytes, bool, tuple, type(None))


def _literal(node) -> tuple[bool, object]:
    """The node's value if it is a literal this rule can compare, else `(False, None)`."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None
    if not isinstance(value, _COMPARABLE):
        return False, None
    return True, value


def _reserved(source_lines: list[str], node) -> bool:
    line = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
    return MARKER in line


def _docstring_reserves(func: ast.AST, name: str) -> bool:
    doc = ast.get_docstring(func) or ""
    return MARKER in doc and name in doc


def collect(files: list[Path]):
    """`(declarations, call values)` over the whole set, keyed by `(function, parameter)`.

    ONE PASS OVER EVERY FILE, because the question is tree-wide by nature: a parameter is
    dead only if NOWHERE moves it, and a per-file scan can only ever say "not here".
    """
    declarations: dict[tuple[str, str], tuple[Path, int, bool]] = {}
    values: dict[tuple[str, str], set] = defaultdict(set)
    unknown: set[tuple[str, str]] = set()

    for path in files:
        source = path.read_text()
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                defaulted = list(zip(args.args[len(args.args) - len(args.defaults):],
                                     args.defaults, strict=False))
                defaulted += list(zip(args.kwonlyargs, args.kw_defaults, strict=False))
                for arg, default in defaulted:
                    if default is None:
                        continue
                    ok, value = _literal(default)
                    if not ok:
                        continue
                    key = (node.name, arg.arg)
                    reserved = (_reserved(lines, arg) or _reserved(lines, node)
                                or _docstring_reserves(node, arg.arg))
                    declarations[key] = (path, node.lineno, reserved)
                    values[key].add(("default", value))
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if name is None:
                    continue
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    key = (name, keyword.arg)
                    ok, value = _literal(keyword.value)
                    if ok:
                        values[key].add(("call", value))
                    else:
                        unknown.add(key)
    return declarations, values, unknown


def findings(files: list[Path] | None = None) -> list[str]:
    files = python_files() if files is None else files
    declarations, values, unknown = collect(files)
    out = []
    for key, (path, lineno, reserved) in sorted(declarations.items(),
                                                key=lambda item: str(item[1][0])):
        function, parameter = key
        if reserved or key in unknown:
            continue
        distinct = {value for _kind, value in values[key]}
        if len(distinct) != 1:
            continue
        only = next(iter(distinct))
        out.append(
            f"{path.relative_to(ROOT)}:{lineno}: {function}(..., {parameter}=) is only "
            f"ever {only!r} -- across every call in this tree, nothing moves it. A knob "
            f"with one value is a constant: inline it, or declare it with a "
            f"`{MARKER} <reason>` comment saying which shape it is held for.")
    return out


def test_fixtures() -> int:
    positive = FIXTURES / "constant_parameter_bad.py"
    negative = FIXTURES / "constant_parameter_ok.py"
    for path in (positive, negative):
        if not path.is_file():
            print(f"SELF-TEST FAILED: missing fixture {path}", file=sys.stderr)
            return 1
    fired = findings([positive])
    quiet = findings([negative])
    if not any("dead_knob" in f for f in fired):
        print(f"SELF-TEST FAILED: no finding on {positive.name}'s dead knob", file=sys.stderr)
        return 1
    if quiet:
        print(f"SELF-TEST FAILED: {negative.name} fired: {quiet}", file=sys.stderr)
        return 1
    if any("fixtures/" in f for f in findings()):
        print("SELF-TEST FAILED: a fixture leaked into the real scan", file=sys.stderr)
        return 1
    print("constant_parameters.py --test-fixtures: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-fixtures", action="store_true")
    args = parser.parse_args()
    if args.test_fixtures:
        return test_fixtures()
    found = findings()
    for finding in found:
        print(f"  {finding}")
    print(f"constant_parameters: {len(found)} finding(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
