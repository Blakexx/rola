#!/usr/bin/env python3
"""READABILITY SPACING LINT (gating). Two checks, csrc (`.cu`/`.cuh`/`.cpp`/`.h`/`.hpp`, excluding
csrc/third_party) AND Python (`rola/`, `tools/`, `tests/`, `benchmarks/`):

1. A function/kernel BODY >= 40 LINES with ZERO BLANK LINES inside it. This
   does not dictate WHERE to segment (declarations / loops / branches, this
   codebase's own convention) -- it only asks that a body long enough to need
   segmentation has at least one. Companion to `.clang-format`'s
   `MaxEmptyLinesToKeep: 1` / `SeparateDefinitionBlocks: Always` (this stage,
   config only): that setting caps blank lines FROM ABOVE, this rule sets the
   floor.
2. MORE THAN ONE CONSECUTIVE BLANK LINE, CSRC ONLY (>= 2 blank lines in a
   row) -- the textual mirror of `.clang-format`'s `MaxEmptyLinesToKeep: 1`
   (this stage, config only), scoped exactly the same way that setting is
   (`.clang-format`'s own header: "Scope: csrc/rola only"). NOT applied to
   Python: PEP8/pycodestyle's convention is 2 blank lines between top-level
   defs, which `>1` would flag as a violation of a style Python does not
   have -- ruff's `E303` ("too many blank lines", default max 2) is Python's
   own belt for this, already selected via pyproject.toml's full "E" family
   and gating today; measured while writing this check (1,421 "findings",
   effectively all of them the ordinary two-blank-line convention between
   top-level defs) before this scope note was added.

METHOD for csrc FUNCTION BODIES (heuristic -- same brace-counting technique as
`lint_standards.py::device_code_line_ranges`, generalized past `__global__`/
`__device__` to ANY function-shaped signature, since this check is about
readability of host code too): a line whose last non-whitespace character is
`{` and whose look-back (up to 3 lines) contains a `)` -- i.e. it closes a
parameter list -- opens a candidate body; braces are counted to the matching
close. Control-flow blocks (`if`/`for`/while`/`switch`/`catch`) are excluded
by keyword. This under- and over-matches at the same margin
`lint_standards.py` already documents for its own brace count (string/char
literals with unbalanced braces, multi-line templates); it is a lint, not a
compiler.

METHOD for Python: the real `ast` module -- exact, not heuristic. A
`FunctionDef`/`AsyncFunctionDef` node's `lineno`/`end_lineno` bounds its body.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "rola"
PY_ROOTS = [ROOT / "rola", ROOT / "tools", ROOT / "tests", ROOT / "benchmarks"]
CSRC_SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

MIN_BODY_LINES = 40

CONTROL_KEYWORDS = re.compile(r"\b(if|for|while|switch|catch|else)\s*\(")
TAIL_WORDS_RE = re.compile(r"[\w\s]*")


def sig_close(line: str) -> bool:
    """"...) [const] [noexcept] {": the text after the LAST `)` is words and spaces and the
    line ends in `{`. Linear: the first form, `\)\s*(\w+\s*)*\{\s*$`, backtracked
    exponentially on a `)` followed by words and no `{` -- every `__launch_bounds__(...) void
    kernel(` line -- and hung the commit gate for minutes."""
    s = line.rstrip()
    if not s.endswith("{"):
        return False
    k = s.rfind(")")
    return k >= 0 and TAIL_WORDS_RE.fullmatch(s[k + 1 : -1]) is not None


def csrc_files():
    for suf in CSRC_SUFFIXES:
        for p in CSRC.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


def py_files():
    #: `tools/lint/fixtures/**` is deliberately excluded (K46's ast-grep bug:
    #: a fixture's own "must fire" violation re-triggering the real report).
    for root in PY_ROOTS:
        if root.is_dir():
            for p in root.rglob("*.py"):
                if "fixtures" in p.relative_to(ROOT).parts:
                    continue
                yield p


def consecutive_blank_findings(path: Path, lines: list[str]) -> list[str]:
    findings = []
    run_start = None
    run_len = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len > 1:
                findings.append(f"{path}:{run_start + 1}: {run_len} consecutive "
                                 f"blank lines (> 1)")
            run_start = None
            run_len = 0
    if run_len > 1:
        findings.append(f"{path}:{run_start + 1}: {run_len} consecutive blank lines (> 1)")
    return findings


def csrc_body_findings(path: Path, lines: list[str]) -> list[str]:
    """Function bodies of `MIN_BODY_LINES` or more with no blank line. The brace depth at
    each line's end is summed ONCE (`depth_end`), so a candidate whose braces never balance
    (a `{` in a string or a comment, a straddling macro) is skipped in constant time; the
    first form re-scanned to the end of the file per candidate and took minutes on a
    1,500-line kernel -- long enough that the commit gate looked hung."""
    findings = []
    n = len(lines)
    depth_end = [0] * n
    depth = 0
    for j, l in enumerate(lines):
        depth += l.count("{") - l.count("}")
        depth_end[j] = depth

    i = 0
    while i < n:
        line = lines[i]
        # Candidate signature close: look at this line and up to 2 above joined.
        window = " ".join(l.strip() for l in lines[max(0, i - 2) : i + 1])
        if (sig_close(line) and "(" in window
                and not CONTROL_KEYWORDS.search(window)
                and not window.strip().endswith("; {")):
            base = depth_end[i - 1] if i > 0 else 0
            close = None

            #: the body opens on this line or the next few; a candidate that never returns
            #: to its base depth is unbalanced text, not a function.
            if depth_end[n - 1] <= base:
                j = i
                started = False
                while j < n and (started or j - i <= 3):
                    if "{" in lines[j]:
                        started = True
                    if started and depth_end[j] <= base:
                        close = j
                        break
                    j += 1

            if close is not None:
                body = lines[i + 1 : close]
                if len(body) >= MIN_BODY_LINES and not any(b.strip() == "" for b in body):
                    findings.append(f"{path}:{i + 1}: function body spans "
                                     f"{len(body)} lines (>= {MIN_BODY_LINES}) with "
                                     f"zero blank lines")
                i = close + 1
                continue
        i += 1
    return findings


def py_body_findings(path: Path, source: str, lines: list[str]) -> list[str]:
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first_body = node.body[0]
        start = first_body.lineno  # 1-indexed, first body statement
        end = node.end_lineno  # 1-indexed, inclusive, whole function
        body_lines = lines[start - 1 : end]
        if len(body_lines) >= MIN_BODY_LINES and not any(b.strip() == "" for b in body_lines):
            findings.append(f"{path}:{node.lineno}: def '{node.name}' body spans "
                             f"{len(body_lines)} lines (>= {MIN_BODY_LINES}) with "
                             f"zero blank lines")
    return findings


FIXDIR = ROOT / "tools" / "lint" / "fixtures"


def self_test() -> int:
    """Excluded from the commit gate (K46 convention)."""
    ok = True

    cu = FIXDIR / "readability_long_body.cu"
    lines = cu.read_text().splitlines()
    found = csrc_body_findings(cu, lines)
    # 'bad' opens at line 3 (`__device__ int bad(int x) {`); 'ok' opens after
    # 'bad' closes -- only 'bad' (zero blank lines in its 40-line body) fires.
    if not any(":3:" in f for f in found):
        print("SELF-TEST FAILED: csrc 'bad' body did not fire", file=sys.stderr)
        ok = False
    if len(found) != 1:
        print(f"SELF-TEST FAILED: expected exactly 1 csrc body finding, got {found}",
              file=sys.stderr)
        ok = False

    blank_cu = FIXDIR / "readability_blank_run.cu"
    blank_found = consecutive_blank_findings(blank_cu, blank_cu.read_text().splitlines())
    if not blank_found:
        print("SELF-TEST FAILED: csrc blank-run fixture did not fire", file=sys.stderr)
        ok = False

    py = FIXDIR / "readability_long_body.py"
    py_source = py.read_text()
    py_found = py_body_findings(py, py_source, py_source.splitlines())
    if not any("'bad'" in f for f in py_found):
        print("SELF-TEST FAILED: python 'bad' body did not fire", file=sys.stderr)
        ok = False
    if any("'ok'" in f for f in py_found):
        print("SELF-TEST FAILED: python 'ok' body fired unexpectedly", file=sys.stderr)
        ok = False

    if not ok:
        return 1
    print("readability_spacing.py --test-fixtures: PASS")
    return 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()

    body_findings = []
    blank_findings = []

    for p in csrc_files():
        rel = p.relative_to(ROOT)
        lines = p.read_text().splitlines()
        body_findings += [f.replace(str(p), str(rel)) for f in csrc_body_findings(p, lines)]
        blank_findings += [f.replace(str(p), str(rel)) for f in consecutive_blank_findings(p, lines)]

    for p in py_files():
        rel = p.relative_to(ROOT)
        source = p.read_text()
        lines = source.splitlines()
        body_findings += [f.replace(str(p), str(rel)) for f in py_body_findings(p, source, lines)]
        # blank-line spacing is NOT checked for Python here -- ruff's E303
        # already owns it at PEP8's threshold (2), not this rule's csrc-only 1.

    print(f"readability_spacing: {len(body_findings)} function/kernel body finding(s) "
          f"(>= {MIN_BODY_LINES} lines, zero blank lines)")
    for f in body_findings:
        print(f"  {f}")
    print(f"readability_spacing: {len(blank_findings)} consecutive-blank-line finding(s) (> 1)")
    for f in blank_findings:
        print(f"  {f}")

    #: GATING. A body of forty lines with no blank line in it is a body a reader has to
    #: re-derive the structure of; the rule asks for segmentation and does not say where,
    #: so satisfying it is a judgement the author makes once and the reader keeps.
    return 1 if (body_findings or blank_findings) else 0


if __name__ == "__main__":
    sys.exit(main())
