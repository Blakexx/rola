#!/usr/bin/env python3
"""R9 ENFORCEMENT LINT (LINT2 item 6, KERNEL_STANDARDS "§R9 ENFORCEMENT: A
STRUCTURE BRANCH FAILS LOUD AT EVERY LAYER", G4 DRIFT GUARD "the R9
enforcement clause is MECHANICAL", G_FOUNDATION G5). REPORT-ONLY.

R9 says a CTA-uniform branch on structure (a runtime switch/if-chain over an
admissible set the declaration fixes -- a "geometry-block field": a decay
kind, a level index, an owner-derived case -- never a measured statistic) is
legal only with (1) generated cases, (2) a compile-time exhaustiveness assert,
(3) a launch-time refusal, and (4) a TRAPPING DEFAULT (`__trap()`, never a
fallthrough) plus a coverage sweep. This lint mechanizes (4) and part of (1):
it cannot see a compile-time `static_assert` or a generator's own case list,
but it CAN see, textually, whether a switch/if-chain's fallback branch calls
`__trap()` and whether the construct is ANNOTATED as drawing its cases from a
generator.

THE MARKER THIS LINT DEFINES (none exists in the tree yet -- G2/G5 are the
stages that adopt it): a comment `// R9-CASES: <case-set name>` on the line
immediately above the `switch`/first `if` of a structure branch, naming the
generated set of admissible cases it covers (e.g. the term-kind enumeration a
future `decode_lattice` generator emits). A structure branch with a trapping
default AND this marker is COMPLIANT; either one missing is a finding.

WHAT COUNTS AS A "STRUCTURE BRANCH" HERE (heuristic; two shapes):
  (a) `switch (x) { case A: ...; case B: ...; default: ...; }` in a
      `__device__`/`__global__` function.
  (b) An `if (x == A) {...} else if (x == B) {...} ... else {...}` chain that
      tests the SAME identifier `x` by `==`/`!=` at least twice.
`if constexpr` chains are EXCLUDED: R9 is about a RUNTIME branch (the compiled
binary carries every arm); `if constexpr` is resolved at compile time -- the
untaken arms never exist in the binary -- so it is already exactly the
"branch on structure the declaration fixes" R9 asks for, with nothing left for
a trap to guard (CUTE_NO_UNROLL-style metaprogramming, not a dispatch).

MEASURED ON THIS TREE (2026-08-29, before G2 lands): every `else if` chain
under csrc/rola/ is `if constexpr` (excluded); the one `switch` on a runtime
value is `decode_lattice.cuh::kind_bits`'s `switch (kind)` -- three explicit
cases plus a bare `default: return r & ~w;` (the fourth legitimate case,
`kKindRmW`, folded into the default rather than named) and no marker. This is
exactly the shape R9's addendum forbids ("a trapping default, never a
fallthrough"); reported here, not fixed (report-only; G2/G5 fix or ratify it).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "rola"

SWITCH_RE = re.compile(r"^\s*switch\s*\(\s*(\w+)\s*\)")
IF_RE = re.compile(r"^\s*(?:}\s*else\s+)?if\s*(constexpr)?\s*\(\s*(\w+)\s*(==|!=)")
MARKER_RE = re.compile(r"//\s*R9-CASES:")
TRAP_RE = re.compile(r"__trap\s*\(")
DEFAULT_RE = re.compile(r"^\s*default\s*:")
ELSE_RE = re.compile(r"^\s*}?\s*else\s*\{?\s*$")


def device_files():
    for suf in (".cu", ".cuh"):
        for p in CSRC.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


def has_marker_above(lines: list[str], idx: int, lookback: int = 3) -> bool:
    return any(MARKER_RE.search(lines[j]) for j in range(max(0, idx - lookback), idx))


def switch_findings(path: Path, lines: list[str]) -> list[str]:
    findings = []
    for i, line in enumerate(lines):
        m = SWITCH_RE.match(line)
        if not m:
            continue

        # Find the switch body's extent by brace counting from this line.
        depth = 0
        started = False
        j = i
        close = None
        while j < len(lines):
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
                    if started and depth == 0:
                        close = j
                        break
            if close is not None:
                break
            j += 1
        if close is None:
            continue
        body = lines[i : close + 1]
        default_idx = next((k for k, l in enumerate(body) if DEFAULT_RE.match(l)), None)
        if default_idx is None:
            continue  # a switch with no default at all is out of this lint's scope
        # Does the default clause's body (up to the next `case`/closing brace)
        # call __trap()?
        default_body = []
        for l in body[default_idx + 1 :]:
            if re.match(r"^\s*(case\s|default\s*:|\})", l):
                break
            default_body.append(l)
        trapping = any(TRAP_RE.search(l) for l in default_body) or TRAP_RE.search(body[default_idx])
        marker = has_marker_above(lines, i)
        if not trapping or not marker:
            missing = []
            if not trapping:
                missing.append("default is not __trap()")
            if not marker:
                missing.append("no `// R9-CASES:` marker above the switch")
            findings.append(f"{path}:{i + 1}: switch ({m.group(1)}) -- {', '.join(missing)}")
    return findings


def if_chain_findings(path: Path, lines: list[str]) -> list[str]:
    findings = []
    i = 0
    n = len(lines)
    while i < n:
        m = IF_RE.match(lines[i])
        if not m or m.group(1) == "constexpr":
            i += 1
            continue

        var = m.group(2)
        chain_start = i
        # Walk forward: does this if start a chain testing the same var at
        # least twice, ending in a bare `else` with no `__trap()`?
        count = 1
        j = i + 1
        has_final_else = False
        # Best-effort: scan subsequent lines for `} else if (var ==/!=` or a
        # terminal `} else {` at the SAME nesting depth as this if's own open
        # brace -- approximated by scanning forward for the next occurrence of
        # either pattern before an unrelated `if`/`switch`/function boundary.
        while j < min(n, i + 4000):
            l = lines[j]
            im = IF_RE.match(l)
            if im and im.group(1) != "constexpr" and im.group(2) == var:
                count += 1
                j += 1
                continue
            if re.match(r"^\s*}\s*else\s*\{?\s*$", l) and count >= 2 and not has_final_else:
                has_final_else = True
                j += 1
                continue
            if has_final_else:
                # first non-brace-only line after the terminal else: check for __trap
                if l.strip() and l.strip() not in ("{", "}"):
                    trapping = bool(TRAP_RE.search(l))
                    marker = has_marker_above(lines, chain_start)
                    if not trapping or not marker:
                        missing = []
                        if not trapping:
                            missing.append("final else is not __trap()")
                        if not marker:
                            missing.append("no `// R9-CASES:` marker above the chain")
                        findings.append(f"{path}:{chain_start + 1}: if-chain on "
                                        f"'{var}' ({count} arms) -- {', '.join(missing)}")
                    break
                j += 1
                continue
            if re.match(r"^\s*(switch|for|while)\s*\(", l) or l.strip() == "":
                break
            j += 1
        i = chain_start + 1
    return findings


def self_test() -> int:
    """Excluded from the commit gate (K46 convention)."""
    fixture = ROOT / "tools" / "lint" / "fixtures" / "r9_switch.cuh"
    lines = fixture.read_text().splitlines()
    found = switch_findings(fixture, lines)
    bad_hits = [f for f in found if f.startswith(f"{fixture}:4:")]
    ok_hits = [f for f in found if not f.startswith(f"{fixture}:4:")]
    if not bad_hits:
        print(f"SELF-TEST FAILED: bad_switch did not fire: {found}", file=sys.stderr)
        return 1
    if ok_hits:
        print(f"SELF-TEST FAILED: ok_switch fired unexpectedly: {ok_hits}", file=sys.stderr)
        return 1
    print("r9_enforcement.py --test-fixtures: PASS")
    return 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()

    all_findings = []
    for p in device_files():
        rel = p.relative_to(ROOT)
        lines = p.read_text().splitlines()
        for f in switch_findings(p, lines) + if_chain_findings(p, lines):
            all_findings.append(f.replace(str(p), str(rel)))

    print(f"r9_enforcement: {len(all_findings)} structure-branch finding(s) "
          f"(missing __trap() default and/or the R9-CASES marker this lint defines)")
    for f in all_findings:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
