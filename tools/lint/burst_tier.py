#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BURST TIER LINT (KERNEL_STANDARDS §23: decide coarsely, burst statically). Gated by
`tools/lint/ratchet.py burst_tier`.

A function that issues an MMA belongs to the burst tier and says so: a `//: @burst` line in its
declaration block (the `//:` comments directly above its signature). A burst function contains
nothing data-dependent: no vote or ballot, no warp sync, no shared-memory barrier, no CTA
rendezvous, no `while`, no `if` that is not `if constexpr`, and no `#pragma unroll 1`. A
function that issues MMAs and is not on the tier says why with `//: @burst-exempt <reason>`,
which the ratchet holds as backlog: an exemption is a debt the docs name, not a way out.

Findings are `path:line: message`, one a site, on stdout, exit 0 (the ratchet judges them). The tree's pre-§23 MMA sites (the decode and
intra families, the readout's box loop) are the baseline; a new one fails the gate.

    python tools/lint/burst_tier.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "csrc" / "rola" / "src"

SIGNATURE = re.compile(r"^\s*(?:__device__|__global__)\s+.*?\b(\w+)\s*\(")
DECL_COMMENT = re.compile(r"^\s*//")
MMA_CALL = re.compile(r"(?<![\w:])(?:ops::)?mma\w*\s*\(|\basm\b[^;]*\bmma\.")
FORBIDDEN = (
    (re.compile(r"__(ballot|any|all|match_any|match_all)_sync\s*\("), "a vote in a burst function"),
    (re.compile(r"__syncwarp\s*\("), "a warp sync in a burst function"),
    (re.compile(r"\bmbar_\w+\s*\("), "a shared-memory barrier in a burst function"),
    (re.compile(r"\brendezvous\w*\s*\("), "a CTA rendezvous in a burst function"),
    (re.compile(r"^\s*while\s*\("), "a `while` in a burst function"),
    (re.compile(r"^\s*(?:\}\s*)?(?:else\s+)?if\s*\((?!\s*constexpr)"), "a data-dependent `if` in a burst function"),
    (re.compile(r"^\s*#pragma\s+unroll\s+1\b"), "`#pragma unroll 1` in a burst function"),
)


def functions(lines: list[str]):
    """(name, decl_start, body_start, body_end) for every device function: the declaration block is the run of
    comment and `template` lines directly above the signature; the body is brace-matched from the signature."""
    i = 0
    while i < len(lines):
        m = SIGNATURE.match(re.sub(r"__launch_bounds__\([^)]*\)", "", lines[i]))
        if not m or lines[i].rstrip().endswith(";"):
            i += 1
            continue
        name = m.group(1)
        decl = i
        while decl > 0 and (DECL_COMMENT.match(lines[decl - 1]) or lines[decl - 1].lstrip().startswith("template")):
            decl -= 1
        depth, start, end = 0, None, None
        for j in range(i, len(lines)):
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    start = j if start is None else start
                elif ch == "}":
                    depth -= 1
                    if start is not None and depth == 0:
                        end = j
                        break
            if end is not None:
                break
        if start is None or end is None:
            i += 1
            continue
        yield name, decl, i, start, end
        i = end + 1


def main() -> int:
    findings = []
    for path in sorted(SRC.rglob("*.cu*")):
        rel = path.relative_to(ROOT)
        lines = path.read_text().splitlines()
        for name, decl, sig, start, end in functions(lines):
            block = "\n".join(lines[decl:sig])
            burst = "@burst" in block and "@burst-exempt" not in block
            exempt = "@burst-exempt" in block
            body = lines[start : end + 1]
            issues = [start + k for k, line in enumerate(body)
                      if MMA_CALL.search(line) and not SIGNATURE.match(line)]
            if issues and not burst and not exempt:
                findings.append(f"{rel}:{issues[0] + 1}: `{name}` issues an MMA outside the burst tier "
                                f"(no `//: @burst` in its declaration block; `//: @burst-exempt <reason>` names a debt)")
            if burst:
                for k, line in enumerate(body):
                    for pattern, what in FORBIDDEN:
                        if pattern.search(line):
                            findings.append(f"{rel}:{start + k + 1}: {what} (`{name}`)")
    for f in findings:
        print(f)
    print(f"burst_tier: {len(findings)} finding(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
