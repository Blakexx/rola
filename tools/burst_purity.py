#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BURST PURITY GATE (KERNEL_STANDARDS §23): the SASS of every `//: @burst` function holds no
vote, barrier, convergence bracket, call or branch -- what the source lint cannot see, because
the compiler inserts them (a collective after a value it cannot prove uniform goes through the
convergence-checking path: fifty-one calls in a form of the readout, once). Read off a cubin with line information:
an instruction belongs to a burst function when any frame of its inline chain lies in that
function's lines.

    python tools/burst_purity.py --arm 0 --source csrc/rola/src/carry/carry_kernel.cuh   # compiles the arm with -lineinfo
    python tools/burst_purity.py --cubin <cubin> --source ...

Prints a row a burst function (instructions, HMMAs, shuffles, the forbidden opcodes) and exits 1
on any forbidden opcode. Docs: docs/internals/tools/burst_purity.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lint"))

import burst_tier  # noqa: E402
import life_ranges  # noqa: E402
import sass  # noqa: E402

FORBIDDEN = re.compile(r"^(?:@!?U?P\w+\s+)?(VOTE\w*|BAR\w*|BSSY|BSYNC|CALL\w*|WARPSYNC|MEMBAR\w*|BRA\w*|EXIT|RET\w*)\b")


def burst_functions() -> list[tuple[str, str, int, int]]:
    """(file name, function, first line, last line) of every `//: @burst` function in the sources."""
    out = []
    for path in sorted(burst_tier.SRC.rglob("*.cu*")):
        lines = path.read_text().splitlines()
        for name, decl, sig, start, end in burst_tier.functions(lines):
            block = "\n".join(lines[decl:sig])
            if "@burst" in block and "@burst-exempt" not in block:
                out.append((path.name, name, sig + 1, end + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None, help="compile this carry arm with -lineinfo and read it")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--source", type=Path, required=True, help="the kernel source the burst functions live in")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    if (a.cubin is None) == (a.arm is None):
        ap.error("give exactly one of --cubin or --arm")
    cubin = a.cubin if a.cubin else life_ranges.compile_arm(a.arm, a.arch)

    text = sass.disassemble(cubin, "--print-line-info-inline", "-gi")
    frames = sass.frames(text)
    instr = {}
    for line in text.splitlines():
        m = sass.INSTRUCTION.match(line)
        if m:
            instr[int(m.group(1), 16)] = ((m.group(2) or "").strip() + " " + m.group(3)).strip()
    functions = burst_functions()

    rows = {name: collections.Counter() for _, name, _, _ in functions}
    for off, ins in instr.items():
        chain = frames.get(off, [])
        for file, name, lo, hi in functions:
            if any(f == file and lo <= ln <= hi for f, ln in chain):
                c = rows[name]
                c["instr"] += 1
                op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0] if ins else "?"
                if op.startswith("HMMA"):
                    c["hmma"] += 1
                if op.startswith("SHFL"):
                    c["shfl"] += 1
                if FORBIDDEN.match(ins):
                    c["forbidden"] += 1
                    c[f"op:{op}"] += 1
                break

    red = False
    for _, name, lo, hi in functions:
        c = rows[name]
        bad = {k[3:]: v for k, v in c.items() if k.startswith("op:")}
        red |= c["forbidden"] > 0
        print(f"{'RED ' if c['forbidden'] else 'ok  '} {name} ({lo}-{hi}): instr {c['instr']} hmma {c['hmma']} shfl {c['shfl']}"
              + (f" :: {bad}" if bad else ""))
    if not functions:
        print("no `//: @burst` functions in the sources")
    if a.json:
        a.json.write_text(json.dumps({name: dict(c) for name, c in rows.items()}, sort_keys=True) + "\n")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
