#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS CONTROL WORDS, and the loop a function compiles to. Every instruction on Volta, Turing and
Ampere carries, in bits 105..125 of its 128-bit encoding, what ptxas decided about it: the stall
count before the next issue, a yield bit, the scoreboard barrier it sets on completion (write) and
the one that guards its source registers (read), the mask of barriers it waits on, and the reuse
flags. `cuobjdump -sass` prints the raw words; this module decodes them, and finds the SASS a source
function's lines compile to, from its first instruction to its last backward branch: one loop
iteration. The layout is Jia et al.'s (Volta §2.1), validated here by the reuse bits matching the
disassembly's `.reuse` operands. Docs: docs/internals/tools/burst_gate.md#control
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lint"))

import burst_tier  # noqa: E402
import sass  # noqa: E402

#: THE MACHINE, an architecture: the cycles an HMMA holds the tensor pipe a scheduler (sm_86: bf16 with
#: fp32 accumulate at half rate, calibration `hmma_1w` 32.4; sm_80 full rate) and the issue-to-result
#: latency of the scoreboard classes (calibration.md's rows where one exists, else the microbenchmark
#: literature, bounded above), and the HMMAs the pipe queues past the one it executes (`hmma_queue`: the
#: calibration row with four loads after each burst hid 35 of their 42 cycles, one slot's worth, so two in
#: flight). A cubin of an architecture without a row is refused.
MACHINES = {
    "sm_86": {"hmma_pipe": 32.5, "latency": {"LDSM": 32, "LDS": 26, "LDG": 400, "SHFL": 26, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 5, "DEFAULT": 6}, "read": 6, "hmma_queue": 0,
              "schedulers": 4, "mem_wavefront": 1.0, "mem_sector": 2.0, "mem_queue": 4.0, "issue": {"RED": 12, "ATOM": 12, "ATOMG": 12,
                                                                                    "STG": 4, "LDG": 4}},
    "sm_80": {"hmma_pipe": 16.0, "latency": {"LDSM": 32, "LDS": 26, "LDG": 400, "SHFL": 26, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 5, "DEFAULT": 6}, "read": 6, "hmma_queue": 0,
              "schedulers": 4, "mem_wavefront": 1.0, "mem_sector": 2.0, "mem_queue": 4.0, "issue": {"RED": 12, "ATOM": 12, "ATOMG": 12,
                                                                                    "STG": 4, "LDG": 4}},
}


def mem_cost(op: str, text: str, mach: dict) -> float:
    """The cycles an instruction holds the SM's memory pipe: a 128-byte shared wavefront a cycle (`mem_wavefront`:
    the SM's shared-memory bandwidth, calibration.md's `matrix_load` rows), a global sector `mem_sector`. The
    counts are the instruction's own: an `ldmatrix` of `k` matrices `k` wavefronts, a 128-bit shared access four,
    a coalesced global access its width's sectors -- a divergent one is not read off the text."""
    if op == "LDSM":
        m = re.search(r"\.(\d)\s", text + " ")
        return (int(m.group(1)) if m else 4) * mach["mem_wavefront"]
    if op in ("LDS", "STS", "LDGSTS"):
        width = 4 if ".128" in text else 2 if ".64" in text else 1
        return width * mach["mem_wavefront"]
    if op in ("SHFL", "ATOMS"):
        return mach["mem_wavefront"]
    if op in ("LDG", "STG"):
        return (4 if ".128" in text else 2 if ".64" in text else 1) * mach["mem_sector"]
    if op in ("RED", "ATOM", "ATOMG"):
        return mach["mem_sector"]
    return 0.0


ENC = re.compile(r"\s*/\*([0-9a-f]{4,5})\*/\s+(.*?);\s*/\*\s*0x([0-9a-f]{16})\s*\*/")
ENC_HIGH = re.compile(r"\s*/\*\s*0x([0-9a-f]{16})\s*\*/")


def control(high: int) -> dict:
    """The control word off the encoding's high 64 bits: bits 41..61 of it."""
    c = high >> 41
    return {"stall": c & 0xF, "yield": (c >> 4) & 1, "wr": (c >> 5) & 7, "rd": (c >> 8) & 7,
            "wait": (c >> 11) & 0x3F, "reuse": (c >> 17) & 0xF}


def encoded(cubin: Path) -> dict[int, tuple[str, dict]]:
    """address -> (instruction text, control bits)."""
    out: dict[int, list] = {}
    addr = None
    for line in subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True,
                               check=True).stdout.splitlines():
        m = ENC.match(line)
        if m:
            addr = int(m.group(1), 16)
            out[addr] = [m.group(2).strip(), None]
            continue
        m = ENC_HIGH.match(line)
        if m and addr is not None and out[addr][1] is None:
            out[addr][1] = control(int(m.group(1), 16))
    return {a: (t, c) for a, (t, c) in out.items() if c is not None}


def opcode(text: str) -> str:
    return re.sub(r"^@!?U?P\w+\s+", "", text).split()[0].split(".")[0] if text else "?"


def branch_target(text: str) -> int | None:
    m = re.search(r"\b(0x[0-9a-f]+)\s*$", text)
    return int(m.group(1), 16) if m else None


def machine(cubin: Path) -> str:
    """The cubin's architecture off its disassembly; refused unless `MACHINES` describes it."""
    m = re.search(r"\.target\s+(sm_\d+)", sass.disassemble(cubin)[:4000])
    arch = m.group(1) if m else "?"
    if arch not in MACHINES:
        raise SystemExit(f"no machine model for {arch}; the models are {sorted(MACHINES)}")
    return arch


def span_of(source: Path, function: str) -> tuple[int, int]:
    lines = source.read_text().splitlines()
    for name, decl, sig, start, end in burst_tier.functions(lines):
        if name == function:
            return sig + 1, end + 1
    raise SystemExit(f"{source.name} has no function {function}")


def attributed(cubin: Path, source: Path, function: str) -> list[tuple[int, str, dict]]:
    """The instructions any inline frame of which lies in `function`'s lines, in address order."""
    lo, hi = span_of(source, function)
    frames = sass.frames(sass.disassemble(cubin, "--print-line-info-inline", "-gi"))
    enc = encoded(cubin)
    return [(a, *enc[a]) for a in sorted(enc)
            if any(f == source.name and lo <= ln <= hi for f, ln in frames.get(a, []))]


def loop_of(cubin: Path, source: Path, function: str) -> list[tuple[int, str, dict]]:
    """One iteration of the function's loop: its attributed instructions from the target of the last backward
    branch to that branch; the whole attribution when there is no loop."""
    body = attributed(cubin, source, function)
    if not body:
        raise SystemExit(f"no SASS attributed to {function}")
    back = [i for i, (a, t, _) in enumerate(body)
            if opcode(t) == "BRA" and branch_target(t) is not None and branch_target(t) < a]
    if back:
        target = branch_target(body[back[-1]][1])
        first = min(i for i, (a, _, _) in enumerate(body) if a >= target)
        body = body[first : back[-1] + 1]
    return body
