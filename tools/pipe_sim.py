#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STATIC PIPE SIMULATOR (KERNEL_STANDARDS §23): a burst loop's steady-state tensor-pipe
occupancy, computed from its SASS alone -- the control bits ptxas wrote into every instruction
(stall count, yield, the scoreboard barriers it sets and waits on, the reuse flags) and a latency
table from this card's calibrations -- for one warp and for two warps sharing a scheduler.

    python tools/pipe_sim.py --arm 0 --loop common/burst.cuh:run [--warps 1|2] [--iterations 6] [--trace]

THE MODEL. A warp issues in order. Instruction i issues at the latest of: the previous issue
plus the previous instruction's stall count; the completion of every scoreboard barrier its wait
mask names (a barrier is set by a producer and completes at the producer's issue plus its
latency: `LATENCY` by opcode class, calibration.md); and, for an HMMA, the tensor pipe's next
free slot -- the pipe takes one HMMA every `HMMA_PIPE` cycles a scheduler and a warp cannot issue
past an HMMA that has no slot. Two warps run the same loop offset by half a period and share the
pipe and the issue port. The loop is the SASS whose inline chain touches the named function's
lines, from its first instruction to its backward branch. Occupancy = HMMAs x HMMA_PIPE over the
steady-state period.

What it is for: the ratcheted floor per burst loop (the pipe stays full or the build fails), and
reading a lost form without a trace. What it is not: a cycle-accurate machine -- memory latency
is a constant a class, the issue port is one instruction a cycle, and the pair's offset is an
assumption; its numbers are read against the trace once (`--calibrate` prints both).
Docs: docs/internals/tools/pipe_sim.md.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lint"))

import burst_tier  # noqa: E402
import life_ranges  # noqa: E402
import sass  # noqa: E402

#: THE MACHINE, an architecture: the control word's layout (Volta through Ampere share it; Hopper's
#: wgmma is asynchronous and this model does not describe it), the cycles an HMMA holds the tensor pipe
#: a scheduler (sm_86: bf16 with fp32 accumulate at half rate, calibration hmma_1w 32.4; sm_80 full
#: rate), and the issue-to-result latency by opcode class for the scoreboard classes (the fixed-latency
#: classes are the stall counts' job) -- from calibration.md's rows where a row exists, else the
#: microbenchmark literature. A cubin of an architecture without a row here is refused.
MACHINES = {
    "sm_86": {"hmma_pipe": 32.5, "latency": {"LDSM": 30, "LDS": 25, "LDG": 400, "SHFL": 24, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 5, "DEFAULT": 6}},
    "sm_80": {"hmma_pipe": 16.0, "latency": {"LDSM": 30, "LDS": 25, "LDG": 400, "SHFL": 24, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 5, "DEFAULT": 6}},
}
HMMA_PIPE = 32.5
LATENCY = MACHINES["sm_86"]["latency"]


def machine(cubin: Path) -> str:
    """The cubin's architecture off its disassembly header; refused unless `MACHINES` describes it."""
    head = sass.disassemble(cubin)[:4000]
    m = re.search(r"\.target\s+(sm_\d+)", head)
    arch = m.group(1) if m else "?"
    if arch not in MACHINES:
        raise SystemExit(f"pipe_sim: no machine model for {arch}; the models are {sorted(MACHINES)}")
    return arch


ENC = re.compile(r"\s*/\*([0-9a-f]{4,5})\*/\s+(.*?);\s*/\*\s*0x([0-9a-f]{16})\s*\*/")
ENC_HIGH = re.compile(r"\s*/\*\s*0x([0-9a-f]{16})\s*\*/")


def control(high: int) -> dict:
    """The control word: bits 105..125 of the 128-bit instruction (41..61 of the high word)."""
    c = high >> 41
    return {"stall": c & 0xF, "yield": (c >> 4) & 1, "wr": (c >> 5) & 7, "rd": (c >> 8) & 7,
            "wait": (c >> 11) & 0x3F, "reuse": (c >> 17) & 0xF}


def encoded(cubin: Path) -> dict[int, tuple[str, dict]]:
    """address -> (instruction text, control bits) off cuobjdump's raw words."""
    out = {}
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


def loop_of(cubin: Path, source: Path, function: str) -> list[tuple[int, str, dict]]:
    """The instructions attributed to `function` (any inline frame in its lines), from the first to the last
    backward branch inside them: one iteration of the loop, in address order."""
    lines = source.read_text().splitlines()
    span = next(((sig + 1, end + 1) for name, decl, sig, start, end in burst_tier.functions(lines) if name == function),
                None)
    if span is None:
        raise SystemExit(f"{source.name} has no function {function}")
    frames = sass.frames(sass.disassemble(cubin, "--print-line-info-inline", "-gi"))
    enc = encoded(cubin)
    inside = [a for a in sorted(enc) if any(f == source.name and span[0] <= ln <= span[1] for f, ln in frames.get(a, []))]
    if not inside:
        raise SystemExit(f"no SASS attributed to {function}")
    body = [(a, *enc[a]) for a in inside]
    back = [i for i, (a, t, _) in enumerate(body) if opcode(t) == "BRA" and _target(t) is not None and _target(t) < a]
    if back:
        first = min(i for i, (a, _, _) in enumerate(body) if a >= _target(body[back[-1]][1]))
        body = body[first : back[-1] + 1]
    return body


def _target(text: str) -> int | None:
    m = re.search(r"\b(0x[0-9a-f]+)\s*$", text)
    return int(m.group(1), 16) if m else None


def simulate(body, warps: int, iterations: int, trace: bool) -> dict:
    """Issue times of `iterations` trips of the loop on `warps` warps sharing one pipe and one issue port."""
    n = len(body)
    period_guess = sum(HMMA_PIPE for _, t, _ in body if opcode(t) == "HMMA") * warps
    starts = [w * period_guess / warps for w in range(warps)]
    pipe_free = 0.0
    port_busy: set[int] = set()
    issued = []
    state = [{"t": starts[w], "bars": [0.0] * 6, "i": 0, "it": 0, "last": None} for w in range(warps)]
    marks = {w: [] for w in range(warps)}

    while any(s["it"] < iterations for s in state):
        w = min((s for s in state if s["it"] < iterations), key=lambda s: s["t"])
        s = state[state.index(w)]
        a, text, c = body[s["i"]]
        op = opcode(text)
        ready = s["t"]
        for b in range(6):
            if (c["wait"] >> b) & 1:
                ready = max(ready, s["bars"][b])
        if op == "HMMA":
            ready = max(ready, pipe_free)
        t = int(ready)
        while t in port_busy:
            t += 1
        port_busy.add(t)

        if op == "HMMA":
            pipe_free = t + HMMA_PIPE
        lat = LATENCY.get(op, LATENCY["DEFAULT"])
        if c["wr"] < 6:
            s["bars"][c["wr"]] = t + lat
        if c["rd"] < 6:
            s["bars"][c["rd"]] = t + min(lat, 20)
        if trace:
            issued.append((t, state.index(s), s["it"], a, text[:48]))
        s["t"] = t + max(1, c["stall"])
        s["i"] += 1
        if s["i"] == n:
            s["i"] = 0
            s["it"] += 1
            marks[state.index(s)].append(t)

    # steady state: the last two iterations' span on warp 0
    m0 = marks[0]
    period = (m0[-1] - m0[-3]) / 2 if len(m0) >= 3 else (m0[-1] - m0[0]) / max(1, len(m0) - 1)
    hmmas = sum(1 for _, t, _ in body if opcode(t) == "HMMA")
    return {"instructions": n, "hmmas": hmmas, "warps": warps, "period": period,
            "occupancy": hmmas * warps * HMMA_PIPE / period if period else float("nan"), "trace": issued}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--source", type=Path, default=Path("csrc/rola/src/common/burst.cuh"))
    ap.add_argument("--function", default="run", help="the function whose lines hold the loop")
    ap.add_argument("--warps", type=int, default=0, help="1 or 2; 0 for both")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--trace", action="store_true", help="print every issue")
    a = ap.parse_args()
    if (a.cubin is None) == (a.arm is None):
        ap.error("give exactly one of --cubin or --arm")
    cubin = a.cubin if a.cubin else life_ranges.compile_arm(a.arm, a.arch)
    global HMMA_PIPE, LATENCY
    arch = machine(cubin)
    HMMA_PIPE, LATENCY = MACHINES[arch]["hmma_pipe"], MACHINES[arch]["latency"]
    print(f"machine {arch}: an HMMA holds the pipe {HMMA_PIPE} cycles a scheduler")
    body = loop_of(cubin, a.source, a.function)
    print(f"loop of {a.source.name}:{a.function}: {len(body)} instructions, "
          f"{sum(1 for _, t, _ in body if opcode(t) == 'HMMA')} HMMAs, {body[0][0]:#07x}..{body[-1][0]:#07x}")
    for warps in ([a.warps] if a.warps else [1, 2]):
        r = simulate(body, warps, a.iterations, a.trace)
        print(f"  {warps} warp(s): period {r['period']:.0f} cycles, pipe occupancy {100 * r['occupancy']:.1f}%")
        if a.trace:
            for t, w, it, addr, text in r["trace"]:
                if it == a.iterations - 1:
                    print(f"    t {t:6d} w{w} {addr:#07x} {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
