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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lint"))

import life_ranges  # noqa: E402
import sass_control as sc  # noqa: E402

#: the machine row the last `main` selected; `simulate` reads these
HMMA_PIPE = sc.MACHINES["sm_86"]["hmma_pipe"]
LATENCY = sc.MACHINES["sm_86"]["latency"]
HMMA_QUEUE = sc.MACHINES["sm_86"]["hmma_queue"]


def select(arch: str) -> None:
    global HMMA_PIPE, LATENCY, HMMA_QUEUE
    m = sc.MACHINES[arch]
    HMMA_PIPE, LATENCY, HMMA_QUEUE = m["hmma_pipe"], m["latency"], m["hmma_queue"]


def simulate(body, warps: int, iterations: int, trace: bool) -> dict:
    """Issue times of `iterations` trips of the loop on `warps` warps sharing one pipe and one issue port."""
    n = len(body)
    period_guess = sum(HMMA_PIPE for _, t, _ in body if sc.opcode(t) == "HMMA") * warps
    starts = [w * period_guess / warps for w in range(warps)]
    pipe_done: list[float] = []  #: the completion times of the HMMAs in the pipe, in order
    port_busy: set[int] = set()
    issued = []
    state = [{"t": starts[w], "bars": [0.0] * 6, "i": 0, "it": 0, "last": None} for w in range(warps)]
    marks = {w: [] for w in range(warps)}

    while any(s["it"] < iterations for s in state):
        w = min((s for s in state if s["it"] < iterations), key=lambda s: s["t"])
        s = state[state.index(w)]
        a, text, c = body[s["i"]]
        op = sc.opcode(text)
        ready = s["t"]
        for b in range(6):
            if (c["wait"] >> b) & 1:
                ready = max(ready, s["bars"][b])
        if op == "HMMA":
            #: a slot in the queue: the pipe executes one and holds `HMMA_QUEUE` behind it
            pending = [d for d in pipe_done if d > ready]
            if len(pending) > HMMA_QUEUE:
                ready = pending[-HMMA_QUEUE - 1]
        t = int(ready)
        while t in port_busy:
            t += 1
        port_busy.add(t)

        if op == "HMMA":
            pipe_done = [d for d in pipe_done if d > t]
            pipe_done.append(max(t, pipe_done[-1] if pipe_done else t) + HMMA_PIPE)
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
    hmmas = sum(1 for _, t, _ in body if sc.opcode(t) == "HMMA")
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
    arch = sc.machine(cubin)
    select(arch)
    print(f"machine {arch}: an HMMA holds the pipe {HMMA_PIPE} cycles a scheduler")
    body = sc.loop_of(cubin, a.source, a.function)
    print(f"loop of {a.source.name}:{a.function}: {len(body)} instructions, "
          f"{sum(1 for _, t, _ in body if sc.opcode(t) == 'HMMA')} HMMAs, {body[0][0]:#07x}..{body[-1][0]:#07x}")
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
