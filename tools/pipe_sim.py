#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STATIC PIPE SIMULATOR (KERNEL_STANDARDS §23): a burst loop's steady-state tensor-pipe
occupancy, computed from its SASS alone -- the control bits ptxas wrote into every instruction
(stall count, yield, the scoreboard barriers it sets and waits on, the reuse flags) and a latency
table from this card's calibrations -- for one warp and for two warps sharing a scheduler.

    python tools/pipe_sim.py --arm 0 --loop common/burst.cuh:run [--warps 1|2] [--sm-warps 8] [--iterations 6] [--trace]
    python tools/pipe_sim.py --calibrate rola_cu13/_C_parts.abi3.so      # the model against the probe's rows

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
MACHINE = sc.MACHINES["sm_86"]
HMMA_PIPE = MACHINE["hmma_pipe"]
LATENCY = MACHINE["latency"]
HMMA_QUEUE = MACHINE["hmma_queue"]


def select(arch: str) -> None:
    global MACHINE, HMMA_PIPE, LATENCY, HMMA_QUEUE
    MACHINE = sc.MACHINES[arch]
    HMMA_PIPE, LATENCY, HMMA_QUEUE = MACHINE["hmma_pipe"], MACHINE["latency"], MACHINE["hmma_queue"]


def simulate(body, warps: int, iterations: int, trace: bool, sm_warps: int = 0) -> dict:
    """Issue times of `iterations` trips of the loop on `warps` warps a scheduler sharing its pipe and issue port,
    `sm_warps` warps in all running the loop on the SM (a scheduler apiece up to `schedulers`), every warp's
    memory instructions through ONE memory pipe in issue order (`sass_control.mem_cost`) behind a queue of
    `mem_queue` cycles: a load issues when the queue has room and completes when the pipe has taken it."""
    n = len(body)
    sm_warps = max(sm_warps, warps)
    schedulers = min(MACHINE["schedulers"], -(-sm_warps // warps))
    total = min(sm_warps, schedulers * warps)
    pipe_done: list[list[float]] = [[] for _ in range(schedulers)]  #: HMMA completion times a scheduler, in order
    port_busy: list[set[int]] = [set() for _ in range(schedulers)]
    mem_free = 0.0  #: when the SM's memory pipe takes its next instruction
    issued = []
    #: every warp starts at once: the probe rows read the same with the pair half a period apart, and the
    #: kernel's trace has the warps in lockstep (fold starts spread ~150 cycles of a 100K window)
    state = [{"w": w, "sched": w % schedulers, "t": 0.0, "bars": [0.0] * 6, "i": 0, "it": 0}
             for w in range(total)]
    marks = {w: [] for w in range(total)}

    while any(s["it"] < iterations for s in state):
        s = min((s for s in state if s["it"] < iterations), key=lambda s: s["t"])
        k = s["sched"]
        a, text, c = body[s["i"]]
        op = sc.opcode(text)
        ready = s["t"]
        for b in range(6):
            if (c["wait"] >> b) & 1:
                ready = max(ready, s["bars"][b])
        if op == "HMMA":
            #: a slot in the queue: the pipe executes one and holds `HMMA_QUEUE` behind it
            pending = [d for d in pipe_done[k] if d > ready]
            if len(pending) > HMMA_QUEUE:
                ready = pending[-HMMA_QUEUE - 1]
        cost = sc.mem_cost(op, text, MACHINE)
        if cost:
            #: the memory pipe's queue holds `mem_queue` cycles of work: a memory instruction does not issue
            #: while the backlog is longer (the profiler's `mio_throttle`), so a warp's loads are taken soon
            #: after they issue and its next loads wait at issue, not at their landing
            ready = max(ready, mem_free - MACHINE["mem_queue"])
        t = int(ready)
        while t in port_busy[k]:
            t += 1
        #: a global access holds the scheduler's issue port `issue` cycles (its lanes' addresses and data
        #: through the LSU: calibration.md's reduction rows, ~12 a coalesced reduction), other instructions one
        for i in range(MACHINE["issue"].get(op, 1)):
            port_busy[k].add(t + i)

        if op == "HMMA":
            pipe_done[k] = [d for d in pipe_done[k] if d > t]
            pipe_done[k].append(max(t, pipe_done[k][-1] if pipe_done[k] else t) + HMMA_PIPE)
        lat = LATENCY.get(op, LATENCY["DEFAULT"])
        done = t + lat
        if cost:
            taken = max(float(t), mem_free)
            mem_free = taken + cost
            done = taken + lat
        if c["wr"] < 6:
            s["bars"][c["wr"]] = done
        if c["rd"] < 6:
            s["bars"][c["rd"]] = t + min(lat, 20)
        if trace:
            issued.append((t, s["w"], s["it"], a, text[:48]))
        s["t"] = t + max(1, c["stall"])
        s["i"] += 1
        if s["i"] == n:
            s["i"] = 0
            s["it"] += 1
            marks[s["w"]].append(t)

    # steady state: the last two iterations' span on warp 0
    m0 = marks[0]
    period = (m0[-1] - m0[-3]) / 2 if len(m0) >= 3 else (m0[-1] - m0[0]) / max(1, len(m0) - 1)
    hmmas = sum(1 for _, t, _ in body if sc.opcode(t) == "HMMA")
    return {"instructions": n, "hmmas": hmmas, "warps": warps, "sm_warps": total, "period": period,
            "occupancy": hmmas * warps * HMMA_PIPE / period if period else float("nan"), "trace": issued}


#: the calibration rows the model is read against (`measure/harness/bench_carry_calib.py`): row name -> the probe
#: kernel's (warps, mode, burst); a row's measured value is cycles an op a warp, an op the row's HMMA or load
CALIBRATION_ROWS = {
    "hmma_1w": (4, 0, 1), "hmma_2w": (8, 0, 1),
    "hmma_load_1": (8, 10, 1), "hmma_load_4": (8, 10, 4), "hmma_load_4_1w": (4, 10, 4), "hmma_load_4_free": (8, 11, 4),
    "hmma_reduce_1": (8, 12, 1), "hmma_reduce_2": (8, 12, 2), "hmma_reduce_4": (8, 12, 4), "hmma_reduce_8": (8, 12, 8),
    "hmma_reduce_div_4": (8, 13, 4), "hmma_reduce_div_8": (8, 13, 8),
    "hmma_frag_1w": (4, 14, 18), "hmma_frag_2w": (8, 14, 18), "hmma_frag_chain_1w": (4, 15, 18),
    "hmma_frag_chain_2w": (8, 15, 18),
    "hmma_alu_32_1w": (4, 21, 32), "hmma_alu_32_2w": (8, 21, 32), "hmma_alu_64_1w": (4, 21, 64),
    "hmma_alu_64_2w": (8, 21, 64), "hmma_queue_16_1w": (4, 22, 16), "hmma_queue_40_1w": (4, 22, 40),
    "hmma_queue_80_1w": (4, 22, 80), "hmma_queue_40_2w": (8, 22, 40),
    "hmma_frag_chain_hooked_1w": (4, 23, 18), "hmma_frag_chain_hooked_2w": (8, 23, 18),
    "hmma_frag_chain_hooked2_1w": (4, 24, 18), "hmma_frag_chain_hooked2_2w": (8, 24, 18), "hmma_latency": (4, 25, 18),
    "hmma_operands_1w": (4, 26, 18), "hmma_operands_2w": (8, 26, 18),
    "matrix_load_4": (8, 9, 4), "matrix_load_16": (8, 9, 16),
    "matrix_load_rows_4": (8, 20, 4), "matrix_load_rows_16": (8, 20, 16), "matrix_load_rows_4_1w": (4, 20, 4),
}


def probe_loops(so: Path) -> dict[tuple[int, int, int], list]:
    """(warps, mode, burst) -> the loop of that `calib_kernel` instantiation in the part harness's module: its
    instructions with control bits, from the target of its last backward branch to the branch."""
    import re
    import subprocess

    import sass

    cubin = [c for c in sass.cubins(so) if "carry_parts" in c.name][0]
    select(sc.machine(cubin))
    text = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    funcs: dict[str, list] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        m = sc.ENC.match(line)
        if m:
            funcs[cur].append([int(m.group(1), 16), m.group(2).strip(), None])
            continue
        m = sc.ENC_HIGH.match(line)
        if m and funcs[cur] and funcs[cur][-1][2] is None:
            funcs[cur][-1][2] = sc.control(int(m.group(1), 16))
    out = {}
    for name, body in funcs.items():
        m = re.search(r"calib_kernelILi(\d+)ELi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        body = [(a, t, c) for a, t, c in body if c is not None]
        back = [i for i, (a, t, _) in enumerate(body)
                if sc.opcode(t) == "BRA" and sc.branch_target(t) is not None and sc.branch_target(t) < a]
        if not back:
            continue
        target = sc.branch_target(body[back[-1]][1])
        first = min(i for i, (a, _, _) in enumerate(body) if a >= target)
        out[tuple(map(int, m.groups()))] = body[first : back[-1] + 1]
    return out


def calibrate(so: Path) -> int:
    """THE MODEL AGAINST THE PROBE (the `--calibrate` step): every calibration row's loop simulated as the probe
    runs it (its warps on the SM, `warps / 4` a scheduler) and printed beside the row's latest measured value from
    the results store. A row the model misses names what the model lacks; a floor is ratcheted only on loops
    whose rows it reproduces (KERNEL_STANDARDS §23)."""
    from rola_results import outputs

    measured: dict[str, tuple[float, str]] = {}  #: a row's LATEST clocked sample, by its stamp
    for _, _, out in outputs("calibration"):
        if isinstance(out, dict) and out.get("clock"):
            for r in out.get("rows", []):
                if out.get("utc", "") >= measured.get(r["name"], (0.0, ""))[1]:
                    measured[r["name"]] = (r["cycles_an_op_a_warp"], out.get("utc", "?"))
    loops = probe_loops(so)
    print(f"machine: an HMMA holds the pipe {HMMA_PIPE} cycles a scheduler; a shared wavefront "
          f"{MACHINE['mem_wavefront']}, a global sector {MACHINE['mem_sector']} cycles of the SM's memory pipe")
    print(f"{'row':22s} {'loop':>5s} {'ops':>4s} {'model':>8s} {'measured':>9s} {'miss':>7s}  measured at")
    worst = 0.0
    for name, key in CALIBRATION_ROWS.items():
        loop = loops.get(key)
        if loop is None:
            print(f"{name:22s} (no such instantiation in {so.name})")
            continue
        warps, _, _ = key
        r = simulate(loop, max(1, warps // MACHINE["schedulers"]), 8, False, sm_warps=warps)
        #: a row's op: its HMMA where it has one, else its load
        ops = sum(1 for _, t, _ in loop if sc.opcode(t) == "HMMA") or sum(1 for _, t, _ in loop if sc.opcode(t) == "LDSM")
        model = r["period"] / ops if ops else float("nan")
        if name in measured:
            m, at = measured[name]
            miss = 100 * (model - m) / m
            worst = max(worst, abs(miss))
            print(f"{name:22s} {len(loop):5d} {ops:4d} {model:8.1f} {m:9.1f} {miss:+6.1f}%  {at}")
        else:
            print(f"{name:22s} {len(loop):5d} {ops:4d} {model:8.1f} {'-':>9s}")
    print(f"worst miss {worst:.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--source", type=Path, default=Path("csrc/rola/src/common/burst.cuh"))
    ap.add_argument("--function", default="run", help="the function whose lines hold the loop")
    ap.add_argument("--warps", type=int, default=0, help="1 or 2; 0 for both")
    ap.add_argument("--sm-warps", type=int, default=0,
                    help="warps on the SM running the loop, sharing its memory pipe (default: the scheduler's)")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--trace", action="store_true", help="print every issue")
    ap.add_argument("--calibrate", type=Path, default=None, metavar="PARTS_SO",
                    help="read the model against the calibration probe's rows in this part harness module")
    a = ap.parse_args()
    if a.calibrate is not None:
        return calibrate(a.calibrate)
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
        r = simulate(body, warps, a.iterations, a.trace, sm_warps=a.sm_warps)
        print(f"  {warps} warp(s) a scheduler, {r['sm_warps']} on the SM: period {r['period']:.0f} cycles, "
              f"pipe occupancy {100 * r['occupancy']:.1f}%")
        if a.trace:
            for t, w, it, addr, text in r["trace"]:
                if it == a.iterations - 1:
                    print(f"    t {t:6d} w{w} {addr:#07x} {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
