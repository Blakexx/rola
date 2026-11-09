#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BURST GATE (KERNEL_STANDARDS §23): what the SASS of every burst loop must satisfy, read off a
cubin with line information, deterministically.

  PURITY   -- a `//: @burst` function's instructions hold no vote, barrier, convergence bracket,
              call or branch: the ones the compiler inserts, which the source lint cannot see.
  COVERAGE -- a named loop's period for ONE warp alone, simulated from the control words with the
              tensor pipe as a resource (`pipe_sim`), against its HMMAs' pipe time -- the occupancy --
              and the exposed latency the control words admit, each named dependence: a READING. The
              gate on every MMA loop of every kernel is the SASS gate's step cover (KERNEL_STANDARDS §32).
  REGISTERS -- the arm's allocation against the thresholds a warp a scheduler costs: 248 buys two,
              168 three.

    bazel run //tools:burst_gate -- --arm 0 --source csrc/rola/src/carry/ampere/carry_kernel.cuh \
        [--loop FILE:FUNCTION ...]

The loops are named by their user functions (`--loop`), since every `Burst::run` user inlines the same lines; a name
the source no longer defines is refused. Exit 1 on a purity red.
Docs: docs/internals/tools/burst_gate.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "lint"))

import burst_tier  # noqa: E402
import life_ranges  # noqa: E402
import pipe_sim  # noqa: E402
import sass  # noqa: E402
import sass_control as sc  # noqa: E402

FORBIDDEN = re.compile(r"^(?:@!?U?P\w+\s+)?(VOTE\w*|BAR\w*|BSSY|BSYNC|CALL\w*|WARPSYNC|MEMBAR\w*|BRA\w*|EXIT|RET\w*)\b")
REGISTER_FILE = 16384       #: 32-bit registers a scheduler partition on this generation
ALLOCATION_GRAIN = 8        #: registers a thread, the allocation unit


def burst_functions() -> list[tuple[str, str, int, int]]:
    out = []
    for path in sorted(burst_tier.SRC.rglob("*.cu*")):
        lines = path.read_text().splitlines()
        for name, decl, sig, start, end in burst_tier.functions(lines):
            block = "\n".join(lines[decl:sig])
            if "@burst" in block and "@burst-exempt" not in block:
                out.append((path.name, name, sig + 1, end + 1))
    return out


def purity(cubin: Path, functions, kernel: str | None = None) -> tuple[bool, list[str], dict]:
    frames = sass.frames(sass.section(sass.disassemble(cubin, "--print-line-info-inline", "-gi"), kernel, cubin.name))
    enc = sc.encoded(cubin, kernel)
    rows = {name: collections.Counter() for _, name, _, _ in functions}
    owner: dict[int, str] = {}
    for off in enc:
        chain = frames.get(off, [])
        for file, name, lo, hi in functions:
            if any(f == file and lo <= ln <= hi for f, ln in chain):
                owner[off] = name
                break

    for off, (ins, _c) in enc.items():
        name = owner.get(off)
        if name is None:
            continue
        c = rows[name]
        c["instr"] += 1
        op = sc.opcode(ins)
        c["hmma"] += op.startswith("HMMA")
        c["shfl"] += op.startswith("SHFL")
        if FORBIDDEN.match(ins):
            #: a branch out of the function is the enclosing loop's own control (a burst's exit or latch
            #: inlined under the burst's frames), not a branch in the burst; a vote on the constant
            #: predicate is the uniform datapath materializing a warp-uniform value, not a data vote
            #: (a branch's own frame, the innermost, says whose it is: a burst function's frames sit
            #: under the primitive's when its branch is the primitive's)
            inner = (frames.get(off) or [("", 0)])[0]
            span = next((lo, hi, file) for file, n, lo, hi in functions if n == name)
            loop_control = op == "BRA" and not (inner[0] == span[2] and span[0] <= inner[1] <= span[1])
            uniform_fact = op.startswith("VOTEU") and ins.rstrip(" ;").endswith("PT")
            if not loop_control and not uniform_fact:
                c["forbidden"] += 1
                c[f"op:{op}"] += 1
    lines, red = [], False
    for _, name, lo, hi in functions:
        c = rows[name]
        bad = {k[3:]: v for k, v in c.items() if k.startswith("op:")}
        red |= c["forbidden"] > 0
        lines.append(f"{'RED ' if c['forbidden'] else 'ok  '} purity {name} ({lo}-{hi}): instr {c['instr']} "
                     f"hmma {c['hmma']} shfl {c['shfl']}" + (f" :: {bad}" if bad else ""))
    return not red, lines, {n: dict(c) for n, c in rows.items()}


def exposed_latency(body, mach: dict) -> tuple[float, list[str]]:
    """Two iterations of the loop by the control words alone: the second's exposed cycles -- each consumer's wait
    on a barrier whose producer's latency the placed distance did not cover -- with the worst dependences named."""
    lat, read_lat = mach["latency"], mach["read"]
    bars: dict[int, tuple[float, str, int]] = {}
    t = 0.0
    exposed = 0.0
    named = collections.Counter()
    for it in range(2):
        for i, (_a, text, c) in enumerate(body):
            op = sc.opcode(text)
            for b in range(6):
                if (c["wait"] >> b) & 1 and b in bars:
                    ready, producer, pidx = bars[b]
                    if ready > t:
                        if it == 1:
                            exposed += ready - t
                            named[f"{producer}@{pidx}->{op}@{i}"] += ready - t
                        t = ready
            if c["wr"] < 6:
                bars[c["wr"]] = (t + lat.get(op, lat["DEFAULT"]), op, i)
            if c["rd"] < 6:
                bars[c["rd"]] = (t + read_lat, op, i)
            t += max(1, c["stall"])
    return round(exposed, 1), [f"{k}: {v:.0f}" for k, v in named.most_common(4)]


def coverage(body, mach: dict, arch: str) -> dict:
    """One warp a scheduler, one on every scheduler of the SM sharing its memory pipe: the loop's period with the
    tensor pipe as a resource (`pipe_sim.simulate`) and the share of it the pipe is busy; beside it the exposed
    latency the control words admit."""
    pipe_sim.select(arch)
    alone = pipe_sim.simulate(body, 1, 6, False, sm_warps=pipe_sim.MACHINE["schedulers"])
    exposed, worst = exposed_latency(body, mach)
    hmmas = alone["hmmas"]
    return {"instructions": alone["instructions"], "hmmas": hmmas, "period": round(alone["period"], 1),
            "occupancy": round(100 * alone["occupancy"], 1),
            "pipe": round(sum(sc.mma_cost(t, mach) for _, t, _ in body), 1),
            "exposed": exposed, "worst": worst}


def registers(cubin: Path) -> dict:
    """The kernel's register allocation and the warps a scheduler it leaves room for."""
    out = subprocess.run([sass.tool("cuobjdump"), "-res-usage", str(cubin)], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+)", out)
    if not m:
        return {}
    regs = int(m.group(1))
    grain = -(-regs // ALLOCATION_GRAIN) * ALLOCATION_GRAIN
    fit = REGISTER_FILE // (grain * 32)
    threshold = (REGISTER_FILE // ((fit + 1) * 32)) // ALLOCATION_GRAIN * ALLOCATION_GRAIN
    return {"registers": regs, "allocated": grain, "warps_a_scheduler": fit, "next_threshold": threshold}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None, help="compile this carry arm with -lineinfo and read it")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--source", type=Path, required=True, help="the kernel source the burst functions live in")
    ap.add_argument("--loop", action="append", default=[], metavar="FILE:FUNCTION",
                    help="a loop to read for coverage: FILE:FUNCTION, the function whose lines the loop's "
                         "instructions are attributed to (a `Burst::run` user, since every user inlines the "
                         "same lines of `common/burst.cuh`)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--kernel", default=None,
                    help="the function to read, a regex over the cubin's (needed where it holds several)")
    a = ap.parse_args()
    if (a.cubin is None) == (a.arm is None):
        ap.error("give exactly one of --cubin or --arm")
    cubin = a.cubin if a.cubin else life_ranges.compile_arm(a.arm, a.arch)
    arch = sc.machine(cubin)
    mach = sc.MACHINES[arch]
    print(f"machine {arch}: an HMMA holds the pipe {mach['hmma_pipe']} cycles a scheduler")

    ok, lines, purity_rows = purity(cubin, burst_functions(), a.kernel)
    for line in lines:
        print(line)

    loops = [tuple(x.rsplit(":", 1)) for x in a.loop]
    cov = {}
    for file, function in loops:
        source = Path(file) if Path(file).exists() else a.source.parent / file
        key = f"{source.name}:{function}"
        r = coverage(sc.loop_of(cubin, source, function, a.kernel), mach, arch)
        cov[key] = r
        print(f"read coverage {key}: instr {r['instructions']} hmma {r['hmmas']} alone period {r['period']:.0f} of "
              f"pipe {r['pipe']:.0f} = {r['occupancy']:.1f}%; exposed latency {r['exposed']:.0f}"
              + (f" :: {r['worst']}" if r["worst"] else ""))

    regs = registers(cubin)
    if regs:
        print(f"registers {regs['registers']} (allocated {regs['allocated']}): {regs['warps_a_scheduler']} warp(s) a "
              f"scheduler; one more at <= {regs['next_threshold']}")
    if a.json:
        a.json.write_text(json.dumps({"machine": arch, "purity": purity_rows, "coverage": cov, "registers": regs},
                                     sort_keys=True) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
