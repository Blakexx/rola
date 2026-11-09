#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name]
[--max-per-hmma N]`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass):
  local memory  -- LDL/STL instructions, or a nonzero stack frame (`cuobjdump -res-usage`)
  outlined collectives -- a `CALL.REL.NOINC` into a convergence subroutine (a short block, `_SUBROUTINE`
                  instructions at most, that syncs the warp, runs one collective and returns) that a CONVERGED
                  warp executes. ptxas guards the ones it emits: a `BRA.DIV` into a fallback block, or a
                  `BRA.CONV` past the call, so a converged warp runs the inline collective; a guarded call is cold
                  code, counted (`guarded`) and not red. A call to an arithmetic slow path is not one, nor a long
                  region ptxas reconverges once on entry
  predicated HMMA -- `@P HMMA`: if-converted tensor work
  instructions per HMMA -- the static ratio, a ceiling the arm states (issue-bound above it)
  the conversion unit in a step -- F2I/I2F/F2F/MUFU inside an MMA loop: 16 lanes a cycle an SM where the FMA pipe's
                  forms (a magic-constant FFMA, a PRMT) run at 64 (KERNEL_STANDARDS section 29)
  global memory in a step -- LDG/STG/RED/ATOMG inside an MMA loop: a step's operands are staged ahead by `cp.async`
                  (LDGSTS, allowed), and a reduction to global memory belongs to a drain outside the steps
  the step cover -- each innermost MMA loop simulated from ptxas's control words (`pipe_sim`) with the warps a
                  scheduler its registers admit sharing its issue port and the SM's memory pipe, its pipe share
                  under `STEP_LINE` (KERNEL_STANDARDS section 32); a debt kernel held at its floor (`--write-floors`
                  records them)
On a whole binary it also reds a PTX image (closed-world rule 2: cubins only), and on a whole extension every kernel
the source defines (`tools/kernels.py`) that no cubin carries: a gate that cannot see a kernel has not gated it. An instrument's library (`--instrument`, bazel/defs.bzl's `rola_instrument`)
carries its own kernels and owes none of the source's. Its kernels are PROBES, and a probe that keeps a step
signature on purpose (a global reduction between its HMMAs, the cost it measures) declares it in the instrument's
DECLARATIONS, by function (`sm_80 <pattern>` where one architecture's SASS keeps it alone), with that cost; an
undeclared signature is red, and so is a declaration an architecture the binary carries does not show. It prints the
counts it reads, the same way `ratify` prints registers and
spills, so a clean run is a line of zeros and a red names the signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import kernels  # noqa: E402 -- path insert must precede these imports
import sass  # noqa: E402
import sass_control  # noqa: E402

#: an instruction line's opcode, off the shared grammar (`tools/sass.py`)
INSTR = sass.INSTRUCTION
#: the same line whole: address, predicate, opcode, operands
WHOLE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.$]+)\s*([^;]*);")
_COLLECTIVE = re.compile(r"^(SHFL|VOTE|MATCH|REDUX|BAR)")

#: the always-false predicate: ptxas keeps a branch under it that it never takes
_NEVER = "@!PT"
#: an unconditional transfer: nothing falls through it
_TERMINATOR = re.compile(r"^(BRA|EXIT|RET|JMP|BRX|JMX)$|^(EXIT|RET)\.")
#: a convergence subroutine's length at most: its operand moves, `WARPSYNC`, the collective (`RET` not counted)
_SUBROUTINE = 8


def cubins_of(path: Path, member: str | None) -> list[Path]:
    return [image for image in sass.cubins(path) if member is None or member in image.name]


_RES = re.compile(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+)")


def resources(cubin: Path) -> dict[str, tuple[int, int]]:
    """`{function: (registers, stack frame bytes)}` off `cuobjdump -res-usage`."""
    text = subprocess.run([sass.tool("cuobjdump"), "-res-usage", str(cubin)], capture_output=True,
                          text=True, check=True).stdout
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in _RES.finditer(text)}


#: the last `gate` call's signature per function, for `--json`
STATS: dict[str, dict] = {}


def convergence_calls(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """``(hot, guarded)``: the calls into a convergence subroutine a converged warp executes, and those it never does
    (in a block a `BRA.DIV` enters, past a `BRA.CONV` in the call's block, or in a dead block: one only a branch under
    the always-false predicate `@!PT` enters, right after an unconditional transfer). ``lines`` are ``(address,
    opcode, operands, predicate)``, the function's own; ``labels`` ``{name: line index}``. A target is an address
    (`cuobjdump`) or a label (`nvdisasm`: `` `($__internal_1_$__cuda_sm70_shflsync_bfly_p) ``)."""
    at = {addr: k for k, (addr, _, _, _) in enumerate(lines)}

    def resolve(operands: str) -> int | None:
        address = re.match(r"(?:~URZ,\s*)?(0x[0-9a-f]+)", operands)
        label = re.search(r"\(([^)]+)\)", operands)
        return at.get(int(address.group(1), 16)) if address else labels.get(label.group(1)) if label else None

    def subroutine(start: int) -> bool:
        block = []
        for _, op, _, _ in lines[start:start + _SUBROUTINE + 1]:
            if op.startswith("RET"):
                return "WARPSYNC" in block and any(_COLLECTIVE.match(o) for o in block)
            block.append(op)
        return False

    def targets(prefix: str, live: bool) -> set[int]:
        found = (resolve(o) for _, op, o, p in lines if op.startswith(prefix) and (p != _NEVER) == live)
        return {k for k in found if k is not None}

    taken = targets("BRA", True)
    starts = {0, *labels.values()} | taken | targets("BRA", False)
    fallback = targets("BRA.DIV", True)
    dead = {k for k in starts - taken
            if k > 0 and lines[k - 1][3] == "" and _TERMINATOR.match(lines[k - 1][1])}
    hot = guarded = 0
    block, conv = 0, False
    for k, (_, opcode, operands, predicate) in enumerate(lines):
        if k in starts:
            block, conv = k, False
        if predicate == _NEVER:
            continue
        conv = conv or opcode.startswith("BRA.CONV")
        target = resolve(operands) if opcode.startswith("CALL.REL") else None
        if target is not None and subroutine(target):
            if block in fallback or block in dead or conv:
                guarded += 1
            else:
                hot += 1
    return hot, guarded


_MMA = re.compile(r"^(HMMA|IMMA)")


def _resolver(lines: list[tuple[int, str, str, str]], labels: dict[str, int]):
    at = {addr: k for k, (addr, _, _, _) in enumerate(lines)}

    def resolve(operands: str) -> int | None:
        address = re.match(r"(0x[0-9a-f]+)", operands)
        label = re.search(r"\(([^)]+)\)", operands)
        return at.get(int(address.group(1), 16)) if address else labels.get(label.group(1)) if label else None

    return resolve


#: `mma_loops` once a function: four lints ask for it; the entry holds its `lines` so the id is never reused while
#: cached. -- see docs/internals/tools/sass_gate.md#cost
_LOOPS: dict[int, tuple[list, list[tuple[int, int]]]] = {}


def mma_loops(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> list[tuple[int, int]]:
    """The innermost loops that issue MMAs, as (first line, backward branch): a loop a backward branch, its body the
    lines from its target to it, innermost when no other loop that issues MMAs lies inside. A loop inside it that
    issues none -- a compare-and-swap spin (sm_80/86's shared fp32 add), a one-trip guard -- is part of its body, read
    once. A divergence fallback's return (the cold stub a `BRA.DIV` enters, read to its unconditional branch back into
    the code it left, however long: ptxas copies a step's remaining work into it) is no loop. Targets resolve as `convergence_calls`' do."""
    hit = _LOOPS.get(id(lines))
    if hit is not None and hit[0] is lines:
        return hit[1]
    loops = _mma_loops(lines, labels)
    _LOOPS[id(lines)] = (lines, loops)
    return loops


def _mma_loops(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> list[tuple[int, int]]:
    resolve = _resolver(lines, labels)
    cold = set()
    for _, opcode, operands, predicate in lines:
        stub = resolve(operands.split(",")[-1].strip()) if opcode.startswith("BRA.DIV") else None
        for j in range(stub, len(lines)) if stub is not None else ():
            cold.add(j)
            if lines[j][3] == "" and _TERMINATOR.match(lines[j][1]):
                break
    back = []
    for k, (_, opcode, operands, predicate) in enumerate(lines):
        target = resolve(operands) if opcode == "BRA" and predicate != _NEVER and k not in cold else None
        if target is not None and target <= k:
            back.append((target, k))
    mma = [(t, k) for t, k in back if any(_MMA.match(x[1]) for x in lines[t:k + 1])]
    return [(target, k) for target, k in mma
            if not any(target <= t and kk < k for t, kk in mma if (t, kk) != (target, k))]


def mma_gap(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """KERNEL_STANDARDS 27's interleave as ptxas emitted it: the longest run of instructions without an MMA in an
    innermost MMA loop's body, and the run around the loop (its tail into its head -- a rolled step loop's own gap, an
    unrolled one's fill and drain). In-order issue makes a run the time a warp leaves the tensor pipe to the
    scheduler's other warps."""
    inside = around = 0
    for target, k in mma_loops(lines, labels):
        at = [j for j in range(target, k + 1) if _MMA.match(lines[j][1])]
        inside = max(inside, *(b - a - 1 for a, b in zip(at, at[1:])), 0)
        around = max(around, k - target - (at[-1] - at[0]))
    return inside, around


#: KERNEL_STANDARDS section 29's debts: the kernels built before the step lints whose steps still read global memory
_STEP_GLOBAL_DEBT = {
    "carry_kernel": "the fold's steps past a list's 16th entry read records from global memory; the readout-first "
                    "rebuild stages every step's records",
    "produce_kernel": "126 global accesses inside its MMA loop: its operands are not staged by cp.async",
}


def step_classes(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """KERNEL_STANDARDS section 29: inside the innermost MMA loops (`mma_loops`), the instructions on the conversion
    unit and the global memory accesses other than `cp.async` (LDGSTS)."""
    xu = memory = 0
    for target, k in mma_loops(lines, labels):
        for j in range(target, k + 1):
            _, op, _, pred = lines[j]
            if pred == _NEVER:
                continue
            kind = sass.opcode_class(op)
            xu += kind == "conversion unit"
            memory += kind == "global" and not op.startswith("LDGSTS")
    return xu, memory


#: a register copy: what ptxas puts between a loop's exit test and its back-edge for a carried value it did not coalesce
_COPY = re.compile(r"^(MOV|IMAD\.MOV(\.U32)?)$")


def mma_loop_branches(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> int:
    """KERNEL_STANDARDS 27: the branches inside a step -- a convergence barrier (BSSY) or a forward conditional BRA in an
    innermost MMA loop (`mma_loops`; the BRA.DIV / BRA.CONV guards of a collective's cold fallback not counted). Each
    one ends ptxas's scheduling block inside the step. The loop's own exit laid out as its LATCH -- out of the loop,
    followed by nothing but register copies (the loop's carried values' moves on the back path) and an unconditional
    back-edge -- ends the block at the step's end, not inside it: not counted.
    -- see docs/internals/tools/sass_gate.md#step-branches"""
    resolve = _resolver(lines, labels)
    found = set()
    for target, k in mma_loops(lines, labels):
        latch = lines[k][3] == ""
        for j in range(target, k):
            _, op, ops, pred = lines[j]
            if pred == _NEVER:
                continue
            forward = resolve(ops) if op == "BRA" and pred else None
            copies = all(_COPY.match(lines[i][1]) and lines[i][3] == "" for i in range(j + 1, k))
            if forward is not None and latch and forward > k and copies:
                continue
            if op == "BSSY" or (forward is not None and forward > j):
                found.add(j)
    return len(found)


#: THE STEP LINE (KERNEL_STANDARDS section 32): the share of a step loop's time its tensor pipe can be kept busy, at
#: least 1 / section 28's line
STEP_LINE = 100 / 1.15
#: the kernels whose step loops are under the line, each with what replaces them; each function held at its floor
#: (`STEP_FLOORS`, by architecture and mangled name: a ratchet), and the list only shrinks
_STEP_COVER_DEBT = {
    "carry_kernel": "the value-carve readout is shared-bound by design (~10 shared cycles an MMA, lab read_value): "
                    "its tile loop's cover is held under the line by that pipe; the next tile's rows and entries "
                    "loaded under its MMAs (read_value: -8-12%), the readout composed with the fold on each "
                    "scheduler's two warps, and sm_90's wgmma (a warpgroup's operand read once) replace it; and the "
                    "fold chain's",
    "produce_kernel": "its operands staged by cp.async a step ahead (with section 29's global-memory debt)",
    "intra_kernel": "its writer rows' bytes: the dense Gram ~24 cycles an MMA against its 16.9 with the refill under "
                    "the slabs (0e0163e0), bytes an MMA the lever (the per-call reader block, the u8 dense levels' "
                    "format, proposed); read under the line once the cover read its warps together",
}
STEP_FLOORS = Path(__file__).absolute().parent / "budgets" / "step_floors.json"
_PARTITION_REGISTERS = 16384


def step_cover(cubin: Path, name: str, lines: list[tuple[int, str, str, str]], labels: dict[str, int],
               registers: int) -> dict | None:
    """KERNEL_STANDARDS section 32 on the SASS: each innermost MMA loop's period for one warp alone, simulated from
    ptxas's control words with the tensor pipe as a resource (`pipe_sim`, the model `burst_gate` reads), its pipe
    time's share of it (the occupancy), and the COVER: the pipe's share with the warps a scheduler the registers admit
    running the loop together -- one issue port a scheduler, one memory pipe the SM's every warp shares -- at most
    the share alone times those warps (a warp sharing its scheduler never runs faster than alone) and 100. The worst
    loop's reading, or None without an MMA loop."""
    loops = mma_loops(lines, labels)
    if not loops:
        return None
    import burst_gate
    import pipe_sim

    arch = sass_control.machine(cubin)
    mach = sass_control.MACHINES[arch]
    encoded = sass_control.encoded(cubin, f"^{re.escape(name)}$")
    grain = -(-registers // 8) * 8
    warps = max(1, _PARTITION_REGISTERS // (grain * 32)) if registers else 1
    worst = None
    each = []
    for target, k in loops:
        body = [(a, *encoded[a]) for a, _, _, _ in lines[target:k + 1] if a in encoded]
        c = burst_gate.coverage(body, mach, arch)
        together = pipe_sim.simulate(body, warps, 6, False, sm_warps=pipe_sim.MACHINE["schedulers"] * warps)
        c["together"] = round(100 * together["occupancy"], 1)
        c["cover"] = round(min(100.0, c["occupancy"] * warps, c["together"]), 1)
        c["warps"] = warps
        #: the loop's memory-pipe cycles an iteration (`sass_control.mem_cost`): with the pipe cycles, the caps --
        #: issue `pipe / instr`, memory `pipe / (schedulers memory)` -- name what binds
        memory = sum(sass_control.mem_cost(sass_control.opcode(text), text, mach) for _, text, _ in body)
        each.append({"first": lines[target][0], "last": lines[k][0], "instr": k - target + 1,
                     "hmma": sum(1 for x in lines[target:k + 1] if _MMA.match(x[1])), "memory": round(memory, 1),
                     **{key: c[key] for key in ("cover", "together", "period", "pipe", "exposed", "worst")}})
        if worst is None or c["cover"] < worst["cover"]:
            worst = c
    worst["loops"] = len(loops)
    worst["each"] = each
    worst["key"] = f"{arch} {name}"
    return worst


def gate(cubin: Path, kernel: str | None, max_per_hmma: float | None) -> tuple[bool, str]:
    global ARCH
    _LOOPS.clear()
    here = ARCH = sass_control.machine(cubin)
    funcs: dict[str, list[str]] = {}
    whole: dict[str, list[tuple[int, str, str]]] = {}
    labels: dict[str, dict[str, int]] = {}
    cur = None
    for line in sass.disassemble(cubin).splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            whole[cur] = []
            labels[cur] = {}
            continue
        m = re.match(r"^(\$?[\w.$]+):\s*$", line)
        if m and cur is not None:
            labels[cur][m.group(1)] = len(whole[cur])
            continue
        m = INSTR.match(line)
        if m and cur is not None:
            predicate = (m.group(2) or "").strip()
            if predicate != _NEVER:
                funcs[cur].append(predicate + " " + m.group(3))
            w = WHOLE.match(line)
            whole[cur].append((int(w.group(1), 16), w.group(3), w.group(4).strip(), predicate) if w else
                              (int(m.group(1), 16), m.group(3), "", predicate))

    ok = True
    report = []
    STATS.clear()
    frame = resources(cubin)
    floors = json.loads(STEP_FLOORS.read_text()) if STEP_FLOORS.exists() else {}
    for name, ins in funcs.items():
        if kernel and not re.search(kernel, name):
            continue
        n = len(ins)
        registers, stack = frame.get(name, (0, 0))
        ldl = sum(1 for i in ins if re.search(r"\b(LDL|STL)\b", i))
        calls, guarded = convergence_calls(whole[name], labels[name])
        branches = mma_loop_branches(whole[name], labels[name])
        gap, around = mma_gap(whole[name], labels[name])
        xu, memory = step_classes(whole[name], labels[name])
        phmma = sum(1 for i in ins if re.match(r"@!?P\d+ HMMA", i))
        hmma = sum(1 for i in ins if " HMMA" in i or i.startswith("HMMA"))
        per = (n / hmma) if hmma else float("nan")
        reds = []
        if ldl:
            reds.append(f"local memory ({ldl} LDL/STL)")
        if stack:
            reds.append(f"stack frame ({stack} B)")
        if calls:
            reds.append(f"convergence subroutines a converged warp executes ({calls})")
        if phmma:
            reds.append(f"predicated HMMA ({phmma})")
        #: an instrument's probe keeps a step signature only where its declarations name it (the module docstring)
        kept = {(key, lint) for key, lints in INSTRUMENT.items() if _declares(key, name, here) for lint in lints}
        exhibited = {lint for lint, count in (("step_branches", branches), ("step_xu", xu), ("step_global", memory))
                     if count}
        held = {lint for _, lint in kept}
        if branches and "step_branches" not in held:
            reds.append(f"branches inside an MMA step ({branches})")
        if xu and "step_xu" not in held:
            reds.append(f"the conversion unit inside an MMA step ({xu})")
        if memory and "step_global" not in held and not any(debt in name for debt in _STEP_GLOBAL_DEBT):
            reds.append(f"global memory inside an MMA step ({memory})")
        if max_per_hmma is not None and hmma and per > max_per_hmma:
            reds.append(f"static instructions per HMMA {per:.1f} > {max_per_hmma}")
        cover = step_cover(cubin, name, whole[name], labels[name], registers)
        debt = next((d for d in _STEP_COVER_DEBT if d in name), None)
        if cover is not None and cover["cover"] < STEP_LINE:
            exhibited.add("step_cover")
        if cover is not None and cover["cover"] < STEP_LINE and "step_cover" not in held:
            if debt is None:
                reds.append(f"a step loop's pipe cover {cover['cover']}% < {STEP_LINE:.1f}%")
            elif cover["key"] not in floors:
                reds.append(f"a step loop's pipe cover {cover['cover']}% with no floor (`--write-floors`)")
            elif cover["cover"] < floors[cover["key"]]:
                reds.append(f"a step loop's pipe cover {cover['cover']}% below its floor {floors[cover['key']]}%")
        ok &= not reds
        STATS[name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                       "stack": stack, "calls": calls, "guarded": guarded, "pred_hmma": phmma,
                       "mma_loop_branches": branches, "mma_gap": gap, "mma_gap_around": around,
                       "step_xu": xu, "step_global": memory, "registers": registers,
                       "step_cover": None if cover is None else {k: cover[k] for k in (
                           "cover", "occupancy", "warps", "period", "pipe", "exposed", "worst", "loops", "key",
                           "each")},
                       "reds": reds,
                       "kept": sorted([key, lint] for key, lint in kept if lint in exhibited)}
        short = name[:70]
        report.append(f"{'RED ' if reds else 'ok  '} {short}: regs {registers} instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {stack} calls {calls} guarded {guarded} predHMMA {phmma} "
                      f"mma-loop-branches {branches} mma-gap {gap} around {around} step-xu {xu} step-global {memory}"
                      + ("" if cover is None else f" step-cover {cover['cover']}% (alone {cover['occupancy']}% x "
                                                  f"{cover['warps']} warps, exposed {cover['exposed']})")
                      + (" :: " + "; ".join(reds) if reds else ""))
        if LOOPS:
            report.extend(loop_report(whole[name], labels[name], cover))
    return ok, "\n".join(report)


#: `--loops`: each function's report is followed by each innermost MMA loop's reading and the lines its lints count
LOOPS = False
#: `--instrument`: an instrument's declarations, `{name pattern: {step lint: the cost the probe measures}}` (the module
#: docstring); empty for the extension, whose functions keep no step signature but a named debt
INSTRUMENT: dict[str, dict[str, str]] = {}
#: the architecture of the cubin `gate` read last, which a cached verdict carries with it
ARCH = ""


def _declared(key: str) -> tuple[str | None, str]:
    """A declaration's architecture and pattern: `sm_80 <pattern>` keeps its signature on that architecture's SASS
    alone (ptxas's output is an architecture's), a bare pattern on every one."""
    arch, _, pattern = key.partition(" ")
    return (arch, pattern) if pattern and re.fullmatch(r"sm_\d+a?", arch) else (None, key)


def _declares(key: str, function: str, arch: str | None) -> bool:
    on, pattern = _declared(key)
    return (on is None or on == arch) and re.search(pattern, function) is not None


def loop_report(lines: list[tuple[int, str, str, str]], labels: dict[str, int], cover: dict | None) -> list[str]:
    """Each innermost MMA loop of a function: its span, its instructions and MMAs, its step cover alone and the
    stalls the model exposes, then each line inside it that a step lint counts -- a branch or convergence barrier, the
    conversion unit, global memory other than `cp.async`."""
    resolve = _resolver(lines, labels)
    out = []
    for (target, k), c in zip(mma_loops(lines, labels), (cover or {}).get("each", [])):
        out.append(f"     loop {c['first']:#x}-{c['last']:#x}: {c['instr']} instr, {c['hmma']} MMA, cover {c['cover']}% "
                   f"(period {c['period']}, pipe {c['pipe']}, exposed {c['exposed']}) {c['worst']}")
        for j in range(target, k + 1):
            addr, op, ops, pred = lines[j]
            forward = resolve(ops) if op == "BRA" and pred and pred != _NEVER else None
            kind = sass.opcode_class(op)
            if op == "BSSY" or (forward is not None and forward > j) or kind == "conversion unit" or (
                    kind == "global" and not op.startswith("LDGSTS")):
                out.append(f"       {addr:#06x} {pred + ' ' if pred else ''}{op} {ops}")
    return out


def missing_kernels(seen: set[str]) -> list[str]:
    """The source's kernels (`tools/kernels.py`) whose name no gated function carries."""
    names = set(kernels.demangled_names(sorted(seen)))
    return [f"{k.where()}: `{k.name}` ({k.family}) has no SASS in this binary" for k in kernels.kernels()
            if k.name not in names]


#: THE VERDICT CACHE: a cubin's verdict, report and statistics under the key of everything that decides them -- the
#: cubin's bytes, the gate's own source closure (every module of `tools/` it has loaded), the step floors, the
#: disassemblers' identity and the arguments -- so an unchanged cubin's verdict is the reading of identical bytes under
#: an identical gate, replayed (ratify's per-unit cache makes the same argument). `--no-cache` reads every cubin.
#: -- see docs/internals/tools/sass_gate.md#cache
_CLOSURE: str | None = None


def _closure() -> str:
    """The gate's source and toolchain identity, once a process: every loaded module under `tools/`, the step floors,
    `nvdisasm` and `cuobjdump` by their version lines."""
    global _CLOSURE
    if _CLOSURE is None:
        import burst_gate  # noqa: F401 -- `step_cover` imports it lazily; its source belongs in the key

        tools = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for path in sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                            if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(tools)}):
            h.update(str(path.relative_to(tools)).encode() + b"\0" + path.read_bytes())
        h.update(STEP_FLOORS.read_bytes() if STEP_FLOORS.exists() else b"")
        for tool in ("nvdisasm", "cuobjdump"):
            h.update(subprocess.run([sass.tool(tool), "--version"], capture_output=True, text=True,
                                    check=True).stdout.encode())
        _CLOSURE = h.hexdigest()
    return _CLOSURE


def cached_gate(cubin: Path, kernel: str | None, max_per_hmma: float | None, use: bool) -> tuple[bool, str]:
    """`gate`, replayed from the cache when the key matches; a miss reads the cubin and records the verdict."""
    if not use:
        return gate(cubin, kernel, max_per_hmma)
    key = hashlib.sha256(json.dumps([_closure(), hashlib.sha256(cubin.read_bytes()).hexdigest(), kernel,
                                     max_per_hmma, LOOPS, sorted(INSTRUMENT.items())]).encode()).hexdigest()
    entry = sass.scratch("sass_gate_verdicts") / f"{key}.json"
    global ARCH
    if entry.exists():
        doc = json.loads(entry.read_text())
        STATS.clear()
        STATS.update(doc["stats"])
        ARCH = doc["arch"]
        return doc["ok"], doc["report"]
    ok, report = gate(cubin, kernel, max_per_hmma)
    tmp = entry.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"ok": ok, "report": report, "stats": STATS, "arch": ARCH}, sort_keys=True))
    tmp.replace(entry)
    return ok, report


def rekey_floors(binary: Path) -> int:
    """THE RENAME RULE ON THE FLOORS: each `sm_XX mangled` key moved to the binary's entry point of the same
    signature once its `BuildTag` argument is dropped (`sass_bodies.by_signature`), its value unchanged -- a kernel
    templated on its build keeps its ratchet. A floor with no such entry point is refused, never dropped."""
    import sass_bodies

    floors = json.loads(STEP_FLOORS.read_text())
    by_sig = {}
    for key in sass_bodies.bodies(binary):
        (sig,) = sass_bodies.by_signature({key: ""}, True)
        by_sig[sig] = key
    out = {}
    for key, value in floors.items():
        arch, name = key.split(" ", 1)
        (sig,) = sass_bodies.by_signature({f"{arch}:{name}": ""}, True)
        if sig not in by_sig:
            raise SystemExit(f"{key}: no entry point of its signature in {binary}")
        out[by_sig[sig].replace(":", " ", 1)] = value
    STEP_FLOORS.write_text(json.dumps(dict(sorted(out.items())), indent=1) + "\n")
    print(f"{len(out)} floors rekeyed in {STEP_FLOORS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel", default=None, help="narrow to the functions matching this regex (a measurement)")
    ap.add_argument("--member", default=None, help="narrow to the cubins whose name holds this (a measurement)")
    ap.add_argument("--instrument", type=Path, default=None, metavar="DECLARATIONS",
                    help="an instrument's library: no kernel of the source is owed, and its probes keep the step "
                         "signatures DECLARATIONS names (a JSON file, docs/internals/tools/sass_gate.md#instrument)")
    ap.add_argument("--max-per-hmma", type=float, default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write every function's signature here, by cubin")
    ap.add_argument("--loops", action="store_true",
                    help="after each function, each innermost MMA loop's reading and the lines its step lints count")
    ap.add_argument("--no-cache", action="store_true",
                    help="read every cubin, the verdict cache neither replayed nor written (docs/internals/tools/"
                         "sass_gate.md#cache)")
    ap.add_argument("--write-floors", action="store_true",
                    help="record each step-cover debt's worst reading as its floor (a ratchet: a later build may not "
                         "read lower)")
    ap.add_argument("--cuobjdump", default=None, help="the graph's cuobjdump (with --nvdisasm: the graph's toolkit, "
                    "scratch of this run's own, no verdict cache)")
    ap.add_argument("--nvdisasm", default=None, help="the graph's nvdisasm")
    ap.add_argument("--rekey-floors", action="store_true",
                    help="move each floor to the binary's function of the same signature, a Build argument dropped "
                         "(the rename rule, docs/internals/common/hardware.md#builds-renamed); values unchanged")
    a = ap.parse_args()
    if (a.cuobjdump is None) != (a.nvdisasm is None):
        raise SystemExit("--cuobjdump and --nvdisasm name the graph's toolkit together")
    if a.cuobjdump:
        sass.use_graph(cuobjdump=a.cuobjdump, nvdisasm=a.nvdisasm)
        a.no_cache = True
    if a.rekey_floors:
        return rekey_floors(a.binary)
    global LOOPS, INSTRUMENT
    LOOPS = a.loops
    INSTRUMENT = json.loads(a.instrument.read_text())["kernels"] if a.instrument else {}
    allok = True
    doc = {}
    seen: set[str] = set()
    guarded = 0
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = cached_gate(cub, a.kernel, a.max_per_hmma, not a.no_cache)
        guarded += sum(st["guarded"] for st in STATS.values())
        if rep:
            print(f"== {cub.name}\n{rep}")
        doc[cub.name] = {"ok": ok, "arch": ARCH, "functions": dict(STATS)}
        seen |= set(STATS)
        allok &= ok
    if not seen:
        print(f"RED: no function matched (--kernel {a.kernel!r}, --member {a.member!r}): a gate with no target does "
              "not pass")
        allok = False
    #: CLOSED-WORLD RULE 2 on the binary itself: no PTX image a driver could just-in-time compile for an architecture
    #: nobody measured
    if a.binary.suffix != ".cubin":
        listed = subprocess.run([sass.tool("cuobjdump"), "--list-ptx", str(a.binary)], capture_output=True, text=True,
                                check=True).stdout
        for line in (line for line in listed.splitlines() if line.startswith("PTX file")):
            print(f"RED  rule 2: {line.strip()} -- a binary carries cubins alone (docs/build.md#what-the-build-checks)")
            allok = False
    if a.binary.suffix != ".cubin" and a.kernel is None and a.member is None and not a.instrument:
        for line in missing_kernels(seen):
            print(f"RED  {line}")
            allok = False
    #: a declaration an architecture does not show keeps nothing there: red, so the declarations only say what is so
    #: -- a plain one on every architecture the binary carries, an `sm_XX` one on its own (an iteration build of one
    #: card reads none of another's)
    if a.instrument and a.kernel is None and a.member is None:
        used = {(cubin["arch"], *k) for cubin in doc.values() for st in cubin["functions"].values() for k in st["kept"]}
        carried = sorted({cubin["arch"] for cubin in doc.values()})
        for key, lints in sorted(INSTRUMENT.items()):
            on = _declared(key)[0]
            for arch in [on] if on else carried:
                for lint in sorted(lints):
                    if arch in carried and (arch, key, lint) not in used:
                        hint = "" if on else " (`sm_XX <pattern>` declares a signature one architecture keeps)"
                        print(f"RED  {a.instrument}: `{key}` keeps {lint}, which no function of {arch}'s SASS shows"
                              + hint)
                        allok = False
    if a.json:
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    if a.write_floors:
        floors = json.loads(STEP_FLOORS.read_text()) if STEP_FLOORS.exists() else {}
        for cubin in doc.values():
            for name, st in cubin["functions"].items():
                if any(d in name for d in _STEP_COVER_DEBT) and st["step_cover"] is not None:
                    floors[st["step_cover"]["key"]] = st["step_cover"]["cover"]
        STEP_FLOORS.write_text(json.dumps(dict(sorted(floors.items())), indent=1) + "\n")
        print(f"step floors written to {STEP_FLOORS}")

    print(f"guarded convergence fallbacks (cold code, not red): {guarded}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
