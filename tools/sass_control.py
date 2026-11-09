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

import functools
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "lint"))

import burst_tier  # noqa: E402
import sass  # noqa: E402

#: THE MACHINE, an architecture: the cycles an HMMA holds the tensor pipe a scheduler (sm_86: bf16 with
#: fp32 accumulate at half rate, calibration `hmma_1w` 32.4; sm_80 full rate, 8 cycles an m16n8k16 from the A100's
#: 1024 FMA an SM a clock, unmeasured here), an fp16-accumulate HMMA's (`hmma_f16_pipe`: lab `tensor_tax` 16.1), an
#: m16n8k8 half its m16n8k16 (lab `mma_k8`), an IMMA's by shape (`imma_pipe`) and the issue-to-result
#: latency of the scoreboard classes (calibration.md's rows where one exists -- the dependent-chain rows put a
#: shared load, a matrix load and a shuffle at 25, 25 and 24 on sm_86, 2026-09-19; an HMMA on its accumulator 24
#: and an FFMA 4, lab `issue_latency` 2026-10-16 (an LDSM.x4 there 29: owed to the memory side); `REDUX` 40, the `redux_chain` row
#: less its link's two fixed-latency instructions; `R2UR` 39 is measured in place,
#: the census's scoreboard wait at its consumer on two cells, and owes a probe row -- else the microbenchmark
#: literature, bounded above), and the HMMAs the pipe queues past the one it executes (`hmma_queue`: the
#: calibration row with four loads after each burst hid 35 of their 42 cycles, one slot's worth, so two in
#: flight). A cubin of an architecture without a row is refused.
MACHINES = {
    "sm_86": {"hmma_pipe": 32.5, "hmma_f16_pipe": 16.1, "imma_pipe": {"16832": 16.0, "16816": 8.0},
              #: a scheduler's issue pipes and the cycles a warp instruction holds one (16-lane integer ALU, the
              #: FMA pair taking an FP32 instruction a cycle, IMAD on the heavy half), shared by its warps; and the
              #: tensor pipe's extension a cycle an FP32 or IADD3 instruction costs it (probe `imma_pattern`
              #: probe4, 2026-10-01: the IMMA side's extra cycles with that class run by the partner warp)
              "pipes": {"alu": {"IADD3": 2, "LOP3": 2, "PRMT": 2, "SHF": 2, "ISETP": 2, "SEL": 2, "IMNMX": 2,
                                "I2IP": 2, "HMNMX2": 2, "LEA": 2, "FLO": 2, "POPC": 2, "PLOP3": 2, "I2FP": 4,
                                "FSEL": 2, "FMNMX": 2, "MOV": 2},
                        "fma": {"FFMA": 1, "FMUL": 1, "FADD": 1, "HFMA2": 1, "HADD2": 1, "HMUL2": 1, "IMAD": 2}},
              "tensor_interference": {"FFMA": 1.15, "FMUL": 0.85, "FADD": 1.0, "IADD3": 1.45, "IMAD": 0.23}, "latency": {"LDSM": 25, "LDS": 25, "LDG": 400, "SHFL": 24, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 24, "IMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 4, "R2UR": 39, "REDUX": 40, "DEFAULT": 6}, "read": 6, "hmma_queue": 0,
              "schedulers": 4, "mem_wavefront": 1.0, "mem_sector": 2.0, "mem_queue": 4.0, "issue": {"RED": 12, "ATOM": 12, "ATOMG": 12,
                                                                                    "STG": 4, "LDG": 4}},
    "sm_80": {"hmma_pipe": 8.0, "hmma_f16_pipe": 8.0, "imma_pipe": {"16832": 8.0, "16816": 4.0}, "latency": {"LDSM": 25, "LDS": 25, "LDG": 400, "SHFL": 24, "STS": 20, "S2R": 20,
                                              "VOTE": 20, "HMMA": 24, "IMMA": 35, "MUFU": 20, "I2F": 12, "F2I": 12, "F2F": 12,
                                              "HFMA2": 6, "FFMA": 4, "R2UR": 39, "REDUX": 40, "DEFAULT": 6}, "read": 6, "hmma_queue": 0,
              "schedulers": 4, "mem_wavefront": 1.0, "mem_sector": 2.0, "mem_queue": 4.0, "issue": {"RED": 12, "ATOM": 12, "ATOMG": 12,
                                                                                    "STG": 4, "LDG": 4}},
}


#: a warp's shared write by its width in words, cycles of the memory pipe (lab `smem_rates`, sm_86: `st.shared.u32`
#: and a lane-a-bank `red.shared.add` 2.003 a wavefront, `st.shared.v2` 1.064 of two, `st.shared.v4` 1.143 of four)
SHARED_WRITE = {1: 2.0, 2: 2.13, 4: 4.57}


def mem_cost(op: str, text: str, mach: dict) -> float:
    """The cycles an instruction holds the SM's memory pipe: a 128-byte shared wavefront a cycle (`mem_wavefront`:
    the SM's shared-memory bandwidth, calibration.md's `matrix_load` rows), a shared write or atomic its measured
    `SHARED_WRITE`, a global sector `mem_sector`. The counts are the instruction's own: an `ldmatrix` of `k`
    matrices `k` wavefronts, a 128-bit shared load four, a coalesced global access its width's sectors -- a bank
    conflict or a divergent access is not read off the text."""
    if op == "LDSM":
        m = re.search(r"\.(\d)\s", text + " ")
        return (int(m.group(1)) if m else 4) * mach["mem_wavefront"]
    width = 4 if ".128" in text else 2 if ".64" in text else 1
    if op in ("STS", "ATOMS"):
        return SHARED_WRITE[width] * mach["mem_wavefront"]
    if op in ("LDS", "LDGSTS"):
        return width * mach["mem_wavefront"]
    if op == "SHFL":
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


def encoded(cubin: Path, kernel: str | None = None) -> dict[int, tuple[str, dict]]:
    """address -> (instruction text, control bits), over the one function `kernel` names (`sass.one_kernel`: offsets
    restart in every function). The cubin's `cuobjdump -sass` is read and parsed once a process for each version of the
    file (`_encoded_all`), where the gate read it once a function. -- see docs/internals/tools/sass_gate.md#cost"""
    st = Path(cubin).stat()
    every = _encoded_all(str(cubin), st.st_size, st.st_mtime_ns)
    want = sass.one_kernel(list(every), kernel, Path(cubin).name)
    return every[want]


@functools.lru_cache(maxsize=16)
def _encoded_all(path: str, _size: int, _mtime_ns: int) -> dict[str, dict[int, tuple[str, dict]]]:
    text = subprocess.run([sass.tool("cuobjdump"), "-sass", path], capture_output=True, text=True,
                          check=True).stdout
    every: dict[str, dict[int, list]] = {}
    out: dict[int, list] | None = None
    addr = None
    for line in text.splitlines():
        f = sass.FUNCTION.match(line)
        if f:
            out, addr = every.setdefault(f.group(1), {}), None
            continue
        if out is None:
            continue
        m = ENC.match(line)
        if m:
            addr = int(m.group(1), 16)
            out[addr] = [m.group(2).strip(), None]
            continue
        m = ENC_HIGH.match(line)
        if m and addr is not None and out[addr][1] is None:
            out[addr][1] = control(int(m.group(1), 16))
    return {name: {a: (t, c) for a, (t, c) in fn.items() if c is not None} for name, fn in every.items()}


#: THE MMA OPCODES the tensor pipe takes, and the cycles one holds it a scheduler: an HMMA `hmma_pipe`, an IMMA
#: by its shape (`imma_pipe`: sm_86 m16n8k32 16, probe `imma_pattern` 2026-10-01; sm_80 the int8 rate, twice the
#: fp16 one, from the literature until a row is measured there).
MMA = ("HMMA", "IMMA")


def mma_cost(text: str, mach: dict) -> float:
    op = opcode(text)
    if op == "HMMA":
        pipe = mach["hmma_f16_pipe"] if re.search(r"HMMA\.\d+\.F16\b", text) else mach["hmma_pipe"]
        return pipe / 2 if "HMMA.1688" in text else pipe
    if op == "IMMA":
        m = re.search(r"IMMA\.(\d{5})", text)
        return mach["imma_pipe"][m.group(1) if m else "16832"]
    return 0.0


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


def attributed(cubin: Path, source: Path, function: str, kernel: str | None = None) -> list[tuple[int, str, dict]]:
    """The instructions of the kernel `kernel` names any inline frame of which lies in `function`'s lines, in address
    order."""
    lo, hi = span_of(source, function)
    frames = sass.frames(sass.section(sass.disassemble(cubin, "--print-line-info-inline", "-gi"), kernel, cubin.name))
    enc = encoded(cubin, kernel)
    return [(a, *enc[a]) for a in sorted(enc)
            if any(f == source.name and lo <= ln <= hi for f, ln in frames.get(a, []))]


def loop_of(cubin: Path, source: Path, function: str, kernel: str | None = None) -> list[tuple[int, str, dict]]:
    """One iteration of the function's burst loop: among the backward branches in its attributed instructions,
    the loop holding the most HMMAs (a tail or a guard loop after the burst is not the burst), from the branch's
    target to the branch; the whole attribution when there is no loop."""
    body = attributed(cubin, source, function, kernel)
    if not body:
        raise SystemExit(f"no SASS attributed to {function}")
    #: the loops are found over the WHOLE kernel (a primitive's loop branch carries the primitive's frames,
    #: not its user's) and the one chosen is the DENSEST in the function's HMMAs -- the burst itself, never a
    #: phase loop around it
    enc = encoded(cubin, kernel)
    addrs = sorted(enc)
    mine = {a for a, t, _ in body if opcode(t) in MMA}
    best = None
    for a in addrs:
        t = enc[a][0]
        if opcode(t) == "BRA" and branch_target(t) is not None and branch_target(t) < a:
            target = branch_target(t)
            loop = [(x, *enc[x]) for x in addrs if target <= x <= a]
            held = sum(1 for x, _, _ in loop if x in mine)
            if held and (best is None or (held / len(loop), held) > best[0]):
                best = ((held / len(loop), held), loop)
    return best[1] if best else body
