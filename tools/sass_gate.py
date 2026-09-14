#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built
extension, red on any of them. Runs on every iteration build (`tools/sass_gate.py <cubin|.so>
[--kernel regex] [--max-per-hmma N]`); exit 1 on a red.

What it reads (per kernel function):
  local memory  -- LDL/STL instructions, or a nonzero stack frame in the ELF's `.nv.info`
  outlined collectives -- `CALL.REL.NOINC $__internal_*` (a convergence subroutine)
  predicated HMMA -- `@P HMMA`: if-converted tensor work
  instructions per HMMA -- the static ratio, a ceiling the arm states (issue-bound above it)
It prints the counts it reads, the same way `ratify` prints registers and spills, so a clean run
is a line of zeros and a red names the signature.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import

NVDISASM = dev_config.cuda_bin("nvdisasm")
CUOBJDUMP = dev_config.cuda_bin("cuobjdump")
INSTR = re.compile(r"^\s*/\*[0-9a-f]{4,5}\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.$]+)")


def cubins_of(path: Path, member: str | None) -> list[Path]:
    if path.suffix == ".cubin":
        return [path]
    out = Path(tempfile.mkdtemp(prefix="sass_gate_"))
    names = subprocess.run([CUOBJDUMP, "-lelf", str(path)], capture_output=True, text=True, check=True).stdout
    picked = []
    for line in names.splitlines():
        m = re.search(r"ELF file\s+\d+:\s+(\S+)", line)
        if m and (member is None or member in m.group(1)):
            picked.append(m.group(1))
    subprocess.run([CUOBJDUMP, "-xelf", "all", str(path.resolve())], cwd=out, capture_output=True, check=True)
    return [out / n for n in picked]


#: the last `gate` call's signature per function, for `--json`
STATS: dict[str, dict] = {}


def gate(cubin: Path, kernel: str | None, max_per_hmma: float | None) -> tuple[bool, str]:
    sass = subprocess.run([NVDISASM, str(cubin)], capture_output=True, text=True, check=True).stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = INSTR.match(line)
        if m and cur is not None:
            funcs[cur].append((m.group(1) or "").strip() + " " + m.group(2))

    ok = True
    report = []
    STATS.clear()
    for name, ins in funcs.items():
        if kernel and not re.search(kernel, name):
            continue
        n = len(ins)
        ldl = sum(1 for i in ins if re.search(r"\b(LDL|STL)\b", i))
        calls = sum(1 for i in ins if "CALL.REL.NOINC $__internal" in i or "$__cuda_sm" in i)
        phmma = sum(1 for i in ins if re.match(r"@!?P\d+ HMMA", i))
        hmma = sum(1 for i in ins if " HMMA" in i or i.startswith("HMMA"))
        per = (n / hmma) if hmma else float("nan")
        reds = []
        if ldl:
            reds.append(f"local memory ({ldl} LDL/STL)")
        if calls:
            reds.append(f"convergence subroutines ({calls} CALL $__internal)")
        if phmma:
            reds.append(f"predicated HMMA ({phmma})")
        if max_per_hmma is not None and hmma and per > max_per_hmma:
            reds.append(f"static instructions per HMMA {per:.1f} > {max_per_hmma}")
        ok &= not reds
        STATS[name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                       "calls": calls, "pred_hmma": phmma, "reds": reds}
        short = name[:70]
        report.append(f"{'RED ' if reds else 'ok  '} {short}: instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} calls {calls} predHMMA {phmma}" + (" :: " + "; ".join(reds) if reds else ""))
    return ok, "\n".join(report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel", default="carry_kernel")
    ap.add_argument("--member", default="carry_arm")
    ap.add_argument("--max-per-hmma", type=float, default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write every function's signature here, by cubin")
    a = ap.parse_args()
    allok = True
    doc = {}
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = gate(cub, a.kernel, a.max_per_hmma)
        print(f"== {cub.name}\n{rep}")
        doc[cub.name] = {"ok": ok, "functions": dict(STATS)}
        allok &= ok
    if a.json:
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
