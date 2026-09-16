#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGION LEDGER: a kernel's executed instructions and stall samples per COMPONENT -- the
device function (or struct method) each SASS instruction inlines from -- against the budgets in
`tools/budgets/<kernel>.json`. KERNEL_STANDARDS §19: a component over budget is red.

    tools/region_ledger.py --csv <ncu SourceCounters csv> --so <extension .so> \\
        --source csrc/rola/src/carry/carry_kernel.cuh --budget tools/budgets/carry.json \\
        --per <cta-windows in the launch>

The csv is `ncu --section SourceCounters --page source --print-source sass --csv` of ONE launch
(rola-ledger skill). Attribution: the innermost frame in the kernel's source of each SASS line
(`nvdisasm --print-line-info-inline`), the function by the nearest preceding definition --
`__device__` functions, `__global__` kernels, `struct` blocks and their methods.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sass  # noqa: E402

DEF = [re.compile(r"^__device__ __forceinline__ \S+ (\w+)\("), re.compile(r"^__global__ .* void (\w+)\("),
       re.compile(r"^(?:template <[^>]*>\s*)?struct (\w+)"), re.compile(r"^  __device__ __forceinline__ (?:\w+ )?(\w+)\(")]


def function_map(src_lines: list[str]) -> list[tuple[int, str]]:
    """(line, name) per definition; a struct's methods as `Struct::method`, so a stream's
    steps attribute to the stream."""
    out = []
    struct = None
    for i, l in enumerate(src_lines):
        for k, rx in enumerate(DEF):
            m = rx.match(l)
            if m:
                name = m.group(1)
                if k == 2:
                    struct = name
                elif k == 3 and struct:
                    name = f"{struct}::{name}"
                else:
                    struct = None
                out.append((i + 1, name))
                break
    return out


def component_of(fn: str, components: set[str] | None) -> str | None:
    """The component a function name belongs to: itself, or its struct."""
    if components is None:
        return fn
    if fn in components:
        return fn
    head = fn.split("::")[0]
    return head if head in components else None


def fn_of(fmap: list[tuple[int, str]], ln: int) -> str:
    name = "?"
    for start, n in fmap:
        if ln >= start:
            name = n
    return name


def frames_map(so: Path, member: str, arch: str) -> dict[int, list[tuple[str, int]]]:
    """SASS offset -> the inline frame chain at that instruction, innermost first, as (file name, line) pairs
    (`tools/sass.py`, the one reader of the disassemblers)."""
    return sass.frames(sass.disassemble(sass.cubin(so, member, arch), "--print-line-info-inline", "-gi"))


class Attribution:
    """Which COMPONENT an instruction belongs to -- the outermost frame in the kernel's source
    (the component function the call chain inlines into) -- and whether it is a barrier SPIN
    (an innermost frame inside `mbar_wait`). A spin is the wait for another agent's work and
    is reported apart from the component's own instructions."""

    def __init__(self, so: Path, member: str, arch: str, source: Path, ops: Path):
        self.fmap = function_map(source.read_text().splitlines())
        self.omap = function_map(ops.read_text().splitlines())
        self.source, self.ops = source.name, ops.name
        self.so, self.member = so, member
        self.arch = arch
        self.frames = frames_map(so, member, arch)

    def of(self, offset: int, components: set[str] | None = None) -> tuple[str, bool]:
        """The component: the OUTERMOST frame in the kernel's source whose function is one of
        `components` (the budget's names), else the outermost frame's function -- in the
        composed kernel every chain ends in the kernel itself, which is no component."""
        chain = self.frames.get(offset, [])
        comp = "?"
        for f, ln in chain:
            if f == self.source:
                c = component_of(fn_of(self.fmap, ln), components)
                if c is not None:
                    comp = c
        if comp == "?":
            for f, ln in chain:
                if f == self.source:
                    comp = fn_of(self.fmap, ln)
        spin = any(f == self.ops and fn_of(self.omap, ln) == "mbar_wait" for f, ln in chain)
        return comp, spin


def line_map(so: Path, member: str, arch: str, source_name: str) -> dict[int, int | None]:
    """SASS offset -> innermost line in `source_name` (None when the chain has none)."""
    out = {}
    for off, chain in frames_map(so, member, arch).items():
        ck = [ln for f, ln in chain if f == source_name]
        out[off] = ck[0] if ck else None
    return out


def read_source_counters(csv_path: Path) -> list[tuple[int, int, int, dict]]:
    """(address, instructions executed, stall samples, stall reasons) per SASS line of an
    `ncu --section SourceCounters --page source --print-source sass --csv` dump."""
    rows = []
    hdr = None
    with csv_path.open() as f:
        for r in csv.reader(f):
            if r and r[0] == "Address":
                hdr = {k: i for i, k in enumerate(r)}
                continue
            if hdr and len(r) == len(hdr) and r[0].startswith("0x"):
                st = {k[6:]: int(r[hdr[k]] or 0) for k in hdr if k.startswith("stall_") and "Not Issued" not in k}
                rows.append((int(r[0], 16), int(r[hdr["Instructions Executed"]] or 0),
                             int(r[hdr["Warp Stall Sampling (All Samples)"]] or 0), st))
    return rows


def _num(x: str) -> int:
    return int(x.replace(",", "")) if x not in ("", "-") else 0


def read_source_detail(csv_path: Path) -> dict[int, tuple[str, int, int]]:
    """address -> (opcode, shared wavefronts, ideal shared wavefronts) per SASS line of the
    same dump: the opcode classes an instruction (HMMA or not), the wavefronts above the ideal
    are bank conflicts (KERNEL_STANDARDS §22: a layout's store side is gated by this count)."""
    out = {}
    with csv_path.open() as f:
        hdr = None
        for r in csv.reader(f):
            if r and r[0] == "Address":
                hdr = {k: i for i, k in enumerate(r)}
                continue
            if hdr and len(r) == len(hdr) and r[0].startswith("0x"):
                text = r[hdr["Source"]].strip()
                op = text.split()[0] if text else "?"
                if op.startswith("@"):
                    op = text.split()[1] if len(text.split()) > 1 else op
                out[int(r[0], 16)] = (op, _num(r[hdr["L1 Wavefronts Shared"]]),
                                      _num(r[hdr["L1 Wavefronts Shared Ideal"]]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--so", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--member", default="carry_arm")
    ap.add_argument("--arch", default=None, help="the cubin's architecture (default: this machine's GPU)")
    ap.add_argument("--ops", type=Path, default=Path("csrc/rola/src/common/ops.cuh"))
    ap.add_argument("--budget", type=Path, default=None)
    ap.add_argument("--cell", default=None, help="the budget cell to compare against")
    ap.add_argument("--per", type=float, default=1.0, help="divisor: CTA-windows in the launch")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the components, the phase census and the wavefront census here")
    a = ap.parse_args()
    attr = Attribution(a.so, a.member, a.arch or sass.device_arch(), a.source, a.ops)
    rows = read_source_counters(a.csv)
    base = rows[0][0]

    budget_doc = json.loads(a.budget.read_text()) if a.budget else {}
    components = set(budget_doc.get("components", {}).values()) if budget_doc else None
    agg: dict[str, list] = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    tot = samp = 0
    for addr, n, sm, st in rows:
        fn, spin = attr.of(addr - base, components)
        if spin:
            fn = fn + " [spin]"
        e = agg[fn]
        e[0] += n
        e[1] += sm
        e[2].update(st)
        tot += n
        samp += sm
    cell_budget = budget_doc.get("cells", {}).get(a.cell, {}) if budget_doc else {}
    by_fn = {fn: cell_budget.get(comp) for comp, fn in budget_doc.get("components", {}).items()} if budget_doc else {}
    red = False

    print(f"instructions/unit {tot / a.per:.0f}  samples {samp}")
    for fn, (n, sm, st) in sorted(agg.items(), key=lambda x: -x[1][0])[: a.top]:
        top = " ".join(f"{k}:{100 * v / max(1, sm):.0f}%" for k, v in st.most_common(3))
        b = by_fn.get(fn)
        flag = ""
        if b is not None and n / a.per > b:
            flag = f"  RED > budget {b}"
            red = True
        print(f"  {fn:22s} {n / a.per:9.0f} ({100 * n / tot:4.1f}%)  samples {100 * sm / max(1, samp):5.1f}%  {top}{flag}")
    doc = census(rows, attr, components, a.per, read_source_detail(a.csv), a.source, a.top)
    if a.json:
        doc["per"] = a.per
        doc["instructions_per_unit"] = round(tot / a.per, 1)
        doc["samples"] = samp
        doc["components"] = {fn: {"instr_per_unit": round(n / a.per, 1), "samples": sm, "stalls": dict(st),
                                  "budget": by_fn.get(fn)} for fn, (n, sm, st) in agg.items()}
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    return 1 if red else 0


def census(rows, attr: Attribution, components: set[str] | None, per: float,
           detail: dict[int, tuple[str, int, int]], source: Path, top: int) -> dict:
    """THE PHASE CENSUS (§22): per component, the HMMAs a unit COUNTED from the dump, the
    other instructions, and the stall samples split between the HMMA instructions and the
    rest -- the build ledger turns these into tensor utilization a phase and a warp's time
    at HMMA against its other work. Then THE WAVEFRONT CENSUS: shared-memory wavefronts above
    the ideal (bank conflicts) a unit, by source line."""
    base = rows[0][0]
    cen: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0, 0])
    waves: dict[int | None, list] = collections.defaultdict(lambda: [0, 0, 0])
    lm = line_map(attr.so, attr.member, attr.arch, source.name)
    for addr, n, sm, _st in rows:
        fn, spin = attr.of(addr - base, components)
        if spin:
            continue
        op, wf, ideal = detail.get(addr, ("?", 0, 0))
        e = cen[fn]
        if op.startswith("HMMA"):
            e[0] += n
            e[2] += sm
        else:
            e[1] += n
            e[3] += sm
        if wf > ideal:
            w = waves[lm.get(addr - base)]
            w[0] += wf
            w[1] += ideal
            w[2] += n

    print("census (a unit): component hmma other samples_at_hmma samples_at_other")
    for fn, (h, o, sh, so_) in sorted(cen.items(), key=lambda x: -(x[1][0] + x[1][1])):
        print(f"  census {fn} {h / per:.1f} {o / per:.1f} {sh} {so_}")
    tot = sum(w[0] - w[1] for w in waves.values())
    print(f"wavefronts above ideal (a unit): {tot / per:.0f}")
    src = source.read_text().splitlines()
    lines = []
    for ln, (wf, ideal, n) in sorted(waves.items(), key=lambda x: -(x[1][0] - x[1][1])):
        text = src[ln - 1].strip()[:60] if ln else "?"
        lines.append({"line": ln, "excess_per_unit": round((wf - ideal) / per, 1), "per_instr": round(wf / max(1, n), 2),
                      "text": text})
    for w in lines[:top]:
        print(f"  wave {w['line']} {w['excess_per_unit']:.0f} {w['per_instr']:.1f}/instr {w['text']}")
    return {"census": {fn: {"hmma_per_unit": round(h / per, 1), "other_per_unit": round(o / per, 1), "samples_at_hmma": sh,
                            "samples_at_other": so_} for fn, (h, o, sh, so_) in cen.items()},
            "wavefronts": {"excess_per_unit": round(tot / per, 1), "lines": lines}}


if __name__ == "__main__":
    sys.exit(main())
