#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGION LEDGER: a kernel's executed instructions and stall samples per COMPONENT -- the
device function (or struct method) each SASS instruction inlines from -- against the budgets in
`tools/budgets/<kernel>.json`. KERNEL_STANDARDS §19: a component over budget is red.

    tools/region_ledger.py --csv <ncu SourceCounters csv> --so <extension .so> \\
        --member <the cubin member> --source <the kernel's file> <its components' files>... \\
        --budget tools/budgets/<kernel>.json --per <cta-windows in the launch>

The csv is `ncu --section SourceCounters --page source --print-source sass --csv` of ONE launch
(rola-ledger skill). Attribution (`Attribution`): the innermost frame in the kernel's sources (the kernel's file
first, then the files of its components) whose function is a budget component, walked outward no further than the
kernel's own body (`nvdisasm --print-line-info-inline`); a frame's function is the nearest preceding definition --
every `__device__` / `__host__` / `constexpr` function whatever its return type, `__global__` kernels, `struct`
blocks and their methods (`DEF`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import sass  # noqa: E402

#: A DEFINITION's first line, by kind: a function at column 0 (any qualifiers, any return type: a `__host__ __device__
#: constexpr` helper missed here books its lines to the definition before it), a kernel, a struct at any depth, and a
#: struct's method two spaces deeper than its struct (`__device__` required, `static` and the other qualifiers before
#: or after it, so a body's local `constexpr` is never one) -- a kernel wrapped in a class keeps its components'
#: attribution (docs/internals/tools/budgets.md#building-a-budget)
_QUAL = r"(?:__host__|__device__|__forceinline__|__noinline__|constexpr|static|inline)"
_RET = r"(?:[\w:<>,*&]+\s+)*?[*&]*"
_NAME = r"(\w+|operator\S*?)"
DEF = [re.compile(rf"^(?:{_QUAL}\s+)+{_RET}{_NAME}\("), re.compile(r"^__global__ .* void (\w+)\(")]
STRUCT = re.compile(r"^( *)(?:template <[^>]*>\s*)?struct (?:alignas\(\d+\) )?(\w+)\b(?!;)")
METHOD = re.compile(rf"^( +)(?:(?:__host__|static|inline|constexpr)\s+)*__device__\s+(?:{_QUAL}\s+)*{_RET}{_NAME}\(")


def function_map(src_lines: list[str]) -> list[tuple[int, str]]:
    """(line, name) per definition; a struct's methods as `Struct::method` (`Outer::Inner::method` nested), so a
    stream's steps attribute to the stream."""
    out = []
    scopes: list[tuple[int, str]] = []
    for i, l in enumerate(src_lines):
        m = STRUCT.match(l)
        if m and not l.rstrip().endswith(";"):
            depth = len(m.group(1))
            scopes = [s for s in scopes if s[0] < depth]
            scopes.append((depth, m.group(2)))
            out.append((i + 1, "::".join(name for _, name in scopes)))
            continue
        m = METHOD.match(l)
        if m and scopes:
            depth = len(m.group(1))
            outer = [s for s in scopes if s[0] < depth]
            if outer and outer[-1][0] == depth - 2:
                scopes = outer
                out.append((i + 1, "::".join([name for _, name in scopes] + [m.group(2)])))
                continue
        for rx in DEF:
            m = rx.match(l)
            if m:
                scopes = []
                out.append((i + 1, m.group(1)))
                break
    return out


def component_of(fn: str, components: set[str] | None) -> str | None:
    """The component a function name belongs to: itself, its innermost struct that is one, or -- a method of a class
    wrapping a kernel (`IntraAt::gram_dense`) -- its own name."""
    if components is None:
        return fn
    if fn in components:
        return fn
    parts = fn.split("::")
    for struct in reversed(parts[:-1]):
        if struct in components:
            return struct
    return parts[-1] if len(parts) > 1 and parts[-1] in components else None


def owner_of(chain: list[tuple[str, int]], fmaps: dict, components: set[str], glue: str | None) -> str | None:
    """The component owning an instruction of inline chain `chain` (innermost frame first): walking outward to the
    first frame in the kernel's own body (`glue`'s function), the first frame in `fmaps`' files whose function is one
    of `components`, else None (the glue's). The walk stops at the kernel's body because the line table's chain is a
    call chain only up to there: beyond it the table can splice another call site's frames (`Attribution`). No
    component calls another, and every component is called from the kernel's body."""
    for f, ln in chain:
        if f in fmaps:
            fn = fn_of(fmaps[f], ln)
            if fn == glue:
                return None
            c = component_of(fn, components)
            if c is not None:
                return c
    return None


def helper_of(chain: list[tuple[str, int]], fmaps: dict, glue: str | None) -> str | None:
    """The outermost function below the kernel's body in `chain` (the work no component owns, named where it is
    written), else None."""
    out = None
    for f, ln in chain:
        if f in fmaps:
            fn = fn_of(fmaps[f], ln)
            if fn == glue:
                break
            out = fn
    return out


def fn_of(fmap: list[tuple[int, str]], ln: int) -> str:
    name = "?"
    for start, n in fmap:
        if ln >= start:
            name = n
    return name


def frames_map(so: Path, member: str, arch: str, kernel: str | None) -> dict[int, list[tuple[str, int]]]:
    """SASS offset -> the inline frame chain at that instruction, innermost first, as (file name, line) pairs, over the
    one function `kernel` names (`tools/sass.py`, the one reader of the disassemblers: offsets restart in every
    function, so two kernels' maps never merge)."""
    listing = sass.disassemble(sass.cubin(so, member, arch), "--print-line-info-inline", "-gi")
    return sass.frames(sass.section(listing, kernel, member))


class Attribution:
    """Which COMPONENT an instruction belongs to -- `owner_of`: the innermost frame in the kernel's sources whose
    function is a component, walking outward no further than the kernel's own body; work no component owns under its
    own function's name (`helper_of`); the kernel's body itself its glue -- and whether it is a barrier SPIN (an
    innermost frame inside `mbar_wait`), the wait for another agent's work, reported apart from the component's own
    instructions. The walk stops at the kernel's body because the line table's chain is a call chain only up to there:
    beyond it the table splices other call sites' frames in (a readout instruction read `read_chunk <- the kernel <-
    store_batches' static_for <- the kernel`), and an outermost rule booked each to the wrong component."""

    def __init__(self, so: Path, member: str, arch: str, sources: list[Path], barrier: Path,
                 kernel: str | None = None):
        self.fmaps = {p.name: function_map(p.read_text().splitlines()) for p in sources}
        self.texts = {p.name: p.read_text().splitlines() for p in sources}
        #: the barrier atom whose `mbar_wait` frames mark a spin; a file without it would mark none, silently
        self.bmap = function_map(barrier.read_text().splitlines())
        if not any(name == "mbar_wait" for _, name in self.bmap):
            raise SystemExit(f"{barrier} defines no `mbar_wait`: no barrier spin could be found (--barrier)")
        self.barrier = barrier.name
        self.so, self.member, self.kernel = so, member, kernel
        self.arch = arch
        self.frames = frames_map(so, member, arch, kernel)

    def of(self, offset: int, components: set[str] | None = None, glue: str | None = None) -> tuple[str, bool]:
        """The component: the INNERMOST frame in the kernel's sources whose function is one of `components` (the
        budget's names) other than `glue` (the kernel's own body, every chain's last frame), else the outermost frame's
        function -- the kernel itself, its glue (with no budget, always the outermost frame's function)."""
        chain = self.frames.get(offset, [])
        comp = "?"
        if components is not None:
            comp = owner_of(chain, self.fmaps, components, glue) or helper_of(chain, self.fmaps, glue) or "?"
        if comp == "?":
            for f, ln in chain:
                if f in self.fmaps:
                    comp = fn_of(self.fmaps[f], ln)
        spin = any(f == self.barrier and fn_of(self.bmap, ln) == "mbar_wait" for f, ln in chain)
        return comp, spin


def line_map(so: Path, member: str, arch: str, source_name: str, kernel: str | None) -> dict[int, int | None]:
    """SASS offset -> innermost line in `source_name` (None when the chain has none)."""
    out = {}
    for off, chain in frames_map(so, member, arch, kernel).items():
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


#: a shared-memory instruction's kind for the census: a load, a write (a store or an atomic: the pipe twice a 32-bit
#: wavefront), an asynchronous copy (its column the global lines it touched, `read_source_detail`), the rest
def _wave_kind(op: str) -> str:
    if op.startswith(("LDS", "LDSM")):
        return "loads"
    if op.startswith(("STS", "ATOMS", "REDS")):
        return "writes"
    if op.startswith("LDGSTS"):
        return "copies"
    return "other"


def _num(x: str) -> int:
    return int(x.replace(",", "")) if x not in ("", "-") else 0


def read_source_detail(csv_path: Path) -> dict[int, tuple[str, int, int]]:
    """address -> (opcode, shared wavefronts, ideal shared wavefronts) per SASS line of the
    same dump: the opcode classes an instruction (HMMA or not), the wavefronts above the ideal
    are bank conflicts on a shared store or load (KERNEL_STANDARDS §22: a layout's store side is
    gated by this count) and the global lines touched on an asynchronous copy (`LDGSTS`)."""
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
    ap.add_argument("--source", type=Path, nargs="+", required=True,
                    help="the kernel's file (its line views), then the files its components live in")
    ap.add_argument("--member", required=True, help="the cubin member's name")
    ap.add_argument("--kernel", default=None,
                    help="the profiled kernel, a regex over the member's functions (needed where it holds several)")
    ap.add_argument("--arch", default=None, help="the cubin's architecture (default: this machine's GPU)")
    ap.add_argument("--barrier", type=Path, default=Path("csrc/rola/src/common/atoms/barrier.cuh"),
                    help="the atom whose `mbar_wait` frames mark a barrier spin")
    ap.add_argument("--budget", type=Path, default=None)
    ap.add_argument("--cell", default=None, help="the budget cell to compare against")
    ap.add_argument("--per", type=float, default=1.0, help="divisor: CTA-windows in the launch")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the components, the phase census and the wavefront census here")
    ap.add_argument("--lines", default=None, metavar="LO-HI",
                    help="also print the source lines in this range: instructions a unit, samples, opcodes, stall reasons")
    ap.add_argument("--component", default=None,
                    help="also print one component's instructions by their innermost frame in ANY file (its helpers', "
                         "the headers'): instructions a unit, samples, opcodes, stall reasons")
    a = ap.parse_args()
    attr = Attribution(a.so, a.member, a.arch or sass.device_arch(), a.source, a.barrier, a.kernel)
    kernel_source = a.source[0]
    rows = read_source_counters(a.csv)
    base = rows[0][0]

    budget_doc = json.loads(a.budget.read_text()) if a.budget else {}
    components = set(budget_doc.get("components", {}).values()) if budget_doc else None
    glue = budget_doc.get("components", {}).get("glue") if budget_doc else None
    agg: dict[str, list] = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    tot = samp = 0
    for addr, n, sm, st in rows:
        fn, spin = attr.of(addr - base, components, glue)
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
    detail = read_source_detail(a.csv)
    if a.lines:
        lo, hi = (int(x) for x in a.lines.split("-"))
        print_lines(rows, attr, detail, kernel_source, lo, hi, a.per)
    if a.component:
        print_component(rows, attr, detail, components, glue, a.component, [*a.source, a.barrier], a.per)
    doc = census(rows, attr, components, a.per, detail, kernel_source, a.top, glue)
    if a.json:
        doc["per"] = a.per
        doc["instructions_per_unit"] = round(tot / a.per, 1)
        doc["samples"] = samp
        doc["components"] = {fn: {"instr_per_unit": round(n / a.per, 1), "samples": sm, "stalls": dict(st),
                                  "budget": by_fn.get(fn)} for fn, (n, sm, st) in agg.items()}
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    return 1 if red else 0


def print_lines(rows, attr: Attribution, detail: dict, source: Path, lo: int, hi: int, per: float) -> None:
    """THE LINE VIEW: the source lines `lo..hi` of the kernel's file, each with its instructions a unit, its stall
    samples and their top reasons, and its opcodes -- the isolating measurement below a component's row."""
    base = rows[0][0]
    src = source.read_text().splitlines()
    lines: dict[int, list] = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
    total = sum(sm for _, _, sm, _ in rows)
    here = 0
    for addr, n, sm, st in rows:
        chain = attr.frames.get(addr - base, [])
        inner = [ln for f, ln in chain if f == source.name]
        if not inner or not (lo <= inner[0] <= hi):
            continue
        e = lines[inner[0]]
        e[0] += n
        e[1] += sm
        e[2].update(st)
        e[3][detail.get(addr, ("?", 0, 0))[0]] += n
        here += sm
    print(f"lines {lo}-{hi}: {here} samples ({100 * here / max(1, total):.1f}% of the launch); instructions a unit, "
          f"samples, opcodes, stall reasons")
    for ln in sorted(lines):
        n, sm, st, ops = lines[ln]
        opstr = " ".join(f"{k}:{v / per:.0f}" for k, v in ops.most_common(4))
        stall = " ".join(f"{k}:{100 * v / max(1, sm):.0f}%" for k, v in st.most_common(3))
        print(f"  {ln:5d} {n / per:8.0f} {sm:8d}  {opstr:50s} {stall:36s} | {src[ln - 1].strip()[:56]}")


def print_component(rows, attr: Attribution, detail: dict, components: set[str] | None, glue: str | None, name: str,
                    sources: list[Path], per: float) -> None:
    """THE COMPONENT VIEW: one component's instructions grouped by their INNERMOST frame in any file -- so the work its
    helpers and the headers inline into it shows where it is written, which the line view of one file cannot."""
    base = rows[0][0]
    text = {p.name: p.read_text().splitlines() for p in sources}
    frames: dict[tuple[str, int], list] = collections.defaultdict(
        lambda: [0, 0, collections.Counter(), collections.Counter()])
    here = 0
    for addr, n, sm, st in rows:
        if attr.of(addr - base, components, glue)[0] != name:
            continue
        chain = attr.frames.get(addr - base, [])
        e = frames[chain[0] if chain else ("?", 0)]
        e[0] += n
        e[1] += sm
        e[2].update(st)
        e[3][detail.get(addr, ("?", 0, 0))[0]] += n
        here += n
    print(f"component {name}: {here / per:.0f} instructions a unit, by innermost frame")
    for (f, ln), (n, sm, st, ops) in sorted(frames.items(), key=lambda x: -x[1][0]):
        if n == 0:
            continue
        opstr = " ".join(f"{k}:{v / per:.0f}" for k, v in ops.most_common(4))
        stall = " ".join(f"{k}:{100 * v / max(1, sm):.0f}%" for k, v in st.most_common(3))
        src = text[f][ln - 1].strip()[:48] if f in text and 0 < ln <= len(text[f]) else ""
        print(f"  {f}:{ln:<5d} {n / per:8.0f} {sm:7d}  {opstr:46s} {stall:34s} | {src}")


def census(rows, attr: Attribution, components: set[str] | None, per: float,
           detail: dict[int, tuple[str, int, int]], source: Path, top: int, glue: str | None = None) -> dict:
    """THE PHASE CENSUS (§22): per component, the HMMAs a unit COUNTED from the dump, the
    other instructions, and the stall samples split between the HMMA instructions and the
    rest -- the build ledger turns these into tensor utilization a phase and a warp's time
    at HMMA against its other work; beside them each component's shared-memory wavefronts a unit by kind (`_WAVE_KIND`:
    loads, writes -- stores and atomics, which hold the pipe twice a 32-bit wavefront, lab `smem_rates` -- and the rest)
    and its MMAs as m16n8k16 equivalents (an m16n8k8 a half), so a phase's shared traffic an MMA reads off the census;
    and each component's loads and writes by SITE, the innermost frame in the kernel's sources (its top 16).
    Then THE WAVEFRONT CENSUS: shared-memory wavefronts above the ideal (bank conflicts) a unit, by source line."""
    base = rows[0][0]
    cen: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0, 0])
    kinds: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    #: a component's shared traffic by its innermost frame in the kernel's sources: (file, line) -> [loads, writes, n]
    sites: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    waves: dict[int | None, list] = collections.defaultdict(lambda: [0, 0, 0])
    lm = line_map(attr.so, attr.member, attr.arch, source.name, attr.kernel)
    for addr, n, sm, _st in rows:
        fn, spin = attr.of(addr - base, components, glue)
        if spin:
            continue
        op, wf, ideal = detail.get(addr, ("?", 0, 0))
        e = cen[fn]
        if op.startswith("HMMA"):
            e[0] += n
            e[2] += sm
            kinds[fn]["mma_k16"] += n * (0.5 if ".1688" in op else 1.0)
        else:
            e[1] += n
            e[3] += sm
        kind = _wave_kind(op)
        kinds[fn][kind] += wf
        if kind in ("loads", "writes") and wf:
            chain = attr.frames.get(addr - base, [])
            site = next(((f, ln) for f, ln in chain if f in attr.fmaps), chain[0] if chain else ("?", 0))
            s = sites[fn][site]
            s[0 if kind == "loads" else 1] += wf
            s[2] += n
        if wf > ideal:
            w = waves[lm.get(addr - base)]
            w[0] += wf
            w[1] += ideal
            w[2] += n

    print("census (a unit): component hmma other samples_at_hmma samples_at_other | shared wavefronts a unit "
          "loads writes copies other, per m16n8k16-equivalent")
    for fn, (h, o, sh, so_) in sorted(cen.items(), key=lambda x: -(x[1][0] + x[1][1])):
        k = kinds[fn]
        waves_txt = " ".join(f"{w}:{k[w] / per:.0f}" for w in ("loads", "writes", "copies", "other"))
        mma = k["mma_k16"]
        each = (f" ({(k['loads'] + k['writes'] + k['other']) / mma:.2f} an MMA, writes {k['writes'] / mma:.2f})"
                if mma else "")
        print(f"  census {fn} {h / per:.1f} {o / per:.1f} {sh} {so_} | {waves_txt}{each}")
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
    traffic = {}
    for fn, by in sites.items():
        top = sorted(by.items(), key=lambda x: -(x[1][0] + 2 * x[1][1]))[:16]
        rows_ = []
        for (f, ln), (ld, wr, n) in top:
            text = attr.texts.get(f, [])
            src_line = text[ln - 1].strip()[:70] if 0 < ln <= len(text) else ""
            rows_.append({"file": f, "line": ln, "loads": round(ld / per, 1), "writes": round(wr / per, 1),
                          "instr": round(n / per, 1), "text": src_line})
        traffic[fn] = rows_
    for fn in sorted(traffic, key=lambda f: -kinds[f]["mma_k16"]):
        if not kinds[fn]["mma_k16"]:
            continue
        print(f"shared traffic by site, {fn} (a unit; cycles = loads + 2 writes):")
        for r in traffic[fn][:10]:
            print(f"  {r['file']}:{r['line']:<5d} ld {r['loads']:7.1f} wr {r['writes']:6.1f} n {r['instr']:7.1f} | {r['text']}")
    return {"traffic_sites": traffic,
            "census": {fn: {"hmma_per_unit": round(h / per, 1), "other_per_unit": round(o / per, 1), "samples_at_hmma": sh,
                            "samples_at_other": so_, "mma_k16_per_unit": round(kinds[fn]["mma_k16"] / per, 1),
                            "waves_per_unit": {w: round(kinds[fn][w] / per, 1)
                                               for w in ("loads", "writes", "copies", "other")}}
                       for fn, (h, o, sh, so_) in cen.items()},
            "wavefronts": {"excess_per_unit": round(tot / per, 1), "lines": lines}}


if __name__ == "__main__":
    sys.exit(main())
