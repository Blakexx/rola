#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTER LIFE-RANGE REPORT (KERNEL_STANDARDS §22): peak LIVE registers per source region
of a kernel, beside ptxas's allocation. A budget argument cites the peak live at the program
point it concerns, never the allocation. The reads and writes are `nvdisasm --print-life-ranges`'s
marks; the liveness is this tool's own, over the function's real control flow
(docs/internals/tools/life_ranges.md#liveness).

    bazel run //tools:life_ranges -- --cubin <cubin> --source csrc/rola/src/carry/ampere/carry_kernel.cuh
    bazel run //tools:life_ranges -- --so <extension .so> --source ... \
        [--member carry_arm] [--kernel carry_kernel] [--top N]
    bazel run //tools:life_ranges -- --arm 0 --source ... [--arch sm_86]   # compiles the carry arm with -lineinfo first
    bazel run //tools:life_ranges -- --arm 0 --kernel carry_kernel \
        --source csrc/rola/src/carry/ampere/carry_kernel.cuh \
        --source csrc/rola/src/carry/ampere/parts.cuh ... --budget tools/budgets/carry.json   # the register budget

The cubin must carry line info (`-lineinfo`); an extension built without it reports the peak
and the histogram but no regions. Regions are the innermost source function (struct methods as
`Struct::method`, as `tools/region_ledger.py` names them).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import sass  # noqa: E402
from region_ledger import fn_of, function_map, owner_of  # noqa: E402


def compile_arm(arm: int, arch: str) -> Path:
    """The carry arm's translation unit compiled alone with line info, against the build's generated headers."""
    import dev_config

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from measure.hw_profile import current

    arch = arch or "sm_" + current().rsplit("-sm", 1)[1]
    cubin = sass.scratch("life_ranges") / f"carry_arm_{arm}.{arch}.cubin"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([dev_config.cuda_bin("nvcc"), "-O3", "-std=c++20", f"-arch={arch}", "--expt-relaxed-constexpr",
                    "-lineinfo", f"-DROLA_CARRY_BUILD_{arm}", "-I", "csrc/rola/src", "-I", "build/generated", "-cubin",
                    f"csrc/rola/src/instantiations/carry_arm_{arm}.cu", "-o", str(cubin)], cwd=root, check=True,
                   capture_output=True, timeout=1800)
    return cubin


def component_peaks(cubin: Path, kernel: str | None, rows: list[tuple[int, str, int]], sources: list[Path],
                    components: dict[str, str]) -> dict[str, int]:
    """THE REGISTER BUDGET's reading (KERNEL_STANDARDS section 32): each budget component's peak live registers, an
    instruction attributed as `tools/region_ledger.py` attributes it (`owner_of`: the innermost inline frame in
    `sources` whose function is a component's, walking outward no further than the kernel's own body, else that body,
    its `glue`)."""
    frames = sass.frames(sass.section(sass.disassemble(cubin, "--print-line-info-inline", "-gi"), kernel, cubin.name))
    fmaps = {p.name: function_map(p.read_text().splitlines()) for p in sources}
    by_function = {fn: name for name, fn in components.items()}
    glue = next((name for name, fn in components.items() if fn.endswith("_kernel")), None)
    peaks: dict[str, int] = collections.defaultdict(int)
    glue_fn = components.get(glue) if glue is not None else None
    for offset, _ins, live in rows:
        c = owner_of(frames.get(offset, []), fmaps, set(by_function), glue_fn)
        owner = by_function[c] if c is not None else glue
        if owner is not None:
            peaks[owner] = max(peaks[owner], live)
    return dict(peaks)


def cubin_of(args) -> Path:
    if args.arm is not None:
        return compile_arm(args.arm, args.arch)
    if args.cubin:
        return Path(args.cubin)
    return sass.cubin(Path(args.so), args.member, args.arch or sass.device_arch())


def line_map(cubin: Path, source_name: str, kernel: str | None) -> dict[int, int | None]:
    """SASS offset -> innermost line in `source_name` of the function `kernel` names (the life-range dump drops the line
    comments, so the map comes from a separate line-info dump)."""
    frames = sass.frames(sass.section(sass.disassemble(cubin, "--print-line-info-inline", "-gi"), kernel, cubin.name))
    return {offset: next((line for name, line in chain if name == source_name), None)
            for offset, chain in frames.items()}


#: a life-range listing's row: the instruction (address, predicate, opcode, operands), then after `// |` its marks --
#: nvdisasm's live count, then a column a register: `^` written, `v` read, `x` both, `:` live through
ROW = re.compile(r"^\s*/\*([0-9a-f]{4,5})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)([^;]*);?.*?// \|(.*)$")
LABEL = re.compile(r"^([.$\w]+):")
TARGET = re.compile(r"`\(([^)]+)\)")
TARGETS = re.compile(r'BRANCH_TARGETS ([^"]+)"')
#: THE REGISTER COUNT nvdisasm prints life ranges for: it refuses a function at 255 (`Invalid register count`), so such
#: a function is read from a copy whose EIATTR_REGCOUNT says 254 -- its columns still reach R253, past the R252 a
#: 255-register function's code reaches (two of the count are reserved)
EIATTR_REGCOUNT, EIFMT_SVAL, MAX_LISTED = 0x2F, 0x04, 254


def listable(cubin: Path) -> Path:
    """`cubin`, or a scratch copy whose 255-register functions' EIATTR_REGCOUNT says `MAX_LISTED` (`.nv.info`'s
    attributes: a format byte, an id byte, a u16 size, then for EIFMT_SVAL the function's symbol and the value)."""
    data = bytearray(cubin.read_bytes())
    shoff, = struct.unpack_from("<Q", data, 0x28)
    entsize, count, names = struct.unpack_from("<HHH", data, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(count)]
    strtab = headers[names][4]
    patched = False
    for h in headers:
        if data[strtab + h[0]: data.index(b"\0", strtab + h[0])] != b".nv.info":
            continue
        p, end = h[4], h[4] + h[5]
        while p < end:
            fmt, attr, size = data[p], data[p + 1], struct.unpack_from("<H", data, p + 2)[0]
            if fmt == EIFMT_SVAL and attr == EIATTR_REGCOUNT and struct.unpack_from("<I", data, p + 8)[0] > MAX_LISTED:
                struct.pack_into("<I", data, p + 8, MAX_LISTED)
                patched = True
            p += 4 + (size if fmt == EIFMT_SVAL else 0)
    if not patched:
        return cubin
    out = sass.scratch("life_ranges") / f"{cubin.stem}.listable.cubin"
    out.write_bytes(bytes(data))
    return out


def live_counts(cubin: Path, kernel: str | None) -> list[tuple[int, str, int]]:
    """(offset, instruction, live registers) per SASS instruction of the function `kernel` names (`live_rows`)."""
    return live_rows(sass.section(sass.disassemble(listable(cubin), "--print-life-ranges"), kernel, cubin.name)
                     .splitlines(), cubin.name)


def live_rows(text: list[str], where: str) -> list[tuple[int, str, int]]:
    """(offset, instruction, live registers) per instruction of one function's life-range listing, its subroutines'
    bodies left out. The reads and writes are nvdisasm's marks (`listing_rows`); the liveness is this tool's, over the
    real control flow (`successors`, docs/internals/tools/life_ranges.md#liveness): a write under a predicate ends no
    range, and a register counts where it is live and some write reaches it."""
    rows, labels = listing_rows(text, where)
    succ = successors(rows, labels)
    n, inside = len(rows), set()
    for name, start in labels.items():
        if name.startswith("$"):
            end = next(k for k in range(start, n) if rows[k]["op"].startswith("RET"))
            inside.update(range(start, end + 1))

    live = fixed_point(n, range(n - 1, -1, -1), lambda k, live: rows[k]["r"] | (
        _union(live[s] for s in succ[k]) & ~(0 if rows[k]["pred"] else rows[k]["w"])))
    preds = [[] for _ in range(n)]
    for k, ss in enumerate(succ):
        for s in ss:
            preds[s].append(k)
    written = fixed_point(n, range(n), lambda k, written: _union(written[q] | rows[q]["w"] for q in preds[k]))
    return [(row["off"], row["text"], bin((live[k] | row["w"]) & (written[k] | row["w"])).count("1"))
            for k, row in enumerate(rows) if k not in inside]


def _union(sets) -> int:
    out = 0
    for x in sets:
        out |= x
    return out


def fixed_point(n: int, order, step) -> list[int]:
    """Each row's set, iterated in `order` until no row's `step(k, sets)` changes."""
    sets = [0] * n
    changed = True
    while changed:
        changed = False
        for k in order:
            new = step(k, sets)
            changed |= new != sets[k]
            sets[k] = new
    return sets


def listing_rows(text: list[str], where: str) -> tuple[list[dict], dict[str, int]]:
    """A life-range listing's rows -- each its offset, predicate, opcode, text, and the registers it writes (`w`) and
    reads (`r`) by its marks under the register header's columns -- and each label's row."""
    head = next(l for l in text if "# 0 1 2" in l)
    head = head[head.index("// |") + 4:]
    cols = [(m.start(), m.group()) for m in re.finditer(r"\S+", head)]
    cols = cols[1:[t for _, t in cols].index("|")]
    if [t for _, t in cols] != [str(r) for r in range(len(cols))]:
        raise SystemExit(f"{where}: an unexpected register header in the life-range listing")
    at = [start + len(t) - 1 for start, t in cols]

    rows, labels, pending = [], {}, []
    for l in text:
        m = LABEL.match(l)
        if m:
            pending.append(m.group(1))
            continue
        m = ROW.match(l)
        if not m:
            continue
        for name in pending:
            labels[name] = len(rows)
        pending = []
        if any(int(r) >= len(cols) for r in re.findall(r"\bR(\d+)\b", m.group(4))):
            raise SystemExit(f"{where}: an instruction names a register past the listing's {len(cols)} columns")
        marks = [m.group(5)[c] if c < len(m.group(5)) else " " for c in at]
        rows.append({"off": int(m.group(1), 16), "pred": bool(m.group(2)), "op": m.group(3),
                     "text": (m.group(3) + m.group(4)).strip(),
                     "w": sum(1 << r for r, mark in enumerate(marks) if mark in "^x"),
                     "r": sum(1 << r for r, mark in enumerate(marks) if mark in "vx")})
    return rows, labels


def successors(rows: list[dict], labels: dict[str, int]) -> list[list[int]]:
    """Each row's successors. A call's marks are nvdisasm's ABI clobbers, never what the call does: a call to a local
    label (a loop's exit through a convergence barrier) is a jump and reads and writes nothing; a call to a subroutine
    (a collective's slow path) takes its body's reads and writes and continues."""
    n = len(rows)
    out = []
    for k, row in enumerate(rows):
        op, nxt = row["op"], [k + 1] if k + 1 < n else []
        target = TARGET.search(row["text"])
        target = target.group(1) if target else None
        if op.startswith("CALL"):
            row["w"] = row["r"] = 0
            if target.startswith(".L"):
                out.append([labels[target]] + (nxt if row["pred"] else []))
                continue
            j = labels[target]
            while not rows[j]["op"].startswith("RET"):
                row["w"] |= rows[j]["w"]
                row["r"] |= rows[j]["r"]
                j += 1
            row["r"] |= rows[j]["r"]
            out.append(nxt)
        elif op.startswith("BRX"):
            out.append([labels[t.strip()] for t in TARGETS.search(row["text"]).group(1).split(",")])
        elif op.startswith("BRA"):
            out.append([labels[target]] + (nxt if row["pred"] or op != "BRA" else []))
        elif op.startswith(("EXIT", "RET", "BPT")):
            out.append(nxt if row["pred"] else [])
        elif op.startswith(("JMP", "JMX", "CALL.ABS")):
            raise SystemExit(f"life_ranges: an unmodelled control instruction {row['text']!r}")
        else:
            out.append(nxt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None, help="compile this carry arm with -lineinfo and read it")
    ap.add_argument("--arch", default=None, help="with --arm: the architecture (default: this machine's GPU)")
    ap.add_argument("--member", default="carry_arm")
    ap.add_argument("--kernel", default=None,
                    help="the function to read, a regex over the cubin's (needed where it holds several)")
    ap.add_argument("--source", type=Path, required=True, action="append",
                    help="the kernel's source (first) and, with --budget, its components' sources")
    ap.add_argument("--budget", type=Path, default=None,
                    help="a budget (tools/budgets/<kernel>.json): red where a component's peak live registers pass its "
                         "`registers` line, or it has none")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--json", type=Path, default=None, help="also write the peak and every region's live counts here")
    a = ap.parse_args()
    if sum(x is not None for x in (a.cubin, a.so, a.arm)) != 1:
        ap.error("give exactly one of --cubin, --so or --arm")
    cubin = cubin_of(a)
    rows = live_counts(cubin, a.kernel)
    if not rows:
        print("no life-range rows: is this a cubin of the kernel?")
        return 1

    source = a.source[0]
    lmap = line_map(cubin, source.name, a.kernel)
    fmap = function_map(source.read_text().splitlines())
    peak = max(r[2] for r in rows)
    hist = collections.Counter((r[2] // 20) * 20 for r in rows)
    print(f"instructions {len(rows)}  peak live registers {peak}")
    print("live histogram (instructions per 20):",
          " ".join(f"{k}-{k + 19}:{v}" for k, v in sorted(hist.items())))
    per = collections.defaultdict(list)
    for off, _ins, live in rows:
        ln = lmap.get(off)
        per[fn_of(fmap, ln) if ln else "?"].append(live)
    if a.json:
        a.json.write_text(json.dumps({"instructions": len(rows), "peak_live": peak, "line_info": any(lmap.values()),
                                      "by_region": {fn: {"peak": max(v), "mean": round(sum(v) / len(v), 1), "instr": len(v)}
                                                    for fn, v in sorted(per.items())}}, sort_keys=True) + "\n")
    if not any(lmap.values()):
        print("no line info in this cubin: build with -lineinfo for regions")
        return 1 if a.budget else 0
    red = False
    if a.budget:
        budget = json.loads(a.budget.read_text())
        lines = budget.get("registers", {})
        peaks = component_peaks(cubin, a.kernel, rows, a.source, budget["components"])
        print(f"\n{'component':14s} {'peak live':>9s} {'line':>5s}")
        for name in budget["components"]:
            line, got = lines.get(name), peaks.get(name, 0)
            over = line is None or got > line
            red |= over
            print(f"{'RED ' if over else 'ok  '}{name:10s} {got:9d} {'-' if line is None else line:>5}")
        if red:
            print(f"RED: a component past its `registers` line in {a.budget}, or with none")
    print(f"\n{'region':26s} {'peak':>5s} {'mean':>6s} {'instr':>6s}")
    for fn, v in sorted(per.items(), key=lambda x: -max(x[1]))[: a.top]:
        print(f"{fn:26s} {max(v):5d} {sum(v) / len(v):6.1f} {len(v):6d}")
    print("\npeak program points:")
    for off, ins, live in sorted(rows, key=lambda r: -r[2])[:6]:
        ln = lmap.get(off)
        print(f"  {off:#07x} live {live}  {fn_of(fmap, ln) if ln else '?'}:{ln}  {ins[:60]}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
