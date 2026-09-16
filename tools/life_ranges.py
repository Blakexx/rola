#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTER LIFE-RANGE REPORT (KERNEL_STANDARDS §22): peak LIVE registers per source region
of a kernel, beside ptxas's allocation, off `nvdisasm --print-life-ranges`. A budget argument
cites the peak live at the program point it concerns, never the allocation: the P81 fold's
two-deep form was abandoned at "255 registers" when its HMMAs ran at 127 live.

    python tools/life_ranges.py --cubin <cubin> --source csrc/rola/src/carry/carry_kernel.cuh
    python tools/life_ranges.py --so <extension .so> --source ... [--member carry_arm] [--top N]
    python tools/life_ranges.py --arm 0 --source ... [--arch sm_86]   # compiles the carry arm with -lineinfo first

The cubin must carry line info (`-lineinfo`); an extension built without it reports the peak
and the histogram but no regions. Regions are the innermost source function (struct methods as
`Struct::method`, as `tools/region_ledger.py` names them).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sass  # noqa: E402
from region_ledger import fn_of, function_map  # noqa: E402

INSTR_RE = re.compile(r"^\s*/\*([0-9a-f]{4,5})\*/\s+(\S.*?)\s*;?\s*// \|\s*(\d+)")


def compile_arm(arm: int, arch: str) -> Path:
    """The carry arm's translation unit compiled alone with line info, against the build's generated headers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measure"))
    import dev_config

    from measure.hw_profile import current

    arch = arch or "sm_" + current().rsplit("-sm", 1)[1]
    cubin = dev_config.scratch("life_ranges") / f"carry_arm_{arm}.{arch}.cubin"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([dev_config.cuda_bin("nvcc"), "-O3", "-std=c++17", f"-arch={arch}", "--expt-relaxed-constexpr",
                    "-lineinfo", f"-DROLA_CARRY_BUILD_{arm}", "-I", "csrc/rola/src", "-I", "build/generated", "-cubin",
                    f"csrc/rola/src/instantiations/carry_arm_{arm}.cu", "-o", str(cubin)], cwd=root, check=True,
                   capture_output=True, timeout=1800)
    return cubin


def cubin_of(args) -> Path:
    if args.arm is not None:
        return compile_arm(args.arm, args.arch)
    if args.cubin:
        return Path(args.cubin)
    return sass.cubin(Path(args.so), args.member, args.arch or sass.device_arch())


def line_map(cubin: Path, source_name: str) -> dict[int, int | None]:
    """SASS offset -> innermost line in `source_name` (the life-range dump drops the line
    comments, so the map comes from a separate line-info dump)."""
    frames = sass.frames(sass.disassemble(cubin, "--print-line-info-inline", "-gi"))
    return {offset: next((line for name, line in chain if name == source_name), None)
            for offset, chain in frames.items()}


def live_counts(cubin: Path) -> list[tuple[int, str, int]]:
    """(offset, instruction, live registers) per SASS instruction."""
    rows = []
    for l in sass.disassemble(cubin, "--print-life-ranges").splitlines():
        m = INSTR_RE.match(l)
        if m:
            rows.append((int(m.group(1), 16), m.group(2).strip(), int(m.group(3))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--arm", type=int, default=None, help="compile this carry arm with -lineinfo and read it")
    ap.add_argument("--arch", default=None, help="with --arm: the architecture (default: this machine's GPU)")
    ap.add_argument("--member", default="carry_arm")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--json", type=Path, default=None, help="also write the peak and every region's live counts here")
    a = ap.parse_args()
    if sum(x is not None for x in (a.cubin, a.so, a.arm)) != 1:
        ap.error("give exactly one of --cubin, --so or --arm")
    cubin = cubin_of(a)
    rows = live_counts(cubin)
    if not rows:
        print("no life-range rows: is this a cubin of the kernel?")
        return 1

    lmap = line_map(cubin, a.source.name)
    fmap = function_map(a.source.read_text().splitlines())
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
        return 0
    print(f"\n{'region':26s} {'peak':>5s} {'mean':>6s} {'instr':>6s}")
    for fn, v in sorted(per.items(), key=lambda x: -max(x[1]))[: a.top]:
        print(f"{fn:26s} {max(v):5d} {sum(v) / len(v):6.1f} {len(v):6d}")
    print("\npeak program points:")
    for off, ins, live in sorted(rows, key=lambda r: -r[2])[:6]:
        ln = lmap.get(off)
        print(f"  {off:#07x} live {live}  {fn_of(fmap, ln) if ln else '?'}:{ln}  {ins[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
