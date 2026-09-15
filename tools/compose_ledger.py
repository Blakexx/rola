#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMPOSER (KERNEL_STANDARDS §22 (9)): a phase's time attributed to its parts by composition.

    python tools/compose_ledger.py --cells flagship-dense,nl64k-alt-k4 \\
        --ladder readout:stream,loads,drain --ladder fold:pool,ring,loads

The carry kernel's MMA phases are built from PARTS (`gen_shards.CARRY_PARTS`); a build may stub
any of them (`ROLA_CARRY_PARTS`): a stub keeps the part's HMMAs -- the same count, atom and
accumulators -- and drops its other work. A LADDER over a phase builds the kernel with none of
that phase's parts real (every other part real), then adds the parts back in the order named, one
rung a build; the last rung is the kernel. Each rung is read in wall time by the phase ledger (a
warm-up launch, then `--launches`), so a part's cost is the step between two rungs -- no constant,
no reading of samples. Each rung is CHECKED: the profiler's HMMA count for the cell must equal the
kernel's (a stub that changed the workload fails its rung), and the build's device code must differ
from the kernel's (a build that did not take the mask fails). The kernel is rebuilt with every part
real at the end and its device code must hash to the value taken before the first rung.
Report: stored through `rola_results` at `compose_ledger`, its markdown in the scratch directory. Serial builds (the host's
memory watchdog kills parallel ones); measured work takes the GPU lock through the tools it drives.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dev_config  # noqa: E402
import gen_shards  # noqa: E402
import toolchains  # noqa: E402

PY = sys.executable
CUOBJDUMP = dev_config.cuda_bin("cuobjdump")
NVDISASM = dev_config.cuda_bin("nvdisasm")
#: where the human-readable report goes; the report itself is stored (`rola_results`)
SCRATCH = dev_config.scratch("compose_ledger")


def run(cmd: list[str], env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def extension_so() -> Path:
    return toolchains.built_extension(ROOT)


def device_hash(arch: str = "sm_86") -> str:
    """The carry arm's device INSTRUCTIONS, disassembled, hashed: a fact about the installed
    binary that no path or timestamp can fake."""
    with tempfile.TemporaryDirectory(prefix="compose_") as tmp:
        subprocess.run([CUOBJDUMP, "-xelf", "all", str(extension_so())], cwd=tmp, capture_output=True, check=True)
        cub = next(p for p in Path(tmp).iterdir() if p.name.startswith("carry_arm") and arch in p.name)
        sass = subprocess.run([NVDISASM, "-gi", str(cub)], capture_output=True, text=True, check=True).stdout
    #: instructions only: the debug line table moves with every source line added above a site
    #: and is not code (a byte-identical kernel differed there alone, 2026-09-12).
    text = "\n".join(re.sub(r"//.*$", "", l).rstrip() for l in sass.splitlines()
                     if re.match(r"\s*/\*[0-9a-f]+\*/\s+[A-Z@]", l))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build(real: tuple[str, ...]) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("ROLA_BUILD_PARTS", None)
    env["ROLA_CARRY_PARTS"] = "all" if set(real) == set(gen_shards.CARRY_PARTS) else (",".join(real) or "none")
    return run([PY, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps", "-q"], env=env, timeout=3600)


PHASE_NAMES = ("head", "readout", "fold", "snapshot", "edges", "sweep", "head_words", "head_scans")


def _instrument(args: list[str], timeout: int = 3600) -> dict:
    """Run one of the tree's instruments with `--json` and return what it wrote (a failure is a row, never a stop)."""
    with tempfile.TemporaryDirectory(prefix="compose_") as tmp:
        out = Path(tmp) / "out.json"
        rc, text = run([PY, *args, "--json", str(out)], timeout=timeout)
        return json.loads(out.read_text()) if out.exists() else {"rc": rc, "raw": text[-800:]}


def phases(cell: str, launches: int) -> dict:
    """Cycles a warp a window per phase (warm-up + `launches`), and every phase's per-warp row."""
    doc = _instrument(["tools/phase_ledger.py", cell, "--launches", str(launches), "--warmup", "1"])
    if "per_phase" not in doc:
        return doc
    return {**doc["per_phase"], "total": doc["total"], "per_warp": doc["per_warp"]}


def counters(cell: str) -> dict:
    """The rung's resource counters (`tools/pipe_counters.py`), a CTA-window where a sum."""
    doc = _instrument(["tools/pipe_counters.py", cell])
    if "launch" not in doc:
        return doc
    per = doc["cta_windows"]
    vals = {k: v / per if k.endswith(".sum") else v for k, v in doc["launch"].items()}
    vals["hmma_launch"] = doc["launch"].get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0.0)
    return vals


def census(cell: str, total_cycles: float | None) -> dict:
    """The rung's stall census (`tools/stall_census.py`): every stall sample of one launch by reason, and by component
    and reason, as cycles a warp a window (the sample share times the phase ledger's total)."""
    doc = _instrument(["tools/stall_census.py", cell])
    if "components" not in doc:
        return doc
    scale = (total_cycles or 0.0) / max(1, doc["samples"])
    by_reason: dict[str, float] = {}
    for comp in doc["components"].values():
        for k, v in comp["stalls"].items():
            by_reason[k] = by_reason.get(k, 0.0) + v * scale
    return {"samples": doc["samples"],
            "by_reason": dict(sorted(by_reason.items(), key=lambda x: -x[1])),
            "by_component": {c: {k: v * scale for k, v in sorted(d["stalls"].items(), key=lambda x: -x[1])}
                             for c, d in doc["components"].items()}}


def registers() -> dict:
    """Peak live registers of the rung's arm (`tools/life_ranges.py --arm 0`, against the rung's parts header)."""
    doc = _instrument(["tools/life_ranges.py", "--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], 1800)
    return {"peak_live": doc.get("peak_live"),
            "by_region": {k: v["peak"] for k, v in doc.get("by_region", {}).items()}} if "peak_live" in doc else doc


def timeline(cell: str) -> dict:
    """The rung's pipe timeline summary (`tools/pipe_timeline.py`, stored scale)."""
    import pipe_timeline as T
    with tempfile.TemporaryDirectory(prefix="compose_tl_") as tmp:
        out = Path(tmp) / "tl"
        rep = T.capture(cell, out)
        tl = T.parse(*T.export(rep))
    summ = T.summary(tl, T.stored_scale())
    return {"summary": summ, "tensor_5us": T.bins(tl["series"].get("tensor", []), 5.0, tl["duration_us"])}


TIMELINE = False


def measure(cells: list[str], launches: int) -> dict:
    """Everything a rung records: per cell the phases (with per-warp rows), the resource counters, the stall
    census and (with --timeline) the pipe timeline; once, the register life ranges."""
    out = {"phases": {}, "counters": {}, "census": {}, "timeline": {}}
    for c in cells:
        out["phases"][c] = phases(c, launches)
        out["counters"][c] = counters(c)
        out["census"][c] = census(c, out["phases"][c].get("total"))
        if TIMELINE:
            out["timeline"][c] = timeline(c)
    out["registers"] = registers()
    out["hmma"] = {c: int(out["counters"][c].get("hmma_launch", 0)) for c in cells}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="flagship-dense,nl64k-alt-k4")
    ap.add_argument("--ladder", action="append", required=True,
                    help="PHASE:part,part,... -- the phase's parts in the order they are added back")
    ap.add_argument("--launches", type=int, default=2)
    ap.add_argument("--timeline", action="store_true", help="capture every rung's pipe timeline (PM sampling)")
    a = ap.parse_args()
    global TIMELINE
    TIMELINE = a.timeline
    cells = [c for c in a.cells.split(",") if c]
    everything = gen_shards.CARRY_PARTS

    ladders = []
    for spec in a.ladder:
        phase, names = spec.split(":", 1)
        order = [f"{phase}.{n}" for n in names.split(",") if n]
        bad = sorted(set(order) - set(everything))
        if bad or len(set(order)) != len([x for x in everything if x.startswith(phase + ".")]):
            ap.error(f"--ladder {spec}: name every part of {phase} once ({[x for x in everything if x.startswith(phase + '.')]})")
        others = tuple(x for x in everything if not x.startswith(phase + "."))
        rungs = [others + tuple(order[:i]) for i in range(len(order) + 1)]
        ladders.append((phase, order, rungs))

    report = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ"),
              "sha": run(["git", "rev-parse", "--short=7", "HEAD"])[1].strip(), "cells": cells,
              "launches": a.launches, "ladders": [], "kernel": {}}

    rc, out = build(everything)
    if rc:
        print(out[-2000:])
        return 1
    ref = device_hash()
    report["kernel"]["hash"] = ref
    report["kernel"].update(measure(cells, a.launches))
    print(f"kernel {ref}: " + "; ".join(f"{c} {report['kernel']['phases'][c].get('total')}" for c in cells), flush=True)

    for phase, order, rungs in ladders:
        rows = []
        for real in rungs[:-1]:
            rc, out = build(real)
            row = {"real": sorted(real), "stubbed": sorted(set(everything) - set(real)), "build_rc": rc}
            if rc:
                row["raw"] = out[-1500:]
                rows.append(row)
                continue
            row["hash"] = device_hash()
            row["mask_took"] = row["hash"] != ref
            row.update(measure(cells, a.launches))
            row["hmma_ok"] = all(row["hmma"][c] == report["kernel"]["hmma"][c] for c in cells)
            rows.append(row)
            print(f"{phase} rung {len(rows) - 1} real={[x for x in order if x in real]} hash={row['hash']} "
                  f"took={row['mask_took']} hmma_ok={row['hmma_ok']}: "
                  + "; ".join(f"{c} {phase} {row['phases'][c].get(phase)}" for c in cells), flush=True)
        rows.append({"real": sorted(everything), "stubbed": [], "hash": ref, "mask_took": True, "hmma_ok": True,
                     **{k: report["kernel"][k] for k in ("phases", "counters", "census", "registers", "hmma", "timeline")}})
        report["ladders"].append({"phase": phase, "order": order, "rungs": rows})

    rc, out = build(everything)
    report["kernel"]["hash_after"] = device_hash() if rc == 0 else None
    report["kernel"]["restored"] = report["kernel"]["hash_after"] == ref
    #: the kernel read again at the end: the run's drift bound (a rung step smaller than the
    #: kernel's own before/after difference is not a reading).
    report["kernel"]["phases_after"] = {c: phases(c, a.launches) for c in cells} if rc == 0 else {}

    drift = []
    for c in cells:
        b = report["kernel"]["phases"].get(c, {})
        e = report["kernel"]["phases_after"].get(c, {})
        drift.append(f"{c} total {b.get('total')} before, {e.get('total')} after"
                     + "".join(f"; {ph} {b.get(ph)} / {e.get(ph)}" for lad in ladders for ph in [lad[0]]))
    md = [f"# Composer {report['sha']} ({report['utc']})", "",
          f"Kernel device hash {ref}; restored after the ladders: {report['kernel']['restored']}.",
          "Kernel read before and after the ladders (the run's drift bound): " + " | ".join(drift), ""]
    for lad in report["ladders"]:
        phase, order = lad["phase"], lad["order"]
        md.append(f"## {phase}: parts added in order {order}")
        md.append("| rung | real parts of the phase | " + " | ".join(f"{c} {phase} | step | total" for c in cells)
                  + " | HMMA count = kernel | build took the mask | tensor pipe on silicon (cells in order) |")
        md.append("|---|---|" + "---|---|---|" * len(cells) + "---|---|---|")
        prev = None
        for i, r in enumerate(lad["rungs"]):
            here = [x.split(".", 1)[1] for x in order if x in r["real"]]
            cells_md = []
            for c in cells:
                v = (r.get("phases") or {}).get(c, {}).get(phase)
                t = (r.get("phases") or {}).get(c, {}).get("total")
                pv = None if prev is None else (prev.get("phases") or {}).get(c, {}).get(phase)
                step = "" if v is None or pv is None else f"{v - pv:+.0f}"
                cells_md.append(f"{'' if v is None else f'{v:.0f}'} | {step} | {'' if t is None else f'{t:.0f}'}")
            util = " / ".join(
                f"{100 * u:.0f}%" if (u := (((r.get("timeline") or {}).get(c) or {}).get("summary") or {})
                                      .get("utilization_mean_full")) is not None else "-" for c in cells)
            md.append(f"| {i} | {', '.join(here) or '(all stubbed)'} | " + " | ".join(cells_md)
                      + f" | {r.get('hmma_ok')} | {r.get('mask_took')} | {util} |")
            prev = r
        md.append("")
        #: THE PART STEPS: each added part's step in the kernel total, in its stall reasons, in the
        #: resource counters (a CTA-window) and in peak live registers.
        keys = [("shared ld", "smsp__sass_inst_executed_op_shared_ld.sum"),
                ("shared st", "smsp__sass_inst_executed_op_shared_st.sum"),
                ("ldsm", "smsp__sass_inst_executed_op_ldsm.sum"),
                ("global red", "smsp__sass_inst_executed_op_global_red.sum"),
                ("wf ld", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
                ("wf st", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
                ("conflict ld", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
                ("conflict st", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
                ("issue %", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
                ("warps active", "smsp__warps_active.avg.per_cycle_active")]
        for c in cells:
            md.append(f"**{c}, {phase} part steps** (total cycles; stall-reason cycles; counters a CTA-window; peak live):")
            md.append("| part added | total | top stall-reason steps | " + " | ".join(k for k, _ in keys) + " | peak live |")
            md.append("|---|---|---|" + "---|" * len(keys) + "---|")
            rungs = lad["rungs"]
            for i in range(1, len(rungs)):
                a_, b_ = rungs[i - 1], rungs[i]
                added = [x.split(".", 1)[1] for x in order if x in b_["real"] and x not in a_["real"]]
                ta = (a_.get("phases") or {}).get(c, {}).get("total")
                tb = (b_.get("phases") or {}).get(c, {}).get("total")
                ra = ((a_.get("census") or {}).get(c) or {}).get("by_reason", {})
                rb = ((b_.get("census") or {}).get(c) or {}).get("by_reason", {})
                steps = sorted(((k, rb.get(k, 0) - ra.get(k, 0)) for k in set(ra) | set(rb)), key=lambda x: -abs(x[1]))[:4]
                ca = (a_.get("counters") or {}).get(c, {})
                cb = (b_.get("counters") or {}).get(c, {})
                pa = (a_.get("registers") or {}).get("peak_live")
                pb = (b_.get("registers") or {}).get("peak_live")
                md.append(f"| {', '.join(added)} | {'' if ta is None or tb is None else f'{tb - ta:+.0f}'} | "
                          + " ".join(f"{k} {v:+.0f}" for k, v in steps) + " | "
                          + " | ".join(f"{cb.get(m, 0) - ca.get(m, 0):+.1f}" for _, m in keys)
                          + f" | {pa} -> {pb} |")
            md.append("")
    from rola_results import Store, checkout

    name = f"{report['utc']}-{report['sha']}-compose"
    here = checkout(ROOT)
    semantics = {"git_sha": here["git_sha"], "diff_sha256": here["diff_sha256"], "cells": report["cells"],
                 "timeline": TIMELINE}
    sample = Store("compose_ledger").put(semantics, output=report, provenance=here)
    (SCRATCH / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"report: compose_ledger sample {sample['n']}; markdown {SCRATCH / name}.md")
    ok = report["kernel"]["restored"] and all(r.get("hmma_ok") and r.get("mask_took") for lad in report["ladders"] for r in lad["rungs"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
