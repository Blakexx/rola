#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BUILD LEDGER (KERNEL_STANDARDS §22): every measurement a build is judged by, in one run,
as one report diffed against the previous build's. Nothing here is new instrumentation: it drives
the tools the gate already has and reads their output back, so a number is never typed twice.

    python tools/build_ledger.py --tag fold-two-deep \\
        --cells nl64k-alt-k4,flagship-dense,flagship-alt-k4,flagship-cohort-k4 \\
        --baseline worktree:PATH,venv:PATH     # the baseline tree and its venv

The steps, each recorded with its outcome (a failure is a row, never a stop):
  sass        -- `tools/sass_gate.py` on the installed extension (§20 signatures)
  registers   -- `tools/life_ranges.py` on a -lineinfo compile of the flagship arm: peak live per
                 region beside the allocation (a budget cites live, never allocation)
  oracle      -- the scoped fp64 oracle on the built arm's cells
  phases      -- `tools/phase_ledger.py` per cell: cycles a warp a window per phase, per-warp rows
  profile     -- `ncu` SourceCounters + pipe counters per cell, `tools/region_ledger.py` per
                 component with the launch's CTA-window divisor; tensor utilization = HMMA count
                 x `--hmma-cycles` over scheduler cycles (never the profiler's pipe-active ratio,
                 §22); THE PHASE CENSUS (HMMAs a phase counted from the dump, utilization a phase
                 against the phase ledger's cycles, a warp's cycles at HMMA vs its other work,
                 instructions vs budget) and THE WAVEFRONT CENSUS (shared wavefronts above ideal
                 = bank conflicts, by source line)
  timeline    -- `tools/pipe_timeline.py` per cell: PM sampling of one launch, the tensor pipe's true
                 utilization over time on the stored scale (§22 (11)), with a chart
  ab          -- `tools/probe_cells.py` per cell against the baseline RUN WITH ITS OWN SCHEDULE
                 (`box` dense, `sparse-g32` sparse), this tree with `first`
The report is stored through `rola_results` at `build_ledger` and its markdown written to the scratch
directory; the diff is against the newest earlier stored report sharing a cell. Measured work: the GPU lock, quiet GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import toolchains  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: where the human-readable report and the timelines' charts go; the report itself is stored (`rola_results`)
SCRATCH = dev_config.scratch("build_ledger")
PY = sys.executable
PHASES = ("head", "readout", "fold", "snapshot", "edges", "sweep", "head_words", "head_scans")
HMMA_CYCLES, SCHEDULERS = 32, 4  #: the utilization formula's constants (sm_86; --hmma-cycles/--schedulers override the profile step)
PIPE_METRICS = ",".join([
    "gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__cycles_active.avg",
    "smsp__inst_executed.sum", "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_inst_executed_op_ldsm.sum", "smsp__sass_inst_executed_op_global_red.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
])
ORACLE_K = ("(folded_state or carried_state or idle_resident or two_backings or passes_the_state "
            "or flagship-dense-first or flagship-alt-k4-first or flagship-alt-k16-first or "
            "flagship-cohort-k4-first or flagship-both-k4-first or identity or one_token or "
            "two_orders or sparse_form or partial_window or multiwindow or struct) "
            "and not dv32 and not dv128 and not deep")


def run(cmd: list[str], timeout: int = 1800, env: dict | None = None) -> tuple[int, str]:
    import os
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as ex:
        return 124, f"timeout after {timeout}s: {ex.stdout!r}"


def git_sha() -> str:
    rc, out = run(["git", "rev-parse", "--short=7", "HEAD"])
    return out.strip() if rc == 0 else "unknown"


def extension_so() -> Path:
    return toolchains.built_extension(ROOT)


def step_sass(report: dict) -> None:
    doc = instrument(["tools/sass_gate.py", str(extension_so())], 900)
    arm = next((v for k, v in sorted(doc.items()) if k.endswith(".sm_86.cubin")), None) if "rc" not in doc else None
    kernel = next(iter(arm["functions"].values()), None) if arm else None
    report["sass"] = {k: kernel[k] for k in ("instr", "hmma", "per_hmma", "ldl", "calls", "pred_hmma")} if kernel else doc


def instrument(args: list[str], timeout: int = 3600) -> dict:
    """Run one of the tree's instruments with `--json` and return what it wrote, or its rc and output tail."""
    with tempfile.TemporaryDirectory(prefix="build_ledger_") as tmp:
        out = Path(tmp) / "out.json"
        rc, text = run([PY, *args, "--json", str(out)], timeout=timeout)
        return json.loads(out.read_text()) if out.exists() else {"rc": rc, "raw": text[-800:]}


def step_registers(report: dict) -> None:
    doc = instrument(["tools/life_ranges.py", "--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], 1800)
    report["registers"] = {"peak_live": doc["peak_live"], "peak_by_region": {k: v["peak"] for k, v in doc["by_region"].items()}} \
        if "peak_live" in doc else doc


def step_oracle(report: dict) -> None:
    rc, out = run([PY, "-m", "pytest", "tests/oracle/test_carry_vs_oracle.py", "-q", "-k", ORACLE_K],
                  timeout=1800)
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    report["oracle"] = {"rc": rc, "passed": int(m.group(1)) if m else 0,
                        "failed": int(f.group(1)) if f else 0,
                        "failures": re.findall(r"^FAILED (\S+)", out, re.M)[:20]}


def step_phases(report: dict, cells: list[str]) -> None:
    report["phases"] = {}
    for cell in cells:
        doc = instrument(["tools/phase_ledger.py", cell, "--launches", "1"], 900)
        report["phases"][cell] = {"per_phase": doc["per_phase"], "total": doc["total"],
                                  "readout_per_warp": doc["per_warp"]["readout"], "fold_per_warp": doc["per_warp"]["fold"]} \
            if "per_phase" in doc else doc


def step_profile(report: dict, cells: list[str], hmma_cycles: int, sms: int, schedulers: int) -> None:
    """The pipe counters (`tools/pipe_counters.py`) and the stall census (`tools/stall_census.py`) per cell; tensor
    utilization = HMMA count x `--hmma-cycles` over scheduler cycles (never the profiler's pipe-active ratio, §22)."""
    report["profile"] = {}
    for cell in cells:
        counters = instrument(["tools/pipe_counters.py", cell])
        row: dict = {"cta_windows": counters.get("cta_windows"), "counters": counters.get("launch", counters)}
        vals = counters.get("launch", {})
        hmma = vals.get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0.0)
        for name, metric in (("tensor_util_elapsed", "sm__cycles_elapsed.avg"), ("tensor_util_active", "smsp__cycles_active.avg")):
            if vals.get(metric):
                row[name] = hmma * hmma_cycles / (sms * schedulers * vals[metric])
        census = instrument(["tools/stall_census.py", cell])
        if "components" in census:
            samp = max(1, census["samples"])
            row["components"] = {fn: {"instr": round(c["instr_per_unit"]), "samples_pct": 100 * c["samples"] / samp,
                                      "stalls": " ".join(f"{k}:{100 * v / max(1, c['samples']):.0f}%" for k, v in
                                                         sorted(c["stalls"].items(), key=lambda x: -x[1])[:3])}
                                 for fn, c in sorted(census["components"].items(), key=lambda x: -x[1]["instr_per_unit"])}
            row["census"] = {fn: {"hmma": c["hmma_per_unit"], "other": c["other_per_unit"], "s_hmma": c["samples_at_hmma"],
                                  "s_other": c["samples_at_other"]} for fn, c in census["census"].items()}
            row["waves_excess"] = census["wavefronts"]["excess_per_unit"]
            row["waves"] = [{"line": w["line"], "excess": w["excess_per_unit"], "per_instr": w["per_instr"], "text": w["text"]}
                            for w in census["wavefronts"]["lines"][:12]]
        else:
            row["profile_raw"] = census
        report["profile"][cell] = row


def step_timeline(report: dict, cells: list[str], stem: str) -> None:
    """THE PIPE TIMELINE per cell (`tools/pipe_timeline.py`): PM sampling of one launch, the tensor pipe's true
    utilization over time on the stored scale, its mean over full-occupancy samples, and the chart."""
    report["timeline"] = {}
    for cell in cells:
        out = SCRATCH / f"{stem}-timeline-{cell}"
        rc, text = run([PY, "tools/pipe_timeline.py", "--cell", cell, "--out", str(out)], timeout=3600)
        row: dict = {"rc": rc, "html": out.with_suffix(".html").name}
        m = re.search(r"\{.*\}", text, re.S)
        if rc == 0 and m:
            with contextlib.suppress(json.JSONDecodeError):
                row.update(json.loads(m.group(0)))
        else:
            row["raw"] = text[-600:]
        report["timeline"][cell] = row


def step_ab(report: dict, cells: list[str], baseline: str, tag: str) -> None:
    report["ab"] = {}
    for cell in cells:
        sched = "box" if "dense" in cell or "struct" in cell else "sparse-g32"
        rc, out = run([PY, "tools/probe_cells.py", "--bench", "carry_forward", "--cells", cell,
                       "--reps", "3", "--warmup", "1", "--rounds", "3", "--timeout", "900",
                       "--stage", f"ledger-{tag}",
                       "--binary", f"{baseline},label:master,schedule:{sched}",
                       "--binary", f"worktree:{ROOT},venv:{Path(PY).parents[1]},label:this,schedule:first"],
                      timeout=3600)
        row = {"rc": rc, "master_schedule": sched}
        for label in ("master", "this"):
            m = re.search(rf"^{label}\s+\S+\s+([\d.]+)", out, re.M)
            if m:
                row[label] = float(m.group(1))
        if "master" in row and "this" in row and row["master"]:
            row["ratio"] = row["this"] / row["master"]
        if "this" not in row:
            row["raw"] = out[-500:]
        report["ab"][cell] = row


def parse_tree(spec: str) -> Path:
    """The worktree of a `worktree:PATH,venv:PATH` spec."""
    return Path(dict(part.partition(":")[::2] for part in spec.split(","))["worktree"])


def previous_report(cells: list[str], current: tuple[str, int]) -> dict | None:
    """The newest stored build-ledger report sharing a cell with this one, other than the sample just stored."""
    from rola_results import Store

    store = Store("build_ledger")
    samples = sorted((s["utc"], r["key"], s["n"], store.dir / s["output"]) for r in store.records() for s in r["samples"]
                     if s["ok"] and (r["key"], s["n"]) != current)
    for _utc, _key, _n, path in reversed(samples):
        r = json.loads(path.read_text())
        if set(cells) & set(r.get("cells", [])):
            return r
    return None


def markdown(report: dict, prev: dict | None) -> str:
    out = [f"# Build ledger {report['sha']} `{report['tag']}` ({report['utc']})", ""]
    s = report.get("sass", {})
    out.append(f"**SASS** instr {s.get('instr')} hmma {s.get('hmma')} ldl {s.get('ldl')} calls {s.get('calls')} "
               f"predHMMA {s.get('pred_hmma')}")
    r = report.get("registers", {})
    out.append(f"**Registers** peak live {r.get('peak_live')}; by region: "
               + ", ".join(f"{k} {v}" for k, v in list(r.get("peak_by_region", {}).items())[:8]))
    o = report.get("oracle", {})
    out.append(f"**Oracle** {o.get('passed')} passed, {o.get('failed')} failed"
               + (f": {', '.join(o['failures'][:5])}" if o.get("failures") else ""))
    out.append("")

    out.append("| cell | " + " | ".join(PHASES) + " | total | prev total | tensor util (active) | A/B master | this | ratio |")
    out.append("|---|" + "---|" * (len(PHASES) + 6))
    for cell in report["cells"]:
        ph = report.get("phases", {}).get(cell, {}).get("per_phase", {})
        tot = report.get("phases", {}).get(cell, {}).get("total")
        pt = (prev or {}).get("phases", {}).get(cell, {}).get("total") if prev else None
        pr = report.get("profile", {}).get(cell, {})
        ab = report.get("ab", {}).get(cell, {})
        util = pr.get("tensor_util_active")
        out.append(f"| {cell} | " + " | ".join(f"{ph.get(p, 0):.0f}" for p in PHASES)
                   + f" | {tot if tot is None else f'{tot:.0f}'} | {pt if pt is None else f'{pt:.0f}'}"
                   + f" | {'' if util is None else f'{100 * util:.0f}%'}"
                   + f" | {ab.get('master', '')} | {ab.get('this', '')} | {ab.get('ratio', '') if 'ratio' not in ab else f'{ab[chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.3f}'} |")
    out.append("")

    for cell in report["cells"]:
        t = report.get("timeline", {}).get(cell, {})
        if t.get("utilization_mean_full") is not None:
            out.append(f"**{cell}** tensor pipe on silicon (PM sampling, stored scale): true utilization "
                       f"{100 * t['utilization_mean_full']:.1f}% over {t.get('full_occupancy_samples')} full-occupancy "
                       f"1-us samples, idle {100 * (t.get('tensor_idle_share_full') or 0):.1f}% of them; chart {t.get('html')}")
    out.append("")
    budget = json.loads((ROOT / "tools/budgets/carry.json").read_text()) if (ROOT / "tools/budgets/carry.json").exists() else {}
    comp_of = {v: k for k, v in budget.get("components", {}).items()}
    phase_of = {"FoldStream": "fold", "ReadoutStream": "readout", "head": "head", "snapshot_publish": "snapshot"}
    for cell in report["cells"]:
        pr = report.get("profile", {}).get(cell, {})
        cen = pr.get("census")
        ph = report.get("phases", {}).get(cell, {}).get("per_phase", {})
        if not cen or not ph:
            continue
        cb = budget.get("cells", {}).get(cell, {})
        out.append(f"**{cell}** phase census (a CTA-window; util = HMMA x {HMMA_CYCLES} / {SCHEDULERS} / phase cycles; "
                   f"warp cycles split by stall samples at HMMA vs the rest):")
        out.append("| component | cycles | HMMA | util | instr | budget | non-HMMA/HMMA | at HMMA | at other |")
        out.append("|---|---|---|---|---|---|---|---|---|")

        for fn in ("ReadoutStream", "FoldStream", "head", "snapshot_publish"):
            e = cen.get(fn)
            c = ph.get(phase_of[fn])
            if not e or c is None:
                continue
            h, o = e["hmma"], e["other"]
            util = h * HMMA_CYCLES / SCHEDULERS / c if c else 0.0
            sm = e["s_hmma"] + e["s_other"]
            at_h = c * e["s_hmma"] / sm if sm else 0.0
            b = cb.get(comp_of.get(fn, ""), None)
            if fn == "FoldStream" and "fill" in cb:
                b = (b or 0) + cb["fill"]
            out.append(f"| {fn} | {c:.0f} | {h:.0f} | {100 * util:.0f}% | {h + o:.0f} | {'' if b is None else f'{b:.0f}'} "
                       f"| {o / max(1.0, h):.1f} | {at_h:.0f} | {c - at_h:.0f} |")
        wx = pr.get("waves_excess")
        if wx is not None:
            out.append(f"shared wavefronts above ideal a window: **{wx}**; by line: "
                       + "; ".join(f"L{w['line']} {w['excess']} ({w['per_instr']:.0f}/instr)" for w in pr.get("waves", [])[:5]))
        out.append("")

    for cell in report["cells"]:
        comps = report.get("profile", {}).get(cell, {}).get("components")
        if comps:
            out.append(f"**{cell}** components (samples %): "
                       + "; ".join(f"{k} {v['samples_pct']:.1f}% ({v['stalls']})" for k, v in list(comps.items())[:6]))

    if prev:
        out.append("")
        out.append(f"Diff against {prev['sha']} `{prev['tag']}` ({prev['utc']}):")
        for cell in report["cells"]:
            a = report.get("phases", {}).get(cell, {}).get("per_phase", {})
            b = prev.get("phases", {}).get(cell, {}).get("per_phase", {})
            if a and b:
                out.append(f"- {cell}: " + ", ".join(f"{p} {a.get(p, 0) - b.get(p, 0):+.0f}" for p in PHASES))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="a few words naming the build")
    ap.add_argument("--cells", default="nl64k-alt-k4,flagship-dense,flagship-alt-k4,flagship-cohort-k4")
    ap.add_argument("--baseline", default=None, help="worktree:PATH,venv:PATH of the baseline tree (for --ab)")
    ap.add_argument("--skip", default="",
                    help="comma list of steps to skip: sass,registers,oracle,phases,profile,timeline,ab")
    ap.add_argument("--hmma-cycles", type=int, default=32,
                    help="measured cycles one bf16/fp32 m16n8k16 HMMA holds the pipe a scheduler (sm_86: 32)")
    ap.add_argument("--sms", type=int, default=80)
    ap.add_argument("--schedulers", type=int, default=4)
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]
    skip = set(a.skip.split(",")) if a.skip else set()

    report = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ"), "sha": git_sha(),
              "tag": a.tag, "cells": cells}
    if "sass" not in skip:
        step_sass(report)
    if "registers" not in skip:
        step_registers(report)
    if "oracle" not in skip:
        step_oracle(report)
    if "phases" not in skip:
        step_phases(report, cells)
    if "profile" not in skip:
        step_profile(report, cells, a.hmma_cycles, a.sms, a.schedulers)
    if "timeline" not in skip:
        step_timeline(report, cells, f"{report['utc']}-{report['sha']}-{re.sub(r'[^A-Za-z0-9_-]+', '-', a.tag)}")
    if "ab" not in skip:
        if not a.baseline:
            ap.error("--ab needs --baseline worktree:PATH,venv:PATH")
        step_ab(report, cells, a.baseline, a.tag)

    from rola_results import Store, checkout
    from rola_results import key as key_of

    name = f"{report['utc']}-{report['sha']}-{re.sub(r'[^A-Za-z0-9_-]+', '-', a.tag)}"
    here = checkout(ROOT)
    semantics = {"git_sha": here["git_sha"], "diff_sha256": here["diff_sha256"], "tag": a.tag, "cells": cells,
                 "skip": sorted(skip), "baseline": checkout(parse_tree(a.baseline))["git_sha"] if a.baseline else None}
    sample = Store("build_ledger").put(semantics, output=report, provenance=here)
    prev = previous_report(cells, (key_of(semantics), sample["n"]))
    md = markdown(report, prev)
    (SCRATCH / f"{name}.md").write_text(md)
    print(md)
    print(f"report: build_ledger/{key_of(semantics)} sample {sample['n']}; markdown {SCRATCH / name}.md")
    rc, text = run([PY, "tools/dashboard.py"])
    print(text.strip() if rc == 0 else f"dashboard: not rendered (rc {rc}): {text[-400:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
