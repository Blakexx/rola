#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PIPE TIMELINE (KERNEL_STANDARDS §22 (11)): the pipes' activity over one launch, measured on silicon.

    python tools/pipe_timeline.py --cell flagship-dense [--rep existing.ncu-rep] [--html out.html]

The capture is the Nsight Compute CLI's PM sampling (`--section PmSampling`, CLI 2025.3 or later): the profiler
replays the kernel over several passes, each sampling one single-pass counter group at a fixed interval (1 us on
this card, ~1,665 cycles), and aligns the groups by timestamp. The export is the raw page with every metric
instance (`--page raw --print-metric-instances details --csv`): per sample, a timestamp and a value, averaged over
the device's SMs. The series read here are the tensor pipe's active cycles (real time), the ALU, XU and FMA
pipes, warps active and SM cycles active, and the shared LSU's wavefronts, each placed on the time axis of its
own pass group's workload start.

THE SCALE. The tensor series is a percent of the profiler's modeled peak, not of this atom's saturation. The
calibration row (`--scale`) is the MMA-only composition's plateau: with every readout and fold part stubbed, its
MMA phases issue HMMAs as fast as the pipe takes them (measured 31-33 cycles an HMMA a scheduler), so its plateau
in this series is 100% saturation, and a reading divided by it is the pipe's true utilization at that moment.

Reports land beside the capture: `<out>.json` (series, summary) and `<out>.html` (the chart), and `--json` writes the
same document where a caller asks for it. The TIMELINE ITSELF is stored by the `timeline` target of `declare.py` and by
nothing here. A CALIBRATION (`--calibrate`) is the exception and writes `pipe_timeline.scale`, because this tool reads it back
as the scale of every later run (`--no-record` measures one without storing it).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import dev_config  # noqa: E402 -- path insert must precede this import
import toolchains  # noqa: E402
from rola_devtools import process  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NCU = dev_config.get("toolchain.ncu")
#: the `rola_results` location of the calibrations: the MMA-only composition's plateau, per binary
SCALE = "pipe_timeline.scale"
PY = sys.executable

#: the series a timeline reads: (short name, metric column in the raw export, pass-group workload-start key)
SERIES = (
    ("tensor", "pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"),
    ("alu", "SM_A.TriageSCG.sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed"),
    ("xu", "pmsampling:sm__inst_executed_pipe_xu_realtime.avg.pct_of_peak_sustained_elapsed"),
    ("fma_heavy", "SM_C.TriageSCG.smsp__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_elapsed"),
    ("fma_lite", "SM_C.TriageSCG.smsp__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_elapsed"),
    ("warps", "TriageAC.tpc__warps_active_realtime.avg.per_cycle_active"),
    ("sm_active", "SM_A.TriageAC.sm__cycles_active.avg"),
    ("lsu_wavefronts", "SM_A.TriageAC.l1tex__data_pipe_lsu_wavefronts.avg"),
)
_INSTANCE = re.compile(r"(\d{15,}) \(\+\d+\): ([-\d.eE+]+)")


def capture(cell: str, out: Path) -> Path:
    """One launch of the cell through rola's runner under PM sampling; returns the report."""
    from bench.provider import oneshot_argv

    r = process.run([NCU, "--target-processes", "all", "-k", "regex:carry_kernel", "-c", "1", "--section",
                     "PmSampling", "-f", "-o", str(out), *oneshot_argv(cell)], cwd=ROOT, timeout=3600)
    rep = out.with_suffix(".ncu-rep") if out.suffix != ".ncu-rep" else out
    if r.returncode or not rep.exists():
        raise SystemExit(f"pipe_timeline: the capture failed:\n{(r.stdout + r.stderr)[-1500:]}")
    return rep


def export(rep: Path) -> tuple[list[str], list[str]]:
    r = process.run([NCU, "--import", str(rep), "--page", "raw", "--print-metric-instances", "details", "--csv"],
                    timeout=3600)
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.reader(r.stdout.splitlines()))
    if len(rows) < 3:
        raise SystemExit(f"pipe_timeline: the export has no values row:\n{r.stdout[-800:]}{r.stderr[-800:]}")
    return rows[0], rows[2]


def _first_stamp(value: str) -> int | None:
    m = re.search(r"\((\d{15,})", value)
    return int(m.group(1)) if m else None


def parse(hdr: list[str], vals: list[str]) -> dict:
    """Every series as (microseconds since its pass group's workload start, value), inside the workload."""
    col = {h: i for i, h in enumerate(hdr)}
    groups = []
    g = 0
    while f"profiler__timestamp_workload_start_{g}" in col:
        s = _first_stamp(vals[col[f"profiler__timestamp_workload_start_{g}"]])
        e = _first_stamp(vals[col[f"profiler__timestamp_workload_end_{g}"]])
        groups.append((s, e))
        g += 1
    series = {}
    for name, metric in SERIES:
        if metric not in col:
            continue
        pts = [(int(t), float(x)) for t, x in _INSTANCE.findall(vals[col[metric]])]
        window = next(((s, e) for s, e in groups if s is not None and any(s <= t <= e for t, _ in pts)), None)
        if window is None:
            continue
        s, e = window
        series[name] = [((t - s) / 1e3, x) for t, x in pts if s <= t <= e]
    duration = max((e - s) / 1e3 for s, e in groups if s is not None and e is not None) if groups else 0.0
    meta = {k: vals[col[k]] for k in ("profiler__pmsampler_interval_time", "profiler__replayer_passes",
                                      "sm__inst_executed_pipe_tensor.sum", "sm__cycles_elapsed.avg",
                                      "device__attribute_multiprocessor_count") if k in col}
    return {"duration_us": duration, "series": series, "meta": meta}


def bins(points: list[tuple[float, float]], width: float, duration: float) -> list[float | None]:
    out = []
    n = max(1, int(duration / width) + 1)
    acc = [[] for _ in range(n)]
    for t, x in points:
        acc[min(n - 1, int(t / width))].append(x)
    for a in acc:
        out.append(statistics.fmean(a) if a else None)
    return out


def summary(tl: dict, scale: float | None) -> dict:
    """The tensor pipe over the samples where every SM is fully occupied (warps active at their maximum), so a
    partial last wave does not dilute it; with a scale, as true utilization."""
    s = tl["series"]
    ten = s.get("tensor", [])
    war = dict(s.get("warps", []))
    if not ten:
        return {}
    wmax = max(war.values()) if war else None
    full = [x for t, x in ten if wmax is None or war.get(t, 0.0) >= 0.95 * wmax]
    busy = [x for x in full if x > 1.0]
    out = {"samples": len(ten), "full_occupancy_samples": len(full),
           "tensor_mean_full": statistics.fmean(full) if full else None,
           "tensor_p90_full": sorted(full)[int(0.9 * (len(full) - 1))] if full else None,
           "tensor_idle_share_full": (len(full) - len(busy)) / len(full) if full else None}
    if scale:
        out["utilization_mean_full"] = out["tensor_mean_full"] / scale if full else None
    return out


def chart(tl: dict, title: str, scale: float | None) -> str:
    """An inline-SVG chart: the tensor pipe (scaled when a scale is given), the ALU pipe and warps active."""
    d = tl["duration_us"]
    W, H, pad = 1100, 300, 40
    series = tl["series"]

    def path(pts, ymax, color):
        if not pts:
            return ""
        xs = [pad + (t / d) * (W - 2 * pad) for t, _ in pts]
        ys = [H - pad - min(1.0, x / ymax) * (H - 2 * pad) for _, x in pts]
        dd = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        return f'<path d="{dd}" fill="none" stroke="{color}" stroke-width="1"/>'

    ten = series.get("tensor", [])
    if scale:
        ten = [(t, 100.0 * x / scale) for t, x in ten]
    alu = series.get("alu", [])
    war = [(t, x * 12.5) for t, x in series.get("warps", [])]
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" style="max-width:100%;height:auto">',
           f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="currentColor" stroke-width="0.5"/>',
           path(ten, 100.0, "#2a7de1"), path(alu, 100.0, "#d98b2b"), path(war, 100.0, "#6a9a4c"),
           f'<text x="{pad}" y="{pad - 12}" font-size="12" fill="currentColor">{html.escape(title)}</text>',
           f'<text x="{W - pad}" y="{H - pad + 16}" font-size="11" text-anchor="end" fill="currentColor">{d:.0f} us</text>',
           "</svg>"]
    legend = ("<p><span style='color:#2a7de1'>tensor pipe" + (" (true utilization %)" if scale else " (% of modeled peak)")
              + "</span> · <span style='color:#d98b2b'>ALU pipe (% of modeled peak)</span> · "
              "<span style='color:#6a9a4c'>warps active (x12.5, 8 = 100)</span></p>")
    return "\n".join(svg) + legend


def plateau(tl: dict) -> dict:
    """The MMA-only composition's plateau in the tensor series: the median of the full-occupancy samples where the
    pipe is busy (above 5% of the modeled peak), with its spread."""
    ten = tl["series"]["tensor"]
    war = dict(tl["series"].get("warps", []))
    wmax = max(war.values()) if war else None
    busy = sorted(x for t, x in ten if (wmax is None or war.get(t, 0.0) >= 0.95 * wmax) and x > 5.0)
    if not busy:
        raise SystemExit("pipe_timeline: no busy full-occupancy samples to calibrate from")
    return {"plateau": statistics.median(busy), "p10": busy[int(0.1 * (len(busy) - 1))],
            "p90": busy[int(0.9 * (len(busy) - 1))], "samples": len(busy)}


def stored_scale() -> float | None:
    """The newest calibration's plateau (`rola_results` at `pipe_timeline.scale`)."""
    from rola_results import Store

    store = Store(SCALE)
    newest = max(((s["utc"], store.dir / s["output"]) for r in store.records() for s in r["samples"] if s["ok"]), default=None)
    return json.loads(newest[1].read_text())["plateau"] if newest else None


def binary() -> str:
    from rola_results import digest

    return digest(toolchains.built_extension(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--rep", type=Path, help="an existing PmSampling report instead of a new capture")
    ap.add_argument("--out", type=Path, help="report path stem (default: the scratch directory's pipe_timeline/<cell>)")
    ap.add_argument("--scale", type=float, help="the MMA-only plateau in the tensor series (default: the stored row)")
    ap.add_argument("--calibrate", action="store_true",
                    help="this capture is the MMA-only composition (ROLA_CARRY_PARTS=none): store its plateau as the scale")
    ap.add_argument("--no-record", action="store_true",
                    help="store no CALIBRATION: --calibrate then only prints the plateau it measured")
    ap.add_argument("--json", type=Path, help="also write the timeline document here")
    a = ap.parse_args()
    from rola_results import Store, checkout

    out = a.out or dev_config.scratch("pipe_timeline") / a.cell
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = a.rep or capture(a.cell, out)
    tl = parse(*export(rep))
    if a.calibrate:
        import datetime as dt
        row = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ"), "cell": a.cell, "report": rep.name,
               "metric": dict(SERIES)["tensor"], "composition": "ROLA_CARRY_PARTS=none", **plateau(tl)}
        if not a.no_record:
            Store(SCALE).put({"cell": a.cell, "binary": binary(), "composition": row["composition"]}, output=row,
                             provenance=checkout(ROOT))
        print(f"scale: {row['plateau']:.2f} (p10 {row['p10']:.2f}, p90 {row['p90']:.2f}, {row['samples']} samples)")
        a.scale = row["plateau"]
    if a.scale is None:
        a.scale = stored_scale()
    summ = summary(tl, a.scale)
    doc = {"cell": a.cell, "report": rep.name, "scale": a.scale, "summary": summ, **tl}
    out.with_suffix(".json").write_text(json.dumps(doc) + "\n")
    if a.json:
        a.json.write_text(json.dumps(doc) + "\n")
    out.with_suffix(".html").write_text(f"<title>Pipe timeline {html.escape(a.cell)}</title>\n"
                                        + chart(tl, a.cell, a.scale) + "\n")
    print(json.dumps({"cell": a.cell, "duration_us": tl["duration_us"], "series": sorted(tl["series"]), **summ},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
