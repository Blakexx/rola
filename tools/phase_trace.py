#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PHASE TRACE RUN: one carry launch on a cell with the kernel's phase trace bound -- every
warp's stamps (the ledger's laps and the streams' steps, `%clock64`) for the first CTAs -- read
back as what each warp was doing when, and what the scheduler's tensor pipe saw.

    python tools/phase_trace.py nl64k-dense [--ctas 8] [--cap 32768] [--json out.json] [--raw out.pt]

The ledger says how long each phase took a warp; the trace says whether the warps were in it at the
same time. Its readings: the finer ledger (a warp's cycles a window by activity), the SCHEDULER'S
VIEW (two warps a scheduler: the share of time zero, one or both are issuing HMMAs, and what they
are doing when neither is), and the LOCKSTEP (the spread of the warps' phase starts within a window).
Measured work: takes the GPU lock exclusive.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_ledger import PHASES  # noqa: E402

#: the trace's events by code: the eight laps (`CarryPhase`), then the streams' steps (`CarryTraceEvent`).
EVENTS = PHASES + ("fold.walk", "fold.fragment", "fold.wait", "fold.fill",
                   "readout.tile", "readout.issue", "readout.wait", "readout.drain")
LAPS = len(PHASES)
MMA_ACTIVE = frozenset({"fold.fragment", "readout.tile"})
SCHEDULERS = 4  #: an SM's schedulers; warp `w` issues on scheduler `w % 4`
HMMA_CYCLES = 32.5  #: the pipe's cost of the kernel's atom a scheduler (calibration.md)


def intervals(cycles: list[int], events: list[int]) -> list[tuple[int, int, str]]:
    """A warp's stamps as labelled intervals. A stream stamp begins the activity it names; a lap ends a phase,
    so the interval after a lap is the phase the NEXT lap names, or the lead into the next stream step."""
    out = []
    for i in range(len(cycles) - 1):
        ev, nxt = events[i], events[i + 1]
        if ev >= LAPS:
            label = EVENTS[ev]
        elif nxt < LAPS:
            label = PHASES[nxt]
        else:
            label = "lead:" + EVENTS[nxt]
        out.append((cycles[i], cycles[i + 1], label))
    return out


def windows(cycles: list[int], events: list[int]) -> list[tuple[int, int]]:
    """The window boundaries on a warp: the prologue's edge lap and each edge lap that closes a fold."""
    edges = PHASES.index("edges")
    fold = PHASES.index("fold")
    bounds = []
    for i, ev in enumerate(events):
        if ev == edges and (not bounds or events[i - 1] == fold):
            bounds.append(cycles[i])
    return list(zip(bounds, bounds[1:]))


def scheduler_view(per_warp: list[list[tuple[int, int, str]]]) -> tuple[dict, dict]:
    """Two warps a scheduler: cycles with 0 / 1 / 2 of them issuing HMMAs, and what each was doing when neither."""
    hist = defaultdict(int)
    idle_doing = defaultdict(int)
    for s in range(SCHEDULERS):
        pair = [iv for w, iv in enumerate(per_warp) if w % SCHEDULERS == s]
        if len(pair) != 2:
            continue
        a, b = pair
        t0 = max(a[0][0], b[0][0])
        t1 = min(a[-1][1], b[-1][1])
        ia = ib = 0
        t = t0
        while t < t1:
            while a[ia][1] <= t:
                ia += 1
            while b[ib][1] <= t:
                ib += 1
            nt = min(a[ia][1], b[ib][1], t1)
            la, lb = a[ia][2], b[ib][2]
            k = (la in MMA_ACTIVE) + (lb in MMA_ACTIVE)
            hist[k] += nt - t
            if k == 0:
                idle_doing[la] += nt - t
                idle_doing[lb] += nt - t
            t = nt
    return dict(hist), dict(idle_doing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--ctas", type=int, default=8, help="the CTAs traced, the grid's first")
    ap.add_argument("--cap", type=int, default=32768,
                    help="stamps a warp (a dense window is ~110 stamps; 128 windows fit); a warp that fills its row is reported")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--state-arm", choices=("fresh", "null"), default="fresh")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--raw", type=Path, default=None, help="also save the stamp tensor here (torch.save)")
    a = ap.parse_args()

    import torch
    from rola_devtools.locks.gpu import gpu_lock

    from measure.cells import WARPS_PER_CTA, by_name, carry_call
    from rola.ops import carry as c

    spec = by_name(a.cell)
    _drawn, kw = carry_call(spec, 1)
    routes, v = kw.pop("routes"), kw.pop("v")
    owners = math.prod(spec.widths) // 256
    warps = WARPS_PER_CTA
    ctas = min(a.ctas, owners)
    with gpu_lock(mode="exclusive"):
        trace = torch.zeros((ctas, warps, a.cap), dtype=torch.int64, device="cuda")
        plane = c.state_plane(kw["descriptor"], 1) if a.state_arm == "fresh" else None
        ext = c.extension()
        for _ in range(a.warmup):
            c.carry_forward(routes, v, state_out=plane, **kw)
        torch.cuda.synchronize()
        ext.carry_trace_bind(trace, warps)
        c.carry_forward(routes, v, state_out=plane, **kw)
        torch.cuda.synchronize()
        ext.carry_trace_bind(None, warps)
    if a.raw:
        torch.save(trace.cpu(), a.raw)
    rows = trace.cpu()

    by_label = defaultdict(list)     #: label -> cycles a warp a window
    counts = defaultdict(list)       #: label -> events a warp a window
    durations = defaultdict(list)    #: label -> one interval's cycles
    by_group = defaultdict(list)     #: (label, warp >= 4) -> one interval's cycles: the scheduler's first and second warp
    hist_total = defaultdict(int)
    idle_total = defaultdict(int)
    fold_spread, read_spread, fold_len = [], [], []
    overflow = 0
    nwin_seen = []
    for cta in range(ctas):
        per_warp = []
        starts = []
        for w in range(warps):
            raw = rows[cta, w]
            raw = raw[raw != 0]
            if raw.numel() >= a.cap:
                overflow += 1
            cyc = (raw >> 8).tolist()
            ev = (raw & 0xFF).tolist()
            iv = intervals(cyc, ev)
            per_warp.append(iv)
            wins = windows(cyc, ev)
            nwin_seen.append(len(wins))
            wstarts = []
            for (t0, t1) in wins:
                tot = defaultdict(int)
                cnt = defaultdict(int)
                first_fold = first_read = None
                for (s, e, lab) in iv:
                    if s < t0 or s >= t1:
                        continue
                    tot[lab] += e - s
                    cnt[lab] += 1
                    durations[lab].append(e - s)
                    by_group[(lab, w >= SCHEDULERS)].append(e - s)
                    if first_fold is None and lab in ("fold.walk", "fold.wait") :
                        first_fold = s
                    if first_read is None and lab in ("readout.tile", "readout.wait", "lead:readout.issue"):
                        first_read = s
                for lab, x in tot.items():
                    by_label[lab].append(x)
                for lab, x in cnt.items():
                    counts[lab].append(x)
                wstarts.append((first_read, first_fold, t1 - t0))
            starts.append(wstarts)
        nw = min(len(s) for s in starts)
        for k in range(nw):
            reads = [starts[w][k][0] for w in range(warps) if starts[w][k][0] is not None]
            folds = [starts[w][k][1] for w in range(warps) if starts[w][k][1] is not None]
            if len(reads) == warps:
                read_spread.append(max(reads) - min(reads))
            if len(folds) == warps:
                fold_spread.append(max(folds) - min(folds))
            fold_len.append(statistics.mean(starts[w][k][2] for w in range(warps)))
        hist, idle = scheduler_view(per_warp)
        for k, x in hist.items():
            hist_total[k] += x
        for lab, x in idle.items():
            idle_total[lab] += x

    windows_n = statistics.median(nwin_seen) if nwin_seen else 0
    print(f"{a.cell}: {ctas} CTAs x {warps} warps traced, {windows_n:.0f} windows a warp"
          + (f"  ({overflow} warp rows FULL: raise --cap)" if overflow else ""))
    print("cycles a warp a window by activity (mean; events a window):")
    order = sorted(by_label, key=lambda lab: -statistics.mean(by_label[lab]))
    total = 0.0
    for lab in order:
        m = statistics.mean(by_label[lab])
        n = statistics.mean(counts[lab])
        med = statistics.median(durations[lab])
        total += m
        print(f"  {lab:20s} {m:8.0f}   x{n:6.1f}   median one {med:6.0f}")
    print(f"  {'total':20s} {total:8.0f}")
    span = sum(hist_total.values())
    if span:
        print("the scheduler's view (two warps a scheduler, share of time):")
        for k in range(3):
            print(f"  {k} warps issuing HMMAs: {100.0 * hist_total.get(k, 0) / span:5.1f}%")
        active_time = sum(k * x for k, x in hist_total.items())
        print(f"  HMMA-active warp-time a scheduler: {100.0 * active_time / span:5.1f}% of the span")
        print("  when neither is (share of the idle time, both warps counted):")
        idle_span = hist_total.get(0, 0)
        for lab, x in sorted(idle_total.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {lab:20s} {100.0 * x / (2 * idle_span):5.1f}%")
    if fold_spread:
        print("lockstep (cycles, median over windows): "
              f"fold starts spread {statistics.median(fold_spread):.0f}, "
              f"readout starts spread {statistics.median(read_spread) if read_spread else float('nan'):.0f}, "
              f"window {statistics.median(fold_len):.0f}")
    for lab, hmmas in (("fold.fragment", 18), ("readout.tile", 144)):
        if lab in durations:
            med = statistics.median(durations[lab])
            first = statistics.median(by_group[(lab, False)]) if by_group[(lab, False)] else float("nan")
            second = statistics.median(by_group[(lab, True)]) if by_group[(lab, True)] else float("nan")
            print(f"one {lab}: median {med:.0f} cycles (warps 0-3 {first:.0f}, 4-7 {second:.0f}) against "
                  f"{hmmas} HMMAs' pipe time {hmmas * HMMA_CYCLES:.0f} a warp alone, x2 shared "
                  f"(x{med / (hmmas * HMMA_CYCLES):.2f})")
    if a.json:
        a.json.write_text(json.dumps({
            "cell": a.cell, "ctas": ctas, "warps": warps, "cap": a.cap, "overflow_rows": overflow,
            "windows_a_warp": windows_n,
            "by_activity": {lab: {"cycles_a_window": round(statistics.mean(by_label[lab]), 1),
                                  "events_a_window": round(statistics.mean(counts[lab]), 2),
                                  "median_one": statistics.median(durations[lab])} for lab in order},
            "scheduler": {"share_by_active": {k: hist_total.get(k, 0) / span for k in range(3)} if span else {},
                          "idle_doing": {lab: x / (2 * hist_total.get(0, 1)) for lab, x in idle_total.items()}},
            "lockstep": {"fold_start_spread_median": statistics.median(fold_spread) if fold_spread else None,
                         "readout_start_spread_median": statistics.median(read_spread) if read_spread else None,
                         "window_median": statistics.median(fold_len) if fold_len else None},
        }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
