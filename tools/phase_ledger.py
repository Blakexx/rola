#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PHASE LEDGER RUN: one carry launch on a cell with the kernel's phase clock bound, printed
as cycles a warp a window per phase, averaged over the CTAs and warps; and per warp, so an
owner imbalance shows. KERNEL_STANDARDS §19's per-phase instrument for the composed kernel.

    bazel run //tools:phase_ledger -- flagship-alt-k4 [--launches 3]
    bazel run //tools:phase_ledger -- nl64k-alt-k4 --budget tools/budgets/carry.json   # against the predicted phases
    bazel run //tools:phase_ledger -- nl64k-alt-h4-g4 --kernel produce                 # the producer, one cell

With `--budget`, each phase against the cycles the design predicted for it (the cell's `phase_cycles`): red where a
phase takes more than `MISS` times its prediction, or has none (KERNEL_STANDARDS section 32: a miss stops work until
the price or the model that was wrong is named).

The ledger is `rola.ops.carry.LEDGER_PHASES` columns by `LEDGER_WARPS` warps by one row a CTA, a CTA a box of
the carve (docs/internals/carry/ampere/carry_kernel.md#phase-ledger). Measured work: takes the GPU lock exclusive.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


#: the largest measured / predicted a phase passes at (KERNEL_STANDARDS section 32)
MISS = 1.3


def judge(cell: str, per_phase: dict[str, float], budget: Path) -> bool:
    """Each phase's measured cycles a warp a window against the budget's prediction for the cell; True when none
    misses."""
    predicted = json.loads(budget.read_text())["cells"].get(cell, {}).get("phase_cycles", {})
    ok = True
    for name, got in per_phase.items():
        want = predicted.get(name)
        red = want is None or got > MISS * want
        ok &= not red
        shown = "no prediction" if want is None else f"predicted {want:.0f}, {got / want:.2f}x" if want else "0"
        print(f"{'RED ' if red else 'ok  '} {name:8s} {got:7.0f}  ({shown})")
    return ok


def ctas(cell: str) -> int:
    """The launch's CTAs: the carve's boxes a write head, times the heads the cell draws (one batch row)."""
    from measure.cells import by_name
    from measure.subjects import cell_carve

    spec = by_name(cell)
    carve = cell_carve(spec)
    return (spec.dv // carve.channels) * math.prod(w // e for w, e in zip(carve.widths, carve.extents)) * spec.heads


def cta_windows(cell: str) -> int:
    """The launch's CTA-windows, CTAs times windows: the one divisor that turns a launch total into a unit."""
    from measure.cells import by_name
    from rola.ops.window import window_of

    return ctas(cell) * ((by_name(cell).tokens + window_of() - 1) // window_of())


def _intra(spec):
    """The intra's module and ``(operands, 1)`` for a cell: its launch over the facts pass's output on the cell's bundle
    into zeroed planes. Its ledger is ``[warps][phases]`` booked over every CTA, so its one row is the window's CTAs
    summed."""
    import torch

    from measure.cells import realize
    from measure.subjects import bundle_of, lists_form_of
    from rola.ops import facts as f
    from rola.ops import intra as i

    routes, values, layout, statics = bundle_of(spec, realize(spec))
    facts = f.facts(routes, values, layout, *statics, lists_form_of(spec))
    bh = routes.batch * routes.heads
    num = torch.zeros(bh, spec.tokens, i.VALUE_WIDTH, device="cuda")
    den = torch.zeros(bh, spec.tokens, device="cuda")
    return i, ((routes, values, facts, layout, *statics, num, den), 1)


def _produce(construction: str, launches: int, warmup: int, warmup_ms: float) -> tuple[list, int, list]:
    """The producer's ledger on a layer construction's first cell: ``(jobs, tiles, ledger rows)`` -- its launch as the
    layer calls it (`measure.subjects.produce`), the ledger `[jobs, warps, phases]` booked over the token tiles."""
    import torch
    from rola_devtools.locks.gpu import gpu_lock

    from measure.cells import by_name
    from measure.cells.layer import CONSTRUCTIONS, build
    from rola.ops import produce as pr

    c = CONSTRUCTIONS[construction]
    fx = build(by_name(c.cells[0]), c)
    spec, layer = fx["cell"], fx["layer"]
    x = fx["x"][:, :spec.tokens].contiguous()
    jobs = pr.job_table(layer.plan)
    tiles = -(-(spec.B * spec.tokens) // 64)
    with gpu_lock(mode="exclusive"), torch.no_grad():
        ledger = torch.zeros((len(jobs), pr.LEDGER_WARPS, len(pr.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
        start, warmed = time.perf_counter(), 0
        while warmed < warmup or time.perf_counter() - start < warmup_ms / 1e3:
            layer._produce(x)
            torch.cuda.synchronize()
            warmed += 1
        pr.ledger_bind(ledger)
        for _ in range(launches):
            layer._produce(x)
        torch.cuda.synchronize()
        pr.ledger_bind(None)
    return jobs, tiles, (ledger.double() / launches / tiles).cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="the fewest launches run before the ledger is bound (a fresh binary's first launch is not a "
                         "reading)")
    ap.add_argument("--warmup-ms", type=float, default=500.0,
                    help="warm-up launches continue until this long has passed: a GPU just locked reads ~11%% slow "
                         "for its first launches")
    ap.add_argument("--kernel", choices=("carry", "intra", "produce"), default="carry",
                    help="the kernel launched: the carry over its operands or the intra over the facts "
                         "pass's output (the intra's row is every CTA of a window summed)")
    ap.add_argument("--per-warp", default=None,
                    help="the phases whose per-warp rows are printed (default the carry's readout and fold): an imbalance is seen, not inferred")
    ap.add_argument("--json", type=Path, default=None, help="also write the reading here: every phase, every warp")
    ap.add_argument("--per-cta", action="store_true",
                    help="also print the phases of the grid's first wave of CTAs (one an SM) and of the rest apart")
    ap.add_argument("--budget", type=Path, default=None,
                    help="a budget whose cell names `phase_cycles`, each phase's predicted cycles a warp a window")
    ap.add_argument("--forced", default=None,
                    help="the plan form each window takes where it fits (`rola.ops.carry.forced_word`)")
    a = ap.parse_args()

    if a.kernel == "produce":
        from rola.ops import produce as pr

        jobs, tiles, per = _produce(a.cell, a.launches, a.warmup, a.warmup_ms)
        out = {}
        for kind, name in ((0, "route"), (1, "value")):
            rows = [i for i, job in enumerate(jobs) if job[0] == kind]
            if not rows:
                continue
            mean = per[rows].mean(dim=(0, 1))
            out[name] = {n: round(float(x), 1) for n, x in zip(pr.LEDGER_PHASES, mean.tolist())}
            print(f"{a.cell} produce {name} jobs ({len(rows)}): cycles a warp a tile (mean over jobs, warps): "
                  + ", ".join(f"{n} {x:.0f}" for n, x in zip(pr.LEDGER_PHASES, mean.tolist()) if x)
                  + f"  total {float(mean.sum()):.0f}")
        if a.json:
            a.json.write_text(json.dumps({"cell": a.cell, "kernel": "produce", "launches": a.launches, "tiles": tiles,
                                          "per_phase": out}) + "\n")
        return 0

    import torch
    from rola_devtools.locks.gpu import gpu_lock

    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import carry as c
    from rola.ops.window import window_of

    spec = by_name(a.cell)
    if a.kernel == "intra":
        c, (operands, n_ctas) = _intra(spec)
        roles = {"all": (range(c.LEDGER_WARPS), c.LEDGER_PHASES)}

        def launch(*ops):
            return c.intra_forward(*ops)
    else:
        roles = {"all": (range(c.LEDGER_WARPS), c.LEDGER_PHASES)}

        def launch(*ops):
            return c.carry(*ops, forced=a.forced)

        operands = carry_operands(spec)
        n_ctas = ctas(a.cell)
    phases = c.LEDGER_PHASES
    warps = c.LEDGER_WARPS
    per_warp = a.per_warp or {"intra": "gram,wait"}.get(a.kernel, "readout,fold")
    windows = (spec.tokens + window_of() - 1) // window_of()
    with gpu_lock(mode="exclusive"):
        ledger = torch.zeros((n_ctas, warps, len(phases)), dtype=torch.int64, device="cuda")
        start, warmed = time.perf_counter(), 0
        while warmed < a.warmup or time.perf_counter() - start < a.warmup_ms / 1e3:
            launch(*operands)
            torch.cuda.synchronize()
            warmed += 1
        c.ledger_bind(ledger[0] if a.kernel == "intra" else ledger)
        for _ in range(a.launches):
            launch(*operands)
        torch.cuda.synchronize()
        c.ledger_bind(None)
    for role, (rows, booked) in roles.items():
        part = ledger[:, rows.start:rows.stop].double().mean(dim=(0, 1)).cpu() / a.launches / windows
        print(f"{a.cell} {a.kernel}" + (f" {role} warps" if len(roles) > 1 else "")
              + ": cycles a warp a window (mean over CTAs, warps): "
              + ", ".join(f"{n} {x:.0f}" for n, x in zip(phases, part.tolist()) if n in booked)
              + f"  total {part.sum():.0f}")
    per = ledger.double().mean(dim=(0, 1)).cpu() / a.launches / windows
    by_warp = ledger.double().mean(dim=0).cpu() / a.launches / windows
    for ph in per_warp.split(","):
        i = phases.index(ph)
        print(f"  {ph:8s} a warp: " + " ".join(f"{by_warp[w][i]:6.0f}" for w in range(warps)))
    if a.per_cta:
        #: each CTA's phases (mean over its warps), CTAs in launch order: the first `SMs` CTAs are the grid's first wave
        cta = ledger.double().mean(dim=1).cpu() / a.launches / windows
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        for lo, hi in ((0, min(sms, n_ctas)), (min(sms, n_ctas), n_ctas)):
            if hi > lo:
                part = cta[lo:hi].mean(dim=0)
                print(f"  CTAs {lo}-{hi - 1}: " + ", ".join(f"{n} {x:.0f}" for n, x in zip(phases, part.tolist()))
                      + f"  total {part.sum():.0f}")
    if a.json:
        a.json.write_text(json.dumps({"cell": a.cell, "kernel": a.kernel, "launches": a.launches,
                                      "warmup": warmed,
                                      "cta_windows": n_ctas * windows,
                                      "per_phase": {n: round(x, 1) for n, x in zip(phases, per.tolist())},
                                      "total": round(float(per.sum()), 1),
                                      "per_warp": {n: [round(by_warp[w][i].item(), 1) for w in range(warps)]
                                                   for i, n in enumerate(phases)}}) + "\n")
    if a.budget:
        return 0 if judge(a.cell, dict(zip(phases, per.tolist())), a.budget) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
