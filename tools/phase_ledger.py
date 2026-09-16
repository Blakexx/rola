#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PHASE LEDGER RUN: one carry launch on a cell with the kernel's phase clock bound, printed
as cycles a warp a window per phase, averaged over the CTAs and warps; and per warp, so an
owner imbalance shows. KERNEL_STANDARDS §19's per-phase instrument for the composed kernel.

    python tools/phase_ledger.py flagship-alt-k4 [--launches 3] [--state-arm fresh|null]

The launch carries a state plane out (`fresh`, the A/B's arm) unless `--state-arm null`: the
exit sweep is part of the call, and a ledger without it once hid a third of a sparse kernel.
Measured work: takes the GPU lock exclusive.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PHASES = ("head", "readout", "fold", "snapshot", "edges", "sweep", "head_words", "head_scans")


def cta_windows(cell: str) -> int:
    """The launch's CTA-windows, owners times windows: the one divisor that turns a launch total into a unit."""
    from measure.cells import by_name
    from rola.ops import carry as c

    spec = by_name(cell)
    return (math.prod(spec.widths) // 256) * ((spec.tokens + c.WINDOW - 1) // c.WINDOW)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="launches run before the ledger is bound (a fresh binary's first launch is not a reading)")
    ap.add_argument("--state-arm", choices=("fresh", "null"), default="fresh")
    ap.add_argument("--per-warp", default="readout,fold",
                    help="the phases whose per-warp rows are printed: an imbalance is seen, not inferred")
    ap.add_argument("--json", type=Path, default=None, help="also write the reading here: every phase, every warp")
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
    windows = (spec.tokens + c.WINDOW - 1) // c.WINDOW
    with gpu_lock(mode="exclusive"):
        ledger = torch.zeros((owners, warps, len(PHASES)), dtype=torch.int64, device="cuda")
        plane = c.state_plane(kw["descriptor"], 1) if a.state_arm == "fresh" else None
        ext = c.extension()
        for _ in range(a.warmup):
            c.carry_forward(routes, v, state_out=plane, **kw)
        torch.cuda.synchronize()
        ext.carry_ledger_bind(ledger)
        for _ in range(a.launches):
            c.carry_forward(routes, v, state_out=plane, **kw)
        torch.cuda.synchronize()
        ext.carry_ledger_bind(None)
    per = ledger.double().mean(dim=(0, 1)).cpu() / a.launches / windows
    print(f"{a.cell}: cycles a warp a window (mean over CTAs, warps): "
          + ", ".join(f"{n} {x:.0f}" for n, x in zip(PHASES, per.tolist())) + f"  total {per.sum():.0f}")
    by_warp = ledger.double().mean(dim=0).cpu() / a.launches / windows
    for ph in a.per_warp.split(","):
        i = PHASES.index(ph)
        print(f"  {ph:8s} a warp: " + " ".join(f"{by_warp[w][i]:6.0f}" for w in range(warps)))
    if a.json:
        a.json.write_text(json.dumps({"cell": a.cell, "launches": a.launches, "warmup": a.warmup, "state_arm": a.state_arm,
                                      "cta_windows": owners * windows,
                                      "per_phase": {n: round(x, 1) for n, x in zip(PHASES, per.tolist())},
                                      "total": round(float(per.sum()), 1),
                                      "per_warp": {n: [round(by_warp[w][i].item(), 1) for w in range(warps)]
                                                   for i, n in enumerate(PHASES)}}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
