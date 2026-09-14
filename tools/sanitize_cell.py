#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ONE registry cell, ONE carry launch, no test framework -- the racecheck lane's driver.

`tools/sanitize_oracle.py` wraps a PYTEST process for most (family, tool) pairs, which is
right: the cell, its draw and its assertions are the oracle tier's, and a sanitizer row that
re-derived them would be a second definition of the cell. RACECHECK is the exception, and
the reason is measured rather than stylistic: its shadow state is per shared access per
BARRIER INTERVAL, the sparse bodies cross no CTA barrier inside a phase (so a phase is ONE
interval), and the host budget it then wants is larger than this box has -- a pytest process
carrying the module, its 160 parametrised rows and torch's own arenas on top of that is
OOM-killed at 23 GB before the report is written (the K2 journal's open item, and it reappears
under the one-box cell, so the cell was not the whole of it).

This driver runs the SAME cell through the SAME binding (`benchmarks.cells.carry_call`) with
nothing else resident: one draw, one launch, one synchronise. It asserts nothing -- correctness
is the oracle tier's row for the same cell -- and it prints a line naming the launch and a
checksum of its outputs, which is what makes a vacuous pass (a cell whose arm is not built, an
empty selection) impossible to read as coverage.

    python tools/sanitize_cell.py --cell one-box-short --order first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--order", default="first", choices=("first", "identity"),
                    help="the order policy, the launch's one dial")
    ap.add_argument("--phase", default="both", choices=("both", "fold", "readout"),
                    help="which phase's schedule to instrument. racecheck's shadow state is "
                         "per shared access per barrier interval and this host's budget "
                         "reaches ONE PHASE at a time on a real draw; zeroing the OTHER "
                         "side's routing plane makes that phase reap everything, so its "
                         "schedule still runs and its memory traffic is nil. The arithmetic "
                         "is the oracle tier's business, not this driver's.")
    args = ap.parse_args()

    import torch

    from benchmarks.cells import by_name, carry_call
    from rola.ops import carry as carry_ops

    spec = by_name(args.cell)
    drawn, call = carry_call(spec, bh=1)
    routes, v = call.pop("routes"), call.pop("v")
    if args.phase != "both":
        dead = torch.zeros_like(routes.read if args.phase == "fold" else routes.write)
        routes = carry_ops.RoutePlanes(
            read=dead if args.phase == "fold" else routes.read,
            write=dead if args.phase == "readout" else routes.write, gain=routes.gain)
        words = call["liveness"].words.clone()
        words[:, 0 if args.phase == "fold" else 1] = 0
        call["liveness"] = carry_ops.LivenessWords(words=words)
    schedule = carry_ops.CarrySchedule(order=args.order)
    state = carry_ops.state_plane(call["descriptor"], 1)
    num, den = carry_ops.carry_forward(routes, v, state_in=None, state_out=state,
                                       schedule=schedule, **call)
    torch.cuda.synchronize()
    print(f"sanitize_cell: {spec.name} phase={args.phase} "
          f"order={args.order} LAUNCHED "
          f"num={int(num.view(torch.int32).abs().sum()):d} "
          f"den={int(den.view(torch.int32).abs().sum()):d} "
          f"state={int(state.view(torch.int32).abs().sum()):d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
