#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ONE registry cell, ONE carry or intra launch, no test framework -- the racecheck lane's driver.

`tools/sanitize_oracle.py` wraps a PYTEST process for most (family, tool) pairs, which is
right: the cell, its draw and its assertions are the oracle tier's, and a sanitizer row that
re-derived them would be a second definition of the cell. RACECHECK is the exception, and
the reason is measured rather than stylistic: its shadow state is per shared access per
BARRIER INTERVAL, the sparse bodies cross no CTA barrier inside a phase (so a phase is ONE
interval), and the host budget it then wants is larger than this box has -- a pytest process
carrying the module, its 160 parametrised rows and torch's own arenas on top of that is
OOM-killed at 23 GB before the report is written (the K2 journal's open item, and it reappears
under the one-box cell, so the cell was not the whole of it).

This driver runs the SAME cell through the SAME binding (the carry's operands, `measure.subjects.carry_operands`, or
for the intra the bench subject's bundle, `measure.subjects.bundle_of`, and the facts pass over it) with
nothing else resident: one draw, one launch, one synchronise. The intra's row needs it for the
same reason in another form: its oracle test computes the fp64 reference inside the instrumented
process, and racecheck slows every torch kernel of that past the test's timeout. It asserts nothing -- correctness
is the oracle tier's row for the same cell -- and it prints a line naming the launch and a
checksum of its outputs, which is what makes a vacuous pass (a cell whose arm is not built, an
empty selection) impossible to read as coverage.

    bazel run //tools:sanitize_cell -- --cell one-box-short
    bazel run //tools:sanitize_cell -- --kernel intra --cell flagship-alt-k4
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
    ap.add_argument("--kernel", default="carry", choices=("carry", "intra"))
    args = ap.parse_args()

    import torch

    if args.kernel == "intra":
        return _intra(args.cell)

    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import carry as carry_ops

    spec = by_name(args.cell)
    num, _den, check, _ = carry_ops.carry(*carry_operands(spec))
    torch.cuda.synchronize()
    print(f"sanitize_cell: {spec.name} LAUNCHED "
          f"num={int(num.abs().sum()):d} check={float(check):g}")
    return 0


def _intra(cell: str) -> int:
    """The cell's bundle, its facts and one intra launch into zeroed planes."""
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, lists_form_of
    from rola.ops import facts as facts_ops
    from rola.ops import intra as intra_ops

    spec = by_name(cell)
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    unions = facts_ops.facts(routes, values, layout, *statics, lists_form_of(spec)).unions
    bh = routes.batch * routes.heads
    num = torch.zeros(bh, routes.tokens, intra_ops.VALUE_WIDTH, device="cuda")
    den = torch.zeros(bh, routes.tokens, device="cuda")
    intra_ops.intra_forward(routes, values, unions, layout, *statics, num, den)
    torch.cuda.synchronize()
    print(f"sanitize_cell: {spec.name} kernel=intra LAUNCHED "
          f"num={int(num.view(torch.int32).abs().sum()):d} den={int(den.view(torch.int32).abs().sum()):d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
