#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INSTRUMENTS' KERNELS: what a per-launch instrument needs to read one kernel family on a cell -- the profiler's
kernel filter, the provider arm whose one untimed launch it captures, the cubin member and sources the region ledger
attributes against, the budget, and the launch's units (the divisor that turns a launch total into the budget's unit).

    bazel run //tools:kernel_targets -- nl64k-alt-k4      # every family's target on a cell: its arm and its units

`tools/stall_census.py`, `tools/pipe_counters.py` and `tools/pipe_timeline.py` take `--kernel FAMILY` from this table.
The facts pass is its one family on this line: a kernel with a schedule adds its row with the schedule.
Docs: docs/internals/tools/kernel_targets.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Target:
    """One family's instrument target."""

    family: str
    symbol: str
    arm: str
    member: str
    sources: tuple[str, ...]
    budget: str | None
    unit: str


def _windows(cell) -> int:
    from rola.ops.window import window_of

    return -(-cell.tokens // window_of())


def units(target: Target, cell) -> int:
    """The launch's count of the budget's unit on a cell: one batch row, the cell's write heads."""
    return _windows(cell) * cell.heads


def launch(target: Target, name: str) -> tuple[str, str]:
    """``(cell, arm)`` an instrument launches for ``name``: a central cell and the family's provider arm."""
    from measure.provider import arm_name

    return name, arm_name(target.arm)


TARGETS = {
    "facts": Target("facts", "facts_pass", "facts_pass", "facts", ("csrc/rola/src/facts/facts.cu",), None,
                    "windows"),
}


def target(family: str) -> Target:
    if family not in TARGETS:
        raise SystemExit(f"no instrument target for {family!r}; the families are {sorted(TARGETS)}")
    return TARGETS[family]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    a = ap.parse_args()
    from measure.cells import by_name

    cell = by_name(a.cell)
    for t in TARGETS.values():
        print(f"{t.family:6s} arm {t.arm:14s} kernel {t.symbol:26s} {units(t, cell):8d} {t.unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
