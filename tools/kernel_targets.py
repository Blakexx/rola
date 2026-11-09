#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INSTRUMENTS' KERNELS: what a per-launch instrument needs to read one kernel family on a cell -- the profiler's
kernel filter, the provider arm whose one untimed launch it captures, the cubin member and sources the region ledger
attributes against, the budget, and the launch's units (the divisor that turns a launch total into the budget's unit).

    bazel run //tools:kernel_targets -- nl64k-alt-k4      # every family's target on a cell: its arm and its units

`tools/stall_census.py`, `tools/pipe_counters.py`, `tools/pipe_timeline.py` and `tools/phase_ledger.py` take
`--kernel FAMILY` from this table (default the carry). Docs: docs/internals/tools/kernel_targets.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]


def schedule_sources(kernel: str) -> tuple[str, ...]:
    """A kernel's schedule files, derived from the tree: its kernel file `<kernel>_kernel.cuh` FIRST (the instruments
    read the first source as the kernel's own), then every other header of `csrc/rola/src/<kernel>/ampere/` in order,
    so a file a schedule gains is read without a list to keep."""
    folder = ROOT / "csrc/rola/src" / kernel / "ampere"
    own = f"{kernel}_kernel.cuh"
    if not (folder / own).is_file():
        raise SystemExit(f"{folder}: no {own}, the kernel file the instruments read first")
    rest = sorted(p.name for p in folder.glob("*.cuh") if p.name != own)
    return tuple(f"csrc/rola/src/{kernel}/ampere/{name}" for name in (own, *rest))


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


def _heads(cell) -> int:
    """The query heads a call reads: the write heads times each one's group."""
    return cell.heads * max(cell.group, 1)


def units(target: Target, cell) -> int:
    """The launch's count of the budget's unit on a cell: one batch row, the cell's write heads (each read by its
    group's query heads)."""
    import phase_ledger

    if target.family == "carry":
        return phase_ledger.cta_windows(cell.name)
    if target.family == "intra":
        from rola.ops import intra as intra_ops
        from rola.ops.window import window_of

        #: a warp's tile pairs: each 16-reader group against every writer tile at or before its own, times the warps
        #: that split a tile's 64 writers (the card's block rows / 2 writer columns a warp)
        cols = intra_ops.block_rows(0) // 2
        groups = window_of() // 16
        pairs = sum(r // 4 + 1 for r in range(groups)) * (64 // cols)
        return pairs * _windows(cell) * _heads(cell)
    return _windows(cell) * cell.heads


def member(target: Target, cell) -> str:
    """The ELF image an instrument reads for ``cell``: the family's member, or -- for the carry, whose arms are one
    image each (or a shared one, `gen_shards.carry_tu_of`) -- the image of the arm the cell's carve selects, its name
    ended so `carry_arm_1` never matches `carry_arm_10`."""
    if target.family != "carry":
        return target.member
    import gen_shards

    from measure.subjects import cell_carve
    from rola.ops.carry import arm_key

    key = tuple(arm_key(cell_carve(cell)))
    rows = [i for i, row in enumerate(gen_shards.CARRY_ARMS) if tuple(row) == key]
    if len(rows) != 1:
        raise SystemExit(f"{cell.name}: its carve's arm {key} is {len(rows)} rows of the declared arm list")
    return f"carry_arm_{gen_shards.carry_tu_of()[rows[0]]}."


def launch(target: Target, name: str) -> tuple[str, str]:
    """``(cell, arm)`` an instrument launches for ``name``: a central cell and the family's provider arm, or -- for the
    producer, whose arms are a layer construction's -- the construction ``name`` on its first cell."""
    from measure.provider import arm_name

    if target.family == "produce":
        from measure.cells.layer import CONSTRUCTIONS

        return CONSTRUCTIONS[name].cells[0], f"{target.arm}@layer={name}"
    return name, arm_name(target.arm)


def produce_units(name: str) -> int:
    """The producer's launch in its unit: its token tiles times its jobs, for the construction ``name``."""
    from measure.cells import by_name
    from measure.cells.layer import CONSTRUCTIONS, build
    from rola.ops import produce as pr

    c = CONSTRUCTIONS[name]
    fx = build(by_name(c.cells[0]), c)
    return -(-(fx["cell"].B * fx["cell"].tokens) // 64) * len(pr.job_table(fx["layer"].plan))


def function(target: Target, cell) -> str:
    """The instantiation a cell launches, as a regex over the cubin's mangled names (the region ledger's; the
    profiler's filter is the family's `symbol`, which it matches against the plain function name): the intra's depth
    and the card's accumulation (`rola.ops.carry.readout_accumulate`), after its build's tag (the first template
    argument)."""
    if target.family == "intra":
        from rola.ops.carry import readout_accumulate

        return f"intra_kernelI.*ELi{len(cell.widths)}ELb{int(readout_accumulate() == 'fp16')}E"
    return target.symbol


TARGETS = {
    "carry": Target("carry", "carry_kernel", "carry_forward", "carry_arm",
                    schedule_sources("carry"),
                    "tools/budgets/carry.json", "CTA-windows"),
    "intra": Target("intra", "intra_kernel", "intra_forward", "intra", ("csrc/rola/src/intra/ampere/intra_kernel.cuh",),
                    "tools/budgets/intra.json", "warp tile pairs"),
    "facts": Target("facts", "facts_pass", "facts_pass", "facts", ("csrc/rola/src/facts/facts.cu",), None,
                    "windows"),
    "produce": Target("produce", "produce_kernel", "produce", "produce", ("csrc/rola/src/produce/ampere/produce_kernel.cuh",),
                      "tools/budgets/produce.json", "job tiles"),
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
        if t.family == "produce":
            continue
        print(f"{t.family:6s} arm {t.arm:14s} kernel {function(t, cell):26s} {units(t, cell):8d} {t.unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
