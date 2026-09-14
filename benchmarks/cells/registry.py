# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTRY BY NAME: `{name -> (kind, record)}` over both cell kinds, read once.

A name is unique across the registry, so a command line names a cell and its kind follows: `carry` for a record of
`carry_cells.json` (a shape and a draw), `layer` for a record of `layer_cells.json` (a constructor). `GROUPS` names the
slices a run is cited by rather than a list of cells. `runnable` is which of them this checkout's binary runs.
"""
from __future__ import annotations

from benchmarks.cells import carry_cells
from benchmarks.cells.layer import layer_cells

CELLS = {**{c.name: ("carry", c) for c in carry_cells()},
         **{c.name: ("layer", c) for c in layer_cells()}}

GROUPS = {
    "all": list(CELLS),
    "carry": [n for n, (kind, _) in CELLS.items() if kind == "carry"],
    "layer": [n for n, (kind, _) in CELLS.items() if kind == "layer"],
    "decode": [n for n, (kind, c) in CELLS.items() if kind == "layer" and c.decode_steps > 0],
}


def runnable() -> dict:
    """WHICH CELLS THIS CHECKOUT'S BINARY RUNS, from the binary's own answer (`rola.ops.carry.arms()`).

    A binary without an arm its tree ships is REFUSED: it is an iteration build (`ROLA_CARRY_ARMS`), and what it
    measures is not the tree. Otherwise a carry cell runs where the binary carries its arm, and the rest are named by
    why: `undeclared`, an arm the tree does not declare, which no build of it carries; `unbuilt`, a test arm this build
    left out. Layer cells always run.
    """
    import sys
    from pathlib import Path

    from rola.ops.carry import arms

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import gen_shards

    declared = {tuple(row) for row in gen_shards.CARRY_ARMS}
    shipped = {tuple(gen_shards.CARRY_ARMS[i]) for i in gen_shards.CARRY_SHIPPED_ARMS}
    built = {tuple(arm) for arm in arms()}
    missing = sorted(shipped - built)
    if missing:
        raise RuntimeError(f"this binary carries {sorted(built)} and lacks the shipped arms {missing}: an iteration "
                           "build (ROLA_CARRY_ARMS) measures a subset of the tree; build the shipped set")
    carry = {name: spec for name, (kind, spec) in CELLS.items() if kind == "carry"}
    return {"cells": [name for name, (kind, spec) in CELLS.items() if kind != "carry" or spec.arm in built],
            "undeclared": sorted(name for name, spec in carry.items() if spec.arm not in declared),
            "unbuilt": sorted(name for name, spec in carry.items() if spec.arm in declared - built),
            "arms": sorted(built)}
