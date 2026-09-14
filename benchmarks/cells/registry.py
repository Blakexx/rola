# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTRY BY NAME: `{name -> (kind, record)}` over both cell kinds, read once.

A name is unique across the registry, so a command line names a cell and its kind follows: `carry` for a record of
`carry_cells.json` (a shape and a draw), `layer` for a record of `layer_cells.json` (a constructor). `GROUPS` names the
slices a run is cited by rather than a list of cells.
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
