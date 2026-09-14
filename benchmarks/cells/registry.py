# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTRY BY NAME: `{name -> (kind, record)}` over both cell kinds, read once.

A name is unique across the registry, so a command line names a cell and its kind follows: `carry` for a record of
`carry_cells.json` (a shape and a draw), `layer` for a record of `layer_cells.json` (a constructor). `GROUPS` names the
slices a run is cited by rather than a list of cells.

Both files are registries of CELLS in rola-devtools' sense (`rola_devtools.cells`): each names its data provider
(`benchmarks.cells:carry_cell`, `benchmarks.cells.layer:layer_cell`) and every record's fields are that provider's
parameters. `registry` loads them with any other registry files (another repository's cells and the points that group
cells by runner); `bench.provider` is rola's runner.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.cells import carry_cells
from benchmarks.cells.layer import layer_cells

#: this checkout's cell registry files
FILES = tuple(Path(__file__).resolve().parent / name for name in ("carry_cells.json", "layer_cells.json"))

CELLS = {**{c.name: ("carry", c) for c in carry_cells()},
         **{c.name: ("layer", c) for c in layer_cells()}}

GROUPS = {
    "all": list(CELLS),
    "carry": [n for n, (kind, _) in CELLS.items() if kind == "carry"],
    "layer": [n for n, (kind, _) in CELLS.items() if kind == "layer"],
    "decode": [n for n, (kind, c) in CELLS.items() if kind == "layer" and c.decode_steps > 0],
}


def registry(*extra):
    """rola's cells with `extra` registry files (cells and points), as one `rola_devtools.cells.Registry`."""
    from rola_devtools.cells import Registry

    return Registry.load([*FILES, *extra])

