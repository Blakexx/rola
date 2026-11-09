# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ROLA'S READING OF THE CENTRAL CELLS: the cells are rola-devtools' (`rola_devtools.cells`, `carry.json`): every
package's tests and measurements read one set of inputs with one draw, and a cell names nothing a package runs it with.

This module is rola's side of that line: a central cell's typed record by name (`by_name`), the carry cells and their
tiers (`carry_cells`), and the cell's draw (`realize`). What the carry kernel is given for a cell is built where it is
timed (`measure.subjects.carry_forward`). `layer.py` is the other half: RoLA's layer constructions (routing widths,
alpha, logit gain), each named with the central layer inputs it was declared for.
"""
from __future__ import annotations

from rola_devtools.cells import build, central
from rola_devtools.cells.carry import DEPOSIT_TOKEN, CarryCell, RealizedCell, realize


def by_name(name: str):
    """A central cell's typed record: a `CarryCell`, a `LayerCell` or a `QKVCell`."""
    return build(central().cell(name))


def carry_cells(*, tier: str | None = None) -> tuple[CarryCell, ...]:
    """Every central carry cell; ``tier`` keeps the cells sized for that reader (`oracle`, `probe`; `both` is either)."""
    cells = tuple(c for c in map(build, central().cells.values()) if isinstance(c, CarryCell))
    return cells if tier is None else tuple(c for c in cells if c.tier in (tier, "both"))


__all__ = ["DEPOSIT_TOKEN", "CarryCell", "RealizedCell", "by_name", "carry_cells", "realize"]
