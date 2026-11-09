# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""rola's reading of the central cells (`measure/cells`): the window the central cells state their tail and flip
against is this kernel's, and a layer construction names only central inputs. No GPU: nothing here draws a cell."""
from __future__ import annotations

import pytest
import torch
from rola_devtools.cells import central
from rola_devtools.cells.carry import WINDOW
from rola_devtools.cells.layer import LayerCell

from measure.cells import by_name, carry_cells
from measure.cells.layer import CONSTRUCTIONS
from rola.ops.window import window_of


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device's window is the binary's table at its card")
def test_the_device_window_is_a_whole_number_of_the_central_window():
    """The central cells state their tail and flip against a 512-token window; every card's window is a whole number
    of it (at W = 1,024 a flip cell's two halves share one kernel window)."""
    assert window_of() % WINDOW == 0


def test_every_construction_is_declared_for_central_layer_inputs_and_every_input_has_one():
    inputs = {name for name, record in central().cells.items() if record["data"].endswith(":layer_cell")}
    for construction in CONSTRUCTIONS.values():
        assert set(construction.cells) <= inputs, construction.name
        assert all(isinstance(by_name(c), LayerCell) for c in construction.cells)
    assert inputs == {c for construction in CONSTRUCTIONS.values() for c in construction.cells}


def test_the_oracle_tier_is_a_selection_of_the_central_carry_cells():
    oracle = carry_cells(tier="oracle")
    assert oracle and all(c.tier in ("oracle", "both") for c in oracle)
    assert set(oracle) <= set(carry_cells())
