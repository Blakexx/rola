# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""rola's reading of the central cells (`benchmarks/cells`): the window the central cells state their tail and flip
against is this kernel's, the mode word packs the sparsity a cell declares, and a layer construction names only central
inputs. No GPU: nothing here draws a cell."""
from __future__ import annotations

from rola_devtools.cells import central
from rola_devtools.cells.carry import WINDOW
from rola_devtools.cells.layer import LayerCell

from benchmarks.cells import by_name, carry_cells, level_modes
from benchmarks.cells.layer import CONSTRUCTIONS
from rola.ops import carry as carry_ops


def test_the_central_window_is_the_carry_kernels():
    assert WINDOW == carry_ops.WINDOW


def test_the_mode_word_is_the_declared_sparsity():
    read, write = carry_ops.SIDE_SPARSE
    assert level_modes(by_name("deep3-alt-k4")) == write | (read << 2) | (write << 4)
    assert level_modes(by_name("flagship-both-k4")) == read | write
    assert level_modes(by_name("flagship-dense")) == level_modes(by_name("corner-tied-k4")) == 0


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
