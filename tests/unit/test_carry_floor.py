# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The hopper carry's floor model (tools/carry_floor.py): its counts are the expectations they claim, and a cluster
moves the terms it is said to move. No GPU: the rates are read from the H100 labs' committed records."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import carry_floor  # noqa: E402

ALT = dict(widths=(256, 256), read_sparse=(0, 1), write_sparse=(1, 0), k=4, box=(16, 16), dv=64, tokens=65536)


def test_a_sides_counts_are_the_expected_ones():
    #: 4 uniform digits of 16 ranges fall in 16 (1 - (15/16)^4) of them
    ranges = 16 * (1 - (15 / 16) ** 4)
    read = carry_floor.side((256, 256), (0, 1), 4, (16, 16), (16, 1))
    assert read.leaves == 1024
    assert read.boxes == pytest.approx(16 * ranges)
    assert read.clusters == pytest.approx(ranges)
    dense = carry_floor.side((256, 256), (0, 0), 4, (16, 16), (4, 4))
    assert (dense.leaves, dense.boxes, dense.clusters, dense.lists) == (65536, 256, 16, 16)


def test_a_cluster_moves_the_drains_and_the_loads_and_nothing_else():
    none = carry_floor.floor(cluster=(1, 1), **ALT)["terms"]
    rows = carry_floor.floor(cluster=(16, 1), **ALT)["terms"]
    columns = carry_floor.floor(cluster=(1, 16), **ALT)["terms"]
    for name in ("tensor, readout", "tensor, fold", "readout operand, expanded", "fold operand, expanded"):
        assert none[name] == rows[name] == columns[name]
    drains, loads = "sum: drains to global memory", "fold values, loaded"
    assert rows[drains]["cycles"][0] < columns[drains]["cycles"][0] == pytest.approx(none[drains]["cycles"][0] * 16 / 58.25, rel=0.01)
    assert columns[loads]["cycles"][0] < rows[loads]["cycles"][0] < none[loads]["cycles"][0]
    assert none["sum: partials into the cluster"]["cycles"] == [0.0, 0.0]


def test_the_two_sums_move_the_same_bytes():
    s64 = carry_floor.floor(cluster=(16, 1), form="s64", **ALT)["terms"]
    slots = carry_floor.floor(cluster=(16, 1), form="slots", **ALT)["terms"]
    name = "sum: partials into the cluster"
    assert s64[name]["cycles"] == pytest.approx(slots[name]["cycles"])


def test_a_missing_record_row_is_refused():
    with pytest.raises(SystemExit, match="has no row"):
        carry_floor.median("wgmma_rate", "a row no record holds")
