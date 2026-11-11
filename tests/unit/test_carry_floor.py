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
    rows = carry_floor.floor(cluster=(16, 1), form="slots", **ALT)["terms"]
    columns = carry_floor.floor(cluster=(1, 16), form="slots", **ALT)["terms"]
    for name in ("tensor, readout", "tensor, fold", "readout operand, expanded", "fold operand, expanded"):
        assert none[name] == rows[name] == columns[name]
    drains, loads = "sum: drains to global memory", "fold values, loaded"
    assert rows[drains]["cycles"][0] < columns[drains]["cycles"][0] == pytest.approx(none[drains]["cycles"][0] * 16 / 58.25, rel=0.01)
    assert columns[loads]["cycles"][0] < rows[loads]["cycles"][0] < none[loads]["cycles"][0]
    #: a cluster's shared load is an estimate from free to worthless: its high end is every box's own load
    assert columns[loads]["cycles"][1] == rows[loads]["cycles"][1] == none[loads]["cycles"][1]
    assert none["sum: partials into the cluster"]["cycles"] == [0.0, 0.0]


def test_the_cluster_forms_read_the_labs_line():
    hop, rounds = "sum: partials into the cluster", "sum: the rounds' barriers"
    slots = carry_floor.floor(cluster=(16, 1), form="slots", **ALT)["terms"]
    s64 = carry_floor.floor(cluster=(16, 1), form="s64", **ALT)["terms"]
    longer = carry_floor.floor(cluster=(16, 1), form="slots", round_partials=512, **ALT)["terms"]
    assert slots[hop]["rate"] == "hop_slots" and s64[hop]["cycles"][0] > slots[hop]["cycles"][0] > 0.0
    assert longer[hop] == slots[hop]
    assert longer[rounds]["cycles"][0] == pytest.approx(slots[rounds]["cycles"][0] / 8)


def test_a_cluster_with_no_sum_shares_the_loads_and_every_box_drains():
    none = carry_floor.floor(cluster=(1, 1), **ALT)["terms"]
    shared = carry_floor.floor(cluster=(1, 16), **ALT)["terms"]
    drains, loads = "sum: drains to global memory", "fold values, loaded"
    assert shared[drains] == none[drains] and shared["sum: the rounds' barriers"]["cycles"] == [0.0, 0.0]
    assert shared[loads]["cycles"][0] < none[loads]["cycles"][0]


def test_a_missing_record_row_is_refused():
    with pytest.raises(SystemExit, match="has no row"):
        carry_floor.median("wgmma_rate", "a row no record holds")
