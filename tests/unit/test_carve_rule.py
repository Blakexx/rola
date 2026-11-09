# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carve rule: producer v1's boxes for the layouts its rules were chosen on, each side's plan -- the fold's state
carve, the readout's value carve and its token and channel groups -- the pinned plan's check, and the refusals
(docs/internals/engine/rules.md#carve)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rola.engine import Carve, PlanOverrides
from rola.engine.carve import channel_groups, check_carve, plan_carve, token_groups
from rola.engine.rules.carve import carve as carve_rule


def _flags(level_tags):
    return tuple(t[0] == "S" for t in level_tags), tuple(t[1] == "S" for t in level_tags)


@pytest.mark.parametrize("widths, level_tags, extents, channels", [
    ((256, 256), ("SD", "DS"), (16, 32), 64),
    ((32, 32, 32, 32), ("DS", "SD", "DS", "SD"), (32, 32, 1, 1), 32),
    ((32, 32, 32, 32), ("SS", "SS", "DD", "DD"), (1, 1, 32, 32), 32),
    ((32, 32, 32, 32), ("SD", "DS", "DD", "SS"), (4, 8, 32, 1), 32),
    ((256, 256), ("SS", "DD"), (16, 32), 64),
    ((64, 64, 256), ("SD", "DS", "DD"), (4, 8, 32), 32),
    ((16, 256), ("SD", "DS"), (16, 32), 64),
    ((256,), ("DD",), (256,), 64),
])
def test_producer_v1_boxes(widths, level_tags, extents, channels):
    read, write = _flags(level_tags)
    c = plan_carve(widths, read, write, 64)
    assert (c.extents, c.channels) == (extents, channels)
    assert len(widths) == 1 or max(c.extents) <= 32
    check_carve(c, widths, read, write, 64)


@pytest.mark.parametrize("d_v, extents, channels, groups", [
    (32, (32, 32), 32, (2, 4)),
    (64, (32, 16), 64, (1, 8)),
    (128, (16, 16), 128, (1, 8)),
    (256, (16, 16), 128, (1, 8)),
])
def test_channels_toward_d_v(d_v, extents, channels, groups):
    """With no listed level the channel extent is the largest power of two dividing d_v, at most 128; the readout's
    warps are token groups x 8-channel groups."""
    c = plan_carve((256, 256), (False, False), (False, False), d_v)
    assert (c.extents, c.channels, (token_groups(c), channel_groups(c))) == (extents, channels, groups)


@pytest.mark.parametrize("d_v, extents, channels, groups", [
    (32, (32, 32), 32, (2, 4)),
    (64, (32, 16), 64, (1, 8)),
    (128, (32, 16), 64, (1, 8)),
])
def test_listed_levels_span_a_facts_list_group(d_v, extents, channels, groups):
    """Each listed level spans a list group the facts pass ranks (16 or 32 digits), each side its own, and a box with a
    listed side holds at most 64 channels: the alternating box is 32 x 32 x 32 at d_v 32 (two token groups x four
    channel groups) and 32 x 16 x 64 from d_v 64 (one token group x eight)."""
    c = plan_carve((256, 256), (False, True), (True, False), d_v)
    assert (c.extents, c.channels, (token_groups(c), channel_groups(c))) == (extents, channels, groups)


@pytest.mark.parametrize("widths", [(16, 16, 16), (16, 16, 16, 16)])
def test_a_dense_sides_row_level_takes_a_tile_first(widths):
    """Spread evenly over three or four dense levels of 16, 64 channels' 512 leaves would leave the sides' row level
    (level 0) 8 deep, under an MMA tile: it takes its 16 first, the other levels the rest."""
    c = plan_carve(widths, (False,) * len(widths), (False,) * len(widths), 64)
    assert c.channels == 64 and c.extents[0] == 16 and c.leaves == 512
    assert c.fold.tile_rows == c.readout.tile_rows == 16


def test_alternating_box_plans():
    c = plan_carve((256, 256), (False, True), (True, False), 64)
    assert (c.fold.lists, c.fold.operand, c.fold.tile_lists, c.fold.tile_rows) == ((0,), (1,), 4, 16)
    assert (c.readout.lists, c.readout.operand, c.readout.tile_lists, c.readout.tile_rows) == ((1,), (0,), 16, 32)
    assert c.readout.tile_lists * c.readout.tile_rows == c.leaves
    assert c.fold.tile_lists * c.fold.tile_rows * 8 == c.leaves
    assert c.ctas(64) == 8 * 16


def test_dense_dense_box_plans():
    c = plan_carve((32, 32, 32, 32), (True, True, False, False), (True, True, False, False), 64)
    assert (c.fold.tile_lists, c.fold.tile_rows) == (1, 128)
    assert (c.readout.tile_lists, c.readout.tile_rows) == (1, 1024)


@pytest.mark.parametrize("level_tags, fold, readout", [
    (("DD", "DD"), ((), (1, 0)), ((), (1, 0))),
    (("SD", "DD"), ((), (1, 0)), ((0,), (1,))),
    (("DS", "DD"), ((0,), (1,)), ((), (1, 0))),
    (("SD", "DS"), ((1,), (0,)), ((0,), (1,))),
])
def test_a_dense_side_groups_by_the_other_sides_list(level_tags, fold, readout):
    """A side with no list level takes its coefficients from the other side's list level (else its first level)
    (`carve.md#sides`); the readout's warps cover every leaf, the fold's an eighth."""
    read, write = _flags(level_tags)
    c = plan_carve((256, 256), read, write, 64)
    assert ((c.fold.lists, c.fold.operand), (c.readout.lists, c.readout.operand)) == (fold, readout)
    assert c.readout.tile_lists * c.readout.tile_rows == c.leaves
    assert c.fold.tile_lists * c.fold.tile_rows * 8 == c.leaves


def test_refusals():
    with pytest.raises(ValueError, match="every level sparse"):
        plan_carve((256, 256), (True, True), (True, False), 64)
    with pytest.raises(ValueError, match="power of two"):
        plan_carve((256, 200), (False, True), (True, False), 64)
    good = plan_carve((256, 256), (False, True), (True, False), 64)
    with pytest.raises(ValueError, match="plan"):
        check_carve(Carve(good.widths, good.extents, good.channels, good.readout, good.fold),
                    (256, 256), (False, True), (True, False), 64)
    with pytest.raises(ValueError, match="divid"):
        check_carve(Carve(good.widths, (32, 48), 64, good.fold, good.readout),
                    (256, 256), (False, True), (True, False), 64)
    with pytest.raises(ValueError, match="channel extent"):
        check_carve(Carve(good.widths, (32, 4), 256, good.fold, good.readout),
                    (256, 256), (False, True), (True, False), 256)


def test_rule_reads_shapes_and_takes_a_checked_pin():
    routes = SimpleNamespace(amp_read=(torch.empty(1, 1, 1, 256), torch.empty(1, 1, 1, 256)),
                             sparse=((False, True), (True, False)))
    assert carve_rule(routes, 64, None) == plan_carve((256, 256), (False, True), (True, False), 64)
    pinned = plan_carve((256, 256), (False, True), (True, False), 64)
    assert carve_rule(routes, 64, PlanOverrides(carve=pinned)) is pinned
    bad = Carve(pinned.widths, pinned.extents, pinned.channels, pinned.readout, pinned.fold)
    with pytest.raises(ValueError, match="plan"):
        carve_rule(routes, 64, PlanOverrides(carve=bad))
