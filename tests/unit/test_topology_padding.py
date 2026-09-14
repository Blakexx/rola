# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""PADDING AT THE PRODUCER: the descriptor's logical-vs-padded split.

No CUDA/GPU anywhere in this file: `rola.routing.topology.padded_level_width` and
`Topology.padded_widths`/`.padded()` are pure Python/dataclass derivations, and
`rola.ops.padding`'s helpers are plain `torch` tensor ops that run identically on
CPU. `tests/oracle/test_padding_seam.py` is the GPU/kernel-boundary half.

Symbols, each at first use: `b_l` a level's LOGICAL width (any int the caller
authored); `B_l` its PADDED, kernel-facing width (the next power of two at or above
16); `d_v` the caller's logical value width; `DV` an op's padded, shipped one.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.padding import PaddedShape, pad_routes, pad_v, padded_value_width, slice_y
from rola.routing.topology import PADDED_LEVEL_FLOOR, padded_level_width
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology


def test_the_floor_matches_the_states_own_floor():
    """`rola._state.MIN_LEVEL_WIDTH` and `padded_level_width`'s floor are ONE number,
    stated in two files -- this pins them equal so the two never drift apart."""
    from rola._state import MIN_LEVEL_WIDTH

    assert PADDED_LEVEL_FLOOR == MIN_LEVEL_WIDTH == 16


@pytest.mark.parametrize("width,padded", [
    (1, 16), (2, 16), (15, 16), (16, 16), (17, 32), (31, 32), (32, 32),
    (33, 64), (45, 64), (256, 256),
])
def test_padded_level_width_is_the_next_power_of_two_at_or_above_the_floor(width, padded):
    assert padded_level_width(width) == padded


def test_padded_level_width_refuses_a_non_positive_or_inexact_width():
    with pytest.raises(ValueError):
        padded_level_width(0)
    with pytest.raises(ValueError):
        padded_level_width(-3)


def _levels(widths):
    return tuple(IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation())
                for w in widths)


def test_topology_widths_stays_logical_forever():
    """`Topology.widths` is the API surface (`tests/unit/test_api_contract.py`'s
    pinned shape/view contract): padding never touches it."""
    topology = Topology(levels=_levels((31, 45)))
    assert topology.widths == (31, 45)


@pytest.mark.parametrize("widths,padded", [
    ((31, 45), (32, 64)),
    ((17, 16, 33), (32, 16, 64)),
    ((256, 256), (256, 256)),
    ((8, 8, 8, 8), (16, 16, 16, 16)),
])
def test_topology_padded_widths_is_per_level_padded_level_width(widths, padded):
    topology = Topology(levels=_levels(widths))
    assert topology.padded_widths == padded


def test_topology_is_padded_predicate():
    assert Topology(levels=_levels((256, 256))).is_padded
    assert not Topology(levels=_levels((31, 45))).is_padded


def test_topology_padded_returns_an_equivalent_topology_at_the_padded_widths():
    topology = Topology(levels=_levels((31, 45)))
    padded = topology.padded()
    assert padded.widths == (32, 64)
    #: `.at(width)` keeps the operator/activation choice, so the padded Topology
    #: describes the SAME model, never a different one.
    assert type(padded.levels[0]) is type(topology.levels[0])
    assert padded.levels[0].read == topology.levels[0].read
    assert padded.levels[0].write == topology.levels[0].write


def test_topology_padded_is_idempotent_on_an_already_padded_topology():
    topology = Topology(levels=_levels((256, 256)))
    assert topology.padded().widths == topology.widths


# -- rola.ops.padding: the op-level translator ---------------------------------------

def test_padded_shape_of_derives_every_field_from_widths_and_d_v():
    shape = PaddedShape.of((31, 45), 48, shipped_dv=(32, 64, 128))
    assert shape == PaddedShape(widths=(31, 45), padded_widths=(32, 64), d_v=48, DV=64)
    assert not shape.is_padded


def test_padded_shape_is_padded_when_nothing_needs_padding():
    shape = PaddedShape.of((32, 64), 64, shipped_dv=(32, 64, 128))
    assert shape.is_padded


def test_padded_value_width_picks_the_smallest_shipped_width_at_or_above_d_v():
    assert padded_value_width(1, shipped=(32, 64, 128)) == 32
    assert padded_value_width(32, shipped=(32, 64, 128)) == 32
    assert padded_value_width(33, shipped=(32, 64, 128)) == 64
    assert padded_value_width(128, shipped=(32, 64, 128)) == 128


def test_padded_value_width_refuses_past_the_largest_shipped_width():
    with pytest.raises(ValueError, match="exceeds every shipped"):
        padded_value_width(129, shipped=(32, 64, 128))


def test_pad_routes_pads_each_level_to_its_own_padded_width_with_exact_zero():
    levels = (torch.rand(2, 3, 4, 31, dtype=torch.float64),
              torch.rand(2, 3, 4, 45, dtype=torch.float64))
    padded = pad_routes(levels, (32, 64))
    assert padded[0].shape[-1] == 32 and padded[1].shape[-1] == 64
    assert torch.equal(padded[0][..., :31], levels[0])
    assert torch.equal(padded[1][..., :45], levels[1])
    assert (padded[0][..., 31:] == 0).all()
    assert (padded[1][..., 45:] == 0).all()


def test_pad_routes_is_a_no_op_when_a_level_is_already_at_its_padded_width():
    level = torch.rand(2, 3, 4, 32, dtype=torch.float64)
    (padded,) = pad_routes((level,), (32,))
    assert padded is level


def test_pad_v_and_slice_y_round_trip_exactly():
    v = torch.randn(2, 3, 4, 48, dtype=torch.float64)
    padded = pad_v(v, 64)
    assert padded.shape[-1] == 64
    assert torch.equal(padded[..., :48], v)
    assert (padded[..., 48:] == 0).all()
    assert torch.equal(slice_y(padded, 48), v)


def test_pad_refuses_to_pad_downward():
    with pytest.raises(ValueError, match="only ever widens"):
        pad_v(torch.zeros(2, 64), 48)
