# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY'S DEN AND THE SCALE PASS'S BYTES AS THE PREFILL TAKES THEM, AGAINST THE fp64 RECURRENCE: each token's
window-aligned denominator and its scale byte's bound over a whole call, per slot, on every central oracle cell.

``den`` is the inter term's denominator (`tests/oracle/reference.py`'s ``inter_reference`` on the bundle's own
dequantized operands, the state before each window read by the window's tokens): the carry's v+1 channel times the
carry's factor, the prefill's units (`rola.ops.prefill.prefill`). The BOUND is the same product against the mass of
``F = m s`` -- the recurrence run with each writing token's value scale as its one value channel -- which the scale
pass's byte must cover: ``2^(byte + 15) / 1.02`` times the carry's factor at least that bound over the reader's factor.
Both leave the reader's per-token scales out (the kernels' contract); :func:`in_truth` applies them.
RED BY PLAN: a cell the carry or the scale pass does not serve, or whose entry state they cannot take, fails by name.
Symbols: ``W`` the window, ``L`` tokens, ``g`` query heads a write head.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from measure.cells import carry_cells
from measure.subjects import carry_operands
from rola.ops import carry as carry_ops
from rola.ops import scale as scale_ops
from rola.ops.prefill import carry_factor, reader_factor
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close
from tests.oracle.scale_reference import UNIT_HEADROOM, UNIT_LEVELS
from tests.oracle.tolerances import CARRY_DEN

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def run(spec):
    """``(routes, values, den, bytes)``: the scale pass and one carry launch at its bytes over the cell, ``den`` in the
    prefill's units (the carry's v+1 channel times its factor)."""
    routes, values, facts, carve, _zero = carry_operands(spec)
    if spec.state != "fresh":
        raise NotImplementedError(f"{spec.name}: the carry takes no entry state yet")
    factor = carry_factor(routes)
    scale = scale_ops.scale(routes, facts, factor)
    _num, den, _, _ = carry_ops.carry(routes, values, facts, carve, scale)
    torch.cuda.synchronize()
    return routes, values, den[:, :routes.tokens] * factor, scale


def in_truth(x: torch.Tensor, routes) -> torch.Tensor:
    """``[B, T, H]`` fp64: a ``[BH g, L]`` plane over the reader's factor in the recurrence's units."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    return (x.double() * reader_factor(routes).double()).reshape(B, H, T).permute(0, 2, 1)


def refs(routes, values) -> tuple[torch.Tensor, torch.Tensor]:
    """``(den, num)`` ``[B, T, H]`` fp64: the inter term's denominator and the same recurrence with each token's value
    scale as its one value channel."""
    read, write = routes.dequantized()
    num, den, _ = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), values.scale.double().unsqueeze(-1), routes.topology.widths, window_of())
    return den, num[..., 0]


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_denominator_is_the_fp64_reference(spec):
    """Every token's ``den`` over the whole call, in the recurrence's units, against the inter term's denominator."""
    routes, values, den, _bytes = run(spec)
    den_ref, _ = refs(routes, values)
    assert_slots_close(in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref, what=f"{spec.name} den")


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_bound_is_the_fp64_reference(spec):
    """Every token's scale byte covers the read product against the mass of ``F`` -- the recurrence on the value
    scales, every channel's bound: ``2^(byte + 15) / 1.02`` in the recurrence's units at least it (within fp32's sums),
    and no byte where it is zero."""
    routes, _values, _den, scale = run(spec)
    scale = scale[:, :routes.tokens]
    _, num_ref = refs(routes, _values)
    unit = carry_factor(routes) * torch.exp2(scale.double() + UNIT_LEVELS) / UNIT_HEADROOM
    cover = in_truth(unit, routes)
    live = num_ref.abs() > 0
    held = torch.where(live, cover * (1 + 1e-5) >= num_ref.abs(), torch.ones_like(live))
    assert held.all(), f"{spec.name}: {int((~held).sum())} of {int(live.sum())} tokens above their byte's bound"


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on the den: on a real cell the kernel's output passes, and a wiped median slot and the
    smallest slots moved past their allowance fail."""
    #: two windows at least, at the card's W: a one-window call's inter term is identically zero, nothing to plant
    base = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    spec = dataclasses.replace(base, tokens=max(base.tokens, 2 * window_of()))
    routes, values, den, _num = run(spec)
    den_ref, _ = refs(routes, values)
    assert_planted_errors_fail(in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref,
                               what=f"{spec.name} den")
