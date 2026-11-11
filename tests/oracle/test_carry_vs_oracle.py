# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY AGAINST THE fp64 RECURRENCE: the inter term of a whole call, per slot, on every central oracle cell.

THE GATE IS THE TRUTH, NOT A DECLARED ARITHMETIC: this file holds the kernel to `tests/oracle/reference.py`'s inter
term on the bundle's own dequantized operands, so a shared misreading of the math by a kernel and a fidelity reference
has nowhere to hide. ``num`` and ``den`` are compared in the recurrence's units -- the kernel's fp32 sums leave each
token's reader factor ``r`` out (`rola.ops.prefill`), so each is multiplied by it (:func:`in_truth`) -- against each
slot's envelope (the same reference on ``|v|``).

RED BY PLAN: no carry kernel is built on this line, so every launch here fails by name
(`rola.ops._ext.KernelNotBuilt`) until a hopper schedule lands; the decomposition test runs no kernel and holds.

Symbols: ``W`` the window (`rola.ops.window.window_of`, the device's), ``L`` tokens, ``C`` value channels.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from measure.cells import carry_cells, realize
from measure.subjects import bundle_of, cell_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.prefill import reader_factor
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close, relative, with_gain
from tests.oracle.tolerances import CARRY_DEN, CARRY_NUM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def _entry(spec, routes):
    """The cell's drawn entry state ``[B, H_w, N, C + 1]`` fp64 (a write head's), or ``None`` for a fresh cell."""
    entry = realize(spec).entry
    return None if entry is None else entry.double().reshape(routes.batch, routes.write_heads, *entry.shape[-2:])


def run(spec):
    """``(routes, values, num, den)``: one carry launch over the cell, from its bundle, the facts pass's lists in the
    form of the arm its carve selects, and the carve."""
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    carve = cell_carve(spec)
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.lists_form(carry_ops.arm_key(carve)))
    num, den = carry_ops.carry(routes, values, facts, carve)[:2]
    torch.cuda.synchronize()
    return routes, values, num, den


def in_truth(x: torch.Tensor, routes) -> torch.Tensor:
    """The carry's ``num`` ``[BH, L, C]`` or ``den`` ``[BH, L]`` in the recurrence's units, ``[B, T, H(, C)]`` fp64:
    times each token's reader factor, which the kernels leave out of both sums."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    r = reader_factor(routes).double()
    got = x[:, :T].double() * (r if x.dim() == 2 else r.unsqueeze(-1))
    return got.reshape(B, H, T, *x.shape[2:]).transpose(1, 2)


def ref(spec, routes, values, magnitudes=False):
    """The inter term's ``(num, den)`` in fp64 from the bundle's own operands and the cell's entry state; with
    ``magnitudes``, the same on ``|v|`` and ``|entry|``: each slot's envelope."""
    read, write = routes.dequantized()
    v = values.dequantized().double()
    state = _entry(spec, routes)
    if magnitudes:
        v, state = v.abs(), None if state is None else state.abs()
    num, den, _state = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), v,
                                                 routes.topology.widths, window_of(), state_in=state)
    return num, den


def check(spec, routes, values, num, _den):
    """``num`` in the recurrence's units, per slot against the inter term's numerator and its envelope."""
    assert_slots_close(in_truth(num, routes), ref(spec, routes, values)[0], output=CARRY_NUM,
                       envelope=ref(spec, routes, values, magnitudes=True)[0], what=f"{spec.name} num")


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_numerator_is_the_fp64_reference(spec):
    """Every token and channel of ``num`` over the whole call, in the recurrence's units, against the inter term's
    numerator: the state before each window read by the window's tokens, nothing within the window."""
    check(spec, *run(spec))


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_denominator_is_the_fp64_reference(spec):
    """Every token's ``den`` over the whole call, in the recurrence's units, against the inter term's denominator."""
    routes, values, _num, den = run(spec)
    den_ref = ref(spec, routes, values)[1]
    assert_slots_close(in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref, what=f"{spec.name} den")


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on the carry's numerator: on a real cell the kernel's output passes, and a wiped median slot
    and the smallest slots moved past their allowance fail."""
    #: two windows at least, at the card's W: a one-window call's inter term is identically zero, nothing to plant
    base = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    spec = dataclasses.replace(base, tokens=max(base.tokens, 2 * window_of()))
    routes, values, num, _den = run(spec)
    assert_planted_errors_fail(in_truth(num, routes), ref(spec, routes, values)[0], output=CARRY_NUM,
                               envelope=ref(spec, routes, values, magnitudes=True)[0], what=f"{spec.name} num")


def test_the_decomposition_is_exact():
    """``inter(W) + intra(W)`` is the SAME full recurrence for every ``W``, and equals the canonical fp64 oracle: the
    no-double-count claim, and the reason an inter-only kernel can be gated at all. It runs no kernel."""
    from rola.ops.constants import READOUT_EPS
    from rola.ops.naive import naive_rola
    from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology

    spec = next(c for c in ORACLE_CELLS if c.name == "flat-small-dense")
    read, write, gain, v = realize(spec).doubles()
    tot = None
    for W in (1, 8, 64):
        ni, di, _ = reference.inter_reference(read, with_gain(write, gain), v, spec.widths, W)
        na, da = reference.intra_reference(read, with_gain(write, gain), v, spec.widths, W)
        cur = (ni + na, di + da)
        if tot is None:
            tot = cur
        else:
            assert relative(cur[0], tot[0]) < 1e-12
            assert relative(cur[1], tot[1]) < 1e-12
    routing = IndependentRouting(width=1, read=SoftmaxActivation(), write=SoftmaxActivation())
    topology = Topology(levels=tuple(routing.at(w) for w in spec.widths))
    y, _ = naive_rola(v, read, with_gain(write, gain), topology, None, output_final_state=False)
    assert relative(tot[0] / (tot[1][..., None] + READOUT_EPS), y) < 1e-10
