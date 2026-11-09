# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY AGAINST THE fp64 RECURRENCE: the inter term of a whole call, per slot, on every central oracle cell.

THE GATE IS THE TRUTH, NOT A DECLARED ARITHMETIC. The components' gates (`test_carry_readout.py`, `test_carry_fold.py`)
hold the kernel to references of its own rounding; this file holds it to `tests/oracle/reference.py`'s inter term on
the bundle's own dequantized operands, so a shared misreading of the math by a kernel and its fidelity reference has
nowhere to hide. ``num`` is compared in the recurrence's units (`readout_reference.num_in_truth`: the per-token factor
the epilogue cancels against ``den``), against each slot's envelope (the same reference on ``|v|``), at both of the
cross-CTA sum's dtypes (`rola.ops.carry.reduce_dtype_for`): f16 pairs at the den pass's scale (``_units`` stands in
for the pass: any bound on the token's output is a valid range) and fp32.

RED BY PLAN (ruling 2026-09-29): a cell whose carve the carry has no arm for, whose depth is not two, whose length is
off the window, or whose entry state the carry cannot yet take (the segment boundary is unbuilt) fails, by name. The carry's ``den`` and its state reach the oracle through the op
(`test_prefill_vs_oracle.py`): the den pass and the state's hand-off write them.

Symbols: ``W`` the window (`rola.ops.window.window_of`, the device's), ``L`` tokens, ``C`` value channels, ``u(t)`` a token's unit byte
(the f16 sum's scale).
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from measure.cells import carry_cells, realize
from measure.subjects import carry_operands
from rola.ops import carry as carry_ops
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close, relative, with_gain
from tests.oracle.fold_reference import unit_scale
from tests.oracle.readout_reference import (
    UNIT_HEADROOM,
    UNIT_LEVELS,
    UNIT_NONE,
    _read_rows,
    fold_level,
    num_in_truth,
    truth_scale,
)
from tests.oracle.tolerances import CARRY_DEN, CARRY_NUM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def _entry(spec, routes):
    """The cell's drawn entry state ``[B, H_w, N, C + 1]`` fp64 (a write head's), or ``None`` for a fresh cell."""
    entry = realize(spec).entry
    return None if entry is None else entry.double().reshape(routes.batch, routes.write_heads, *entry.shape[-2:])


def _units(spec, routes, values):
    """``[BH, windows x W]`` int8: each token's unit byte as the den pass sizes it (`readout_reference.units`' rule), from the
    TRUE state before its window -- so the gate leans on no fidelity reference, and a partial window is a window."""
    level_s = routes.level_s
    x, a = _read_rows(routes, level_s)
    read, write = routes.dequantized()
    state = reference.inter_reference(tuple(r.double() for r in read), tuple(w.double() for w in write), values.dequantized().double(), routes.topology.widths, window_of(), state_in=_entry(spec, routes), states=True)
    B, T, H, _ = x.shape
    widths, kappa = routes.topology.widths, unit_scale(routes, fold_level(routes, level_s))
    num = torch.zeros(B, T, H, dtype=torch.float64, device=x.device)
    for w, before in enumerate(state):
        before = before.repeat_interleave(routes.group, 1)                         # a query head its write head's
        top = before[..., :-1].abs().amax(-1).reshape(B, H, *widths) / kappa
        top = top if level_s == 0 else top.transpose(-1, -2)                          # [B, H, B_S, B_R]
        sl = slice(w * window_of(), (w + 1) * window_of())
        num[:, sl] = torch.einsum("bthd,bthr,bhdr->bth", x[:, sl], a[:, sl], top)
    u = torch.where(num > 0, torch.ceil(torch.log2(UNIT_HEADROOM * num.clamp_min(1e-300))) - UNIT_LEVELS,
                    torch.full_like(num, UNIT_NONE))
    #: the den pass writes the window grid whole: a partial window's tail reads nothing
    grid = torch.full((B * H, len(state) * window_of()), UNIT_NONE, dtype=torch.int8, device=x.device)
    grid[:, :T] = u.clamp(-128, 127).to(torch.int8).permute(0, 2, 1).reshape(B * H, T)
    return grid


def run(spec, reduce_dtype=None, with_den=False):
    """``(routes, values, unit, num)`` (and ``den`` where asked): one carry launch over the cell at ``reduce_dtype``
    (None: the auto rule's), the f16 sum's scale bytes a bound from the true state (:func:`_units`)."""
    routes, values, facts, carve, zero = carry_operands(spec)
    if routes.topology.D != 2:
        #: the launch first, so the carry's own refusal or fault is what the cell reports where it has one
        carry_ops.carry(routes, values, facts, carve, zero)
        torch.cuda.synchronize()
        raise NotImplementedError(f"{spec.name}: the carry's level S is defined at depth two (`carry_level_s`)")
    unit = _units(spec, routes, values)
    num, den, _, _ = carry_ops.carry(routes, values, facts, carve, unit, reduce_dtype=reduce_dtype)
    torch.cuda.synchronize()
    return (routes, values, unit, num, den) if with_den else (routes, values, unit, num)


def den_in_truth(den: torch.Tensor, routes) -> torch.Tensor:
    """``[B, T, H]`` fp64: the carry's ``den`` (each CTA's at its tokens' units already) times the per-token factor
    the epilogue cancels against ``num`` (`readout_reference.truth_scale` at unit 0)."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    got = den[:, :T].double().reshape(B, H, T).permute(0, 2, 1)
    return got * truth_scale(routes, torch.zeros((B * H, T), dtype=torch.int8, device=den.device), routes.level_s)


def ref(spec, routes, values, magnitudes=False):
    """The inter term's numerator in fp64 from the bundle's own operands and the cell's entry state; with
    ``magnitudes``, the same on ``|v|`` and ``|entry|``: each slot's envelope."""
    read, write = routes.dequantized()
    v = values.dequantized().double()
    state = _entry(spec, routes)
    if magnitudes:
        v, state = v.abs(), None if state is None else state.abs()
    num, _den, _state = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), v, routes.topology.widths, window_of(), state_in=state)
    return num


def check(spec, routes, values, unit, num):
    """``num`` in the recurrence's units, per slot against the inter term's numerator and its envelope."""
    assert_slots_close(num_in_truth(num, routes, unit), ref(spec, routes, values), output=CARRY_NUM,
                       envelope=ref(spec, routes, values, magnitudes=True), what=f"{spec.name} num")


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_numerator_is_the_fp64_reference(spec, reduce_dtype):
    """Every token and channel of ``num`` over the whole call, in the recurrence's units, against the inter term's
    numerator: the state before each window read by the window's tokens, nothing within the window -- at each of the
    cross-CTA sum's dtypes."""
    check(spec, *run(spec, reduce_dtype))


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_denominator_is_the_fp64_reference(spec):
    """Every token's ``den`` over the whole call -- the v+1 channel the carry reads out beside ``num``
    (`docs/internals/carry/ampere/handoff.md#den`) -- in the recurrence's units, against the inter term's denominator."""
    from tests.oracle.test_den_vs_oracle import refs

    routes, values, _unit, _num, den = run(spec, with_den=True)
    den_ref, _ = refs(routes, values)
    assert_slots_close(den_in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref, what=f"{spec.name} den")


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on the carry's numerator: on a real cell the kernel's output passes, and a wiped median slot
    and the smallest slots moved past their allowance fail."""
    #: two windows at least, at the card's W: a one-window call's inter term is identically zero, nothing to plant
    base = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    spec = dataclasses.replace(base, tokens=max(base.tokens, 2 * window_of()))
    routes, values, unit, num = run(spec)
    assert_planted_errors_fail(num_in_truth(num, routes, unit), ref(spec, routes, values), output=CARRY_NUM,
                               envelope=ref(spec, routes, values, magnitudes=True), what=f"{spec.name} num")


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
