# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OP AGAINST THE fp64 ORACLE: `rola.interface.rola_op`'s output and its handed-back state, per slot.

THE REFERENCE IS `rola.ops.naive.naive_rola` DIRECTLY, on the bundle's own dequantized levels and values -- the op gate
compares on the same quantized inputs the kernels read (ruling 2026-09-26). The facts pass, the carry, the den pass,
the intra and the epilogue run as one call and the whole answer is compared against the recurrence itself, so a
window-grid disagreement, a double-counted diagonal or a dropped tile pair has nowhere to hide.

RED BY PLAN (ruling 2026-09-29): a cell outside the built envelope fails by its refusing clause -- a carve no carry
arm carries, depth three or four, a value width other than 64, a carried state. The rule's own teeth
and the decomposition's non-vacuity run no kernel.

Symbols: ``W`` the window, ``L`` tokens, ``y`` the op's output ``[B, T, H, d_v]``.
"""
from __future__ import annotations

import pytest
import torch

import rola
from measure.cells import by_name, carry_cells, realize
from measure.subjects import bundle_of
from rola.interface import rola_op
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_slots_close, oracle_run
from tests.oracle.tolerances import PREFILL_READOUT, PREFILL_STATE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: every oracle cell a fresh call can run: a carried cell's entry state reaches the op through a chained call
CELLS = tuple(c for c in carry_cells(tier="oracle") if c.state == "fresh")
IDS = [c.name for c in CELLS]


def operands(spec, tokens: slice = slice(None)):
    """``(routes, values)``: the cell's draw over ``tokens`` in the format the kernels read."""
    routes, values, _layout, _statics = bundle_of(spec, realize(spec), tokens)
    return routes, values


def oracle(routes, values, entry=None):
    """`tests.oracle.fixtures.oracle_run` on the bundle's own operands, chained from ``entry``."""
    read, write = routes.dequantized()
    return oracle_run(values.dequantized().double(), tuple(x.double() for x in read), tuple(x.double() for x in write), routes.topology, entry=entry)


@pytest.mark.parametrize("spec", CELLS, ids=IDS)
def test_the_op_matches_the_fp64_oracle(spec):
    """``y`` of a stateless call, every token and channel, against the recurrence's readout and its envelope."""
    routes, values = operands(spec)
    y, state = rola_op(routes, values)
    assert state is None
    want = oracle(routes, values)
    assert_slots_close(y, want.y, output=PREFILL_READOUT, envelope=want.y_envelope, what=f"{spec.name} y")


@pytest.mark.parametrize("spec", CELLS, ids=IDS)
def test_the_op_hands_back_the_fp64_state(spec):
    """A stateful call binds the state and hands back the recurrence's final state, leaf by leaf in canonical order
    (`RoLAState.materialize`), against the oracle's and its envelope."""
    routes, values = operands(spec)
    state = rola.state()
    rola_op(routes, values, state=state)
    want = oracle(routes, values)
    assert_slots_close(state.materialize(), want.state, output=PREFILL_STATE, envelope=want.state_envelope,
                       what=f"{spec.name} state")


def test_a_chained_call_equals_one_long_call():
    """THE STATE IS THE WHOLE CONTRACT of a chunked prefill: two calls over the halves, one state between them, give
    the long call's readout and final state, each against the oracle's one long run."""
    spec = by_name("flagship-alt-k4-L4096")
    half = spec.tokens // 2
    state = rola.state()
    ys = [rola_op(*operands(spec, part), state=state)[0] for part in (slice(0, half), slice(half, spec.tokens))]
    routes, values = operands(spec)
    want = oracle(routes, values)
    assert_slots_close(torch.cat(ys, dim=1), want.y, output=PREFILL_READOUT, envelope=want.y_envelope,
                       what="chained y")
    assert_slots_close(state.materialize(), want.state, output=PREFILL_STATE, envelope=want.state_envelope,
                       what="chained state")


def test_the_per_slot_rule_fails_what_a_global_max_passes():
    """The rule the cells are graded in can fail (docs/testing.md): a small slot moved by a hundredth of the readout's
    largest magnitude passes a global max-relative test at 2% and fails per slot."""
    routes, values = operands(by_name("flagship-alt-k4"))
    want = oracle(routes, values)
    planted = want.y.clone()
    i = int(torch.argmin(want.y.abs().flatten()))
    planted.view(-1)[i] += 0.01 * float(want.y.abs().max())
    assert float((planted - want.y).abs().max()) <= 0.02 * float(want.y.abs().max())
    with pytest.raises(AssertionError, match="off the oracle"):
        assert_slots_close(planted, want.y, output=PREFILL_READOUT, envelope=want.y_envelope, what="planted")


def test_the_intra_term_is_actually_present():
    """NON-VACUITY: the inter term alone fails the op's gate on a multi-window cell, so a call that dropped the intra
    (or the carry) could not pass. Runs no kernel."""
    routes, values = operands(by_name("flagship-alt-k4"))
    read, write = routes.dequantized()
    num, den, _ = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), values.dequantized().double(), routes.topology.widths, window_of())
    want = oracle(routes, values)
    with pytest.raises(AssertionError, match="off the oracle"):
        assert_slots_close(num / (den[..., None] + 1e-5), want.y, output=PREFILL_READOUT, envelope=want.y_envelope,
                           what="inter alone")


def test_the_op_refuses_what_it_has_no_kernel_for():
    """THE REFUSAL IS STRUCTURAL: a decay, which the prefill envelope does not carry, is refused by name before any
    kernel runs."""
    from rola.routing.types import LeafMassDecay

    routes, values = operands(by_name("flagship-alt-k4"))
    dials = tuple(torch.full((routes.write_heads, w), 0.5, device="cuda") for w in routes.topology.widths)
    with pytest.raises((NotImplementedError, ValueError), match="decay"):
        rola_op(routes, values, LeafMassDecay(dials=dials))
