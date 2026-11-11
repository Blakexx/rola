# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STATE'S BACKING IS INVISIBLE AND A CONTINUATION IS ONE CALL, at the op (`rola.interface.rola_op`).

The paged arena is the default backing and `PlanOverrides(paging=False)` pins the dense plane (docs/api.md section 5):
the two give the same readout and the same state, bit for bit, on a fresh call and on a continuation. A stateless call
hands no state back. A call whose writes reach a strict subset of the leaves a previous call wrote leaves every other
leaf exactly as it found it -- an idle resident page is neither loaded nor stored.

RED BY PLAN (ruling 2026-09-29) until the prefill composition and the carry's segment boundary are ported: every cell
fails by name.
"""
from __future__ import annotations

import pytest
import torch

import rola
from measure.cells import by_name, realize
from measure.subjects import bundle_of
from rola.expert import PlanOverrides
from rola.interface import rola_op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: cells with sparse, structured and dense support: every page touched, some pages never touched, a split read side
CELLS = ("flagship-alt-k4", "flagship-cohort-k4", "flagship-struct25", "flat-small-multiwindow")
DENSE = PlanOverrides(paging=False)


def operands(spec, tokens: slice = slice(None)):
    routes, values, _layout, _statics = bundle_of(spec, realize(spec), tokens)
    return routes, values


def calls(spec, parts, expert=None):
    """``(ys, state)``: the op over ``parts`` of the cell's tokens, one state carried through them."""
    state = rola.state()
    ys = [rola_op(*operands(spec, part), state=state, expert=expert)[0] for part in parts]
    return ys, state


@pytest.mark.parametrize("name", CELLS)
def test_the_paged_backing_is_bit_identical_to_the_dense_one(name):
    spec = by_name(name)
    (y_paged,), paged = calls(spec, [slice(None)])
    (y_dense,), dense = calls(spec, [slice(None)], DENSE)
    assert paged.paged and not dense.paged
    assert torch.equal(y_paged, y_dense)
    assert torch.equal(paged.materialize(), dense.materialize())


@pytest.mark.parametrize("name", ("flagship-alt-k4-L4096", "flagship-cohort-k4-L4096"))
def test_a_continuation_is_bit_identical_across_the_two_backings(name):
    """The cell run as two calls through one state, once paged and once dense: the same readouts and the same
    state."""
    spec = by_name(name)
    parts = [slice(0, spec.tokens // 2), slice(spec.tokens // 2, spec.tokens)]
    ys_paged, paged = calls(spec, parts)
    ys_dense, dense = calls(spec, parts, DENSE)
    for got, want in zip(ys_paged, ys_dense):
        assert torch.equal(got, want)
    assert torch.equal(paged.materialize(), dense.materialize())


def test_a_stateless_call_carries_no_state():
    """The predicate is "a state was passed": a stateless call hands back ``None``."""
    _y, state = rola_op(*operands(by_name("flagship-alt-k4")))
    assert state is None


def test_an_idle_resident_page_is_neither_loaded_nor_stored():
    """The second call's writes reach a strict subset of the leaves the first call's wrote: every leaf outside that
    subset is bit for bit what the first call left, on both backings. A cohort draw (tokens of a cohort share their
    sparse write digits) leaves most leaves unwritten by a window; a dense write level is worded whole and pages every leaf."""
    spec = by_name("flagship-cohort-k4")
    half = spec.tokens // 2
    first, second = operands(spec, slice(0, half)), operands(spec, slice(half, spec.tokens))
    widths = second[0].topology.widths
    touched = None
    for level, live in enumerate(second[0].live("write")):
        digits = live.any(dim=(0, 1, 2))
        shape = [1] * len(widths)
        shape[level] = widths[level]
        touched = digits.view(shape) if touched is None else touched & digits.view(shape)
    untouched = ~touched.reshape(-1)
    assert bool(untouched.any()), "the cell's second half writes every leaf: the claim is vacuous"
    for expert in (None, DENSE):
        state = rola.state()
        rola_op(*first, state=state, expert=expert)
        before = state.materialize().clone()
        rola_op(*second, state=state, expert=expert)
        after = state.materialize()
        assert torch.equal(after[..., untouched, :], before[..., untouched, :])
