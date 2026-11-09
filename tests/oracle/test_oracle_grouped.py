# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""GROUPED QUERIES IN THE ORACLE: H query heads read H_w write heads, and that is the ungrouped layer with each write
head's side repeated for its query heads -- THE IDENTITY GATE every grouped kernel is held to.

Query head h reads write head h // (H / H_w) (consecutive). A grouped run equals the ungrouped run whose write side,
values, gain, decay dials and initial state are `repeat_interleave`d H / H_w times, per slot, and the final state is
the write heads' (each query head's ungrouped state its write head's). The gate must be able to fail: the other
assignment (h reads h % H_w) moves the output. No device: the oracle is fp64 torch.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.types import LeafMassDecay
from tests.oracle.fixtures import with_gain
from tests.oracle.oracle_fixtures import _decay_dials, _simplex, _topology

_B, _T, _DV = 2, 12, 5
_RTOL, _ATOL = 1e-12, 1e-14


def _draw(widths, H, H_w, decayed, seed):
    gen = torch.Generator().manual_seed(seed)
    read = tuple(_simplex((_B, _T, H, w), 0.5, gen, "cpu") for w in widths)
    write = tuple(_simplex((_B, _T, H_w, w), 1.0, gen, "cpu") for w in widths)
    v = torch.randn((_B, _T, H_w, _DV), generator=gen, dtype=torch.float64)
    g_write = 0.5 + torch.rand((_B, _T, H_w), generator=gen, dtype=torch.float64)
    N = 1
    for w in widths:
        N *= w
    state = torch.rand((_B, H_w, N, _DV + 1), generator=gen, dtype=torch.float64)
    dials = _decay_dials(widths, H_w, gen, "cpu") if decayed else None
    return read, write, v, g_write, state, dials


def _run(widths, read, write, v, g_write, state, dials):
    decay = None if dials is None else LeafMassDecay(dials=dials)
    return naive_rola(v, read, with_gain(write, g_write), _topology(widths), decay, initial_state=state, output_final_state=True)


def _spread(x, g, dim, consecutive=True):
    """Each write head's tensor for its query heads: `repeat_interleave` (h reads h // g), or the other order."""
    if consecutive:
        return x.repeat_interleave(g, dim=dim)
    reps = [1] * x.ndim
    reps[dim] = g
    return x.repeat(*reps)


def _ungrouped(widths, read, write, v, g_write, state, dials, g, consecutive=True):
    def spread(x, dim):
        return _spread(x, g, dim, consecutive)

    return _run(widths, read, tuple(spread(w, 2) for w in write), spread(v, 2), spread(g_write, 2),
                spread(state, 1), None if dials is None else tuple(spread(d, 0) for d in dials))


@pytest.mark.parametrize("widths", [(4, 4), (3, 5)])
@pytest.mark.parametrize("H_w,g", [(1, 2), (2, 2), (1, 3), (2, 4)])
@pytest.mark.parametrize("decayed", [False, True])
def test_a_grouped_run_is_the_ungrouped_run_with_the_write_side_repeated(widths, H_w, g, decayed):
    draw = _draw(widths, H_w * g, H_w, decayed, seed=17 + 7 * g + H_w)
    y, final = _run(widths, *draw)
    y_ref, final_ref = _ungrouped(widths, *draw, g)
    assert y.shape == (_B, _T, H_w * g, _DV) and final.shape == (_B, H_w, final_ref.shape[2], _DV + 1)
    torch.testing.assert_close(y, y_ref, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(final.repeat_interleave(g, dim=1), final_ref, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("widths", [(4, 4)])
def test_the_gate_fails_on_the_other_head_assignment(widths):
    H_w, g = 2, 2
    draw = _draw(widths, H_w * g, H_w, True, seed=5)
    y, _ = _run(widths, *draw)
    y_other, _ = _ungrouped(widths, *draw, g, consecutive=False)
    assert not torch.allclose(y, y_other, rtol=_RTOL, atol=_ATOL)


def test_the_write_heads_must_divide_the_query_heads():
    read, write, v, g_write, state, _ = _draw((4, 4), 3, 2, False, seed=3)
    with pytest.raises(ValueError, match="must divide the query heads"):
        _run((4, 4), read, write, v, g_write, state, None)


def test_the_decay_is_the_write_heads():
    read, write, v, g_write, state, _ = _draw((4, 4), 4, 2, False, seed=4)
    dials = _decay_dials((4, 4), 4, torch.Generator().manual_seed(4), "cpu")
    with pytest.raises(ValueError, match="write heads H_w=2"):
        _run((4, 4), read, write, v, g_write, state, dials)
