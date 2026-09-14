# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""R4's decomposition gate: the reverse pass splits on the forward's window edge.

The mirror of ``test_carry_vs_oracle.py::test_m2_decomposition_is_exact``. The
CARRY half plus the INTRA half of every cotangent equals ``torch.autograd``
through the whole operator, in fp64 -- so a window-grid disagreement, a
double-counted diagonal or a dropped tile pair in the GRADIENT has nowhere to hide
either.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.constants import READOUT_EPS
from tests.oracle.backward_reference import (
    carry_backward_reference,
    intra_backward_reference,
    readout_seeds,
)
from tests.oracle.fixtures import simplex
from tests.oracle.reference import inter_reference, intra_reference

#: `L % W == 0` is the combined operator's own refusal, so the cells honor it.
CELLS = [
    #: (widths, T, W, d_v, k_tok, gain_is_one)
    ((16, 16), 32, 8, 4, None, True),
    ((16, 16), 32, 8, 4, None, False),
    ((16, 16), 24, 8, 4, 3, False),
    ((8, 8, 8), 16, 8, 4, None, False),
    ((8, 8, 8), 16, 8, 4, 2, False),
]


def _draw(widths, T, d_v, k_tok, gain_is_one, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (1, T, 1)
    read = tuple(simplex(shape + (w,), k_tok, gen, device) for w in widths)
    write = tuple(simplex(shape + (w,), k_tok, gen, device) for w in widths)
    if gain_is_one:
        gw = torch.ones(*shape, device=device, dtype=torch.float64)
    else:
        gw = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(*shape, d_v, device=device, dtype=torch.float64, generator=gen)
    return read, write, gw, v, gen


def _combined(read, write, gw, v, widths, window, state_in):
    num, den, state = inter_reference(read, write, gw, v, widths, window, state_in)
    n_i, d_i = intra_reference(read, write, gw, v, widths, window)
    return num + n_i, den + d_i, state


@pytest.mark.parametrize("widths,T,window,d_v,k_tok,gain_is_one", CELLS)
@pytest.mark.parametrize("chained", [False, True])
def test_backward_decomposition_is_exact(widths, T, window, d_v, k_tok, gain_is_one, chained):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    read, write, gw, v, gen = _draw(widths, T, d_v, k_tok, gain_is_one, 7, device)
    N = 1
    for w in widths:
        N *= w
    state_in = (0.1 * torch.randn(1, 1, N, d_v + 1, device=device, dtype=torch.float64,
                                 generator=gen)) if chained else None

    leaves = [t.clone().requires_grad_(True) for t in read + write]
    gwv = gw.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    sv = None if state_in is None else state_in.clone().requires_grad_(True)
    D = len(widths)
    num, den, state = _combined(tuple(leaves[:D]), tuple(leaves[D:]), gwv, vv, widths,
                                window, sv)

    d_num = torch.randn_like(num)
    d_den = torch.randn_like(den)
    d_state_out = torch.randn_like(state)
    inputs = leaves + [gwv, vv] + ([sv] if sv is not None else [])
    got = torch.autograd.grad((num, den, state), inputs, (d_num, d_den, d_state_out))

    c_read, c_write, c_gw, c_v, c_sin = carry_backward_reference(
        read, write, gw, v, widths, window, d_num, d_den, state_in, d_state_out)
    i_read, i_write, i_gw, i_v = intra_backward_reference(
        read, write, gw, v, widths, window, d_num, d_den)

    want = ([c_read[l] + i_read[l] for l in range(D)]
            + [c_write[l] + i_write[l] for l in range(D)]
            + [c_gw + i_gw, c_v + i_v] + ([c_sin] if sv is not None else []))
    names = ([f"d_read[{l}]" for l in range(D)] + [f"d_write[{l}]" for l in range(D)]
             + ["d_gw", "d_v"] + (["d_state_in"] if sv is not None else []))
    for name, a, b in zip(names, got, want):
        scale = max(1e-30, float(b.abs().max()))
        err = float((a - b).abs().max()) / scale
        assert err < 1e-12, f"{name}: relative {err:.3e}"


def test_readout_seeds_match_autograd_through_the_divide():
    """The seam's own formulas, against autograd through ``num/(den+eps)``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(3)
    num = torch.randn(2, 5, 3, 4, device=device, dtype=torch.float64, generator=gen)
    den = torch.rand(2, 5, 3, device=device, dtype=torch.float64, generator=gen)
    dy = torch.randn_like(num)
    n = num.clone().requires_grad_(True)
    d = den.clone().requires_grad_(True)
    y = n / (d[..., None] + READOUT_EPS)
    g_num, g_den = torch.autograd.grad(y, (n, d), dy)
    s_num, s_den = readout_seeds(num, den, dy, READOUT_EPS)
    assert float((g_num - s_num).abs().max()) < 1e-14
    assert float((g_den - s_den).abs().max()) / max(1e-30, float(g_den.abs().max())) < 1e-12
