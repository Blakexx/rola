# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference for K31 the INTER term, and the decomposition that proves it.

TWO INDEPENDENT ROUTES TO THE SAME NUMBER, and both are used:

1. :func:`inter_reference` runs the windowed recurrence directly -- per window,
   read against state-at-window-start, then fold that window's writes. This is the
   study's section 2.1 semantics stated as code.
2. :func:`intra_reference` computes the WITHIN-window pairs alone, and
   ``full_oracle - intra == inter`` is asserted on the small cell. That is the
   decomposition the study's exactness clause claims ("inter = cross-window pairs,
   intra = within-window pairs, same global edge -> no double count"), and a gate
   that only ever ran route 1 would never test it.

Everything here derives its own leaf addresses (MSB-first mixed radix), exactly as
``rola/ops/naive.py`` does and for the same reason: a reference that imported
production's map would make a wrong convention agree with itself.

Symbols at first use: ``widths`` per-level digit counts, ``N = prod widths``,
``d_v`` value channels, ``W`` the global window length in tokens, ``ell`` the
canonical leaf index.
"""
from __future__ import annotations

import math

import torch


def radix_strides(widths):
    return tuple(math.prod(widths[l + 1:]) for l in range(len(widths)))


def leaf_product(levels, widths, sl=None):
    """``[B,t,H,N]`` -- ``prod_l levels[l][..., d_l(ell)]`` over the token slice."""
    N = math.prod(widths)
    strides = radix_strides(widths)
    idx = torch.arange(N, device=levels[0].device)
    out = None
    for level, stride, width in zip(levels, strides, widths):
        x = level if sl is None else level[:, sl]
        sel = x.double().index_select(-1, (idx // stride) % width)
        out = sel if out is None else out * sel
    return out


def deposit_value(v):
    Bn, T, H, d_v = v.shape
    return torch.cat((v.double(), torch.ones(Bn, T, H, 1, device=v.device, dtype=torch.float64)),
                     dim=-1)


def inter_reference(read_levels, write_levels, g_write, v, widths, window,
                    state_in=None):
    """``(num, den, state_out)`` in fp64, canonical leaf order.

    ``num`` is ``[B,T,H,d_v]``, ``den`` ``[B,T,H]``, ``state_out``
    ``[B,H,N,d_v+1]``. The readout is against state-at-window-start and carries
    NO within-window pair -- that is the whole content of the inter/intra split.
    """
    Bn, T, H, d_v = v.shape
    N = math.prod(widths)
    vt = deposit_value(v)
    state = (torch.zeros(Bn, H, N, d_v + 1, device=v.device, dtype=torch.float64)
             if state_in is None else state_in.double().clone())
    num = torch.zeros(Bn, T, H, d_v, device=v.device, dtype=torch.float64)
    den = torch.zeros(Bn, T, H, device=v.device, dtype=torch.float64)
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        R = leaf_product(read_levels, widths, sl)
        out = torch.einsum("bthn,bhnc->bthc", R, state)
        num[:, sl] = out[..., :d_v]
        den[:, sl] = out[..., d_v]
        del R, out
        Wm = g_write[:, sl].double()[..., None] * leaf_product(write_levels, widths, sl)
        state = state + torch.einsum("bthn,bthc->bhnc", Wm, vt[:, sl])
        del Wm
    return num, den, state


def intra_reference(read_levels, write_levels, g_write, v, widths, window):
    """The WITHIN-window pairs, staircase-masked (inclusive read-after-write).

    Materializes the ``[w, w]`` similarity per window, which is exactly what the
    kernel must never do -- a reference is allowed to be slow and obvious.
    """
    Bn, T, H, d_v = v.shape
    vt = deposit_value(v)
    num = torch.zeros(Bn, T, H, d_v, device=v.device, dtype=torch.float64)
    den = torch.zeros(Bn, T, H, device=v.device, dtype=torch.float64)
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        n = sl.stop - sl.start
        R = leaf_product(read_levels, widths, sl)
        Wm = g_write[:, sl].double()[..., None] * leaf_product(write_levels, widths, sl)
        A = torch.einsum("brhn,bchn->bhrc", R, Wm)
        mask = torch.tril(torch.ones(n, n, device=v.device, dtype=torch.float64))
        A = A * mask
        out = torch.einsum("bhrc,bchv->brhv", A, vt[:, sl])
        num[:, sl] = out[..., :d_v]
        den[:, sl] = out[..., d_v]
    return num, den
