# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference for R4's reverse pass, split on the forward's own window edge.

Two routes to one number, as ``reference.py`` has for the forward: the CARRY route
(:func:`carry_backward_reference`, the cross-window pairs, which is what the two
scans compute) plus the INTRA route (:func:`intra_backward_reference`, the
within-window pairs) equals ``torch.autograd`` through the whole operator. The
gate that states this is `test_prefill_backward_reference.py`.

Symbols at first use: ``widths`` per-level digit counts, ``N = prod widths``,
``d_v`` value channels, ``W`` the global window length in tokens, ``cols = d_v+1``,
``dn`` the seed pair ``(d_num, d_den)``, ``S_entry`` the state at window entry,
``Lambda`` the adjoint at window exit, ``LOO`` a level's leave-one-out product.
"""
from __future__ import annotations

import math

import torch

from tests.oracle.reference import deposit_value, leaf_product, radix_strides


def readout_seeds(num, den, dy, eps):
    """The division's backward: ``dy`` on ``y = num/(den+eps)`` into ``(d_num, d_den)``.

    The floor is ADDITIVE, so there is no kink and no ``den``-dependent branch; the
    ``d_den`` form is the stable one, taken against the operator's own output.
    """
    dt = den.double() + eps
    y = num.double() / dt[..., None]
    d_num = dy.double() / dt[..., None]
    d_den = -(dy.double() * y).sum(-1) / dt
    return d_num, d_den


def _digit_index(widths, l, device):
    """``[N]`` -- leaf ``s``'s level-``l`` digit, MSB-first mixed radix."""
    N = math.prod(widths)
    stride = radix_strides(widths)[l]
    return (torch.arange(N, device=device) // stride) % widths[l]


def _loo(levels, widths, l, sl=None):
    """``[B,t,H,N]`` -- ``prod_{l' != l} levels[l'][..., d_l'(s)]``; never a division."""
    #: the surviving levels keep their OWN strides, so the leaf axis is re-derived
    #: against the full width tuple rather than against the reduced one.
    N = math.prod(widths)
    strides = radix_strides(widths)
    idx = torch.arange(N, device=levels[0].device)
    out = None
    for i, (level, width) in enumerate(zip(levels, widths)):
        if i == l:
            continue
        x = level if sl is None else level[:, sl]
        sel = x.double().index_select(-1, (idx // strides[i]) % width)
        out = sel if out is None else out * sel
    if out is None:
        base = levels[0] if sl is None else levels[0][:, sl]
        return torch.ones(base.shape[:-1] + (N,), device=base.device, dtype=torch.float64)
    return out


def _scatter_digits(per_leaf, widths, l):
    """``[..., N] -> [..., width_l]``: sum a per-leaf quantity into its level-``l`` digit."""
    digit = _digit_index(widths, l, per_leaf.device)
    out = torch.zeros(per_leaf.shape[:-1] + (widths[l],),
                      device=per_leaf.device, dtype=torch.float64)
    return out.index_add_(-1, digit, per_leaf)


def carry_backward_reference(read_levels, write_levels, g_write, v, widths, window,
                             d_num, d_den, state_in=None, d_state_out=None):
    """The CROSS-WINDOW half of every gradient -- what Scan A and Scan B compute.

    Returns ``(d_read, d_write, d_gw, d_v, d_state_in)`` with ``d_read``/``d_write``
    tuples of ``[B,T,H,width_l]``, ``d_gw`` ``[B,T,H]``, ``d_v`` ``[B,T,H,d_v]`` and
    ``d_state_in`` ``[B,H,N,cols]`` in CANONICAL leaf order.
    """
    Bn, T, H, d_v = v.shape
    N = math.prod(widths)
    D = len(widths)
    dev = v.device
    vt = deposit_value(v)
    dn = torch.cat((d_num.double(), d_den.double()[..., None]), dim=-1)  # [B,T,H,cols]

    d_read = [torch.zeros(Bn, T, H, w, device=dev, dtype=torch.float64) for w in widths]
    d_write = [torch.zeros(Bn, T, H, w, device=dev, dtype=torch.float64) for w in widths]
    d_gain = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)
    d_value = torch.zeros(Bn, T, H, d_v, device=dev, dtype=torch.float64)

    # ---- Scan A: the ascending replay, and the read side -------------------
    state = (torch.zeros(Bn, H, N, d_v + 1, device=dev, dtype=torch.float64)
             if state_in is None else state_in.double().clone())
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        dR = torch.einsum("bthc,bhnc->bthn", dn[:, sl], state)
        for l in range(D):
            d_read[l][:, sl] = _scatter_digits(dR * _loo(read_levels, widths, l, sl), widths, l)
        Wm = g_write[:, sl].double()[..., None] * leaf_product(write_levels, widths, sl)
        state = state + torch.einsum("bthn,bthc->bhnc", Wm, vt[:, sl])
        del dR, Wm

    # ---- Scan B: the descending adjoint, and the write side ----------------
    adjoint = (torch.zeros(Bn, H, N, d_v + 1, device=dev, dtype=torch.float64)
               if d_state_out is None else d_state_out.double().clone())
    for t0 in range(((T - 1) // window) * window, -1, -window):
        sl = slice(t0, min(t0 + window, T))
        U = leaf_product(write_levels, widths, sl)
        gw = g_write[:, sl].double()
        dW = torch.einsum("bhnc,bthc->bthn", adjoint, vt[:, sl])
        for l in range(D):
            d_write[l][:, sl] = gw[..., None] * _scatter_digits(
                dW * _loo(write_levels, widths, l, sl), widths, l)
        acc = torch.einsum("bthn,bhnc->bthc", U, adjoint)
        d_value[:, sl] = gw[..., None] * acc[..., :d_v]
        d_gain[:, sl] = (acc * vt[:, sl]).sum(-1)
        R = leaf_product(read_levels, widths, sl)
        adjoint = adjoint + torch.einsum("bthn,bthc->bhnc", R, dn[:, sl])
        del U, dW, acc, R

    return tuple(d_read), tuple(d_write), d_gain, d_value, adjoint


def intra_backward_reference(read_levels, write_levels, g_write, v, widths, window,
                             d_num, d_den):
    """The WITHIN-WINDOW half of every gradient -- what the intra backward computes.

    Returns ``(d_read, d_write, d_gw, d_v)``; the within-window term touches no state.
    """
    Bn, T, H, d_v = v.shape
    D = len(widths)
    dev = v.device

    d_read = [torch.zeros(Bn, T, H, w, device=dev, dtype=torch.float64) for w in widths]
    d_write = [torch.zeros(Bn, T, H, w, device=dev, dtype=torch.float64) for w in widths]
    d_gain = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)
    d_value = torch.zeros(Bn, T, H, d_v, device=dev, dtype=torch.float64)

    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        n = sl.stop - sl.start
        gw = g_write[:, sl].double()
        grams = [torch.einsum("brhe,bche->bhrc", read_levels[l][:, sl].double(),
                              write_levels[l][:, sl].double()) for l in range(D)]
        prod = grams[0].clone()
        for l in range(1, D):
            prod = prod * grams[l]
        mask = torch.tril(torch.ones(n, n, device=dev, dtype=torch.float64))
        #: Gd'[t,j] = <dn[t,:], vt[j,:]>, the ONE new pairwise object the backward needs.
        gd = (torch.einsum("brhv,bchv->bhrc", d_num[:, sl].double(), v[:, sl].double())
              + d_den[:, sl].double().permute(0, 2, 1)[..., None])
        dA = gd * mask
        A = prod * gw.permute(0, 2, 1)[:, :, None, :] * mask
        d_value[:, sl] += torch.einsum("bhrc,brhv->bchv", A, d_num[:, sl].double())
        d_gain[:, sl] += torch.einsum("bhrc,bhrc->bhc", dA, prod).permute(0, 2, 1)
        for l in range(D):
            #: the leave-one-out over LEVELS, as a product and never as A / G_l:
            #: G_l is exactly zero wherever the two supports are disjoint.
            loo = torch.ones_like(prod)
            for l2 in range(D):
                if l2 != l:
                    loo = loo * grams[l2]
            dG = dA * gw.permute(0, 2, 1)[:, :, None, :] * loo
            d_read[l][:, sl] += torch.einsum("bhrc,bche->brhe", dG,
                                             write_levels[l][:, sl].double())
            d_write[l][:, sl] += torch.einsum("bhrc,brhe->bche", dG,
                                              read_levels[l][:, sl].double())
        del grams, prod, gd, dA, A

    return tuple(d_read), tuple(d_write), d_gain, d_value
