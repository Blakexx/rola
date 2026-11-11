# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LISTS' MODEL: the facts pass's per-(window, digit) lists -- the pure-torch spec the kernel is graded against
(docs/internals/facts/facts.md#lists-out).

A side's LIST LEVEL (its one sparse level) gets, per window and LIST GROUP -- ``G`` consecutive digits, the side's
``list_group`` (16 or 32: a box's extent at the level, `LIST_GROUPS`) -- a UNION -- the tokens live in any of the
group's digits -- RANKED: ``union[slot]`` is the token of union rank ``slot`` (``GAP`` past the union), ``count`` the
union's EXTENT, its size in whole steps of 8. Per digit, its LIST: its tokens in token order (slot order), no pads.
Its BOUNDARIES are its entries before union slot ``64 (j + 1)``, ``j < 8`` (the last its length), out in PREFIX FORM
over the group's lists modulo 2^16 (``pre[l][j]`` the boundaries of lists ``< l`` summed, row ``G`` the group's), beside
the group's exact TOTALS (``totals[j]``, for a planner): any chunk of the union whose bounds are multiples of 64 finds
each list's run and its place among the lists' runs by four loads, each a difference within the chunk (which holds
fewer than 2^16 entries), and no chunk size of a consumer's reaches this pass.

AN ENTRY IS A SLOT AND A BYTE, one int32: the token's union slot in the low 16 bits, its level byte in bits 16 to 23,
on either side. Nothing of a card is in it -- no exponent, no coefficient in an operand's format, no layout offset --
and its token is ``union[slot]``.

Mirrors ``csrc/rola/src/facts/facts.cuh`` (``kListCap``, ``kListAt``, ``kStep``, ``kListBound``, ``kGap``).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import liveness as lv

#: a list's boundaries: one every 64 union slots, the last its length; a window of W slots holds W / 64 (`list_at`)
LIST_BOUND = 64
#: A SIDE'S LIST GROUP: the digits one union and its prefix rows span, a liveness group (`lv.GROUP_DIGITS`) or its half
LIST_GROUPS = (16, 32)
#: the step: a union's extent is its size in whole steps
STEP = 8
#: a gap slot's token: token 0 flagged (``0x8000`` as int16), so a row fetch masked to the window stays inside it
GAP = -0x8000


def list_at(window: int) -> int:
    """A window's list boundaries: one every `LIST_BOUND` of its ``window`` union slots."""
    return window // LIST_BOUND


class Lists(NamedTuple):
    """One side's lists at a window of ``W`` tokens and list group ``G``: ``entries``
    ``[BH, windows, B, W]`` int32, ``slot | byte << 16`` (a list holds at most a window's slots), ``groups = ceil(B / G)``, ``pre``
    ``[BH, windows, groups, G + 1, W / 64]`` int16 the boundaries
    in prefix form modulo 2^16, ``totals`` ``[BH, windows, groups, W / 64]`` int32 the group's exact, ``union``
    ``[BH, windows, groups, W]`` (a slot's token, ``GAP`` in a gap) and
    ``count`` ``[BH, windows, groups]`` the unions' extents. Past a list's end and a union's extent nothing is
    defined."""

    entries: torch.Tensor
    pre: torch.Tensor
    totals: torch.Tensor
    union: torch.Tensor
    count: torch.Tensor


def columns(routes, values) -> tuple[torch.Tensor, torch.Tensor]:
    """``(m, m s)`` ``[B, T, H]`` fp32, multiplied as the facts pass does: each write level's scale in order (the
    write side's shared scale), then the value scale."""
    m = routes.scale_write[0].float()
    for scale in routes.scale_write[1:]:
        m = m * scale.float()
    return m, m * values.scale.float()


def final_lists(bits: torch.Tensor, layout: lv.LivenessLayout, level: int, *, rows, window: int,
                list_group: int) -> Lists:
    """One side's lists at ``level`` from its unpacked words ``bits`` ``[BH, rows, L]`` (:func:`lv.unpack`'s side slice)
    and the level's ``rows`` ``[B, T, H, width]`` u8, at a window of ``window`` tokens and a list group of
    ``list_group`` digits (`LIST_GROUPS`).

    GROUPED READS: ``bits`` ``[B H_w, g, rows, L]``, a write head's g query heads' read slots, are one union a (window,
    group) over the window's g W rows, head-major (row ``j W + i``, its head's tokens in order then the next head's):
    a list's entries in row order, boundaries every 64 slots of the merged union (``g LIST_AT`` of them)."""
    if bits.dim() == 3:
        bits = bits.unsqueeze(1)
    if list_group not in LIST_GROUPS:
        raise ValueError(f"a side's list group is one of {LIST_GROUPS} digits, got {list_group}")
    BH, heads, L, B, G = bits.shape[0], bits.shape[1], layout.L, layout.B[level], list_group
    windows, groups = -(-L // window), -(-B // G)
    columns_at = heads * list_at(window)
    base = layout.row_base(level)
    entries = torch.zeros((BH, windows, B, heads * window), dtype=torch.int32)
    bounds = torch.zeros((BH, windows, groups, G, columns_at), dtype=torch.int64)
    union_out = torch.full((BH, windows, groups, heads * window), GAP, dtype=torch.int64)
    count = torch.zeros((BH, windows, groups), dtype=torch.int64)
    rows = rows.cpu()
    H = rows.shape[2] // heads

    for bh in range(BH):
        b, h = divmod(bh, H)
        for w in range(windows):
            t0, t1 = w * window, min(L, (w + 1) * window)
            for g in range(groups):
                lo, hi = base + g * G, base + min(B, g * G + G)
                union = [(j, t) for j in range(heads) for t in bits[bh, j, lo:hi, t0:t1].any(0).nonzero().flatten().tolist()]
                slot_of = {row: s for s, row in enumerate(union)}
                count[bh, w, g] = -(-len(union) // STEP) * STEP
                union_out[bh, w, g, :len(union)] = torch.tensor([j * window + t for j, t in union], dtype=torch.int64)
                for dd in range(hi - lo):
                    digit = g * G + dd
                    slots = [slot_of[(j, t)] for j in range(heads)
                             for t in bits[bh, j, lo + dd, t0:t1].nonzero().flatten().tolist()]
                    words = [s | int(rows[b, t0 + union[s][1], h * heads + union[s][0], digit]) << 16 for s in slots]
                    if words:
                        entries[bh, w, digit, :len(words)] = torch.tensor(words, dtype=torch.int32)
                    bounds[bh, w, g, dd] = torch.tensor([sum(s < LIST_BOUND * (k + 1) for s in slots)
                                                         for k in range(columns_at)])
    zero = torch.zeros_like(bounds[..., :1, :])
    pre = torch.cat((zero, bounds.cumsum(3)), 3)
    return Lists(entries, pre & 0xFFFF, pre[..., -1, :], union_out, count)
