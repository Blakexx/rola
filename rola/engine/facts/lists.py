# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LISTS' MODEL: the facts pass's per-(window, digit) lists in the final form the carry reads, and the window's
gains pairs -- the pure-torch spec the kernel is graded against (docs/internals/facts/facts.md#lists-out, #gains).

A side's LIST LEVEL (its one sparse level) gets, per window and LIST GROUP -- ``G`` consecutive digits, the side's
``list_group`` (16 or 32: the carry arm's box extent at the level, `LIST_GROUPS`) -- a UNION -- the tokens live in any
of the group's digits -- RANKED: ``union[slot]`` is the token of union rank ``slot`` (``GAP`` past the union), ``count``
the union's EXTENT, its size in whole steps of 8. Per digit, its LIST: its tokens in token order (slot order), no pads.
Its BOUNDARIES are its entries before union slot ``64 (j + 1)``, ``j < 8`` (the last its length), out in PREFIX FORM
over the group's lists modulo 2^16 (``pre[l][j]`` the boundaries of lists ``< l`` summed, row ``G`` the group's), beside
the group's exact TOTALS (``totals[j]``, for the carry's planner): any chunk of the union whose bounds are multiples of
64 finds each list's run and its place among the lists' runs by four loads, each a difference within the chunk (which
holds fewer than 2^16 entries), and no chunk size of the carry's reaches this pass.

An entry is four int16 words, its two 32-bit halves the carry's ``uint2``:
- READ: the slot's pool row offset in the union in 16-byte chunks, the row's swizzle folded in (the carry takes the
  slot back from it); its token's window place (the carry looks up the token's scale byte by it); the coefficient
  ``f16(byte / 256)``, 0. A chunk starting at a multiple of 64 finds its rows by subtracting its first slot's offsets:
  the swizzles depend on a slot's low bits only.
- WRITE: the slot, 0; ``c_v = f16(fp32(b m s) 2^(4 - E_v))``, ``c_m = f16(fp32(b m) 2^(4 - E_m))`` -- ``b`` the token's
  level byte, ``m`` and ``m s`` the facts pass's fp32 columns, ``E = max(K, -123)`` the list's exponent.

The GAINS: each token's ``(m, m s)`` as f16 at its window's powers of two, ``2^s`` with ``s = -4 - floor(log2(max))``
(the largest of each in ``[2^-4, 2^-3)``; a zero largest ``s = 0``).

Mirrors ``csrc/rola/src/facts/facts.cuh`` (``kListCap``, ``kListAt``, ``kCoefTop``, ``kUnitMin``, ``kStep``) and the lists'
addresses in ``csrc/rola/src/carry/carry.cuh`` (``kListBound``, ``kGap``, ``pool_chunk``).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import liveness as lv

#: a list's boundaries: one every 64 union slots, the last its length; a window of W slots holds W / 64 (`list_at`)
LIST_BOUND = 64
#: A SIDE'S LIST GROUP: the digits one union and its prefix rows span, a liveness group (`lv.GROUP_DIGITS`) or its half
LIST_GROUPS = (16, 32)
COEF_TOP = 4
UNIT_MIN = COEF_TOP - 127
#: the carry's step: a union's extent is its size in whole steps
STEP = 8
#: a gap slot's token: token 0 flagged (``0x8000`` as int16), so a row fetch masked to the window stays inside it
GAP = -0x8000


def list_at(window: int) -> int:
    """A window's list boundaries: one every `LIST_BOUND` of its ``window`` union slots."""
    return window // LIST_BOUND


class Lists(NamedTuple):
    """One side's lists in the final form at a window of ``W`` tokens and list group ``G``: ``entries``
    ``[BH, windows, B, W, 4]`` int16 (a list holds at most a window's slots), ``groups = ceil(B / G)``, ``pre``
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


def chunk0(row_bytes: int, slot: int) -> int:
    """A pool row's chunk 0 after the swizzle (the carry's ``pool_chunk``): an 8-chunk row XORs by the slot's residue,
    a row of an odd number of 64-byte quarters by the residue's high bits within each four."""
    return slot & 7 if row_bytes == 128 else (slot >> 1) & 3


def _f16_bits(x: float) -> int:
    return int(torch.tensor(x, dtype=torch.float32).half().view(torch.int16))


def final_lists(bits: torch.Tensor, layout: lv.LivenessLayout, level: int, *, row_bytes: int, rows, cols,
                list_exp, window: int, list_group: int) -> Lists:
    """One side's lists at ``level`` from its unpacked words ``bits`` ``[BH, rows, L]`` (:func:`lv.unpack`'s side slice),
    the read side's pool rows of ``row_bytes``, at a window of ``window`` tokens and a list group of ``list_group``
    digits (`LIST_GROUPS`). Both sides' coefficients read the level's ``rows`` ``[B, T, H, width]``
    u8; the WRITE side (``cols`` given) also takes ``cols`` (:func:`columns`) and the pass's ``list_exp``
    ``[BH, windows, B, 2]``; the READ side passes both as ``None``.

    GROUPED READS: ``bits`` ``[B H_w, g, rows, L]``, a write head's g query heads' read slots, are one union a (window,
    group) over the window's g W rows, head-major (row ``j W + i``, its head's tokens in order then the next head's):
    a list's entries in row order, the entry's token field its row, boundaries every 64 slots of the merged union
    (``g LIST_AT`` of them), a read entry's row offset modulo 2^16 (the carry takes it back by its chunk's first)."""
    if bits.dim() == 3:
        bits = bits.unsqueeze(1)
    if list_group not in LIST_GROUPS:
        raise ValueError(f"a side's list group is one of {LIST_GROUPS} digits, got {list_group}")
    BH, heads, L, B, G = bits.shape[0], bits.shape[1], layout.L, layout.B[level], list_group
    windows, groups = -(-L // window), -(-B // G)
    columns_at = heads * list_at(window)
    base = layout.row_base(level)
    write = cols is not None
    entries = torch.zeros((BH, windows, B, heads * window, 4), dtype=torch.int16)
    bounds = torch.zeros((BH, windows, groups, G, columns_at), dtype=torch.int64)
    union_out = torch.full((BH, windows, groups, heads * window), GAP, dtype=torch.int64)
    count = torch.zeros((BH, windows, groups), dtype=torch.int64)
    rows = rows.cpu()
    if write:
        m, ms = (c.cpu() for c in cols)
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
                    words = [_entry(s, union[s], write, row_bytes, rows, b, h * heads + union[s][0], t0, digit,
                                    m if write else None, ms if write else None,
                                    list_exp[bh, w, digit] if write else None, window) for s in slots]
                    if words:
                        entries[bh, w, digit, :len(words)] = torch.tensor(words, dtype=torch.int16)
                    bounds[bh, w, g, dd] = torch.tensor([sum(s < LIST_BOUND * (k + 1) for s in slots)
                                                         for k in range(columns_at)])
    zero = torch.zeros_like(bounds[..., :1, :])
    pre = torch.cat((zero, bounds.cumsum(3)), 3)
    return Lists(entries, pre & 0xFFFF, pre[..., -1, :], union_out, count)


def _entry(s, row, write, row_bytes, rows, b, h, t0, digit, m, ms, list_exp, window):
    """One entry's four int16 words, ``s`` its slot, ``row`` its (head in the group, window place) and ``h`` its
    query head; a read entry's token field its head above the ``window``'s token bits."""
    def i16(x: int) -> int:
        return x - 0x10000 if x >= 0x8000 else x

    j, token = row
    byte = float(rows[b, t0 + token, h, digit])
    if write:
        t = t0 + token
        e_m = max(int(list_exp[0]), UNIT_MIN)
        e_v = max(int(list_exp[1]), UNIT_MIN)
        cm = _f16_bits((byte * float(m[b, t, h])) * 2.0 ** (COEF_TOP - e_m))
        cv = _f16_bits((byte * float(ms[b, t, h])) * 2.0 ** (COEF_TOP - e_v))
        return [i16(s), 0, cv, cm]
    pool = s * row_bytes // 16 + chunk0(row_bytes, s)
    return [i16(pool), j * window + token, _f16_bits(byte / 256), 0]


def gains_shift(top: torch.Tensor) -> torch.Tensor:
    """The window's power of two for each channel: ``-4 - floor(log2(top))`` (the exponent field's), clamped to
    [-126, 127], 0 for a zero largest."""
    e = (top.float().view(torch.int32) >> 23) - 127
    return torch.where(top > 0, (-e - 4).clamp(-126, 127), torch.zeros_like(e))


def gains(routes, values, L: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gains16, gains_exp)``: ``[BH, L, 2]`` f16 each token's ``(m, m s)`` at its window's powers of two, and
    ``[BH, windows, 2]`` those powers' exponents, at a window of ``window`` tokens."""
    m, ms = columns(routes, values)
    B, T, H = m.shape
    both = torch.stack((m, ms), -1).permute(0, 2, 1, 3).reshape(B * H, T, 2).cpu()
    windows = -(-L // window)
    pad = torch.cat((both, both.new_zeros(B * H, windows * window - T, 2)), 1)
    top = pad.reshape(B * H, windows, window, 2).amax(2)
    shift = gains_shift(top)
    scale = torch.ldexp(torch.ones_like(top), shift.float()).repeat_interleave(window, 1)[:, :T]
    return (both * scale).half(), shift
