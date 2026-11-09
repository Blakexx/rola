# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY'S PLANS, modeled: per (batch-head, box class, window) the record `csrc/rola/src/carry/ampere/schedule.cuh`'s plan
pass writes from the facts pass's counts -- the plan word (the forced form's where it fits, else the rule's) and a step
word a step by the rule -- word for word, so a gate asserts the kernel's records and the forms its cell exercises
(docs/internals/carry/ampere/schedule.md#planner).

Symbols: ``W`` the window (the device's, :func:`rola.ops.window.window_of`), ``U`` a side's union slots in a window, ``S[j]`` a side's box lists' entries
before union slot ``64 (j + 1)``, ``B`` the area's bytes, ``n`` a window's entries (its read chunks, then its fold
chunks), ``t`` a step (``n`` the step that only retires the last entry).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of

LIST_BOUND = 64


def list_at() -> int:
    """The device window's list boundaries, one every `LIST_BOUND` union slots."""
    return window_of() // LIST_BOUND


#: the image's bytes whole and a half's: a box's digits x K x channels f16, the state's 32,768 values at every box
IMAGE_WHOLE, IMAGE_RING = 32 * 2048, 16 * 2048


def panel_row(channels: int, reduce: int) -> int:
    """A read slot's num panel row at the reduce dtype (0 f16 pairs, 1 fp32)."""
    return channels * (4 if reduce else 2)


def den_row(channels: int) -> int:
    """A read slot's den words, one fp32 a readout channel group of 8."""
    return channels // 2


WHOLE, HALVES = 0, 1
TOP = 0


class Counts(NamedTuple):
    """A window's counts: each side's union and segment sums, whether the write side lists (a side with no list
    level's fold buffer holds each slot's factors in its entries' place), the call's reduce dtype (0 f16 pairs, 1
    fp32: a read slot's panel row), the box's channels (32 the shipped 32 x 32 x 32 box's) and whether the read side
    is dense (its buffer a slot's exit words where a listed side's entries are)."""

    ru: int
    rs: tuple
    fu: int
    fs: tuple
    wl: bool = True
    reduce: int = 0
    channels: int = 32
    read_dense: bool = False


def image_bytes(mode: int) -> int:
    """`image_bytes`: the image's bytes at a mode."""
    return IMAGE_WHOLE if mode == WHOLE else IMAGE_RING


class Win(NamedTuple):
    """A window as the plan pass sees it: its counts, its plan word, its entries and read entries, its image's bytes."""

    c: Counts
    word: int
    n: int
    nr: int
    image: int


def _a128(x: int) -> int:
    return (x + 127) // 128 * 128


def read_bytes(slots: int, entries: int, read_row: int, reduce: int = 0, channels: int = 32,
               read_dense: bool = False) -> int:
    return (_a128(slots * read_row) + _a128(slots * panel_row(channels, reduce)) + _a128(slots * den_row(channels))
            + _a128((slots if read_dense else entries) * 8))


def fold_bytes(slots: int, entries: int, write_row: int, listed: bool = True) -> int:
    return _a128(slots * write_row) + _a128((entries if listed else slots) * 8)


def _sum_at(S, j: int) -> int:
    return int(S[j]) if 0 <= j < len(S) else 0


def entries_of(S, first: int, slots: int, total: int) -> int:
    """`entries_of`: ``S`` a side's segment sums, 8 a query head of its union."""
    hi = len(S) - 1 if first + slots >= total else (first + slots) // LIST_BOUND - 1
    return _sum_at(S, hi) - _sum_at(S, first // LIST_BOUND - 1)


def chunk_max(S, m: int, total: int) -> int:
    step = LIST_BOUND * m
    return max((entries_of(S, f, min(step, total - f), total) for f in range(0, total, step)), default=0)


def largest_chunk(u: int, S, room: int, nbytes) -> int:
    """The whole union where one buffer of it fits ``room``; else the largest ``64 m`` below it whose largest buffer
    fits -- the fewest chunks (`largest_chunk`)."""
    if nbytes(u, _sum_at(S, len(S) - 1)) <= room:
        return u
    one, m = 1, 1
    while LIST_BOUND * m < u:
        one = m if nbytes(LIST_BOUND * m, chunk_max(S, m, u)) <= room else one
        m += 1
    return LIST_BOUND * one


def tables_bytes(heads: int) -> int:
    """The read tables' bytes at ``heads`` query heads a write head (`tables_of`): the union, the prefix rows, the scale
    pass's bytes and the plan records, each two windows'."""
    W = window_of()
    return (2 * heads * W * 2 + -(-2 * 33 * list_at() * heads * 2 // 16) * 16 + 2 * heads * W
            + 2 * carry_ops.plan_words(heads, W) * 4)


def area_of(smem: int, area_one: int, heads: int) -> int:
    """The call's area (`tables_of`): from the fixed tenants' end -- the census's one-head area below its tables -- to
    the tables at the top of shared memory."""
    top = lambda g: (smem - tables_bytes(g)) // 128 * 128  # noqa: E731
    return area_one - (top(1) - top(heads))


def chunks_of(u: int, chunk: int) -> int:
    return -(-u // chunk) if chunk > 0 else 0


def plan_word(mode: int, rc: int, fc: int, ru: int, fu: int) -> int:
    """`plan_word`: mode | read chunk / 64 << 1 | fold chunk / 64 << 9 | read chunks << 14 | fold chunks << 22."""
    return (mode | -(-rc // LIST_BOUND) << 1 | -(-fc // LIST_BOUND) << 9 | chunks_of(ru, rc) << 14
            | chunks_of(fu, fc) << 22)


def chunks(word: int) -> tuple[int, int, int, int, int]:
    """``(mode, read chunk, fold chunk, read chunks, fold chunks)`` of a plan word."""
    return (word & 1, LIST_BOUND * (word >> 1 & 0xFF), LIST_BOUND * (word >> 9 & 0x1F), word >> 14 & 0xFF,
            word >> 22 & 0x1F)


def rule_plan(c: Counts, read_row: int, write_row: int, area: int) -> int:
    """`rule_plan`: the read chunk beside the halves, the image whole where it fits beside the largest read buffer, the
    fold's largest chunk in the area."""
    def nbytes(s, e):
        return read_bytes(s, e, read_row, c.reduce, c.channels, c.read_dense)

    rc = largest_chunk(c.ru, c.rs, area - image_bytes(HALVES), nbytes) if c.ru > 0 else 0
    nr = chunks_of(c.ru, rc)
    most = 0
    if nr > 0:
        most = nbytes(min(rc, c.ru), chunk_max(c.rs, rc // LIST_BOUND, c.ru) if rc < c.ru
                      else _sum_at(c.rs, len(c.rs) - 1))
    whole = nr > 0 and image_bytes(WHOLE) + most <= area
    fc = largest_chunk(c.fu, c.fs, area, lambda s, e: fold_bytes(s, e, write_row, c.wl)) if c.fu > 0 else 0
    return plan_word(WHOLE if whole else HALVES, rc, fc, c.ru, c.fu)


def forced_plan(forced: int, c: Counts, rule: int, read_row: int, write_row: int, area: int) -> int:
    """`forced_plan`: the forced form's plan where every chunk fits alone beside its image and a whole image has reads
    to serve; else the rule's."""
    if forced < 0:
        return rule
    r_mode, r_rc, r_fc, _, _ = chunks(rule)
    fm, fr, ff = forced & 3, forced >> 2 & 0xFF, forced >> 10 & 0xF
    mode = r_mode if fm == 0 else fm - 1
    rc = 0 if c.ru == 0 else r_rc if fr == 0 else c.ru if fr == 0xFF else LIST_BOUND * fr
    fc = 0 if c.fu == 0 else r_fc if ff == 0 else c.fu if ff == 0xF else LIST_BOUND * ff
    image = image_bytes(mode)
    fits = mode == HALVES or c.ru > 0
    for first in range(0, c.ru, rc or 1):
        s = min(rc, c.ru - first)
        fits = fits and image + read_bytes(s, entries_of(c.rs, first, s, c.ru), read_row, c.reduce, c.channels,
                                           c.read_dense) <= area
    for first in range(0, c.fu, fc or 1):
        s = min(fc, c.fu - first)
        fits = fits and fold_bytes(s, entries_of(c.fs, first, s, c.fu), write_row, c.wl) <= area
    return plan_word(mode, rc, fc, c.ru, c.fu) if fits else rule


def win_of(c: Counts, forced: int, read_row: int, write_row: int, area: int) -> Win:
    """`win_of`: a window's plan word and what the rule reads off it."""
    word = forced_plan(forced, c, rule_plan(c, read_row, write_row, area), read_row, write_row, area)
    mode, _, _, nr, nf = chunks(word)
    return Win(c, word, nr + nf, nr, image_bytes(mode) if nr > 0 else 0)


def drained_bytes(x: Win, t: int, read_row: int) -> tuple[int, int]:
    """Read entry ``t``'s PANEL in its buffer, what its drain reads (`drained_of`): its offset and bytes, the num panel
    and the den words."""
    _, rc, _, _, _ = chunks(x.word)
    slots = min(rc, x.c.ru - t * rc)
    return (_a128(slots * read_row),
            _a128(slots * panel_row(x.c.channels, x.c.reduce)) + _a128(slots * den_row(x.c.channels)))


def entry_bytes(x: Win, t: int, read_row: int, write_row: int) -> int:
    """Entry ``t``'s buffer bytes (`entry_bytes_of`): its chunk's slots and entries from the exact totals."""
    _, rc, fc, _, _ = chunks(x.word)
    read = t < x.nr
    chunk, total = (rc, x.c.ru) if read else (fc, x.c.fu)
    first = (t if read else t - x.nr) * chunk
    slots = min(chunk, total - first)
    if read:
        return read_bytes(slots, entries_of(x.c.rs, first, slots, total), read_row, x.c.reduce, x.c.channels,
                          x.c.read_dense)
    return fold_bytes(slots, entries_of(x.c.fs, first, slots, total), write_row, x.c.wl)


def slot_place(slot: int, nbytes: int, base: int, area: int) -> int:
    return area - nbytes if slot == TOP else base


def disjoint(a: int, ab: int, b: int, bb: int) -> bool:
    return a + ab <= b or b + bb <= a


def fits_beside(at: int, nbytes: int, live_at: int, live_bytes: int, image: int, area: int) -> bool:
    return at >= image and at + nbytes <= area and disjoint(at, nbytes, live_at, live_bytes)


class Record(NamedTuple):
    """One window's record as the plan pass writes it: its plan word and its steps, each ``(at, ahead, next, reuse)``
    -- the entry's place, filled ahead, the next entry filled now, the rendezvous between the retired read's drain and
    the step's own fill where it takes the drained panel's bytes."""

    word: int
    steps: tuple

    def words(self) -> tuple:
        """The record's 32-bit words the plan pass writes: the plan word, then the steps' 16-bit words in pairs."""
        packed = [at // 128 | ahead << 11 | nxt << 12 | reuse << 13 for at, ahead, nxt, reuse in self.steps]
        packed += [0] * (len(packed) & 1)
        return (self.word,) + tuple(lo | hi << 16 for lo, hi in zip(packed[::2], packed[1::2]))


def records(wins: list[Win], read_row: int, write_row: int, area: int) -> list[Record]:
    """A box class's records over its windows (`window_plan_kernel`): the slot parity an exclusive scan of the entry
    counts' parities, each window's first entry filled ahead where the window before leaves it room, its steps by the
    rule."""
    B = area

    def eb(x, t):
        return entry_bytes(x, t, read_row, write_row)

    out, par = [], 0
    for w, x in enumerate(wins):
        ahead, at = False, 0
        if x.n > 0:
            b0 = eb(x, 0)
            at, ahead = slot_place(par, b0, x.image, B), True
            if w > 0 and wins[w - 1].n > 0:
                pv = wins[w - 1]
                ppar, lt = par ^ (pv.n & 1), pv.n - 1
                lread, lb = lt < pv.nr, eb(pv, lt)
                lbase = pv.image if lread else 0
                if not lread and lt == pv.nr and pv.nr > 0:
                    rb = eb(pv, pv.nr - 1)
                    rat = slot_place(ppar ^ ((pv.nr - 1) & 1), rb, pv.image, B)
                    lbase = pv.image if fits_beside(slot_place(ppar ^ (lt & 1), lb, pv.image, B), lb, rat, rb,
                                                    pv.image, B) else 0
                ahead = fits_beside(at, b0, slot_place(ppar ^ (lt & 1), lb, lbase, B), lb,
                                    pv.image if lread else 0, B)
        nx = wins[w + 1] if w + 1 < len(wins) else None
        steps, gone, gone_b = [], 0, 0
        for t in range(x.n + 1):
            work, read = t < x.n, t < x.nr
            cross = t + 1 >= x.n and (work or t == 0)
            slot = par ^ (t & 1)
            b = eb(x, t) if work else 0
            if work and not ahead:
                at = slot_place(slot, b, x.image if read else 0, B)
            nxt, nat, nb = False, 0, 0
            if not cross and work:
                nb = eb(x, t + 1)
                nat = slot_place(slot ^ 1, nb, x.image if t + 1 < x.nr or read else 0, B)
                nxt = fits_beside(nat, nb, at, b, x.image if read else 0, B)
            elif cross and nx is not None and nx.n > 0:
                nb = eb(nx, 0)
                nat = slot_place(slot ^ 1 if work else slot, nb, nx.image, B)
                nxt = fits_beside(nat, nb, at if work else 0, b, x.image if work and read else 0, B)
            drained = t > 0 and t - 1 < x.nr
            #: the carry fills each entry at its own step: the rendezvous where that fill meets the drained panel
            reuse = drained and work and not disjoint(at, b, gone, gone_b)
            steps.append((at if work else 0, int(ahead), int(nxt), int(reuse)))
            #: what a drain reads of a read entry: its panel
            pn = drained_bytes(x, t, read_row) if work and read else (0, b)
            gone, gone_b, ahead, at = at + pn[0], pn[1], nxt, nat
        out.append(Record(x.word, tuple(steps)))
        par ^= x.n & 1
    return out


def _side(lists, carve_plan, routes, w: int, bh: int, group: int):
    """``(U, S)``: a side's union slots and segment sums in window ``w`` at a box whose list level's group is
    ``group``; every window token and no entries for a side with no list level."""
    if not carve_plan.lists:
        return min(window_of(), routes.tokens - w * window_of()), (0,) * list_at()
    totals = lists.totals[bh, w, group].long().cpu().numpy()
    return int(lists.count[bh, w, group]), tuple(int(v) for v in np.asarray(totals))


def plans(routes, facts, carve, forced: str | None = None) -> list[list[Record]]:
    """Every box class's records over the call's windows (a class per pair of the two list levels' groups, each
    batch and write head, in the plan tensor's order), as the plan pass on this card writes them (its area and pool rows from
    the arm's census)."""
    f = carry_ops.form(carry_ops.arm_key(carve))
    area, read_row, write_row = area_of(f.smem, f.area, routes.group), f.read_row, f.write_row
    reduce = int(carry_ops.reduce_dtype_for(carve) == torch.float32)
    word = carry_ops.forced_word(forced)
    windows = -(-routes.tokens // window_of())
    write = facts_ops.side_lists(facts, "write") if carve.fold.lists else None
    read = facts_ops.side_lists(facts, "read") if carve.readout.lists else None
    #: a side's groups at its list level, the arm's list group a union (`plan.md#list-groups`)
    group = {id(carve.readout): f.read_list_group, id(carve.fold): f.write_list_group}
    groups = lambda p: max(1, carve.widths[p.lists[0]] // group[id(p)]) if p.lists else 1  # noqa: E731
    out = []
    for bh in range(routes.batch * routes.write_heads):
        for gr in range(groups(carve.readout)):
            for gw in range(groups(carve.fold)):
                wins = []
                for w in range(windows):
                    ru, rs = _side(read, carve.readout, routes, w, bh, gr)
                    fu, fs = _side(write, carve.fold, routes, w, bh, gw)
                    wins.append(win_of(Counts(ru, rs, fu, fs, bool(carve.fold.lists), reduce, carve.channels,
                                              not carve.readout.lists),
                                       word, read_row, write_row, area))
                out.append(records(wins, read_row, write_row, area))
    return out


def forms(box_records: list[list[Record]]) -> set[str]:
    """The forms a call exercises: each image mode read under, and ``<mode>-chunks`` where a window reads more than one
    chunk under it; ``fold-chunks``; ``ahead`` (an entry filled while the one before computes), ``next`` (a window's
    first entry filled by the window before), ``deferred`` (an entry filled at its own step), ``reuse`` (a fill that
    waits on the retired read's drain), ``empty`` (a window of no entries)."""
    out = set()
    for box in box_records:
        for w, rec in enumerate(box):
            mode, _, _, nr, nf = chunks(rec.word)
            name = ("whole", "halves")[mode]
            if nr > 0:
                out.add(name)
            if nr > 1:
                out.add(f"{name}-chunks")
            if nf > 1:
                out.add("fold-chunks")
            if nr + nf == 0:
                out.add("empty")
            for t, (_, ahead, _, reuse) in enumerate(rec.steps):
                if t < nr + nf:
                    out.add(("deferred", "next" if t == 0 and w > 0 else "ahead")[ahead])
                if reuse:
                    out.add("reuse")
    return out
