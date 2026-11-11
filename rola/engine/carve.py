# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARVE PLANNER -- producer v1 and the check every carve passes: the carry's box and each side's operation
plan, from the declaration: the fold a state carve (its warps own lists), the readout a value carve (its warps own
channels over every leaf).

The planner is a pure fold over the declaration (the shipped widths, each level's two sparse flags, ``d_v``), so a
better planner replaces this one without touching a kernel: every kernel that depends on the geometry takes the
:class:`~rola.engine.types.Carve` it returns and refuses one it was not built for (docs/internals/engine/carve.md).
"""
from __future__ import annotations

from functools import cache

from rola.engine.types import Carve, SidePlan

#: THE BOX'S STATE, the architecture's register budget for it: 8 warps x 32 lanes x 128 fp32 registers a lane. Its
#: leaves are the state over the channel extent, and no level's extent exceeds the square root of 1,024 leaves.
WARPS = 8
BOX_VALUES = WARPS * 32 * 128
#: THE CHANNEL EXTENT: the readout's channel groups are 8 channels (an MMA's N), at most a warp each, so a box holds
#: 32 (4 groups, two token groups) to 128 (8 groups of 16)
CHANNELS = 32
MAX_CHANNELS = 128
GROUP_CHANNELS = 8
CAP = 32
#: THE LISTS' GROUPS: the facts pass ranks a side's lists in a group of 16 or 32 digits
#: (`rola.engine.facts.lists.LIST_GROUPS`), the arm reading its own (docs/internals/carry/ampere/plan.md#list-groups), so a
#: listed level spans one of them (or its whole width) in the box, each side its own
LIST_GROUPS = (16, 32)
#: THE LISTED BOXES' CHANNELS: the components serve 32 or 64 channels where a side lists (a 128-channel box needs the
#: fold's k8 and m8 forms and a readout warp of two channel groups)
LISTED_CHANNELS = 64
#: A SIDE'S DENSE EXTENT is an MMA tile of rows at least.
MIN_DENSE = 16

SIDES = ("read", "write")


def _log2(n: int) -> int:
    return n.bit_length() - 1


def tags(read_sparse, write_sparse) -> tuple[str, ...]:
    """Each level's tag, its read flag then its write flag: ``S`` sparse, ``D`` dense."""
    return tuple(("S" if r else "D") + ("S" if w else "D") for r, w in zip(read_sparse, write_sparse))


def _allocate(widths, level_tags, total: int, cap: int, rows) -> list[int] | None:
    """Producer v1's extents as exponents, or None when the box cannot be filled under the cap: each dense side's row
    level (``rows``) an MMA tile first; DD levels as much as fits, spread evenly; DS and SD by turns, DS first (the
    readout's dense level, its K, before its list level), each concentrated on its widest level; SS last."""
    room = [min(_log2(w), _log2(cap)) for w in widths]
    bits = [0] * len(widths)
    left = total

    def give(level: int) -> bool:
        nonlocal left
        if left == 0 or bits[level] == room[level]:
            return False
        bits[level] += 1
        left -= 1
        return True

    def widest(tag: str) -> int | None:
        open_ = [lv for lv, t in enumerate(level_tags) if t == tag and bits[lv] < room[lv]]
        return max(open_, key=lambda lv: (room[lv], -lv)) if open_ else None

    for lv in rows:
        while bits[lv] < _log2(MIN_DENSE) and give(lv):
            pass
    dd = [lv for lv, t in enumerate(level_tags) if t == "DD"]
    while left and any(give(lv) for lv in sorted(dd, key=lambda lv: bits[lv])):
        pass
    while left:
        moved = False
        for tag in ("DS", "SD"):
            lv = widest(tag)
            moved |= lv is not None and give(lv)
        if not moved:
            break
    while left and (lv := widest("SS")) is not None:
        give(lv)
    return bits if left == 0 else None


def _operand(lists, operand, coef=None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A side's ``(lists, operand)``: a side with a sparse level lists it and forms its operand from the dense ones; a
    side with none takes one dense level as its rows and the other as its coefficients, named in that order -- its
    coefficients the other side's list level where that side lists one of its levels, else its first dense level
    (docs/internals/engine/carve.md#sides)."""
    if not lists and len(operand) == 2:
        c = coef if coef in operand else operand[0]
        return (), (operand[1] if c == operand[0] else operand[0], c)
    return tuple(lists), tuple(operand)


def _product(levels, extents) -> int:
    n = 1
    for lv in levels:
        n *= extents[lv]
    return n


def _fold(lists, operand, extents, leaves: int) -> SidePlan:
    """THE FOLD'S STATE CARVE: a warp owns ``leaves / 8`` leaves, its dense rows whole while they fit (``tile_rows``),
    the rest in lists (``tile_lists``)."""
    rows = _product(operand if lists else operand[:1], extents)
    tile_rows = min(rows, leaves // WARPS)
    return SidePlan(lists=lists, operand=operand, tile_lists=leaves // WARPS // tile_rows, tile_rows=tile_rows)


def _readout(lists, operand, extents) -> SidePlan:
    """THE READOUT'S VALUE CARVE: every warp holds every leaf of the box for its channels -- ``tile_rows`` the
    operand's rows (the MMA's K: a listed side's dense levels, a dense side's row level) by ``tile_lists`` digits (a
    listed side's list levels' values, a dense side's coefficient level's)."""
    rows, digits = (operand, lists) if lists else (operand[:1], operand[1:])
    return SidePlan(lists=lists, operand=operand, tile_lists=_product(digits, extents),
                    tile_rows=_product(rows, extents))


def channel_groups(carve: Carve) -> int:
    """The readout's channel groups: 8 channels a group, at most a warp each."""
    return min(WARPS, carve.channels // GROUP_CHANNELS)


def token_groups(carve: Carve) -> int:
    """The readout's token groups: the warps left over the channel groups, each its own tokens' panel and pool share."""
    return WARPS // channel_groups(carve)


def _sides(read_sparse, write_sparse):
    """The fold's and the readout's ``(lists, operand)`` (`_operand`), each side's coefficients the other's one list
    level where it has one."""
    D = len(read_sparse)
    write_lists = [lv for lv in range(D) if write_sparse[lv]]
    read_lists = [lv for lv in range(D) if read_sparse[lv]]
    fold = _operand(write_lists, [lv for lv in range(D) if not write_sparse[lv]],
                    coef=read_lists[0] if len(read_lists) == 1 else None)
    readout = _operand(read_lists, [lv for lv in range(D) if not read_sparse[lv]],
                       coef=write_lists[0] if len(write_lists) == 1 else None)
    return fold, readout


def _row_levels(read_sparse, write_sparse) -> list[int]:
    """Each side's row level where it lists none (its operand's first): the levels `_allocate` gives a tile first."""
    return sorted({operand[0] for lists, operand in _sides(read_sparse, write_sparse) if not lists})


def _build(widths, read_sparse, write_sparse, extents, channels: int) -> Carve:
    leaves = _product(range(len(widths)), extents)
    fold, readout = _sides(read_sparse, write_sparse)
    return Carve(widths=tuple(widths), extents=tuple(extents), channels=channels,
                 fold=_fold(*fold, extents, leaves), readout=_readout(*readout, extents))


def _channels(d_v: int) -> int:
    """The channel extent toward ``d_v``: the largest power of two dividing it, at most `MAX_CHANNELS`."""
    c = MAX_CHANNELS
    while d_v % c:
        c //= 2
    return c


def _admitted(carve_: Carve, read_sparse, write_sparse) -> bool:
    """Whether the box is one the kernels take: every listed level spans a list group the facts pass ranks
    (`LIST_GROUPS`, or its whole width), each side its own; a box with a listed side at most `LISTED_CHANNELS`; and each
    side's warps own a tile of dense rows (`MIN_DENSE`)."""
    listed = [(w, e) for w, r, wr, e in zip(carve_.widths, read_sparse, write_sparse, carve_.extents) if r or wr]
    spans = all(any(e == min(g, w) for g in LIST_GROUPS) for w, e in listed)
    channels = not listed or carve_.channels <= LISTED_CHANNELS
    return spans and channels and min(carve_.fold.tile_rows, carve_.readout.tile_rows) >= MIN_DENSE


def _declared(widths, read_sparse, write_sparse, d_v: int) -> None:
    if len(read_sparse) != len(widths) or len(write_sparse) != len(widths):
        raise ValueError(f"a carve needs both sides' flags a level: widths {widths}, read {read_sparse}, "
                         f"write {write_sparse}")
    for w in widths:
        if w < MIN_DENSE or w & (w - 1):
            raise ValueError(f"a shipped level width is a power of two at least {MIN_DENSE}, got {w}")
    for side, flags in zip(SIDES, (read_sparse, write_sparse)):
        if all(flags):
            raise ValueError(f"the {side} side declares every level sparse; every side has a dense level")
    if d_v < CHANNELS or d_v % CHANNELS:
        raise ValueError(f"d_v is a multiple of the box's {CHANNELS} channels, got {d_v}")


def plan_carve(widths, read_sparse, write_sparse, d_v: int) -> Carve:
    """Producer v1: the channel extent toward ``d_v`` (`_channels`), halved until the box is one the kernels take
    (`_admitted`: every listed level spans a facts list group, each side's dense rows an MMA tile); the box's
    leaves from the tags under the per-level cap, the cap lifted to the widths when the box cannot be filled under it.
    A pure function of the declaration,
    planned once a declaration (the `Carve` is frozen): every call of a shape after its first costs no host time
    (docs/internals/engine/carve.md#planned-once)."""
    return _planned(tuple(widths), tuple(read_sparse), tuple(write_sparse), int(d_v))


@cache
def _planned(widths, read_sparse, write_sparse, d_v: int) -> Carve:
    _declared(widths, read_sparse, write_sparse, d_v)
    channels = _channels(d_v)
    while True:
        total = _log2(BOX_VALUES // channels)
        bits = _allocate(widths, tags(read_sparse, write_sparse), total, CAP, _row_levels(read_sparse, write_sparse))
        if bits is None:
            bits = [_log2(w) for w in widths]
            while sum(bits) > total:
                bits[max(range(len(bits)), key=lambda lv: (bits[lv], lv))] -= 1
        carve_ = _build(widths, read_sparse, write_sparse, [1 << b for b in bits], channels)
        if channels == CHANNELS or _admitted(carve_, read_sparse, write_sparse):
            break
        channels //= 2
    check_carve(carve_, widths, read_sparse, write_sparse, d_v)
    return carve_


def check_carve(carve: Carve, widths, read_sparse, write_sparse, d_v: int) -> None:
    """Refuse a carve the declaration does not admit, naming what is wrong (the kernels refuse the same)."""
    _declared(widths, read_sparse, write_sparse, d_v)
    if carve.widths != tuple(widths) or len(carve.extents) != len(widths):
        raise ValueError(f"the carve is for widths {carve.widths}, the declaration's are {tuple(widths)}")
    for w, e in zip(widths, carve.extents):
        if e < 1 or e & (e - 1) or w % e:
            raise ValueError(f"an extent is a power of two dividing its level's width: {carve.extents} of {widths}")
    if (carve.channels < CHANNELS or carve.channels > MAX_CHANNELS or carve.channels & (carve.channels - 1)
            or d_v % carve.channels):
        raise ValueError(f"the channel extent is a power of two from {CHANNELS} to {MAX_CHANNELS} dividing d_v {d_v}, "
                         f"got {carve.channels}")
    if carve.leaves * carve.channels > BOX_VALUES:
        raise ValueError(f"the box holds {carve.leaves} x {carve.channels} values, over the state's {BOX_VALUES}")
    want = _build(widths, read_sparse, write_sparse, carve.extents, carve.channels)
    for side, plan in (("fold", carve.fold), ("readout", carve.readout)):
        if plan != getattr(want, side):
            raise ValueError(f"the {side}'s plan {plan} is not the box's {getattr(want, side)}")
        if plan.tile_rows < MIN_DENSE:
            raise ValueError(f"the {side}'s warps own {plan.tile_rows} dense rows, under an MMA tile of {MIN_DENSE}")
