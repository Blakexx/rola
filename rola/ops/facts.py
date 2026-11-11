# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS, as a caller reaches it: one launch over the bundle's rows for both sides' class-1 words, the
window's columns and each side's lists, an entry a slot and a byte.

The words' LAYOUT, their reader, the folds over them and the pure-torch MODEL the kernel is graded against live in
:mod:`rola.engine.facts.liveness`; the lists' model in :mod:`rola.engine.facts.lists`. What is here is the call: the
layout and the two sides' static operands a bundle's topology implies (:func:`call_layout`, :func:`call_statics`),
packed into the bits R13 requires so no logical width crosses the seam, and the launch
(docs/internals/facts/facts_api.md). The lists carry nothing of a consumer's but each side's list group
(:class:`ListsForm`): the digits one union spans, a box's extent along the side's list level.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import lists as list_model
from rola.engine.facts.liveness import LivenessLayout, SideStatics, mask_words, side_statics
from rola.ops._ext import extension

__all__ = [
    "COMPOSITE",
    "LEDGER_PHASES",
    "LEDGER_WARPS",
    "WHOLE_GROUPS",
    "Facts",
    "ListsForm",
    "call_layout",
    "call_statics",
    "facts",
    "ledger_bind",
    "list_level",
    "side_lists",
]


#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `FactsPhase` order, booked over every CTA.
LEDGER_PHASES = ("load", "vote", "ranks", "lists")
LEDGER_WARPS = 8


class Facts(NamedTuple):
    """One call's facts, a write head's (``BH = B H_w``): ``words`` the ``[BH, g + 1, rows, ceil(L / 32)]`` int32
    class-1 table (:class:`LivenessLayout`; slots ``0 .. g - 1`` the query heads' read side, ``g`` the write side;
    ``[BH, 2, ...]`` ungrouped), ``unions`` the ``[BH, g + 1, groups, ceil(L / 64)]`` int32 tile unions
    (``tile_unions``), and ``fm`` the ``[BH, L, 2]`` fp32 factor products, each token's ``(F, m)`` = ``(m s, m)``,
    ``m`` the write levels' scales multiplied (a write gain among them) and ``s`` the value scale. A DENSE level's rows and groups
    of ``words`` and ``unions`` are left unwritten: they are its width mask over the live tokens, which a reader takes
    from the side's statics. ``write`` and ``read`` each side's lists at its list level
    (:class:`rola.engine.facts.lists.Lists`; width 0 for a side with no sparse level, ``None`` for a composite side,
    :func:`side_lists`) (docs/internals/facts/facts.md#lists-out)."""

    words: torch.Tensor
    unions: torch.Tensor
    fm: torch.Tensor
    write: list_model.Lists | None
    read: list_model.Lists | None


class ListsForm(NamedTuple):
    """Each side's LIST GROUP -- the digits one union and its prefix rows span, a box's extent along the side's list
    level (:data:`rola.engine.facts.lists.LIST_GROUPS`; docs/internals/facts/facts.md#list-groups)."""

    read_list_group: int
    write_list_group: int


#: each side's lists in whole 32-digit groups, a liveness group a union
WHOLE_GROUPS = ListsForm(read_list_group=32, write_list_group=32)


def call_layout(routes) -> LivenessLayout:
    """The layout a bundle's words are read in: every level at its padded width, the call's tokens."""
    widths = routes.topology.padded_widths
    return LivenessLayout(D=len(widths), B=widths, L=routes.tokens)


def call_statics(routes, layout: LivenessLayout) -> tuple[SideStatics, SideStatics]:
    """``(read, write)``: each side's static operands as the bundle declares them -- a level not flagged ``sparse`` is
    DENSE (neither read nor written: its words are the width mask), and the rows' own widths are the logical ones."""
    return tuple(side_statics(layout, tuple(not flag for flag in flags), routes.topology.widths)
                 for flags in routes.sparse)


#: A SIDE WITH TWO OR MORE SPARSE LEVELS: its lists are composite (keyed by a digit of each), which this pass does not
#: build (:func:`side_lists` refuses them).
COMPOSITE = -2


def list_level(statics: SideStatics) -> int:
    """A SIDE'S LIST LEVEL: its one sparse level (the carve's ``lists`` for the side, by definition:
    docs/internals/engine/carve.md); ``-1`` for a side with none, whose every token is a slot; :data:`COMPOSITE` for a
    side with two or more."""
    sparse = [level for level, dense in enumerate(statics.dense_levels) if not dense]
    return sparse[0] if len(sparse) == 1 else (-1 if not sparse else COMPOSITE)


def side_lists(facts: Facts, side: str) -> list_model.Lists:
    """``side``'s (``"write"`` or ``"read"``) lists, refusing a side the pass built none for because they are composite
    (docs/internals/facts/facts.md#lists-out)."""
    lists = getattr(facts, side)
    if lists is None:
        raise ValueError(f"the facts pass builds no composite lists: the {side} side has two or more sparse levels, "
                         "and its lists would be keyed by a digit of each")
    return lists


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the kernel skips reads on."""
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def facts(routes, values, layout: LivenessLayout, read_statics: SideStatics, write_statics: SideStatics,
          form: ListsForm) -> Facts:
    """THE PASS: both sides' words and tile unions, the window's columns and each side's lists from the bundle
    (``routes``, ``values``) in ONE launch, each side's lists in ``form``'s list group.

    Same words at every sparse level, bit for bit, as :func:`rola.engine.facts.liveness.liveness_words` over the
    bundle's rows, and their unions as :func:`rola.engine.facts.liveness.tile_unions`; the lists
    :func:`rola.engine.facts.lists.final_lists` at each side's list
    group, bit for bit. The rows are read where the bundle holds them, any offset or stride of the format, each level's
    ``[B, T, H, w_l]`` view.

    GROUPED QUERIES (docs/api.md#grouped-queries): everything is a write head's -- ``words`` and ``unions``
    ``[B H_w, g + 1, ...]`` (slots ``0 .. g - 1`` its query heads' read side, slot ``g`` its write side), the columns,
    the write lists -- and its read lists one union a (window, group) over its g query heads' rows.
    """
    level, read_level = list_level(write_statics), list_level(read_statics)
    count = mask_words(layout.rows)
    words, unions, fm, *lists = extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), values.scale,
        list(layout.B), _dense_bits(read_statics), [read_statics.mask_word(i) for i in range(count)],
        _dense_bits(write_statics), [write_statics.mask_word(i) for i in range(count)], max(level, -1),
        max(read_level, -1), form.read_list_group, form.write_list_group)
    write, read = list_model.Lists(*lists[:5]), list_model.Lists(*lists[5:])
    return Facts(words=words, unions=unions, fm=fm, write=None if level == COMPOSITE else write,
                 read=None if read_level == COMPOSITE else read)


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/facts/facts.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles, or
    ``None``."""
    extension().rola_facts_ledger_bind(ledger)
