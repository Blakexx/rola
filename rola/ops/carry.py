# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY, as a caller reaches it. No carry kernel is built on this line: :func:`carry` raises
:class:`rola.ops._ext.KernelNotBuilt`, and :func:`arms` is the binary's own answer to what it carries (nothing).

What stands is the carve's vocabulary the planner and the envelope read: the carve
(:class:`rola.engine.types.Carve`) names the box -- an extent a level and the channel extent -- and each side's list
levels and operand levels, and its arm key (:func:`arm_key`) is what a compiled body is selected by.
"""
from __future__ import annotations

from rola.engine.types import Carve
from rola.ops import facts as facts_ops
from rola.ops._ext import extension, not_built

__all__ = ["arm_key", "arms", "carry", "lists_form", "same_split", "sm_clock_ghz"]


def _outer(plan) -> set[int]:
    """A side's levels outside its dense rows: its lists, or a side with none its coefficient levels."""
    return set(plan.lists if plan.lists else plan.operand[1:])


def same_split(carve: Carve) -> int:
    """Whether the fold's lists are the readout's digits (1: both sides list the same levels) or its K (0);
    a box whose sides share part of a level set has neither."""
    read = carve.readout
    fold, read_rows = _outer(carve.fold), set(read.operand if read.lists else read.operand[:1])
    if fold == _outer(read):
        return 1
    if fold == read_rows:
        return 0
    raise ValueError(f"the fold's lists {sorted(fold)} are neither the readout's digits nor its K "
                     f"{sorted(read_rows)}: no hand-off form carries them")


def arm_key(carve: Carve) -> tuple[int, ...]:
    """The compiled body a carve selects: ``(fold_rows, fold_factors, read_rows, read_factors, channels,
    same_split)`` -- each side's dense rows (the fold's warp tile, the readout's K) and its operand's level count, the
    channel extent and the hand-off's form (:func:`same_split`; ``tools/gen_shards.py``'s ``ARM_KEY``)."""
    return (carve.fold.tile_rows, len(carve.fold.operand), carve.readout.tile_rows, len(carve.readout.operand),
            carve.channels, same_split(carve))


def carry(routes, values, facts, carve: Carve, *args, **kwargs):
    """THE LAUNCH: not built on this line (:func:`rola.ops._ext.not_built`)."""
    not_built("carry")


def arms() -> list[tuple[int, ...]]:
    """The arms THIS binary carries, each its :func:`arm_key`: the binary's own answer, never the source tree's."""
    return [tuple(row) for row in extension().carry_arms()]


def lists_form(key: tuple[int, ...]) -> facts_ops.ListsForm:
    """The lists the built arm ``key`` reads, the facts pass's call: no arm is built on this line
    (:func:`rola.ops._ext.not_built`)."""
    not_built("carry")


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
