# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The two paging decisions: which backing a state binds to, and where a run lands.

AdmissionRule is the one rule that splits (design.md §B2.3): everything that MOVES
BYTES is the arena's node, and the placement arithmetic -- the extent alignment and
the all-or-nothing ceiling -- is this fold. The reset-vs-union half stays at the
arena, because the reset CLEARS THE PAGE TABLE and the table is what sizes the run
this rule then places (docs/internals/engine/rules.md).
"""
from __future__ import annotations

from typing import Any

from rola.engine import PlanOverrides, rule


@rule(gives=("paging",), needs=("expert",))
def paging_preference(expert: Any) -> bool:
    """Whether a state bound by this call gets a PAGED backing. Default: yes.

    The expert may set ``paging=False`` and only False -- ``paging=True`` is refused
    by :class:`rola.engine.types.PlanOverrides` itself, because there is nothing to
    turn on. That is the "expert overrides the MAPPING only" ruling, made mechanical.
    """
    if expert is None:
        return True
    return PlanOverrides.of(expert).paging is not False


@rule(gives=("first_slot",), needs=("allocated", "capacity", "extent_atoms", "count"))
def admission(allocated: int, capacity: int, extent_atoms: int, count: int) -> int:
    """The first slot a run of `count` atoms takes, ALIGNED when it opens an extent.

    **ALL OR NOTHING**: a demand the ceiling cannot serve RAISES, naming the demand
    and the arena size, and moves nothing. There is no partial grant -- a prefix grant
    silently computes a different function.

    The alignment rule is one line and is the whole policy: a batch that fits the
    current extent's remainder goes there (so small admissions do not waste a whole
    extent), and a batch that does not starts a new extent (so a large admission -- an
    owner's ``QPO`` atoms, or a decode step's pre-growth run -- is contiguous rather
    than straddling). Both branches are STRUCTURAL: they compare a count against a
    remainder, never a size against a threshold.
    """
    if count <= 0:
        return allocated
    remainder = (-allocated) % extent_atoms
    first = allocated + remainder if (remainder and count > remainder) else allocated
    if first + count > capacity:
        raise RuntimeError(
            f"the page arena cannot serve this launch: {count} new atoms demanded, "
            f"{capacity - first} free in an arena of {capacity} atoms. "
            "The arena grants ALL or refuses -- there is no partial grant and no "
            "dropped write. The capacity IS the dense limit, so a demand it cannot "
            "serve was not derived from this arena's own keying.")
    return first
