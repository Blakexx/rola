# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts engine's value types: the call class, the mask it selects, the arm, the
carry's carve, the pins a caller may set over axes the op decides, and the refusal every
no-backward clause raises.

Every fact is an immutable value. Tensor-valued facts are held by REFERENCE and
nothing here allocates (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

#: the MMA k-extent, and the page granule. The Python mirror of
#: the MMA k-extent (16, the ampere atom's `m16n8k16`, `common/atoms/mma.cuh`), which a C++
#: ``constexpr`` cannot supply here; :meth:`rola.ops.paging.PageArena.__init__` checks
#: it against the atom geometry it addresses.
MMA_K_QUANTUM = 16


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only kernel surface."""


class Mask(IntEnum):
    """The call classes a DAG specializes over -- never a per-fact bool.

    Specialization is per CALL CLASS so the built set stays fixed and small; a bool
    per fact would be `2^n` of them, and intra-mask variation is a runtime-null
    output skip instead (`state_in is None`).
    """

    M1 = 1
    """PREFILL_STATELESS -- `state_in=0 state_out=0 paged=0`."""
    M2 = 2
    """PREFILL_STATEFUL -- `state_out=1`, `state_in` and `paged` free."""
    M3 = 3
    """DECODE_DENSE -- one token onto a dense plane: no probe, no gate, no admission."""
    M4 = 4
    """DECODE_PAGED -- one token onto an arena: the probe, the gate and the admission.

    A GROWTH STEP IS NOT A MASK. It is M4 run twice around an admission, which is why
    the growth path selects no different subgraph (docs/internals/engine/decode_dag.md).
    """


@dataclass(frozen=True, slots=True)
class CallClass:
    """What KIND of call this is, in the bits every downstream decision folds.

    These are ENGAGEMENT bits: `state_in` says this call was handed a state to
    continue, which is a different question from whether an entry-state TENSOR reaches
    the launch. The latter is `residency`'s answer, and it is what the skip predicate
    folds (`rola/engine/rules/run_table.py`).

    The mask these bits select is :func:`rola.engine.rules.mask.mask` -- a rule, not a
    property here: a decision folded from a fact belongs to band 2 wherever it is
    written, and one written as a method on the fact is a rule that no listing of
    `rola/engine/rules/` would show.
    """

    state_in: bool
    state_out: bool
    paged: bool
    grad: bool


@dataclass(frozen=True, slots=True)
class SidePlan:
    """One side's operation plan in the carry's box: the side's SPARSE levels index its lists and
    its DENSE levels its operand rows (formed in level order, the canonical product), and a warp
    covers ``tile_lists`` lists x ``tile_rows`` rows of the box -- the fold's warps a share of its
    leaves, the readout's every leaf for their channels (docs/internals/engine/rules.md#carve).
    """

    lists: tuple[int, ...]
    operand: tuple[int, ...]
    tile_lists: int
    tile_rows: int


@dataclass(frozen=True, slots=True)
class Carve:
    """The carry's carve: a CTA's box (an extent a level of the shipped ``widths``, and a channel
    extent) and each side's operation plan -- the fold the write side's, the readout the read
    side's. Every kernel that depends on the geometry takes this one descriptor
    (docs/internals/engine/rules.md#carve).
    """

    widths: tuple[int, ...]
    extents: tuple[int, ...]
    channels: int
    fold: SidePlan
    readout: SidePlan

    @property
    def leaves(self) -> int:
        """The box's leaves."""
        n = 1
        for e in self.extents:
            n *= e
        return n

    def ctas(self, d_v: int) -> int:
        """The CTAs a head: one a box position, a channel group."""
        n = d_v // self.channels
        for w, e in zip(self.widths, self.extents):
            n *= w // e
        return n


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """Pins over axes the op would otherwise decide.

    ``paging``: the state's BACKING, and the expert overrides the MAPPING ONLY.
    A stateful call pages by default -- one page is one MMA atom, pages are admitted
    by the write-side plan's exact bitmap, and an atom with no page reads as zeros, so
    paging is BITWISE INVISIBLE and there is nothing for a caller to turn on.
    ``None`` (the default) is that default; ``False`` pins the DENSE plane, which is
    what the bit-identity gate compares against; ``True`` is REFUSED, because a knob
    whose only value is the default is a knob that lies about being a decision.

    ``carve``: the carry's box and operation plans, pinned; the rule checks it against the
    declaration exactly as it checks its own (docs/internals/engine/rules.md#carve).

    A pin is engine vocabulary and lives here with the facts it folds into.
    :mod:`rola.expert` is its public address (docs/api.md §6), which is a re-export.
    """

    paging: bool | None = None
    carve: Carve | None = None

    def __post_init__(self) -> None:
        if self.paging is not None and type(self.paging) is not bool:
            raise TypeError(f"paging must be None or a bool, got {self.paging!r}")
        if self.carve is not None and type(self.carve) is not Carve:
            raise TypeError(f"carve must be None or a Carve, got {type(self.carve).__name__}")
        if self.paging is True:
            raise ValueError(
                "paging is ON by default for every stateful call, so there is nothing "
                "to turn on: pass paging=None (or omit it) for the paged backing, or "
                "paging=False to pin the dense plane. The expert overrides the MAPPING "
                "only (docs/internals/state.md).")

    @classmethod
    def of(cls, value: Any) -> PlanOverrides:
        """The overrides, or a hard failure naming what was passed instead."""
        if type(value) is cls:
            return value
        raise TypeError(
            "the `expert` argument takes a rola.expert.PlanOverrides -- the marked advanced "
            f"surface -- got {type(value).__name__}. Its axes are ones the op decides; "
            "pinning one is a deliberate act with a name.")


#: THE SUPPLIABLE SET IS CLOSED. A caller may substitute production -- the routing
#: bundle and the packed amplitude planes it stands for -- and nothing below it.
#: Everything downstream of that boundary is one canonical, un-overridable
#: implementation, gated once, so supplying one of those facts RAISES rather than
#: forking fact production.
SUPPLIABLE = frozenset({"routes", "read_plane", "write_plane"})
