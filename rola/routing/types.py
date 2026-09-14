"""Canonical, ops-internal routing plan types.

These types deliberately describe routing execution rather than the layer-facing
configuration vocabulary.  In particular, union routing is a fixed operation,
not a parameterized activation-family choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, TypeAlias

import torch

from rola.routing.activations import (
    ActivationProperties,
    ActivationTag,
    properties_for,
    validate_alpha,
)
from rola.routing.topology import (
    padded_level_width,
    validate_level_width,
    validate_routing_shape,
)

ROUTE_DESCRIPTOR_VERSION = 2
ROUTE_TILE_SIZE = 32
READ_SUPPORT = 1
WRITE_SUPPORT = 2


def level_logit_slice(branches: tuple[int, ...], index: int) -> slice:
    """The column span level ``index`` occupies in a packed per-side logit tensor.

    THE definition of the packed router-logit layout: one side's levels laid end to
    end, level ``l`` owning ``branches[l]`` columns, with no padding to
    ``max(branches)``. Free-standing (rather than only a ``ResolvedRouting`` method)
    because ``diagnostics/telemetry.py`` is handed the branch widths alone and must
    read the same layout -- a second offset expression there would be a second
    definition of the layout, and two definitions drift.
    """
    start = sum(branches[:index])
    return slice(start, start + branches[index])


def activation_properties(activation: Any, *, owner: str = "activation") -> ActivationProperties:
    """The registry row for any activation object, by its `tag`.

    The ONE way any rule in this module reads an activation's properties. Nothing
    reads a property off the activation object itself, because then an object could
    disagree with the table and the table would stop being the source of truth.
    """
    tag = getattr(activation, "tag", None)
    if tag is None:
        raise TypeError(
            f"{owner} must be a routing activation carrying a registry `tag`, got {activation!r}")
    return properties_for(tag, owner=owner)


def _validated_alpha(alpha: float, owner: str) -> float:
    """Kept as the module's spelling of the check; the check itself is the registry."""
    return validate_alpha(alpha, owner=owner)


@dataclass(frozen=True, slots=True)
class SoftmaxActivation:
    """The dense routing activation. A plain tagged constructor.

    Carries NO property metadata -- properties come from
    `rola.routing.activations.ACTIVATION_REGISTRY`, keyed on :attr:`tag`. Metadata a
    caller could supply is metadata a caller could supply wrong, and a validator
    reading it would be checking a claim instead of a fact.
    """

    @property
    def tag(self) -> ActivationTag:
        return ("softmax", None)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="SoftmaxActivation")

    @property
    def can_zero(self) -> bool:
        """The `exact_zeros` column, under the name the rest of the tree uses."""
        return self.properties.exact_zeros


@dataclass(frozen=True, slots=True)
class EntmaxActivation:
    """An exact entmax routing activation, at a registry-ratified `alpha`.

    `alpha` is validated by REGISTRY LOOKUP (`("entmax", alpha)`), so the set of
    admissible alphas is the set of rows and there is no second list to keep in step.
    """

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha, owner="EntmaxActivation"))

    @property
    def tag(self) -> ActivationTag:
        return ("entmax", self.alpha)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="EntmaxActivation")

    @property
    def can_zero(self) -> bool:
        return self.properties.exact_zeros


def entmax(alpha: float = 1.5) -> EntmaxActivation:
    """User-facing constructor. `entmax(2)` is sparsemax."""
    return EntmaxActivation(alpha)


def softmax() -> SoftmaxActivation:
    """User-facing constructor for the dense activation."""
    return SoftmaxActivation()


RoutingActivation: TypeAlias = SoftmaxActivation | EntmaxActivation

# ``Activation`` is the spec-facing alias for this union. Same type.
Activation: TypeAlias = RoutingActivation


@dataclass(frozen=True, slots=True)
class RouteDescriptor:
    """Versioned compact routing ABI shared by every routing consumer.

    Values are physically tile-major: ``[BH, ceil(T/32), sum_width, 32]``.
    Support is packed once per zero-capable level as signed ``int32`` storage
    carrying uint32 bit patterns in
    ``[BH, ceil(T/32), sum_sparse_width]``.  There is deliberately no token-major
    view, padded branch width, digits table, or per-level support tensor in this
    contract.  A cleared support bit guarantees that the corresponding stored
    side factor is zero; a set bit may conservatively accompany a stored zero.
    """

    read_values: torch.Tensor
    write_values: torch.Tensor
    support_words: torch.Tensor
    level_offsets: tuple[int, ...]
    support_offsets: tuple[int, ...]
    level_widths: tuple[int, ...]
    radix_strides: tuple[int, ...]
    support_side_flags: tuple[int, ...]
    token_origin: int
    token_count: int
    topology_stamp: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, tensor in (
            ("read_values", self.read_values),
            ("write_values", self.write_values),
            ("support_words", self.support_words),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.read_values.ndim != 4 or self.write_values.ndim != 4:
            raise ValueError("route values must have rank 4")
        if self.support_words.ndim != 3:
            raise ValueError("support_words must have rank 3")

        offsets = tuple(self.level_offsets)
        support_offsets = tuple(self.support_offsets)
        widths = tuple(self.level_widths)
        strides = tuple(self.radix_strides)
        flags = tuple(self.support_side_flags)
        stamp = tuple(self.topology_stamp)
        d = len(widths)

        if d == 0 or len(offsets) != d + 1:
            raise ValueError("level_offsets must contain D + 1 entries for a nonempty routing plan")
        validate_routing_shape(widths, owner="RouteDescriptor")
        if any(type(value) is not int for value in (*offsets, *support_offsets, *widths, *strides, *flags, *stamp)):
            raise TypeError("RouteDescriptor metadata must contain exact ints")
        if len(support_offsets) != d or len(strides) != d or len(flags) != d:
            raise ValueError("RouteDescriptor metadata must contain one entry per routing level")
        if offsets[0] != 0 or any(end - start != width or width < 2 for start, end, width in zip(offsets, offsets[1:], widths)):
            raise ValueError("level_offsets must be compact slices matching level_widths >= 2")
        expected_support_offset = 0
        for offset, width, flag in zip(support_offsets, widths, flags):
            if flag not in (0, READ_SUPPORT, WRITE_SUPPORT, READ_SUPPORT | WRITE_SUPPORT):
                raise ValueError("support_side_flags entries must be READ=1 and/or WRITE=2")
            if flag == 0:
                if offset != -1:
                    raise ValueError("dense routing levels must use support offset -1")
            else:
                if offset != expected_support_offset:
                    raise ValueError("support offsets must compactly represent each sparse level once")
                expected_support_offset += width
        expected_strides = tuple(prod(widths[index + 1:]) for index in range(d))
        if strides != expected_strides:
            raise ValueError("radix_strides must be canonical MSB-first mixed-radix strides")
        if type(self.token_origin) is not int or self.token_origin < 0:
            raise ValueError("token_origin must be a nonnegative exact int")
        if type(self.token_count) is not int or self.token_count < 0:
            raise ValueError("token_count must be a nonnegative exact int")
        route_tiles = (self.token_count + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE
        expected_value_shape = (self.read_values.shape[0], route_tiles, offsets[-1], ROUTE_TILE_SIZE)
        if (
            self.read_values.dtype != torch.bfloat16
            or self.write_values.dtype != torch.bfloat16
            or self.read_values.shape != expected_value_shape
            or self.write_values.shape != expected_value_shape
            or not self.read_values.is_contiguous()
            or not self.write_values.is_contiguous()
        ):
            raise ValueError("route values must be contiguous bf16 [BH, route_tiles, sum_width, 32] arenas")
        if self.read_values.device != self.write_values.device:
            raise ValueError("read and write route arenas must share a device")
        expected_support_shape = (self.read_values.shape[0], route_tiles, expected_support_offset)
        if (
            self.support_words.dtype != torch.int32
            or self.support_words.shape != expected_support_shape
            or not self.support_words.is_contiguous()
            or self.support_words.device != self.read_values.device
        ):
            raise ValueError(
                "support_words must be contiguous int32 uint32-bit patterns "
                "[BH, token_words, sum_sparse_width]")
        expected_stamp = (ROUTE_DESCRIPTOR_VERSION, d, *widths, *strides, *flags)
        if stamp != expected_stamp:
            raise ValueError("topology_stamp does not match the RouteDescriptor ABI metadata")
        object.__setattr__(self, "level_offsets", offsets)
        object.__setattr__(self, "support_offsets", support_offsets)
        object.__setattr__(self, "level_widths", widths)
        object.__setattr__(self, "radix_strides", strides)
        object.__setattr__(self, "support_side_flags", flags)
        object.__setattr__(self, "topology_stamp", stamp)


@dataclass(frozen=True, slots=True)
class IndependentRouting:
    """One routing LEVEL: its branch width and its two per-duty activations.

    THE LEVEL'S WHOLE SPEC IS THIS OBJECT. ``width`` lives here rather than in a
    separate wrapper because every derived quantity a plan needs -- ``N``, the packed
    router width, the logit offsets, the topology -- is a function of the level LIST,
    and a list whose width and whose operator arrive in two different containers can be
    assembled inconsistently. The activations stay width-free registry keys: an
    activation is a function on a row of any length, so a width on one would be a
    second place the same number lives.
    """

    width: int
    read: RoutingActivation
    write: RoutingActivation

    def __post_init__(self) -> None:
        validate_level_width(self.width, owner="IndependentRouting")
        for name, activation in (
            ("read", self.read),
            ("write", self.write),
        ):
            # Admission is a REGISTRY lookup, not an isinstance list. An isinstance
            # list would be a second closed world, and a variant with a ratified row
            # would still be unconfigurable until someone remembered to widen it --
            # which is precisely the "one row and it participates in everything"
            # property the per-duty surface is built to give.
            activation_properties(activation, owner=f"IndependentRouting.{name}")
        if (
            activation_properties(self.read).exact_zeros
            and activation_properties(self.write).exact_zeros
        ):
            raise ValueError(
                "IndependentRouting only permits two exact-zero (`exact_zeros`) sides when "
                "the two duties are the SAME solve; use TiedRouting or UnionRouting for "
                "sparse/sparse routing. Stated against the registry column rather than "
                "against a list of activation names, so a new zero-capable variant is "
                "covered the moment its row exists")

    def activation(self, duty: str):
        """This level's activation for one DUTY, `'read'` or `'write'`.

        Per-level AND per-duty configuration is what makes level-split
        (`docs/open-work.md` M-3) pure configuration rather than a code path, so the
        duty axis gets a named accessor instead of two attribute reads that a caller
        has to remember the spelling of.
        """
        if duty == "read":
            return self.read
        if duty == "write":
            return self.write
        raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")

    def at(self, width: int) -> IndependentRouting:
        """This level's operator choice at another branch width."""
        return IndependentRouting(width=width, read=self.read, write=self.write)

    def duty_simplex_output(self, duty: str) -> bool:
        """Whether this cell's own output already lands on the simplex.

        Independent routing hands each duty one WHOLE solve, so the duty's answer is
        the activation's registry column unchanged.
        """
        return activation_properties(self.activation(duty)).simplex_output


@dataclass(frozen=True, slots=True)
class UnionRouting:
    """Exact entmax shared support followed by read/write role splitting.

    The INTERSECTION GUARANTEE -- that one solve determines a support both duties
    then split across -- requires the activation to produce EXACT structural zeros.
    That precondition is checked against the registry's `exact_zeros` column, not
    against a list of activation names, so a future zero-capable family is admitted
    by its row and a future non-zero-capable one is refused by its row.
    """

    width: int
    alpha: float = 1.5

    def __post_init__(self) -> None:
        validate_level_width(self.width, owner="UnionRouting")
        alpha = validate_alpha(self.alpha, owner="UnionRouting")
        object.__setattr__(self, "alpha", alpha)
        if not properties_for(("entmax", alpha), owner="UnionRouting").exact_zeros:
            raise ValueError(
                f"UnionRouting(alpha={alpha}) names an activation whose registry row says "
                "`exact_zeros=False`. Union routing IS the shared-support construction: "
                "without exact structural zeros there is no support to intersect, only "
                "small numbers, and the read/write split would be silently meaningless")

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(("entmax", self.alpha), owner="UnionRouting")

    def activation(self, duty: str) -> EntmaxActivation:
        """Union routing runs ONE solve, so both duties name the same activation."""
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return EntmaxActivation(self.alpha)

    def at(self, width: int) -> UnionRouting:
        """This level's operator choice at another branch width."""
        return UnionRouting(width=width, alpha=self.alpha)

    def duty_simplex_output(self, duty: str) -> bool:
        """Whether this cell's own output already lands on the simplex.

        The write duty receives the shared solve WHOLE, so it is the activation's
        registry column. The read duty receives a SPLIT of that solve's support and
        therefore a strict share of its mass: measured across the shipped census, the
        read sum runs as low as 7e-4, so the column does not transfer to it.
        """
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return duty == "write" and self.properties.simplex_output


@dataclass(frozen=True, slots=True)
class TiedRouting:
    """One routing LEVEL, ONE activation, ONE solve: the read factor IS the write factor.

    A first-class type in its own right: the level-type vocabulary is exactly
    ``IndependentRouting | TiedRouting | UnionRouting``, with tying named by which type
    a level IS rather than by a flag on ``IndependentRouting`` that happened to collapse
    its two independent solves into one. The stored field is named ``op`` rather than
    ``activation`` because :meth:`activation` is the per-duty ACCESSOR every routing
    level answers to (``Topology.activation`` calls ``level.activation(duty)``
    generically); a field and a method cannot share one name on a ``slots=True``
    dataclass, and the accessor is the name the rest of the tree already dispatches on.

    A tied SOFTMAX level is legal and is DENSE_BOTH: validation is a registry lookup
    (``activation_properties``), not a list of admissible activation names, so any row
    -- entmax or softmax -- is accepted the moment it exists in
    ``rola/routing/activations.py``.
    """

    width: int
    op: RoutingActivation

    def __post_init__(self) -> None:
        validate_level_width(self.width, owner="TiedRouting")
        activation_properties(self.op, owner="TiedRouting.op")

    @property
    def properties(self) -> ActivationProperties:
        return activation_properties(self.op, owner="TiedRouting")

    def activation(self, duty: str) -> RoutingActivation:
        """Tied routing runs ONE solve, so both duties name the same activation."""
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return self.op

    def at(self, width: int) -> TiedRouting:
        """This level's operator choice at another branch width."""
        return TiedRouting(width=width, op=self.op)

    def duty_simplex_output(self, duty: str) -> bool:
        """Whether this cell's own output already lands on the simplex.

        Tied routing hands both duties the SAME whole solve, so the duty's answer is
        the activation's registry column unchanged -- identical to
        :meth:`IndependentRouting.duty_simplex_output` on a shared solve.
        """
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return self.properties.simplex_output


LevelRouting: TypeAlias = IndependentRouting | TiedRouting | UnionRouting


@dataclass(frozen=True, slots=True)
class ResolvedRouting:
    """Validated routing topology and packed projection layout for one RoLA plan."""

    levels: tuple[LevelRouting, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("ResolvedRouting.levels must be nonempty")
        for level in levels:
            if type(level) not in (IndependentRouting, TiedRouting, UnionRouting):
                raise TypeError(f"ResolvedRouting.levels contains an invalid level: {level!r}")
        branches = tuple(level.width for level in levels)
        validate_routing_shape(branches, owner="ResolvedRouting")
        if any(branch < 2 for branch in branches):
            raise ValueError(f"ResolvedRouting branch widths must be >= 2, got {branches!r}")
        object.__setattr__(self, "levels", levels)

    @property
    def branches(self) -> tuple[int, ...]:
        """One branch width per level -- READ OFF the levels, never carried beside them.

        Every packed-layout property below is a function of this tuple, and the level
        list is where a width is declared; a second stored copy is a second place the
        same number can be wrong.
        """
        return tuple(level.width for level in self.levels)

    @property
    def D(self) -> int:
        return len(self.levels)

    @property
    def state_count(self) -> int:
        return prod(self.branches)

    @property
    def untied_levels(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, level in enumerate(self.levels)
            if not isinstance(level, TiedRouting)
        )

    @property
    def packed_slot_count(self) -> int:
        return self.D + len(self.untied_levels)

    @property
    def packed_read_slot_lookup(self) -> tuple[int, ...]:
        """Packed source slot for each level's read logits."""
        read_slots = {level: self.D + slot for slot, level in enumerate(self.untied_levels)}
        return tuple(read_slots.get(level, level) for level in range(self.D))

    @property
    def packed_slot_level(self) -> tuple[int, ...]:
        """The routing level each packed slot projects, one entry per slot.

        Slots ``0..D-1`` are the write slots, in level order; the remaining slots are
        the untied levels' read slots, also in level order. That ordering is what makes
        the write side's columns the leading contiguous block of ``route_W`` -- see
        :attr:`packed_router_width`.
        """
        return tuple(range(self.D)) + self.untied_levels

    @property
    def packed_slot_widths(self) -> tuple[int, ...]:
        """Consumed columns per packed slot: the width of the level that slot feeds."""
        return tuple(self.branches[level] for level in self.packed_slot_level)

    @property
    def packed_slot_offsets(self) -> tuple[int, ...]:
        """``packed_slot_count + 1`` cumulative column offsets into ``route_W``."""
        offsets = [0]
        for width in self.packed_slot_widths:
            offsets.append(offsets[-1] + width)
        return tuple(offsets)

    @property
    def packed_router_width(self) -> int:
        """Columns in the stored ``route_W`` -- every one of them consumed.

        THE definition of the router parameter's column layout. ``route_W`` is
        ``[H, hidden_size, packed_router_width]``: each packed slot owns exactly the
        ``branches[level]`` columns the level it feeds consumes, laid end to end on
        :attr:`packed_slot_offsets`, with NO padding to ``max(branches)``.

        The old layout stored ``[H, packed_slots, hidden_size, max(branches)]``, so every
        slot whose level was narrower than the widest carried dead columns: dead
        parameters, dead optimizer moments, dead gradient buffers, and a per-step
        ``torch.cat`` gather whose only job was to strip them before the GEMM. At the
        parity cell (H=8, hidden=512, branches (256, 64), untied -> 4 slots) that is
        ``4 * 256 = 1024`` stored columns per head where ``256 + 64 + 256 + 64 = 640``
        are consumed -- 37.5% dead. The waste GROWS with branch-width imbalance, so it
        is a scaling defect, not a constant.

        Because the write slots come first and in level order, the write side's columns
        are ``[0, packed_logit_width)`` -- one contiguous SLICE, so the write-side
        gather does not merely shrink, it disappears.
        """
        return self.packed_slot_offsets[-1]

    def packed_slot_slice(self, slot: int) -> slice:
        """The column span packed slot ``slot`` occupies in ``route_W``."""
        if type(slot) is not int or not 0 <= slot < self.packed_slot_count:
            raise ValueError(
                f"slot must be an exact int in [0, {self.packed_slot_count}), got {slot!r}")
        offsets = self.packed_slot_offsets
        return slice(offsets[slot], offsets[slot + 1])

    def packed_column_runs(self, slots: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """``slots``' column spans, with abutting spans merged into single runs.

        The projection gathers one side's slots into one weight. Merging is what turns
        that gather into a VIEW whenever the slots happen to be adjacent in storage --
        which, under this layout, is every structurally common case: the write side is
        always ``range(D)``, a fully tied plan's read side is the same run, and a fully
        untied plan's read side is the trailing run. Only a MIXED plan's read side
        interleaves the two regions and still needs a copy.
        """
        runs: list[list[int]] = []
        for slot in slots:
            span = self.packed_slot_slice(slot)
            if runs and runs[-1][1] == span.start:
                runs[-1][1] = span.stop
            else:
                runs.append([span.start, span.stop])
        return tuple((start, stop) for start, stop in runs)

    @property
    def shared_solve_levels(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, level in enumerate(self.levels)
            if isinstance(level, TiedRouting)
        )

    @property
    def read_level_can_zero(self) -> tuple[bool, ...]:
        """Exact-zero capability for every level on the read side.

        Read generically off ``level.activation('read')`` rather than an
        ``IndependentRouting``-only ``.read`` attribute, so ``TiedRouting`` (whose one
        activation answers both duties through the same accessor) needs no separate
        branch here.
        """
        return tuple(
            True if isinstance(level, UnionRouting)
            else activation_properties(level.activation("read")).exact_zeros
            for level in self.levels
        )

    @property
    def write_level_can_zero(self) -> tuple[bool, ...]:
        """Exact-zero capability for every level on the write side. See :attr:`read_level_can_zero`."""
        return tuple(
            True if isinstance(level, UnionRouting)
            else activation_properties(level.activation("write")).exact_zeros
            for level in self.levels
        )

    @property
    def level_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for width in self.branches:
            offsets.append(offsets[-1] + width)
        return tuple(offsets)

    @property
    def packed_logit_width(self) -> int:
        """Columns in ONE side's router-logit tensor -- tightly packed, no per-level padding.

        The router logits are packed on the same offsets the route arena already uses
        (``level_offsets``), NOT padded to ``max(branches)`` per level, and
        :attr:`packed_router_width` now packs the STORED weight on those same offsets
        too, so the parameter allocation and the compute layout are one layout rather
        than two related by a per-step gather.

        Equal to ``packed_slot_offsets[D]``: the write slots are ``0..D-1`` in level
        order, so one side's logit offsets ARE the leading column offsets of ``route_W``.
        """
        return self.level_offsets[-1]

    def level_logit_slice(self, index: int) -> slice:
        """The column span level ``index`` occupies in a packed per-side logit tensor.

        Delegates to the module-level :func:`level_logit_slice` so the production
        solve, the fp64 reference and telemetry all read ONE definition of the
        packed layout.
        """
        return level_logit_slice(self.branches, index)

    @property
    def radix_strides(self) -> tuple[int, ...]:
        return tuple(prod(self.branches[index + 1:]) for index in range(self.D))

    @property
    def support_side_flags(self) -> tuple[int, ...]:
        return tuple(
            (READ_SUPPORT if read_can_zero else 0) | (WRITE_SUPPORT if write_can_zero else 0)
            for read_can_zero, write_can_zero in zip(
                self.read_level_can_zero, self.write_level_can_zero)
        )

    @property
    def support_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        next_offset = 0
        for width, flag in zip(self.branches, self.support_side_flags):
            if flag == 0:
                offsets.append(-1)
            else:
                offsets.append(next_offset)
                next_offset += width
        return tuple(offsets)

    @property
    def topology_stamp(self) -> tuple[int, ...]:
        return (
            ROUTE_DESCRIPTOR_VERSION,
            self.D,
            *self.branches,
            *self.radix_strides,
            *self.support_side_flags,
        )

    @property
    def read_can_zero(self) -> bool:
        return any(self.read_level_can_zero)

    @property
    def write_can_zero(self) -> bool:
        return any(self.write_level_can_zero)

    def validate_packed_router(self, route_W: torch.Tensor, route_bias: torch.Tensor | None = None) -> int:
        """Validate packed router tensors and return their derived head count."""
        if not isinstance(route_W, torch.Tensor):
            raise TypeError(f"route_W must be a torch.Tensor, got {type(route_W).__name__}")
        if route_W.ndim != 3:
            raise ValueError(
                "route_W must have shape [H, hidden_size, packed_router_width], "
                f"got {tuple(route_W.shape)}")
        heads, _, width = route_W.shape
        if heads < 1 or width != self.packed_router_width:
            raise ValueError(
                "route_W must have shape "
                f"[H>=1, hidden_size, {self.packed_router_width}], "
                f"got {tuple(route_W.shape)}")
        if route_bias is not None:
            if not isinstance(route_bias, torch.Tensor):
                raise TypeError(f"route_bias must be a torch.Tensor or None, got {type(route_bias).__name__}")
            expected_bias_shape = (heads, self.packed_router_width)
            if route_bias.shape != expected_bias_shape:
                raise ValueError(
                    f"route_bias must have shape {expected_bias_shape}, got {tuple(route_bias.shape)}")
        return heads


# ---------------------------------------------------------------------------
# The config contract's routing types.
#
# ``Activation`` (above), ``LevelRouting`` (below, already ``Independent |
# Union``), ``IndependentRouting``, and ``UnionRouting`` are reused verbatim --
# they already satisfy the contract's shapes. Only the topology/decay
# wrappers are new. These are producer-agnostic: they describe the routing
# contract the naive oracle (``rola/ops/naive.py``) consumes, not a
# projection ABI. They deliberately do not reuse ``ResolvedRouting`` /
# ``RouteDescriptor``, which are packed-router / kernel ABI types tied to the
# earlier Q/K-shaped producer and are not part of this contract.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OpaqueActivation:
    """The activation of a cell this package did not solve.

    A producer is duck-typed, so a level may be filled by a map with no registry row of
    its own. Its cell still has to answer every question the rules ask -- density,
    ratification, normalization -- and it answers them from ONE row, ``("opaque",
    None)``, whose columns are argued in ``rola/routing/activations.py``.
    """

    @property
    def tag(self) -> ActivationTag:
        return ("opaque", None)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="OpaqueActivation")

    @property
    def can_zero(self) -> bool:
        return self.properties.exact_zeros


@dataclass(frozen=True, slots=True)
class OpaqueRouting:
    """The routing tag of a level produced by a foreign producer.

    It carries the level's WIDTH and nothing else -- there is no operator for the
    library to configure, because the producer owns the map, but the width is a
    structural fact the topology derives ``N`` and the radix strides from. Both duties
    report the opaque activation, so
    every rule that quantifies over ``(level, duty)`` cells covers a foreign level
    without having to know it is one.
    """

    width: int

    def __post_init__(self) -> None:
        validate_level_width(self.width, owner="OpaqueRouting")

    def activation(self, duty: str) -> OpaqueActivation:
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return OpaqueActivation()

    def at(self, width: int) -> OpaqueRouting:
        """The same opaque declaration at another branch width."""
        return OpaqueRouting(width=width)

    def duty_simplex_output(self, duty: str) -> bool:
        """The opaque row's column: a foreign producer's output is never vouched for."""
        return activation_properties(self.activation(duty)).simplex_output


LevelRouting: TypeAlias = IndependentRouting | TiedRouting | UnionRouting | OpaqueRouting


@dataclass(frozen=True, slots=True)
class Topology:
    """The routing topology, and ONLY routing.

    ``D = len(levels)`` is validated in ``[1, 4]`` and every level's width in
    ``[1, 256]`` by ``validate_routing_shape`` -- the same validator the earlier form's
    ``ResolvedRouting`` uses, so the two structural invariants are not
    reimplemented here.

    **The value width is not here, and that is the point.** ``d_v`` is the width of
    the state each leaf holds, which is a fact about the VALUE stream: the layer
    sizes ``v_proj``/``o_proj`` with it and the op reads it off ``v.shape[-1]`` at
    call time. Nothing on this object ever used it -- it was validated here and read
    from here by five consumers that all had ``v`` in hand.
    """

    levels: tuple[LevelRouting, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        for index, level in enumerate(levels):
            if type(level) not in (IndependentRouting, TiedRouting, UnionRouting, OpaqueRouting):
                raise TypeError(
                    f"Topology.levels[{index}] must be an IndependentRouting, TiedRouting, "
                    f"UnionRouting or OpaqueRouting -- the level's whole spec, width included -- "
                    f"got {level!r}")
        validate_routing_shape(tuple(level.width for level in levels), owner="Topology")
        object.__setattr__(self, "levels", levels)

    @property
    def D(self) -> int:
        return len(self.levels)

    @property
    def widths(self) -> tuple[int, ...]:
        """The LOGICAL, caller-authored per-level widths (``b_l``) -- unchanged by
        padding. A model built with ``IndependentRouting(width=31, ...)`` reads
        ``31`` here forever; see :attr:`padded_widths` and :meth:`padded` for the
        kernel-facing, padded ones (``B_l``, padded at the producer)."""
        return tuple(level.width for level in self.levels)

    @property
    def padded_widths(self) -> tuple[int, ...]:
        """Per-level ``B_l``: the next power of two at or above 16 that admits ``b_l``.

        The KERNEL-FACING widths (``StateFormat.B``'s law): every ``B_l`` this
        property names is already a legal ``rola._state.StateFormat`` level width.
        Derived from :attr:`widths` alone -- never a second input a caller supplies,
        so a padded width can never disagree with the logical one it pads.
        """
        return tuple(padded_level_width(width) for width in self.widths)

    @property
    def is_padded(self) -> bool:
        """Whether every level is ALREADY at its padded width -- the dense fast path.

        ``widths == padded_widths`` for an ideal (power-of-two >= 16, every level)
        model: :meth:`padded` still returns an equivalent ``Topology`` in that case,
        so a caller may always call it uniformly (KERNEL_STANDARDS §R9: a structure
        branch is a derived fact, never a data-dependent shortcut) -- this property
        exists for callers that want to skip a genuinely zero-cost pad.
        """
        return self.widths == self.padded_widths

    def padded(self) -> Topology:
        """A new ``Topology``, one level's ``.at(padded_width)`` per level.

        The SAME model described at its kernel-facing widths: same per-level
        operator and activation choice (``IndependentRouting``, ``UnionRouting`` and
        ``OpaqueRouting`` all carry an ``.at(width)`` that keeps everything but the
        width), so the levels this returns describe "the b real digits plus dead pad
        digits", never a different model. A convenience for a caller building the
        padded-shape SIDE of an equivalence directly (``tests/oracle/
        test_padding_seam.py``'s fp64 prefill proof); the op call surfaces that
        actually pad a live call (``rola.ops.carry``/``decode``/``intra``) derive
        their own padded widths per call instead, via ``rola.ops.padding.PaddedShape``
        -- they need the value width alongside the levels, which a ``Topology`` alone
        does not carry.
        """
        return Topology(levels=tuple(
            level.at(padded_level_width(level.width)) for level in self.levels))

    @property
    def N(self) -> int:
        """Leaf capacity: ``N = prod_l width_l`` -- derived, never configured."""
        return prod(self.widths)

    @property
    def radix_strides(self) -> tuple[int, ...]:
        """MSB-first mixed-radix strides: ``stride_l = prod(width_{l+1..D-1})``."""
        widths = self.widths
        return tuple(prod(widths[index + 1:]) for index in range(len(widths)))

    # --- the per-level per-duty activation surface, read generically ----------

    def activation(self, level: int, duty: str):
        """The activation configured for one `(level, duty)` cell."""
        return self.levels[level].activation(duty)

    def activation_properties(self, level: int, duty: str) -> ActivationProperties:
        """That cell's registry row -- the ONLY source of activation properties."""
        return activation_properties(
            self.activation(level, duty), owner=f"Topology.levels[{level}].{duty}")

    def duty_simplex_output(self, level: int, duty: str) -> bool:
        """Whether that cell's own map already lands on the simplex.

        THE fold's licence to skip a level. It is a routing-form
        question and not only an activation one, because a routing form may hand a
        duty a share of a solve rather than the whole of it.
        """
        return self.levels[level].duty_simplex_output(duty)

    def duty_cells(self) -> tuple[tuple[int, str], ...]:
        """Every `(level, duty)` cell, so a rule can quantify over the whole grid
        instead of over a hand-written pair of loops per rule."""
        return tuple((level, duty) for level in range(self.D) for duty in ("read", "write"))

    @property
    def unratified_activations(self) -> tuple[tuple[int, str, ActivationTag], ...]:
        """The cells whose registry row says `kerneled=False`.

        A config naming one of these is a LABELED RESEARCH SURFACE -- it runs
        on the reference path carrying an explicit `not_kerneled` flag. It is not a
        kernel fallback, because the kernel is never asked and never declines;
        `launch_consumer` remains the single authority on supported shapes.
        """
        return tuple(
            (level, duty, self.activation(level, duty).tag)
            for level, duty in self.duty_cells()
            if not self.activation_properties(level, duty).kerneled
        )

    @property
    def kerneled(self) -> bool:
        """Whether every configured activation has a ratified kernel row."""
        return not self.unratified_activations


@dataclass(frozen=True, slots=True)
class LeafMassDecay:
    """Complete-leaf write-conditioned decay dials.

    ``dials[l]`` has shape ``[H, width_l]`` with entries ``delta_l[d] in [0, 1)``.
    The upper bound is strict and load-bearing: it keeps
    ``rate_leaf[s] = prod_l delta_l[d_l(s)] < 1`` for every leaf ``s``, so
    ``keep = (1 - rate_leaf) ** c = 0`` is unreachable by construction.

    **This check is necessary but NOT sufficient.** It runs in
    ``dial``'s own dtype, which may be fp64; the kernel only ever reads an fp32
    cast of these dials, and an fp64 value within ~3e-8 of 1 is strictly ``< 1``
    here yet rounds to exactly ``1.0f`` on cast, which NaNs the kernel
    (``log1pf(-1.0f) = -inf``, then ``expf(0 * -inf)``). The invariant is
    re-checked in fp32 at the cast boundary in
    ``rola/internal/decay.py::leaf_rate_dials`` -- that second check, not this
    one, is what the kernel's correctness actually depends on.
    """

    dials: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        dials = tuple(self.dials)
        if not dials:
            raise ValueError("LeafMassDecay.dials must be nonempty")
        heads = None
        for level, dial in enumerate(dials):
            if not isinstance(dial, torch.Tensor):
                raise TypeError(
                    f"LeafMassDecay.dials[{level}] must be a torch.Tensor, got {type(dial).__name__}")
            if dial.ndim != 2:
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] must have shape [H, width_l], "
                    f"got {tuple(dial.shape)}")
            if heads is None:
                heads = dial.shape[0]
            elif dial.shape[0] != heads:
                raise ValueError("LeafMassDecay.dials must share one head count H across levels")
            if torch.any(dial < 0) or torch.any(dial >= 1):
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] entries must satisfy the strict invariant "
                    "0 <= delta < 1 (upper bound strict; keeps rate_leaf < 1 so keep=0 is "
                    "unreachable by construction)")
        object.__setattr__(self, "dials", dials)

    def validate_against(self, topology: Topology) -> None:
        """Check level count, per-level width, and the WRITE-side mass precondition.

        **Mass decay requires a `normalized` write side.** The clock is
        `c[t,s] = prod_l p_write_stored[t, d_l(s)]` and the survival factor is
        `(1 - rate_leaf)**c` -- an expression that prices a decay per unit of MASS
        DEPOSITED. If the write side does not land on the simplex, `c` is no longer
        a unit-mass share and the same routing decays by an amount that depends on
        the un-normalized scale of the logits, which is not a modeling choice anyone
        made. Stated against the registry column, so it covers every present and
        future activation without naming one.
        """
        widths = topology.widths
        unnormalized = [
            (level, topology.activation(level, "write").tag)
            for level in range(topology.D)
            if not topology.activation_properties(level, "write").normalized
        ]
        if unnormalized:
            raise ValueError(
                f"LeafMassDecay requires a `normalized` write side on every level; "
                f"levels {unnormalized} name activations whose registry row says "
                "`normalized=False`. The decay clock prices survival per unit of "
                "DEPOSITED MASS, so an un-normalized write side makes the amount "
                "decayed depend on the scale of the logits rather than on the "
                "routing. Configure a normalized write activation, or do not "
                "configure mass decay")
        if len(self.dials) != len(widths):
            raise ValueError(
                f"LeafMassDecay.dials must contain one tensor per level (D={len(widths)}), "
                f"got {len(self.dials)}")
        for level, (dial, width) in enumerate(zip(self.dials, widths)):
            if dial.shape[-1] != width:
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] must have width {width} matching "
                    f"Topology.levels[{level}].width, got {dial.shape[-1]}")


# ``kind: None | LeafMassDecay``. ``FactorizedMassDecay`` was removed and must not be
# reintroduced here.
DecayConfig: TypeAlias = None | LeafMassDecay
