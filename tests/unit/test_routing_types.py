"""Focused tests for the compact RouteDescriptor ABI."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from rola.routing.types import (
    READ_SUPPORT,
    ROUTE_DESCRIPTOR_VERSION,
    WRITE_SUPPORT,
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    RouteDescriptor,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
)


def _descriptor(plan, *, tokens=33):
    tiles = (tokens + 31) // 32
    values = torch.empty(2, tiles, plan.level_offsets[-1], 32, dtype=torch.bfloat16)
    support = torch.empty(
        2, tiles,
        sum(width for width, offset in zip(plan.branches, plan.support_offsets) if offset >= 0),
        dtype=torch.int32,
    )
    return RouteDescriptor(
        values, values, support,
        plan.level_offsets, plan.support_offsets, plan.branches,
        plan.radix_strides, plan.support_side_flags, 17, tokens, plan.topology_stamp,
    )


def test_union_routing_has_only_width_and_validated_alpha_and_fixed_split_semantics():
    union = UnionRouting(width=4)
    assert [f.name for f in fields(UnionRouting)] == ["width", "alpha"]
    assert union.alpha == 1.5
    assert UnionRouting(width=4, alpha=2.0) == UnionRouting(width=4, alpha=2.0)
    assert UnionRouting(width=4, alpha=2.0) != union
    assert not hasattr(union, "tied")
    assert not hasattr(union, "read")
    assert not hasattr(union, "write")


def test_tied_routing_has_only_width_and_one_validated_activation():
    tied = TiedRouting(width=4, op=EntmaxActivation(1.5))
    assert [f.name for f in fields(TiedRouting)] == ["width", "op"]
    assert tied.op == EntmaxActivation(1.5)
    assert tied.activation("read") is tied.activation("write") is tied.op
    assert TiedRouting(width=4, op=EntmaxActivation(1.5)) == tied
    assert TiedRouting(width=4, op=EntmaxActivation(2.0)) != tied
    with pytest.raises(ValueError, match="duty"):
        tied.activation("both")
    # A tied SOFTMAX level is legal and reports DENSE_BOTH (no exact zeros, simplex
    # output on both duties).
    dense_tied = TiedRouting(width=3, op=SoftmaxActivation())
    assert dense_tied.duty_simplex_output("read") is dense_tied.duty_simplex_output("write") is True
    assert not dense_tied.properties.exact_zeros


def test_tied_routing_at_another_width_keeps_the_activation():
    tied = TiedRouting(width=4, op=EntmaxActivation(1.5))
    assert tied.at(8) == TiedRouting(width=8, op=EntmaxActivation(1.5))


def test_resolved_routing_derives_compact_descriptor_metadata():
    plan = ResolvedRouting(
        (
            TiedRouting(width=2, op=SoftmaxActivation()),
            IndependentRouting(width=3, read=EntmaxActivation(2.0), write=SoftmaxActivation()),
            UnionRouting(width=5),
            IndependentRouting(width=4, read=EntmaxActivation(1.5), write=SoftmaxActivation()),
        ),
    )
    assert plan.level_offsets == (0, 2, 5, 10, 14)
    assert plan.radix_strides == (60, 20, 4, 1)
    assert plan.support_offsets == (-1, 0, 3, 8)
    assert plan.support_side_flags == (0, READ_SUPPORT, READ_SUPPORT | WRITE_SUPPORT, READ_SUPPORT)
    assert plan.topology_stamp == (
        ROUTE_DESCRIPTOR_VERSION, 4, 2, 3, 5, 4, 60, 20, 4, 1,
        0, READ_SUPPORT, READ_SUPPORT | WRITE_SUPPORT, READ_SUPPORT,
    )
    assert not hasattr(plan, "kernel_width")
    assert not hasattr(plan, "support_stream_map")
    assert not hasattr(plan, "support_stream_count")
    with pytest.raises(FrozenInstanceError):
        plan.levels = ()


def test_tied_routing_has_one_packed_slot_and_is_the_shared_solve_level():
    """A `TiedRouting` level has ONE packed slot -- excluded from `untied_levels` and
    counted in `shared_solve_levels` -- because it names one solve for both duties.
    `IndependentRouting` carries no shared-solve arm any more (that asymmetric case is
    deleted), so there is no longer a second construction to compare this against."""
    tied_plan = ResolvedRouting(
        (TiedRouting(width=4, op=EntmaxActivation(1.5)),
         UnionRouting(width=5)),
    )
    assert tied_plan.untied_levels == (1,)
    assert tied_plan.shared_solve_levels == (0,)
    assert tied_plan.packed_slot_widths == (4, 5, 5)
    assert tied_plan.read_level_can_zero == (True, True)
    assert tied_plan.write_level_can_zero == (True, True)


def test_route_descriptor_has_one_compact_tile_major_and_support_arena():
    plan = ResolvedRouting(
        (IndependentRouting(width=3, read=EntmaxActivation(1.5), write=SoftmaxActivation()),
         UnionRouting(width=5, alpha=2.0)),
    )
    descriptor = _descriptor(plan)
    assert descriptor.read_values.shape == descriptor.write_values.shape == (2, 2, 8, 32)
    assert descriptor.read_values.dtype == descriptor.write_values.dtype == torch.bfloat16
    assert descriptor.support_words.shape == (2, 2, 8)
    assert descriptor.support_words.dtype == torch.int32
    assert descriptor.read_values is descriptor.write_values
    assert descriptor.token_origin == 17
    assert descriptor.token_count == 33
    assert not hasattr(descriptor, "read")
    assert not hasattr(descriptor, "write")
    assert not hasattr(descriptor, "digits")
    assert not hasattr(descriptor, "support_stream_map")
    with pytest.raises(FrozenInstanceError):
        descriptor.token_count = 1


def test_route_descriptor_rejects_noncompact_or_inconsistent_metadata():
    values = torch.empty(1, 1, 4, 32, dtype=torch.bfloat16)
    support = torch.empty(1, 1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="topology_stamp"):
        RouteDescriptor(
            values, values, support, (0, 4), (0,), (4,), (1,), (READ_SUPPORT | WRITE_SUPPORT,),
            0, 1, (ROUTE_DESCRIPTOR_VERSION, 1, 4, 1, READ_SUPPORT),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("level_offsets", (0, True)),
        ("support_offsets", (0.0,)),
        ("level_widths", (4.0,)),
        ("radix_strides", (1.0,)),
        ("support_side_flags", (3.0,)),
        ("topology_stamp", (float(ROUTE_DESCRIPTOR_VERSION), 1, 4, 1, 3)),
    ],
)
def test_route_descriptor_requires_exact_integer_metadata(field, value):
    plan = ResolvedRouting((UnionRouting(width=4),))
    descriptor = _descriptor(plan, tokens=1)
    with pytest.raises(TypeError, match="exact ints"):
        replace(descriptor, **{field: value})


@pytest.mark.parametrize("field", ["token_origin", "token_count"])
def test_route_descriptor_rejects_non_exact_integer_token_metadata(field):
    plan = ResolvedRouting((UnionRouting(width=4),))
    descriptor = _descriptor(plan, tokens=1)
    with pytest.raises(ValueError, match="exact int"):
        replace(descriptor, **{field: True})


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: EntmaxActivation(True), TypeError),
        (lambda: EntmaxActivation(1), ValueError),
        (lambda: EntmaxActivation(1.8), ValueError),
        (lambda: UnionRouting(width=4, alpha=True), TypeError),
        (lambda: UnionRouting(width=4, alpha=float("nan")), ValueError),
        (lambda: UnionRouting(width=4, alpha=1), ValueError),
        (lambda: UnionRouting(width=4, alpha=1.8), ValueError),
        (lambda: IndependentRouting(width=4, read="softmax",
                                    write=SoftmaxActivation()), TypeError),
        (lambda: IndependentRouting(width=4, read=EntmaxActivation(1.5),
                                    write=EntmaxActivation(1.5)), ValueError),
        (lambda: IndependentRouting(width=4, read=EntmaxActivation(1.5),
                                    write=EntmaxActivation(2.0)), ValueError),
        (lambda: IndependentRouting(width=4, read=SoftmaxActivation(),
                                    write=SoftmaxActivation(), tied=False), TypeError),
        (lambda: TiedRouting(width=0, op=SoftmaxActivation()), ValueError),
        (lambda: TiedRouting(width=4, op="softmax"), TypeError),
        (lambda: ResolvedRouting(()), ValueError),
    ],
)
def test_invalid_routing_values_are_rejected(factory, error):
    with pytest.raises(error):
        factory()


@pytest.mark.parametrize("read_alpha,write_alpha", [(1.5, 1.5), (1.5, 2.0)])
def test_independent_routing_rejects_distinct_sparse_support_solves(read_alpha, write_alpha):
    with pytest.raises(ValueError, match="UnionRouting"):
        IndependentRouting(
            width=4,
            read=EntmaxActivation(read_alpha),
            write=EntmaxActivation(write_alpha),
        )


def test_sparse_branch_width_is_bounded_without_limiting_total_capacity():
    sparse = IndependentRouting(width=1, read=EntmaxActivation(1.5), write=SoftmaxActivation())
    with pytest.raises(ValueError, match="width must be <= 256"):
        sparse.at(257)
    assert ResolvedRouting(
        (sparse.at(256), sparse.at(256), sparse.at(256))).state_count == 256 ** 3


def test_packed_router_shape_validation_derives_heads():
    plan = ResolvedRouting(
        (IndependentRouting(width=2, read=SoftmaxActivation(), write=SoftmaxActivation()),
         UnionRouting(width=3)),
    )
    # Both levels are untied (an `IndependentRouting` level always is, and so is a
    # union level), so the packed slots are write-0, write-1, read-0, read-1 with widths
    # 2, 3, 2, 3 -- 10 columns, not the 4 * max(2, 3) = 12 the padded layout stored.
    assert plan.packed_router_width == 10
    route_W = torch.randn(2, 5, 10)
    route_bias = torch.randn(2, 10)
    assert plan.validate_packed_router(route_W, route_bias) == 2
    with pytest.raises(ValueError, match="packed_router_width"):
        plan.validate_packed_router(torch.randn(2, 4, 5, 3))
    with pytest.raises(ValueError, match="10"):
        plan.validate_packed_router(torch.randn(2, 5, 12))
    with pytest.raises(ValueError, match="route_bias"):
        plan.validate_packed_router(route_W, torch.randn(2, 12))


@pytest.mark.parametrize(
    "tied,expected_slot_widths",
    [
        ((True, True, True), (4, 2, 3)),
        ((False, False, False), (4, 2, 3, 4, 2, 3)),
        ((True, False, True), (4, 2, 3, 2)),
    ],
)
def test_packed_router_columns_are_exactly_the_columns_each_slot_consumes(tied, expected_slot_widths):
    """The stored router weight carries NO column that is not read.

    The padded layout gave every slot ``max(branches)`` columns, so a narrow level's
    slot carried dead parameters, dead optimizer moments and dead gradients. The packed
    layout gives each slot its own level's width, and the offsets are the running sum --
    which is what makes the write side (slots ``0..D-1``, in level order) the LEADING
    contiguous run, and therefore a slice rather than a gather.
    """
    plan = ResolvedRouting(
        tuple(
            TiedRouting(width=w, op=SoftmaxActivation()) if flag
            else IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation())
            for flag, w in zip(tied, (4, 2, 3))
        ),
    )
    assert plan.packed_slot_widths == expected_slot_widths
    assert plan.packed_slot_count == len(expected_slot_widths)
    offsets = plan.packed_slot_offsets
    assert offsets == tuple(
        sum(expected_slot_widths[:index]) for index in range(len(expected_slot_widths) + 1))
    assert plan.packed_router_width == sum(expected_slot_widths)
    # The write side is always one run, and it is the packed LOGIT layout verbatim.
    assert plan.packed_column_runs(tuple(range(plan.D))) == ((0, plan.packed_logit_width),)
    for level in range(plan.D):
        assert plan.packed_slot_slice(level) == plan.level_logit_slice(level)
    # A fully tied plan's read side is that same single run; a fully untied plan's read
    # side is the trailing run; only a mixed plan interleaves and needs a copy.
    runs = plan.packed_column_runs(plan.packed_read_slot_lookup)
    if not plan.untied_levels or len(plan.untied_levels) == plan.D:
        assert len(runs) == 1
    else:
        assert len(runs) > 1
    assert sum(stop - start for start, stop in runs) == plan.packed_logit_width
