import pytest
import torch

from rola.routing.projection import packed_router_logits
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
)


def _routing_for_untied(levels, untied_levels, branch):
    return ResolvedRouting(
        tuple(
            IndependentRouting(width=branch, read=SoftmaxActivation(), write=SoftmaxActivation())
            if level in untied_levels
            else TiedRouting(width=branch, op=SoftmaxActivation())
            for level in range(levels)
        ),
    )


def _slots(route_W, route_bias, routing, branch):
    """Split the packed column axis back into per-slot blocks.

    `route_W` is stored `[H, hidden, packed_router_width]` with slot `s` owning
    `packed_slot_slice(s)`. These fixtures use UNIFORM branch widths, so every slot owns
    `branch` columns and the split is an exact `view` -- which is itself the claim that
    the packed layout is slot-major on `packed_slot_offsets`, asserted below.
    """
    heads, hidden, columns = route_W.shape
    assert columns == routing.packed_slot_count * branch
    assert routing.packed_slot_offsets == tuple(
        slot * branch for slot in range(routing.packed_slot_count + 1))
    weight = route_W.view(heads, hidden, routing.packed_slot_count, branch)
    bias = None if route_bias is None else route_bias.view(heads, routing.packed_slot_count, branch)
    return weight, bias


def _full_read(weight, bias, routing):
    indices = list(routing.packed_read_slot_lookup)
    return weight[:, :, indices], None if bias is None else bias[:, indices]


def _reference(hidden, weight, bias, heads):
    equation = "bld,hdsc->bhlsc" if hidden.dim() == 3 else "blhd,hdsc->bhlsc"
    logits = torch.einsum(equation, hidden.float(), weight.float())
    if bias is not None:
        logits = logits + bias.float()[None, :, None]
    # The production layout is PACKED (`ResolvedRouting.packed_logit_width`): one side's
    # levels laid end to end on `level_offsets`, with no per-level padding to
    # `max(branches)`. These fixtures use uniform widths, so the slot axis flattens
    # exactly onto those offsets and the reference stays a plain reshape.
    return logits.reshape(
        hidden.shape[0] * heads, hidden.shape[1], weight.shape[2] * weight.shape[3])


def _expected_read(hidden, write_hidden, weight, bias, routing, expected_write, heads):
    if write_hidden is hidden:
        read_W, read_bias = _full_read(weight, bias, routing)
        return _reference(hidden, read_W, read_bias, heads)
    if not routing.untied_levels:
        return expected_write
    read_bias = None if bias is None else bias[:, routing.D:]
    untied = _reference(hidden, weight[:, :, routing.D:], read_bias, heads)
    slots = {level: slot for slot, level in enumerate(routing.untied_levels)}

    def block(source, position):
        width = routing.branches[0]
        return source[:, :, position * width:(position + 1) * width]

    return torch.cat([
        block(untied, slots[level]) if level in slots else block(expected_write, level)
        for level in range(routing.D)
    ], dim=2)


@pytest.mark.parametrize("untied_levels", [(), (0, 1, 2), (0, 2)])
@pytest.mark.parametrize("distinct_write_stream", [False, True])
def test_packed_router_matches_separate_projection_and_gradients(untied_levels, distinct_write_stream):
    torch.manual_seed(31)
    batch, length, heads, levels, hidden, branch = 2, 5, 2, 3, 7, 4
    routing = _routing_for_untied(levels, untied_levels, branch)
    h = torch.randn(batch, length, hidden, dtype=torch.float32, requires_grad=True)
    h_w = torch.randn_like(h, requires_grad=True) if distinct_write_stream else None
    route_W = torch.randn(heads, hidden, routing.packed_router_width, requires_grad=True)
    route_bias = torch.randn(heads, routing.packed_router_width, requires_grad=True)

    read_logits, write_logits = packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w)
    weight, bias = _slots(route_W, route_bias, routing, branch)
    write_hidden = h if h_w is None else h_w
    expected_write = _reference(write_hidden, weight[:, :, :levels], bias[:, :levels], heads)
    expected_read = _expected_read(
        h, write_hidden, weight, bias, routing, expected_write, heads)
    torch.testing.assert_close(read_logits, expected_read, rtol=0, atol=0)
    torch.testing.assert_close(write_logits, expected_write, rtol=0, atol=0)

    loss = read_logits.square().mean() + 0.7 * write_logits.square().mean()
    gradient_inputs = (h, route_W, route_bias) + (() if h_w is None else (h_w,))
    gradients = torch.autograd.grad(loss, gradient_inputs, allow_unused=True)
    expected_loss = expected_read.square().mean() + 0.7 * expected_write.square().mean()
    expected_gradients = torch.autograd.grad(
        expected_loss,
        gradient_inputs,
        allow_unused=True,
    )
    for actual, expected in zip(gradients, expected_gradients):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_tied_routing_reads_off_the_write_projection():
    """A `TiedRouting` level has ONE packed slot (`untied_levels` excludes it): its
    read logits are a VIEW of its write logits, not a second projected column block."""
    torch.manual_seed(7)
    batch, length, hidden = 2, 5, 6
    routing = ResolvedRouting(
        (TiedRouting(width=4, op=EntmaxActivation(1.5)),
         IndependentRouting(width=3, read=SoftmaxActivation(), write=SoftmaxActivation())),
    )
    assert routing.untied_levels == (1,)
    # slot widths: write-0 (tied, w=4), write-1 (w=3), read-1 (untied, w=3) -> 10 columns.
    assert routing.packed_router_width == 10
    h = torch.randn(batch, length, hidden, requires_grad=True)
    route_W = torch.randn(2, hidden, routing.packed_router_width, requires_grad=True)
    read_logits, write_logits = packed_router_logits(h, route_W, routing)
    tied_span = routing.level_logit_slice(0)
    torch.testing.assert_close(
        read_logits[..., tied_span], write_logits[..., tied_span], rtol=0, atol=0)


def test_packed_router_rejects_plan_shape_mismatches():
    routing = ResolvedRouting(
        (IndependentRouting(width=2, read=SoftmaxActivation(), write=SoftmaxActivation()),
         UnionRouting(width=3)),
    )
    h = torch.randn(1, 2, 3)
    # Both levels untied -> slot widths (2, 3, 2, 3), so 10 stored columns.
    assert routing.packed_router_width == 10
    with pytest.raises(ValueError, match="packed_router_width"):
        packed_router_logits(h, torch.randn(1, 4, 3, 3), routing)
    with pytest.raises(ValueError, match="10"):
        packed_router_logits(h, torch.randn(1, 3, 12), routing)


def test_packed_router_rejects_hidden_width_mismatch():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, 3, routing.packed_router_width)

    with pytest.raises(ValueError, match="hidden width"):
        packed_router_logits(torch.randn(1, 2, 4), route_W, routing)


def test_packed_router_rejects_rank4_head_count_mismatch():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, 3, routing.packed_router_width)

    with pytest.raises(ValueError, match="head count"):
        packed_router_logits(torch.randn(1, 2, 1, 3), route_W, routing)


def test_packed_router_rejects_mismatched_write_hidden():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, 3, routing.packed_router_width)
    h = torch.randn(1, 2, 3)

    with pytest.raises(ValueError, match="shape and device"):
        packed_router_logits(h, route_W, routing, h_w=torch.randn(1, 3, 3))


@pytest.mark.parametrize("untied_levels", [(), (0, 1, 2)])
def test_uniform_tie_state_returns_read_write_views_of_one_projection(untied_levels):
    routing = _routing_for_untied(3, untied_levels, 2)
    h = torch.randn(1, 2, 3)
    route_W = torch.randn(1, 3, routing.packed_router_width)
    read_logits, write_logits = packed_router_logits(h, route_W, routing)
    if not untied_levels:
        assert read_logits.untyped_storage().data_ptr() == write_logits.untyped_storage().data_ptr()


@pytest.mark.parametrize("untied_levels", [(), (0,), (0, 1, 2)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("split_stream", [False, True])
def test_the_two_projection_layouts_are_one_gemm_and_agree_at_the_bit(
        untied_levels, with_bias, split_stream):
    """P73 S5: `token_major=True` returns `[B, T, H, W]` instead of `[B*H, T, W]`.

    TWO claims, and they are separate. **The numbers are the same bytes**: a column of
    the output is a dot product over the full hidden dim, so it cannot depend on where
    the column is written -- `torch.equal`, not a tolerance, because a tolerance here
    would hide a real arithmetic change behind an "acceptable" one and this seam's whole
    licence is that it moves no number (`test_projection_tf32_support_gate.py` exists
    because a previous change to this GEMM's arithmetic flipped 14-95 entmax support
    entries).

    **And the token-major form is a VIEW of one GEMM**: it is contiguous, where the flat
    form needs a transpose of the whole plane to make `(B, H)` one axis. That is the
    0.085 ms this layout exists to delete, and asserting contiguity is what keeps a
    future edit from silently reintroducing the copy while still returning the right
    numbers.

    Swept over the three tie states (which decide whether the two sides are one GEMM,
    two, or a concatenation), the bias (a different broadcast per layout), and the split
    read/write stream.
    """
    from rola.routing.projection import _packed_router_logits

    torch.manual_seed(23)
    heads, batch, tokens, hidden = 3, 2, 5, 4
    routing = _routing_for_untied(3, untied_levels, 2)
    h = torch.randn(batch, tokens, hidden)
    h_w = torch.randn(batch, tokens, hidden) if split_stream else None
    route_W = torch.randn(heads, hidden, routing.packed_router_width)
    route_bias = torch.randn(heads, routing.packed_router_width) if with_bias else None

    def project(token_major):
        return _packed_router_logits(
            h, route_W, routing, route_bias=route_bias, h_w=h_w,
            projection_dtype=torch.float32, token_major=token_major)

    #: the third return is the gain span, absent here (`gain_columns` defaults to 0).
    flat = project(False)[:2]
    major = project(True)[:2]
    for side, (flat_side, major_side) in enumerate(zip(flat, major)):
        assert major_side.shape == (batch, tokens, heads, routing.packed_logit_width), (
            f"side {side}: token-major must be [B, T, H, W], got {tuple(major_side.shape)}")
        folded = major_side.permute(0, 2, 1, 3).reshape(
            batch * heads, tokens, routing.packed_logit_width)
        assert torch.equal(folded, flat_side), (
            f"side {side}: the two layouts disagree in "
            f"{int((folded != flat_side).sum())} of {flat_side.numel()} entries. They are "
            "the same dot products written to different addresses; any difference is a "
            "real arithmetic change, and the entmax support boundary is a threshold on "
            "these numbers.")

    #: THE SECOND CLAIM, asserted where it is unambiguous. A fully tied plan is ONE GEMM
    #: with no slicing afterwards, so "token-major is the GEMM's own output" means
    #: exactly contiguity there. The untied plans hand back HALF-SLICES of one GEMM,
    #: which are non-contiguous by construction and were already established as free to
    #: read: the solve takes them through its stride tuple.
    if not untied_levels:
        for side, major_side in enumerate(major):
            assert major_side.is_contiguous(), (
                f"side {side}: a fully tied plan's token-major projection must BE the "
                "GEMM's output; a non-contiguous one means the copy this layout exists "
                "to delete is back")
    #: And whatever the flat form shares between its two sides, the token-major form
    #: shares too -- so the layout change materialized nothing the flat one did not.

    def one_storage(sides):
        return sides[0].untyped_storage().data_ptr() == sides[1].untyped_storage().data_ptr()

    assert one_storage(major) == one_storage(flat), (
        "the token-major layout changed which sides share a storage, so it materialized "
        "something the flat layout did not")
