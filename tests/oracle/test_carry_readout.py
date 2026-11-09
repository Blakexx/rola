"""Gate for THE CARRY'S READOUT (csrc/rola/src/carry/ampere/readout.cuh, the value carve): each slot of ``num`` is the fp64
reference of the readout's declared arithmetic (`tests/oracle/readout_reference.py`: the image at each digit's power of
two, each (token, digit) piece rounded into the panel at the token's scale byte, the CTAs summed in the call's reduce
dtype) within its bound, at both orders of the even box's levels, over four windows, at both reduce dtypes
(docs/internals/carry/ampere/readout.md).

Symbols: ``W`` the window (the device's), ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import carve_at, plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from tests.oracle import carry_plan
from tests.oracle.carry_fixtures import (
    FORM_CELLS,
    FORM_CELLS64,
    assert_box64_arm,
    box64,
    box64_dense,
    box64_same,
    dense_bundle,
    listed_bundle,
    read_listed_bundle,
)
from tests.oracle.readout_reference import num_absolute, readout, units

#: THE EVEN BOX, 32 x 32 x 32 (two channel boxes at d_v 64)
EVEN_BOX = ((32, 32), 32)


def _readout_ratio(widths, L, write_sparse, growth, accumulate, *, same=False, live=4, forced=None,
                   reduce_dtype=torch.float16, box=None, dense_write=False, dense=False):
    """``(ratio, plans)``: the call's largest slot distance from the reference over its bound, and its boxes' plans; at
    the planner's box, or at ``box`` (extents a level, channels); ``dense_write`` a write side dense at every level and
    the read side listing level ``write_sparse``; ``dense`` both sides dense at every level (the planner's box)."""
    if dense:
        routes, values = dense_bundle(widths, L)
    elif dense_write:
        routes, values = read_listed_bundle(widths, L, write_sparse, live=live)
    else:
        routes, values = listed_bundle(widths, L, write_sparse, same=same, live=live)
    at = torch.arange(L, device="cuda") // window_of()
    routes.scale_write[0].mul_(torch.exp2(growth * at.float())[None, :, None])
    widths, d_v = tuple(routes.topology.padded_widths), values.v.shape[-1]
    #: the 32-channel box unless named (at d_v 64 the planner's is the 64-channel one, gated by its own tests); a dense
    #: read side's the planner's
    carve = (plan_carve(widths, *routes.sparse, d_v) if dense
             else carve_at(widths, *routes.sparse, *(box or EVEN_BOX), d_v))
    assert carry_ops.arm_key(carve) in carry_ops.arms(), carry_ops.arm_key(carve)
    key = carry_ops.arm_key(carve)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.lists_form(key))
    #: level S the readout's digits: a dense read side's its operand's second level (`rola.engine.carve._readout`)
    level_s = (carve.readout.operand[1] if dense
               else write_sparse if same or dense_write else 1 - write_sparse)
    scale = units(routes, values, facts.list_exp, level_s, window_of())
    num, _den, _, _ = carry_ops.carry(routes, values, facts, carve, scale, accumulate=accumulate, forced=forced,
                             reduce_dtype=reduce_dtype)
    torch.cuda.synchronize()
    want, bound = readout(routes, values, facts.list_exp, level_s, window_of(), scale, accumulate, reduce_dtype,
                          box=(carve.extents[level_s], carve.extents[1 - level_s]))
    got = num_absolute(num, scale)
    later = slice(window_of(), L)
    assert (want[:, later].abs() > 0).float().mean() > 0.5, "most later slots read nothing: the cell is vacuous"
    assert torch.equal(got[:, :window_of()], torch.zeros_like(got[:, :window_of()])), (
        "the first window read a state before any write")
    #: a slot exactly the reference is within any bound, a zero one included (the first window's, before any write)
    diff = (got - want).abs()
    return torch.where(diff == 0, 0.0, diff / bound).max().item(), carry_plan.plans(routes, facts, carve, forced)


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_the_readout_is_the_reference_per_slot(write_sparse, growth, accumulate, reduce_dtype):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64), both accumulate forms, both reduce dtypes: every token and
    channel of ``num`` within the bound of the reference's readout -- the write gain steady, or scaled 2^growth a
    window, so the fold's lists climb through their units and the image puts them on one scale."""
    ratio, _ = _readout_ratio((64, 64), 2048, write_sparse, growth, accumulate, reduce_dtype=reduce_dtype)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


def _sides(write_sparse):
    """``(read_sparse, write_sparse)`` flags of the alternating bundle at depth two."""
    write = tuple(level == write_sparse for level in range(2))
    return tuple(not flag for flag in write), write


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_the_64_channel_box_reads_out_to_the_reference(write_sparse, growth, accumulate, reduce_dtype):
    """The 64-channel box (one token group of eight channel groups, 16 read digits a box at the facts pass's 16-digit
    list group): every slot of ``num`` within the bound of the reference's readout, both accumulate forms, both reduce
    dtypes, the units steady or climbing."""
    ratio, _ = _readout_ratio((64, 64), 2048, write_sparse, growth, accumulate, reduce_dtype=reduce_dtype,
                              box=box64(write_sparse))
    assert_box64_arm(carve_at((64, 64), *_sides(write_sparse), *box64(write_sparse), 64))
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize(("widths", "live", "image"), [((64, 64), 4, "halves"), ((256, 256), 2, "whole")])
def test_the_same_split_reads_out_to_the_reference(widths, live, image, level, accumulate, reduce_dtype):
    """Both sides listing one level (`[SS, DD]` and `[DD, SS]`): the hand-off's same-split form, the fold's lists the
    readout's digits, in each image mode, every slot within its bound, the units climbing 2^8 a window."""
    ratio, plans = _readout_ratio(widths, 2048, level, 8, accumulate, same=True, live=live, reduce_dtype=reduce_dtype)
    assert image in carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {image}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("read_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_a_dense_write_sides_64_channel_box_reads_out(read_sparse, growth, accumulate, reduce_dtype):
    """A dense write side's 64-channel box (16 lists x 32 rows x 64 channels, two fold lists a warp, the same split):
    every slot of ``num`` within its bound, both accumulate forms and reduce dtypes."""
    ratio, _ = _readout_ratio((64, 64), 2048, read_sparse, growth, accumulate, reduce_dtype=reduce_dtype,
                              box=box64_dense(read_sparse), dense_write=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("read_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_a_dense_write_side_reads_out(read_sparse, growth, accumulate, reduce_dtype):
    """A dense write side's 32-channel box (32 lists x 32 rows x 32 channels, the same split): every slot of ``num``
    within its bound, both accumulate forms and reduce dtypes."""
    ratio, _ = _readout_ratio((64, 64), 2048, read_sparse, growth, accumulate, reduce_dtype=reduce_dtype,
                              box=((32, 32), 32), dense_write=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
def test_one_box_of_a_dense_write_side_reads_out(reduce_dtype):
    """One dense write side's 64-channel box a batch-head (16 listed read digits, 32 dense rows), two windows: every
    slot within its bound -- the sanitizer lane's one-CTA cell for that box."""
    ratio, _ = _readout_ratio((16, 32), 2 * window_of(), 0, 8, carry_ops.readout_accumulate(), reduce_dtype=reduce_dtype,
                              box=box64_dense(0), dense_write=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
def test_one_box_of_the_same_split_reads_out(reduce_dtype):
    """One same-split 64-channel box a batch-head (both sides listing a level of 16, 32 rows), two windows: every slot
    within its bound -- the sanitizer lane's one-CTA cell for that box."""
    ratio, _ = _readout_ratio((16, 32), 2 * window_of(), 0, 8, carry_ops.readout_accumulate(), same=True,
                              reduce_dtype=reduce_dtype, box=box64_same(0))
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
def test_one_box_reads_out_at_the_64_channel_box(reduce_dtype):
    """One box a batch-head (each list level exactly the box's extent: 32 write lists, 16 read digits a facts list
    group): every slot within its bound over two windows -- the sanitizer lane's one-CTA cell for this box."""
    ratio, _ = _readout_ratio((32, 16), 2 * window_of(), 0, 8, carry_ops.readout_accumulate(), reduce_dtype=reduce_dtype,
                              box=box64(0))
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("growth", [0, 8])
@pytest.mark.parametrize("widths", [(64, 64), (128, 32)])
def test_a_dense_read_side_reads_out_to_the_reference(widths, growth, reduce_dtype):
    """Both sides dense at every level (readout_dense.cuh, the box 32 digits x K 16 x 64 channels): every slot within
    the bound of the declared arithmetic -- operand scaling at the token's exponent, one fp32 accumulator a box."""
    ratio, _ = _readout_ratio(widths, 4 * window_of(), 0, growth, "fp32", reduce_dtype=reduce_dtype, dense=True)
    assert ratio <= 1.0, ratio


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
def test_one_box_of_a_dense_read_side_reads_out(reduce_dtype):
    """One box a batch-head on both sides dense (32 digits x K 16, the box's extents): every slot within its bound over
    two windows -- the sanitizer lane's one-CTA cell for the dense readout."""
    ratio, _ = _readout_ratio((32, 16), 2 * window_of(), 0, 8, "fp32", reduce_dtype=reduce_dtype, dense=True)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS64)
def test_every_plan_form_reads_out_at_the_64_channel_box(widths, live, forced, forms):
    """Each of the plan's forms at the 64-channel box: every slot of ``num`` within its bound."""
    ratio, plans = _readout_ratio(widths, 2048, 0, 8, carry_ops.readout_accumulate(), live=live, forced=forced,
                                  box=box64(0))
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("level", [0, 1])
def test_the_same_splits_64_channel_box_reads_out(level, accumulate, reduce_dtype):
    """Both sides listing one level at the 64-channel box (16 of its digits, two fold lists a warp, 32 rows): every
    slot within its bound, the units climbing 2^8 a window."""
    ratio, _ = _readout_ratio((64, 64), 2048, level, 8, accumulate, same=True, reduce_dtype=reduce_dtype,
                              box=box64_same(level))
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS)
def test_every_plan_form_reads_out_to_the_reference(widths, live, forced, forms):
    """Each of the plan's forms (`tests/oracle/carry_fixtures.py` `FORM_CELLS`), its cell taking them by the plans'
    model: every slot of ``num`` within its bound, the units climbing 2^8 a window, the device's accumulate form."""
    ratio, plans = _readout_ratio(widths, 2048, 0, 8, carry_ops.readout_accumulate(), live=live, forced=forced)
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
