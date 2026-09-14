"""Gate for the LIVENESS PASS: the kernel IS the model, bit for bit.

`tests/unit/test_liveness_contract.py` gates the contract and its pure-torch model
without a device; this file gates the ONE pass against that model, which is what makes
every property the model already proves -- sparse rows are the amplitude support, dense
rows are the static width mask and cost no read, pad digits are dead, a token past `L`
votes dead, the folds are exact -- a property of the shipped kernel too.

It also gates the two class-2 folds the pass feeds: the exact per-page byte
(:func:`rola.engine.facts.liveness.page_bits`, which `plan_exact` and residency
consume) against the general box-grain fold at the atom box set and against the
leaf-level truth.

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``b_l`` the LOGICAL
one; ``L`` tokens; ``BH`` = batch x heads; ``N = prod_l B_l`` leaves; atom = 16
canonical leaves = the page granule.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.facts import liveness as lv
from rola.engine.facts import planes
from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN
from rola.ops.liveness import liveness_words

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pass is a CUDA kernel")

#: The contract gate's own shapes, so the two files cover the same topologies: a padded
#: dense level, a two- and a three-level topology, an `L` that is not a whole number of
#: words, and one cell whose atom does not fill its last level.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33))

#: Which levels each side calls DENSE. The pass reads neither side's dense levels.
_MODES = ((), (0,), "all")


def _format(widths, d_v=64, BH=2):
    return StateFormat(D=len(widths), B=tuple(widths), order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=d_v, dtype=SPLIT_PLANE_DTYPE,
                       ids=BH * (prod(widths) >> PAGE_RECTANGLE_BITS))


def _statics(layout, dense, logical=None):
    levels = range(layout.D) if dense == "all" else dense
    return lv.side_statics(layout, tuple(level in levels for level in range(layout.D)),
                           tuple(layout.B) if logical is None else tuple(logical))


def _draw(layout, statics, *, BH=2, seed=0, device="cuda"):
    """A synthetic routing plane: sparse levels a random support with at least one live
    digit per (token, level), dense levels a full row, every pad digit EXACTLY zero."""
    generator = torch.Generator().manual_seed(seed)
    plane = torch.zeros((BH, layout.L, layout.rows), dtype=torch.float32)
    for level in range(layout.D):
        base, width = layout.row_base(level), layout.B[level]
        logical = sum(statics.digit_mask[base:base + width])
        block = torch.rand((BH, layout.L, logical), generator=generator) + 0.5
        if not statics.dense_levels[level]:
            keep = torch.rand((BH, layout.L, logical), generator=generator) < 0.4
            keep[..., 0] |= ~keep.any(dim=-1)
            block = block * keep
        plane[:, :, base:base + logical] = block
    return plane.to(torch.bfloat16).to(device)


def _both(layout, read_statics, write_statics, *, BH=2, seed=0):
    read = _draw(layout, read_statics, BH=BH, seed=seed)
    write = _draw(layout, write_statics, BH=BH, seed=seed + 101)
    return read, write


# --------------------------------------------------------------- the pass is the model


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("dense", _MODES)
def test_the_kernel_is_the_model_bit_for_bit(widths, L, dense):
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout, dense)
    read, write = _both(layout, statics, statics)
    got = liveness_words(read, write, layout, statics, statics)
    want = lv.liveness_words(read.cpu(), write.cpu(), layout, statics, statics)
    assert got.shape == (read.shape[0], lv.SIDES, layout.rows, layout.words)
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_two_sides_are_voted_independently(widths, L):
    """One launch, two sides: the write side's rows must not be the read side's."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    read_statics, write_statics = _statics(layout, ()), _statics(layout, (0,))
    read, write = _both(layout, read_statics, write_statics)
    got = liveness_words(read, write, layout, read_statics, write_statics)
    want = lv.liveness_words(read.cpu(), write.cpu(), layout, read_statics, write_statics)
    assert torch.equal(got.cpu(), want)
    assert not torch.equal(got[:, lv.READ], got[:, lv.WRITE])


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("dense", _MODES)
def test_a_padded_topology_is_the_model_and_its_pad_digits_are_dead(widths, L, dense):
    layout = lv.LivenessLayout.of(_format(widths), L)
    logical = tuple(max(1, width - 3) for width in widths)
    statics = _statics(layout, dense, logical)
    read, write = _both(layout, statics, statics)
    got = liveness_words(read, write, layout, statics, statics).cpu()
    assert torch.equal(got, lv.liveness_words(read.cpu(), write.cpu(), layout, statics, statics))
    bits = lv.unpack(got, layout)
    for level in range(layout.D):
        base = layout.row_base(level)
        pad = bits[:, :, base + logical[level]:base + layout.B[level], :]
        assert not pad.any(), f"a pad digit voted live at level {level}, dense={dense}"


def test_a_dense_level_is_never_read():
    """BLIND the dense amplitudes: a pass that read them would report the level dead."""
    layout = lv.LivenessLayout.of(_format((32, 16)), 64)
    statics = _statics(layout, (0,), logical=(20, 16))
    read, write = _both(layout, statics, statics)
    blind_read, blind_write = read.clone(), write.clone()
    blind_read[:, :, :32] = 0
    blind_write[:, :, :32] = 0
    seen = liveness_words(read, write, layout, statics, statics)
    blinded = liveness_words(blind_read, blind_write, layout, statics, statics)
    assert torch.equal(seen, blinded)
    mask = torch.tensor(statics.digit_mask[:32], dtype=torch.bool)
    dense_rows = lv.unpack(blinded.cpu(), layout)[:, lv.READ, :32]
    assert torch.equal(dense_rows, mask[None, :, None].expand(dense_rows.shape))


def test_a_token_past_the_end_votes_dead_on_the_device():
    layout = lv.LivenessLayout.of(_format((16, 16)), 33)
    statics = _statics(layout, ())
    plane = torch.ones((1, layout.L, layout.rows), dtype=torch.bfloat16, device="cuda")
    got = liveness_words(plane, plane, layout, statics, statics).cpu()
    assert layout.words == 2
    assert torch.equal(got[:, :, :, 1], torch.ones_like(got[:, :, :, 1]))


def test_negative_zero_is_dead_on_the_device_too():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    statics = _statics(layout, ())
    plane = torch.full((1, layout.L, layout.rows), -0.0, dtype=torch.bfloat16, device="cuda")
    plane[0, 0, 5] = -1.0
    bits = lv.unpack(liveness_words(plane, plane, layout, statics, statics).cpu(), layout)
    assert bits[:, lv.READ].sum() == 1 and bits[0, lv.READ, 5, 0]


# --------------------------------------------------------------- the seam's refusals


def test_the_seam_refuses_a_row_that_is_not_the_amplitude_column():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    statics = _statics(layout, ())
    plane = _draw(layout, statics)
    wrong = lv.LivenessLayout(D=2, B=(16, 32), L=32)
    with pytest.raises(RuntimeError, match="a row index IS the column it votes from"):
        liveness_words(plane, plane, wrong, _statics(wrong, ()), _statics(wrong, ()))


def test_the_seam_refuses_a_plane_that_is_not_bf16():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    statics = _statics(layout, ())
    plane = torch.zeros((2, 32, 32), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="bf16 amplitudes"):
        liveness_words(plane, plane, layout, statics, statics)


# --------------------------------------------------------------- the folds it feeds


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_page_byte_is_the_general_fold_at_the_atom_box_set(widths, L):
    """`page_bits` is not a second derivation: it is `page_activity` over `atom_boxes`."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout, ())
    read, write = _both(layout, statics, statics)
    words = liveness_words(read, write, layout, statics, statics).cpu()
    boxes = lv.atom_boxes(layout)
    assert len(boxes) == layout.atoms
    want = lv.page_activity(layout, boxes, lv.box_activity(words, layout, boxes, lv.READ),
                            boxes, lv.box_activity(words, layout, boxes, lv.WRITE))
    assert torch.equal(lv.page_bits(words, layout), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_page_byte_is_the_leaf_level_truth_exactly(widths, L):
    """EXACT, not merely sound: at the atom grain the box->page map is a bijection, so
    a page carries a bit if and only if it holds a live leaf."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout, ())
    read, write = _both(layout, statics, statics)
    words = liveness_words(read, write, layout, statics, statics).cpu()
    got = lv.page_bits(words, layout)
    bits = lv.unpack(words, layout)
    strides = layout.radix_strides
    for side, flag in ((lv.READ, ATOM_READ), (lv.WRITE, ATOM_WRITTEN)):
        truth = torch.zeros_like(got, dtype=torch.bool)
        for bh in range(bits.shape[0]):
            for token in range(layout.L):
                leaves = torch.zeros(1, dtype=torch.int64)
                for level in range(layout.D):
                    base = layout.row_base(level)
                    digits = bits[bh, side, base:base + layout.B[level], token].nonzero().flatten()
                    leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
                truth[bh, torch.unique(leaves // lv.ATOM_LEAVES)] = True
        assert torch.equal((got & flag).bool(), truth)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_activity_byte_entry_is_the_fold_over_the_pass(widths, L):
    """`planes.atom_bits` -- what `plan_exact` and residency consume -- IS this fold."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout, ())
    read, write = _both(layout, statics, statics)
    words = liveness_words(read, write, layout, statics, statics)
    assert torch.equal(planes.atom_bits(write, widths, read).cpu(),
                       lv.page_bits(words.cpu(), layout))
    #: the convenience entry: the write plane stands in for both sides.
    alone = planes.atom_bits(write, widths).cpu()
    assert torch.equal(planes.written_atoms(alone), planes.read_atoms(alone))
