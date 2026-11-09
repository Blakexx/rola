"""Gate for the LIVENESS CONTRACT: the class-1 words and the class-2 folds.

The subject is a CONTRACT and its executable spec, both of which are pure Python over
torch tensors, so the whole file gates without the built extension and without a GPU.
What is NOT gated here is the facts pass KERNEL (`tests/integration/test_facts_pass.py`):
this file is what that kernel is graded against (docs/internals/facts/liveness_contract.md).

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``b_l`` the LOGICAL
one; ``L`` tokens; ``BH`` = batch x heads; ``N = prod_l B_l`` leaves; atom = 16
canonical leaves = the page granule; a BOX = one warp sub-box, a rectangle in digit
space.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from math import prod
from pathlib import Path

import pytest
import torch

from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.facts import liveness as lv
from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN

_HEADER = Path(__file__).resolve().parents[2] / "csrc/rola/src/facts/liveness_contract.cuh"

#: (B_l per level, L) -- a padded dense level, a two-level and a three-level topology,
#: an L that is not a whole number of words, and one cell whose boxes SPLIT an atom.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33))


def _format(widths, d_v=64, BH=2):
    return StateFormat(D=len(widths), B=tuple(widths), order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=d_v, dtype=SPLIT_PLANE_DTYPE,
                       ids=BH * (prod(widths) >> PAGE_RECTANGLE_BITS))


def _draw(layout, statics, *, BH=2, seed=0):
    """A side's synthetic rows, ``[1, L, BH, b_l]`` u8 per level at its logical width: sparse levels get a random
    support with at least one live digit per (token, level), dense levels a full row of nonzero digits."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for level in range(layout.D):
        base = layout.row_base(level)
        logical = sum(statics.digit_mask[base:base + layout.B[level]])
        block = torch.randint(1, 256, (1, layout.L, BH, logical), generator=generator)
        if not statics.dense_levels[level]:
            keep = torch.rand((1, layout.L, BH, logical), generator=generator) < 0.4
            keep[..., 0] |= ~keep.any(dim=-1)
            block = block * keep
        rows.append(block.to(torch.uint8))
    return tuple(rows)


def _truth(rows, layout):
    """``[BH, rows, L]`` bool: each digit's nonzero test, a level's digits past its rows dead."""
    live = torch.zeros((rows[0].shape[2], layout.rows, layout.L), dtype=torch.bool)
    for level, block in enumerate(rows):
        base = layout.row_base(level)
        live[:, base:base + block.shape[-1]] = (block[0] != 0).permute(1, 2, 0)
    return live


def _statics(layout, *, dense=(), logical=None):
    logical = tuple(layout.B) if logical is None else tuple(logical)
    return lv.side_statics(layout, tuple(level in dense for level in range(layout.D)), logical)


def _boxes(layout, spans):
    """Every aligned box of the given per-level spans -- the carve's whole partition."""
    grid = [[]]
    for level in range(layout.D):
        grid = [row + [digit] for row in grid
                for digit in range(0, layout.B[level], spans[level])]
    return [lv.Box(start=tuple(row), span=tuple(spans)) for row in grid]


# --------------------------------------------------------------- the layout


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_layout_rows_are_the_digits_in_canonical_order(widths, L):
    layout = lv.LivenessLayout.of(_format(widths), L)
    assert layout.rows == sum(widths)
    seen = [layout.row_of(level, digit)
            for level in range(layout.D) for digit in range(widths[level])]
    assert seen == list(range(layout.rows))


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_layout_bytes_are_L_times_sum_B_over_eight(widths, L):
    layout = lv.LivenessLayout.of(_format(widths), L)
    if L % lv.TOK_BITS == 0:
        assert layout.side_bytes == L * sum(widths) // 8
    else:
        assert layout.side_bytes == lv.token_words(L) * 4 * sum(widths)
    assert layout.call_words == 2 * layout.side_words


def test_layout_word_index_walks_the_table_exactly_once():
    layout = lv.LivenessLayout.of(_format((16, 16)), 64)
    seen = {layout.word_index(bh, side, row, token * lv.TOK_BITS)
            for bh in range(2) for side in range(lv.SIDES)
            for row in range(layout.rows) for token in range(layout.words)}
    assert sorted(seen) == list(range(2 * layout.call_words))


def test_layout_refuses_a_digit_outside_its_level():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    with pytest.raises(ValueError, match="carries 16 digits"):
        layout.row_of(0, 16)


# --------------------------------------------------------------- the header mirror


def test_header_constants_mirror_the_python_reader():
    text = _HEADER.read_text()
    constants = dict(re.findall(r"constexpr (?:int|uint16_t) (\w+) = (0x[0-9a-fA-F]+|\d+)u?;", text))
    assert int(constants["kTokBits"]) == lv.TOK_BITS
    assert int(constants["kTileTokens"]) == lv.TILE_TOKENS
    assert int(constants["kGroupDigits"]) == lv.GROUP_DIGITS
    assert int(constants["kMaxLevels"]) == 4
    assert re.search(r"kRead = 0, kWrite = 1, kSides = 2", text)
    assert (lv.READ, lv.WRITE, lv.SIDES) == (0, 1, 2)


def test_header_compiles_as_a_standalone_translation_unit(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("no host C++ compiler on this box")
    source = tmp_path / "liveness_contract_tu.cpp"
    source.write_text(f'#include "{_HEADER}"\n'
                      "static constexpr rola::facts::LivenessLayout kL{2, {16, 16, 0, 0}, 64};\n"
                      "static_assert(kL.rows() == 32);\n"
                      "static_assert(kL.words() == 2);\n"
                      "static_assert(kL.row_of(1, 3) == 19);\n"
                      "static_assert(kL.side_bytes() == 64 * 32 / 8);\n"
                      "static_assert(kL.word_index(1, 1, 0, 0) == 3 * kL.side_words());\n"
                      "static_assert(kL.token_bit(33) == 2u);\n"
                      "static_assert(kL.groups() == 2 && kL.group_base(1) == 1 && kL.tiles() == 1);\n"
                      "static_assert(kL.union_index(1, 1, 1, 0) == 3 * 2 + 1);\n"
                      "static_assert(rola::facts::digit_live(uint8_t{1}));\n"
                      "static_assert(!rola::facts::digit_live(uint8_t{0}));\n"
                      "static_assert(rola::facts::mask_bit(0x4u, 34));\n"
                      "int main() { return 0; }\n")
    subprocess.run([compiler, "-std=c++17", "-fsyntax-only", str(source)], check=True)


# --------------------------------------------------------------- the pass model


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_sparse_rows_are_the_amplitude_support(widths, L):
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout)
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    bits = lv.unpack(words, layout)
    assert torch.equal(bits[:, lv.READ], _truth(rows, layout))
    assert torch.equal(bits[:, lv.READ], bits[:, lv.WRITE])


def test_dense_rows_are_the_static_width_mask_and_cost_no_read():
    layout = lv.LivenessLayout.of(_format((32, 16)), 64)
    statics = _statics(layout, dense=(0,), logical=(20, 16))
    rows = _draw(layout, statics)
    #: the dense level's digits are OVERWRITTEN with zeros: a pass that read them
    #: would report the level dead, and the width mask is what makes it not.
    blinded = (torch.zeros_like(rows[0]),) + rows[1:]
    words = lv.liveness_words(blinded, blinded, layout, statics, statics)
    bits = lv.unpack(words, layout)[:, lv.READ, :32]
    mask = torch.tensor(statics.digit_mask[:32], dtype=torch.bool)
    assert torch.equal(bits, mask[None, :, None].expand(bits.shape))


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_pad_digits_are_dead_by_construction(widths, L):
    layout = lv.LivenessLayout.of(_format(widths), L)
    logical = tuple(max(1, width - 3) for width in widths)
    for dense in ((), (0,), tuple(range(len(widths)))):
        statics = _statics(layout, dense=dense, logical=logical)
        rows = _draw(layout, statics)
        bits = lv.unpack(lv.liveness_words(rows, rows, layout, statics, statics), layout)
        for level in range(layout.D):
            base = layout.row_base(level)
            pad = bits[:, :, base + logical[level]:base + widths[level], :]
            assert not pad.any(), f"a pad digit voted live at level {level}, dense={dense}"


def test_a_token_past_the_end_votes_dead():
    layout = lv.LivenessLayout.of(_format((16, 16)), 33)
    statics = _statics(layout)
    rows = tuple(torch.ones((1, layout.L, 1, width), dtype=torch.uint8) for width in layout.B)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    assert layout.words == 2
    tail = words[:, :, :, 1]
    assert torch.equal(tail, torch.ones_like(tail))
    assert lv.unpack(words, layout).shape[-1] == 33


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_a_tile_union_is_its_groups_digits_live_anywhere_in_its_64_tokens(widths, L):
    """The unions against the per-token bits directly: bit ``i`` of (group ``g`` of level ``l``, tile ``t``) is digit
    ``32 g + i`` of level ``l`` live at some token of the tile; a group past a narrow level's digits is dead."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout)
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    bits = lv.unpack(words, layout)
    unions = lv.tile_unions(words, layout)
    assert unions.shape == (2, lv.SIDES, layout.groups, layout.tiles)
    for level, width in enumerate(widths):
        for digit in range(width):
            group, bit = layout.group_base(level) + digit // lv.GROUP_DIGITS, digit % lv.GROUP_DIGITS
            for tile in range(layout.tiles):
                want = bits[:, :, layout.row_of(level, digit), tile * lv.TILE_TOKENS:(tile + 1) * lv.TILE_TOKENS]
                got = (unions[:, :, group, tile] >> bit) & 1
                assert torch.equal(got.bool(), want.any(-1))
        for digit in range(width, -(-width // lv.GROUP_DIGITS) * lv.GROUP_DIGITS):
            group, bit = layout.group_base(level) + digit // lv.GROUP_DIGITS, digit % lv.GROUP_DIGITS
            assert not ((unions[:, :, group] >> bit) & 1).any()


def test_negative_zero_is_dead_like_positive_zero():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    statics = _statics(layout)
    rows = tuple(torch.full((1, layout.L, 1, width), -0.0) for width in layout.B)
    rows[0][0, 0, 0, 5] = -1.0
    bits = lv.unpack(lv.liveness_words(rows, rows, layout, statics, statics), layout)
    assert bits[:, lv.READ].sum() == 1
    assert bits[0, lv.READ, 5, 0]


def test_the_static_mask_packs_into_the_words_the_kernel_reads():
    """`SideStatics` is an OPERAND, not a description: the bits the seam packs are the
    bits `digit_in_mask` reads back, and no logical width crosses with them."""
    layout = lv.LivenessLayout.of(_format((32, 16)), 64)
    statics = _statics(layout, dense=(0,), logical=(20, 9))
    words = [statics.mask_word(index) for index in range(lv.mask_words(layout.rows))]
    assert len(words) == 2
    for row in range(layout.rows):
        read_back = bool((words[row // 32] >> (row % 32)) & 1)
        assert read_back == statics.digit_mask[row]
    assert sum(statics.digit_mask) == 20 + 9


# --------------------------------------------------------------- the folds


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("spans", ("whole", "half"))
def test_box_bits_are_the_or_over_tokens_of_the_and_over_levels(widths, L, spans):
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout, dense=(0,) if len(widths) > 1 else ())
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    bits = lv.unpack(words, layout)[:, lv.READ]
    step = [width if spans == "whole" else max(1, width // 2) for width in widths]
    boxes = _boxes(layout, step)
    got = lv.box_activity(words, layout, boxes, lv.READ, statics)
    for index, box in enumerate(boxes):
        clause = torch.ones((bits.shape[0], layout.L), dtype=torch.bool)
        for level in range(layout.D):
            base = layout.row_base(level) + box.start[level]
            clause &= bits[:, base:base + box.span[level], :].any(dim=1)
        assert torch.equal(got[:, index], clause.any(dim=-1))


def test_a_fully_pad_box_is_dead_and_a_straddling_one_is_not():
    layout = lv.LivenessLayout.of(_format((32, 16)), 64)
    statics = _statics(layout, logical=(20, 16))
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    boxes = _boxes(layout, [8, 16])
    live = lv.box_activity(words, layout, boxes, lv.WRITE, statics)
    for index, box in enumerate(boxes):
        if box.start[0] >= 20:
            assert not live[:, index].any(), "a box of nothing but pad digits must die"
    straddle = [index for index, box in enumerate(boxes) if box.start[0] == 16]
    assert straddle and live[:, straddle].any()


def test_a_box_covers_the_leaves_its_digit_runs_name():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    box = lv.Box(start=(0, 0), span=(1, 16))
    assert torch.equal(box.leaves(layout), torch.arange(16))
    assert torch.equal(box.pages(layout), torch.tensor([0]))
    split = lv.Box(start=(0, 0), span=(2, 8))
    assert sorted(split.pages(layout).tolist()) == [0, 1]


def test_box_refuses_a_run_that_is_not_an_aligned_divisor():
    layout = lv.LivenessLayout.of(_format((16, 16)), 32)
    for bad in (lv.Box(start=(1, 0), span=(2, 16)), lv.Box(start=(0, 0), span=(3, 16))):
        with pytest.raises(ValueError, match="aligned divisor run"):
            bad.check(layout)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_page_bits_are_the_or_of_the_box_bits_over_the_box_to_page_map(widths, L):
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout)
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    read_boxes = _boxes(layout, [max(1, width // 2) for width in widths])
    write_boxes = _boxes(layout, list(widths))
    read_bits = lv.box_activity(words, layout, read_boxes, lv.READ, statics)
    write_bits = lv.box_activity(words, layout, write_boxes, lv.WRITE, statics)
    pages = lv.page_activity(layout, read_boxes, read_bits, write_boxes, write_bits)
    assert pages.shape == (read_bits.shape[0], layout.atoms)
    want = torch.zeros_like(pages)
    for boxes, live, flag in ((read_boxes, read_bits, ATOM_READ),
                              (write_boxes, write_bits, ATOM_WRITTEN)):
        for index, box in enumerate(boxes):
            for page in box.pages(layout).tolist():
                want[:, page] |= live[:, index].to(torch.uint8) * flag
    assert torch.equal(pages, want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_page_bits_cover_every_leaf_the_routing_actually_touches(widths, L):
    """SOUNDNESS: the derived set is a box-grain OR, so it may over-report, but a page
    holding a live leaf is never missed -- allocation and residency depend on it."""
    layout = lv.LivenessLayout.of(_format(widths), L)
    statics = _statics(layout)
    rows = _draw(layout, statics)
    words = lv.liveness_words(rows, rows, layout, statics, statics)
    boxes = _boxes(layout, [max(1, width // 2) for width in widths])
    live = lv.box_activity(words, layout, boxes, lv.WRITE, statics)
    pages = lv.page_activity(layout, boxes, live, boxes, live)
    bits = lv.unpack(words, layout)[:, lv.WRITE]
    strides = layout.radix_strides
    BH = bits.shape[0]
    for bh in range(BH):
        for token in range(layout.L):
            leaves = torch.zeros(1, dtype=torch.int64)
            for level in range(layout.D):
                base = layout.row_base(level)
                digits = bits[bh, base:base + layout.B[level], token].nonzero().flatten()
                leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
            for page in torch.unique(leaves // lv.ATOM_LEAVES).tolist():
                assert pages[bh, page] & ATOM_WRITTEN, (
                    f"page {page} holds a live leaf and the derived bits missed it")
