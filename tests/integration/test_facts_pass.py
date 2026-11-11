"""Gate for the FACTS PASS (csrc/rola/src/facts/): its words ARE the model's, bit for bit, and its lists are the
model's final form, bit for bit.

`tests/unit/test_liveness_contract.py` gates the words' contract and the pure-torch model without a device; this file
holds the one pass to that model on the bundle's own rows -- u8, separate tensors and views of one packed plane
-- at every SPARSE level, which makes every property the model proves (sparse rows are the digits' support, pad digits
are dead, a token past `L` votes dead) a property of the kernel; a dense level's rows and groups are the static width
mask, which the folds take from the statics, so the pass neither reads nor writes them. The lists are the model's
at each list group (docs/internals/facts/facts.md#lists-out). The class-2 folds the words feed are
gated here too: the exact per-page byte against the general box fold at the atom box set and against the leaf-level
truth.

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``b_l`` the LOGICAL one (the rows' own width);
``L`` tokens; ``BH`` = batch x heads; atom = 16 canonical leaves = the page granule.
"""
from __future__ import annotations

import pytest
import torch
import torch.utils.deterministic

from rola.engine.facts import lists as list_model
from rola.engine.facts import liveness as lv
from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from rola.routing import quant
from rola.routing.factors import RouteFactors
from rola.routing.topology import padded_level_width
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pass is a CUDA kernel")

#: (b_l per level, L): one level, two and three levels, an `L` that is not a whole number of words, a level wider
#: than a 32-digit group, and a many-window call with a tail window and a tail word.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33), ((256, 256), 1100))

#: Which levels each side declares DENSE (the pass neither reads nor writes either side's dense levels), every side
#: with at least one (docs/api.md section 2.1): "alternating" -- read-dense on the even levels, write-dense on the odd
#: (and, at depth one, on level 0), so each side has a sparse level from depth two; "tied" -- level 0 dense on both
#: sides, the rest sparse; "all" -- every level dense on both.
_MODES = ("alternating", "tied", "all")

#: EACH SIDE'S LIST GROUP, (read, write) (docs/internals/facts/facts.md#list-groups): a unit's whole liveness group,
#: each side's halves alone, and both halves
_LIST_GROUPS = ((32, 32), (16, 32), (32, 16), (16, 16))


def _facts(routes, values, layout, read_statics, write_statics, list_groups=(32, 32)):
    return facts_ops.facts(routes, values, layout, read_statics, write_statics, facts_ops.ListsForm(*list_groups))


def _flags(D, mode):
    """``(read, write)`` dense flags a level for one of `_MODES`."""
    if mode == "all":
        return (True,) * D, (True,) * D
    if mode == "tied":
        return (True,) + (False,) * (D - 1), (True,) + (False,) * (D - 1)
    return (tuple(level % 2 == 0 for level in range(D)),
            tuple(level % 2 == 1 or D == 1 for level in range(D)))


def _sides(layout, mode, logical):
    """``(read_statics, write_statics)`` of a mode."""
    return tuple(lv.side_statics(layout, flags, logical) for flags in _flags(layout.D, mode))


def _statics(layout, dense, logical):
    """One side's statics at dense levels ``dense`` (a tuple of levels)."""
    return lv.side_statics(layout, tuple(level in dense for level in range(layout.D)), logical)


def _bundle(widths, L, statics, *, B=1, H=2, seed=0, packed=False, group=1):
    """A bundle at logical widths ``widths`` in the format: a sparse level's rows a random support (at least one live
    digit a token) as u8; a dense level's mostly nonzero digits with some zeros -- a value that rounded to zero, which
    the width mask reports live all the same -- as u8 too (`rola.routing.quant`). ``packed`` gives each
    side's levels as views of the producer's one u8 plane, otherwise each level is its own tensor. ``group``: ``H`` write heads each read by ``group`` query heads."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    write_heads = H
    for index, side in enumerate(statics):
        H = write_heads * group if index == 0 else write_heads
        levels = []
        for level, width in enumerate(widths):
            digits = torch.randint(1, 256, (B, L, H, width), generator=gen)
            keep = torch.rand((B, L, H, width), generator=gen) < (0.9 if side.dense_levels[level] else 0.4)
            if not side.dense_levels[level]:
                keep[..., 0] |= ~keep.any(-1)
            digits = digits * keep
            levels.append(digits.to(torch.uint8))
        if packed:
            slots = [quant.row_slot(width) for width in widths]
            offsets = [sum(slots[:i]) for i in range(len(widths))]
            plane = torch.zeros((B, L, H, sum(slots)), dtype=torch.uint8)
            for offset, width, level in zip(offsets, widths, levels):
                plane[..., offset:offset + width] = level
            plane = plane.cuda()
            rows.append(tuple(plane[..., offset:offset + width] for offset, width in zip(offsets, widths)))
        else:
            rows.append(tuple(quant.aligned_rows(B, L, H, w, "cuda").copy_(level)
                              for w, level in zip(widths, levels)))

    def column(heads=write_heads):
        return (torch.rand((B, L, heads), generator=gen) + 0.1).cuda()

    D = len(widths)
    routes = RouteFactors(topology=Topology(levels=tuple(OpaqueRouting(width=w) for w in widths)),
                          amp_read=rows[0], amp_write=rows[1],
                          scale_read=tuple(column(write_heads * group) for _ in range(D)),
                          scale_write=tuple(column() for _ in range(D)),
                          sparse=tuple(tuple(not dense for dense in side.dense_levels) for side in statics))
    values = Values(v=torch.zeros((B, L, write_heads, 8), dtype=torch.bfloat16, device="cuda"), scale=column())
    return routes, values


def _layout(widths, L):
    return lv.LivenessLayout(D=len(widths), B=tuple(padded_level_width(w) for w in widths), L=L)


def _model(routes, layout, read_statics, write_statics):
    return lv.liveness_words(tuple(r.cpu() for r in routes.amp_read), tuple(r.cpu() for r in routes.amp_write),
                             layout, read_statics, write_statics)


def _assert_written(got, want, layout, read_statics, write_statics):
    """The pass's words and unions are the model's at every sparse level of each slot: each query head's read side,
    then the write side."""
    unions = lv.tile_unions(want, layout)
    heads = want.shape[1] - 1
    for slot in range(heads + 1):
        statics = read_statics if slot < heads else write_statics
        for level, dense in enumerate(statics.dense_levels):
            if dense:
                continue
            rows = slice(layout.row_base(level), layout.row_base(level) + layout.B[level])
            groups = slice(layout.group_base(level), layout.group_base(level + 1))
            assert torch.equal(got.words[:, slot, rows].cpu(), want[:, slot, rows]), (slot, level)
            assert torch.equal(got.unions[:, slot, groups].cpu(), unions[:, slot, groups]), (slot, level)


# --------------------------------------------------------------- the words are the model


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("dense", _MODES)
@pytest.mark.parametrize("packed", (False, True))
def test_the_words_are_the_model_bit_for_bit(widths, L, dense, packed):
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, dense, widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics), packed=packed)
    got = _facts(routes, values, layout, read_statics, write_statics)
    assert got.words.shape == (routes.batch * routes.heads, lv.SIDES, layout.rows, layout.words)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_two_sides_are_voted_independently(widths, L):
    """One launch, two sides, each with its own statics and rows: each side's sparse levels are its own rows' votes."""
    layout = _layout(widths, L)
    read_statics = _sides(layout, "alternating", widths)[0]
    write_statics = _sides(layout, "tied", widths)[1]
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
@pytest.mark.parametrize("dense", _MODES)
def test_a_padded_topology_is_the_model_and_its_pad_digits_are_dead(widths, L, dense):
    """Logical widths short of the padded ones: the rows are the logical width, the words the padded layout."""
    logical = tuple(max(1, width - 3) for width in widths)
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, dense, logical)
    routes, values = _bundle(logical, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)
    bits = lv.unpack(got.words.cpu(), layout)
    for side, statics in enumerate((read_statics, write_statics)):
        for level in range(layout.D):
            base = layout.row_base(level)
            if not statics.dense_levels[level]:
                assert not bits[:, side, base + logical[level]:base + layout.B[level], :].any(), (side, level)


def test_a_dense_level_is_neither_read_nor_written():
    """BLIND the dense level's rows and POISON the outputs (torch's deterministic fill of new memory, the integer
    maximum): a pass that read the level would vote it dead and one that wrote it would clear the poison, while the
    sparse level is the model's all the same. The call lists nothing: a list level is a side's one sparse level, never
    a dense one (`facts_ops.list_level`)."""
    widths, L = (32, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (0,), (20, 16))
    read_statics = write_statics = statics
    routes, values = _bundle((20, 16), L, (read_statics, write_statics))
    want = _model(routes, layout, read_statics, write_statics)
    for side in (routes.amp_read, routes.amp_write):
        side[0].zero_()
    poison = torch.iinfo(torch.int32).max
    masks = [statics.mask_word(i) for i in range(lv.mask_words(layout.rows))]
    deterministic, fill = torch.are_deterministic_algorithms_enabled(), torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        words, unions, *_ = facts_ops.extension().rola_facts_forward(
            list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), values.scale,
            list(layout.B), 1, masks, 1, masks, -1, -1, 32, 32)
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill
    _assert_written(facts_ops.Facts(words, unions, None, None, None), want, layout, read_statics, write_statics)
    assert (words[:, :, :layout.B[0]] == poison).all() and (unions[:, :, :layout.group_base(1)] == poison).all()


def test_a_token_past_the_end_votes_dead():
    widths, L = (16, 16), 33
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    for side in (routes.amp_read, routes.amp_write):
        for rows in side:
            rows.fill_(1)
    got = _facts(routes, values, layout, read_statics, write_statics).words.cpu()
    assert layout.words == 2
    for side, statics in enumerate((read_statics, write_statics)):
        for level, dense in enumerate(statics.dense_levels):
            if not dense:
                rows = got[:, side, layout.row_base(level):layout.row_base(level) + layout.B[level], 1]
                assert torch.equal(rows, torch.ones_like(rows)), (side, level)


# --------------------------------------------------------------- the factor products are the declared ones


@pytest.mark.parametrize(("widths", "L"), [((32, 64), 600), ((16,), 1100)])
def test_the_factor_products_are_the_declared_fp32_ones(widths, L):
    """``fm`` holds each token's ``(F, m)`` = ``((sw_0 sw_1) vs, sw_0 sw_1)``, multiplied in fp32 in that order, bit for
    bit, at every token of every window (docs/internals/facts/facts.md#factor-products)."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics).fm
    m = torch.ones_like(routes.scale_write[0].float())
    for scale in routes.scale_write:
        m = m * scale.float()
    F = m * values.scale.float()
    want = torch.stack((F, m), -1).permute(0, 2, 1, 3).reshape(routes.batch * routes.heads, L, 2)
    assert torch.equal(got, want)


# --------------------------------------------------------------- the lists are the model


def _lists_want(routes, values, layout, read_statics, write_statics, list_groups=(32, 32)):
    """``(write, read)``: each side's lists from the model at its list level and its list group (``list_groups``,
    read then write), ``None`` for a side with no sparse level."""
    bits = lv.unpack(_model(routes, layout, read_statics, write_statics), layout)
    wl, rl = facts_ops.list_level(write_statics), facts_ops.list_level(read_statics)
    write = read = None
    if wl >= 0:
        write = list_model.final_lists(bits[:, -1], layout, wl, rows=routes.amp_write[wl], window=window_of(),
                                       list_group=list_groups[1])
    if rl >= 0:
        read = list_model.final_lists(bits[:, :-1], layout, rl, rows=routes.amp_read[rl], window=window_of(),
                                      list_group=list_groups[0])
    return write, read


def _assert_lists(got: list_model.Lists, want: list_model.Lists):
    """Every defined slot of the pass's lists is the model's: the unions and their sizes, the boundaries' prefix rows
    and the groups' totals, and each list to its end -- gathered on the device, so nothing past an end (undefined by the
    contract) is read."""
    assert torch.equal(got.count.cpu().long(), want.count)
    assert torch.equal(got.union.cpu().long(), want.union)
    pre = got.pre.long() & 0xFFFF
    assert torch.equal(pre.cpu(), want.pre)
    assert torch.equal(got.totals.cpu().long(), want.totals)
    BH, windows, width = got.entries.shape[:3]
    lengths = ((pre[..., 1:, -1] - pre[..., :-1, -1]) & 0xFFFF).reshape(BH, windows, -1)[..., :width]
    end = torch.arange(got.entries.shape[3], device=pre.device) < lengths[..., None]
    assert torch.equal(got.entries[end].cpu(), want.entries[end.cpu()])


#: GROUPED QUERIES: shapes whose merged read unions pass a window (every token live in a group at the draw's density),
#: a narrow level, and a many-window call with a tail window
_GROUPED = (((16, 16), 64), ((64, 64), 1100), ((256, 256), 1100))


@pytest.mark.parametrize(("widths", "L"), _GROUPED)
@pytest.mark.parametrize("group", (2, 4))
@pytest.mark.parametrize("list_groups", ((32, 32), (16, 16)))
def test_grouped_reads_are_one_union_a_window_bit_for_bit(widths, L, group, list_groups):
    """GROUPED QUERIES (docs/internals/facts/facts.md#lists-out): ``H`` write heads each read by ``group`` query
    heads. The words and unions are each slot's -- every query head's read side, then the write side -- and the read
    side's lists one union a (window, list group) over the query heads' rows, head-major, the write side's a write
    head's: the model's, bit for bit, the boundaries every 64 slots of the merged union, each list group's slots past
    its own heads before."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics), group=group)
    got = _facts(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    assert read.count.max() > min(L, window_of()), "no merged union passes one head's rows: the cell is vacuous"
    _assert_lists(got.write, write)
    _assert_lists(got.read, read)


@pytest.mark.parametrize(("widths", "L"), [shape for shape in _SHAPES if len(shape[0]) <= 2])
@pytest.mark.parametrize("dense", _MODES)
@pytest.mark.parametrize("list_groups", _LIST_GROUPS)
def test_the_lists_are_the_model_bit_for_bit(widths, L, dense, list_groups):
    """Each side's lists at its list level and list group (docs/internals/facts/facts.md#lists-out, #list-groups): the
    union by rank and its size, each digit's entries in token order (a slot and a byte), and the groups' boundaries
    in prefix form -- bit for bit, at every window."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, dense, widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    for side, want in ((got.write, write), (got.read, read)):
        if want is None:
            assert side.entries.shape[2] == 0
        else:
            _assert_lists(side, want)


@pytest.mark.parametrize("list_groups", _LIST_GROUPS)
def test_the_boundaries_cover_a_whole_window(list_groups):
    """A window whose groups' unions pass the last boundary: every boundary of a list counted, none past its end."""
    widths, L = (64, 64), 1024
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    assert (write.count > 7 * list_model.LIST_BOUND).any(), "no union passes the last boundary: the cell is vacuous"
    _assert_lists(got.write, write)
    _assert_lists(got.read, read)


@pytest.mark.parametrize("list_groups", ((32, 32), (16, 16)))
def test_a_deeper_call_lists_each_sides_one_sparse_level(list_groups):
    """Past depth two a side with one sparse level lists it, the same lists as the model's."""
    widths, L = (16, 32, 16), 96
    layout = _layout(widths, L)
    read_statics = _statics(layout, (0, 2), widths)
    write_statics = _statics(layout, (1, 2), widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, list_groups=list_groups)
    assert facts_ops.list_level(write_statics) == 0 and facts_ops.list_level(read_statics) == 1
    _assert_lists(got.write, write)
    _assert_lists(got.read, read)


def test_a_composite_side_gets_words_and_no_lists():
    """A side with two sparse levels (the alternating split at depth three writes levels 0 and 2) gets its words and
    unions, and its lists are refused by name: they would be keyed by a digit of each level."""
    widths, L = (16, 16, 16), 96
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = _facts(routes, values, layout, read_statics, write_statics)
    assert facts_ops.list_level(write_statics) == facts_ops.COMPOSITE and got.write is None
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)
    with pytest.raises(ValueError, match="composite lists"):
        facts_ops.side_lists(got, "write")
    assert facts_ops.side_lists(got, "read").entries.shape[2] == layout.B[1]


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/facts/facts.md#phase-ledger): every warp books the list items' column staging
    and its units' vote and store, and a bound ledger moves no output."""
    widths, L = (256, 256), 1100
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    plain = _facts(routes, values, layout, read_statics, write_statics)
    ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    facts_ops.ledger_bind(ledger)
    try:
        booked = _facts(routes, values, layout, read_statics, write_statics)
    finally:
        facts_ops.ledger_bind(None)
    want = _model(routes, layout, read_statics, write_statics)
    _assert_written(plain, want, layout, read_statics, write_statics)
    _assert_written(booked, want, layout, read_statics, write_statics)
    assert torch.equal(plain.fm, booked.fm)
    assert (ledger > 0).all(), ledger


# --------------------------------------------------------------- the seam's refusals


def test_the_seam_refuses_rows_that_are_not_the_format():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    wide = [rows.float() for rows in routes.amp_read]
    with pytest.raises(RuntimeError, match="a CUDA u8 tensor"):
        facts_ops.extension().rola_facts_forward(
            wide, wide, list(routes.scale_write), values.scale, list(layout.B), 0,
            [read_statics.mask_word(0)], 0, [write_statics.mask_word(0)], 1, -1, 32, 32)


@pytest.mark.parametrize("list_groups", ((24, 32), (32, 8)))
def test_the_seam_refuses_a_list_group_it_does_not_rank(list_groups):
    """A side's list group is a liveness group or its half (docs/internals/facts/facts.md#list-groups): any other
    refused by name."""
    widths, L = (32, 32), 64
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    with pytest.raises((RuntimeError, ValueError), match="list group is"):
        _facts(routes, values, layout, read_statics, write_statics, list_groups=list_groups)


def test_the_seam_refuses_a_mask_past_the_rows():
    """A mask naming a digit past a level's rows would read another level's digits: refused, not guessed."""
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    wide = _sides(layout, "alternating", widths)
    routes, values = _bundle((12, 16), L, wide)
    with pytest.raises(RuntimeError, match="past its 12-digit rows"):
        _facts(routes, values, layout, *wide)


# --------------------------------------------------------------- the producer's bundle, end to end


def test_the_producers_bundle_through_its_topologys_statics():
    """A real producer's bundle: the layout and statics its topology declares, the kernel against the model."""
    import rola

    torch.manual_seed(0)
    routing = rola.Routing([rola.split_routing(32, sparse_duty="write"), rola.split_routing(32, sparse_duty="read")])
    layer = rola.RoLA(routing, hidden_size=64, num_heads=2, d_v=32).cuda().eval()
    with torch.no_grad():
        routes, values = layer._produce(torch.randn(2, 600, 64, device="cuda"))
    layout = facts_ops.call_layout(routes)
    read_statics, write_statics = facts_ops.call_statics(routes, layout)
    assert read_statics.dense_levels == (True, False) and write_statics.dense_levels == (False, True)
    got = _facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


# --------------------------------------------------------------- the folds it feeds


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_general_fold_at_the_atom_box_set(widths, L):
    """`page_bits` is not a second derivation: it is `page_activity` over `atom_boxes`."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    words = _facts(routes, values, layout, read_statics, write_statics).words.cpu()
    boxes = lv.atom_boxes(layout)
    assert len(boxes) == layout.atoms
    want = lv.page_activity(layout, boxes, lv.box_activity(words, layout, boxes, lv.READ, read_statics),
                            boxes, lv.box_activity(words, layout, boxes, lv.WRITE, write_statics))
    assert torch.equal(lv.page_bits(words, layout, read_statics, write_statics), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_leaf_level_truth_exactly(widths, L):
    """EXACT, not merely sound: at the atom grain the box->page map is a bijection, so a page carries a bit if and only
    if it holds a live leaf."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    words = _facts(routes, values, layout, read_statics, write_statics).words.cpu()
    got = lv.page_bits(words, layout, read_statics, write_statics)
    bits = lv.unpack(words, layout)
    strides = layout.radix_strides
    for side, flag in ((lv.READ, ATOM_READ), (lv.WRITE, ATOM_WRITTEN)):
        truth = torch.zeros_like(got, dtype=torch.bool)
        for bh in range(bits.shape[0]):
            for token in range(layout.L):
                leaves = torch.zeros(1, dtype=torch.int64)
                for level in range(layout.D):
                    base = layout.row_base(level)
                    if (read_statics, write_statics)[side].dense_levels[level]:
                        digits = torch.arange(layout.B[level])
                    else:
                        digits = bits[bh, side, base:base + layout.B[level], token].nonzero().flatten()
                    leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
                truth[bh, torch.unique(leaves // lv.ATOM_LEAVES)] = True
        assert torch.equal((got & flag).bool(), truth)
