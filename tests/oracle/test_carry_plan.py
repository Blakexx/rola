"""Gate for THE PLAN PASS (csrc/rola/src/carry/ampere/schedule.cuh `window_plan_kernel`): every window's record for every
(batch-head, box class) is the plans' model's (`tests/oracle/carry_plan.py`), plan word and step words word for word,
on every form cell and under a spread of forced forms; and the rule the model states never places a fill over anything
live (docs/internals/carry/ampere/schedule.md#pass).

Symbols: ``W`` the window (512 tokens), a box class a pair of the two list levels' 32-digit groups, ``B`` the area's
bytes, ``n`` a window's entries.
"""
from __future__ import annotations

import random

import pytest
import torch

from rola.engine.carve import carve_at, plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from tests.oracle import carry_plan as cp
from tests.oracle.carry_fixtures import (
    FORM_CELLS,
    assert_box64_arm,
    box64,
    box64_dense,
    box64_same,
    dense_bundle,
    listed_bundle,
    read_listed_bundle,
)

#: the forced forms gated: none, then each image mode and each chunk choice at least once
FORCED = (None, "whole/union/union", "halves/64/128", "halves/union/64", "whole/128/192", "rule/64/rule",
          "halves/rule/union")


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize(("widths", "live"), sorted({(cell[0], cell[1]) for cell in FORM_CELLS}))
def test_the_plans_are_the_model(widths, live, forced):
    """Each window's plan word (its mode, chunks and their counts) and every step word (the entry's place, filled ahead,
    the next filled now, the drain's rendezvous) as the model plans them -- so the forms the carry's gates assert by
    are the kernel's."""
    routes, values = listed_bundle(widths, 2048, 0, live=live)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced)


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize(("widths", "live"), sorted({(cell[0], cell[1]) for cell in FORM_CELLS}))
def test_the_64_channel_box_plans_are_the_model(widths, live, forced):
    """The 64-channel box: a read slot's panel and den words twice the 32-channel box's and the read side's box classes
    its 16-digit list groups -- every record the model's word for word."""
    routes, values = listed_bundle(widths, 2048, 0, live=live)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced, box=box64(0))


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize("read_sparse", [0, 1])
def test_a_dense_write_sides_64_channel_box_plans_are_the_model(read_sparse, forced):
    """The dense write side's 64-channel box (16 lists x 32 rows: 224-byte write rows): every record the model's word
    for word."""
    routes, values = read_listed_bundle((64, 64), 2048, read_sparse)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced,
                                box=box64_dense(read_sparse))


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize("widths", [(64, 64), (128, 32)])
def test_a_dense_read_sides_plans_are_the_model(widths, forced):
    """Both sides dense (the 32-digit x K 16 x 64-channel box: a read slot's exit words where a listed side's entries
    are): every record the model's word for word."""
    routes, values = dense_bundle(widths, 2048)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced)


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize("level", [0, 1])
def test_the_same_splits_64_channel_box_plans_are_the_model(level, forced):
    """Both sides listing one level at the 64-channel box (16-digit list groups on both sides): every record the
    model's word for word."""
    routes, values = listed_bundle((64, 64), 2048, level, same=True)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced,
                                box=box64_same(level))


def _assert_plans_are_the_model(routes, values, layout, statics, forced=None, box=None):
    """The plan pass's written words for every (batch-head, box class, window) against the model's, at the planner's
    box or at ``box`` (extents a level, channels); returns the model's records."""
    padded, d_v = tuple(routes.topology.padded_widths), values.v.shape[-1]
    carve = plan_carve(padded, *routes.sparse, d_v) if box is None else carve_at(padded, *routes.sparse, *box, d_v)
    if box is not None:
        assert_box64_arm(carve)
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.lists_form(carry_ops.arm_key(carve)))
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    *_, plans = carry_ops.carry(routes, values, facts, carve, unit, forced=forced)
    torch.cuda.synchronize()
    words = plans.cpu().view(torch.int32).long() & 0xFFFFFFFF
    want = cp.plans(routes, facts, carve, forced)
    classes = plans.shape[2]
    for box, windows in enumerate(want):
        bh, cls = divmod(box, classes)
        for w, rec in enumerate(windows):
            expect = rec.words()
            assert tuple(words[bh, w, cls, :len(expect)].tolist()) == expect, (box, w)
    return want


def test_a_window_that_reads_nothing_is_planned_as_the_model():
    """A window whose box reads nothing (a cohort draw, whose groups go unread for whole windows) before one that reads:
    the records are the model's. Its successor's first read is filled in one of its steps or at its own, all of which
    run past the store, whose work stages the inputs the fill reads (docs/internals/carry/ampere/schedule.md#steps); the
    successor's readout is held to the oracle by `tests/oracle/test_carry_vs_oracle.py`."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of

    spec = by_name("flagship-cohort-k4-L4096")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    want = _assert_plans_are_the_model(routes, values, layout, statics)
    nothing_then_reads = sum(int(cp.chunks(windows[w - 1].word)[3] == 0 and cp.chunks(windows[w].word)[3] > 0)
                             for windows in want for w in range(1, len(windows)))
    assert nothing_then_reads > 0, "no window reading nothing precedes one that reads: the cell is vacuous"


def test_grouped_plans_are_the_model():
    """GROUPED QUERIES (four query heads a write head): the read union is the group's merged rows, so a window's read
    chunks, its records' size and the area are the group's (`tables_of`); the records are the model's word for word."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of

    spec = by_name("flagship-alt-k4-g4")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    assert routes.group == 4
    want = _assert_plans_are_the_model(routes, values, layout, statics)
    assert max(cp.chunks(rec.word)[3] for windows in want for rec in windows) > 1, (
        "no window reads more than one chunk: the group's union never passes one head's buffer")


def _sides(rng: random.Random):
    """A random window's counts: each side's union (none, small or full) and segment sums of up to 32 lists a slot."""
    def side():
        W = window_of()
        u = rng.choice((0, rng.randrange(8, 128, 8), rng.randrange(128, W, 8), W))
        density = rng.choice((1, 4, 16, 32))
        totals, run = [], 0
        for j in range(cp.list_at()):
            slots = max(0, min(u, cp.LIST_BOUND * (j + 1)) - cp.LIST_BOUND * j)
            run += rng.randint(0, density * slots)
            totals.append(run)
        return u, tuple(totals)
    (ru, rs), (fu, fs) = side(), side()
    return cp.Counts(ru, rs, fu, fs)


@pytest.mark.parametrize("seed", range(4))
def test_the_rule_never_fills_over_what_is_live(seed):
    """THE RULE'S SAFETY on random windows of this card's arm: every entry inside the area and every read above its
    image; at each step the entry computing, the entry filled ahead beside it and the live image pairwise clear; a
    deferred fill clear of the image where it is live; the next window's first clear of the last entry and above its own
    image. A fill the rule cannot fit beside the live entry waits for its own step -- never a byte shared. And as the
    carry runs it (every entry filled at its own step, a warp's fill right after its own drain): a fill that meets the
    panel of the read before it -- the bytes slower warps may still be draining -- is marked for the rendezvous."""
    f = carry_ops.form((32, 1, 32, 1, 32, 0))
    area, read_row, write_row = f.area, f.read_row, f.write_row
    rng = random.Random(seed)
    wins = [cp.win_of(_sides(rng), -1, read_row, write_row, area) for _ in range(64)]
    recs = cp.records(wins, read_row, write_row, area)
    for w, (x, rec) in enumerate(zip(wins, recs)):
        for t in range(x.n):
            at, ahead, nxt, _ = rec.steps[t]
            b = cp.entry_bytes(x, t, read_row, write_row)
            read = t < x.nr
            assert at >= 0 and at + b <= area, (w, t)
            if read:
                assert at >= x.image, (w, t, "a read under the image")
            if nxt:
                if t + 1 < x.n:
                    nat, nb = rec.steps[t + 1][0], cp.entry_bytes(x, t + 1, read_row, write_row)
                else:
                    nx = wins[w + 1]
                    nat, nb = recs[w + 1].steps[0][0], cp.entry_bytes(nx, 0, read_row, write_row)
                    assert recs[w + 1].steps[0][1] == 1
                    assert nat >= nx.image, (w, t, "the next window's first under its image")
                assert cp.disjoint(at, b, nat, nb), (w, t, "a fill over the computing entry")
                if read:
                    assert nat >= x.image, (w, t, "a fill over the live image")
            if ahead and t > 0:
                assert rec.steps[t - 1][2] == 1, (w, t, "filled ahead by a step that did not fill it")
            if 0 < t <= x.nr:
                off, nbytes = cp.drained_bytes(x, t - 1, read_row)
                if not cp.disjoint(at, b, rec.steps[t - 1][0] + off, nbytes):
                    assert rec.steps[t][3] == 1, (w, t, "a fill over the panel being drained, unmarked")
