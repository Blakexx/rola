# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY KERNEL AGAINST THE fp64 REFERENCE -- the executable spec of the C3 line.

THE GATE IS THE fp64 ORACLE, NOT A SHIPPED KERNEL. There is no bit-identity target
across a redesign; the only defensible reference is the arithmetic itself, and the band
is the repository's one bf16 family bound.

THE READOUT IS NOT BUILT ON THIS LINE. The kernel carries the frame, the state slice and
the fold; PHASE 1's four slots and the two transit legs are named empty stages, so ``num``
and ``den`` come back as the zeros the seam allocated. Every cell that asserts a readout is
therefore RED and stays red until the readout lands (card
`development/queue/C_CLEAN_SLATE.md`, "TEST-DRIVEN": no skips, no xfails). The cells whose
readout is zero BY THE ARITHMETIC -- a single window over an entry state nothing reads --
are green now and are the executable spec of everything the frame does own.

THE CELLS ARE DATA (rola-devtools' central registry, `rola_devtools.cells.carry`), read
through `benchmarks.cells`, so the oracle tier, the probe and the benches name one set of
cells and one draw. What a record declares is a SHAPE and a DRAW -- never an arm, never a
window, never a ``(k, m)`` box, because the axis law left the kernel nothing else to
be told.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``k_tok`` = a token's nonzero digits per sparse
level (a property of the DRAW, never told to the kernel), ``W`` = the kernel's fixed
window, which the reference mirrors because the inter/intra split is defined on it.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import by_name, carry_call, carry_cells, descriptor, launch, realize
from rola.ops import carry as carry_ops
from rola.ops.paging import bytes_equal
from tests.oracle import reference
from tests.oracle.fixtures import (
    assert_planted_errors_fail,
    assert_slots_close,
    canonical_from_plane,
    plane_from_canonical,
    relative,
)
from tests.oracle.tolerances import CARRY_DEN, CARRY_NUM, CARRY_STATE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def run(spec, state_in=None, state_out=None, page_table=None, bh=1, activity=None,
        schedule=None):
    """One kernel call on ``spec``. A carried cell given no plane binds the entry state drawn with it, advanced in
    place; a plane of the caller's is the caller's."""
    drawn, call = carry_call(spec, bh=bh)
    routes, v = call.pop("routes"), call.pop("v")
    if state_in is None and state_out is None and spec.state == "carried":
        state_in = state_out = plane_from_canonical(drawn.entry, bh)
    if activity is not None:
        call["activity"] = activity
    if state_out is None:
        state_out = carry_ops.state_plane(call["descriptor"], bh)
    num, den = carry_ops.carry_forward(routes, v, state_in=state_in, state_out=state_out,
                                       page_table=page_table, schedule=schedule, **call)
    return drawn, num, den, state_out


def entry(spec, drawn):
    """The entry state `run` binds when the caller gives no plane: the drawn one for a carried cell, else none."""
    return drawn.entry.reshape(1, 1, spec.N, spec.dv + 1) if spec.state == "carried" else None


def ref(drawn, state_in=None, magnitudes=False):
    """``(num, den, state)`` from the fp64 reference; with ``magnitudes``, the same run on ``|v|`` and ``|state_in|``:
    each slot's envelope (`tests.oracle.fixtures.assert_slots_close`)."""
    read, write, gain, v = drawn.doubles()
    if magnitudes:
        v, state_in = v.abs(), None if state_in is None else state_in.double().abs()
    return reference.inter_reference(read, write, gain, v, drawn.cell.widths,
                                     carry_ops.WINDOW, state_in=state_in)


def kernel_slots(spec, num, den, plane, bh=1):
    """The kernel's ``num``, ``den`` and state in the reference's layout."""
    return (num.reshape(bh, 1, spec.tokens, spec.dv).permute(0, 2, 1, 3),
            den.reshape(bh, 1, spec.tokens).permute(0, 2, 1), canonical_from_plane(plane))


def check(spec, drawn, num, den, plane, bh=1):
    """PER SLOT, against each slot's envelope: the numerator, the denominator (a sum of non-negative terms, so its own
    envelope) and the state."""
    want, env = ref(drawn, entry(spec, drawn)), ref(drawn, entry(spec, drawn), magnitudes=True)
    names = ("num", "den", "state")
    for got, w, e, name, output in zip(kernel_slots(spec, num, den, plane, bh), want, env, names,
                                       (CARRY_NUM, CARRY_DEN, CARRY_STATE)):
        w = w.reshape(got.shape) if name == "state" else w
        e = e.reshape(got.shape) if name == "state" else e
        assert_slots_close(got, w, output=output, envelope=e, what=f"{spec.name} {name}")


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on the carry's numerator and state: on a real cell, the kernel's output passes, and a wiped
    median slot and the smallest slots moved past their allowance fail."""
    spec = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    drawn, num, den, plane = run(spec)
    (n_ref, _, s_ref), (n_env, _, s_env) = ref(drawn), ref(drawn, magnitudes=True)
    got_num, _, got_state = kernel_slots(spec, num, den, plane)
    assert_planted_errors_fail(got_num, n_ref, output=CARRY_NUM, envelope=n_env, what=f"{spec.name} num")
    assert_planted_errors_fail(got_state, s_ref.reshape(got_state.shape), output=CARRY_STATE,
                               envelope=s_env.reshape(got_state.shape), what=f"{spec.name} state")


# ------------------------------------------------------------- the registry itself

def test_the_registry_declares_only_lawful_shapes():
    """NON-VACUITY for the gate above: every cell's descriptor, launch shape and
    geometry block are built here, so a cell the kernel boundary would refuse fails HERE,
    where the reason is the registry, instead of as a mystery inside a call."""
    for spec in carry_cells():
        desc = descriptor(spec)
        shape = launch()
        carry_ops.geometry_block(desc, shape)
        assert desc.DV in carry_ops.SHIPPED_DV
        assert shape.warps_per_cta in carry_ops.WARPS_PER_CTA


# ------------------------------------------------- the arithmetic, cell by cell

@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_term_is_the_fp64_reference(spec):
    """``num``, ``den`` AND the state plane, against the recurrence in CANONICAL order.

    The state is asserted on every cell, not only the small ones: it is the accumulator
    in the write carve's order, so a wrong permutation, a wrong swizzle or a dropped mass
    column cannot pass, and it is the only output carrying a cross-window rejoin error
    that the readout of the SAME window would hide."""
    drawn, num, den, plane = run(spec)
    check(spec, drawn, num, den, plane)


#: the cells whose sequence outruns one window, so their fold carries deposits from a
#: window that has already ended -- both backings and the partial trailing window.
MULTI_WINDOW = [cell for cell in ORACLE_CELLS if cell.tokens > carry_ops.WINDOW]
MULTI_WINDOW_IDS = [cell.name for cell in MULTI_WINDOW]


@pytest.mark.parametrize("spec", MULTI_WINDOW, ids=MULTI_WINDOW_IDS)
def test_the_folded_state_is_the_fp64_reference(spec):
    """THE FOLD ALONE, over more windows than one, against the reference's state.

    The state plane IS the fold's output: every deposit of every window lands in it, in
    the write carve's own order, so a wrong coefficient, a wrong token pairing, a stale
    segment slot or a dropped tail window all show here. It is asserted separately from
    ``num``/``den`` because those are the READOUT's, which is not built on this line --
    the cell's own row in `test_the_inter_term_is_the_fp64_reference` stays red until it
    is, and this row is the part the fold owns.
    """
    drawn, _, _, plane = run(spec)
    (_, _, s_ref), (_, _, s_env) = ref(drawn, entry(spec, drawn)), ref(drawn, entry(spec, drawn), magnitudes=True)
    assert_slots_close(canonical_from_plane(plane), s_ref.reshape(1, spec.N, spec.dv + 1),
                       output=CARRY_STATE, envelope=s_env.reshape(1, spec.N, spec.dv + 1), what=f"{spec.name} state")


def test_the_folded_state_is_the_fp64_reference_on_the_paged_backing():
    """The same claim on the PAGED backing, under a permuted slot table: the fold's
    deposits reach the same leaves whether the page is where its atom says or somewhere
    else the table names."""
    spec = next(c for c in MULTI_WINDOW if c.backing == "paged")
    desc = descriptor(spec)
    pages = desc.N // carry_ops.PAGE_LEAVES
    gen = torch.Generator(device="cuda").manual_seed(spec.seed)
    slots = torch.argsort(torch.rand(pages, device="cuda", generator=gen))
    #: a POOL, which is what a table's slots index (`rola.ops.carry._refuse_state`), here holding a page per atom
    pool = carry_ops.state_plane(desc, 1).reshape(pages, carry_ops.PAGE_LEAVES, desc.cols)
    drawn, _, _, pool = run(spec, state_out=pool,
                            page_table=slots.to(torch.int32).reshape(1, pages))
    (_, _, s_ref), (_, _, s_env) = ref(drawn), ref(drawn, magnitudes=True)
    assert_slots_close(canonical_from_plane(pool[slots].unsqueeze(0)), s_ref.reshape(1, spec.N, spec.dv + 1),
                       output=CARRY_STATE, envelope=s_env.reshape(1, spec.N, spec.dv + 1),
                       what=f"{spec.name} paged state")


def test_a_carried_state_is_advanced_in_place():
    """THE CONTINUATION: the same plane in and out, holding the cell's drawn entry state, with the reference chained
    from the same entry. A stateful chain allocates ONE plane for the sequence."""
    spec = next(c for c in ORACLE_CELLS if c.name == "flat-small-carried")
    drawn = realize(spec)
    plane = plane_from_canonical(drawn.entry)
    _, num, den, advanced = run(spec, state_in=plane, state_out=plane)
    assert advanced is plane
    check(spec, drawn, num, den, plane)


def test_an_idle_resident_page_is_neither_loaded_nor_stored():
    """the cell: a call reaching a STRICT SUBSET of the pages a previous one wrote
    must neither load nor store the rest.

    The census is ASSERTED, not inferred: an idle page's bytes must be BIT-identical
    across the second call, and perturbing one must not move the readout. The truncated
    draw (``support`` below one) is what produces an untouched page at all -- an
    unstructured ``k_tok`` still reaches every page at this length."""
    first = next(c for c in ORACLE_CELLS if c.name == "flat-small-dense")
    second = next(c for c in ORACLE_CELLS if c.name == "flat-small-idle-resident")
    plane = carry_ops.state_plane(descriptor(first), 1)
    run(first, state_out=plane)

    before = plane.clone()
    _, num, den, plane = run(second, state_in=plane, state_out=plane)
    moved = (before != plane).any(-1).any(-1)
    assert not bool(moved.all()), "the truncated cell reached every page the first did"

    control = before.clone()
    _, num_c, den_c, _ = run(second, state_in=control, state_out=control)

    perturbed = before.clone()
    perturbed[~moved] = -1234.5
    _, num2, den2, _ = run(second, state_in=perturbed, state_out=perturbed)

    #: THE BOUND IS THE REDUCTION FLOOR, MEASURED HERE, not bit equality. ``num``/``den``
    #: are a plane the readout REDUCES into with one atomic contribution per owner box per
    #: token, and the order of those contributions is the hardware's -- so the same call
    #: twice already differs at fp32 reduction-rounding scale (``num_c`` above measures it
    #: on this very cell). The claim under test is that the perturbed page's CONTENT does
    #: not reach the readout, and the perturbation is ~1e9 in magnitude against a readout of
    #: ~1e-4: a leak of even one leaf would exceed this bound by twelve orders of magnitude,
    #: so the test is sharp, not slackened.
    floor = max(float((num - num_c).abs().max()), float(num.abs().max()) * 2.0 ** -16)
    d_floor = max(float((den - den_c).abs().max()), float(den.abs().max()) * 2.0 ** -16)
    assert float((num - num2).abs().max()) <= floor and float((den - den2).abs().max()) <= d_floor, (
        "perturbing an idle-resident page moved the readout: the entry sweep is reading "
        "a page this call does not read")


# ------------------------------------------------- the dense / paged byte gate

def test_the_two_backings_write_the_same_pages():
    """ONE KERNEL, ONE ABI. The dense backing is the paged one under the identity slot
    table, so a paged run whose table is a PERMUTATION of the slots must write, page for
    page, what the dense run wrote -- to the fold's own floor: the segment locks leave the
    order in which warps add into a state word to timing, so two runs of ONE backing differ
    by rounding, and that measured difference is the bound the other backing sits inside.
    An addressing bug moves whole leaves and exceeds it by orders of magnitude.

    A permutation rather than the identity is the point: an identity table would also
    pass for a kernel that ignored the table entirely, which is the shape this gate
    exists to exclude."""
    spec = next(c for c in ORACLE_CELLS if c.backing == "paged")
    desc = descriptor(spec)
    pages = desc.N // carry_ops.PAGE_LEAVES

    dense = carry_ops.state_plane(desc, 1)
    run(spec, state_out=dense)
    again = carry_ops.state_plane(desc, 1)
    run(spec, state_out=again)

    gen = torch.Generator(device="cuda").manual_seed(spec.seed)
    slots = torch.argsort(torch.rand(pages, device="cuda", generator=gen))
    #: a POOL of one page an atom, which is the paged state's shape (`rola.ops.carry._refuse_state`)
    paged = carry_ops.state_plane(desc, 1).reshape(pages, carry_ops.PAGE_LEAVES, desc.cols)
    run(spec, state_out=paged, page_table=slots.to(torch.int32).reshape(1, pages))

    floor = max(float((dense - again).abs().max()), float(dense.abs().max()) * 2.0 ** -16)
    assert float((paged[slots] - dense[0]).abs().max()) <= floor, (
        "the paged backing wrote different pages than the dense one; the two backings are "
        "one kernel, so a difference past the fold's floor is an addressing bug")


# --------------------------------------------------------------- the lattice

def test_lattice_is_a_permutation():
    for widths, k, m in (((64, 64), 4, 8), ((64, 64), 4, 4), ((256, 256), 4, 8)):
        pi = carry_ops.lattice_of_canonical(widths, k, m)
        n = 1
        for width in widths:
            n *= width
        assert torch.equal(torch.sort(pi).values, torch.arange(n, device=pi.device))
        _, _, _, bc, owners = carry_ops.box_shape(widths, k, m)
        assert int((pi // bc).max()) == owners - 1


def test_the_decomposition_is_exact():
    """``inter(W) + intra(W)`` is the SAME full recurrence for every ``W``, and equals
    the canonical fp64 oracle. This is the study's no-double-count claim, and the reason
    an inter-only kernel can be gated at all. It runs NO kernel, which is why it is the
    one test in this file that is green today."""
    from rola.ops.constants import READOUT_EPS
    from rola.ops.naive import naive_rola
    from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology

    spec = next(c for c in ORACLE_CELLS if c.name == "flat-small-dense")
    drawn = realize(spec)
    read, write, gain, v = drawn.doubles()
    tot = None
    for W in (1, 8, 64):
        ni, di, _ = reference.inter_reference(read, write, gain, v, spec.widths, W)
        na, da = reference.intra_reference(read, write, gain, v, spec.widths, W)
        cur = (ni + na, di + da)
        if tot is None:
            tot = cur
        else:
            assert relative(cur[0], tot[0]) < 1e-12
            assert relative(cur[1], tot[1]) < 1e-12
    routing = IndependentRouting(width=1, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(w) for w in spec.widths))
    y, _ = naive_rola(v, read, write, gain, topo, None, output_final_state=False)
    mine = tot[0] / (tot[1][..., None] + READOUT_EPS)
    assert relative(mine, y) < 1e-10


# ------------------------------------------- what the frame owns, against the oracle

def _honest_activity(drawn, descriptor, bh=1):
    """The activity bits a facts pass would emit for this draw: a page is READ when some
    token's read amplitudes reach one of its leaves, WRITTEN likewise.

    The registry's default is the conservative "every page, both bits", which is the truth
    a caller with no facts pass may state; this is the tight one, and the two must give
    the same answer -- the bits are permission to skip, never a change of arithmetic."""
    from rola.ops.carry import ACTIVITY_READ, ACTIVITY_WRITTEN, PAGE_LEAVES

    read, write, _, _ = drawn.doubles()
    live = torch.zeros(descriptor.N // PAGE_LEAVES, dtype=torch.uint8, device="cuda")
    for side, bit in ((read, ACTIVITY_READ), (write, ACTIVITY_WRITTEN)):
        pages = reference.leaf_product(side, descriptor.B).reshape(-1, descriptor.N)
        touched = (pages != 0).any(0).reshape(-1, PAGE_LEAVES).any(-1)
        live |= touched.to(torch.uint8) * bit
    return live.expand(bh, -1).contiguous()


def _occupied_plane(spec, bh=1):
    """A state plane carrying an arbitrary LAWFUL state: a random fp32 accumulator put
    into the stored split-plane form, so every page holds a state some sequence could
    have written rather than bit patterns no fp32 accumulator ever produces."""
    from rola.ops.paging import to_split_planes

    desc = descriptor(spec)
    gen = torch.Generator(device="cuda").manual_seed(spec.seed)
    logical = torch.randn((bh, desc.N // carry_ops.PAGE_LEAVES, carry_ops.PAGE_LEAVES, desc.cols),
                          device="cuda", generator=gen)
    return to_split_planes(logical).reshape(logical.shape).contiguous()


@pytest.mark.parametrize("name", ["identity-passthrough-dense", "identity-passthrough-paged"])
def test_a_window_with_nothing_live_passes_the_state_through_bit_for_bit(name):
    """THE RECURRENCE'S IDENTITY. No token reads and no token writes, so the state is the
    state: not close to it, THE SAME BYTES.

    Both ends of the entry sweep are covered. Under the tight activity bits the box is
    idle and the prologue dies before it touches shared memory; under the conservative
    bits the box loads its pages, folds nothing and stores them back, and the split-plane
    round trip has to be exact for the bytes to survive. A kernel that rounded anywhere in
    its state path fails the second case while passing the first."""
    spec = by_name(name)
    desc = descriptor(spec)
    drawn = realize(spec)
    pages = desc.N // carry_ops.PAGE_LEAVES
    table = None
    if spec.backing == "paged":
        gen = torch.Generator(device="cuda").manual_seed(spec.seed)
        slots = torch.argsort(torch.rand(pages, device="cuda", generator=gen))
        table = slots.to(torch.int32).reshape(1, pages)

    for activity in (_honest_activity(drawn, desc), None):
        plane = _occupied_plane(spec)
        if table is not None:
            #: with a table the state is the POOL its slots index, one page an atom here
            plane = plane.reshape(pages, carry_ops.PAGE_LEAVES, desc.cols)
        before = plane.clone()
        _, num, den, _ = run(spec, state_in=plane, state_out=plane, page_table=table,
                             activity=activity)
        assert bytes_equal(plane, before), (
            f"the state moved through a window with nothing live (activity="
            f"{'tight' if activity is not None else 'conservative'})")
        assert not num.any() and not den.any(), "a dead read side read something"


def test_one_token_writing_one_leaf_is_the_oracles_single_deposit():
    """THE FOLD'S SINGLE-TILE POINT, against the fp64 recurrence.

    One token, one-hot on digit zero of every write level, so the whole window's deposit is
    one leaf receiving ``gain * [v, 1]``. It is the smallest cell that pins every map the
    fold composes at once: the token's position inside its segment (the k step), the value
    channels (the accumulator's m), the leaf (its n) and the mass column -- a permutation
    of any one of them sends the deposit somewhere else or zeroes it.

    The readout is asserted too, and its expected value is ZERO by the arithmetic and not
    by omission: a single window reads the state it ENTERED with, and this cell enters with
    nothing."""
    spec = by_name("single-deposit")
    drawn, num, den, plane = run(spec)
    check(spec, drawn, num, den, plane)
    assert not num.any() and not den.any()
    canon = canonical_from_plane(plane)[0]
    assert canon[0].any(), "the single deposit did not land"
    assert not canon[1:].any(), "the single deposit landed on more than one leaf"


# ------------------------------------------- the SPARSE bodies, over the density sweep

#: THE TWO ORDER POLICIES a shipped arm can run: the first-live-box sort (the default, the
#: reap) and the identity (token order, every token tiled). Both are the SAME arithmetic;
#: what changes is which (tile, box) pairs the kernel skips, so both must be the fp64
#: reference and the pair is the sort's own A/B.
ORDER_ROWS = [(cell, order) for cell in ORACLE_CELLS for order in carry_ops.ORDER_POLICIES]
ORDER_IDS = [f"{cell.name}-{order}" for cell, order in ORDER_ROWS]


@pytest.mark.parametrize("spec,order", ORDER_ROWS, ids=ORDER_IDS)
def test_the_sparse_form_is_the_fp64_reference(spec, order):
    """THE REAP CHANGES NO SUM. A dead (tile, box) pair's every coefficient is a product one
    of whose factors is an amplitude of exactly zero -- that is the liveness pass's own
    predicate -- so the term the kernel never forms was exactly ``+0.0f``, under either
    order. The cells sweep the DENSITY (dense, ``alt`` at two ``k_tok``, ``cohort``, the two
    structured truncations, the degenerate draws), which is the whole point: an order that
    is right at one density and wrong at another would pass a single-cell row.
    """
    drawn, num, den, plane = run(spec, schedule=carry_ops.CarrySchedule(order=order))
    check(spec, drawn, num, den, plane)


#: the fully dense cell the two orders are compared ON: every (tile, box) pair is live, so
#: the sort reaps NOTHING and the two orders tile the same pairs.
BYTE_GATE_CELL = "flat-small-dense"
ORDER_GATE_CELL = "flat-small-multiwindow"


def test_the_two_orders_agree_where_there_is_nothing_to_reap():
    """THE IDENTITY IS THE REFERENCE. At a fully dense draw every token is live in every box,
    so the two orders tile the same (tile, box) pairs; ``num``/``den`` agree to the reduction
    floor, because the readout's contributions are reduced into the planes in an order the
    hardware picks (the seam's declared contract, K2 06.1), and the STATE -- whose fold
    write-back order across warps the segment locks leave to timing -- to the same floor. A
    leak of one leaf would exceed that floor by orders of magnitude. Two windows, so the
    readout is against a folded state and not the zero one.
    """
    spec = by_name(ORDER_GATE_CELL)
    ident = carry_ops.CarrySchedule(order=carry_ops.ORDER_IDENTITY)
    _, n_u, d_u, s_u = run(spec, schedule=ident)
    _, n_s, d_s, s_s = run(spec, schedule=carry_ops.CarrySchedule())
    #: the floor is MEASURED on this cell, not assumed: two identity runs differ by the
    #: reductions' order alone, and that difference is the bound the sorted run has to sit
    #: inside.
    _, n_c, d_c, s_c = run(spec, schedule=ident)
    s_u, s_s, s_c = (canonical_from_plane(x) for x in (s_u, s_s, s_c))
    for a, b, c in ((n_u, n_s, n_c), (d_u, d_s, d_c), (s_u, s_s, s_c)):
        floor = max(float((a - c).abs().max()), float(a.abs().max()) * 2.0 ** -16)
        assert float((a - b).abs().max()) <= floor


def test_the_selection_rule_picks_the_first_box_order_at_every_mode():
    """The launch's own choice: the first-live-box order whatever the level modes declare
    (the order study, 2026-09-08: within 1.3x of the optimum at alt-k4, 3-4x below token
    order; at dense the two policies tile the same work)."""
    spec = by_name(BYTE_GATE_CELL)
    desc = descriptor(spec)
    for modes in (0, carry_ops.SIDE_SPARSE[0], carry_ops.SIDE_SPARSE[1]):
        assert carry_ops.select_schedule(desc, modes).order == carry_ops.ORDER_FIRST_BOX
