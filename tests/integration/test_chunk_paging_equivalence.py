# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Dense and paged are ONE kernel and ONE base translation -- bit for bit.

The carry kernel's state I/O addresses a page per atom. Which page is the only thing a
backing changes: without a table the slot IS the atom and the plane is the dense
``[BH, N/16, 16, cols]`` one; with a table the slot is what the plan committed and the
plane holds only those pages. Same instruction stream, same accumulation order, same
ABI -- so the claim here is BYTE identity and not a tolerance. A tolerance would be the
wrong instrument entirely: any difference at all is an ADDRESSING difference, and
addressing is exact or wrong.

THE CELLS ARE THE REGISTRY'S (``benchmarks/cells``) and the call is the registry's
``carry_call``, so what runs here is the same declaration, the same draw and the same
operands the oracle tier and the benches run. A record names a SHAPE and a DRAW: the
structured-support records are what make an absent atom possible at all, since
unstructured sparsity at long ``L`` still reaches every atom.

THESE CELLS ARE RED UNTIL THE KERNEL EXISTS (card ``development/queue/C_CLEAN_SLATE.md``,
"TEST-DRIVEN"): every claim below reaches ``rola.ops.carry.carry_forward``, which refuses
by name on this line. That is the honest red -- not a skip, not an xfail.

Vocabulary, defined here and used unqualified below: ``D`` = level count; ``B_l`` =
level ``l``'s padded digit count; ``N = prod_l B_l`` = leaves; ``L`` = tokens;
``BH = batch * heads``; ``DV`` = the padded value width; ``cols = DV + 1``; **atom** =
the 16-leaf page granule (``rola.ops.paging.MMA_K_QUANTUM``); **slot** = the physical
page an atom is mapped to, ``-1`` for an atom the plan committed nothing for;
``box_leaves`` = the leaves one owner block owns, derived from ``(DV, warps_per_cta)``;
``owners = N / box_leaves``.

**THE PLAN IS THE SHIPPED ONE, UNMODIFIED.** The bitmap planned here is
``rola.engine.facts.planes.atom_bits`` exactly as published -- the kernel's own exact
write set, canonically keyed -- and the table is ``PageArena.plan_exact``'s. One
representation, shared by the arena, the descriptor and the decode step, is what makes
this file's byte claim a statement about addressing rather than about a convention.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import carry_call, carry_cells
from rola.engine.facts import planes
from rola.ops import carry as carry_ops
from rola.ops.paging import MMA_K_QUANTUM, PageArena, bytes_equal, from_split_planes
from tests.oracle.fixtures import assert_fresh_binary

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the carry is a CUDA kernel"),
]

_POISON = -1234.5

#: The shapes this file runs over: one cell per topology the registry declares, at full
#: support and the dense backing. The backing is what the file VARIES, so a cell that
#: declares the paged one would be declaring the answer.
SHAPES = tuple(c for c in carry_cells()
               if c.backing == "dense" and c.state == "fresh" and c.support == 1.0
               and c.draw in ("dense", "alt"))
SHAPE_IDS = [c.name for c in SHAPES]

#: The structured-support cells: a truncation to a level's FIRST digits, so a smaller
#: support is a SUBSET of a larger one at every level and therefore over the atoms too.
#: That subset property is what a continuation's second call needs, and it holds across
#: two records because it is a property of the truncation and not of the draw.
STRUCTURED = tuple(c for c in carry_cells() if c.support < 1.0 and c.state == "fresh")
STRUCT_IDS = [c.name for c in STRUCTURED]

#: ``(wider, narrower)`` at one shape -- the continuation pair.
PAIRS = tuple((a, b) for a in STRUCTURED for b in STRUCTURED
              if a.widths == b.widths and a.dv == b.dv and a.support > b.support)
PAIR_IDS = [f"{a.name}->{b.name}" for a, b in PAIRS]


def _cols(spec) -> int:
    return spec.dv + 1


def _box_leaves(spec) -> int:
    return carry_ops.box_leaves(spec.dv, spec.warps_per_cta)


def _activity(spec, drawn, bh=1):
    """The kernel's own EXACT atom-grain ACTIVITY byte, as the shipped pass publishes it.

    Bit 0 WRITTEN, bit 1 READ, over BOTH sides' packed planes. No re-keying: the kernel
    addresses its state by the CANONICAL atom id this byte is indexed by, so the plan the
    engine would build is the plan this file hands the arena and the byte the engine
    would pass is the byte this file hands the kernel.
    """
    def side(levels):
        return torch.cat(levels, -1).permute(0, 2, 1, 3).contiguous()

    bits = planes.atom_bits(side(drawn.write), spec.widths, read_plane=side(drawn.read))
    return bits.reshape(bh, -1)


def _bind(spec, bh=1):
    """The registry's call, with the CELL's own activity in place of the conservative one."""
    drawn, call = carry_call(spec, bh=bh)
    call["activity"] = _activity(spec, drawn, bh)
    return drawn, call


def _arena(spec, bh, pool_slack=0.0):
    return PageArena(widths=spec.widths, BC=_box_leaves(spec), cols=_cols(spec), BH=bh,
                     device=torch.device("cuda"), pool_slack=pool_slack)


def _logical(plane, spec, bh=1):
    """A state plane as the logical ``[BH, N, cols]`` fp32 view."""
    return from_split_planes(plane).view(bh, spec.N, _cols(spec))


def _run(spec, call, *, state_in=None, state_out=None, page_table=None):
    routes, v = call["routes"], call["v"]
    rest = {k: call[k] for k in ("descriptor", "geometry", "liveness", "activity", "launch")}
    return carry_ops.carry_forward(routes, v, state_in=state_in, state_out=state_out,
                                   page_table=page_table, **rest)


def _dense_run(spec, call, *, state_in=None, plane=None, bh=1):
    if plane is None:
        plane = carry_ops.state_plane(call["descriptor"], bh)
    num, den = _run(spec, call, state_in=state_in, state_out=plane)
    return num, den, plane


def _paged_run(spec, call, arena, *, state_in=None, bh=1):
    arena.wait()
    num, den = _run(spec, call, state_in=state_in, state_out=arena.state,
                    page_table=arena.page_table)
    return num, den, arena.materialize().reshape(bh, spec.N, _cols(spec))


def _stored(arena) -> torch.Tensor:
    """The arena's pages gathered into CANONICAL atom order, still in the STORED form.

    ``materialize`` rejoins hi||lo; this one does not, so the comparison it feeds is the
    BYTE claim over the split layout itself -- a backing that agreed on values while
    disagreeing on which plane a half lives in would pass the first and fail this.
    """
    table = arena.page_table
    gathered = arena.state.index_select(0, table.long().clamp_min(0).reshape(-1))
    stored = torch.zeros_like(gathered)
    present = (table >= 0).reshape(-1)
    stored[present] = gathered[present]
    return stored


def _num_is_exact(spec) -> bool:
    """``num`` is fanned in with ``atomicAdd`` across owner blocks, so at ``owners > 1``
    the path is not bit-reproducible against ITSELF and the comparison is against the
    reordering envelope with a null control asserted to fit inside it. The STATE is exact
    on every cell regardless: each atom is stored by exactly one CTA."""
    return spec.N // _box_leaves(spec) == 1


def _envelope(num, spec):
    owners = spec.N // _box_leaves(spec)
    return owners * owners * torch.finfo(torch.float32).eps * num.abs().max()


def _assert_num(dense, paged, control, spec, what):
    if _num_is_exact(spec):
        assert torch.equal(dense, control), "num is not reproducible against itself"
        assert torch.equal(dense, paged), what
    else:
        bound = _envelope(dense, spec)
        assert (dense - control).abs().max() <= bound, "the null control blew its envelope"
        assert (dense - paged).abs().max() <= bound, what


# ---------------------------------------------------------------------------
# 1. dense and paged are the same call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", STRUCTURED + SHAPES, ids=STRUCT_IDS + SHAPE_IDS)
def test_the_paged_backing_is_bit_identical_to_the_dense_one(spec):
    assert_fresh_binary()
    drawn, call = _bind(spec)
    dense_plane = carry_ops.state_plane(call["descriptor"], 1)
    num_dense, _, _ = _dense_run(spec, call, plane=dense_plane)
    num_control, _, _ = _dense_run(spec, call)

    arena = _arena(spec, 1)
    result = arena.plan_exact(planes.written_atoms(call["activity"]))
    num_paged, _, paged = _paged_run(spec, call, arena)
    dense = _logical(dense_plane, spec)

    assert result.allocated_pages > 0 and (paged != 0).any(), (
        "the plan committed nothing, or the pages came back empty")
    identity = torch.arange(arena.logical_pages, device="cuda",
                            dtype=torch.int32).view_as(arena.page_table)
    assert spec.support == 1.0 or not torch.equal(arena.page_table, identity), (
        "the plan mapped every atom to its own id, so this cell never exercised "
        "a translation at all")
    assert bytes_equal(_stored(arena),
                       dense_plane.reshape(-1, MMA_K_QUANTUM, _cols(spec))), (
        "the two backings disagree on the STORED bytes: the split layout is part of "
        "the backing-invisibility claim, not only the values it rejoins to")
    assert bytes_equal(dense, paged), "the paged state differs from the dense one"
    _assert_num(num_dense, num_paged, num_control, spec,
                "the paged num differs from the dense num")


# ---------------------------------------------------------------------------
# 2. a zero page is inert
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", STRUCTURED, ids=STRUCT_IDS)
def test_a_zero_page_is_inert(spec):
    """Force atoms the routing never writes to be resident, and nothing moves.

    This is what licenses every SUPERSET residency derivation as safe-direction, and it
    is the mechanical form of the self-normalizing readout's dead-leaf argument.
    """
    assert_fresh_binary()
    _drawn, call = _bind(spec)
    exact = planes.written_atoms(call["activity"])
    assert not bool(exact.all()), (
        f"{spec.name} declares support {spec.support} and still reaches every atom; "
        f"a structured-support record exists precisely so that it does not")

    exactly = _arena(spec, 1)
    exactly.plan_exact(exact)
    num_exact, _, s_exact = _paged_run(spec, call, exactly)

    superset = _arena(spec, 1)
    superset.plan_exact(torch.ones_like(exact))
    num_super, _, s_super = _paged_run(spec, call, superset)

    #: THE NULL CONTROL is the exact plan run a second time into its own arena: at
    #: ``owners > 1`` the fan-in's atomics make ``num`` non-reproducible against ITSELF,
    #: so the control is what bounds the comparison. The STATE is exact on every cell.
    control = _arena(spec, 1)
    control.plan_exact(exact)
    num_control, _, _ = _paged_run(spec, call, control)

    assert superset.resident_pages > exactly.resident_pages, (
        "the superset admitted nothing extra")
    assert bytes_equal(s_exact, s_super), "a resident never-written page moved the state"
    _assert_num(num_exact, num_super, num_control, spec,
                "a resident never-written page moved num")


# ---------------------------------------------------------------------------
# 3. an absent atom is skipped, not written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", STRUCTURED, ids=STRUCT_IDS)
def test_an_absent_atom_is_skipped_and_never_written(spec):
    """A structured-support run leaves every unadmitted page exactly as it found it.

    Checked against a poison fill of the arena's whole backing: the exit epilogue stores
    only PLANNED atoms, so a page outside the plan keeps its poison, and ``num`` still
    equals the dense run's.
    """
    assert_fresh_binary()
    _drawn, call = _bind(spec)
    bitmap = planes.written_atoms(call["activity"])
    assert not bool(bitmap.all()), f"{spec.name} reaches every atom; none is absent to skip"

    num_dense, _, dense_plane = _dense_run(spec, call)
    num_control, _, _ = _dense_run(spec, call)
    dense = _logical(dense_plane, spec)

    arena = _arena(spec, 1)
    arena.plan_exact(bitmap)
    arena.wait()
    #: THE CLAIM LIVES ON THE RAW POOL, not on ``materialize()``, which gathers through
    #: the table and sends an absent atom to zeros BY CONSTRUCTION. What must not move is
    #: a physical slot the table names for nothing: an atom the plan skipped has no slot,
    #: so a kernel that addressed it anyway would scribble at a NEGATIVE displacement.
    #: The whole pool is poisoned -- planned slots included, which a fresh call rewrites
    #: whole -- and the unmapped slots must come back untouched.
    used = arena.page_table[arena.page_table >= 0].long()
    spare = torch.ones(arena.state.shape[0], dtype=torch.bool, device="cuda")
    spare[used] = False
    assert bool(spare.any()), "this arena's pool is exactly full; no unmapped slot to poison"
    arena.state.fill_(_POISON)
    num_paged, _, paged = _paged_run(spec, call, arena)

    live = bitmap.repeat_interleave(MMA_K_QUANTUM, dim=1)
    assert bytes_equal(dense[live], paged[live]), (
        "a planned atom differs across the backings")
    assert bool((arena.state[spare] == _POISON).all()), (
        "an unmapped slot was written: the exit epilogue is addressing atoms the plan "
        "committed nothing for")
    _assert_num(num_dense, num_paged, num_control, spec,
                "the skipping run's num differs from the dense num")


# ---------------------------------------------------------------------------
# 4. a reserved slack pool is bitwise invisible
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", STRUCTURED, ids=STRUCT_IDS)
def test_a_reserved_slack_pool_is_bitwise_invisible(spec):
    """A pool moves every residency slot and must move NO byte of the state.

    Its slots come off the SAME allocator, ahead of the routing's own, so a pooled arena
    assigns different slot ids to the same atoms -- and "which slot" is exactly what a
    backing is not allowed to be able to say.
    """
    assert_fresh_binary()
    _drawn, call = _bind(spec)
    bitmap = planes.written_atoms(call["activity"])

    num_dense, _, dense_plane = _dense_run(spec, call)
    num_control, _, _ = _dense_run(spec, call)
    dense = _logical(dense_plane, spec)

    plain = _arena(spec, 1)
    plain.plan_exact(bitmap)
    pooled = _arena(spec, 1, pool_slack=0.25)
    pooled.plan_exact(bitmap)
    num_pooled, _, state_pooled = _paged_run(spec, call, pooled)

    assert pooled.pool_capacity > 0, "this cell needs a pool"
    assert not torch.equal(plain.page_table, pooled.page_table), (
        "the pool did not move a single slot id, so this cell exercised no translation")
    assert bytes_equal(dense, state_pooled), (
        "the reserved pool moved the state: some address is keyed on atom RANK rather "
        "than on the page table")
    _assert_num(num_dense, num_pooled, num_control, spec,
                "the pooled arena's num differs from the dense num")


# ---------------------------------------------------------------------------
# 5. a continuation is bit-identical across the two backings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SHAPES, ids=SHAPE_IDS)
def test_a_continuation_is_bit_identical_across_the_two_backings(spec):
    """The cell run twice through one state, once dense and once in the arena.

    BOTH backings continue IN PLACE: the entry state and the final state are the SAME
    plane, resolved through the same table, which is the aliasing the design permits and
    the case a growing sequence actually runs. The second call's plan is ``fresh=False``
    -- the atoms the first committed stay committed, and only newly written ones are
    admitted and zeroed.
    """
    assert_fresh_binary()
    _drawn, call = _bind(spec)
    written = planes.written_atoms(call["activity"])

    plane = carry_ops.state_plane(call["descriptor"], 1)
    _dense_run(spec, call, plane=plane)
    s1 = _logical(plane, spec).clone()
    #: THE SAME PLANE IN AND OUT: a stateful chain allocates ONE plane for the SEQUENCE.
    num2_dense, _, _ = _dense_run(spec, call, state_in=plane, plane=plane)
    s2 = _logical(plane, spec).clone()

    arena = _arena(spec, 1)
    arena.plan_exact(written)
    _paged_run(spec, call, arena)
    p1 = arena.materialize().reshape(1, spec.N, _cols(spec)).clone()
    assert bytes_equal(s1, p1), "the first call's state differs across the backings"

    arena.plan_exact(written, fresh=False)
    num2_paged, _, p2 = _paged_run(spec, call, arena, state_in=arena.state)
    assert bytes_equal(s2, p2), "the continued state differs across the backings"
    _assert_num(num2_dense, num2_paged, num2_dense, spec, "the second call's num differs")


# ---------------------------------------------------------------------------
# 6. a stateless call carries no state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SHAPES, ids=SHAPE_IDS)
def test_a_stateless_call_carries_no_state(spec):
    """The predicate is "a plane was passed", at this seam as at the facade's.

    A call handing NEITHER plane takes the kernel's null-state path, allocates nothing,
    and computes the same ``num`` a fresh stateful call does -- a fresh entry state is
    zero, and the store is the only other thing that path drops.
    """
    assert_fresh_binary()
    _drawn, call = _bind(spec)
    num_state, _, _ = _dense_run(spec, call)
    num_none, _ = _run(spec, call)
    _assert_num(num_state, num_none, num_state, spec,
                "the null-state path's num differs from the stateful one's over a zero "
                "entry state")


# ---------------------------------------------------------------------------
# 7. a RESIDENT BUT IDLE atom costs a continuation nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("wider", "narrower"), PAIRS, ids=PAIR_IDS)
def test_a_resident_but_idle_atom_is_neither_loaded_nor_stored(wider, narrower):
    """The second call's routing touches a STRICT SUBSET of the first call's atoms.

    Residency is the arena's fact and activity is the liveness pass's: the union
    residency a continuation plans keeps the first call's atoms mapped, and the second
    call must touch none of the ones it neither reads nor writes. The claim is made
    twice, because the two halves of the predicate fail differently:

    * THE STORE SKIP is observable, and is checked against a poison fill of exactly the
      idle-resident slots -- they must come back holding it.
    * THE LOAD SKIP is not observable by construction: an idle atom is not read, so its
      contents cannot reach ``num`` whether the sweep loads them or not. That is the
      whole point (skipping is exact, not an approximation), and what it saves is DRAM
      traffic -- measured by the probe's ncu pass, never by an assertion here.
    """
    assert_fresh_binary()
    _d1, first = _bind(wider)
    _d2, second = _bind(narrower)

    resident = planes.written_atoms(first["activity"])
    active = second["activity"] != 0
    idle = resident & ~active
    assert bool(idle.any()), (
        "the narrower call reaches every atom the wider one committed; this pair is "
        "not a continuation over idle residency")
    assert bool((planes.written_atoms(second["activity"]) & ~resident).sum() == 0), (
        "the narrower call writes an atom the wider one did not: the truncation's subset "
        "property does not hold for this pair")

    plane = carry_ops.state_plane(first["descriptor"], 1)
    _dense_run(wider, first, plane=plane)
    num2_dense, _, _ = _dense_run(narrower, second, state_in=plane, plane=plane)
    s2 = _logical(plane, narrower).clone()

    arena = _arena(wider, 1)
    arena.plan_exact(resident)
    _paged_run(wider, first, arena)
    arena.plan_exact(planes.written_atoms(second["activity"]), fresh=False)
    num2_paged, _, p2 = _paged_run(narrower, second, arena, state_in=arena.state)
    assert bytes_equal(s2, p2), (
        "the continued state differs across the backings; the two backings ran different "
        "predicates")

    #: THE STORE SKIP, on a second arena carrying the same first call: poison exactly the
    #: idle-resident slots and they must survive the second call untouched.
    poisoned = _arena(wider, 1)
    poisoned.plan_exact(resident)
    _paged_run(wider, first, poisoned)
    poisoned.plan_exact(planes.written_atoms(second["activity"]), fresh=False)
    poisoned.wait()
    slots = poisoned.page_table[idle].long()
    assert int(slots.numel()) > 0, "no idle-resident slot to poison"
    poisoned.state[slots] = _POISON
    num2_poison, _, _ = _paged_run(narrower, second, poisoned, state_in=poisoned.state)
    assert bool((poisoned.state[slots] == _POISON).all()), (
        "an idle-resident atom was written: the exit sweep is storing atoms this call "
        "does not write")
    _assert_num(num2_paged, num2_poison, num2_dense, narrower,
                "poisoning an idle-resident atom moved num: the entry sweep is reading "
                "an atom this call does not read")
