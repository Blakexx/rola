# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""DECODE's paged backing: one walk, one base translation, and growth per step.

The `T = 1` sibling of `test_chunk_paging_equivalence.py`, and it claims the same
three things for the same reason. Which page an atom lives on is the ONLY thing a
backing changes: without a table the slot IS the atom, with one it is what the plan
committed, and the instruction stream, the accumulation order and the ABI are the
same object. So the claim is `torch.equal` and not a tolerance -- any difference at
all would be an ADDRESSING difference, and addressing is exact or wrong.

Vocabulary, defined here and used unqualified below: `N` = leaf count; `d_v = 64`;
`cols = d_v + 1` (the mass column is unconditional); `BH = batch * heads`;
**atom** = 16 consecutive leaves in LATTICE leaf order = the page granule
(`rola.ops.paging.MMA_K_QUANTUM`); **slot** = the physical page an atom is mapped to,
`-1` for an atom no plan committed; `M` = the step's touched-row count.

**FIVE CLAIMS.**

1. *The write-atom derivation is EXACT.* `rola.ops.decode._write_atom_bitmap` is the
   ENUMERATED REFERENCE for the set a step admits -- the shipped path takes that set from
   the step's own device-side condensation, and this is the independent derivation it
   is gated against (`test_decode_growth_trigger.py`). It is checked here against the leaf
   support's own definition
   (`schedule.py`'s `write_mask = tuple(level != 0 ...)`, taken to leaf granularity and
   ORed 16-deep) rather than against a second copy of itself. A bitmap that were merely
   a superset would still be SAFE -- claim 3 is what licenses that -- but it would stop
   residency being the routing's own property, which is the whole contract.

2. *Bit identity.* One step dense and one paged over the same routing, the same `v` and
   the same contents. `y` is equal and the states are equal once the paged pages are
   gathered back through the table. Decode's `y` is deterministic by construction (the
   fixed-order slot combine), so unlike the tiled arm there is no envelope here at all.

3. *An absent atom is inert, in both directions.* It is not written -- every unadmitted
   page of the arena's backing survives a poison fill untouched -- and a READ through it
   contributes exactly zero, which is the same thing the dense backing's never-deposited
   zeros contribute. This is the mechanical form of the self-normalizing readout's
   dead-leaf argument at `T = 1`.

4. *Growth is planned-exact, per step.* A step that deposits into an atom no previous
   step reached admits exactly those atoms and no others, in place, without moving a
   base pointer -- and the answer is the oracle's.

5. *A slack pool changes WHO admits and nothing else.* The same chain with the host
   admitting nothing -- every atom taken by the step itself, from the pool -- reaches the
   oracle's answer and is `torch.equal` to the host-admitted chain. A pool slot is an
   ordinary zeroed arena slot, so there is nothing for an atom to be admitted "into a
   pool" as (`docs/internals/paging/paging.md#the-slack-pool`).

The FACADE-level continuation (prefill on the chunk arm, then decode steps on the same
state object) is `tests/integration/test_decode_continuation.py`; this file stays at
the ABI, where the addressing lives.

Run: ``pytest tests/integration/test_decode_paging.py``
"""
from __future__ import annotations

import math

import pytest
import torch

from rola.ops.decode import _decode_step, _write_atom_bitmap, derive_decode_geometry
from rola.ops.lattice import permutation, to_canonical
from rola.ops.naive import naive_rola
from rola.ops.paging import MMA_K_QUANTUM, PageArena, bytes_equal, from_split_planes, to_split_planes

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

_DV = 64
_COLS = _DV + 1
_POISON = -4321.5

#: DECODE'S TOLERANCE AND ITS TOPOLOGY BUILDER, ADOPTED RATHER THAN INVENTED -- the
#: reasoning `test_decode_vs_oracle.py` states for the first and the routing tag it
#: fixes for the second, imported so there is ONE of each.
from tests.oracle.oracle_fixtures import (  # noqa: E402
    _topology as _build_topology,
)
from tests.oracle.test_decode_vs_oracle import DECODE_RTOL  # noqa: E402

#: `(widths, BC)`. Every leaf count is a whole number of atoms, which is what a paged
#: keying REQUIRES (`AtomKeying.__post_init__`); the depth axis is what varies, because
#: the walk's per-word digit decomposition is the part depth reaches. K53: every row is
#: admission-lawful under the width-16 floor, so a walk unit is exactly one atom in each
#: (`box_shape`'s innermost span is 16 at every depth here, not narrower).
_ARMS = (
    ((64, 64), 128),
    ((16, 16, 16, 16), 128),
    ((16, 16, 16), 128),
    ((16, 16, 16, 16), 128),
)

#: The fraction of each level's digits the WRITE side is allowed to reach. Residency is a
#: property of the write support's STRUCTURE: a thin but unstructured write side still
#: touches every atom, so a cell that wants ABSENT atoms must restrict digits.
_SUPPORTS = {"sparse": 0.25, "half": 0.5, "all": 1.0}


def _topology(widths):
    return _build_topology(widths)


def _simplex(shape, generator, live=None, start=0):
    """Rows on the simplex, optionally supported on the `live` digits from `start`."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if live is not None:
        x[..., :start] = 0.0
        x[..., start + live:] = 0.0
    return x / x.sum(-1, keepdim=True)


def _fixture(widths, *, seed, support=1.0, offset=0.0, B=2, H=3):
    """One decode step's inputs: dense read support, `support`-structured write support.

    `offset` SLIDES LEVEL 0's live digit window instead of widening every level's, which
    is what makes two steps reach different atoms: an atom is 16 consecutive LATTICE
    leaves, so a digit window that stays inside one owner span lands in one sub-box no
    matter how wide it is, and sliding ONE level moves the sub-box while keeping the
    touched set small.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((B, 1, H, w), g) for w in widths)
    write = tuple(_simplex((B, 1, H, w), g, live=max(1, int(w * support)),
                           start=int(w * offset) if level == 0 else 0)
                  for level, w in enumerate(widths))
    g_write = torch.rand((B, 1, H), device="cuda", dtype=torch.float64, generator=g) + 0.5
    v = torch.randn((B, 1, H, _DV), device="cuda", dtype=torch.float64, generator=g)
    return dict(read=read, write=write, g_write=g_write, v=v, B=B, H=H)


def _f32(fixture):
    return (fixture["v"].float(),
            tuple(t.float() for t in fixture["read"]),
            tuple(t.float() for t in fixture["write"]),
            fixture["g_write"].float())


def _config(widths, fixture):
    return derive_decode_geometry(
        _topology(widths), d_v=_DV, decay=False,
        BH=fixture["B"] * fixture["H"], device="cuda")


def _leaf_support(levels, widths, config):
    """`[BH, N]` bool from the DEFINITION, in LATTICE order -- the plane's own order.

    `schedule.py`'s `write_mask` at leaf granularity, in torch, so the derivation under
    test is checked against the definition and not against a second copy of itself. The
    definition is CANONICAL; `pi` is applied here because an atom is 16 consecutive
    LATTICE leaves.
    """
    B, _T, H, _W = levels[0].shape
    N = math.prod(widths)
    strides = [math.prod(widths[l + 1:]) for l in range(len(widths))]
    index = torch.arange(N, device=levels[0].device)
    mask = torch.ones(B * H, N, dtype=torch.bool, device=levels[0].device)
    for level, width in enumerate(widths):
        digit = (index // strides[level]) % width
        folded = levels[level].permute(0, 2, 1, 3).reshape(B * H, width)
        mask &= folded[:, digit] != 0
    pi = permutation(widths, config.lattice_k, config.lattice_m, device=mask.device)
    lattice = torch.empty_like(mask)
    lattice.index_copy_(1, pi, mask)
    return lattice


def _arena(widths, BC, BH, pool_slack=0.0):
    return PageArena(widths=widths, BC=BC, cols=_COLS, BH=BH,
                     device=torch.device("cuda"), pool_slack=pool_slack)


def _seed_from_dense(arena, dense):
    """Copy a dense `[BH, N, cols]` LATTICE-order state into the arena's committed slots."""
    arena.wait()
    BH, N, cols = dense.shape
    #: THE ARENA HOLDS THE STORED FORM (the split planes); `dense` is the logical one.
    atoms = to_split_planes(dense.reshape(BH * (N // MMA_K_QUANTUM), MMA_K_QUANTUM, cols))
    table = arena.page_table.reshape(-1)
    present = table >= 0
    arena.state.index_copy_(0, table[present].long(), atoms[present])


def _dense_masked_to(arena, dense):
    """A LATTICE-order `dense` with every NON-resident atom zeroed -- what the arena
    can represent.

    The two backings can only agree where both hold something, and an absent atom holds
    nothing. Zeroing rather than skipping the comparison is deliberate: it makes the
    dense run compute the SAME function, so claim 2's `torch.equal` is a statement about
    addressing and not about which leaves were compared.
    """
    BH, N, cols = dense.shape
    atoms = dense.reshape(BH, N // MMA_K_QUANTUM, MMA_K_QUANTUM, cols).clone()
    atoms[arena.page_table < 0] = 0.0
    return atoms.reshape(BH, N, cols)


# ---------------------------------------------------------------------------
# 1. the write-atom derivation is the definition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_the_step_write_atom_set_is_exact(arm, support):
    """`_write_atom_bitmap` IS the write support, condensed 16-deep. Exactly.

    This is the gate on the growth derivation's *choice*: a step plans from this object,
    so if it were a superset the pool would commit atoms the routing never reaches, and
    if it were a subset a deposit would land on an absent atom and be dropped.
    """
    widths, _BC = arm
    fixture = _fixture(widths, seed=len(widths) * 31 + 7, support=_SUPPORTS[support])
    config = _config(widths, fixture)
    _v, _read, write, _gw = _f32(fixture)

    got = _write_atom_bitmap(write, config)
    want = _leaf_support(write, widths, config).view(
        -1, config.N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)
    assert torch.equal(got, want), "the step's write-atom set is not the write support"
    if _SUPPORTS[support] < 1.0:
        assert not bool(got.all()), (
            "this fixture admits every atom, so the cell cannot distinguish the exact "
            "derivation from a blanket one")


# ---------------------------------------------------------------------------
# 2. dense and paged are one walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_the_paged_backing_is_bit_identical_to_the_dense_one(arm, support):
    widths, BC = arm
    fixture = _fixture(widths, seed=len(widths) * 97 + BC, support=_SUPPORTS[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    v, read, write, g_write = _f32(fixture)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(_write_atom_bitmap(write, config), fresh=True)
    arena.wait()

    entry = torch.randn(BH, config.N, _COLS, device="cuda", dtype=torch.float32)
    entry[..., _DV] = entry[..., _DV].abs()
    dense = _dense_masked_to(arena, entry)
    _seed_from_dense(arena, dense)

    y_dense, dense_out, _ = _decode_step(
        v, read, write, g_write, config,
        to_split_planes(dense.clone().view(fixture["B"], fixture["H"], config.N, _COLS)))
    y_paged, _, _ = _decode_step(
        v, read, write, g_write, config, arena.state,
        page_table=arena.page_table)

    assert torch.equal(y_dense, y_paged), "the two backings disagree on y"
    assert torch.equal(from_split_planes(dense_out).view(BH, config.N, _COLS),
                       arena.materialize()), "the two backings disagree on the state"
    # NON-VACUITY: pages were committed, they came back nonzero, and (unless the write
    # support is total) the table is NOT the identity map, so a translation ran.
    assert arena.allocated_pages > 0
    assert float(arena.materialize().abs().max()) > 0.0
    if _SUPPORTS[support] < 1.0:
        identity = torch.arange(arena.page_table.numel(), device="cuda",
                                dtype=torch.int32).view_as(arena.page_table)
        assert not torch.equal(arena.page_table, identity), (
            "the page table is the identity map; no base translation was exercised")


# ---------------------------------------------------------------------------
# 3. an absent atom is inert, in both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_an_absent_atom_is_never_written_and_reads_as_zero(arm):
    """The whole backing is poisoned, the step runs, and every unadmitted page survives.

    A `slot < 0` that reached the address expression would land `4160` bytes off the
    FRONT of the backing, which this poison fill is what detects. The READ half is the
    same cell's other assertion: the read support is DENSE here, so the walk reaches
    absent atoms on every step, and `y` still equals the dense run's -- the dense run
    holding zeros exactly where the arena holds nothing.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=len(widths) * 13 + 5, support=0.25)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    v, read, write, g_write = _f32(fixture)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(_write_atom_bitmap(write, config), fresh=True)
    arena.wait()
    resident = arena.allocated_pages
    assert resident < arena.logical_pages, "every atom is resident; nothing is absent"

    #: admission zeroes only what it COMMITS, so the pages beyond the committed run are
    #: free to carry a poison the step must not touch.
    arena.state[resident:] = _POISON
    entry = torch.randn(BH, config.N, _COLS, device="cuda", dtype=torch.float32)
    entry[..., _DV] = entry[..., _DV].abs()
    dense = _dense_masked_to(arena, entry)
    _seed_from_dense(arena, dense)
    table_before = arena.page_table.clone()

    y_dense, _, _ = _decode_step(
        v, read, write, g_write, config,
        to_split_planes(dense.clone().view(fixture["B"], fixture["H"], config.N, _COLS)))
    y_paged, _, _ = _decode_step(
        v, read, write, g_write, config, arena.state,
        page_table=arena.page_table)

    assert bytes_equal(arena.state[resident:],
                       torch.full_like(arena.state[resident:], _POISON)), (
        "the step wrote a page no plan committed")
    assert torch.equal(arena.page_table, table_before), "the step moved the page table"
    assert torch.equal(y_dense, y_paged), (
        "a read through an absent atom was not inert")


# ---------------------------------------------------------------------------
# 4. growth: a step admits its own new atoms, and only those
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_a_step_grows_residency_by_exactly_its_new_atoms(arm):
    """Two steps whose write supports differ: the second admits its NEW atoms and no more.

    The bases do not move (the pool is reserved once), the previously committed atoms
    keep their contents, and the answer is the chained fp64 oracle's. Residency is
    asserted as an EXACT count against the two bitmaps' own set difference, which is
    what makes this a planned-exact claim rather than a "grew somehow" one.
    """
    widths, BC = arm
    config = _config(widths, _fixture(widths, seed=0))
    BH = 2 * 3
    first = _fixture(widths, seed=len(widths) * 41 + 1, support=0.5)
    second = _fixture(widths, seed=len(widths) * 41 + 2, support=0.5, offset=0.25)

    bits = [_write_atom_bitmap(_f32(step)[2], config) for step in (first, second)]
    new_atoms = int((bits[1] & ~bits[0]).sum())
    assert new_atoms > 0, "the second step reaches no new atom; the cell is vacuous"

    arena = _arena(widths, BC, BH)
    arena.plan_exact(bits[0], fresh=True)
    arena.wait()
    base_ptr = arena.state.data_ptr()
    before = arena.page_table >= 0
    table_before = arena.page_table.clone()
    assert torch.equal(before, bits[0]), "the first plan is not the first step's write set"

    ys = []
    for step, bitmap in zip((first, second), bits):
        #: THE GROWTH ORDER, exactly `decode_forward`'s: plan the union on the side
        #: stream, wait once, launch.
        arena.plan_exact(bitmap, fresh=False)
        arena.wait()
        v, read, write, g_write = _f32(step)
        y, _, _ = _decode_step(v, read, write, g_write, config, arena.state,
                               page_table=arena.page_table)
        ys.append(y)

    #: RESIDENCY IS THE TABLE, not the allocator's cursor. `resident_pages` counts slots
    #: the `ExtentAllocator` has CONSUMED, which includes the alignment gap it opens when
    #: a batch does not fit the current extent's remainder -- a placement TENDENCY the
    #: class documents and refuses to promise as an invariant. What is exact, and what
    #: planned-exact admission means, is which ATOMS are mapped.
    after = arena.page_table >= 0
    assert torch.equal(after, bits[0] | bits[1]), (
        "residency after the second step is not the union of the two write sets")
    assert int((after & ~before).sum()) == new_atoms, (
        "the step admitted atoms it does not write")
    assert torch.equal(arena.page_table[before], table_before[before]), (
        "growth moved an already-committed atom's slot")
    assert arena.state.data_ptr() == base_ptr, "growth moved the plane's base pointer"

    #: the fp64 oracle, chained one token at a time -- which is what decode does.
    ref_state = None
    y_ref = None
    for step in (first, second):
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_write"],
            _topology(widths), None,
            initial_state=ref_state, output_final_state=True)
    state = to_canonical(arena.materialize(), widths, config.lattice_k,
                         config.lattice_m).view(2, 3, config.N, _COLS)
    scale_y = max(1e-30, float(y_ref.abs().max()))
    scale_s = max(1e-30, float(ref_state.abs().max()))
    err_y = float((ys[-1].double() - y_ref).abs().max()) / scale_y
    err_s = float((state.double() - ref_state).abs().max()) / scale_s
    assert err_y < DECODE_RTOL, f"the grown step's y left the oracle band: {err_y:.3e}"
    assert err_s < DECODE_RTOL, f"the grown state left the oracle band: {err_s:.3e}"


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_a_pool_admitted_sequence_is_the_oracle_s_and_the_host_admitted_one_s(arm):
    """The same two steps with the host never admitting anything: the DEVICE admits.

    A slack pool makes growth the step's own business, so this cell runs the chain with
    no `plan_exact` between the steps at all -- the arena starts EMPTY and each step takes
    what it needs. Two claims, and the second is the one that matters: the chain is the
    chained fp64 oracle's, and it is `torch.equal` to the host-admitted chain, because a
    pool slot is an ordinary zeroed arena slot and an atom admitted into one is
    indistinguishable from an atom `plan_exact` admitted.
    """
    widths, BC = arm
    config = _config(widths, _fixture(widths, seed=0))
    BH = 2 * 3
    first = _fixture(widths, seed=len(widths) * 41 + 1, support=0.5)
    second = _fixture(widths, seed=len(widths) * 41 + 2, support=0.5, offset=0.25)
    bits = [_write_atom_bitmap(_f32(step)[2], config) for step in (first, second)]
    assert int((bits[1] & ~bits[0]).sum()) > 0, "the second step grows nothing here"

    admitted = _arena(widths, BC, BH)
    pooled = _arena(widths, BC, BH, pool_slack=1.0)
    ys = {}
    for name, arena, pool in (("host", admitted, None), ("pool", pooled, pooled)):
        out = []
        for step, bitmap in zip((first, second), bits):
            if pool is None:
                arena.plan_exact(bitmap, fresh=False)
                arena.wait()
            v, read, write, g_write = _f32(step)
            y, _, _ = _decode_step(v, read, write, g_write, config, arena.state,
                                   page_table=arena.page_table, pool=pool)
            out.append(y)
        ys[name] = out

    assert torch.equal(pooled.page_table >= 0, bits[0] | bits[1]), (
        "the device admitted a different residency than the two steps' write sets")
    assert torch.equal(ys["pool"][-1], ys["host"][-1]), (
        "a pool-admitted step's y differs from a host-admitted one's")
    assert torch.equal(pooled.materialize(), admitted.materialize()), (
        "a pool-admitted state differs from a host-admitted one")

    ref_state = None
    y_ref = None
    for step in (first, second):
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_write"],
            _topology(widths), None,
            initial_state=ref_state, output_final_state=True)
    state = to_canonical(pooled.materialize(), widths, config.lattice_k,
                         config.lattice_m).view(2, 3, config.N, _COLS)
    err_y = float((ys["pool"][-1].double() - y_ref).abs().max()) / max(
        1e-30, float(y_ref.abs().max()))
    err_s = float((state.double() - ref_state).abs().max()) / max(
        1e-30, float(ref_state.abs().max()))
    assert err_y < DECODE_RTOL, f"the pool-admitted y left the oracle band: {err_y:.3e}"
    assert err_s < DECODE_RTOL, f"the pool-admitted state left the oracle band: {err_s:.3e}"
