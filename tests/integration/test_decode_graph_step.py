# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE GATED DECODE STEP: the deposit fact that forced it, and CUDA-graph capture.

S4b left the steady paged step host-wait-bound -- 0.093 ms of a 0.177 ms step was the
host waiting for a 4-byte growth flag to become visible, in front of a 0.007 ms kernel.
P67-a removed the host from the steady step, which is only sound because a step can be
made a device-side NO-OP for the batch-heads it cannot serve. This file gates the whole
of that.

Vocabulary, defined here and used unqualified below: **atom** = 16 consecutive leaves in
LATTICE order = the page granule; **slot** = the physical page an atom maps to (`-1` =
ABSENT); **needy** = a batch-head whose write atoms it can neither address nor admit;
**steady step** = one whose write atoms are all resident; **growth step** = one whose are
not.

**THE FACT THAT DECIDED THE DESIGN (cell 1).** A deposit into an absent atom is DROPPED
by the kernel -- `decode.cu`'s walk takes `if (in_w && atom_present)`. So a host that
checked the growth flag LAZILY (every K steps, a pinned poll) would run steps whose
writes are silently lost: laziness is not an available design at any K > 1, and the only
sound shapes are per-step synchronization or a device-side conditional. The step's own
per-batch-head verdict IS that conditional, and it is reached in the same launch, off the
write map the walk was going to build anyway.

**WHAT THE VERDICT BUYS (cells 2-3).** A needy batch-head touches nothing -- no `y`, no
state, no counter -- so the step can be ENQUEUED before its verdict is host-visible and,
if the verdict says grow, replayed after the admission. The result is bit-identical to
the admit-then-step order, which cell 3 states as `torch.equal` against the enumerated
reference's own admission.

THE VERDICT IS PER BATCH-HEAD, so "touches nothing" is a claim about the NEEDY
batch-heads and the exactly-once contract is what covers the others: a batch-head that
advanced in the first walk carries a done flag into the replay and re-reads for its `y`
without depositing again. Cell 2 makes every batch-head needy so that the claim is the
whole plane; the mixed batch is `test_decode_growth_trigger.py`'s.

**WHAT IT UNLOCKS (cells 4-5).** With no host read inside it, the step is CUDA-graph
capturable. A replay is SELF-CORRECTING: the verdict is re-derived inside the graph, so
after an admission the same captured graph performs the step it had skipped.

Run: ``pytest tests/integration/test_decode_graph_step.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.dags.decode_dag import admit_growth, decode_step_gated, growth_pending
from rola.ops.decode import _decode_step, _write_atom_bitmap
from rola.ops.lattice import to_lattice
from rola.ops.paging import MMA_K_QUANTUM, bytes_equal, from_split_planes, to_split_planes
from rola.routing.factors import RouteFactors
from tests.integration.test_decode_paging import (
    _COLS,
    _DV,
    _arena,
    _config,
    _f32,
    _fixture,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

_WIDTHS, _BC = (64, 64), 128


def _absent_one_written_atom(fixture, config, arena):
    """Admit the step's write set except ONE atom, and say which one. `(bitmap, atom)`."""
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    row, column = (bitmap.nonzero())[0].tolist()
    held_out = bitmap.clone()
    held_out[row, column] = False
    arena.plan_exact(held_out, fresh=True)
    arena.wait()
    return bitmap, (row, column)


def _step(fixture, config, arena, *, workspace=None):
    v, read, write, g_write = _f32(fixture)
    y, _state, workspace = _decode_step(v, read, write, g_write, config, arena.state,
                                        workspace=workspace, page_table=arena.page_table)
    return y, workspace


def _absent_one_atom_per_bh(fixture, config, arena):
    """Admit the step's write set except ONE atom of EVERY batch-head. Returns the set.

    Every batch-head needy is what makes cell 2's claim the WHOLE plane; with the verdict
    per batch-head, holding out one atom of one batch-head would leave the others running
    and the cell would be asserting the mechanism the design replaced.
    """
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    held_out = bitmap.clone()
    for row in range(bitmap.shape[0]):
        held_out[row, int(bitmap[row].nonzero()[0])] = False
    arena.plan_exact(held_out, fresh=True)
    arena.wait()
    return bitmap


def _atom_rows(arena, row, column):
    """The `[16, cols]` page an atom maps to, LOGICAL. The caller must know it is resident.

    The stored page is split bf16 planes, so "this atom is zero" is a claim about
    the values it rejoins to and is read through the gather, not off the raw bytes.
    """
    slot = int(arena.page_table[row, column])
    assert slot >= 0, "the atom is absent; there are no rows to read"
    return from_split_planes(arena.state[slot])


# ---------------------------------------------------------------------------
# 1. the deposit fact
# ---------------------------------------------------------------------------

def test_a_batch_head_that_cannot_address_its_writes_declines_which_is_why_laziness_is_refused():
    """The walk DROPS what it cannot address, so the step refuses to walk at all.

    `if (in_w && atom_present)` means a deposit into an absent atom is lost, not queued.
    So a host that read the growth verdict LAZILY (every `K` steps, a pinned poll) would
    run steps whose writes vanish. The kernel closes that off by ASKING FIRST, in the same
    launch: a batch-head with an unaddressable write atom walks nothing, and the atom that
    would have eaten the deposit is still zero after the admission that follows.

    Both halves are asserted, because "nothing was written" is only interesting beside
    "this step does write it": the same step against a resident atom fills those rows.
    """
    fixture = _fixture(_WIDTHS, seed=11, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]

    declined = _arena(_WIDTHS, _BC, BH)
    bitmap, (row, column) = _absent_one_written_atom(fixture, config, declined)
    _step(fixture, config, declined)
    #: Admitting AFTER the step is exactly what a lazy host would do one token late.
    declined.plan_exact(bitmap, fresh=False)
    declined.wait()
    lost = _atom_rows(declined, row, column)

    kept = _arena(_WIDTHS, _BC, BH)
    kept.plan_exact(bitmap, fresh=True)
    kept.wait()
    _step(fixture, config, kept)
    deposited = _atom_rows(kept, row, column)

    assert float(deposited.abs().max()) > 0.0, "the fixture's held-out atom is never written"
    assert float(lost.abs().max()) == 0.0, (
        "a deposit reached an atom the step could not address; if the kernel ever starts "
        "walking a batch-head it cannot admit, this file's whole design argument changes "
        "and the lazy-poll design reopens")


# ---------------------------------------------------------------------------
# 2. a needy batch-head is a no-op, exactly
# ---------------------------------------------------------------------------

def test_a_step_no_batch_head_can_address_touches_nothing():
    """Every batch-head needy: no state, no `y` semantics, no counter -- it never happened.

    Every batch-head, because the verdict is PER batch-head: holding one atom out of one
    batch-head would leave the others running, and the plane-wide `torch.equal` below
    would then be asserting the global-no-op mechanism the design replaced rather than
    the contract. The mixed batch is `test_decode_growth_trigger.py`'s cell.
    """
    fixture = _fixture(_WIDTHS, seed=12, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]
    arena = _arena(_WIDTHS, _BC, BH)
    bitmap = _absent_one_atom_per_bh(fixture, config, arena)

    before = arena.state.clone()
    _step(fixture, config, arena)
    assert bytes_equal(arena.state, before), "a step no batch-head could address wrote state"

    #: And the same step against the admitted set is the ordinary step, on the same
    #: buffers -- residency is an input, never a mode.
    arena.plan_exact(bitmap, fresh=False)
    arena.wait()
    _step(fixture, config, arena)
    assert not bytes_equal(arena.state, before), "the admitted step did not run"


# ---------------------------------------------------------------------------
# 3. the gated growth order is the admit-first order
# ---------------------------------------------------------------------------

def test_gated_step_then_admit_then_replay_is_bit_identical_to_admitting_first():
    """the host order against S4b's, `torch.equal` on `y` and on the plane.

    The reference admits from `_write_atom_bitmap` -- the enumerated reference, whose
    only callers are gates -- so the two orders are compared through two independent
    derivations of the same write set. One batch-head is needy and the others are not,
    so the equality is also the exactly-once claim: the batch-heads that ran in the first
    walk must re-read in the replay without depositing again.
    """
    fixture = _fixture(_WIDTHS, seed=13, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    reference = _arena(_WIDTHS, _BC, BH)
    reference.plan_exact(bitmap, fresh=True)
    reference.wait()
    y_reference, _ = _step(fixture, config, reference)

    #: The gated order: a batch-head that cannot address its writes runs as a no-op, the
    #: admission follows, and the replay is the step.
    gated = _arena(_WIDTHS, _BC, BH)
    _absent_one_written_atom(fixture, config, gated)
    _y, workspace = _step(fixture, config, gated)
    gated.plan_exact(bitmap, fresh=False)
    gated.wait()
    y_gated, _ = _step(fixture, config, gated, workspace=workspace)

    assert torch.equal(y_reference, y_gated)
    assert bytes_equal(_gathered(reference, BH), _gathered(gated, BH))


def _gathered(arena, BH):
    """`[BH, N, cols]` in LATTICE order from the arena's slots -- what the two backings
    can both hold."""
    table = arena.page_table
    N = table.shape[1] * MMA_K_QUANTUM
    out = torch.zeros(BH, table.shape[1], MMA_K_QUANTUM, _COLS, device="cuda")
    present = table >= 0
    out[present] = arena.state[table[present].long()]
    return out.reshape(BH, N, _COLS)


# ---------------------------------------------------------------------------
# 4-5. capture
# ---------------------------------------------------------------------------

_HIDDEN, _H, _T, _B = 64, 2, 128, 2


def _sequence():
    """A prefilled paged state, its producer and a stream of decode tokens.

    The prefill leg ran on `rola.rola_op` until the K31 deletion batch removed
    the arm (baseline = tag `baseline/pre-k31`). The state is now SEEDED the
    facade's own way minus the launch: `_kernel_entry` binds the arena and the
    fp64 oracle's `output_final_state` fills the committed pages, narrowed to
    the kernel's fp32 and crossed into the lattice leaf order the arena's atoms
    are cut from. What these cells gate (capture, replay, growth) reads the
    arena's base, table and plane, and none of the three cares which arm
    deposited the numbers.

    RESIDENCY IS TOTAL, stated rather than derived: over 16 atoms a 128-token
    prefill leaves nothing absent, and cell 5 MAKES its growth by dropping an
    atom from a full plan. A partial seeding bitmap would be a second leaf-order
    authority for these cells to disagree with.
    """
    import rola
    from rola.ops.naive import naive_rola
    from rola.routing.producer import union_routing
    from tests.conftest import build_routes

    torch.manual_seed(7)
    producer = build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=_WIDTHS,
                           routing=union_routing(1)).cuda()
    torch.manual_seed(507)
    x = torch.randn(_B, _T + 8, _HIDDEN, device="cuda")
    v = torch.randn(_B, _T + 8, _H, _DV, device="cuda")
    with torch.no_grad():
        routes = producer(x[:, :_T])
        config = _config(_WIDTHS, dict(B=_B, H=_H))
        bits = torch.ones(_B * _H, config.N // MMA_K_QUANTUM, dtype=torch.bool,
                          device="cuda")
        state = rola.state()
        _s_in, arena = state._kernel_entry(routes, bits, d_v=_DV, paging=True,
                                           BC=64)
        arena.wait()
        _y_ref, oracle_state = naive_rola(
            v[:, :_T], routes.read_simplex(), routes.write,
            routes.g_write, routes.topology, None,
            output_final_state=True)
        lattice = to_lattice(oracle_state.float(), _WIDTHS, config.lattice_k,
                             config.lattice_m)
        #: THE ARENA HOLDS THE STORED FORM (the split planes).
        atoms = to_split_planes(lattice.reshape(_B * _H, -1, MMA_K_QUANTUM, _COLS))
        table = arena.page_table
        present = table >= 0
        arena.state[table[present].long()] = atoms[present]
    return producer, x, v, state


def _static(routes, v):
    """A bundle over buffers that never move -- what capture requires of its inputs."""
    static = RouteFactors(
        topology=routes.topology,
        read=tuple(t.clone() for t in routes.read), write=tuple(t.clone() for t in routes.write),
        g_write=routes.g_write.clone(),
        read_mass=None if routes.read_mass is None else routes.read_mass.clone())
    return static, v.clone()


def _fill(static, static_v, routes, v):
    for buffer, level in zip(static.read, routes.read):
        buffer.copy_(level)
    for buffer, level in zip(static.write, routes.write):
        buffer.copy_(level)
    static.g_write.copy_(routes.g_write)
    if static.read_mass is not None:
        static.read_mass.copy_(routes.read_mass)
    static_v.copy_(v)


def _capture(static, static_v, state):
    """Warm up on a side stream, then capture ONE gated step. Returns `(graph, y)`."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            decode_step_gated(static, static_v, state)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, _ = decode_step_gated(static, static_v, state)
    return graph, y


@torch.no_grad()
def test_the_captured_step_is_the_eager_step_bit_for_bit():
    """Same kernels, same order, same bytes -- the capture owes exactly this."""
    producer, x, v, state = _sequence()
    token = slice(_T + 4, _T + 5)
    routes = producer(x[:, token])
    static, static_v = _static(routes, v[:, token])
    _fill(static, static_v, routes, v[:, token])
    graph, y_graph = _capture(static, static_v, state)

    #: Make the token's write set resident FIRST, so neither run is the vacuous no-op
    #: cell 5 is about.
    decode_step_gated(static, static_v, state)
    if growth_pending(state):
        admit_growth(state)
    plane = state._arena.state
    snapshot = plane.clone()

    graph.replay()
    torch.cuda.synchronize()
    captured_y, captured_plane = y_graph.clone(), plane.clone()
    assert not bytes_equal(captured_plane, snapshot), "the compared step did nothing"

    plane.copy_(snapshot)
    eager_y, _ = decode_step_gated(static, static_v, state)
    torch.cuda.synchronize()
    assert torch.equal(captured_y, eager_y)
    assert bytes_equal(captured_plane, plane)


def _make_growth(state, static):
    """Drop one atom this token writes from the plan, and say which. `(row, column)`.

    A 128-token prefill over 16 atoms leaves nothing absent, so the growth is MADE.
    Re-planning also reassigns slots, which is the stronger cell -- the captured graph
    holds the arena's BASE and its table, and neither moves.
    """
    written = _write_atom_bitmap(static.write, _config(_WIDTHS, dict(B=_B, H=_H)))
    keep = (state._arena.page_table >= 0) & written
    row, column = (keep.nonzero())[0].tolist()
    keep[row, column] = False
    state._arena.plan_exact(keep, fresh=True)
    state._arena.wait()
    return row, column


@torch.no_grad()
def test_a_growth_step_lands_under_the_graph_by_replay():
    """The replay is SELF-CORRECTING: no-op, admit, replay, and it is the eager answer.

    THE COMPARISON IS OF THE WHOLE PAIR, on two independently prefilled sequences, and
    that is a restatement forced by the verdict being PER BATCH-HEAD. The cell used to
    replay, admit, replay, then restore the plane and run ONE eager step against the
    result -- which was sound only while the first walk deposited nothing at all. It now
    deposits for every batch-head it can serve, so a restored plane is not a common
    starting point for those batch-heads and a single eager step would deposit into them
    twice. What the design actually claims is that the PAIR advances every sequence
    exactly once, and two identical sequences walking the pair -- one captured, one eager
    -- is that claim stated without a restore.
    """
    producer, x, v, state = _sequence()
    token = slice(_T + 5, _T + 6)
    routes = producer(x[:, token])
    static, static_v = _static(routes, v[:, token])
    _fill(static, static_v, routes, v[:, token])
    graph, y_graph = _capture(static, static_v, state)

    row, column = _make_growth(state, static)
    plane = state._arena.state
    snapshot = plane.clone()

    graph.replay()
    torch.cuda.synchronize()
    assert growth_pending(state), "the fixture's growth token is already resident"
    #: THE FIRST WALK, per batch-head: the one that cannot address its writes touches
    #: nothing. (The plane-wide equality this cell used to assert is the global-no-op
    #: mechanism, and the design replaced it.)
    assert int(state._arena.page_table[row, column]) < 0, (
        "the held-out atom is resident; the fixture made no growth")
    resident = state._arena.page_table[row][state._arena.page_table[row] >= 0].long()
    assert bytes_equal(plane[resident], snapshot[resident]), (
        "the needy batch-head deposited under the graph")

    admit_growth(state)
    graph.replay()
    torch.cuda.synchronize()
    assert not growth_pending(state), "the replay did not clear the verdict"
    landed_y, landed = y_graph.clone(), state.materialize().clone()

    #: The same pair, eagerly, on its own sequence. `_sequence` is seeded, so the two
    #: states are the same state and the two tokens are the same token.
    producer2, x2, v2, eager_state = _sequence()
    eager_routes = producer2(x2[:, token])
    eager_static, eager_static_v = _static(eager_routes, v2[:, token])
    _fill(eager_static, eager_static_v, eager_routes, v2[:, token])
    assert _make_growth(eager_state, eager_static) == (row, column), (
        "the two sequences are not the same sequence")
    decode_step_gated(eager_static, eager_static_v, eager_state)
    assert growth_pending(eager_state)
    admit_growth(eager_state)
    eager_y, _ = decode_step_gated(eager_static, eager_static_v, eager_state)
    torch.cuda.synchronize()

    assert torch.equal(landed_y, eager_y)
    assert torch.equal(landed, eager_state.materialize())
