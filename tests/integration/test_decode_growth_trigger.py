# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The DEVICE-SIDE growth verdict: does this step reach an atom it cannot address?

A paged decode step's exact admission preamble -- the write-atom derivation,
`plan_exact`, the cross-stream event -- costs roughly ten times the kernel it precedes
against the launch it precedes. It is also usually unnecessary: a sequence whose routing
has settled writes atoms that are already resident, and there is nothing to admit. The
step answers that question ITSELF, inside its own prologue, off the write map it had to
build anyway, and the whole of its correctness is one claim:

  **THE VERDICT FIRES IFF THE PREAMBLE WOULD ADMIT SOMETHING THE STEP CANNOT ADMIT.**

Both halves matter and this file sweeps both. A verdict that MISSED would let a step
deposit into an absent atom -- silently dropped, since a `slot < 0` row is never formed. A
verdict that fired SPURIOUSLY would only cost the preamble it exists to skip, but it would
also mean the two derivations disagree, and a disagreement that is harmless in one
direction is not evidence about the other.

The claim is cheap to make exactly because the two sides SHARE a definition rather than
mirror one: the residency test reads the very units the walk walks, resolved from the same
per-level factors, and `_write_atom_bitmap` is the torch derivation the exact preamble
plans from. So this file
checks an EQUALITY the call graph already forces, which is the only kind of agreement
worth gating: the failure it can catch is a condensation or a residency test that drifted,
not a second copy of the support rule.

**THE VERDICT IS PER BATCH-HEAD, and cells 6-7 are what makes that non-vacuous.** A step
over a mixed batch advances the batch-heads it can address and no-ops the others; the
caller's replay after the admission advances the rest and RE-READS the ones that already
ran, so every sequence advances EXACTLY ONCE across the pair. The contract is
exactly-once, never "the batch skipped" -- a cell that asserted the old global no-op
mechanism is restated here against that contract.

**THE SLACK POOL (cells 8-9)** is the same verdict with a capacity: where the state
reserves slack slots, a growing batch-head admits itself and the verdict does NOT fire --
there is nothing for the host to do. Exhaustion degrades to exactly the pool-free path.

Vocabulary, defined here and used unqualified below: `N` = leaf count; `d_v = 64`;
`cols = d_v + 1`; `BH = batch * heads`; **atom** = 16 consecutive leaves in LATTICE
leaf order = the page granule; **slot** = the physical page an atom is mapped to, `-1`
for an atom no plan committed; **needy** = a batch-head whose write set it can neither
address nor admit.

Run: ``pytest tests/integration/test_decode_growth_trigger.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.decode import (
    DecodeScratch,
    _decode_step,
    _write_atom_bitmap,
)
from rola.ops.paging import MMA_K_QUANTUM, PageArena, bytes_equal
from tests.integration.test_decode_paging import (
    _ARMS,
    _config,
    _f32,
    _fixture,
    _leaf_support,
)

_SUPPORT = {"sparse": 0.25, "half": 0.5, "all": 1.0}

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]


def _step(fixture, config, arena, *, workspace=None, pool=None):
    """One step at the ABI, on the residency the arena has. `(y, workspace)`."""
    v, read, write, g_write = _f32(fixture)
    BH = fixture["B"] * fixture["H"]
    if workspace is None:
        workspace = DecodeScratch.build(config, BH, torch.device("cuda"))
    y, _state, workspace = _decode_step(
        v, read, write, g_write, config, arena.state, workspace=workspace,
        page_table=arena.page_table, pool=pool)
    return y, workspace


def _verdict(fixture, config, arena, *, pool=None):
    """The step's own answer, and the workspace it ran on. A FRESH workspace each call.

    Fresh because the done flags are per-STEP state: a workspace carried across two
    independent calls would answer for the first one's batch-heads.
    """
    _y, workspace = _step(fixture, config, arena, pool=pool)
    workspace.growth_host.copy_(workspace.growth_any)
    fired = bool(workspace.growth_host[0])
    #: The reduction the host reads and the per-`bh` vector it reduces must agree, on
    #: every cell this file already sweeps.
    assert fired == bool(workspace.growth.any())
    return fired, workspace


def _would_admit(fixture, config, arena) -> int:
    """What the EXACT preamble would admit, run here and nowhere near the shipped path."""
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    return int((bitmap & (arena.page_table < 0)).sum())


def _arena(widths, BC, BH, **kwargs):
    return PageArena(widths=widths, BC=BC, cols=65, BH=BH, device=torch.device("cuda"),
                     **kwargs)


def _gathered(arena, BH):
    """`[BH, N, cols]` from the arena's slots -- what two backings can both hold."""
    table = arena.page_table
    out = torch.zeros(BH, table.shape[1], MMA_K_QUANTUM, 65, device="cuda")
    present = table >= 0
    out[present] = arena.state[table[present].long()]
    return out.reshape(BH, -1, 65)


# ---------------------------------------------------------------------------
# 1. the two directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
@pytest.mark.parametrize("support", ("sparse", "half", "all"))
def test_the_verdict_fires_exactly_when_the_preamble_would_admit(arm, support):
    """Empty table -> must fire; then the step's own atoms admitted -> must not.

    The two states are the two ends of a real sequence: the first step after a prefill
    that never reached these atoms, and every settled step after it. The assertion is
    the same in both, stated against what the preamble WOULD do rather than against a
    second expectation of what the routing looks like.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=hash((widths, support)) % 10_000, support=_SUPPORT[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)

    #: EMPTY: nothing is mapped, so every atom this step writes is new.
    arena = _arena(widths, BC, BH)
    admits_before = _would_admit(fixture, config, arena)
    assert admits_before > 0, "this cell needs a step that writes something"
    fired, _ = _verdict(fixture, config, arena)
    assert fired, (
        f"the verdict missed a step the preamble would admit {admits_before} atoms for; "
        "the step would have deposited into an absent atom and lost it")

    #: SETTLED: this exact write set is now resident, so there is nothing to admit.
    arena.plan_exact(_write_atom_bitmap(write, config), fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    assert _would_admit(fixture, config, arena) == 0, "the fixture's own precondition"
    fired, _ = _verdict(fixture, config, arena)
    assert not fired, (
        "the verdict fired on a step with nothing to admit; the preamble it exists to "
        "skip would be paid on every settled step of the sequence")


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_one_absent_atom_is_enough_to_fire(arm):
    """The verdict is an ANY, and this is its tightest cell.

    The step's whole write set is admitted and then EXACTLY ONE of its atoms is evicted
    from the table. Nothing else moves. A verdict that reduced over the wrong axis, or
    that condensed 32 leaves to an atom instead of 16, or that tested `>= 0` on the wrong
    row of the table, passes the coarse cells above and fails here.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=907 + len(widths), support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(bitmap, fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    assert not _verdict(fixture, config, arena)[0], "the fixture's own precondition"

    written = bitmap.nonzero()
    for pick in (0, written.shape[0] // 2, written.shape[0] - 1):
        bh, atom = (int(x) for x in written[pick])
        slot = int(arena.page_table[bh, atom])
        arena.page_table[bh, atom] = -1
        fired, _ = _verdict(fixture, config, arena)
        arena.page_table[bh, atom] = slot
        assert fired, (
            f"evicting the single atom ({bh}, {atom}) this step writes did not fire the "
            "verdict")
    assert not _verdict(fixture, config, arena)[0], "restoring the slot did not settle it"


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_an_absent_atom_the_step_does_not_write_is_not_growth(arm):
    """Residency is not the question; the step's OWN write set against residency is.

    A sparsely-resident arena is the normal case -- that is what paging is for -- so a
    verdict keyed on "is anything absent" rather than "is anything I WRITE absent" would
    fire on every step forever and quietly restore the preamble this stage removed.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=311 + len(widths), support=0.25)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(bitmap, fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    absent = int((arena.page_table < 0).sum())
    assert absent > 0, "this cell needs a routing that leaves atoms absent"
    assert not _verdict(fixture, config, arena)[0], (
        f"{absent} atoms are absent and the verdict fired, but the step writes none of "
        "them")


# ---------------------------------------------------------------------------
# 2. the step's own condensation is the step's own support
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
@pytest.mark.parametrize("support", ("sparse", "half", "all"))
def test_the_published_condensation_is_what_a_growth_step_admits(arm, support):
    """`atom_bits` IS the write-atom set, against TWO independent derivations.

    A growth step plans from this buffer and derives nothing on the host, so the buffer
    is not an internal -- it is the admission's input, and a wrong bit in it is either a
    page committed for nothing (a bit set too many) or a deposit silently dropped into an
    absent atom (a bit missing). It is checked against `_write_atom_bitmap` -- the
    Cartesian-product definition in torch, with no share of the kernel's expansion -- and
    against `_leaf_support` (`schedule.py`'s `write_mask` taken to leaf granularity) ORed
    16-deep, so what is compared is the SUPPORT RULE and not two copies of one condensation.

    The leaf map itself is no longer published by anything: the step's two maps live and
    die inside the CTA that built them, which is what the absorbed verdict buys, so the
    leaf-granular reference reaches the assertion THROUGH the condensation.

    The arena is EMPTY here, which is the growth step's own condition -- every atom the
    step writes is new -- so the equality is asserted exactly where the shipped path
    consumes it.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=419 + len(widths), support=_SUPPORT[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)

    arena = _arena(widths, BC, BH)
    fired, workspace = _verdict(fixture, config, arena)
    torch.cuda.synchronize()
    reference = _write_atom_bitmap(write, config)
    assert fired, "an empty table and a step that writes: this cell IS a growth step"
    assert reference.any(), "a vacuous cell would compare two empty bitmaps"
    assert torch.equal(workspace.atom_bits, reference), (
        "the set the step published is not the set the definition derives, so a growth "
        "step would admit the wrong atoms")
    assert torch.equal(
        reference, _leaf_support(write, widths, config).view(
            BH, config.N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)), (
        "the enumerated reference and the leaf support's own definition condense "
        "differently")

    #: and the admission agrees with the verdict: planning from the published set leaves
    #: nothing for the next step to find.
    arena.plan_exact(workspace.atom_bits, fresh=False)
    arena.wait()
    torch.cuda.synchronize()
    assert not _verdict(fixture, config, arena)[0], (
        "admitting the step's own published set did not settle the verdict")


# ---------------------------------------------------------------------------
# 3. the verdict is PER BATCH-HEAD
# ---------------------------------------------------------------------------

def _hold_out_one_atom_of_one_bh(fixture, config, arena):
    """Admit the whole write set except one atom of ONE batch-head. `(bitmap, bh)`."""
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    row, column = (bitmap.nonzero())[0].tolist()
    held_out = bitmap.clone()
    held_out[row, column] = False
    arena.plan_exact(held_out, fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    return bitmap, row


@pytest.mark.parametrize("arm", _ARMS[:1], ids=lambda a: "x".join(map(str, a[0])))
def test_a_mixed_batch_advances_the_resident_batch_heads_and_no_others(arm):
    """ONE batch-head needy, the rest resident: the verdict fires and the rest RUN.

    This is the per-batch-head design's non-vacuity. The old mechanism was a global
    no-op -- one absent atom anywhere stopped the whole batch -- and a cell asserting
    THAT would pass here for the wrong reason. What is asserted is the CONTRACT: the
    needy batch-head's rows are untouched, and every other batch-head's advanced.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=2311, support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    assert BH > 1, "a mixed-batch cell needs more than one batch-head"

    arena = _arena(widths, BC, BH)
    _bitmap, needy = _hold_out_one_atom_of_one_bh(fixture, config, arena)
    before = _gathered(arena, BH)
    fired, workspace = _verdict(fixture, config, arena)
    torch.cuda.synchronize()
    after = _gathered(arena, BH)

    assert fired, "the held-out atom did not fire the verdict"
    assert int(workspace.growth.sum()) == 1, (
        "exactly one batch-head cannot address its write set; the verdict vector says "
        f"{int(workspace.growth.sum())}")
    assert bool(workspace.growth[needy]), "the wrong batch-head is flagged"
    assert bytes_equal(after[needy], before[needy]), (
        "the needy batch-head deposited anyway -- its write would be lost into the atom "
        "it cannot address")
    others = [i for i in range(BH) if i != needy]
    assert not bytes_equal(after[others], before[others]), (
        "the resident batch-heads did NOT advance: this is the global-no-op mechanism, "
        "and the contract is per batch-head")


@pytest.mark.parametrize("arm", _ARMS[:1], ids=lambda a: "x".join(map(str, a[0])))
def test_the_admission_and_replay_advance_every_sequence_exactly_once(arm):
    """no-op, admit, replay == admit first. The whole of "exactly once", as `torch.equal`.

    The mixed batch is what makes it a claim: the batch-heads that ran in the first walk
    must NOT deposit again in the second, and the batch-head that could not run must
    deposit once. The reference admits everything up front and runs one step.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=2317, support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    reference = _arena(widths, BC, BH)
    reference.plan_exact(bitmap, fresh=True)
    reference.wait()
    y_reference, _ = _step(fixture, config, reference)

    replayed = _arena(widths, BC, BH)
    _hold_out_one_atom_of_one_bh(fixture, config, replayed)
    _y, workspace = _step(fixture, config, replayed)
    replayed.plan_exact(workspace.atom_bits, fresh=False)
    replayed.wait()
    y_replayed, _ = _step(fixture, config, replayed, workspace=workspace)
    torch.cuda.synchronize()

    assert torch.equal(y_reference, y_replayed), (
        "the replay's `y` is not the admit-first `y`: a batch-head either deposited "
        "twice or was read before it deposited")
    assert bytes_equal(_gathered(reference, BH), _gathered(replayed, BH)), (
        "the replayed state is not the admit-first state -- exactly-once failed")


# ---------------------------------------------------------------------------
# 4. the slack pool: the same verdict with a capacity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_a_covering_pool_admits_the_growth_itself_and_raises_no_verdict(arm):
    """Pool-covered growth is not the host's business, and the state is the same state.

    The reference admits from the host with no pool at all; the pooled arena starts empty
    and lets the step admit itself. Both must reach the same state, because a pool slot is
    an ordinary zeroed arena slot and an atom admitted into one is indistinguishable from
    an atom `plan_exact` admitted.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=4801 + len(widths), support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    reference = _arena(widths, BC, BH)
    reference.plan_exact(bitmap, fresh=True)
    reference.wait()
    y_reference, _ = _step(fixture, config, reference)

    pooled = _arena(widths, BC, BH, pool_slack=1.0)
    fired, workspace = _verdict(fixture, config, pooled, pool=pooled)
    torch.cuda.synchronize()
    assert not fired, (
        "a pool that covers the step's demand still sent the host a verdict; there is "
        "nothing for the host to do and the step already admitted itself")
    assert int((pooled.page_table >= 0).sum()) == int(bitmap.sum()), (
        "the device claim did not land in the table")
    assert pooled.pool_headroom() == pooled.pool_capacity - int(bitmap.sum(1).max()), (
        "the pool cursor did not advance by the batch-head's own demand")
    y_pooled, _ = _step(fixture, config, pooled, workspace=workspace, pool=pooled)
    torch.cuda.synchronize()
    assert not torch.equal(y_pooled, y_reference), (
        "the second pooled step must differ -- it deposits on top of the first")

    fresh = _arena(widths, BC, BH, pool_slack=1.0)
    y_first, _ = _step(fixture, config, fresh, pool=fresh)
    torch.cuda.synchronize()
    assert torch.equal(y_first, y_reference), (
        "a pool-admitted atom is distinguishable from a host-admitted one; it must not be")
    assert bytes_equal(_gathered(fresh, BH), _gathered(reference, BH)), (
        "the pool-admitted state is not the host-admitted state")


@pytest.mark.parametrize("arm", _ARMS[:1], ids=lambda a: "x".join(map(str, a[0])))
def test_an_exhausted_pool_degrades_to_the_host_path_exactly(arm):
    """A pool smaller than one step's demand fires the verdict and admits NOTHING.

    All-or-nothing per batch-head: a partial claim would deposit part of a step and leave
    the rest to a replay that deposits it again. The degrade is the pool-free path, and
    the chain it ends in is the admit-first state.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=4909, support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    demand = int(bitmap.sum(1).max())

    reference = _arena(widths, BC, BH)
    reference.plan_exact(bitmap, fresh=True)
    reference.wait()
    y_reference, _ = _step(fixture, config, reference)

    #: A pool one slot short of the thinnest demand: every batch-head is needy.
    slack = (demand - 1) / config.N * MMA_K_QUANTUM
    starved = _arena(widths, BC, BH, pool_slack=max(slack, 0.0))
    assert starved.pool_capacity < demand, "this cell needs a pool that cannot cover"
    before = starved.state.clone()
    fired, workspace = _verdict(fixture, config, starved, pool=starved)
    torch.cuda.synchronize()
    assert fired, "an exhausted pool must degrade to the host verdict"
    assert int((starved.page_table >= 0).sum()) == 0, (
        "an exhausted pool admitted a PREFIX of the demand; the claim is all-or-nothing")
    assert bytes_equal(starved.state, before), "a needy step wrote state"

    starved.plan_exact(workspace.atom_bits, fresh=False)
    starved.wait()
    y_replayed, _ = _step(fixture, config, starved, workspace=workspace, pool=starved)
    torch.cuda.synchronize()
    assert torch.equal(y_replayed, y_reference)
    assert bytes_equal(_gathered(starved, BH), _gathered(reference, BH))
