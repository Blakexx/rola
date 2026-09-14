"""Gate for the paging ATOM re-key (docs/internals/paging/).

The subject is the arena's KEYING and its allocator policy, both of which are now pure
Python over torch tensors and are therefore gateable without the built extension. What
is NOT gateable here -- the consumer kernel's startup slot resolution -- is recorded in
the findings doc as an unassigned hand-off, not asserted around.

Symbols: ``N`` leaves; ``BC`` leaves per owner; ``cols`` = ``d_v`` (+1 under global);
``BH`` = batch x heads; ``MMA_K_QUANTUM = 16``; atom = one 16-leaf block in canonical
leaf order; ``QPO = BC / 16`` atoms per owner.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.ops.paging import (
    MMA_K_QUANTUM,
    AtomKeying,
    ExtentAllocator,
    PageArena,
    bytes_equal,
    to_split_planes,
)

_DEVICE = torch.device("cpu")

#: THE INSTANTIATED `BC` plus one hypothetical. Every quantity the arena stores is
#: `BC`-INVARIANT after the re-key, so a `BC`-derived literal shows up as a shape
#: mismatch on the fake row -- or, worse, as a silent pass on all three, which is what
#: the alternating-BC gate below is for.
_BCS = [16, 64, 32]
_BC_IDS = ["BC16-instantiated", "BC64-instantiated", "BC32-FAKE"]


def _arena(widths=(8, 16), BC=64, cols=4, BH=2, **kwargs):
    return PageArena(widths=widths, BC=BC, cols=cols, BH=BH, device=_DEVICE, **kwargs)


# ---------------------------------------------------------------------------
# 1. The keying arithmetic
# ---------------------------------------------------------------------------


def test_the_atom_keying_is_the_spec_expression_with_no_derived_literals():
    """Atom id, row, and the shifted address, checked against the ARITHMETIC form.

    The kernel-side expression uses shifts; the definition uses division. Asserting one
    against the other is what pins that the shift constants are derived from
    ``MMA_K_QUANTUM`` and not from a 4 and a 15 that happen to agree with it today.
    """
    keying = AtomKeying(N=128, cols=5, BH=3)
    assert keying.atoms_per_bh == 128 // MMA_K_QUANTUM
    assert keying.logical_atoms == 3 * (128 // MMA_K_QUANTUM)
    for bh in range(3):
        for leaf in range(128):
            assert keying.atom_of(bh, leaf) == bh * keying.atoms_per_bh + leaf // MMA_K_QUANTUM
            assert keying.row_of(leaf) == leaf % MMA_K_QUANTUM
            for v in (0, 4):
                assert keying.address(7, leaf, v) == (
                    (7 * MMA_K_QUANTUM + leaf % MMA_K_QUANTUM) * 5 + v)

    with pytest.raises(ValueError, match="whole number of"):
        AtomKeying(N=127, cols=4, BH=1)
    with pytest.raises(ValueError, match="whole number of"):
        keying.atoms_per_owner(24)


@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_every_stored_object_is_BC_invariant(BC):
    """THE property the re-key exists for, asserted directly.

    ``BC`` is a re-derivable COMPUTE tile, a page is a MEMORY-RESIDENCY atom. After
    the re-key the table's shape, the pool's shape and the address expression mention
    only ``N``, ``cols`` and ``MMA_K_QUANTUM``. A single
    ``BC``-keyed survivor would show up here as a shape that moved with ``BC``.
    """
    widths, cols, BH = (8, 16), 4, 2
    arena = _arena(widths=widths, BC=BC, cols=cols, BH=BH)
    N = prod(widths)
    assert tuple(arena.page_table.shape) == (BH, N // MMA_K_QUANTUM)
    assert tuple(arena.state.shape) == (arena.logical_pages, MMA_K_QUANTUM, cols)
    assert arena.logical_pages == BH * N // MMA_K_QUANTUM
    assert arena.dense_bytes() == BH * N * cols * 4
    assert arena.atoms_per_owner == BC // MMA_K_QUANTUM


def test_two_sequences_at_alternating_BC_produce_bit_identical_residency():
    """The alternating-``BC`` gate: a ``BC`` flip is a no-op for residency.

    ``BC`` is PER LAUNCH, so this is not a hypothetical: the caller may hand
    consecutive launches different tiles over the same live state. The re-key's whole
    claim is that those launches address the SAME bytes of the SAME slots. Here the same
    write support is admitted twice, at ``BC = 16`` and at ``BC = 64``, and the page
    table and the pool contents must come out bit-identical.

    Before the re-key this test could not even be written: the two arenas had different
    table SHAPES.
    """
    widths, cols, BH = (8, 16), 4, 2
    N = prod(widths)
    generator = torch.Generator().manual_seed(4)
    atoms = torch.rand((BH, N // MMA_K_QUANTUM), generator=generator) < 0.4

    results = []
    for BC in (16, 64):
        arena = _arena(widths=widths, BC=BC, cols=cols, BH=BH)
        arena.plan_exact(atoms)
        # Write a value through the ADDRESS EXPRESSION, so the comparison covers the
        # addressing and not merely the table.
        for bh in range(BH):
            for leaf in range(N):
                slot = int(arena.page_table[bh, leaf // MMA_K_QUANTUM])
                if slot >= 0:
                    flat = arena.state.reshape(-1)
                    flat[arena.keying.address(slot, leaf, 0)] = float(bh * N + leaf)
        results.append((arena.page_table.clone(), arena.state.clone(),
                        arena.last_result))

    (table_a, state_a, res_a), (table_b, state_b, res_b) = results
    assert torch.equal(table_a, table_b), "the page table moved with BC"
    assert bytes_equal(state_a, state_b), "the addressed bytes moved with BC"
    assert res_a.allocated_pages == res_b.allocated_pages == int(atoms.sum())


# ---------------------------------------------------------------------------
# 2. Planned-exact admission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_planned_exact_sizes_the_pool_to_the_popcount_and_zeroes_what_it_admits(BC):
    """The popcount of the atom condensation IS the footprint.

    Not a bound, not a growth target -- the exact number, known before the launch. And
    an admitted slot must read as ZERO whatever was in it before, because a fresh atom
    is a fresh state; the pool is deliberately pre-dirtied here so a missing zero-init
    cannot pass.
    """
    widths, cols, BH = (8, 16), 4, 2
    N = prod(widths)
    arena = _arena(widths=widths, BC=BC, cols=cols, BH=BH)
    arena.state.fill_(7.0)
    generator = torch.Generator().manual_seed(5)
    atoms = torch.rand((BH, N // MMA_K_QUANTUM), generator=generator) < 0.3

    result = arena.plan_exact(atoms)
    touched = int(atoms.sum())
    assert result.allocated_pages == touched
    assert result.realized_owners == touched
    assert result.allocated_bytes == touched * MMA_K_QUANTUM * cols * 4
    assert (arena.page_table >= 0).sum() == touched
    assert torch.equal(arena.page_table >= 0, atoms)
    assert float(arena.state[:touched].abs().max()) == 0.0, (
        "an admitted slot kept stale bytes; a fresh atom must read as a fresh state")


def test_a_continuation_keeps_what_it_already_holds_and_a_fresh_call_does_not():
    """The ``fresh=`` contract, unchanged by the re-key. No implicit third option."""
    widths, cols, BH = (8, 16), 4, 2
    N = prod(widths)
    arena = _arena(widths=widths, cols=cols, BH=BH)
    first = torch.zeros((BH, N // MMA_K_QUANTUM), dtype=torch.bool)
    first[0, :2] = True
    second = torch.zeros_like(first)
    second[0, 2:4] = True

    arena.plan_exact(first)
    arena.plan_exact(second, fresh=False)
    assert int((arena.page_table >= 0).sum()) == 4, "a continuation dropped live atoms"
    arena.plan_exact(second, fresh=True)
    assert int((arena.page_table >= 0).sum()) == 2, "a fresh call carried state forward"


def test_an_exhausted_allocator_REFUSES_and_never_grants_a_prefix():
    """A demand the pool cannot serve RAISES; it never grants a prefix.

    RETIRED IN PART WITH THE CAP: the arena has no ceiling parameter any
    more, so exhaustion is no longer reachable through `PageArena` at all -- the
    capacity IS the dense limit and the plan's demand is a subset of it by
    construction. The invariant itself still has an owner, and this is it: the
    ALLOCATOR refuses all-or-nothing, naming the demand and the pool, and leaves
    itself exactly as it was so a caller that catches it has not lost state.
    """
    allocator = ExtentAllocator(4, 4)
    with pytest.raises(RuntimeError, match=r"8 new atoms demanded.*arena of 4 atoms"):
        allocator.take(8)
    assert allocator.allocated == 0, (
        "a refused grant moved the allocator; refusal must be all-or-nothing")
    assert allocator.take(4) == 0 and allocator.allocated == 4, (
        "the same pool must serve a demand that FITS, or the refusal is keyed on the "
        "pool being small rather than on the demand")


# ---------------------------------------------------------------------------
# 3. The extent allocator (policy only)
# ---------------------------------------------------------------------------


def test_the_extent_allocator_keeps_a_batch_contiguous_and_is_invisible_otherwise():
    """Runs of ``C_EXTENT_ATOMS``, and a batch that does not fit opens a new one.

    Both branches are STRUCTURAL -- a count against a remainder, never a size against a
    threshold -- and neither is observable in the table FORMAT, which is why the gate
    reads slot ids directly instead of inferring contiguity from anything else.
    """
    allocator = ExtentAllocator(capacity=32, extent_atoms=4)
    assert allocator.take(3) == 0           # opens extent 0, leaves 1 slot
    assert allocator.take(1) == 3           # FITS the remainder: no waste
    assert allocator.take(4) == 4           # exactly one extent
    assert allocator.take(3) == 8           # opens extent 2
    assert allocator.take(3) == 12, "a batch larger than the remainder straddled"
    assert allocator.allocated == 15

    # The ceiling REFUSES rather than granting a prefix, and leaves the cursor put.
    small = ExtentAllocator(capacity=5, extent_atoms=4)
    with pytest.raises(RuntimeError, match="9 new atoms demanded"):
        small.take(9)
    assert small.allocated == 0, "a refused take moved the allocator cursor"
    assert small.take(5) == 0

    with pytest.raises(ValueError, match="extent_atoms"):
        ExtentAllocator(capacity=8, extent_atoms=0)


@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_the_default_extent_is_one_owners_worth_of_atoms(BC):
    """``C_EXTENT_ATOMS`` defaults to ``QPO``, i.e. today's behaviour exactly.

    A ``BC``-keyed page gave an owner's leaves contiguously for free. The default extent
    reproduces that and no more, so a WRONG placeholder cannot be worse than the thing
    the re-key replaces -- which is the rule for choosing a placeholder default.
    """
    arena = _arena(BC=BC)
    assert arena.extent_atoms == BC // MMA_K_QUANTUM


def test_an_owners_atoms_land_contiguous_in_the_common_case():
    """The tendency the extent policy promises -- and it is a tendency, not an invariant.

    One owner's ``QPO`` atoms admitted in one call must take consecutive slots, because
    that is what lets a CTA resolve ONE base at startup instead of ``QPO`` lookups.
    """
    widths, BC, cols, BH = (8, 16), 64, 4, 1
    N = prod(widths)
    arena = _arena(widths=widths, BC=BC, cols=cols, BH=BH)
    atoms = torch.zeros((BH, N // MMA_K_QUANTUM), dtype=torch.bool)
    qpo = BC // MMA_K_QUANTUM
    atoms[0, :qpo] = True                       # owner 0's whole run
    arena.plan_exact(atoms)
    slots = [int(arena.page_table[0, a]) for a in range(qpo)]
    assert slots == list(range(slots[0], slots[0] + qpo)), (
        f"an owner's {qpo} atoms did not land in one extent: {slots}")


# ---------------------------------------------------------------------------
# 4. The dense bridges, re-keyed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_a_fully_resident_arena_materializes_the_canonical_leaf_order(BC):
    """The dense bridge OUT is LOSSLESS across the re-key, at every ``BC``.

    Admit every atom, write a dense state through the slots the table names, and
    ``materialize`` must return it unchanged: the atom layout is a RESHAPE of the
    canonical leaf order, so a round trip that lost a byte would mean the two disagree
    about which leaf a row is. The write goes through ``page_table`` rather than
    assuming an identity mapping, which keeps the claim about the LAYOUT rather than
    about the allocator's cursor.
    """
    widths, cols, BH = (8, 16), 4, 2
    N = prod(widths)
    arena = _arena(widths=widths, BC=BC, cols=cols, BH=BH)
    dense = torch.arange(BH * N * cols, dtype=torch.float32).reshape(BH, N, cols)
    arena.plan_exact(torch.ones((BH, N // MMA_K_QUANTUM), dtype=torch.bool))
    arena.state.index_copy_(
        0, arena.page_table.reshape(-1).long(),
        to_split_planes(dense.reshape(BH * N // MMA_K_QUANTUM, MMA_K_QUANTUM, cols)))
    assert bytes_equal(arena.materialize(), dense)
    assert arena.allocated_pages == arena.logical_pages


def test_materialize_leaves_unmapped_atoms_as_zeros():
    """An atom with no slot reads as a ZERO state, not as whatever a slot held."""
    widths, cols, BH = (8, 16), 4, 1
    N = prod(widths)
    arena = _arena(widths=widths, cols=cols, BH=BH)
    arena.state.fill_(3.0)
    atoms = torch.zeros((BH, N // MMA_K_QUANTUM), dtype=torch.bool)
    atoms[0, 1] = True
    arena.plan_exact(atoms)
    dense = arena.materialize()
    live = dense[0, MMA_K_QUANTUM:2 * MMA_K_QUANTUM]
    assert float(live.abs().max()) == 0.0
    dead = torch.cat([dense[0, :MMA_K_QUANTUM], dense[0, 2 * MMA_K_QUANTUM:]])
    assert float(dead.abs().max()) == 0.0


def test_the_atom_condensation_shape_is_checked_rather_than_broadcast():
    """A caller handing the OWNER condensation instead of the ATOM one must fail loudly."""
    widths, cols, BH = (8, 16), 4, 2
    arena = _arena(widths=widths, cols=cols, BH=BH)
    owners = torch.zeros((BH, prod(widths) // arena.BC), dtype=torch.bool)
    with pytest.raises(ValueError, match="atom condensation"):
        arena.plan_exact(owners)
