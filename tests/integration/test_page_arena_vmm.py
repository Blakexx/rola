# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The arena's VMM backing: the physical-memory receipt, and that it changes nothing else.

Paging's claim is a claim about PHYSICAL MEMORY. An arena whose pool is one allocation at
the dense limit satisfies every equivalence gate in this directory and saves nothing --
which is exactly the state the tree was in before this file existed. So the receipt is
the point, and it is taken from the driver's own accounting (`VmmOwner::committed_bytes`,
the sum of what `cuMemCreate` handed over) rather than from a process-wide instrument:
`nvidia-smi` reports the caching allocator's reservations too and cannot attribute bytes
to one arena.

Vocabulary, defined here and used unqualified below: `N` = leaf count; `cols = d_v + 1`;
`BH = batch * heads`; **atom** = 16 consecutive leaves = the page granule; **slot** = the
physical page an atom is mapped to; a **chunk** = one driver allocation, the unit
commitment moves in; RESERVED means address space exists, COMMITTED means physical memory
does.

**FOUR CLAIMS.**

1. *Reserve is the dense limit; commitment is the plan.* The virtual extent covers every
   slot the topology could ever need -- address space is free -- while the committed
   bytes track what `plan_exact` handed out, rounded to the driver's own granularity.

2. *Growth extends commitment and never moves a base.* This is what makes a stable
   pointer compatible with a growing footprint, and it is the whole reason the backing is
   VMM rather than a reallocating pool.

3. *The two backings compute the same function, bit for bit.* Where the memory came from
   is not a semantic axis, so `y` and the state are `torch.equal` and not merely close.

4. *There is no silent fallback.* A CUDA device whose driver cannot serve VMM raises; the
   dense allocation is reachable only by asking for it by name.

Run: ``pytest tests/integration/test_page_arena_vmm.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.decode import _decode_step, _write_atom_bitmap
from rola.ops.paging import MMA_K_QUANTUM, PageArena, to_split_planes, vmm_supported
from tests.integration.test_decode_paging import _ARMS, _config, _f32, _fixture

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the VMM backing is CUDA-only"),
    pytest.mark.skipif(not (torch.cuda.is_available() and vmm_supported("cuda")),
                       reason="this device's driver does not serve VMM"),
]

_COLS = 65
_PAGE_BYTES = MMA_K_QUANTUM * _COLS * 4

#: A pool whose dense limit is LARGE against the driver's 2 MiB allocation granularity.
#: The receipt is only meaningful when the pool is many chunks wide: an arena smaller than
#: one chunk commits its whole dense limit in a single mapping and saves nothing, which is
#: a hardware fact about the granularity rather than a defect in the arena.
_BIG = dict(widths=(128, 128), BC=256, BH=32)


def _granular(nbytes: int, owner) -> int:
    """`nbytes` rounded up to the driver's allocation granularity -- the LAST rounding.

    Every extent the owner maps passes through it, so the receipt's arithmetic must too;
    a target below one granule is not a smaller mapping, it is the same one.
    """
    unit = owner.allocation_granularity
    return -(-nbytes // unit) * unit


def _big_arena(**kwargs) -> PageArena:
    return PageArena(cols=_COLS, device=torch.device("cuda"), **_BIG, **kwargs)


def _sparse_bitmap(arena: PageArena, every: int) -> torch.Tensor:
    bitmap = torch.zeros(arena.BH, arena.keying.atoms_per_bh, dtype=torch.bool, device="cuda")
    bitmap[:, ::every] = True
    return bitmap


# ---------------------------------------------------------------------------
# 1. the receipt
# ---------------------------------------------------------------------------

def test_a_sparse_plan_commits_sparse_physical_memory():
    """A quarter-resident arena costs a quarter of the dense footprint, plus granularity.

    The whole of P67 in one assertion. `dense_bytes` is the counterfactual -- what an
    unpaged `[BH, N, cols]` state occupies, and what the arena's own pool cost before this
    backing existed -- and `committed_bytes` is what the driver actually handed over.
    """
    arena = _big_arena()
    owner = arena._owner
    chunk_bytes = owner.chunk_pages * _PAGE_BYTES

    #: RESERVE IS THE DENSE LIMIT. Address space is free, so the arena never has to decide
    #: how large it might get -- which is what removes the ceiling parameter.
    assert owner.dense_limit_pages == arena.logical_pages
    assert owner.virtual_bytes >= arena.dense_bytes()
    assert owner.chunk_pages < arena.logical_pages, (
        "this cell needs a pool many chunks wide; a one-chunk pool cannot show a receipt")

    result = arena.plan_exact(_sparse_bitmap(arena, every=4), fresh=True)
    arena.wait()
    torch.cuda.synchronize()

    #: The three quantities, and the ONE inequality between them that is the claim.
    assert result.allocated_pages == arena.logical_pages // 4
    assert arena.resident_pages == result.allocated_pages, (
        "the plan opened an alignment gap; this cell's admission is one contiguous run")
    assert result.committed_bytes < result.dense_bytes, "the arena committed the dense limit"
    assert result.committed_fraction < 0.35, (
        f"a quarter-resident arena committed {result.committed_fraction:.1%} of the dense "
        "footprint; the savings are being eaten by chunk granularity")

    #: EXACTLY, not approximately: commitment is the allocator's cursor rounded UP to a
    #: chunk boundary, plus the owner's one-chunk lookahead, and finally to the driver's
    #: allocation granularity -- which is the last rounding and the one nothing can undo.
    pages = result.allocated_pages
    chunks = -(-pages // owner.chunk_pages)
    expected = min(arena.logical_pages, (chunks + 1) * owner.chunk_pages) * _PAGE_BYTES
    assert result.committed_bytes == _granular(expected, owner)
    assert 0 < result.committed_bytes - pages * _PAGE_BYTES <= 2 * chunk_bytes


def test_the_bridge_commits_the_dense_limit_and_says_so():
    """The counterfactual, measured rather than asserted.

    The dense allocation is the state the tree shipped in before the VMM backing, and its
    `committed_bytes` is the dense limit BY CONSTRUCTION -- one `torch.zeros` over every
    slot the topology could need. Naming it in the same units as the VMM arena is what
    makes the receipt above a comparison instead of a number.
    """
    arena = _big_arena(backing="dense")
    result = arena.plan_exact(_sparse_bitmap(arena, every=4), fresh=True)
    assert arena.backing == "dense" and arena._owner is None
    assert result.committed_bytes == result.dense_bytes
    assert result.committed_fraction == 1.0
    assert result.allocated_pages == arena.logical_pages // 4, (
        "the two backings plan the same admission; only the physical memory differs")


# ---------------------------------------------------------------------------
# 2. growth
# ---------------------------------------------------------------------------

def test_growth_extends_commitment_and_never_moves_the_base():
    """Admitting more commits more, and every pointer already handed out stays valid.

    The base pointer is checked across a growth that crosses several chunk boundaries,
    which is the case a reallocating pool cannot serve: the reservation is what makes the
    address stable while the physical memory behind it changes.
    """
    arena = _big_arena()
    owner = arena._owner
    first = arena.plan_exact(_sparse_bitmap(arena, every=16), fresh=True)
    arena.wait()
    base = arena.state.data_ptr()
    committed = first.committed_bytes
    table_before = arena.page_table.clone()
    present_before = table_before >= 0

    grown = arena.plan_exact(_sparse_bitmap(arena, every=2), fresh=False)
    arena.wait()
    torch.cuda.synchronize()

    assert grown.allocated_pages == arena.logical_pages // 2
    assert grown.committed_bytes > committed, "growth committed nothing"
    assert grown.committed_bytes >= grown.allocated_pages * _PAGE_BYTES, (
        "the allocator handed out slots the driver has not mapped")
    assert owner.mapped_capacity_pages >= grown.allocated_pages
    assert arena.state.data_ptr() == base, "growth moved the plane's base pointer"
    #: The atoms already resident keep their slots: growth is an EXTENSION of residency
    #: and never a re-placement of it.
    assert torch.equal(arena.page_table[present_before], table_before[present_before])
    assert arena.resident_pages == grown.allocated_pages


def test_commitment_survives_reset_and_rollback_returns_it():
    """`reset` is LOGICAL; `rollback_to` is the only thing that gives memory back.

    Dropping the table makes every atom unmapped and the allocator start again, but the
    driver mappings and their addresses survive -- a sequence that ends does not pay to
    re-map for the next one. `rollback_to` is the deliberate shrink, and it refuses
    everything it cannot prove: a target inside a chunk, and the initial chunk itself.
    """
    arena = _big_arena()
    owner = arena._owner
    arena.plan_exact(_sparse_bitmap(arena, every=4), fresh=True)
    arena.wait()
    base = arena.state.data_ptr()
    committed = arena.committed_bytes
    mapped = arena.mapped_pages

    arena.reset()
    assert arena.allocated_pages == 0 and arena.resident_pages == 0
    assert arena.committed_bytes == committed, "reset returned physical memory"
    assert arena.state.data_ptr() == base, "reset moved the base pointer"

    owner.rollback_to(owner.chunk_pages)
    assert arena.mapped_pages == owner.chunk_pages < mapped
    assert arena.committed_bytes == _granular(owner.chunk_pages * _PAGE_BYTES, owner)
    assert arena.state.data_ptr() == base, "rollback moved the base pointer"

    with pytest.raises(RuntimeError, match="not a committed chunk boundary"):
        owner.rollback_to(owner.chunk_pages // 2)
    with pytest.raises(RuntimeError, match="initial headroom chunk"):
        owner.rollback_to(0)

    #: The arena is REUSABLE after both: the next plan re-commits and re-admits.
    arena.plan_exact(_sparse_bitmap(arena, every=8), fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    assert arena.resident_pages == arena.logical_pages // 8
    assert arena.state.data_ptr() == base


# ---------------------------------------------------------------------------
# 3. the two backings compute the same function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_the_two_backings_are_bit_identical(arm):
    """Same routing, same `v`, same contents -- VMM-backed and bridge-backed.

    `torch.equal` and not a tolerance: where the bytes came from is not a semantic axis,
    the addressing is identical under both, and decode's `y` is deterministic by
    construction. Any difference at all would be a real one.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=41 + len(widths), support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    v, read, write, g_write = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    outputs = []
    for backing in ("vmm", "dense"):
        arena = PageArena(widths=widths, BC=BC, cols=_COLS, BH=BH,
                          device=torch.device("cuda"), backing=backing)
        assert arena.backing == backing
        arena.plan_exact(bitmap, fresh=True)
        arena.wait()
        #: The SAME contents in both, written through the table so the slot layouts agree
        #: by admission order rather than by assumption.
        table = arena.page_table
        present = table >= 0
        seed = torch.arange(int(present.sum()) * MMA_K_QUANTUM * _COLS, device="cuda",
                            dtype=torch.float32).view(-1, MMA_K_QUANTUM, _COLS) * 1e-3
        arena.state.index_copy_(0, table[present].long(), to_split_planes(seed))
        y, _, _ = _decode_step(v, read, write, g_write, config, arena.state,
                               page_table=table)
        outputs.append((y, arena.materialize(), arena))

    assert torch.equal(outputs[0][0], outputs[1][0]), "the two backings disagree on y"
    assert torch.equal(outputs[0][1], outputs[1][1]), "the two backings disagree on the state"
    #: NON-VACUITY: the run committed pages, they came back nonzero, and the VMM arena
    #: really did commit less than the dense one.
    assert float(outputs[0][1].abs().max()) > 0.0
    assert outputs[0][2].backing == "vmm" and outputs[0][2]._owner is not None
    assert outputs[0][2].allocated_pages == outputs[1][2].allocated_pages > 0


# ---------------------------------------------------------------------------
# 4. the backing is a detected capability, not a fallback
# ---------------------------------------------------------------------------

def test_auto_resolves_to_vmm_on_cuda_and_dense_off_it():
    """The default is the feature; the bridge is the host's only option, not a rescue."""
    assert _big_arena().backing == "vmm"
    host = PageArena(widths=(16, 16), BC=64, cols=_COLS, BH=2, device=torch.device("cpu"))
    assert host.backing == "dense" and host._owner is None
    with pytest.raises(ValueError, match="CUDA-driver memory"):
        PageArena(widths=(16, 16), BC=64, cols=_COLS, BH=2, device=torch.device("cpu"),
                  backing="vmm")
    with pytest.raises(ValueError, match="'auto', 'vmm' or 'dense'"):
        _big_arena(backing="progressive")


def test_the_capability_is_proven_once_and_cached():
    """The probe is a driver ROUND TRIP, not an attribute read, so it is asked once.

    `rola_vmm_probe` reserves, maps, grants access to, unmaps and releases one
    granularity-sized allocation before it says yes -- which is the only answer worth
    having, and far too expensive to repeat per arena.
    """
    from rola.ops import paging
    from rola.ops._ext import extension

    probe = extension().rola_vmm_probe(device=torch.cuda.current_device())
    assert probe["supported"] and probe["allocation_granularity"] > 0
    assert "probe passed" in probe["reason"]
    assert torch.cuda.current_device() in paging._VMM_PROBE, (
        "the arena did not cache the device's capability")
