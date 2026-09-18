# `rola/ops/paging.py` — page commitment, keyed to the MMA atom

Moved from the module docstring; the module keeps the vocabulary and the
one-sentence purpose.  Companion: `vmm_owner.md` (the VMM backing).

## What Changed, And Why It Is Not A Tuning Choice

``page_states := MMA_K_QUANTUM`` (was ``BC``). The ruling: **``BC`` is a re-derivable
COMPUTE tile; a page is a MEMORY-RESIDENCY atom.** Their consumers are different
objects with different lifetimes -- admission bitmaps, arena pools, residency ledgers,
warm reuse across sequences -- and every one of them must survive a ``BC`` change
unbiased. The re-key was ruled while a PLANNER chose ``BC`` per launch: a
``BC``-keyed page would have to be rebuilt whenever the planner moved it, which is
exactly the arena-rebuild class the re-key kills. That planner is retired
(``DELETIONS.md``) and ``BC`` is now a property of the built chunk arm, which
weakens the URGENCY of the argument and none of its force -- ``BC`` still varies
across arms and across rebuilds of the matrix, and a residency structure keyed to
a compute tile would still be re-keyed by each such move.

Every legal ``BC`` is a whole number of atoms by construction (``16 | BC``), so an
owner spans exactly ``QPO`` atoms and a CTA resolves ``QPO`` slots at startup instead
of one page. Residency tests move to atom granularity -- the SAME atom as the
emptiness-skipping quantum -- so ONE predicate serves both.

**THE ADDRESS.** Logical atom id ``bh * (N / MMA_K_QUANTUM) + leaf // MMA_K_QUANTUM``;
``table[atom id] = slot | ABSENT``; the value address is
``((slot << LOG2_ATOM | row) * cols + v)`` with ``row = leaf & (MMA_K_QUANTUM - 1)``.
Both shifts are DERIVED from the named constant (invariant I1) -- see
:data:`_LOG2_ATOM_LEAVES` -- so there is no ``4`` and no ``15`` anywhere in this file.

**WHO RESOLVES IT.** The chunk consumer, at the two ends of its state I/O and
nowhere else -- that kernel touches pages only there, so the resolve is one
``int32`` load per atom rather than a per-CTA-startup cost. It resolves
``slot = page_tbl ? page_tbl[atom] : atom``, which makes the DENSE plane the same
kernel with a null table (the slot IS the atom), and paging ONE base translation
rather than a second path. ``slot < 0`` -- an atom the plan committed nothing for
-- is skipped at both ends, exactly and not defensively, because the plan's bitmap
is the write set exactly (``../facts/liveness.md``). ``chunk_kernel.md`` carries the
addressing, the CTA-uniformity of the barriers around that skip, and the 4 GiB
plane bound the 32-bit displacement form owes the caller.

## Allocation: Planned-Exact-Async

The pool is sized EXACTLY, not grown: the popcount of the atom condensation is the
number of atoms this call will touch, so the arena knows its own footprint before the
consumer launches, and there is nothing to discover. The condensation comes from the
call's own arm and never from a second derivation — the liveness pass at `T > 1`
([`../facts/liveness.md`](../facts/liveness.md)), the decode step's own published `atom_bits` at `T = 1`
([`decode/decode_api.md` §4](../decode/decode_api.md#the-verdict)) — and both are EXACT
rather than a bound, which is what makes `resident == realized` a fact instead of a hope.

Allocation and zero-init issue on a SIDE STREAM as soon as the epilogue lands, and the
consumer launch waits on ONE event. That is the whole of the asynchrony: no polling, no
second sync, and the zeroing overlaps the schedule build rather than serializing with
it.

**THERE IS NO CAP, AND THAT IS THE POINT.** With the footprint known exactly, an
allocator ceiling is either never-binding or a self-inflicted refusal, so the arena
carries none: its capacity IS the dense limit and it COMMITS only what the plan
admits — the limit is RESERVED virtually and committed physically per plan, which is
[the two backings](#backings) below. The former ``CAPPED`` mode (partial grant plus
an admit mask that annihilated the denied atoms' writes) was DELETED before it: it
computed a DIFFERENT function from the one the caller asked for, and the closed-world
standard is that a demand the binary cannot serve is refused at the host rather than
silently degraded. The ``max_physical_pages`` parameter that survived it is deleted
with the paged-state-I/O integration.

The consequence downstream is that admission is TOTAL: every atom the routing realizes
is resident, so ``admit bit set``, ``resident`` and ``page_table[atom] >= 0`` are one
predicate and the arena publishes exactly one of them (the table).

## <a id="the-slack-pool"></a>The Slack Pool — A Decode Step Admits Itself

`plan_exact` is a HOST call: the admission it performs is exactly what a paged decode
step needs when it grows, and every growth step before this feature reached the host for
it, once. `rola.state(pool_slack=...)` is the escape from that reach, and it is storage
mechanics rather than a mode — the same class of fact as ``strategy`` or the backing
itself.

**SIZING.** ``pool_slack`` is a FRACTION of full residency, resolved PER `bh`:
`PageArena.pool_capacity = ceil(pool_slack * atoms_per_bh)` slots are reserved for EACH
batch-head, not for the arena as a whole. Per `bh`, because the device claim below is per
`bh` too — every CTA of one batch-head derives its slots from that batch-head's own
region and cursor, and a shared cursor across batch-heads is exactly the cross-CTA
communication the design has to avoid. `pool_slack = 0.0`, the default, reserves nothing:
`pool_capacity` is `0`, the three pool tensors are `None`, and every growth step goes to
the host exactly as it always did. This is EXACT COMMITMENT, restated: the pool is an
addition to it, never a replacement, and turning it off costs nothing that was not already
being paid.

**THE SLOTS ARE ORDINARY, PRE-COMMITTED AND PRE-ZEROED.** A pool's slots come from the
same `ExtentAllocator` every other admission draws from — taken and committed at
`PageArena` construction (and again at `reset()`, since the allocator cursor they came
from is rewound there too) — and zeroed on the host's own time, off the step's critical
path. An atom the device later claims into one of them is therefore BYTE-INDISTINGUISHABLE
from one `plan_exact` admitted: no migration, no copy, and the dense-vs-paged equality
gate is untouched by the pool existing.

**THE DEVICE CLAIM is replicated and canonical, with no atomic and no cross-CTA
communication.** Every CTA of a batch-head independently computes the SAME mapping: the
batch-head's touched-unmapped atoms, in ASCENDING ATOM ID, taken from `pool_slots[bh]`
starting at `pool_cursor[bh]`, via a block scan every CTA runs identically over the same
inputs. It is all-or-nothing per batch-head — a CTA either finds the demand fits under
`pool_cap - cursor` and claims the whole of it, or claims nothing and the batch-head is
`needy` exactly as an unpooled one would be.

<a id="the-shadow-row"></a>**WHY IT WRITES A SHADOW ROW, AND NOT THE TABLE.** The claim's
predicate is `page_tbl[a] < 0` — an atom not yet mapped. Writing a claimed slot into the
real table during the prologue would move that predicate out from under a SIBLING CTA
still computing its own claim, and two CTAs' independently-replicated claims would then
diverge over which atoms remain unclaimed. The claim instead writes `pool_map[bh]` — a
WHOLE SHADOW COPY of that batch-head's table row, the claim merged in — and the walk
addresses through THAT row instead of the table, by choosing one base pointer once,
before the walk starts. The walk's own body knows nothing about pools: it is the same
loop over `page_row[atom]` whether `page_row` is the table or the shadow. The batch-head's
LAST-ARRIVING CTA — elected from the same split-K `ctr[bh]` the combine uses — folds the
shadow row's new entries into the real table and advances the cursor, once, after every
sibling CTA is done reading either row.

**EXHAUSTION DEGRADES TO THE POOL-FREE PATH, EXACTLY.** A batch-head whose demand exceeds
`pool_cap - cursor` claims nothing and is `needy`: the verdict, the per-`bh` no-op, the
host's `admit_growth`, and the replay are the same four steps an unpooled growth step
always took. There is no new failure mode and no partial claim to unwind.

**POOL-COVERED GROWTH RAISES NO VERDICT.** The step admitted itself, so there is nothing
urgent for the host — `growth_pending` reads `False`. What a serving loop watches instead
is `rola.engine.dags.decode_dag.pool_headroom(state)`: the MINIMUM free slots over
batch-heads (the first one to run out is the one that will need the host). It IS a
device-to-host read and it does synchronize, which is why it is not on the step's path
and why the pool exists in the first place: the caller polls it on its own schedule,
between steps, at a cadence the pool's depth makes safe. `refill_pool(state)` takes fresh
slots for exactly the positions the device spent, commits and zeroes them, and rewinds the
cursors — on whatever schedule the caller likes, entirely off the step's path. A claimed
slot is ordinary residency the moment the table fold lands, so a refill never reclaims
anything; it only restocks.

**`PageArena.reset()` RE-RESERVES THE POOL.** The pool's slots came from the allocator
cursor `reset()` rewinds, so leaving them in place would let a future admission hand out
slots the pool still names as its own — the one way a device claim could alias a host
admission. `reset()` therefore re-runs the same reservation `__init__` did, on the freshly
rewound allocator.

## <a id="backings"></a>The Two Backings, And The Receipt

Paging's claim is a claim about PHYSICAL MEMORY, and a pool allocated at the dense limit
satisfies every equivalence gate while saving none of it. So the arena's device backing is
the CUDA driver's virtual memory manager ([`vmm_owner.md`](vmm_owner.md)): the dense limit
is RESERVED as address space, which costs nothing, and physical memory is committed as
:class:`ExtentAllocator` hands out slots. ``PageArena.committed_bytes`` reads that
commitment out of the owner's own accounting rather than out of ``nvidia-smi``, which
reports the caching allocator's reservations too and cannot attribute bytes to one arena.

``backing='dense'`` is one ``torch.zeros`` over every slot the topology could need. It is
the HOST backing — a CPU device has no driver VMM at all — and the instrument the
equivalence gates measure the VMM arena against. It is not a fallback, and the difference
matters: ``backing='auto'`` (the default) resolves to the VMM backing on CUDA and RAISES on
a CUDA device whose driver cannot serve it, naming the driver's own reason. A measurement
must not be able to quote the dense allocation as the feature.

The capability is a device fact, proven once per device index by the driver round trip
:func:`vmm_supported` runs and cached (``rola.ops.paging._VMM_PROBE``) — never re-asked per
arena.

**COMMITMENT FOLLOWS THE CURSOR, NOT THE RESIDENCY.** The two differ: the residency is the
page table's popcount and the cursor is what :class:`ExtentAllocator` has handed out,
including the alignment gap it opens when a batch does not fit an extent's remainder. The
cursor is a host integer and the residency is a device reduction, so the cursor is what the
arena's own logic runs on and what sizes the commitment; ``allocated_pages`` is that
number, and ``resident_pages`` is the exact one, a REPORTING surface that syncs and never
appears on a measured path.

## <a id="granule"></a>The Savings-Claim Rule: State The Pool Size

**A paging-savings statement that does not state its pool size is not a result.** The
driver commits in ALLOCATION GRANULES (2 MiB on sm_86), so a pool smaller than one
granule commits its whole dense limit in a single mapping and saves NOTHING, however
sparse its residency. Measured, at `BH = 8`, `widths = (16, 16)`, `cols = 65`: a 0.51 MiB
dense limit reports `committed 2.00 MiB` — 394% of dense — while the same routing at
`widths = (8, 8, 8, 8)` (8.12 MiB dense) reports 10.00 MiB, and the `(128, 128)`,
`BH = 32` pool of `tests/integration/test_page_arena_vmm.py` reports **27.7% committed
against a 25% plan** (37,748,736 B of a 130 MiB dense limit — `ceil(8192/504) = 17`
chunks plus the owner's one-chunk lookahead, asserted exactly rather than as a band).

Two consequences, both binding on anything that quotes a number:

* every savings claim carries its pool size and its dense counterfactual — which is why
  `PagingResult` reports `committed_bytes` and `dense_bytes` beside `committed_fraction`,
  and why a paging measurement reports all three;
* The small topologies the test suite uses (0.5–8 MiB pools) are inside the granule
  regime and CANNOT show a saving. They gate correctness, never the claim. The claim is
  measured at pools many chunks wide, which is the standing
  design-for-1M-scale rule arriving in the memory system.

A second floor sits above the granule one and is a property of the ROUTING rather than
the driver: with unstructured sparsity at long `L`, every atom is live (the measurement,
reproduced by the since-deleted paging bench at `logit_gain = 8`: 2048/2048 atoms resident). Sparse
RESIDENCY needs STRUCTURED support and does not follow from entmax alone.

## Extents Are Allocator Policy And Nothing Else

The pool is carved into runs of ``C_EXTENT_ATOMS`` consecutive slots, and atoms
admitted together take consecutive slot ids -- so an owner's ``QPO`` atoms land
contiguous in the common case and the CTA resolves ONE base instead of ``QPO``
independent lookups. This is INVISIBLE to the table format and to the kernel: the table
is still ``atom -> slot``, and a caller cannot tell an extent-allocated arena from a
first-fit one except by reading the slot ids. Transfers batch at extent granularity.

## Scope

The arena stays CALL-SCOPED by default (``fresh=`` contract unchanged):
:meth:`plan` resets before every call unless the caller opts into continuation.
:meth:`adopt_dense` / :meth:`materialize` remain the explicit, lossless bridges to the
dense cache contract.

Arena exclusion is respected literally: the arena holds persistent cross-chunk
state pages ONLY.
