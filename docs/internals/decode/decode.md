<!-- ABSENT: K31_R0_ABI_RIPPLE_SPEC cum intra_panel kPanelMaxExponent GLA_MASS_DECAY_SPEC MmaPrec PREC PANEL_MAX_EXPONENT -->
# `csrc/rola/src/decode/decode.cuh` + `csrc/rola/src/decode/decode.cu` — the `T = 1` decode path

This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the GEMV kernel, its dispatch and its host entry. The
mirror rule is one doc per source basename, so `decode.cuh` and `decode.cu` land
here together. **Part A** is the header; **Part B** is the translation unit.

`decode_lattice.cuh` — the `(k, m)` box lattice's factor tables the prologue
builds and the walk resolves a unit from — has its own doc,
[`decode_lattice.md`](decode_lattice.md). The host-facing declaration split is
[`decode_api.md`](decode_api.md).

Symbols used throughout: `N` is the leaf count; `D` the number of routing levels
(`1 <= D <= 4`); `b_l` level `l`'s width; `BH = B * H` the flattened batch-head
axis; `d_v` the value width and `cols = d_v + 1` (the mass column is
unconditional; raw normalization is dead — there is no `GLOBAL` axis anywhere in
this repository, decode included); `n_split` the split-K degree; `k`, `m`,
`s_l`, `g_l`, `BC` the `(k, m)` lattice's own symbols
([`decode_lattice.md`](decode_lattice.md), spec `K31_R0_ABI_RIPPLE_SPEC.md`
§2.1–2.5); `BT` is the tiled consumer's token tile, which appears here only to
be ruled out.

**The tiled consumer, in this document, is a RETIRED arm.** It was the prefill
kernel this path was designed against; P67 D2 deleted its shipped shape whole
and K31 R2 deleted what remained of it entirely, extracting what decode and
paging still need into `csrc/rola/src/facts/` (`DELETIONS.md`; revival by tag
`baseline/pre-k31` or the deletion commits' parents). The contrasts drawn
against it below are kept deliberately and read in the past tense: decode's
shape is what it is BECAUSE of what it declined to inherit, and a design record
that erases the thing declined leaves the decision unexplained. Where a section
says a constant is restated rather than included, the header it was not
included from is one of the deleted ones, and the restatement is now simply
this file's own constant. There is no surviving prefill consumer to name a
current doc for; the prefill arm is deleted pending K31 the rebuild.

---

# Part A — the header (`decode.cuh`)

## <a id="no-stage-2"></a>1. `T = 1` has no stage 2

The decode design ruled that at `T = 1` **every** stage-2 decision is
degenerate:

| stage-2 decision | why it is degenerate at `T = 1` |
|---|---|
| `BT` | one token, so there is no tiling |
| `P` | nothing to split |
| packing | `K` is one token's support, and the masks subsume it |
| statistics | a population of one |

So the decode path bypasses selection ARCHITECTURALLY — not by taking a cheap
default — and its config freezes at the prefill→decode boundary. What is left per
step is data plane only, and its shape is the one §6b ratified: a
**token-stationary, state-streaming gather-GEMV**.

## <a id="kronecker-support"></a>2. The structure, and the one fact it rests on

At `T = 1` a leaf `s` is in the token's read support iff EVERY level's digit
`d_l(s)` has a nonzero read amplitude, so

```
supp(R) = prod_l supp_l(R)                      (a Cartesian product)
```

a Cartesian product at EVERY granularity the `(k, m)` box names: an owner
coordinate, a fixed run offset, an innermost digit. That is the one fact the
walk rests on, and it prunes at each granularity in turn rather than at one:

- an owner coordinate dead on a side kills every unit under it — it never
  enters that side's owner list at all
  ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists));
- a UNIT whose fixed levels have an absent digit on a side is dead on that side
  WHOLE — one register test kills every leaf it would have held
  (`Unit.live_r`/`.live_w`, [`decode_lattice.md#the-unit`](decode_lattice.md#the-unit));
- what survives that is walked over the SET BITS of the innermost level's
  `s_{D-1} <= kDecodeMaxSpan`-wide slice — a fixed, capped cost per live unit,
  not a cost that grows with `N`.

**The leaf order IS the lattice, so there is no map over the leaf space to
consult, exact or bounded.** The step's factor tables — `sum_l ceil(B_l/32)`
mask bits and at most `sum_l g_l` owner-list entries per side
([`decode_lattice.md`](decode_lattice.md)) — are the whole apparatus the walk
reaches a leaf through; nothing here is a term in `N`, and nothing here is
materialized and then swept the way a leaf-granular bitmap would be. The walk
composes a leaf from three small factors instead of testing membership in a
big one.

## <a id="bc-absent"></a>3. `BC` names TWO different things, and only one of them is decode's

Owner-blocked AMORTIZATION is a property of how the TILED consumer shares a
`[BC][cols]` block's cost across many tokens; with one token there is nothing
to amortize, and `DecodeGeometry`'s own docstring is explicit that `BC` MUST
NOT become one of its fields — that arm never existed here
([`engine/plan.py`](../../../rola/engine/plan.py)).

The `(k, m)` lattice's `BC = prod_l s_l` is a different quantity that DOES
exist on this path: the leaf count one OWNER holds, i.e. one unit of the grid
partition the walk enumerates ([`decode_lattice.md`](decode_lattice.md)).
Nothing about it is a launch shape or an amortization decision — it is a
LAYOUT fact, the same one the tiled consumer's `[BC][cols]` block address used
before it (spec §2.6's survivor table: `λ / BC = O`, by construction rather
than by a checked property). The DENSE state address is `(bh * N + leaf) *
cols + c`, leaf-major already, and the PAGED one is keyed to the 16-leaf ATOM
([§4](#paged-address)) rather than to the owner — so even where `BC` names a
real quantity here, no address in either backing is computed FROM it directly;
it is the walk's unit boundary, not a term either address expression carries.

## <a id="paged-address"></a>4. Paged and dense are ONE walk

<a id="the-atom"></a>The page granule is the **atom** — `kAtomLeaves = 16`
consecutive leaves in LATTICE leaf order, spec §4.5's
`page_states := MMA_K_QUANTUM`. Under the mixed-radix intra-box order an atom is 16
consecutive `local` values at a multiple of 16, so it is an ALIGNED DIGIT RECTANGLE for
every admissible span vector — the levels whose fields lie above bit 4 are fixed, the
level straddling it keeps an aligned contiguous sub-run, and the levels below are free
(`carry/box.cuh`, K3/K4). One condition survives, `16 | BC` (an atom lies inside one
owner), checked at [`fill_lattice`](decode_lattice.md#the-refusals) where a page table is
present; it is what lets the candidate-atom test ([§4b](#candidate-atoms)) read a unit's
membership out of ONE aligned window of its innermost write slice. `kAtomLeaves` is geometry-owned and
fixed across every config forever, which is exactly why the paged address addresses
through the atom rather than through `BC` ([§3](#bc-absent)); it is mirrored in
`decode.cuh` rather than included, for the same reason `kMaxLevels` is
([§5](#max-levels)): the decode TUs share no header with the tiled consumer.

The walk resolves an atom's first row as an ELEMENT offset from `p.state`, and the
two backings differ in that offset and in nothing else:

| backing | `page_tbl` | atom base | slot |
|---|---|---|---|
| dense | `nullptr` | `bh * N + (atom << 4)` | the atom itself |
| paged | `[BH, N/16]` int32 | `slot << 4` | `page_tbl[bh * atoms_per_bh + atom]` |

so `row = state + atom_base + (leaf & 15) * cols` under both, and the dense form
recombines to `bh * N + leaf` — today's address, bit for bit. Writing the dense arm
through the atom rather than through the leaf costs nothing and buys the property
that matters: a topology whose `N` is NOT a whole number of atoms (`(3, 5, 7)`,
`(7, 9)`, `(33,)` all appear in the gates) still addresses correctly densely, and it
simply has no paged keying at all — `AtomKeying` refuses to build one.

**The resolve is once per touched ATOM, not per row.** It is a value-keyed cache
(`if ((leaf >> 4) != cached_atom) reload`), so it costs at most one `int32` load per
16-leaf atom, and — the part worth stating — it is correct whatever order the walk
reaches leaves in. The lattice's own ASCENDING `C` order ([§26](#aligned-subranges)) is
what makes 16 consecutive leaves share a slot and makes the cache CHEAP; nothing about
its correctness depends on that order.

The table row is pinned as a plain origin and everything per atom is a displacement
from it, which is the addressing discipline the chunk consumer's SASS gate forced and
this path adopts. The widening multiply that forms `atom_base` is paid once per atom
where the previous code paid one per ROW, so the change strictly reduces it.

<a id="absent-atom"></a>**An absent atom (`slot < 0`) reads as zeros and is not
written.** That is exact rather than defensive: a step's plan admits its write set
EXACTLY (the step's own published condensation, [§4b](#atom-bits)), and a batch-head
whose write set it cannot cover does not walk at all ([§4c](#per-bh-verdict)), so an atom
with no slot is outside the write set, and a never-deposited leaf holds zeros in the dense backing too. The
self-normalizing readout then contributes exactly nothing through it — the
mechanical form of the dead-leaf argument, gated by
`tests/integration/test_decode_paging.py`.

## <a id="growth"></a>4b. Growth — the step's own new work, and its own question

A step may deposit into an atom that is not resident. At `T = 1` the write support is
a Cartesian product ([§2](#kronecker-support)), so the step's write-ATOM set is the
per-level exact-nonzero indicators' product ORed 16-deep.

<a id="absorbed-verdict"></a>**THE STEP ASKS THE QUESTION ITSELF, IN ITS OWN PROLOGUE.**
the exact admission is dear against the launch it precedes, and a sequence whose routing
has settled writes atoms that are already resident. What makes the question nearly free is
that the step has already built the objects it is asked about: the write-side factor tables
([`decode_lattice.md`](decode_lattice.md)) are the ones the walk itself resolves units from,
so the residency test is a pass over the WRITE list's candidate atoms
([§4b](#candidate-atoms)) and a page-table indirection each, and no second launch, no second
fold and no second factor pass exist at all. There is nothing left for a probe kernel to be.

So the order per step is ONE kernel, and then, only where the answer was yes and no pool
covered it, `rola_op`'s order at `L = 1`:

```
decode_step        ->  fold, expand, THE VERDICT, walk, and the write-ATOM set,
                       per batch-head, in one launch
growth_host.copy_  ->  the ONE 4-byte read, after the launch that produced it
  no  -> done                      ->  the walk ran, against the residency it had
  yes -> state._decode_entry_arena ->  plan_exact(scratch.atom_bits, fresh=False)
         arena.wait()              ->  the ONE event the launch waits on
         _decode_step              ->  the same step again, and the batch-heads that
                                       already ran do not run twice ([§4c](#done-flags))
```

<a id="the-table-test"></a>**THE TABLE TEST.** The count of atoms that are written and unmapped
is a block reduction (`cub::BlockScan::ExclusiveSum`) over one `page_row[a] < 0` test per
candidate atom. Every CTA of a batch-head computes it, independently and identically: the
two objects it reads are the CTA's own factor tables and a page table that no launch
mutates until every CTA of that batch-head has arrived ([§4c](#the-last-arriver)).
Recomputing beat publishing when it was measured — the split CTAs' recompute of the fold
and the factor tables is what removed the launch — and the same argument governs here.

<a id="candidate-atoms"></a>**THE CANDIDATE ATOMS ARE ENUMERATED FROM THE SAME LATTICE
PRODUCT THE WALK USES, RESTRICTED TO THE WRITE LIST.** A unit's leaves are a CONTIGUOUS
ALIGNED RUN ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so an atom and a
unit nest one way or the other and the index says which: the low `atoms_per_unit_shift`
bits of `i` pick an atom INSIDE a unit, and `lat_units_per_atom_shift` consecutive units
make ONE atom. Exactly one of the two shifts is ever non-zero, and both are PAGED-ONLY
quantities the host derives once (`decode_lattice.md#the-refusals`).

`candidate_atom(i)` therefore resolves the write-side unit (or the `1 <<
lat_units_per_atom_shift` consecutive ones) through `resolve_unit` over the CANDIDATE TERM
SLOT `D + 1` — the write product in the `live owners x run radix` space, which is the only
one monotone in atom id and is exactly the enumeration this claim was ratified on
([`decode_lattice.md#the-candidate-slot`](decode_lattice.md#the-candidate-slot)); it shares
the walk's owner lists and needs no list plane of its own — and an atom is live where a resolved unit is `valid && live_w` and the
ALIGNED WINDOW of its innermost write slice that the atom covers is non-zero. At the
flagship spans `(8, 16)` a unit IS an atom, both shifts are zero, and the test is "the
unit's 16-bit write slice is non-zero".

The whole enumeration is
`n_candidates = (units[D + 1] >> lat_units_per_atom_shift) << atoms_per_unit_shift` —
a PARTITION mapping monotonically onto ascending atom ids by construction, because unit
rank and the atom index inside a unit are BOTH ascending in the lattice's own leaf order.
That
ordering is not a convenience: the pool claim's canonical ranks ([§4e](#the-claim)) are
defined in ascending atom id and this enumeration is where that order comes from. The same
enumeration serves the table test, the claim and the condensation below.

<a id="atom-bits"></a>**THE CONDENSATION IS PUBLISHED, and it is the only thing that is.**
the test already has the write-atom set in hand, so writing it out costs one byte per atom
and removes the host derivation from a growth step entirely: `plan_exact` plans from
`scratch.atom_bits` and derives no support. One CTA per batch-head (`seg == 0`) publishes
it, UNCONDITIONALLY — a dead atom's `false` is as much of the set as a live one's `true`,
and publishing only the growing batch-heads would leave the buffer carrying an older step's
answer for the others, which is the very buffer the admission then plans from. It is two
passes, a clear over the row and a set over the candidate rectangle, because "everything
outside the write support is false" is a statement about the whole row while the support
is not. The clear is the one remaining term in a steady step that scales with `N`, and it
is one byte per atom on one CTA per batch-head.

The leaf-granular map is published by NOTHING. The factor tables live and die inside the
CTA that built them ([§12](#factor-tables)); the step's one leaf-level output is this
16-deep condensation of the write side's.

`rola.ops.decode._write_atom_bitmap` survives as the ENUMERATED REFERENCE for that
condensation — the same set derived from the Cartesian-product definition in torch, with
the comparison in `tests/integration/test_decode_growth_trigger.py` as its only caller. It
is not a fallback and not a second path a caller may select; it is what makes the gate a
check rather than the kernel agreeing with itself ([§39](#second-derivation)).

## <a id="the-gate"></a>4c. The verdict is PER BATCH-HEAD, and the done flags are what make a replay idempotent

<a id="per-bh-verdict"></a>**WHY A VERDICT AT ALL.** A deposit into an ABSENT atom is
DROPPED ([§4](#absent-atom)) — `if (in_w && atom_present)`. So a host that read the growth
verdict LAZILY (every `K` steps, a pinned poll) would run steps whose writes are silently
lost. Laziness is not an available design at any `K > 1`; the only sound shapes are
per-step synchronization, or a step that DECLINES TO RUN where its writes are not yet
addressable. This is the second, and it is what turns the verdict from a host decision into
a device one — which is in turn what lets the step be enqueued before any host has seen it,
and captured in a graph ([§4d](#capture)).

A NEEDY batch-head — one whose write set it can neither address nor admit — runs nothing:
no walk, no `y`, no state, no counter beyond its arrival. Its verdict lands in `growth[bh]`
and its write-atom set in `atom_bits`, which is exactly what the host admission needs and
all of it.

The skip is PER BATCH-HEAD and not all-or-nothing, and the thing that used to make that
unsafe is the done flags below. A batch that shares one verdict wastes every resident
sequence's step on the rarest one's growth; a batch that skips per sequence must answer
what a replay does to the sequences that already deposited.

<a id="done-flags"></a>**THE DONE FLAGS.** `DecodeScratch.done` is `[BH]` int32, zeroed
ONCE at build. A batch-head that advances sets its flag. On the caller's replay a done
batch-head **walks and reads but does not deposit** — and reproduces its `y` BIT FOR BIT
while doing so, because [update-then-read](#update-then-read) means the rows it would
deposit into already hold the deposit and the same expression is their readout. So `y` is
complete for the whole batch after the replay without `y` ever becoming a buffer that
survives between steps, and the contract the caller sees is:

> gated step -> `growth_pending` -> `admit_growth` -> replay
> advances every sequence EXACTLY ONCE.

`tests/integration/test_decode_growth_trigger.py` states it as `torch.equal` on `y` AND on
the gathered plane against the admit-everything-first order, over a MIXED batch — one
batch-head needy, the rest not — which is the cell that can tell the per-batch-head
contract from the global skip it replaced.

<a id="the-last-arriver"></a>**EVERY CROSS-LAUNCH MUTATION HAPPENS AT ONE POINT, AND THAT
IS THE WHOLE RACE ARGUMENT.** The done flag, the page table's fold ([§4e](#the-claim)) and
the pool cursor are all read by the PROLOGUE of every CTA of a batch-head, and all written
by that batch-head's LAST-ARRIVING CTA — elected from `ctr[bh]`, the split-K counter that
already existed, which the needy path increments too so that the election is the same
statement in both branches. A CTA arrives only after it has finished reading, so no CTA of
a launch can observe a value its own launch wrote. **IT IS ONE WARP THAT DOES THE
WRITING**, not the CTA: the elected CTA's other warps have already retired at the arrival
count ([§29](#the-warp-arrival)), so the page table's fold is lane-strided over the row and
its size folds by a warp shuffle where a block scan used to be — the barrier deletion's
price, paid on the claiming step alone. Without that, a done flag set early
would make a sibling CTA skip its own share of the SAME step, and two CTAs' claims would
disagree about which atoms are free.

<a id="growth-reduction"></a>**THE REDUCTION, AND THE RESET.** The last of those per-batch-head
writers to arrive at `growth_ctr` ORs `growth` into the `[1]` `growth_any` the host reads.
Nothing needy is the same statement as every batch-head done, so that one test also decides
whether the done flags are zeroed for the next step: the flags reset themselves exactly
when the round is over, no `memset` launch stands between two decode steps, and the whole
thing is safe to capture. `growth_ctr` resets itself the same way `ctr` does.

## <a id="the-claim"></a>4e. The slack pool — a growth step that admits itself

Where the state reserves slack ([`paging.md#the-slack-pool`](../paging/paging.md#the-slack-pool)),
a batch-head whose write set is not resident does not have to go to the host at all: it
takes slots from its own pool region and advances. `pool_cap = 0` — the shipped default —
is exact commitment, and the paragraphs below describe machinery that does not run.

**THE ASSIGNMENT IS REPLICATED, NOT COMMUNICATED.** Every CTA of the batch-head computes
the same mapping: the touched-unmapped atoms in ASCENDING ATOM ID take
`pool_slots[bh][cursor + rank]`, with `rank` from a block scan over that batch-head's
atoms. No atomic claims a slot, nothing spins, and nothing is broadcast — which is what
keeps the step one launch and keeps it capturable. It is ALL-OR-NOTHING: a demand larger
than the batch-head's remaining pool is a needy batch-head and the host path exactly
([§4c](#per-bh-verdict)), because a partial claim would deposit part of a step and leave
the replay to deposit the rest twice.

**IT WRITES A SHADOW ROW, NOT THE TABLE, AND THAT IS FORCED.** The claim's predicate is
`page_tbl[a] < 0`. A prologue write into the table would change a sibling CTA's predicate,
which would change its ranks, and two CTAs would then assign the same slot to different
atoms — the ordering ambiguity that makes replicated assignment unsound. So the claim
writes `pool_map[bh]`: a WHOLE copy of that batch-head's table row with the claim merged
in, every atom present, claimed or resident or absent. The walk then addresses through
THAT ROW INSTEAD of the table, by one pointer chosen once in the prologue, and the walk's
own body knows nothing about pools — no extra test, no extra load, no register held across
it. The batch-head's last-arriving CTA folds the row's new entries into the real table and
advances the cursor ([§4c](#the-last-arriver)), so by the next step the claim is ordinary
residency and the shadow row is dead.

**AN ADMITTED ATOM IS AN ADMITTED ATOM.** Pool slots are ordinary arena slots, committed
and ZEROED by the host before they are named, so an atom the device admitted is
indistinguishable from one `plan_exact` admitted: same bytes, same table format, no
migration and no copy anywhere. That is what keeps the dense-vs-paged `torch.equal` gate a
statement about ADDRESSING, and it is asserted directly — a pooled arena and a
host-admitted one reach the same state and the same `y`.

## <a id="capture"></a>4d. CUDA-graph capture — `decode_step_gated`

With no host read inside it the step is capturable, and `rola.engine.dags.decode_dag.decode_step_gated`
is that entry: the same kernel in the same order as `decode_forward` minus the verdict
read. The CALLER owns the verdict, at whatever synchronization it already has (a decode
loop syncs once per token to sample):

```
y, state = decode_step_gated(...)   # or graph.replay()
if growth_pending(state):           # one 4-byte read, at the caller's own sync
    admit_growth(state)             # plan_exact from the step's own condensation
    y, state = decode_step_gated(...)  # or graph.replay(); now it runs
```

**A REPLAY IS SELF-CORRECTING**, which is what makes one captured graph enough: the verdict
is re-derived inside it, so after the admission it is 0 and the same graph performs the
step it had skipped — for the batch-heads that had skipped it, and for no others
([§4c](#done-flags)). Nothing in the loop reallocates — the workspace is built once, every
counter and flag is reset by the last CTA that reads it, and the arena's base address and
page table are objects an admission writes IN PLACE (the reservation never moves), so a
graph captured over them survives growth. The eager `decode_forward` remains THE base
path; capture is an optimization the caller selects, not a fallback either way.

Measured before the absorption (RTX 3080 Ti, medians of 15 x 50, one steady paged step,
`d_v = 64`, `BH = 6`), a loop of steps with one synchronize at the end, against the
admit-preamble-every-step order that preceded the device-side verdict:

| cell | steady, S4b order | steady, verdict order | steady, CAPTURED |
|---|---|---|---|
| 16x16 BC64    | 0.2468 | 0.2046 | **0.0367** |
| 8x8x8x8 BC128 | 0.3254 | 0.2220 | **0.0573** |
| 64x64 BC256   | 0.2811 | 0.2127 | **0.0553** |

The host time the caller's thread spends per steady step falls from 0.259–0.318 ms to
0.008–0.010 ms under capture: the host-wait floor S4b measured (launch-to-visible ~0.05–0.08
ms on this WSL2 host) is not tuned away, it is left with nothing to sit in front of.

Host-side ADMISSION is what the slack pool removes for a growth step that fits it
([§4e](#the-claim)); a step that overruns the pool, or a state with no pool at all,
surfaces to the host exactly as it always did.

## <a id="max-levels"></a>5. `kMaxLevels` — restated, not included

Spec §1 gives `1 <= D <= 4`. `decode.cuh` states that bound itself rather than
inheriting it from a prefill header: it was restated rather than included from the
tiled consumer's `consumer.cuh` because the decode TUs shared no other object with
that kernel and a one-way include would have implied they did. There is no prefill
consumer left to include it from at all today (`DELETIONS.md`), which makes the same
choice load-bearing for a different reason now: the SPEC is the shared authority here,
a header is not, and a header cannot be even where one exists.

## <a id="level-width-capacity"></a>6. The level-width capacity is a GATE, not a convenience

The CEILING below is the producer's; the FLOOR is `rola.ops.decode._admit` (K53), which
refuses a `B_l` below the one admission law's floor through `rola._state.StateFormat`,
the same check prefill's seam runs, before a call reaches this capacity check at all.

`rola/routing/topology.py`'s `MAX_BRANCH_WIDTH = 256` is the widest level the
PRODUCER will emit. A decode path whose capacity were narrower would accept a
topology the producer can build and then read past the end of a staged amplitude
row. So the relation `kDecodeMaxLevelWidth >= kProducerMaxBranchWidth` is
enforced in two halves:

- The **compile-time** half is the `static_assert` in `decode.cuh`;
- The **runtime** half is `rola_decode_capacity()`, bound into the extension so
  `tests/unit/test_decode_capacity.py` can assert the two constants agree.

A C++ assertion cannot read a Python constant. The pair is what makes this
checkable rather than merely written down — and see
[§33](#producer-width-mirror) for why the mirror is exposed separately from the
capacity.

## <a id="decode-threads"></a>7. `kDecodeThreads` — the width IS the arm

The step launches ONE WAVE, SM-aligned: the host takes the declared residency off the
kernel's own launch bound and sizes `n_split` from it ([§23](#the-split-policy)), so the
width is the only constant a shape change touches and `n_split` follows.

* **PRIMARY ARM, `kDecodeThreads = 1024`** — 32 warps, ONE CTA per SM. The per-CTA
  prologue and the ADMISSION are then derived once per SM instead of once per thin CTA,
  split 32 ways.
* **COMPARISON ARM, `kDecodeThreads = 512`** — 16 warps, two CTAs per SM. The same 1024
  resident threads and the same 64-register cap; the two differ only in the single wave's
  tail, because with no barrier below the prologue there is nothing left for a second
  resident CTA to interleave WITH.

At `d_v = 64` a warp is exactly two output columns per lane either way, with 128 B
coalesced row loads — the lane→column map is a property of the warp, not of the CTA.

## <a id="params-amps"></a>9. `DecodeParams` — the operands, as addresses

`read[l]` / `write[l]` / `g_write` / `v` are `Span`s: a base pointer and the `(b, h, w)`
strides of the CALLER'S OWN `[B, 1, H, *]` view. There is no `[BH, b_l]` plane between the
producer and the walk — the step folds these into shared memory itself
([`decode_fold.md`](decode_fold.md)), so the `[B, 1, H, W] -> [BH, W]` fold is an ADDRESS
the kernel computes rather than a pass that ran.

`normalize[l]` says whether read level `l` still carries its own mass, so the fold divides
it by its row sum. `levels_bf16`, `g_bf16` and `v_bf16` name each operand family's storage
type; the fold branches on them once per side, never per element
([`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)).

`level_amp_offset[l]` is where level `l`'s staged amplitude row starts inside the flat
`sum_l b_l` SMEM arrays. It is host-computed and carried rather than recomputed, because
the kernel needs it on the innermost path.

The remaining per-level integers are `level_width[l] = b_l` and
`level_row_offset[l]`, an offset into the flattened per-head dial array. Everything a
LEAF-order walk would need beyond that — a level's stride or span in the flattened leaf
space — has no counterpart here: the walk never addresses a leaf as a flat integer, only
as the `(owner, fixed run offsets, innermost digit)` decomposition the `(k, m)` box's fields carry
([`decode_lattice.md`](decode_lattice.md), and `lat_*` fields on this same struct).

## <a id="detached-clock"></a>10. The detached clock, and why there is no sweep

The clock is the STORED WRITE ALLOCATION ITSELF — `c[t, s] = prod_l p_write[t, d_l(s)]`,
carrying no side gain, because `g_write` scales the deposit and not the clock. It is
therefore not a stream of its own: the step reads the same staged write rows the deposit
reads, and a second fold of one fact would be a second derivation of it.

Its `stored != 0 => live != 0` guarantee is what makes §2.4's no-sweep claim EXACT: a leaf
outside `supp(W)` has a zero factor at some level, so its clock is `0`, so `keep^0 == 1`
exactly and its stored state is unchanged. Decay therefore fires only on written rows —
and written rows are streamed anyway. **There is no entry sweep and no exit sweep in this
kernel.**

## <a id="state-in-place"></a>11. `state` is read and written in place

`state` is the dense `[BH, N, cols]` plane or the arena's `[slots, 16, cols]` one
([§4](#paged-address)), with `cols = d_v + 1` (the mass column is
unconditional — raw is dead), and it is read AND
written in place: a row's read contribution and its write-back are performed by
the SAME lane in program order. The read-then-write ordering the tiled kernel
spends a barrier on per tile costs one register here. The semantics of that order
are [§27](#update-then-read).

`y` is `[BH, d_v]` — the FUSED ratio readout, `num / (den + eps)`.

## <a id="factor-tables"></a>12. The factor tables never leave the CTA that builds them

[§2](#kronecker-support)'s exact support is served by the `(k, m)` lattice's factor tables
— the digit masks and the two owner-coordinate lists
([`decode_lattice.md`](decode_lattice.md)) — built in SHARED MEMORY by the CTA that is
about to walk them. NOTHING PUBLISHES ANYTHING: the step's one leaf-level output is the
write map's 16-deep condensation ([§4b](#atom-bits)), and the tables themselves die with
the CTA.

The walk is COUNTED, NOT SCANNED, in the same sense a leaf-space bitmap walk never was:
`resolve_unit` decomposes a unit index by shifts into an owner rank and the fixed run
offsets directly ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so there is no
prefix-sum structure over the leaf or row space left to consult here at all — the row
count the tiled consumer's `cum`/coarse-index apparatus existed to answer is not a
question this design asks. What replaces it is the owner-list's own realized count `n_o`
per (list, level) — a single small array the block-scan in
[`compact_owner_lists`](decode_lattice.md#the-owner-lists) already produces, consulted
directly at unit-resolve time rather than through a second index over it.

## <a id="splitk-workspace"></a>13. The split-K workspace

`ws` is `[BH, n_split, cols]` and `ctr` is `[BH]`. The workspace is SLOT-ORDERED
and summed by the last-arriving CTA, which is what makes decode's `y`
RUN-REPRODUCIBLE. Why that diverges from the tiled consumer on purpose is
[§16](#determinism).

## <a id="extract-bits"></a>14. `extract_bits` — the funnel shift, written once

Extract `count` bits of a packed bitmap starting at BIT index `start`, returned in
the low `count` bits (`count <= 32`). This is the shift-and-merge every lattice
slice ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)'s
`level_slice`) is read through — an owner's span at any level, one extract — and
it is written once because it is the one place a funnel shift can be off by one.
A SINGLE DIGIT never straddles a word, so the unit's per-level membership tests
go through `mask_bit` (one load, one shift) instead.

<a id="shift-32-ub"></a>The `(start & 31) == 0` fork is **not** an aligned fast path bolted onto a general
one — it is the C++ shift operator's undefined behaviour at a shift width of 32,
avoided at the one site that can reach it (`lo >> 0 | hi << 32`). Both arms
compute the same function. The standing *"no branch to dodge edge cases"* rule is
satisfied because there is ONE strategy here — gather 32 bits from at most two
source words — and the fork is inside it.

---

# Part B — the translation unit (`decode.cu`)

## <a id="gemv-shape"></a>15. Why it is a GEMV, and where the reductions are not

§6b's ratified shape, literally: the token's routing weights and value are
RESIDENT (SMEM/registers), the STATE streams past, and the grid partitions the
lattice's UNIT RANK SPACE ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)).

What makes it a GEMV rather than a GEMM is that there is one token, so the
contraction index is the LEAF and the output index is the COLUMN. That is why
there is no MMA here **and no warp reduction either**: lanes are mapped to OUTPUT
COLUMNS, so each lane accumulates its own partials in registers for the whole row
stream and the dot never crosses a lane.

The only reductions in the file are the 8-way cross-warp fold and the
`n_split`-way cross-CTA combine, both over a small FIXED index set. Both are
fixed-order sums rather than tree primitives because each is `cols` INDEPENDENT
reductions over 8 (or `n_split`) values — a shape no block-reduce primitive has.

## <a id="determinism"></a>16. Determinism is a shipped property here, and it diverges from the tiled consumer on purpose

The tiled consumer emits each token's output as a split-K partial through
`atomicAdd` into `num`/`den`, so its `y` is not reproducible against its own
binary wherever a `bh` holds more than one owner. **Measured: 384 of 976 fixture
cases differ between two runs of one binary.**

Prefill's `y` feeds a loss or an eval aggregate, and that is tolerable. Decode's
`y` feeds an argmax or a sample that picks the NEXT TOKEN, so a last-bit
difference in `den` flips a near-tie logit and the divergence compounds
autoregressively for the rest of the generation.

At `T = 1` the partials are `n_split * cols` floats per `bh`, so the fixed-order
combine costs one workspace slot per CTA, one `__threadfence()` and one
`atomicAdd` on an INT counter — an atomic that defines the SCHEDULE, not the
arithmetic. See [§30](#slot-order).

The STATE is unconditionally reproducible for a stronger reason: the row
partition is a PARTITION, so every written row has exactly one owning CTA, warp
and lane, and no atomic touches `state` at all.

## <a id="no-barrier"></a>17. No barrier per LEAF, and that is the axis that matters

The fold and factor-table prologue costs a FIXED number of barriers and the rest of
the step costs NONE. Four are unconditional — the read side's row sums, both sides'
rows/value/dials, the digit masks and the zeroed counts, the owner liveness bitmasks —
then one closing [`publish_term_layout`](decode_lattice.md#the-owner-lists); the
admission adds its own where it claims, and the last of them publishes the claim row
and the zeroed warp arrival count to everything below. The per-(list, level) barriers
the compacted owner lists needed are gone with the block scans
([2.1](decode_lattice.md#the-owner-lists)), so the count no longer grows with `D` at
all — and it never grew with `N` or with a leaf-space bitmap's word count, which is
the axis the tiled consumer's per-candidate-tile barriers scaled on.

**BELOW THE PROLOGUE THERE IS NOT ONE.** The WALK never needed one: a lane's read
contribution and its write-back are one statement in program order, and the ordering
is UPDATE-THEN-READ because §2's survival is inclusive at `j = t`
([§27](#update-then-read)). The cross-warp fold and the split-K hand-off used to need
five between them; they are now an ARRIVAL COUNT ([§29](#the-warp-arrival)).

## <a id="decode-max-exponent"></a>18. `kDecodeMaxExponent` — the same convention, structurally inert here

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as the
tiled consumer's `intra_panel.cuh` carried in `kPanelMaxExponent` — restated here
rather than included, because that header was that kernel's panel algebra and
decode shares none of it. The header went with the arm (`DELETIONS.md`); the
convention is the spec's rule 3, and this file states it for itself.

On this path the clamp is **structurally inert**, and that is worth saying rather
than leaving a reader to wonder: the exponent is `clock * log1pf(-rate)` with
`clock >= 0` and `rate in (0, 1)`, so `log1pf(-rate) < 0` and the exponent is
non-positive. `expf` can underflow to zero (the correct limit) but cannot
overflow.

It is written anyway so that ONE convention governs every survival exponential in
the repository, and so a future decay form that CAN go positive finds the clamp
already in place. `tests/unit/test_decode_capacity.py` pins the two constants
equal.

## <a id="template-axes"></a>20. Why there is no `GLOBAL` axis, and what `NC` is

Raw normalization is dead everywhere in this repository (revival = git;
[`DELETIONS.md`](../DELETIONS.md)): the mass column and the `num/den` divide are
UNCONDITIONAL, so `cols = DV + 1` at every instantiation and there is nothing left
for a `GLOBAL` template axis to select between. `decode_step_kernel<DV, D,
DECAY>` takes exactly three template parameters.

`NC = DV / 32` is columns per lane. The row load is `NC` fully coalesced 128 B
transactions and the lane's accumulators are its own output columns — the whole
reason no reduction is needed on the dot ([§15](#gemv-shape)).

## <a id="smem-ledger"></a>21. The SMEM ledger, and the fold's `+3` pitch

| block | extent | what it holds |
|---|---|---|
| `sa_r`, `sa_w` | `[total_rows]` fp32 | both sides' amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `sd` | `[total_rows]` fp32 | this head's dial row; `DECAY` only |
| `v_tok` | `[DV]` fp32 | the token's value, resident |
| `fold` | `[warps][cols + 3]` fp32 | the cross-warp partials |
| `mask_r`, `mask_w` | `[mask_total]` u32 | one bit per digit, per side ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)) |
| `omask` | `[4][omask_total]` u32 | the four KINDS' owner LIVENESS BITS, one per owner coordinate, per level — the BUILDER of the two lists below ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists)) |
| `olist` | `[4][olist_total]` i32 | those bits compacted: the i-th live owner, one indexed load |
| `dlist` | `[dlist_total]` i32 | the WRITE side's live digits, OUTER levels only — term 0's own grain ([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)) |

Every one of them is FILLED BY THIS CTA, from the raw operands and from nothing else: the
step allocates no plane, reads no `[BH, b_l]` intermediate and publishes no bitmap. That is
the whole of [§21b](#per-cta-prologue).

The amplitude rows are indexed by DIGIT and are not compacted: **the digit mask
carries membership, so the digit IS the index**, and no compaction, id list or
flag packing exists on this path.

<a id="fold-bank-padding"></a>The `+3` pads the fold's row stride to `cols + 3` so the eight warps' stores land
on distinct banks. The fold reads DOWN the warp axis, so an unpadded stride
sharing a factor with 32 would collide. `gcd(cols + 3, 32)` is 4 at `cols = 65`
and 1 at `cols = 32` and at `cols = 64 + 3`.

## <a id="per-cta-prologue"></a>21b. The prologue is per-CTA, and that is the design

The step is one kernel with three phases — fold, factor, walk — and the grid is
`(n_split, BH)`, so the first two run once per SPLIT-K CTA rather than once per batch-head.
Every CTA re-derives the same amplitude rows and the same factor tables for its `bh`, then
walks only its own share of the unit rank space.

That redundancy is the price of two properties:

- **The factor tables never reach DRAM.** A shared per-`bh` table would have to be written
  by someone and read by everyone, which is a global round trip on the walk's hottest
  index — or a launch boundary to order the write against the reads, which is the launch
  this design exists to delete.
- **The split partition stays.** Collapsing the grid to one CTA per batch-head
  (`n_split = 1`) costs no recompute at all, but the walk is row-starved without the
  split — a topology's `N` is exactly what decides how starved, so there is no regime
  where collapsing the grid is the better shape.

**Nothing in the recompute's fixed cost is a term in `N`.** Building both masks is
`sum_l ceil(B_l/32)` words per side; building both owner lists is a block scan per
(list, level), `D` levels, over an array at most `lat_g[l] <= kDecodeThreads` wide
([`decode_lattice.md`](decode_lattice.md)) — bounded by the LEVEL WIDTHS and the OWNER-GRID
EXTENTS, never by the leaf space and never by the state, which is why paying it `n_split`
times still lands ahead of paying one extra launch once. This is a structural property of
the walk itself, not a tuning outcome: the leaf-space bitmap this design replaced (its
construction, its cost and its removal are recorded in `DELETIONS.md`) DID carry a term in
`N`, which is the gap this design exists to close.

<a id="the-split-policy"></a>**The split is ONE WAVE of the declared residency.** Because
the prologue is per-CTA, a grid deeper than the machine can hold at once re-pays that
whole prologue in a second wave for work the first wave could have finished, and the
`beyond-realized CTAs exit early` argument does not cover it: a CTA cannot exit before it
knows its own range, and knowing its range means running the prologue. So
`rola.ops.decode.derive_n_split` takes `SMs * residency / BH`, with the residency read
back off the launch bound itself
([`decode_api.md` §2](decode_api.md#residency)), bounded below by the UNIT SUPPLY so a
small topology does not launch CTAs there is no work for. Both bounds are static facts —
an occupancy fact and a geometry fact.

<a id="retirement"></a>**THE CEILING IS WHAT THE HOST CAPTURES; THE DEVICE IS WHAT SPENDS
IT.** `n_split` is derived from the DENSE unit ceiling because that is the only count a
host can know — the count a step actually enumerates is a function of ITS OWN SUPPORT and
exists only on the device, once the factor tables do
([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)). Under the
union-of-products enumeration the two diverge by up to 11x, and at small `BH` — where
`n_split` is largest — the ceiling over-launches by 5x. So the split policy stays exactly
what it was, frozen for the sequence and baked into the captured launch, and the SEGMENTS
RETIRE:

* every CTA derives `n_live = clamp(ceil(max_t units[t] / kDecodeWarps), 1, n_split)` from
  the counts its own prologue produced — the walk's stride is `n_split * kDecodeWarps`, so
  the segments with an iteration are exactly the PREFIX `[0, n_live)`;
* a segment at or past it returns THERE, before the admission question — no walk, no
  partial, no arrival, and it is not summed;
* and it returns before the admission on purpose: the table test and the pool claim read a
  page table the last-arriving live segment folds into ([§4e](#the-claim)), so a retired
  segment must not be inside that window at all.

**IT IS THE ARRIVAL AND THE COMBINE THAT RETIREMENT ACTUALLY BUYS, NOT THE PROLOGUE**, and
that was measured rather than assumed. A variant that retired the walk but still wrote
exact zeros and arrived bought NOTHING; the per-segment cost a large split carries is
`n_split` atomics on one counter followed by `n_split` `__ldcg` loads per column on ONE
CTA ([§30](#slot-order)). The measured effect of retiring completely, at `BH = 1` where
`n_split = 256`, is 10-12 % off every sparse cell.

**AND IT IS NOT A THRESHOLD.** `n_live` is the walk's own loop bound, asked before the
loop. It is also why a PER-BATCH-HEAD PRE-PASS is not shipped: publishing the factor
tables and the counts from a `BH`-CTA launch so segments could retire before folding
anything was built, gated green and measured, and it LOST — at `BH = 1` the pre-pass is a
single CTA whose whole latency is exposed, and it cost 3-7 us against a saving the
in-kernel retirement already collects.

## <a id="row-scales"></a>22. `sr` and `sw` — the two per-token row scales

At `T = 1` there is no owner rectangle, so the per-LEVEL factors the tiled kernel
hoists into `sr`/`sw` for the fixed levels `l < j` stay in the per-LEAF product
instead. The split between `R[0,s]` and `sr` is an owner-rectangle artefact, and
the product is the same product either way. What remains here is exactly the
per-TOKEN part: the side gains. There is no read-side gain at all: the ratio
readout fixes it to 1 identically, in every arm.

## <a id="aligned-subranges"></a>26. The unit's leaves are walked in ascending order, over the innermost run's set bits

<a id="the-term-loop"></a>**THE WALK'S OUTER LOOP IS THE TERM**, over the `D + 1`
products whose disjoint union is the set the step touches
([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)), and each term's index
space is partitioned across the grid on its own. That is a LATENCY decision, not
bookkeeping: inside the term loop every level's KIND is loop-invariant, so the owner-list
bases, the realized counts, the padded field widths and the membership predicates all
hoist out of the unit loop and the per-unit dependent chain is the one the
product-of-unions walk had. The alternative — laying the terms end to end in ONE index
space and naming each unit's term by a scan over `D + 1` offsets — was built and measured
at **+2.3 ns of dependent latency per unit**, which on the scattered cell cost more than
the units the decomposition saves. The loop is not unrolled over `D`, so one copy of the
walk body serves every term.

Term 0 IS the write product, and it is the ONE term whose rank space is not
`live owners x the full run radix`: under the split-level declaration its outer support is
sparse INSIDE the owners it touches, so that space enumerated 2048 units for 256 live ones
at the flagship's scattered cell — and the dead ones, interleaved one-in-eight, idled seven
of every CTA's eight warps. It enumerates LIVE DIGITS instead
([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)). The
CANDIDATE-ATOM enumeration ([§4d](#candidate-atoms)) does NOT follow it there: a flat digit
rank is not monotone in atom id, and the pool claim's canonical ranks are defined in that
order, so a term slot `D + 1` holds the write product in the old space and the candidates
read that. Two enumerations of one set, each in the order its consumer needs.

<a id="the-unit-partition"></a>The grid partitions each term's UNIT INDEX SPACE directly — `for
(unit = seg*kDecodeWarps + warp; unit < units[term]; unit += n_split*kDecodeWarps)` — so there
is no separate row-range-to-word resolution step: a unit IS the warp's own share, by
construction, and `resolve_unit`
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) is what turns a unit index
into its `(owner, r_0 .. r_{D-2})` pair before any leaf inside it is touched. Every
touched leaf therefore has exactly one owning CTA, warp and lane — the unit partition is a
PARTITION, so no atomic ever touches `state` ([§16](#determinism)).

**A WARP TAKES A WHOLE UNIT, NOT A WHOLE OWNER, AND THAT IS A MEASURED CHOICE.** A unit is
already the DRAM granule: under the mixed-radix order its leaves are the innermost level's
contiguous run, which at the flagship spans is one 16-leaf ATOM, so owner-contiguity
across units buys nothing on top of it — the sectors are whole either way. Partitioning at
OWNER grain instead caps the walk's parallelism at the LIVE OWNER COUNT and hands each
warp the entire `BC`-leaf box to walk serially; measured on the wide sparse arm that cost
roughly a factor of two, and at `N = 256` (two live owners against eight warps) rather
more. The CTA's warps take CONSECUTIVE units, so a CTA still covers one contiguous stretch
of the plane per pass.

Within one resolved unit the walk visits the SET BITS of THAT TERM'S slice — `cw` for the
write product, `cr` for a read-remainder term, already conditioned by `resolve_unit`
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — ascending, by `__ffs` — ascending bit index is ascending LEAF order, because the unit's
leaves ARE the innermost level's run in order. A dead innermost digit costs one `__ffs`
step and no address at all, and the two sides' membership comes off the already-resolved
`cr`/`cw` bitfields with no division anywhere. That ascending order is what makes
`cached_atom` — the paged base translation's one piece of reuse ([§4](#paged-address)) —
correct and cheap at once: at the flagship spans the whole unit is ONE atom, so the
translation is resolved once per unit.

The trip count per unit is `popcount(cr | cw) <= s_{D-1} <= kDecodeMaxSpan`
([`decode_lattice.md#capacities`](decode_lattice.md#capacities)) — a small,
compile-time-boundable cost per LIVE unit, never a cost that varies with `N`.

EVERY LEVEL BUT THE INNERMOST LEAVES THIS LOOP. A unit fixes their digits, so their
amplitude product (and the decay rate's) is formed ONCE in `resolve_unit` and a slot costs
ONE innermost amplitude rather than `D` of them.

A leaf outside the lattice's own partition of `N` is never visited: `owners * BC == N`
exactly (`decode_lattice.md#the-refusals`), so the walk needs no leaf bound test and never
fabricates a row.

## <a id="update-then-read"></a>27. Update first, then read the updated row — and that order IS the semantics

Not an optimization. §2's survival is INCLUSIVE at `j = t`: the token's own
deposit is read by the token itself with survival exactly `1`.

The tiled consumer expresses that as TWO terms, because at `BT > 1` the deposit
and the readout live in different GEMMs: the readout against the decayed entry
state, plus the causal intra Gram's self term `G[0,0] * Vw`. At `T = 1` those two
are

```
a_r * keep^c * M_entry  +  a_r * a_w * sw * v   ==   a_r * S_exit
```

so **the whole intra Gram collapses into reading the row after it is written**.
One pass over the row, one register, and the causal structure is exact rather than
reconstructed.

A row in `supp(R) \ supp(W)` has `keep == 1` and no deposit, so
`S_exit == M_entry` and the same expression is its readout too — one path, no arm.

Both off-by-ones — decaying after the deposit, and reading before it — are `O(1)`
errors that G4/G6 discriminate at the fp32 tolerance.

## <a id="decay-at-read"></a>28. Decay at read, via the detached clock, with no sweep and no exponent split

At `T = 1` the tiled kernel's `GLA_MASS_DECAY_SPEC.md` §12 rule-1 half-mass shift
is exactly the IDENTITY. With `P_ref = P[0]/2`:

```
readout:  fr * s0                 collapses to   keep^c
fold:     s0 * (s0 M + W fw Vw)   collapses to   keep^c M + W Vw
```

So on this path there is no clock prefix buffer, no `p_ref`, no `s0`, no division
by a cumulative survival, and no `fw` underflow class — **ONE `expf` where the
tiled kernel evaluates three.**

`log1pf` IS the composition (never a rounded linear-domain `1 - rate`) and there
is no epsilon floor: both of §12 rule 2's prohibitions, inherited verbatim.

The mass column is the `v := 1` column of the identical recurrence, unconditional
on every arm — one lane-0 statement, never a second path.

## <a id="fold-and-sr"></a>29. <a id="the-warp-arrival"></a>The cross-warp fold is an ARRIVAL COUNT, and where `sr` is applied

Every warp of a live CTA deposits its `cols` partials into its OWN row of `fold`,
fences its own stores, and bumps a shared arrival count with one atomic; the warp that
reads back `kDecodeWarps - 1` is the last arriver, and it alone sums the rows. No
barrier is involved and none is available: the count itself is the proof that every row
is written. Its zero is published by the PROLOGUE's last barrier — shared memory is not
zeroed at CTA start, and that is the one place a zero costs nothing.

**DETERMINISM SURVIVES BECAUSE THE ORDER IS FIXED, NOT BECAUSE THE ARRIVALS ARE.** The
sum runs over warps `0 .. kDecodeWarps - 1` per column, which is exactly the order the
barrier-stepped tree it replaces used, so `y` is byte-reproducible at a fixed `n_split`
as before ([§30](#slot-order)).

**A WARP THAT WALKS NOTHING STILL DEPOSITS AND ARRIVES.** Retirement at warp grain is
the walk a warp does not run — its accumulators are exact zeros and adding them changes
no bit — never an early `return`, which would leave the count short. Retirement at CTA
grain is unchanged and still returns before the admission ([§22](#retirement)).

**EVERY LANE FENCES ITS OWN ACCESSES**, on both sides of the count: a fence orders only
the fencing thread, so a leader-only fence before a one-lane atomic would cover one lane
of thirty-two (`KERNEL_STANDARDS.md` §12).

**AND THE PAIRING IS PROVEN IN THE PTX MEMORY MODEL, WHICH IS NOT WHAT THE SANITIZER
CHECKS.** The emitted PTX is `membar.cta` / `bar.warp.sync` / `atom.shared.add.u32` /
`membar.cta`. A fence followed in program order by a strong write on the counter is a
RELEASE PATTERN on it, and a strong read of the counter followed by a fence is an
ACQUIRE PATTERN (PTX ISA §8.8, whose own examples are `fence.release; atom.relaxed [M];`
and `atom.relaxed [M]; fence.acquire;`). Every warp's arrival precedes the last
arriver's in OBSERVATION ORDER through the counter's atomic RMW chain (§8.9.2), which is
what makes release synchronize with acquire (§8.9.4); the lanes that are not lane 0 reach
the release through `bar.warp.sync`'s own memory ordering (§9.7.14.2). Two consequences
are load-bearing and easy to lose: the arrival MUST keep using its return value, because
a fire-and-forget `red` forms no acquire pattern at all (§8.11.1), and the fences must
stay on EVERY lane. `compute-sanitizer racecheck` reports 48 RAW hazards here regardless
— it models barriers, `__syncwarp`, `cuda::barrier`, cluster barriers and async-copy
waits, and a fence plus an atomic is none of those. That is adjudicated, with its proof
and its exit condition, by `tools/sanitize_oracle.py`'s `ASSERTED`; the measurements
behind it are in FINDINGS `## K47`.

`sr` is applied by the summing warp, before the split-K store, exactly as the tiled
kernel emits `sr * y` into `num` and `sr * den_acc` into `den`. That makes the combine
below a plain sum, and keeps the two consumers' dividend and divisor the same objects.

## <a id="slot-order"></a>30. The deterministic split-K combine, over the LIVE segments

SLOT ORDER, always. This is the whole determinism argument: the addends are summed in a
FIXED order, which gives one answer per binary where an `atomicAdd` race gives one answer
per arrival order.

THE RANGE IS `n_live`, NOT `n_split` ([§21](#retirement)): retired segments write no
partial and take no arrival, so both the arrival target and the summed slot range are
`[0, n_live)`. Every CTA of the batch-head derives that bound identically from the same
published unit counts, so nothing is communicated and nothing is ordered — and the sum is
still ASCENDING SLOT ORDER over a range that is a function of the inputs alone, which is
exactly what G1 asserts (`y` byte-reproducible at a FIXED `n_split`). `y` still moves
across `n_split`, as it always did, because fp32 addition is not associative.

`__ldcg` bypasses L1 so the reader sees the other CTAs' stores that the
`__threadfence` above made visible.

`n_live == 1` takes an early return before any of this — not a special case, the
same expression with zero combine steps. A batch-head whose support is empty still owes a
`y`, which is why `n_live` is clamped up to one and segment 0 always runs: `y` is then the
ratio `0 / (0 + eps)` it writes.

The counter is reset to `0` by the last CTA, so no `memset` launch stands between
two decode steps and the counter is safe to capture in a CUDA graph.

## <a id="declared-residency"></a>30b. The residency is DECLARED, and it is the whole
perf story of the fusion

`decode_step_kernel` carries
`__launch_bounds__(kDecodeThreads, decode_step_blocks_per_sm(DECAY, D))`, and that second
argument is not decoration: it is a REGISTER CAP the assembler must meet, and the census
is what proves it met it without spilling. `decode_step_blocks_per_sm` is
`(decay || levels >= kMaxLevels) ? 4 : 5` — TWO TEMPLATE AXES move the declared residency,
each by exactly one CTA:

- **DECAY** carries a second per-leaf product (the decay rate) and a survival
  exponential through the same loop as the undecayed arm's amplitude product;
- **DEEPEST** (`D == kMaxLevels == 4`) resolves one more level per unit — one more owner
  coordinate, one more fixed digit and one more amplitude factor folded into `Unit<D>`
  ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — held live across the
  same loop.

The measured census is what settles both numbers, and it did move once in wave 2c: an
earlier draft of the unit carried a RUNTIME loop level between the unit and its slots, and
spilled 16–30 stores at `D = 2` and `D = 3` against the 48-register cap the `5` implies.
Collapsing the unit to the innermost run — one shape, uniformly
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — returned every arm to zero
spill at the SAME declared residency, which is the outcome the *"ideal regime is not
negotiable"* rule asks for: the budget was not the thing to relax.

Both are compile-time functions of a template axis rather than a threshold read off a
measurement, which is what the standing *"branch on structure, not thresholds"* rule asks
for: a config that is DECAY or `D == 4` gets one fewer CTA/SM, unconditionally, and every
other config gets the wider budget.

**The claim's SIZE comes out of the fold that applies it** — an atom the shadow row maps
and the table does not IS a claimed atom ([§4e](#the-claim)) — rather than being carried
across the walk or re-derived from the write-support enumeration afterwards. Both
alternatives hold state alive across the loop the register budget is tight in; counting it
where the fold already touches every atom holds nothing at all.

## <a id="instantiation-matrix"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) = 16`. Note what is **not** an axis: no `BT` (one token), no `BC`
(no owner-blocked amortization — [§3](#bc-absent)), no `MmaPrec`/precision split (there is
no MMA, and the tf32x3/bf16 split died with the `PREC` axis everywhere else —
`DELETIONS.md`), no `GLOBAL` (raw normalization is dead; the mass column is unconditional
— [§20](#template-axes)), no OPERAND STORAGE TYPE (the fold reads three independent dtype
families as runtime flags selecting a typed loop instead — a template axis there would
multiply the matrix by eight for a few hundred bytes of gather,
[`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)), and — the axis this rung
removed — no ADDRESS-SPACE arm: the factor tables are shared memory unconditionally, at
every shipped topology, so there is no second arm for a template parameter to select
between. `d_v` is restricted to `{32, 64}` by a `TORCH_CHECK`, and `D` outside `[1, 4]` is
refused in the dispatch's `default`.

## <a id="producer-width-mirror"></a>33. Why the producer-width mirror is bound separately

`rola_decode_producer_width_mirror()` exposes `kProducerMaxBranchWidth` separately
from the capacity because the two are different claims:

- `capacity >= producer` is the SAFETY property, and the `static_assert`
  ([§6](#level-width-capacity)) covers it;
- `mirror == the real Python constant` is what stops that assertion from being
  satisfied by a STALE mirror after the producer's limit moves.

Only a test that can see both languages can check the second, which is why this is
bound at all.

## <a id="empty-not-zeros"></a>34. `y` is `at::empty`, not `at::zeros`

Every element is written by the `n_split == 1` CTA or by the last-arriving CTA of
the combine, so a zero-fill would be a launch that the launch-latency-bound
small-batch regime pays for nothing — and an unwritten element showing as garbage
rather than as a plausible zero is what a gate wants.

## <a id="smem-refusal"></a>35. The staged-amplitude SMEM budget is a declared capacity

The host entry queries `cudaDevAttrMaxSharedMemoryPerBlock` and `TORCH_CHECK`s
`decode_step_smem_bytes(...)` against it, naming `sum_l width_l` in the message. That
function's total is the staged amplitude rows, the dial row, the token value, the
cross-warp fold buffer and BOTH factor-table blocks (the digit masks, the owner liveness
bitmasks and the two lists they compact into —
`2 * mask_total + 4 * omask_total + 4 * olist_total + dlist_total` words,
[`decode_lattice.md`](decode_lattice.md)):
everything a step's prologue puts in shared memory, all of it. It is **refused loudly
rather than taken as a slower path** — there is no spill-to-global arm for any of it, and
there is no second address space for the factor tables to fall back to either.

---

# Part C — the host side (`rola/ops/decode.py`, `rola/engine/dags/decode_dag.py`)

The step's ORDER, the two memo disciplines and the capturability declaration are the
DAG's page ([`engine/decode_dag.md`](../engine/decode_dag.md)); what follows is what the
KERNELS impose on the host.

## <a id="decay-domain"></a>36. Decode's decay domain is strictly larger, as a consequence

The tiled consumer refused a routing whose per-leaf reference exponent exceeded
`PANEL_MAX_EXPONENT`, because its rule-1 lowering exponentiated `keep^{+P}` and
`keep^{-P}` separately and their product is wrong — not merely rounded — once one
saturates while the other underflows. At `T = 1` that rule-1 half-mass shift is EXACTLY
the identity: the readout's `fr * s0` collapses to `keep^c`, and the fold's
`s0 * (s0 M + W fw Vw)` to `keep^c M + W Vw`. So decode forms ONE exponential, never a
quotient of two, and has no admissibility precondition to check — which is why it does
not import one.

## <a id="forward-only"></a>37. Forward only, and the guard is defence in depth

The decode kernels have no backward. The layer's decode envelope check already
refuses training mode and gradient reachability before the decode fork is reached, so
the module's own check exists for a direct caller.

## <a id="verdict-read"></a>38. The verdict's read is ONE BLOCKING pinned copy, and it cannot be anything else

The step publishes `growth_any` at the end of its own launch, so the eager path's read is
behind it BY DATA DEPENDENCE and there is nothing left to enqueue it in front of: the step
is the only launch either way. It is one `[1]` int32 into a workspace-owned PINNED buffer,
copied blocking. That was already the measured choice for a read standing in front of a
kernel (0.093 ms against 0.152 for an asynchronous copy plus an event, RTX 3090, medians of
15 × 100 — the host must wait for the kernel either way, and the event adds a record and a
query on top of the wait the copy already performs), and with the ordering forced there is
no second shape to compare.

Under capture the read leaves the step entirely — the caller performs it, at its own
synchronization ([§4d](#capture)) — which is where the host's participation in a steady
step actually goes.

## <a id="second-derivation"></a>39. `_write_atom_bitmap` is the gate's independent second derivation

The shipped path derives the write-atom set on the device (the step publishes it), so
`_write_atom_bitmap` has no production caller and is not meant to acquire one. It
survives deliberately as a SECOND derivation of an object the kernel computes, written
from the definition rather than from the kernel's structure, so the equality
`tests/integration/test_decode_growth_trigger.py` asserts is a real check and not the
kernel agreeing with itself. The decode paging and VMM gates plan from it for the same
reason: a fixture using the kernel's own answer could not catch the kernel.

The derivation, from the definition: at `T = 1` a leaf is written iff every level's write
digit is nonzero, so the write support is the Cartesian product of the per-level
exact-nonzero indicators and the atom set is that product ORed 16-deep. It is the same
object `rola.engine.facts.planes.atom_bits` derives from the facts pass and the same one
the step condenses, reached by neither's route.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kdecodemaxexponent"></a>
### `kDecodeMaxExponent`

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
`intra_panel.cuh`'s `kPanelMaxExponent`, restated because decode shares none of that
header's panel algebra. On this path the clamp is STRUCTURALLY INERT -- the exponent
cannot go positive -- and is written anyway so ONE convention governs every survival
exponential in the repository: docs/internals/decode/decode.md#decode-max-exponent

<a id="decode-stamp-kernel-note-l63"></a>
### in `decode_stamp_kernel`, near line 63

THE ENUMERATION'S SHAPE AND THE LAUNCH'S WIDTH ARE PART OF THE FOOTPRINT, alongside
the two struct sizes and the span ceiling. The kind and term counts are what a walk's
unit index MEANS, and the warp count is HOW MANY CTAs derive the step -- and a binary
that disagrees about either is a different step however its structs measure. Both
lessons were paid for: the union-of-products enumeration moved every walk in the
extension without moving one byte of either struct, and the fat-CTA launch shape moved
the derivation count from 256 to 80 without moving the enumeration. IT IS THE MOST
SIGNIFICANT FIELD of the same positional scheme, so the three older fields still
divide out unchanged and the inversion's plausibility bound still holds
(`kKinds * kMaxTerms * kDecodeWarps` is 192 at eight warps, 768 at thirty-two).
THE PARTITION'S GRANULARITY RIDES THE SAME FIELD, additively, because a binary that
cuts the unit space at a different block width walks a different step however its
width and its structs measure -- the round-4 lesson applied to the second of the two
knobs the launch shape now carries:
docs/internals/decode/decode_api.md#build-stamp

<a id="decode-step-kernel-note-l137"></a>
### in `decode_step_kernel`, near line 137

THE TERM TABLE, ONE PACKED WORD PER (term, level): the field's position, its width and
the level's realized count in one read where three arrays used to be read separately.
`D + 2` rows -- the walk's `D + 1` terms and the admission's candidate slot:
docs/internals/decode/decode_lattice.md#the-terms

<a id="decode-step-kernel-note-l151"></a>
### in `decode_step_kernel`, near line 151

TWO BARRIERS FOR THE WHOLE PROLOGUE. The read side's row sums are the only thing
anything downstream waits on, so they take the first; everything else -- both sides'
rows, the value, the dials -- is independent and shares the second.
docs/internals/decode/decode_fold.md

<a id="sr"></a>
### `sr`

The two per-token row scales. At `T = 1` there is no owner rectangle, so what
remains here is exactly the per-TOKEN part -- the side gains; the per-LEVEL factors
the tiled kernel hoists stay in the per-LEAF product.
docs/internals/decode/decode.md#row-scales

<a id="umax"></a>
### `umax`

`n_split` is FROZEN for the sequence and baked into the captured launch
([`decode.md#the-split-policy`](#the-split-policy)), and it is sized from the DENSE
unit ceiling because that is the only count a host can know. The count this step
ACTUALLY enumerates is a property of ITS OWN SUPPORT and exists only here, once the
factor tables do. So the ceiling is what the host captures and the DEVICE is what
spends it: a segment with no unit in any term retires, and it retires COMPLETELY --
no walk, no partial, no arrival, and it is not summed.
THE LIVE SEGMENTS ARE A PREFIX. A segment owns the `kUnitBlockWarps`-wide block at
`seg * kUnitBlockWarps` of every period, so it has an iteration exactly when
`seg * kUnitBlockWarps < max_t units[t]`,
and `n_live` is that bound clamped into `[1, n_split]`. Every CTA of the batch-head
derives it from the same published counts, so the arrival target and the summed slot
range agree by construction and no CTA has to be told.
IT IS NOT A THRESHOLD -- it is the walk's own loop bound, asked before the loop.
AT LEAST ONE, ALWAYS: a batch-head whose support is empty still owes a `y`, and `y`
is the ratio `0 / (0 + eps)` that segment 0 writes.
IT RETIRES BEFORE THE ADMISSION, and that ordering is load-bearing rather than
incidental: the table test and the pool claim read a page table the last-arriving
LIVE segment folds into, so a retired segment must not be inside that window at all.
docs/internals/decode/decode.md#retirement
ONE SHARED READ AND ONE COMPARE. `publish_term_layout` folds the maximum by `atomicMax`
inside its own barrier, so learning it costs neither a second block synchronisation
(which the dense cell, where nothing retires, measured at ~1 %) nor `D + 1` redundant
reads per thread. The test is written as the COMPARE rather than as `seg >= n_live`
because they are the same question -- `seg >= clamp(ceil(umax / U), 1, n_split)` is
`seg > 0 && seg * U >= umax` at `U = kUnitBlockWarps`, since `seg < n_split` always --
and this form keeps the
division and the two clamps off the path every CTA walks, on the cell where nothing
retires. `n_live` itself is only needed by the epilogue.

<a id="win-bits"></a>
### `win_bits`

THE CANDIDATE ATOMS, enumerated from the same lattice product the walk uses, and a
PARTITION of the write side's atoms with no straddle case at all -- which is what the
pool claim's canonical ranks need. A unit's leaves are a CONTIGUOUS ALIGNED run, so an
atom and a unit nest one way or the other and the index says which: the low
`atoms_per_unit_shift` bits pick an atom INSIDE a unit, and `lat_units_per_atom_shift`
consecutive units make ONE atom. Exactly one of the two is non-zero.
THE WINDOW IS THE UNIT'S RUN, CLIPPED TO AN ATOM. At the flagship spans `(8, 16)` a
unit IS an atom: `n_units` and the atom index are both one, and membership is the
unit's write slice being non-zero.
IT NEEDS NO SECOND ENUMERATION AT ALL, because TERM 0 IS THE WRITE PRODUCT: the walk's
own first term is the write side, in its own order, with `live_w`/`cw` the
conditioning that term carries. docs/internals/decode/decode.md#candidate-atoms

<a id="paged"></a>
### `paged`

Three outcomes PER BATCH-HEAD: advance, no-op, or re-read without depositing:
docs/internals/decode/decode.md#absorbed-verdict
NO 64-BIT ADDRESS MAY SURVIVE INTO THE WALK -- each place that needs this
batch-head's table row forms it again, at two registers of the declared residency:
docs/internals/decode/decode.md#declared-residency

<a id="a"></a>
### `a`

THE TABLE TEST: how many atoms this batch-head's write side reaches that the page
table does not map. Every CTA of the batch-head computes it independently from the
same two objects -- its own factor tables and a table no launch mutates until
every CTA of that batch-head has arrived -- so the answer is a RECOMPUTE and never
a communication. docs/internals/decode/decode.md#the-table-test

<a id="syncthreads"></a>
### `__syncthreads`

THE CLAIM, replicated and canonical: the batch-head's touched-unmapped atoms
take pool slots in ASCENDING ATOM ID from its own cursor, so every CTA of the
batch-head derives the identical mapping with no atomic and no arrival.

IT WRITES THE CLAIMS AND NOTHING ELSE. Not the page table, because the table is
the claim's own predicate and a prologue write would move a sibling CTA's ranks;
and not a whole shadow ROW either, which is the defect this deletes.
THE ROW-COPY WAS A CROSS-CTA RACE. Every CTA of the batch-head copied the table
into this row -- writing `-1` for every atom the table does not map yet -- while
its siblings were already writing their claimed SLOTS into the same addresses,
with no ordering between CTAs at all. The barrier below orders one CTA's copy
against its own claim writes and says nothing about anyone else's: a sibling's
`-1` could land on top of a claimed slot, and a third CTA reading that address
in the window took the ABSENT path and skipped its deposit. The final value was
always right, which is what the old comment's "identical mapping" was true of;
the TRANSIENT was not, and correctness that survives only because one wave of
CTAs happens to march in step is not correctness.
SO THE ROW IS CLAIMS-ONLY, every write is the same value from every CTA, and the
walk reads TWO SOURCES instead of one (`decode.md#the-claim`). It also deletes an
`atoms_per_bh`-element bulk copy per CTA per claiming step.

<a id="decode-step-kernel-note-l355"></a>
### in `decode_step_kernel`, near line 355

THE CONDENSATION, published UNCONDITIONALLY by one CTA per batch-head: a dead
atom's `false` is as much of the set as a live one's `true`, and publishing only
the growing batch-heads would leave the buffer carrying an older step's answer for
the others -- which is the set a host admission then plans from.

<a id="syncthreads-2"></a>
### `__syncthreads`

THE PROLOGUE ENDS ON THIS BARRIER AND THERE IS NO OTHER BELOW IT. It publishes the
claim row to the walk and the zeroed warp arrival count to the combine -- shared memory
is not zeroed at CTA start, and this is the one place a zero can be published for free.
docs/internals/decode/decode.md#no-barrier

<a id="want-live"></a>
### `want_live`

A NEEDY BATCH-HEAD RUNS NOTHING: no walk, no `y`, no state, no counter beyond its
arrival -- which is what lets the step be enqueued before the admission it needs is
known, and what makes the caller's replay THE SAME STEP rather than a second one.
docs/internals/decode/decode.md#per-bh-verdict
`n_live` -- the same bound the retirement test above asked as a COMPARE, formed here
because the combine and the arrival need the NUMBER rather than the answer.

<a id="decode-step-kernel-note-l386"></a>
### in `decode_step_kernel`, near line 386

THE WARP'S OWN PARTIAL, and it exists whether or not the warp walks anything: the
combine is an ARRIVAL COUNT, so every warp of a live CTA deposits and arrives, and a
warp whose units are all beyond this term's count deposits exact zeros. Retirement at
warp grain is therefore the walk it does not run, never an early `return`:
docs/internals/decode/decode.md#the-warp-arrival

<a id="decode-step-kernel-note-l397"></a>
### in `decode_step_kernel`, near line 397

THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is
pinned here (a plain origin, no per-atom widening multiply); the cache is keyed on
the atom VALUE, so it is correct whatever order the walk reaches leaves in.
`atom_base` is the atom's first row as an ELEMENT offset from `p.state`: dense is
`bh * N + (atom << 4)` (which is why a topology whose `N` is not a whole number of
atoms still addresses correctly -- the two halves recombine to `bh * N + leaf`),
paged is `slot << 4`. docs/internals/decode/decode.md#paged-address
A CLAIMING STEP READS TWO SOURCES: the page table first, and this step's CLAIM ROW
only where the table says absent. That is the whole of the pool inside the walk, and
it is the shape the race-free claim row forces -- the row no longer carries the
table's own entries, so it cannot answer for an atom it did not claim.
THE TABLE COMES FIRST BECAUSE IT ANSWERS ALMOST EVERY ATOM: a claiming step's newly
claimed atoms are by construction the ones the table does NOT map, so the fallback is
taken only on them and every resident atom costs one load, as before.
THE TWO CAN NEVER DISAGREE. `_admit_pages` writes only atoms the table has as absent
and there is no eviction, so a slot never moves once assigned; a claim row entry from
an earlier step therefore equals the table's. The one path that could break that is
`PageArena.reset`, which clears the table and rewinds the allocator -- so it now
clears this row too (`rola/ops/paging.py`). docs/internals/decode/decode.md#the-claim

<a id="units"></a>
### `units`

THE UNIT IS THE (owner, outer-run) PAIR and the grid partitions ITS OWN index
space -- not the owner's -- so every touched leaf has exactly one owning CTA, warp
and lane and the state needs no atomic:
docs/internals/decode/decode.md#the-unit-partition
A WARP TAKES A WHOLE UNIT, and a unit is the DRAM granule itself: under the
mixed-radix order a unit's leaves are the innermost level's contiguous run, which at
the flagship spans is one 16-leaf ATOM. Owner-contiguity across units buys nothing
on top of that -- the sectors are already whole -- while partitioning at owner grain
would cap the walk's parallelism at the LIVE OWNER COUNT and give each warp the
whole `BC`-leaf box to walk serially. The CTA's warps take CONSECUTIVE units, so a
CTA still covers one contiguous stretch of the plane per pass.
EVERY LEVEL BUT THE INNERMOST LEAVES THE SLOT LOOP AND THAT IS THE UNIT'S POINT:
a unit fixes their digits, so their amplitude product is formed ONCE, in
`resolve_unit`, and every slot below costs ONE innermost amplitude rather than `D`.
THE OUTER LOOP IS THE TERM, and it is a LATENCY decision: every level's kind is
loop-invariant inside it, so the list bases, the counts, the field widths and the
membership predicates all hoist and the per-unit dependent chain is the union walk's.
The alternative -- one index space, term named per unit -- was BUILT AND MEASURED at
`+2.3 ns` per unit, which cost the scattered cell more than the decomposition saves.
docs/internals/decode/decode.md#the-term-loop

<a id="units-2"></a>
### `units`

THE TERM'S ROW OF THE PACKED TABLE, hoisted once per term: every level's kind,
field width, field position and realized count is loop-invariant inside the term,
so the resolver below reads ONE word per level instead of three:
docs/internals/decode/decode_lattice.md#the-terms

<a id="period"></a>
### `period`

THE BLOCK, THE REPEAT AND THE PERIOD. A warp's low `kUnitBlockWarpsBits` name its
unit inside the segment's block; the rest name WHICH of the CTA's `kUnitRepeats`
blocks it serves, one period apart. `seg * kUnitBlockWarps + ub` sweeps one whole
period bijectively, so the partition is the same one-owner-per-row partition it
was, cut at a granularity the CTA's width no longer sets.

<a id="u"></a>
### `u`

TWO RESOLVERS, ONE WALK. The write product is enumerated from live DIGITS and
every other term from live owners crossed with the run radix; both hand back the
identical `Unit<D>`, and the selector is the term loop's own invariant.
docs/internals/decode/decode_lattice.md#the-write-grain

<a id="decode-step-kernel-note-l474"></a>
### in `decode_step_kernel`, near line 474

THE ENUMERATED RUN IS THIS TERM'S. Term 0 walks the write run; every other term
is disjoint from `supp(W)` by construction and walks the read remainder it names.
A UNION `br | bw` WOULD DOUBLE-COUNT: a write unit whose innermost read digits
leave `W` reaches leaves that the innermost split term also enumerates.
`valid` needs no test here -- `resolve_unit` has already zeroed the slices of a
unit its term does not realize. docs/internals/decode/decode_lattice.md#the-unit

<a id="t"></a>
### `t`

THE SLOT LOOP IS THE INNERMOST LEVEL'S RUN, in ASCENDING lattice order -- the
unit's own contiguous run -- iterated over the TERM's set bits, so a dead
innermost digit costs one `__ffs` step and no address at all. Ascending is what
makes `cached_atom` the walk's only reuse and enough of it:
docs/internals/decode/decode.md#aligned-subranges
NO SPECULATIVE UNROLLING: the trip count is a runtime popcount, and an unrolled
body duplicates every live value of the walk's tightest scope.

<a id="decode-step-kernel-note-l516"></a>
### in `decode_step_kernel`, near line 516

AN ABSENT ATOM READS AS ZEROS AND IS NOT WRITTEN. Exact, not defensive: the
plan admits the step's write set exactly, so an atom with no slot is outside
it, and a never-deposited leaf holds zeros in the dense backing too. The
self-normalizing readout then contributes exactly nothing through it.
docs/internals/decode/decode.md#absent-atom

<a id="decode-step-kernel-note-l532"></a>
### in `decode_step_kernel`, near line 532

UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
not an optimization: survival is INCLUSIVE at `j = t`, so the whole causal
intra Gram collapses into reading the row after it is written. A row in
`supp(R) \ supp(W)` has `keep == 1` and no deposit, so the same expression is
its readout too -- one path, no arm.
A DONE BATCH-HEAD TAKES THE READ AND NOT THE DEPOSIT, and reproduces its `y`
BIT FOR BIT while doing so: docs/internals/decode/decode.md#done-flags
HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read

<a id="decode-step-kernel-note-l541"></a>
### in `decode_step_kernel`, near line 541

DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
SPLIT: at `T = 1` the tiled kernel's rule-1 half-mass shift is exactly the
identity, so ONE `expf` replaces its three. `log1pf` IS the composition
(never a rounded `1 - rate`) and there is no epsilon floor -- both of that
rule's rule 2's prohibitions: docs/internals/decode/decode.md#decay-at-read

<a id="fold-pitch"></a>
### `fold_pitch`

ZERO `__syncthreads` BELOW THE PROLOGUE, and this is the mechanism that removes the
last of them. Each warp deposits its `cols` partials into ITS OWN shared row, fences,
and bumps the CTA's arrival count with one atomic; the LAST ARRIVER -- and no barrier
-- is what makes "every row is written" true, and it alone sums them.
DETERMINISM SURVIVES BECAUSE THE ORDER IS FIXED, not because the arrivals are: the sum
runs over warps `0 .. kDecodeWarps - 1` per column, exactly the order the barrier-stepped
tree it replaces used, so `y` is byte-reproducible at a fixed `n_split` as before.
EVERY LANE FENCES ITS OWN STORES before the one-lane arrival, and every lane fences
again after the broadcast before it reads a sibling's row (KERNEL_STANDARDS 12: a fence
orders only the fencing thread).
HAZARD fold-bank-padding -- docs/internals/decode/decode.md#fold-bank-padding

<a id="decode-step-kernel-note-l619"></a>
### in `decode_step_kernel`, near line 619

OVER THE LIVE SEGMENTS, NOT OVER THE CEILING -- and this is where retirement is
actually paid for. The per-segment cost a large split carries is the ARRIVAL AND
THE COMBINE (`n_split` atomics on one counter, then `n_split` `__ldcg` loads per
column on ONE warp), not the replicated prologue: a variant that retired the walk
but still wrote exact zeros and arrived was BUILT AND MEASURED AT ZERO. The order
is still a FIXED ASCENDING SLOT ORDER over a range every CTA of the batch-head
derives identically from the published counts, which is the whole of G1; `y` moves
across `n_split` exactly as it always did (fp32 addition is not associative) and is
byte-identical at a fixed one. docs/internals/decode/decode.md#slot-order
A NEEDY BATCH-HEAD'S CTAs ARRIVE ANYWAY and deposit nothing. The arrival is what
elects one CTA to publish the verdict below, and electing it from the same counter the
combine uses is what makes "no CTA of this batch-head is still reading" the SAME
argument in both cases: docs/internals/decode/decode.md#the-last-arriver

<a id="decode-step-kernel-note-l655"></a>
### in `decode_step_kernel`, near line 655

UNROLLED FOR MEMORY-LEVEL PARALLELISM, and the factor is a MEASURED one. The trip
count is dynamic, so KERNEL_STANDARDS 6 asks for the pragma to be explicit -- and
at `1` it is a correctness defect on this loop, not a style choice: the `n_live`
loads are INDEPENDENT (only the adds chain), and forbidding the unroll leaves one
load in flight, which measured 26.6 us of a 93 us dense step on ONE warp of ONE CTA
while 79 SMs idled -- 50 % of the scattered step.
docs/internals/decode/decode.md#slot-order

<a id="decode-step-kernel-note-l667"></a>
### in `decode_step_kernel`, near line 667

THE MASS COLUMN IS ONE LANE'S HERE, not every lane's -- the opposite of the
CTA fold above, and for the opposite reason: this loop's length IS the step's
tail, so a broadcast load every lane repeats is a third of the only critical
path left in the kernel. It is broadcast once, at the divide.

<a id="row-base"></a>
### `row_base`

EVERY MUTATION A SIBLING CTA COULD HAVE READ HAPPENS HERE AND NOWHERE ELSE, which
is why it is safe: docs/internals/decode/decode.md#the-last-arriver
IT IS ONE WARP'S WORK NOW, not the CTA's, and that is the barrier deletion's price
paid where it is cheapest: the CTA's other warps have already retired at the arrival
count, so the fold is lane-strided over the page table row and its size is folded by a
warp shuffle instead of a block scan. It runs only on a CLAIMING step.

<a id="row-base-2"></a>
### `row_base`

THE CLAIM'S SIZE FALLS OUT OF THE FOLD ITSELF -- an atom the shadow row maps and
the table does not IS a claimed atom -- so it is neither carried across the walk
nor re-derived from the write-support enumeration. Both of those keep values alive
through the loop the register budget is tight in, and both were measured to spill:
docs/internals/decode/decode.md#declared-residency

<a id="decode-arm-0"></a>
### `DECODE_ARM_0`

THE DECLARED ARM LIST, AS DATA. The decode family's instantiation matrix is the
product `d_v(2) x D(4) x decay(2)`, and it is written out one row per arm so a
BUILD can name a subset of it by index -- `ROLA_DECODE_ARMS=5` compiles arm 5 alone
and nothing else. That is an ITERATION knob and only that: the default below is
every declared arm, so a plain build and every gate build are the closed-world one,
and a subset binary is refused a shape it does not carry rather than mis-launching
it. docs/build.md, docs/internals/decode/decode_api.md#arm-subset
THE LAUNCH IS STILL WRITTEN ONCE. `dispatch_switch.cuh` refuses the MACRO LADDER --
a `#define` per axis whose BODY restates the `<<<...>>>` -- and this is not one: the
macro carries the arm's three CONSTANTS and its comparison, the launch itself is the
single generic lambda below, and no `<<<...>>>` is spelled twice.

<a id="fill-lattice"></a>
### `fill_lattice`

THE (k, m) BOX, FILLED AND REFUSED HERE. The derivation is `carry/box.cuh`'s
`BoxPlan`, evaluated on runtime widths instead of template arguments -- decode takes
its lattice as data, because one built binary serves every topology the producer can
emit. Every constraint is that header's `static_assert` restated as a runtime check.
docs/internals/decode/decode_lattice.md#the-refusals

<a id="floor-inner"></a>
### `floor_inner`

`BoxPlan`'s section 2.5 as the atom-rectangle law generalizes it: the INNERMOST
level takes as much capacity as it can carry -- never less than its balanced share,
never less than an atom asks, never more than its own width admits -- and the
rest of `j` is spread balanced over the outer levels, innermost first.  The atom
floor is a PREFERENCE and the width ceiling is the law: a level narrower than an
atom spills its capacity outward and the atom then spans more than one level.

<a id="fill-lattice-note-l960"></a>
### in `fill_lattice`, near line 960

THE UNIT, and the atom it is reconciled with. The unit is the innermost level's run
-- ALWAYS, at every topology -- and the two shifts below are how an atom is named in
terms of it: a run at or above `kAtomLeaves` holds whole atoms, a run below it means
an atom is that many consecutive units. Exactly one of the two is non-zero, and only
the CANDIDATE enumeration reads them; the walk is the same three loops either way.

<a id="rola-decode-producer-width-mirror"></a>
### `rola_decode_producer_width_mirror`

The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately from the
capacity because the two are DIFFERENT CLAIMS -- the second is what stops the
`static_assert` being satisfied by a stale mirror:
docs/internals/decode/decode.md#producer-width-mirror

<a id="rola-decode-residency"></a>
### `rola_decode_residency`

THE STEP'S DECLARED RESIDENCY, in CTAs per SM, read off the same `constexpr` the
`__launch_bounds__` are built from. The split policy needs it to size one wave, and a
python mirror of a launch bound is exactly the constant that drifts:
docs/internals/decode/decode_api.md#residency

<a id="note-l1083"></a>
### near line 1083

THE TWO BACKINGS, validated where they differ and nowhere else. A page table names
the leaf space (`[BH, N / kAtomLeaves]`) and the plane holds only the slots a plan
committed; without one the plane IS the leaf space. Slot VALUES are not
bounds-checked -- that would be a device-side read of the table, and the plan owns
the bound. docs/internals/decode/decode_api.md#paged-state

<a id="note-l1131"></a>
### near line 1131

THE ADMISSION BUFFERS, checked whichever backing this is: the step's own verdict
lands in them under a page table and they are untouched without one, and validating
them either way is what keeps the two backings ONE ABI:
docs/internals/decode/decode_api.md#the-step

<a id="torch-check"></a>
### `TORCH_CHECK`

THE ATOM'S PLACE IN THE LATTICE, required of a PAGED backing only, and now ONE
condition where the interleaved order needed two: a unit is a contiguous aligned run,
so an atom is a whole number of units or a unit is a whole number of atoms, and only
the second makes the candidate enumeration a partition. The interleave's `k^D >= 16`
and `log2(k) | 4` are discharged by the mixed-radix order itself (`box.cuh`, K3/K4).
A dense backing never asks an atom a liveness question and carries no constraint:
docs/internals/decode/decode.md#candidate-atoms

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kmaxlevels"></a>
### `kMaxLevels`

THE PAGE GRANULE AND THE OWNER'S SHAPE COME FROM THE BOX, not from a mirror of it:
`carry/box.cuh` is where the lattice's addressing constants and its constexpr
`BoxPlan` arithmetic live, and the two families write and read ONE plane in ONE order.
What this path adds is the RUNTIME carrier below -- decode takes `(k, m)` and the
per-level widths as data rather than as template arms, so it holds `BoxPlan`'s fields
as scalars under `BoxPlan`'s own names and derives them by `BoxPlan`'s own formulas.

<a id="kproducermaxbranchwidth"></a>
### `kProducerMaxBranchWidth`

THE LEVEL-WIDTH CAPACITY, a GATE and not a convenience: below the producer's
`MAX_BRANCH_WIDTH` it would accept a topology the producer can emit and then
over-read a staged row. Compile-time half here, runtime half in a bound accessor:
docs/internals/decode/decode.md#level-width-capacity

<a id="kdecodethreads"></a>
### `kDecodeThreads`

THE STEP LAUNCHES ONE FAT CTA PER SM, and that is a SHARING decision before it is an
occupancy one. Every CTA of a step re-derives the identical per-(step, batch-head)
setup -- the fold, the digit masks, the four owner lists, the term table, the write
digit list, and the ADMISSION with its candidate-atom enumeration and page-table probes
-- and those CTAs run PHASE-SYNCHRONIZED, so the duplication has no mutual coverage at
all. THE COST IS WAVES, NOT COPIES: a derivation duplicated across CONCURRENT SMs is
wall-free, and it is paid only where the live CTAs exceed the machine and the copies
SERIALIZE -- 256 thin CTAs over 80 SMs is 3.2 derivations deep per SM, and that depth,
not the count, is what a wider CTA removes. Measured on this path, that derivation is
51-87 % of the step at `BH = 1` and 18-59 % at `BH = 8`.
WIDENING THE CTA IS WHAT SHARES IT. Intra-CTA sharing is native -- shared memory and a
barrier, which the derivation is already written as -- so a 32-warp CTA derives ONCE
what four 8-warp CTAs derived four times, and derives it on a chain four times shorter
because every loop in it is `for i = tid; i < n; i += blockDim`.
docs/internals/decode/decode.md#decode-threads
THE WIDTH IS THE ARM, and D1 measures two of them: the PRIMARY arm is the fat CTA
(1024 threads, 32 warps, one CTA per SM) and the COMPARISON arm is F512 (two CTAs per
SM). With no barrier after the prologue the two differ only in the single wave's tail,
which is what the ten-pair table decides between:
docs/internals/decode/decode.md#decode-threads

<a id="kunitblockwarps"></a>
### `kUnitBlockWarps`

THE UNIT PARTITION'S GRANULARITY, AND IT IS NOT THE CTA'S WARP COUNT. Widening the CTA
is a decision about WHERE THE DERIVATION IS SHARED; it is not a decision about how
coarsely the unit space is cut, and tying the two together is what made a wider CTA
retire whole SMs. The block a segment owns stays `kUnitBlockWarps` units wide at every
width, so `n_live = ceil(umax / kUnitBlockWarps)` -- the live SEGMENT count, and with it
the machine's coverage -- is the same number at 256 threads and at 1024.
A CTA THEN SPENDS ITS EXTRA WARPS ON MORE BLOCKS, NOT ON A WIDER ONE: its
`kUnitRepeats` groups of `kUnitBlockWarps` warps take `kUnitRepeats` blocks one whole
period apart, so the walk covers the identical unit set in the identical order per
block, every warp is spent wherever units allow, and each block is still the CONTIGUOUS
run the write grain's rotation is built on.
RETIRE WORK, NOT CTAs. docs/internals/decode/decode.md#retirement

<a id="kdecodemaxspan"></a>
### `kDecodeMaxSpan`

THE LATTICE'S ONE DECLARED CAPACITY, refused loudly at the host entry: an owner's
per-level span is read as a SINGLE 32-bit slice, both when a level's owner grid is
built and when a unit's innermost run is resolved. The interleaved order's second and
third ceilings (a sub-box's leaves, the capacity multiplier) died with it -- under
mixed radix a unit is a contiguous run and there is no sub-box scan to bound.
docs/internals/decode/decode_lattice.md#capacities

<a id="kdecoderesidentthreads"></a>
### `kDecodeResidentThreads`

THE DECLARED RESIDENCY of the step kernel, in CTAs per SM, and it is a BUDGET rather
than a hope: `__launch_bounds__` turns it into a register cap the assembler must meet,
and the census is what proves it met it without spilling. The walk is a latency-bound
stream over state rows, so residency is the one thing it trades registers FOR.
IT IS ONE, AND IT NO LONGER HAS TEMPLATE AXES. At `kDecodeThreads = 1024` a CTA is the
SM's whole resident thread budget that the register file admits (65536 / 1024 = 64
registers per thread), so the DECAY and DEEPEST arms -- which used to buy their extra
live state by giving up a CTA of residency -- keep the same 64-register cap the widest
arm needs, and the shallow arms gain headroom rather than losing it (they ran at a cap
of 51 when five 256-thread CTAs shared the file). ONE is also what makes the derivation
shared: the host's split policy sizes ONE WAVE of this number, so `n_split` becomes
`SMs / BH` fat CTAs and the per-(step, batch-head) setup is derived ONE WAVE DEEP
instead of 3.2. docs/internals/decode/decode.md#declared-residency
IT IS DERIVED FROM THE WIDTH, not written down twice: the register file admits 64
registers per thread across 1024 resident threads, so a CTA of `kDecodeThreads` leaves
room for exactly `1024 / kDecodeThreads` of itself. At 1024 that is the one fat CTA per
SM; at 512 it is two, and both put the same 1024 threads and the same 64-register cap on
the SM. The arm's width is then the ONLY constant a shape change touches.

<a id="katomshift"></a>
### `kAtomShift`

The atom as a SHIFT and a MASK. `kAtomLeaves` itself is the box's
(`rola::carry::kAtomLeaves`, imported above): it is geometry-owned and fixed across
every config forever, which is what makes the paged address BC-free.
docs/internals/decode/decode.md#the-atom

<a id="decodeparams-note-l152"></a>
### in `DecodeParams`, near line 152

THE RAW PER-TOKEN OPERANDS, ADDRESSED RATHER THAN COPIED: both sides' routing
levels, the write gain and the token's value, each as the caller's own view. There
is no `[BH, b_l]` plane between the producer and the walk -- the step folds these
into shared memory itself: docs/internals/decode/decode_fold.md

<a id="decodeparams-note-l174"></a>
### in `DecodeParams`, near line 174

THE (k, m) BOX, `BoxPlan`'s fields as runtime scalars under `BoxPlan`'s names:
`lat_s[l] = k * m_l` is an owner's span at level `l`, `lat_g[l] = B_l / lat_s[l]`
its owner-grid extent, and a leaf's lattice index is
`lambda = O * BC + sum_l r_l << lat_run_shift[l]` -- the owner in mixed radix over
`lat_g`, the OWNER-LOCAL index in mixed radix over the runs `lat_s`, most
significant level first. `lat_run_shift[l] = log2 prod_{l' > l} s_l'` is
`BoxPlan::run_shift`, and every suffix product is a SHIFT because every span is a
power of two. docs/internals/decode/decode_lattice.md

<a id="decodeparams-note-l190"></a>
### in `DecodeParams`, near line 190

THE WALK'S UNIT, and the whole of it: a unit is `(owner, r_0 .. r_{D-2})` -- an owner
together with every run offset but the innermost -- so its leaves are the innermost
level's whole run, `1 << lat_inner_bits` of them, CONTIGUOUS and aligned in lattice
order. At the flagship spans `(8, 16)` that run IS an atom, at full occupancy, and
the unit's 16 slots are level 1's digits.
THERE IS NO SECOND UNIT SHAPE. A topology whose innermost span is thinner than an
atom does not get a deeper walk -- it gets SEVERAL UNITS PER ATOM, which the
candidate enumeration absorbs (`lat_units_per_atom_shift`) and the walk never sees.
docs/internals/decode/decode_lattice.md#the-unit

<a id="decodeparams-note-l209"></a>
### in `DecodeParams`, near line 209

THE OWNER LIVENESS BITMASK'S word layout: one bit per OWNER COORDINATE, per level,
per kind -- `omask_words[l] = ceil(lat_g[l] / 32)` words at `omask_off[l]` of an
`omask_total`-word plane, one plane per kind. It replaces the four COMPACTED owner
lists: a rank selects its owner by find-nth-set instead of by an indexed load, which
is what deletes the prologue's block scans and their barriers:
docs/internals/decode/decode_lattice.md#the-owner-lists
THE WRITE SIDE'S LIVE-DIGIT LIST NEEDS NO PLANE AT ALL: the write digit mask IS its
bitmask, so term 0's grain is the same find-nth-set over `mask_w`:
docs/internals/decode/decode_lattice.md#the-write-grain

<a id="decodeparams-note-l221"></a>
### in `DecodeParams`, near line 221

THE COMPACTED LISTS THE BITMASK PRODUCES, and they are read exactly as they were
before D1: `olist[kind][level][i]` is the i-th live owner, one indexed load. The
bitmask is the BUILDER, not the reader -- a find-nth-set at use time was measured at
+25 % of the dense step, because the admission resolves one rank per candidate atom
per CTA: docs/internals/decode/decode_lattice.md#the-owner-lists

<a id="decodeparams-note-l233"></a>
### in `DecodeParams`, near line 233

THE ATOM AND THE UNIT, RECONCILED, and PAGED BACKING ONLY. Exactly one of these is
non-zero: a unit holds `1 << atoms_per_unit_shift` atoms, or an atom holds
`1 << lat_units_per_atom_shift` consecutive units. The candidate enumeration is a
PARTITION of the write side's atoms either way -- which is what the pool claim's
canonical ranks need -- and the WALK is the same three loops in both cases.
The host refuses a page table under a lattice whose OWNER is thinner than an atom,
because then an atom would straddle two owners and no grouping of units recovers it:
docs/internals/decode/decode.md#candidate-atoms

<a id="decodeparams-note-l250"></a>
### in `DecodeParams`, near line 250

`[BH, atoms_per_bh]` int32, `-1` = ABSENT; `nullptr` = the dense backing, where the
slot IS the atom. ONE base translation for both:
docs/internals/decode/decode.md#paged-address
MUTABLE because a pool claim folds into it, and only ever at the point where the
batch-head's last CTA has arrived: docs/internals/decode/decode.md#the-claim

<a id="decodeparams-note-l265"></a>
### in `DecodeParams`, near line 265

`[BH]` int32, THE DONE FLAGS -- `1` where a batch-head has already deposited this
step. A replay re-walks a done batch-head for its `y` and deposits NOTHING, which
is what makes "exactly once per sequence" hold across the admission the replay
exists for: docs/internals/decode/decode.md#done-flags

<a id="decodeparams-note-l275"></a>
### in `DecodeParams`, near line 275

THE SLACK POOL -- physical slots pre-committed and pre-zeroed by the host, from
which a growth step admits ITSELF. `pool_slots` is `[BH, pool_cap]` slot ids,
`pool_cursor` `[BH]` how many of each batch-head's are spent, `pool_map`
`[BH, atoms_per_bh]` the claim the walk addresses through before the fold.
`pool_cap = 0` (and three null pointers) is EXACT-COMMIT: every growth step goes to
the host, which is the shipped default: docs/internals/decode/decode.md#the-claim

<a id="extract-bits-2"></a>
### `extract_bits`

EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, returned in
the low `count` bits (`count <= 32`) -- the shift-and-merge every lattice slice is
read through, written once because it is the one place a funnel shift can be off by
one. docs/internals/decode/decode.md#extract-bits

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode.cu` condensed at their call site into a short pointer.

<a id="decode-step-kernel-note-l95"></a>
### in `decode_step_kernel`, near line 95

---- SMEM ledger ----------------------------------------------------------
  sa_r, sa_w   [total_rows] fp32   both sides' amplitude rows, UNCOMPACTED and
                                   indexed by DIGIT: the digit IS the index.
  sd           [total_rows] fp32   this head's decay dials; DECAY only
  v_tok        [DV]         fp32   the token's value, resident
  fold  [warps][cols+3]     fp32   the cross-warp partials, bank-padded
  mask_r,mask_w [mask_total] u32   one bit per digit, per side
  omask  [4][omask_total]   u32    the four kinds' owner LIVENESS BITS, per level
  olist  [4][olist_total]   i32    those bits COMPACTED: the i-th live owner
  dlist  [dlist_total]      i32    the WRITE side's live digits, OUTER levels only

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs share.

T = 1 HAS NO STAGE 2: every stage-2 decision is degenerate, so this path bypasses
selection ARCHITECTURALLY and its config freezes at the prefill->decode boundary.
What is left per step is data plane only, in the ratified shape: a
TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.

Three facts a reader must not miss:
  1. THE LEAF ORDER IS THE LATTICE, in the ONE order `carry/box.cuh` defines: a leaf is
     named by its owner and by its per-level RUN OFFSETS in mixed radix, and the step
     reaches it by WALKING those factors -- there is no leaf map, no bitmap over the
     leaf space, and no term in `N` in the step's fixed cost.
  2. supp(R) = prod_l supp_l(R) at T = 1, so the touched set is a PRODUCT at every
     lattice granularity and the walk prunes at each one.
  3. PAGED AND DENSE ARE ONE WALK. The page granule is the ATOM (16 leaves), which is
     what keeps the address BC-free; a null page table means the slot IS the atom and
     the bytes are the dense path's, exactly.

The lattice, the walk's cost accounting and the paged base translation:
docs/internals/decode/decode.md

<a id="near-line-51"></a>
### near line 51

NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a hash that
depends on the BUILD PATH, so a manifest ratified in one worktree certifies a name no
other checkout's build produces. A ratifiable kernel needs a named symbol.

<a id="decodestampkernel"></a>
### `decode_stamp_kernel`

THE STEP'S OWN FOOTPRINT, recompiled with this translation unit, so a test that reads
it back is reading a fact only the loaded binary can produce:
docs/internals/decode/decode_api.md#build-stamp

<a id="cols"></a>
### `cols`

The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a
template axis and not a runtime test: docs/internals/decode/decode.md#template-axes

<a id="nc"></a>
### `NC`

Columns per lane: `NC` fully coalesced 128 B transactions, and the lane's
accumulators are its own output columns.

<a id="sumax"></a>
### `s_umax`

The retirement bound, folded by `publish_term_layout`'s own `atomicMax` -- which is why
it is zeroed HERE, behind the fold's barrier: docs/internals/decode/decode.md#retirement

<a id="near-line-122"></a>
### near line 122

One unit count per TERM. The walk's outer loop is the term, so each term's rank space
is partitioned across the grid on its own and the counts never need summing.

<a id="sarrive"></a>
### `s_arrive`

THE CTA'S WARP ARRIVAL COUNT, and the whole of the combine's synchronisation. It is
zeroed in the PROLOGUE, behind the prologue's own last barrier, which is what lets the
combine below need no barrier at all: docs/internals/decode/decode.md#the-warp-arrival

<a id="near-line-169"></a>
### near line 169

`sum_l B_l` bits and at most `sum_l g_l` entries: nothing in this prologue is sized
in the leaf space: docs/internals/decode/decode_lattice.md

<a id="candfld"></a>
### `cand_fld`

THE CANDIDATE SLOT'S ROW of the same packed term table -- the ADMISSION's rank space,
which is the write product in the OLD owner-and-run order because only that order is
monotone in atom id: docs/internals/decode/decode_lattice.md#the-candidate-slot

<a id="ncandidates"></a>
### `n_candidates`

FROM THE CANDIDATE SLOT, not from the walk's write term: the two enumerate the same
set and only this one is monotone in atom id, which is what the claim's canonical
ranks are defined in: docs/internals/decode/decode_lattice.md#the-write-grain

<a id="log1pf"></a>
### `log1pf`

THE CLOCK IS THE STORED WRITE ALLOCATION ITSELF -- no side gain, which is
what makes "stored != 0 => live != 0" an identity:
docs/internals/decode/decode.md#detached-clock

<a id="near-line-466"></a>
### near line 466

THE MASS COLUMN IS READ BY EVERY LANE, not by lane 0 alone: the value is the same
broadcast load for all of them and it is the denominator every lane divides by, so
carrying it in all 32 registers costs one load and saves the shuffle.

<a id="near-line-503"></a>
### near line 503

SLOT ORDER, always -- the whole determinism argument. `__ldcg` bypasses L1 so
the reader sees the stores the `__threadfence` above made visible.
HAZARD slot-order -- docs/internals/decode/decode.md#slot-order

<a id="lane"></a>
### `lane`

Reset for the next step: no `memset` launch between two decode steps, and the
counter is safe to capture in a CUDA graph.

<a id="seen"></a>
### `seen`

THE REDUCTION THE HOST READS, and the done flags' self-reset. Nothing needy means
every batch-head is done, so the two answers are ONE test and no launch stands
between two steps: docs/internals/decode/decode.md#growth-reduction

<a id="roladecodearms"></a>
### `rola_decode_arms`

THE BUILT SET, READ BACK OFF THE SAME LIST: `[d_v, D, decay]` per arm, in the order
the binary carries them, so a subset build is SELF-IDENTIFYING rather than merely
smaller: docs/internals/decode/decode_api.md#arm-subset

<a id="torchcheck"></a>
### `TORCH_CHECK`

The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16`. What is NOT an axis --
`BT`, `BC`, the lattice's `(k, m)` and every operand's storage type -- is the
interesting part: docs/internals/decode/decode.md#instantiation-matrix

<a id="launch"></a>
### `launch`

ONE LAUNCH SITE. Each axis arrives as an `integral_constant`, so the three
`::value`s are template arguments and the `<<<...>>>` configuration cannot drift
between arms: docs/internals/dispatch_switch.md#one-launch

<a id="isbf16"></a>
### `is_bf16`

ONE FLOATING AXIS OVER THE CLOSED SET THE PRODUCER AND THE VALUE PROJECTION EMIT: an
unmatched dtype is a REFUSAL rather than a silent widening on the host.
docs/internals/decode/decode_fold.md#the-dtype-arms

<a id="spanof"></a>
### `span_of`

The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. `T != 1` is refused -- the fold
is defined on one token and a second one would be silently dropped:
docs/internals/decode/decode_fold.md#the-identity

<a id="gsuf"></a>
### `gsuf`

`BoxPlan::run_shift`: level `l`'s run offset sits at `log2 prod_{l\' > l} s_l\'` in the
owner-local index, so the whole mixed radix is shifts and masks.

<a id="filllevels"></a>
### `fill_levels`

The per-level geometry every decode entry validates the same way, filled into `p` and
reported back as `total_rows`.

<a id="roladecodebuildstamp"></a>
### `rola_decode_build_stamp`

THE BUILD STAMP: a path or hash check cannot catch a stale `.so` at the right path,
and only a device-side fact can: docs/internals/decode/decode_api.md#build-stamp

<a id="y"></a>
### `y`

`empty`, not `zeros`: every element is written, so a zero-fill is a launch the
launch-latency-bound small-batch regime pays for nothing:
docs/internals/decode/decode.md#empty-not-zeros

<a id="pooled"></a>
### `pooled`

THE SLACK POOL, present or absent as ONE decision -- three buffers and a capacity,
or four zeros and exact-commit. There is no half-configured pool:
docs/internals/paging/paging.md#the-slack-pool

<a id="kmaxlevels-2"></a>
### `kMaxLevels`

`1 <= D <= 4`. The same bound the consumer's `kMaxLevels` states, restated
rather than included -- the decode TUs share no other object with the tiled
consumer. docs/internals/decode/decode.md#max-levels

<a id="span"></a>
### `Span`

ONE OPERAND'S ADDRESS, as the producer left it: the base and the three strides the
fold walks over a `[B, 1, H, W]` view. `sw` is `0` for the per-token scalars, which
have no last axis. docs/internals/decode/decode_fold.md#the-identity

<a id="decodeparams"></a>
### `DecodeParams`

EVERY POINTER AND EXTENT ONE DECODE STEP NEEDS, and nothing the kernel could cheaply
recompute. Field by field: docs/internals/decode/decode.md#params-amps

<a id="near-line-98"></a>
### near line 98

Per READ level: does it still carry its own mass, so the fold divides it by its row
sum? The write side is never normalized.

<a id="levelsbf16"></a>
### `levels_bf16`

The storage type of each operand family, `1` = bfloat16 and `0` = float. Branching
on STORAGE selects a typed gather loop once per side, never a test per element:
docs/internals/decode/decode_fold.md#the-dtype-arms

<a id="near-line-129"></a>
### near line 129

`r_l`'s shift inside the unit index: `lat_run_shift[l]` re-based on the innermost run,
so a unit decomposes by shifts alone and no device division exists on this path.

<a id="near-line-132"></a>
### near line 132

The digit masks' word layout: level `l` owns `mask_words[l]` words at `mask_off[l]`
of a `mask_total`-word plane, one plane per side.

<a id="near-line-150"></a>
### near line 150

THE WRITE SIDE'S LIVE-DIGIT LIST, OUTER LEVELS ONLY -- term 0's own grain, compacted
from `mask_w` by the same prefix popcount:
docs/internals/decode/decode_lattice.md#the-write-grain

<a id="state"></a>
### `state`

DENSE `[BH, N, cols]` or PAGED `[slots, kAtomLeaves, cols]`, `cols = d_v + 1`.
Read AND written in place by the SAME lane in program order:
docs/internals/decode/decode.md#state-in-place

<a id="growth-2"></a>
### `growth`

`[BH]` int32, THE PER-BATCH-HEAD VERDICT: `1` where this step writes an atom no
table entry and no pool slot can address, `0` where it advanced. Written by the
batch-head's last-arriving CTA: docs/internals/decode/decode.md#per-bh-verdict

<a id="growthany"></a>
### `growth_any`

`[1]` int32, the OR of `growth` over batch-heads, and the `[1]` int32
self-resetting arrival counter of the CTA that reduces it:
docs/internals/decode/decode.md#growth-reduction

<a id="atombits"></a>
### `atom_bits`

`[BH, atoms_per_bh]` bool, this step's own write set condensed to atoms. It is what
a growth step's `plan_exact` admits from, so the host never re-derives the set the
step has already expanded: docs/internals/decode/decode.md#atom-bits

<a id="ws"></a>
### `ws`

Split-K workspace. Slot-ordered and summed by the last-arriving CTA, which is what
makes decode's `y` RUN-REPRODUCIBLE -- and that diverges from the tiled consumer
on purpose: docs/internals/decode/decode.md#determinism

<a id="split-planes"></a>
### The split bf16 planes, and the unit that reads one

THE STATE'S PAGE HOLDS SPLIT bf16 PLANES --
`[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]`, the geometry
`common/state_page.md` defines and the carry addresses with the same blocks. The
walk's per-leaf row is therefore two 16-bit runs rather than one fp32 run, and the
plane count is the leaf's own fact for this step:

* a leaf this step WRITES is read `hi || lo` -- the exact fp32 it carried -- updated,
  and stored back to both planes, so a sequence's carried precision is untouched. The
  plane choice keys on `in_w`, the step's own write set, and NOT on whether THIS PASS
  performs the deposit: a gated step's third outcome is "re-read without depositing"
  (`done[bh]`), and a re-read of a leaf the step wrote must see the same bytes the
  depositing pass saw, or two admission orders would compute different `y` for the same
  step. `deposit` gates the STORE and nothing else;
* a leaf this step only READS pulls the **hi plane alone**. That is a DECLARED
  arithmetic statement, not an approximation of one: the readout operand is the hi
  plane in every kernel (prefill's MMA operand has been `hi_bf16x2` of the
  accumulator since F1b), and decode now states the same thing about its own
  readout. The fp64 oracle REPORTS the difference; the gate asserts against the
  bf16-mirrored band, which is the dual-run protocol the oracle doctrine names.

The address is a BYTE offset now: `slot * Blocks<DV>::kBytes` under a page table, and
`(bh * atoms_per_bh + atom) * kBytes` dense -- the dense backing is the same
atom-major page sequence with every page resident, which is what makes the two
backings one addressing law rather than two.

PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the descriptor. `N`
comes from the level widths and must be a whole number of pages, which a lawful routing
makes it by construction; a ragged `N` is REFUSED at this seam rather than padded, and
decode carries no shape, tolerance or topology prefill does not. The dense plane is
`[BH, N, cols]` — the same page sequence with every page resident.
