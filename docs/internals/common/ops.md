# `csrc/rola/src/common/ops.cuh` — the device op helpers

The ladder's instruction-level helpers: `mma` (m16n8k16 bf16→fp32),
`load_frag`/ldmatrix wrappers, `cp_async` staging, `gather_ptr`/`pin_address`
(the pinned-base + zero-extended-32-bit-displacement addressing discipline the
SASS gate patrols), ballot/scan primitives, packed bf16 arithmetic. Overlaps
in role with this tree's `mma.cuh`/`cp_async.cuh`; the dedup is the e2e
cleanup step's, after which one helper family serves both consumers.

## Layer 2b -- the baseline operations

HOW EACH STEP IS PERFORMED.  Layer 2a (`design.cuh`) says what the design
LOOKS LIKE on this architecture; this file says how each of its steps is
EXECUTED, as operations with contracts.  Every entry is defined by WHAT THE
ALGORITHM NEEDS, never by what an instruction does: "stage this tile and tell
me when it is visible", not "issue a cp.async".  Get that level wrong and the
abstraction leaks on the first new architecture -- TMA cannot implement "issue
a cp.async", but it can implement `stage_tile`.

The kernel above this header names no architecture and issues no inline PTX.
It reads layer 2a for its design and calls this file to execute it.

This file is also where THE CONSISTENCY BETWEEN 2a AND 2b is checked (see the
section at the foot): a selection is invalid if no available operation can
serve it, and that is what the assertion says -- "selection unsatisfiable
given available operations", never "this architecture lacks instruction X".
Where the capability row offers a better implementation than the one built,
the gap is ANNOUNCED rather than taken silently.

### Operations and contracts

```
  smem_addr(p)              the CTA-local address of a shared object; the
                            address type every operation here consumes.
  stage_tile(dst, src)      copy `kStageBytes` bytes global -> shared without
                            occupying the issuing thread.  Membership of a
                            group is implicit: every stage_tile since the last
                            stage_commit.  Completion is SIGNALLED, not waited.
  stage_commit()            close the current group.
  stage_wait<n>()           all but the newest `n` groups have landed FOR THE
                            ISSUING THREAD.  CTA-wide visibility of another
                            thread's stage requires a following rendezvous();
                            this is a property of the contract, not of the
                            sm_80 implementation, and TMA/mbarrier satisfies
                            it the same way.
  rendezvous()              every thread of the CTA arrives; all prior shared
                            accesses by any thread are ordered before all
                            subsequent ones.  Also the WAR fence that lets a
                            ring slot be overwritten.
  load_frag / load_frag_t   four 8x8 b16 tiles from shared memory, landed in
                            the register order `mma` consumes.  `_t`
                            transposes each 8x8, which is what lets one shared
                            tile serve as an operand of either handedness.
  mma(acc, a, b)            acc += A(16x16) . B(16x8), bf16 operands, fp32
                            accumulator, accumulator PRIVATE to the issuing
                            unit, and BOTH operands read from that unit's
                            registers.  That last property is what serves
                            `STATE_RESIDENCY = REGISTERS`, which is layer 2a's
                            selection and not a flag; which SIDE a given
                            kernel's accumulator lands on is that kernel's
                            clause, asserted against this operation at its own
                            site.
  pack_bf16x2(lo, hi)       two fp32 -> the b32 that `mma` reads as two
                            consecutive k-elements of an operand.  This is
                            the accumulator-to-operand conversion; it is the
                            only reason the state never touches shared memory.
  splat_bf16x2(x)           one bf16 into both halves of a packed pair, so a
                            scalar can be an operand of `mul_bf16x2`.
  mul_bf16x2(a, b)          elementwise product of two packed bf16 pairs.
                            The archetypal SELECTING entry: sm_90 has one
                            instruction for it, sm_80/86/89 must go through
                            fp32 at 1 FMUL + 1/2 pack per element.
  gather_ptr(base, off)     the address of a row selected by an INDEX rather
                            than reached by an induction: a 64-bit base formed
                            once plus a 32-bit byte displacement, ZERO
                            EXTENDED.  The contract is the ARITHMETIC and not
                            the address -- writing `base + i` at element type
                            lets the compiler strength-reduce every access to a
                            widening multiply, which is precisely what the
                            addressing rule forbids, so the widening is stated
                            here once as an extension.  A displacement that
                            does not fit 32 bits is the caller's clause, and
                            the caller that has one states its bound.  Stated
                            by per-side compaction, its first caller: a gather
                            is an indirection and must not become per-element
                            64-bit arithmetic.
  pin_address(p)            a 64-bit address the compiler may not RE-DERIVE.
                            "Form the base once and afterwards only advance
                            it" is a rule about EMITTED code, and a compiler
                            under register pressure satisfies the source while
                            breaking the rule: it drops the live base and
                            recomputes it inside the loop, which is the same
                            widening multiply moved rather than removed.  This
                            is where the rule is stated to the compiler; it
                            costs the two registers the base occupies, which is
                            what keeping it live means.  Stated by per-side
                            compaction, whose gathered bases have no induction
                            to be carried by.
  bit_count(x)              the number of set bits.
  first_set(x)              the index of the lowest set bit, or `kBallotLanes`
                            when there is none.  With `bit_count` this is what
                            turns a lockstep vote into POSITIONS: the count of
                            set lanes below a lane is that lane's rank among
                            the live ones, and the first set lane of a "would
                            overflow" vote is where a run must be cut.  Stated
                            by per-side compaction's cook.
  lane_ballot(pred)         the predicate of every lane of the LOCKSTEP unit,
                            one bit per lane, in lane order.  Nothing is
                            written and nothing is ordered: the result is the
                            unit's own agreement about a property of the data
                            its lanes hold.  The lockstep unit is `kBallotLanes`
                            wide and is NOT the MMA unit -- a warpgroup issues
                            one MMA and still votes in lanes of 32 -- so the
                            two widths are stated separately.  Stated by the
                            ladder's token schedule, its first caller: a group
                            of tokens is live for a block when ANY of the
                            tokens the unit holds is, and a vote is how a unit
                            answers that without a memory round trip.
  fan_in_add(p, x)          add ONE contribution into an accumulator that
                            other units also contribute to.  The issuing
                            thread does not consume a result and must not wait
                            for one -- that is what makes this a reduction
                            rather than an atomic read-modify-write, and it is
                            the whole contract -- stated as `red.global.add.f32`
                            outright, because `atomicAdd` with an unused result
                            is only SOMETIMES lowered to a reduction: one
                            restructuring of the readout's scatter turned it
                            into returning atomics and a 2.5x step.
  No ordering across
                            contributors is defined and no contributor
                            observes another's value, so a caller that needs a
                            reproducible summation order needs a different
                            operation.  Stated by the ladder's capacity carve,
                            which is its first caller: the N/BC blocks of a
                            slot axis each compute a complete partial output.
  red_shared_add_f32(a, x)  `fan_in_add`'s contract with the meeting held in
                            SHARED memory: add one contribution into an
                            accumulator other units of the same block also
                            contribute to, return nothing, wait for nothing.
                            Stated by the readout's fan-in ring, its first
                            caller: a token's partials meet inside the CTA and
                            ONE reduction per token then leaves it.
  red_shared_add_u32(a, x)  the same over the counting monoid.  The ring's
                            arrival count is incremented once per contributing
                            warp-row and read by whoever wants to reclaim the
                            row; the incrementer never learns the total.
  cas_shared_u32(a, c, v)   the ONE shared word that is EXCHANGED rather than
                            reduced: the previous value is the answer and the
                            caller does wait for it.  Stated by the ring's
                            reclaim, its only caller: exactly one of the warps
                            that find a row stale may flush it, and a
                            compare-and-swap is what "exactly one" means.
  load_shared_f32(a)        one fp32 cell of a shared accumulator, ADDRESSED
                            rather than indexed, for the reason every other
                            shared access here is: a swizzled address is not an
                            affine function of an index.
  publish_fence()           the order between a unit's shared writes and the
                            flag that publishes them, at BLOCK scope and not
                            wider.  Stated by the ring: its producers and
                            consumers are warps of one CTA talking through
                            shared memory, so a device-scope fence would order
                            traffic no participant can observe.
  store_f32(p, x)           one fp32, at fp32 element addressing.  The
                            counterpart of `store_bf16` for a buffer whose
                            consumer is `fan_in_add` rather than an mma
                            operand: a summand that will be added to N/BC
                            others cannot be rounded to the operand format
                            first without the fan-in's error exceeding the
                            design's own.
  or_run<n>(p)              the bitwise OR of `n` contiguous bf16 in their
                            storage form, at the widest access that covers
                            them.  A magnitude test on the result answers "is
                            ANY of these nonzero" for the whole run at once,
                            because OR cannot clear a bit that was set.
                            Stated by the ladder's membership test, its first
                            caller: a block's span at a routing level is a
                            contiguous run of amplitudes and the test is
                            whether the run is entirely zero.
  store_bf16x2(p, v)        one packed bf16 pair, addressed as bf16 elements.
  store_bf16(p, x)          one fp32 -> one bf16 element, same round-to-
                            nearest-even as `pack_bf16x2` per element.  A
                            TRANSPOSED output accumulator holds two
                            token-adjacent values of ONE d_v column, and in a
                            [token][d_v] buffer those are different rows, so
                            the pair cannot be published as a b32.
  load_vec16 / store_vec16  one naturally aligned 16-byte value, moved in a
                            single access.  These are the SYNCHRONOUS pair of
                            `stage_tile`: 16 B is `kStageBytes`, the widest
                            access this baseline has anywhere, and a row that
                            is COMPUTED rather than staged must be published at
                            the same width the staged rows arrive at, or the
                            two would need different bank analyses of the same
                            buffer.  Stated by the ladder's routed coefficient
                            production, which is their first caller.
  copy_shared_to_global_16b one fully coalesced 128-bit store, shared -> global.
                            The composition of the two above, kept as its own
                            name because it is the whole of the epilogue's
                            contract and reads as such at the call site.
```

### <a id="reg-transpose"></a>The register transpose, and the candidate it replaced

`transpose_frag_b16(x)` transposes an 8x8 b16 tile where it already lies, across
the issuing unit's own registers. It exists for exactly one caller: the adjoint's
CHANNEL CONTRACTION (`backward.cuh`), whose `k` axis is `d_v` — the axis the fold
accumulator carries on its `m` side. The contract is about MAPS, not about data.

For `m16n8k16`, the accumulator spreads its two axes over the lanes as
`(m = t/4 [+8], n = (t%4)*2 + {0,1})` and a `B` operand spreads its two as
`(k = (t%4)*2 + {0,1} [+8], n = t/4)`. The two are the same matrix through a
transposed lane map, so an accumulator can be an operand of a contraction over its
own `m` axis only through this operation.

**MEASURED, not read off the ISA.** A standalone probe loaded a known 16x8 matrix
into a thread's registers using the accumulator map, fed it as `B` against
`A = I16`, and read the mma's own output back:

| candidate | verdict |
|---|---|
| the accumulator map already IS the `B` map | REFUTED: 112 of 128 elements differ, and the difference is exactly the transpose |
| `movmatrix.sync.aligned.m8n8.trans.b16` | 0 of 128 differ. Two instructions per (16 `d_v` x 8 slot) tile, on top of the `pack_bf16x2` the readout already pays |
| a shared-memory round trip through the padded buffers | not needed, not built |

The capability is a REQUIRED clause of the baseline (`arch_caps.cuh`) rather than a
selecting one, because the alternative is not a slower implementation of the same
operation — it is a different algorithm, with a `BCg x d_v` staging buffer that
does not fit beside the ring.

The SASS gate needs no carve-out for it: `MOVM` is not one of the signatures
`tools/sass_gate.py` reds on (local memory, convergence subroutines, predicated
HMMA, instructions-per-HMMA).

### Operations this kernel does not have, and why not

`warp_reduce` and a `publish`/`acquire` pair weaker than a full rendezvous are
absent, because nothing here needs them: the private d_v carve removes every
cross-lane reduction, and every buffer a producer publishes is read by the
whole CTA after a barrier that the ring already required.  There is no
`compact` either, and that is a statement about WHERE the compaction lives: a
caller that gathers tokens builds an INDEX LIST once and then addresses
through `gather_ptr`, so what it needs from this layer is the vote, the two
bit counts and the address -- not a data-moving primitive.  An operation
invented before it has a caller cannot be defined by what the algorithm needs,
because no algorithm needs it yet.

### What is deliberately not a capability

The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
layout knob in the capability table would make every future measurement
unfalsifiable.  It is a free axis, and `design.cuh` says why it stays one.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/common/ops.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kmmaunitthreads"></a>
### `kMmaUnitThreads`

THE BUILT `mma`'s UNIT: the threads that jointly issue one MMA and jointly
own its accumulator.  This is a property of the IMPLEMENTATION below, not of
any architecture -- the capability row states what the architecture's unit
is, and the consistency section at the foot of this file is where the two
have to agree.

<a id="gather-ptr"></a>
### `gather_ptr`

A GATHERED ROW'S ADDRESS: a base formed once, displaced by a 32-bit BYTE
offset that is zero-extended rather than multiplied out.  The extension is
written here explicitly because the arithmetic is the contract: at element
type the compiler is entitled to re-derive `base + index` as a widening
multiply per access, and it does -- measured at `30x IMAD.WIDE` in a loop that
the form below emits none for.  The caller owns the 32-bit bound.

THE BASE IS A GLOBAL ADDRESS, AND SAYING SO IS PART OF THE OPERATION.  An
address the compiler cannot see through is an address whose storage class it
cannot infer, and the difference is not cosmetic: a generic pointer turns the
fan-in's reduction into an atomic read-modify-write and every gathered load
into a generic one.  The assumption restores what the arithmetic hid.

<a id="atom-activity"></a>
### `atom_activity`

THE FACTS PASS' TWO ORTHOGONAL PER-ATOM BITS, and the one activity
predicate every body with state sweeps derives from them.  `kAtomWritten` says
this call's routing WRITES the atom; `kAtomRead` says it READS it.  They are
never pre-ORed into a single "active" flag by the producer: the read-only
distinction is what the touched-atom census and the continuation accounting are
about, and ORing saves nothing -- one byte per atom is read either way.

THE TWO PREDICATES, identical under the dense and the paged backing:
LOAD an atom iff it is RESIDENT (`page_slot >= 0`; dense = null table = all
resident) AND `(read | written)`; STORE it iff `written`.  Residency is the
arena's fact and says only WHERE an atom is; activity is the facts pass' fact
and says WHETHER it is touched.  So the two backings touch the identical atom
set, which is what makes `tests/integration/test_chunk_paging_equivalence.py`'s
`torch.equal` a certificate of the skip logic and not merely of the addressing.

A NULL `atom_bits` IS THE UNIVERSAL FACT, the same convention a null page table
keeps: a caller that supplies no activity fact asserts nothing about it, and
every resident atom is read and written.  That is the pre-P80b behaviour, and it
is what the reverse family's adjoint planes require -- an adjoint's activity is
the DOWNSTREAM call's fact, never this call's (docs/internals/carry/backward_kernel.md).

<a id="page-slot"></a>
### `page_slot`

THE STATE PLANE'S ONE BASE TRANSLATION (docs/internals/paging/paging.md).
`slot = page_tbl ? page_tbl[atom] : atom`, which makes the DENSE plane THIS SAME
kernel with a null table -- the slot IS the atom -- so paging is one base
translation and never a second path.  A NEGATIVE slot is an atom the plan
committed nothing for and is skipped at BOTH ends of the state I/O, exactly and
not defensively, because the plan's bitmap is the write set exactly.

THE PREDICATE IS CTA-UNIFORM, and that is what licenses skipping the barriers
inside the sweep along with the traffic: an atom belongs to exactly one owner
block, and the atom id every consumer forms is derived from `blockIdx` alone.

<a id="stage-run"></a>
### `stage_run`

THE SAME COPY AT A NARROWER QUANTUM.  `kStageBytes` is the widest the staging
issues; a run whose whole length is shorter than that is still ONE asynchronous
access, and a caller with a compile-time run length gets exactly one issue rather
than a padded 16-byte one that would read a neighbour's bytes.  The four- and
eight-byte forms are the cache-ALL variant because the sixteen-byte bypass has no
narrower member; both src and dst are naturally aligned, which is the caller's
contract and the reason no size is ever split here.

<a id="stage-tile-if"></a>
## `stage_tile_if` — a predicated staging quantum

`cp.async` with its source run predicated: a live quantum copies its sixteen bytes, a dead
one ZERO-FILLS its destination and reads nothing. That is what lets a partial tile — the
last segment of a window shorter than the window — be the same code as a full one instead of
a tail case, and what keeps it from reading past an operand's end to get there.

<a id="rendezvous-group"></a>
### `rendezvous_group`

`rendezvous_group(id, n)` -- a barrier over `n` threads that arrive at the
SAME named barrier `id`, leaving the other warps of the CTA unwaited.  The
contract is rendezvous()'s restricted to a warp-aligned subset; id 0 is the
CTA barrier's and is not offered here.  sm_80/86/90 all provide sixteen.

<a id="kballotlanes"></a>
### `kBallotLanes`

THE LOCKSTEP UNIT'S WIDTH, and it is deliberately not `kMmaUnitThreads`.  The
two coincide on the architecture built here and do not in general: a
warpgroup issues ONE MMA across 128 threads and still votes in lanes of 32.
A caller that conflates them would silently read a quarter of a vote.

<a id="lane-ballot"></a>
### `lane_ballot`

ONE BIT PER LANE, IN LANE ORDER.  The unit's own agreement about a property
of the data its lanes hold, reached without a memory round trip and without
ordering anything.  Expressed as the intrinsic rather than as a named
instruction: what the operation promises is the bit vector, not the vote.

<a id="bit-count"></a>
### `bit_count`

THE TWO READINGS OF A VOTE.  A bit vector answers "how many" and "where
first"; together they turn a lockstep unit's agreement into POSITIONS, which
is what a caller needs when the lanes are producing a compacted list rather
than a single yes or no.

<a id="lane-prefix-incl"></a>
### `lane_prefix_incl`

THE UNIT'S OWN RUNNING TOTAL.  `N` must be a power of two no wider than the
unit; lanes at or above it contribute whatever they hold and are the caller's
to mask.  Every lane executes every step, so the whole unit is converged at
each `__shfl_up_sync`.

<a id="load-frag"></a>
### `load_frag`

THE `memory` CLOBBER IS PART OF THE CONTRACT.  `ldmatrix` READS SHARED
MEMORY, so it has to be ordered against the stores that filled it: without the
clobber `asm volatile` pins the instruction against other volatile asm but leaves
the compiler free to move it across a `__syncthreads`/`__syncwarp`, and a warp could
then pull rows another warp has not published.  Every producer/consumer pair in this
family is stores -> barrier -> `ldmatrix`, so the clobber belongs here, once.  It is
stated as a CONTRACT and not as a fix: adding it did not change the `## R-B` layout
probe's open failure, and no miscompile has been demonstrated against it.

<a id="transpose-frag-b16"></a>
### `transpose_frag_b16`

AN 8x8 b16 TILE, TRANSPOSED WHERE IT ALREADY LIES.  The contract is about
MAPS and not about data: an accumulator spreads its two axes over the unit's
lanes one way and a `k`-side operand spreads its two the other way, so a
contraction over an accumulator's OWN m axis cannot read it without this.
MEASURED, not read off the ISA -- the probe and the refuted alternative are
in docs/internals/common/ops.md.

<a id="mma"></a>
### `mma`

THE ATOM IS CHOSEN BY SHAPE FAMILY, NOT BY NAME.  `kMmaM/N/K` is the shape
the carve and every fragment index are written in; an architecture whose
register-source family is shaped differently (Hopper's `_RS` atoms are
64 x {8..256} x 16 with a K-major register A) substitutes its own member of
the family, and the constraints that keeps -- M a multiple of 64, A K-major
-- are properties of the family rather than of a hand-picked instruction.

<a id="hi-bf16"></a>
### `hi_bf16`

THE ACCUMULATOR'S TOP HALVES, TAKEN AS A bf16 PAIR.  A bf16 IS the top
sixteen bits of an fp32, so a pair of accumulator registers becomes an MMA
operand with ONE `prmt.b32` -- byte selector `0x7632` picks bytes 2,3 of the low
source and 6,7 of the high one -- against `pack_bf16x2`'s conversion pair.

THE ARITHMETIC IS NOT THE SAME AND THAT IS DECLARED, NOT INCIDENTAL.  This
TRUNCATES where `pack_bf16x2` ROUNDS TO NEAREST EVEN, so the operand can differ
by one bf16 ulp (relative 2^-8 rather than 2^-9) and it is biased toward zero.
It is the K35 read path's declared rounding, chosen because the hi-plane hybrid
publishes exactly these bytes: the strip a warp writes and the fragment it would
have built on the fly are then BIT-IDENTICAL, so the hybrid is a placement change
and never an arithmetic one.  The oracle mirror carries the same truncation.
THE SAME TRUNCATION FOR ONE FLOAT.  The exchange places each leaf's channel on its
own, so a fragment element reaches shared memory as a 2-byte store rather than as
half of a packed pair; the value stored is bit-for-bit what `hi_bf16x2` would have
put in that half.

<a id="lo-bf16x2"></a>
### `lo_bf16x2`

The two floats' LOW halves packed the same way (`prmt` 0x5410): the residue beneath the
bf16 plane. The carry's exit sweep stages a state row's low plane with it, so the page's
lo row is written as whole lines beside the hi row `hi_bf16x2` stages.

<a id="splat-bf16x2"></a>
### `splat_bf16x2`

One bf16 into BOTH halves of a packed pair, so that a scalar can be an
operand of `mul_bf16x2`.  The ladder's coefficient production multiplies a
whole run of amplitudes by a single per-digit amplitude; the packed operand
is what makes that one operation rather than a scalar special case, on every
architecture including the one where `mul_bf16x2` is a single instruction.

<a id="splat-bf16x2-bits"></a>
### `splat_bf16x2_bits`

the same splat over a raw bf16 BIT PATTERN.  A run of amplitudes held in a
register array must never be ADDRESS-TAKEN -- one `&run[i]` puts the whole run
in local memory, whatever the index's constness -- so a caller holding the bits
splats them by value rather than by reinterpreting a pointer.

<a id="mul-bf16x2"></a>
### `mul_bf16x2`

THE ARCHETYPAL SELECTING ENTRY.  Every implementation satisfies ONE contract:
the elementwise product of two packed bf16 pairs, rounded to nearest-even in
bf16.  There are THREE routes and the architecture row picks one at the
primitive's definition (KERNEL_STANDARDS section 8), never at a call site.

THE PTX ISA GATES THE MULTIPLY AND THE FMA DIFFERENTLY, and that is the whole
reason this entry has three arms rather than two: `mul.rn.bf16x2`
requires `.target sm_90`, but `fma.rn.bf16x2` is available from sm_80.  So on
Ampere the product is still ONE instruction -- an HFMA2 with a ZERO ADDEND --
and the fp32 route below is needed by no built architecture at all.

THE ADDEND IS `-0.0` AND ITS SIGN IS LOAD-BEARING.  `fma.rn` rounds the exact
product plus the addend ONCE, so `a*b + (-0.0)` is bit-identical to
`mul.rn(a, b)` INCLUDING the signed zeros: `(-0) + (-0) = -0` and
`(+0) + (-0) = +0` under round-to-nearest-even, whereas a `+0.0` addend would
turn every `-0` product into `+0`.  Nothing downstream reads a zero's sign, but
an arithmetic identity that holds only for the values we happen to feed it is
not an identity, and the oracle mirror is written against the exact product.

THERE IS NO fp32 FALLBACK, because after the row above there is no
architecture that would take one: `tabulated()` is closed-world and every row
in it carries `bf16_simd_fma`.  An untabulated architecture fails the build,
which is the refusal this family wants -- not a slower path nobody measured.
WHAT THE DELETED fp32 ROUTE COST, MEASURED (K35 STAGE-LEDGER S2): the fetcher-side
expansion ran at 3.9x its instruction floor, and the two named lines were
`cuda_bf16.hpp:660` (`PRMT`, 1.27 M) and this function's `FMUL` pairs
(1.32 M) -- about four instructions per pair where one HFMA2 does.

<a id="pack-lo"></a>
### `pack_lo`

THE STATE'S LO HALVES, PACKED.  The top 16 bits of an fp32 ARE its bf16
truncation, so a state word splits exactly into the `hi` the exchange publishes and
the `lo` its owner keeps.  Two lows ride one register; `join_hi_lo` puts a word
back together bit-for-bit, so the round trip is lossless by construction.

<a id="store-shared-u16"></a>
### `store_shared_u16`

A SEQUENTIAL SHARED STORE, ADDRESSED RATHER THAN INDEXED.  This is the store
side of the hazard the gather comment above names: `dst[k++]` through a
`uint16_t*` re-derives `dst + k` as a WIDENING MULTIPLY every iteration, so a
loop whose work is one `STS.U16` per live bit measures `IMAD:114, IADD3:69` and
no `STS` in its top opcodes at all -- the schedule walk's scatter, 447 warp
instructions per (owner, window) for about two stores a lane.  Carrying the
shared address in a 32-bit register and advancing it by the element size makes
the per-iteration cost one add.

<a id="or-shared-u32"></a>
### `or_shared_u32`

ONE CONTRIBUTION INTO A SHARED BITSET, FIRE AND FORGET -- `fan_in_add`'s
contract over the bitwise monoid.  The per-tile liveness mask has one writer per
(token, stream) row and one reader per stream, so what it needs is a reduction and
never a read-modify-write: nothing waits and no lane learns what anyone else set.

<a id="publish-fence"></a>
### `publish_fence`

THE ORDER BETWEEN A UNIT'S SHARED WRITES AND THE FLAG THAT PUBLISHES THEM, and
nothing wider: the fold's producers and consumers are warps of ONE CTA talking
through shared memory, so the scope that has to be fenced is the block's.  A
device-scope fence here would order traffic no participant can observe.

<a id="store-bf16"></a>
### `store_bf16`

One element, rounded to nearest-even exactly as `pack_bf16x2` rounds each of
its two.  The single-element form exists because a transposed accumulator's
adjacent pair is adjacent in TOKEN, and a [token][d_v] buffer puts those in
different rows.

<a id="fan-in-add"></a>
### `fan_in_add`

ONE CONTRIBUTION INTO A SHARED ACCUMULATOR, FIRE AND FORGET.  The result is
not returned, and that is the operation's substance rather than a convenience
-- an implementation that hands back the previous value has to keep the
issuing thread waiting for it, and the whole point of a reduction is that
nothing waits.  Expressed as the intrinsic rather than as a named instruction
so that an architecture whose best reduction is wider, or is performed by a
different unit, is served by its own compiler; the SASS gate for this kernel
checks that what was emitted is in fact the non-returning form.

<a id="load-vec16"></a>
### `load_vec16`

The 16-byte quantum, synchronously.  `kStageBytes` is the width of every
asynchronous access this baseline issues, and these move the same width for
data the kernel PRODUCED rather than staged, so one bank analysis covers a
buffer whichever way its rows got there.

<a id="or-run"></a>
### `or_run`

THE OR OF A CONTIGUOUS RUN, AT THE WIDEST ACCESS THAT COVERS IT.  `n` is a
compile-time run length and the run is naturally aligned, so 1, 2, 4 and 8
amplitudes are one access each and longer runs are a whole number of 16-byte
accesses.  The reduction is bitwise because the CALLER's question is whether
anything in the run is nonzero, which needs no arithmetic and no ordering.

<a id="static-assert"></a>
### `static_assert`

`design.cuh` claims REGISTERS is a residency this baseline builds.  It is
built by an `mma` that reads an operand out of the issuing unit's own
registers, and by `pack_bf16x2`, which is the accumulator-to-operand
conversion that keeps the state out of shared memory.  If either went away
the claim would be false, and this is where that would be caught.

<a id="static-assert-2"></a>
### `static_assert`

THE CARVE'S UNIT.  Not a capability and not an architecture clause: `mma`,
`load_frag` and the d_v carve above are written for a 32-thread unit owning
a 16-row slice of the state, and a 128-thread warpgroup unit owns 64 and
issues a differently shaped atom.  So this is a consistency clause between
the design's carve and the operations that execute it, and its failure means
NOT BUILT rather than UNSUPPORTED.

<a id="note-l562"></a>
### near line 562

THE ANNOUNCEMENT SITES.  Each is a case where the capability row records a
better implementation of a baseline operation than the one built here.  None
of them is an error: the built path is correct everywhere it compiles.  What
is forbidden is taking the slower path SILENTLY.

`stage_tile` on a part with bulk asynchronous copy: TMA satisfies the same
contract with one descriptor issue instead of one per 16 bytes.

<a id="kstagebytes"></a>
### `kStageBytes`

Bytes moved by one `stage_tile`.  A FACT OF THE OPERATION, not of the
kernel: it is the widest single asynchronous copy the staging implementation
issues, and the staging loop's shape is derived from it.

<a id="kmmaacceptsregistera"></a>
### `kMmaAcceptsRegisterA`

The operand sources the built `mma` accepts.  A kernel whose accumulator
must land on a particular side asserts against THESE, at its own site: which
side is a property of that kernel and not of the baseline.

<a id="kstagingisbulk"></a>
### `kStagingIsBulk`

The built `stage_tile` moves `kStageBytes` per issue.  A bulk (TMA) form
would satisfy the same contract with one descriptor issue for a whole tile;
it is not built.  See the announcement in the consistency section.

<a id="t"></a>
### `T`

A BASE THE COMPILER MUST KEEP RATHER THAN RE-DERIVE.  The value is unchanged;
what changes is that it becomes the result of an operation ptxas cannot see
through, so rematerialising it is not available and the base stays live.

<a id="firstset"></a>
### `first_set`

`kBallotLanes` for an empty vector, so "nowhere" compares greater than every
lane and needs no separate case at the call site.

<a id="lastset"></a>
### `last_set`

THE OTHER END OF THE SAME VOTE, for a caller that consumes its lanes from
the high side.  `-1` for an empty vector, so "nowhere" compares BELOW every
lane and needs no separate case, exactly as `kBallotLanes` compares above.

<a id="knegzeropair"></a>
### `kNegZeroPair`

the packed pair of bf16 NEGATIVE ZEROS, which is the sign-preserving
identity element of `fma.rn`'s addend.

<a id="storesharedu32"></a>
### `store_shared_u32`

A SHARED WORD ACCESS, ADDRESSED RATHER THAN INDEXED for the same reason
`store_shared_u16` is: a swizzled address is not an affine function of an index, so
indexed form would re-derive a widening multiply per access.

<a id="redsharedaddu32"></a>
### `red_shared_add_u32`

THE SAME OVER THE COUNTING MONOID.  A stream's arrival count is incremented once per
contributing fetcher and read by whichever executor is waiting on the segment; the
incrementer never learns the total, so this is a reduction and not an exchange.

<a id="storesharedvec16"></a>
### `store_shared_vec16`

the 16-byte quantum into shared memory, addressed.  The segment's dead rows are
cleared at this width so a whole slot row is a constant number of stores.

<a id="loadsharedu16"></a>
### `load_shared_u16`

one amplitude out of a staged factor row, by ADDRESS.  Its index is a digit, so
the access is dynamic and an indexed form would re-derive a widening multiply.

<a id="storef32"></a>
### `store_f32`

One fp32 element, stored where it will later be read as a summand rather than
as an mma operand.  There is no rounding here BY CONSTRUCTION: see
`fan_in_add`.

<a id="kbf16magnitudemask"></a>
### `kBf16MagnitudeMask`

The magnitude bits of a packed bf16 pair.  A run of bf16 is entirely zero
exactly when the OR of its storage words has no magnitude bit set, since the
only zeros bf16 has are +0 and -0 and OR cannot clear a set bit.

<a id="koctets"></a>
### `kOctets`

§6: `N` is this function's template parameter (renamed from lowercase `n`
so the unroll-discipline lint reads it as compile-time); `N / 8` is still an
arithmetic expression the heuristic can't parse, so name it too.

<a id="stage-run-if"></a>
## `stage_run_if`

`stage_run`'s predicated form, the narrow-quantum twin of `stage_tile_if`: the copy issues
with its SOURCE RUN SIZE zeroed, so a dead quantum reads nothing and zero-fills its
destination. The fold's factor staging is what needs it — a level run is `span_l * 2` bytes
and only the flagship's is sixteen — and the zero fill is what makes a partial window
arithmetic rather than a tail case: a token past the window's end lands zero amplitudes, and
a zero amplitude is a zero coefficient.

<a id="stage-run-lanes"></a>
## `stage_run_lanes`

`stage_run`'s LANE-predicated form: the copy is issued by the live lanes and a dead lane issues
nothing, so its destination keeps what it holds. It exists for the calibration rows that asked
whether the fill's dead lanes -- landed zero-size on the pool's zero row, which is zero already --
cost the memory pipe: they do not on the sixteen-byte copies (calibration.md, the dead-lane rows:
36 wavefronts a copy against 5, 55.3 against 56.9 cycles), and ~60 cycles a copy on the four-byte
`.ca` ones, under the A/B's drift for the fill. The kernel's copies stay on `stage_run_if`, whose
zero fill the readout's ring rows past the live count rely on.

<a id="uniform-warp"></a>
## `uniform_warp`

The warp's index as a value ptxas knows to be warp-uniform: three ballots on the thread
index's bits (`popc(ballot(bit)) >> 5`), because `threadIdx.x >> 5` is a per-thread value
to its analysis and a loop bounded by anything derived from one is a divergent loop, after
which every shuffle is fenced and outlined (`carry/carry_kernel.md#uniform`).

<a id="warp-sync"></a>
## `warp_sync()` -- the warp's edge as one instruction

`bar.warp.sync 0xffffffff`. The carry's warp-private rings and staging tiles are ordered by
warp edges; `__syncwarp()` at a site the compiler cannot prove convergent (after a uniform
early return, a lane-predicated deposit) is lowered to a CALL into `__cuda_sm70_warpsync`,
a subroutine inside the tile loop. Stating the instruction keeps the edge a single WARPSYNC.

<a id="store-shared-f32x2"></a>
## `store_shared_f32x2(a, x, y)` -- an fp32 pair to shared memory

`st.shared.v2.f32` at an 8-byte aligned address: the readout's staging stores, where a C
fragment register pair is two consecutive channels of one token.

<a id="load-shared-v4"></a>
## `load_shared_v4(a)` -- one 16-byte quantum from shared memory

`ld.shared.v4.u32` at a 16-byte aligned address: the readout reads a tile's sixteen token ids
as two of these, every lane the same address (a broadcast, conflict free).


<a id="mbarrier"></a>
## `mbar_init`, `mbar_arrive_when_landed`, `mbar_arrive`, `mbar_wait` — a shared-memory barrier object

An `mbarrier` (sm_80+) completes a phase at `count` arrivals and flips its parity. A thread's
`cp.async` copies arrive when they LAND (`cp.async.mbarrier.arrive.noinc`: the arrival is one
of the counted ones, so the count includes it); a thread's own arrive releases its prior
shared stores; a wait acquires. `mbar_wait` is a `test_wait` spin: cheap when the phase is
already complete, which is the case the design arranges for. `mbar_test` asks once and
returns the answer, for a stream that does other work when its copies have not landed (the
carry's interleaved loop; the answer is voted over the warp before it steers a branch, so
the compiler sees a uniform predicate). The carry's pool uses a full/empty pair a slot
(`carry_kernel.md#pool`) so that warps drift freely by a chunk, where a CTA barrier would
hold every warp to the slowest.

<a id="atom-shared-add"></a>
## `atom_shared_add_u32` — a shared counter's take

`atom.shared.add.u32`, the old value returned: the carry's readout deals tiles off a shared
counter with it. Stated shared, so ptxas emits no address-space probe (the generic atomic's
lowering is `red-global`'s story).

<a id="once"></a>
## `once` — an opaque one-trip loop bound

`for (int n = ops::once(cond); n > 0; --n)` is a real branch where `if (cond)` on a
warp-uniform value would be if-converted and every warp would issue the body predicated off
(KERNEL_STANDARDS §20; the fill's round ownership ran as four passes of zero-size copies on
every warp until this). The bound is hidden from ptxas by an empty `asm volatile`, so the
loop cannot be recognized as 0-or-1 trips and flattened.

<a id="nth-set-bit"></a>
## `nth_set_bit` — the n-th set bit of a word

A branch-free binary search on popcounts, five steps. About thirty instructions; the
intrinsic `__fns` is a software loop that cost 21% of a dense window once. The carry keeps
it off its hot paths (the fold's slot map is a ring store and a load instead) and uses it
where a map of sixteen set bits to lanes is unavoidable.

<a id="cycles"></a>
## `cycles` — the SM's cycle counter

`%clock64` by `mov`, for the phase ledger (`carry_kernel.md#phase-ledger`); the intrinsic
`clock64` is not declared in every host-pass compilation of a header that both the launch
and the part harness include.

<a id="red-global"></a>
## `red_global_add_f32` — a global reduction stated global

The readout's rows leave `num` and `den` by reduction across owner CTAs. Written as a generic
`atomicAdd` on the kernel's pinned base pointer, the add lowered to a generic-space atomic: an
address-space probe, a shared-memory CAS spin and a local fallback beside a RETURNING atomic, whose L2 round trip the next branch then waited on -- 30% of a dense window's stall
samples sat on those branches. The explicit `red.global.add.f32` on the address converted to
the global window is fire-and-forget and carries no scoreboard.
