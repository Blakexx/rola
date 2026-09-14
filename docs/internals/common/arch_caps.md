# `csrc/rola/src/chunk/arch_caps.cuh` — the capability rows

The ladder's per-architecture capability table, imported with
the chunk family: one `Caps` row per SM generation (async copy, tile loads,
MMA operand sourcing, TMEM, SMEM/register budgets), consumed by `design.cuh`'s
derivations. The sm_90 row builds and announces capabilities the 32-thread
consumer does not yet use (wgmma-class residency, TMA staging) — an
announcement, not a refusal, by the ladder's sm_90 ruling.

## Layer 1 -- the architecture capability facts

LAYER 1 -- THE ARCHITECTURE CAPABILITY FACTS.

THE THREE LAYERS.  This file is the first of them and it is the only one that
knows an architecture exists:

  1  CAPABILITY FACTS      this file.  Objective, per architecture, no
                           judgment.  Does the MMA accept an operand from the
                           issuing unit's registers?  Does the accumulator
                           live in TMEM?  How many bytes of shared memory does
                           an SM have?  Nothing here decides anything.
  2a DERIVED SELECTIONS    `design.cuh`.  What the design LOOKS LIKE on this
                           architecture -- state residency, staging depth --
                           each COMPUTED from layer 1 rather than looked up
                           per architecture.  A per-arch table of selections
                           would be an architecture branch in a costume.
  2b BASELINE OPERATIONS   `ops.cuh`.  HOW each step is performed, as
                           operations with contracts, plus the statement of
                           which implementations are actually built.
  3  THE KERNEL            reads 2a for its design and calls 2b to execute.
                           It touches this file never and names an
                           architecture never.

THIS COSTS NO SPECIALISATION.  Every fact below is a constant of the compile
target, so layers 1 and 2b collapse to a single arm per compilation; only the
FREE axes (the chunk size, the shared-memory layout) multiply the binary, and
those are measured rather than derived.  Nothing here is ever a template
parameter.

THE KINDS OF ENTRY.  A fact is one of:

  REQUIRED   a capability whose absence would change the STRUCTURE, not the
             implementation.  Never a branch: `supported()` goes false and the
             build stops with the clause that failed.
  SELECTING  a capability that implements a baseline operation more
             efficiently.  A compile-time bool.  Code branches on the FLAG,
             never on an architecture number.
  RESIDENCY  where an accumulator can live and where an operand can be
             sourced from.  These are INSTRUCTION facts under their honest
             names, and they are NOT read by any kernel: layer 2a turns them
             into `StateResidency`, which is what the algorithm actually
             requires.  Named as instructions at the level a kernel sees, a
             part whose accumulator lives in TMEM would be REFUSED for having
             the wrong storage class while meeting the real requirement.
  FACT       a quantitative resource (bytes, registers, threads).  NOT a
             branch under any circumstances.  It is an INPUT TO DERIVATION:
             residency, staging depth, the register budget.

THERE IS NO OTHER KIND, and in particular there are no fallback chains.  A
capability must never silently select a slower route that then ships
unnoticed; that is the difference between this table and a fallback chain.
Where a better implementation exists and is not built, layer 2a ANNOUNCES it
at compile time -- a documented gap a reader sees is a stated engineering
decision, and a silent slower path is the forbidden thing.

PER-SM FACTS LIVE HERE; PER-DEVICE FACTS DO NOT.  SM count is not an
architectural constant -- GA102 ships at 80 and 82 SMs, GA100 at 108 -- so it
is a runtime `cudaDeviceProp` query, never a table row.  Everything in
`Caps` is a property of the compute capability itself.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/common/arch_caps.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l163"></a>
### near line 163

The conjunction of the REQUIRED clauses.  There is no partial support.
`mma_a_from_regs` is NOT among them and must not be: whether the state can be
an operand without a memory round trip is a RESIDENCY question, answered by
`design.cuh` from the residency facts, and an architecture that answers it a
different way is a derivation rather than a refusal.

<a id="note-l203"></a>
### near line 203

THE PER-ARCH REGISTER BUDGET.  The largest per-lane register count that
still admits `target_ctas` CTAs of `threads` threads, honouring the
allocation quantum.  This is the number that must NEVER be a global
literal: at 256 threads and 3 CTAs it is 80 on every part with 65536
registers per SM, and the arithmetic that produced it is the table's, not a
measurement session's.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/common/arch_caps.cuh` condensed at their call site into a short pointer.

<a id="reg-transpose-b16"></a>
### `reg_transpose_b16`

an 8x8 b16 tile spread across the MMA unit's own
registers is transposed IN PLACE.  A clause of the
baseline because the adjoint's contraction runs
over the accumulator's OWN m axis, which puts the
accumulator on the k side, and the accumulator map
and the k-side map differ by exactly this
transpose (measured on the retired consumer's
catalogue; baseline = tag baseline/pre-k31).

<a id="mma-a-from-regs"></a>
### `mma_a_from_regs`

the A operand may be supplied from the issuing
unit's OWN registers.  Warpgroup MMAs keep a
register-source A (the `_RS` atom family), so this
holds on every tabulated architecture.

<a id="mma-b-from-regs"></a>
### `mma_b_from_regs`

the same for B.  A warpgroup MMA sources B ONLY
from a shared-memory descriptor.  Which SIDE a
given kernel's accumulator lands on is a clause of
THAT kernel, asserted against the built operation
at its own site.

<a id="accum-in-tmem"></a>
### `accum_in_tmem`

the MMA's accumulator lives in a dedicated tensor
memory rather than in the unit's register file
(`tcgen05`).  False on every row below; a Blackwell
row would set it, and the residency DERIVATION --
not a new refusal -- is what would then have to
hold.

<a id="mma-a-from-tmem"></a>
### `mma_a_from_tmem`

the A operand may be sourced from that same tensor
memory, which is what lets the invariant survive
with no register-resident state at all.

<a id="bf16-simd-arith"></a>
### `bf16_simd_arith`

elementwise bf16 arithmetic on a packed pair
(`mul.rn.bf16x2`).  MEASURED 2026-08-09: ptxas
rejects it below sm_90 -- "requires .target sm_90
or higher".  Ampere has native bf16 CONVERSION and
native bf16 MMA but no bf16 SIMD ARITHMETIC, so
the same product costs 1 FMUL + 1/2 pack per
element there against ~1/2 instruction on sm_90.
Same baseline operation, different implementation,
no structural change: the archetypal flag.

<a id="bf16-simd-fma"></a>
### `bf16_simd_fma`

FUSED elementwise bf16 arithmetic on a packed pair
(`fma.rn.bf16x2`).  MEASURED 2026-08-28:
this one IS available at sm_80, unlike the plain
`mul.rn.bf16x2` above -- the PTX ISA gates the two
differently -- so a PRODUCT is one instruction here
via a `-0.0` addend, which is bit-identical to
`mul.rn.bf16x2` (a single rounding of the exact
product; the addend's SIGN is what preserves a
signed zero, `+0.0` would flip `-0`).  Ampere thus
has no bf16 SIMD MULTIPLY but does have bf16 SIMD
FMA, which is the cheaper route to the same
baseline operation.

<a id="async-bulk-copy"></a>
### `async_bulk_copy`

TMA.  Would implement `stage_tile`'s contract with
one descriptor issue instead of one per 16 B.

<a id="warpgroup-mma"></a>
### `warpgroup_mma`

`wgmma`.  Present on sm_90 and the reason
`mma_b_from_regs` is false there.

<a id="dyn-reg-realloc"></a>
### `dyn_reg_realloc`

`setmaxnreg`: a warpgroup may GIVE UP or CLAIM
registers at run time, so two groups of the same
CTA can hold different per-lane allocations.
Without it, allocation is uniform per thread across
the whole kernel and sized by the MAXIMUM live set
over the paths a thread takes -- which is why a
thread that produces and then consumes pays
`max(producer, consumer)` and reuses the registers,
and why DEDICATING a unit to production cannot be
paid for by shrinking it.

<a id="mma-unit-threads"></a>
### `mma_unit_threads`

threads that jointly issue one MMA and jointly own
its accumulator.  32 (warp) or 128 (warpgroup).
The carve is expressed in THESE units, not warps.

<a id="driver-smem-reserve"></a>
### `driver_smem_reserve`

per-CTA reserve the driver takes off the top; the
residency condition is on smem_per_cta + this

<a id="tensor-f32-accum-pct"></a>
### `tensor_f32_accum_pct`

fp32-accumulate tensor rate as a percentage of
the fp16-accumulate rate.  50 on consumer
Ampere/Ada, 100 on GA100.  Not a branch: it is
why a tensor-bound baseline (FA) is flattered by
a consumer part and ratios must be quoted from
the target.

<a id="note-l125"></a>
### near line 125

Field order, for every row below:
  cc |  REQUIRED: async_copy, tile_load, mma_bf16_f32, bf16_cvt_packed,
                  reg_transpose_b16
     | RESIDENCY: mma_a_from_regs, mma_b_from_regs, accum_in_tmem,
                  mma_a_from_tmem
     | SELECTING: bf16_simd_arith, bf16_simd_fma, async_bulk_copy,
                  warpgroup_mma, dyn_reg_realloc
     |      FACT: mma_unit_threads, smem_per_sm, smem_per_cta_max,
                  smem_granularity, regs_per_sm, regs_per_lane_max,
                  reg_alloc_unit, max_threads_per_sm, max_ctas_per_sm,
                  driver_smem_reserve, tensor_f32_accum_pct

<a id="caps-of-note-l147"></a>
### in `caps_of`, near line 147

sm_90 is TABULATED AND NOT BUILT.  Every REQUIRED clause holds and
the residency derivation succeeds -- `wgmma` keeps a
register-source A operand (the `_RS` atom family), so the state
can be an operand without a round trip there too.  What is
missing is an IMPLEMENTATION: the built `mma`, `load_frag` and
carve are written for a 32-thread unit and a warpgroup unit is
128.  A capability GAP, not a structural refusal.

<a id="tabulated"></a>
### `tabulated`

The RATIFIED SET.  A compute capability that is not a row here fails the
build; it is never approximated by a neighbouring row.

<a id="regspercta"></a>
### `regs_per_cta`

Registers a CTA actually reserves.  Allocation is per warp in `reg_alloc_unit`
quanta, so a lane asking for 129 costs the same as one asking for 136.

<a id="min4"></a>
### `min4`

Resident CTAs per SM: the binding constraint among shared memory, registers,
threads and the hard CTA cap.

<a id="capsdigest"></a>
### `caps_digest`

A digest of the whole active row, written by a kernel and checked on the
host.  A binary compiled against a different table cannot masquerade as this
one -- no path or file hash can establish that, only a device-side fact.
