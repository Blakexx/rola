# Instruction budgets: the method, the prices, the ledger

KERNEL_STANDARDS [§29](../../KERNEL_STANDARDS.md#s29) asks for every kernel's instructions to be budgeted by opcode
class before code. This page is how a budget is built, checked and corrected, and holds the two pieces of data the
method runs on: the PRICES of the primitives kernels are written in, and the LEDGER of stated budgets against measured
counts. A lab states its budget with `Bench::budget` (per MMA, or per declared unit with `Bench::units`); a production
kernel in `tools/budgets/<kernel>.json` ([phase_ledger.md](phase_ledger.md) reads those).

## Building a budget

1. Name the kernel's PHASES as its source does: each phase is a `//:` heading in the kernel body (a device function is
   its own phase). The ledger attributes instructions by these headings, so the budget and the code share one outline.
2. For each phase, its TRIPS per unit of work on each cell, from the experiment's mirror (its `.inputs.py`) or the
   facts it stands for -- a lane-divergent loop at the warp's longest lane, never the mean (§29.12).
3. For each trip, the PRICE: the primitives the phase is written in, each from the table below for the architecture.
   A primitive missing from the table is compiled alone and its SASS read (`tools/lab.py sass`) before it is priced;
   it then enters the table with the experiment it was read from.
4. The line item is trips x price, by class; the budget is the sum, stated before the code (`Bench::budget`).
5. A production kernel's budget also states, before the code, each component's peak live registers (`registers`,
   gated by `tools/life_ranges.py --budget`, [`life_ranges.md#budget`](life_ranges.md#budget)) and, per cell, each
   ledger phase's predicted cycles a warp a window (`phase_cycles`, gated by `tools/phase_ledger.py --budget`,
   [`phase_ledger.md#budget`](phase_ledger.md#budget)): KERNEL_STANDARDS §32.

A COMPONENT is a definition the region ledger's function map finds (`tools/region_ledger.py`): a function, a struct or a struct's method, nested structs included; a kernel whose device code is a class over its plan (the intra's `IntraAt<Walk<B>>`) keeps its components by their own names.

## Checking it

`tools/lab.py run <experiment>` gates the executed counts by class (ncu's per-opcode counts of the first launch). When
a kernel is over, `tools/lab.py ledger <experiment> [--kernel REGEX]` attributes the executed instructions to its
phases, so the miss is a phase and a price rather than a total. The attribution is ncu's SourceCounters of one launch
mapped to source lines (the lab builds carry `-lineinfo`); an inlined helper's instructions land on the helper's own
lines, so a helper is a phase of its own.

`tools/region_ledger.py` maps a production launch's samples the same way, over the one function `--kernel` names: a
member holding several (a carry arm holds the carry and its plan pass) is refused without it, since offsets restart in
every function and two functions' frames would overwrite each other's.

Every ledger entry is classified (the method `tools/region_ledger.py` applies to production kernels):
- (a) inherent: the design's own cost, near its floor;
- (b) the mechanism is right and its implementation above its floor -- fix to the floor;
- (c) work the cell does not need.

Instruction counts never stand alone: time, IPC and stall reasons go beside them (§21).

## Correcting it

A restated budget names the price that was wrong and corrects the table in the same commit; a restatement that
corrects no price is a guess replaced by a measurement, and teaches the next budget nothing (§29.13). The ledger below
records each lab's stated line against its measured count.

<a id="prices"></a>
## Prices

SASS instructions per use, sm_86 (ptxas of the pinned toolkit), with where each was read. A price is the instructions
the warp issues; a rate (cycles) belongs to `common/hardware.json` and its labs -- each pipe's, the shared-memory
pipe's by kind of traffic ([hardware.md#shared](../common/hardware.md#shared)) -- and a lab prints every pipe's cycles
from both ([pipes.md](pipes.md)).

| primitive | price | read in |
|---|---|---|
| a warp reduction (`__reduce_*_sync`) used by every lane | 2: `REDUX` + a move from its uniform register | `color_union` |
| a warp collective inside a loop ptxas cannot prove convergent | +1 `BRA.DIV` per collective | `color_union` |
| a conditional single-lane or selected-value store (`if (lane == 0)`, `in ? a : b`) | +3-4: `BSSY`, `BRA`, `BSYNC` | `color_union` |
| a lane-dependent select between two expressions (`lane & s ? f(x) : g(x)`) | both expressions, predicated | `color_union` |
| a counter bumped by a uniform index into k registers | k x (compare + select + add) | `color_union` |
| a 32 x 32 bit transpose by shuffles | 5 x (rotate + mask + `SHFL` + merge) = 20 | `color_union` |
| a shared-memory address through `tile_at`'s XOR swizzle with a runtime row | 3 | `color_union` |
| a lane-divergent loop | its body x the warp's longest lane | `color_union` |
| a shared-atomic claim on a digit, withdrawn on failure | ~10 per digit per pass (claim, compare, withdrawal, full mark) | `color_union` |
| ptxas's additions around `cp.async`: three never-executed `LDS` before each `LDGSTS`, the loop's exit `CALL`, alignment `NOP`s, `UIADD3` fillers between dependent MMAs | as counted | `read_image` |
| a 4-lane reduce-scatter of 4 values (lane `t` ends with value `t`'s max over its quad) | 12: 3 `SHFL` + 6 `SEL` (which value each lane keeps and sends) + 3 `IMNMX` | `post_keys` (production, 2026-11-01) |
| a per-slot fact handed to the slot's lanes (a lane forms it, a step reads it) | 1 `SHFL` a fact a step + 1 `IADD3` (the step's source lane) | `drain` (production, 2026-11-02) |
| a 64-bit address from a 32-bit word offset (`word_at`) | 2: `LEA` + `LEA.HI.X` (ptxas turns `mad.wide` by an immediate 4 into the shift-adds) | `drain` (production, 2026-11-02) |
| an f16 pair times one half (`ops::hmul2_by`) | 1: `HMUL2` with the half in its operand selector (`.H0_H0`); on a packed `splat_u16` operand +2 a broadcast (`PRMT` or `LOP3`, and `IMAD`) | compiled alone, and the dense readout's carry kernels (production, 2026-11-09; sm_80 not read) |

<a id="forms"></a>
## Forms

The cheapest known form for each recurring job, what it costs and where that was measured. A budget line names the
form it uses, or why it departs (KERNEL_STANDARDS §29.15); a form found cheaper replaces its row in the commit that
measures it. Pipe cycles are an SM's, from `common/hardware.json` ([pipes.md](pipes.md) prices them).

| job | the form | its cost | measured in |
|---|---|---|---|
| a maximum over values held in different units (scaled exponents) | integer keys: the fp32 bits plus the unit's shift in the exponent field (zero the least key, the shift clamped), compared as integers | an integer add and max a value, no conversion | `handoff` |
| a reduction across warps into a slot an item | one shared atomic (max, add) a lane at the item's slot, then one wide load of a lane's slots | 2 pipe cycles an atomic instruction; one `LDS.128` for four slots | `handoff`, `smem_rates` |
| an address at every access of a level | a base held for the level plus compile-time offsets (§29.3); a lane's constants held (`lab::held`) | 0-1 instructions an access | `read_image`, `handoff` |
| a lane's several words into shared memory | the widest store the layout allows | a 4-byte store 2 pipe cycles a wavefront, 8 or 16 bytes ~1.0-1.14 | `smem_rates` |
| global rows into shared memory | `cp.async.cg` | 0.5 pipe cycles a 128 B (`.ca` 5.4; a load into registers 1.6-1.7 and its registers in flight) | `smem_rates` |
| a stream's per-step scalars (coefficients, tokens, entries) | in the record riding the stage copy, read lane-ready | one 8-byte load a lane a step | `row_stream`, `read_stream` |
| a pad in a stream | not copied: its zero coefficient cancels the stale row | nothing | `row_stream` |
| fp32 to integer | one FFMA adding 1.5 x 2^23, a pair packed by one PRMT (§29.6) | 1.5 instructions a value, the FMA pipe | `read_image` |
| an accumulation into shared memory from many lanes, sm_80/86 | the 32-bit integer `red.shared.add` (their only native shared atomic add) | 2 pipe cycles an instruction | `read_image`, `smem_rates` |
| a branch on loaded data around collectives | a vote (`__any_sync`), a loop bottom-tested on it (§29.9) | no convergence barrier | `read_image` |
| a value every lane needs from a warp reduction | `__reduce_*_sync` | 2 instructions (its uniform-register move) | `color_union` |

<a id="ledger"></a>
## Ledger

Stated lines against measured counts, by lab. A miss over 1.3x means the table was wrong somewhere; its correction is
the entry's last column.

| lab | stated | measured | correction |
|---|---|---|---|
| `read_image` readout (instructions an MMA) | ~9 | 18.7 -> 8.2 | the §29.9 source patterns; ptxas's additions priced above |
| `color_union` serial (a box-window) | ~45 a token + 1.15K | ~64 a token + 2.6K | `REDUX` moves, uniform-index counters, `BRA.DIV`; the fixed part itemized |
| `color_union` batched (a box-window) | 1.8K / 3.7K / 1.1K | 3.8K / 7.8K / 1.6K | the warp's longest lane, swizzled addresses, both-sided selects, conditional stores |
| `handoff` 16-byte form, integer/address (a CTA-window) | 960 | 1,085 -> budget 1,088 | each scale's float bits are one instruction (16 a warp), priced first as address forms a held base removes |
| the carry's hand-off at nl64k-alt-k4 (a CTA-window: keys, exponents, stores) | 4,360 | 4,431 | the exponents 10 -> 14 a warp: the two parities' addresses, the warp's sync |
| the carry at nl64k-alt-k4 (a warp-window) | ~2,020 | 3,884 | open: every copy's token, source and swizzle re-derived; plan and window fields re-read at each use ([`carry_kernel.md#budget`](../carry/ampere/carry_kernel.md#budget)) |
| the producer at nl64k-alt-h4-g4 (a job tile) | 17.5K | 39.1K -> 29.5K -> 27.0K | the solve's member mask a value a pass and its selects (an ALU-pipe instruction each) removed; a u8 digit one FFMA (`F2I` and two multiplies before); open: the k-loop's 5.6K besides its HMMAs, the gain's serial loads ([`produce.md#budget`](../produce/ampere/produce_kernel.md#budget)) |
