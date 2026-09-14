# `csrc/rola/src/intra/` — the K31 window-intra kernel

The within-window causal term, as its own deterministic side kernel. It computes
nothing the carry kernel computes: the carry kernel's readout sees the state as
of a window's start, and every pair of tokens INSIDE one window belongs here.
The design it implements is `development/plans/K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md`
§2.1 (the global window grid) and §2.5 (the factorized, windowed, tiled intra).

## What it computes

For a window `w` of `W` tokens (`W` and the level width `B` are the instantiation axes;
the built arms are `(2, 256, 384)`, `(2, 256, 512)` — the two ends of the architecture
doc's band — `(2, 256, 64)`, the conformance cell, and the deep pair `(3, 16, 64)` and
`(4, 8, 64)`. They pair with the carry family's arms, because the two kernels share ONE
global window grid),

    A[t, t'] = gain(t') * prod_l G_l[t, t'],   G_l = R_l · W_l^T over level l's B digits
    o[t]   += sum_{t' <= t in w} A[t, t'] · v[t']
    den[t] += sum_{t' <= t in w} A[t, t']

The gain is a per-COLUMN scalar applied AFTER the level product, which is the
shipped kernel's gain law: support bits are ungained. `den` is the same `A`
summed — the mass the ones-column of the oracle's recurrence produces — so the
kernel carries no second matrix for it.

`[W x W]` is never materialized. The window's `(W/64) x (W/64)` tile grid of `[64 x 64]`
output blocks is walked lower-triangular (21 tile-pairs at `W = 384`, 36 at `W = 512`);
the diagonal blocks carry the within-tile staircase. Per tile-pair the per-level Gram
accumulates in registers over the level's whole digit axis, the levels are
Hadamard-multiplied there, and the result becomes the A operand of the value MMA
**in place** — the accumulator-to-A fragment identity, no shared-memory round trip
on sm_80.

## The output block is C = 128 reader tokens

The unit of the walk is not the tile pair but the BLOCK: two row tiles, 128 reader
tokens, sweeping the column tiles together. One staged column tile — its `W_l`
operand slabs, its value rows, its gains — is consumed by both row tiles, so a
window of 8 tiles stages 20 column tiles instead of 36. What the block does NOT
change is the row side: each row tile's operands are still re-staged per column
tile. Cutting that needs two column tiles per block as well, which multiplies the
Gram tiles a warp must hold; the census already refuses two of them (below).

A window whose grid is a single tile has no pair to share with, so the block
collapses onto it and the CTA halves with it — `row_tiles_of(W)`, `rows_of(W)` and
`threads_of(W)` are one derivation, and the arm's tile count is what picks it.

## The carve, and why it has no reductions

One CTA per `(bh, window)`, one warp per `kWarpRows = 16` rows of the block —
eight warps at the paired arms, four at a single-tile one. A warp holds the Gram
of ONE tile pair and owns 16 output rows and every column; the row tiles are
shared ACROSS warps rather than stacked in one warp's registers, which is what
keeps the accumulator budget independent of `C`. Output rows are disjoint per
warp and the causal triangle's imbalance lands nowhere — neither between warps
nor between CTAs; the block's last column tile leaves the low half of the warps
idle for that one sweep, which is the strictly-upper pair not being computed.

The block's whole reader side is resident, so the footprint clears the 48 KB a
kernel gets unasked (65 664 B at `W = 512`) and the arms launch on opted-in
dynamic shared memory, re-declared per launch because the opt-in is per device.

Write-back is `fan_in_add` (a non-returning RED) into caller-owned `o`/`den`
planes, so the kernel is composable with a concurrent carry kernel and with
itself; the caller owns zeroing.

## The pipeline stage is as wide as the level allows

A stage is one `cp.async` fill and one pair of barriers, and several `kSlabDigits = 16`
MMA steps ride it, so barriers are per stage and not per MMA step. At `B = 256` a stage
is `kStageCap = 64` digits and four steps ride it: 16 barriers per tile-pair rather than
64. Measured on sm_86, narrowing the stage to the MMA step cost 1.66x on the dense cell.
A level that fits inside one stage IS one stage, which is what the deep arms are.

A LEVEL NARROWER THAN THE MMA'S CONTRACTION STEP IS CONTRACTED PADDED. There is no
`k < 16` tensor op, so at `B = 8` a level's slab is 16 digits of which 8 are real: the
tail is zeroed once at entry and never staged, and the Gram MMA runs at
`kSlabDigits / B = 2x` the arithmetic it needs on that level. It costs nothing else, it
is uniform (a no-op wherever `kSlabDigits` divides `B`), and the deep conformance cells
are its gate — a stale byte in the pad is a wrong Gram, not a crash.

Operand rows are padded (`operand_pitch(B)`, `kValuePitch`) so that the eight row
addresses `ldmatrix` reads per 8x8 tile land in eight distinct banks, while every
row base stays 16-byte aligned for `cp.async`.

## The slab gate

The one sparsity tier the study kept. Before the tile walk, the CTA summarizes
each `(tile, level, side)` as a `B`-bit digit union over that tile's 64 tokens. A
slab's MMA group issues only when the row side's 16-bit window AND the column
side's is nonempty; the staging for a whole stage is skipped when all four of its
slabs are dead.

THE UNION IS READ, NEVER RE-DERIVED. The support is a producer artifact: the entmax
solve that made the amplitudes freezes the same exact-nonzero test into a
32-token-blocked word (`entmax/factor.cu:219-241` — one bit per `(token, digit)`, bit
`token & 31`, digits contiguous), and `factor_backward_kernel` already unpacks that
word rather than re-testing, which is why the producer's two directions cannot
disagree about the support set. The intra takes the same word, one per side, in the
plane's own digit order: `sread`/`swrite`, `[BH, ceil(L/32), D*B]` int32, so a word's
column IS the plane's digit.

That layout is a zero-transpose fit for a TILE-grain union. A 64-token tile is
EXACTLY TWO WORDS per digit, so the union is `(w0 | w1) != 0`; the work axis is
(tile, certificate word) and the unit is a WARP holding 32 CONSECUTIVE DIGITS, one per
lane, which makes each of the two loads one 128-byte line and makes the `__ballot` of
the per-lane test the certificate word itself — no atomic, no cross-lane reduction, no
transpose. A level narrower than one 32-digit word is ONE unit and a wide level
several, the same loop at every width; digits past `B` (a padded level's tail)
contribute a cleared bit, as they must.

The bytes are the point. Re-deriving the union scans the tile's bf16 AMPLITUDES —
64 tokens x `B` digits x 2 B = **65.5 KB per (tile, level, declared side)** at
`B = 256`, which at `L = 65536` measured as **67.1 MB of the intra forward's 574.2 MB
DRAM read**, the whole previously-unattributed 12%. The word answers the identical
question in `2 x B x 4 B` = **4 KB**, a 16x cut on that term, for a signature change
and no new format.

A producer whose solve is shared across a level's two duties hands the SAME word to
both sides. The `RouteDescriptor` contract allows a set bit to accompany a stored
zero, and such a bit costs one gated MMA group whose product is exactly zero — it can
move no number, which is the same argument that makes the gate a certificate.

A side the launch's mode word does not declare sparse is never summarized and
answers all-ones, so the gate self-deactivates at dense routing rather than
becoming a second code path. The modes are a LAUNCH ARGUMENT, not a compile-time
pattern axis: `DENSE_BOTH`, `READ_SPARSE`, `WRITE_SPARSE`, `BOTH_SPARSE` (union
routing, which runs on exactly this machinery).

The gate is an exact zero certificate, so it cannot move a number — the battery
asserts bitwise equality between a gated and an ungated run of the same operands.

## Where the time goes (sm_86, measured)

The Π→Σ collapse that makes this term cheap in FLOPs also collapses its
arithmetic intensity. Per window the Gram issues `W^2 · sum_l B_l` flops against
`4 · W · sum_l B_l` operand bytes, so the ideal intensity is `W/4` — 128 flop/byte
at `W = 512`, against a 3080 Ti's 94 flop/byte compute/bandwidth balance. Ideal
intensity is therefore NOT what binds; the tile walk's operand RE-READ is, and it
grows with `W`, not away from it.

Both levers are pulled. `W` is an instantiation axis and the flagship is 384–512
(the architecture doc's sizing law); the output block is `C = 128`. The re-read
each addresses is a tile-staging count against the window's distinct tiles — at
`W = 512`, 16 distinct tiles against 72 stagings before the block and 56 after
(the row side's 36 are untouched, the column side's 36 became 20).

Measured at `W = 512`, `BH = 1`, `L = 65536`, dense (`ncu`, four launches, and the
free-running wall beside it):

| | dur ms | wall ms | DRAM read MB | vs ideal | tensor | occ | long-scoreboard |
|---|---|---|---|---|---|---|---|
| one row tile per sweep | 1.322 | 1.33 | 646.7 | 4.5x | 17.2% | 13.2% | 18.8% |
| `C = 128` | 0.938 | 0.74 | 507.2 | 3.6x | 26.0% | 16.6% | 10.1% |

The ideal is 142.7 MB — the two operand planes, the value plane and the gains,
read once. Traffic fell 1.28x and the wall fell 1.8x, because the block also
halves the column stagings and the barriers per unit of issued MMA: the same
Gram work now rides half as many `cp.async` fills on the column side. The term is
no longer bandwidth-dominated at the dense cell.

What remains is the ROW side's 4.5x, and what would cut it is more tile pairs
per warp — which is the budget the census refuses. Stacking just the block's two
row tiles in one warp's registers (the four-warp form of the same block)
assembles 255 registers with 72 spill stores and 17 in-loop local accesses at
`W = 512`; two column tiles as well would want twice that again. The shipped
block, sharing the row tiles across warps instead, is register-clean at 168
registers (`W = 512` and `W = 384`), 96 (`W = 64`), zero spill, zero local, on both
ratified architectures.

The CTA is register-bound to one per SM (256 threads x 168 registers), so a grid
of 128 CTAs — `BH = 1` at `L = 65536` — runs 1.6 waves and pays a tail. Cells short
enough to be tail-dominated show it: the cohort-64 cells run 0.42/0.46 ms against
0.39/0.43 before at `BH = 1`, while the same cells at `BH = 8` run 2.85/3.10 ms
against 3.34/3.90. Dense and scattered are faster at both.

The slab gate, measured, is a CLUSTERING mechanism and not a sparsity one: at
`k_tok = 4` SCATTERED the kernel runs 1.09 ms against dense's 0.94 (near
indistinguishable), and at cohort 64 it runs 0.53 ms with DRAM read down 2.5x.
That is the same fact the ballot-intersection tier's `k_tok ~ 7` crossover
records, seen from the built side, and it is why that tier is not built.

READING THE CERTIFICATE INSTEAD OF REBUILDING IT is worth a fixed ~65 MB of the
read, at `W = 512`, `BH = 1`, `L = 65536` (`ncu`, one settled launch, paired against
the scan form on the same draws):

| cell | DRAM read MB | `ncu` dur ms | free-running median ms |
|---|---|---|---|
| `k_tok = 4` scattered, scan | 583.6 | 1.040 | 0.820 |
| `k_tok = 4` scattered, word | **511.1** (−72.5, −12.4%) | **0.963** (−7.4%) | **0.756** (−7.9%) |
| `k_tok = 4` cohort 64, scan | 205.9 | 0.483 | 0.372 |
| `k_tok = 4` cohort 64, word | **142.6** (−63.2, −30.7%) | **0.407** (−15.8%) | **0.298** (−19.8%) |

The cut is an ABSOLUTE constant in the bytes, not a fraction: the scan reads a tile's
whole amplitude rows whatever the routing puts in them, so both cells shed the same
~65 MB and only the denominator differs. The wall follows at about 0.5–0.6 of the byte
ratio, not 1:1 — the term is 62–65% of DRAM peak but its #1 issue stall is
`math_pipe_throttle`, so a byte saved is not a cycle saved. The dense cell moves by
nothing measurable, as it must: no side is declared, so no certificate is built at all.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/intra/intra_kernel.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l31"></a>
### near line 31

THE OUTPUT BLOCK IS C = `kTile * row_tiles_of(W)` READER TOKENS.  The column
operands staged for one column tile serve every row tile of the block, which
is what takes the tile-pair grid's operand re-read down; a window with a
single tile has no pair to share with, so the block collapses onto it.

<a id="kwarpspertile"></a>
### `kWarpsPerTile`

ONE WARP PER `kWarpRows` OF THE BLOCK: a warp holds the Gram of ONE tile pair,
so the block's accumulator budget per thread does not grow with C -- the row
tiles are shared across warps, not stacked in one warp's registers.  The CTA
width is therefore a function of the arm, like every other tile count here.

<a id="kstagecap"></a>
### `kStageCap`

A LEVEL'S STAGED WIDTH is its digit count rounded up to the MMA's contraction
step: a level narrower than one step is contracted PADDED, because there is no
`k < 16` tensor op.  The pad is zeroed once and never staged, so it costs MMA
throughput on that level and nothing else -- exactly `kSlabDigits / B` when
`B < kSlabDigits`, and nothing at all otherwise.

<a id="kstagecap-2"></a>
### `kStageCap`

THE PIPELINE STAGE IS AT MOST AS WIDE AS THE LEVEL.  A stage is one cp.async
fill and one pair of barriers, and several `kSlabDigits` steps ride it, so the
barrier count per tile-pair is the stage count and not the MMA-step count.  A
level that fits inside one stage IS one stage.

<a id="alignas"></a>
### `alignas`

THE OPERAND SLABS ARE PER-TILE-PAIR AND DO NOT GROW WITH THE WINDOW: only the
zero-certificate masks do, at one word per (tile, level, side, 32 digits).
The block's whole reader side is resident, which puts the footprint past the
48 KB a kernel gets unasked, so the arm launches on opted-in dynamic shared
memory.

<a id="build-masks"></a>
### `build_masks`

One bit per digit, unioned over a tile's 64 tokens -- the exact zero
certificate the slab gate reads.  A side the mode word does not declare
sparse is never summarized and answers all-ones, which is what makes the
gate self-deactivating rather than a second code path.
-- docs/internals/intra/intra_kernel.md#the-slab-gate

<a id="kfill"></a>
### `kFill`

THE PAD IS ZEROED ONCE, AND ONLY WHERE THERE IS ONE.  A level narrower than
the MMA's contraction step leaves the slab's tail unstaged, and a zero there is
what makes the padded contraction equal the `B`-wide one.  A level that fills
its staged width has no tail, so the fill is not merely skipped at runtime --
it is not compiled, because the registers it costs are live across the whole
tile-pair loop.

<a id="tilesof"></a>
### `tiles_of`

THE WINDOW IS AN ARM, so the tile counts it induces are functions of it and
never file-scope constants -- the carry's `W` template argument, mirrored.

<a id="operandpitch"></a>
### `operand_pitch`

The operand rows are padded so the eight row addresses ldmatrix reads per
8x8 tile land in eight distinct banks; the pad is in elements and keeps every
row base 16-byte aligned, which cp.async requires.

<a id="intraparams"></a>
### `IntraParams`

`sread`/`swrite`: the producer's frozen support word for the plane of the same
name, `[BH, token_words, row_width]` int32, bit `token & 31` of word
`token / 32`, so a word's COLUMN is the plane's digit.

<a id="koctets"></a>
### `kOctets`

§6: `N` is this function's template parameter (renamed from lowercase `n`
so the unroll-discipline lint reads it as compile-time); `N / 8` is still an
arithmetic expression the heuristic can't parse, so name it too.

<a id="lane"></a>
### `lane`

no zero-fill and no leading barrier: every DECLARED (tile, level, side, word)
is written once by one lane, and `slab_bits` answers an undeclared side from
the mode word without reading the array.

<a id="near-line-185"></a>
### near line 185

One bit per row tile of the block: the pairs of this stage with work in them.
A strictly-upper pair is never set, which is how the causal grid is walked.

<a id="krowtiles"></a>
### `kRowTiles`

§6: row_tiles_of(W) is compile-time (W is this function's template parameter)
but the heuristic can't see through the function call, so name it.

<a id="kcoldigitoctets"></a>
### `kColDigitOctets`

§6: kColDigits/8 and kStage/16 are compile-time but the heuristic only reads a
bare identifier, so name them.

<a id="near-line-218"></a>
### near line 218

a level's REAL digits only.  Where `B` is narrower than the MMA's contraction
step the slab's tail was zeroed at entry and is never refilled, which is what
makes the padded contraction exact.

<a id="kcolchanneloctets"></a>
### `kColChannelOctets`

§6: kColChannels/8 is compile-time but the heuristic only reads a bare
identifier, so name it.

<a id="deposit"></a>
## The deposit, at the line law

A warp's output block is `kWarpRows` rows by `kValueWidth` channels plus each row's mass,
final in the MMA accumulator after the block's last column tile. It leaves through the
warp's staging tile -- `kWarpRows` rows of one 32-channel segment, the mass, and a pad to
`kStageStride = 40` floats, so a fragment's 64-bit pair stores are bank free and a row is
contiguous -- a segment at a time: the segment's four n-tiles stored, the warp edge, then
one `red.global.add.f32` per row covering its 32 consecutive channels, one 128-byte line an
instruction, and the sixteen masses one line more. The tile is laid in the operand slabs,
dead by then: the block's last stage ended on a CTA edge after every warp's last MMA and the
next block's fill begins on one. The fragment-direct reduction it replaced touched eight
lines an instruction, the form the carry measured and rejected (`carry/carry_kernel.md#readout`).
The intra's cost is not here (1.6% of the tensor peak at `W = 512`); its rebuild on the
carry's form is owed.
