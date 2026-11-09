# `csrc/rola/src/intra/ampere/intra_kernel.cuh` — the intra kernel

Mirror doc for the kernel that computes every pair of tokens inside one window over the bundle as the producer wrote
it: each level's Gram as f16 MMAs on the rows as stored (a sparse level's u8 digits widened in registers, a dense
level's f16 rows as they are), the value product in f16 weights against the f16 values, the rows reduced into the
output planes the carry reduces into beside it. The carry's readout sees the state as of a window's start, so the
within-window pairs are this kernel's and only this kernel's (ruled 2026-10-31: the intra is fp16, on the planes as
stored). Its constants and operand block are [`../intra.md`](../intra.md), its host seam [`../intra_api.md`](../intra_api.md); the
arithmetic it is graded against is `intra_bundle_reference` in `tests/oracle/int_reference.py` -- the fp64 intra on the
same stored operands -- itself graded against the fp64 intra of the bundle's dequantized levels;
`tests/integration/test_intra_kernel.py` holds the kernel to it per slot within the declared roundings
([below](#numerics)).

Vocabulary, defined here and used unqualified below: `D` = routing depth, one or two (the arm); `w_l` = level `l`'s
width in digits (a whole number of 16-digit granules up to 256); `L` = tokens; `BH` = batch × query heads; `H_w` = write
heads; **window** = the device's W tokens ([`../../common/window.md`](../../common/window.md)); **tile** = 64 consecutive
tokens; **block** = the card's reader rows a CTA owns ([`../intra.md`](../intra.md#block)); **form** = a level's (read, write)
storage, u8 or f16 each; **stage** = 128 bytes of a writer tile's rows -- 128 u8 or 64 f16 digits of one level;
**slab** = 32 digits of one level, the gate's grain and two k-steps; **HMMA** = one `mma.sync.m16n8k16` on f16
operands, 2,048 multiply-adds.

## What it computes

For a window, a reader `t` and a writer `t' <= t` in it:

    G_l[t, t']  = Σ_d r_l[t, d] · w_l[t', d]                       level l, the rows' unit values
    A[t, t']    = m(t') · Π_l G_l[t, t']                           m the write gain times the write levels' scales
    den[t]     += Σ_t' A[t, t']
    num[t]     += Σ_t' A[t, t'] · s(t') · v(t')                    s the value scale, v the f16 values

A u8 digit's unit value is `d / 255`, an f16 dense coefficient's its own. `(m s, m)` is the facts pass's `fm`, a
writer's two factors read as one 8-byte pair. The READER's factor is not applied: it is per token and cancels in
`num / den`, the output contract the carry keeps too. A grouped call's writer side -- rows, values, `fm` -- is its
write head's, read by `H / H_w` consecutive query heads.

<a id="arms"></a>

## The arms, the forms and the widths

The arm is the depth, the accumulation and the call's evenness, `intra_kernel<D, kF16, kEven>`: `D` = 1 or 2 fixes the
gate's level loop; `kF16` the MMAs' accumulator, fp16 (1) where the device's fp32-accumulate rate is half its fp16 one,
fp32 (0) elsewhere -- the carry readout's rule
([`../../carry/carry_api.md#carry-readout-accumulate`](../../carry/carry_api.md#carry-readout-accumulate)); `kEven` whether
the call is a whole number of windows ([`#partial-window`](#partial-window)).
A level's form is runtime, one of four: the tile's stages are walked LEVEL-MAJOR (`tile_stages`), each level's under
one `rola::uniform_switch` on its form, so a stage's Gram knows its form at compile time and the four forms are four
straight-line bodies, not a test a k-step. The widths are the call's ([`../intra.md`](../intra.md#stages)): a level's stages
and slabs come from its width and its writer's storage. Templating the forms per level as well would be `4^D` bodies
per stage count -- ~80 straight-line Gram bodies at depth two -- for the dispatch's one switch a level.

<a id="walk"></a>

## The walk and the kernel

The device code is `IntraAt<Wk>`: the kernel's functions and types as members over the walk `Wk = Walk<B>`'s sizes,
their bodies the same at every build, and `intra_kernel<BuildTag<B>, D, kF16, kEven>` the grid's entry calling its
`run`. A build's arms compile in its image alone (`launch<B, D, kF16>` under `in_image`).

<a id="the-grid"></a>

## The grid: a CTA a piece, the longest first

A block `b`'s readers pair with writer tiles `0 .. (b + 1) R - 1` (`R` the block's row tiles). A CTA is one PIECE of
a block's sweep — at most `kPieceTiles` = 4 of its tiles, the diagonal pair always in the last piece
([`../intra.md`](../intra.md#pieces)) — longest first so the short ones fill in behind. `blockIdx.y` is the query head. Each
piece stages its block's readers and reduces its partial sums into the output planes, so a row takes a reduction
from each of its block's pieces. Measured on the int8 form (ncu, base clock): pieces of 4 tiles took `flagship-dense`
from 71 to 42 µs for 9% at `L = 64K`; on this kernel at `nl64k-dense` a piece of a whole 64-reader block (up to 8
tiles) runs 14% slower than pieces of 4.

<a id="partial-window"></a>

A call whose `L` is not a whole number of W -- its PARTIAL LAST WINDOW -- runs the arms `kEven` false of the one body
(`intra_kernel<D, kF16, kEven>`), compiled in their own translation unit (`intra_partial.cu`,
[`intra_partial.md`](intra_partial.md)) -- the way FlashAttention compiles its uneven lengths as arms of a template
flag. In it `tiles` = a window's tiles before the call's end
(the last one partial where `L` ends inside it), a piece's writer tiles are clipped to them, and a piece whose
readers or writers all lie past them exits before its first barrier. Every copy of a row past the call's last token
-- a writer row of a stage, a reader row of a part, a value row, a writer's factors -- zero-fills and reads nothing
(`Bases::lw`, `lr`; `Copier::lw`); a block the call ends inside lands every reader part in the prologue's groups, so
the first tile's steps carry no bound; the gate clears every pair with a tile past them and reads the union table a
word a tile (a head's row of unions need not start on 16 bytes), a writer's factors a run of 8 bytes for the same
reason; the deposit writes only the rows before the end. A writer past the end pairs only with readers before it,
above the diagonal, so the epilogue's mask takes it out; zero-filled, its factors and values are finite and the
masked products exactly zero.

Why arms, not a runtime test: the bounds compiled into every call cost the whole-window calls 2-4% on sm_86 at every
form tried (each copy predicated, the writer rows clamped, the reader parts moved to the prologue), ptxas
re-scheduling the Gram's steps around them; the arms `kEven` true are the body as it was, their SASS the same
instructions (one compare scheduled earlier outside the steps), their phase ledger the same at `nl64k-dense` (406.1K
against 406.4K cycles a warp a window). A partial call pays its bounds on every window, measured on no cell yet.

<a id="warps"></a>

A CTA is the card's warps ([`../intra.md`](../intra.md#block)): a warp owns 16 readers against `kWarpCols` of a tile's 64
writers -- all 64 on a 128-reader card (8 warps), 32 on a 64-reader card (two warps a reader group, 8 warps), so each
scheduler keeps two warps. A warp's rows and columns are its own for the whole sweep: its output is final in registers
at the end, partial over its columns where two warps share its rows, and both reduce into the same planes.

<a id="smem"></a>

## Shared memory

| tenant | bytes | holds |
|---|---|---|
| reader rows | `rows · pitch` | the block's reader rows, loaded once, every stage's slot in its reader's storage |
| `stage[3]` | 3 × 8 KB | three writer stages in flight |
| `values` | 8 KB | the writer tile's f16 values |
| `fm` | 512 | the writer tile's `(m s, m)` |
| `den` | `warps · 64` | a warp's 16 partial row sums, added across tiles |
| `gate` | `(W / 64) · R · 4` | the slab gate's table |
| `words` | 128 | each stage's word: the write and read side's first byte and limit, its level, reader chunk, reader part's width and slab bits (`build_words`) |
| barriers, counters, clock | ~0.5 KB | the values' full mbarrier and release counters, the phase clock |

The reader rows follow the fixed tenants (`reader_rows`), so the launch's dynamic shared memory is
`sizeof(Smem) + rows · pitch`. The reader region is dead after the sweep and holds the deposit's staging tiles; its
pitch is at least three stages and room for every warp's tile. At (256, 256) f16 on both sides a 64-reader block's rows
are 64 KB and the footprint ~99 KB: the sm_86 block is the most its 99 KB opt-in holds.

<a id="swizzle"></a>

## The swizzle

A row is 16-byte chunks; chunk `c` of row `r` sits at `c ^ swizzle(r)`, `swizzle(r) = (r & 3) << 1 | (r >> 2) & 1`. Its
low three bits are a bijection of `r & 7`, so the eight rows an `ldmatrix` reads at one chunk lie in eight bank groups;
and four consecutive rows' 32-byte pairs (`c`, `c + 1` for even `c`) lie in four distinct pairs, so a half-warp's
64-bit pair reads -- four rows, four lanes each -- are one wavefront. A copy's source pair keeps its 32-byte phase (the
swap of a pair's halves on rows 4-7 is the one move inside a pair that keeps a single sector request:
`cp-async` coalescing, measured 2026-10-04). The value rows, 128 bytes of 64 f16 channels, take the same swizzle;
`ldmatrix.trans` reads eight consecutive tokens at one chunk.

<a id="copies"></a>

## The copies

Every copy is `cp.async.cg` of one 16-byte chunk. THE STAGES AND THE READER ROWS are copied by every thread, its
share of the copy group of its stage: a stage's 64 writer rows of 128 bytes are two copies a thread (rows `tid / 8`
and `+ 32` at chunk `tid % 8`, `stage_share`), a reader part a thread's chunk `tid % 16` of rows `tid / 16 + 16 i`
(`reader_part`, its 4, 8 or 16 chunks a row the stage's digits at the read side's width). A thread's rows are formed
from bases made once (`Bases`: each level's write rows at the window's first token, read rows at the block's first
reader, and their strides), a copy their base plus one 32 x 32 -> 64 product of its token and the stride, and its
reader copies step by fixed strides. A copy is predicated, never branched around: a chunk past its level's bytes (a
16-digit tail's second half, a short stage's empty slots) has a zero source size, so it lands as zeros and reads
nothing; a stage no warp reads, or past the piece, the same; a reader chunk a part does not have is not issued
(`stage_tile_when`), since its destination is another part's. THE VALUES AND FACTORS are one warp's, a lane's place
fixed (row `lane / 8 + 4 i`, chunk `lane % 8`), arriving on the values' mbarrier with `cp.async.mbarrier.arrive.noinc`
when they land; a stage row `j + 4 i`'s swizzled chunk is `k ^ 2 j ^ (i & 1)`. The rows are read where the bundle
holds them, by their byte strides on 16-byte boundaries; the producer's rows are head-major, so a CTA's tokens are one
contiguous run a row.

<a id="stages"></a>
<a id="pipeline"></a>

## The pipeline

KERNEL_STANDARDS section 4's sm_80 form (ruled 2026-11-01 with the corrected SASS gate, which reads the step): the
writer side streams the piece's stages, level by level and tile by tile, through three buffers, each stage ONE COPY
GROUP of every thread. A step (`tile_stage`): `cp.async.wait_group 1` -- the thread's copies of this stage landed --
and one CTA barrier, which carries both facts: every thread's copies of the stage landed, and every warp done with
the stage before; then the stage's Gram with the refill under it -- the stage two on into the buffer just released
(`Ring`: the refill's stage, its word and its tile's gate word are read a step ahead), a thread's two rows a half after
each half of a straight-line Gram's slabs (`gram_dense`'s `refill`), whole after a gated or dead stage's -- committed as
the next group. The refill sits under the MMAs, not before them, because every warp's copies issued at the stage's
start delay the Gram that opens it: the lab's ring runs the dense Gram at 28.0 cycles an MMA with the refill first,
24.0 with it whole after the MMAs and 23.8 a piece after each slab, the u8 Gram 23.4, 22.9 and 20.0
([`intra_stream`](../../../lab/intra_stream.md)); in the kernel (sm_86) the halves took the Gram from 304.6K to 266.3K
cycles a warp a window at `nl64k-dense` (the intra -8.9%), 276.7K to 259.4K at `nl64k-alt-k4` (-4.2%), the refill
whole after the MMAs a percent behind on every cell. No mbarrier
in a step: a per-lane spin compiles to a convergence barrier, and an mbarrier address from a ring index ptxas cannot
prove uniform to a loop over distinct values (both measured on sm_80 the same day), and the SASS gate reds both.
THE READER ROWS land once, a stage's part in the copy group of the piece's first tile's same stage: the first two with
the first two stages before the loop, the rest with the ring's refills in the first tile (`tile_stages<..., true>`),
so the first Gram waits for one part, not the block's rows, and the later tiles' steps carry no reader code. A step
carries the part two stages on only where it is its own level's (`reader_part<D, F>`: its chunks a row known, every copy
live); each level's first two parts, which a step of the level before would carry, ride the first two groups instead
(`reader_part_any`), so a step's reader copies are never shaped at run time. A stage's
slabs and the tile's gate come from the stage words (`build_words`), read a step ahead. The tile's values keep their
own mbarrier, outside the steps: the last warp to release them copies the next tile's.

The protocol's measured shape (`nl64k-dense`, one head, sm_86, CUDA events at boost), before the step form: a per-stage
runtime loop with the form tested a stage, 676 µs; level-major with the form a level, ring counters for the divide by
three and the Gram's zeroing and fold out of the stage loop, 501 µs; the refill by the last releaser three on in place
of the first warp's claim one on, 513 µs; five buffers in place of three on `nl64k-alt-k4`, 471 against 457 µs; an L2
prefetch of the stage six on, 652 µs.

<a id="slab-gate"></a>

## The slab gate

A tile pair whose product is exactly zero is not computed, and a slab whose digits meet in no token pair is not
multiplied. The certificate is the facts pass's TILE UNIONS
([`../facts/liveness_contract.md`](../../facts/liveness_contract.md#tile-unions)): per side, 32-digit group and 64-token tile,
the digits live anywhere in the tile -- in a grouped table, a write head's query heads' read sides and then its write
side. Warp 0 reads, a lane a slab, the reader side's unions for the block and the writer side's for the window and
builds `gate[tile][row tile]`: bit `s` set when slab `s`'s reader and writer unions meet. Slabs are numbered across the
levels, and slab `s` of level `l` reads the union table's group `group0_l + s - slab0_l`. A level with no bit set makes
the pair's product zero, so the whole word clears, and a pair above the diagonal is zero too. A side's dense level
answers all-ones and is not read.

The table drives everything else, uniformly across the CTA: a stage is copied only when a row tile reads one of its
slabs, the tile's values only when a row tile's pair is live (the piece's first stages, copied before the table
exists, always); a warp computes the tile only when its pair is live and multiplies only the slabs it reads, the
form chosen once a level outside the steps: a level whose every slab the warp reads, its stages each one straight-line
block (`gram_dense`), otherwise a loop over each stage's live slabs whose body is one slab's loads and MMAs
(`gram_slabs`, the step, branch-free). A skipped slab's products are all zero and
a skipped pair adds exact zeros: `test_the_slab_gate_is_an_exact_certificate` holds the gated run to the ungated
run's sums.

<a id="gram"></a>

## The Gram

A slab is two k-steps of the warp's 16 readers against its `kWarpCols` writers (one `IMMA` k-step at a u8 x u8 level,
[below](#widening)): `load_slab` then `mma_slab`, each slab's loads issued before the slab ahead of it multiplies --
in the straight-line form slab by slab, in the slab loop two slabs a trip with the odd one peeled after
(KERNEL_STANDARDS §27.5). The K order is the level's form's:

* **f16 on both sides (natural):** the MMA's own order, `ldmatrix.x4` for both operands (A a k-step, B two n-tiles a
  k-step).
* **a u8 side:** `ldmatrix` gives a lane four consecutive digits of a row, `4t .. 4t + 3`, which no `ldmatrix` pattern
  gives an f16 fragment. The u8 fragment's order is the level's: slots `2t, 2t + 1` take digits `4t, 4t + 1`, slots
  `2t + 8, 2t + 9` take `4t + 2, 4t + 3`, and the f16 side is read in it -- a writer's by one 64-bit read an n-tile
  (both B words in one load), a reader's by two 64-bit reads a k-step, rows `g` and `g + 8` (words `0, 2` and `1, 3`
  of the fragment). Each is a half-warp's pair read, one wavefront a phase (the swizzle). Four 32-bit reads cost two
  each: a read's 32 lanes take the words at bytes `0` and `8` (or `4` and `12`) of a chunk, the sixteen even (or odd)
  banks, so eight rows hold each bank twice under any swizzle. The pair read lands the fragment's middle words
  swapped; the floor trick's rescale writes the A registers in order.

The level's Gram accumulates in fp16 pairs or fp32 by the arm, zeroed before the level's stages and folded after
(`fold_level`): the first level lifted to digit units by an exact power of two, each later one multiplied in.

<a id="widening"></a>

## The Gram by form, and the widening

Each level's Gram is the MMA its form's operands want (ruled 2026-11-01):

* **u8 x u8 (a level sparse on both sides, a tied level):** `IMMA` m16n8k32 u8 x u8 into s32, one an n-tile a slab
  (the slab's 32 digits its K) -- exact, no widening: the u8 fragments' K order (a lane's four consecutive digits) is
  the int8 MMA's own. The s32 Gram shares the f16/fp32 Gram's registers (`Acc`'s union) and becomes fp32 once a level
  in the fold (`gram_values`, outside the steps); its digit scale is 1. `Σ d d'` stays under 2^31 at any width the
  kernel reads (`256 · 255² < 2^24`).
* **a mixed level (one side u8, the other dense f16, floored at 2^-13 of its row's max):** f16 MMAs. A u8 word's four
  digits become two f16 pairs, exactly, in registers as they are used -- never in shared memory, which would double a
  u8 level's rows. At fp32 accumulation a digit enters as the f16 subnormal `d · 2^-24` -- the byte alone under a zero
  byte, one `prmt` a pair -- and the products are exact in the fp32 sums (the tensor cores keep f16 subnormal inputs).
  An f16 READER against a u8 WRITER (the u8 side widened every MMA) takes THE FLOOR TRICK at either accumulation: the
  writer's digits the same free subnormals and the reader's f16 times 2^15 once a k-step (`hmul2`, exact), so a
  product is `a d 2^-9` -- within `[2^-22, 2^-1]` by the floor, under 128 summed over a 256-digit level, its weakest
  products subnormal but at least 2^-13 below their row's strongest. The rescale also writes the A registers in
  order from the pair reads; at fp32 without it ptxas rebuilt each fragment by moves, ~two an MMA. A u8 READER against an f16 writer widens only A, once a
  k-step, and keeps `d / 16` (`prmt` under `0x54`, one `hfma2` less 64, exact): the scale under which a level's Gram
  stays within f16 at any support (`256 · 255 / 16 < 65,504`).
* **f16 x f16:** the natural MMA, no widening.

The host (`intra.cu`'s `units`) carries the scales: level 0's Gram is lifted by its form's digit scale (`16^n` or
`2^(24 n)` for `n` widened u8 sides, `2^9` for the floor trick, 1 for `IMMA`), and the rest of every level's digit scale
and `255^-n` ride `unit`. The fp16 widening of a u8 writer cost ~4 ALU instructions an MMA, which on sm_80's 8-cycle
MMA held the slab loop at 73-82% of the step line; the floor trick and `IMMA` take it to one `prmt` a pair or none.
`tests/integration/test_intra_kernel.py` holds every form at both accumulations per slot, its margins in the oracle
margins file (`--oracle-margins`, rows `intra num` / `intra den`).

<a id="epilogue"></a>

## The epilogue

At the tile's last level the product becomes `A`: each lane reads its columns' `(m s, m)` (one 16-byte read two
columns), and on the diagonal tile a column above its row is zero. The row sums `A m · unit` go to `den` (two quad
shuffles, one shared add a row), and `A m s` -- the value product's `P` -- replaces the product. `den` stays an fp32
sum: a ones column in the value product would sum `P = A m s`, the value scale riding num's weights only.

<a id="value-product"></a>

## The value product

Each reader's `P` at its own power of two over the warp's columns -- its largest in `[1, 2)`, an exact exponent shift
-- rounded to f16 and packed from the accumulator's registers into the A fragments (an n-tile pair's C words are a
k-step's A, FA2's relayout), the values by `ldmatrix.trans` of the token-major rows, a channel pair's MMAs into a fresh
fragment (fp16 or fp32 by the arm), and the sums back at the reader's power of two (times `unit`) into the output
registers.

<a id="numerics"></a>

## The numerics

Every Gram term is non-negative, so a rounding's error is relative to a partial sum that never exceeds the final one.
At fp16 accumulation each k-step's sum rounds once to f16: a level's Gram is within `k_l U16` of its value over its
`k_l` k-steps (`U16 = 2^-11`). The weights round to f16 (`U16`, or within half a subnormal of the reader's largest), and
at fp16 the value product's four k-steps round their sums. The gate bounds each slot by these against its ENVELOPE --
the same intra on `|v|` (`den` against itself) -- plus fp32 accumulation; a planted 1% error in one slot fails it.
Measured (`L = 1024`, two heads, three draws): at fp32 the worst slot uses 0.53-0.63 of its bound (the weights'
rounding, 3.9e-4 of the envelope at worst), `den` 0.007-0.04; at fp16 0.11-0.18 of the bound, the worst slot 3.2e-3 of
its envelope and `den` 2.3e-3 of itself.

<a id="deposit"></a>

## The deposit

After its last tile every warp passes one CTA barrier -- two warps share a reader group's rows, so the reader region is
dead only once all are done -- and stages its 16 rows × 64 channels in a staging tile of its own in the dead reader
rows, at a 72-float pitch (the pair stores bank free), then reduces each row into `num` as two 128-byte
`red.global.add.f32` runs, lanes on consecutive channels, and its 16 partial row sums into `den`. A RED, not a store:
the carry reduces into the same planes on the same stream, and the caller owns zeroing them.

<a id="phase-ledger"></a>

## The phase ledger

`rola_intra_ledger_bind` binds a `[8][7]` int64 buffer (warp × `reader`, `gate`, `wait`, `gram`, `epilogue`, `value`,
`deposit`) every CTA's warps add their cycles into (`rola.ops.intra.ledger_bind`); a bound ledger moves no output. A
phase is booked at the lap that ends it; no lap sits inside a step (a gated lap is a branch), so a tile's stages,
their waits and copies included, book `gram`, and `wait` is the tile's values' wait.

<a id="where-the-time-goes"></a>
<a id="region-ledger"></a>

## The line and where the time goes

The line (the plan card's DESIGN block for the fp16 intra), a head at `nl64k`, W = 512, (256, 256): 5.3M HMMAs, ~150 µs of
tensor time on sm_86 at fp16 accumulation (16 cycles a scheduler) and ~70 µs on the A100; ~139 µs of DRAM bytes on
sm_86; the line 1.3× the larger floor, ~200 µs on sm_86.

Measured (sm_86, one head, CUDA events at boost), fp16 / fp32 accumulation: `nl64k-dense` 505 / 577 µs,
`nl64k-alt-k4` 455 / 550 µs -- 2.5× the line. Neither memory binds (ncu at base clock: DRAM 161 MB a head at 24% of
peak, L2 hits 66%, the tensor pipe busy 28%), and a call whose 32 heads alias one head's rows, all of it L2-resident,
runs exactly as fast as 32 distinct heads. With the Gram skipped the rest costs 422 of the 513 µs.

What binds is the SM's copy rate beside the MMAs against the bytes an MMA takes. The region ledger
(`tools/region_ledger.py` against `tools/budgets/intra.json`) and the lab's `intra_stream` probe
([`../../../lab/intra_stream.md`](../../../lab/intra_stream.md)) agree: the Gram alone on resident stages runs at the tensor
rate, but an SM's copy path beside it delivers ~8 bytes a cycle on both cards whatever the ring's depth (3, 6 or 10
stages measured the same) or who issues the refills, and an 8 KB f16 stage feeds a 64-reader block only 128 HMMAs --
~16 bytes a cycle at the tensor rate. So the dense f16 Gram runs at about twice its floor by construction, a u8 writer
level (half the bytes a digit) near it. Fewer bytes an MMA -- more readers a writer byte, or narrower rows -- is what
lowers it; the readers' A fragments in registers, built to free the area for a deep ring, measured slower (683 against
505 µs at `nl64k-dense`, its depth-two arms spilling at 255 registers).

## Other architectures

On sm_80 the block is 128 readers (8 warps of 64 writers), the stage feeds twice the MMAs a byte, and the tensor floor
is half sm_86's. On sm_90 the writer stages move by TMA boxes and the Gram and value product are `wgmma` from shared
memory, the producer a warpgroup that gives its registers back to the consumers; the gate and the forms' K orders carry
over unchanged. sm_100's `tcgen05` takes the same shared-memory operands with the accumulator in tensor memory.

## Where the mechanism came from

The int8 intra it replaces (2026-09-29, rewritten 2026-11-01; [`DELETIONS.md`](../../DELETIONS.md)) walked the same
pieces, staged the block's readers once, streamed three stage buffers and gated slabs on the facts pass's tile unions;
those stay. What changed is the operands: the int8 form needed u8 rows on both sides and s8 value digits, which the
bundle stores for no dense level and no values since dense rows became native f16 (2026-10-04 write, 2026-10-07 read)
-- so this kernel reads the planes as they are, at f16 MMAs' twice the bytes a digit of a dense row and half the IMMA's
rate.
