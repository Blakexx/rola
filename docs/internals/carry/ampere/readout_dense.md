# The readout over a dense read side

Mirrors `csrc/rola/src/carry/ampere/readout_dense.cuh`. A dense read side lists no level: every token of a read chunk reads
every digit of the box. The value carve holds (`readout.md#warps`): a warp its 8 channels over every leaf of the box,
one token group of eight channel groups at the arm `[16, 2, 16, 2, 64, 1]` (both sides dense, the same split, the box
32 digits x K 16 x 64 channels: `plan.md#box`). Its output leaves through the chunk's panel and today's drain
(`panel.md#drain`), so a token exits once a box whatever the image's mode.

Symbols: `D` the box's digits (the read side's coefficient level), `K` its other level (the read rows, the MMA's K),
`a[t, d]` a token's coefficient at digit `d`, `b[t, l]` its row, `e_d` digit `d`'s image exponent (`handoff.md#maxima`),
`e_t` a token's exponent over the box.

<a id="row"></a>
## A dense read row

A slot's row is `kReadRow = 2 K + 2 D` bytes (96 at the arm): the token's K rows (the MMA's A), then its digits'
coefficients from `kCoefAt = 2 K`, both copied by the read fill (`parts.md#fill`, a dense side's two factor slices) on
the row's swizzle (an odd number of 32 bytes: `parts.md#swizzle`). The read buffer keeps, where a listed side keeps its
entries, two EXIT WORDS a slot (8 bytes: `schedule.cuh`'s `read_bytes`): the num's factor and the den's. A dense side's
union is its window's slots, no gaps: a slot is its token (`fill`'s and `drain`'s unlisted case). The facts pass checks
a read row's shape only for a listed read side (`facts.md`): a dense side has no entries to place in it.

<a id="factors"></a>
## The factors

`dense_factors`, once a chunk after its copies land: four lanes a token, eight tokens a warp, 64 a step over the CTA.
A lane takes 8 of the token's digits (one 16-byte load), each `y = a 2^(e_d)` exactly in fp32; the token's exponent is
`e_t = ceil_exp(max |y|)` -- the largest `e_d + ceil_exp(|a|)` -- by two shuffles across its quad; its coefficients are
rewritten in place as `c[t, d] = f16(y 2^(-e_t))` (so `|c| <= 1`), and the quad's first lane writes the exit words: the
num's `2^(e_t - kImageTop - kCoefTop)` (the state's coefficient top undone; at the f16 panel with the token's scale
byte's `2^(-scale - 1)` in the same exponent, so a tiny num under a large frame stays in range) and the den's
`2^(e_t)`; a token reading nothing writes zeros. Every term is a select (`pow2_sel`): a branch before a warp collective
is a convergence region a step, which the first forms paid (`read_dense`'s lab, below).

<a id="nest"></a>
## The nest

`read_dense<P, F32>`, a call's digits (the whole image or a half, `schedule.md#planner`): for each BATCH of 64 tokens (four
tiles of 16, their fp32 accumulators held) -> each TRANSIT GROUP of 4 digits (their slices in registers: K 16 x the
warp's 8 channels, an m16n8k16 B each, two digits an `ldmatrix.x4.trans`; 4 digits because the fold's 128 state
registers stay live beside the nest -- 8 spilled) -> each TILE (its rows once, one `ldmatrix.x4`; its two rows'
coefficients for the group, 8 bytes each) -> each digit: the rows scaled by the tokens' `c[t, d]` (four HMUL2, the
lane's rows `g` and `g + 8` each its half broadcast) and one MMA into the tile's accumulator. ONE fp32 ACCUMULATOR over
all `K D` leaves: 512 products of image values up to `2^10` reach `2^19`, past f16, so a dense side accumulates in fp32
on every card (sm_86's half rate is its floor). On GeForce that departs from the constitution's R17 (fp16 where fp32
accumulate is half rate), which the image's `2^10` top serves for a listed tile's 32 products: owed, fp16 sums kept
inside that bound, straight into the f16 panel. A batch's accumulators times each token's num exit word are added into
its panel words (the fill zeroed them; an image half's call adds onto the other's): `panel_word<128>` at f16,
`panel_word<256>` at fp32. A slot past the chunk is the chunk's last, read and never added.

<a id="den"></a>
## The den

`dense_den`, once a chunk over the whole den image (`handoff.md#den`; at 8 groups `den_block(d) = d >> 3`, so a block
is 8 consecutive digits): warp `w` takes the tiles `i = w mod 8`, apart from the nest (a den MMA under the nest's tile
loop would be a warp-uniform branch around an MMA, which the SASS gate refuses). A tile's den MMA a block on its
UNSCALED K rows, its 8 columns times `c[t, d]` and the digit's `2^(dE - kImageTop - e_d)` (its den factor, the
coefficient's scaling undone; the den state carries no coefficient top), the quad's columns summed by two shuffles,
times the token's `2^(e_t)`, into the warp's den word of the slot (the drain sums a slot's den words).

<a id="cost"></a>
## Cost

MEASURED (the part harness [`read_dense`](../../../lab/read_dense.md), sm_86, a 128-slot chunk over the image in halves,
cycles a warp-window): 21,578 against the tensor floor's 16,640 (fp32 accumulate: 32 cycles an m16n8k16 a scheduler),
1.30x; the nest alone ~19.1K (1.15x), the factors 1,444, the den 1,108. ACROSS CARDS: sm_80 and sm_86 run this form
(`mma.sync`, `ldmatrix`); sm_80's fp32 accumulate is full rate (8 cycles), so its floor is a quarter of sm_86's.
sm_90 and sm_100 take the same slices and rows; their asynchronous MMAs consume them from shared memory.
