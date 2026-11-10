# `carry/ampere/readout.cuh` — the value-carve readout

Mirrors `csrc/rola/src/carry/ampere/readout.cuh`. The readout of a LISTED read side, a VALUE CARVE: every warp reads every
digit of the box and every leaf of it, for its own channels and its own tokens -- where the fold is a state carve (its
warps own lists, `fold.md`). The fold hands its state over through the image (`handoff.md#image`); the readout's
output goes to the panel and from there to `num` and `den` (`panel.md`). A side with no list level (the dense
readout: its nest of accumulator groups, operand scaling) is the next component, and `Plan` refuses an arm whose
`read_factors` is not 1.

THE OUTPUT: each token's partials over the CTA's box, `P(t, c) = sum_{d, r} x(t, d) a(t, r) S(d, r, c) / kappa` --
`S` the state before the token's window (digit `d`, list `r`), `x` the read coefficient of the digit as stored (a
sparse level's byte / 256), `a` the read row of the list level as stored (f16), `kappa` the state's byte scale (the
fold's operand an f16 dense level, its coefficient on the byte scale: 1/255, `rola.ops.prefill.carry_factor`). The
read side's per-token scales and `kappa` are common to `num` and `den` and cancel in the epilogue. Gated per slot
against the fp64 reference of this arithmetic (`tests/oracle/readout_reference.py`'s `readout`,
`tests/oracle/test_carry_readout.py`): both orders of the alternating box and both same-split boxes, both accumulate
forms, both reduce dtypes, each image mode.

<a id="warps"></a>
## The warps

At the 32-channel box the CTA's 8 warps are 2 TOKEN GROUPS x 4 CHANNEL GROUPS of 8 channels (`kTokenGroups`,
`kChannelGroups`; the planner's `rola.engine.carve.token_groups` and `channel_groups`): warp `w` is channel group `w
mod 4` of token group `w / 4`. A channel group is one MMA's N; a token group a chunk's slots on one side of
`group_split` -- a multiple of 64, the granule the facts pass's prefix rows count a list's tokens at, so a list's run
splits where its prefix rows hold the count below the split (`Run::n0`, `parts.md#entries`). A chunk of at most 64 slots
is token group 0's alone. The groups' slots are disjoint, so each group has its own panel rows and den words and no
two warps add into one word: the panel needs no atomics (sm_80/86's shared atomic add is integer only). All 8 warps
read the same digit's slices at the same time, so the image is read by the CTA as one stream, a digit at a time.

At the 64-channel box (32 lists x 16 digits x 64 channels) the warps are 1 token group x 8 channel groups: every warp
takes all of a digit's run (`read_run`'s whole count; no split, so no 64-slot imbalance between groups) for its 8
channels, the den of its group's `n_d` contiguous digits (`handoff.md#den`). The tile, its K of 32 lists and its two MMAs are the 32-channel
box's.

<a id="factors"></a>
## The entries' factors

`read_factors`, once a chunk: after the chunk's copies land (a warp's own wait) and before the rendezvous that opens
its readout, each warp rewrites the entries of its own digits `warp + 8 j` -- the runs its fill copied
(`parts.md#entries`), so no other warp's copies are read and no barrier is added -- in place: its KEY (`entry_key`), at
32 channels the row's byte offset in the chunk's pool (its row offset in 16-byte units taken back by the chunk's first
slot, times 16: `64 s + 16 ((s >> 1) & 3)` for slot `s` at the 64-byte row, under 2^16 since a chunk holds at most a
window's 1,024 slots; the f16 panel row's base too, `panel.md#words`), at 64 channels the slot's f16 panel base in
16-byte units, `8 s + (s & 7)` (13 bits at W = 1,024, where a byte offset would pass 16; the pool row
`(key << 3) & ~15` and the den row `(key << 2) & ~31` one shift and mask from it) and its coefficient (f16, the high half, the den's) in the first word, its NUM FACTOR in the
second, fp32: `c 2^(e_d - kImageTop - kCoefTop)` (the image holds digit `d` at `2^(kImageTop - e_d)` of the fold's
state, itself `2^kCoefTop` the state: `handoff.md#maxima`), on the f16 panel times the token's scale `2^(-s - 1)`
(`token_scale`, its exponent field built directly). The same products in the same order as at a tile, once a chunk
where they were once a channel group: the four channel groups of a token group read every entry, and the token's scale
byte, the exponent build and the coefficient's conversion were a quarter of a tile's instructions.

<a id="chunk"></a>
## A chunk

`read_chunk<P, F32, F16>` reads a chunk's slots over digits `lo .. hi - 1` (the whole box, or a half where the image is
stored in halves, `schedule.md#planner`; the kernel's one call site, its passes a loop), `F32` the panel's dtype (the
call's reduce dtype), `F16` the MMAs' accumulate form (fp16 where the card's fp32 accumulate runs at half rate,
[`../carry_api.md#carry-readout-accumulate`](../carry_api.md#carry-readout-accumulate)). Each lane holds its digit's RUN WORD
in the warp's token group (`read_run`, once a chunk: the digit, its entries' place << 5, its count << 21), so a shuffle
from it takes its source lane from the low bits. It runs two PASSES (`read_pass`), each digit with entries in the token
group in one of them: the digits whose den the warp's channel group reads (`d mod 4` at 32 channels, its group's
contiguous `n_d` at 64: `handoff.md#den`) with their den, then the others with the num
alone -- the den's MMAs only where they are kept (a warp-uniform `if` around them is if-converted to predicated HMMAs,
which the ptxas gate refuses, `../../../KERNEL_STANDARDS.md#s20`; two loops are not).

A PASS walks HALVES in digit order -- up to 8 entries of what remains of a digit's run, one word like the run's -- two a
TILE, tokens on M: rows 0-7 the first half's entries and rows 8-15 the second's, of any digits. At alt-k4's ~4 entries
a (digit, token group) a tile of one digit held a quarter of its rows; two halves a tile halve the row loads, the entry
loads and the panel's adds a run, at the same MMAs a run: each half's digit's slice against all 16 rows (two m16n8k16
each, `tile_mma`), its own 8 rows kept. Lane `d` holds, once a pass, the run of the next digit the pass names above `d`
(none, a count of 0, past the last), so a half's successor is its run advanced by 8, else one shuffle from lane `d`.
`tile_in` loads a tile's rows a TILE AHEAD -- lane `l`'s entry (half `l & 8`, entry `l & 7`; a lane past its half's
count loads the half's first and adds nothing), the rows by two `ldmatrix.x4` (`chunk_addr` at the entry's row offset):
each trip of the pass is two tiles, each's row loads issued before the other's MMAs and adds, so no operand moves
between registers and the trip's one exit is at its end (no branch among its MMAs; the second tile's adds predicated
off where it is past the pass's last; the exit's flag shuffled from lane 0, so ptxas sees it warp-uniform). The pass
hand-rolls what `Burst` units do (KERNEL_STANDARDS §24's form), a debt the burst lint cannot see: it reads only a
function that issues an MMA itself, and the pass's MMAs are `tile_mma`'s, a burst function. `tile_late`
loads the rest just before the next tile's rows: each half's digit's slice (the 32 lists x 8 channels of the warp's
channel group, one `ldmatrix.x4.trans` at `slice_at`: the lane's row and the digit's plane and swizzle bits apart, one
product, a mask and an XOR), its den factor, and the lane's two tokens' entries (row `g` the first half's, row `g + 8`
the second's) -- held a tile ahead they were the values ptxas copied on the composed kernel's back-edges, whose fold
state is 128 registers a lane. The den pass's den block is the warp's channel group's (`handoff.md#den`: the 8 digits it
owns), loaded once a pass, so one den MMA pair a tile serves both halves (rows 0-7 the first digit's column, rows 8-15
the second's). `tile_out`: the MMAs, then each half's token in turn -- its panel word one XOR of its row offset
(`panel.md#words`), its load, its add, its store; the den's column of its digit (`den_col`'s low bit picks in the lane's
pair) at its coefficient times the digit's den factor into the slot's den word `cg`, by the lanes holding the column
(`tq == den_col(d) >> 1`) -- with a warp sync between the halves: a token may be in both. Every load whose store is
predicated off is predicated too (`ops::load_shared_u32_if`), so a dead lane reads no word a live lane writes; a warp
sync a tile orders one tile's adds before the next's, whose tokens may be the same in other lanes.

THE f16 PANEL AT THE TOKEN'S SCALE BYTE: the scale pass's byte bounds the token's `num` under `2^(s + 15)` with 2%
headroom ([`scale.md`](../../scale/scale.md)), so every piece and every partial sum of the token's panel word lies within f16's range at
`2^(-s - 1)`, the top bit kept for the cross-CTA sum; a piece under the frame's 2^-24 rounds into the subnormals, which
the reference's bound carries (a term a piece and an add). The fp32 panel is absolute. Each piece is rounded once into
the panel's dtype and each add once more.

MEASURED (the part harness read_alone, sm_86, cycles a warp-window at nl64k-alt-k4's
draw, the f16 panel at fp16 accumulate): 23.9K as first built (the den on every warp, each token's factor a channel
group, one tile at a time); the den on its own channel group 21.4K; the factors once a chunk and the tiles a tile ahead
14.5K; the two tokens' adds overlapped 13.5K; the next digit a lane's, the den in its digits' num tiles, the panel's word
one XOR of the row key and a tile its run's remainder (no offset carried) 10.7K; two halves a tile, the den block a
channel group's 8.2K -- against its floors there (tensor 4.0K, shared 8.1K at the first form's traffic). The halves'
form against the one-digit tile at denser lists: alt-k16's density (18 / 256) 11.6K -> 11.1K, 40 / 256 17.9K -> 18.8K
(+5%: a run of 16 or more a tile of two halves of one digit, its MMAs and slices twice). A table of the pass's halves
(tile `k` lanes `2 k`, `2 k + 1`, no successor chain) measured 9.6K: its ~340-cycle build four times a window, every
scheduler's two warps in it together. In the composed kernel (`tools/phase_ledger.py`, the readout's one call site)
the readout phase 11.0K of a 29.2K window at the one-digit tile, from 13.1K of 31.6K and 24.9K of 41.8K. AT THE 32 x 16
x 64 BOX (the planner's at nl64k-alt-k4 and flagship-alt-k16; the harness at the cell's chunk): the den digits a
group's two contiguous, one den pass a chunk, took the harness 7.5K -> 6.3K (flagship-alt-k16's chunk 11.2K -> 10.6K)
and the composed readout 8.35K -> 7.09K of a 24.19K -> 23.34K window (flagship-alt-k16 22.3K -> 20.8K of 56.1K ->
55.4K); the rest went to the half's rendezvous, where the warps without a den wait. THE SASS GATE holds its step loops at their floor (`tools/budgets/step_floors.json`,
`tools/sass_gate.py`'s `_STEP_COVER_DEBT`); what replaces the debt is composition with the fold (each scheduler's two
warps in different phases) and sm_90's `wgmma` (a warpgroup's operand read once).

ACROSS CARDS: sm_80 and sm_86 run this form (`mma.sync` m16n8k16, `ldmatrix`); sm_86's half-rate fp32 accumulate takes
the fp16 form. sm_90 and sm_100 take the same slices and entries; their asynchronous MMAs (`wgmma`, `tcgen05`) consume
the slice from shared memory without the `ldmatrix`, the panel's adds unchanged.
