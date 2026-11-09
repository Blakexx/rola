# `carry/ampere/handoff.cuh` — the hand-off

Mirrors `csrc/rola/src/carry/ampere/handoff.cuh`: the fold's state to the readout's image every window -- the digits' keys
and exponents, the image's slices, a batch's stores and the den's tables, each in the sides' two forms
(`#same-split`). The fold is `fold.md`'s, the readout `readout.md`'s; the state's registers and the den's register
places are `parts.md`'s.

<a id="maxima"></a>
## The digit maxima

The hand-off's first step (the f16 image at a power of two a digit). A lane's registers hold its warp's lists' state,
each list in its own running unit ([`fold.md#units`](fold.md#units)), so the lists meet on one scale through an integer
KEY: a largest |accumulator| at its list's unit as its fp32 bits plus (the list's unit less the box's top unit,
`box_top`: a lane a list, a lane past a 16-list box's lists a never-live unit) x 2^23 -- the exponent field shifted, the
mantissa kept, a zero `INT_MIN` -- so the largest over the lists is
one integer max. A key is exact while the live units span under `kKeySpan` (a live unit lies between `kUnitMin` and
128; a never-live list's state is zero). `post_keys` posts each digit's key at the window's parity, at the digit's
place in lane order (`exp_slot`: digit `16 M + 8 H + g` at `4 g + 2 M + H`), in the hand-off's form (`#same-split`):

- THE ALTERNATING FORM (`post_row_keys`): a digit is a fold row, across every warp's lists. A digit row's largest over
  the lane's eight values a list (seven FMNMX), the bits joined across the row's four lanes by a quad reduce-scatter (a
  list's lanes share its unit, so the max commutes with the shift and a zero stays zero: the zero test and the shift
  run once a lane a row, on its own list `t`), the four lists' keys by two more shuffles, and one shared atomic max a
  lane.
- THE SAME SPLIT (`post_list_keys`): a digit is a fold list, every row of it its warp's. A list's largest over the
  lane's 32 values and the warp (`__reduce_max_sync`), keyed at its unit; lane `j` stores list `w + 8 j`'s key, the
  one writer, no atomic.

After the barrier `post_exps` turns lane `i`'s key into its digit's exponent: `(key + 0x7fffff) >> 23`, the exponent
field rounded up past a nonzero mantissa, plus the top less 127 -- exactly `ceil_exp` of the digit's largest |state| on
one scale -- at least `kImageTop - 127`, recorded in the warp's OWN ROW of `kExps` (`exps_of`; digit `d` at
`exp_slot(d)`) for its stores and its readout. Every warp records the same values and reads only its own: the readout
reads every digit's on every warp, and one shared row written by all eight was an unsynchronized write beside every
read (benign, the same values, but a racecheck record a byte). Warp 0 resets the other parity's keys for the next
window (a warp a window ahead posts into the other parity while a slower one still reads these). The image holds a digit at `2^(kImageTop - e)`: its largest
value at most 2^10, so a 32-list sum of image times read row (each at most 1) stays within 2^15, inside an fp16
accumulator, and one placement serves the fp32- and fp16-accumulate readouts. Measured (2026-10-12, the image's quality
model, per token against fp64 on the flagship cells): the placement at 2^10 and at 2^15 give the same error, as does a
block prescale; a leaf keeps 11 bits down to 24 octaves under its digit's largest. The form was labbed first
([`docs/lab/handoff.md`](../../../lab/handoff.md), form 4).

<a id="image"></a>
## The image

The value-carve readout's (`readout.md`): digit `d`'s plane is a SLICE a readout channel group -- 32 lists x 8 channels
f16, 512 bytes (`kSliceBlock`) -- list `l`'s 16-byte row at `l ^ (d & 7)` (`slice_row`), the four slices a 2 KB plane.
The swizzle serves both sides: a hand-off store is a lane's channel pair (4 bytes) and a warp's 32 lanes are 8 lists'
rows x 4 words, distinct rows by `l ^ (d & 7)` -- 32 banks; a readout's `ldmatrix.x4.trans` reads 8 lists' rows of
one digit, their `l ^ (d & 7)` distinct -- 32 banks. A part's planes hold a digit at plane `d & 31` (the whole image) or
`d & 15` (a half: `schedule.md#planner`). The channel pair a store holds is channels `8 n + 2 t, + 1` of slice `n`, so
the readout's MMA output pairs a token's channels `(c, c + 1)` in natural order, the panel's and `num`'s
(`panel.md#words`).

THE DEN IMAGE (`kDenImage`, 2 KB): the den state at its digit's scale in blocks of 8 digit columns (the den MMA's N),
each digit's den read by one readout channel group (`cg` groups, `n_d = D / cg` digits a group, `8 / n_d` groups a
block). At 4 groups (32 x 32 x 32) digit `d` is the group `d mod 4`'s, a block a group's 8 digits, block `d & 3` column
`d >> 2`, list `l`'s 16-byte row at position `l ^ (4 ((l >> 4) & 1) | (d & 3))` (`den_row`), a permutation inside each
8-row group. At 8 groups (the 64-channel boxes) a group's digits are CONTIGUOUS -- digit `d` the group `d / n_d`'s,
block `d >> 3`, column `d & 7` (`den_block`, `den_col`; 1 KB a block) -- so a warp's den digits lie in one image half
and its den pass runs once a chunk: at `d mod 8` each half held one of a warp's two digits, a den pass a half whose
two-tile trip was mostly dead ([`read_alone`](../../../lab/read_alone.md), 2026-11-07). Row `l` at `l ^ (4 ((l >> 4) & 1)
| 2 ((l >> 3) & 1) | block)`: a store's 32 lanes (lists `w + 8 t`, digit `g + 8 i` at block `i` and column `g`) meet 16
banks, two lanes a word, one wavefront, and inside an 8-row group the XOR is one constant, so the reader's `ldmatrix`
stays one wavefront a matrix. At K 16 (the dense box `[16, 2, 16, 2, 64, 1]`: 16 rows a block, 256 bytes) the same
map: the reader's 8-row groups one wavefront each; the writer's lanes (list `w + 8 t` the digit, at block `t`, rows `g +
8 i`) meet 8 banks four ways -- two stores a lane once a window. So do the same split's 64-channel boxes'
(`[32, 1, 32, 1, 64, 1]`, `[32, 2, 32, 1, 64, 1]`): a store's 32 lanes are one digit column (`w`), 16 K rows in each of
two blocks, the row's XOR leaving 8 banks four words each -- four wavefronts a store, two stores a lane once a window
(read off `den_place` and `den_row`, not measured). The XOR is for the writer (`den_tables_out`): a store's 32 lanes hold two lists 16 apart and 16 digits
across the four blocks, which on unpermuted rows met 4 banks (8 wavefronts a store, 2026-11-06 census); permuted they
meet 16, two lanes a word, one wavefront. The reader's `ldmatrix` rows stay inside their 128-byte groups, so it stays
conflict-free. So a warp's den block is the same for every tile of its den pass,
loaded once a pass by one `ldmatrix.x4.trans`, and one den MMA pair a tile gives both halves' digits' columns
(`readout.md#chunk`); as a block of 8 consecutive digits it took a load and a pair a half.

<a id="same-split"></a>
## The same split

The hand-off has two forms, the arm key's `same_split` (`Plan::kSameSplit`; `rola.ops.carry.same_split`, refused at the
seam when the call's levels disagree, `../carry.md#refusals`): where the sides list different levels (the alternating box)
the fold's lists are the readout's K and its rows the digits; where both list the same level the fold's lists ARE the
readout's digits and its rows the K. A dense fold side takes its coefficients from the read side's list level
([`../engine/carve.md#sides`](../../engine/carve.md#sides)), so its box is the same split too. The image, the readout and
the den's readout are the same in both: only the keys (`#maxima`), the stores (`#store`) and the den's tables
(`#den`) read the fold's state the other way round. The readout reads every leaf for its channels, so the hand-off is
always the image -- at dense its stores are ~1.0K cycles of a ~70K window (`tools/design_floor.py`).

ARMS: `[32, 1, 32, 1, 32, 0]` the alternating box, `[32, 1, 32, 1, 32, 1]` both sides listing one level (`[SS, DD]`),
`[32, 2, 32, 1, 32, 1]` a dense write side over a listed read side (`[SD, DD]`).

<a id="store"></a>
## A batch's stores

`store_batches<P, Mask>`: every warp stores its state of the batches' digits (a batch 8 digits) into the image, each
value times `2^(u + kImageTop - e)` for its list's unit `u` and its digit's exponent `e`, packed f16, a lane's channel
pair of a slice row one 4-byte store at its held base and a compile-time offset (`Mask` picks the plane). Its frame
(`store_frame`): the base, opaque so its terms are formed at each call rather than hoisted across the windows; its four
exponents, one 16-byte load from its warp's row of `kExps`; its lists' units. The factor (`list_scale`) is two integer multiply-adds on
the FMA pipe and two clamps, its multiplier and bias loaded (`kBits`) so ptxas cannot turn them into ALU shifts; it is
held to a normal float's range (below it the f16 is zero; above it the list's accumulators are under 2^-117 of its
unit). Each batch's code is compiled and selected by a uniform test: the state's registers are indexed by the batch.

- THE ALTERNATING FORM (`store_row_batches`): batch `2 M + H` is rows `16 M + 8 H + g` of every list; list `w + 8 j`'s
  value goes to its digit's K row `(w ^ g) + 8 j` (digit `g`'s swizzle), the lane's four exponents its rows'.
- THE SAME SPLIT (`store_list_batches`): batch `j` is list `w + 8 j`, its plane the list's; row `16 M + 8 H + g` goes
  to K row `16 M + 8 H + (g ^ w)` (the list's swizzle, its low bits the warp), the lane's four exponents its lists'.

<a id="den"></a>
## The den's tables

The den -- each token's read product against the MASS of the windows before, the v+1 channel ([`fold.md#den`](fold.md#den))
-- is read out beside the channels (`readout.md#chunk`). THE DEN'S TABLES, a window's, from the den's registers
(`State::den`, `fold.md#state`): a window's read chunks run before its fold chunks, so the registers hold the state
before the window's writes, each word at its place (`parts.md#den-place`: a fold list and a fold row). A word's readout digit
and K are its fold row and list in the alternating form, its list and row in the same split (`den_digit`, `den_k`).
Before the keys each digit's largest goes into `kDenMax` by an integer max (`den_tables_max`: a non-negative float's
bits order as integers; cleared before the window's top barrier); after the exponents each digit's FACTOR `2^(dE -
kImageTop)` (`kDenFactor`; `pow2_wide`) and the den image at `2^(kImageTop - dE)` (`den_tables_out`, four 16-bit stores
a lane). Every den term is ABSOLUTE: its coefficient times the digit's factor times the slice's products. While the
window folds the den image is free (its readers are done at the window's last drain), so a dense fold exchanges its
den tiles through it (`fold.md#dense`).

| buffer | written by | read by | shape | banks |
|---|---|---|---|---|
| `kDenMax` u32 `[32]` | `den_tables_max`: a lane its four words' digits, every warp, `red.shared.max` | `den_tables_out`: a lane its words' digits, lane `l` digit `l`'s factor | scalar | 16 digits two lanes each (once a window) |
| `kDenImage` f16 `[4][32][8]` | `den_tables_out`: a lane its four words, 16-bit | the readout: a digit's block, one `ldmatrix.x4.trans` | 16-bit / 16-byte rows | the stores once a window; the loads 8 rows of 16 bytes: one wavefront |
