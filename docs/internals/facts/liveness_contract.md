# `csrc/rola/src/facts/liveness_contract.cuh` — the class-1 words' contract

Mirror doc for the contract header and for its Python twin
`rola/engine/facts/liveness.py`. The PASS that writes the words is the facts pass
([`facts.md`](facts.md)), which neither file owns; what is written down here is the
shape of its output, the two folds over that output, and why each is the grain it is.
The Python module carries a pure-torch MODEL of all three — the executable spec the
kernel is graded against — and `tests/unit/test_liveness_contract.py` is its gate.

Vocabulary, defined here and used unqualified below: `D` = routing depth;
`B_l` = level `l`'s PADDED width (a power of two at or above 16 — the descriptor's,
K50); `b_l` = its LOGICAL width, which never leaves the host; `L` = tokens in the call;
`BH` = batch × heads; `N = prod_l B_l` = leaves; **atom** = 16 consecutive leaves in
canonical order, which is the page granule; **box** = one warp sub-box, an aligned
rectangle in digit space that the geometry block derives
([`common/geom.md`](../common/geom.md)).

## Two classes, and nothing between them

Blake's ruling of 2026-08-29 (the foundation card, "ONE GEOMETRY-INDEPENDENT
LIVENESS PASS") admits exactly two classes of fact.

* **CLASS 1 — the raw bitlists**, `O(Σ_l B_l × L)` per side, stored once. One bit per
  (side, token, level, digit), in canonical digit order. No span, no carve, no `B`
  in the pass's template: this is the whole reason there is ONE pass rather than one
  per consumer, and it is what makes the words reusable across kernels and across
  calls whose routes are unchanged.
* **CLASS 2 — the `O(N)`-class facts**, aggregated over tokens: the PRIMARY per-box
  READ/WRITTEN bits over the WHOLE CALL, and the DERIVED per-page bits the host ORs
  out of them. Nothing sits between: a box's liveness *for one token* is the AND over
  levels, computed on the fly by the owning warp and never stored, because
  `O(#boxes × L)` has no consumer.

The int8 rebuild's ruling of 2026-09-28 widened what the facts kernel owns to anything
determinable from the raw produced values, so two per-grain aggregates now sit beside
the classes and are written by the same pass: the list exponents, per (window, list digit)
([`facts.md`](facts.md#lists)), and the [tile unions](#tile-unions), per 64 tokens. Neither is a class-2 fact — neither is
over the whole call nor `O(N)` — and neither is derived by a consumer: each is the OR or
the maximum of what the pass already holds when it holds it.

## <a id="max"></a>The digits: each against its row's max

Python's statement of the digits is `rola.routing.quant` (`AMP_LEVELS`, `DENSE_FLOOR`, the `SPARSE` and `DENSE` forms);
`tests/unit/test_api_contract.py` holds the two to the same constants.

Every level's row is stored unit-max: u8 digits with the row's largest at 255 (`kRowMaxDigit`), the row's max beside
it as an fp32 scale. A digit `b` is the amplitude `b / 255 × scale`. An all-zero row, its scale 0, is a dead token at
that level.

## <a id="alignment"></a>The row alignment law

Every level's rows start on 32 bytes and step batch, token and head by multiples of 32 bytes (a dimension of length one
is free), in `RouteFactors`' check and the facts seam's; `quant.ROW_ALIGN` and the facts header's
`kRowAlign` state it. The producer and `RouteFactors.from_levels` pack each side's levels in 32-byte slots
(`rola.routing.quant.row_slot`: a level's width rounded up), the slot's tail zero — at power-of-two widths of 32 and up
the slots are the widths, so the law costs nothing there. It is what lets a kernel move a token's run as 16-byte
asynchronous quanta whose source is 16-byte aligned and whose 32-byte run is one sector: the facts pass's ring
([`facts.md`](facts.md#load)). A level whose width is not a multiple of 16 is copied whole
up to its last digit and no further (a partial quantum zero-fills the rest).

## <a id="floor"></a>The dense floor

A DENSE level is one whose activation has no exact zeros (the bundle's `sparse` flag unset: softmax, not entmax),
and it is never zero (ruled 2026-10-04): no coefficient of a live row is below `kDenseFloor` = 2^-13 of the row max,
the numerically safe minimum (f16's least normal is 2^-14). A dense level's rows, on either side, are stored f16 at
the row max with the floor already taken — the producer writes `max(w / max, 2^-13)` — so its consumers read them as
they are (the write side ruled 2026-10-04, the read side 2026-10-07: the readout loads the read rows into its MMAs
unconverted). The floor applies only in a live row (scale > 0): a dead row stays zero everywhere, including where a
consumer divides the row by its own sum (the decay's shares). A sparse level's zero is exact and stays 0, and its rows are u8 on both
sides.

What it buys: the fp16 fold scales each write list by one power of two, and the readout publishes each 64-leaf block
at one; both keep 11 bits a term but only a bounded range below their largest. With the floor, the list's strongest
token writes every leaf of its list at no less than 2^-13 of the list's top, and a reader weights the leaves of a
block within 2^13 of each other, so what fp16 loses below its range is a small fraction of every leaf it touches
whatever the gain spread. The cost to the model: a peaked row's mass grows by at most 255 × 2^-13 ≈ 3% of its max,
inside the uncertainty its own zero digits already carry.

Every consumer reads it the same way: a dense level's f16 rows as stored, the floor taken, on either side -- the
`DENSE` form's `dequantize`, the decode fold, the fold and readout, the intra and the oracle's references. Numerical safety asks for one dense level a side (the alternating split, or fully dense);
sparse-sparse routing has no floor to lean on and is declared at the caller's risk.

<a id="layout"></a>

## The class-1 word layout

    [BH][side][row][word]        side: 0 = read, 1 = write
    row  = row_base(level) + digit,  row_base(l) = Σ_{i<l} B_i
    word = token / 32,  bit = token % 32

`rows = Σ_l B_l`, `words = ceil(L / 32)`, and the words are `uint32_t` (torch carries
them as `int32`, the same 32 bits, exactly as the union table already did). Per side
per batch-head that is `Σ_l B_l × L / 8` bytes whenever `L` is a whole number of
words, and `ceil(L/32) × 4 × Σ_l B_l` in general — it does NOT grow with `N`.

**A row is a level's digit, in canonical order** — the column a packed plane of the
levels holds it at, which is how the producer writes the bundle (`[B, T, H, Σ_l b_l]`,
the levels on `level_offsets`). Tokens are the minor axis: a word is 32 consecutive
tokens of one row, so a consumer walking a window reads a row's words in order, and a
warp's 32 tokens are one word.

**A token past the end votes dead.** The tail word's high bits are zero by
construction, so a fold may OR whole words without masking.

**The nonzero test is on the digit** (`digit_live`): a u8 digit is live when it is
nonzero. The model applies `!= 0` to whatever rows it is given, which is the same test on
u8 digits and lets a drawn call state its support from its floating levels (`-0` dead like
`+0`).

<a id="statics"></a>

## Dense levels: the static width mask, and no logical width in the signature

A dense level — an activation with no exact zeros (softmax), as the topology declares
it — is NOT read by the pass, and its rows are NOT WRITTEN: they are the STATIC WIDTH MASK,
ones for digits below `b_l` over the live tokens, zeros for the pad, which every reader of
the words — the folds (`box_activity`, `atom_activity`, `page_bits`, given the side's
statics) and the kernels — takes from the statics, never from the table. The model
(`liveness_words`) states those rows all the same, as the definition the folds reproduce. That is exact up to a sound over-report: only a pad digit's
logit is `-inf`, and a digit whose probability rounded to a zero digit in the bundle's
format is reported live — work a consumer spends, never a contribution it misses.
"Absent means full" would have marked a fully-pad box live and allocated pages of
nothing but pad; the mask is a handful of words, not a read of `Σ_dense b_l` digits a
token.

The mask reaches the kernel as BITS, never as a width. `SideStatics` carries
`dense_levels` (a bitmask over levels — a MODE fact, `level_is_dense` in the header)
and `digit_mask` (one bit per row, `mask_bit`). The two contracts put the translation
at the seam and keep logical-width fields out of every kernel entry signature;
`rola.engine.facts.liveness.side_statics` is that translation, and it is the only place
`b_l` appears.

<a id="tile-unions"></a>

## The tile unions

    [BH][side][group][tile]      group: a level's digits in 32s, group_base(l) = Σ_{i<l} ceil(B_i / 32)
    tile = token / 64,  bit i of (group g of level l) = digit 32 g + i

A word per (side, 32-digit group of a level, 64-token tile): bit `i` set when that digit is
live at some token of the tile — the OR of the tile's two class-1 words, a row a bit. A
level narrower than 32 digits is one group with its high bits clear; `groups = Σ_l
ceil(B_l / 32)` a side, `tiles = ceil(L / 64)`, and a tile past `L`'s last word ORs only
the words that exist. `union_address` in the header is the table's address from its
extents, which `LivenessLayout::union_index` and a kernel with compile-time extents both
call; `tile_unions` in the Python module is the model, gated against the per-token bits.

The intra's slab gate is the consumer (`intra/ampere/intra_kernel.md#slab-gate`): a CTA reads 40
bytes a lane of it where reading the words themselves took a 64-byte run from each of 512
rows 8 KB apart — measured at `L = 64K`, 10% of the intra's warp cycles against 3.5%.

<a id="folds"></a>

## The folds, and why the AND is exact

**Per box, per side, over the whole call** (the PRIMARY bits):

    live(box, token) = AND over levels l of ( OR over d in box.run(l) of bit(side, l, d, token) )
    READ/WRITTEN(box) = OR over tokens of live(box, token)

The AND is not a relaxation. A token's live leaf set is the PRODUCT of its per-level
live digit sets — the routing is a Kronecker product — so "some live leaf inside the
box" and "every level has a live digit inside its run" are the same statement. The
consuming side therefore ORs its own span's digits per level and then ANDs the levels;
that word count is the fold's floor term (the note).

The bits are per box over the WHOLE CALL and never per window: the box lives in
registers from the call's entry to its exit, so the gate is `LOAD on resident ∧ (read ∨
written)`, `STORE on written`, once each. A window with nothing live for a box is a
schedule fact read off these same rows while walking, never a stored bit.

**Per page** (the DERIVED bits), on the host:

    page.READ    = OR over read-side  boxes intersecting the page of READ(box)
    page.WRITTEN = OR over write-side boxes intersecting the page of WRITTEN(box)

through the box→page map the descriptor gives (a box's leaves are the product of its
digit runs; the page is the trailing four canonical bits). The byte is the one
`rola/engine/facts/planes.py` already publishes — `ATOM_READ | ATOM_WRITTEN`, never
pre-ORed — so `plan_exact` allocates on the written half, residency checks read the
read half, and a decode step consumes this set because its unit is the page. Being a
box-grain OR it may over-report a page whose live leaves belong to a box that
straddles it; it never under-reports, which is the property allocation depends on and
which the gate asserts against the leaf-level truth.

## What padding does, and does not, reach

The kernel never learns that padding exists. Pad digits lie past a level's rows (the
bundle holds `b_l` digits a level) and the mask never names them, so their class-1 bits
are zero on a sparse level and zero by the width mask on a dense one; a box of nothing but pad digits therefore has both call-level
bits clear and dies at the prologue, and a live box that straddles the pad boundary
folds zeros on its pad slots. There is no descriptor-keyed dense shortcut in the
kernel and no host-side grid compaction.
