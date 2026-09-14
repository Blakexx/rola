# `csrc/rola/src/facts/liveness_contract.cuh` — the liveness pass's contract

Mirror doc for the contract header and for its Python twin
`rola/engine/facts/liveness.py`. The PASS is a kernel neither file owns (C2a builds
it); what is written down here is the shape of its output, the two folds over that
output, and why each is the grain it is. The Python module carries a pure-torch MODEL
of all three — the executable spec the kernel is graded against — and
`tests/unit/test_liveness_contract.py` is its gate.

Vocabulary, defined here and used unqualified below: `D` = routing depth;
`B_l` = level `l`'s PADDED width (a power of two at or above 16 — the descriptor's,
K50); `b_l` = its LOGICAL width, which never leaves the host; `L` = tokens in the call;
`BH` = batch × heads; `N = prod_l B_l` = leaves; **atom** = 16 consecutive leaves in
canonical order, which is the page granule; **box** = one warp sub-box, an aligned
rectangle in digit space that the geometry block derives
([`common/geom.md`](../common/geom.md)).

## Two classes, and nothing between them

Blake's ruling of 2026-08-29 (card `G_FOUNDATION.md`, "ONE GEOMETRY-INDEPENDENT
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

<a id="layout"></a>

## The class-1 word layout

    [BH][side][row][word]        side: 0 = read, 1 = write
    row  = row_base(level) + digit,  row_base(l) = Σ_{i<l} B_i
    word = token / 32,  bit = token % 32

`rows = Σ_l B_l`, `words = ceil(L / 32)`, and the words are `uint32_t` (torch carries
them as `int32`, the same 32 bits, exactly as the union table already did). Per side
per batch-head that is `Σ_l B_l × L / 8` bytes whenever `L` is a whole number of
words, and `ceil(L/32) × 4 × Σ_l B_l` in general — it does NOT grow with `N`.

**A row index IS the packed amplitude column.** The routing planes arrive as
`[BH, L, Σ_l B_l]` with the levels packed on `level_offsets`, so the pass reads column
`r` and writes row `r`, and no consumer needs a second layout to relate a bit to the
amplitude that voted it. Tokens are the minor axis because the vote is a warp ballot:
one row, one lane per token, one word written by lane 0 — the shape the shipped union
table already had.

**A token past the end votes dead.** The tail word's high bits are zero by
construction, so a fold may OR whole words without masking.

**The nonzero test is on the bf16 MAGNITUDE** (`0x7fff`), so an amplitude of `-0` is
dead exactly like `+0`.

<a id="statics"></a>

## Dense levels: the static width mask, and no logical width in the signature

A dense (softmax) level is NOT read by the pass — today's skip stands. Its rows are
the STATIC WIDTH MASK: ones for digits below `b_l`, zeros for the pad. That is EXACT
rather than conservative, because a softmax amplitude is never zero and only a pad
digit's logit is `-inf`; "absent means full" would have marked a fully-pad box live
and allocated pages of nothing but pad. Unpadded dense levels are bit-identical to
the all-ones the old pass implied, and the traffic is unchanged: the mask is a
handful of words, not a read of `Σ_dense B_l` bf16 per token.

The mask reaches the kernel as BITS, never as a width. `SideStatics` carries
`dense_levels` (a bitmask over levels — a MODE fact) and `digit_mask` (one bit per
row). The two contracts put the translation at the seam and keep logical-width
fields out of every kernel entry signature; `rola.engine.facts.liveness.side_statics`
is that translation, and it is the only place `b_l` appears.

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

The kernel never learns that padding exists. Pad digits get amplitude exactly zero at
the producer, so their class-1 bits are zero on a sparse level and zero by the width
mask on a dense one; a box of nothing but pad digits therefore has both call-level
bits clear and dies at the prologue, and a live box that straddles the pad boundary
folds zeros on its pad slots. There is no descriptor-keyed dense shortcut in the
kernel and no host-side grid compaction.
