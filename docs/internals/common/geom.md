# `geom.cuh` — the addressing block

The mirror of `csrc/rola/src/common/geom.cuh`. One derivation, computed once per launch
on the host, consumed by every kernel of the family.

**Symbols.** `D` levels; `B_l` level `l`'s digit width; `BC` the leaves one owner (one
CTA's box) holds; `MO` the per-level mode word (two bits a level, read then write);
`s(l)` the OWNER span; `g(l) = B_l / s(l)` the owner grid; `NS` a side's stream count;
`sv(l)` the WARP SUB-BOX's span; `rank(l)` a level's position in a side's private order
(0 innermost); `shift(l)` the slot index's bit offset for level `l`; `kAtomLeaves = 16`
the page granule.

## What is compile-time and what is not (the axis law)

SHAPE is compile-time because it sizes registers and shared memory. G2 read that too
narrowly and G2b corrects it (the "PROPOSED CORRECTION TO R12", ruled 2026-08-29): the
warp's leaf-slot → digit map is baked into the unrolled expansion, so **the sub-box's
spans and slot layout ARE register shape**, exactly like leaves-per-warp and `DV`. Made
runtime they force either a dynamically indexed local array (a spill) or a case
enumeration — G2 measured the second at 2x instructions, 5.9x I-cache stalls and a 63 →
190 KB kernel.

So: compile-time is `D`, `DV`, `W`, `C`, the CTA's warps, and the two SPAN LAWS below
(the warp sub-box from `(D, DV)`, the owner box from `(D, BC)`). Runtime is the widths
`B_l` — strides, column bases, id weights, the owner grid — and the per-side ORDER, which
reaches the kernel only as CARVE OFFSETS: which stream index bit fixes which level's
digits. Every runtime value here is CTA-UNIFORM, so it lives in the uniform datapath and
the vector register count does not move — which is the census this gates on.

The consequence is the ship principle: a user of the Python package never registers an
arm. One binary per shape serves every topology and every declaration of its depth.

<a id="arch-constants"></a>
## The arch's primary constants, in derivation order

1. `leaves_per_sm(dv)` — what the SM's STATE CAPACITY admits at this value width (the
   register file on sm_80/sm_90, TMEM on sm_100). See [the state fraction](#state-fraction)
   for where the capacity number comes from.
2. `warps_per_sm` — the sub-box owners are WARPS on every architecture.
3. The SM box is `leaves_per_sm` balanced over `D`; the carve divides it, in the side's
   own order, into `warps_per_sm` SEGMENTS.
4. `leaves_per_warp = leaves_per_sm / warps_per_sm` is the warp's SEGMENT, derived.
5. The COMPILED SHAPE is the MMA ACCUMULATOR TILE's leaf layout — the slot → digit map
   that fixes which leaf row of the tile each expanded coefficient (the panel row) lands
   on. On sm_80/sm_90 that tile is the warp's register fragment (32 leaves at `DV = 64`);
   on sm_100 it is the TMEM tile the MMA addresses in place (`tcgen05.mma` accumulates
   into TMEM and reads A from TMEM, so the state never passes through registers to
   compute). The map is the same object either way.
6. The warp's SEGMENT is its tiles, and TILES PER SEGMENT is derived: it is `1` here, one
   tile being the whole segment. A warp that walks several tiles does so as a runtime trip
   count over identical tile bodies — not a shape, and not in this table.

The SHAPE COUNT is flat across architectures: it depends on `D` and on how many bits the
carve takes (`log2 warps_per_sm`), not on the capacity. What scales with
`leaves_per_warp` is each generated body's SIZE.

<a id="state-fraction"></a>
## The state's share of the register file

`leaves_per_sm(dv) = state_elems_per_sm(caps) / dv`, and
`state_elems_per_sm = regs_per_sm * kStateFileNumerator / kStateFileDenominator` — the
capability row's own register count times the share the design gives the STATE, one
quarter. At 65,536 registers that is 16,384 fp32 elements, hence 256 leaves at `DV = 64`
and 32 per warp.

The share is a DESIGN NUMBER and the census is what checks it. A warp holding 32 leaves of
65 channels holds 2,080 fp32 elements, 65 per lane of a 256-register budget at one CTA per
SM: 25.4 per cent, allocated as 80 registers per lane once the channel count is padded to
the MMA's m granule. The quarter is therefore the share the design INTENDS and the census
reports the share it got; a stage that parks part of the state elsewhere revises the
constant upward, and an architecture whose accumulator lives outside the register file
(TMEM) derives the number from that capacity instead.

The fraction is one value across the tabulated set, so it is written as a constant rather
than a per-architecture table; a part that changes it gets a row when it gets a capability
row.

<a id="cluster"></a>
## The cluster level

`kClusterCtas` is the number of CTAs one cluster's box is partitioned across. It is
arch-deterministic and it is 1 on every architecture this build tabulates, so every cluster
term folds away — a cluster is one CTA and a CTA is one SM.

The level exists as a NAME because it is real where a cluster is real: a two-SM tensor-core
mode executes one accumulation across a clustered CTA pair, with the accumulator split over
the pair and one memory fetch landing in both SMs. The hierarchy is therefore state box ->
cluster box -> CTA box -> warp box -> accumulator boxes, truthfully, on every part; what is
NOT built is any machinery for a cluster larger than one — no distributed shared memory, no
multicast, no pair issue. Deleting the level instead would hard-code one architecture into
the vocabulary, which is the failure the portability rule exists to prevent.

<a id="warp-box-set"></a>
## The generated set of sub-box shapes, and why it is a SET

The carve is TOP-DOWN (Blake, 2026-08-29): the box's spans are divided level by level in
the SIDE'S OWN order — its sparse levels outermost — until the box count reaches the
warp count; what remains is that side's sub-box. `[4,4,4,4]` at eight warps goes
`[1,4,4,4]` (four boxes) then `[1,2,4,4]` (eight), and `[1,2,4,4]` is the shape.

Because the result depends on the order, and the order comes from the MODE WORD, there is
not one shape but a SET: `sub_boxes(D, box_leaves, warps)` runs the carve over EVERY
level order and de-duplicates. It is a generated constexpr table, never hand-written, and
`sb_index` is the question "is this carved shape a member" asked of the generator itself.
At `D = 2` the set has two members, at `D = 3` four, at `D = 4` seven.

Each side selects its member at launch from its own order; the two sides therefore no
longer share a slot layout, which is what restores the two-sided carve's liveness
selectivity (the `## G2b` finding) and what makes the transit's row map a per-side
scatter rather than a per-warp base.

<a id="canon-leaf"></a>
## The canonical-leaf map, and why it is compile-time in D

`geom_canon_leaf` walks the levels with `#pragma unroll 1` over the block's RUNTIME depth,
so every span and run-shift is an array load — and the map is called per atom by the state
page sweep. The parent computed the same map from `BoxPlan` as a folded bit permutation
and spent 4,096 instructions on it; the runtime form spends 1,402,880 on the k16_w384 cell, a
342x blow-up and the single largest term in that cell's instruction delta.

`geom_canon_leaf_t<D, BC>` is the same law with the DEPTH and the OWNER SPANS compile-time
— which they are, `f(D, BC)` — leaving only the width-derived `owner_div_bits`, `grid` and
`weight` runtime. R12 is explicit that a register-shaped loop carries no runtime trip
count, and the kernel knows `D` and `BC` as template arguments, so the untemplated form was
a standards violation as well as a cost. The runtime `geom_canon_leaf` stays for the host
seam, which genuinely does not know `D` at compile time.

<a id="row-swizzle"></a>
## The region's swizzle, and why it is free

The bank of a word in the transit region is `(36 * row + col) mod 32`, and `36 * row mod
32 = 4 * (row mod 8)` — so an access is conflict-free exactly when its rows are distinct
`mod 8`. The region's row is the leaf's READ-ORDER INDEX (`geom_read_index`: the read
side's carve order from the mode word, rank 0 innermost) under
`sb_swizzle(row) = row ^ (((row >> 3) ^ (row >> 6) ^ (row >> 9)) & 7)`: every three-bit
group of the index folded into its low three bits, so index bit `j` lands on bank group
`e(j mod 3)` and ANY three consecutive index bits are independent.

Why the read order, not the physical one. The readout's pull (`ldmatrix.trans`) takes
eight consecutive READ-ORDER leaves; in the physical order those step rows by a power of
two whenever a read-sparse level sits inside the run level (the alternating draw: sixteen
rows apart, four-way conflicted, 16 wavefronts a load where 4 are ideal, measured 2026-09-07
as 16% of every shared wavefront in the kernel). No GF(2)-linear swizzle of the physical
row clears every order the mode words reach — the deep-4 orders force four row bits into
one coset of the bank group, and any three of them then collide with the bit-0 pull — and
neither does any `row ^ f(row >> 3)` with `f` free (both exhaustive searches, journal
section 38). In the read order the pull is `{0, 1, 2}` always, and the write side's
accesses — a member's scatter of eight consecutive innermost digits — are a contiguous
bit triple wherever the rank order puts that level, which the folded swizzle clears.
The exchange-law model test (tests/unit) walks every access at every reachable (read order, write member).

Two properties keep it free rather than a cost, both asserted: it is a BIJECTION of
`[0, BC)`, so no leaf is lost or aliased; and every step of the map — the member's
scatter, the read-order permutation, the swizzle — is GF(2)-LINEAR in the slot's bits, so
`row(strm, P) = row(strm, 0) ^ XOR_{b in P} row(bit b)`: a warp held its base row and one
row per slot bit, and a fragment's row was an XOR selection by the slot's compile-time
bits (the transit that used this is gone with the state carve's snapshot, DELETIONS.md). The address arithmetic composes with `^` and never with `+`.

<a id="cta-box-first"></a>
## The derivation is CTA-BOX-FIRST

`warps_per_sm` splits as `ctas_per_sm × warps_per_cta` (Blake, 2026-08-29).
`warps_per_cta` is a COMPILE-TIME axis with the two-value set `{8, 4}` — launch shapes
`(1, 8)` and `(2, 4)` — and the member set is `f(D, leaves_per_warp, warps_per_cta)` with
THE CTA'S BOX as the unit: the box a CTA owns is `leaves_per_warp × warps_per_cta` leaves,
balanced over `D`, and each side carves THAT box among the CTA's own warps. `kWarps` never
enters a shape; it fixes which box the shape law is run on.
The arch law — at `warps_per_cta == warps_per_sm` the arm's set IS the SM-grain set — is
asserted where the set can be reached from, which on this branch is the geometry test
(see "What consumes this block today"), and in the body's own static_assert once a body
consumes the block.

**`(1, 8)` DOES NOT FIT sm_86, and the number is the panel ring.** The readout law ties
the batch tile to the warps (`kWarps * kMmaN == C`), so eight warps means `C = 64`, and
`panel_bytes = 2 * kWarps * C * LdS * 2` is therefore QUADRATIC in the warp count:
81,920 B at `(1, 8)` against 20,480 B at `(2, 4)`. With the readout block and two V slots
the arm asks **117,760 B against sm_86's 99,328 B per-CTA maximum** and the body's own
`static_assert(design::smem_fits(bytes))` refuses it at compile time. It fits sm_90's
227 KB, so `(1, 8)` is the design target the panel law has to be re-shaped for, not a row
that can simply be added to the arm table.

**The grain this branch compiles.** `warps_per_sm` is eight — one CTA per SM, R15 — but
these arms compile `kWarps = 4`, so the CTA box is HALF the SM box (`BC = 128` of the SM's
256 leaves) and the set is the same law run on the half. The two sets are NOT the same
shapes: at `D = 2` the SM box `[16,16]` yields `[2,16]` and `[16,2]`, while the half box
`[8,16]` yields `[2,16]` and `[8,4]` — and `[16,2]` does not fit a box that is 8 wide at
level 0. A four-warp CTA cannot be a contiguous subset of the SM's eight sub-boxes with
BOTH sides covering the same leaves, because the two sides carve different levels. The set
is generated from the box the CTA actually owns for exactly that reason.

<a id="warp-box"></a>
## The warp sub-box, and the canonical slot layout

`leaves_per_warp(DV) = 2048 / DV` is the register law: 32 leaves of state at `DV = 64`,
16 at `DV = 128`. Its `log2` is split BALANCED over the `D` levels, remainder
innermost-first — `owner_span_bits` — and that is the whole of `sv(l)`. At `DV = 64`:

| `D` | `sv` | `shift` |
|---|---|---|
| 1 | `[32]` | `[0]` |
| 2 | `[4, 8]` | `[3, 0]` |
| 3 | `[2, 4, 4]` | `[4, 2, 0]` |
| 4 | `[2, 2, 2, 4]` | `[4, 3, 2, 0]` |

<a id="canonical-slots"></a>
A slot index is the MEMBER'S OWN digits in canonical order, innermost lowest, so
`shift(l) = Σ_{l' > l} log2 sv(l')` — `sb_shift`, evaluated on the member that side
selected. G2b shared ONE layout across both sides and G2c un-shared it: sharing made the
transit's rows an aligned run, but it also forced both sides to carve the same levels,
which is the selectivity the two-sided carve exists for. What
survives the correction:

* The produce's span and shift are IMMEDIATES on both sides — a member is a compile-time
  case of the structure switch — so the expansion is the parent's code at the parent's
  instruction count;
* The transit's row map is a per-side SCATTER (see below) whose lane-dependent part is one
  base word and whose per-fragment part is constant;
* `BC = leaves-per-warp × kWarps` is DERIVED, never an axis: the carve stacks warp
  sub-boxes along the levels where the owner box is wider than the sub-box.

<a id="carve"></a>
<a id="span-rule"></a>
## The span rule, and why the atom is a constraint and not an afterthought

K50 (Blake, 2026-08-29): "balance BC over levels; a level narrower than its balanced
share takes its full width and the remainder balances over the others ... Refusal only if
`Π_l B_l < BC`." The same card's amended ownership ruling names the invariant the balance
runs under: **a box is a union of whole atoms**.

Balance ALONE does not keep that invariant. An atom is 16 CONSECUTIVE CANONICAL leaves
and it is the page granule — the state's entry load and exit store address the plane by
canonical atom id, at page grain. At `D = 3, B = 16` pure balance gives `[4, 4, 8]` and at
`D = 4, B = 8` it gives `[2, 4, 4, 4]`; in both, an atom straddles TWO owners, and each of
them stores the whole page. That is measured, not argued: the unconstrained form failed
nine fp64 cells, STATE PLANE only, on BOTH bodies, at exactly those two topologies.

So the rule is balance UNDER the atom invariant:

1. reserve the atom innermost-outward — the levels a 16-leaf granule spans are held WHOLE;
2. balance `log2 BC` over the levels, remainder INNERMOST-FIRST;
3. raise any level to its atom floor and cap it at its own width;
4. take the surplus back from the WIDEST level above its floor, ties to the OUTERMOST, so
   the levels above the atom stay balanced among themselves; give any shortfall to the
   innermost level with room.

This reproduces the compile-time `BoxPlan` it replaces at every built topology, which is
what makes the model test a PARITY PROOF rather than a re-declaration:

| topology | derived | the compile-time law |
|---|---|---|
| `D = 2, B = 256` | `[8, 16]` | `[8, 16]` |
| `D = 2, B = 64` | `[8, 16]` | `[8, 16]` |
| `D = 3, B = 16` | `[2, 4, 16]` | `[2, 4, 16]` |
| `D = 4, B = 16` | `[2, 2, 2, 16]` | — (the parent's `B = 8` topology is unlawful under the floor) |

Under the restored floor (`B_l >= 16`, power of two) no level width can bind the
balance, so this law is compile-time `f(D, BC)` — which is what lets the channel-split
body's slice map be immediates too. A level narrower than its span is REFUSED, by name;
there is no narrower slot layout to fall back to.

The CORRECTION rules the eventual relaxation — "prefill owns whole ROWS ... the fix is
ROW-GRAIN activity masks (16 bits per atom) from the facts pass" — and that machinery is
the page I/O and the facts bit emission. Until it exists, the atom invariant is what the
page-grain store requires, and the unconstrained balance is a follow-on gated on it.

<a id="refusals"></a>
## The refusal set, and what is a census fact instead

Refused at the seam, each by name: a width that is not a power of two; **a level
narrower than the owner box's span there** (which subsumes `Π_l B_l < BC` and names the
level that cannot hold its share); a value width that is not a multiple of sixteen; a
packed amplitude row that is not a whole number of sixteen-byte granules; a level whose
column base is not a whole number of the runs read out of it (the wide loads' alignment);
and a stream count the grid cannot partition.

`min_l B_l` is REPORTED, not refused, and it stays a census fact: the width law above is
the one the shape actually owns, and under the floor (`B_l >= 16`) it never fires for a
lawful state. The one topology it removed at G2b is `D = 4, B = 8`, whose conformance
cells moved to the lawful `(16, 16, 16, 16)`.

<a id="local-row"></a>
<a id="digit"></a>
## The owner-local canonical index, and the digit it carries

`geom_local_row(side, stream, P)` is the transit row map WITHOUT the swizzle: the stream's
carve offset and the slot's digit placed at each level's own bit offset in the owner-local
canonical index. It is written once and reused three ways, which is the point — one law,
not three:

* the transit's row is this index swizzled;
* the state's PAGE and ROW are this index split at the page rectangle, `page = local >> 4`
  and `row = local & 15`, which holds because the owner span law gives the innermost level
  at least the page's four bits and no shift below it;
* `geom_digit(g, O, local, l)` recovers level `l`'s ABSOLUTE digit from it — the owner's
  grid digit above the box's span, the slot's run below — which is what turns a lane's leaf
  into a column of the packed factor plane.

<a id="row-map"></a>
## The transit's row map

The region is indexed by the OWNER-LOCAL CANONICAL INDEX, so each side reaches a leaf
through its OWN base word and its own member's fixed shifts:

    geom_xch_row(g, side, stream, P) = swz( Σ_l ( off_l(stream) + dig_l(P) ) << run_shift(l) )

with `off_l = ((stream >> grid_div_bits[l]) & (grid[l]-1)) << sub_bits[l]` the stream's
own carve offset at that level, `dig_l = (P >> sub_shift[l]) & (sv(l)-1)` the slot's digit
there, and `swz` the swizzle above. The two sides' members generally DIFFER — that is what
the per-side sub-box buys — so a leaf keeps neither its slot nor its warp across the
transit, and the map is a scatter rather than a per-warp base plus the identity.

Both halves are GF(2)-composable: the lane's stream offsets are a runtime base word taken
once, every per-fragment slot term is a compile-time constant, and the swizzle distributes
over the two because it is linear.

`tests/unit/test_exchange_law.py` proves the law rather than the code: each side's map
is its own members' scatter and the two agree on the region, the exchange relays the state
leaf for leaf, the swizzle is a bijection of the region and is linear, and every
`(read order, write member)` pair a mode word can select -- over every topology in the case
list -- is conflict-free on all four accesses. That sweep carries a FLOOR on how many pairs
it reaches (`>= 14`), because the pair set collapses under the innermost-digit invariant
and a sweep that stopped reaching pairs would otherwise pass silently.

The identity case is the former `kTied`. K50 DELETED the elided body — "one transit,
always; the tied case is a transit whose row map is the identity, same code, same register
plan" — so `identity_map` is a reported FACT that makes a wasted transit visible in the
ledger, and never a second code path.

<a id="on-the-tip"></a>
## What consumes this block today

NOTHING in the shipped kernel bodies. The block landed as a FOUNDATION piece
(G2-EXTRACT): the parent carry bodies on this branch are the MO/B-templated ones whose
geometry is compile-time, and they are the parity reference that the pipelined body (F-P)
is measured against — "the old body is never retrofitted again" (Blake's ruling on the foundation card, "THE CUT").
What held the two derivations together while the parent still built was
`tests/unit/test_carry_geometry.py`'s live re-read of the parent binary's own compile-time
census, arm by arm; C0 deleted `carry_build_stamp`/`carry_arms` from the extension
entirely, so as of C1a that file's parity claim is checked against `ARM_PARAMS`/
`PARENT_CENSUS`, a table TRANSCRIBED from the parent tip (335c2b9) rather than re-derived
from a live binary — the A-side of a parity claim is a pinned tip, per `## G1` SKILLS
FEEDBACK 3, and this is that prediction landing.

Two consequences worth stating for the reader who arrives here first:

* `box.cuh` carries its own `kAtomLeaves` for the parent body, and this header carries
  one too. They are the same constant with the same value, both at namespace scope in
  `rola::carry`, so a translation unit that includes BOTH does not compile. No TU does
  today; the body that adopts this block deletes the other copy.
* The `sb_matches_arch_law` static_assert that G2c put in the carry body — at
  `warps_per_cta == warps_per_sm` the arm's set IS `sub_boxes(D, leaves_per_sm(DV),
  warps_per_sm)` — has no body to live in here, so the arch law is asserted in the
  geometry test instead, over the generated set the binding exposes.

<a id="c1a-member-sets"></a>
**C1a PINS THE PER-SIDE MEMBER SET AT BOTH `warps_per_cta`.** The set is
`f(D, leaves_per_warp, warps_per_cta)` (`## warp-box-set` above), and the 2x4 ruling
gives `warps_per_cta` exactly two values, `{8, 4}` — the shipped one-CTA-per-SM shape and
the `(2, 4)` diagnostic arm — so `test_the_sub_box_set_is_generated_and_deduplicated` in
`tests/unit/test_carry_geometry.py` enumerates, de-duplicates and counts the generated set
at BOTH, not only the `4` these arms happen to compile: `{1, 2, 4, 7}` members at
`D = 1..4, warps_per_cta = 4` and `{1, 2, 5, 8}` at `warps_per_cta = 8`. The arch-law
assert — at `warps_per_cta == warps_per_sm` the CTA-box law IS the SM-grain law — is what
`test_the_shape_set_is_the_arch_law_at_one_cta_per_sm` already checks at `warps_per_cta =
8` (`warps_per_sm`, `## arch-constants`); the two tests together are the "enumerated,
deduplicated, counted, with the arch-law assert at 8" claim.

<a id="carve-order"></a>
## The carve order

The order each side carves its levels in is a RUNTIME INPUT: `derive_carry_geom` takes a
`CarveOrder` — both sides' levels ranked, rank 0 innermost — and `derive_side` walks it
BACKWARDS, dividing the outermost level first, until the box count reaches the side's stream
count. Nothing about the order is compile time; it selects a member of the same generated
shape set the compile-time switch already covers.

`carve_order_of_modes` is the STRUCTURAL DEFAULT the seam applies when a caller names none:
a side's SPARSE levels rank outermost, so the carve divides them first and the dense levels
stay whole until the digits run out. The boxes are uniform in SIZE either way — that is the
shape law — and what the order shapes is the uniformity of their ACTIVITY, which is what
reaping reads. A call that declares every level dense on both sides gets the canonical order,
which is always lawful.

<a id="innermost-invariant"></a>
## The innermost digits are the vector width

`carve_sub_bits` never divides the CANONICAL INNERMOST level below the page rectangle
(`kAtomLeaves`). Those digits are the fastest varying ones — adjacent digits there are
adjacent leaves — so a warp box narrower than the rectangle owns a STRIDE rather than a RUN:
its leaves would be scattered across every page of the box, its factor columns would no
longer be one contiguous run per level, and the sixteen-leaf tile that guarantees the
nesting law would not fit inside it. When an order names that level, the carve passes over it
and divides the next one instead — the same shape as the owner law's own floor — and a
stream count no order can reach under the floor is REFUSED by name rather than rounded.

The invariant is visible in the generated shape set: every member ends in the same innermost
span, so the orders that differ only in how they would have split it collapse onto one
member. At the flagship arm the set is one shape rather than two; at depth four it is three
rather than eight.
