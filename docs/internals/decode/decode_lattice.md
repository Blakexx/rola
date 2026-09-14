# `csrc/rola/src/decode/decode_lattice.cuh` — the `(k, m)` box lattice and the T=1 factor tables

The leaf order for `T = 1` decode IS the `(k, m)` box lattice, in the ONE order
the addressing block defines and the carry family writes (`csrc/rola/src/common/geom.cuh`;
the `(k, m)` box lattice itself is on the `k35-final` tip, C0): a leaf is
named by its owner coordinate `o_l` at every level and by its per-level RUN
OFFSET `r_l = d_l - o_l * s_l`, composed in MIXED RADIX most significant level
first,

```
lambda = O * BC + sum_l r_l * prod_{l' > l} s_l'
```

and this file is the device object the step's walk reaches a leaf THROUGH — by
composing those factors — rather than by consulting a map over the leaf space.
Nothing here is sized in `N`: one bit per DIGIT (`sum_l ceil(B_l/32)` words per
side) and one entry per live OWNER COORDINATE (`sum_l g_l` at most, per list) is
the whole apparatus.

THE INTERLEAVE IS DEAD. The R0 spec's LATTICE-1 §2.2 named a second intra-box
order — the `[u_0..u_{D-1} | c_0..c_{D-1}]` interleave, in which a sub-box `k^D`
was a contiguous run — and an earlier draft of this walk enumerated
`(owner, sub-box)` units because of it. Mixed radix is the shipped order (the
one-order ruling; `box.cuh`'s `run_of`), a sub-box is a scattered rectangle
under it, and the walk's unit moved accordingly (§3). The measured reason is
recorded in `docs/internals/DELETIONS.md`'s wave-2c rows and in the batch's
findings: under the interleave a token's run over a level that is DENSE for its
side straddles atoms at 25% occupancy at EITHER span vector; under mixed radix
with `s_{D-1} >= kAtomLeaves` it FILLS them; where the level is narrower than
an atom the rectangle spans two levels and the fill is partial
(`carry/box.md`).

Symbols, restated from the spec at first use here: `D` the number of routing
levels; `B_l` level `l`'s width; `k = 2^kappa` the uniform per-level sub-box
span; `m = prod_l m_l` the capacity multiplier, distributed innermost-first to
the atom floor, capped by the level's own width and balanced over the rest
(spec §2.5 as the atom-rectangle law generalizes it, `carry/box.md`);
`s_l = k * m_l` an owner's span at level `l`; `g_l = B_l / s_l` the owner-grid
extent; `BC = prod_l s_l` the leaves one owner holds; `atom` the 16-leaf page
granule (`rola::carry::kAtomLeaves`).

`decode.cuh` carries the box's fields on `DecodeParams` under `BoxPlan`'s own
names (`lat_k`, `lat_s[]`, `lat_g[]`, `lat_gsuf[]`, `lat_run_shift[]`,
`lat_local_bits`, and the unit's own `lat_inner_bits`, `lat_outer_bits`,
`lat_oshift[]`) because decode takes `(k, m)` and the widths as DATA rather than
as template arms — one built binary serves every topology the producer can emit,
so `BoxPlan`'s constexpr fields become runtime scalars derived by `BoxPlan`'s own
formulas. This doc is their authority; `decode.cuh` merely carries them.

---

## <a id="the-digit-mask"></a>1. `build_digit_mask` — one bit per digit, race-free without atomics

A side's DIGIT MASK is `sum_l ceil(B_l/32)` words, one bit per digit of every
level, packed level by level (`mask_off[l]` / `mask_words[l]` locate a level's
slice, `mask_total` the whole plane). One thread owns one destination word and
reads only the amplitudes that word covers — the exact-nonzero test spec
LATTICE-1 assumes throughout, never a threshold — so the pass needs no atomic
and no second pass to merge partial words.

`level_slice` reads `count <= kDecodeMaxSpan` digits of one level as a single
word through `extract_bits` (`decode.md#extract-bits` — the funnel shift is
written once and this file is one more caller of it, not a second copy of it).
An owner's span and a sub-box's span are each one such extract, which is what
`kDecodeMaxSpan` is a capacity FOR.

## <a id="the-terms"></a>2. The enumeration: the UNION OF THE PRODUCTS, as a disjoint sum

At `T = 1` each side's support is a product, `supp(R) = prod_l R_l` and
`supp(W) = prod_l W_l` over per-level DIGIT sets. The set one step touches is
therefore `(prod_l R_l) u (prod_l W_l)`. What an earlier walk ENUMERATED was
`prod_l (R_l u W_l)` — the product of the unions — and the two are not the same
object. Under the split-level declaration every level has a dense side, so every
level's union is FULL and the product of unions is the DENSE product at every
support: 4096 units at the flagship whether the token routes to 65536 leaves or
to 512 (k31_decocc M1.1 measured exactly that).

The union of two products is not a product. It IS a DISJOINT UNION of `D + 1` of
them, and that is what makes it walkable by the same three loops:

```
(prod_l W_l)  u  [ (prod_l R_l) \ (prod_l W_l) ]
   = TERM 0   (+)  (+)_{l} TERM (l + 1)

TERM 0     = prod_l W_l
TERM l + 1 = [prod_{l' < l} (R_l' & W_l')] x (R_l & ~W_l) x [prod_{l' > l} R_l']
```

EXHAUSTIVE: a leaf of `prod R` outside `prod W` has a SMALLEST level `l` whose
digit is not in `W_l`; that `l` names its term. DISJOINT: term 0's level-`l`
digit is in `W_l` and term `l + 1`'s is not, and terms `i < j` disagree at level
`i - 1`. So every leaf either side reaches is enumerated EXACTLY ONCE — which is
the whole reason this is a legal replacement and not merely a smaller one:
UPDATE-THEN-READ stays a single visit per leaf (`decode.md#update-then-read`),
the deposit stays a PARTITION over write owners because term 0 alone deposits
(`decode.md#the-unit-partition`), and not one byte of state traffic moves.

**What it buys, and where it does not.** The decomposition prunes OWNERS. A term
is empty the moment any of its levels' digit sets is, so `R_l <= W_l` at every
level — the dense cell — collapses every read term to nothing and leaves the walk
exactly the write product. Measured at the flagship, `BH = 8`, units per
batch-head against the product-of-unions' 4096:

| cell | T0 | T1 | T2 | total | vs union |
|---|---|---|---|---|---|
| dense | 4096 | 0 | 0 | 4096 | 1.00x |
| ALT k16 scattered | 2048 | 4096 | 0 | 6144 | **0.67x (worse)** |
| ALT k4 scattered | 512 | 1024 | 0 | 1536 | 2.67x |
| ALT k16 clustered | 512 | 512 | 0 | 1024 | 4.00x |
| ALT k4 clustered | 128 | 256 | 0 | 384 | **10.67x** |

`T2` is empty at every ALT cell because level 1's write side is dense, so
`R_1 \ W_1` is empty. THE SCATTERED CELL IS THE LOSS CASE AND IT IS STRUCTURAL:
level 0's read support there is 240 digits spread over all 32 owners, so
`R_0 \ W_0` spans the same owners as `R_0` and the two products simply ADD. A
support that is sparse in LEAVES but dense in OWNERS has nothing for an
owner-granular enumeration to prune, on either enumeration.

<a id="the-refused-fallback"></a>**THE FIFTH KIND WAS BUILT AND REFUSED.** Since
both counts — the decomposition's `sum_t units[t]` and the product-of-unions'
`units` — are EXACT WORK COUNTS this block already computes, a fifth `R_l | W_l`
owner list plus a fallback term walking it whenever it is smaller is a policy
with no threshold in it at all, and it targets exactly the scattered cell above.
It was built, gated green and benchmarked, and it LOST:

* its fifth list costs one more `cub::BlockScan` per level in a prologue that is
  the step's dominant cost wherever `BH` is small (`decode.md#per-cta-prologue`),
  which cost every cell 1 to 5 %;
* and the unit count it selects on DOES NOT PREDICT THE WALL — at `BH = 1` the
  scattered cell ran 21 % SLOWER on the fallback's 4096 units than on the
  decomposition's 6144.

It is recorded here rather than shipped, and `kKinds` stays four.

## <a id="the-write-grain"></a>2.2 The write product's own grain — live DIGITS, not live owners crossed with a run radix

Every rank space in §2 is `live owners x the FULL outer run radix`: the owner list prunes
OWNERS and the run field **never prunes at all**. That is exact wherever a level's support
is dense inside the owners it touches, and it is the union walk's own shape — which is why
the union walk never paid for it: its level-`l` support is `R_l u W_l`, and under the
split-level declaration one side of every level is dense.

**TERM 0 IS THE ONE TERM THAT IS SPARSE INSIDE ITS OWNERS BY CONSTRUCTION.** Under ALT the
write side is the sparse side at the outer level, and its live digits scatter across the
owner grid. Measured at the flagship, `ALT k16 scattered`, per batch-head:

| term | rank space | units | live | idle | leaves per LIVE unit |
|---|---|---|---|---|---|
| 0 (write product) | owners x run radix | 2048 | 256 | **87.5 %** | 16.00 |
| 1 (read remainder) | owners x run radix | 4096 | 2640 | 35.5 % | 1.45 |

87.5 % is not merely wasted `resolve_unit` calls. The dead units are INTERLEAVED with the
live ones — one live unit in every eight — so seven of a CTA's eight warps idle while the
eighth walks sixteen slots. That warp-level imbalance is most of the cost, which is why
removing it bought far more than the unit count alone predicts (§the ledger in the batch
findings: `-7.3 us` measured against `+0.4 us` predicted from `c_u` alone).

**SO TERM 0 ENUMERATES LIVE DIGITS** at every level but the innermost, from a per-level
WRITE DIGIT LIST (`dlist`, `sum_{l < D-1} B_l` entries, built by the same block scan that
builds the owner lists — a third 10-bit field in the same packed `int`). The innermost
level keeps its owner, because it contributes a whole run as ONE SLICE rather than a digit.
A digit splits back into `(owner, run)` by shifts (`o = d >> lat_sbits[l]`,
`r = d & (lat_s[l] - 1)`), so the unit's base is assembled exactly as before and a unit is
still one CONTIGUOUS ALIGNED RUN.

| cell | term 0 before | term 0 after | total before | total after | union |
|---|---|---|---|---|---|
| dense | 4096 | 4096 | 4096 | 4096 | 4096 |
| ALT k16 scattered | 2048 | **256** | 6144 | **4352** | 4096 |
| ALT k4 scattered | 512 | **64** | 1536 | **1088** | 4096 |
| ALT k16 clustered | 512 | **256** | 1024 | **768** | 4096 |
| ALT k4 clustered | 128 | **64** | 384 | **320** | 4096 |

The dense cell is untouched — every digit is live, so live digits and
`owners x run radix` are the same 4096 — which is the test that this is a change of GRAIN
and not of density: there is no branch and no threshold anywhere in it.

**THE REMAINING TERMS KEEP THE OWNER-AND-RUN SPACE.** The same ledger prices them: they
idle 1.6 % to 35.5 %, never the majority that term 0 did, and a second digit list per kind
is shared memory the `D = 4` residency cannot afford.

### Two orderings, both DRAM decisions, both measured

The flat digit field cannot reproduce the union walk's fully sequential sweep — ascending
`base` needs `(o_0, o_1, r_0)` lexicographic and a flat digit bundles `(o_0, r_0)` — so the
order was chosen by measurement:

* **FIELD ORDER.** Term 0 keeps the other terms' order: innermost level LEAST significant,
  level 0 most significant. Putting the outer digit field lowest instead sweeps one atom in
  every `s_0` across the whole plane before returning for the next, and measured **+2.6 %
  on the dense cell**; this order consumes one level-0 owner's whole region before moving on.
* **<a id="the-rotation"></a>WARP ROTATION.** With the innermost owner lowest, consecutive
  units are one owner box apart, so a CTA's eight warps reach eight atoms `BC` leaves apart.
  The low `log2(kDecodeWarps)` bits of the loop index are therefore ROTATED into the bottom
  of the finest outer level's digit field: a CTA's warps then take CONSECUTIVE ATOMS while
  the CTA's stride still sweeps one region at a time. Worth a further **0.7 %** on the dense
  cell at `BH = 1`. It is a BIT PERMUTATION of a power-of-two index space, hence a
  bijection, so the walk still visits every unit exactly once and the deposit partition —
  one owning CTA, warp and lane per row — is exactly the one it was.

### <a id="the-candidate-slot"></a>The candidate atoms keep the old space, and that is the whole of the compatibility argument

A flat digit rank is **not monotone in atom id**: at the flagship spans
`base = 2048*o_0 + 128*o_1 + 16*r_0`, so ascending `(d_0, o_1)` runs `..., 1920, 16, ...`.
The walk does not care. The CANDIDATE-ATOM enumeration does — the pool claim's canonical
ranks are defined in ASCENDING ATOM ID ([`decode.md#the-claim`](decode.md#the-claim)) — so
term slot `D + 1` holds the write product AGAIN in the old `owners x run radix` space, and
`candidate_atom` reads that slot through `resolve_candidate_unit` -- a resolver of its own,
because the candidate enumeration asks the write mask a MEMBERSHIP question and nothing
else. It needs neither side's amplitude product, neither decay rate and not the read slice,
and its kind is `kKindW` at every level by definition, so the term table and the
kind-indexed addressing collapse to constants. Resolving it through the WALK's resolver
made it pay for all of that once per candidate rank per CTA -- sixteen ranks per thread at
the dense cell -- and `ncu` priced that at **5.9 % of the step's whole instruction count**. It is the enumeration the claim was ratified on, byte for
byte; nothing about the claim, the shadow row or the atom bits moved. `tests/integration/
test_decode_paging.py` is what holds that: it exercises growth, the pool claim and the
replay path, and it passes unedited.

## <a id="the-owner-lists"></a>2.1 `build_owner_mask` — four kinds, one ballot per kind per level

The terms ask for four per-level digit sets — `W_l`, `R_l`, `R_l & W_l`,
`R_l & ~W_l` — so there are four OWNER LIVENESS BITMASKS, `omask[kind][level]`,
one bit per owner coordinate, with `n_o` the realized count. THE DIGIT MASKS STAY
TWO: a kind's bits are a boolean function of the read and write bits
(`kind_bits`), formed where they are read, on single digits and on whole owner
spans alike.

Each thread reads ITS OWNER'S SPAN ONCE per side (`lat_g[l] <= kDecodeThreads`
follows from the level-width capacity, `decode.md#level-width-capacity`, so one
block covers a whole level's owner grid) and all four kinds fall out of that one
pair of slices — exactly as they did when this function compacted them.

**THE BALLOT IS WHAT REPLACED THE BLOCK SCAN.** A warp's 32 lanes already hold 32
consecutive owners' answers, so one `__ballot_sync` per kind IS that level's
32-owner word and its lane 0 stores it; the realized count rides the same word as
a `__popc` folded by one shared atomic. The four `cub::BlockScan` exclusive sums
that used to compact those bits into lists — and the TWO BARRIERS PER LEVEL they
carried — are gone, and with them the only reason the prologue's cost grew with
the live owner count rather than with the owner grid.

**THE BALLOT WORDS COMPACT THEMSELVES.** `compact_owner_lists` runs on the other
side of one barrier: an owner the ballot found live writes ITSELF into
`olist[kind][level][rank]`, where `rank` is the popcount of its kind's bits below
it — at most `kDecodeMaxLevelWidth / 32 = 8` `POPC`s, and exactly one at every
topology whose level is 32 owners or narrower, which the flagship's two levels
both are. The WRITE DIGIT LIST is compacted the same way straight off `mask_w`,
its count being that mask's popcount ([2.2](#the-write-grain)).

**WHY THE LIST SURVIVES AND ONLY THE SCAN DIES — MEASURED.** Reading the bitmask
directly at USE time (`select(i-th set bit)` by `__fns`, no list plane at all) was
built and benchmarked, and it LOST BADLY: `__fns` is not one instruction on sm_86
but roughly 56 SASS with reconvergence-stack ops, and the ADMISSION resolves one
owner rank per LEVEL per CANDIDATE ATOM per CTA per step — 4096 candidates at the
flagship — so the select cost **+2.42 M instructions on the dense `BH = 1` step,
+25 % of the whole step**, against a prologue saving roughly a tenth of that. The
per-CTA replication is what makes a use-time cost so much more expensive than a
build-time one, and it is the same fact [`decode.md#per-cta-prologue`] states about
the derivation generally.

<a id="ballot-uniformity"></a>**HAZARD — ballot uniformity.** `build_owner_mask`'s
`__ballot_sync(0xffffffff, ...)` runs with the WHOLE CTA present: the level loop is
unrolled over the template `D` and no thread branches around it. An owner beyond
`lat_g[l]` contributes zero slices and therefore a zero bit, and the store is
predicated AFTER the ballot on `lane == 0 && word < omask_words[l]` — never before
it. Predicating the collective instead would split the warp at an instruction that
needs every lane (`KERNEL_STANDARDS.md` §12).

**Each level's live count is padded to the next power of two.** A level's field
width is `ceil_log2(n_o[kind][l])`, its position the running sum of the levels
below it, and a unit's owner rank decomposes into per-level fields by SHIFTS AND
MASKS alone — never a division. A rank in the padded-but-unrealized part of a
level's field is `i >= n_o[kind][l]`, marked `invalid` and skipped rather than
wrapped: the padding costs some dead ranks in exchange for never issuing a device
integer division in the walk's innermost enumeration.

`term_layout` turns those counts into each term's rank space and its unit count,
and `publish_term_layout` calls it ONE THREAD PER TERM because the terms are
independent — `D + 2` threads doing at most `D` `ceil_log2_dev`s each, rather than one
thread doing twenty-four in series while the rest wait at the barrier. The unit
count is zero wherever any of that term's kinds is empty, so an absent owner set
kills a term outright rather than walking zero-width ranges.

**THE THREE FIELDS RIDE ONE PACKED WORD.** `s_fld[term][level]` carries the field's
position (5 bits), its width (5 bits) and the level's realized count (the rest), so
the resolver's per-unit read is ONE shared load where the separate `bits`, `shift`
and `n_o` arrays were three. The table stays a per-CTA prologue object rather than
a per-warp register set, and that is a MEASURED register-budget fact: holding `D`
field words live across the walk spilled 48 words at the `D = 4` arm against the
64-register cap that 1024 resident threads impose.

## <a id="the-unit"></a>3. `Unit` / `resolve_unit` — one unit is an `(owner, r_0 .. r_{D-2})` pair

A UNIT is what the grid partitions: one owner coordinate `O` together with
EVERY RUN OFFSET BUT THE INNERMOST. Under the mixed-radix intra-box order the
innermost level's run is the CONTIGUOUS TAIL of the owner-local index, so a
unit's leaves are `s_{D-1}` consecutive lattice indices at an `s_{D-1}`-aligned
base. At the flagship spans `(8, 16)` that run IS one 16-leaf atom, at full
occupancy, and the unit's 16 slots are level 1's digits — which is the whole
reason the unit is this shape rather than the interleave's `(owner, sub-box)`.

`resolve_unit` decomposes a unit index by shifts alone WITHIN ITS TERM: the OWNER
RANK occupies the high bits, one padded field per level (§2.1), and the low
`lat_outer_bits` are the fixed run offsets in the SAME mixed radix the owner-local
index uses, so `outer << lat_inner_bits` is the unit's base inside its owner with
no multiply. The term is the WALK'S OUTER LOOP and is passed in
(`decode.md#the-term-loop`), which makes every level's kind — and therefore its
list base, its count and its field widths — loop-invariant.

THE UNIT IS HANDED BACK ALREADY IN THE WALK'S VOCABULARY, and that is what keeps
the term out of the tight loop entirely. The slot loop iterates one slice and
tests `in_r`/`in_w` exactly as the product-of-unions walk did, because
`resolve_unit` CONDITIONS the two slices on the way out:

* TERM 0 is the write product, so `cw` IS the enumerated run, every one of its
  leaves is in `supp(W)`, and `cr`/`live_r` answer the read question unchanged. A
  dead write digit above the innermost level zeroes both slices — the unit dying.
* TERM `l + 1` is disjoint from `supp(W)` and inside `supp(R)` by construction, so
  it reports `live_w = false` (nothing of it is written), `live_r = true`, and its
  enumerated run in `cr`.

So the walk needs NO THIRD SLICE REGISTER and no term arm: `bits` is `cw` for
term 0 and `cr` otherwise, and the register budget the tight loop is measured on
(`decode.md#declared-residency`) is the one it always had.

The result is a `Unit<D>`: `owner`, `base` (the unit's first leaf as a lattice
index), a `valid` flag (false where the padded rank space overshoots a level's
realized count, §2), `live_r`/`live_w` (false where some FIXED level's digit is
absent on that side — the pruning the product structure
`supp(R) = prod_l supp_l(R)` buys: one absent level kills the whole unit for
that side, in ONE test rather than a per-leaf one), the innermost level's
staging base `dlast`, both sides' `s_{D-1}`-wide slices `cr`/`cw`, and the
FIXED LEVELS' AMPLITUDE PRODUCTS `ar`/`aw`/`rate`.

Those products are the unit's second point. Because a unit fixes every level but
the innermost, their amplitudes are resolved ONCE HERE and a slot below costs a
single innermost amplitude rather than `D` of them — where the interleaved
walk's unit cut every level and had to re-form the whole product per leaf.

THERE IS NO SECOND UNIT SHAPE. A topology whose innermost span is thinner than
an atom does not get a deeper walk: it gets SEVERAL UNITS PER ATOM, which the
CANDIDATE enumeration absorbs (`lat_units_per_atom_shift`,
`decode.md#candidate-atoms`) and the walk never sees. An earlier draft made the
unit's depth a runtime quantity so that a unit was always at least an atom; it
was measured to SPILL the step's register budget at `D = 2` and `D = 3` — the
extra loop level held every live value of the walk's tightest scope across a
trip count of one at every shipped topology — and the batch's findings carry the
census rows. This is the "no branch to dodge edge cases" rule applied: one
strategy, uniformly, and the degenerate config pays at the candidate enumeration
instead of in the walk.

`Unit` is the object the step's walk (`decode.md#the-unit-partition`) and the
candidate-atom enumeration (`decode.md#candidate-atoms`) both resolve a unit
into; neither reads the mask or owner-list arrays through any other route.

## <a id="capacities"></a>4. The lattice's declared capacity

`kDecodeMaxSpan = 32` (`decode.cuh`), and it is now the ONLY one. An owner's
span `s_l` is read as a SINGLE 32-bit slice — both when a level's owner grid is
built (§1) and when a unit's innermost run is resolved (§3) — so the span is a
gate the host refuses past, never a slower fallback, at
[`fill_lattice`](#the-refusals). `rola.ops.lattice` mirrors it (`_MAX_SPAN`) so
a topology the device would refuse is refused on the host, once, before a
launch, rather than negotiated at one; the decode build stamp carries it, so a
host that outgrew the device's box fails a device-side check
(`decode_api.md#build-stamp`).

<!-- ABSENT: kDecodeMaxCap kDecodeMaxSubBox -->
the interleave's other two ceilings died with it: `kDecodeMaxCap` bounded the
capacity radix a unit's low bits carried, and `kDecodeMaxSubBox` bounded the
intra-sub-box scan. Under mixed radix a unit is a contiguous run, there is no
capacity radix separate from the run offsets, and there is no sub-box scan to
bound.

## <a id="the-refusals"></a>5. `fill_lattice` — the box's constraints, as runtime checks

Decode takes its lattice as DATA, not as a template arm, so every constraint
`box.cuh`'s `BoxPlan` states as a `static_assert` is a `TORCH_CHECK` in
`decode.cu::fill_lattice`:

- `k` and `m` are exact powers of two;
- `k <= kDecodeMaxSpan`;
- per level, `s_l = k * m_l <= kDecodeMaxSpan` and `B_l % s_l == 0` (K6: an
  owner sits inside a level — `m_l` is spec §2.5 as the atom-rectangle law
  generalizes it, innermost first to the atom floor, capped by the level's own
  width and balanced over the rest, restated in `fill_lattice`'s loop and in
  `rola.ops.lattice.box_shape`);
- `g_l <= kDecodeThreads` — a level's owner grid is built in one block pass;
- `owners * BC == N` — the lattice PARTITIONS the leaf space exactly, with no
  remainder leaf and no double-covered one.

THE ATOM FLOOR IS A PREFERENCE, THE WIDTH CEILING IS THE LAW. A topology whose
widths cannot reach `s_{D-1} >= kAtomLeaves` (an odd level, a level narrower
than an atom) still has a lattice and still decodes: the floor is taken as far
as `log2 m` and the level's width allow, the capacity spills outward, and the
atom then spans more than one level. What IS refused is a capacity no span
vector can carry — `sum_l a_l != log2 m`.

A PAGED backing adds ONE more, checked only where `page_table` is present
(`rola_decode_forward`, cross-referenced from `decode.md#candidate-atoms`):
`BC >= kAtomLeaves`, i.e. an owner holds a whole number of atoms. That is what
makes an atom and a unit nest one way or the other, which is what makes the
candidate enumeration a PARTITION. The interleave needed two clauses here
(`k^D % kAtomLeaves == 0` and `4 % log2(k) == 0`); mixed radix discharges both
by the order itself — an atom is 16 consecutive `local` values at a multiple of
16, hence an aligned digit rectangle for EVERY admissible span vector
(`box.cuh`, K3/K4). A DENSE backing never asks an atom a liveness question and
carries no constraint.

`rola.ops.lattice.derive_lattice` and `box_shape` are the SAME arithmetic,
host-side and static, and are what choose `(k, m)` for a topology before any of
this runs — `fill_lattice` re-derives nothing; it validates the pair the host
already committed to and fills the suffix products and shift widths
`resolve_unit` reads (§3). `rola.ops.carry` reads its box from the same module,
so the two families cannot disagree about where a leaf is.

## <a id="the-seam"></a>6. `rola.ops.lattice` — the oracle and test seam, and nowhere else

`permutation(widths, k, m)` builds `pi`, an `[N]` int64 tensor with
`pi[canonical_leaf] = lattice_leaf` — spec §2.4's theorem that `pi` is a FIXED
BIT PERMUTATION, computed here as the explicit mixed-radix formula (spec
§2.3's forward map) rather than assumed. `to_lattice`/`to_canonical` are the
one legal place a state plane crosses between the oracle's canonical leaf
order and the lattice order the kernel is BORN in: `to_lattice` via
`index_copy_(1, pi, canonical)`, the inverse via `index_select`.

No shipped kernel path ever calls `permutation` or converts a plane — a
kernel never HOLDS a plane in canonical order to begin with, so there is
nothing on the production path to convert. The seam exists for the oracle
(which is canonical forever, `docs/internals/DELETIONS.md`'s standing rule)
and for a test that must compare the two: `_write_atom_bitmap`
(`decode.md#second-derivation`) and the fp64 oracle battery both cross through
here exactly once, on the way into or out of the comparison, and never inside
a measured step.

`LATTICE_CAP_BC = 128` is the one number in `derive_lattice` that is a CHOICE
rather than a consequence: the leaves ONE OWNER holds, which is what the carry
family's register budget bounds, and the ceiling the solver searches down from.
It is stated as `BC` rather than as a capacity multiplier because `BC = k^D * m`
grows with DEPTH — a ceiling on `m` bounds the box at one `D` only, and `m = 8`
is a box of 128 at `D = 2` and 2048 at `D = 4`. The objective is spec §7.3's in
its order: maximize `BC` under the cap, then the SHALLOWEST atom rectangle, then
the largest `k`. That reproduces the built arms (`(64,64)` and `(256,256)` both
`(4, 8)`) and selects `k = 2` at `(16,16,16)`, whose one-level rectangle the
`k = 4` box cannot reach.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode_lattice.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kkindw"></a>
### `kKindW`

THE FOUR PER-LEVEL DIGIT SETS the terms are built from, as the index of one owner-list
plane each. `kKindW` is also TERM 0's kind at every level, so the write product -- the
set the candidate atoms and the pool claim are enumerated from -- is a term of the walk
and needs no second list plane.
THERE IS NO FIFTH, UNION KIND, and that is a MEASURED refusal rather than an omission:
a `R_l | W_l` list plus a fallback term that walks it whenever it enumerates fewer
units than the decomposition was built, gated and benchmarked, and LOST on every cell
(its fifth list costs a block scan per level in a prologue-bound step, and the unit
count it selects on does not predict the wall -- the scattered cell at `BH = 1` was
21 % SLOWER on the enumeration with 1.5x FEWER units).
docs/internals/decode/decode_lattice.md#the-refused-fallback

<a id="kmaxterms"></a>
### `kMaxTerms`

`D + 2` term slots: the write product, one per level, and -- LAST -- the CANDIDATE
LAYOUT. The candidate slot is the write product again, enumerated the way it always
was (live owners crossed with the FULL run radix), and it exists because the walk's
own write term no longer is: docs/internals/decode/decode_lattice.md#the-write-grain

<a id="build-owner-mask"></a>
### `build_owner_mask`

THE OWNER LIVENESS BITMASK: per KIND and level, ONE BIT PER OWNER COORDINATE, set
where that kind's digit set has any digit inside the owner's span. Thread `tid` owns
owner `tid` on both sides, reads its span ONCE, and all four kinds fall out of that one
pair of slices exactly as they did; `lat_g[l] <= kDecodeThreads` follows from the
level-width capacity, so one pass covers every owner. The caller's barrier before this
must have published both digit masks and zeroed the two count rows.

THE BALLOT IS WHAT REPLACES THE BLOCK SCAN, and that is this function's whole subject.
The compacted lists this supersedes cost FOUR `cub::BlockScan`s and TWO BARRIERS PER
LEVEL to build a table whose only use is `list[i] -> owner`; a warp's 32 lanes already
hold 32 consecutive owners' answers, so ONE `__ballot_sync` per kind IS that level's
word, written by its lane 0, and `list[i]` becomes a find-nth-set over the same word
(a find-nth-set over that word). The realized counts ride the same
word as a popcount, so the level's `n_o` needs no second pass either.
THE COUNTS ARE FOLDED BY ATOMIC, NOT SCANNED: a count is the popcount SUM over a level's
at most eight words, and the eight lane-zeros that hold them are in different warps.
THE DIGIT LIST HAS NO PLANE. Term 0's outer levels enumerate live WRITE DIGITS, and the
write digit mask already IS one bit per digit -- so the list is `mask_w` itself, its
count is `mask_w`'s popcount, and the digit list's scan field disappears with it.

<a id="compact-owner-lists"></a>
### `compact_owner_lists`

THE COMPACTION, AND IT COSTS ONE `POPC` PER WORD. An owner that the ballot found live
writes ITSELF into its rank slot: its rank is the popcount of its kind's bits below it,
which is at most `kDecodeMaxLevelWidth / 32 = 8` popcounts and, at every topology whose
level is 32 owners or narrower, exactly one. The caller's barrier before this must have
published `omask`.

WHY NOT SELECT AT USE TIME. Reading the bitmask directly -- `select(i-th set bit)` by
`__fns` -- deletes these planes entirely, and was BUILT AND MEASURED: `__fns` is not one
instruction on sm_86 but ~56 SASS with reconvergence-stack ops, and the ADMISSION
resolves one owner rank per LEVEL per CANDIDATE ATOM per CTA per step (4096 candidates
at the flagship), so the select cost +2.3M instructions on the dense step -- +25 % of the
whole step, against a prologue saving of a tenth of that. THE SCAN IS STILL GONE: what
this replaces is `cub::BlockScan`, four of them per level with two barriers, not the
list. docs/internals/decode/decode_lattice.md#the-owner-lists

<a id="kfieldwidthmax"></a>
### `kFieldWidthMax`

A LEVEL'S RANK FIELD, PACKED INTO ONE REGISTER, and the packing is a REGISTER BUDGET
decision measured on the deepest arm: three parallel `int[D]` arrays cost twelve live
registers at `D = 4` and spilled the `d_v = 64` walk against its 64-register cap, while
one packed word per level costs four and unpacks in two ALU ops. The three fields are
the padded field's POSITION in the term's rank (5 bits -- at most `D` levels of at most
`kFieldWidthMax` bits each), its WIDTH (5 bits) and the level's REALIZED COUNT (the rest),
which a padded-past-the-end index is refused against.

<a id="term-layout"></a>
### `term_layout`

THE TERM LAYOUT: each term's own rank space, DERIVED INTO REGISTERS BY WHOEVER NEEDS
IT. `fld[l]` packs level `l`'s field position, its power-of-two padding and its realized
count, so a rank decomposes by shifts and nothing is divided around; the
return is the term's unit count, and a term whose kind is empty at any level contributes
NO units at all -- which is how `R_l <= W_l` at every level (the dense cell) collapses
the whole read remainder to nothing and leaves the walk exactly the write product.

ONE PACKED WORD PER LEVEL IS WHAT THE WALK READS, and it replaces THREE reads -- the
field width, the field position and the level's realized count were three separate
shared-memory arrays. The table stays a PER-CTA PROLOGUE OBJECT rather than a per-warp
register set, and that is a REGISTER BUDGET fact, measured: holding `D` field words live
across the walk spilled 48 words at the `D = 4` arm against the 64-register cap that
1024 resident threads impose, while the packed table costs no register at all and one
third of the loads. docs/internals/decode/decode_lattice.md#the-owner-lists

THREE LAYOUTS, NOT ONE, and the difference between them is this file's subject.
  * TERM 0, THE WRITE PRODUCT, enumerates LIVE DIGITS at every level but the innermost
    and live OWNERS at the innermost. Its outer fields are the LOW ones so consecutive
    units still walk consecutive atoms. Why: every other rank space is
    `live owners x the FULL run radix`, and the run radix NEVER PRUNES -- a write side
    that is sparse in digits WITHIN its owners (which is what the split-level
    declaration makes it, by construction) then enumerates `span` units for every live
    one. Measured at the flagship: 2048 units for 256 live ones.
    docs/internals/decode/decode_lattice.md#the-write-grain
  * TERMS `1..D`, the read remainders, keep the owner-and-run rank space. The ledger
    says they idle 1.6-35.5 %, never the >50 % that would pay for a second grain.
  * THE CANDIDATE SLOT `D + 1` is the write product AGAIN, in the OLD owner-and-run
    space, because the candidate-atom enumeration and the pool claim's canonical ranks
    are defined in ASCENDING ATOM ID and only that space is monotone in it
    ([`decode.md#candidate-atoms`]). The walk needs no such order and does not pay for
    one; the claim keeps exactly the enumeration it was ratified on.

THE FIELD ORDER IS INNERMOST-LEAST-SIGNIFICANT at every term, and it is a DRAM decision:
putting term 0's outer DIGIT field lowest instead sweeps one atom out of every `s_0`
across the whole plane before returning for the next, which measured +2.6 % on the dense
cell; this order consumes one level-0 owner's whole region before moving on.

<a id="publish-term-layout"></a>
### `publish_term_layout`

THE WHOLE TABLE AND THE RETIREMENT BOUND, published once per CTA. ONE THREAD PER TERM,
because the terms are independent: `D + 2` threads do at most `D` `ceil_log2_dev`s each
instead of one thread doing twenty-four in series while the rest wait. `umax` -- the
largest count the WALK's terms reach, which is the retirement bound -- is folded by
`atomicMax` inside this function's OWN barrier, so no CTA pays a second block
synchronisation to learn it: docs/internals/decode/decode.md#retirement
the candidate slot is not the walk's, so it is not in the retirement bound.

<a id="unit"></a>
### `Unit`

ONE UNIT of the walk: the `(owner, r_0 .. r_{D-2})` pair a rank names -- an owner
together with EVERY run offset but the innermost. Under the mixed-radix intra-box order
the innermost level's run is the CONTIGUOUS tail of the owner-local index, so a unit's
leaves are `s_{D-1}` consecutive lattice indices at an aligned base: at the flagship
spans `(8, 16)` that is EXACTLY ONE 16-leaf atom, at full occupancy, with the innermost
level's 16 digits as the unit's slots.

What the unit carries is therefore ASYMMETRIC BY CONSTRUCTION, and that asymmetry IS
the unit's point: every level but the innermost contributes ONE FIXED DIGIT, so its
amplitude and its membership are resolved ONCE HERE and the slot loop below costs one
innermost amplitude per leaf rather than `D` of them. Only the innermost level
contributes a SLICE (`cr`/`cw`, one 32-bit extract) that the slot loop iterates by
`__ffs`. `valid` is false where the padded rank space overshoots a level's realized
count. docs/internals/decode/decode_lattice.md#the-unit

<a id="resolve-unit"></a>
### `resolve_unit`

THE UNIT INDEX, decomposed by shifts alone within its TERM. THE TERM IS THE WALK'S
OUTER LOOP AND THAT IS A LATENCY DECISION, not a bookkeeping one: it makes the kind of
every level -- and therefore the list base, the count, the field widths and every
predicate below -- LOOP-INVARIANT, so the per-unit dependent chain is the one the union
walk had. Laying the terms end to end in one index space and naming the term per unit
was measured at `+2.3 ns` of dependent latency per unit, which on the scattered cell
cost more than the units the decomposition saves:
docs/internals/decode/decode.md#the-term-loop
the grid partitions EACH TERM'S index space by the modular map it always used, so the
deposit -- which lives in term 0 alone -- is still one owning CTA, warp and lane per row.
Within a term the high bits are the OWNER RANK, one padded field per level, and the low
`lat_outer_bits` are the fixed run offsets in the SAME mixed radix the owner-local index
uses, so `outer << lat_inner_bits` is the unit's base inside its owner without a
multiply. A level whose digit is dead in the TERM'S OWN KIND kills the unit, which is
the pruning the product structure buys.

WHAT THE UNIT HANDS BACK IS ALREADY IN THE WALK'S VOCABULARY, and that is deliberate:
the slot loop iterates `live_w ? cw : cr` and tests `in_r`/`in_w` exactly as a union
walk does, because the two slices are CONDITIONED here so the term never reaches it.
  * TERM 0 is the write product: `cw` IS the enumerated run, every one of its leaves is
    in `supp(W)`, and `cr`/`live_r` answer the read question as they always did. A dead
    write digit above the innermost level zeroes both slices, which is the unit dying.
  * TERM `l + 1` is disjoint from `supp(W)` by construction and inside `supp(R)`, so it
    reports `live_w = false` (no leaf of it is written), `live_r = true`, and its
    enumerated run in `cr` (where the union walk's read slice would have been).
The walk therefore has NO TERM ARM, NO EXTRA SLICE REGISTER and one expression per question -- the
arithmetic the tight loop's register budget is measured on is unchanged.
docs/internals/decode/decode_lattice.md#the-unit
`DECAY` is a template arm so the dial product is not merely multiplied by one.

<a id="candidateunit"></a>
### `CandidateUnit`

THE CANDIDATE ATOM'S UNIT, and it is deliberately NOT a `Unit<D>`. The candidate
enumeration asks the write mask a MEMBERSHIP question and nothing else -- which atom
does this rank touch, and does the write side reach it -- so it needs neither side's
amplitude product, neither decay rate, and not the read slice. Resolving it through the
walk's resolver made it pay for all five, once per candidate rank per CTA, which at the
dense cell is sixteen ranks per thread: measured at 5.9 % of the step's whole
instruction count. Its kind is `kKindW` at every level by definition, so the term table
and the kind-indexed addressing collapse to constants here too.
docs/internals/decode/decode_lattice.md#the-candidate-slot

<a id="resolve-write-unit"></a>
### `resolve_write_unit`

THE WRITE PRODUCT'S UNIT, resolved from LIVE DIGITS. It is `resolve_unit`'s sibling and
it hands back the identical `Unit<D>`, so the walk that consumes it has no idea which
one produced it -- the whole difference is the rank space
([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)).

Every level but the innermost takes its digit from that level's WRITE DIGIT LIST rather
than from an owner list crossed with a full run radix, and the digit splits back into
`(owner, run)` by shifts because a span is a power of two: `o = d >> lat_sbits[l]`,
`r = d & (lat_s[l] - 1)`. So the unit's base is assembled exactly as before -- the owner
in mixed radix over `lat_g`, the runs at `lat_run_shift` -- and a unit is still ONE
CONTIGUOUS ALIGNED RUN of the innermost level.
`live_w` IS TRUE BY CONSTRUCTION: this enumeration visits write digits only, so the
membership test the owner-and-run space has to make per level does not exist here. The
read side still answers for itself, digit by digit.

<a id="krotlevel"></a>
### `kRotLevel`

THE WARP-LOCALITY ROTATION, and it is the second DRAM decision in this resolver.
The field order above puts the innermost OWNER lowest, so consecutive units are one
owner box apart and a CTA's eight warps reach eight atoms `BC` leaves apart. The
order that fixes THAT -- outer digits lowest -- sweeps one atom in every `s_0` across
the whole plane before returning, which is worse still. Both were measured.
SO THE LOW `log2(kUnitBlockWarps)` BITS OF THE LOOP INDEX ARE ROTATED INTO THE BOTTOM OF
THE FINEST OUTER LEVEL'S FIELD: a block's warps then take CONSECUTIVE ATOMS, while the
CTA's own stride still sweeps one owner region at a time. It is a BIT PERMUTATION of
an index space whose extent is a power of two, so it is a bijection and the partition
-- one owning CTA, warp and lane per deposited row -- is exactly the one it was.
docs/internals/decode/decode_lattice.md#the-write-grain
the index is CLAMPED, not guarded: at `D == 1` there is no outer level to rotate into
and the block below does not run, but the subscript is still compiled.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode_lattice.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE T=1 FACTOR TABLES -- the per-level objects the step's lattice walk is enumerated
from, and nothing else.

Three facts a reader must not miss:
  1. NOTHING HERE IS SIZED IN `N`. One bit per DIGIT (`sum_l B_l` bits per side) and
     one entry per live OWNER COORDINATE (`sum_l g_l` at most) is the whole apparatus;
     the leaf space is never represented.
  2. Membership is the EXACT-NONZERO test on the amplitude, never a threshold -- the
     same defining expression the schedule's support bitmap uses.
  3. THE RANK SPACE IS A BIT FIELD, not a mixed radix over the realized counts: each
     level's live count is padded up to a power of two so a unit decomposes by shifts.
     A device integer division is tens of cycles and the enumeration runs once per
     candidate atom per CTA, which is where it would be paid.

THE ENUMERATION IS THE UNION OF THE TWO PRODUCTS, NOT THE PRODUCT OF THE TWO UNIONS,
and that is this header's whole subject. `prod_l (R_l u W_l)` is the DENSE product
wherever every level has a dense side -- which the split-level declaration makes true
at EVERY level -- while `(prod_l R_l) u (prod_l W_l)` is the set the step actually
touches. The two are separated by a factor of up to `2^D` at the flagship.

The union of two products is not a product, but it IS a DISJOINT UNION of `D + 1` of
them, which is what makes it walkable by the same three loops:

    (prod_l W_l)  u  [ (prod_l R_l) \ (prod_l W_l) ]
       = TERM 0   ]+[ (+)_{l=0}^{D-1} TERM (l + 1)
    TERM 0     = prod_l W_l
    TERM l + 1 = [prod_{l' < l} (R_l' & W_l')] x (R_l & ~W_l) x [prod_{l' > l} R_l']

EXHAUSTIVE: a leaf of `prod R` outside `prod W` has a SMALLEST level `l` whose digit is
not in `W_l`, and that `l` names its term. DISJOINT: term 0's level-`l` digit is in
`W_l` and term `l + 1`'s is not; terms `i < j` disagree at level `i - 1`. So every leaf
either side reaches is enumerated EXACTLY ONCE -- which is what keeps UPDATE-THEN-READ
a single visit and the deposit a partition over write owners.

FOUR KINDS, ONE PER PER-LEVEL DIGIT SET the terms ask for (`W`, `R`, `R & W`,
`R & ~W`), each with its own OWNER LIVENESS BITMASK -- one bit per owner coordinate, not
a compacted list; the term layout says which kind each level of each term takes. The two
DIGIT MASKS stay two -- a kind's bits are a boolean function of the read and write bits,
formed where they are read.

The lattice's formulas, the capacities and the cost accounting:
docs/internals/decode/decode_lattice.md

<a id="termkind"></a>
### `term_kind`

THE TERM TABLE, as arithmetic rather than as a table: term `0` is the write product,
and term `j >= 1` splits at level `j - 1`.

<a id="kindbits"></a>
### `kind_bits`

A KIND'S BITS, from the two sides' bits. One expression, used on single digits and on
whole owner spans alike, which is what keeps the digit masks at two planes.

<a id="near-line-53"></a>
### near line 53

One side's DIGIT MASK, one bit per digit of every level, packed level by level. One
thread owns one destination word and reads the amplitudes it covers, so the pass is
race-free without atomics and costs `sum_l ceil(B_l / 32)` words.

<a id="levelslice"></a>
### `level_slice`

Level `l`'s digits `[lo, lo + count)` as one word. `count <= kDecodeMaxSpan` is the
host's declared capacity, which is what makes an owner's whole span a single extract.

<a id="maskbit"></a>
### `mask_bit`

ONE DIGIT'S MEMBERSHIP, as ONE load and a shift. `level_slice` reads two words and
funnel-shifts them because an owner's SPAN can straddle a word boundary; a single digit
never can, and the unit asks this question `2 * (D - 1)` times.

<a id="base"></a>
### `base`

NO `owner` FIELD. The owner's mixed-radix index is a resolver LOCAL that dies into
`base`; carrying it in the struct kept a register live across the walk's tightest
scope for nothing, and with two resolvers in one kernel that cost two.

<a id="alive"></a>
### `alive`

THE CONDITIONING the walk's one expression rests on, and the whole of the term's
presence in this file's contract.

<a id="kdm1"></a>
### `kDm1`

§6: `D` is this function's template parameter, so `D - 1` is compile-time;
name it so the heuristic reads it as such.

<a id="near-line-421"></a>
### near line 421

THE SAME CONDITIONING `resolve_unit` applies to the write term, for the same reason:
the walk iterates one slice and asks `in_r`/`in_w` with one expression each.
