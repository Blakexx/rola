# The architecture

This is the foundation document: what the kernel family computes, the address space
it computes it over, the format of the state it carries, the grains its work is
partitioned into, which quantities are compiled and which are addressed at runtime,
what each kernel does, and how every structural mechanism maps onto the
architectures the repository ships for.

**Authority.** `docs/KERNEL_STANDARDS.md` is the constitution and the laws; this
document is the architecture those laws describe; `docs/internals/` is the
per-source-file mirror beneath it, and `docs/architecture/` holds the individual
decisions (the self-masking theorem, the affine recurrence, the factored
representation, fp32 accumulation, closed-world codegen) whose arguments this
document assumes rather than repeats. Where the three disagree, the constitution
wins, then this document, then the mirror. Nothing outside the repository is an
authority for anything stated here.

**Status marks.** A mechanism is either **in the tree** (the mirror page describes
running code), **ruled, not yet built** (the law is settled and this document states
it; the mirror page for the mechanism says what is in the tree today), or
**UNSPECIFIED (ruling pending)** — a place where a reader would reasonably expect a
law and there is not one yet. The third mark is deliberate: an invented rule in a
foundation document is worse than an acknowledged hole.

---

## 1. What the kernel computes

The consumer is a **gated linear-attention recurrence over a routed leaf address
space**. Once the routing allocations are materialized it is textbook gated linear
attention — one state row per leaf, a diagonal gate, a rank-one deposit, a linear
readout — and the fp64 oracle (`rola/ops/naive.py`) is written as exactly that. The
oracle is the executable specification; the kernels are optimizations of it, never a
second definition.

**Symbols, defined here and used unqualified below.** `B` batch, `H` heads,
`T` tokens, `D` routing levels, `B_l` the digit width of level `l`,
`N = Π_l B_l` the leaf count, `DV` (`d_v`) the value width,
`cols = DV + 1`, `BH = B·H`, `L` the sequence length of a call, `n` a leaf index,
`W` the window length in tokens, `C` the BATCH TILE — the tokens one fold or readout
batch carries, which is the token dimension of its matrix multiply and one slot of the
value ring, so a window's write list of `nw` entries is `ceil(nw / C)` batches —
`BC` the leaves one owner holds, `s(l)` an owner's span at level `l`, `g(l) = B_l / s(l)` the owner grid at
that level.

Per token `t` and leaf `n`:

    keep[t,n] = (1 - rate[n]) ^ c[t,n]                       the diagonal gate
    S[t,n,:]  = keep[t,n] · S[t-1,n,:] + Wt[t,n] · v~[t,:]   the rank-one deposit
    r[t,:]    = Σ_n R[t,n] · S[t,n,:]                        the readout, read-after-write
    y[t,:]    = r[t, 0:DV] / (r[t, DV] + eps)                the ratio

`R[t,n] = Π_l p_read[t,l,d_l(n)]` and `Wt[t,n] = g_write[t] · Π_l p_write[t,l,d_l(n)]`
are the per-leaf products of the per-level routing allocations — each level's
allocation is a normalized simplex over that level's `B_l` digits, and `d_l(n)` is
leaf `n`'s digit at level `l`. `rate` is the per-leaf decay built the same way out of
per-level dials; the clock `c` is the write product, detached. There is no query, no
key and no feature map anywhere in the family: the routing IS the addressing.

**The mass rides the state.** `v~` is the value row with a column of ones appended,
so the denominator of the ratio is the same recurrence run on ones — one state, one
recurrence, one readout, `cols = DV + 1` logical columns. The ratio readout is the
only normalization the family has; there is no second recurrence and no raw mode.

**Declared arithmetic** (KERNEL_STANDARDS §R17). Operands are bf16 everywhere —
there is no float operand path in any kernel. Accumulation is fp32, and that is
closed (`docs/architecture/fp32-accumulators.md`). The state's logical element is
fp32. **The readout operand is the top half of the fp32 accumulator, taken by
truncation, in every kernel** — the truncation is declared, not incidental: it makes
the hybrid split of §3 exact on rejoin, and the ratio cancels its uniform bias. The
oracle reports the difference in fp64 and the gate asserts against the bf16-mirrored
band (`docs/testing.md`). A round-half-up variant with an invertible rejoin is
designed and not built; it is reached only if a training result ever shows a
state-scale bias.

**The window decomposition.** A call's token axis is cut into fixed windows of `W`
tokens, global to every owner. The carry kernel computes each window's readout
against the state as of that window's start and then folds that window's writes; the
pairs of tokens INSIDE one window are the intra kernel's term. `inter(W) + intra(W)`
is `W`-invariant and equals the oracle — the identity that licenses two kernels
(`docs/internals/intra/intra_kernel.md`; the carry half is off this line, §6).

---

## 2. The leaf lattice

**Levels and widths.** The address space is a `D`-level mixed radix. Level `l` has
width `B_l`, a power of two; `N = Π_l B_l`. The widths are per level and are a
property of the state, not of the binary (§5).

**The canonical order is MSB-first mixed radix**: leaf `n` decomposes as
`d_l(n) = (n / stride_l) mod B_l` with `stride_l = Π_{l' > l} B_{l'}`, so level `0`
is the most significant digit and level `D-1` varies fastest. Every derivation in
the repository — the oracle's own strides, the page table, the atom keying, the
kernels' addressing — is this one order, and it is carried as a FIELD of the state's
descriptor rather than assumed, so that a second order would be a refusal and never a
silent reinterpretation of somebody's bytes (`docs/internals/state.md`).

**The page rectangle.** The storage granule is the **atom**: 16 consecutive leaves in
canonical order, aligned to 16. In digit terms it is the rectangle formed by the
trailing four canonical bits — the innermost levels taken whole plus, where the four
bits fall inside a level, an aligned contiguous sub-run of that level's digits, with
every level above fixed. That the atom is a digit rectangle at every admissible span
vector is a theorem of the order, not a constraint on the carve
(the reference body's `carry/box.cuh` on the parity-reference tip; this line re-derives it in
`csrc/rola/src/common/geom.cuh`).

**The owner lattice.** An owner (§4) holds a rectangle of digits per level: span
`s(l)` at level `l`, grid `g(l) = B_l / s(l)`, `BC = Π_l s(l)` leaves. The map from
the canonical index to the lattice index is a fixed BIT PERMUTATION — in the
canonical index level `l`'s field splits as `(o_l | r_l)`, the owner coordinate and
the offset inside its run; in the lattice index the same fields are regrouped as
`[o_0 … o_{D-1}][r_0 … r_{D-1}]`, every field keeping its width. Two consequences the
kernels live on: the owner index is a bit-field extract and never a division chain,
and an owner's leaves are one contiguous `BC`-aligned run by construction. Inside an
owner the order is mixed radix over the runs (`local = Σ_l r_l · Π_{l'>l} s(l')`), so
a bucket keyed on one level's digit owns a contiguous run of leaves.

**Spans are balanced under the width floor.** Every level is at least 16 wide
(`B_l` a power of two ≥ 16; a one-level state needs `B ≥ 32`), so the warp's spans —
compile-time from `(D, leaves_per_warp, warps_per_cta)` (§5) — never exceed a level's
width and no narrow-level case exists. The earlier atom-floored convention is
superseded: what it protected (the innermost run that amortizes a resolve and lengthens
a vector load) is a measured efficiency term, never a law, and decode's unit is the atom
regardless of prefill's spans (§4).

---

## 3. The state

The carried state is a paged tensor of `[B, H, N, cols]` logical shape and nothing
more. Its full contract — the membership test that keeps launch machinery off it,
binding, the backings, clone, the layer's continuation container — is
`docs/internals/state.md`; this section states the architecture that contract
encodes.

### 3.1 The format descriptor, and one admission law

**The state owns its format** (KERNEL_STANDARDS §R13). A `StateFormat` binds at the
state's first call, from the call itself, and carries:

| field | what it is |
|---|---|
| `D`, `B_l` | the depth and the per-level widths, each a power of two |
| `order` | the canonical order of §2, as a field rather than an assumption |
| `page_bits` | the page rectangle: the trailing four canonical bits, which may span levels |
| `DV` | the value width; `cols = DV + 1` is the page's LOGICAL width, not its layout |
| `dtype` | fp32 as split bf16 planes (§3.2) |
| `ids` | the page-table id space, `BH · N / 16` |

**Later calls are CHECKED against it, and a mismatch is a refusal at the seam naming
the field that differs.** The descriptor is fixed for the state's life — a re-layout
is an explicit state operation, never something a call does — and it travels with the
layer's continuation. Each kernel derives its addressing from (descriptor, its own
shape) in its prologue.

**Prefill and decode share ONE admission law: the descriptor.** Decode accepts
exactly what prefill accepts — the same depth, the same widths, the same `N`, the
same value width, bf16 operands, therefore the same topologies. There is no
decode-only shape, no decode-only tolerance and no decode-only admission branch
anywhere, tests included: a second admission law is a second contract to keep in step
with the first, and the two kernels partition ONE paged state.

**There is no ragged state.** `N = Π_l B_l` over powers of two, so a lawful state is
page-aligned by construction; a leaf axis that is not a whole number of pages is
REFUSED — not padded, not carried in a tail page, not served by a second layout.

### 3.2 Split planes, and the page

A page is one atom — 16 leaves — stored as **planes, not rows**:

    [16 × DV hi][16 × DV lo][16 mass hi][16 mass lo]

where `hi || lo` is the fp32 word exactly. The hi plane of a row is `2·DV` bytes,
which at `DV = 64` is one 128 B line; the interleaved fp32 row it replaces was
`(DV+1)·4` bytes and line-aligned only by accident. The mass rides its own blocks
rather than a column of every row. The conversion each way is one integer byte-permute
per pair of elements — no shared-memory traffic and no arithmetic
(`docs/internals/common/state_page.md`).

**The page stride is padded**: `page_stride = round_up(hi + lo + mass, 128 B)`, a pure
function of `DV` (4,224 B at `DV = 64`; 8,320 at 128; 2,176 at 32), so that a hi row
is line-aligned in every slot rather than in even slots only. The mass stays inside
the page; the descriptor's arena sizing and the dense backing use the same stride.
The cost is 1.5% of capacity at `DV = 64`. *(Ruled; the unpadded 4,160 B stride is
what is in the tree today, and the measured price of it is +9.6% of the DRAM traffic
on written atoms.)*

**Hi-only reads.** The readout operand is the hi plane (§1), so an atom a call only
READS has a lo plane no arithmetic can observe, and not moving it is exact rather than
approximate. A read-only atom therefore moves half the bytes; a written atom moves and
stores both planes and rejoins exactly. This is one mechanism in both kernels: prefill
gates its state slice, decode gates its unit.

### 3.3 Two backings, one set of bytes

**Paged** is the default: a page arena, a `[BH, N/16]` page table of slot ids, and
residency planned exactly from the call's own write set. Resident state scales with
the atoms the routing REALIZES, not with the `N` the topology provisions. **Dense** is
the same page sequence with every page resident, in the same split-plane atom-major
layout. Dense and paged are ONE kernel, one ABI, one accumulation order and one base
translation, and the gate between them is BYTE identity — asserted on an integer view
of the storage, never with a float comparison, because a stored word is two
neighbouring elements' halves and a byte-identical pair can read as NaN
(`docs/internals/paging/paging.md`). A caller cannot observe which backing a state
has except by asking.

`materialize()` is the `hi || lo` gather back to the logical `[B, H, N, cols]`
tensor. It is for the oracle, a cross-arm hand-off and a test — never on a measured
path, since allocating that tensor is exactly what paging exists to avoid.

### 3.4 Activity bits: two sets, one grain each

Residency is the arena's fact; **activity is the facts pass's fact**, and only the
second decides what a call TOUCHES (KERNEL_STANDARDS §R14).

* **PRIMARY bits — the kernel's grain.** Per (warp box, side), over the WHOLE CALL:
  one READ bit and one WRITTEN bit, never pre-ORed, emitted at the carve's own grain.
  The state slice is loaded once at the call's entry and stored once at its exit — the
  box lives in registers across every window — so the gate is per call: LOAD on
  `resident ∧ (read ∨ written)`, STORE on `written`. Nothing is kept per window: a
  window with nothing live for a box is a schedule fact, read off the ballots while
  walking, never stored. Because the gate never references the
  atom, prefill needs no page alignment and no row masks: owners splitting an atom
  write disjoint rows. *(Ruled, not yet built: the emission consumes the carve as a
  kernel input, which arrives with the runtime-addressing work of §5.)*
* **DERIVED bits — the page grain.** Per page, host-side: the OR of the primary bits
  over the warp boxes the page intersects, through the box→page map the descriptor
  gives. Exact allocation admits a page iff an intersecting box is WRITTEN this call;
  residency checks read the derived READ set; a decode step consumes this set, its
  unit being the page. *(In the tree.)*

Three properties follow. Dense and paged touch the IDENTICAL atom set, so the byte
gate certifies the skip logic and not merely the addressing. A resident-but-idle atom
costs no state traffic at all, so a call's state traffic is proportional to what IT
touches rather than to what the sequence has ever touched. And the price is the
in-place law: an atom the exit sweep skips must already hold what it carried in, so a
gated continuation's exit plane must BE its entry plane — the seam refuses two
distinct planes when an activity map is supplied.

---

## 4. The grains, and how they nest

Five grains, from the storage outward. **They nest on the page and nowhere else**
(KERNEL_STANDARDS §R14): every other pair of grains is free to cut across.

| grain | what it is | who owns it |
|---|---|---|
| **page** | 16 canonical leaves = one atom = one slot | the state's format |
| **decode unit** | one atom | the decode step |
| **prefill box** | the CTA's `BC` leaves: one contiguous lattice run | the carry launch |
| **stream box** | a stream's rectangle of digits inside the CTA's box | the carve |
| **warp box** | the leaf sub-box one warp holds in registers | the register shape |

**The decode unit is the atom** — not the innermost span. Decode has no boxes; it has
units, which are contiguous runs of consecutive canonical leaves with the outer digits
fixed (one memoized outer product; one absent digit kills the unit). Defining the unit
as the innermost span would shrink it whenever prefill balances its spans, so the unit
is fixed at the atom regardless of what prefill does; when the atom spans two levels
its 16-slot mask is the outer product of the two innermost digit masks, one extra
multiply per slot, once per unit. **Prefill and decode ownership are decoupled**: two
kernels partitioning the same paged state, whose entire shared contract is the atom,
the page table and the activity bits.

**Prefill owns whole ROWS.** A leaf row is a line; two owners splitting an atom write
disjoint rows, so nothing masks and nothing clobbers. Residency stays per page.

**Streams are an axis.** `S = kWarps / WPS`, where `S` is the stream count (the
state matrix of §1 never appears in the same expression), `kWarps` the CTA's warps
and `WPS` the warps consuming one stream. Each stream has a box — the tokens live in
it, from its own ballot — and a private segment; the `WPS` warps consuming a stream partition ITS box among
themselves, each warp taking the leaf sub-box it already has. **`S = 1` is the union
point**: one box (the CTA's), one union list, one shared segment, each warp its own
leaves — the base case, and the control row forever. `S` is a distribution knob, not a
fallback and not a threshold; the same code is correct and at floor at every value up
to one stream per warp. Streams are decoupled from the warp carve: raising `S`
re-partitions the token lists, not the leaf ownership. *(Ruled; `S = 1` is what is in
the tree. `S` is DERIVED: as large as shared memory admits at residency, from the
descriptor and the architecture's shared-memory size — the premise being that a stream
costs close to nothing, which the private-stream measurement is the standing test of;
never a density, never a runtime input.)*

**The grains are the same on both sides.** Stream box, accumulator tile, warp segment
and multiprocessor box are one vocabulary for the fold (the write side) and the readout
(the read side); each side derives its own carve of the same box, and the grains nest the
same way. The sides differ only in implementation: the readout has one stream per
accumulator tile, always — a reader's stream is its own tile's token list, nothing to
share — while a fold stream may be shared by the warps whose tiles consume it, because
the fold stages value rows into a stream's segment.

**The stream box is sized to its consumers' accumulator tile.** A stream's segment —
its expanded coefficient rows and value rows — covers exactly one accumulator tile's
leaves, the multiply its consumers issue. Where a warp's segment is one tile the stream
box is at least the warp box and the warps sharing a stream partition it; where a warp's
segment is many tiles (tensor memory) the stream box is smaller than the warp box and the
warp consumes the stream once per tile — the same token list, coefficients expanded for
that tile's leaves, the value rows staged once — so liveness is tested at tile grain and
shared memory is sized to one tile's stream, never to the segment. The tile count per
segment is a runtime trip count; it is one on the architectures in the tree. *(Ruled;
built with the pipelined body.)*

**Per-side carves.** Read and write are carved independently: a stream owns an affine
rectangle of digits per level, per side, and the private slot order puts the carved
(sparse) levels outermost. Liveness is the exact AND over levels of that side's own
amplitude runs — a value test, so a dense level's run is nonzero for every token and
drops out of the conjunction on its own, with no structure fact carried into the
kernel and no branch on a computed statistic. Per level at most one side is sparse, so
there is one stored row set per level and a dense side is absent-means-full; the two
sides' clause sets are different level subsets, which is why the schedule is two-sided
with independent counts. A read carve and a write carve that coincide are a transit
whose row map is the identity — the same code, never a second body (the transit's row
map is derived in `docs/internals/common/geom.md`; the body is off this line, §6).

**The read side has no stream axis: one stream per warp, always.** The write side has
a stream count only because staging `d_v`-wide value rows into private segments
duplicates bytes per stream; the read side duplicates nothing, so every warp reads its
own box's tokens.

---

## 5. The axis law

KERNEL_STANDARDS §R11 and §R12 in architectural form. **A compile-time axis is
legitimate only as a deterministic function of user input that the selector can pick;
everything else is derived, addressed at runtime, or constant.**

| class | quantities | why |
|---|---|---|
| **compiled** | `D`, `DV` | the register shape — the accumulator a warp holds is leaves-per-warp × `DV`, and the level nest is unrolled over `D`. Both are chosen by the user's model. |
| **derived** | `BC`, the spans, `kWarps`, `S` | from residency and the descriptor: leaves-per-warp = state registers / `DV`; `BC` = leaves-per-warp × `kWarps`; spans balance `BC` over the levels (§2); `S = kWarps / WPS`. |
| **runtime addressing** | the per-side modes, `B_l`, the orders, the row maps | CTA-uniform values read from the descriptor in the prologue: a few shifts, spans, per-warp offsets, row-map deltas and swizzle words per level. |
| **constant** | `W`, `C` | kernel internals; a short sequence runs partial windows. |

`W`, `C`, `warps_per_cta`'s `{8, 4}` set, `BC`'s derivation and the per-arch launch
table are declared once in `csrc/rola/src/common/constants.cuh` / `rola.engine.constants`
(`docs/internals/common/constants.md`); the S-from-SMEM rule for `S` lives there too.

**Shape is the warp's slot map, and it is compiled.** The quantity that fixes which
register holds which factor in the unrolled expansion — the map from a worker's leaf
slots to per-level digits — is REGISTER SHAPE, not addressing: a runtime span would
index a register array dynamically (a spill) or a switch over every layout in the hot
loop (measured 2× instructions). The derivation is generic in the architecture's state
capacity: `leaves_per_sm` (the register file on sm_80/90, tensor memory on sm_100)
balanced over the D levels is the SM's box; `warps_per_sm` is the constant that carves
it — a warp owns a segment of the box on every architecture; each SIDE's level order
divides the box, sparse levels first, until the warp count is reached, and what remains
is that side's warp segment. The compiled shape is the leaf layout of the MMA ACCUMULATOR TILE —
the slot map that fixes which leaf row of the tile each expanded coefficient lands on.
On sm_80/90 that tile is the warp's register fragment and today it is the whole segment;
on sm_100 it is the tensor-memory tile the MMA accumulates into and reads its A operand
from in place, and a warp walks its segment tile by tile (a runtime trip count over
identical bodies; the state never passes through registers to compute). The set
of shapes a side can select is generated from (D, log₂ warps_per_sm, the box's bit
vector) — its count is flat across architectures — and the selection per side is a
`uniform_switch` over that set outside the window loop. A CTA holds a contiguous subset
of the SM's warp segments; how many is a launch parameter and never enters the shape. *(Ruled; landing with the runtime-addressing work.)*

**One geometry block, one derivation.** The modes, the widths, the spans, the row maps
and the decode lattice are derived in exactly ONE place, from the descriptor, and
every consumer reads that block. A second derivation site is the defect the oracle's
own independence rule exists to catch elsewhere — inside the kernel it is simply a
way for two agreeing wrong conventions to pass every gate.

**What the move costs and what it does not.** The immediates become uniform-datapath
registers; the instruction is the same shift or logical operation with a register
operand, at the same throughput. What is lost is the constant folding of span-1 and
shift-0 levels — a few instructions per token per level, a ledger line, not a design
change. The static nest over levels stays static: only the immediates become
registers. The vector register count must not rise beyond noise, and the census is the
gate on that. *(Ruled, not yet built: today the modes and widths are template
arguments and the shipped arm set multiplies with them.)*

**SMEM is launch-sized.** Every shared tenant whose size depends on a derived
quantity — the transit region, the pools, the staging tiles, the live words — is sized
by the host from the descriptor and passed as
DYNAMIC shared memory, with offsets as uniform values. Never max-sized. Above the
48 KB static limit the one-time function attribute is set, on every architecture.
**Residency is ASSERTED at the seam from the launch size, as a refusal** — the launch
does not quietly get fewer blocks than the shape was derived for.

**No runtime trip count in a register-shaped loop.** The spans enter the hot loops only
as shifts and masks; the fold and readout loops iterate leaves-per-warp × `DV`, which
is compile-time. The census — vector registers flat, zero spill — is the proof.

**Structure branches, and the four layers that enforce them** (KERNEL_STANDARDS §R9
and its enforcement clause). A CTA-uniform branch on STRUCTURE is legal — a switch
over the admissible (span, shift) pairs is structure the declaration fixes; a branch
on a measured statistic (a density, a live count, "small L") never is. A legal
structure branch carries all four of: cases GENERATED from the arm's shape with a
static assertion on the count, never hand-enumerated; a compile-time exhaustiveness
assertion, so an arm that could receive an uncovered case does not build; a
launch-time refusal at the seam naming the uncovered case; and a trapping default,
with a post-build sweep of every arm at every admissible case on a tiny cell.

---

## 6. The kernels

Four kernels and one host-side derivation. They share the window grid, the descriptor
and the atom; they share no code path that could make a wrong convention agree with
itself.

**Both of them have since landed, and the two sections below were their
specification while they did not.** The clean-slate line (`development/queue/C_CLEAN_SLATE.md`)
deleted the stats passes and the carry body from this branch; §6.1's liveness pass is
now built (`csrc/rola/src/facts/liveness.{cu,cuh}`, `docs/internals/facts/liveness.md`)
and a carry body has shipped (`csrc/rola/src/carry/`, `docs/internals/carry/carry_kernel.md`),
though the pipelined body's actual mechanism diverges in places from §6.2's blueprint
below (its window/snapshot/head/pool structure is not identical to the fold-with-streams
and hybrid-transit design stage 3 and stage 4 describe, and the shipped arm's layout is
PLAIN rather than a `uniform_switch` over a generated member set) — §6.2 is left as the
design record it was written as rather than rewritten stage by stage against the shipped
mechanism, which this pass could not responsibly complete; `docs/internals/carry/carry_kernel.md`
is the authority on what actually runs. The deleted bodies are readable on the `k35-final` branch
tip, which is never edited again and which is the parity reference every stage of the
rebuild is graded against from its first probe. §6.3 and §6.4 — decode and intra —
describe kernels that are here and unchanged.

**Nothing is ported, and nothing is a prototype.** The rebuild reads the old bodies and
the measured records as SOURCES OF MECHANISM — each stage below names the file or the
record its mechanism was proven in — and then writes that mechanism in its final form,
at floor, inside the pipelined structure, from the first commit. There is no
correct-first-fast-later stage: a stage that loses to the reference on the mechanism it
replaces is red, and bad performance is a report, never a scope decision. "Ground up"
means the structure is designed for the whole design from the start, not that what has
been proven is discarded. *(Ruled.)*

### 6.1 The liveness pass — the bits

One pass over the routing amplitude planes emits everything the downstream launches
need in place of an amplitude scan: per side, per token, per level, ONE BIT PER DIGIT
in canonical order — a sparse level's bits voted from that side's own amplitudes, a
dense level's bits its static width mask — plus the per-level width plan and the
activity bits of §3.4, folded from the same words on the host. The emitted size per
side is `L × Σ_l B_l / 8` bytes: a function of the sequence and the address space's
WIDTHS, never of `N`. Because a pad digit's amplitude is exactly zero by
construction — the routing producer pads a level's logits with negative infinity before
the activation, and the descriptor records the logical width beside the padded one
(§3.1) — pad digits are never live, never mark a box and never allocate a page; the
kernel sees a dead digit like any other.

Why the bits, and why per DIGIT: a consumer's liveness test becomes word logic over
data it already holds, instead of the amplitude scan it replaces, which read
`2 Σ_l s_l` bf16 values per token per owner. The digit grain is what makes the pass
CARVE-INDEPENDENT — the carve is a runtime fact now (§5), so the pass cannot pre-vote
at it, and each consumer instead ORs the digits of its own span at each level and ANDs
the levels together, from bits that are the same for every consumer. That is the one
structural difference from the pass this replaces, which emitted rows already voted at
a compile-time carve; the property that vote had — a side spanning a level whole
contributes nothing, so a stream tests exactly the levels it carves — is preserved,
because a dense level's bits are all ones and drop out of the conjunction on their own.
Liveness therefore stays a VALUE test: no structure fact and no measured statistic
enters a kernel (§R9). The consumers are the carry's window schedule (its per-side,
per-window live-token union), the host folds that produce the primary and derived
activity bits of §3.4, and decode's paged admission.

*Where the mechanism was proven:* the union table, the atom-grain vote and the
block-grain liveness epilogue on the `k35-final` tip's
csrc/rola/src/facts/facts_kernel.cuh and `facts_launch.cuh`; the two-bit activity law
and the byte gate that certifies it at tag `record/p80b-activity-bits`. *The floor:*
each amplitude plane read exactly once (unique bytes — the pass is DRAM-bound); the
emitted bytes as above; the host folds linear in boxes × levels; and, on the consuming
side, the word operations per (warp, window, side) that the span OR and the level AND
cost, which is the term the layout is answerable for.

*(In the tree: the pass was the FIRST foundation feature of the clean-slate line, built
and tested on its own before the kernel body, with its word layout, its Python reader
and its host folds pinned by the contract stage as an executable model. Decode's own
step condenses its write-atom set on device rather than depending on this pass for it
(`docs/internals/decode/decode.md#atom-bits`), so decode's paged path was never actually
blocked on it the way this note once anticipated — `docs/open-work.md`.)*

### 6.2 Carry — the window body

Per window, per owner, against the state as of the window's start:

    Oᵀ[:, t]   += Sᵀ[:, box] · R~ᵀ[box, t]      PHASE 1, the readout
    Sᵀ[:, box] += V~ᵀ[:, t] · W~[t, box]        PHASE 2, the fold

**The ordering law is window-grain read-before-write on the owner's own slots** — a
statement about the window, not about a batch. A structure that read and folded per
batch would let a later batch's readout see an earlier batch's folded writes, and both
are inside one window, so those pairs belong to the intra term and would be counted
twice. The batch line therefore IS the window line, except across windows where the
owner is side-dead, which is the whole-window-skip harvest.

**The launch surface takes the strict shape and refuses the rest.** The API accepts any
shape and serves it by padding; the kernel accepts only what the descriptor declares —
powers of two `≥ 16`, a shipped `DV`, `N` a multiple of 16, bf16 operands, a descriptor
matching the arm — and refuses everything else by name. It never pads and never masks,
and the seam is the only translator (`docs/api.md`, KERNEL_STANDARDS §R13). Until a
body exists the surface fails for ONE honest reason, that there is no implementation:
no stub computes a wrong answer and no test is skipped to hide the absence.

**The body is built in seven stages**, each one a shipped increment on the foundation's
features, each gated by the oracle, a sanitizer run, a census reporting the state
fraction of the register file, and a probe against the reference body on `k35-final`.
The pipeline is not the last stage: the structure the later stages fill — one CTA of
eight warps at 256 registers, double buffering, split-phase edges — exists from the
skeleton onward, so no stage retrofits a schedule.

**1. The skeleton.** The descriptor read and its admission refusal; the geometry
block's values derived ONCE in the prologue into uniform-datapath registers; the page
I/O — the box loaded once at the call's entry and stored once at its exit, gated by the
primary activity bits (§3.4) — the die-fast prologue that exits before touching state
when a box is neither read nor written this call; and the launch surface above, on a
trivial one-window cell against the oracle. *Proven form:* the dispatch and its
by-name refusal, and the state load/store sweeps, in the reference's
`csrc/rola/src/carry/carry.cu` and `carry_kernel.cuh`; the addressing block itself
needs no reference — it is alive in this tree at `csrc/rola/src/common/geom.cuh`
(`docs/internals/common/geom.md`). *Floor:* the prologue's uniform-register and
instruction counts; zero spill at the register law's allocation; the state fraction;
state bytes = the box, once in and once out.

**2. The fold, with cooperative staging.** Three parts, not two: the live-token union
derived ONCE per side per window; a COOPERATIVE fetch of un-expanded factor rows and
value rows, by executor share, into ONE shared staging, with the expansion memoized in
registers; then per-stream sends from each stream's own ballot into its private
segment, and seal-and-consume, after which consumers are pure matrix multiply. A union
over (token, stream) pairs would collapse the first two parts and re-fetch every token
once per stream; separating them fetches each token once per window and pays the
per-stream cost only on the sends. Ownership of a derivation is one warp's or one
CTA's, never replicated per warp to dodge an edge the kernel already pays, and the edge
staging introduces is absorbed by the covers and split-phase edges of stage 6 — which
is why the earlier reading, that cooperative staging forces a barrier the warps wait
at, is superseded rather than weighed again. The staged batch `C` is a file constant
(§5) and is never tied to the warp count. *Proven form:* the fold and its ballot walk
in the reference's `carry_kernel.cuh` (find its sections by the file's own banners, not
by line numbers); the ballot and owner-row grain in `carry_rows.cuh`; the three-part
schedule's measured record at tag `record/f3-a247916`. *Floor:* one fetch per (token,
window) — a re-fetch factor of exactly one in the ledger; sends counted per (stream,
live token); DRAM and L2 counted as UNIQUE bytes; the expansion's arithmetic issued
between multiply issues rather than before them.

**3. The readout and the hybrid transit.** The readout's B-operand fragment IS the
state accumulator's TOP HALVES — a byte permute of the fp32 pair into a packed bf16
pair, the truncating read the declared arithmetic specifies (§R17) — so no separate
bf16 image of the state is ever built, and the hi plane the pages store (§3.2) is the
same bits by the same rule. The two sides are carved independently, so the phase change
re-lays the state into the other side's order, and the hi plane MOVES: the writers
publish it into the shared transit region and DROP it, holding only the lo plane
through PHASE 1; the region holds it, in the READ carve's order, for the whole phase,
and the readout reads it there in place as its B operand, stream by stream, with one
transposing matrix load per (k-step, channel m-tile) — no second bf16 stack exists and
no register copy of it either — and after PHASE 1 the writers pull those same bits back
in write layout and rejoin, `(hi << 16) | lo`, exactly, with no arithmetic. The
register footprint through PHASE 1 is 0.5× the state — the lo plane — plus the
readout's own working set; a copy form keeping both planes live in registers is
rejected because the register wall is the binding one. The publisher places each
leaf's row at the READER's slot, and the region is XOR-swizzled and conflict-free for
the transposing load. The per-side member choice is ONE
`uniform_switch<Set>` over the generated member set, placed OUTSIDE the loop — one
specialized body per member, the transit itself outside the switch, since an in-loop
switch merges every member's live ranges and interleaves their code (KERNEL_STANDARDS
§R9 enforcement; §5). *Proven form:* the carved body's readout, the
channel-split readout that proved the fragment identity, and the exchange row-map
section, all in the reference's `carry_kernel.cuh` (the uncarved channel-split body,
retired by §R18, is beside it in `carry_split_kernel.cuh`); the `(k, m)` lattice and
the leaf bit permutation in `carry/box.cuh`, whose arithmetic is most legible in that
tip's Python mirror of the same law; the swizzle and its bank-conflict ledger at tag
`record/g2d-4705709`; the row map's derivation is in this tree
(`docs/internals/common/geom.md`). *Floor:* 1.0× state registers through PHASE 1; zero
bank conflicts on both legs; one transposing load per contiguous reader segment; and,
because the switch changes an ownership partition, a SELECTIVITY line and a
BANK-BEHAVIOUR line beside its per-member instruction count.

**4. The streams.** `S` is derived from shared memory (§4), and a stream's segment
covers exactly one accumulator tile's leaves — the multiply its consumers issue. A warp
segment of `k` tiles consumes `k` tile-streams in sequence, `k` being a runtime trip
count that is one on the architectures in the tree; the loop is written with that trip
count from the first commit, never assuming that a tile and a segment coincide. The
read side has no stream axis: one stream per accumulator tile, always. *Proven form:*
the private-segment form and its stream-count axis at tag `record/f3-a247916`, whose
per-stream value segments fit the one CTA's shared memory — the shared-tile deviation
it recorded was a symptom of the two-CTA assumption (§7). *Floor:* bytes staged per
stream = one tile's expanded coefficient rows plus its value rows; the duplicated bytes
each added stream costs; shared memory filled to the residency target and not past it.

**5. The fan-in.** The window's output reduction is a native global reduction, and its
price is HOW MANY LEAVE THE SM: measured on sm_86, the reductions of a form that issued
one per (stream, token) line were a quarter of the whole kernel while no port on the
way to L2 was above 55%, and neither their MMAs nor their staging cost 4% — the
reduction count, not the issue count, is the figure of merit, with the 128-BYTE LINE
PER INSTRUCTION as the second. The readout therefore carves by POSITION: a warp owns a
slice of the batch's positions and every channel of their rows, sums across every
stream in its MMA accumulator, and the rows leave once per batch per CTA — one
instruction per position per 32 contiguous channels, through a per-warp staging tile
in TOKEN-ROW layout, drained progressively under the MMAs that follow. Issuing straight
from the accumulator fragment is fewer instructions and was measured SLOWER, because
the fragment touches eight lines an instruction; the shared-memory float add is a
compare-and-swap spin on this architecture and is not in the path. *Proven form:* the
position-carved readout with its staged, progressive drain, in the reference's
`carry_kernel.cuh`; the law the measurement stands on — staged versus straight is
decided by LATENCY, not by counts — is KERNEL_STANDARDS §R7. *Floor:* reduction
instructions per token per CTA per batch = `DV` / 32, each covering one full 128 B
line, once, whatever the stream multiplicity.

**6. The pipeline.** The body hides its own latency by construction: a batch's rows are
filled asynchronously while the batch before is consumed, a tile's gather and its B
fragments are issued ahead of the MMAs that use them, the readout's write-out is
drained under the MMAs that follow it — the next segment's, then the next batch's —
and the edges are named barriers crossed once per batch. `__syncthreads` survives only where a
write-after-read edge on a region genuinely needs every warp finished. Each transit leg
covers itself, and the ORDER is the whole mechanism: stores, then work that does not
depend on the pulled data, then the edge, then the pull — a cover issued BEFORE the
store chain overlaps nothing, which is a defect the sequence shows and the stall
counters do not. A named barrier does not help on that leg, because an arrival and a
wait executed by the SAME threads are two phases of one barrier rather than a split
edge, and every warp there is both publisher and puller; the asynchronous barrier is
what lets a reader pull a writer's rows as soon as that writer has arrived, at a finer
grain than the leg. The launch is one CTA per multiprocessor (§7), and exposed latency
is a COVERAGE finding, never grounds to split the CTA. On sm_80 the warps are
symmetric and the pipelining is within each warp; on sm_90 and sm_100 the same schedule
is expressed as producer/consumer specialization with register repartitioning (§9).
*Proven form:* the reference's serial per-batch structure — wait, rendezvous, produce,
load, multiply — and its measured issue rate are the control this stage is graded
against, not a template. *Floor:* exposed cycles per batch ≈ 0 under the multiplies, by
the stall mix; a stated target issue rate; every edge per window counted and justified.

**7. The parked lo plane, and the fill rule.** The last stage spends what the earlier
ones freed: during the readout the lo plane is idle state, so parking it in shared
memory — aliasing the transit region, dead at that moment — lets its registers house
the readout's output fragments, the allocation (a maximum over the body's moments)
falls, and the freed registers become LEAVES. Every moment then fills shared memory to
the residency target, by value per kilobyte. Both are stated as law in §7, with the
per-architecture declaration of where parking costs no residency. *Floor:* registers
per leaf before and after; the leaves per multiprocessor that buys; shared memory
occupancy at every moment.

**The reverse pass follows the same way, afterwards** — the same features, the same
grading against the reference, once the forward body stands. Its proven form is the
zero-snapshot two-scan reverse in the reference's `backward_kernel.cuh` and
`backward.cu`. **UNSPECIFIED (ruling pending):** whether the reverse carries the
forward's per-side carve and hybrid transit unchanged or derives its own, and what its
schedule is in the pipelined structure; today's reverse is built at depth two only, and
the obstruction is two static assertions on the digit reduction's row and warp maps
(`docs/open-work.md`).

**UNSPECIFIED (ruling pending): skip-induced wave drift.** Every owner reads the SAME
amplitude planes, and the design pays for that by having all owners resident on the
same window at the same time, so one token's runs are fetched once and hit in L2 for
every other owner. Owners that take the whole-window skip advance along the token axis
while busy owners lag, the multiprocessors de-phase, and that shared working set
dissolves; the fixed global window grid does not prevent it. The measured price of the
de-phasing collapsed once liveness came from ballots rather than amplitudes, so what
remains is the drift itself, unpriced. There is no phase-coherence law here — the
candidates on the record are a grid-wide per-window edge, a shared per-window liveness
scan, and a skip that advances the owner without advancing its stream position — and
the rebuild does not invent one (`docs/open-work.md`).

### 6.3 Decode — one wave of fat CTAs

At `T = 1` every tiling decision is degenerate, so decode is its own kernel: a routed
GEMV per step. Units are spread across CTAs first, then warps, so a small live set
spreads over the machine instead of concentrating on a few multiprocessors. The
admission — candidate atom enumeration, page-table probes, the block scan of demand,
the pool claim by canonical rank — is computed once per CTA, cooperatively, with
barriers CONFINED TO THE PROLOGUE; after the prologue there are none. Each warp then
derives only what its own units need, walks them (resolve the atom, cache the address,
read the row, update-then-read, accumulate in registers), and the combine is an
arrival count: each warp deposits its partial into its own shared row and bumps a
counter, and the last arriver sums the rows in fixed warp order, which is what keeps
the step byte-reproducible. The launch is ONE WAVE, aligned to the multiprocessor
count. **Read-only units pull the hi plane only** (§3.2), which is close to halving
decode's state traffic as `N` grows. `update-then-read` — the deposit lands before the
readout reads the row — IS the semantics, not an optimization
(`docs/internals/decode/decode.md`).

### 6.4 Intra — the within-window term

The causal term inside a window, as its own deterministic side kernel, reducing into
the same output and mass buffers the carry kernel allocated. Per window it walks the
lower-triangular tile grid of `[64 × 64]` output blocks; per tile pair the per-level
Gram matrices accumulate in registers over the level's digit axis, the levels are
multiplied together there, and the result becomes the A operand of the value multiply
IN PLACE — the accumulator-to-A fragment identity, with no shared-memory round trip.
The `[W × W]` matrix is never materialized. Where it touches the paged state it does
so through the same descriptor as every other kernel, and its ownership grain is its
own — the kernel in the tree reads no state at all, since every leaf it touches is
written and read inside the one window. *(**UNSPECIFIED (ruling pending):** what that
grain is once the intra term owns state, and whether the intra family's instantiation
axes fold into the `D × DV` shipped set of §8 — its window and level width are
template axes today.)*

---

## 7. Launch shape and the CTA policy

**The register law.** Eight warps per multiprocessor at 256 registers per lane,
`__launch_bounds__(256, 1)`, the register file filled exactly (8 × 32 × 256 = 65,536).
The census reports the STATE FRACTION of the register file beside the register count on
every stage, because the lever that raises leaves per multiprocessor is registers per
leaf — the working set — and not the block count.

**Registers per leaf is the lever, and the lo plane is its largest term.** The register
allocation is the maximum over the body's moments, and during the readout the lo plane
is idle state: parking it in shared memory for that moment (aliasing the transit region,
which is dead then) lets its registers house the readout's output fragments instead, so
the allocation falls and the freed registers become LEAVES — a bigger box per
multiprocessor, which is the fan-in and DRAM lever. It composes with the hybrid transit
(the writers drop the hi plane at the phase change; parking drops the rest). The
companion rule is that every moment FILLS shared memory to the residency target, spending
it by value per kilobyte: streams first in the fold, the parked lo plane first in the
readout. *(Ruled, not yet built: lands with the pipelined body; per-architecture
declaration where it does not cost residency.)*

**The CTA is the multiprocessor** (KERNEL_STANDARDS §R15). One CTA per multiprocessor
with all the warps the register file admits; streams partition inside it. More CTAs per
multiprocessor duplicate the cooperative fetch and expansion, run the transit once
each, and forbid cross-warp sharing of the readout; streams share all of it. **Latency
hiding is the kernel's job** — covers and split-phase edges — so exposed latency is a
coverage finding and never grounds to split the CTA.

**One CTA per multiprocessor is the GOAL, not a measured choice.** The block count
does not change the leaves resident per multiprocessor (one block of eight warps and two
of four hold the same registers); a two-block launch is only a DIAGNOSTIC — it says how
far the body is from hiding its own latency — and never becomes the law. Every
measurement the repository has was taken at two blocks of four warps, where the second
block was covering every edge; the one-block figure is the number the pipelined body is
graded against.

The per-arch `(ctas_per_sm, warps_per_cta)` launch table this section describes is data
in `csrc/rola/src/common/constants.cuh` / `rola.engine.constants`
(`docs/internals/common/constants.md#launch-table`), not a per-kernel branch.

---

## 8. Shipping

**A user of the Python package never registers an arm** (KERNEL_STANDARDS §R16). The
shipped set is `D × DV` per architecture — a dozen or so binaries — and every other
shape the family serves is served by runtime addressing over the descriptor, checked at
the seam. An unknown descriptor is admitted by refusal-checked derivation, not by
enumeration: it is either lawful (powers of two, `Π_l B_l ≥ BC`, `D` and `DV` matching
a shipped arm, the residency assertion of §5 passing) and it runs, or the refusal names
the field.

**The derivation is ratified as code plus golden outputs.** Closed-world codegen is the
rule for the binaries — pinned assembler, ahead-of-time only, a per-architecture
ratified manifest recording registers, spills, residency, compile time and the
per-arm launch table, and a device-side stamp asserted before any gate
(`docs/ratification.md`, `docs/architecture/closed-world-codegen.md`). A derivation that
now runs at launch instead of at compile time is inside that closed world only if its
OUTPUTS are certified too, which is what the golden outputs are for.

**One body per operation family** (KERNEL_STANDARDS §R18). The channel-split body is
deleted; dense is the `S = 1` point of the one body. A tied carve is a transit with the
identity row map. A single-valued axis is not plumbed as a parameter — it is deleted,
and the ratio readout is simply what the readout is.

---

## 9. Portability

Every structural mechanism states its form on each architecture; a mechanism that
exists on only one is a workaround, not a design (KERNEL_STANDARDS §R8). A primitive
may select different NATIVE instructions per architecture; it may never select a
wider-type emulation.

| mechanism | sm_80 / sm_86 | sm_90 | sm_100 |
|---|---|---|---|
| routing arithmetic, expansion | bf16x2 native fused multiply-add | same | same |
| operand supply to the tensor pipe | register fragments, transposing shared loads | shared-memory-sourced wgmma operands | tensor-memory-sourced tcgen05 operands |
| state in the register file | symmetric warps, in-warp pipelining (a producer warp would waste an eighth of the multiprocessor's state) | producer/consumer specialization with per-warp register repartitioning | same |
| operand staging from DRAM | asynchronous copy into shared memory | tensor-memory-accelerator box loads, descriptors built at runtime | same, plus multicast to a cluster |
| runtime addressing | uniform-datapath registers from the prologue | the same, pairing naturally with runtime descriptors | the same |
| the transit | shared region, XOR-swizzled, transposing loads | the same region; the exchange can ride distributed shared memory across a cluster | the same |
| leg edges | a CTA edge with the cover before it; asynchronous barriers (arrive, issue, wait at use) as the split-phase primitive from sm_80 up | the same object, its completion transaction-counted by the copy engine | the same |
| the fan-in | native global reduction from an aligned layout | the same, vector reduction where the width admits it | the same |
| the box grain | one CTA per multiprocessor | a two-CTA cluster sharing staging through distributed shared memory | the same |
| decode's walk and combine | per-warp gathers, arrival-count combine | bulk copies per warp, an asynchronous-barrier arrival | the same |
| the closed world | ratified manifest per architecture, device stamp | the same | the same |

---

## 10. Index: mechanism → page

| mechanism | page |
|---|---|
| the recurrence and the ratio readout, as the executable spec | `rola/ops/naive.py` |
| the update is affine in the input and never in the state | `docs/architecture/no-delta-rule.md` |
| accumulation precision | `docs/architecture/fp32-accumulators.md` |
| storing factors, expanding transiently | `docs/architecture/factored-representation.md` |
| dead routes are exact zeros | `docs/architecture/self-masking-theorem.md` |
| the state contract, descriptor, backings, activity bits | `docs/internals/state.md` |
| the page's plane geometry and its byte permutes | `docs/internals/common/state_page.md` |
| the arena, the page table, the slack pool | `docs/internals/paging/paging.md` |
| the addressing block: spans, sub-boxes, the transit row map | `docs/internals/common/geom.md`, `docs/internals/common/geom_api.md` |
| the window body and the liveness pass: what is built, in order, and where each mechanism was proven | §6.1, §6.2 |
| the deleted bodies themselves — the carved window body, the arms, the stats passes | the `k35-final` branch tip, the parity reference (never edited); their mirror pages went with them |
| the architecture capability table and the closed-world refusal | `docs/internals/common/arch_caps.md` |
| decode: the step, the walk, growth, capture | `docs/internals/decode/decode.md` |
| decode's lattice and factor tables | `docs/internals/decode/decode_lattice.md` |
| the within-window term | `docs/internals/intra/intra.md`, `docs/internals/intra/intra_kernel.md` |
| the host order of a call | `docs/internals/engine/chunk_dag.md`, `docs/internals/engine/decode_dag.md` |
| the public surface | `docs/api.md`, `docs/internals/rola_api.md` |
| what a manifest is and what it refuses | `docs/ratification.md`, `docs/architecture/closed-world-codegen.md` |
| the tiers, the tolerances, the conformance regimes | `docs/testing.md` |
| the one stopwatch and the measurement ledger | `docs/measurement.md` |
| what is knowingly unfinished | `docs/open-work.md` |
| what was removed, and why it was safe | `docs/internals/DELETIONS.md` |
