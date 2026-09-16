# `csrc/rola/src/facts/liveness.cuh` / `liveness.cu` — the ONE liveness pass

Mirror doc for the pass itself. What it WRITES — the class-1 word layout, the two
class-2 folds over it, and why each is the grain it is — is
[`liveness_contract.md`](liveness_contract.md), which this kernel does not restate and
must not contradict; its host seam is
[`liveness_api.md`](liveness_api.md). `rola.engine.facts.liveness.liveness_words` is
the pure-torch MODEL this kernel is graded against and
`tests/unit/test_liveness_pass.py` is where the two are held equal, bit for bit.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B_l` =
level `l`'s PADDED width; `b_l` = its LOGICAL width, which never reaches this file;
`L` = tokens in the call; `BH` = batch × heads; `rows = Σ_l B_l`;
`words = ceil(L/32)`.

## What makes it ONE pass

There is one pass because there is nothing geometry-dependent in it to specialize. No
span, no carve, no `BC` and no compile-time `B` appears in the kernel or in its launch:
`D` and every `B_l` arrive at runtime in the operand block, the digit order is
canonical, and the only compile-time facts are the contract's own (`kMaxLevels`,
`kTokBits`, `kSides`). Every consumer's coarser grain — the carry's per-run schedule
rows, the per-box call-level READ/WRITTEN bits, the per-page activity byte, decode's
digit masks — is a FOLD over these words rather than a second producer, which is the
whole of Blake's ruling of 2026-08-29 (`development/queue/G_FOUNDATION.md`, "ONE
GEOMETRY-INDEPENDENT LIVENESS PASS; EVERY GRAIN IS A FOLD OVER IT").

<a id="the-grain"></a>

## The grain: one warp per token word, one lane per token

`kLivenessWarps` warps to a CTA; a warp owns one token word `w`, and lane `t` owns
token `w·32 + t`. A row's word is then one `__ballot_sync` and one store by lane 0 —
the shape the shipped union table already had (`k35-final`'s
`facts/facts_kernel.cuh`, `chunk_facts_tables`), kept because it is the form that
makes the vote a single instruction and the layout's token-minor axis its natural
output.

A lane past `L` reads **token zero's** row rather than branching, and its bit is
masked out of the ballot by `real`, so the tail word's high bits are zero by
construction and a fold may OR whole words without masking. A warp whose `w` is past
`words` returns as a whole lockstep unit before any ballot.

<a id="the-read"></a>

### The read: sixteen digits per lane per transaction

Lane `t` reads its OWN token's row, so the 32 lanes of one ballot are `rows·2` bytes
apart — the axis the layout makes minor is tokens, and the axis a lane walks is digits.
A scalar 2-byte load at that stride touches 32 distinct sectors for 64 useful bytes and
leans on L1 to recover the other fifteen-sixteenths of each sector across the `rows`
iterations; measured, L1 does not recover it once `rows` grows, and DRAM traffic runs
2.5–3.8× the unique bytes.

An earlier `kVecDigits = 8` form (one 16-byte `uint4` per lane) cut that to 1.77–1.92×
of the unique bytes: a lane's 16-byte load fills only half of the 32-byte sector its
address falls in, and closing the other half depends on it surviving in cache until the
adjacent lane's group is read, which at this occupancy it mostly does not.

So a lane loads `kVecDigits = 16` digits — two consecutive `uint4`s, one whole 32-byte
sector — and votes all sixteen from registers. That is one full sector per lane per
sixteen digits instead of two half-used ones, and it is UNIFORM rather than a fast path
with a narrow fallback: `B_l` is a power of two at or above 16 by the descriptor's law,
so every level's width and every row base is a whole number of sixteen-digit groups,
and the pass's seam REFUSES a width that is not
([`liveness_api.md`](liveness_api.md#the-plan)). There is no remainder loop because
there is no lawful remainder.

The sixteen amplitudes are extracted from the eight `uint4` members (two loads of four
each) by shift, never through a pointer into the loaded value, so the group stays in
registers.

*The store pattern.* Consecutive rows are `words` apart, so lane 0's stores are strided
4-byte writes. The written volume is `rows · L / 8` bytes per side, an eighth of the
amplitude bytes read, so the pass stays read-dominated. Coalescing them would mean
staging votes in shared memory to gather several token words per store; that is a
mechanism this pass does not carry, and it is the one candidate its ledger points at.

<a id="passplan"></a>

## `PassPlan` — the whole operand block, by value

`D`, the per-level widths, `rows`, `L`, `words`, `side_words`, and per side the dense
level bitmask and the digit mask's words. It is a kernel parameter, not a device
allocation: the masks are at most `kMaxMaskWords` words per side (four levels of the
widest branch, packed), which is a few hundred bytes of parameter space, so the pass
takes no allocation, no extra memcpy and no second launch to learn its own shape. The
`SideStatics` the contract defines is constructed on the device over those parameter
words, so the kernel addresses the mask THROUGH the contract's type and never through
a private copy of the rule.

**No logical width crosses into it.** `b_l` reaches the digit mask at the host seam
(`rola.engine.facts.liveness.side_statics`) and nowhere else — the two contracts keep
width fields out of a kernel entry signature, and a bit operand is the only form that
satisfies both that and the invariant that a dense level's rows ARE the width mask.

<a id="the-kernel"></a>

## The kernel body

Two sides, `D` levels, `B_l` digits — three runtime loops, and one branch per level
that is warp-uniform and loop-invariant:

* a **sparse** level's digit is voted from the amplitude at column `row`, because the
  contract makes the row index and the packed amplitude column the same number; there
  is no per-level column arithmetic at all;
* a **dense** level is NOT READ. Its rows are the width mask, and since the mask bit is
  uniform across the warp the whole level costs one ballot for `real` plus one store
  per digit — no load, no per-digit vote. That the two branches are separate loops
  rather than a ternary is what guarantees the zero traffic.

The nonzero test is the contract's `amplitude_live` on the bf16 MAGNITUDE, so `-0` is
dead like `+0`; the plane is addressed as `uint16_t` because the test is on bits and
never on a bf16 value.

<a id="the-launch"></a>

## The launch

`grid = (ceil(words / kLivenessWarps), BH)`, one block of `kLivenessThreads`. Both
sides are done by the same warp in the one launch: they share the token word, the
`real` mask and the ballot's `alive`, and their two planes are read from the same
token row, so splitting them would double the launch and the tail arithmetic to save
nothing.

## Where the mechanism was proven

<!-- ABSENT: facts_launch -->
the ballot grain, the token-minor layout and the past-the-end lane's dead vote:
`k35-final`'s csrc/rola/src/facts/facts_kernel.cuh (`chunk_facts_tables`) and
`facts_launch.cuh`, which are never edited again. What is NEW here is that the vote is
per DIGIT rather than at a compile-time carve, which is what removes `B`, the spans and
`BC` from the family's template axes and leaves one pass where there were four kernels.
