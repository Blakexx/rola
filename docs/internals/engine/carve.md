# `rola/engine/carve.py` — the carve planner

Mirror of the planner the `carve` rule calls ([`rules.md`](rules.md#carve) says what the
carve is and why one descriptor). This page explains how producer v1 builds a carve and how
the check refuses one; the code is the authoritative statement of each step.

## <a id="budget"></a>The budget

The box's state is the CTA's fp32 accumulators: `WARPS` (8) warps x 32 lanes x 128
registers a lane, `BOX_VALUES` = 32,768. Its leaves are that over its channel extent: 1,024
at `CHANNELS` (32), 512 at 64, 256 at `MAX_CHANNELS` (128). `CAP` (32) bounds any one level's
extent, and `MIN_DENSE` (16) is a warp's least operand rows, one MMA tile.

## <a id="producer-v1"></a>Producer v1

The channel extent comes first: the largest power of two dividing `d_v`, at most
`MAX_CHANNELS` (`_channels`), so a token's output is a whole box's channels and the
readout's warps each own theirs -- halved until the box is one the kernels take
(`_admitted`): every listed level spans a list group the facts pass ranks (`LIST_GROUPS`, 16
or 32 digits, or the level's whole width), each side its own -- the lists come ranked in
groups, and a box over part of a group would read and drain tokens it never touches; a box
with a listed side holds at most `LISTED_CHANNELS` (64: a 128-channel box needs the fold's k8
and m8 forms); and each side's warps own an MMA tile of dense rows (`MIN_DENSE`). So at `d_v`
64 a dense-dense declaration takes 64 channels, and so does the alternating one: 32 write
lists x 16 read digits x 64 channels, its read side ranked in 16-digit groups, its write side
in 32; at `d_v` 32 it is 32 x 32 x 32. The leaves are what the state holds over it,
allocated as powers of two, one bit at a time, under the cap and each level's width:

0. Each side with no list level, its row level (its operand's first: `_row_levels`) to an
   MMA tile, `MIN_DENSE` -- spread evenly, three dense levels of 16 at 64 channels would be
   8 x 8 x 8, a row level under a tile; first, they are 16 x 8 x 4.
1. `DD` levels, round robin, as long as one has room.
2. `DS` then `SD`, by turns, each to its tag's widest level with room: the readout's dense
   level (its K) before its list level, since the readout is the costlier phase (Blake,
   2026-11-06).
3. `SS` levels, the widest first.

When the bits run out before the box is full, the cap lifts and the box takes the widest
levels whole, smaller than the budget if the levels are narrow (a depth-one declaration is
its own whole box).

## <a id="sides"></a>Each side's plan

Each side's plan follows from the box. A side with a sparse level lists it and forms its
operand from its dense levels, both in level order (`_operand`). A side with no sparse level
and two dense levels names one as its rows and the other as its coefficients, in that order:
its coefficients the other side's list level where the other side lists exactly one level and
it is one of this side's, else its first dense level.

- THE FOLD is a state carve (`_fold`): a warp owns `box leaves / 8` leaves -- every channel of
  them, in registers -- its operand rows whole while they fit (`tile_rows`), the rest in lists
  (`tile_lists`).
- THE READOUT is a value carve (`_readout`): every warp holds every leaf of the box for its
  channels, `tile_rows` the operand's rows (the MMA's K: a listed side's dense levels, a dense
  side's row level) by `tile_lists` digits (a listed side's list values, a dense side's
  coefficient values). Its warps are `channel_groups` of 8 channels (at most 8) by
  `token_groups` (the warps left over: 2 at 32 channels, 1 at 64 and 128).

The two sides never own the same leaves a warp, so the state always reaches the readout
through the image.

## <a id="check"></a>The check

`check_carve` refuses, naming the reason: a declaration with a side of only sparse levels,
a width that is not a shipped power of two, an extent that does not divide its width, a
channel extent outside 32 to 128 or not dividing `d_v`, a box over the state budget, a side's
plan that is not the box's, and a warp with fewer operand rows than an MMA tile. The
planner's own carves pass through it.

## <a id="planned-once"></a>Planned once a declaration

`plan_carve` is a pure function of the declaration (the widths, each side's per-level flags and `d_v`) and its
`Carve` is frozen, so it plans once a declaration (`functools.cache` on `_planned`) and every later call of the
shape returns the same object. Planning in Python -- the build, the check's rebuild, each side's warp ownership over
the box's 1,024 leaves -- ran twice a prefill call (the envelope's clauses and the facts node) before the first
kernel launched, the GPU idle meanwhile (`tools/op_timeline.py` found it). Planned once: the A100's sparse prefill at
`nl64k-alt-k4-h4` 10.50 -> 8.03 ms and layer g16 11.27 -> 8.72 ms, interleaved in one bench run against the tree
before (2026-11-06). A refused declaration raises every time (an exception is not cached).

## <a id="open"></a>Open

A single `DD` level's share is the cap's, not density's: at very low density a larger
share wins (by about a third on a tied layout), which a declared budget and the structural
cost model may later take. A box of fewer than 128 leaves (a narrow depth-one declaration)
is refused: it leaves a warp under an MMA tile of rows, and needs a plan that splits the
channels across warps.
