# `carry/carry.cu` and `carry/carry.cuh` — the carry's constants, operand block and host seam

Mirrors `csrc/rola/src/carry/carry.cuh` (the constants and the operand block every component reads) and
`csrc/rola/src/carry/carry.cu` (the kernel boundary's refusals, the block's derivation from the carve, the dispatch
over the built arms). The body has its own page, `ampere/carry_kernel.md`; the components `ampere/parts.md` and `ampere/fold.md`; an arm's
plan `ampere/plan.md`.

The carry is the inter-window term: each CTA owns a BOX of leaves (the carve's, `docs/internals/engine/rules.md#carve`,
`docs/internals/engine/carve.md`) and walks a call's windows in order. Every window it reads the state before the
window's writes out to the window's readers (the readout), then folds the window's writers into it (the fold). The
state never leaves the registers inside a call; the readout reads an f16 image of it (`ampere/handoff.md#image`), each warp every
leaf of the box for its channels (`ampere/readout.md`).

<a id="constants"></a>
## The constants

- `kWarps = 8`, `kThreads = 256`: one CTA an SM, its register file holding the box's state.
- The window W is the build's (`Plan<Arm, B>::kWindow`, [`../common/window.md`](../common/window.md)): the facts
  pass, the scale pass and the intra read the same trait, so the inter/intra split is one recurrence.
- THE BOX'S CAPS, every arm's: `kMaxExtent = 32` (a level's extent at most: the facts pass's group of digits, a lane
  a list), `kStateValues` = 32,768 values = 256 lanes x 128 fp32 registers. An arm's own box is its plan's
  (`ampere/plan.md#box`): its fold's lists and rows and its channels, their product the state.
- `kListBound = 64`, `kListAt = 8`: a list's boundaries, its entries before every 64th union slot (the last its
  length), so the carry chunks a union at any multiple of 64 the facts pass never sees, out in prefix form over the
  list group's lists modulo 2^16 (`G + 1` rows, a row a list, then the group's, at most `kPreRows = 33`:
  `ampere/plan.md#list-groups`) beside the group's exact totals;
  `kListCap = kWindow` entries a list.
- `max_steps(g)`, `plan_words(g)`: a window's plan record for a box class at `g` query heads a write head, its plan
  word and a 16-bit step word a step (`ampere/schedule.md#plan`).
- `kSliceBytes = 64`: a level's slice in a pool row, the box's 32 digits at f16.
- `kMaxLevels = 4`: the depth the operand block addresses.

<a id="sides"></a>
## A side of the carve

`Side` is one side's plan as the kernel addresses it. A side with a sparse level lists it (`list`) and forms its rows
from its other levels (`operand`, in level order). GROUPED QUERIES: the CTA is a write head's (`H` in the block, the
values' and the write rows' heads); its read rows are its `g` query heads', `g` by the read rows' heads over `H`, so a
read union's slot holds a ROW, `j 512 + t` -- query head `j` of the write head (the call's `bh g + j`), token `t` --
and a grouped read side lists at a level (refused otherwise, by name). A side with no sparse level (`list` -1) names two dense levels in
`operand`: its row level, then its coefficient level, whose digit is each slot's coefficient for a list or a digit
(`docs/internals/engine/carve.md#sides`). Each operand level carries its plane and element bytes. A level's slice is the
box's digits of it from the box's first digit: `origin_of` decomposes the CTA's box index in level order, the channel
group innermost.

<a id="lists"></a>
## The lists

`SideLists` is the facts pass's output at a side's list level in its final form, per (batch-head, window), merged
over `heads` query heads' rows (the read side's a write head's `g`, the write side's one;
[`facts.md`](../facts/facts.md#lists-out)):

- `entries`: a list's 8-byte entries in slot order, `kListCap` a query head, no pads. A read entry is its slot's pool row
  offset in the union in 16-byte chunks with its swizzle folded in (the first word's low half), its token (its high
  half: its query head's row, `j 512 + t`, its place in the scale bytes) and the coefficient (f16, the second word's low
  half): a chunk from a multiple of 64 takes its pool back by its first slot's offset and its slot from the row
  (`ampere/readout.md#chunk`). A write entry is the slot, then `c_v` and `c_m`. A step's lanes past a chunk's entries are
  masked by the carry (a read lane's stores predicated off, a write lane's row 0 with c = 0).
- `pre` (int16, the boundaries in prefix form modulo 2^16, the arm's list group plus one rows of `8 heads` columns) and
  `totals` (int32,
  the group's exact): `chunk_runs` turns a chunk's slots into each list's run and its place, `side_counts` reads the
  totals.
- `union_tok` and `union_count` (int32): per list group each slot's row (its query head and window-relative token,
  `#sides`), and the slots' extent; a GAP slot holds `kGap` (bit 15 set), and a fill zero-fills its row.

The addresses both kernels compute live here, one copy: `kGap` and `pool_chunk` (a pool row's 16-byte chunk swizzled by
the slot's residue, or its high bits within each four for a row of an odd number of 64-byte quarters); `parts.cuh`'s
`chunk_at` is `pool_chunk` at a compiled row.

<a id="params"></a>
## The operand block

`CarryParams` reaches the kernel as a grid constant. The box is the carve's: `extents` a level, `channels`, and
`stride`, a level's step in the box index (the channel groups innermost). Every pointer is the bundle's own memory:
the rows and values at their strides (the row alignment law, `docs/internals/facts/facts.md`), the facts pass's
lists, the cross-CTA sum's scale bytes (`scale`, an exponent byte a query head's token, a window wide so a window's
copy is aligned: the scale pass's bound on |num(t)|, `scale/scale.md`, read only where `reduce` is f16 pairs). The
outputs are `den` (each token's v+1 channel, `ampere/handoff.md#den`), `num` (a query head's: f16 pairs at each token's scale, or
fp32 channels, by `reduce` -- `carry_api.md#carry-reduce`, `ampere/panel.md#drain`) and `check` (the final state's sum).
`plans` are the plan pass's records (`ampere/schedule.md#pass`), a window's for each box class.
`readout_f16` is the readout's accumulation (`carry_api.md#carry-readout-accumulate`); `forced` a plan form every
window takes where it fits, -1 for none (`ampere/schedule.md#planner`). A bound capture copies the block and its grid out before the
launch (`CarryCapture`, `carry_api.md#carry-capture-bind`).

<a id="refusals"></a>
## The kernel boundary refuses

`carry_forward` takes the strict form and refuses everything else, naming the field:

- depth 1 to 4; every width a power of two from 16, every extent a power of two dividing it; the box's leaves times
  its channels exactly a CTA's state; the channel extent a power of two from 32 to 128, dividing `d_v` (the built
  arms 32 and 64); the arm key's rows from 16 and channels from 32.
- a side lists at most one level and forms its operand from all the others; a list level spans the arm's list group
  (the box's lists or digits, `ampere/plan.md#list-groups`), and the side's lists are shaped at it; a side with no list level names exactly a row level and a coefficient level; an operand slice is a 4-byte
  copy at least and `kSliceBytes` at most.
- an fp16 readout on a dense read side, whose readout accumulates fp32 alone (`ampere/readout_dense.md#nest`).
- every tensor's dtype, contiguity and exact shape (the lists' five, `scale` `[BH, windows x W]`, `num` int32 `[BH,
  T, d_v / 2]` for f16 pairs or fp32 `[BH, T, d_v]`); the arm key's channels and factor counts agreeing with the
  carve, and its hand-off form (`same_split`) with the levels the sides list (`ampere/handoff.md#same-split`); a write head's
  `num` rows under 2^32 values (the drain's offsets); the readout's accumulation fp32 (0), fp16 (1) or the device's rule (-1); the cross-CTA sum f16
  pairs (0) or fp32 (1); a forced form -1 or a
  mode (0 to 2) with two chunk fields in 14 bits; a window's token stride under 2^31 / `kWindow` (a window's offsets
  are 32-bit).

The closed-world arch refusal (`check_arch_table`) is the first statement of every entry.

## Dispatch

The key is `rola.ops.carry.arm_key(carve)`; `arm_switch<CarryArmSet>` resolves it to the generated arm's launch or
refuses naming the field no built arm carries (`docs/internals/common/arm_switch.md`). The grid is
`(boxes, batch-heads)`. The phase ledger, when bound by `carry_ledger_bind`, must cover the grid.
`carry_build_stamp` reads four facts back off a kernel this binary compiled at the current device's build (its window,
the capability digest, `kStateValues`, `kListBound`), the device-side check a stale binary cannot pass.
