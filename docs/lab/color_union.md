# `color_union`

Mirrors `measure/lab/experiments/color_union.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What the facts pass's COLORING costs (KERNEL_STANDARDS §30's producer rule). A slot's residue mod 8 is its pool row's
bank group, so a box-window's union slots are colored so that each digit's tokens spread over the residues. The same
problem holds on both sides: the readout gathers the read union's rows, the fold the write union's, 8 tokens a matrix.
One warp colors one box-window, from its bit tile in shared memory (the facts pass's `tile_at` swizzle) to every union
token's slot. Two forms are measured: SERIAL (`color_union.inputs.py`'s `color`, every output checked exactly) and
BATCHED (its `batched`; the claims race on shared atomics, so outputs are checked by invariants). The facts pass no
longer colors (2026-10-13: the composed carry on sm_86 gained nothing from it,
[`facts.md`](../internals/facts/facts.md#ranks)); this page stays the record of what a coloring costs.

THE CHOICE, made on the CPU against the bank model (`read_image.inputs.emit`'s readout wavefronts, tokens on N), read
and write unions alike. In the table, "gathers" is wavefronts per `ldmatrix` matrix, so 1.0 means no conflicts:

| coloring | alt-k4 gathers | alt-k16 gathers | dependent steps (alt-k4 / alt-k16) |
|---|---|---|---|
| none (the union's rank order) | 1.94 | 1.82 | 0 |
| digit by digit, each digit water-filled | 1.014-1.042 | 1.158 | 32 |
| per-digit rounds, each digit coloring its next token (parallel) | 1.067-1.177 | -- | ~14 |
| SERIAL: multi-digit tokens one at a time, then each digit's singles | 1.000 | 1.011 | 40 / 317 |
| BATCHED: multi-digit tokens 32 at a time under capacities | 1.000 | 1.006 | 2.4 / 11 |

`csrcolor` (cuSPARSE, deprecated) solves a different problem: a proper coloring with as many colors as it needs, one
large graph a call. This one needs exactly 8 colors, a capacity per list and ~1,000 small problems a head that live in
shared memory.

THE BATCHED FORM:
1. A digit of n tokens allows each color ceil(n / 8) times, its CAPACITY.
2. Tokens in two or more digits are queued, those in three or more first. A batch of 32 (a lane each) sums its digits'
   counts, ORs their full colors, and takes the least-used allowed color (any color when none is: BLOCKED). It claims
   the color on every digit by a shared atomic. A claim that finds a digit full is withdrawn and requeued; a batch that
   keeps nothing runs its first token alone next.
3. Each digit's single tokens fill its room first-fit from an even or an odd start (any fill within capacity is
   conflict-free). Every digit starts even; digits whose odd start moves the parity count the needed way flip, in
   order, until half the imbalance (one prefix count).
4. THE PARITY LAYOUT: a slot's position fixes only its color's parity; the color's other two bits are the row's
   16-byte swizzle, folded into the entry's row offset. Positions are dense within each parity, so the pool runs over
   its size only by the parities' difference (~1 slot on average).
5. ROUND PLACEMENT: a digit's step i takes each color's i-th token, so a step never repeats a color; the steps are the
   largest color count, the same as an even fill's. Under it both forms measure exactly 1.000 gathers and adds; a
   blocked token costs a step, not a conflict.

The readout's panel must follow the parity layout (derived, not yet in the bank model): 64-byte rows by position with
a 32-byte swizzle bit, in place of 80-byte rows by slot.

Measured (sm_86, 12 warps an SM, a warp a box-window; budget lines restated from the counts):

| | alt-k4 | alt-k16 | thinned 1/16 |
|---|---|---|---|
| multi-digit tokens a box-window | 39.6 | 317 | 0.13 |
| serial: instructions / SM-cycles a box-window | 6,545 / 3,197 | 23,007 / 11,324 | 2,596 / 1,266 |
| batched: instructions / SM-cycles a box-window | 3,829 / 2,354 | 7,835 / 5,648 | 1,630 / 909 |
| batched: tokens over a capacity a box-window | 0 | 0.21 | 0 |

THE BUDGET LINES MISSED. The serial form was stated at ~45 instructions a multi-digit token and ~1.15K fixed, and
measured ~64 and ~2.6K:
- each `redux.sync` lands in a uniform register and costs a move back;
- the residue's count update is per word;
- divergence checks before the reductions;
- fixed: the plans, the 32-digit tie pass, the slot pass.

The batched form was stated at ~1.8K, ~3.7K and ~1.1K, and measured 3.8K, 7.8K and 1.6K:
- integer/address is 2-2.5x the line: swizzled tile addresses, u16 count unpacks and the slot pass's position
  arithmetic;
- control is 3-5x at alt-k16: three divergent loops over a token's digits a batch, each running the batch's largest
  digit count.

Two first drafts were dropped on their counts:
- the first serial form was 1.25-1.8x the final one: a ballot walk, counts bumped by shared atomics, branches around
  single-lane stores;
- the first transposes and singles cost ~2x their final form: ptxas computed both sides of a lane-dependent select;
  water-fill levels.

`redux.sync` measured 48 cycles of latency and 2 SM-cycles a warp-instruction of throughput, as `shfl`.

What it buys: on A100 the colored readout's shared time is 2.61 cycles a token against 3.36 uncolored (`read_image`),
for each of the 16 CTAs reading a box-window's union. At 1,024 box-windows a head the batched form costs ~17 us at
alt-k4 and ~42 us at alt-k16 on the 3080 Ti standalone. Inside the facts pass, its union, list lengths and rank walks
are work that pass already does; the batches and the transposes are the new part.

Record: `measure/lab/results/color_union/sm_86.json`.
