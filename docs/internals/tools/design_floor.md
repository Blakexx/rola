# `design_floor.py`

Mirrors [`tools/design_floor.py`](../../../tools/design_floor.py).

THE DESIGN'S FLOORS: the value-carve carry (ruled 2026-11-06) counted a CTA-window over a cell's real lists at any
box, priced on each card. It is the design's estimate before there is a kernel, so every number it prints is
ESTIMATED: the counts are exact over the facts pass's lists, and the instruction budgets and card constants are stated
here. The readout's latency is not modelled; the lab measures it.

<a id="box"></a>
## The box

`--box LxDxC` (repeatable; default `32x16x64` and `32x32x32`): `L` lists (the fold's list level, the readout's dense
level and K), `D` digits (the readout's list level, the fold's dense level), `C` channels, `L D C` = 32,768 (the
state's 8 warps x 32 lanes x 128 registers). The readout's warps are `C / 8` channel groups (at most 8) times
`8 / that` token groups. A token's group is its output line's (`128 / 2C` tokens a line) modulo the groups.

<a id="lists"></a>
## The lists

`carry_operands` builds the cell's bundle and runs the facts pass at the planned carve (512 tokens a window). Each
side's lists become a membership `M[bh, window, digit, token]` (a read entry's token is its window place; a write
entry's its union slot's token), merged `card window / 512` windows at a time (their tokens disjoint) and split into
the box's digits at the side's list level. A dense side holds every token in every digit. Since the read side's lists
depend only on its list level's box and the write side's only on its own, a CTA-window is one (read box, write box)
pair; every pair of every window is priced.

<a id="phases"></a>
## The phases

Each phase counts warp instructions, shared-pipe cycles (a 128-byte load one, a store two -- lab `smem_rates`),
MMA cycles (an m16n8k16 at fp32 accumulate: 8 cycles a scheduler on sm_80, 32.1 on sm_86; m16n8k8 half) and the
global path (128-byte lines from L2, atomic requests).

- READOUT, a listed side: per digit and token group, `n` entries in `ceil(n / 16)` tiles (tokens on M); each of the
  group's channel-group warps loads the tiles' rows (`ceil(n / 8) 8` rows of `2L` bytes), its slice of the digit
  (`L x 8` f16) once, the entries' words, and adds its tiles into its panel (load, store; `--panel f16|fp32`).
  `READ_TILE` = 23 instructions a tile a warp, `READ_DIGIT` = 8 a live digit a warp.
- READOUT, a dense side: the value-carve readout's nest -- a token batch of `--batch`
  tokens a warp, a transit group of `--transit` digits; rows once a (tile, transit group), slices once a (batch,
  digit), the coefficients on the rows (HMUL2), no panel; the exit staged to full lines.
- DEN, in the num's form: one 8-wide tile a tile in the same nest, its columns up to
  8 digits (1 at a listed side), its word a token a warp in the panel; `DEN_TILE` = 9.
- DRAIN, a listed side: a token's channels and its warps' den words read and cleared by stores, a full-line RED a
  128 bytes of num, a den request a 16 tokens.
- FOLD (the state carve, unchanged): per list `n` entries in 16-token steps and an 8-token tail, `D / 16 (C / 8 + 1)`
  MMAs a step (the den a fifth n-tile), its coefficient rows (`D x 16`) and values (`16 x C`) loaded a step (a dense
  side's values once a step for the warp's lists), the entries' words; `FOLD_STEP` = 5 beyond the MMAs and loads,
  `FOLD_LIST` = 10.
- HAND-OFF: the image's stores (the state's and the den's, f16) and the maxima (`MAXIMA` = 150 a warp).
- FILLS: the read union's rows (`2L` bytes, a dense side's coefficient slice beside them), the write union's
  (`2D + 2C`), every entry's 8 bytes; a copy's write half a shared cycle a 128 bytes.

<a id="floors"></a>
## The floors

Per CTA-window: OVERLAPPED the largest of the window's totals (issue over 4 schedulers, shared, tensor over 4, the
global path); IN TURN each phase's largest summed, with the global path beside it. The printed floor is the mean over
the CTA-windows; CTA SPREAD is the slowest box's whole call over the mean box's (the wave's tail). On sm_80 each cell
prints twice: the global path at sm_86's L2 line and at half (`l2_bracket`, the A100's L2 serving more an SM). TODAY
is the A100 phase ledger at W = 1,024, measured 2026-11-06 (`TODAY_SM80`); the milliseconds use its
alt-k4 scaling (24.2K cycles a warp-window <-> 5.09 ms) and print for that cell only.
