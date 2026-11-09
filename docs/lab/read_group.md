# `read_group`

Mirrors `measure/lab/experiments/read_group.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When a warp's read digits are taken M at a time -- a group's tiles the 8-token tiles of its digits' union, each union
token with each digit's coefficient (zero where the digit does not read it), the M digits summed into one exit (Blake,
2026-10-18: "when m=2, 2 digits accumulate into one accumulator") -- what does a group-tile cost, and what does the
readout cost a window against each digit's own tiles?

The form is [`read_stream`](read_stream.md)'s staged readout: a ring of 8 stages a warp, a stage a tile's 8 read rows
and its record, one record stream a warp across windows and passes, pads not copied, every group's first tile in one
block then each group's rest, the panel drained a pass. A group-tile loads its rows once, runs 4 MMAs a digit against
each digit's state, and leaves its outputs once: 8 conversions, 4 packs, 4 shared atomic adds. Two scalings:
- OPERAND: the lane's read-row column (token g) scaled by each digit's coefficient, 4 bf16x2 FMAs a digit, the MMAs
  summed in one accumulator; the record carries each position's M coefficients as bf16x2.
- OUTPUT: an accumulator a digit, each digit's coefficients folded into the exit's conversions (an FFMA a digit an
  output, the exact pair's 1.5 x 2^23 addend carried along the chain).

- UNIFIED (ruled 2026-10-18: one form for every M, output scaling): OUTPUT with the record the production readout
  can be fed -- each position's panel field and its token's unit (`field << 16 | unit`), each lane's two tokens' M
  coefficients contiguous as f16 pairs (one load at every M) -- and each (token, digit) factor the coefficient times
  the digit's scale (held, loaded each pass as a window's exponents are) and the token's `2^-unit`, multiplied on the
  FMA pipe so a zero coefficient stays zero. The inputs give tokens units 0-2 and digits exponents 0-1, each f16
  coefficient stored so its factor is the target exactly. M = 1 is its degenerate case.

M = 2 pairs a warp's digits w, w + 8 and w + 16, w + 24; M = 4 takes all four. The inputs
(`read_group.inputs.py`) are `read_stream`'s three cells, the same coloring, rows and panels; each union placed as a
digit's span is (`place`). Every form's drained panel is checked word for word against the same panel: only the sums'
grouping changes.

The cells' tiles, a box-window (the inputs' mirror):

| | M = 1 | M = 2 | M = 4 |
|---|---|---|---|
| alt-k4: tiles (MMAs) | 45.0 (180) | 38.7 (309) | 34.9 (558) |
| alt-k16 | 155.7 (623) | 138.0 (1,104) | 123.8 (1,980) |
| cohort-k16 | 177.6 (710) | 144.6 (1,157) | 124.2 (1,987) |

A group's union saves 14-22% of the tiles at alt-k4 and 19-30% at cohort-k16 (fewer pads, shared tokens) and runs
1.7-3.1x the MMAs (every digit over every union token).

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window, every form exact:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| M = 1 (each digit its tiles) | 4,804 | 12,189 | 19,069 |
| M = 2, output scaling | 4,913 | 13,447 | 21,277 |
| M = 2, operand scaling | 5,021 | 13,752 | 21,669 |
| M = 4, output scaling | 7,553 | 22,738 | 33,354 |
| M = 4, operand scaling | 7,118 | 21,116 | 31,566 |
| UNIFIED, M = 1 | 4,868 | 12,544 | 19,676 |
| UNIFIED, M = 2 | 5,400 | 14,991 | 22,555 |
| UNIFIED, M = 4 | 7,442 | 22,240 | 33,087 |

(Three variants' samples hold one outlier near 2x, a foreign consumer's moment on the card; their medians match the
first record's. M = 1 here runs 6% over `read_stream`'s staged form, 4,549 at alt-k4: it loads the warp's four states before the
first tiles, where `read_stream` interleaves each with its digit's first tile.)

Each form's pipes a CTA-window ([`tools/pipes.py`](../internals/tools/pipes.md), the binding one priced on each card):

| alt-k4 / cohort-k16 | sm_86: tensor | shared | issue | sm_80: tensor | shared (binds) |
|---|---|---|---|---|---|
| M = 1 | 1,445 / 5,645 | 2,030 / 5,751 | 929 / 3,051 | 360 / 1,406 | 2,029 / 5,748 |
| M = 2, operand | 2,485 / 9,117 | 1,975 / 5,347 | 948 / 2,970 | 619 / 2,270 | 1,973 / 5,344 |
| M = 4, operand | 4,470 / 15,415 | 1,947 / 5,067 | 1,046 / 3,151 | 1,113 / 3,838 | 1,946 / 5,064 |

What the shared pipe holds a CTA-window at alt-k4, M = 1 (priced cycles): loads 1,191 -- the four digits' states
reloaded each pass (512 wavefronts), the drain's panel loads (2 wavefronts each: below), the rows' `ldmatrix`, the
records -- the drain's clearing stores 306, the exit's atomics 360, the copies 81. Bank conflicts: 271 wavefronts.

THE UNIFIED FORM on each card. On sm_80 the shared pipe binds every form, and the unified form's record (f16
coefficients, a 32-byte field part) is the smallest, so it prices lowest at every M: alt-k4 2,017 / 1,933 / 1,937
cycles a CTA-window (M = 1 / 2 / 4) against the best other form's 2,025 / 1,973 / 1,946, cohort-k16 5,681 / 5,171 /
5,013 against 5,740 / 5,345 / 5,064. On sm_86 it costs 1-3% over the idealized M = 1 (whose record carries each fp32
factor ready), 6-11% over M = 2's output scaling and 4-5% over M = 4's operand scaling: ~14 instructions a tile more at
M = 1 and ~32 at M = 4 -- the factor a (token, digit): the f16 pair decoded, the digit's scale times the token's
`2^-unit`, and that times the coefficient (alt-k4 a tile 96 against 83 at M = 1, 152 against 120 at M = 4).

Instructions a CTA-window: alt-k4 3,716 (M = 1), 3,627 / 3,792 (M = 2 output / operand), 4,347 / 4,182 (M = 4); a
group-tile 83, 94 / 98, 125 / 120 with the window's fixed work spread over it. Stalls: M = 1 the shared queue 19%,
fixed latency 19%, the pass barrier 12% (cohort-k16: the barrier 40%); M = 4 operand the math pipe 38%.

What it settles:

- On sm_86 grouping costs: the tensor pipe binds every grouped form, its MMAs 1.7-3.1x M = 1's. M = 2 runs 2-14%
  slower, M = 4 48-75%. The planner's merge width for a read side on sm_86 is 1 at these densities.
- On sm_80 the shared pipe binds every form and grouping saves 3-12% of it (M = 4 operand: 4% at alt-k4, 12% at
  cohort-k16), its MMAs still under the shared floor. The saving is the exits' atomics and the rows a tile: the
  shared pipe at sparse is mostly fixed a window -- the states' reload once a pass (512 wavefronts a CTA-window, ruled
  fixed 2026-10-14) and the drain -- which grouping does not touch.
- Operand scaling is the cheaper exit at M = 4 (6-8% under output scaling on sm_86, 122 registers against 152); at
  M = 2 output scaling is 2% cheaper. Both exact here (coefficients exact in bf16); in the carry operand scaling puts
  each digit's exponent into bf16 operands (the range fp32's; 8 bits of mantissa a scaled row), output scaling stays
  fp32.
- THE UNIFIED FORM holds at every M: exact, the cheapest on sm_80, 1-3% over the idealized form at sm_86's M = 1. Its
  cost on sm_86 is the per-(token, digit) factor; folding the token's unit into the record's coefficient where the
  record's producer knows it would take the token multiplies and the unit's extraction off the tile.
- THE DRAIN'S OWN SHARED COST exceeds the exit's atomics at alt-k4: each lane's panel word load and clearing store
  pairs slots (s, s + 1), whose 16 words sit 20 words apart at the panel's pitch -- 4 banks shared, 2 wavefronts an
  access (the bank model; slots (s, s + 4) sit 80 words = 16 banks apart, 1 wavefront), and each clearing store is
  2 cycles. Those are the sparse readout's next shared-pipe lever on sm_80, before grouping.

Record: `measure/lab/results/read_group/sm_86.json`.
