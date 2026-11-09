# `handoff`

Mirrors `measure/lab/experiments/handoff.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the carry's hand-off cost a window on sm_86 -- each digit's exponent from the fold's state (its largest value
over the box's 32 lists, each list at its own unit) and the state stored into the image, f16 at each digit's power of
two -- in the carry's form and in a lean form, the image whole (sm_80's form) or in two halves with a barrier between
(sm_86's), and how near its floor does it run?

8 warps hold a box's state in their fold accumulators (lists `w + 8 j`, 32 digits, 32 channels; the inputs
`handoff.inputs.py`: units a list, one list never live, values that span octaves by list and digit, digits zero in some
lists and one zero in all), made opaque each window so nothing is hoisted. A window is the maxima, a CTA barrier, the
exponents, the stores and a barrier; a sample is a CTA's span over 64 windows, 80 CTAs. Every form's image and
exponents are checked byte for byte against the mirror's, which follows `csrc/rola/src/carry/ampere/parts.cuh`'s rules.

- THE CARRY'S FORM before the 16-byte image (`handoff.cu`'s `carry_maxima`, `carry_digit_exp`, `carry_stores`, its
  mirror): a list's largest value a digit row as
  `ceil_exp` plus its unit, two shuffles a row, posted int16; after the barrier lane `d` the largest of the 8 warps'
  posts; a scale a (digit row, list) from its clamped exponent, each channel pair scaled, packed and stored 4 bytes at
  an address formed for it.
- LEAN: a value compared across lists as an integer KEY -- its fp32 bits plus (its unit less the box's top unit) x 2^23,
  so the exponent field moves and the mantissa stays; a zero INT_MIN -- one integer max a list, the digit row's 4 lanes
  joined by two shuffles, one shared atomic max a lane at its digit's slot (`4 g + 2 M + H`); after the barrier each
  lane's 4 keys in one 16-byte load and its exponents `(key >> 23) - 127 + (mantissa != 0)` plus the top unit, exact
  while the live units span under 255; the stores at 4 held bases (a channel tile's) and compile-time offsets.
- LEAN INTO THE 16-BYTE IMAGE (`unit16_of`): a unit holds a writer lane's 8 channels of a (digit, list) -- chunk `t`
  holds channels `8 n + 2 t + b` at position `2 n + b` -- so a lane stores a (digit row, list) as one 16-byte store at
  its one held base plus a compile-time offset; the digit's swizzle moves its low bit to bit 2 (`(d & 1) << 2 |
  (d >> 1) & 3`), so a quarter-warp's stores (two digits, four chunks) take eight bank groups. The maxima start from
  a row's first pair (7 FMNMX, not 8), the exponents are `((key + 0x7fffff) >> 23)` plus the top unit less 127 (the
  exponent field rounded up past a nonzero mantissa), and each scale's difference and float bits are integer
  multiply-adds (the FMA pipe; the bits' constants loaded, so ptxas cannot fold them into an ALU shift), its two
  clamps the only ALU work. The same image bytes and exponents as the rule; the readout's `ldmatrix.trans` reads the
  same units (eight lists of a chunk) and its outputs pair channels (c, c + 2) where today's pair (c, c + 8).

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a window (a CTA's):

| form | the image whole | two halves |
|---|---|---|
| the carry's | 3,025 | 3,180 |
| lean | 2,242 | 2,252 |
| lean, 16-byte units | 1,648 | 1,705 |
| bound: lean, the maxima alone | 569 | |
| bound: lean, the stores alone | 1,394 | |

(The carry's and lean forms read 3,059 / 3,184 and 2,248 / 2,266 at the first record, 434a3c1f.)

Instructions a CTA-window (ncu): the carry's 5,894, lean 4,688 -- of which on the ALU pipe 3,250 and 2,619 (the
maxima's float max, the packs, the integer work), the FMA pipe 1,650 and 1,371, the load/store pipe 833 and 657.
Issue active 46% and 49%; the stalls a warp's: the shared queue full 15% and 17% (the stores alone 31%), a dependent
result 16% and 12%, the math pipe busy 10% and 11%, the barriers 6-8%.

Per CTA-window, each pipe's cycles ([`tools/pipes.py`](../internals/tools/pipes.md); the bound names the largest):

| form | ALU pipe | issue | shared pipe (an SM) | FMA pipe | largest stalls |
|---|---|---|---|---|---|
| the carry's | 1,625 | 1,473 | 1,401 | 485 | dependent result 16%, shared queue 15% |
| lean | 1,310 | 1,172 | 1,273 | 343 | shared queue 17%, not selected 15% |
| lean, 16-byte units | 1,166 | 1,028 | 827 | 396 | not selected 20%, math pipe 14%, barrier 8% |

The 16-byte form's window loop is 514 instructions a warp against lean's 585: 16 stores for 64, 112 FMNMX for 128,
the scales' 16 shifts moved to the FMA pipe. On the ALU pipe stay 292 a warp: 112 FMNMX and 64 packs (the format's:
fp32 state to f16 at an exact exponent), 60 integer maxima (the scales' clamps, the keys' maxima and floors), 35
compares and selects (a zero row's key below every real one: the exponent exact for any unit span under 255), ~21
adds and logic. Its bank conflicts: 0.02 a window (the bank model's 4 wavefronts a store, `tools/bank_model.py`).

What it settles:

- The lean form forms the carry's image and exponents exactly in 0.73x its cycles (0.71x in halves), 0.80x its
  instructions.
- Its floor is the shared pipe and the ALU pipe, not issue. The image is 512 stores of 4 bytes a lane a window at 2
  cycles each ([`smem_rates`](smem_rates.md)), 1,024 cycles; the maxima's float max, the packs and the integer work run
  on sm_86's half-rate ALU pipe, ~1,310 cycles a scheduler a window.
- In halves the hand-off alone costs 1% more (the carry's 4%): sm_86's halves cost their readout between them, not
  their barrier.
- THE 16-BYTE IMAGE (ruled 2026-10-18): 0.74x lean's cycles (0.54x the carry's form), exact, its stores
  conflict-free. It runs 1.41x its ALU pipe's 1,166 cycles: the barrier between the maxima and the stores runs a
  phase of only ALU work (the maxima) and then one that shares it with the FMA pipe and the stores, so the two do not
  overlap within a window. In the carry the hand-off's ALU work sits beside the fold's and the readout's MMAs (other
  pipes); that overlap is the composed window's, not this lab's.
- The format sets the floor: 176 of the 292 ALU instructions a warp are the maxima's FMNMX and the f16 packs. The
  exponent is what lets the readout accumulate in fp16 (a 32-list sum under 2^10 at `kImageTop`), twice the fp32
  rate on GeForce; a bf16 image would need no exponent and no maxima but only accumulates in fp32 -- the same on
  sm_80, the readout's tensor time doubled on sm_86.
- The hand-off alone in the carry's form runs 3.06K cycles, against 3.7K for the maxima and the stores in the composed
  window (the window lab's form 6, its record `measure/lab/results/carry_window`): the rest is the maxima's barrier waiting on the slowest fold warp
  and the next window's staging, which the store phase holds.

Record: `measure/lab/results/handoff/sm_86.json`.
