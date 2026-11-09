# `read_step`

Mirrors `measure/lab/experiments/read_step.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The readout's step on the even carve: a warp owns 4 level-S read digits x 32 level-R digits x 32 channels, its slice
as f16 B fragments (K = level R, permuted so a lane's four k rows are one token's four consecutive digits) beside the
fold's 128 fp32 accumulators (registers 240-254); a step's A = 16 tokens' level-R bytes by one `ldmatrix.x4` (96-byte
read rows) and a PRMT + HFMA2 a register at the token's level-S coefficient; 8 fp16-accumulate MMAs and 2 for den; the
outputs to s16 pairs (two channels a word) added by `red.shared.add` (ATOMS) into a 128-slot panel of 17-word rows; the
drain two slots a warp request by `red.global.add.u32`, each word cleared. PER DIGIT steps each digit over its own
gathered tokens and adds after each; MERGED sums the warp's 4 digits over the union in registers and adds once. The
block's span a pass against its tensor floor (the busiest scheduler's MMAs at 16 cycles). sm_86, cycles a pass:

| form | steps | steps alone | + panel adds | + drain burst | + drain spread | + burst, aligned | + spread, aligned |
|---|---|---|---|---|---|---|---|
| merged | 8, 16 live | 12.0K | 11.8K | 13.6K | 14.6K | 13.0K | 13.8K |
| per digit | 8 a digit, 16 live | 14.6K | 31.2K | 32.0K | 36.5K | 31.3K | 35.9K |
| per digit | 2 a digit, 16 live | 3.75K | 8.11K | 9.54K | 10.6K | 8.93K | 9.92K |
| per digit | 1 a digit, 8 live | 1.54K | 3.27K | 5.61K | 5.77K | 4.23K | 4.68K |

"Aligned" drains the channel words alone to 64-byte target rows (den carried elsewhere); the other drains write 17-word
rows (channels and den) to a 17-word pitch, den a request of its own: ncu counts 316 L2 reduction requests a CTA a pass
for their 128 RED instructions -- a two-token request spans three or four lines, den two. The panel adds above are each
a branch on its token (`if (in) red.shared`), issued right after their own step's MMAs; merged adds once a token for
its four digits. The drain costs ~1.0K cycles a 128-slot pass aligned (2.3K misaligned with den), and spreading it
through the next pass's steps under a double panel and mbarriers costs 0.4-1K more than the burst, not less: the
drain's instructions and waits land in the steps. The coefficient loads are unconditional (a pad's slot is a real row,
its value masked), which took the steps alone from 17.1K / 4.39K / 1.66K.

THE DEN PASS'S STEP (8 MMAs, den out), its adds issued after the next step's MMAs:

| per digit | 8 a digit | 2 a digit | 1 a digit, 8 live |
|---|---|---|---|
| MMA floor | 8.19K | 2.05K | 1.02K |
| adds a step behind, each a branch | 26.4K (3.2x) | 6.75K (3.3x) | 2.60K (2.5x) |
| + operands a step ahead, a switch on the digit and a guarded prefetch | 34.9K | 8.92K | 3.79K |
| + no branch: loads unconditional, adds a value masked to zero | 16.9K (2.1x) | 4.60K (2.2x) | 2.32K (2.3x) |
| + the sequence unrolled: rings at fixed registers | 12.8K (1.56x) | 3.42K (1.67x) | 2.03K (1.99x) |
| + the coefficient after the MMAs, in the conversion | 12.2K (1.49x) | 3.31K (1.62x) | 2.11K (2.06x) |
| the same, converted by F2I (an f16 multiply, then the conversion unit) | 12.7K | 3.49K | 2.47K |
| the coefficient after the MMAs, entries as byte offsets | 12.4K (1.52x) | 3.25K (1.59x) | 1.94K (1.89x) |
| + the conflict-free panel (20-word rows, pads to their own rows) | 12.4K (1.52x) | 3.26K (1.59x) | 1.67K (1.64x) |
| isolation: without the conversion (values wrong) | 9.60K (1.17x) | 2.79K (1.36x) | 1.46K (1.43x) |
| isolation: without the adds (values wrong) | 8.89K (1.09x) | 2.34K (1.14x) | 1.22K (1.19x) |
| isolation: the conversion independent of the outputs | 11.7K (1.43x) | 3.14K (1.54x) | 1.89K (1.85x) |
| isolation: half the values converted | 11.4K (1.39x) | 2.96K (1.44x) | 1.54K (1.51x) |
| isolation: the conversion replaced by as many dependent LOP3/PRMT | 13.6K (1.66x) | 3.78K (1.85x) | 1.76K (1.72x) |
| isolation: those integer operations independent of the outputs | 14.4K (1.76x) | 3.69K (1.80x) | 1.95K (1.90x) |
| 12, MMA j reading output word j - 1 masked to zero | 12.2K (1.49x) | 3.15K (1.54x) | 1.75K (1.71x) |
| the same, each word also reading an MMA's output | 12.7K (1.55x) | 3.24K (1.58x) | 1.77K (1.73x) |
| 12, a scheduler's two warps taking turns at the MMAs (named barriers) | 14.2K (1.73x) | 3.60K (1.76x) | 1.89K (1.85x) |
| 12 with the level-R side native f16 (A loaded as it is, no unpack) | 10.7K (1.31x) | 2.89K (1.41x) | 1.57K (1.53x) |
| the same, accumulating in fp32 (half rate on this card) | 17.6K (2.15x) | 4.48K (2.19x) | 2.31K (2.25x) |

Merged, adds a step behind: 10.5K (1.28x). The aligned burst drain adds ~1K a pass. The last form converts on the FMA
pipe (each half to fp32, x 64 + 1.5 x 2^23, the low halves packed: round-to-nearest, exhaustively checked over every
in-range half) in place of `F2I.S16.F16` on the quarter-rate conversion unit; it moved nothing at 8 steps a digit
(16.9K either way) -- issue binds, not that unit. ncu, the first branchy form: issue 40%, tensor 31%, 163 instructions
a digit-step, stalls on fixed-latency and short-scoreboard waits and branch resolution; the unrolled form issues 132
a digit-step, against 128 cycles of MMAs a warp-step: two warps a scheduler leave it at the edge of issue-bound.
Applying each token's level-S coefficient after the MMAs (it factors out of the sum over level-R digits: A is then
the raw bytes, unpacked by PRMT less 1024, and the coefficient times the unit rides in the conversion's FFMA, a pad's
scale zero, so nothing is masked) issues 118 a digit-step; the rest is the conversion (40: each half to fp32, the
FFMA, the pack, for 16 values a step), the unpack (16), the MMAs and adds (16) and addresses. A shared f16x2 atomic
(`red.shared.add.noftz.f16x2`) compiles on sm_86 to a compare-and-swap loop (`ATOMS.CAST.SPIN`), not an add, so an
f16 panel is no way around the conversion here. Converting by F2I instead (24 instructions a step against the FMA
path's 40) is slower at every
density: the quarter-rate conversion unit binds before issue does. Entries carrying their byte offsets (a token's row
with the swizzle folded in,
its panel row and a pad bit, as the facts pass could emit them) bring the step to 105 instructions a digit-step:
issue fits under the MMAs (two warps 210 against 256 cycles). What remains is the measured tax (KERNEL_STANDARDS 26.3
a): beside fp16-accumulate MMAs an FP32 instruction costs ~1.0 cycle of the chain, so the conversion's 16 FFMAs and
the unpack's packed adds and permutes add ~30 cycles to a 128-cycle step -- a tax-inclusive floor of ~1.23x against
the 1.5x measured at 8 steps a digit.

THE GAP TAKEN APART. The panel's 17-word rows put the 8 tokens of an atomic add on overlapping banks (~3 wavefronts an
add, 33 a step against 9 without adds); 20-word rows land slots of distinct residue mod 8 on disjoint 4-bank groups (the
tokens of each step's aligned 8 already hold distinct residues -- the facts pass's residue order), and pads adding zero
each to slot 0 serialized on one word, so they add to 8 dummy rows: 0 atomic bank conflicts, 17 wavefronts a step,
-14% at 8 live tokens, nothing at full steps -- shared memory was not what bound them. Removing the conversion (the
words added raw) or the adds as well isolates the rest: at 8 steps a digit the operands and MMAs run 1.09x, the adds
take it to 1.17x and the conversion to 1.52x -- 34 of the 52 points; at 2 steps a digit 14 / 22 / 23, at 1 step 19 / 24
/ 21.

THE CONVERSION TAKEN APART (8 steps a digit; a scheduler's floor is two warps' 8 MMAs, 257 cycles a step). Its 40
instructions a warp-step (16 `HADD2.F32` to fp32, 16 `FFMA`, 8 `PRMT`) cost 0.345 of the floor over the raw adds.
Computed on values independent of the MMA outputs (each word XORed with it, so the same instructions issue, none on
the output-to-atomic chain), they cost 0.261; half of them on the chain 0.215. So: the chain's latency 0.08, the
instructions 0.26, linear in their count (0.08 + 0.13 at half). The tensor_tax rates beside fp16-accumulate MMAs
price the instructions at 16 x 1.0 + 16 x 0.13 + 8 x 0.22 = 20 cycles a warp-step, 0.155 at two warps; the other
~0.1 is issue pressure -- the off-chain form issues 94% of the floor's issue capacity (ncu: issue 65.5% at 1.43x;
`not_selected` 0.46 an issue against 0.30 without the conversion). The tax itself is not issue: in `tensor_tax` one
filler warp beside one MMA warp a scheduler leaves 15 free slots an MMA, where collisions would cost ~1/16 cycle an
instruction, and classes of one issue cost tax 1.0 (FP32), 0.48 (packed half), 0.13-0.22 (conversions, LOP3, PRMT).
The integer class is no escape on sm_86: LOP3 and PRMT run at half rate (16 lanes), so 40 of them a step load the ALU
pipe past the floor (ncu: 67.7% of its peak at 1.76x, 119% of the floor's time) -- in the chain 1.66x, off it 1.76x.
At 2 steps a digit the same split: chain 0.055, instructions 0.17.

THE INTERLEAVE IS PTXAS'S. Mode 12's SASS bursts 24 MMAs (about 4 instructions apart), then runs ~200 instructions
without one. Written MMA by MMA in the source (each MMA, then one of the last step's output words), the SASS came out
identical to 12's. Per-instruction samples (ncu source counters) put a warp at an MMA 49% of the time -- 28% waiting
on the busy tensor pipe, 7% ready but not selected while the other warp issued (~13% of a scheduler's cycles with
the pipe free) -- and the conversion mostly issuing: no instruction class favors a ready MMA. A zero-valued
dependency places the MMAs (MMA j's accumulator takes output word j - 1 ANDed with a run-time zero, `passes >> 31`):
the longest MMA-free run in the step falls from 208 to 83 instructions, not-selected MMAs to 3.8%, and the step to
1.49x. Tied both ways (each word also reads an MMA's output) it falls to 54, but the MMA latency joins the chain:
1.55x. Turn-taking -- the two warps of a scheduler alternating at a step's MMAs under named barriers (`bar.sync`,
MMAs, `bar.arrive`), each converting while the other's MMAs run -- does not survive ptxas: the barriers fence memory
only, the MMAs move outside them, and the barriers cost 1.73x. `tools/sass_gate.py` reports the MMA gap (the longest
run in a step loop's body, and around it) and `lab.py sass` prints it.

INSTRUCTIONS AN MMA, NOT THEIR ORDER. The step loop's non-MMA instructions an MMA, the same compiled for sm_80 and
sm_86 (`lab.py sass read_step --arch sm_80`): 12.2 for mode 12; 9.6 with the level-R side native f16 (rows of 32
halves, 64 bytes, chunks swizzled by `(slot >> 1) & 3`; two `ldmatrix` a step and no PRMT/HADD2 unpack); 7.6 with
that and fp32 accumulation (no `HADD2.F32`; outputs converted by the FFMA and packed). Removing the unpack's ~21
instructions a step took the step from 1.52x to 1.31x at 8 steps a digit (ncu: 693M instructions against 864M,
tensor 75% against 65%) -- more than the tax rates price the unpack at (~0.05), since near issue saturation each
instruction also costs contention. The fp32-accumulate form runs at half rate here (32-cycle MMAs, 2.15x the fp16
floor, 1.07x its own); a full-rate card (A100, A6000) is where it would run, and its time there is unmeasured. sm_80's
shadow is 8 cycles an MMA, 7 issue slots: 7.6 is the first form near it.

THE PANEL AGAINST ADDING TO GLOBAL ROWS (2026-10-14, modes 24 and 25; Blake: "Lab first, then decide"). Mode 24 is 12
with its panel drained in a burst at the pass's end (every slot, two a warp request); 25 is 12 with each step's output
words added straight to their tokens' rows (`red.global.add`), no panel and no drain. Both add into 4,096 rows every
CTA shares, L2-resident, as a window's boxes all add into its 512 tokens' rows. Cycles a pass:

| step | 12 without adds | 12 (panel adds) | 24 (+ drain) | 25 (global adds) |
|---|---|---|---|---|
| 8 steps a digit, 16 live | 8,887 | 12,417 | 11,700 | 81,219 |
| 2 steps a digit, 16 live | 2,341 | 3,256 | 3,778 | 19,877 |
| 1 step a digit, 8 live (alt-k4's lists) | 1,218 | 1,674 | 2,358 | 10,710 |

Adding each entry's partials to global memory runs 4.5x the panel and its drain at alt-k4's step, 7x at 8 steps: an
entry's adds are as many requests as a slot's drain, every box's land on the same rows, and the L2's atomics on them
serialize. The panel merges a slot's digits first, so the carry keeps the panel and drains a slot once.

A per-digit step's gathered slots are distinct (a list never holds a token twice). The gate: `read_step`'s branchy
forms declare their branches (KERNEL_STANDARDS 27). Records: `measure/lab/results/read_step/sm_86.json`,
`sm_86.counters.json`, `sm_86.source.json` (the per-instruction samples, by class and stall reason).
