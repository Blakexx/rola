# `read_image`

Mirrors `measure/lab/experiments/read_image.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's readout step in its ruled final form (lab-local code: no production readout yet), the state's operand from
a bf16 image in shared memory, on nl64k-alt-k4's real read lists and a dense cell. One CTA an SM a box of 32 lists x
32 read digits x 32 channels; a warp reads digits w, w + 8, w + 16, w + 24, every digit's first step in one
straight-line block. The entries arrive in the facts pass's final form, built by `read_image.inputs.py` standing in for
the facts pass:
- the pool row's byte offset with its swizzle folded in, its spare high half holding the digit's step count and a
  flag for steps past the staged entries;
- the coefficient times the unit;
- the panel row's offset less the lane offset of the entry's token in its step.

The entries are conflict-free by construction (KERNEL_STANDARDS §30):
- **Coloring:** a box-window's union slots are colored, so a digit's tokens have distinct residues mod 8 (a row's
  bank group). 99.75% of digits with at most 8 tokens get them.
- **Placement:** each token sits at the position its residue requires in the instruction that touches it.
- **Pads:** a pad takes a free residue's row and adds coefficient 0 to that residue's dummy panel row.

Each output pair converts on the FMA pipe to the exact integer hi x 2^16 + lo, added by `red.shared` into its token's
panel row. A warp's first 16 entries a digit (32 at W = 1024) arrive a window ahead by `cp.async`, a 16-byte chunk a
lane. Lane constants are held in registers (`lab::held`), and every branch on loaded data is a vote (§29.9). There are
two forms: tokens on M (16-position steps, the state the B operand) and tokens on N (8-token tiles, the state the A
operand, every digit's first tile in one straight-line block).

The check runs the even channels negative, so a low half below zero carries into the high half and an inexact pack
fails. It uses two coefficients by slot and digit, and every output word is exact. Cycles are the CTA's span
(`WarpSpan`), measured against the tensor floor that matches the variant's synchronization: each window's busiest
scheduler with the pass barrier, and the busiest over all windows when warps run free (§29.10).

| sm_86 | + panel adds, barrier | free-running | + the drain | instr / MMA | wavefronts / MMA (conflicts) |
|---|---|---|---|---|---|
| alt-k4, tokens on M (256.8 MMAs a CTA-window) | 1.26x | 1.07x (2198 cycles) | 1.78x | 8.2 | 4.83 (0.02) |
| alt-k4, tokens on N (180.3) | 1.44x | 1.30x (1896 cycles) | 2.19x | 10.8 | 6.39 (0.03) |
| dense, tokens on M | 1.03x | | 1.07x | 5.4 | 2.13 (0) |
| dense, tokens on N | 1.04x | | 1.08x | 5.9 | 2.13 (0) |

Before the coloring and placement, conflicts ran 1.59 per MMA on M and 2.14 on N, and wavefronts 6.40 and 8.53. The
bank model predicted the drop before the run (`banks.json`).

Per token at alt-k4, tokens on N cost less on both cards:
- **sm_86:** 3.70 cycles against M's 4.29 (0.86x).
- **A100 (shared-memory bound):** 2.61 predicted shared cycles against 2.93 (0.89x).

On sm_86, N saves less than its 0.70x MMAs suggest. Shared memory is one pipe an SM, and N's 7.4 cycles per MMA add up
to ~92% of its tensor floor, so the two pipes are co-bound. Dense ties: same MMAs, same wavefronts.

Every variant meets its instruction and wavefront budgets. The forms this replaced (F2I and PRMT packs, which are
inexact under u32 adds when a low half is negative; a state ring; half steps; select-routed pads; uncolored slots)
are recorded at carry-k5 8434df71 and 66c79def.

Record: `measure/lab/results/read_image/sm_86.json`.
