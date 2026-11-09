# `list_fold`

Mirrors `measure/lab/experiments/list_fold.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The fold's step forms for the carve re-price, 8 warps (two a scheduler), fp32 accumulate, the kernel's 128 accumulator
registers live, 256 rows resident (160 bytes: level S's 32 f16 digits, 32 f16 values, level R's 32 u8 digits; the
halves of each 32-byte pair swapped on rows with bit 2 set). PER LIST: a warp steps each list over its own tokens'
rows (gathered by hash, or the pass's own contiguous 16), a row past the list's live count reading row 0 with c = 0;
MERGED: a warp steps the union, every list's MMAs on every token, the operands loaded once. Two steps' operands loaded
ahead of their MMAs; the block's span a pass against its tensor floor (the busiest scheduler's two warps' MMAs at 32
cycles). sm_86:

| form (a warp's lists x level-S digits x channels) | steps | live | rows | x floor |
|---|---|---|---|---|
| merged, 8 x 16 x 32 | 8 | 16 | contiguous | 1.055 |
| merged, 4 x 32 x 32 | 8 | 16 | contiguous | 1.081 |
| per list, 4 x 32 x 32 (each list reloading the same rows) | 8 a list | 16 | contiguous | 1.262 |
| the same, staged (step s + 1's loads before step s's MMAs) | 8 a list | 16 | contiguous | 1.349 |
| per list, 8 x 16 x 32 | 8 a list | 16 | contiguous | 1.977 |
| per list, 4 x 32 x 32 | 2 a list | 16 | gathered | 1.424 |
| per list, 4 x 32 x 32 | 2 a list | 8 | gathered | 1.167 |
| per list, 4 x 32 x 32 | 1 a list | 8 | gathered | 1.166 |
| per list, 8 x 16 x 32 | 2 a list | 16 | gathered | 2.160 |
| per list, 8 x 16 x 32 | 1 a list | 8 | gathered | 1.706 |
| per list, 4 x 32 x 32 | 2 a list | 16 | gathered, residue-ordered | 1.263 |
| per list, 4 x 32 x 32 | 2 a list | 8 | gathered, residue-ordered | 1.107 |
| per list in one block, c precomputed, 4 x 32 x 32 | 8 a list | 16 | contiguous | 1.015 |
| the same | 2 a list | 16 | gathered, residue-ordered | 1.017 |
| the same | 2 a list | 8 | gathered, residue-ordered | 1.017 |
| the same | 1 a list | 8 | gathered, residue-ordered | 1.030 |

At dense, per list (all four lists reloading the same rows) is 25.8K cycles a pass against the merged form's 18.8K at
the same tile: 1.37x, from four times the operand loads and a mass atom a list step. Per list at today's 16 x 32 tile
is operand-bound (1.7-2.2x); at 32 x 32 a half-full step (alt-k4's ~8 tokens a list a window) runs 1.17x its floor
even with gathered rows, a full gathered step 1.42x (16 random rows share bank groups where 8 pads on row 0
broadcast). The staged form does not interleave: ptxas keeps the next step's loads in a block ahead of the MMAs, each
entry's pad check a branch that fences the scheduler (its SASS), and it is slower.

THE PER-LIST STEP AT ITS FLOOR (form 5, 2026-10-07). KERNEL_STANDARDS 27's shape -- a pass one block (every list's
steps unrolled, step i + 1's operands loaded before step i's MMAs into a two-slot ring at fixed registers), entries
as byte offsets (`at(row, 0)`: a lane's chunk bit is lane-fixed, so its row is an XOR and the m-tile and n-chunk
offsets immediates), pads masked (row 0, c = 0), each token pair's c precomputed as f16 `{c_v, c_m}` pairs (what the
facts pass would emit, so the step only multiplies A by it), the mass accumulated in place (no fresh fragment and its
adds) -- with gathered rows residue-ordered (each 8 of a step distinct mod 8: the rows' bit-2 swap then lands an
`ldmatrix`'s 8 rows on 8 bank groups). Non-MMA instructions an MMA in the step loop, the same for sm_80 and sm_86:
2.5 / 3.4 / 4.7 at 8 / 2 / 1 steps a list, against 13.4-17.0 for form 1 and 5.6 for the merged form -- under sm_80's 7
issue slots an 8-cycle MMA at every density. Timed, 1.015-1.030x. Residue order alone takes form 1's full gathered
step from 1.374x to 1.263x (ncu: 58M bank conflicts in 145M wavefronts hashed, 0.06M ordered); the shape and the
precomputed c take the rest. Tensor active 49% of the fp16 peak, the fp32-accumulate pipe's whole rate on this card.

THE COEFFICIENT FORMED IN THE FOLD (forms 6 and 7, 2026-10-07): what the fold pays when the facts pass emits 2-byte
entries and the warp forms each token's c itself -- the list digit's byte from the token's pool row (f16 through the
mantissa: `0x6400 | b`, less 1024), times the token's `(F, m)` f16 pair, two HMUL2s a token pair. The row now holds
level R's 28 digits and that pair unswizzled after the swizzled level-S and value chunks (form 5 unchanged: 1.015 /
1.018 / 1.030). Form 6's entries are slots (the row's offset and the gains' index computed from each); form 7's are
the rows' offsets, the gains read from the row. Against form 5 at 8 / 2 / 1 steps a list:

| form | sm_86 x floor | non-MMA instructions an MMA (sm_80 = sm_86 SASS) |
|---|---|---|
| 5, c in the entry | 1.015 / 1.018 / 1.030 | 2.46 / 3.44 / 4.70 |
| 6, slot entries | 1.039 / 1.048 / 1.062 | 6.11 / 7.05 / 8.30 |
| 7, offset entries, gains in the row | 1.032 / 1.042 / 1.051 | 5.31 / 6.25 / 7.50 |

Forming c costs ~2.9 instructions an MMA at best (the loads, the byte's conversion, the pairs' PRMTs and HMUL2s);
slots add ~0.8 of address arithmetic. sm_86's fp32-accumulate MMA (32 cycles) hides it, +1.6-2.4%; sm_80's 8-cycle MMA
leaves 7 issue slots, which form 7 fills at two steps a list and overruns at one. The numerics differ too: c from
f16 gains at the window's scale, not fp32 gains at the list's own, so a list of light tokens loses its scaling.
Records: `measure/lab/results/list_fold/sm_86.json`, `sm_86.counters.json`.
