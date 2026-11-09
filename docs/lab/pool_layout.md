# `pool_layout`

Mirrors `measure/lab/experiments/pool_layout.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The owner part's fill into 160-byte pool rows plain, with the 16-byte halves of each 32-byte pair swapped on rows with
bit 2 set, or 176 bytes apart; and the steps' in-place reads of a row (two `ldmatrix.x4.trans` over the values, four
coefficient bytes a lane). sm_86: L2 sectors (`ncu`) 7 a token plain and swapped, 9 at 176 bytes (odd rows 16 bytes
off phase); the fill a pass from DRAM ~3.6K cycles in every layout, from L2 1.70K / 1.71K / 2.28K; the reads 192 cycles
a step plain (2-way bank conflicts: 160 bytes puts every row's chunk on the same parity of 16-byte bank group), 119
swapped, 118 at 176 bytes. Swapping halves inside an aligned 32-byte pair keeps cp.async's sector requests and removes
the conflict. Records: `measure/lab/results/pool_layout/sm_86.json`, `sm_86.counters.json`.
