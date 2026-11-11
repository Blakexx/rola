# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum s64 | slots] [--dv 64] [--tokens 65536] [--json floor.json]

THE SHAPE. A level is `--widths` digits, sparse on a side where its flag is 1 (`--k` live digits a token, drawn
uniformly, so every count is an expectation). A box is `--box` digits of each level and a cluster `--cluster` boxes
along each level. The tool's docstring states each term; what it leaves out is listed there too -- the hand-over
between producer and consumer warps, the den, the divergent carve's register swap, paging, and every other kernel.

THE CLUSTER'S SUM takes one of two forms. `s64`: each box adds its partial as 64-bit integers into the cluster's
collector (an integer reduction into another CTA's shared memory is an instruction of sm_90; a float one is not),
which costs a conversion a value and a scale agreed a window ([`arith_model`](arith_model.md#sums) measures what that
scale costs in accuracy). `slots`: each box stores its fp32 partial into its own slot of the collector's shared
memory and the collector adds a token's slots in registers -- the fp32 sum itself, with no scale and nothing to
agree, at the price of reading the bytes back. Both move the same bytes through shared memory.

MODELLED (2026-11-11; the H100's records for the pipe, the shared access rate and the drain; a load request, the
issue rate, the clock and the memory bandwidth ESTIMATED; alt = read side sparse on one level, write side on the
other, 4 live digits; boxes 16 x 16; `--sum s64`; SM-cycles a token, low to high, and the 64K call):

| the shape | boxes read | clusters read | the sum (partials, drains) | fold values | total | the call |
|---|---|---|---|---|---|---|
| 256 x 256 alt, no cluster | 58.2 | 58.2 | 0, 1,176 to 2,726 | 320 to 640 | 1,590 to 3,460 | 440 to 950 us |
| ... a cluster the 16 boxes one read range's tokens read | 58.2 | 3.64 | 200, 74 to 170 | 88 to 176 | 450 to 730 | 125 to 200 us |
| ... a cluster 8 x 2 | 58.2 | 6.62 | 200, 134 to 310 | 82 to 165 | 510 to 860 | 140 to 240 us |
| ... a cluster 4 x 4 | 58.2 | 10.9 | 200, 221 to 512 | 60 to 120 | 570 to 1,020 | 160 to 280 us |
| 64 x 64 alt (4,096 leaves), clusters of 4 | 10.9 | 2.73 | 38, 55 to 128 | 29 to 59 | 145 to 260 | 40 to 72 us |
| 16^4 alt (two sparse levels a side), the same clusters | 58.2 | 3.64 | 200, 74 to 170 | 352 to 703 | 990 to 1,530 | 270 to 420 us |
| 256 x 256 dense on both sides, clusters 4 x 4 | 256 | 16 | 878, 323 to 749 | 1,280 to 2,560 | 8,300 to 10,300 | 2.3 to 2.9 ms |

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and the 256 x 256 alt row's cost a token does not move.

WHAT IT SHOWS. The tensor work is 64 of the alt rows' cycles; the rest is bytes, counted a (token, box): the partials
into the cluster, the drains, the values. A cluster is worth a factor of three to five, and its shape trades the
readout's drains against the fold's value loads -- which side to favour rests on the ESTIMATED load request, the
least known rate here. A deeper routing at the same box reads the same boxes with four times the leaves, and its
fold sits in four times the lists a box. Dense on both sides at as many leaves as tokens is over FlashAttention's
floor by the tensor work alone (4,096 against 2,049). `--sum slots` reads the same totals less the conversion and the
scan.
