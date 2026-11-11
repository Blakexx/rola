# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum none | slots | s64] [--round 64] [--dv 64] [--tokens 65536] [--json floor.json]

WHY IT EXISTS. The hopper carry's design is a choice of box and cluster, and each choice moves several costs at
once: a box read is a partial to sum, a box written is a set of operands to load, and a cluster shares one or the
other. The model makes that trade one calculation. Its inputs are the declaration's structure alone (the levels'
widths and which side is sparse on which), so a result holds for any draw of that structure, as an expectation.

WHAT IT LEAVES OUT: the hand-over between producer and consumer warps, the den, the divergent carve's register swap,
paging, and every other kernel. Its rates are the H100's; no blackwell card is priced (none has been measured), and
a form that names an sm_90 instruction states no sm_100 form here -- owed with that family's first lab.

THE CLUSTER. By default (`--sum none`) a cluster shares the fold's value loads and nothing else: every box drains
its own partial. THE CLUSTER'S SUM, asked for, takes one of two forms, each summing a token's partials inside a
cluster before one fp32 drain to global memory. `slots` is the fp32 sum the format states
([precision.md](../../architecture/precision.md)):
each box stages its partial and sends it by a bulk copy into its own slot of a collector's shared memory, and the
collector adds a token's slots in registers. `s64` sends 64-bit integers by a bulk reduction, at a scale agreed a
window -- outside the fp32 sum the format states, counted for comparison ([`arith_model`](arith_model.md#sums)
measures its accuracy). Both are MEASURED by [`cluster_sum`](../../lab/cluster_sum.md) at two round sizes, and the
model reads that record's line: a cost a round a CTA whatever the round holds (`--round` partials a CTA hand over
between barriers; beyond the lab's 64 the line is UNMEASURED), and a cost a partial. The lab's partial includes its
collectors' drain, an eighth of one, which the model's drains count again: a row with a cluster's sum is over by that much. The
warps' own fp32 add into a mate's shared memory, read at eighteen times a store, is no form here.

MODELLED (2026-11-11; the H100's records for the pipe, the shared access rate, the drain and the cluster's sum; a load
request, the clock and the memory bandwidth ESTIMATED; `--read-sparse 0 1 --write-sparse 1 0 --k 4
--box 16 16` unless the row says, the 16^4 row `--read-sparse 0 0 1 1 --write-sparse 1 1 0 0 --k 2 --box 4 4 4 4`;
SM-cycles a token, low to high, and the 64K call):

| the shape | boxes read | the sum (partials, rounds, drains) | fold values | total | the call |
|---|---|---|---|---|---|
| 256 x 256, no cluster | 58.2 | 0, 0, 1,176 to 2,726 | 320 to 640 | 1,590 to 3,460 | 440 to 950 us |
| ... `--cluster 16 1 --sum slots`, rounds of 64 | 58.2 | 897, 6,594, 74 to 170 | 88 to 640 | 7,740 to 8,390 | 2.1 to 2.3 ms |
| ... `--cluster 16 1 --sum slots --round 512` | 58.2 | 897, 824, 74 to 170 | 88 to 640 | 1,970 to 2,620 | 545 to 725 us |
| ... `--box 256 1`: a box the whole read-dense level, no cluster | 4.0 | 0, 0, 80 to 186 | 5,120 to 10,240 | 5,290 to 10,520 | 1.5 to 2.9 ms |
| ... `--box 256 1 --cluster 1 16`: those boxes sharing value loads | 4.0 | 0, 0, 80 to 186 | 320 to 10,240 | 490 to 10,520 | 135 us to 2.9 ms |
| 64 x 64 (4,096 leaves), no cluster | 10.9 | 0, 0, 221 to 512 | 80 to 160 | 320 to 690 | 90 to 190 us |
| 16^4 (two sparse levels a side, 2 live digits, boxes 4^4), no cluster | 49.0 | 0, 0, 989 to 2,293 | 1,280 to 2,560 | 2,360 to 4,940 | 650 to 1,360 us |
| 256 x 256 dense on both sides, no cluster | 256 | 0, 0, 5,169 to 11,981 | 20,480 to 40,960 | 31,500 to 58,800 | 8.7 to 16 ms |

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and the 256 x 256 alt row's cost a token does not move.

WHAT IT SHOWS. The tensor work is 64 of the sparse rows' cycles; the rest is bytes, counted a (token, box): the
output's partials and the fold's values. A CLUSTER DOES NOT PAY: a partial costs as much into a mate's shared memory
as to global memory, and the round's barriers come on top -- at the lab's round the clustered row is two to five times the
unclustered one, and at a round of 512 partials (unmeasured, and more shared memory than the slots have) it is no
better than the unclustered row's low end. So the output's sum is priced by the count of partials a token, 58 at
these boxes, and at 65,536 tokens the unclustered row stands around FlashAttention's floor, not under it. What
lowers it is fewer partials a token, and the box's shape does that: a box the whole of the read side's dense level
is read by 4 boxes a token, not 58, and its sum falls to 80 to 186. The cost moves to the fold, whose every token
then writes into all 256 boxes and owes each its value. A cluster of those boxes taking one value load between them
is the form's whole case, and that load is the model's widest ESTIMATE -- from one request a cluster to one a box,
490 to 10,520 cycles a token -- because no lab has landed a load in every CTA of a cluster. The model also charges
a value once a (token, list), four a box here, where one landed copy may serve a box's lists. Dense on both sides at as many leaves as tokens is over FlashAttention's floor by
the tensor work alone (4,096 against 2,049).
