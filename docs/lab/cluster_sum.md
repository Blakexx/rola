# `cluster_sum`

Mirrors `measure/lab/experiments/cluster_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A token's output is a sum of partials that different CTAs hold, and each partial added to global memory by itself is
the carry's largest cost in [`carry_floor`](../internals/tools/carry_floor.md). What does a token cost a CTA when a
cluster sums its partials in one CTA's shared memory first, by each of five forms, beside every CTA draining its
own -- and how much of that is the round's fixed cost? And can one collector take a cluster's partials, or must the
collecting be spread?

THE FORM. Clusters of 8 CTAs, 4 warps each, as many clusters as the card's SMs hold. A round is a partial of 16 or of
64 tokens × 64 channels a CTA, from registers: a form's cost a token at two round sizes separates what it pays a
round from what it pays a token.
- `direct`: every CTA adds its partial to its cluster's sum in global memory.
- `slots`: every CTA's warps store its partial into its own slot of the collector's shared memory; the collector
  adds a token's 8 slots in registers and adds the sum to global memory. The fp32 sum itself.
- `atomic`: every CTA's warps add fp32 into the collector's one panel; the collector reads it, clears it, drains it.
- `bulk slots`: the slot form with the partial staged in the CTA's own shared memory and sent by one bulk copy a
  collector, whose bytes complete the collector's barrier.
- `bulk integers`: the partial staged as 64-bit integers at a fixed scale and sent by one bulk reduction into the
  collector's one panel, converted back by the collector.
- `warp integers`: the warps' own 64-bit integer adds into that panel.
- `hand-over`: the barriers of a round of the warps' forms (an arrival a CTA at the collector, an arrival a collector
  at every CTA) with no partial sent or summed. A bulk form's round completes its collector's barrier by bytes, with
  a fence for the bulk engine and a re-arm, and no row isolates that.
Each cluster form with ONE collector (rank 0 takes every token) and with EIGHT (a token's collector is its index
modulo 8). Barriers a round order sources and collectors -- every thread that stored or added into a mate, and every
collector's thread after its reads and clears, fences at cluster scope; an arrival releases to the cluster, a wait
acquires from it -- and the cluster's own barrier its
start and its end. A mate's addresses are mapped once, before the clock. Every row's timed round includes its
collectors' drain to global memory. Every sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a warp's store into a cluster-mate's shared memory is a
generic store (`ST.E`). A warp's float add there is `ATOM.E.ADD.F32` on one path with compare-and-swap loops on two
others, chosen by the address at run time; its 64-bit integer add is `ATOM.E.ADD.64`. The bulk copy is one
`UBLKCP.S.S`, the bulk integer reduction one `UBLKRED.S.S.ADD.U64`.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60242098, driver 580.173.02; 2026-11-11, the experiment at
34960275; 7 repetitions, every sum exact; cycles a token a CTA, the median, the spread under 1.0%; THE ROUND and THE
TOKEN are the two columns' split, cost = the round / its tokens + the token):

| the form | 16 tokens a round | 64 tokens a round | the round | the token |
|---|---|---|---|---|
| every CTA drains its own | 36.7 | 23.5 | 280 | 19.2 |
| fp32 slots stored by the warps, one collector | 570.8 | 176.7 | 8,407 | 45.3 |
| fp32 slots stored by the warps, eight collectors | 568.7 | 155.3 | 8,820 | 17.5 |
| fp32 atomic adds into one panel, one collector | 862.4 | 463.0 | 8,520 | 329.9 |
| fp32 atomic adds into one panel, eight collectors | 1007.1 | 488.4 | 11,064 | 315.6 |
| fp32 slots by a bulk copy, one collector | 491.1 | 163.7 | 6,985 | 54.6 |
| fp32 slots by a bulk copy, eight collectors | 468.2 | 128.6 | 7,245 | 15.4 |
| 64-bit integers by a bulk reduction, one collector | 490.2 | 270.5 | 4,687 | 197.3 |
| 64-bit integers by a bulk reduction, eight collectors | 480.1 | 147.7 | 7,091 | 36.9 |
| 64-bit integer adds by the warps, one collector | 584.6 | 305.8 | 5,948 | 212.9 |
| 64-bit integer adds by the warps, eight collectors | 605.6 | 195.2 | 8,754 | 58.5 |
| the round's barriers alone, no partial, one collector | 453.7 | 113.4 | 7,259 | 0.0 |
| the round's barriers alone, no partial, eight collectors | 481.1 | 120.3 | 7,697 | 0.0 |

WHAT IT SHOWS. NO CLUSTER FORM IS CHEAPER THAN EVERY CTA DRAINING ITS OWN at either round, and the split says the
best of them would need a round its shared memory cannot hold. A cluster form pays twice: a round's barriers and
fences, 4,687 to 11,064 cycles a CTA whatever the round holds, and then a token's partial, which by itself costs
about as much through a mate's shared memory as straight to global memory -- 15.4 cycles by a bulk copy into slots
and 17.5 by the warps' stores, eight collectors, their drain included, against the direct 19.2. The integer forms
cost more (36.9 by a bulk reduction, 58.5 by the warps' adds), the warps' fp32 add far more (316), and one
collector up to five times eight (the fp32 add alone the same either way). The hand-over row's token is zero, as it
must be, and its round is the warps' forms' round. So a larger round only brings a cluster form down TO the direct
drain: the bulk-copied slots would pass it at a round near 1,925 tokens (the split is two points on a line, a
round beyond 64 tokens UNMEASURED), whose slots are 3.9 MB of a cluster's shared memory against the 1.8 MB its
eight CTAs have. On this card the output's sum is priced by its partials, a (token, CTA), and summing them in a
cluster first buys nothing. The warps' 64-bit integer add sums exactly under these barriers.

Record: `measure/lab/results/cluster_sum/sm_90a.json`.
