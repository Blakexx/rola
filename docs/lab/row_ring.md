# `row_ring`

Mirrors `measure/lab/experiments/row_ring.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Producer warps expand u8 rows, landed in shared memory by a bulk copy, into a ring of wgmma B operands -- bf16, each
row times its fp32 scale. How many cycles a tile does a producer take against the warpgroup consuming it, and how many
producers keep one warpgroup fed?

THE FORM. A CTA an SM: warps 0 to 3 are a consuming warpgroup (one `m64n64k16` wgmma a tile, A in registers), then
1, 2 or 4 producer warps over a ring of 4 operand slots and 8 slots of bytes. A tile is 64 tokens × 16 K: 1 KB of
bytes, 2 KB as the operand. A producer's tile: its bytes arrive by `cp.async.bulk` under an mbarrier, issued before
the tile ahead of it is expanded, so two loads are in flight; each lane expands two tokens and issues the writers'
proxy fence; a full and an empty barrier a slot hand the tile over and back. Each side also runs alone, the other
doing nothing but the barriers' arrivals (the consumer alone has no loads and no expansion beside it). The scales are powers of two, so every product is exact in bf16 and the readout's accumulators are
compared exactly with the host's.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): the expansion is off the conversion unit. A byte becomes
the fp32 `2^23 + byte` by one `PRMT` against a constant, the product by one `FFMA` (times the scale, less `2^23` scales,
rounded once), and a pair becomes a bf16 word by one `F2FP.BF16.F32.PACK_AB`: a lane's 32 elements are 2 `LDS.128`, 32
`PRMT`, 32 `FFMA`, 16 packs and 4 `STS.128` -- one FFMA, one PRMT and half a pack an element. A bulk load is one
`UBLKCP.S.G`.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60165274, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the accumulators exact against the host's at every producer count; cycles a tile -- 1,024 elements -- the
median, the spread under 0.3%):

| | cycles a tile |
|---|---|
| the producer alone, 1 warp (the consumer only arrives) | 773.1 |
| the consumer alone (the producer only arrives) | 289.7 |
| together, 1 producer warp | 760.6 |
| together, 2 producer warps | 390.9 |
| together, 4 producer warps | 303.7 |

WHAT IT SHOWS. One producer warp makes a tile in about 770 cycles, 0.75 a warp an element, and the ring runs at the
producers' rate up to two warps (760, then 391, half of it). At four it runs at 304, the consumer's own 290: a
warpgroup under this hand-over takes a tile in 290 cycles, where one wgmma with its group's wait is 96 and the
pipe's own time 32 ([`wgmma_rate`](wgmma_rate.md)). What the other 190 cycles are -- the full and empty barriers'
polled waits, the descriptor formed a tile, the rest of the loop -- no row here separates.

BESIDE IT: [`byte_expand`](byte_expand.md) reads the expansion alone at 202 cycles a tile a warp, in a loop that
differs from this producer's (it stores its bytes itself, where these arrive by a bulk copy). So most of a producer's
773 cycles here is not the expansion; what it is, among the barrier waits, the warp's synchronization and the loads'
issue, no row separates.

Record: `measure/lab/results/row_ring/sm_90a.json`; its counters are the written exception (no profiler on that box).
