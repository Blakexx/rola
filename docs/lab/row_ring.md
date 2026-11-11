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

UNMEASURED: no hopper card has run it. A producer's and the consumer's cycles a tile, alone and together at 1, 2 and 4
producers, are owed to its first run (`bazel run //tools:lab -- pack row_ring --arch sm_90a`, then `carried` on the
card, under its kill timeout).
