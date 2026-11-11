# `byte_expand`

Mirrors `measure/lab/experiments/byte_expand.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a warp take to expand a tile of u8 bytes into bf16 times each token's fp32 scale, with nothing else in its
loop -- no ring, no barrier?

THE FORM. 1, 4 or 8 warps a CTA, a CTA an SM. Each warp expands its own 1 KB of bytes (64 tokens × 16 K, in shared
memory) into its own 2 KB operand tile, over and over: [`row_ring`](row_ring.md)'s expansion -- a lane's two tokens, a
byte to `2^23 + byte` by one `PRMT`, the product by one `FFMA`, a pair to a bf16 word by one conversion. A lane stores
its bytes from registers every round and loads them back (ptxas forwards one of the two loads). The last tile is
checked against the host's. FOUR MORE FORMS at 4 and 8 warps each drop one instruction class, to find which one holds
the SM's rate: the pair packed by truncation (a `PRMT` in place of the conversion), no product, no store and reload of
the bytes, no store of the tile (ptxas then forwards both loads as well) -- UNMEASURED, owed to their first run.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60165936, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the last tile exact against the host's; cycles a 1,024-element tile a warp, the median, no spread):

| warps a CTA | cycles a tile a warp | cycles a tile an SM |
|---|---|---|
| 1 | 202.1 | 202 |
| 4 | 224.1 | 56 |
| 8 | 448.1 | 56 |

WHAT IT SHOWS. A warp alone expands a tile in 202 cycles; four warps an SM each take 224 and eight each take twice
that, so this form's SM expands a tile every 56 cycles however many warps do it past four -- 57% of the rate the
tensor pipe takes one ([`wgmma_rate`](wgmma_rate.md): 32). WHAT SATURATES IS NOT KNOWN: the loop is about 90
instructions a warp a tile, under half of what four schedulers issue in those cycles, so it is not the SM's issue;
the forms that each drop one class are there to find it, and until they run 56 is this form's reading, not the
expansion's floor.

Record: `measure/lab/results/byte_expand/sm_90a.json`; its counters are the written exception (no profiler on that
box).
