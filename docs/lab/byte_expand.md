# `byte_expand`

Mirrors `measure/lab/experiments/byte_expand.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a warp take to expand a tile of u8 bytes into bf16 times each token's fp32 scale, with nothing else in its
loop -- no ring, no barrier?

THE FORM. 1, 4 or 8 warps a CTA, a CTA an SM. Each warp expands its own 1 KB of bytes (64 tokens × 16 K, in shared
memory) into its own 2 KB operand tile, over and over: [`row_ring`](row_ring.md)'s expansion -- a lane's two tokens, a
byte to `2^23 + byte` by one `PRMT`, the product by one `FFMA`, a pair to a bf16 word by one conversion. A lane stores
its bytes from registers every round and loads them back (ptxas forwards one of the two loads). The checked forms'
last tile is the host's. Four forms each drop a class: the pair packed by truncation (a `PRMT` in place of the
conversion), no product, no store and reload of the bytes, no store of the tile (ptxas then forwards the remaining
load too, so that form drops the tile's stores AND the load). One more form changes only which tokens a lane takes
-- `lane` and `lane + 32` in place of two neighbours -- so a warp's 16-byte accesses fall 16 bytes apart, on all
eight bank groups.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60171071, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the checked forms exact against the host's; cycles a 1,024-element tile a warp, the median, no spread):

| the form | 128-bit shared stores a lane a tile | 1 warp | 4 warps | 8 warps |
|---|---|---|---|---|
| the expansion, a lane's tokens neighbours | 6 | 210.1 | 224.1 | 448.1 |
| ... the pair packed by truncation | 6 | | 224.1 | 448.1 |
| ... no product | 6 | | 224.1 | 448.1 |
| ... no store and reload of the bytes | 4 | | 128.1 | 256.1 |
| ... no store of the tile | 2 | | 64.1 | 128.1 |
| THE EXPANSION, A WARP'S LANES ON EIGHT BANK GROUPS | 6 | 67.1 | 99.1 | 192.1 |

WHAT IT SHOWS.
- The arithmetic is not what the expansion costs: dropping the conversion or the product moves no row by a tenth of a
  cycle. With a lane's tokens neighbours the rows follow the shared-memory accesses alone -- at four warps 32 cycles a
  warp for each of 7, 4 and 2 (the stores, pinned by the SASS, and a load ptxas leaves in the first three forms;
  the loads' count is read from the disassembly, not gated).
- THE LAYOUT IS THE COST. The same instructions with a warp's 32 accesses 16 bytes apart -- four lanes a bank group
  where neighbours put eight on each of four -- run a lone warp's tile in 67 cycles against 210, and an SM's in 24 to
  25 (99 a warp at four, 192 at eight) against 56.
- AN SM EXPANDS A TILE EVERY 24 CYCLES, under the 32 its tensor pipe takes one ([`wgmma_rate`](wgmma_rate.md)), and
  no faster past four warps. What holds it at 24 is not separated; the accesses' wavefronts are the candidate.

Record: `measure/lab/results/byte_expand/sm_90a.json`; its counters are the written exception (no profiler on that
box).
