# `read_stream`

Mirrors `measure/lab/experiments/read_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the readout's rows are streamed a digit at a time into each warp's ring (merge width 1) instead of gathered from a
staged pool, what does the stream cost beyond the readout, at what ring depth, and in what form?

The step is the readout's tokens-on-N form ([`read_image`](read_image.md)): a tile is 8 tokens, 4 bf16 MMAs against
the digit's state from the image, each output pair exact and added into its token's panel row. A box-window's colored
slots run in passes of 256 (a panel's rows): each pass steps every digit's first tile in one straight-line block, then
each digit's rest, then drains its panel (`red.global.add`) between two CTA barriers. The inputs
(`read_stream.inputs.py`) are three cells' read lists, colored and placed per pass as the facts pass emits them:
nl64k-alt-k4 over 64 windows, flagship-alt-k16-L16384 and flagship-cohort-k16-L16384 over 32. Each digit steps at least
one tile a pass (an empty digit's all pads, as the readout steps every digit's first). The 80 CTAs are boxes 0 .. 79 of
one head, channel tile fastest; a box's read row for a token is its level-0 slice group's 64 bytes, fetched by token
from global memory, so L2 serves what one box fetched to the others. Each CTA's drained panel is checked word for word.

| a CTA-window | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| tiles: mean; the busiest warp's a pass, x 8 | 45.0; 55.8 | 155.8; 180.8 | 175.7; 355.9 |
| passes | 1.00 | 2.00 | 1.98 |

STREAMED: each warp's ring of `D` stages, a stage a tile's 8 read rows and its 8 entries lane-ready (fp32 coefficient,
panel offset: 576 bytes, the rows' 32-byte phase kept); the producer walks its window's token record (each stage's 8
tokens, a pad the tile's first live token), copied a window ahead with the stage group it is issued in, and copies each
stage's rows and entries, one stage a tile, its addresses running pointers. The consumer's tile reads its two tokens'
entries with one 16-byte load (no shuffles, no unpacking), its row address and entry offset held in registers; the
drain is unrolled on held addresses, a request a 128-byte line.

STAGED (the stream's floor form): each warp's ring of 8 stages of 608 bytes -- a tile's 8 read rows, its 8 entries
lane-ready, the 8 tokens of the stage 8 on -- its stages one record stream across windows and passes (a pre-pass's
plan, tokens global), so the producer reads a stage's tokens from the slot it refills, copies its rows and its 96-byte
record (the stage's entries, the tokens 8 on) in one burst, and checks no window. A pad (a tile's slot past its
digit's entries) is not copied: the slot keeps an earlier stage's finite row, which its zero coefficient cancels.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| streamed, ring 4 (the first tiles' block holds the ring, nothing ahead) | 5,357 |  |  |
| streamed, ring 5 | 5,124 | 12,327 | 21,775 |
| streamed, ring 6 | 5,401 |  |  |
| streamed, ring 5, copies allocated in L1 (`.ca`) | 5,119 | 12,353 | 19,871 |
| bound: the readout with the producer, no row copies | 4,801 | 12,115 | 20,172 |
| bound: the copies alone | 2,706 |  |  |
| bound: the readout alone, no producer (its entries from global memory a tile ahead) | 3,759 | 9,463 | 15,395 |
| bound: the same, no drain | 2,788 | 7,640 | 13,496 |
| bound: the stream with no waits on its stages | 5,108 |  |  |
| STAGED, ring 8 (pads not copied) | 4,549 | 11,676 | 18,670 |
| staged, ring 8, pads copied as zeros | 4,564 | 11,731 | 18,811 |
| bound: staged, the readout with the producer, no row copies (pads as zeros) | 4,014 | 10,212 | 16,140 |
| bound: staged, the copies alone (pads as zeros) | 1,906 |  |  |

Instructions a MMA (ncu): the readout alone 16.5 at alt-k4, 14.0 at alt-k16, 13.0 at cohort-k16 (25.2 at alt-k4
before the floor pass), of which the drain 2.8, 1.7, 1.4 (11.0 before: 12 of an iteration's 15 instructions rebuilt
its 64-bit addresses); streamed 25.1, 20.2, 19.9 (34.9 at alt-k4 before). A rest tile with no producer runs 42
instructions for its 4 MMAs: the output path 16 (8 FFMA scaling and converting each output, 4 packs, 4 shared
atomics), the MMAs and the row `ldmatrix` 5, the entries' prefetch 9, panel bases and row 3, slot and loop 6, the
rest 3; at alt-k4 a warp holds ~5.6 tiles a window, so its window's fixed work (4 digits' state loads, the digits'
loops, the passes' barriers, the drain, the record) spreads over ~22 MMAs. Staged: 20.4, 17.7, 17.2 -- its stream
3.9, 3.8, 4.2 a MMA over the readout alone; a rest tile 57 instructions (the producer 20, ~34 streamed: its token,
its row's source in a shift and a carry, the pad test, the two copies, the record pointer, ptxas's 3 fillers; the
consumer 37, the output path 18 of them). Shared-memory wavefronts a MMA: the readout alone 9.5, 7.0, 6.6; streamed
10.5, 8.0, 7.7; staged 10.2, 7.9, 7.4. L2 sectors a MMA: streamed 4.8, 4.7, 5.1; staged 3.6, 4.1, 3.6.

What it settles:

- The stream costs the readout 1.36x at alt-k4, 1.30x at alt-k16 and 1.41x at cohort-k16 (streamed at ring 5 against
  the readout with no producer, drains included): its producer's instructions between tiles of 4 MMAs, not its waits
  (with no waits the stream runs the same, 5,108 against 5,124). Its forms measured at carry-k5 a43afc27 -- the
  consumer loading its entries, the token a stage ahead, two tiles a stage -- do not help.
- Staged, it costs 1.21x, 1.23x and 1.21x: one record stream (no window checks), the record riding in the stage's
  copy and pads not fetched take the producer from ~8.6 instructions a MMA to ~4 and a quarter of the stream's L2
  sectors. The record's running pointer (5 instructions fewer a tile than a multiply-add from the stage's index)
  costs cohort-k16 1.6% (18,670 against 18,371), the cause not pinned.
- The readout's floor at sparse is its output path: each output leaves registers after K = 32 (two MMAs), at 2.5
  instructions a value -- 4 a MMA -- before the stream and the window's fixed work.
- A ring of 5 stages: 4 costs 5%; 6 costs 6%, its records only a window ahead (a window holds 4 stages at least,
  fewer than the 5 groups ring 6 waits on, so its entry drains the ring).
- Allocating the copies in L1 (`.ca`) serves a box's re-fetches from L1 (a token two of its digits read): 32%
  fewer L2 sectors at alt-k4, 42% at alt-k16, 49% at cohort-k16 (ncu). Cohort-k16 runs 8.2% faster -- not by
  bandwidth, its L2 at 34% of peak with `.cg`; the cause is not pinned -- alt-k4 and alt-k16 the same. But a `.ca`
  copy takes 5.4 shared-memory pipe cycles a 128 bytes against `.cg`'s 0.5 ([`smem_rates`](smem_rates.md)), and the
  pipe binds sm_80: `.cg`.

Record: `measure/lab/results/read_stream/sm_86.json`.
