# `intra_stream`

Mirrors `measure/lab/experiments/intra_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION: the fp16 intra ([`docs/internals/intra/ampere/intra_kernel.md`](../internals/intra/ampere/intra_kernel.md)) moves ~5
bytes a cycle an SM, ~6.8 with its Gram skipped, where `cp.async` streams 18 ([`smem_rates`](smem_rates.md)). What does
its stage ring deliver an SM beside a Gram, and which way of issuing the refills keeps the copies under the MMAs?

THE FORM: one CTA an SM (99 KB of dynamic shared memory), 600 stages over L2-resident rows (4 MB: 4,096 tokens of 1
KB, the dense plane's token stride), a stage the 64 rows' 128-byte slice (the intra's 8 KB stage), a ring of 3, 6 or 10
buffers. Eight compute warps each await a stage, run `SLABS` slabs of the dense Gram on it -- a slab two k-steps, an
`ldmatrix.x4` of a static A and two of the stage's B a k-step, 8 fp16-accumulate MMAs a slab (the 8-warp walk's 4
n-tiles a warp) -- and release it. The issue forms: the LAST RELEASER of a stage refills its buffer a ring on (the
intra's first form); EVERY WARP, at a stage's start, awaits the previous stage's `empty` mbarrier and copies its share of
the stage two on; a PRODUCER warp, a ninth, copies every stage on each buffer's `empty`; the same every-warp ring
STREAMED, a warp's share a piece after each slab (awaiting the buffer's `empty` at the first piece); and the SM80
MULTISTAGE ring (CUTLASS's mainloop, the production intra's since 9d12ff39): a commit group a stage, each thread
awaiting its own copies (`cp.async.wait_group`) and one `bar.sync` a stage, the refill at the stage's start (the
intra's), whole after the stage's MMAs, or a piece after each slab. Controls: the Gram alone on resident stages (no
copies), the same Gram while every warp streams its share into spare buffers after each slab with nothing waiting on
the copies, and a stage's rows one contiguous 8 KB run. A sample is the CTA's span over the bytes moved and over
the MMAs issued (cycles an MMA a scheduler), the mean over SMs.

THE RECORD: `measure/lab/results/intra_stream/sm_86.json` (RTX 3080 Ti). The A100's (SXM4 40 GB, a rented host whose
counters are closed, so the driver's ncu pass cannot run there) is the same binary at three buffers run directly,
2026-11-01:

| form, 3 buffers | 0 slabs, B/cycle | 2 slabs (dense f16): B/cycle, cycles/MMA | 4 slabs (u8): B/cycle, cycles/MMA |
|---|---|---|---|
| the Gram alone (no copies) | -- | --, **16.9** | --, 17.0 |
| last releaser refills | 10.3 (A100 10.3) | 6.4, 40.2 (A100 6.3, 40.4) | 4.9, 26.2 (A100 5.0, 25.7) |
| every warp a share | 13.6 (13.8) | 6.9, 37.1 (7.8, 33.0) | 4.8, 26.5 (5.8, 22.3) |
| a producer warp | 10.7 (10.9) | 7.9, 32.3 (7.8, 32.6) | 6.8, 18.9 (6.6, 19.5) |
| the producer, contiguous rows | 10.4 | 7.9, 32.2 | -- |

| a producer warp, by ring depth | 0 slabs, B/cycle | 2 slabs: B/cycle, cycles/MMA | 4 slabs |
|---|---|---|---|
| 3 buffers (24 KB) | 10.7 | 7.9, 32.3 | 6.8, 18.9 |
| 6 buffers (48 KB) | 10.8 | 7.9, 32.2 | -- |
| 10 buffers (80 KB) | 10.6 | 8.0, 32.0 | 6.7, 19.0 |

2026-11-05, sm_86, the forms that put the copies under the slabs (B/cycle, cycles an MMA a scheduler):

| form | 2 slabs (dense f16) | 4 slabs (u8) |
|---|---|---|
| every warp streams into spare buffers, nothing waits on the copies (10 buffers) | **13.0, 19.7** | 7.2, 17.8 |
| the every-warp ring streamed, `kB - 1` ahead: 3 / 6 / 10 buffers | 7.5, 34.1 / 7.8, 33.0 / 9.5, 26.9 | 5.5, 23.3 (3) |
| the every-warp ring streamed, 2 ahead: 4 / 6 / 10 buffers | 9.0, 28.5 / 7.6, 33.7 / 9.3, 27.5 | 5.6, 22.8 (6) |
| every warp a share at the stage's start: 6 / 10 buffers | 6.9, 37.0 / 7.0, 36.8 | -- |
| multistage, the refill at the stage's start (the intra's), 3 buffers | 9.2, 28.0 | 5.5, 23.4 |
| multistage, the refill whole after the stage's MMAs, 3 buffers | 10.7, 24.0 | 5.6, 22.9 |
| multistage, the refill a piece after each slab: 3 / 4 / 6 buffers | **10.8, 23.8** / 10.9, 23.6 / 11.1, 23.1 | **6.4, 20.0** (3) / 6.1, 21.0 (6) |

2026-11-06, the A100 (SXM4 40 GB, rented; counters closed, the rows the driver prints before its ncu pass), 5
repetitions:

| form | 2 slabs (dense f16) | 4 slabs (u8) |
|---|---|---|
| the Gram alone (no copies) | --, 13.0 | --, 12.1 |
| every warp streams into spare buffers, nothing waits on the copies (10 buffers) | 16.6, 15.4 | 8.3, 15.4 |
| a producer warp, 3 buffers | 7.8, 32.6 | -- |
| multistage, the refill at the stage's start (the intra's), 3 buffers | 12.3, 20.9 | 7.1, 18.1 |
| multistage, the refill whole after the stage's MMAs, 3 buffers | 12.6, 20.3 | 7.0, 18.3 |
| multistage, the refill a piece after each slab: 3 / 6 buffers | **13.4, 19.1** / 13.6, 18.8 | **8.0, 16.0** (3) / 7.8, 16.4 (6) |

WHAT IT SETTLES: the Gram's code runs at the tensor rate (16.9 cycles an MMA against sm_86's 16), the token stride costs
nothing, and the bytes in flight do not bind: a ring of 3, 6 or 10 stages delivers the same rate. The ~8 bytes a cycle
the first forms reached beside a Gram is the forms', not the SM's (corrected 2026-11-05): with nothing waiting on the
copies, every warp streaming its share under its slabs, the SM copies 13 bytes a cycle while the dense Gram runs at
19.7 cycles an MMA. Waiting on the copies is what costs: the mbarrier rings, streamed or not, stay at 7.5-9.5 bytes a
cycle; the multistage ring's one `bar.sync` a stage reaches 10.8. Within it, WHERE the refill is issued is the lever:
at the stage's start (before the Gram, every warp's copies at once) 28.0 cycles an MMA, after the Gram's MMAs 24.0, a
piece after each slab 23.8 -- and for the u8 stage, four slabs, only the pieces help (23.4, 22.9, 20.0). In the
kernel the refill in halves under the slabs measured a percent ahead of it whole after the MMAs on every cell, so the
production ring issues it in halves (see [`intra_kernel.md#pipeline`](../internals/intra/ampere/intra_kernel.md#pipeline)). The dense f16 stage feeds a 64-reader block
128 MMAs an 8 KB, ~16 bytes a cycle at the tensor rate: fewer bytes an MMA -- more readers a writer byte, or narrower
rows -- is what takes the rest; not a deeper ring (above) nor the readers' A fragments in registers to make room for
one (built 2026-11-01 and slower: 683 against 505 us at `nl64k-dense`, spilling at depth two).
