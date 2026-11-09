# `publish_rounds`

Mirrors `measure/lab/experiments/publish_rounds.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The even carve's publish: the fold's warps own 4 level-R digits x 32 level-S x 32 channels (fp32), the readout's 4
level-S x 32 level-R x 32 channels (f16 B fragments), a transpose of 32K values a window through `[S][R][channel]` (an
S block 16 bytes past 2 KB, channel chunks swizzled by `(R >> 1) & 3`), whole (64 KB) or in four rounds of 16 KB (a
level-S quarter, two readout warps' slices each) under mbarriers. In windows of fold steps (32 fp32-accumulate MMAs a
step) and readout tiles (32 fp16-accumulate MMAs a tile), the block's span a window, sm_86:

| publish | dense window | sparse window |
|---|---|---|
| none | 96.6K | 24.4K |
| whole, barriers (64 KB) | 99.5K | 25.8K |
| whole, mbarrier (64 KB) | 99.5K | 26.0K |
| four rounds, one 16 KB buffer | 100.6K | 26.8K |
| four rounds, two buffers (32 KB) | 100.0K | 26.3K |

Rounds double-buffered cost 0.4-0.6K a window over the whole publish, single-buffered 1.0-1.1K: the rounds'
handshakes, not their bytes. Record: `measure/lab/results/publish_rounds/sm_86.json`.
