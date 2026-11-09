# `state_publish`

Mirrors `measure/lab/experiments/state_publish.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The composed owner kernel's hand-off each window: the fold's warps own 8 R-lists x 16 S-digits (128 fp32 registers),
the readout's 2 S-digits x 64 R-leaves (64 f16 registers), so the publish transposes 1024 leaves x 32 channels across
warps: f16 stores at `[S][R][channel]` (swizzled; a padded variant gives each S-digit's block 16 more bytes), a barrier
or an mbarrier, each warp's slice pulled by `ldmatrix.x4.trans`. sm_86: the publish alone 1572 cycles a window,
padded or not -- the padding removes the stores' 2-way bank conflicts (ncu) and changes nothing, since 64 KB written and
64 KB read (~1K cycles at 128 bytes a clock) plus two syncs bind it. In windows of fold and readout MMAs (a dense window
32 fold steps and 64 read tiles a warp, a sparse one 8 and 16), the block's span a window:

| window | dense | sparse |
|---|---|---|
| fold + readout | 98.1K | 24.3K |
| + publish (barrier) | 99.7K | 25.7K |
| + boundary work (barriers) | 100.0K | 26.2K |
| + boundary, then publish | 102.1K | 28.1K |
| + boundary interleaved with publish | 101.9K | 28.0K |
| + publish by mbarrier over the read start-up (padded) | 101.0K | 27.2K |

After barriers (the stand-in lists phase), the warps reach the publish together and its whole cost shows; behind an
mbarrier, with the readout's start-up between a warp's stores and its pull, about half hides (+1.0K dense, ~1%; +0.95K
sparse, ~4%). Registers peak at 222-224 (the fold state and the pulled slice), no spills. Records:
`measure/lab/results/state_publish/sm_86.json`, `sm_86.counters.json`.
