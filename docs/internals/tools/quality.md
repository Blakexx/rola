# `tools/quality.py` — the output's error against fp64, by accumulation

The constitution's R17 declares each component's accumulation precision, and [precision.md](../../architecture/precision.md)
records it; this instrument is how a choice of accumulation is priced in the output, not in a model. It reads a
stateless prefill's per-token error against the fp64 recurrence with each MMA leg -- the carry's readout and the
intra -- at fp32 or fp16 (`rola.ops.prefill`'s `accumulate` and `intra_accumulate`); a dense read side's readout
accumulates fp32 alone and the carry refuses an fp16 request there, so only the intra turns there.

    bazel run //tools:quality -- flagship-alt-k4 [--json quality.json]

`declare.py` runs it on the carry cells sized for both the oracle and a measurement (tier `both`), stored at
`rola/quality`. It holds the GPU shared: it is correctness work, not a timing.

What it measured, and what that decided, is on [precision.md](../../architecture/precision.md#quality).
