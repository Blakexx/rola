# `tools/quality.py` — the output's error against fp64, by accumulation

The ampere family's kernels, still in this tree, take an accumulation precision a component (the fp16 family's
switch, which [precision.md](../../architecture/precision.md) records as closed); this instrument priced a choice of
accumulation in the output, not in a model. It reads a
stateless prefill's per-token error against the fp64 recurrence with each MMA leg -- the carry's readout and the
intra -- at fp32 or fp16 (`rola.ops.prefill`'s `accumulate` and `intra_accumulate`); a dense read side's readout
accumulates fp32 alone and the carry refuses an fp16 request there, so only the intra turns there.

    bazel run //tools:quality -- flagship-alt-k4 [--json quality.json]

`declare.py` runs it on the carry cells sized for both the oracle and a measurement (tier `both`), stored at
`rola/quality`. It holds the GPU shared: it is correctness work, not a timing.

MEASURED (sm_86, 2026-11-09, the median a token; both legs fp32 / the carry's readout fp16 / the intra fp16):
flagship-alt-k4 0.032 / 0.032 / 0.057%, flagship-alt-k16 0.044 / 0.045 / 0.087%, flagship-cohort-k16 0.045 / 0.045 /
0.088%, flagship-dense 0.054 / (its readout fp32 alone) / 0.111%. The listed readout's fp16 accumulation cost nothing
measurable; the intra's doubled the error.
