# `tools/quality.py` — the output's error against fp64, by accumulation

The constitution's R17 declares each component's accumulation precision, and [precision.md](../../architecture/precision.md)
records it; this instrument is how a choice of accumulation is priced in the output, not in a model. It reads a
stateless prefill's per-token error against the fp64 recurrence with each MMA leg -- the carry's readout and the
intra -- at fp32 or fp16 (`rola.ops.prefill`'s `accumulate` and `intra_accumulate`); a dense read side's readout runs
fp32 whatever it is told, so only the intra turns there.

    bazel run //tools:quality -- flagship-alt-k4 [--json quality.json]

`declare.py` runs it on the carry cells sized for both the oracle and a measurement (tier `both`), stored at
`rola/quality`. It holds the GPU shared: it is correctness work, not a timing.

MEASURED (sm_86, the 3080 Ti, 2026-11-09), the median a token -- both fp32 / the carry fp16 / the intra fp16:
flagship-alt-k4 0.032% / 0.032% / 0.057%; flagship-alt-k16 0.044% / 0.045% / 0.087%; flagship-cohort-k16 0.045% /
0.045% / 0.088% (all listed read sides); flagship-dense 0.054% / the same (its readout fp32) / 0.111%. The listed
readout's fp16 accumulation costs nothing measurable at any of the three (p99 within 0.003%); the intra's doubles the
error. A one-window cell (flat-small-alt-k4) says nothing of the carry: its first window reads no state, so its
output is the intra's.
