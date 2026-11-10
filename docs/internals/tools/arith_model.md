# `tools/arith_model.py` — the bf16 family's arithmetic, modelled

The op's accuracy class is a format decision ([precision.md](../../architecture/precision.md#quality) records it);
this model prices the family before any kernel of it exists: rows u8 at every level, values and MMA operands bf16,
every sum fp32. Its truth is the fp64 oracle's -- summed a window at a time
so a 64K cell fits, and checked against `rola.ops.naive.naive_rola` over the first two windows on every run -- and it
rounds only where such a kernel would (the tool's docstring lists the four places), each alone and all together.

    bazel run //tools:arith_model -- flagship-alt-k4 nl64k-dense [--window 1024] [--json model.json]

MEASURED (2026-11-09, one head, the median / the 99th percentile a token):

| cell (window) | all | fold alone | state as operand alone | read coefficient alone | intra alone |
|---|---|---|---|---|---|
| flagship-alt-k4 (512) | 0.134% / 0.202% | 0.015% / 0.066% | 0.015% / 0.066% | 0.015% / 0.062% | 0.114% / 0.202% |
| flagship-dense (512) | 0.132% / 0.191% | 0.001% / 0.003% | 0.000% / 0.002% | 0.001% / 0.004% | 0.132% / 0.191% |
| nl64k-alt-k4 (1,024) | 0.115% / 0.174% | 0.067% / 0.098% | 0.066% / 0.097% | 0.057% / 0.081% | 0.021% / 0.162% |
| nl64k-dense (1,024) | 0.021% / 0.167% | 0.003% / 0.003% | 0.001% / 0.015% | 0.002% / 0.003% | 0.020% / 0.167% |

WHAT IT SHOWS. A rounding of a (token, leaf) coefficient or of a state value averages over the leaves a token reads,
so the fold, the state and the read coefficient cost little where a token reads many leaves (dense) and more where
it reads few (alt-k4). The intra rounds once a token pair, and the cells' values are random-signed, so a token's
output is a small remainder of its terms (about 7% of their size at flagship-dense) while the roundings do not
cancel: the intra carries the error on a two-window cell and the worst tokens everywhere (a window's early tokens,
whose output is a few intra terms). With no rounding on the model is the truth exactly.
