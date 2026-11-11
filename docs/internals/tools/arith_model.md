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

## <a id="sums"></a>The cross-box sum

The readout's output is a sum over boxes that different CTAs hold. The model sums each box's fp32 partial three ways,
with nothing else rounded: box after box in fp32; as integers of 32 or 64 bits at a scale `2^q` taken from a bound
(the tool's docstring lists the bounds: one a call, two a window, two a cluster and window); and NORMALIZED -- each
box's num over its cluster's den for the token, a share of a weighted average of values, as integers at the call's
largest value scale, turned back and multiplied by the cluster's den. A box is `--box FIRST SECOND` digits a level; a cluster
is the boxes sharing one range of level `--along`, summed as integers, the clusters then in fp32. Three stresses move
what a scale must hold: `--scale-spread` (write scales rising over the call), `--value-spread` (value scales rising),
`--leaf-spread` (write scales apart across one cluster's leaves).

    bazel run //tools:arith_model -- nl64k-alt-k4 --along 1 [--scale-spread 40 | --value-spread 16 | --leaf-spread 24]

MEASURED (2026-11-11, one head, boxes 16 x 16, clusters along the read-sparse level; the sum's own error, the worst
token; the fp32 sum reads 4e-05% to 7e-05% in every column):

| the integer sum | nl64k-alt-k4 | ... write scales x 2^40 | ... value scales x 2^16 | corner-tied-k4, leaves 2^24 apart |
|---|---|---|---|---|
| 32 bits, one scale a call | 99% | 161% | 110% | |
| 32 bits, a window's coarse bound (largest token x tokens so far) | 0.28% | 1.3% | 0.85% | |
| 32 bits, the cluster's state a window | 0.0001% | 0.0001% | 0.0001% | 2.0% |
| 32 bits, the cluster's planned bound a window | 0.007% | 0.002% | 0.003% | 4.2% |
| 32 bits, normalized in the cluster | 0.0002% | 0.00006% | 2.3% | 0.00001% |
| 64 bits, one scale a call | 0.000008% | 107% | 0.0002% | |
| 64 bits, a window's coarse bound | 0.000008% | 0.000008% | 0.000008% | 0.00001% |
| 64 bits, normalized in the cluster | 0.00002% | 0.00002% | 0.00003% | 0.00001% |

WHAT IT SHOWS. On the cells as drawn every form but a 32-bit call-wide or coarse scale adds less than the fp32 sum's
own error. Each 32-bit form has one stress that takes it out of the family's 0.2%: a scale from the cluster's state
fails where a cluster's leaves hold magnitudes far apart and a token reads only the small ones; the normalized sum
fails where value scales spread over the call. The 64-bit sums at a window's coarse bound and normalized in the
cluster hold in every column measured. A token of nl64k-alt-k4 reads 3.66 of its clusters.
