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

WHY IT IS MODELLED. The readout's output is a sum over boxes that different CTAs hold, and
[precision.md](../../architecture/precision.md) states that sum in fp32. A float cannot be summed between CTAs before
it reaches global memory -- the instruction set's reductions into another CTA's shared memory are integer -- so every
box's partial would cross to global memory by itself. An integer partial can be summed in a cluster first. The model
asks what an integer sum costs in accuracy, under each way of choosing its scale, before any kernel takes one: each
box's fp32 partial summed in fp32, as 32- or 64-bit integers at a scale from a bound, or normalized by its cluster's
den (the tool's docstring states each form). A box is `--box FIRST SECOND` digits a level; a cluster is the boxes
sharing one range of level `--along`. Four stresses move what a scale must hold: `--scale-spread` (write scales rising
over the call), `--value-spread` (value scales rising), `--leaf-spread` (write scales apart across one cluster's
leaves) and `--spike` (the call's first 16 tokens far over the rest).

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
| 64 bits, the cluster's state a window | 0.00002% | 0.00002% | 0.00002% | 0.00001% |
| 64 bits, normalized in the cluster | 0.00002% | 0.00002% | 0.00003% | 0.00001% |

THE SPIKE (the first 16 tokens at 2^B over the rest; the 64-bit forms, the worst token):

| B, cell | a window's coarse bound | the cluster's state a window | normalized in the cluster |
|---|---|---|---|
| 40, corner-tied-k4 | 0.21% | 0.0002% | 0.000007% |
| 48, corner-tied-k4 | 27% | 0.045% | 0.000007% |
| 40, flagship-both-k4-L4096 | 1.9% | 0.0003% | 0.00003% |
| 48, flagship-both-k4-L4096 | 127% | 0.13% | 0.00003% |
| 40, flagship-alt-k4-L4096 | 0.000008% | | 0.00002% |

The 64-bit normalized sum under value scales rising 2^32, 2^40 and 2^48 (flagship-alt-k4-L4096): 0.00002%, 0.0002%,
0.03%.

WHAT IT SHOWS. On nl64k-alt-k4 as drawn the 64-bit sums add less than the fp32 sum's own error, and the 32-bit sums at
a cluster's scale or normalized add one to three times it -- all far inside the family's 0.2%; a 32-bit call-wide or
coarse scale is outside it. Every 32-bit form has a stress that takes it past 0.2%. Among the 64-bit forms the coarse
bound gives out at a spike near 2^40 where later tokens' reads miss the spike's leaves (none in the alternating cell,
whose every reader meets every writer); the cluster's state scale holds to a spike or a leaf spread near 2^48; the
normalized sum holds everywhere but a value-scale spread near 2^48. A token of nl64k-alt-k4 reads 3.66 of its
clusters.
