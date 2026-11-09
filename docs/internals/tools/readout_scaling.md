# `readout_scaling.py`

Mirrors [`tools/readout_scaling.py`](../../../tools/readout_scaling.py).

THE READOUT'S TWO SCALINGS against the fp64 oracle on a cell's real state: how a box's readout from the f16 image
loses accuracy under each way of applying the read coefficients, for the value-carve readout's dense side.

<a id="forms"></a>
## The two forms

Both read the IMAGE: each digit of each box at its own power of two, `f16(S 2^(10 - e_d))`, `e_d` the digit's
largest |state| over the box's lists and channels rounded up (the readout reference's placement,
[`tests/oracle/readout_reference.py`](../../../tests/oracle/readout_reference.py)), and the read rows `a` as stored
(f16).

- OUTPUT SCALING: each digit's sum over the box's lists in fp32 (the MMA's products exact, its sum fp32), times
  `x 2^(e_d - 10)` in fp32, the digits summed in fp32.
- OPERAND SCALING: each token's exponent `e_t` the largest of `e_d + ceil(log2 x)` over the box's digits it reads;
  each row scaled `y = f16(a x 2^(e_d - e_t))` (one rounding a product, f16's subnormals kept); ONE fp32 sum over every
  leaf of the box, times `2^(e_t - 10)`. `e_t` places a token's largest digit near 1, so f16's range costs only the
  digits far under it.

<a id="output"></a>
## What it prints

The state comes from the fold reference (`fold_state`), the coefficients and rows from the bundle
(`readout_reference._read_rows`), level S the readout's digits. Per slot, the error over its magnitude
`sum |x| |a| |S|` (the readout reference's): the maximum, the 99.9th and 50th percentiles, for each form; and the
share of operand products under f16's normal range and flushed to zero. A box's in-CTA sum only -- the cross-CTA sum
rounds the same in both. `--windows` prices a subset of a long cell's windows (the state is folded through all of
them); `--box` sets the box (lists along level R, digits along level S, channels).
