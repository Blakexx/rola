# Precision

> **Every accumulator is fp32 or an exact integer.** Storage and operands are one family on every card: u8 rows,
> bf16 values and operands (`docs/CONSTITUTION.md#r17`).

The distinction is the whole document, so state it first.

| | What it means | Rule |
|---|---|---|
| **ACCUMULATION** | the format a value is *added into* across steps and carried: the fold's state, the mass (the state's v+1 channel), the readout's and the den's sums, the output's sum across CTAs | fp32 or exact integer, closed. Not a default, not a knob, not a card's. |
| **STORAGE and OPERANDS** | the format a value is *held in* and read, without being accumulated into: routing rows, values, the stored state between calls, an MMA's operands, the output | declared per component, each with its quality measured against fp64. |

A read-only value that is 1 ULP wrong contributes an error bounded by its own magnitude, once. An accumulator that is
1 ULP wrong embeds that error and carries it for the state's lifetime: normalization bounds what reads contribute;
nothing bounds what accumulation embeds.

<a id="components"></a>
## The components

| component | storage and operands | accumulation |
|---|---|---|
| the route bundle | every level, either side, dense or sparse: u8 at its row max (255), a zero a zero and no floor under it; an fp32 scale a (token, level). A dense level differs from a sparse one only in having no list | -- |
| the values | bf16 at the token's largest channel = 1, an fp32 scale a token | -- |
| the stored state | s24 a leaf-channel (an s16 hi plane and a u8 lo plane) at an s8 power-of-two exponent a (leaf, 32-channel group); the mass fp32 | -- |
| the fold | the coefficient of a (token, leaf) -- its levels' bytes with the token's scales, multiplied in fp32 where the bytes are expanded -- rounded to bf16 once; the values bf16 as stored | fp32 |
| the readout | the state as a bf16 operand; the read coefficient bf16 | fp32 |
| the intra | a window's token pairs' products exact (u8 rows, integer MMAs); the pair's weight bf16; the values bf16 | fp32 |
| the output's sum | a token's partial sums from every CTA that reads it | fp32, in shared memory and across CTAs |
| decode | the bundle's token, the stored state | each written element updated in fp32 and rounded to its group's unit (2^-23 of the group's largest) |
| the output | `y = P / den`, bf16 | -- |

A u8 byte is exact in bf16 (8 significant bits), so a row costs no rounding as an operand; a scale, a coefficient or a
power of two is multiplied in fp32 where the byte is expanded. Nothing in the family depends on the card: there is no
image of the state at a chosen exponent, no pass that bounds a token's sum before it is formed, and no switch of
accumulation by a card's MMA rates or by a side's density.

WHY NOT 16-BIT FLOAT OPERANDS WITH 5 EXPONENT BITS. The fp16 family (the branch `ampere`'s) kept an unbounded fp32
state and a token's unbounded sum inside f16's range by machinery: the state's image at an exponent a digit, list
exponents, pre-scaled coefficients, a dense row's floor at 2^-13, a pass that bounded each token's sum, and a switch of
accumulation by card. Its last measurement (2026-11-09, sm_86) closed it: with each (token, digit)'s factor on the
read rows before the MMA, the rows under f16's normal floor 2^-14 were 32% at flagship-alt-k4, 71% at alt-k16, 57% at
cohort-k16 and 100% at flagship-dense. bf16's 8 exponent bits hold every one of those values with no machinery, at 8
significant bits where f16 has 11.

<a id="quality"></a>
## Quality

Per token, `|dy| / |y|` against fp64 on continuous values (`docs/testing.md`).

MEASURED on kernels (the October tables, 2026-10-03 and 2026-10-12, the carry's own output on their draws, median a
token): a bf16 family 0.16-0.20% on the flagship cells; the fp16 family 0.018-0.029%. The class is accepted on every
card (Blake, 2026-11-09): it is the bf16 output store's own.

MODELLED for this family before any kernel of it exists ([`tools/arith_model.py`](../internals/tools/arith_model.md),
2026-11-09: the fp64 recurrence on the formatted operands, rounded only where a kernel of the family rounds; the
median / the 99th percentile a token): flagship-alt-k4 0.134 / 0.202%, flagship-dense 0.132 / 0.191%, nl64k-alt-k4
0.115 / 0.174%, nl64k-dense 0.021 / 0.167%. The intra's weight, rounded once a token pair, carries the worst tokens
everywhere; the fold's coefficient, the state as an operand and the read coefficient average over the leaves a token
reads, and each alone costs under 0.02% at the 99th percentile on the dense cells and at most 0.10% on the alt-k4
ones.

THE SUM ACROSS CTAs is fp32 because an f16 one costs by the CTAs it joins: 0.012% x sqrt(CTAs), about 0.14% at the 128
CTAs a 64K call reduces over (MEASURED, sm_86), against ~1e-4% for fp32.

<a id="why"></a>
## Why accumulation stays fp32

A narrowed resident state is refused because the error it embeds walks. Measured on a bf16 state: the re-quantization
walk is `sqrt(T)`-exact (1.06e-2 at T = 512 against a 1e-2 budget, ~1.8e-1 extrapolated to N = 65K), composing in
quadrature (log-log slope +0.490 against theory's +0.5, 150 paired runs); decay does not flatten it, since decay
shrinks the signal and the old noise together while fresh rounding arrives at the current magnitude.

Three arguments make the rule permanent, because they are about what a measurement can see:

1. **The governing write count is a lifetime, not a launch.** A store-back at higher precision preserves an embedded
   error; the state carries it across calls, and decode compounds it.
2. **Nothing bounds a burst.** A burst of small writes re-quantizes repeatedly without decaying.
3. **The oracle cannot see it.** The reference is per call; cross-call compounding is invisible to the gate that
   would have to catch it.

The same rule holds outside the state: decay dials accumulate in fp32, because `1 - rate` near 1 cancels
catastrophically inside a narrow product (`rola/routing/`).

<a id="gates"></a>
## What a precision change owes

- **The gate is written before the change**, with a length-scaling arm (`T` in {512, 2048, 8192}) and a halt on any
  monotone trend: a gate without it cannot see a `sqrt(T)` accumulation at all.
- **The gate is shown capable of failing.** An output-charged assert whose maximum sits at token 0, where the state has
  been re-quantized zero times, cannot detect a resident-state walk of any size.
