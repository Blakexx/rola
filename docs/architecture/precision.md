# Precision

> **Every accumulator that carries state is fp32 or an exact integer.** Storage and operand precision are declared
> per component, one fp16 family (`docs/CONSTITUTION.md#r17`).

The distinction is the whole document, so state it first.

| | What it means | Rule |
|---|---|---|
| **ACCUMULATION** | the format a value is *added into* across steps and carried: the fold's state, the mass (the state's v+1 channel), the scale pass's write masses | fp32 or exact integer, closed. Not a default, not a knob. |
| **STORAGE and OPERANDS** | the format a value is *held in* and read, without being accumulated into: routing rows, values, the stored state between calls, the readout's image, the output | declared per component, each with its quality measured against fp64. |

A read-only value that is 1 ULP wrong contributes an error bounded by its own magnitude, once. An accumulator that is
1 ULP wrong embeds that error and carries it for the state's lifetime: normalization bounds what reads contribute;
nothing bounds what accumulation embeds.

<a id="components"></a>
## The components

| component | storage and operands | accumulation |
|---|---|---|
| the route bundle | a dense level f16 at its row max, floored at 2^-13 of it; a sparse level u8 at its row max, exact zeros; an fp32 scale a (token, level) | -- |
| the values | f16 at the token's largest channel = 1, an fp32 scale a token | -- |
| the stored state | s24 a leaf-channel (an s16 hi plane and a u8 lo plane) at an s8 power-of-two exponent a (leaf, 32-channel group); the mass fp32 | -- |
| the fold | A the slots' dense slices f16 times c (f16 at `2^(E - unit)`), B the values f16 | fp32 on every card, each list in its running unit |
| the readout | the state's f16 image at a power of two a digit, the digit's largest value under 2^10; read rows f16; the coefficient f16 | a listed read side: fp32 where fp32 accumulate is full rate (A100, A6000, A40), fp16 on GeForce, chosen per product from `hardware.json`; a dense read side: fp32 on every card today, a departure from this rule on GeForce (its one accumulator over all 512 leaves of a token batch can reach 2^19, past the 2^15 bound the image's 2^10 top gives a listed tile's 32 products; [`carry/ampere/readout_dense.md`](../internals/carry/ampere/readout_dense.md)), owed: fp16 sums kept inside that bound; each token's num into the panel as f16 pairs at its scale byte (or fp32, the declared `reduce_dtype`), added across CTAs by `red.global.add.noftz.f16x2` (or `.f32`; [`carry/ampere/panel.md#drain`](../internals/carry/ampere/panel.md#drain)) |
| the scale pass | the rows as stored (a u8 digit over 255, a dense f16), `F` | fp32, the byte by the carry's rule |
| decode | the bundle's token, the stored state | each written element updated in fp32 and rounded to its group's unit (2^-23 of the group's largest) |
| the output | `y = P x unit / den`, bf16 | -- |

The image's 2^10 top keeps a 32-list sum of image times read row under 2^15, inside an fp16 accumulator, so one
placement serves both readout accumulations (`docs/internals/carry/ampere/handoff.md#maxima`). The fp16 readout accumulation is
not a state accumulator: it sums one tile's 32 lists, and the panel takes its result at the token's scale.

OPERAND SCALING AT THE TOKEN'S SCALE DOES NOT FIT f16 (measured 2026-11-09 on the readout reference's states, sm_86):
with each (token, digit)'s factor `c 2^(e_d - kImageTop - s - 1)` on the read rows before the MMA, the scaled rows under
f16's normal floor 2^-14 are 32% at flagship-alt-k4, 71% at alt-k16, 57% at cohort-k16 and 100% at flagship-dense
(3.7%, 11%, 7.6% and 38% with the token's byte left off). The scale byte's frame is the scale pass's bound, about 2^9
over the real sums, and the image keeps its 2^10 top, so the whole shift lands on the rows. The listed readout
therefore scales its accumulator, and the dense readout scales its rows at a per-token exponent of its own
([`readout_scaling.py`](../internals/tools/readout_scaling.md) prices that form).

The readout's partials are an output's, not a state's: each token's num into the panel and across CTAs as f16 pairs at each token's scale byte (or fp32, the declared `reduce_dtype`), added across CTAs by `red.global.add.noftz.f16x2` (or `.f32`),
the scale pass's byte bounding the token's whole num so the f16 sum stays in range (Blake, 2026-11-06: the readout's
exits straight to HBM by f16x2 REDs, no integer panel; the dtype declared 2026-10-19,
[`carry_api.md#carry-reduce`](../internals/carry/carry_api.md#carry-reduce)). The constitution's R17 still names them
exact s16 pairs: a departure its text owes.

<a id="quality"></a>
## Quality

Measured per token as `|dy| / |y|` against fp64 on continuous values, median (`docs/testing.md`): the fp16 family
0.018-0.029% on the flagship cells, bf16 0.16-0.20%; fp16 accumulation of the readout 0.018-0.035%. The f16 image at a
power of two a digit matches a block prescale and beats a bf16 image 2.6x at alt-k4 and ~8x under a heavy history.
Those are the October tables' (2026-10-03, 2026-10-12: the carry's own output on their draws). The op's whole output
by each MMA leg's accumulation ([`tools/quality.py`](../internals/tools/quality.md), sm_86, 2026-11-09), the median a
token -- both fp32 / the carry's readout fp16 / the intra fp16: flagship-alt-k4 0.032 / 0.032 / 0.057%,
flagship-alt-k16 0.044 / 0.045 / 0.087%, flagship-cohort-k16 0.045 / 0.045 / 0.088%, flagship-dense 0.054 / (its
readout fp32 alone) / 0.111%. The listed readout's fp16 accumulation costs nothing measurable; the intra's doubles the
error.

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
