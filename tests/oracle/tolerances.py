"""THE NUMERIC BAND, stated ONCE for every tier that grades against the oracle.

The comparison basis is fixed by the arms, not by a file: the fp64 oracle
(`rola.ops.naive.naive_rola`) is the reference, the kernel arm stores bf16, and
the derived budget is therefore the **bf16 family bound**

    BF16_RTOL = 1e-2      (u_bf16 = 2^-8 = 3.9e-3 per requantization)

Every suite that compares a kernel launch to the oracle imports this number.
That is the point of the module: a tolerance restated per file is a tolerance
that can be widened in one file by somebody who did not have to say so in a
diff a reviewer of the numeric contract would read.

THE RULE IS PER SLOT (`tests.oracle.fixtures.assert_slots_close`): the kernel's
error on each output slot is at most `BF16_RTOL` times that slot's ENVELOPE,
the sum of the sizes of the terms the slot adds up, which the same fp64
reference computes when run on `|v|` (`fixtures.oracle_run`; every routing
weight, gain and decay factor is non-negative). A readout `num / (den + eps)`
adds `|y|` for its denominator's rounding; a denominator or a mass column is its
own envelope. A gate whose output is not a sum of non-negative weights times
values (the entmax solve, its gradients, the producer's amplitudes) compares per
element in `torch.testing.assert_close`'s form and states its own derived
tolerances beside the claim.

MEASURED 2026-09-14 (`pytest --oracle-margins=<file>` records every comparison):
the worst error over its allowance under its budget, and the smallest uniform
relative error the rule sees on some slot of the output.

    carry numerator      CARRY_READOUT_RTOL   0.69 (corner-tied-k4)   20% at most
    carry denominator    CARRY_READOUT_RTOL   0.39                    2%
    carry state plane    CARRY_FOLD_RTOL      0.91 (corner-tied-k4)   0.8%
    prefill readout      CARRY_READOUT_RTOL   under 0.5               4% at most
    decode y (a step)    BF16_RTOL            0.11                    41% at most
    decode state         BF16_RTOL            < 1e-3                  1%
    intra output / mass  BF16_RTOL            0.39 / < 1e-3           1%
    projection logits    2^-8 (derived)       0.42                    1.2%

A slot whose sum is one term carries its output's whole rounding chain, which is
why the carry's budgets are counted from the kernel below rather than taken from
the family budget: the tied corner's numerator and state slots sum one or two
terms each. One decode step's `y` reads a random-signed entry state over up to
65536 leaves, so its terms cancel to about a fortieth of their sizes and that
step alone sees only a uniform error above 41%; the same gate's state sees 1%.
The TEETH are measured on the kernels' own outputs: the carry, prefill, decode,
intra and projection gates each wipe the slot of median size and move the
smallest slots past their allowance (`fixtures.assert_planted_errors_fail`),
and the prefill gate zeroes every readout slot under one percent of the largest,
which a global maximum passes by construction.
"""
from __future__ import annotations

#: The derived bf16 family budget. See the module docstring for the derivation.
BF16_RTOL = 1e-2

#: One bf16 rounding's largest relative error: a round to nearest moves an 8-significant-bit value by at most 2^-8
#: of itself, and keeping a float's HIGH HALF (a truncation) by at most two of them.
U_BF16 = 2.0 ** -8

#: THE CARRY'S CHAINS, counted in `csrc/rola/src/carry/carry_kernel.cuh`, where a term's error is the sum of its
#: roundings (every bf16 x bf16 product then lands exactly in an fp32 accumulator):
#: the FOLD rounds a deposit twice before it reaches the fp32 state -- the outer pair times the gain, and the inner
#: tile times that (`fold_fragment`'s two `mul_bf16x2`) -- so a state slot is off by at most two u of its terms;
#: the READOUT reads the state's high half (`snapshot_publish`'s and the masses' `hi_bf16x2`, a truncation: two u)
#: and rounds the inner tile times the outer pair once more (`readout_tile`'s `mul_bf16x2`), so `num`, `den` and a
#: readout carry the fold's two, those two and one: five u. A slot whose sum is one term carries the whole chain,
#: which the `corner-tied-k4` cell reaches (its numerator slots sum one or two terms): 1.34% measured, inside
#: five u (1.95%) and outside the family budget above.
CARRY_FOLD_RTOL = 2 * U_BF16
CARRY_READOUT_RTOL = 5 * U_BF16

#: Where the per-slot check records its margins: `--oracle-margins`, set by `tests/conftest.py`; None records nothing.
MARGINS_FILE = None
