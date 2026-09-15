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
the worst error over its allowance, and the smallest uniform relative error the
rule sees on some slot of the output.

    carry numerator (every oracle cell)        0.47     7.7% at most
    carry denominator                          0.45     1%
    carry state plane                          0.62     1%
    combined prefill readout                   0.27     2%
    decode y (every step of every gate)        0.11     41% at most
    decode state                               < 1e-3   1%
    intra output / mass                        0.39 / < 1e-3   1%
    projection logits (rtol 2^-8, derived)     0.42     1.2%

One decode step's `y` reads a random-signed entry state over up to 65536
leaves, so its terms cancel to about a fortieth of their sizes and that step
alone sees only a uniform error above 41%; the same gate's state sees 1%.
The TEETH are measured on the kernels' own outputs: the carry, prefill, decode,
intra and projection gates each wipe the slot of median size and move the
smallest slots past their allowance (`fixtures.assert_planted_errors_fail`),
and the prefill gate plants a `sqrt(t)` drift that a global maximum passes.
"""
from __future__ import annotations

#: The derived bf16 family budget. See the module docstring for the derivation.
BF16_RTOL = 1e-2

#: Where the per-slot check records its margins: `--oracle-margins`, set by `tests/conftest.py`; None records nothing.
MARGINS_FILE = None
