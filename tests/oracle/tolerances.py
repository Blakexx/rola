"""THE NUMERIC BAND, stated ONCE for every tier that grades against the oracle.

The comparison basis is fixed by the arms, not by a file: the fp64 oracle
(`rola.ops.naive.naive_rola`) is the reference, the kernel arm stores bf16, and
the derived budget is therefore the **bf16 family bound**

    BF16_RTOL = 1e-2      (u_bf16 = 2^-8 = 3.9e-3 per requantization)

Every suite that compares a kernel launch to the oracle imports this number.
That is the point of the module: a tolerance restated per file is a tolerance
that can be widened in one file by somebody who did not have to say so in a
diff a reviewer of the numeric contract would read.

THE RULE IS PER SLOT (`tests.oracle.fixtures.assert_slots_close`), and every
kernel-vs-oracle gate names the OUTPUT KIND it grades (`Output` below), which
carries the whole rule for that output:

- ITS CLAUSES, the shared form (Blake, 2026-09-15). A clause is an `(rtol, atol)`
  pair and FAILS a slot whose relative error exceeds `rtol` AND whose absolute
  error exceeds `atol`, so a slot of size `s` passes within
  `min_i max(rtol_i * s, atol_i)`: one clause is `allclose`'s own rule, a clause
  at `rtol = 0` is a pure absolute cap, and each further clause is another hinge
  in that curve. Every gate in the tree is held this way -- the carry, prefill,
  decode and intra kernels, the entmax solve and its gradients, the producer's
  stored levels and routes -- so one rule reads every oracle comparison and no
  gate states tolerances of its own.
- ITS ENVELOPE, where the output is MULTILINEAR: at most `rtol` times the sum of
  the sizes of the terms the slot adds up, which the same fp64 reference computes
  when run on `|v|` (`fixtures.oracle_run`; every routing weight, gain and decay
  factor is non-negative). A readout `num / (den + eps)` adds `|y|` for its
  denominator's rounding; a denominator or a mass column is its own envelope.
  No curve over slot size can separate a slot that is small because its terms
  cancelled from one that is small because its terms were small; the envelope
  gives each slot its own allowance from its own terms, which is why it sits
  beside the clauses rather than replacing them.

The tightest of the two binds, and a failure names which one it was.

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

from dataclasses import dataclass

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

#: THE PRODUCTION ENTMAX SOLVE'S RATIFIED TOLERANCES against the fp64 reference solve (values and their gradients),
#: and the stored bf16 levels' (`tests/integration/test_production_levels.py` states their derivation beside the seam).
#: Never loosened.
ENTMAX_RTOL, ENTMAX_ATOL = 2e-5, 2e-6
ENTMAX_GRAD_RTOL, ENTMAX_GRAD_ATOL = 5e-5, 5e-6
STORED_ROUTE_RTOL, STORED_ROUTE_ATOL = 4e-3, 2e-3


@dataclass(frozen=True)
class Output:
    """ONE KIND OF ORACLE-GRADED OUTPUT: the clauses every slot of it is held to, and on a multilinear output the
    envelope budget. A clause ``(rtol, atol)`` FAILS a slot whose relative error exceeds ``rtol`` AND whose absolute
    error exceeds ``atol`` (Blake, 2026-09-15: ``OR(AND(r1, a1), AND(r2, a2), ...)``), so a slot of size ``s`` passes
    within ``min_i max(rtol_i·s, atol_i)``: one clause is `allclose`'s rule, each further clause another hinge in that
    curve, and a clause at ``rtol = 0`` is a pure absolute cap. ``rtol`` is the multilinear envelope budget
    (`fixtures.allowances`), None for an output that has no envelope. Every gate names its output kind here; no gate
    states its own numbers.
    """

    name: str
    clauses: tuple[tuple[float, float], ...]
    rtol: float | None = None


#: THE CLAUSES ARE MEASURED, and these are the measurements (`pytest --oracle-margins=FILE`, read with
#: `fixtures.frontier_rtols`): for each relative error, the largest ABSOLUTE error any slot past it took on the whole
#: battery, from the comparisons that passed. A clause declares that relative error with twice that absolute error, so
#: no measured slot is inside a clause it fails; clauses below the output's own declared budget are not declared at all,
#: because a draw that reached the budget would fail one. A kind's numbers are re-measured when the cells or the
#: hardware change, and tightening is always the safe direction. MEASURED 2026-09-15 on RTX 3080 Ti (sm_86), the oracle,
#: integration and entmax tiers: 2,700 comparisons, the worst slot at 0.91 of its allowance (the carry state).
CARRY_NUM = Output("carry numerator", rtol=CARRY_READOUT_RTOL,
                   clauses=((0.0, 6.24e-2), (1e-1, 1.05e-3), (1e1, 3.96e-4), (1e2, 3.17e-5)))
CARRY_DEN = Output("carry denominator", rtol=CARRY_READOUT_RTOL, clauses=((0.0, 5.47e-2), (1e-1, 0.0)))
CARRY_STATE = Output("carry state", rtol=CARRY_FOLD_RTOL,
                     clauses=((0.0, 4.02e-2), (CARRY_FOLD_RTOL, 8.34e-3), (1.0, 4.01e-3), (1e1, 1.90e-3),
                              (1e2, 4.44e-4)))
PREFILL_READOUT = Output("prefill readout", rtol=CARRY_READOUT_RTOL,
                         clauses=((0.0, 2.86e-2), (1e-1, 1.24e-2), (1.0, 6.10e-3), (1e1, 3.01e-3), (1e2, 6.39e-4)))
PREFILL_STATE = Output("prefill state", rtol=CARRY_FOLD_RTOL,
                       clauses=((0.0, 4.02e-2), (CARRY_FOLD_RTOL, 8.34e-3), (1.0, 1.96e-3), (1e2, 4.44e-4)))
DECODE_Y = Output("decode y", rtol=BF16_RTOL, clauses=((0.0, 3.52e-3), (1.0, 1.25e-3), (1e2, 3.35e-4)))
DECODE_STATE = Output("decode state", rtol=BF16_RTOL, clauses=((0.0, 4.38e-7), (1e-2, 7.80e-9), (1e-1, 0.0)))
INTRA_OUTPUT = Output("intra output", rtol=BF16_RTOL, clauses=((0.0, 1.66e-2), (1e-1, 7.74e-3)))
INTRA_MASS = Output("intra mass", rtol=BF16_RTOL, clauses=((0.0, 9.88e-7), (1e-2, 0.0)))
PROJECTION_LOGITS = Output("projection logits", rtol=U_BF16, clauses=((0.0, 1.32e-1), (1.0, 6.42e-2), (1e1, 0.0)))
ENTMAX_VALUES = Output("entmax values", clauses=((0.0, 8.34e-2), (ENTMAX_RTOL, ENTMAX_ATOL)))
ENTMAX_GRADIENTS = Output("entmax gradients", clauses=((0.0, 5.96e-5), (ENTMAX_GRAD_RTOL, ENTMAX_GRAD_ATOL)))
STORED_LEVELS = Output("stored bf16 levels", clauses=((0.0, 1.52e-1), (BF16_RTOL, BF16_RTOL)))
STORED_ROUTE = Output("stored route", clauses=((0.0, 2 * STORED_ROUTE_ATOL), (STORED_ROUTE_RTOL, STORED_ROUTE_ATOL)))

#: EVERY OUTPUT KIND, for the test that holds each to declaring its clauses and their measurement.
OUTPUTS = (CARRY_NUM, CARRY_DEN, CARRY_STATE, PREFILL_READOUT, PREFILL_STATE, DECODE_Y, DECODE_STATE, INTRA_OUTPUT,
           INTRA_MASS, PROJECTION_LOGITS, ENTMAX_VALUES, ENTMAX_GRADIENTS, STORED_LEVELS, STORED_ROUTE)

#: Where the per-slot check records its margins: `--oracle-margins`, set by `tests/conftest.py`; None records nothing.
MARGINS_FILE = None
