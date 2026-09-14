"""THE NUMERIC BAND, stated ONCE for every tier that grades against the oracle.

The comparison basis is fixed by the arms, not by a file: the fp64 oracle
(`rola.ops.naive.naive_rola`) is the reference, the kernel arm stores bf16, and
the derived budget is therefore the **bf16 family bound**

    BF16_RTOL = 1e-2      (u_bf16 = 2^-8 = 3.9e-3 per requantization)

Every suite that compares a kernel launch to the oracle imports this number.
That is the point of the module: a tolerance restated per file is a tolerance
that can be widened in one file by somebody who did not have to say so in a
diff a reviewer of the numeric contract would read. The band's TEETH -- that it
is not slack enough to swallow a format two rungs below bf16, and that a
genuinely wrong arm breaks it -- are `tests/oracle/test_bf16_family_gate.py`'s
M1/M2/M3 mutants, which live beside the claim they mutate.

Extracted here at P67 D2-b: the number's previous home was
the retired `test_consumer_vs_oracle.py`, the TILED consumer's oracle
gate, which retires with the tiled consumer. The band is a property of the
numeric contract, not of that gate, so it outlives it.
"""
from __future__ import annotations

#: The derived bf16 family budget. See the module docstring for the derivation.
BF16_RTOL = 1e-2
