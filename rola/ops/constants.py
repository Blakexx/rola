# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Constants that are part of the DEFINITION, not of any one implementation.

A number in this file changes what RoLA computes. That is the entry criterion:
a tolerance, a tile size or a launch heuristic belongs to the thing that chose
it, but a constant that appears inside the mathematical statement has exactly
one home, because two homes that merely happen to agree is a defect the
one-API rule exists to forbid.
"""

from __future__ import annotations

#: The readout denominator's floor: ``y[t] = num[t] / (den[t] + READOUT_EPS)`` --
#: the readout is a ratio, and this is the one form it has.
#:
#: It is a DEFINITION constant, not a guard. ``den[t] = sum_s R[t,s] * mass[t,s]``
#: is zero exactly when the token reads only leaves no token has written yet --
#: the cold-read regime, which is ordinary rather than exceptional at the start
#: of a sequence and permanent for a leaf the routing never selects. The floor
#: fixes the value there at ``num/eps`` (itself zero, since an unwritten leaf
#: contributes no mass and no value), so the cold read is a defined zero instead
#: of a NaN. Changing it changes the function everywhere, not just at zero: the
#: readout is a ratio, so every output is scaled by ``den/(den + eps)``.
#:
#: The value is shared by the oracle (``rola/ops/naive.py``) and decode
#: (``rola/ops/decode.py``); the deleted chunk consumer read it too, and the intra
#: will again. It is NOT on the public surface: ``rola_op`` takes no ``eps``,
#: because a number that changes the function is not a caller's argument. The
#: kernel holds no copy either -- the decode config takes ``eps`` as a runtime
#: argument -- so this constant is the single origin of the value on every path.
READOUT_EPS = 1e-5

__all__ = ["READOUT_EPS"]
