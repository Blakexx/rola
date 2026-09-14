# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""MaskRule -- which specialized subgraph this call class runs, per kernel family."""
from __future__ import annotations

from rola.engine import Mask, rule


@rule(gives=("mask",), needs=("call_class",))
def mask(call_class) -> Mask:
    """The DAG band this call runs: the stateful band the moment any state bit is engaged.

    Specialization is per CALL CLASS and never per fact: a bool per fact would be
    `2^n` built subgraphs, and intra-mask variation is a runtime-null output skip
    instead (docs/internals/engine/rules.md).
    """
    return (Mask.M2 if (call_class.state_in or call_class.state_out or call_class.paged)
            else Mask.M1)


@rule(gives=("mask",), needs=("paged",), masks=(Mask.M3, Mask.M4))
def decode_mask(paged) -> Mask:
    """The decode band this step runs: the BACKING is the whole of the distinction.

    A dense-backed step addresses the leaf space directly and has nothing to admit; a
    paged one asks the device whether its write atoms are resident. That is the only
    difference in the node set, and a growth step selects no different subgraph -- it is
    M4 walked twice around an admission (docs/internals/engine/decode_dag.md).
    """
    return Mask.M4 if paged else Mask.M3
