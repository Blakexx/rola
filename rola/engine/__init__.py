# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The engine: ONE structure carries every kernel family from raw inputs to a launch.

    Context -> DAG -> Plan -> execute(plan)

This ROOT holds the shared CORE and nothing family-specific: the DAG framework
(:mod:`rola.engine.runner`), the value types (:mod:`rola.engine.types`) and the
per-family plans (:mod:`rola.engine.plan`). The DOMAINS are its subpackages --
:mod:`rola.engine.facts` (the fact primitives and their bodies),
:mod:`rola.engine.rules` (the band-2 folds) and :mod:`rola.engine.dags` (the
per-family compositions).

The core is re-exported here because that is the layering statement: a domain reaches
it THROUGH this module and never through a sibling domain, so `rule` is a property of
the engine rather than of the fact-finder that happened to define it
(docs/internals/engine/engine.md).
"""

from rola.engine.plan import ChunkPlan, DecodeGeometry, DecodePlan
from rola.engine.runner import (
    Node,
    node,
    rule,
    run,
    validate_capturable,
    validate_dag,
)
from rola.engine.types import (
    MMA_K_QUANTUM,
    SUPPLIABLE,
    Arm,
    BackwardNotImplemented,
    CallClass,
    Mask,
    PlanOverrides,
)

__all__ = [
    "MMA_K_QUANTUM",
    "SUPPLIABLE",
    "Arm",
    "BackwardNotImplemented",
    "CallClass",
    "ChunkPlan",
    "DecodeGeometry",
    "DecodePlan",
    "Mask",
    "Node",
    "PlanOverrides",
    "node",
    "rule",
    "run",
    "validate_capturable",
    "validate_dag",
]
