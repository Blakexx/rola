# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The per-family compositions: one DAG each, read top to bottom.

A DAG is a literal ordered tuple of band-1 fact nodes (:mod:`rola.engine.facts.nodes`) and
band-2 rules (:mod:`rola.engine.rules`), walked once by :func:`rola.engine.runner.run`; its
terminal node folds what the walk produced into the family's PLAN, and the kernels
read only the plan (docs/internals/engine/engine.md).
"""

from rola.engine.dags.decode_dag import (
    DECODE_STEP,
    DecodeContext,
    admit_growth,
    decode_step_gated,
    growth_pending,
)
from rola.engine.dags.prefill_dag import (
    PREFILL_DAG,
    PrefillContext,
    run_prefill,
    validate_context,
)

__all__ = [
    "DECODE_STEP",
    "PREFILL_DAG",
    "DecodeContext",
    "PrefillContext",
    "admit_growth",
    "decode_step_gated",
    "growth_pending",
    "run_prefill",
    "validate_context",
]
