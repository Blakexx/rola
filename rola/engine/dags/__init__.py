# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The per-family compositions: one DAG each, read top to bottom.

A DAG is a literal ordered tuple of band-1 fact nodes (:mod:`rola.engine.facts.nodes`) and
band-2 rules (:mod:`rola.engine.rules`), walked once by :func:`rola.engine.runner.run`; its
terminal node folds what the walk produced into the family's PLAN, and the kernels
read only the plan (docs/internals/engine/engine.md).
"""

from rola.engine.dags.chunk_dag import (
    CHUNK_DAG,
    ChunkContext,
    build_chunk_plan,
    validate_context,
)
from rola.engine.dags.decode_dag import (
    DECODE_STEP,
    DECODE_VERDICT,
    DecodeContext,
    admit_growth,
    decode_forward,
    decode_step_gated,
    growth_pending,
)

__all__ = [
    "CHUNK_DAG",
    "DECODE_STEP",
    "DECODE_VERDICT",
    "ChunkContext",
    "DecodeContext",
    "admit_growth",
    "build_chunk_plan",
    "decode_forward",
    "decode_step_gated",
    "growth_pending",
    "validate_context",
]
