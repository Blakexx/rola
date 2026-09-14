# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer: band 2, the pure host folds from facts to execution decisions.

Rule engines are PLURAL over ONE shared fact set. None of them owns a fact, none of
them recomputes one, and none of them may issue a launch or an allocation. Each is
declared with :func:`rola.engine.runner.rule`, which is what makes this directory's
listing the layer's inventory (docs/internals/engine/rules.md).
"""

from rola.engine.rules.arm import CHUNK_TOKENS, arm
from rola.engine.rules.envelope import arm_envelope, envelope, training_envelope
from rola.engine.rules.mask import decode_mask, mask
from rola.engine.rules.paging import admission, paging_preference
from rola.engine.rules.run_table import run_table

__all__ = [
    "CHUNK_TOKENS",
    "admission",
    "arm",
    "arm_envelope",
    "decode_mask",
    "envelope",
    "mask",
    "paging_preference",
    "run_table",
    "training_envelope",
]
