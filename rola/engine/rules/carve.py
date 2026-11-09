# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""CarveRule -- the carry's box and each side's operation plan: the expert's pinned carve, checked, else producer v1's."""
from __future__ import annotations

from typing import Any

from rola.engine import Carve, PlanOverrides, rule
from rola.engine.carve import check_carve, plan_carve


@rule(gives=("carve",), needs=("routes", "d_v", "expert"))
def carve(routes: Any, d_v: int, expert: Any) -> Carve:
    """The carry's carve: the expert's pinned one, checked, else producer v1's -- from the rows' shipped widths and
    each level's flags, never their contents (docs/internals/engine/rules.md#carve)."""
    widths = tuple(int(rows.shape[-1]) for rows in routes.amp_read)
    read_sparse, write_sparse = routes.sparse
    pinned = None if expert is None else PlanOverrides.of(expert).carve
    if pinned is not None:
        check_carve(pinned, widths, read_sparse, write_sparse, d_v)
        return pinned
    return plan_carve(widths, read_sparse, write_sparse, d_v)
