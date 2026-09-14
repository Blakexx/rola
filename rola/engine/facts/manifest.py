# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""What this BINARY was built for: the arm census, read off the extension.

A fact about the build rather than about the call, and the only input an arm or
envelope decision has that is not the call's own. It is READ, never restated -- a
second copy of the matrix in python is a second source that agrees until it does not
(docs/internals/engine/engine.md).

The read is CACHED because it is a function of the loaded binary and of nothing
else, which is the primitive declaring its own caching rather than each call site
carrying a memo. The result is immutable for the same reason a fact is:
a shared cached value a caller could mutate would not be one.
"""
from __future__ import annotations

from functools import cache

from rola.ops._ext import extension

#: An arm as every rule reads it: `(C, BC, D, B)`. The census rows carry the launch
#: geometry after those four, which is band-2 material for `LaunchRule` and not for
#: anything declared today.
Arm4 = tuple[int, int, int, int]


@cache
def chunk_arms() -> tuple[Arm4, ...]:
    """The `(C, BC, D, B)` arms the chunk CONSUMER is built for."""
    return tuple(tuple(row[:4]) for row in extension().chunk_arms())
