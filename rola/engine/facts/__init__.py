# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts: what is TRUE about one call, and the primitives that find it.

A fact is an immutable value produced by exactly one declared node. This domain holds
the node library (:mod:`rola.engine.facts.nodes`) and the BODIES its nodes are the
smallest wrappers over -- the packing and the write-atom bitmap
(:mod:`rola.engine.facts.planes`), the build's arm censuses
(:mod:`rola.engine.facts.manifest`), the operand list
(:mod:`rola.engine.facts.operands`), the bundle adapters
(:mod:`rola.engine.facts.call`).

Nothing is re-exported here: the value types, the plans and the runner are the
ENGINE's core and are imported from :mod:`rola.engine`, which is what keeps a domain
from being the route to something it does not own (docs/internals/engine/engine.md).
This is the one domain that reaches the extension, and it is reached by module.
"""
