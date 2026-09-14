# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The DAG framework: a declared node, a declared rule, the structural check, the walk.

A DAG is a literal ordered tuple of nodes and **the order in the tuple IS the
execution order**, chosen by the DAG's author. `needs` exists to make that order
CHECKABLE and readable, not to compute it: :func:`validate_dag` refuses a DAG whose
node reads a fact no earlier node produced, at import time rather than at launch
(docs/internals/engine/engine.md).

A RULE is a band-2 node: a pure host fold from facts to a decision, declared with
:func:`rule` so the discipline is mechanically visible rather than conventional
(docs/internals/engine/rules.md).

Memoization is a node ATTRIBUTE here rather than a per-site convention: `pure_of`
names the facts a pure host derivation is a function OF, and `per_state` binds a
device allocation to the lifetime of the one sequence that owns it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from weakref import WeakKeyDictionary

from rola.engine.types import Mask

_STREAMS = ("host", "main", "side")


def _nvtx_push(name: str) -> None:
    """Open an NVTX range named for the DAG node about to execute.

    Instrumented ONCE here, in the walk every DAG (`chunk_dag`, `decode_dag`,
    any future family) goes through, rather than per node function — a new
    node needs no opt-in to appear in `nsys`. Guarded so a CPU-only import (no
    CUDA context) never raises: NVTX ranges are a profiling aid, never a
    correctness dependency, and their absence must never change behavior
    (docs/measurement.md's "the sanitizer gap"-adjacent instrument for
    "what is the span" arguments).
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@dataclass(frozen=True, slots=True)
class Node:
    """One declared unit of the walk: its facts, its edges and its discipline."""

    fn: Callable[[dict], Any]
    name: str
    gives: tuple[str, ...]
    needs: tuple[str, ...]
    stream: str
    masks: tuple[Mask, ...]
    waits: tuple[str, ...]
    host_sync: bool
    pure_of: tuple[str, ...]
    per_state: bool
    #: this node may stand inside a CUDA-graph capture: it reads no device value on the
    #: host and allocates nothing per step.
    capturable: bool
    #: this node exists to be an independent second derivation for a gate, and no
    #: shipped DAG may contain it (docs/internals/engine/decode_dag.md).
    reference_only: bool
    memo: Any = field(compare=False)
    #: the band-2 fold this node adapts, or None for a band-1 fact node. Its presence
    #: IS the rule marker: `rola/engine/rules/` is the set of functions that carry one.
    pure: Callable[..., Any] | None = None

    def __call__(self, facts: dict) -> Any:
        return self.fn(facts)

    def table(self, facts: dict) -> dict | None:
        """The cache this call's output belongs in, or None when the node is uncached.

        A `per_state` node's table is WEAK-keyed on the state object, and the weakness
        is required rather than incidental: a dropped sequence must drop its buffers,
        and two live sequences must never share one.
        """
        if self.per_state:
            return self.memo.setdefault(facts["state"], {})
        return self.memo if self.pure_of else None


def node(*, gives: tuple[str, ...] = (), needs: tuple[str, ...] = (),
         stream: str = "host", masks: tuple[Mask, ...] = (Mask.M1, Mask.M2),
         waits: tuple[str, ...] = (), host_sync: bool = False,
         pure_of: tuple[str, ...] = (), per_state: bool = False,
         capturable: bool = False, reference_only: bool = False,
         ) -> Callable[[Callable], Node]:
    """Declare a node. `host_sync` marks the ONE node permitted a device-to-host read."""

    def declare(fn: Callable[[dict], Any]) -> Node:
        if stream not in _STREAMS:
            raise ValueError(f"{fn.__name__}: stream={stream!r} is not one of {_STREAMS}")
        if per_state and "state" not in needs:
            raise ValueError(f"{fn.__name__}: a per_state node must read `state`")
        if set(pure_of) - set(needs):
            raise ValueError(
                f"{fn.__name__}: pure_of names {sorted(set(pure_of) - set(needs))} it "
                f"does not read; a memo key that is not an input is not a key")
        if capturable and host_sync:
            raise ValueError(
                f"{fn.__name__}: a host_sync node reads a device value on the host, "
                f"which is exactly what a capture forbids; it cannot be capturable")
        return Node(fn=fn, name=fn.__name__, gives=tuple(gives), needs=tuple(needs),
                    stream=stream, masks=tuple(masks), waits=tuple(waits),
                    host_sync=host_sync, pure_of=tuple(pure_of), per_state=per_state,
                    capturable=capturable, reference_only=reference_only,
                    memo=WeakKeyDictionary() if per_state else {})

    return declare


def rule(*, gives: tuple[str, ...], needs: tuple[str, ...],
         masks: tuple[Mask, ...] = (Mask.M1, Mask.M2)) -> Callable[[Callable], Callable]:
    """Declare a RULE: a pure host fold from facts to a decision.

    **A rule may read a fact's SHAPE, DTYPE, DEVICE and PRESENCE, and never its
    CONTENTS. It issues no launch and allocates nothing.** That contract is not
    machine-checkable, which is exactly why the marker exists: `rola/engine/rules/` holds
    functions carrying it and nothing else, so the discipline is reviewable by
    listing a directory (docs/internals/engine/rules.md).

    The decorated function keeps its plain signature -- a rule is called directly,
    with values, which is what makes it testable with no device present -- and gains
    `.node`, the band-2 node a DAG walks it as. The parameter names ARE the fact
    names, refused here if they disagree, so the adapter is a lookup and not a
    convention.
    """

    def declare(fn: Callable[..., Any]) -> Callable[..., Any]:
        from inspect import signature

        params = tuple(signature(fn).parameters)
        if params != tuple(needs):
            raise ValueError(
                f"{fn.__name__}: the rule reads {list(needs)} but its parameters are "
                f"{list(params)}; a rule's parameter names are the FACT names it folds")

        def walk(facts: dict) -> Any:
            return fn(**{name: facts[name] for name in needs})

        walk.__name__ = fn.__name__
        #: A rule reads no tensor CONTENTS, issues no launch and allocates nothing, so
        #: every rule is capturable by its own contract -- there is nothing else for it
        #: to be.
        fn.node = replace(
            node(gives=gives, needs=needs, masks=masks, capturable=True)(walk), pure=fn)
        return fn

    return declare


def validate_dag(dag: tuple[Node, ...], seeds: tuple[str, ...]) -> None:
    """Refuse a DAG that is not walkable in its own order. Import-time, per DAG."""
    available = set(seeds)
    for step in dag:
        missing = [name for name in step.needs if name not in available]
        if missing:
            raise ValueError(
                f"{step.name} reads {missing} but no earlier node or seed produces "
                f"them: the tuple's order is the execution order, so a need that "
                f"arrives later is a DAG authoring error")
        available.update(step.gives)

    joins = [step.name for step in dag if step.waits]
    if len(joins) > 1:
        raise ValueError(
            f"{joins}: exactly one Join per DAG. A second event is a second "
            f"synchronization discipline.")
    syncs = [step.name for step in dag if step.host_sync]
    if len(syncs) > 1:
        raise ValueError(f"{syncs}: at most one device-to-host read per DAG")
    enumerated = [step.name for step in dag if step.reference_only]
    if enumerated:
        raise ValueError(
            f"{enumerated}: a reference_only node is a gate's INDEPENDENT second "
            f"derivation of a fact the shipped path derives its own way. Walking one "
            f"here would delete that independence, which is the whole of its value.")


def validate_capturable(dag: tuple[Node, ...], name: str) -> None:
    """Refuse a recipe that could not stand inside a CUDA-graph capture.

    Structural, not measured: a node that reads a device value on the host, or that
    allocates per step, breaks a capture whether or not today's fixture notices
    (docs/internals/engine/decode_dag.md).
    """
    bad = [step.name for step in dag if not step.capturable]
    if bad:
        raise ValueError(
            f"{name} is the captured recipe and {bad} is/are not declared capturable: "
            f"a host read or a per-step allocation inside the captured region is what "
            f"this declaration exists to refuse.")


def run(dag: tuple[Node, ...], facts: dict) -> dict:
    """Walk `dag` in order over the seeded fact bag, and return it."""
    for step in dag:
        mask = facts.get("mask")
        if mask is not None and mask not in step.masks:
            facts.update({name: None for name in step.gives if name not in facts})
            continue
        if step.gives and all(name in facts for name in step.gives):
            continue
        absent = [name for name in step.needs if name not in facts]
        if absent:
            raise RuntimeError(f"{step.name} was reached without {absent}")
        table = step.table(facts)
        key = tuple(facts[name] for name in step.pure_of)
        if table is not None and key in table:
            produced = table[key]
        else:
            _nvtx_push(step.name)
            try:
                produced = step(facts)
            finally:
                _nvtx_pop()
            if table is not None:
                table[key] = produced
        if len(step.gives) == 1:
            facts[step.gives[0]] = produced
        elif step.gives:
            facts.update(dict(zip(step.gives, produced, strict=True)))
    return facts
