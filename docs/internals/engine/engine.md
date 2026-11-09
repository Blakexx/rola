# The engine

Mirror of `rola/engine/` — the umbrella's ROOT (the runner, the value types, the
plans) and its `facts/` domain. This page explains WHY the structure is shaped as it
is and how its pieces connect; the code is the authoritative statement of how each
body works. What is DECIDED from facts is [the rule layer's page](rules.md).

Symbols: `B` batch, `T` tokens, `H` heads, `D` routing depth, `N` leaf count, `d_v`
value width, `BC` the carry's box in leaves (the arena's allocation extent), `BH = B*H`.

## <a id="the-three-bands"></a>1. What the engine is

One structure carries every kernel family from a call's raw inputs to its launch
arguments:

```
Context  ->  DAG  ->  launch
```

* a **Context** (`PrefillContext`, `DecodeContext`) is a frozen record of the RAW
  inputs — the routing bundle, the values, the decay dial, the state, the expert
  overrides. Nothing derived.
* a **DAG** is an ordered tuple of declared nodes in THREE BANDS. Band 1 nodes may
  launch and may allocate, and produce FACTS: what is true about this call. Band 2
  nodes are RULES — pure host folds from facts to decisions, which launch nothing and
  allocate nothing ([`rules.md`](rules.md)). They are interleaved, not phased.
* a **Plan** is band 3 where a family has one: the rules' verdicts plus references to
  the terminal tensor facts, and no third kind of field (`plan.py`'s `DecodePlan`,
  which the decode DAG's `step` node launches).
* the launch node reads its plan or its facts and decides nothing. A launch site that
  decides anything is the condition this engine exists to remove.

The engine puts under one declaration what would otherwise be five conventions agreeing
by hand: the facade's host order, the prefill seam's argument threading, the decode
geometry object, the state's residency entry, and the arena's side-stream/event pattern.

## <a id="the-split-rule"></a>2. What lives where

ONE umbrella package with three domains under it, and one sentence decides what is in
it at all:

> **What crosses the EXTENSION BOUNDARY lives in `rola/ops/`. What a DAG walk produces
> or decides lives in `rola/engine/`.**

* `rola/engine/` (the ROOT) — the SHARED CORE, which belongs to no domain: the DAG
  framework (`runner.py` — `Node`, `node`, `rule`, `run`, `validate_dag`,
  `validate_capturable`), the value types, the pins a caller may set over axes the op
  decides — `PlanOverrides` — the carry's carve — `Carve`, `SidePlan` — and the
  refusal every no-backward clause raises —
  `BackwardNotImplemented` — (`types.py`), and the per-family plans (`plan.py`).
* `rola/engine/facts/` — band 1, BODIES INCLUDED. The packing and the write-atom
  bitmap (`planes.py`), the liveness words and their folds (`liveness.py`), the facts
  pass's list model (`lists.py`), the operand list (`operands.py`), the adapters that read those facts off a routing
  bundle and the
  envelope authority they fold into — `envelope_refusal` and `require_envelope`
  (`call.py`) — and the declared nodes over them (`nodes.py`). A fact primitive is not
  a wrapper that defers its logic elsewhere; the primitive IS here.
* `rola/engine/rules/` — band 2, one declared fold per decision.
* `rola/engine/dags/` — the per-family compositions: the tuple, its context, its
  terminal node.
* `rola/ops/` — the launches and the bodies around them. `decode.py` is the geometry,
  the scratch and `step`, plus `_decode_step` — those same bodies in the step's order
  for a caller holding tensors rather than a bundle; the fold and the factor tables are
  IN the step's one launch and have no separate body here
  ([`decode/decode_fold.md`](../decode/decode_fold.md), [`decode/decode_lattice.md`](../decode/decode_lattice.md)).

**A domain reaches the core through the ROOT, never through a sibling domain.**
`rola/engine/__init__.py` re-exports the core for exactly that: `from rola.engine
import rule` is the import a rule module writes, so `rule` is a property of the engine
rather than of whichever domain happened to define it. The core itself imports no
domain, which is what keeps that re-export from being a cycle; `rola/engine/facts/`
re-exports nothing, because a domain that is also a route to something it does not own
is the mis-homing this layout removes. `rola.engine.rules` therefore never acquires a
dependency on the extension: the primitives that DO reach it are `rola.engine.facts`
modules, imported by module.

## <a id="a-dag-is-code"></a>3. A DAG is code

There is no graph library and no topological sort. A DAG is a literal tuple and
**the order in the tuple IS the execution order**, chosen by the DAG's author for an
efficient issue order. `needs` exists to make that order CHECKABLE and readable:
`validate_dag` refuses, at import time, a DAG whose node reads a fact no earlier node
or seed produces. A missing edge is therefore an authoring error, never a runtime
surprise.

The runner is the whole execution model:

* a node whose mask does not contain the call's MASK — the MaskRule's own output,
  read as a fact — is SKIPPED, and its facts are
  seeded null — the runtime-null output skip;
* a node whose facts are already in the bag is skipped, which is how a SUPPLIED fact
  short-circuits every node upstream of it;
* otherwise its needs are asserted present and it is called.

## <a id="masks"></a>4. Masks, and why not per-fact bools

Specialization is per CALL CLASS, and the set is CLOSED AT FOUR: `M1`
prefill-stateless and `Mask.M2` prefill-stateful for the prefill family, `Mask.M3` decode-dense and
`M4` decode-paged for the decode family. Each family has its own MaskRule over its own
facts — `mask` folds `CallClass`'s four bits, `decode_mask` folds the one bit a decode
step has — and a mask earns its name only by selecting a DIFFERENT node set. A growth
step does not: it is `M4` walked twice around an admission
([`decode_dag.md` §1](decode_dag.md#the-two-masks)).

A bool per fact would be `2^n` specializations of everything downstream, so intra-mask
variation is carried as a null plan FIELD instead — `state_in is None`, a dense step's
`page_table is None` — which is free and costs no hot-loop register.

## <a id="the-suppliable-set"></a>5. The suppliable set is closed

Production is the sole pluggable node. A caller may substitute the routing bundle and
the packed amplitude planes it stands for; everything DOWNSTREAM of that boundary —
the tables, the bitmaps, the liveness bits, the residency plan — is one canonical
implementation, un-overridable and gated once. A context that supplies a canonical
fact RAISES rather than forking fact production, because two ungated producers of one
fact is exactly what the two-purposes doctrine forbids.

The round trip through DRAM at that boundary is not a cost to engineer away; it IS
the pluggability seam.

## <a id="streams-and-the-join"></a>6. Streams, the join, and the one host read

Node metadata declares the discipline the arena implements:

* `stream="side"` — issued on the arena's side stream.
* `waits=(...)` — a Join. **There is exactly one Join per DAG** and `validate_dag`
  asserts it, because a second event would be a second synchronization discipline.
* `host_sync=True` — the ONE node permitted a device-to-host read.
* `capturable=True` — this node may stand inside a CUDA-graph capture: it reads no
  device value on the host and allocates nothing per step. Declaring it beside
  `host_sync` is refused; the two are one claim written from two ends, and
  `validate_capturable` refuses a captured recipe holding a node without it
  ([`decode_dag.md` §4](decode_dag.md#capturable)).
* `reference_only=True` — this node is a GATE's independent second derivation of a
  fact the shipped path derives its own way. `validate_dag` refuses any DAG containing
  one, which is what keeps that independence from being deleted by a well-meaning
  wiring ([`decode_dag.md` §6](decode_dag.md#reference-only)).

Where a Join sits is a correctness fact, not a scheduling preference: see
[the prefill DAG's page](prefill_dag.md).

## <a id="memoization"></a>7. Memoization is a node attribute

Not a per-site convention:

* `pure_of=(...)` names the facts a pure host derivation is a function OF, and gives
  the node a plain dict cache. It is for CONFIG derivations — the geometry of a
  shape, the arm of a topology — whose answer is identical for identical keys.
* `per_state=True` gives the node a `WeakKeyDictionary` keyed on the state object,
  for device allocations belonging to ONE live sequence. The weakness is required
  rather than incidental: a dropped sequence must drop its buffers, and two live
  sequences must never share one.

A memo key that is not one of the node's own inputs is refused at declaration.

**A `per_state` cache holds REBUILDABLE TRANSIENTS ONLY**, and the state is its KEY
rather than one of its inputs: every other need is a declared memo key, so what is
cached is a function of facts a rebuild has, and the body never reads the sequence at
all. That is what makes the handle-IS-the-sequence invariant survive caching — two
handles onto one geometry get interchangeable objects, and a fork or a rollback
inherits nothing through a cache (`docs/internals/state.md` §9.1).
`tests/unit/test_facts_memo.py` drives every shipped `per_state` node with a state that
raises on any attribute read, so the property is proven per node rather than declared.

A PRIMITIVE whose answer is a function of the loaded binary rather than of any fact is
not a node and no node memo reaches it: the binary's arm census (`rola.ops.carry.arms`) is
read by the envelope adapter in `facts/call.py` at each call. The intra's admitted widths are not one: they are a law
over the widths and the value width (`rola.ops.intra.admits`).

## <a id="refusals"></a>8. Refusals are one call at the head

Each kernel family has ONE `validate_context(ctx)`, run immediately after the context
is built and before any fact node. It consolidates the operand-type refusals, the
configuration envelope, the gradient refusal and the suppliable-set closure. Scattering preconditions across the DAG
would allow a refusal three
launches in, with the operands already consumed; one call at the head makes that
unreachable by construction.

Every refusal names the clause that has no arm, and the message texts are gated
(`tests/unit/test_api_contract.py`).

The envelope half of that call is `facts/call.py`'s `require_envelope`, and it is
reached DOWNWARD: the engine composes the rule layer's clauses over its own adapters
and the facade calls the engine, never the reverse. `envelope_refusal` beside it is the
same composition without the raise, which is what lets a caller ask before calling —
`rola.expert` re-exports it as the public address (`docs/api.md` §5), and that
re-export is the only direction the dependency runs.

`PlanOverrides` is the same statement from the other end. A pin is a VALUE a rule
folds — `rules/paging.py` reads it and returns a bool — so it is engine vocabulary and
`types.py` is where it is defined; `rola.expert` re-exports it as the marked door
(`docs/api.md` §5). NO MODULE UNDER `rola/engine/` IMPORTS `rola.interface` OR
`rola.expert`, at import time or inside a body: a deferred import in a function body
hides an inversion from the import graph without removing it, so the rule layer holds
none.

## <a id="the-plan"></a>9. What a plan may hold

A host field is some rule's verdict; a tensor field is a reference to a terminal
fact. **No plan field may be `O(N * L)`**: a plan is held for as long as its call is,
so a field added for one launch's convenience is retained with it.

A fact can also be an OPERAND, and the engine does not pretend otherwise: the
state-out plane is both a launch argument and the call's result.

## <a id="neutrality"></a>10. The neutrality gate

A host refactor that claims neutrality is answerable to what the extension RETURNED —
`y`, the state plane and the facts themselves — off a fixed cell set, compared across
two checkouts. Comparing the returned objects rather than re-deriving them is what
makes the comparison independent of whatever host structure produced the launch
arguments. The mechanism is the diff node
([`docs/measurement.md`](../../measurement.md#the-build-everything-declared-a-run-pruned-by-label)):
two sides, the cells, a rule and an expectation, so "against `HEAD~1`" is a second
checkout the suite's root already composes rather than a second tool.
