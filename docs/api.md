# The API

RoLA is linear attention under a structured feature map. That is not a slogan about the
paper; it is the specification of this API. A head's recurrent state is expanded into
many sub-states that share the head's projections, and a learned routing decides which
sub-states each token reads and writes — so the public contract is *linear attention's*
contract, with exactly **one** new concept: the feature map.

A user who knows linear attention should find one unfamiliar thing on this page.

```python
import rola

routes   = rola.RouteProducer(rola.uniform(2, rola.union_routing(64, alpha=1.5)),
                              hidden_size=1024, num_heads=8)
decay    = rola.LearnedDecay(2 ** -8, widths=routes.widths, num_heads=8)

layer = rola.RoLA(routes, d_v=64, decay=decay, layer_idx=0).cuda().eval()
y, _ = layer(hidden_states)                     # stateless: (y, None)
```

or, without the layer:

```python
factors = routes(hidden_states)          # the (q, k) pair, after the feature map
y, _ = rola.rola_op(factors, v, decay())
```

---

## 1. The op

```python
rola_op(routes: RouteFactors, v: Tensor, decay: LeafMassDecay | None = None, *,
        state=None, expert=None) -> (y, state)
```

`routes` is the `(q, k)` pair after the feature map. `v` is `[B, T, H, d_v]`. `decay` is
the recurrence dial. `y` is `[B, T, H, d_v]`.

`state` is `None` (a stateless call, returning `(y, None)`) or a `rola.state()` object,
which is mutated in place and handed back. THE PREDICATE IS `state is not None` — there
is no second flag, because a stateless call runs a kernel variant with no state I/O
compiled into it at all. Read the carried state with
`state.materialize() -> [B, H, N, d_v + 1]`, where `N = prod(widths)` is the leaf
capacity; that is a bridge for the oracle and for a test, never a measured path. See
[`docs/internals/state.md`](internals/state.md).

`rola.state(pool_slack=0.0)` is the paged backing's other storage-mechanics knob:
a fraction of full residency, reserved per batch-head as physical slots a `T = 1` decode
step can admit into itself, with no host round trip. `0.0` — the default — is exact
commitment, unchanged from before the knob existed. Two calls read and refill it off the
step's own path: `rola.engine.dags.decode_dag.pool_headroom(state)` (the slack slots the
thinnest batch-head still has) and `refill_pool(state)` (restock what the device spent).
See [`internals/state.md`](internals/state.md#the-slack-pool) and
[`internals/paging/paging.md`](internals/paging/paging.md#the-slack-pool).

There is no plan argument, no configuration argument, and no statistics argument. Those
are not model facts — they are how the op RUNS, and the op decides how it runs, per
call, from the routing it was actually given. They live behind
[`rola.expert`](#5-the-expert-surface).

**Any level width, any `d_v`.** A `RouteProducer` level's width and a layer's `d_v` are
never required to be a power of two or a shipped value — host-side padding at the op's
own call surface serves the rest, with no kernel change
([`docs/internals/padding.md`](internals/padding.md)). `rola.rola_op`'s CUDA arm is
currently deleted pending K31 R2, so this is not yet reachable end to end through the
layer; the padding seam is built and tested at `rola.ops.carry`/`rola.ops.decode`/
`rola.ops.intra` ahead of the pipelined kernel body that will consume it.

### 1.1 The built envelope IS the surface

The arm is the chunk consumer, and the chunk consumer is a BUILT MATRIX. **Everything
outside it raises.** There is no second arm behind this surface: the fp64 oracle
(`rola.ops.naive.naive_rola`) is the executable spec — what tests and ratification
call directly — and it is not reachable by dispatch, because a dispatch to it at
inference is a silent ~1000x-slower route shipping unnoticed, which is the exact thing
the no-fallback rule forbids.

| axis | the built matrix | outside it |
|---|---|---|
| a gradient anywhere in the operands | none | `BackwardNotImplemented` |
| the module's mode (layer only) | `eval()` | `BackwardNotImplemented` |
| device | CUDA | refused |
| `d_v` | `64` | refused |
| widths | uniform (`len(set(widths)) == 1`) | refused (a jagged topology) |
| `(D, B)` | the manifest's combinations (`rola.engine.facts.manifest`'s `chunk_arms()`) | refused |
| `T` | any length | — |
| `decay` | not implemented in the chunk arm (decode has it) | refused |
| an unratified activation | — | refused |

Ask before calling: `rola.expert.envelope_refusal(routes, v, decay)` returns the
sentence the refusal will carry, or `None`. `rola.expert.require_envelope(...)` is the
raising form the op itself uses.

**This is a real capability narrowing, and it is stated rather than softened.** Shapes
outside the matrix — including the flagship `widths = (8, 16)`, which is refused for
being non-uniform — had a kernel before the tiled consumer's retirement, and there is
NO TRAINING PATH through this surface at all until the native backward lands. The cards
that widen the matrix and bring the backward are in
[`docs/open-work.md`](open-work.md).

`rola_op` is the **single autograd boundary**, and it is a refusal rather than a
branch: the failure it prevents — a forward-only kernel reached from a gradient
context returns a tensor with no `grad_fn` and severs the graph silently — is
structural, not a bug in any one call site. The deleted consumer's autograd node was the last
line, loud if the first is ever bypassed.

### 1.2 Continuation is LAYERED

The op's state is the paged tensor and nothing else
([`docs/internals/state.md`](internals/state.md)). Each level of the stack returns its
OWN thin bundle, and the rule is a membership test:

> **An object holds exactly the recurrences ITS level introduced.**
> Op: the routed state. Layer: `LayerContinuation`. Model: the per-layer bundles.

So `RoLA.forward` returns `(y, LayerContinuation)` — always the container — and takes
one back. `LayerContinuation.state` is the op's paged tensor; it is the container's
only field today, and any future layer-level recurrence is a field ON IT, never on the
op's state. The container is the contract in both directions: a bare `RoLAState` or the
retired `(state, conv_rings)` tuple is refused by name, because accepting a lookalike
would make the layer's own fields silently optional to carry.

This is not the old pooled state object recreated. That object's defect was LAYER
LUGGAGE INSIDE THE OP's state; keeping each level's object pure is what leaves
accretion nowhere to land.

### 1.3 TWO CONTRACTS, ONE TRANSLATOR

There are two boundaries in this library and they accept different things. Confusing
them is what produces a kernel that quietly pads, masks or half-serves a shape, so they
are stated apart.

| | what it accepts | what it does with the rest |
|---|---|---|
| **the API** (this page: `rola_op`, `RoLA`, `rola.state()`) | any level width `b_l >= 2`, any `d_v`, any length `L` | **pads** — level logits to the next power of two at or above 16 with `-inf`, `v` to the next shipped `DV` with zero columns |
| **the kernel** (`rola.ops.carry`, `rola.ops.decode`) | only the state descriptor's strict shape: `B_l` a power of two at or above 16, a shipped `DV`, `N` a multiple of 16, bf16 operands | **refuses**, by name — it never pads and never masks |

**The seam is the only translator.** A padded digit gets amplitude *exactly* zero
(`-inf` before the entmax or softmax solve), so it is never live, never read, never
written, never marks a box and never causes a page to be allocated: the kernel sees a
dead digit like any other and no kernel code knows padding happened. `d_v` padding is
the same mechanism on the value side — the projection writes the first `d_v` columns of
a `DV`-wide buffer and `y` comes back as a view of the first `d_v` — so a kernel never
sees a width other than its own.

**Both boundaries are tested.** API tests prove a padded run is bit-identical to the
same shape declared natively; kernel-boundary tests call the launch surface directly
with unpadded shapes and assert the refusal
(`tests/unit/test_carry_surface.py`). A logical-width field in a kernel entry
signature is a lint finding, because a kernel that could read one could pad.

**What the descriptor fixes.** A state binds its format at its first call — `D`, the
per-level `B_l`, the canonical leaf order, the page rectangle, `DV`, and the stored form
(split bf16 planes) — and every later call is *checked* against it rather than assumed
to fit. A call that disagrees is refused at the seam, naming the field, instead of
addressing somebody else's bytes inside a kernel. Re-laying out a state is an explicit
op, never a side effect of a call. See
[`internals/state.md`](internals/state.md#format).

**Status.** The kernel half of this contract is live: the carry launch surface and its
three refusals are built and tested. The *body* behind it is not — the carry family is
being rebuilt from the ground up (card `development/queue/C_CLEAN_SLATE.md`), so a carry
call today reaches the surface, passes its refusals and then fails with one honest
reason: there is no implementation on this line. The envelope table in
[1.1](#11-the-built-envelope-is-the-surface) is what is *built*; this section is the law
it is built against.

## 2. The feature map

### 2.1 The bundle IS the contract

```python
RouteFactors(topology, read, write, g_write, read_mass=None)
```

**The stored form is bf16.** The routing sides have one consumer, and it reads bf16, so
that is what the shipped producer's solve WRITES -- there is no fp32 plane between the
solve and the operator, and no narrowing pass. `read_simplex()` divides in at least
fp32 whatever the levels are stored in, because a bf16 quotient of a bf16 sum is three
roundings where one is forced. The bundle's own invariant is only that every level
shares ONE dtype, so a bundle in any dtype is legal; bf16 is what the shipped producer
emits and what every kernel-vs-oracle tolerance in the tree is derived against
(`tests/oracle/tolerances.py`).

`RouteFactors` is the op's first operand and the whole producer contract: **any callable
returning a well-formed one is a producer.** There is no producer base class, no
per-level tier and no assembly gateway — none of them was a boundary anything checked.
What is checked is the bundle's own invariants, and they are STRUCTURAL rather than
provenance-based.

**The bundle is a record of tensors, and it is constructible.** Its fields are the
stored levels, the side gains, the read mass and the topology they are shaped by —
nothing else, and in particular no declaration a caller could pair with tensors that do
not describe it. `RouteFactors.__post_init__` therefore checks the STRUCTURE rather than
the provenance: one level per topology level at that level's width, one shared
`[B, T, H]` prefix, one device, one dtype,
`g_write` and (when present) `read_mass` shaped `[B, T, H]`. There is no `g_read`
field at all — the ratio readout fixes the read-side gain to 1 identically, so there
is nothing for a column to override. A test that plants a
specific routing pattern builds one directly and is held to exactly those invariants.
Unit-sum-ness is deliberately not among them: it is a device reduction per level per
call, it is the shipped producer's own output guarantee, and conformance fixtures plant
non-canonical amplitudes on purpose. The op still type-gates on the class, so handing it
bare tensors is still a clean refusal.

A producer also owns the topology (`routes.topology`, `routes.widths`), because that is
a fact about the map rather than about the layer
wrapping it. It does NOT own `d_v`: the value width is the width of the state each leaf
holds, which is a fact about the value stream — the layer sizes `v_proj`/`o_proj` with
it and the op reads it off `v.shape[-1]` at call time.

**The side gain is COLUMNS OF THE ROUTER GEMM.** The gain is a function of the read
stream and so are the routing logits, so `route_W` carries both: the routing slots
first, the gain's span last. One parameter, one cast of the source stream, one
projection. `gain=False` removes the column rather than pinning it, so a disabled gain
is a narrower parameter and not a parameter with an identically zero gradient.

**The side gain is routing's, and it rides the bundle.** Two things put it there
rather than on the layer. There is no read-side gain at all — the ratio readout fixes
it to 1 identically — so the write side carries the only gain column there is, and
a layer-owned gain producer would have to be told that in order to know its own
columns, which is the coupling running backwards. And the gain is where the
per-level mass product lands: the producer folds
each level's mass into it, so the gain and the routing mass are the same number by the
time any consumer sees either. It rides the bundle rather than arriving as a third op
argument because `RouteFactors.tensors()` is the autograd-reachability enumeration, and
an operand outside that flat list is invisible to autograd — the exact failure the
single autograd boundary exists to make impossible.

### 2.2 The mass fold, and what each duty does with it

Levels need not be unit-sum: each is factored into `(mass, simplex)`. The two sides then
take the mass differently, because the two duties use their levels differently. The
WRITE side APPLIES it — the stored level is the simplex and the magnitude folds into
`g_write`, which is where a per-level magnitude belongs, and the decay clock is defined
on the stored level. The READ side CARRIES it — the readout is a ratio, so a per-token
scale common to every leaf is homogeneous of degree zero through it, and
`RouteFactors.read` is stored with the mass still on it while `RouteFactors.read_mass`
holds the per-token product (`None` when there is none). `RouteFactors.read_simplex()`
divides it back out for every consumer defined on the simplex instead: the fp64 oracle
and its input contract, the reference recurrence. The decode arm is defined on the
simplex too, and divides INSIDE its operand fold off the same predicate --
`RouteFactors.read_needs_normalization()`, which `read_simplex` reads as well -- because
a decode step's whole budget is launches
([`internals/decode/decode_fold.md`](internals/decode/decode_fold.md)).

**The read mass reaches the readout as a scaled floor.** The one term that does not
scale with a ratio is its floor, so it is handed over pre-scaled: the consumer computes
`m·num / (m·den + READOUT_EPS·m)`, which is `num / (den + READOUT_EPS)` exactly — the
same function, not a differently conditioned one. The alternative — dropping the mass
and leaving the floor alone — is an *effective* floor of `READOUT_EPS / m`, and the
union read mass runs as low as 7e-4 across the gain sweep, so it is a change to the
definition rather than to its conditioning. `READOUT_EPS` is a DEFINITION constant
(`rola/ops/constants.py`); the scaled floor is what keeps it one.

**The mass is only COMPUTED where it is not declared to be 1, and only there.** Each
`(level, duty)` cell answers `simplex_output` — "does this cell's own output already
land on the simplex?" — and where it answers yes, the mass is 1, so its `sum` is
omitted, it contributes nothing to the product, and the write side's `div` goes with it.
The answer is a declaration read off the activation registry
(`rola/routing/activations.py`) through the level's routing form, never a runtime test
of the tensor: a `sum(...) == 1` check is the very reduction the skip exists to delete,
and it would make the bundle's arithmetic depend on its data rather than on its
configuration. It is asked per *duty* and not per *activation*, because a routing form
may hand one duty a share of a solve rather than the whole of it — union routing's read
half is exactly that, so it answers `False` while the entmax it names answers `True`.
A FOREIGN producer folds its own mass, because it returns the finished bundle: the
opaque row's `simplex_output = False` is what keeps `normalized` from being read
circularly for a cell nobody vouched for, and it reaches the shipped fold only through
a topology a foreign producer declared. Measured across the shipped census (`D` in 1..4, fp32 and fp64), every cell answering
`True` has `max |sum − 1| ≤ 2.4e-7` in fp32 and `≤ 4.4e-16` in fp64.

**The producer owns packing.** The solve emits `[BH, T, width]`, which is already what
the kernel's view wants, and the public `[B, T, H, width_l]` layout is a permute of it —
so exactly one of the two can be the materialized tensor, and it is the kernel's. The
public tensors are views. Materializing the public layout instead and folding it back is
an inverse copy pair. It was priced at 2.82 % of the whole step at `L = 4096`
(`workflows/refactor/p12_lane3.md`), and the same harness measures it directly against
the production call it rides: 3.085 ms vs 2.732 ms at `L = 4096` and 0.942 ms vs
0.835 ms at `L = 1024` — 12.9 % and 12.7 % of production-plus-fold, at both lengths.

### 2.3 The shipped producer

```python
RouteProducer(levels, *, hidden_size, num_heads,
              bias=False, gain=True, gain_bias_init=None, gain_config=None)
```

A learned packed per-head router projection, solved with the exact production
entmax/softmax closed form, folded, gained, and returned as one bundle. Construction
validates the whole assembly — widths in range, a buildable topology, a gain layout —
so an unbuildable model fails on the line that built it.

**`levels` is the configuration, and it is a LIST.** One entry per routing level,
outermost first, each carrying that level's WHOLE spec:

```python
IndependentRouting(width, read, write)   # two logit slices, two activations, two solves
TiedRouting(width, op)                   # one logit slice, one activation, one solve
UnionRouting(width, alpha=1.5)           # one solve, support split
```

`TiedRouting` is a first-class type: the read factor IS the write factor, the same
tensor, because there is only one duty's worth of output. It differs from
`UnionRouting`: union routing splits ONE shared support across two roles, so the read
and write factors differ (a share each); tied routing hands both duties the SAME tensor
outright. `op` is any registry activation — a tied *softmax* level is legal and reports
dense on both sides. The vocabulary is exactly these three types — there is no
compatibility flag on `IndependentRouting` for the tied case.

Depth, leaf count `N`, the packed router width, the per-level logit offsets and the
topology are all DERIVED from that list. None of them is a second input, so none of them
can disagree with it. The five spellings the shipped configurations use:

```python
dense_routing(width)                      # softmax on both duties
union_routing(width, alpha=1.5)           # one exact entmax solve, shared support
split_routing(width, alpha, sparse_duty)  # entmax one duty, softmax the other
tied_routing(width, activation=entmax(1.5))  # one activation, one solve, shared tensor
uniform(depth, level)                     # `depth` levels spelled the same way
```

`uniform` is a list builder, not a broadcast: a producer takes a list and nothing else,
because a broadcast is a rule the reader has to know before they can read the widths off
the call.

**Every level is one launch group, not one launch each.** The levels share a packed
`route_W` and are dispatched per level inside the solve kernel with no privileged level,
so a model pays one GEMM and one solve group for the whole topology. Levels sharing an
operator, an alpha, a width class and an operand set share a single launch; levels that
do not are genuinely different `(operator, source, destination)` triples and issue
separately.

Activation *properties* are never supplied by a caller — they come from the closed
registry in `rola/routing/activations.py`, keyed on the tag. Metadata a caller could
supply is metadata a caller could supply wrong, and a validator reading it would be
checking a claim instead of a fact. Activations stay width-free: an activation is a
function on a row of any length, so a width on one would be a second place the same
number lives.

### 2.4 Foreign producers

Any callable returning a well-formed bundle works, with no registration and no tier.
This is sound because no producer-specific execution path exists: the consumer eats
tensors and cannot tell what produced them — no producer identity reaches the kernel's
parameters at all. Support bits are computed FROM the tensors, so a foreign producer
changes nothing about correctness. Its only cost is inside its own module. The layer's
`routes` slot checks STRUCTURE for the same reason: callable, plus the three facts the
layer reads (`hidden_size`, `num_heads`, `topology`).

A level filled by a producer that does not name one of our activations is *opaque*: it
declares `OpaqueRouting(width)` and reads the `("opaque", None)` registry row, whose
columns are argued in `rola/routing/activations.py`. What that costs it is the
activation questions the topology can answer about it — ratification, in particular —
and nothing else.

### 2.5 The one modelling consequence to know

The decay clock is defined on the STORED (normalized) level. A level whose write-side
mass differs from 1 has therefore moved magnitude out of the clock and into `g_write`.
That is the defined semantics rather than a defect, but it is a choice a level's author
is making, so it is stated here rather than buried.

## 3. The decay-source slot

Where the recurrence's forgetting rates come from is a CHOICE, and the recurrence cannot
tell the answers apart:

```python
rola.ConstantDecay(rate, widths=..., num_heads=...)                    # user-fixed
rola.LearnedDecay(rate, widths=..., num_heads=..., scope='state')      # 'state' | 'global'
```

Both emit a `LeafMassDecay` whose `dials[l]` is `[H, width_l]` in `[0, 1)`. A source is
validated against the topology at layer construction. Duck-typed like a producer:
declare `widths` and `num_heads`, return a `LeafMassDecay`.

`state` and `global` are the two IDENTIFIABLE resolutions: under `keep = (1 - prod_l
delta_l)**c` (one write-conditioned clock per leaf), a rate only ever enters the
recurrence through that per-leaf product, so per-digit and per-head are the two degrees
of freedom the product structure lets the recurrence actually distinguish.

`ConstantDecay` set high is a legitimate performance lever: the state forgets fast and
the realized routing shows it, through exactly the same door a learned source uses.
Nothing about it is a special case. (Today the chunk arm has no decay and refuses it
— §1.1 — so the lever reaches production only through the decode arm, which does have
it, until the kernel-side decay retirement is reopened.)

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
RoLA(routes, *, d_v=64, decay=None, layer_idx=None, expert=None)

y, continuation = layer(hidden_states, continuation=LayerContinuation(state=rola.state()))
y, continuation = layer(next_tokens, continuation=continuation)
y, _ = layer(hidden_states)                            # stateless -> (y, LayerContinuation(state=None))
```

`forward(x, attention_mask=None, continuation=None) -> (y, LayerContinuation)` — ALWAYS
the container. See §1.2; a bare `RoLAState` or the retired `(state, conv_rings)` tuple
is refused by name rather than adopted.

`routes` is the only positional argument, and it is the only declaration of
`hidden_size`/`num_heads`: the layer reads both off the producer rather than taking a
second copy to check against. `d_v` is the per-head value width — the layer's, because
it sizes `v_proj`/`o_proj`; the default is the width the chunk matrix is built at.

`value_conv`/`route_conv` (a causal short convolution over the value/routing-source
stream) were DELETED at K45 (docs/internals/DELETIONS.md): they were never reviewed as
a mechanism and are not paper-1 scope. Re-adding a convolution hook is a field on
`LayerContinuation`, not a return to the old ad hoc ring tuple.

The layer adds exactly one clause the op cannot know: `self.training` REFUSES,
independently of gradient reachability, because reentrant gradient checkpointing runs
its first pass under `no_grad()` with the module still in training mode — and a
forward that pass could take would be a forward no backward can follow.

`layer.last_execution_backend` names the arm the last forward took — `'cuda'` or
`'decode'`, and `None` until one completes. It is a fact about the branch rather than an
inference from a timing, and it is published for exactly that: a benchmark process
cannot spy on the call it is timing, and wall-clock cannot tell two arms apart. It is a
SCALAR, not a tally — a caller that needs how many times an arm ran spies on the arm.

### 4.1 The bf16 projection arm

`v_proj` and `o_proj` run as bf16 GEMMs
with weights cast per call, and `v` FLOWS bf16 straight into the op. The arm is
UNCONDITIONAL because the op's envelope makes it so: there is no CPU lowering, so a
non-CUDA forward refuses at `rola_op` and no call off CUDA ever observes a projected
value. Round-tripping
it through the module dtype was MEASURED to cost two elementwise casts per call — a `v`
downcast and a `y` upcast, ~12-13% of a small cell's op time — which is why the layer
narrows once, at the projection, rather than at the op boundary. The GEMMs use cuBLAS's
fp32 compute type (the accumulate-in-fp32 ruling); the global
reduced-precision-reduction knob is deliberately left alone, so this is the same
environment the FlashAttention comparison arm runs in.

There is no dtype exception left on this arm: `value_conv` — the one case that used to
round-trip `v` back to the module dtype ahead of a convolution — was DELETED at K45
(docs/internals/DELETIONS.md). `o_proj`'s operand cast is exact regardless: `y` leaves
the kernel already bf16-rounded.

## 5. The expert surface

`rola.expert` holds everything BELOW the contract line — the dispatch's own answer,
the pins over axes the op otherwise decides, and the fine-grained gain layout behind
the layer's one public flag:

```python
from rola.expert import PlanOverrides, envelope_refusal
layer = rola.RoLA(routes, expert=PlanOverrides())
```

| name | what it is |
|---|---|
| `PlanOverrides` | the pins: `paging` |
| `envelope_refusal(routes, v, decay=None)` / `require_envelope(...)` | whether a call has an arm, asked before making it — the refusing clause in its own words, or `None`. `v` is named rather than assumed: the `d_v` clause is a fact about the value stream, so the answer is a property of the configuration AND of `v.shape[-1]` |
| `requires_backward` | the gradient-reachability predicate the envelope's first clause is built from |
| `GainConfig` | the two-boolean gain layout behind the public `gain` flag |

The contract for everything there is **stable construction, advisory contents**:
build these objects and pass them; do not assert on their fields. What a `Dispatch`
reports is the answer to "how will this call run", and that answer moves with the
machine and with the built matrix. The model facts it carries are the exception,
because they are not decisions.

**Every pin is a declared restriction rather than a dial, and each refuses by name
rather than being ignored.** `state_block` (`BC`) is not
host-selectable: the chunk arm's `BC` belongs to the built arm the manifest carries
for a topology, so any value but `None` is refused at the call — a knob that
silently does nothing is how a caller comes to believe a launch was shaped a way it
never was.
the fp64 reference path. `paging` overrides the MAPPING only: `None` is the paged
default, `False` pins the dense plane the bit-identity gate compares against, and
`True` is refused because paging is on by default and bitwise invisible.

**There is no plan object and no derive/execute seam.** `Plan`, `Schedule`,
`derive_plan` and `execute` were the tiled consumer's, and went with it in P67 D2
([`internals/DELETIONS.md`](internals/DELETIONS.md)). The geometry they carried is
now a property of the built arm rather than of a per-launch derivation, so there is
no launch-shaping object for a caller to hold, synthesize or staleness-test. What
replaces the adversarial use they enabled: a conformance family drives
the envelope's own rule (`rola.engine.rules.envelope`) directly and declares the refusal
clause it expects, and
`envelope_refusal` answers the same question a synthetic plan used to be built to probe.

## 6. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** RoLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.

## Gradients are not bitwise reproducible

`rola_op` trains, and its gradients are **not** run-to-run deterministic. Both scans
assemble their outputs through fp32 `atomicAdd` fan-ins whose order is the
scheduler's, and the two scans walk DIFFERENT partitions of the token axis by
construction — each closes its chunks by its own walk — so nothing about the reverse
pass is bit-reproducible even on identical inputs on identical hardware. The forward
already had this property through the same fan-ins; the backward inherits it in a
worse regime (more contributors, narrower destinations).

Two consequences, stated rather than discovered:

* a training run cannot be reproduced bit-for-bit by re-running it, only
  statistically;
* a gate on a gradient must be a tolerance, never an equality.

**Training is STATELESS.** A call carrying both a gradient and a `state=` plane is
refused by name: the state is caller-owned and mutated in place, and the reverse
pass has no `d initial_state` — it rebuilds from zero. **Decode has no reverse pass
at all**, so a grad-bearing single-token continuation is refused too. And the
reverse pass carries a NARROWER arm matrix than the forward: a grad-bearing call on
an arm it does not carry is refused at the boundary rather than three launches later.
