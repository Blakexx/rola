# The API

RoLA is linear attention under a structured feature map. That is not a slogan about the
paper; it is the specification of this API. A head's recurrent state is expanded into
many sub-states that share the head's projections, and a learned routing decides which
sub-states each token reads and writes — so the public contract is *linear attention's*
contract, with exactly **one** new concept: the feature map.

A user who knows linear attention should find one unfamiliar thing on this page.

```python
import rola

routing = rola.Routing(rola.uniform(2, rola.union_routing(64, alpha=1.5)))
decay   = rola.LearnedDecay(2 ** -8, widths=routing.widths, num_heads=8)

layer = rola.RoLA(routing, hidden_size=1024, num_heads=8, d_v=64, decay=decay, layer_idx=0).cuda().eval()
y, _ = layer(hidden_states)                     # stateless: (y, None)
```

or, without the layer, from levels you computed yourself -- the format's own utilities build the bundles:

```python
factors = rola.RouteFactors.from_levels(topology, read_levels, write_levels)
y, _ = rola.rola_op(factors, rola.Values.from_projection(v), decay())
```

---

## 1. The op

```python
rola_op(routes: RouteFactors, values: Values, decay: LeafMassDecay | None = None, *,
        state=None, expert=None) -> (y, state)
```

`routes` is the `(q, k)` pair after the feature map. `values` is the value stream in the
internal format, `Values.from_projection(v)` over a `[B, T, H, d_v]` projection (§4.1).
`decay` is the recurrence dial. `y` is `[B, T, H, d_v]`.

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
commitment. Two calls read and refill it off the
step's own path: `rola.engine.dags.decode_dag.pool_headroom(state)` (the slack slots the
thinnest batch-head still has) and `refill_pool(state)` (restock what the device spent).
See [`internals/state.md`](internals/state.md#the-slack-pool) and
[`internals/paging/paging.md`](internals/paging/paging.md#the-slack-pool).

There is no plan argument, no configuration argument, and no statistics argument. Those
are not model facts — they are how the op RUNS, and the op decides how it runs, per
call, from the routing it was actually given. They live behind
[`rola.expert`](#5-the-expert-surface).

**Any level width, any `d_v`.** A `Routing` level's width and a layer's `d_v` are
never required to be a power of two or a shipped value — host-side padding at the op's
own call surface serves the rest, with no kernel change
([`docs/internals/padding.md`](internals/padding.md)): the prefill's carry leg and the
decode step pad below the op, and the intra reads the rows at their own widths.

### 1.1 The built envelope IS the surface

The prefill is the facts pass, the carry and the intra over a BUILT MATRIX
(`rola.engine.dags.prefill_dag`). **Everything outside it raises.** There is no second arm behind this surface: the
fp64 oracle
(`rola.ops.naive.naive_rola`) is the executable spec — what tests and ratification
call directly — and it is not reachable by dispatch, because a dispatch to it at
inference is a silent ~1000x-slower route shipping unnoticed, which is the exact thing
the no-fallback rule forbids.

| axis | the built matrix | outside it |
|---|---|---|
| a gradient anywhere in the operands | none | `BackwardNotImplemented` |
| the module's mode (layer only) | `eval()` | `BackwardNotImplemented` |
| device | CUDA | refused |
| `d_v` | the intra's value width (`rola.ops.intra.VALUE_WIDTH`), which the binary's carry arms at the depth must reach | refused |
| widths | one or two levels, each a whole number of 16-digit granules up to 256, any mix (`rola.ops.intra.admits`) | refused |
| `T` | any (a partial last window runs inside the kernels; the layer's segments are whole windows, §4.2) | — |
| `decay` | not implemented in the prefill (decode has it) | refused |
| an unratified activation | — | refused |

Ask before calling: `rola.expert.envelope_refusal(routes, v, decay)` returns the
sentence the refusal will carry, or `None`. `rola.expert.require_envelope(...)` is the
raising form the op itself uses.

**The envelope is narrow, and that is stated rather than softened.** Shapes outside the
matrix are refused, and there is NO TRAINING PATH through this surface until the native
backward lands; what widens it is in [`docs/open-work.md`](open-work.md).

`rola_op` is the **single autograd boundary**, and it is a refusal rather than a
branch: the failure it prevents — a forward-only kernel reached from a gradient
context returns a tensor with no `grad_fn` and severs the graph silently — is
structural, not a bug in any one call site. The prefill's autograd node, whose backward
raises `BackwardNotImplemented`, is the last line, loud if the first is ever bypassed.

### 1.2 Continuation is LAYERED

The op's state is the paged tensor and nothing else
([`docs/internals/state.md`](internals/state.md)). Each level of the stack returns its
OWN thin bundle, and the rule is a membership test:

> **An object holds exactly the recurrences ITS level introduced.**
> Op: the routed state. Layer: `LayerContinuation`. Model: the per-layer bundles.

So `RoLA.forward` returns `(y, LayerContinuation)` — always the container — and takes
one back. `LayerContinuation.state` is the op's paged tensor; it is the container's
only field today, and any future layer-level recurrence is a field ON IT, never on the
op's state. The container is the contract in both directions: a bare `RoLAState` or a
`(state, conv_rings)` tuple is refused by name, because accepting a lookalike would make
the layer's own fields silently optional to carry. Keeping each level's object pure is
what leaves layer luggage nowhere to land inside the op's state.

### 1.3 TWO CONTRACTS, ONE TRANSLATOR

There are two boundaries in this library and they accept different things. Confusing
them is what produces a kernel that quietly pads, masks or half-serves a shape, so they
are stated apart.

| | what it accepts | what it does with the rest |
|---|---|---|
| **the API** (this page: `rola_op`, `RoLA`, `rola.state()`) | any level width `b_l >= 2`, any `d_v`, any length `L` | **pads** — level logits to the next power of two at or above 16 with `-inf`, `v` to the next shipped `DV` with zero columns |
| **the kernel** (`rola.ops.carry`, `rola.ops.decode`) | only the state descriptor's strict shape: `B_l` a power of two at or above 16, a shipped `DV`, `N` a multiple of 16, the internal format's rows and values (§2.1, §4.1) | **refuses**, by name — it never pads and never masks |

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
(`tests/oracle/test_padding_seam.py`). A logical-width field in a kernel entry
signature is a lint finding, because a kernel that could read one could pad.

**What the descriptor fixes.** A state binds its format at its first call — `D`, the
per-level `B_l`, the canonical leaf order, the page rectangle, `DV`, and the stored form
(`rola._state.STATE_DTYPE`, [`internals/common/state_page.md`](internals/common/state_page.md)) — and every later call
is *checked* against it rather than assumed
to fit. A call that disagrees is refused at the seam, naming the field, instead of
addressing somebody else's bytes inside a kernel. Re-laying out a state is an explicit
op, never a side effect of a call. See
[`internals/state.md`](internals/state.md#format).

**Status.** `rola_op` validates the call against the envelope and walks the prefill DAG.
The DAG's carry leg (`rola.ops.prefill`) is not yet re-pointed at the rebuilt carry's
launch surface (`rola.ops.carry.carry`), so a prefill does not yet run end to end
([`docs/open-work.md`](open-work.md)).

## 2. The feature map

### 2.1 The bundle IS the contract, and it is THE INTERNAL FORMAT

```python
RouteFactors(topology, amp_read, amp_write, scale_read, scale_write, source=None, sparse=None)
RouteFactors.from_levels(topology, read, write, *, sparse=None)        # the framework's epilogue
RouteFactors.from_rows(topology, amp_read, amp_write, scale_read, scale_write, source=None, *,
                       sparse=None)                                    # rows already in the format
```

**The stored form is the internal quantized format** (ruled 2026-09-26). Each level's rows are
UNIT-MAX -- u8 with the row's largest digit at 255, except a DENSE level's rows, f16 at 1 (the write side ruled
2026-10-04, the read side 2026-10-07); a sparse level's zero digit is dead; a dense level is never zero -- its rows are
stored with the floor taken, 2^-13 of the row's max (the dense floor, ruled 2026-10-03/04;
`docs/internals/facts/liveness_contract.md#floor`); an all-zero row is a dead token at that level -- with the row's maximum
carried beside them as its fp32 scale (`amp_read`/`scale_read`, `amp_write`/`scale_write`, one
tensor per level, `[B, T, H, width]` read and `[B, T, H_w, width]` write, starting on and stepping by 32 bytes: the
row alignment law, `docs/internals/facts/liveness_contract.md#alignment`). Every level is stored as rows, once (ruled 2026-10-01: the
intra and decode read rows, and the carry reads rows through the facts pass's words). `sparse` is
`(read, write)`, a bool a level: the levels whose support the facts pass words because they may hold
exact zeros worth skipping. It is set at production -- the producer flags a side whose activation
has exact zeros, and `from_levels`/`from_rows` default to that rule -- and read by every consumer;
no consumer derives it from the activations. A flag cannot disagree with the rows: a sparse-flagged
level with no zeros costs its words, and a dense-flagged level with zeros is read whole.
EVERY SIDE HAS A DENSE LEVEL: a `sparse` that flags every level of a side is refused -- a side's dense levels are
the carry's operand on that side and carry the floor (`docs/internals/engine/rules.md#carve`).
<a id="grouped-queries"></a>
GROUPED QUERIES: the read side is the `H` query heads' (`heads`), the write side the `H_w` write heads' (`write_heads`):
its rows and scales (`[B, T, H_w]`), and with it the values, the decay dials and the state. `H_w` divides
`H`, and query head `h` reads write head `h // group` (`group = H / H_w`, consecutive: PyTorch SDPA's `enable_gqa`
order); `H_w = H` is the ungrouped layer. The oracle reads grouped bundles (`tests/oracle/test_oracle_grouped.py`: a
grouped run is the ungrouped run with each write head's side repeated for its query heads). The facts pass reads them,
a write head's read side one union a window over its query heads' rows (`docs/internals/facts/facts.md#lists-out`), and
a carry reads them a CTA a write head, its query heads' rows with it, as does the intra, so the prefill reads them. The decode step and the carried state read one head count
today, and each refuses a grouped bundle by name (`RouteFactors.refuse_grouped`).
`RouteFactors.rows(level, side)` is a level's rows, and its scale the row's max: the level as its producer handed it
over, whatever its sum ([§2.2](#22-the-scaled-block-and-the-gain)). This is what
the int carry, the intra and the decode kernels read, and it is written by the framework's producer
alone: the fused kernel's epilogue, and the reference's (`rola/routing/quant.py`, `RouteFactors.from_levels`),
the first gated against the second where it is built. The op's oracle takes the
bundle itself as its input.

**Nothing floating is stored.** `RouteFactors.dequantized()` computes the levels, `rows x scale`
in fp32, at the call: a consumer that still computes on floating levels calls it at its seam, so
it computes on the quantized VALUES and nothing in the tree sees a number the kernel would not;
the call goes when that consumer reads the rows. `source` is the producer's floating levels, held
ONLY when they require a gradient -- the anchor `dequantized()` steps the gradient straight
through the rounding to (`quant.straight_through`); an inference bundle holds the rows and nothing
else.

**The bundle is a record of tensors, and it is constructible.** Its fields are the
stored levels, their scales and the topology they are shaped by — nothing else, and in
particular no declaration a caller could pair with tensors that do not describe it.
`RouteFactors.__post_init__` therefore checks the STRUCTURE rather than the provenance: one
level per topology level at that level's width, one `sparse` flag a level a side, one shared
prefix a side, one device, one dtype, each scale fp32 at its side's prefix. A test that plants
a specific routing pattern builds one directly and is held to exactly those invariants.
Unit-sum-ness is not among them: no level need sum to 1 ([§2.2](#22-the-scaled-block-and-the-gain)).
The op still type-gates on the class, so handing it bare tensors is still a clean refusal.

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

**The write gain is the producer's, and it rides write level 0's scale.** There is no
read-side gain at all — the ratio readout cancels a per-token read factor but in the floor —
so the write side carries the only gain column there is, and a layer-owned gain producer would
have to be told that in order to know its own columns, which is the coupling running
backwards. The producer folds the gain into write level 0's scale, in the job that writes that
level, so the format has no gain field and no consumer reads one; it rides the bundle's own
tensors, which `RouteFactors.tensors()` enumerates for autograd.

### 2.2 The scaled block, and the gain

A level is numbers in a representation, not a distribution (ruled 2026-11-07,
[`CONSTITUTION.md` R23](CONSTITUTION.md#r23)). `q` (read), `k` (write) and `v` are each a
SCALED BLOCK: unit-max rows in a FORM that states its storage type, its signedness and its
floor -- u8 nonnegative at 255 (a sparse routing level, a zero dead), f16 nonnegative at 1
with the dense floor taken (a dense routing level), f16 signed in `[-1, 1]` (the values) --
with the row's max as its fp32 scale. The forms are `rola.routing.quant`'s `Form` (`SPARSE`, `DENSE`, `SIGNED`),
quantizing into and dequantizing out of each written once on it, and the factory refuses an entry its form does not
hold -- a negative or non-finite routing level, a non-finite value -- with one host read a call. A side's shared scale is the product of its levels'
scales, STORED FACTORED: each level's scale is written by the producer job that quantizes that
level and multiplied by the reader as it loads them, so no job waits on another. The op every
consumer computes, and the fp64 oracle with them:

    W(t, s) = prod_l k_l(t, d_l(s))          R(t, s) = prod_l q_l(t, d_l(s))
    S_t(s)  = S_{t-1}(s) + W(t, s) v(t)      den_t(s) likewise with v = 1
    y(t)    = sum_s R(t, s) S_t(s) / (sum_s R(t, s) den_t(s) + eps)

No level is required to sum to 1, on either side, and no flag says whether one does. A
producer's magnitude is its own: ours, the write gain `softplus(gain logit + bias)`, is folded
into write level 0's scale (§2.1); a caller's own levels carry whatever magnitude they have into
their scales, `RouteFactors.from_levels` quantizing each against its own max and nothing else.

**The read scales are kept, for the floor.** A per-token read factor common to every leaf
cancels in the ratio everywhere but the floor, so the read scales' product `r` is read only
there: the prefill readout is `num' / (den' + eps / r)` (`rola/ops/prefill.py`'s `readout`),
which is the formula above exactly, and it is what makes `eps` mean the same thing on every
routing. `READOUT_EPS` is a DEFINITION constant (`rola/ops/constants.py`). Decode computes the
same `y`: neither normalizes a read level.

**The decay computes its own shares** ([§3](#3-the-decay-source-slot)): its clock is each leaf's
SHARE of the token's write, `c = prod_l k_l(d_l(s)) / sum_d k_l(d)`, a row sum a level and
scale-free, so no stored level need be a simplex and a gain folded into a scale is invisible to
it.

**The producer owns packing.** The solve emits `[BH, T, width]`, which is already what
the kernel's view wants, and the public `[B, T, H, width_l]` layout is a permute of it —
so exactly one of the two can be the materialized tensor, and it is the kernel's. The
public tensors are views. Materializing the public layout instead and folding it back is
an inverse copy pair. A harness measures it directly against the production call it
rides: 3.085 ms vs 2.732 ms at `L = 4096` and 0.942 ms vs
0.835 ms at `L = 1024` — 12.9 % and 12.7 % of production-plus-fold, at both lengths.

### 2.3 The declared routing, and the producer

```python
Routing(levels, *, gain=True, gain_bias_init=None, gain_config=None, bias=False)
RoLA(routing, *, hidden_size, num_heads, d_v=64, decay=None, layer_idx=None, ...)
```

RULED 2026-09-29: the producer is a DECLARED CONFIGURATION. A `Routing` is data -- the levels, the write gain and
the projection bias -- and the layer compiles it once into its producer plan and owns every weight it implies: the
packed per-head routing projection `route_W`, its bias, the write gain's bias, `v_proj` and `o_proj`. One producer
call (`rola.ops.produce.produce`) turns a stretch of tokens into the bundle -- the projection, the exact
entmax/softmax solves, the write gain, the unit-max rows and the f16 values -- and on CUDA with no gradient
that call is ONE LAUNCH of the fused producer kernel, the bundle's rows and scales views of the planes it writes --
a kernel not built on this line, so the call raises `rola.ops._ext.KernelNotBuilt` there
([`open-work.md`](open-work.md)). The torch composition behind the same call is the reference: fp64, the host,
and -- until the native backward -- the path a gradient takes. Construction validates the whole declaration --
widths in range, a buildable topology, a gain layout -- so an unbuildable model fails on the line that built it.

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

`uniform` is a list builder, not a broadcast: a routing takes a list and nothing else,
because a broadcast is a rule the reader has to know before they can read the widths off
the call.

**Every level, every head and the values are one launch.** The plan compiles into the
kernel's job table -- a job per head's solve and per group of value heads -- so a model pays
one launch for the whole topology, whatever its levels' operators, alphas and widths.

<a id="projection-weight"></a>**The projection reads a laid-out copy of `route_W` when no gradient flows.** The GEMM's
operands are bf16 and the kernel reads each column's hidden run contiguous (`hidden_contiguous`), so an inference
step would otherwise cast the whole fp32 parameter and re-lay it out on every call -- measured at
~55 us of GPU a decode step at `H = 16`, `(256, 256)`, more than the decode kernel. The copy
is made once and reused while the parameter's version counter, storage, device and the
projection dtype are unchanged; an in-place update (an optimizer step, `load_state_dict`)
moves the version and so the copy. A step that trains casts inside the graph, since its
gradient must reach the parameter and the parameter changes every step anyway. (Writing
through `route_W.data` bypasses the version counter, as it does for every autograd cache.)
The layer's value and output projections read their bf16 weights the same way (another ~6 us
a step and two launches at `H = 16`); the one mechanism is `rola/_inference_copy.py`.

Activation *properties* are never supplied by a caller — they come from the closed
registry in `rola/routing/activations.py`, keyed on the tag. Metadata a caller could
supply is metadata a caller could supply wrong, and a validator reading it would be
checking a claim instead of a fact. Activations stay width-free: an activation is a
function on a row of any length, so a width on one would be a second place the same
number lives.

### 2.4 Production lives in the framework

RULED 2026-09-26, and 2026-09-29: there is no duck-typed producer, and the layer takes no callable feature map. A
new kind of production is added to the framework -- its activation to the closed registry, its solve to the
producer, its reference and its oracle -- and what it returns is the internal format. A caller of the ops with its
own levels builds the bundles through the format's own utilities (`RouteFactors.from_levels`,
`Values.from_projection`), which are the same epilogue the producer applies; nothing else constructs a bundle the op
admits: the format's fields are required, checked structurally, and a bundle assembled from bare floating levels is
refused by name. Correctness is still producer-blind -- every support bit is derived from the rows (a zero digit is
dead) and nothing about a producer's identity reaches a kernel -- but the FORMAT is ours, because it is what every
kernel reads and what the oracle gates against.

A level filled by a production that names none of our activations is *opaque*: it declares
`OpaqueRouting(width)` and reads the `("opaque", None)` registry row. What that costs it is the
activation questions the topology can answer about it -- ratification, in particular -- and
nothing else.

### 2.5 The one modelling consequence to know

The decay clock is each leaf's SHARE of the token's write ([§2.2](#22-the-scaled-block-and-the-gain)),
scale-free: a magnitude a write level carries -- a producer's gain, a level summing past one --
scales the deposit and not the clock. That is the defined semantics rather than a defect, but it
is a choice a level's author is making, so it is stated here rather than buried.

## 3. The decay-source slot

Where the recurrence's forgetting rates come from is a CHOICE, and the recurrence cannot
tell the answers apart:

```python
rola.ConstantDecay(rate, widths=..., num_heads=...)                    # user-fixed
rola.LearnedDecay(rate, widths=..., num_heads=..., scope='state')      # 'state' | 'global'
```

Both emit a `LeafMassDecay` whose `dials[l]` is `[H, width_l]` in `[0, 1)`, `H` the write heads (the decay is the
state's: [grouped queries](#grouped-queries)). A source is
validated against the topology at layer construction. Duck-typed like a producer:
declare `widths` and `num_heads`, return a `LeafMassDecay`.

`state` and `global` are the two IDENTIFIABLE resolutions: under `keep = (1 - prod_l
delta_l)**c` (one write-conditioned clock per leaf, `c` the leaf's share of the token's write:
[§2.2](#22-the-scaled-block-and-the-gain)), a rate only ever enters the
recurrence through that per-leaf product, so per-digit and per-head are the two degrees
of freedom the product structure lets the recurrence actually distinguish.

`ConstantDecay` set high is a legitimate performance lever: the state forgets fast and
the realized routing shows it, through exactly the same door a learned source uses.
Nothing about it is a special case. The prefill refuses a decay today (§1.1), so the
lever reaches production through the decode step.

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
RoLA(routing, *, hidden_size, num_heads, d_v=64, decay=None, layer_idx=None, expert=None, memory_budget=None)

y, continuation = layer(hidden_states, continuation=LayerContinuation(state=rola.state()))
y, continuation = layer(next_tokens, continuation=continuation)
y, _ = layer(hidden_states)                            # stateless -> (y, LayerContinuation(state=None))
```

`forward(x, attention_mask=None, continuation=None) -> (y, LayerContinuation)` — ALWAYS
the container (§1.2).

`routing` is the declared `Routing` (§2.3), and the layer owns every weight it implies.
`d_v` is the per-head value width — the layer's, because it sizes `v_proj`/`o_proj`; the
default, 64, is the intra's value width.

<a id="one-token-step"></a>**One token of a carried sequence is `_step`, then the verdict.** `RoLA._step(x, state)`
is the whole token with no host read inside it -- the producer's one launch, the gated decode
step, the output projection -- so a caller can capture it in a CUDA graph and replay it a
token. A batch-head whose write atoms the state can neither address nor admit skips itself;
`forward` then asks `growth_pending(state)`, admits and steps again, which advances every
sequence exactly once (`docs/internals/decode/decode.md#capture`). Reading the verdict is a
4-byte copy and a sync, which a serving loop pays at the sync that samples its token.

A convolution over the value or routing-source stream, if one is added, is a field on
`LayerContinuation`.

The layer adds exactly one clause the op cannot know: `self.training` REFUSES,
independently of gradient reachability, because reentrant gradient checkpointing runs
its first pass under `no_grad()` with the module still in training mode — and a
forward that pass could take would be a forward no backward can follow.

`layer.last_execution_backend` names the arm the last forward took — `'cuda'` or
`'decode'`, and `None` until one completes. It is a fact about the branch rather than an
inference from a timing, and it is published for exactly that: a benchmark process
cannot spy on the call it is timing, and wall-clock cannot tell two arms apart. It is a
SCALAR, not a tally — a caller that needs how many times an arm ran spies on the arm.

### 4.1 The projections and the value format

`v_proj` and `o_proj` run as bf16 GEMMs (their weights read through the inference copy when no
gradient flows, §2.3), and the projection
is written to the value format at the layer: `Values.from_projection(v)` (`rola/values.py`)
holds each token's values `v` in f16 against its largest channel (at 1); that magnitude is
the token's fp32 `scale`. The op takes the bundle, not the tensor. The plane and the scale are
what the kernels read; `Values.dequantized()` is
the fp32 view a consumer still computing on floating values takes at its seam, with the
projection's gradient stepped straight through when the projection requires one. The arm is
UNCONDITIONAL because the op's envelope makes it so: there is no CPU lowering, so a
non-CUDA forward refuses at `rola_op` and no call off CUDA ever observes a projected
value. Round-tripping
it through the module dtype was MEASURED to cost two elementwise casts per call — a `v`
downcast and a `y` upcast, ~12-13% of a small cell's op time — which is why the layer
narrows once, at the projection, rather than at the op boundary. The GEMMs use cuBLAS's
fp32 compute type (the accumulate-in-fp32 ruling); the global
reduced-precision-reduction knob is deliberately left alone, so this is the same
environment the FlashAttention comparison arm runs in.

`o_proj` rounds the op's fp32 `y` to its bf16 operand, the one rounding on the output side.

### 4.2 Segments: the layer bounds its own working set

A prefill runs as SEGMENTS of `T` tokens (ruled 2026-09-29): per segment the layer projects that
stretch's routes and values, runs `rola_op` on it and carries the state to the next, so the projections
are bounded along with the op -- which an op-level split cannot do once the whole call's bundle exists.
The op runs one segment and knows nothing of segmentation.

`T` is DERIVED, never a knob: the caller declares `memory_budget` (bytes); the layer takes the largest
whole number of the device's windows whose working set (`RoLA.segment_bytes_per_token`, an upper bound on
every per-token tensor a segment allocates, gated against the measured peak) fits beside the state
(`RoLA.state_bytes`, full residency), and REFUSES a budget below one window beside the state.
`memory_budget=None` is one segment. The window grid is global, so a boundary on a window boundary moves
no pair: a segmented call is the whole call (the op's readout to fp32 ULPs,
`tests/integration/test_layer_segments.py`). A stateless call that spans segments carries an internal
state and hands none back.

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
| `PlanOverrides` | the pins: `paging`, `carve` |
| `envelope_refusal(routes, v, decay=None)` / `require_envelope(...)` | whether a call has an arm, asked before making it — the refusing clause in its own words, or `None`. `v` is named rather than assumed: the `d_v` clause is a fact about the value stream, so the answer is a property of the configuration AND of `v.shape[-1]` |
| `requires_backward` | the gradient-reachability predicate the envelope's first clause is built from |
| `GainConfig` | the two-boolean gain layout behind the public `gain` flag |

The contract for everything there is **stable construction, advisory contents**:
build these objects and pass them; do not assert on their fields. What a `Dispatch`
reports is the answer to "how will this call run", and that answer moves with the
machine and with the built matrix. The model facts it carries are the exception,
because they are not decisions.

**Every pin is a declared restriction rather than a dial, and each refuses by name
rather than being ignored.** `PlanOverrides` carries two fields. `paging` is the state's
BACKING, never a launch shape: `None` is the paged default, `False` pins the dense plane
the bit-identity gate compares against, and `True` is refused because paging is on by
default and bitwise invisible. `carve` pins the carry's box and operation plans; the rule
checks a pinned carve against the declaration exactly as it checks its own
([`internals/engine/rules.md#carve`](internals/engine/rules.md#carve)).

**There is no plan object and no derive/execute seam.** The carry's launch geometry is
the carve — the rule's, or a pinned one checked the same way — so there is no
launch-shaping object for a caller to hold or synthesize. A conformance family drives the
envelope's own rule (`rola.engine.rules.envelope`) directly and declares the refusal
clause it expects, and `envelope_refusal` answers the same question before a call.

## 6. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** RoLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.
