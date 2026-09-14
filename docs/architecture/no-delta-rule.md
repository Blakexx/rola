# The affine recurrence — no delta rule

> **The state update is affine in the input and never a function of the state.**
> Nothing the recurrence does may depend on what the state currently holds.

Formally, the sanctioned update is

```
M[t,s,:] = keep[t,s] * M[t-1,s,:] + W[t,s] * v[t,:]
d[t,s]   = keep[t,s] * d[t-1,s]   + W[t,s]
```

where `keep`, `W` and `v` are functions of the **inputs** (routing amplitudes,
decay dials, values) and of nothing else. A delta-rule-style update — one whose
write depends on what is already stored, e.g. erase-then-write against a read of
`M` — is a standing user veto. It is not
a performance preference and it is not open for a benchmark: it changes what the
engine is.

## Why: three payoffs

### 1. Commutativity

Writes to a leaf compose by an operation that does not care about their order
relative to *other* leaves, and the per-leaf recurrence is a linear scan with an
associative combine. That is what makes the whole computation expressible as
GEMMs over a chunk: the intra-chunk fold is **one matrix product**, not a
sequential solve. A state-dependent write destroys this immediately — DeltaNet's
WY/UT in-chunk solve is sequential *because* the update reads the state — and the
chunk fold stops being a GEMM.

### 2. Chunkability

Because the transition does not read the state, a chunk's contribution can be
formed **without knowing the state at the chunk boundary**, and the boundary is
applied afterwards. Two consequences the engine depends on:

- The backward pass gains **no snapshot pressure**: with a state-dependent write,
  the state at every write point is needed to differentiate, and that has to be
  stored or recomputed. Here it does not.
- **capacity is routing-derivable**: what a token writes, and where, is a
  function of routing alone. Under state-dependent writes the effect of a write
  becomes trajectory-dependent, and every capacity statement becomes a statement
  about a particular history rather than about the architecture.

### 3. Schedulability — the payoff usually left out

This is the one that pays for the whole execution model, and it is easiest to
state as a negative: **state-dependent routing would force in-kernel
rescheduling.** You would have to consume a schedule in order to compute the next
schedule, with a global barrier between them, inside the kernel.

Because the recurrence is input-affine, **the plan is complete before launch.**
the work list is therefore a *free permutation*: the kernel touches the owner
list only through a flat `blockIdx`
(the only two `p.owners` reads in the kernel, at CTA entry and both indexed by
`blockIdx`), so any
reordering of that list is a legal execution. Everything the engine gets from
that follows:

- **sequence parallelism** — the flat `(owner, segment)` work list and its
  segment machinery of the retired tiled consumer, and the chunk arm's own
  schedule after it;
- **split-K** — partial outputs over an arbitrary partition of the contraction,
  reduced afterwards;
- **the paged arena** — residency decided from the plan, ahead of execution,
  rather than discovered during it;
- **plan-side specialization in general** — including the largest measured win in
  the performance record to date, a *one-line reordering* of the owner list for
  L2 locality: parity step 71.98 → 67.60 ms, DRAM traffic 20.27 → 6.38 GB, with
  **zero lines of kernel source changed** (journal, 2026-08-04, head-major owner
  order; landed as commit `309bfbe`). That entire class of optimization exists
  only because order is not load-bearing, and order is not load-bearing only
  because the recurrence is affine.

**The precision that matters**: "routing exists before launch" means the
schedule's **factors** — per-level digit bitmaps and
slice unions, sized by the *sum* of the widths — are complete and immutable at
launch. The per-owner schedule is **never stored**; the kernel evaluates it from
those factors with a warp-local AND chain, measured at ~0.2% of the instruction
stream, barrier-free. Storing the per-owner schedule would be the `O(L·N)`
materialization wall. **Schedulability is evaluation, not generation** — see
[factored-representation.md](factored-representation.md), of which this is one
instance.

## Where expressiveness goes instead

The veto is a routing decision about *where new capability is added*, not a
refusal to add any. Three sanctioned directions:

1. **Producers.** The routing producers (routers, per-level activations, learned
   `α`) are kernel-invariant and unrestricted. Anything expressible as "compute a
   better `W` and `R`" is free.
2. **`N`, depth, and routing quality.** More states, more levels, better
   selection — the capacity axis the architecture is *for*.
3. **The inter-layer loop.** State-dependent computation is legitimate *between*
   layers: layer ℓ's readout conditions layer ℓ+1's producers. This is where
   trajectory dependence lives, and it costs the recurrence nothing.

And two things that are explicitly **not** missing:

- **Erasure exists** — it is decay and gating, driven by the input
  (`keep = (1 - rate)^c` with `c` the write clock). What is refused is erasure
  computed *from the state*.
- **Selection exists** — sparsity comes from the routing activation's exact
  zeros. What is refused is selection conditioned on stored content.

## What would reopen it

Nothing short of a user ruling; this is a veto, not a measurement. But the record
should be honest about what the veto costs and what would make the cost visible:
the delta-rule wing of the literature buys expressiveness per state, and if a
capacity-matched comparison ever showed RoLA needing materially more states to
reach the same quality, that is the number that would have to be argued about.
The engine's own position is that the answer to "more expressiveness per state"
is **more states**, and that the affine recurrence is what makes more states
affordable — see [design for scale](guarantees-choices-preferences.md), the
standing directive that all kernel and benchmark design targets `N ≳ 65K`.

## Provenance

- User veto and its four-pillar argument: ruled 2026-07-31, restated
  2026-08-01 (development journal; theory notes §10i.25/§10i.25a); this page
  is the standing statement of it.
- The schedulability argument and its precision clause: campaign journal,
  2026-08-05, "affine-schedulability observation (paper material)".
- The reordering win it licenses: campaign journal, 2026-08-04, head-major owner
  order; `test_the_owner_order_is_head_major`, which retired with
  `tests/unit/test_planner.py` in P67 D2 (`git show
  c7eaeb9:tests/unit/test_planner.py`); commit `309bfbe`.
