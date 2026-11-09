# Factored representation

> **Store the factors. Expand transiently, where they are consumed. Never
> materialize the expansion.**

Every large object in this engine is a Kronecker-style product of small
per-level objects. The rule is that it stays that way in memory: what is written
down is the `Σ_l b_l` factors, and the `Π_l b_l` expansion exists only inside the
consumer that needs it, in registers or shared memory, for as long as one tile
takes.

It is a **unification**: the memory rule, the schedule rule and the
schedulability rule are one principle, not three. *The routing matrix and its
support are both Kronecker-factored objects — store the factors, expand
transiently where consumed (panel in registers, candidacy in SMEM), never in
HBM.*

## Why it is one principle and not four coincidences

The generative fact is that the leaf index is a **mixed-radix digit string**:
leaf `s` decomposes into per-level digits `d_l(s)` via the topology's radix
strides, and every per-leaf quantity in the model is a product over levels of a
per-digit quantity:

```
W[t,s] =  Π_l  k[l][t, d_l(s)]     # a producer's write gain is a factor of k[0]
R[t,s] =  Π_l  q[l][t, d_l(s)]     # (docs/CONSTITUTION.md#r23)
rate[s] = Π_l  delta_l[d_l(s)]
```

(`rola/ops/naive.py`, `_leaf_product`.) A product over levels is a rank-1 object
per level; storing the expansion stores `Π b_l` numbers to represent `Σ b_l`
numbers. At the design scale — `N ≳ 65K` states, the standing target — the
difference is the difference between a plan that fits and a plan that does not.

The **address side factors too**: a leaf's digit at a level is a pure function of
the leaf index that reads no routing data (`rola/ops/naive.py::_leaf_product`'s
strides; the kernels' shifts over power-of-two widths). So the support's
*bitmap* is likewise a Kronecker product of the per-level indicators
([`decode/decode.md`](../internals/decode/decode.md), which expands two `N`-bit maps by
Kronecker broadcast rather than storing one).

## The instances it unifies

| Object | Factored form stored | Expansion, and where it lives |
|---|---|---|
| **Routing gates** | the bundle's per-level rows `[B, T, H, b_l]` | the carry's operand, a token's dense slices times its list coefficient, formed in registers for one MMA ([`carry/ampere/fold.md#step`](../internals/carry/ampere/fold.md#step)); the intra's per-level Grams; decode's per-unit amplitude products. The gates are **never materialized** as `[L, N]`. |
| **The Kronecker ledger** — the leaf-index arithmetic | the widths and the box's extents | shifts and masks evaluated where a leaf is addressed; decode's unit fixes every level but the innermost, so their product is formed once a unit ([`decode/decode_lattice.md#the-unit`](../internals/decode/decode_lattice.md#the-unit)). |
| **Liveness words** | per-level digit words and tile unions, sized by `Σ_l b_l` | a box's liveness for a token is **evaluated** as the AND over levels by the consumer, never stored ([`facts/liveness_contract.md`](../internals/facts/liveness_contract.md)). |
| **The schedule** | the same per-level bitmaps | the per-owner work set, re-evaluated in kernel. Storing it would be the `O(L·N)` wall. |
| **The token-gather ballots** | the same factor words | live-token masks computed from the support words already loaded. |

The negative cases are as instructive; both are priced and refused:

- **Precomputing the `W` panel to VRAM** — `O(candidates × tokens × box leaves)`,
  gigabytes at the parity cell. Refused (2026-08-04). The
  implication of "the operand is computed" is *make it cheap* (affine addressing)
  and *hide it* (pipeline), never *materialize it*. This is FlashAttention's
  founding lesson applied to the operand instead of to the score matrix.
- **Per-candidate compaction** — the same `O(L·N)` wall in a different coordinate
  (8.59 GB at parity). Per-**token** compaction, `O(BH · T · support)`, is
  `N`-independent (13.5 MB against 320 MiB dense) and is therefore the admissible
  form (measured 2026-08-04).

## When it does NOT apply

The rule is about **representation**, and it earns its keep only when the
expansion is (a) larger than the factors by a factor that grows with a design
axis, and (b) consumed in a bounded window. Where either fails, factoring is
waste or worse:

1. **When the expansion is consumed repeatedly at a distance.** The rule says
   expand *where consumed*. If the same expansion is needed by many consumers
   spread over time, re-expansion is recomputation, and the trade becomes a real
   measurement rather than an obvious win. The engine's own case where the
   staging transform *loses*: an amplitude staging buffer that reshapes the
   per-element gather has no reuse to exploit, because each element is loaded
   **exactly once per visit** — a theorem, confirmed by probe at a measured
   load-reduction factor of `1.0030` work-weighted and exactly `1.0000` at the
   flagship (measured 2026-08-04, and the staging declined). *There is
   nothing to amortize.*
2. **When the factored form is smaller than the hardware's quantum.** Below a
   32-byte sector or a 16-row MMA quantum, a finer representation cannot save a
   byte or an instruction. Measured: with fp32 amplitudes, 8 digits are one 32 B
   sector, so support granularities `G ∈ {1,2,4,8}` have *identical* sector
   counts; and `kMmaKQuantum = 16` makes `G ∈ {1..16}` identical in MMA rows
   (measured 2026-08-04). **The continuum collapses
   to the hardware's grid.**
3. **When the expansion is dense.** Factoring is a statement that most of the
   product is structurally absent. Where occupancy approaches 1, the dense form
   *is* the compact form, and the factored path pays index arithmetic for nothing.
   This is why the carve takes each level's representation from its declared flag
   rather than a universal default: a sparse level is a list, a dense level is
   expanded into the operand ([`rules.md#carve`](../internals/engine/rules.md#carve)).
4. **When it would make a declaration execution-affecting.** Expansion may be
   skipped only where the realized masks still decide membership; see the scope
   boundary of the [self-masking theorem](self-masking-theorem.md#7-scope-boundary).

## Provenance

- The Cartesian-product support structure:
  [`docs/internals/decode/decode.md`](../internals/decode/decode.md#kronecker-support),
  [`docs/internals/decode/decode_lattice.md`](../internals/decode/decode_lattice.md); reference semantics:
  `rola/ops/naive.py::_leaf_product`.
