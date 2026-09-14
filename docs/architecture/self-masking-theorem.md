# The self-masking theorem

> **Dead routes are exact zeros, and zeros are inert — in every accumulator, in
> The mass channel, and in the clock.**

This is the most-used single fact in the engine: three separate design
conclusions rest on it, each of which reduces to it independently (§5). It has a
name and a document so that the next "do we have to check X?" is answered by
citation rather than re-derived.

---

## 1. Statement

Let `s` be a leaf, `t` a token. Write `W[t,s]` for the write amplitude, `R[t,s]`
for the read amplitude, `M[t,s,:]` for the value state, `d[t,s]` for the mass
state, and `c[t,s]` for the decay clock exponent.

**Theorem.** If `W[t,s] = 0` exactly, then step `t` leaves both accumulators of
leaf `s` **bit-unchanged**:

```
c[t,s]    = 0                       (the clock does not tick)
keep[t,s] = (1 - rate[s])^0 = 1     exactly, for every rate in [0,1)
M[t,s,:] <- 1 * M[t-1,s,:] + 0 * v[t,:]   = M[t-1,s,:]
d[t,s]   <- 1 * d[t-1,s]   + 0            = d[t-1,s]
```

and, on the read side, a leaf whose state is identically zero contributes
`R[t,s] * 0 = 0` to the numerator and `R[t,s] * 0 = 0` to the mass denominator —
it is absent from the readout ratio rather than diluting it.

Call this **write-conditioned quiescence**: a route that is not taken costs
nothing and *changes* nothing, down to the last bit.

## 2. Proof by citation — the oracle

The normative reference implementation is `rola/ops/naive.py`
(`naive_rola`, fp64, the eternal spec). Three lines carry the theorem.

**(a) The clock is the write product, not a time counter.** The docstring at
`rola/ops/naive.py:137` states the spec's definition:

```
clock reads (spec §2: ``c[t,s] = stop_gradient(prod_l p_write_stored[...])``);
```

and the implementation is `:207`:

```python
c = _leaf_product(stored64, strides, N).detach()   # [B,T,H,N]
```

`c[t,s]` is the same product of per-level write factors that forms `W[t,s]`
(`:197`, `W = g_write64[..., None] * _leaf_product(write64, strides, N)`), with
gradient detached. **Therefore `W[t,s] = 0` and `c[t,s] = 0` have the same
cause** — some level's factor at `s` is an exact zero.

**(b) A zero clock is an exact identity survival factor.** `:208`:

```python
keep = torch.pow(1.0 - rate, c)                    # [B,T,H,N]
```

`x ** 0 == 1.0` exactly in IEEE arithmetic for every finite `x`, including
`x = 0`. No rounding, no tolerance. And when `decay is None`, `:203` sets `keep`
to exact ones unconditionally, so the no-decay path is the same statement
trivially.

**(c) The update is therefore the identity.** `:219`:

```python
state = keep[:, t, ..., None] * state + W[:, t, ..., None] * deposit_value[:, t, :, None, :]
```

Substituting `keep[t,s] = 1` and `W[t,s] = 0` gives `state[s] <- state[s]` —
multiplication by exactly `1.0` and addition of exactly `0.0`, both
bit-preserving. **Nothing in the leaf's row moves.**

There is ONE such line and not two. The mass state is the same recurrence run on
a value of all ones, so it is the state's last
column (`deposit_value` carries a ones column) rather than a second update with
its own copy of the argument. The substitution therefore covers value and mass
together, which is the strongest form the citation can take: there is no second
line where the theorem could fail to hold.

The property is therefore not asserted about the kernel and hoped for in the
reference: it is a readable consequence of three lines of the reference, and the
kernel is gated against that reference. In-tree verification:
`workflows/research/api-contract-audit.md` §3.9.

## 3. Proof by citation — the kernel

The oracle establishes the *semantics*; the kernel has to reproduce the exact
zero in floating point, which is a separate claim with two load-bearing parts.

**(a) The amplitude is never logged.** The coefficient is built as a direct
product of staged per-level factors (`build_wt_rows` in the retired tiled
consumer's `consumer_kernel.cuh`; the chunk arm's panel production is the same
product form), so a structurally dead route is an exact `0.0f`. The alternative log-space
lowering — log the factors, select with a binary MMA, exponentiate in the
epilogue — would turn that zero into `-inf`, and a clamped `-inf` exponentiates
to `6e-39`, a *small* number rather than a zero. The decision not to take that
form is recorded, with its arithmetic, at
the tiled consumer's `intra_panel.md#no-log-space`, retired with that arm
([`DELETIONS.md`](../internals/DELETIONS.md)); the rule it recorded is the spec's
and binds every arm.
**That is a live instance of a design choice made to keep this theorem true.**

**(b) The other factor is finite by construction.** `0 * finite == 0` exactly;
`0 * inf == NaN`. The spec's rule-3 read-side format guard
(`clamp_bf16_finite`, carried by the tiled consumer's `readout_mma.cuh` until P67
D2 retired that arm) is what kept the multiplicand finite on that narrowing arm.
On the CHUNK arm the multiplicand is a bf16 simplex amplitude produced host-side —
finite by construction rather than by a read-side clamp — and
`chunk_kernel.cuh`'s panel production says so at the site: a zeroed panel row
propagates through "a multiplication whose other operand is a finite simplex
amplitude". The composition is the same one the tiled arm's `build_wt_rows`
stated: rule 4's
zero-route annihilation is satisfied *branchlessly and structurally* because the
coefficient is a product of amplitudes, the factor it multiplies is finite by
rule 3, and `0 * finite == 0`. **There is no test on the amplitude anywhere on
that path** — the annihilation is arithmetic, not a branch, which is why it costs
nothing and cannot be forgotten under a code motion that preserves the product.

## 4. Precondition — write-conditioned decay

**The decay must be mass (write-conditioned) decay, not time decay.** Under a
time decay the exponent would be `1` regardless of whether a write occurred, so
untouched states would decay and the update would not be the identity. The whole
theorem rests on `c` being the *write* product.

This precondition is not luck: mass decay is fixed by the semantics authority
(`GLA_MASS_DECAY_SPEC.md` §2, the normative state-math spec) and time decay is
outside the sanctioned design. But the dependency must travel with the theorem,
because several other decisions now rest on the theorem.

**A second, narrower precondition** is the storage boundary itself. The clock reads
the STORED write levels -- there is one representation, bf16, and both the oracle and
the kernel are fed it -- so the theorem holds exactly while the stored representation
has the **same zero set** as the trained one. A quantization that turned an exact zero
into a denormal, or a nonzero into a zero, would break the coupling in (a) above. That
is why the solve WRITES the stored form rather than being rounded into it afterwards:
the exact structural zeros the entmax closed form produces are written as zeros, so the
zero set is preserved by construction rather than by a check. Any future
storage-precision boundary on the write levels inherits the same constraint, and it
belongs with the theorem because it is not visible from the boundary's own side.

## 5. The three standing consequences

This is why it earns a name rather than a paragraph. Each of the following stands
on its own argument, and each of those arguments reduces to the theorem.

### (1) OVER-FETCH is free of correctness risk

A kernel may read **more routes than are live**, at any granularity it likes,
because the dead ones contribute exactly nothing. Fetching a coarser rectangle
than the true support costs bandwidth and never correctness. This is what makes
the engine's granularity coarsening legal:

- The **paging atom** is one `MMA_K_QUANTUM`-leaf granule, not the true support
  ([`paging/paging.md`](../internals/paging/paging.md), the atom re-key);
- **chunk unions** over a chunk's tokens are an OR, deliberately coarser than the
  per-token AND (the `BT` token tile the tiled consumer took this union over is
  retired; the chunk arm's super-chunk is what takes it now);
- The **block rectangle** is the granularity of a visit: every leaf of the block
  is resident for the whole visit whether or not the visit's support names it, so a
  leaf outside the support is read, multiplied by an exactly-zero coefficient, and
  contributes nothing to either the readout or the fold.

The design rule that follows: **granularity is a performance axis, not a
correctness axis.** Coarsen it with a cost argument alone.

### (2) IDENTITY-STORE — absent and never-written are the same thing

A state that has never been written may be **stored or skipped with the same
result**. Its row is bit-identically zero either way, and a store-back of an
untouched row is an identity RMW: read, multiply by exactly `1.0`, add exactly
`0.0`, round back to the same bits.

Two live uses:

- **Paging is bitwise invisible.** An atom with no write candidate has no
  committed slot (`kAbsentPage = -1`); it reads as zeros and is never stored
  back, which is what makes the paged path *bitwise* equal to the unpaged one
  rather than approximately equal
  ([`paging/paging.md`](../internals/paging/paging.md); gated by
`tests/integration/test_chunk_paging_equivalence.py`).
- **Untouched rows do not walk.** A row whose writes are all zero is left
  bit-identical, so repeated store/reload cannot drift it. This is what separates
  the *store* count from the *visit* count in the state-precision analysis: the
  store of an untouched leaf is a bitwise identity, so the re-quantization count is
  the count of actual writes, not of visits (journal, per-leaf visit probe).

### (3) TRUSTLESS CAPABILITY FLAGS

A capability flag may be **conservatively wrong in the inclusive direction
without changing a result**, so such a flag can be *trusted outright*, with no
verification: an over-approximated support costs the declaring caller performance and
nobody else correctness. That licence is what makes a declared-density column a
legitimate pricing input for any future cost model, and it is exactly the argument the
membership-authority rule below turns on.

The same reasoning licenses the emission-side `if (out != 0.0f)` skips: adding
`0.0f` is the identity, so including the term is a no-op and the test is
over-inclusion-safe in the direction that matters
(the membership-authority rule, stated for the surviving arm at
[`carry/carry_kernel.md`](../internals/carry/carry_kernel.md)).

**The direction is asymmetric and that asymmetry is the whole content.**
Over-approximating support is always safe. **Under-approximating it is wrong** —
and no declaration in the API is permitted to shrink a support. Membership is
decided by the realized support bits and by nothing else.

## 6. When it would fail

A theorem that cannot fail is a slogan. These are the changes that would break
it. Each is a live proposal in the design space, and each is refused *because* it
breaks this.

| Change | Why the theorem dies |
|---|---|
| **An epsilon floor** on routing weights (`max(p, eps)`, or an STE-style relaxed support) | There are no exact zeros left. Every dead route becomes a small live route: the clock ticks, `keep < 1`, untouched states decay, over-fetch changes results, identity-store drifts. Standing ruling: sparsity comes from the activation's exact zeros only — never a floor, never a threshold. |
| **A log-space amplitude** | `log 0 = -inf`. Either `-inf * 0 = NaN`, or the clamp turns the zero into `6e-39` and the route is no longer dead. The arithmetic is in the retired `intra_panel.md#no-log-space` (`git show c7eaeb9:docs/internals/intra_panel.md`). |
| **A non-zero-preserving normalization** — per-state normalization that divides by the state's own magnitude, or any readout of the form `q · S / ‖S‖` | `0/0`. A never-written state has no unit direction, so reading it is undefined rather than inert. The shipped readout is a **global ratio** `Σ r̃ᶜ Nᶜ / (Σ r̃ᶜ dᶜ + ε)` in which the explicit `dᶜ` factor kills the dead leaf in numerator *and* denominator; that structure is what makes the theorem survive normalization at all. |
| **A mean-style reduction** — averaging over the *live* entries, or any readout whose denominator counts contributors rather than summing their mass | A zero is then not inert; it is a vote for zero. A dead leaf would drag an `O(1)` average down by an amount proportional to how many leaves are dead — data-dependent, silent, and absent from the reference. This is the exact reason the "average over only-written states" variant is classified a **modeling change** rather than an optimization: it alters the dense result too. |
| **Time decay** in place of mass decay | The exponent is 1 whether or not a write occurred. Untouched states decay. See §4. |
| **A non-finite multiplicand** — removing rule 3's clamp, or applying it on the arm that does not narrow | `0 * inf = NaN`. The annihilation is `0 * finite`, and "finite" is an enforced property, not an assumption. |

Any proposal in this table is a proposal to change the engine, not to tune it.
It requires re-deriving every consequence in §5 from scratch.

## 7. Scope boundary

**The theorem is about the kernel's arithmetic.** It says that computing with a
zero is the same as not computing. It says nothing about a component that reads
a *declaration* instead of the realized masks, because such a component never
performs the multiplication the theorem is about.

Concretely: the trustless-flag consequence (§5.3) is proved **against today's
kernel**, in which the density declaration selects a code path but membership is
still decided by the support bits the kernel reads. A future factored feeder that
let a declaration become execution-affecting — deciding what is fetched *without*
consulting the realized masks — would step outside the theorem and need its own
argument. This coupling is flagged in `workflows/research/api-contract-audit.md` §5.4.

State it as a rule: **the kernel is self-masking; anything that skips reading the
masks is outside the theorem.**
