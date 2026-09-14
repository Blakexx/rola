# fp32 accumulators

> **Accumulators are fp32, permanently.** Storage precision is a live,
> measurable axis. Accumulation precision is closed.

The distinction is the whole document, so state it first.

| | What it means | Status |
|---|---|---|
| **ACCUMULATION precision** | the format a value is *added into* across steps: the value state, the mass state, the decay clock, the split-K emission accumulators | **closed at fp32.** Not a default, not a tuning knob — a ruling. |
| **STORAGE precision** | the format a value is *held in* and later read, without being accumulated into: amplitudes, the value tensor, projection GEMM inputs, the `y` output cast | **live.** Each surface gets a precision gate written first, then an oracle A/B, then a performance measurement. |

The two look alike in a dtype table and behave nothing alike. A read-only tensor
that is 1 ULP wrong contributes an error bounded by its own magnitude, once. An
accumulator that is 1 ULP wrong **embeds** that error and carries it forward for
the lifetime of the state.

## <a id="compensated-pair"></a>A compensated pair is not a plain narrowing, and the two are governed separately

**HISTORICAL, and kept because the ruling it tests is not.** The mechanism this
section governs was the TILED consumer's, retired whole in P67 D2
([`DELETIONS.md`](../internals/DELETIONS.md); revival `git show c7eaeb9:<path>`).
The prefill arm in the tree today is the chunk consumer, whose state plane is
plain fp32 (`chunk_kernel.cuh`'s `s_in`/`s_out`) — there is no compensated pair in
this tree now, and the paragraphs below are the record of how the split was
applied when there was one. What is NOT historical is the distinction, the ruling,
and the gate: `tests/oracle/test_bf16_family_gate.py` re-anchored onto
the chunk arm in the same phase and still runs the `sqrt(T)` walk over
`T in {512, 2048, 8192}`.

The tiled consumer's resident state was two bf16 planes (`Shi`, a value's bf16
rounding, and `Slo`, its residual) rather than one fp32 plane. That was NOT the
single-narrowing "bf16 resident state" the measured reason below refuses: that
proposal halved the state's footprint (one bf16 plane at 2 bytes/element
against fp32's 4) to buy a larger `BC`. The plane pair costs the SAME bytes per
element as the fp32 form it replaced (two bf16 words = 4 bytes = one fp32 word) and
was built for a different reason entirely — an `ldmatrix`-fetchable B operand for
GEMM 1 — so it did not touch the tradeoff this document's measurement declined.

The ACCUMULATION/STORAGE split above applied to it, and governed the plane form:
the fold's read-modify-write read the pair, summed in fp32 registers, and split
back to `Shi`/`Slo` at its one writeback — the accumulator stayed fp32, exactly as
this ruling requires. The state's STORAGE precision between accumulation events was
a compensated pair (~16 mantissa bits) where it had been single fp32 (~24) — a live
axis under this document's own table, and this page's own `sqrt(T)`-walk gate
(`tests/oracle/test_bf16_family_gate.py::test_family_deviation_is_flat_in_T`,
run against the real kernel's state output) was run against it and held: every swept
`T in {512, 2048, 8192}` cleared `BF16_RTOL`, and the fitted log-log slope cleared
the `< 0.25` bar against the resident-state walk's measured `+0.49`. That is exactly
what distinguishes a compensated pair from a single narrowing — 16 mantissa bits
leave enough headroom that the walk does not surface inside the swept range. A plain
single-narrowing bf16 state remains refused for the reason measured below; the same
gate is what killed it, and that gate now reports on the chunk arm (slope `+0.053`
over the same three lengths, P67 D2-b).

## The measured reason

A bf16 resident state — halving the state's shared-memory footprint — is
**refused**, and the numbers that refuse it are these (campaign journal,
2026-08-03, "batch-1 verdict"):

- The re-quantization walk is **real and `sqrt(T)`-exact**: `1.06e-2` at `T=512`
  against a `BF16_RTOL` budget of `1e-2` — already over — and extrapolating to
  `~1.8e-1` at `N ≈ 65K`. The standing design-for-scale rule is what makes this
  visible: a `T=37` fixture passes it, which is why the suite now carries a
  512 / 2048 / 8192 arm (`docs/testing.md`, Tier 1's length axis) and charges the
  output per token segment rather than against one global maximum.
- The composition law is measured, not assumed (150 paired runs, 10 cells × 5 `T`
  × 3 seeds): the walk composes in **quadrature**, log-log slope `+0.490` against
  theory `+0.5`, with the quadrature constant flat in `V` (spread 1.06×) where a
  linear law spreads 1.35×.
- **Decay does not flatten it.** Decay shrinks the signal and the old noise
  together, while fresh rounding always arrives at the current magnitude.
- **An fp32 mass column is exactly the right exception.** Keeping the mass column
  fp32 leaves results bit-unchanged, because a row-uniform error in the value
  columns cancels in the readout ratio while a per-element error in the
  denominator has no cancelling partner.

Three adjacent justifications for a bf16 state are each refuted by their own
measurement: DRAM traffic (state is 0.36% of it), occupancy (0.85 eligible warps
per scheduler; the kernel is not occupancy-starved), and ring-depth enablement
(depth 2 is not shared-memory-blocked; freeing bytes makes depth 2 available on
*fewer* instantiations, 148 → 128).

## What the ruling forgoes

The cost is real and is paid knowingly: the freed shared memory would have
purchased a larger `BC`, and with it a designed path from ~18.4 ms to ~9.2 ms on
the consumer kernel. That path is forgone because the failure it risks is silent.

## Why the ruling is permanent, not a tuning default

The ruling (campaign journal, 2026-08-05 late, "bf16 state accumulation killed")
rests on three arguments that no measurement can reverse, because they are about
what a measurement can *see*:

1. **The governing write count is a lifetime, not a launch.** The walk does not
   reset at an fp32 boundary: an fp32 store-back does not repair a bf16-embedded
   error, it preserves it. The cache carries it across calls and decode compounds
   it.
2. **Nothing bounds a burst.** Per-write mass decay gives no guarantee of
   attenuation per event: a burst of small writes re-quantizes repeatedly without
   decaying. A "small writes are cheap" defence requires a discount that the
   convexity of the composition law does not permit.
3. **The oracle structurally cannot see it.** The reference is per-call. Cross-call
   compounding is invisible to the gate that would have to catch a regression.
   A property no gate can see cannot be an engineering choice; it has to be a
   structural one.

The standing conclusion these compose to: **normalization bounds what reads
contribute; nothing bounds what accumulation embeds.**

There is a second, independent instance of the same rule already in tree, and it
is worth citing because it is not about the state: the decay **dials** are
accumulated in fp32 and never bf16, because `rate` near 1 makes `1 - rate`
catastrophically cancelling and `log1pf` cannot repair a cancellation that has
already happened inside the product (the tiled consumer's decay dials, retired with
that arm — [`DELETIONS.md`](../internals/DELETIONS.md); the rule is the spec's, and
`rola/routing/`'s dial construction still holds it host-side).

## What stays open

The bf16 **read-only memory** family is untouched by this ruling, and is
strengthened by contrast — no accumulation, therefore no walk. It is one
decision over five surfaces, all keyed to the MMA precision axis so that the
exact arm stays fp32 end to end (journal, 2026-08-04 night, phase-2 scope):

1. spanned-level amplitudes;
2. The value tensor's HBM storage;
3. `v_proj` / `o_proj` as bf16 GEMMs with fp32 accumulation;
4. `y` cast to bf16 *after* the divide — the divide, the numerator and the
   denominator accumulators all stay fp32;
5. The host-boundary fp32-only check, relaxed.

Note what items 3 and 4 preserve: **bf16 operands feeding fp32 accumulation is
not a violation of this document.** The MMA already works that way. The rule is
about the accumulator, not the operand.

Two conditions govern any of these landing:

- **The precision gate is written before the change**, not after, and it carries a
  `T`-scaling arm (`T ∈ {512, 2048, 8192}`) with a halt on any monotone trend.
  A gate without a `T`-scaling arm cannot see a `sqrt(T)` accumulation at all, so
  a working build is not evidence of its absence.
- **A gate must be shown capable of failing.** The committed output-charged assert
  in that same family is **bit-identical between the fp32 and bf16 arms in 150/150
  runs**, because its maximum is pinned at token 0 — where the carried mass is
  still tiny and the state has been re-quantized zero times. It cannot detect a
  resident-state walk of any size at any `T`. A gate that cannot fail is not
  evidence.

## Provenance

- The walk measurement and the declined build: campaign journal, 2026-08-03; the
  preserved instruments `i6_walk.py` / `i6_numbers` / `occ32` and the declined
  patch `rung1_bf16_state_DECLINED.patch`.
- Composition law, `V_max` derivations and the gate defect: `vmax_budget.md`
  (150 paired runs), journal 2026-08-04.
- Final kill and the standing conclusion: campaign journal, 2026-08-05 late.
- The dials instance: the tiled consumer's `consumer.md#dials-fp32`, recoverable
  with the arm at `git show c7eaeb9:docs/internals/consumer.md`.
