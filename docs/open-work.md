# Open work

What is knowingly unfinished, one row an item, each pointing at the page that carries its contract. A row leaves this
list when its work lands; git keeps what it said. The per-kernel obligations a kernel still owes — an oracle test, a
sanitizer row, a budget line, a phase ledger, a ratified manifest entry, an overlap row — are declared in
`tools/coverage.json` with the reason each is owed, and are not repeated here
([`internals/tools/coverage.md`](internals/tools/coverage.md)).

## The prefill

| # | Open | Where |
|---|---|---|
| **C-b** | **The native backward.** No reverse pass exists; `rola_op` refuses a gradient by name. | [`api.md` §1.1](api.md#11-the-built-envelope-is-the-surface) |
| <a id="carry-components"></a>**C-c** | **The carry's components owe their pipelining.** The window is an item machine under a per-window plan (the largest chunks that fit, two in flight); the readout's tiles and the fold's steps are latency-bound (one dependent chain a warp, a list or a digit at a time), and at nl64k-alt-k4 the whole union's fold buffer misses fitting beside the readout by ~6.9 KB on sm_86, so its fill is exposed at the fold. Each lands against its budget and the per-slot oracle. | [`internals/carry/ampere/carry_kernel.md`](internals/carry/ampere/carry_kernel.md), [`internals/carry/ampere/schedule.md#planner`](internals/carry/ampere/schedule.md#planner), [`internals/carry/ampere/readout.md`](internals/carry/ampere/readout.md) |
| **C-e** | **The carry's state I/O.** The rebuilt carry reads and writes no state plane, so the envelope refuses a stateful prefill; the entry and exit sweeps owe the in-place contract. | [`internals/state.md#the-dense-continuation`](internals/state.md#the-dense-continuation) |
| **C-f** | **The carry arm set is three rows.** `tools/manifests/shipped_set.json` declares the alternating `D = 2` box and the two same-split ones (both sides listing one level; a dense write side over a listed read side); a dense read side's arms come with the dense readout, and the committed manifests ratify one carry arm until the next milestone re-ratifies. | [`build.md#carry-arm-lists`](build.md#carry-arm-lists), [`KERNEL_STANDARDS.md#s16`](KERNEL_STANDARDS.md#s16) |
| **C-g** | **Depth three and four.** The intra reads one or two levels (`rola.ops.intra.MAX_LEVELS`), so the prefill refuses deeper topologies by name. | [`internals/intra/intra.md`](internals/intra/intra.md) |
| **C-h** | **Decay in the prefill.** The prefill envelope refuses a decay; decode has it. | [`api.md` §3](api.md#3-the-decay-source-slot) |
| **C-i** | **Whether the other families' instantiation axes join the carry's declaration.** `shipped_set.json` is the carry's; the producer's solves, the decode step and the intra key their own closed sets. Unruled; `tools/lint/drift_guards.py`'s `rule_arm_axes` reads carry kernels only until it is. | [`ARCHITECTURE.md` §5](ARCHITECTURE.md#5-the-axis-law) |

## The state

| # | Open | Where |
|---|---|---|
| **S-a** | **Lifting the 4 GiB state-plane bound.** State is addressed by a 32-bit byte displacement from a pinned origin; the dense and paged backings need different lifts. | [`internals/state.md`](internals/state.md) §9.2 |
| **S-b** | **Copy-on-write state branching.** `strategy="cow"` is refused by name; the semantics are settled, and the per-step table copy wants a two-level page table. | [`internals/state.md#cow`](internals/state.md#cow) |

## Measurement and gates

| # | Open | Where |
|---|---|---|
| **M-a** | **A layer subject.** The timed subjects are kernel-level and the two op-level paths, the stateless `prefill` and decode's step (`measure/subjects.py`); a layer segments a sequence into stateful calls, which wait on C-e. | [`measurement.md`](measurement.md) |
| **M-b** | **Decode bench cells.** Decode's step is timed as a subject; no latency-regression rows exist for it. | [`internals/decode/decode.md`](internals/decode/decode.md) |
| **M-c** | **Skip-induced wave drift.** Boxes that skip windows race ahead of busy ones and de-phase the CTAs' shared working set in L2; measure on the rebuilt carry before designing a fix (a grid-wide window edge, a shared liveness scan, or a skip that keeps its stream position). | [`internals/carry/ampere/carry_kernel.md`](internals/carry/ampere/carry_kernel.md) |
| **M-d** | **The class-drift check.** No gate asserts that a two-branches-agree class's declared members still cover the code's branches. | [`testing.md`](testing.md) |
| **M-e** | **Spec requirement IDs.** Every normative clause an immutable `[SPEC-...]` ID, tests citing IDs only, CI checking each cited ID exists once and pinning its clause's content hash. | — |

## The surface

| # | Open | Where |
|---|---|---|
| **A-a** | **`torch.library` registration and `_fake` meta kernels.** The op's signature carries structured objects (`RouteFactors`, `Values`, the state) that no schema expresses; a flattening layer would buy `torch.compile` and meta-device tracing across the op boundary. | [`api.md`](api.md) |
| **A-b** | **Shapes outside the built envelope**, each justified by a measurement at scale: a non-uniform topology, a `d_v` the intra does not read, a level-split arm. | [`api.md` §1.1](api.md#11-the-built-envelope-is-the-surface) |
