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
| <a id="carry-components"></a>**C-c** | **No carry, intra or producer kernel is built.** The ampere family's are the branch `ampere`'s; each kernel's hopper schedule is a design of its own, written into its page before it is built, the carry's first. Until then `rola_op` refuses every prefill for want of a carry arm, the ops raise `rola.ops._ext.KernelNotBuilt`, and the tests and comparisons that reach them fail by that name. | [`ARCHITECTURE.md` §9](ARCHITECTURE.md#9-portability), [`KERNEL_STANDARDS.md`](KERNEL_STANDARDS.md) |
| **C-d** | **The data model in the shared code.** The line's operands are u8 rows at every level and bf16 values (CONSTITUTION §R17, §R23); the factory still forms a floored f16 dense level and f16 values, the facts pass still emits list exponents, coefficients in an f16 operand's format and pool offsets for a carry arm's form, and decode still reads the f16 dense rows. | [`architecture/precision.md`](architecture/precision.md#components), [`internals/facts/facts.md`](internals/facts/facts.md) |
| **C-e** | **The carry's state I/O.** A prefill that hands a state in or out is refused by the envelope; a carry's entry and exit sweeps owe the in-place contract. | [`internals/state.md#the-dense-continuation`](internals/state.md#the-dense-continuation) |
| **C-f** | **The carry arm set is empty.** `tools/manifests/shipped_set.json` declares no row; its key stands as the carve's operation-plan pair until a hopper carry's design re-keys it. | [`internals/tools/gen_shards.md`](internals/tools/gen_shards.md) |
| **C-g** | **Depth three and four.** The intra's shape law reads one or two levels (`rola.ops.intra.MAX_LEVELS`), so the prefill refuses deeper topologies by name. | `rola/ops/intra.py` |
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
| **M-c** | **Skip-induced wave drift.** Boxes that skip windows race ahead of busy ones and de-phase the CTAs' shared working set in L2; measure on a built carry before designing a fix (a grid-wide window edge, a shared liveness scan, or a skip that keeps its stream position). | [`ARCHITECTURE.md` §6.3](ARCHITECTURE.md#63-the-carry--the-inter-window-term) |
| **M-d** | **The class-drift check.** No gate asserts that a two-branches-agree class's declared members still cover the code's branches. | [`testing.md`](testing.md) |
| **M-e** | **Spec requirement IDs.** Every normative clause an immutable `[SPEC-...]` ID, tests citing IDs only, CI checking each cited ID exists once and pinning its clause's content hash. | — |

## The surface

| # | Open | Where |
|---|---|---|
| **A-a** | **`torch.library` registration and `_fake` meta kernels.** The op's signature carries structured objects (`RouteFactors`, `Values`, the state) that no schema expresses; a flattening layer would buy `torch.compile` and meta-device tracing across the op boundary. | [`api.md`](api.md) |
| **A-b** | **Shapes outside the built envelope**, each justified by a measurement at scale: a non-uniform topology, a `d_v` the intra does not read, a level-split arm. | [`api.md` §1.1](api.md#11-the-built-envelope-is-the-surface) |
