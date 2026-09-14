# `docs/architecture/` — the permanent decisions

This tree holds the decisions that outlive any particular kernel: the things a
contributor must know before changing code, and the reasons behind them, written
for a reader who was not here. Nothing here is a status report. If a statement is
true only of the current implementation, it belongs in
[`docs/internals/`](../internals/README.md) — the per-source-file mirror, which
carries the reasons attached to a *line*. If it is the record of who measured
what on which day, it belongs in the campaign journal, which is deliberately out
of this tree (the performance-campaign journal, an
append-only session log). Each document here states a decision and its argument,
with citations back to the journal entry, the workflow report, the test, or the
source line, so that a future reader can **re-derive** rather than trust.

Everything here is falsifiable. Where the record is genuinely uncertain, the
document says so and names the measurement that would settle it.

| Document | What it settles |
|---|---|
| [self-masking-theorem.md](self-masking-theorem.md) | Dead routes are exact zeros and zeros are inert. The licence for over-fetch, identity-store, and trustless capability flags — and the conditions under which it would fail. |
| [no-delta-rule.md](no-delta-rule.md) | The state update is affine in the input and never in the state. Commutativity, chunkability, schedulability. Where expressiveness goes instead. |
| [factored-representation.md](factored-representation.md) | Store the factors; expand transiently where they are consumed. Never materialize the expansion. |
| [fp32-accumulators.md](fp32-accumulators.md) | Accumulation precision is closed at fp32. Storage precision is a live axis. |
| [closed-world-codegen.md](closed-world-codegen.md) | Only measured binaries run: pinned assembler, AOT-only, ratified per-arch manifest, runtime refusal. Why JIT compilation is rejected. |
| [planner.md](planner.md) | Derive launch-static geometry from launch-static facts; never score candidates, never read a routing statistic to fix a layout. The planner MECHANISM is retired; the page carries the rule and what decides a launch in its place. |
| [guarantees-choices-preferences.md](guarantees-choices-preferences.md) | The taxonomy every proposed change is sorted into, populated with real entries, and what each category demands of a contributor. |

## Reading order

A contributor changing the kernel should read
[guarantees-choices-preferences.md](guarantees-choices-preferences.md) first: it
tells you which of the other documents your change is arguing against, and what
kind of argument would be admissible.
