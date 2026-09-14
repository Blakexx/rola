# The planner — RETIRED, and the rule it leaves behind

> **Derive STATE-LAYOUT geometry from launch-static facts. Never score
> candidates, never read a routing statistic to decide a launch-static axis.**

The rule above is permanent and still governs this tree. The MECHANISM this
document used to describe is not: `rola/planner/` — `derive_bc` and its three
ceilings, `build_plan` / `plan_and_schedule`, `Plan` / `Schedule`,
`select_window` and the calibration instruments — was deleted whole by P67 D2
([`../internals/DELETIONS.md`](../internals/DELETIONS.md)). The full text of the
derivation, with each ceiling's falsifier and the `GT` window measurements, is
recoverable at `git show c7eaeb9:docs/architecture/planner.md`, and the code it
described at `git show c7eaeb9:rola/planner/`.

This page is kept rather than deleted because the rule outlived the machine that
enforced it, and a tree with no cost model should say WHY it has none.

## What decides the launch now

The leaves one owner block owns are no longer derived per launch, because
that is no longer a host-selectable quantity at all. It is a property of the
BUILT ARM: `tools/gen_shards.py` owns the closed-world `CARRY_ARMS` table of
`(D, DV, warps_per_cta)` tuples, emits it as the generated `carry_selection.inc`
(one per-arm translation unit, `carry_arm_N.cu`, per shipped row), and
`carry/carry.cu`'s dispatch runs an arm only if the selection header carries it.
A topology with no built arm gets no kernel: `rola.engine.rules.envelope` names
the refusing clause and `rola_op` raises it ([`../api.md`](../api.md) §1.1, the
built envelope).

That is the same category of answer the old derivation gave, taken one step
further. The closed form read launch-static facts to choose among compiled
blocks; the manifest lookup reads a compiled fact and chooses nothing. Neither
reads a per-step routing distribution, which is the property the rule is about.

The **expert pin** went with the derivation it replaced: `PlanOverrides` no
longer carries `state_block`, and passing one is refused by name rather than
silently ignored, because a pin over an axis the binary fixes would be inert.

## What stayed true

1. **Structure, not thresholds.** An axis is dispatched on a structural fact —
   a tile count, a level count, a built arm — never on a threshold over a
   measured value. The refusal clauses in `rola.engine.rules.envelope` are all of the
   first kind.
2. **A feasibility refusal is loud.** A topology outside the built matrix gets a
   named refusal and a reference execution, never a silently substituted arm.
   This is the shape the old `box_admitted` `ValueError` had, kept.
3. **Exact routing statistics are MEASUREMENT instruments, not planner inputs.**
   The census and the union statistics that answered "what did this routing do"
   were bench instruments no dispatch path reached; they were deleted with the
   planner, and the surviving statistics surface (`rola.engine.facts.planes.atom_bits`, the
   write-atom set the page plan consumes) keeps the same discipline — it is
   keyed to the 16-leaf quantum, not to `BC`, so it is a fact about the routing
   alone and is legitimately taken BEFORE an arm is chosen.
4. **Where this sits relative to the field.** FlashAttention, xformers and
   TensorRT-LLM encode "which kernel arm" as a static condition on shape, dtype
   and architecture, and none reads a per-step distribution. RoLA's routing IS a
   per-step distribution, and the geometry still does not need to see it.

## The one axis the rule did not govern, and its verdict

`GT`, the window (the token quantum one visit consumed), was the deliberate
exception: it was selected by reading a routing statistic, because it changed
nothing about the state's layout or the kernel's numerics — a pure performance
selection over an already-derived `BC`.

Its measured verdict is worth keeping even though the axis is gone: a wider
window never won the residency it cost on `sm_86`, and the loss TRACKED THE
RESIDENCY GIVEN UP rather than the wider Gram's arithmetic (`BC = 16`: 4 CTAs ->
2, +43%; `BC = 64`: 3 CTAs -> 1, +84% on the dense `N = L = 4096` cell, +145% at
`atscale-L2048`), and it survived a `cp.async` prefetch of the amplitude gathers
that cut `long_scoreboard` 26-46% — so the residency was buying something other
than the panel builds. Records: `workflows/perf/p36_quantum.md`,
`workflows/perf/p39_prefetch.md`. The (since-deleted) chunk arm's own swizzle
axis met the same kind of end for the opposite reason — a receipt of
-2.1..+2.2%, noise, no winner (`git show c7eaeb9:instantiations/chunk_arms.inc`).
