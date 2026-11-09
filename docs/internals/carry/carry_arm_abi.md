# `carry/carry_arm_abi.cuh` — one arm's ABI

Mirrors `csrc/rola/src/carry/carry_arm_abi.cuh`: pointers and scalars only, so a per-arm translation unit never
parses the torch headers and the dispatch unit never parses a kernel body.

<a id="form"></a>
## The form

`CarryForm` is an arm's plan on the current device's card (`ampere/plan.md#plan`): its shared bytes, the area its windows'
plans place buffers in, a read and a write pool row's bytes, and each side's list group (`ampere/plan.md#list-groups`). The
facts pass addresses the read entries' rows in the read row and ranks each side's lists in its list group
(`rola.ops.carry.lists_form`).

<a id="census"></a>
## The census

`ArmCensus` is what the device build reports for one arm: registers a thread, local bytes (zero, or the SASS gate is
red), the block's thread limit, the binary version and its form. `carry_census` returns one row an arm, its index and
key first, the form's fields after the binary version. `ROLA_CARRY_ARM_DECLARE` declares an arm's launch and census for the dispatch unit.
