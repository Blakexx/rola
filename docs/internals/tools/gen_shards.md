# `tools/gen_shards.py` — the one enumeration of every arm family's arms

Mirrors `tools/gen_shards.py`. The ratification's view of the same machinery is
[`docs/ratification.md`](../../ratification.md#what-the-shipped-set-is-and-where-it-is-declared).

## Membership is declared; enumeration is code

`tools/manifests/shipped_set.json` declares WHICH arms exist, keyed by `ARM_KEY`: the carve's operation-plan pair
([`engine/rules.md#carve`](../engine/rules.md#carve)). This module decides what a row MEANS: which translation unit
carries it, what the selection header says about it, what the ratification's `shards`
block records. The split is what lets the library's arm gate compare the two as
independent statements instead of restating one twice.

<a id="families"></a>
## Arm families

Every kernel with declaration arms plugs in as an ARM FAMILY (`ArmFamily`): its key, its declared rows
(`tools/manifests/<family>_set.json`, the carry's `shipped_set.json`), the generated selection header a build's subset
is written to and who includes it. The reader, the refusals, the selection header (`ROLA_<FAMILY>_BUILT`,
`ROLA_<FAMILY>_BUILD_<i>`, the arm set `ROLA_<FAMILY>_ARM_SET_X` with its counts) and its schema self-test are one code
path over the family; a build's scope comes from its spec (`family_arms`: `--//:<family>_arms`), never a flag on every
unit's argv. The built table is checked against the declaration, and the binary asked what it carries, by
`check_built_arm_table` and `sweep_binary`, which the library's arm gate (`bazel/gates.py`) runs. The families: `carry` (one translation unit an arm group, below) and `decode` (one unit,
`decode/decode.cu`).

## What it produces

| output | checked in? | written by | consumed by |
|---|---|---|---|
| `csrc/rola/src/instantiations/carry_arm_<i>.cu` | yes — what is reviewed is what compiles | `--write` | the build's source list |
| `csrc/rola/units.bzl` | yes, checked by `--check` and the graph's build gates | `--write` | the graph's units: `tools/kernels.py`'s hand units, the runtime's among them, and each arm unit with the arms it carries |
| `build/generated/carry_selection.inc` | never | the graph's header action (`bazel/headers.py`) into its own output, which `bazel run //:library` copies here; `tools/ratify.py`'s compile path, on every run | the per-arm TUs, the carry dispatch TU, `common/arm_switch.cuh` |
| `build/generated/decode_selection.inc` | never | the graph's header action into its own output, which `bazel run //:library` copies here; `tools/ratify.py`'s compile path, on every run | `decode/decode.cu` |
| `build/generated/builds.inc` | never | the graph's header action into its own output, which `bazel run //:library` copies here; `write_builds`, from `tools/ratify.py`'s compile path on every run | `common/select.cuh` ([`../common/select.md#carried`](../common/select.md#carried)) |
| the `shards` block | recorded in each `tools/manifests/<toolchain>/sm_XX.json` | `tools/ratify.py --write` | the manifest digest |

The selection header carries two things: the `ROLA_CARRY_BUILD_<i>` selection, and the
ARM SET `arm_switch` dispatches over — `ROLA_CARRY_ARM_SET_X(F)`, plus
`ROLA_CARRY_ARM_COUNT` beside `ROLA_CARRY_ARM_DECLARED`. Both counts, because the
refusal has to tell "this wheel does not carry that arm" from "nothing builds that arm
in any configuration".

It is written unconditionally on every run, never accumulated, and only the family's own
translation units include it (`ArmFamily.includer`) — which is what keeps an arm-set change
from perturbing every other translation unit's argv. Decode's is the same header for its
family (`decode_selection.inc`, `ROLA_DECODE_ARM_SET_X`).

## What it refuses

`_load_arm_set` reads a family's declaration field by field and names the field it
refuses on: an unreadable schema, the wrong family, a re-keyed declaration, a domain
that does not name the key's fields or is empty, a row of the wrong width, a value
outside its declared domain, a row declared twice, a missing `shipped` list, and
rows out of canonical order. The last is the one worth stating: an arm's INDEX is
named by the selection header, by the per-arm translation units and by
`--//:carry_arms`, so it must be a property of the declaration and not of where a row
was typed — a row inserted in the middle would silently renumber every arm.

## The self-test

`bazel run //tools:gen_shards -- --self-test` requires the reader to refuse ten malformed
declarations BY NAME (the message must name the field), refuses a missing declaration,
and then installs a synthetic three-row table to show the selection header, the arm set
and the shard block rendering a subset scope (two of three rows built), which the live
declaration's default build, every row, never exercises.

`--check` regenerates into memory and diffs, and additionally asserts the selection
header's schema directly — the header is never checked in, so there is no file to diff
it against.
