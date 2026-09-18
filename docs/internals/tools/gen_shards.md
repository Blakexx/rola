# `tools/gen_shards.py` — the one enumeration of the carry family's arms

Mirrors `tools/gen_shards.py`. The ratification's view of the same machinery is
[`docs/ratification.md`](../../ratification.md#what-the-shipped-set-is-and-where-it-is-declared).

## Membership is declared; enumeration is code

`tools/manifests/shipped_set.json` declares WHICH arms exist, keyed
`(D, DV, warps_per_cta)`. This module decides what a row MEANS: which translation unit
carries it, what the selection header says about it, what the ratification's `shards`
block records. The split is what lets `setup.py`'s post-build check compare the two as
independent statements instead of restating one twice.

## What it produces

| output | checked in? | written by | consumed by |
|---|---|---|---|
| `csrc/rola/src/instantiations/carry_arm_<i>.cu` | yes — what is reviewed is what compiles | `--write` | the build's source list |
| `build/generated/carry_selection.inc` | never | `setup.py`, `tools/ratify.py`, on every run | the per-arm TUs, the carry dispatch TU, `common/arm_switch.cuh` |
| the `shards` block | recorded in each `tools/manifests/<toolchain>/sm_XX.json` | `tools/ratify.py --write` | the manifest digest |

The selection header carries two things: the `ROLA_CARRY_BUILD_<i>` selection, and the
ARM SET `arm_switch` dispatches over — `ROLA_CARRY_ARM_SET_X(F)`, plus
`ROLA_CARRY_ARM_COUNT` beside `ROLA_CARRY_ARM_DECLARED`. Both counts, because the
refusal has to tell "this wheel does not carry that arm" from "nothing builds that arm
in any configuration", which is the same distinction the battery's `require_arm` makes
and the reason an undeclared arm is never a skip.

It is written unconditionally on every run, never accumulated, and only the carry
family's own translation units include it — which is what keeps an arm-set change from
perturbing every other translation unit's argv.

## What it refuses

`_load_shipped_set` reads the declaration field by field and names the field it
refuses on: an unreadable schema, the wrong family, a re-keyed declaration, a domain
that does not name the key's fields or is empty, a row of the wrong width, a value
outside its declared domain, a row declared both shipped and test, a missing list, and
rows out of canonical order. The last is the one worth stating: an arm's INDEX is
named by the selection header, by the per-arm translation units and by
`ROLA_CARRY_ARMS`, so it must be a property of the declaration and not of where a row
was typed — a row inserted in the middle would silently renumber every arm.

## The self-test

`python tools/gen_shards.py --self-test` requires the reader to refuse ten malformed
declarations BY NAME (the message must name the field), refuses a missing declaration,
and then installs a synthetic three-row table to show the selection header, the arm set
and the shard block actually rendering rows. The live declaration carries one row today
(`tools/manifests/shipped_set.json`'s `[2, 64, 8]`), which exercises the single-row case;
the self-test's synthetic three-row table is what shows the renderer working at a width
the live declaration does not yet reach, so a change to the multi-row path is not first
seen the day a second arm ships.

`--check` regenerates into memory and diffs, and additionally asserts the selection
header's schema directly — the header is never checked in, so there is no file to diff
it against.
