# `tools/ratify.py` — what the gate measures beyond the codegen manifest

Mirrors `tools/ratify.py`. The manifest itself, its outcomes, its self-test and the
closed-world rules are [`docs/ratification.md`](../../ratification.md); this page
covers the two records that are not per-arch codegen, the generated headers the
gate must write before it measures anything, and the gate on the graph's build.

## Three artifacts, three lifetimes

| artifact | scope | written by | checked by |
|---|---|---|---|
| `tools/manifests/<toolchain>/sm_XX.json` | per toolchain, per arch | `--write` | every run |
| `tools/manifests/derivation.json` | arch-independent | `--write-derivation`, and `--write` | every run |
| `build/generated/*.inc` | per run, never checked in | the compile path, and `bazel run //:library` | — (they are inputs) |

`--write-derivation` exists as its own mode because the derivation is host code
measured by one small compile: refreshing it should not require a whole codegen
ratification, and a codegen ratification must not be able to leave it behind. `--write`
therefore does both, outside the per-arch loop, so a re-ratification can never file
codegen numbers from one tree beside derivation goldens from another.

## The generated headers are written BEFORE anything is measured

`measure_units` writes `carry_selection.inc` and `csrc_stamp.inc` from the same
functions the graph's header and stamp actions run. Both are unconditional overwrites: `arms=None` means the
shipped set, never "whatever the last build left behind". A gate that measured against
headers it did not write would be measuring a tree nobody builds — and an arm-subset
build that leaked its scope into a later ratification is a shape this project has
already shipped twice.

## What the derivation record does not cover

The derivation's code hash is over the files `derivation_cases.json` DECLARES
(`common/geom.cuh`), not over `csrc/`. Widening it to the whole tree would expire the
record on every kernel edit, and a record that expires constantly is one people
re-write without reading. The wholesale trigger for the CODEGEN remains
`csrc_digest()`; these are two triggers for two things, and both are checked.

## <a id="graph"></a>On the graph's build: `--graph`

`--graph CENSUS --runfiles DIR` measures the graph's build and compiles nothing. The compile path recompiles the
build's units under the build's flags so that its numbers describe the shipped binary. On the graph, those units'
compiles are the build: each `cuda_unit` keeps ptxas's report beside its object, so reading the two measures the
binary itself, by the same two readers the compile path applies to its own compile. There is one compile, not a
second one the census must agree with. The census record (`bazel/defs.bzl`'s `ratify_census`, every path relative to
`DIR`) names each unit the library links, the graph's ptxas, disassemblers and mold, the derivation harness the graph
built, and the build's parameters; the readers read with the graph's tools.

- THE SCOPE IS THE BUILD'S PARAMETERS: `--arch`, `--shard` and `--arm` are refused, and an iteration build is
  decided as the build's record decides it (`gen_shards.is_iteration`, `bazel/record.py`). An iteration build is
  never verified or written; its `--census` is a measurement.
- THE DERIVATION HARNESS is built by the graph: `tools/derivation.py` renders it from the declared cases with the
  standard library only, so a build action can run it before any venv, and tools/BUILD.bazel's
  `derivation_harness` compiles it as the extension's units are compiled and links it. Ratify runs that binary and
  refuses it unless its source is this tree's rendering.
- PROVENANCE: `toolchain.mold` is the graph's mold, the linker of the measured library; the compile path links
  nothing and records none.

`bazel test //:ratified` is the verdict in a copy of the tree, and `bazel run //:ratify
-- --write` (or `-- --census --json PATH`) runs in the checkout
([`build-graph.md`](../../build-graph.md#ratify)).
