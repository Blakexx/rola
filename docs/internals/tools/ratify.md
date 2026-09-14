# `tools/ratify.py` — what the gate measures beyond the codegen manifest

Mirrors `tools/ratify.py`. The manifest itself, its outcomes, its self-test and the
closed-world rules are [`docs/ratification.md`](../../ratification.md); this page
covers the two records that are not per-arch codegen and the generated headers the
gate must write before it measures anything.

## Three artifacts, three lifetimes

| artifact | scope | written by | checked by |
|---|---|---|---|
| `tools/manifests/<toolchain>/sm_XX.json` | per toolchain, per arch | `--write` | every run |
| `tools/manifests/derivation.json` | arch-independent | `--write-derivation`, and `--write` | every run |
| `build/generated/*.inc` | per run, never checked in | `setup.py` and this gate | — (they are inputs) |

`--write-derivation` exists as its own mode because the derivation is host code
measured by one small compile: refreshing it should not require a whole codegen
ratification, and a codegen ratification must not be able to leave it behind. `--write`
therefore does both, outside the per-arch loop, so a re-ratification can never file
codegen numbers from one tree beside derivation goldens from another.

## The generated headers are written BEFORE anything is measured

`measure_units` writes `carry_selection.inc` and `csrc_stamp.inc` from the same
functions `setup.py` calls. Both are unconditional overwrites: `arms=None` means the
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
