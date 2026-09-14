# Provenance

This repository's git history begins at the extraction. That is deliberate, and
this document is the record that makes it safe: **the full development history of
everything here is preserved elsewhere, and this file names exactly where.**

## Where this code came from

| | |
|---|---|
| **Source repository** | `github.com/Blakexx/flash-linear-attention-rola` — a fork of [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) |
| **Source branch** | `rola-v3` |
| **Source commit** | `e27d4acc` — *"Retire the spike naming: the probe kernel is the production consumer, by name too"* |
| **Merge commit on the fork's mainline** | `b71608f9` — *"Merge rola-v3 into the mainline, ADDITIVELY"* (parents `3c78af94`, `e27d4acc`) |
| **Extraction date** | 2026-08-01 |

The `rola-v3` branch carries 503 commits of RoLA V3 development. It was merged
into the fork's mainline **before** this extraction, specifically so that the fork
retains the complete history whatever this repository's history policy is. The
merge is additive: the branch's development-time purge of the upstream
flash-linear-attention code was reverted inside the merge commit, so the fork's
mainline is the union of upstream FLA and RoLA V3.

## Why the history starts here rather than being filtered in

Considered and rejected: `git filter-repo` of the RoLA paths into this repository.

The extraction **renamed essentially every file** (`v3_consumer.cu` →
`csrc/rola/src/consumer.cu`, `ops/rola/routing/schedule_v3.py` →
`rola/planner/schedule.py`, and so on for ~50 paths — both of those destinations
have since been deleted outright, P67 D2), and it also renames the C++
symbols the ratification manifest is keyed on. `git blame` continuity — the only
real benefit of filtering — is therefore not actually delivered without a large,
error-prone, and unverifiable `--path-rename` script. Meanwhile the filtered
history would carry the entire superseded V2/CuTe/Triton lineage, the `spike`
naming era, and hundreds of interleaved commits belonging to flash-linear-attention
rather than to RoLA.

Starting fresh means every commit in this repository is a commit that passed the
release bar. It is also the reversible choice: a filter-repo import can still be
performed later; a filter-repo on a published repository cannot be undone.

## Licensing lineage

This package is licensed **Apache-2.0** (see `LICENSE`, `NOTICE`).

The upstream project, flash-linear-attention, is MIT-licensed. The RoLA CUDA
kernels, routing producers, ratification tooling, and evidence machinery
are original work and are not derived from it. The files whose lineage does trace
to the fork — chiefly the layer scaffolding conventions and small cache helpers —
are noted in `NOTICE`; MIT permits the relicensing of derivative distributions
provided the original copyright and permission notice are retained, which `NOTICE`
does.

### The vendored CUTLASS/CuTe subtree

`csrc/third_party/cutlass` is NVIDIA CUTLASS, BSD-3-Clause, vendored **unmodified**:
the `include/` subtree there is byte-identical to upstream tag `v3.5.1`, commit
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`, which is what makes the pin checkable
against something other than itself (`git clone --branch v3.5.1 && diff -r`).

It is vendored rather than fetched because the build is closed-world: a codegen input
that a network fetch supplies at build time is an input the ratification manifest
cannot describe. `csrc/third_party/cutlass/PIN.json` records the tag, the commit, the
licence and a digest of the whole subtree; `setup.py::_gate_vendored` (closed-world
rule 5) re-computes that digest before every compile and refuses a build whose
vendored headers drifted, and the pin is stamped into `rola/_build_config.py` so an
installed wheel answers "which CUTLASS is inside this `.so`?" with no checkout.

BSD-3-Clause requires the copyright notice, the condition list and the disclaimer to
be retained in redistributions; `csrc/third_party/cutlass/LICENSE.txt` is the upstream
text verbatim and `NOTICE` points at it.

## Related repositories

| Repository | Role |
|---|---|
| `Blakexx/flash-linear-attention-rola` | The FLA fork. Full RoLA development history; the HuggingFace `RoLAConfig`/`RoLAForCausalLM` model classes; the one-clone reproduction vehicle for the paper. Consumes this package. |
| `Blakexx/rola-zoology` | Synthetic-task (MQAR and relatives) evaluation harness. |
| `Blakexx/rola-bench` | Language-modelling benchmark suite and named experiment configurations. |
| `Blakexx/rola-paper` | The paper. The canonical **claims** document; `docs/spec.md` here is the canonical engineering **contract** document. They cite each other; neither subsumes the other. |
