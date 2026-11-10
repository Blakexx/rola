# `tools/` — the map: what each entry point is, who calls it, and what it records

The tools inventory. Every script under `tools/` is here with one line of what it is, what runs it, and
what it writes; a tool with more to say has its own page beside this one, and a tool without one is fully stated by its
own module docstring. A tool that nothing runs is a tool to delete (`DELETIONS.md`), so the CALLED BY column is the
column that matters.

**No tool writes a measurement record except through a declared target.** A measurement is a target of
`declare.py` (`rola_devtools.build`), its result is stored by a store target beside it, and the tool itself prints or
writes `--json`. The two exceptions are named below: `compose_ledger` is a driver that builds what it measures, and a `--calibrate`
run of `pipe_timeline` writes the plateau it reads back as every later run's scale.

## The build

| tool | what it is | called by | writes |
|---|---|---|---|
| [`dev.py`](dev.md) | the one command that sets up, checks and describes a machine (a host, or the devbox container), and checks the devbox image | a person, through its run targets; the commit gate's §23 check | the `environment` record (its container check), `tools/devbox_image.json` |
| [`dev_config.py`](dev_config.md) | every machine-dependent value, read from one directory of JSON files | every tool here, the tests | — |
| [`build_flags.py`](../../build.md) | THE ONE SOURCE of the nvcc flag list | the graph (`bazel/flags.py`), `ratify.py`'s compile path | — |
| `build_record.py` | the build record `_build_config.py`, rule 3 (which assembles the ratification it names) and rule 5 (the vendored tree's pin) | the graph (`bazel/record.py`), `ratify.py` | the record |
| `csrc_stamp.py` | the csrc content hash and the header that compiles it into the binary as the device stamp | `ratify.py`, the graph's stamp action | the stamp header |
| [`bootstrap.py`](dev_config.md#bootstrap) | what a tool knows before the dev config's reader is importable: the sibling checkouts, the rola-devtools checkout; run, the machine's one step outside the graph (the pinned bazelisk) | `dev_config.py`, the graph's front door, a person setting up a machine | the `bazel` link |
| [`gen_shards.py`](gen_shards.md) | THE ONE ENUMERATION of the carry family's instantiation matrix | `ratify.py`, the provider, the graph's header, record and arm-gate actions (`bazel/headers.py`, `bazel/record.py`, `bazel/gates.py`) | the generated per-arm units, `csrc/rola/units.bzl` and the selection headers |
| [`toolchains.py`](toolchains.md) | each toolchain as a named record of the assembler a fatbin is built and ratified under | `ratify.py`, `dev.py`, the graph's flags and record (`bazel/flags.py`, `bazel/record.py`) | — |
| `mold_toolchain.py` | the linker's provenance: the graph's mold, by its version and digest | `ratify.py` | — |
| [`wheels.py`](wheels.md) | the graph's library of a clean, ratified commit, packaged as the pure-Python wheel and the binary plugin | a person, at a release (`bazel run //:wheels`) | the wheels |
| `no_source_build.py` | `pyproject.toml`'s build backend, which refuses a source install by name | pip | — |
| `supported.py` | the README's supported-configuration table — a REPORT, not a claim | a person, after an arm change | the table |
| [`kernels.py`](kernels.md) | THE KERNEL LIST: every device unit the build compiles and every kernel the source defines, derived from the tree | the graph's unit list (`gen_shards.py`), `ratify.py`, every kernel gate | — |
| `build/dag_to_dot.py` | a DAG tuple (`rola.engine.dags.*`) as Graphviz DOT and SVG | a person, after a DAG change | the `.svg` beside its page |
| `git-hooks/pre-commit` | THE COMMIT GATE: `bazel test //:gate` on the commit as git will write it ([`build-graph.md`](../../build-graph.md#checks)) | git, every commit | — |
| `git-hooks/post-commit` | THE ADVISORY REVIEW: `review_gate.py` on HEAD in the background (`review_gate.md`) | git, every commit | — |
| [`review_gate.py`](review_gate.md) | a commit, range or staged diff reviewed by an LLM against CONSTITUTION and KERNEL_STANDARDS, each finding citing a rule and a line; advisory | the post-commit hook, a person | `rola-review/` in the git common dir |

## Ratification and the assembled code

| tool | what it is | called by | writes |
|---|---|---|---|
| [`ratify.py`](ratify.md) | the spill manifest, the census, the SASS-body gate and the derivation | a person at a milestone, the graph's `//:ratified`, `//:ratify` and `//:wheels` | `tools/manifests/` |
| [`devbox_packages.py`](../../setup.md#the-devbox-image) | the devbox image's package layer adds packages, never changes the base's, and carries nothing at its merged-/usr links | the graph's `//:devbox_packages` | — |
| [`oci_layout.py`](../../setup.md#the-devbox-image) | an OCI image layout, read: its one image's manifest digest (the devbox image's identity), the tree's tag for it, and its layers | `dev.py`'s container commands, the graph's `//:image_tag`, `usrmerge_layer.py`, `devbox_packages.py` | — |
| [`usrmerge_layer.py`](../../setup.md#the-devbox-image) | the package layer for a merged-/usr base: what the packages ship under the base's top-level links into /usr, read from the base, goes under /usr | the graph's `//:devbox_debs` | the image's package layer |
| [`derivation.py`](ratify.md#graph) | the derivation's golden harness, rendered from the declared cases | `ratify.py`, the graph's `derivation_harness_source` genrule | the harness source |
| `sass.py` | THE DISASSEMBLER SEAM: the one `cuobjdump`/`nvdisasm` invocation and line grammar; a frame map is one function's (`section`: offsets restart in every function) | `ratify.py`, `sass_gate.py`, `region_ledger.py`, `life_ranges.py` | — |
| `kernel_targets.py` | the instruments' kernel families: each one's profiler filter, provider arm, cubin member, sources, budget and unit (`kernel_targets.md`) | `stall_census.py`, `pipe_counters.py`, `phase_ledger.py` | — |
| [`scale_slack.py`](scale_slack.md) | whether a bound looser than the exact read product can be the carry's f16 scale: each candidate's slack over it, and the carry's spread at its scale against two runs at the exact bound's (the scale pass's own bytes among them) | a person | `--json` |
| `quality.py` | the output's per-token error against the fp64 recurrence at each pairing of the carry's and the intra's accumulations (`quality.md`) | `declare.py` (`quality`) | `rola/quality` |
| `realized_support.py` | a layer construction's realized routing support: each level's and side's nonzero digits a token in the bundle the producer emits (`realized_support.md`) | `declare.py` (`support`) | `rola/support` |
| `sass_gate.py` | the ptxas signatures of KERNEL_STANDARDS §20, red on any of them; guarded convergence fallbacks counted; an unchanged cubin's verdict replayed (`sass_gate.md`) | every iteration build; the graph's SASS gate action (`--cuobjdump`/`--nvdisasm`, its toolkit); the `sass` instrument | `--json`, `--loops` (each step loop's reading and the lines its lints count), `--no-cache` |
| `sass_bodies.py` | per-instantiation SASS body hashes, the codegen-equivalence instrument | a person, comparing two builds | `--json` |
| `sass_control.py` | the SASS control words (stall count, yield, scoreboards, reuse) and the loop a function compiles to; its `MACHINES` table is the static simulator's per-architecture model | `pipe_sim.py`, `burst_gate.py` | — |
| [`burst_gate.py`](burst_gate.md) | what every burst loop's SASS must satisfy, deterministically (KERNEL_STANDARDS §23) | the `sass` instrument | exit 1 when red |
| [`pipe_sim.py`](pipe_sim.md) | a burst loop's steady-state tensor-pipe occupancy from its SASS alone | a person, before a kernel runs | `--json` |
| `header_selfcheck.py` | every header compiles alone | a person, after moving a declaration | — |

## The instruments (each a target of `declare.py`)

| tool | what it is | called by | writes |
|---|---|---|---|
| [`phase_ledger.py`](phase_ledger.md) | one carry launch with the kernel's phase clock bound | the `phases` target | `--json` |
| [`op_timeline.py`](op_timeline.md) | a subject's calls under torch's profiler: kernel busy time against the GPU's idle, the gaps and the host work in them | a person | — |
| [`pipe_counters.py`](pipe_counters.md) | the profiler's pipe and resource counters for one launch | the `counters` target | `--json` |
| [`pipes.py`](pipes.md) | one counter set for any launch and its pricing per card: each pipe's cycles, the binding one | `lab.py`, `pipe_counters.py` | -- (a module) |
| [`carve_model.py`](carve_model.md) | the carry's mechanics for any carve on any card, over a real cell's lists (KERNEL_STANDARDS §32.2) | -- (in build: the carve study) | prints |
| [`design_floor.py`](design_floor.md) | the value-carve carry's counts and floors a CTA-window over a cell's real lists, any box, any card (ESTIMATED: no kernel yet) | a person (the carry's design) | prints |
| [`readout_scaling.py`](readout_scaling.md) | the readout's output and operand scaling against the fp64 oracle on a cell's real state (the dense side's accumulator form) | a person (the carry's design) | prints |
| [`overlap_gate.py`](overlap_gate.md) | each kernel's perf rows against the largest of their floors (§28) | -- | exit 1 when red |
| [`stall_census.py`](stall_census.md) | every warp-stall sample of one launch, by source line | the `census` target | `--json` |
| [`pipe_timeline.py`](pipe_timeline.md) | the pipes' activity over one launch, measured on silicon | the `timeline` target | `--json`; a `pipe_timeline.scale` record under `--calibrate`, which it reads back as later runs' scale |
| [`life_ranges.py`](life_ranges.md) | peak LIVE registers per source region, beside ptxas's allocation | the `registers` target | `--json` |
| `region_ledger.py` | executed instructions and stall samples per COMPONENT, against `tools/budgets/` | a person, attributing a phase | `--json` |
| [`compose_ledger.py`](compose_ledger.md) | a phase's time attributed to its parts by composition — a DRIVER: it builds each part | a person, after a component lands | its own `compose_ledger` record |
| `roofline.py` | the intra's wall time on one cell beside the device's own u8 IMMA ceiling — the fraction of the machine, not the wall time | the `roofline` target | `--json` |
| [`budgets/`](budgets.md) | each kernel's instruction budget by class, per MMA (KERNEL_STANDARDS §29) | `region_ledger.py`, the lab | — |
| [`bank_model.py`](bank_model.md) | the shared-memory wavefronts a warp instruction costs, from its lanes' byte addresses (§30) | a person, before a GPU run; the lab | — |
| [`lab.py`](lab.md) | the lab: small SASS-gated CUDA experiments whose measurements become kernel rules ([`docs/lab/`](../../lab/README.md)) | a person | `measure/lab/results/` |

## The oracle's own gates

| tool | what it is | called by | writes |
|---|---|---|---|
| `sanitize_oracle.py` | the compute-sanitizer gate on the facts, intra, decode, producer and entmax oracle cells | the sanitizer lane, `//tests:sanitizer`, which passes it the toolkit's own compute-sanitizer | — |
| `sanitize_cell.py` | one cell, one launch, no test framework — the racecheck lane's driver | `sanitize_oracle.py` | — |
| [`docrefs.py`](docrefs.md) | every citation a doc makes, pinned to the content it was read against | the commit gate | `docs/references.lock.json` |

`tools/lint/` is its own family with its own page ([`lint.md`](lint.md)): the ratchets, the standards lint and their
baselines, and the coverage declaration `tools/coverage.json` ([`coverage.md`](coverage.md)).
