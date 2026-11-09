# The build graph

The build graph is Bazel, run through bazelisk and the graph's front door. Its workspace is `MODULE.bazel` (the rule
sets and the hermetic CUDA toolkit), `.bazelrc` (the graph's settings), `.bazelversion` and `.bazeliskrc` (Bazel's
release and its digest), `bazel/` (the graph's machinery: its rules, the environment, the platform and the scripts its
actions run) and `bazel/bazel` (the front door); each `BUILD.bazel` sits beside what it describes. The graph builds the
extension (`//:library`, [below](#compile)), gates it ([below](#gates)) and runs the checks ([below](#checks)).
The commit gate is `bazel test //:gate` (`tools/git-hooks/pre-commit`, on the commit as git will write it), and the
test tiers are the graph's targets too (`//:lane`, [below](#checks)), and so are the two wheels (`bazel run //:wheels`,
[`build.md`](build.md#wheels)); ratify keeps its compile path for a new architecture's first manifest
([below](#ratify)). Nothing builds outside the graph: `pip install .` refuses (`tools/no_source_build.py`). CI runs the commit gate's checks through the graph
(`.github/workflows/lint.yml`, on a fresh runner as `.github/actions/graph` sets one up). Measurement is
`rola_devtools.build`'s
([`measurement.md`](measurement.md)), whose timing sessions hold the GPU and the clock as machine-wide counting
semaphores, which Bazel cannot do for an action.

## <a id="environment"></a>The environment's layers

| layer | made by | stored |
|---|---|---|
| the host's NVIDIA driver and GPU | the host | -- |
| the machine (base OS libraries, git, `python3`, the host C++ compiler nvcc drives) | the host, or the devbox image (`//:image`, docs/setup.md#the-devbox-image) | the image |
| bazelisk, Bazel's launcher | `tools/bazelisk_pin.json`, installed by `python3 tools/bootstrap.py` under `host.tools_dir` by its pin (never a machine key), which links `bazel` on PATH to it (`toolchain.bazel`; `check` has the row), on a host and in the devbox container alike | `host.tools_dir` |
| Bazel | `.bazelversion`, verified against `.bazeliskrc`'s digest | bazelisk's cache |
| the graph's CUDA toolkit | NVIDIA's redistributable components at release 13.0.0 (`MODULE.bazel`'s `cuda.redist_json`: nvcc 13.0.48), only the compiler's (~75 MB of the 4.7 GB set) | the shared repository cache |
| the Python environment | `requirements.lock`, synced with `--require-hashes` by the pinned uv (rola-devtools' `pins/uv_pin.json`, the graph's `@uv`, which `//:dev_env` writes venvs with too) into the pinned CPython (3.11.13), and rola-devtools from its module's source, by rola-devtools' environment extension (`MODULE.bazel`'s `environment.python` tag, `@rola_env`; `rola_devtools.graph.make_env`) | `host.build_cache/bazel/store`, one environment a key (the lock, rola-devtools' content, the Python, uv and the torch index), under a file lock; uv's cache there hardlinks the packages; `@rola_env//:flags.bzl`'s `VENV` names it, so a command naming it is keyed by the environment |
| the linker | mold at `tools/mold_pin.json`'s release (`@mold`) | the shared repository cache |
| the lints' analyzer | clang-tidy at `tools/clang_tidy_pin.json`'s wheel, its binary's digest checked (`@clang_tidy`) | the shared repository cache |
| the pinned toolkit | `tools/cuda_toolkit_pin.json`: NVIDIA's redistributable manifest by its digest (the compiler's, held equal by `tests/unit/test_cuda_toolkit_pin.py`), every component it names by the manifest's digest, unpacked into one tree as init installs it outside the graph (`@cuda_tools`, file for file init's copy): the disassemblers the SASS gate reads with (cuobjdump, nvdisasm), compute-sanitizer (the sanitizer lane's) and the whole toolkit the test tiers hand their tests (`ROLA_CUDA_HOME`); beside it cuRAND's headers, the graph's alone (clang's CUDA front end includes one) | the shared repository cache |
| the host C++ compiler's identity | its version and the digests of gcc, g++ and cc1plus (`@host_cc`), an input of every compile | the output base |
| the devices' identity | each device's name, UUID, driver and compute capability from `nvidia-smi` (`@host_gpu`, read at every fetch; a machine that names no device refuses), an input of every device test tier | the output base |

## <a id="front-door"></a>The front door

bazelisk runs `bazel/bazel` (`bazel/front_door.py`; the wrapper directory `.bazeliskrc` names, in place of
bazelisk's default `tools/`) in place of the Bazel it downloaded, so every Bazel command
in a checkout -- typed, or run by a tool -- goes through it. It runs under the machine's `python3` before any venv
exists: a stub that runs rola-devtools' front door (`rola_devtools.graph.front_door`, every RoLA repository's) from the
rola-devtools checkout `tools/bootstrap.py` finds.

- THE SHARED MACHINERY IS ROLA-DEVTOOLS' MODULE: its environment extension and generic rules (`tree_test`,
  `checkout_test`, `tool_test`, `tool_run`, `tree_part`, `pytest_tier`, `device_lane`; `bazel/defs.bzl` hands each
  rola's environment). `MODULE.bazel` pins the module at a public commit (`git_override`), the one pin of
  rola-devtools; the front door overrides that pin with the checkout it found, in an rc file of this workspace's own
  (`--override_module`), so a change made there is the one the graph runs. A machine with no checkout clones one at
  the pin first (`tools/bootstrap.py --devtools-pin`), so every graph runs on an override and `MODULE.bazel.lock` is
  the overridden resolution's (Bazel records one extension digest differently under the bare pin).

- THE MACHINE'S SETTINGS come from the dev config: the shared disk and repository caches under `host.build_cache`
  (`bazel/disk`, `bazel/repository`), in an rc file it regenerates on every run and hands Bazel after the workspace's
  (`--bazelrc`). Nothing in the checkout names a machine.
- EVERY BUILD RUNS UNDER THE HOST BUDGET (`rola_devtools.locks.host`): a command's build phase holds the slots free
  when it starts, at least one, and runs `slots / host.nvcc_threads` jobs (nvcc runs one cicc an architecture). A
  `test` command builds under the budget, then runs its tests holding none; a `run` command writes its
  launch script under the budget (`--script_path`), then runs it holding none. A test or a measurement takes its own
  locks, as outside the graph: a GPU test under a held budget would deadlock against a measurement, which takes the
  GPU before the host's slots. A timing session holds every slot, so no build runs during one.
- Run directly, without bazelisk, it refuses; so does `bazel coverage`, whose instrumented build would differ from the
  budgeted one and run outside the budget (the graph has no coverage target).

## <a id="compile"></a>The extension's compile

`//:library` is `rola_cu13/_C.abi3.so` and its build record `rola_cu13/_build_config.py`; `bazel run //:library` copies
both into the checkout's `rola_cu13/` (a copy: the library is imported from the checkout), and the build's generated
headers into `build/generated`.

- THE BUILD'S PARAMETERS are the root package's settings: `--//:archs` (default `@rola_env//:flags.bzl`'s
  `RATIFIED_ARCHS`, the toolchain's ratified manifests), `--//:carry_arms` and `--//:decode_arms` (arm indices; empty is
  every arm) and `--//:carry_parts` (`tools/gen_shards.py`'s `carry_parts_spec`). Any value but a default makes an
  iteration build, which its record says.
- THE FLAGS ARE `tools/build_flags.py`'S: the environment repository writes `@rola_env//:flags.bzl` from it
  (`bazel/flags.py`) -- the nvcc codegen flags and torch's extension defines, the host C++ flags -- and the graph states
  none of its own. rules_cuda's own additions are off on every unit (its extended lambdas, compilation-mode flags,
  random seed and host toolchain flags), so nvcc's command line is the build flags, the `-gencode` targets, `-x cu`,
  `-ccbin` and the include roots.
- EACH UNIT IS ONE ACTION OF THE GRAPH'S OWN RULE (`bazel/cuda.bzl`'s `cuda_unit`, which runs
  `bazel/cuda_compile.py`): rules_cuda's toolchain, flags and command line, as its `cuda_library` builds them, one
  position-independent object a unit, always linked. nvcc writes a dependency file (`-MD -MF`), and every declared
  header the unit did not read becomes the action's `unused_inputs_list`, so the unit's cache key holds what it read:
  a header edit recompiles the units that include it and no other. The file
  lists what every pass read, the host's and each architecture's device pass (nvcc 13.0.48: a header included only
  under `__CUDA_ARCH__ == 860` is listed when sm_86 compiles and not when sm_80 alone does). Only declared headers are
  prunable: the toolkit, the host compiler and its identity (rule 4) key every compile. The action keeps nvcc's
  diagnostics, ptxas's `-v` report among them, as `<unit>.ptxas.txt` (the `ptxas_reports` output group); the console
  shows them without the report's lines.
- ONE ARCHITECTURE SPEC A CARD: the library's transition writes `--@rules_cuda//cuda:archs` as
  `compute_80:sm_80;compute_86:sm_86`; a comma list (`sm_80,sm_86`) is one spec, which compiles every card from the
  lowest virtual architecture.
- THE INCLUDE ORDER IS SETUP.PY'S: the repository's headers, then CUTLASS's, then the generated headers (and the stamp,
  for the one unit that reads it), then torch's, Python's and the toolkit's, so a repository header wins any name
  collision (`bazel/defs.bzl`'s `include_root`).
- THE GENERATED HEADERS are actions over their generators: the arm selections, the carry parts and the carried builds
  (`rola_headers`, `bazel/headers.py`, from the settings) and the device stamp (`tools/csrc_stamp.py` over every file
  under `csrc/rola/src` but its markdown -- the digest ratify computes). Only `build_stamp.cu`'s unit reads the stamp
  and depends on it (`stamp_unit`): a digest of every source, as any unit's input it would recompile every unit on
  any edit.
- ONE UNIT LIST: `csrc/rola/units.bzl`, written and checked by `tools/gen_shards.py`, names `tools/kernels.py`'s hand
  units, the runtime's among them (`runtime_units`, `//csrc/rola:runtime`), and each generated carry arm unit with the
  arms it carries. Each arm unit is its own target, and
  `carry_arm_set` hands the link only the units `--//:carry_arms` names (`gen_shards`' grammar: indices or `all`), so an
  unwanted arm is never compiled.
- `--threads` IS THE MACHINE'S `host.nvcc_threads`, which the front door hands the graph (`ROLA_NVCC_THREADS`): the
  cicc a job runs is the count its job budget divides by.
- THE LINK is the pinned mold's, of every unit's objects (each target `alwayslink`, so every object is linked)
  and, by path, torch's `libc10.so` and `libtorch_cpu.so`, the CUDA runtime and the driver's stub: each NEEDED by its
  soname, no search path written (no unit carries a dynamic library; the units' CUDA runtime is its headers,
  `//bazel:cuda_headers` -- rules_cuda 0.3.0's own `@cuda//:cuda_headers` names a component CUDA 13 lacks).
- AN INSTRUMENT'S LIBRARY (`bazel/defs.bzl`'s `rola_instrument`): a measurement's own kernels as
  `rola_cu13/<module>.abi3.so` beside the extension, linked as it is and SASS-gated as it is (`--instrument`: it owes
  none of the source's kernels, and its probes keep only the step signatures their declarations name, the rule's
  `kept`, [`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#instrument)), for the architectures `--//:archs` names; never shipped, so no record, arm gate or
  ratify. `//:parts_library` is the part harness's driver (`_C_parts`, KERNEL_STANDARDS section 22): the calibration
  kernels (`//measure/harness/carry_parts`, with `-lineinfo` for the ledgers) and the runtime's units, the library's own
  objects.
## <a id="gates"></a>The library's gates

Building `//:library` runs the gates below; their verdicts are its outputs and its runfiles, so neither a build nor
`bazel run` hands over a library one of them has not passed. Each reruns only when its inputs change. Ratify is not
among them: the manifests are stale between milestones by design (KERNEL_STANDARDS section 16), so its verdict is a
target of its own ([below](#ratify)).

- THE BUILD GATES AND THE RECORD (`bazel/record.py`): rule 1 (`tools/toolchains.py`'s `rule1`) -- the
  graph's ptxas is the toolchain record's and the manifests' byte for byte (the tie between that record and
  `MODULE.bazel`'s toolkit pin), its cuobjdump and nvdisasm are the record's byte for byte too (the tie between the
  pinned toolkit's manifest and the compiler's, which `tests/unit/test_cuda_toolkit_pin.py` holds equal), and the environment's torch is the toolchain's CUDA major --; rule 3, every architecture ratified; rule 5, the vendored
  tree its `PIN.json`'s (`tools/build_record.py`'s `vendored_pin`); the shards, the checked-in generated units the arm
  list's (`tools/gen_shards.py`'s `check_generated`); then the record, `tools/build_record.py`'s writer.
- THE SASS GATE (KERNEL_STANDARDS section 20, `tools/sass_gate.py`) on the linked library, with the graph's
  `cuobjdump` and `nvdisasm` (`@cuda_tools`, the pinned toolkit, one `bin/` for both as an installed toolkit has
  them) and no verdict cache: Bazel's is the cache. It reds a PTX image too: closed-world rule 2 on the binary itself.
- THE ARM GATE (`bazel/gates.py`): the built table against the declaration, and the coverage sweep -- the binary asked
  what it carries, loaded by the environment's interpreter against the toolkit's driver stub, so the answer is the
  binary's on any machine. It runs outside the sandbox, where the environment's packages are, the environment's
  store path on its command line (the path carries the environment's key).

## <a id="ratify"></a>Ratify on the build

`bazel test //:ratified` is ratify's verdict on the library the graph built, every refusal of
[`ratification.md`](ratification.md#what-the-gate-refuses): `tools/ratify.py --graph` in a copy of the tree, cached on
the census, the tree and the manifests. It is tagged `manual`, so neither the gate nor the lane runs it: it runs at a
milestone and before a wheel. `bazel run //:ratify -- --write` re-ratifies in the checkout, and `-- --census --json
PATH` measures. Neither compiles: the census (`//:census`, `bazel/defs.bzl`'s `ratify_census`) is the units' own
compiles, each unit's ptxas report and object (the `cuda_unit` rule's), with the derivation harness the graph builds
(tools/BUILD.bazel's `derivation_harness`, [`internals/tools/ratify.md`](internals/tools/ratify.md#graph)). On
2026-11-08 the census equalled the compile path's (`python tools/ratify.py --census`: 750 entries, every register,
spill, local-access count and body hash) and the verdict was the same line for line. The census's disassemblers run
a unit a host-budget slot: the machine's budget under `bazel run //:ratify`, a pool of the sandbox's own (the default
slot count) under the sandboxed `//:ratified`.

THE GAP: the graph ratifies the shipped set, the architectures the toolchain's manifests already name
(`RATIFIED_ARCHS`), so an architecture's FIRST manifest has no graph form -- `--//:archs` naming a new one makes an
iteration build, which is never ratified. Until it has one, a first manifest is the compile path's (`python
tools/ratify.py --arch XX --write`, [`bringup.md`](bringup.md)), and rule 3's refusal names it.

## <a id="checks"></a>The checks

`bazel test //:gate` runs every check the graph holds -- `//:lints`, the checks over files (the repo's own and
`//csrc/rola:lints`); `//tools/lint:lint_fixtures`, each lint firing on its own fixture; and `//:commit_checks`, the
checks of the commit -- and is the commit gate:
`tools/git-hooks/pre-commit` runs it with the changes not staged for the commit set aside, and the tree the commit's
index (`GIT_INDEX_FILE`, which `@rola_tree` reads when the hook hands it in).

- THE TREE is `//:tree`: every tracked file (`git ls-files`, read by the environment extension's `@rola_tree`, which
  reads again when git's index moves), each package's part its `tracked` filegroup (`bazel/defs.bzl`'s `tree_part`,
  which names its Python files `tracked_python` and the parts a check reads); `//:tree_python` is every Python file.
  A file is in the tree once git tracks it.
- A CHECK OVER FILES IS A TEST OVER THE PART OF THE TREE IT READS (`bazel/defs.bzl`'s `tree_test`): its `data` names
  the parts it reads -- a package's tracked files, the Python files, a named directory, a configuration file -- or the
  whole tree for a check of every tracked file, so an edit reruns only the checks that read the edited file. Its
  inputs are copied into a real tree in the test's own temporary directory, the environment's tools on
  PATH and no machine's dev config; a check that asks git what is tracked gets the copy as a git index of exactly
  those files. Cached on what it reads, it reruns only when one of those files changes.
- THE LINTS live beside what they check, and the linters in `tools/lint` (its package: the scripts, their
  configuration, the modes a person runs by hand and each linter's self-test on its fixtures). `//csrc/rola:lints` are
  the kernel tree's own: clang-format, ast-grep (`tools/lint/ast_grep_gate.py`), the unused instantiations and the
  unit lints below. `//:lints` adds the repo's policies over the whole tree: ruff and its blank-line spacing, the
  standards lint (`tools/lint/lint_standards.py`), the readability spacing, the agent files' portability
  (`tools/lint/portable_claude.py`), the mirror's declarations (`rola_devtools.mirror --check`), and each ratchet
  (`tools/lint/ratchet.py <lint>`: the lint's findings against its baseline, a new finding and a fixed one each red).
- THE UNIT LINTS (`//csrc/rola:clang_tidy_device`, `clang_tidy_host`, `host_warnings`; `bazel/defs.bzl`'s `unit_lints`
  aspect and `unit_lint_test`): clang-tidy's two profiles and the host warnings, each run on one unit's own compile
  action -- its command line and its inputs -- so a lint reads what the build compiled
  ([`internals/tools/lint.md`](internals/tools/lint.md#unit-lints)). The units are the library's own (the aspect follows
  its deps), read in one configuration a ratified architecture (a split transition), whose generated headers are that architecture's build set; each verdict is
  cached on its unit's compile, and states the seconds it took. `bazel run //:compile_commands` writes the editor's
  compile database from the same commands, for the first ratified architecture.
- THE TEST TIERS (`//tests:cpu`, `unit_cuda`, `oracle`, `integration`; `//:lane` all four; `bazel/defs.bzl`'s
  `pytest_tier`): each tier one target, as pytest runs it ([`testing.md`](testing.md#running-it)), over the tree and the
  built library copied into a real tree, outside the sandbox (the machine's locks live in `/tmp`). The CPU tier is
  sharded by Bazel's protocol (`rola_devtools.graph.pytest_shards`), each shard drawing a host slot, and sees no device; a device
  tier runs alone, takes the machine's GPU lock itself, and has the devices' identity among its inputs (`@host_gpu`).
  `--test_filter` is pytest's `-k`, `--test_arg` passes pytest an argument. Every tier reads the machine's dev config
  (its locks), and hands its tests the graph's copy of the pinned toolkit (`@cuda_tools`, an input of the tier, so a
  new toolkit reruns it) as `ROLA_CUDA_HOME`, which `dev_config.cuda_home()` takes over init's copy. The sanitizer lane (`//tests:sanitizer`,
  `bazel/defs.bzl`'s `device_lane`, `manual`) runs as a device tier does, passing the toolkit's own compute-sanitizer
  by its path ([`testing.md`](testing.md#the-sanitizer-lane)).
- THE CHECKS OF THE COMMIT (`//:commit_checks`, `bazel/defs.bzl`'s `checkout_test`): what git knows rather than
  a file set -- no ratchet's baseline grown against HEAD's (`tools/lint/ratchet.py --grown`), every doc reference
  resolving (`tools/docrefs.py`), the devbox image the one the tree names and checked (`tools/dev.py container
  gate`, which builds `//:image`). Each runs in the
  checkout, outside the sandbox and never cached, its input the repository's state.

## <a id="packages"></a>The packages

The tree keeps the layout of a Python package with a native extension, as FlashAttention and PyTorch do (ruled
2026-11-08): `rola/` beside `csrc/`, and the tests that drive the kernels through the op against the oracle in `tests/`,
since the kernels serve the op alone and those tests grade the two as one. A BUILD file sits beside what it describes,
its lints among it, and `bazel/` holds only the graph's own machinery:
- the root package: the build's parameters, the extension and the part harness's library, the devbox image, the
  repo-wide checks and the gate;
- `csrc/rola`: the kernel tree, its units and its own lints; `csrc/third_party`: the vendored trees;
- `tools`: the pins and records the graph reads, the generators it runs, and each tool's run target; `tools/lint`: the
  linters, the modes a person runs by hand and each linter's self-test;
- `measure/harness`: the part harness's run target; `measure/harness/carry_parts`: its driver's calibration kernels;
- `tests`: the tiers.

A C++ API of the kernels' own would make a split by language right.

## <a id="rules"></a>The rules

Each is a trap a setup like this falls into, and what avoids it (checked against Bazel 9.3.1's own options where it
names one).

1. BZLMOD ONLY, PINNED: `MODULE.bazel` and its `MODULE.bazel.lock` are committed with `.bazelversion`; the commit gate
   runs with `--lockfile_mode=error` (`tools/git-hooks/pre-commit`), so a dependency resolved differently from the
   lock refuses the commit rather than drifts.
2. NO MACHINE IN THE CHECKOUT: `.bazelrc` holds the graph's settings and names no path; the machine's come from the dev
   config through the front door. Startup options are never inside a `--config`.
3. THE DISK CACHE IS BOUNDED, AND ITS BOUND RUNS: `--experimental_disk_cache_gc_max_size` caps the shared cache at
   5 GB, the graph's budget on every machine: a full build of `//:library` puts 34 MB in it (measured 2026-11-08), so
   the bound holds ~150 builds' outputs across every worktree of a machine, and the collector runs only after
   the server has idled `--experimental_disk_cache_gc_idle_delay` (5 minutes), so the server's `--max_idle_secs` (15
   minutes) stays above that delay -- below it the cache grows without bound.
   The repository cache's path never moves: Bazel 9 keeps every fetched repository's contents inside it
   (`contents/`), and each output base links into them, so a moved cache breaks every output base until it is
   expunged.
4. EVERY TOOL AN ACTION RUNS IS DECLARED: hermetic (the toolkit), or its identity is an input of the actions that run
   it (the host C++ compiler's version). A tool the graph does not see is one whose change shares stale results between
   machines. An identity is never a header: a compile drops a declared header it did not read from its key (rules_cc's
   C++ compile by its dependency file, as the CUDA units by theirs: a header nothing includes was edited and nothing
   recompiled, 2026-11-08), so `@host_cc//:identity.txt` is a file each compile takes outright (`host_cc` on a CUDA
   unit, `additional_compiler_inputs` on a C++ one).
5. A GPU TEST'S RESULT IS THE MACHINE'S: its key carries the devices' identity (`@host_gpu`: each one's name, UUID,
   driver and compute capability, read at every fetch), so a pass is never replayed for another device; it takes the machine's GPU lock itself, since Bazel serializes tests only within one invocation and
   every worktree runs its own server.
6. NO VOLATILE STAMP IN A COMPILE: the commit and other status reach only the final records (the build record, the
   wheels' metadata), never an action whose output is reused.
7. TESTS ARE ONE TARGET A TIER OR AREA, SHARDED: a test process imports torch (seconds), so one target a file costs
   minutes; the runner honours `$TEST_SHARD_STATUS_FILE`, and each target states its timeout.
8. A RUN TARGET READS `$BUILD_WORKSPACE_DIRECTORY`: `bazel run` starts in the runfiles tree, not the checkout.
9. WORKSPACES AND OUTPUT BASES LIVE ON THE LINUX DISK, never under `/mnt` (WSL's Windows mounts: slow, and file
   watching is undefined there).
10. THE SANDBOX STAYS ON: Bazel 9's hermetic `/tmp` keeps nvcc's temporary files apart, so nothing runs
    `--spawn_strategy=local` to dodge it. Three kinds of work leave it: the arm gate, because it imports the
    environment's packages from the store, which the sandbox does not hold (the store path on its command line carries
    the environment's key); the checks of the commit, whose input is the checkout itself (rule 15); and the test
    tiers, which take the machine's locks in `/tmp`, a sandbox's own.
11. CUDA THROUGH RULES_CUDA'S OWN SETTINGS: the architectures through `--@rules_cuda//cuda:archs`, one spec a card
    (cubins only, no PTX); a host compile flag reaches nvcc's host pass only through rules_cuda's host options; an external repository's
    headers through the target's include options.
12. AN ACTION'S SCRIPT FINDS ITS SIBLINGS UNRESOLVED (`Path(__file__).absolute()`): in the sandbox it is a link, and
    resolving it imports the checkout's files instead of the action's declared inputs.
13. TWO SCHEDULERS SHARE ONE HOST through the host budget: the front door holds it for every build phase and never for
    a test or run phase; a caller that holds host slots never starts a Bazel build, whose actions run in a server that
    does not inherit the caller's hold.
14. A CHECK OVER FILES READS A COPY: the runfiles are links into the checkout, so a tool that resolves a path, walks
    a directory or asks git reads the checkout instead of the check's inputs; `tree_test` copies them into a real tree
    (a git index of exactly them where the check asks git), and the test's own script is not among them.
15. A CHECK OF THE COMMIT IS NEVER CACHED: its input is the repository's state (HEAD, the index, the commits a doc
    pins), which no file set names, so it runs in the checkout tagged `external` -- never as a cached test that
    passes on a stale verdict.

## <a id="use"></a>Using it

    python3 tools/bootstrap.py                                      # the pinned bazelisk, `bazel` linked to it
    bazel run //:init                                               # the machine's dev config, hooks and router
    bazel run //:dev_env                                            # this checkout's venv, which its hooks run from
    bazel test //:gate                                              # every check the graph holds
    bazel run //:compile_commands                                   # the editor's compile_commands.json
    bazel test //:lane                                              # every test tier
    bazel run //:library                                            # the shipped build, into rola_cu13/
    bazel run //:library --//:archs=86 --//:carry_arms=3            # an iteration build
    bazel run //:parts_library --//:archs=86                        # the part harness's driver, beside the library
    bazel run //:image_load                                         # the devbox image, into Docker (docs/setup.md)
    bazel run //:image_push                                         # the devbox image, published to Docker Hub
    bazel run //tools:<tool> -- <its arguments>                     # a tree tool, in the checkout on its own venv
    bazel run //tools/lint:ratchet -- <lint> --shrink               # a ratchet's baseline, shrunk with its fixes
    bazel run //:docrefs -- accept <doc>                            # a doc's citations pinned, read against their targets
    bazel run //:measure -- run declare.py:all --only <glob>        # a measurement session (rola-devtools' build system)
    bazel run //:results -- show|history|latest|verdict|commit      # the measurements store
    bazel run //:mirror -- --check                                  # the public mirror's export (CI's mirror job)

Never pass `--jobs` or a cache path: the front door sets both. Nothing the graph builds lands in the checkout but what a
run target writes there -- the library `bazel run //:library` copies in, with the build's generated headers into
`build/generated` (where the tools that compile the tree's units read them: `tools/life_ranges.py`,
`tools/header_selfcheck.py`; a header only when its text changed), and the database `bazel run //:compile_commands` writes: Bazel's output links are off (`--experimental_convenience_symlinks=clean`).
