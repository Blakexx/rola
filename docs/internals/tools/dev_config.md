# `tools/dev_config.py` — every machine-dependent value, in one directory

KERNEL_STANDARDS §23. A value that depends on the machine lives in the dev config directory and is read through
this loader. Examples are where the CUDA toolkit is, how many compile slots the host has, how the SM clock is
locked, and where the measurements record is. The tree never reads a setting from the process environment, and
never spells a machine path in a tool.

**The directory** is `~/.config/rola/`, or the directory the `ROLA_DEV_CONFIG` variable names (a test). It holds one JSON file per section. A missing file or key takes its declared default. An undeclared file
or key is refused, so a typo fails instead of silently taking the default.

**The reader is rola-devtools'** (`rola_devtools.config`), and so are the `host` and `clock` sections: the machine's
locks (`rola_devtools.locks`) read them, for every repository and container on the host. This loader declares rola's
own sections beside them and, owning the directory, refuses a file no section declares. rola-devtools is a development
dependency, pinned in `MODULE.bazel` (its module's `git_override`): the graph's environment carries it from the module's
source, and `bazel run //:init` links the `workspace.devtools` checkout into every venv on the machine.

| file | keys |
|---|---|
| `toolchain.json` | `cuda_home` (a CUDA toolkit older checkouts' files name; this tree uses the pinned one, `cuda_home()`), `ncu` (a Nsight Compute CLI older checkouts' files name; this tree finds the pinned one by its pin, `ncu()`, which refuses until init installed it), `sccache`, `sccache_enabled` and `mold` (older checkouts' compiler cache and linker; this tree reads none of them, its graph links with its own mold), `clang_tidy` (the host's clang-tidy, which checkouts before the build graph's unit lints install and read; this tree reads none of it), `bazel` (where the bootstrap links the pinned bazelisk on PATH) |
| `host.json` | `nvcc_threads`, `build_jobs`, `budget_slots`, `lock_dir`, `gpu_lock`, `gpu_shared_slots`, `nice`, `locks_trace`, `scratch`, `build_cache` (the build systems' caches: the graph's under `bazel/`), `tools_dir`, `windows_system32`, `wsl_lib` |
| `clock.json` | `ghz`, `lock`, `unlock` (null `ghz`: the host measures unlocked) |
| `store.json` | `root` (the measurements store: a rola-results checkout) |
| `workspace.json` | `suite` (the rola-bench checkout whose root `declare.py` measures this tree), `devtools` (the rola-devtools checkout, the public mirror's declared build system, linked into every venv), `worktrees` (the one folder worktrees live in), `base_venv` (a venv the checkouts on the machine whose hooks run from it read; this tree reads none of it) |

<a id="bootstrap"></a>**Before the reader is importable.** The graph's front door (`bazel/front_door.py`, a stub over
`rola_devtools.graph.front_door`) runs
under the machine's `python3`, before any venv exists. `tools/bootstrap.py` reads one value the standard-library way --
`workspace.devtools`, from `workspace.json` in the directory the reader's own rule names -- and the front door imports
the reader and the host budget from that checkout. Run (`python3 tools/bootstrap.py`), it is the machine's one step
outside the graph: it imports the reader the same way and runs `tools/dev.py`'s `install_bazelisk`, the pinned bazelisk
and the `bazel` link, since nothing can `bazel run` before `bazel` exists. The sibling checkouts' location (`SIBLINGS`, the default of
`store.root`, `workspace.suite` and `workspace.devtools`) is defined there once, for both. Two read-outs serve a
machine that has nothing yet, CI's runner (`.github/actions/graph`): `--devtools-pin`, the rola-devtools repository
and commit `MODULE.bazel` pins, which it clones beside the checkout first, and `--bin-dir`, the directory of
`bazel`'s link (`toolchain.bazel`), for its PATH.

Each key's type, default and meaning are declared in `SCHEMA`. `bazel run //:show` prints every
resolved value and the file it came from ([dev.md](dev.md)).

**The API.**
- `get("section.key")` returns a value.
- `cuda_home()` and `cuda_bin(tool)` resolve the toolkit: the pinned one (`tools/cuda_toolkit_pin.json`) -- in a
  test tier the graph's copy, which the tier hands in (`ROLA_CUDA_HOME`), else init's under `host.tools_dir`, refused
  until init installed it. Every compile, disassembly and assembler check outside the graph's own rules uses them, so
  the ptxas a gate reads is the one the build ran.
- `scratch(name)` returns a tool's scratch directory.
- `locks_mounted()` says whether this machine is a container whose launcher mounted the host's locks at
  `CONTAINER_LOCKS` (from the process's own mount table, never a directory a tool could have made).
- `devbox_image()` returns the devbox image's digest inside such a container, which its launcher mounts read-only at
  `CONTAINER_IMAGE`, else the empty string. A file the launcher mounts, not an environment variable: an editor's
  container would carry a variable as a setting of its own.
- `windows_tool(rel)` returns a Windows tool under WSL.
- `pinned(name)` returns where a pin's binary is installed (`tools/<name>_pin.json`: its folder under `host.tools_dir`,
  the binary in it), the one lookup dev.py and the measurement tools use; `ncu()` returns the pinned Nsight Compute,
  refusing until init installed it.
- `detect()` returns what `init` writes.
- `reload()` rereads the files, here and in the locks' reader, for a test that points `ROLA_DEV_CONFIG` at a
  temporary directory.

**Environment variables that remain.** One list in the loader names them, and nothing else may read one.
`tools/lint/lint_standards.py` enforces this: it checks Python reads through the syntax tree (constants and
f-string prefixes resolved) and shell expansions.
- `HANDOFFS`: values one of the tree's own processes passes to a child it starts.
  - `ROLA_DEV_CONFIG` itself.
  - The toolkit ratify's compile path and the header self-check set for torch's extension module (`CUDA_HOME`).
  - The graph's copy of the pinned toolkit a test tier hands its tests (`ROLA_CUDA_HOME`).
  - pytest-xdist's worker id.
  - The agent harness's campaign directory.
  - The login name printed into the sudoers rule `bazel run //:clock` suggests (`USER`).
  - The search path bazelisk hands what it runs, the front door's directory first (`PATH`; `check` finds the user's
    `bazel` past it).
  - `host.nvcc_threads`, which the front door hands the graph's compile flags (`ROLA_NVCC_THREADS`).

  The Bazel release bazelisk hands the front door (`BAZEL_REAL`) and the shared store the front door hands the
  graph's environment (`ROLA_GRAPH_STORE`) are rola-devtools' to read, not this tree's.

What one build is, is the graph's settings (`--//:archs` and the arm and part settings, `docs/build-graph.md#compile`),
never an environment variable.

**What it replaced**, each variable deleted rather than kept as a fallback:
- `ROLA_NVCC_THREADS`, `MAX_JOBS` as a setting, `ROLA_HOST_BUDGET_SLOTS`, `ROLA_HOST_BUDGET_DIR`,
  `ROLA_GPU_LOCK`, `ROLA_GPU_SHARED_SLOTS`, `ROLA_NO_NICE`, `ROLA_LOCKS_TRACE`.
- `ROLA_SCCACHE`, `ROLA_SCCACHE_BIN` as a setting, `MOLD_PATH`, `CUDA_HOME` as a setting.
- `ROLA_CLOCK_CONFIG`, `ROLA_RESULTS_DIR`, `ROLA_DB`, `ROLA_IMAGE_DIGEST`, `ROLA_FLA_CROSSCHECK`,
  `ROLA_MACHINE_SALT`.
- The shell tools' `ROLA_BASE_VENV`, `ROLA_WORKTREES_DIR`, `ROLA_WORKTREE_BUILD`, `ROLA_BUILD_SLOTS`,
  `ROLA_BUILD_DIR`.
- The `/usr/local/cuda/bin/...` and `/mnt/c/Windows/...` paths spelled in eight tools, and three hardcoded
  `/tmp` scratch directories.
