# `tools/dev_config.py` — every machine-dependent value, in one directory

KERNEL_STANDARDS §23. A value that depends on the machine lives in the dev config directory and is read through
this loader. Examples are where the CUDA toolkit is, how many compile slots the host has, how the SM clock is
locked, and where the measurements record is. The tree never reads a setting from the process environment, and
never spells a machine path in a tool.

**The directory** is `~/.config/rola/`, or the directory the `ROLA_DEV_CONFIG` variable names (a container, a
test). It holds one JSON file per section. A missing file or key takes its declared default. An undeclared file
or key is refused, so a typo fails instead of silently taking the default.

| file | keys |
|---|---|
| `toolchain.json` | `cuda_home` (null takes torch's `CUDA_HOME`), `ncu` (Nsight Compute 2025.3+), `sccache`, `sccache_enabled`, `mold` |
| `host.json` | `nvcc_threads`, `build_jobs`, `budget_slots`, `lock_dir`, `gpu_lock`, `gpu_shared_slots`, `nice`, `locks_trace`, `scratch`, `tools_dir`, `windows_system32`, `wsl_lib` |
| `clock.json` | `ghz`, `lock`, `unlock` (null `ghz`: the host measures unlocked) |
| `store.json` | `root` (the measurements store: a rola-results checkout) |
| `workspace.json` | `suite` (the rola-bench checkout whose `rola_bench/suite` measures this tree), `worktrees` (the one folder worktrees live in), `base_venv` (the venv pointer venvs borrow from) |
| `environment.json` | `image_digest` (inside the dev container), `fla_crosscheck` |

Each key's type, default and meaning are declared in `SCHEMA`. `python tools/dev.py show` prints every
resolved value and the file it came from ([dev.md](dev.md)).

**The API.**
- `get("section.key")` returns a value.
- `cuda_home()` and `cuda_bin(tool)` resolve the toolkit. Every compile, disassembly and assembler check uses
  them, so the ptxas a gate reads is the one the build ran.
- `scratch(name)` returns a tool's scratch directory.
- `windows_tool(rel)` returns a Windows tool under WSL.
- `detect()` returns what `init` writes.
- `load.cache_clear()` rereads the files, for a test that points `ROLA_DEV_CONFIG` at a temporary directory.
  The `dev_config_env` fixture in `tests/conftest.py` does this.

**Environment variables that remain.** Two lists in the loader name them, and nothing else may read one.
`tools/lint/lint_standards.py` enforces this: it checks Python reads through the syntax tree (constants and
f-string prefixes resolved) and shell expansions.
- `HANDOFFS`: values one of the tree's own processes passes to a child it starts.
  - `ROLA_DEV_CONFIG` itself.
  - The lock-held markers (`ROLA_HOST_BUDGET_HELD`, `ROLA_GPU_LOCK_HELD`, `ROLA_FILE_LOCK_HELD_*`).
  - The gated `nvcc` and `sccache` that `setup.py` passes to the compiler wrapper.
  - The inputs `setup.py` sets for torch (`CUDA_HOME`, `MAX_JOBS`, `PYTORCH_NVCC`).
  - pytest-xdist's worker id.
  - The agent harness's campaign directory.
- `BUILD_PARAMETERS`: what one build is, set per invocation, because pip passes nothing else to `setup.py`:
  `ROLA_CUDA_ARCHS`, `ROLA_CARRY_ARMS`, `ROLA_DECODE_ARMS`, `ROLA_CARRY_PARTS`, `ROLA_BUILD_PARTS`,
  `ROLA_NO_EXTENSION`, `ROLA_STRICT_MANIFEST`, `ROLA_SKIP_POST_BUILD_RATIFY`.

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
