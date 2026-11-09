# `tools/dev.py` — the dev environment, set up and checked by one command

KERNEL_STANDARDS §23. One command sets up a machine for this tree (a host, or the devbox container), checks it and
describes it. It
replaced tools/build/new_worktree.sh, tools/setup_clock_lock.py and tools/build/build_lock.sh. The
settings it writes and reads are the dev config ([dev_config.md](dev_config.md)).

    python3 tools/bootstrap.py               # the pinned bazelisk and `bazel` (install_bazelisk), before any target
    bazel run //:init [-- --force]
    bazel run //:check
    bazel run //:show
    bazel run //:get -- <section.key>
    bazel run //:clock -- --mhz 1665
    bazel run //:dev_env                     # dev-env, given the checkout and the graph's environment
    bazel run //:worktree -- <name> <sha>
    bazel run //:container_check             # the devbox image's clean room
    bazel run //:container_compose           # the devbox image for an editor's dev container

Each target is this file's command of the same name (`bazel/defs.bzl`'s `dev_run`): run in the checkout on the
graph's environment interpreter, its arguments after `--`.

**`init`** writes this machine's config:
- The pinned CUDA toolkit (`TOOLKIT`, `tools/cuda_toolkit_pin.json`), where it is missing or its nvcc does not report
  its pin's version, which every compile, disassembly and assembler check outside the graph's own rules uses
  (`dev_config.cuda_home()`), on no PATH.
- The pinned devbox tools (`DEV_TOOLS`: hyperfine, difftastic, Nsight Compute and Systems), each where it is missing or
  does not report its pin's version; the measurement tools find the pinned Nsight Compute by its pin
  (`dev_config.ncu()`).
- The Windows System32 mount under WSL.
- In a container, the lock directories its launcher mounted (`dev_config.locks_mounted()`, from the process's own
  mount table), rewritten at every run and removed where the mount is gone (`_follow_lock_mount`), so a container
  never keeps its own locks beside the host's; `check`'s row is red where the config and the mount disagree.

A key already in a file is kept unless `--force`. A pinned tool that is missing, or does not report its pin's
version, is downloaded from the pin's release URL (the devbox tools' pins by init, `tools/bazelisk_pin.json` by
`install_bazelisk`, the bootstrap's), asked again after a server error or a dropped connection, and
then from its `mirror_urls` (the same asset through GitHub's API: github.com's release downloads returned gateway
timeouts from its own edge for hours on 2026-11-08 while the API served the same assets). Its release sha256 is
verified, and so is its binary
sha256 where the pin has one. A toolkit pin names NVIDIA's redistributable manifest by its digest instead, and the
components it takes: init lays them out as the graph's own copy does (`@cuda_tools`), file for file, so a tool outside
the graph and a test tier run the same toolkit. The whole release -- a tarball, a bare binary under its release's name, or the
toolkit -- lands under `host.tools_dir`, because the layout
is part of the tool: the toolkit's nvcc finds cicc and ptxas by its own place. A pinned tool is found by its pin
(`dev_config.pinned`: under `host.tools_dir`, in its pin's folder, by its release's name), never recorded in the
machine's files -- every checkout loads them, and an older loader refuses a key it does not declare. `install_bazelisk` links `toolchain.bazel` (`~/.local/bin/bazel`) to the pinned bazelisk -- a link it made to another pinned release is replaced, anything else at the path refused. `init` wires this checkout's commit gate (the git-config keys `core.hooksPath` per worktree, whose pre-commit hook runs `bazel test //:gate`, and rola.venv naming the checkout's own `bazel run //:dev_env` venv, which the post-commit review runs from; a checkout with no venv of its own is named and left as it is) in every worktree of the repository that carries the tracked hook, turns the repository's shared hook into a refusal (a checkout reaches it only unwired), installs the agent skill router where Claude Code keeps a user configuration (KERNEL_STANDARDS §31: the repo's `.claude/hooks/skill_router_bootstrap.py` copied to the user's `~/.claude/hooks/skill_router.py` where it differs, and its PreToolUse entry added to the user's `~/.claude/settings.json` where absent, every other setting kept; the image has no `~/.claude` and gets neither), and runs `check`, whose row says whether the router is installed and current. The
clock is not set by `init`, because locking it asks for administrator consent once.

**`check`** prints one row per dependency and exits 1 on any red:
- The config parses.
- The pinned toolkit's nvcc reports its pin's version.
- Each devbox tool reports its pin's version (Nsight Compute among them).
- The commit gate is wired.
- The lock directory, scratch directory and GPU lock directory are writable.
- The clock is set, or the host is marked unlocked.
- The worktrees folder exists; `store.root` is a rola-results checkout, and `rola_results` and `rola_devtools` import
  from `store.root` and `workspace.devtools` in the checkout's own venv.

**`clock --mhz N`** locks the SM clock, reads the driver's clock under the lock, and releases it: the round trip
the harness makes (`rola_devtools.locks.clock`, [sm_clock.md](../common/sm_clock.md)). Only after the lock is proven
does it write `clock.json`.
- **Under WSL:** it registers two elevated scheduled tasks, `gpu-lock` and `gpu-unlock`. Each runs a hidden
  `wscript` launcher that calls `nvidia-smi.exe` and exits with its code, and registering them asks for administrator
  consent once. The lock and unlock commands start the task and return when it has finished, failing with its result:
  `schtasks /run` returns before the task runs, so a proof read raced the lock, and a driver's refusal read as success.
- **On Linux:** it writes `sudo -n nvidia-smi -lgc` commands and prints the sudoers rule they need.

**`dev-env`** is `bazel run //:dev_env`'s implementation, which passes the checkout and the graph's environment
(its store venv, interpreter and uv). It writes the checkout's venv, `venv-<checkout>` beside it, on the environment's
interpreter. Its site holds five `.pth` files, read in name order: the checkout's root (`aa_rola_<name>.pth`), the
co-developed sibling checkouts (`rola_devtools.pth`, `rola_results.pth`, `LINKED`), the environment's site-packages
(`zz_rola_env.pth`, the venv's marker) and the bytecode prefix (`zz_rola_pycache.pth`: `sys.pycache_prefix` under
`host.build_cache`, so no `__pycache__` lands in the checkout). An activated venv is a developer's shell, so its bin
carries every command the devbox has: the environment's (ruff, pytest, ast-grep, clangd, py-spy, ...), run by the
venv's interpreter so they import what its site names, and the devbox's pinned tools (hyperfine, difft, ncu, nsys),
linked from their folders under `host.tools_dir`, one install a machine whichever checkout links it. It refuses
until init has installed them. It replaces only a venv it wrote, wires the checkout's commit gate to run from it, and refuses unless `rola`, `rola_devtools`, `rola_results` and `torch` each import from
where those files say.

**`worktree <name> <sha>`** creates the worktree in `workspace.worktrees`, then runs `bazel run //:dev_env` and
`bazel run //:library` in it: its venv and commit gate, then its extension, which the graph serves from its shared
cache when any checkout built that commit's library. It imports `rola` and the extension from inside the worktree.

**`container`** keeps the devbox image a supported environment (§23 (3), `docs/setup.md` §2-3). Each action takes the
image's OCI layout (`//:image`), whose manifest digest is the image's identity, and `check` and `compose` the image as
Docker loads it (`//:image_tarball`), tagged `rola-devbox:<digest[:12]>`.
- `check` (`bazel run //:container_check`) is the clean room: `CONTAINER_LOOP` run step by step by `docker exec` in a
  container on fresh volumes of its own, which clones the checkouts from their git directories mounted read-only
  (`CLONED`, this checkout with its staged change), each step timed; `COLD_START` names the steps a fresh machine pays
  before it can work. It stores the result through `rola_results` at `environment`, one record per image digest, and
  a green check writes the digest to `tools/devbox_image.json` and removes its volumes.
- `compose` (`bazel run //:container_compose`) writes `.devcontainer/compose.local.yaml`: the image's tag, the
  devbox's own volumes and the host's mounts (`_mounts`: the lock directories at `dev_config.CONTAINER_LOCKS`, the WSL
  driver's libraries, and the image's digest, read inside by `dev_config.devbox_image()`).
- `gate` is the commit gate's check: the image the graph builds is the one `tools/devbox_image.json` names, and a
  commit changing the name has a passing record for the new digest. A commit that leaves the image alone needs no
  Docker.

**The measurements store** (`store.root`, a rola-results checkout) is importable from every interpreter that runs
here: `init` links `rola_results` into each checkout's own venv (it names the environment's site-packages as a plain
path, which reads none of its `.pth` files), by a one-line `rola_results.pth`, and `dev-env` links each venv it
writes. The store's own command line is `bazel run //:results`
(`commit -m` commits the records).

`init`, `check`, `show`, `get`, `clock`, `dev-env` and `worktree` run from whichever python starts them (`dev-env`'s is the environment's, through its run target); `container` re-executes itself under the checkout's own venv (`bazel run //:dev_env`), because the container reads `rola_results`, which it links; the host's own python carries nothing of this project. The sibling repositories gate through their own graphs (`bazel test //:gate`, rola-devtools' commit gate), from no venv of this tree's.

**`get`** prints one value, for a person or a shell (the `rola-dev` skill's way to read one setting).

**Running a CPU-heavy command under the host budget** is `python -m rola_devtools.locks.host [--slots N]
[--exclusive] -- <cmd>`, niced per `host.nice`. The graph's front door takes it for every build.
