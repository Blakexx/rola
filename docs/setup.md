# Developer setup — the dev container

This page is the whole story for a DEVELOPER's machine: someone who is going to
compile the kernels, run the battery, and use the reading/searching/profiling
tools the docs and skills use. It is not for a USER of the package — a user
installs the prebuilt wheel (`README.md`), takes the ratified manifest inside it
on faith, and never runs anything in this file.

The closed-world build (`docs/build.md`) is gated on an EXACT `ptxas` string and an exact set of the surrounding
tools the docs and skills name (`clangd`, `ast-grep`, `hyperfine`, ...). Reproducing that by hand on a new machine is
exactly the kind of silent drift the rest of this repository refuses, so every one of them is a pin the graph or init
installs, on a host and in the devbox container alike (`//:image`, `.devcontainer/`): an isolated, portable machine for
a box with no host of its own to set up, which runs the host's same loop.

## 0. The dev environment: `tools/dev.py`

Everything this tree needs from a machine -- the CUDA toolkit, Nsight Compute,
the host's compile slots, the lock files, the SM
clock, the measurements store and the worktrees folder -- is one
directory of JSON files, the dev config (`~/.config/rola/`,
`docs/internals/tools/dev_config.md`, KERNEL_STANDARDS §23). One command sets a
machine up and checks it (`docs/internals/tools/dev.md`):

Every setup command is a `bazel run` target but one: `python3 tools/bootstrap.py`, by the machine's own python3,
installs the pinned bazelisk and links `bazel` to it, since nothing can `bazel run` before `bazel` exists. The bootstrap and the
graph's front door import the dev config's reader (`rola_devtools.config`, which also holds the machine's locks) from
the rola-devtools checkout beside this tree, where `workspace.devtools` points by default: clone it there first. The
targets run on the graph environment's interpreter, which carries rola-devtools from its module (`MODULE.bazel`'s pin, or
the checkout the front door overrides it with). A
fresh machine runs on the reader's defaults until `init` writes its config.

```bash
python3 tools/bootstrap.py              # the one step outside the graph: the pinned bazelisk, `bazel` linked to it
bazel run //:init                       # detect, install pinned tools, wire the commit gate and agent router, check
bazel run //:clock -- --mhz 1665        # once per host: lock the SM clock (one administrator prompt under WSL)
bazel run //:check                      # every dependency, one row each; exit 1 on a red
bazel run //:dev_env                    # this checkout's venv, which its commit gate runs from
bazel run //:worktree -- <name> <sha>   # a worktree, its venv and gate, its extension
```

## 1. Host prerequisites

* **An NVIDIA driver for the toolchain's CUDA major.** The declared toolchain is `cu13` (CUDA 13.0,
  `tools/toolchains/cu13.json`), so any driver >= 580 works. This repo's own dev box runs driver `595.95` against an
  RTX 3080 Ti (`sm_86`).
* **Docker**, with the NVIDIA container runtime wired in:
  * **Linux (native):** install `nvidia-container-toolkit`
    (https://github.com/NVIDIA/nvidia-container-toolkit), then
    `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
  * **WSL2 (this box's platform):** install the Windows NVIDIA driver (which
    already carries WSL2 CUDA support — do **not** install a Linux NVIDIA
    driver inside WSL2 itself); install Docker Desktop with the WSL2 backend
    enabled, or Docker Engine directly inside the WSL2 distro; then
    `nvidia-container-toolkit` exactly as the native-Linux step above, run
    from inside the WSL2 distro. Verify with:
    ```bash
    docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
    ```
    A working WSL2 GPU pass-through prints the host's GPU table from inside
    the container. Run it BEFORE anything else on a fresh machine;
    `bazel run //:container_check` (step 3) is the whole proof.
* **git**, to clone the repo and (for a developer) create worktrees.

## <a id="the-devbox-image"></a>2. The devbox image

The devbox image (`//:image`) is a graph target and a thin wrapper around the bootstrap: what a host supplies itself
before the graph, and nothing the graph or the bootstrap installs. rules_oci writes it as files in the standard image
format, so Docker only runs it; `bazel run //:image_load` hands it to Docker as `rola-devbox:latest`. Its base is
`buildpack-deps:jammy` by digest (MODULE.bazel's `devbox_base`: a Docker Official Image, Ubuntu 22.04 with git, gcc,
python3 and curl); on it, the OS packages the base lacks (vim, less, jq, fd, ctags, graphviz, sqlite3, clang-query),
pinned: rules_distroless resolves them against Ubuntu's archive snapshot of the base image's day, its indexes' digests
recorded in MODULE.bazel.lock and every package fetched by the digest they list; then the packages' commands under the
names their alternatives give them, and `~/.local/bin` first on PATH, where the bootstrap links `bazel` as on a host.
A package layer carries its whole closure, the base's core packages among them, so `bazel test //:devbox_packages`
holds that every package the base already has is at the base's version (2026-11-08: 182 packages, 130 of them the
base's, none at another version). The base merges /usr (its top-level links into it, read from the base itself:
`/bin`, `/lib`, `/lib32`, `/lib64`, `/libx32`, `/sbin`), and a layer whose directories sit there would replace the
links, so the packages' files go under /usr as dpkg writes them through the links (`tools/usrmerge_layer.py`), and the
check refuses a layer with anything left there. 1.4 GB.

Its identity is its content digest, which the tree names in `tools/devbox_image.json`. The commit gate builds the
image and refuses a tree whose image is another, and a change to the name without a passing container check for the
new digest (KERNEL_STANDARDS §23 (3)). A commit that leaves the image alone needs no Docker, which a rented box (itself
a container) does not have.

**Published:** after a green check that renames it, `bazel run //:image_push` publishes the image to Docker Hub as
`blakeresearch/rola-devbox:<the digest's first 12 digits>`, public like the namespace's other images. A rented box
starts from `blakeresearch/rola-devbox@<the digest tools/devbox_image.json names>` and runs the loop inside (§3).

## 3. Prove it, then open it

```bash
bazel run //:container_check
```

is the clean room: the host's own loop, step by step, in the image on fresh volumes of its own, each step timed. The
container clones the checkouts from their git directories, mounted read-only: rola-devtools, rola-results and
rola-bench at their commits, and this checkout at its commit with its staged change applied, so its tree is the commit
as git will write it. Inside it then runs `python3 tools/bootstrap.py`, `bazel run //:init`, `bazel run //:dev_env`,
`bazel run //:check` (init's own check runs before the checkout's venv exists, so its gate row is red by design; this
one is the verdict) and the commit gate; then the GPU and a call through torch's flash backend, a lock the HOST holds
seen as held inside (so one lock governs host and container, KERNEL_STANDARDS §14 addendum), and an iteration build of
the library for the device's architecture, imported. The result, each step's time and the cold start (the clone, the
bootstrap, init and the venv: what a fresh machine pays before it can work) are stored through `rola_results` at
`environment` under the image's digest, and a green check writes the digest to `tools/devbox_image.json`. A green
run's volumes go with it; a red run's stay for a look until the next run.

**What a container shares with the host:** the host's `host.lock_dir` at `/run/rola/locks` and its GPU lock's
directory at `/run/rola/gpu` (`dev_config.CONTAINER_LOCKS`, which the container's init detects and records at every
run, so the lock directory follows the mount: a container started without it uses its own locks, and `check` is red
where the config and the mount disagree), and under
WSL the directory above `host.wsl_lib` (`/usr/lib/wsl`) read-only: the image's loader path starts at
`/usr/lib/wsl/lib`, and without the Windows driver's libraries there CUDA fails to initialize inside the container with
error 500 even though `nvidia-smi` works (measured 2026-09-12, Docker Desktop 20.10.17, driver 595.95). The whole
directory, not `lib/` alone: that `libcuda` is a loader that opens the driver store beside it. Nothing else: the
container's checkouts, tools, caches and environment are its own, a second copy of the downloads on Docker's disk.
The launcher also mounts the image's digest (at `dev_config.CONTAINER_IMAGE`, read by `dev_config.devbox_image()`),
which ratify records as where a manifest was measured.

**VS Code / Claude Code:** "Reopen in Container". `.devcontainer/devcontainer.json` runs `bazel run
//:container_compose` as its `initializeCommand`, which hands the image to Docker and writes
`.devcontainer/compose.local.yaml` (never committed): the image's tag, the devbox's volumes and the mounts above. The
`rola` service of `.devcontainer/compose.yaml` then runs on the devbox's own volumes, once a machine: `rola-devbox-home`
at `/root` (the pinned tools, Bazel's caches, the environment store) and `rola-devbox-work` at `/workspace` (the
checkouts). Inside, clone the checkouts into `/workspace` (rola beside rola-devtools, rola-results and rola-bench, where
the reader's defaults look), then run the host's loop in `/workspace/rola`.

**Without an editor:** `bazel run //:container_compose`, then
`docker compose -f .devcontainer/compose.yaml -f .devcontainer/compose.local.yaml run --rm rola bash`.

Inside, the SM clock is not locked: the lock is a host act (`bazel run //:clock`), and rows measured in the container
record `clock_locked = false`.

## 3b. The linker is the graph's

`mold` is THE extension's linker unconditionally (Blake, 2026-08-28), no fallback to `ld.bfd`/`ld.gold`. The graph
fetches the pinned release itself (`@mold`, by `tools/mold_pin.json`'s digest) and links with it, so the container and
the host link with the byte-identical binary and no machine installs one.

## 3c. The checkout's venv: many worktrees on one environment

Each checkout needs its OWN `rola` (its own library, matching its own commit), but installing the whole
`requirements.lock` for every worktree is minutes of identical bytes. The graph builds the environment once into the
shared store, keyed by the lock (docs/build-graph.md#environment), and `bazel run //:dev_env` writes each checkout a
venv over it, `venv-<checkout>` beside the checkout ([`internals/tools/dev.md`](internals/tools/dev.md)). It holds the
invariants that make it safe:

- **It is the environment's interpreter**: the venv is made on the store's Python, so the packages it names are the
  ones that interpreter's ABI built.
- **The `.pth` order puts the checkout first**: `site.py` reads `.pth` files in filename order, so the checkout's
  `rola` and the co-developed sibling checkouts resolve before the environment's copy of anything with the same name.
- **Nothing is installed into it**: its packages are the environment's, and its bin carries the environment's
  commands, so an activated venv has `ruff`, `pytest` and the rest; the checkout's library is the one
  `bazel run //:library` copies into it.

`bazel run //:worktree -- <name> <sha>` adds the worktree and runs both targets in it. It verifies from inside the
new worktree: `python -c` puts its own working directory first on `sys.path`, so a check launched from another worktree
imports that worktree's `rola`.

## 4. First build of the SHIPPED set

```bash
bazel run //:library      # every ratified architecture, every arm: the shipped build, into rola_cu13/
bazel test //:ratified    # ratify's verdict on it: the build's SASS bodies and spills against the manifests
```

The library's gates run on every build (`docs/build-graph.md#gates`): the toolchain the record's (rule 1), every
architecture ratified (rule 3), the vendored CUTLASS its pin's (rule 5), the shards their generator's, the SASS gate
and the arm gate. Ratify is a target of its own (`docs/build-graph.md#ratify`), since the manifests are stale between
milestones by design; it reads the build's own compiles, so it needs the toolchain and no GPU.

## 5. Run the battery

```bash
bazel test //:lane     # every test tier: cpu, unit_cuda, oracle, integration
```

This is the battery `docs/testing.md` "Running it" names; the session-scoped
`gpu_lock()` fixture in `tests/conftest.py` serializes GPU-touching runs the
way it does on the bare host — the container adds no scheduling of its own. Expect
~3.6 min total on hardware comparable to this repo's dev box (`docs/testing.md`,
measured 2026-08-06).

## 6. Measure this checkout

```bash
bazel run //:measure -- run declare.py:all --only 'rola/phases' --only 'rola/sass'
```

rola-devtools' build system runs this checkout's declarations (`declare.py`): the build graph's library
(`bazel run //:library`), SASS, and a timing
session of the carry arms on the cell under the GPU and clock locks, each result stored. A timing entry this binary
cannot run is recorded as that entry's failure in the session's output, not skipped — which is the check this step
stands in for: that the graph's build in the container carries the arms a session measures.

## <a id="the-devbox"></a>What's in the image, and why: the devbox

The devbox has every tool a developer needs, each pinned, and every tool outside the graph pinned into the image or the
dev setup. The devbox's own tools (`tools/dev.py`'s `DEV_TOOLS`) are one pin each, which init installs into its own
versioned folder under `host.tools_dir`, one copy a machine (in the container, on its home volume); each checkout's venv
links them (`bazel run //:dev_env`), so an activated venv has them first on PATH and the system's own tools are
untouched. The environment's developer commands come with
`requirements.lock`, on the same venv's bin. The tree's own code reaches a pinned tool by its pin or recorded path; a
check finds the environment's commands (ast-grep, ruff) on the PATH the graph or the venv sets, which is the
environment's own. THE GAP: the rows sourced from the OS's packages are pinned in the image (its package layer) but
on a host they are the host OS's own, which init does not install; `tools/build/dag_to_dot.py`'s optional `--svg` runs
that graphviz's `dot` from PATH.

| Tool | Source | Role |
|---|---|---|
| the CUDA toolkit: `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm`, `cu++filt`, `compute-sanitizer` | `tools/cuda_toolkit_pin.json` (NVIDIA's redistributable components at the manifest the graph's toolkit is read by, by its digest), init, at `dev_config.cuda_home()` and on no PATH (torch takes its toolkit from the nvcc on PATH without following links) | the ratified toolchain (`docs/build.md` rule 1, `cu13`: CUDA 13.0, V13.0.48) for every tool outside the graph's own rules; the graph's copy of the same pin (`@cuda_tools`) is the test tiers' and the binary gates' |
| `mold` 2.42.0 | `tools/mold_pin.json`, the graph's `@mold` | the extension's only linker |
| Nsight Compute 2025.3, Nsight Systems 2025.3 | `tools/nsight_compute_pin.json`, `tools/nsight_systems_pin.json` (NVIDIA's redistributable components at the toolkit's manifest, by digest), init | PM sampling (pipe timeline) and the measurement tools' Nsight Compute, found by its pin; timelines |
| `hyperfine`, `difftastic` (`difft`) | `tools/hyperfine_pin.json`, `tools/difftastic_pin.json` (pinned GitHub releases, sha256-verified), init | ad-hoc wall-clock comparisons outside the declared timing sessions; structural diffs |
| `ast-grep`, `py-spy`, `clangd` 22, `clang-format` 19.1.7 | `requirements.lock` (the environment) | structural code search; live Python profiling; the editor's language server, at the pinned clang-tidy's LLVM major; the formatter `.clang-format` names |
| `graphviz`, `sqlite3` | the image's package layer; a host's own | dependency/DAG rendering, the store's record index inspection |
| `universal-ctags` | the image's package layer; a host's own | cross-referencing the `csrc/` tree |
| `jq`, `fd` (via `fd-find`) | the image's package layer; a host's own | the shell one-liners this repo's own docs use |
| `clang-tools-14` (clang-query) | the image's package layer; a host's own | AST queries the skills use; clang-tidy is the graph's pinned one (`@clang_tidy`), run by the unit lints |
| Python 3.11.13, `requirements.lock` | the graph's environment (`@rola_env`, the lock generated by `uv pip compile`, see `requirements.in`'s header), each checkout's venv over it (`bazel run //:dev_env`) | the tree's one Python environment, the tests' `pytest-timeout` and `hypothesis` among it |

## Reproducibility as a statement

`tools/ratify.py`'s written manifests carry `source.image_digest`: the devbox image's digest, which the
container's launcher mounts (step 3), or the empty string if a ratification ran on a bare host. It sits beside
`source.csrc_sha256`/`source.date`, not inside `toolchain` — it identifies
WHERE a measurement ran, not WHAT was measured, so it is recorded and reported
on drift, never part of `manifest_digest()`'s
ratchet-gated hash. Re-ratifying inside this container and comparing
`source.image_digest` across two runs is how "reproduced under the same
pinned environment" becomes a value you can diff instead of a claim you take
on faith.
