# Developer setup — the dev container

This page is the whole story for a DEVELOPER's machine: someone who is going to
compile the kernels, run the battery, and use the reading/searching/profiling
tools the queue's briefs assume. It is not for a USER of the package — a user
installs the prebuilt wheel (`README.md`), takes the ratified manifest inside it
on faith, and never runs anything in this file.

The container (`Dockerfile`, `.devcontainer/devcontainer.json`) exists because
the closed-world build (`docs/build.md`) is gated on an EXACT `ptxas` string, an
exact `sccache`, and now an exact set of the surrounding tools every brief
names by name (`clangd`, `ast-grep`, `hyperfine`, ...). Reproducing all of that
by hand on a new machine is exactly the kind of silent drift the rest of this
repository refuses; the container makes it one build instead of a checklist.

## 0. The dev environment: `tools/dev.py`

Everything this tree needs from a machine -- the CUDA toolkit, Nsight Compute,
the pinned `sccache` and `mold`, the host's compile slots, the lock files, the SM
clock, the measurements store, the worktrees folder and base venv -- is one
directory of JSON files, the dev config (`~/.config/rola/`,
`docs/internals/tools/dev_config.md`, KERNEL_STANDARDS §23). One command sets a
machine up and checks it (`docs/internals/tools/dev.md`):

`tools/dev.py` reads the dev config through rola-devtools (`rola_devtools.config`, which also holds the machine's
locks), so a new machine's first `init` runs with a rola-devtools checkout on the path: clone it beside this tree, where
`workspace.devtools` points by default, and run `PYTHONPATH=../rola-devtools python tools/dev.py init`. Init links
the checkout into every venv here, so nothing after it needs the path.

```bash
python tools/dev.py init          # detect, install the pinned tools, wire the commit gate, check
python tools/dev.py clock --mhz 1665   # once per host: lock the SM clock (one administrator prompt under WSL)
python tools/dev.py check         # every dependency, one row each; exit 1 on a red
python tools/dev.py worktree <name> <sha>   # a worktree, its pointer venv, its gate, its extension
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
    `python tools/dev.py container check` (step 3) is the whole proof.
* **git**, to clone the repo and (for a developer) create worktrees.

## 2. Build the image

```bash
python tools/dev.py container build
```

The image is built from the environment's inputs alone -- `Dockerfile`, `.devcontainer/`, `requirements.lock`, the
tool pins and the `tools/dev.py` files (`ENV_INPUTS`), copied into a throwaway context, never the whole tree -- and
tagged `rola-dev:<first 12 of the environment key>`. The key is sha256 over those inputs; the image records it as
`environment.image_digest`, which `tools/ratify.py` stamps as `source.image_digest` and every record row carries.

Its last step is `tools/dev.py init --image`: the same one init a bare host runs, writing the image's dev config
(`ROLA_DEV_CONFIG=/opt/rola/dev-config`) for its fixed layout, installing the pinned `sccache` and `mold` from the
repo's pins, and running `check --image`. A red check fails the image build -- the image built on 2026-08-28 went two
weeks without `cuobjdump`, the Nsight Compute the pipe timeline needs, or flash-attn, and nothing said so.

The image is built for a toolchain record (`tools/toolchains/<name>.json`; `--toolchain NAME` when more than one is
declared). `container build` passes the record's base image, its exact apt package pins and its torch index as build
arguments, and the Dockerfile fails the build unless `ptxas --version` is the record's string byte for byte -- the same
comparison `setup.py`'s rule 1 makes, so a base tag carrying a newer patch (every CUDA 12.4 tag shipped V12.4.131
against a V12.4.99 pin, measured 2026-08-29) cannot slip through. `cuobjdump` is one of the record's pins and
`nsight-compute-2025.3.0` is pinned beside them. The attention reference needs no package: it is torch's own flash
backend (`docs/measurement.md`).

## 3. Prove it, then open it

```bash
python tools/dev.py container check
```

runs the proofs inside the image, with the host's mounts, and stores the result through `rola_results` at
`environment`, under the environment key: `tools/dev.py check` inside; the GPU and a call through torch's flash backend; a lock
the HOST holds is seen as held inside (so one lock governs host and container, KERNEL_STANDARDS §14 addendum); and an
iteration build of a copy of the checkout, imported. **A commit that changes any environment input needs a passing
record for the key of what it stages -- the commit gate refuses it otherwise (KERNEL_STANDARDS §23 (3)).**

**The mounts come from the host's dev config.** `python3 tools/dev.py container compose` writes
`.devcontainer/compose.local.yaml` (never committed) binding, onto the image's fixed layout: this checkout at
`/workspace/rola`, the host's `host.lock_dir` at `/run/rola/locks` and its GPU lock's directory at `/run/rola/gpu`,
`store.root` at `/workspace/store`, `workspace.suite` at `/workspace/suite`, and `workspace.worktrees` at
`/workspace/worktrees`. It refuses when no image
exists for the current key. Under WSL it also mounts the directory above `host.wsl_lib` (`/usr/lib/wsl`) read-only:
the image's loader path starts at `/usr/lib/wsl/lib`, and without the Windows driver's libraries there CUDA fails to
initialize inside the container with error 500 even though `nvidia-smi` works (measured 2026-09-12, Docker Desktop
20.10.17, driver 595.95 -- the 2026-08-28 image had never been run against the GPU). The whole directory, not `lib/`
alone: that `libcuda` is a loader that opens the driver store beside it.

**VS Code / Claude Code:** "Reopen in Container". `.devcontainer/devcontainer.json` runs `container compose` as its
`initializeCommand`, then the `rola` service of `.devcontainer/compose.yaml` with the generated mounts, and runs
`python tools/dev.py check` once created.

**Without an editor:** `docker compose -f .devcontainer/compose.yaml -f .devcontainer/compose.local.yaml run --rm rola bash`.

Inside, the SM clock is not locked: the lock is a host act (`tools/dev.py clock`), and rows measured in the container
record `clock_locked = false`.

The history this replaced, kept because it was measured: the first devcontainer (2026-08-29) mounted worktrees through
a `${localEnv:...}` default nested inside another variable, which `@devcontainers/cli` does not expand, then through a
`-v`-style string the CLI forwards as a malformed `--mount`; it ended as a static empty named volume, so host worktrees
were never visible inside. Generating the mounts from the dev config removes the variable and the question.

## 3b. Building on the bare host (no container) needs `mold`

K39 (Blake ruling, 2026-08-28): `mold` is THE extension's linker unconditionally
-- `setup.py` refuses to build without it, no fallback to `ld.bfd`/`ld.gold`
(`tools/mold_toolchain.py`, `tools/mold_pin.json`). On a bare host with no root
and no apt package for it (this repo's dev box: `apt-cache policy mold` has no
candidate on Ubuntu 22.04), `python tools/dev.py init` downloads the pinned
release, verifies its sha256 against `tools/mold_pin.json`, unpacks the whole
release (`bin/mold` beside `bin/ld.mold`, which `-fuse-ld=mold` selects) under
`host.tools_dir`, and records it as `toolchain.mold`. The container and the host
link with the byte-identical binary.

## 3c. The pointer venv, the fast path (many worktrees on one bare host)

K46 (2026-08-29, "agents stop reverse-engineering it from sibling venvs" --
recurring SKILLS FEEDBACK from K42 and K45). Sections 3b/4 are the FIRST build
of a single checkout. This is the DIFFERENT question a bare host with many
concurrent git worktrees (`worktrees-live-in-one-folder`) actually has: each
worktree needs its OWN `rola` editable install (its own compiled `.so`,
matching its own SHA and its own arm subset), but reinstalling `torch` +
`einops` + `entmax` + the rest of `requirements.lock` from scratch for every
worktree is minutes of wasted work repeated dozens of times for bytes that are
identical across all of them. The POINTER VENV is a small venv per worktree
that BORROWS a shared base venv's already-installed packages and adds only
what is actually per-worktree: this checkout's own root, and this checkout's
own editable-installed extension.

`python tools/dev.py worktree <name> <sha>` does the whole sequence below in one
call (`docs/internals/tools/dev.md`); read it before doing this by hand a second time.

**1. The shared base venv, built ONCE** (recorded as `workspace.base_venv` in the dev config):

```bash
uv venv --python 3.11 --seed /path/to/shared/base-venv
/path/to/shared/base-venv/bin/python -m pip install -r requirements.lock
```

This is a normal, complete venv — nothing pointer-shaped about it yet. It is
the thing every worktree's pointer venv borrows from, so building it well once
(the full `requirements.lock`, the right CUDA-enabled `torch`) is what makes
every later worktree venv cheap.

**2. Per worktree, a new SEEDED venv:**

```bash
uv venv --python 3.11 --seed /path/to/worktrees/venv-<name>
```

**FOOTGUN 1 — a seedless venv silently falls through to a system python.**
Omitting `--seed` leaves no `pip` binary inside the new venv at all. A later
bare `pip install ...` (as opposed to `<venv>/bin/python -m pip install ...`)
then resolves to whatever `pip` happens to be first on `PATH` OUTSIDE the
venv — installing into a completely different python's site-packages with no
error, no warning, and an import that "worked" moments ago now resolving to
the wrong build. This is the same trap family as the device-side-stamp
extension trap (`extension-trap-device-side-check`): nothing about the venv's
existence proves what actually received the install. `--seed` (or always
spelling the install as `python -m pip install`) is not optional.

**3. Two `.pth` files in the new venv's `site-packages`, pointing OUT:**

```bash
SITE=/path/to/worktrees/venv-<name>/lib/python3.11/site-packages
echo /path/to/worktrees/<name>                              > "$SITE/aa_rola_<name>.pth"
echo /path/to/shared/base-venv/lib/python3.11/site-packages  > "$SITE/zz_rola_base.pth"
```

`site.py` processes `.pth` files in the sorted order of their filenames, so the
`aa_`/`zz_` prefixes are load-bearing, not decoration: this worktree's own root
(`aa_...pth`) and its own editable install (`__editable__...pth`, added by step
4 below, which alphabetically sorts after both) resolve BEFORE the shared base
venv's copy of anything with the same name. A worktree's own compiled `rola`
extension is what gets imported, never the base venv's.

**4. The in-tree editable build, `--no-deps`:**

```bash
cd /path/to/worktrees/<name>
/path/to/worktrees/venv-<name>/bin/python -m pip install -e . --no-deps --no-build-isolation
```

**FOOTGUN 2 — `pip install -e .` without `--no-deps` pulls a fresh,
incompatible torch and yields an ABI-mismatched `.so`.** `pyproject.toml`
declares `torch` as a dependency; without `--no-deps`, `pip`/`uv pip` resolves
and installs its OWN copy of `torch` into the pointer venv — not necessarily
the same build (CPU-only, a different CUDA minor version, a different ABI) as
the one the shared base venv already carries and the extension is about to
compile its C++/CUDA ABI against via `torch.utils.cpp_extension`. The result
imports without error and then crashes or misbehaves at the first real call,
because the compiled `.so` and the `torch` actually resolved at import time
were built against two different ABIs. The shared base venv's `torch` (reached
through the `zz_rola_base.pth` file in step 3) IS the dependency resolution
here; `--no-deps` is not optional either. `setup.py` takes the machine-wide
host budget itself (`docs/build.md`, `rola-build` skill) — a pointer-venv build
is still a build, and still shares the machine-wide compile budget.

Verify with `python -c "import rola; from rola_cu13._build_config import
BUILD_CONFIG; print(BUILD_CONFIG['archs'])"` against the new venv's python —
the device-side stamp, never a path check (`extension-trap-device-side-check`)
-- but run it FROM the new worktree's own directory. **FOOTGUN 3, found
writing the worktree tool: `python -c` puts `''` (the
process's OWN current directory) first in `sys.path`.** A verification launched
from inside a DIFFERENT rola worktree (which also carries a `rola/` package)
resolves `import rola` to THAT worktree instead of the new one — `''` beats
every `.pth`-added path because it is `sys.path[0]`, checked before any of
them. `tools/dev.py worktree` runs its verification from inside the new worktree
for exactly this reason; do the same by hand.

## 4. First build of the SHIPPED set

Inside the container (or on the bare host with `mold` vendored per 3b above):

```bash
pip install -e . --no-build-isolation
```

This runs the closed-world gates in order (`docs/build.md`): rule 3 (ratified
archs), rule 1 (toolchain — verified against the SAME manifest string the box
this container was authored on already matches), rule 5 (vendored CUTLASS),
the shard-generation check, the compile itself (`sccache`-cached per
`docs/build.md#sccache` once a cache is warm), the fatbin gate, and the
post-build ratification (`tools/ratify.py`) — which is the gate that answers
"does this container's compile reproduce the committed manifest's SASS
hashes". A clean build reports:

```
rola-standalone $ python tools/ratify.py --arch 80 --arch 86
...
SASS-body gate: all <N> ratified bodies unchanged; PASS
```

`--arch 80` requires an ampere-class-or-later device to actually launch
against, but the RATIFICATION itself (recompiling the shipped TUs and hashing
their SASS) needs only the toolchain, not a matching physical GPU — see
`tools/ratify.py`'s own `--arch` handling.

## 5. Run the battery

```bash
pytest tests/unit tests/oracle -m "not cuda"                       # CPU-only, no device, no build
pytest tests/unit -m cuda                 # tests/unit's CUDA-marked rows
pytest tests/oracle                       # kernels vs the fp64/torch baselines
pytest tests/integration                  # composition, dispatch, two-branches-agree
```

This is the same battery `docs/testing.md` "Running it" names; the session-scoped
`gpu_lock()` fixture in `tests/conftest.py` serializes GPU-touching runs the
way it does on the bare host — the container adds no scheduling of its own. Expect
~3.6 min total on hardware comparable to this repo's dev box (`docs/testing.md`,
measured 2026-08-06).

## 6. Measure this checkout

```bash
python -m rola_devtools.build run declare.py:all --only 'rola/phases' --only 'rola/sass'
```

rola-devtools' build system runs this checkout's declarations (`declare.py`): the gated build, SASS, and a timing
session of the carry arms on the cell under the GPU and clock locks, each result stored. A timing entry this binary
cannot run is recorded as that entry's failure in the session's output, not skipped — which is the check this step
stands in for: that the container's compile produced a binary carrying the arms a session measures.

## What's in the image, and why

| Tool | Source | Role |
|---|---|---|
| `ptxas`/`nvcc` | the toolchain record's apt pins (`cu13`: CUDA 13.0, V13.0.48) | the ratified toolchain (`docs/build.md` rule 1) |
| `cuobjdump`, Nsight Compute 2025.3 | apt, pinned versions | cubin extraction (SASS gate, region ledger, composer); PM sampling (pipe timeline) |
| `sccache` 0.17.0 | `tools/dev.py init --image` from `tools/sccache_pin.json`, verified | the compiler cache `docs/build.md#sccache` measures |
| `ninja` | apt | the build's actual compile driver (`docs/build.md` step 5) |
| `mold` 2.42.0 | `tools/dev.py init --image` from `tools/mold_pin.json`, verified | the extension's only linker |
| `graphviz`, `sqlite3` | apt | dependency/DAG rendering, the store's record index inspection |
| `clangd`, `clang-tools-14` (clang-query) | apt | the editor/AST tooling briefs assume is present |
| `universal-ctags` | apt | cross-referencing the `csrc/` tree |
| `jq`, `fd` (via `fd-find`) | apt | the shell one-liners this repo's own docs use |
| `hyperfine` | pinned GitHub release, sha256-verified | ad-hoc wall-clock comparisons outside the declared timing sessions |
| `ast-grep-cli`, `py-spy` | pipx (real PyPI packages) | structural code search; live Python profiling |
| `difftastic` (`difft`) | pinned GitHub release, sha256-verified | structural diffs — **not** pipx: it has no PyPI package, verified at authoring time; installed the same way as `mold`/`hyperfine` instead of being forced through a tool that cannot actually install it |
| `pre-commit` | pipx | the repo's lint/format hooks |
| Python 3.11, `requirements.lock` | `uv venv` + `uv pip install -r` (lock generated by `uv pip compile`, see `requirements.in`'s header) | replaces the hand-grown dev venv this box iterated in; includes `pytest-timeout` and `hypothesis` per this card |

## Reproducibility as a statement

`tools/ratify.py`'s written manifests carry `source.image_digest`: the image's
`environment.image_digest`, the environment key it was built from (step 2), or the
empty string if a ratification ran on a bare host. It sits beside
`source.csrc_sha256`/`source.date`, not inside `toolchain` — it identifies
WHERE a measurement ran, not WHAT was measured, so (like `sccache`'s version)
it is recorded and reported on drift, never part of `manifest_digest()`'s
ratchet-gated hash. Re-ratifying inside this container and comparing
`source.image_digest` across two runs is how "reproduced under the same
pinned environment" becomes a value you can diff instead of a claim you take
on faith.
