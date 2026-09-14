# `tools/dev.py` — the dev environment, set up and checked by one command

KERNEL_STANDARDS §23. One command sets up a host or a container for this tree, checks it and describes it. It
replaced `tools/build/new_worktree.sh`, `tools/setup_clock_lock.py` and `tools/build/build_lock.sh`. The
settings it writes and reads are the dev config ([dev_config.md](dev_config.md)).

    python tools/dev.py init [--force]
    python tools/dev.py check
    python tools/dev.py show
    python tools/dev.py get <section.key> | cuda-home
    python tools/dev.py clock --mhz 1665
    python tools/dev.py worktree <name> <sha> [--build]
    python tools/dev.py container compose|build|check|gate
    python tools/dev.py store commit -m <msg>

**`init`** writes this machine's config:
- The CUDA toolkit on `PATH`, or torch's.
- The newest Nsight Compute CLI under `/usr/local/cuda-*`.
- `sccache` and `mold` where they are installed.
- The Windows System32 mount under WSL.
- The base venv of the running interpreter: a pointer venv's `zz_rola_base.pth` target, or the venv itself.

A key already in a file is kept unless `--force`. A pinned tool that is missing, or does not report its pin's
version, is downloaded from the pin's release URL (`tools/sccache_pin.json`, `tools/mold_pin.json`). Its
tarball sha256 is verified, and so is its binary sha256 where the pin has one. The whole release is unpacked
under `host.tools_dir`, because the layout is part of the tool: `-fuse-ld=mold` selects the `ld.mold` next to
`mold`. `init` then wires this checkout's commit gate (`core.hooksPath` per worktree) and runs `check`. The
clock is not set by `init`, because locking it asks for administrator consent once.

**`check`** prints one row per dependency and exits 1 on any red:
- The config parses.
- nvcc, ptxas, nvdisasm and cuobjdump exist in the toolkit.
- `ncu` is 2025.3 or later.
- sccache and mold pass the build's own gates (`sccache_toolchain.gate`, `mold_toolchain.link_flags`), so
  `check` refuses exactly what a build would.
- The commit gate is wired.
- The lock directory, scratch directory and GPU lock directory are writable.
- The clock is set, or the host is marked unlocked.
- The base venv and the worktrees folder exist; `store.root` is a rola-results checkout, and `rola_results` imports
  from it in the base venv.

**`clock --mhz N`** locks the SM clock, reads the driver's clock under the lock, and releases it: the round trip
the harness makes (`tools/clock_lock.py`, [sm_clock.md](../common/sm_clock.md)). Only after the lock is proven
does it write `clock.json`.
- **Under WSL:** it registers two elevated scheduled tasks, `gpu-lock` and `gpu-unlock`. Each runs a hidden
  `wscript` launcher that calls `nvidia-smi.exe`, and registering them asks for administrator consent once.
- **On Linux:** it writes `sudo -n nvidia-smi -lgc` commands and prints the sudoers rule they need.

**`worktree <name> <sha>`** creates the worktree in `workspace.worktrees` and wires its commit gate before
anything is built. It makes a seeded pointer venv beside it with two `.pth` files: this worktree's root
(`aa_`), then the base venv's site-packages (`zz_`). Then it installs the package editable with `--no-deps`.
The extension is copied from the main checkout when that checkout is at the same commit and built; `--build`
builds it anyway. It refuses unless the editable finder maps `rola` to the new worktree, and it imports `rola`
from inside the worktree. The three footguns this avoids are in `docs/setup.md` §3c.

**The base venv** is part of the machine: `check` compares `workspace.base_venv` against `requirements.lock`, where a
pin the venv lacks or carries at another version is drift and extras the lock does not name are not. `init` installs
the missing pins with `uv pip install -r requirements.lock`, adding without removing. flash-attn is pinned there by
release-wheel URL and sha256.

**`container`** keeps the dev image a supported environment (§23 (3), `docs/setup.md` §2-3). Its inputs are
`ENV_INPUTS`: the Dockerfile, `.devcontainer/devcontainer.json` and `compose.yaml`, `requirements.lock`, the tool pins,
and this file with the modules it imports. Their sha256, path and bytes each, is the ENVIRONMENT KEY.
- `build` copies only those inputs into a context and builds `rola-dev:<key[:12]>`. The image's last step is
  `init --image`, which writes the fixed layout `IMAGE_LAYOUT`, installs the pins and runs `check --image`. A red row
  fails the build.
- `compose` writes `.devcontainer/compose.local.yaml`, binding the host's checkout, lock directories, store and
  worktrees onto that layout, and refuses without an image for the key.
- `check` runs `CONTAINER_PROOFS` inside the image with those mounts: `dev.py check`, the GPU with torch and
  flash-attn, a lock held by a host process seen as held, and an iteration build of a copy of the checkout. It
  stores the result through `rola_results` at `environment`, one record per key, every check a sample.
- `gate` is the commit hook. When the staged files touch an input, it computes the key of the staged blobs and
  refuses the commit without a passing sample for that key.

**`store`** keeps the measurements store (`store.root`, a rola-results checkout): `init` links `rola_results` into
every interpreter that measures here -- the base venv, each worktree's pointer venv (it borrows the base as a plain
path, which reads none of the base's `.pth` files), python3's user site, and in the image the base venv at the store's
mount point -- by a one-line `rola_results.pth`, and `worktree` links each new venv; `commit -m` commits the records (`python -m
rola_results commit`).

**`get`** prints one value for a shell caller. `cuda-home` resolves the toolkit, which the clang-tidy device
scripts pass as `--cuda-path`.

**Running a CPU-heavy command under the host budget** is `python tools/host_budget.py [--slots N]
[--exclusive] -- <cmd>`, niced per `host.nice`. `setup.py` takes the budget itself.
