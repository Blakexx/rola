# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# THE DEV CONTAINER (K37). USERS get a wheel (prebuilt per-arch binaries, a driver
# dependency, closed-world -- see README.md). DEVELOPERS get this: a box that can
# compile the kernels under the SAME toolchain the ratified manifests were measured
# under, plus the reading/searching/profiling tools the queue's briefs assume.
#
# THE TOOLCHAIN COMES FROM ITS RECORD (tools/toolchains/<name>.json, docs/internals/tools/toolchains.md).
# `tools/dev.py container build` passes the record's `container.base_image` (CUDA_BASE_IMAGE), its exact apt
# `container.packages` (CUDA_PACKAGES), its `torch_index` (TORCH_INDEX) and its name (ROLA_TOOLCHAIN). A base tag's own
# nvcc is not trusted: the packages are installed at their pinned versions whatever the tag ships (a tag can carry a
# newer patch -- measured for 12.4, whose every tag shipped V12.4.131 against a V12.4.99 pin), and the build fails
# unless `ptxas --version` is the record's string byte for byte, the same comparison setup.py's rule 1 makes.
# ubuntu22.04 for the apt package surface the rest of this file needs (deadsnakes, clangd-14, ...).
#
# sccache and mold are installed by `tools/dev.py init --image` from the repo's own pins (`tools/sccache_pin.json`,
# `tools/mold_pin.json`) -- the same command and the same pins a bare host uses, so the image and the host cannot
# disagree on either. The container-only tools below (uv, hyperfine, difftastic) are pinned the same way: an exact
# version, an exact upstream release asset, and a sha256 verified before the archive is trusted.
#
# THE IMAGE IS KEPT CURRENT MECHANICALLY (KERNEL_STANDARDS §23 (3)): the environment key hashes this file and every
# other input `tools/dev.py` lists in ENV_INPUTS; a commit that changes one carries a passing
# `python tools/dev.py container check` for its key, or the commit gate refuses it.
ARG CUDA_BASE_IMAGE
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIPX_HOME=/opt/pipx \
    PIPX_BIN_DIR=/usr/local/bin

# --------------------------------------------------------------------------
# APT PACKAGES -- everything with a real Ubuntu-22.04 (jammy) package. Left
# out on purpose: `docker.io`/nvidia-container-toolkit (host-side,
# docs/setup.md), `hyperfine`/`mold`/`difftastic` (not in 22.04's repos --
# pinned release binaries below). CUDA/ptxas is a SEPARATE, later RUN block
# (below): the base image's own nvcc/ptxas are not trusted, see the
# FROM-line comment at the top of this file.
#
# PYTHON 3.11 IS NOT IN JAMMY'S DEFAULT REPOS (jammy ships 3.10). The
# deadsnakes PPA is the standard, widely-used source for it; added here
# rather than building CPython from source, which would be a second,
# unpinned toolchain to maintain for no benefit over a maintained PPA build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg2 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        ninja-build \
        graphviz \
        sqlite3 \
        libsqlite3-dev \
        clangd-14 \
        clang-tools-14 \
        clang-tidy-14 \
        universal-ctags \
        jq \
        fd-find \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        pipx \
        vim \
        less \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/clangd-14 /usr/local/bin/clangd \
    && ln -sf /usr/bin/clang-query-14 /usr/local/bin/clang-query \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# THE EXACT ASSEMBLER -- the toolchain record's packages at their pinned versions (`--allow-downgrades`: a base tag can
# ship a newer patch than the record). cuobjdump is one of them (the SASS gate, the region ledger and the composer
# extract cubins with it); Nsight Compute 2025.3 (the PM sampling the pipe timeline reads) is pinned beside them.
ARG CUDA_PACKAGES
RUN apt-get update && apt-get install -y --allow-downgrades ${CUDA_PACKAGES} \
        nsight-compute-2025.3.0=2025.3.0.19-1 \
    && rm -rf /var/lib/apt/lists/*
#: FAILS THE BUILD, LOUDLY, IF THE ASSEMBLER IS NOT THE RECORD'S: a build that appears to succeed while carrying the
#: wrong assembler is exactly the silent-drift shape the closed-world design refuses.
ARG ROLA_TOOLCHAIN
COPY tools/toolchains/ /opt/rola/bootstrap/tools/toolchains/
RUN python3.11 -c "import json, subprocess, sys; \
want = ' '.join(json.load(open('/opt/rola/bootstrap/tools/toolchains/${ROLA_TOOLCHAIN}.json'))['ptxas'].split()); \
have = ' '.join(subprocess.run(['ptxas', '--version'], capture_output=True, text=True).stdout.split()); \
sys.exit(0 if have == want else 'FATAL: ptxas is not toolchain ${ROLA_TOOLCHAIN}: ' + have)"

# --------------------------------------------------------------------------
# uv -- the Python resolver/installer this repo's ONE lock file
# (`requirements.lock`, generated by `uv pip compile`, see `requirements.in`'s
# header and `docs/setup.md`) is built with. Pinned release, sha256-verified
# against upstream's own published `.sha256` (same recipe as
# `tools/sccache_pin.json`).
ARG UV_VERSION=0.12.7
ARG UV_SHA256=788f18abea7c5f55d6216e4f5613fd89d4d59b631efeec117b2b07fe72f1da21
RUN curl -sSL -o /tmp/uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

# --------------------------------------------------------------------------
# hyperfine -- pinned release, sha256-verified (not in 22.04's apt repos).
ARG HYPERFINE_VERSION=1.20.0
ARG HYPERFINE_SHA256=63ad53934062118f5b0be11785e0bb1603d4b91667d1921f2fd8df9a8712040a
RUN curl -sSL -o /tmp/hyperfine.tar.gz \
        "https://github.com/sharkdp/hyperfine/releases/download/v${HYPERFINE_VERSION}/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${HYPERFINE_SHA256}  /tmp/hyperfine.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/hyperfine.tar.gz -C /tmp \
    && install -m 0755 "/tmp/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu/hyperfine" /usr/local/bin/hyperfine \
    && rm -rf /tmp/hyperfine.tar.gz "/tmp/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu"

# --------------------------------------------------------------------------
# difftastic -- pinned release, sha256-verified. NOT installed via pipx: it
# has no PyPI package (verified against the index at authoring time -- a
# Rust binary with no Python wheel, unlike `ast-grep-cli`/`py-spy` below), so
# it is pinned the same way as mold/hyperfine rather than forced through a
# tool that cannot actually install it.
ARG DIFFT_VERSION=0.70.0
ARG DIFFT_SHA256=2997d2bbe620534edbd79b0049f00ce84eef3fedb15c7822456d58e38d8b05c9
RUN curl -sSL -o /tmp/difft.tar.gz \
        "https://github.com/Wilfred/difftastic/releases/download/${DIFFT_VERSION}/difft-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${DIFFT_SHA256}  /tmp/difft.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/difft.tar.gz -C /usr/local/bin \
    && chmod 0755 /usr/local/bin/difft \
    && rm -f /tmp/difft.tar.gz

# --------------------------------------------------------------------------
# pipx tools -- ast-grep-cli, py-spy: real PyPI packages, so pipx is the
# actual right tool for them (difftastic is not one, see above).
# pre-commit is also installed here (a dev tool, not a repo runtime
# dependency) so it is present even before the Python lock's venv exists.
#
# clang-format==19.1.7 IS PINNED, NOT A FREE CHOICE: `.clang-format`'s own
# header states "Pinned tool: clang-format 19.1.7 (pip package
# `clang-format==19.1.7`)" and `tools/gen_shards.py`'s `--check` shells out to
# whatever `clang-format` resolves on PATH to compare against the committed
# generated shards -- found the hard way (2026-08-29): this image originally
# carried ONLY apt's `clang-tools-14` family (clangd/clang-tidy/clang-query,
# all fine for editor tooling) and NO `clang-format` at all, so a fresh
# in-container build's `pip install -e .` failed at `setup.py`'s
# `_gate_shards` with a generic "shards do not match the arm list" message
# that was actually `FileNotFoundError: clang-format` swallowed by the
# subprocess-returncode check -- a version 14 apt package would have been
# the WRONG fix anyway, since a formatter-version mismatch against files
# committed under 19.1.7 reads as a false "shards drifted" failure, the same
# failure shape as a real drift. pipx installs the exact pinned PyPI package.
RUN pipx install ast-grep-cli \
    && pipx install py-spy \
    && pipx install pre-commit \
    && pipx install clang-format==19.1.7

# --------------------------------------------------------------------------
# THE PYTHON ENVIRONMENT -- ONE LOCK FILE, `requirements.lock` (generated by
# `uv pip compile requirements.in`, `uv` being present on this box -- the
# card's "uv lock, or pip-compile if not" -- see `requirements.in`'s own
# header and `docs/setup.md`), replacing the hand-grown dev venv this box
# iterated in. `rola` itself is installed separately, `--no-build-isolation`,
# for the same reason `docs/build.md` states it: pip's isolation would fetch
# a CPU-only torch and build the extension against the wrong ABI.
WORKDIR /workspace
ARG TORCH_INDEX
COPY requirements.lock /tmp/requirements.lock
RUN uv venv /opt/venv --python 3.11 --seed \
    && . /opt/venv/bin/activate \
    && uv pip install -r /tmp/requirements.lock \
        --extra-index-url "${TORCH_INDEX}" \
        --index-strategy unsafe-best-match

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# --------------------------------------------------------------------------
# THE IMAGE'S DEV CONFIG -- `tools/dev.py init --image`, the one init every machine runs, from a bootstrap copy of the
# files it needs: the image's fixed layout (the mount points below, which `python tools/dev.py container compose` binds
# the host's lock directories, store and worktrees onto), the pinned sccache and mold installed under /opt/rola/tools,
# then `check --image`. A red check FAILS THE IMAGE BUILD: a tool the tree needs cannot go missing from the image
# unnoticed again. No home path is baked in (§17): mount sources are host-side, never compiled into the image.
#
# ROLA_ENV_KEY is the environment key the image is built from (`container build` passes it): sha256 over the
# environment's inputs, recorded as `environment.image_digest`, which `tools/ratify.py` stamps as
# `source.image_digest`. Declared HERE, after every heavy layer, so a new key rebuilds only this step.
ARG ROLA_ENV_KEY=""
ENV ROLA_DEV_CONFIG=/opt/rola/dev-config
#: Under WSL the Windows driver's Linux libraries (libdxcore, libcuda) are mounted read-only at /usr/lib/wsl/lib
#: (`host.wsl_lib`); without them first on the loader path CUDA fails to initialize with error 500 (measured
#: 2026-09-12 on Docker Desktop 20.10.17, driver 595.95). On native Linux the directory does not exist and changes nothing.
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
COPY requirements.lock /opt/rola/bootstrap/requirements.lock
COPY tools/dev.py tools/dev_config.py tools/sccache_toolchain.py tools/mold_toolchain.py \
     tools/sccache_pin.json tools/mold_pin.json /opt/rola/bootstrap/tools/
RUN mkdir -p /workspace/rola /workspace/store /workspace/suite /workspace/worktrees /run/rola/locks /run/rola/gpu \
    && /opt/venv/bin/python /opt/rola/bootstrap/tools/dev.py init --image --env-key "${ROLA_ENV_KEY}"

CMD ["/bin/bash"]
