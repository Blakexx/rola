# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""NO SOURCE BUILD: `pyproject.toml`'s build backend, which refuses. rola builds through its graph alone and ships as
two wheels (docs/build.md#wheels), so a source install is not supported: `pip install .` stops here with the reason
rather than building a package without its kernels."""

REFUSAL = ("rola does not build from source: `bazel run //:library` builds a checkout's extension and `bazel run "
           "//:wheels` its wheels (docs/build.md#wheels); install `rola[cu13]` from those")


def build_wheel(*_args, **_kwargs):
    raise SystemExit(REFUSAL)


def build_sdist(*_args, **_kwargs):
    raise SystemExit(REFUSAL)


def build_editable(*_args, **_kwargs):
    raise SystemExit(REFUSAL)
