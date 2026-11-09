# `tools/wheels.py` — the two wheels

Mirrors `tools/wheels.py`. It packages the graph's library of a clean, ratified commit as the two distributions
([`../../build.md#wheels`](../../build.md#wheels)): `rola`, pure Python, and `rola-cu13`, the binary plugin.
`bazel run //:wheels` runs it (`bazel/defs.bzl`'s `rola_wheels`), handing it the library, its record and the build's
census from its runfiles.

## The refusals

A wheel names a commit and ships a ratified build, so the tool refuses before it packages anything: a tree with
changes (`_clean_commit`); a build ratify's verdict does not pass (`_ratified`, `bazel test //:ratified`'s, which also
refuses an iteration build); and a record that is not this version's build under this toolchain, gated against the
manifests the plugin will carry (`_verify`).

## The packages

The two wheels are one build's output split by package, so `rola` never carries a binary and `rola-cu13` never carries
the Python package: `rola` is the tracked `rola/` package and the project's metadata, which `metadata` writes from
`pyproject.toml`'s `[project]` in the form setuptools wrote it, so a resolver reads the same fields it read before;
`rola-cu13` is the tracked `rola_cu13/` package, the graph's library and record, and the manifests that record hashes,
with metadata that requires `rola` at exactly its version. A commit's wheels are the same bytes each time (fixed member
order and timestamps).

## The check

`check` makes a scratch venv whose only borrowed package directory is the one holding the running interpreter's torch,
installs `rola` alone and runs `_PROBE` (a kernel call must refuse, naming `rola[cu13]`), then installs `rola-cu13` and
runs it again (the binary must load and list its carry arms). It prints both verdicts and `wheels check OK` or `FAIL`.
