# `tools/wheels.py` — the two wheels

Mirrors `tools/wheels.py`. It turns one gate build of a clean commit into the two distributions of wheel naming D
([`../../build.md#wheels`](../../build.md#wheels)): `rola`, pure Python, and `rola-cu13`, the binary plugin.

## The build

`_build` runs `pip wheel .` on the checkout with the `ITERATION_KNOBS` removed from the environment (the arch, arm,
part and skip variables that make an iteration build or skip a gate) and `ROLA_STRICT_MANIFEST=1`, so the post-build
gates fail the build instead of warning. The result is one combined wheel holding both packages. `setup.py` writes the
build record into that wheel's build tree as well as the checkout's: the package's files were copied before the
record existed.

## The refusals

`_clean_commit` refuses a tree with changes: a wheel names a commit. `_verify` refuses a combined wheel whose binaries
are anything but `rola_cu13/_C.abi3.so` (a part harness driver is an instrument), and one whose record is not a
shippable build of this version: `version`, `iteration` false, `archs` equal to the toolchain's ratified architectures,
`toolchain`, and `manifest_sha256` equal to `ratify.manifest_digest` over the manifest files the plugin will carry.

## The split

`split` gives `rola` the `rola/` package and the build's own metadata (which declares the `cu13` extra pinned to this
version), with a `py3-none-any` WHEEL. It gives `rola-cu13` the `rola_cu13/` package, the manifests under
`rola_cu13/manifests/`, and metadata of its own: its name and version, the build's author, licence and Python floor,
`Requires-Dist: rola==<version>` and the build's torch requirement, under `PLUGIN_TAG`. `_wheel_file` writes RECORD and
stores members in name order with the fixed `EPOCH` timestamp, so a commit's wheels are the same bytes each time.

## The check

`check` makes a scratch venv whose only borrowed package directory is the one holding the running interpreter's torch,
installs `rola` alone and runs `_PROBE` (a kernel call must refuse, naming `rola[cu13]`), then installs `rola-cu13` and
runs it again (the binary must load and list its carry arms). It prints both verdicts and `wheels check OK` or `FAIL`.
