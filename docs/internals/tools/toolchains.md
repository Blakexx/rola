# `tools/toolchains.py` — the toolchain records

Mirrors `tools/toolchains.py`. A **toolchain** is one exact assembler a fatbin is built and ratified under. Everything
that asks "which compiler, which manifests" reads it through this module: `setup.py`'s rules, `tools/ratify.py`,
`tools/supported.py` and the lints.

## The record

`tools/toolchains/<name>.json`, exactly four fields:

| field | meaning |
|---|---|
| `ptxas` | the whole `ptxas --version` output, whitespace-normalized: the string closed-world rule 1 compares |
| `cuda_major` | the CUDA major torch must be built for; torch's extension builder only refuses another major once it compiles, the build refuses before |
| `torch_index` | the PyTorch wheel index serving torch for that major |
| `container` | the dev container that carries this assembler: `base_image` (an `nvidia/cuda` devel tag pinned by its digest, `name:tag@sha256:...`: a tag can be re-pushed, the digest is the image) and `packages`, the exact apt `name=version` pins that put the record's `ptxas` in it (`tools/dev.py container build` passes both, and the image refuses a `ptxas` that is not the record's) |

Two records naming one assembler are refused: one assembler is one toolchain. The record's name is the manifest
directory: `tools/manifests/<name>/sm_XX.json`, one file per architecture
([`../../ratification.md`](../../ratification.md)). The declarations beside those directories
(`shipped_set.json`, `derivation_cases.json`, `derivation.json`) describe the source, not a compiler, and stay in
`tools/manifests/`.

## Selection

The build takes no toolchain parameter. The configured toolkit (dev config `toolchain.cuda_home`) is the choice, and the
record whose `ptxas` equals that toolkit's is the toolchain (`for_ptxas`); a toolkit no record names refuses, naming the
declared ones. A second spelling of the choice would be one that can disagree with the compiler that actually runs.
`setup.py` resolves it at import, except for a Python-only install (`ROLA_NO_EXTENSION=1`), which compiles nothing;
`tools/ratify.py` resolves it per run; a lint that checks a built binary takes the toolchain from that binary's own
`BUILD_CONFIG["ptxas"]`, never from whatever toolkit the checking host has.

`rola/_build_config.py` records the toolchain's name beside its `ptxas`, and `tools/supported.py` writes one README row
per ratified (toolchain, architecture).

Adding or changing a toolchain is [`../../bringup.md`](../../bringup.md#changing-the-toolchain).
