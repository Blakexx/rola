# `common/build_stamp.cu` — the extension's freshness fact

Mirrors `csrc/rola/src/common/build_stamp.{cu,cuh}`.

A stale binary is the failure this repository has re-learned more times than any
other, and every check that sits one level above it passes on the shape that keeps
recurring: the right file, at the right path, built from sources that have since
moved. An import path, a file hash and an mtime are all one level too high. The only
check that cannot be fooled is a fact the DEVICE produces out of a constant the
compile put there.

<a id="csrc-build-stamp"></a>
## `csrc_build_stamp()`

A one-thread kernel returns `ROLA_CSRC_STAMP`: 60 bits of the sha256 of
`csrc/rola/src`, the same digest `tools/ratify.py` records as `source.csrc_sha256`.
The value is written by the device out of a constant this translation unit compiled
in, so it is a fact about the loaded fatbin and not about the host process that asks.

`tests/unit/test_build_stamp.py` is what makes it a gate rather than a number: the
value the fatbin returns must EQUAL what this tree hashes to right now. The test reads
the digest function out of `tools/ratify.py` by path rather than importing the
installed package — the fact wanted is a property of THIS repository, and a tool that
imports the package to learn it can be answered by another worktree's build.

## Why a generated header and not a define

`build/generated/csrc_stamp.inc` is written by `setup.py` before the build compiles
and by `tools/ratify.py` before it measures, from one function. A `-D` on the
extension's flag list reaches every translation unit's argv, so every `csrc/` edit
would rebuild the whole extension; a header included only here rebuilds one file, and
ninja hashes it through the depfile like any other include. A flag ninja does not hash
is worse than either: the build reports success and the binary silently stays the
previous variant.

The digest covers `csrc/rola/src` and this header lives outside it, so the constant
never feeds back into its own input.

## What it is not

It is not the families' footprint stamps (`intra_build_stamp`,
`rola_decode_build_stamp`). Those fold struct sizes and launch widths into a value a
test decodes arithmetically, and they answer a different question: whether the loaded
binary's shapes are the ones this source declares. Both are wanted, and neither
subsumes the other — a footprint stamp cannot see a change that leaves the shapes
alone, and a content hash cannot say which shape moved.
