# `csrc/rola/src/instantiations/` — generated per-arm translation units

Every file here is written by `tools/gen_shards.py --write` from the declared shipped set
(`tools/manifests/shipped_set.json`) and must never be edited: `--check` regenerates them
into memory and diffs, and the commit gate runs it.

One file per arm GROUP — rows that compile to the same kernel share a translation unit —
each defining its arms only when this build's generated selection header
(`build/generated/carry_selection.inc`) names them. That is what makes an arm subset a FILE
subset: an arm the build does not want is a file the build does not compile, and the arms
that are wanted compile in parallel rather than serially inside one front end.

The directory is exempt from the per-file mirror-doc rule because its contents are
generated; this page is what that exemption is paid for with. The mechanism each unit
instantiates is documented at its source: `docs/internals/carry/carry_arm.md` for the launch
and the census, `docs/internals/carry/carry_kernel.md` for the body, and
`docs/internals/tools/gen_shards.md` for the generator.
