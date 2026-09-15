# `carry/carry_arm_abi.cuh` — the per-arm ABI

Mirrors `csrc/rola/src/carry/carry_arm_abi.cuh`.

Pointers and scalars only. The dispatch calls an arm through these declarations, so an arm's
translation unit compiles against the parameter block and CUDA's runtime alone and never
parses a torch header, not even the stable seam's — a torch header was most of an arm's compile time
(`docs/build.md`'s compile ledger).

`ArmCensus` is the row shape [`carry.md#census`](carry.md#census) describes;
`ROLA_CARRY_ARM_DECLARE` is the X-macro form the dispatch expands over the built set.
