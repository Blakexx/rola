# `carry/carry_api.cuh` — the carry family's host entry declarations

Mirrors `csrc/rola/src/carry/carry_api.cuh`.

The entries the registration translation unit registers: the forward pass, the device build
stamp with its per-arm census, and the arm list. This header is the only place the carry
family and `torch` meet in a declaration; the definitions and every refusal are in
[`carry.md`](carry.md), and no arm's translation unit includes this file — that is what
keeps an arm from parsing the torch headers.
