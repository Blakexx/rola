# `common/arm.cuh` — one carry arm's key and constants

Mirrors `csrc/rola/src/common/arm.cuh`. `CarryKey` is a carry arm's key, the carve's operation-plan pair in
`tools/gen_shards.py`'s `ARM_KEY` order -- its last field, `same_split`, whether the fold's lists are the readout's
digits (both sides list one level) or its K; `Arm`
hands one row's fields to a kernel as compile-time constants. The header
includes nothing, so a per-arm translation unit (`csrc/rola/src/instantiations/`) never parses the torch headers the
host dispatch needs (`docs/internals/common/arm_switch.md`).
