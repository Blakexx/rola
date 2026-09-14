# `carry/layout.cuh` — a side's layout against the box

Mirrors `csrc/rola/src/carry/layout.cuh`.

`SideLayout` classifies each level of a side's carve order by where its digit falls against
the sixteen-leaf position bits: INNER within them, OUTER above, STRADDLING across bit four.
It is derived on the host (`make_side_layout`, from the two sides' rank orders and the span
rule) and carried in the parameter block, so every field is a constant-bank operand in the
kernel; the arrays are indexed only with compile-time levels. `digit_of_pos`,
`digit_of_box`, `straddle_digit`, `plane_row` and `local` are the leaf arithmetic the
sweeps, the snapshot and the head's box words use. The level lists (`n_inner`, `in_row`,
`in_span`, `n_outer`, `out_row`, `out_bshift`, `out_mask`) are the head's uniform loops for
a composed layout; `single_inner`, `single_outer`, `inner_row`, `outer_row` are the plain
layout's fast path. `span_total` and `span_base` are the box's routing footprint in
amplitude columns ([`carry_kernel.md#factor-rows`](carry_kernel.md#factor-rows)).
