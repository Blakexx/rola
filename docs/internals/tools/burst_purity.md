# `tools/burst_purity.py` — the burst tier, read off the SASS

The source lint (`tools/lint/burst_tier.py`, KERNEL_STANDARDS §23) refuses a vote, a barrier or
a data-dependent branch written into a `//: @burst` function. This gate refuses the ones the
compiler writes: a collective placed after a value ptxas cannot prove warp-uniform is wrapped in
a convergence bracket (`BSSY`/`BSYNC`) or routed through a subroutine (`CALL`), and a branch the
source never had can appear from if-conversion undone. None of these is visible in the source,
and each puts a serialization point between two bursts.

The cubin needs line information (`--arm N` compiles the carry arm with `-lineinfo`, as
`life_ranges.py` does). An instruction belongs to a burst function when any frame of its inline
chain lies in that function's lines, so the fold's gather and burst are found inside
`Burst::run` inside the stream's step. A row a function: instructions, HMMAs, shuffles (allowed:
a shuffle is data movement, not a vote), and the forbidden opcodes; exit 1 on any.

Forbidden: `VOTE*`, `BAR*`, `BSSY`, `BSYNC`, `CALL*`, `WARPSYNC`, `MEMBAR*`, `BRA*`, `EXIT`,
`RET*`. The primitive's own pair loop lives in `common/burst.cuh` and is not a burst function,
so its loop branch and count guards are not counted against the functions it runs.

It runs as the `purity` instrument beside `registers` and `sass` on every suite run.
