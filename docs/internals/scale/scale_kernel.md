# `csrc/rola/src/scale/scale_kernel.cuh` — the scale pass's kernels

Three kernels, each a grid of independent units (R22), none with a float atomic: the window write masses
(`scale_writes`), their prefix over the windows (`scale_scan`), the bytes (`scale_reads`). The bound and the operand
block are [`scale.md`](scale.md).

<a id="digits"></a>

## A lane's digits

`digits_in<F16>`: lane `l` of a warp holds digits `c0 + 8 l .. + 7` of a row's chunk as unit values -- a u8 row's 8
bytes over 255, an f16 row's 16 bytes at least the floor -- or zeros where the chunk passes the row's width (no load
issued). A row's place is 64-bit (`row_at`: batch, head and token by the plane's strides). `warp_sum` and `warp_max`
are five butterfly rounds, every lane holding the result. Every loop over tokens, chunks and groups is warp-uniform
(the token count `n` and the widths are the CTA's), so the collectives sit under no lane-dependent branch; a token past
the window's last (`n`) loads the last token's row and contributes zero by its factor.

<a id="writes"></a>

## The window's write masses

`scale_writes` (a form a pair of its levels' dtypes), a CTA a (write head, window), level S a 256-digit chunk at a time: a warp its groups of
32 tokens (`g = warp, warp + 8, ..`), four tokens' rows in flight, `acc += w_S c(t)` eight digits a lane. In the FIRST
chunk each token's `c(t) = max_r w_R(t, r) F(t)` is formed beside its level-S loads -- its level-R row loaded a chunk at
a time and maxed in the lane, a warp max, times its `F` (`fm`, zero past the window's last) -- and lane `j` keeps token
`j`'s into `c_tok` (the plan's window wide: 2 KB at W = 512), which the warp's later chunks read back (the same warp's tokens: no barrier).
Each chunk ends with the eight warps' partials into `part` (8 KB), a barrier, thread `i` summing digit `i`'s over the
warps into the window's own masses, a barrier. One pass over the rows: the level-R and level-S loads of a token are in
flight together, where a first form's two passes (level R's maxima, a barrier, then level S) took turns at the DRAM
(MEASURED 2026-11-02, `tools/overlap_gate.py`, nl64k x 4 heads: 1.27x the DRAM floor at alt-k4, 1.21x dense). Bytes:
the window's rows of both write levels once, `fm` a float2 a token; arithmetic two operations a digit.

<a id="scan"></a>

## The prefix

`scale_scan`, a thread a (write head, level-S digit): the windows' masses in order, eight windows' loads issued before
their adds, each replaced by the sum of those before it -- `G_w`, the masses of the windows strictly before `w`, which
are what window `w`'s tokens read (the carry's window-aligned state). `windows x width_S x 8` bytes a head.

<a id="reads"></a>

## The bytes

`scale_reads` (a form a pair of its levels' dtypes), a CTA a (query head, window): its write head's `G_w` into dynamic shared memory (`width_S x
4` bytes); a warp its groups of 32 tokens, four at a time: each token's level-S row against `G_w` (the lane's eight
masses loaded once a chunk for the four tokens) and its level-R row's sum, each total a warp sum, the bound
`sum x dot x inv_factor` turned into the carry's byte (`carry::unit_byte`), lane `j` keeping token `j`'s and storing
it, one 32-byte store a group; a partial window's tail past the call is stored `kUnitByteNone` (the carry reads the window grid whole). Bytes: the window's rows of both read levels once.

THE BUDGET (sm_86, 912 GB/s, nl64k-alt-h4: L 64K x 4 heads at 256 x 256): every row read once -- the write side's
levels S (f16) and R (u8), the read side's S (u8) and R (f16) -- 97 MB a head with `fm` and the masses, ~0.42 ms a call
at the bandwidth floor; the line 1.3x it.
