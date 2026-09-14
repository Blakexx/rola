# `csrc/rola/src/facts/liveness_api.cuh` — the liveness pass's host seam

Mirror doc for the pass's one host entry and for the refusals it makes before a launch
exists. The kernel is [`liveness.md`](liveness.md); the output's shape and meaning are
[`liveness_contract.md`](liveness_contract.md). `rola/ops/liveness.py` is the Python
caller, and it translates the contract's value types into this argument list and does
nothing else.

<a id="liveness-words"></a>

## `liveness_words`

    liveness_words(read_plane, write_plane, widths,
                   dense_read, mask_read, dense_write, mask_write) -> int32 tensor

The two packed `[..., L, Σ_l B_l]` bf16 amplitude planes a producer already emits, the
descriptor's per-level widths, and per side the dense-level bitmask and the digit
mask's words. It returns `[BH, 2, rows, words]` `int32` — the class-1 table, allocated
here because it is this call's output and is owned by the engine DAG for the call.

`BH` is not an argument: it is whatever leading extent the planes carry
(`numel / (L · rows)`), so the same entry serves the `[B, H, T, sumW]` shape
`pack_side` produces and the `[BH, L, rows]` a test builds by hand.

**The masks arrive as words, never as widths.** `dense_*` is a bitmask over levels — a
MODE fact — and `mask_*` is one bit per row, packed by
`rola.engine.facts.liveness.side_statics`, which is the only place `b_l` appears.
R13's two contracts put that translation at the seam and keep logical-width fields out
of every kernel entry signature.

<a id="the-plan"></a>

## The refusals, and why each is a refusal rather than a guess

`build_plan` states them all before a launch exists (KERNEL_STANDARDS §18: a seam
refuses rather than guesses):

* the depth is one to four levels, and **each width is a power of two at or above 16**
  (the descriptor's law, R13) and at most 256 — the pass reads sixteen digits per lane
  in one transaction, and that law is exactly what makes every level's width and row
  base a whole number of those groups. A narrower or non-power-of-two level is refused
  BY NAME rather than served by a second, scalar path: there is no fallback in this
  family;
* `L ≥ 1`;
* each side's dense bitmask names only levels inside the depth;
* each side's digit mask is exactly `mask_words(rows)` words — a short mask would read
  a level's rows out of uninitialized parameter space;
* the planes are contiguous device bf16 of the SAME shape, since the two sides describe
  one call;
* **`Σ_l B_l` equals the packed amplitude row's width.** This is the load-bearing one:
  the contract's whole economy is that a row index IS the column it votes from, so a
  plane whose row is not the widths' sum would have the pass voting one digit's bit
  from another digit's amplitude. It is caught here, not by a wrong answer.

The entry allocates the output and launches; it holds no fold, because every class-2
grain is a host fold over what it returns
([`liveness_contract.md`](liveness_contract.md#folds)).
