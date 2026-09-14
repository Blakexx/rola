# `csrc/rola/src/entmax/entmax.{cuh,cu}` — the producer's union entmax solve

Hand CUDA over CUB warp primitives: the α-entmax solve that turns per-token read
and write LOGITS into route VALUES, a midpoint amplitude, and §4.1's packed
per-leaf support bitmap. `entmax.cuh` carries the host declaration pair
(`union_forward` / `union_backward`) AND [the warp-row vocabulary](#warp-row) the
whole entmax family speaks; `entmax.cu` is the union split's implementation — two
kernels and their host dispatch.

The family's other member is [`factor.md`](factor.md): the independent per-side
solves, whose forward is this file's phases B/C/D fed one logit tensor.

Design of record: this document.
Gates: `tests/oracle/test_entmax_cuda_gates.py`, against the **fp64 oracle only** —
never kernel-vs-kernel (trackE.md §4).

Symbols: a *stream* is one (batch, head)-like routing stream; a *token* is one row
of the solve; `width` is the level's branch width and `col` indexes it; `LW` is the
LOGICAL warp width (a power of two ≤ 32); `IPT` is items per lane, so
`W_PAD = LW * IPT` is the padded row width; `alpha` is 1.5 (α-entmax) or 2.0
(sparsemax); `tau` is the solve's threshold; *support* is the set of columns with
`midpoint > tau`.

---

## <a id="why-cuda"></a>1. Why this exists as CUDA at all

A Triton lowering of this solve measures **255 registers WITH SPILLS at 16.67%
occupancy** as written, and **115 reg / 33.33%** hand-restructured — where it still
carries an uncontrollable `tl.sort` scratch layout with **75%
bank-conflict-excessive wavefronts**. It is also a JIT against a closed-world AOT
policy that forbids exactly that.

Provenance: the producer bring-up measurements and
`producer-p3.md`.

## <a id="one-row-per-warp"></a>2. The structural idea: one row per (logical) warp

Every collective in the entmax solve — the sort, both prefix scans, every reduction
— runs ALONG ONE TOKEN'S ROW. Triton can only express them as block-scope
collectives over a 2D `(ROW_TILE, BLOCK_COLS)` tile, which is what

1. welds register demand to the CTA tile, so `num_warps` is not a lever, and
2. produces the bank-conflicted shared scratch.

Assigning one row to one warp makes every collective warp-scope and shuffle-based.
The consequences are all structural:

- **register demand becomes `O(items-per-lane)` and decouples from the warp count**
  — so the warp count is a free occupancy knob, which it provably is not under
  Triton (`producer-p3.md` §1);
- **the solve needs ZERO shared memory and ZERO block barriers.** `cub::WarpScan` /
  `cub::WarpReduce` `TempStorage` measures **1 byte** here — the pure-shfl
  specializations — so the 75%-bank-conflict `tl.sort` scratch is ELIMINATED, not
  tuned;
- **exactly ONE `__syncthreads` in the whole kernel** (§8, the epilogue).

CUB's warp primitives are the library half of the spec's "hand-roll semantics, never
primitives" rule.

The per-thread `TempStorage` declarations are only legal because CUB resolves all of
them to the pure-shfl specializations. That is MEASURED, not assumed
(`<scratchpad>/cub_sizeof.cu`: 1 byte for every `WarpScan` / `WarpReduce`
instantiation this kernel uses), and the `static_assert`s in both kernels turn a
future CUB that changes its mind into a compile error rather than a race.

## <a id="frozen-support-word"></a>3. The frozen support-word layout

`WORD_TOKENS = 32` is FROZEN (§4.1's per-leaf write bitmap, **nine downstream
consumers**). One CTA therefore owns exactly the 32 tokens of one support word, so
the `1 << (token % 32)` packing needs **no global atomics and no zero-init pass**.

`__launch_bounds__(128, 8)` — i.e. `MIN_BLOCKS_PER_SM = 8` — states the 64-register
target to ptxas, so a regression becomes a compile-time allocation decision instead
of a silent spill; the ratification manifest (`tools/ratify.py`) then fails the
BUILD on spill growth.

## <a id="strides"></a>4. `Strides` — one address expression, two route-value layouts

`read_values` / `write_values` accept BOTH route-value layouts through one stride
5-tuple `(stream, token, col, tile, lane)`:

| layout | `stream` | `token` | `col` | `tile` | `lane` |
|---|---|---|---|---|---|
| rank-3 dense level | `stride(0)` | `stride(1)` | `stride(2)` | 0 | 0 |
| rank-4 tile-major arena slice | `stride(0)` | 0 | `stride(2)` | `stride(1)` | `stride(3)` |

The single address expression

```
stream*s + tok*t + (tok>>5)*tile + col*c + (tok&31)*lane
```

is correct for both, because `five()` already zeroes the members the layout does not
use, deriving them from the tensor's own RANK. **Layout is therefore data, not a
codegen axis** — the Triton lowering this replaced needed a `tl.where(TILE_MAJOR,
...)` select and a specialization axis for it.

## <a id="stream-split"></a>4.2 The stream axis is TWO strides

A *stream* is one `(batch, head)` pair, and the grid indexes it as one number,
`b*H + h`. The logits do not have to be laid out that way, and the packed router GEMM
does not lay them out that way: it is a single `[B·L, d] x [d, H·C]` sgemm whose output
is `[B, L, H, C]`. Flattening `(B, H)` into one axis of THAT is not a view — it needs a
transpose of the whole plane.

**What that transpose cost, measured** (the P73 cell's projection, `[2,2048,512] x
[8,512,256]`, paired medians of 51): the `bhlc` einsum plus its reshape is 0.5149 ms,
where the `blhc` einsum is 0.4301 ms and returns a CONTIGUOUS VIEW of the one GEMM — no
copy at all. **−0.085 ms, and BIT-IDENTICAL**: a column of the output is a dot product
over the full hidden dim, so it does not depend on where the column is written. S4
measured the two other candidate reorders (`unsqueeze`/`expand` matmul) at +23% and
refused them; this is the form its §5 named as the one that would actually work.

So `Strides` carries the stream axis as a PAIR, `stream` and `stream2`, and each kernel
splits `blockIdx.y` once per block:

```
sb = stream / heads          //  b
sh = stream - sb * heads     //  h
addr_split = sb*s.stream + sh*s.stream2 + tok*s.token + col*s.col
```

Three things about that are load-bearing, and each was MEASURED rather than reasoned:

1. **`addr_split` is a SECOND expression, not a widening of `addr`.** Only the logit
   plane (and its gradient) needs two products; every other operand keeps the one-stride
   `addr` it always had. Widening all of them cost the family its 64-register bound —
   **4 spilling instantiations became 56**, with 180–244 byte spills at `IPT = 8`.
2. **The split is UNCONDITIONAL, and `heads >= 1` always** — `heads == 1` is one head,
   for which this is `sb = stream, sh = 0`. Guarding it as `heads ? stream/heads : stream`
   spilled the family just as badly, and the reason is not the division's latency:
   `blockIdx.y` is a special register, free to re-read, and a plain quotient of it is
   cheap to rematerialize, but a value that arrives through a branch is neither, so it
   is pinned in registers across the whole kernel.
3. **Hoisting the stream term per operand made it worse, not better** (56 → 58): six
   live `int64_t` bases is more expensive than rematerializing each from `blockIdx.y`.

The grid's stream extent is `logits.size(0) * heads`, which is `B*H` for the token-major
plane and `size(0)` for the flat one, since a flat plane declares `heads == 1`.

The producer emits the token-major plane only where the CUDA solve will read it (CUDA
fp32). The torch lowering and the fp64 reference solve read a level by SLICING, with no
stride tuple to read a head axis through, so they keep the flat plane — **no reference's
accepted shape changed**, and no dual run is owed.

## <a id="warp-row"></a>4.1 The warp-row vocabulary lives in the header

`Strides`/`addr`, `store_value`/`load_value`, `Descending`, `Pair`/`PairAdd`, the row
reductions `Max`/`Min`/`Sum`, `bf16_nonzero`, `logical_warp_mask`, `five` and the block constants are declared in
`entmax.cuh` rather than in either `.cu`, because `factor.cu`'s four kernels are the
same machine over a different phase set and a second copy would be a second thing to
keep in step.

They sit behind `#ifdef __CUDACC__`. That guard is load-bearing and not defensive:
`csrc/rola/rola_api.cpp` includes this header to reach the host declarations below,
and it is compiled by the HOST compiler, which knows nothing of `__device__` or
`__nv_bfloat16`.

<a id="reductions"></a>**The row reductions are rola's, not CCCL's.** `Max`, `Min` and `Sum` carry the exact
bodies and signatures CCCL 2.x shipped as `cub::Max` (`(b > a) ? b : a`), `cub::Min` (`(b < a) ? b : a`) and
`cub::Sum` (`a + b`): forwarding references, `common_type` results. CCCL 3 (CUDA 13) removed those three in favor of
`cuda::maximum`, `cuda::minimum` and `cuda::std::plus`, and `cuda::minimum` is `(a < b) ? a : b`: on a tie (`-0.0`
against `+0.0`) or a NaN it returns the other operand. Owning the bodies keeps every warp-row reduction's RESULT fixed
across CCCL versions.

**Its codegen is not fixed, and that is a measured, accepted cost.** The functor's type identity reaches `ptxas`'s
allocation: under one compiler (CUDA 12.4), these bodies in place of `cub::Max`/`Min`/`Sum` moved 512 entmax kernel
bodies and the register sum 31746 -> 31608, while `using Max = cub::Max` aliases moved none; spill and in-loop locals
were unchanged. Timed, the port costs 3-5% on a few entmax cells (deep-D4-w8, dense-D2, dense-NL4096) under 12.4 and
13.0 alike, and CCCL 3's own functors move the cost between cells without removing it. No identity-preserving spelling
exists once CCCL 3 deleted the originals. Accepted for the CUDA 13 move; the root cause is the K54 card.

The namespace is NAMED (`rola::entmax::detail`), so moving these out of `entmax.cu`
moved no mangled symbol and changed no generated code — `tools/ratify.py` reported
"the source digest moved but NO ratified body did" across all 380 bodies at the
move, which is the check that says so rather than the claim that it should.

## <a id="launch-shape"></a>5. Launch shape and the width class

The grid is `(words, streams, LEVELS)`: one CTA owns one 32-token support word of one
stream of one level, and `blockIdx.z` indexes a [`LevelTable`](factor.md#batched-levels)
that arrives in the kernel's parameter space. So one launch covers every union level
that selects the same template arm — same padded width, same alpha, same output dtype —
and `production.py`'s `_Plan` is what groups a walk into those launches.

**Why the class is in the grouping rule and the width is not.** `LW` and `IPT` are what
`w_pad = next_pow2(width)` selects, and they set how `WarpMergeSort` permutes the row and
how the `WarpScan` associates the prefix sums. Running a width-3 level in a width-64 arm
would therefore round `tau` differently in fp32 and could move a support boundary — it
would not be the loop it replaces. Widths INSIDE a class may differ freely; the kernel
reads each level's own `width` from the table. `build_table` refuses a mixed-class table
rather than padding to the largest.

That this is exactly the loop is gated, at `rtol = 0`:
`test_the_batched_solve_is_the_per_level_loop_bit_for_bit` runs four levels of one class
as one launch and as four, and asserts `torch.equal` on values, second output, midpoints
and support words.

## <a id="overflow-free-delta"></a>6. `overflow_free_delta` — the read-write difference

Verbatim from the shipped semantics: two opposite-sign finite extrema overflow a raw
subtraction while their MIDPOINT stays representable, so the MAGNITUDE is clamped
instead. Gate A1 holds this **bit-identical** against plain fp32 torch — it is pure
elementwise, so no reassociation exists.

The `saturated` flag it returns is not cosmetic: the backward zeroes `d_delta`
wherever it fires, because at saturation the difference is no longer a
differentiable function of the two logits.

`log_write_gate` is the companion: `log(sigmoid(-delta)) == -softplus(delta)`,
finite for every finite input, computed as `-(max(d,0) + log1p(exp(-|d|)))`.
**`log1pf` rather than `logf(1+x)`** because the governing baseline is the fp64
oracle, which uses `logsigmoid`, and `log1pf` is the closer composition to it.

## <a id="smem-slot-alias"></a>7. HAZARD — the support-word staging aliases the sort scratch

The `__shared__` slot is sized `max(SORT_BYTES, STAGE_BYTES)` per PHYSICAL warp and
serves both purposes in sequence. **No barrier is needed before the staging write**,
and that is what gets the whole kernel down to ONE `__syncthreads`: each physical
warp writes only into ITS OWN slot, and its own sorts are already retired
warp-synchronously by then, so there is no cross-warp hazard.

Introducing a shared slot, widening one warp's write past its slot, or letting a
warp read another's slot before the barrier all reintroduce a race the compiler will
not flag.

`SLOT` is `RAW_SLOT` rounded up to 16 B so warp `w`'s slot stays 16 B aligned:
`WarpMergeSort::TempStorage` is `4 + 4*W_PAD` bytes, i.e. NOT a multiple of its own
alignment.

Since the row sort became `RowSort` (§7.1), `SORT_BYTES` is `RowSort<LW, IPT>::kSlotBytes`,
which is **zero** for the register network. At `W_PAD <= 128` the slot is therefore the
staging alone and the alias is vacuous; at `W_PAD = 256` it is the arrangement above,
unchanged.

## <a id="row-sort"></a>7.1 `RowSort` — the row sort, and why there are two networks

Phase B needs one token's row in descending order. `RowSort<LW, IPT>::sort` is the ONE
entry point; which network it lowers to is decided by the WIDTH CLASS at compile time —
the same axis `int_switch` already derives `(LW, IPT)` from, so this is not a new
dispatch axis, it is the existing one.

| `W_PAD` | `IPT` | network | why |
|---|---|---|---|
| 2 – 128 | 1, 2, 4 | register-resident bitonic, shuffles only, zero shared memory | 14–22% faster |
| 256 | 8 | `cub::WarpMergeSort` | the bitonic's Theta(N log^2 N) work loses 10-14% at every register budget |

### The bitonic network

The element's rank is the BLOCKED index `lane*IPT + i`, which is the layout the prefix
walk and its `WarpScan` association require (§9). A standard bitonic network compares
indices differing in one bit, so the partner distance `j` splits the stages in two:

- `j < IPT` — partner is another ITEM of the same lane: a register compare-exchange, no
  shuffle at all.
- `j >= IPT` — partner is another lane, `lane ^ (j/IPT)`, and the WHOLE `IPT` block moves
  together. The direction bit `k` is then strictly above every item bit, so `desc` is
  uniform across the lane and no per-item direction is needed.

`W_PAD = 64` is 21 stages, 15 of them cross-lane; the merge sort it replaces staged the
row through shared memory once per merge round.

### Why the split is a complexity-class fact and not a tuning knob

The first explanation shipped here -- that the bitonic spills against the 64-register
`__launch_bounds__(128, 8)` -- was measured false by the 2026-08-17 unification probe
(`workflows/p74-producer/rowsort_unify.md`): the bitonic compiles with ZERO spills at 64
registers and at every raised budget to 174, and it loses at all of them (best variant
+10.3%, 86x the probe's noise band; a hybrid with one shared-memory exchange stage loses
more, and its elaborate form converges to `cub::WarpMergeSort` itself).

The honest wall is arithmetic: a bitonic network is Theta(N log^2 N) -- 204
compare-exchanges per lane at `W_PAD = 256` -- against the merge sort's
Theta(N log N) at ~64 per lane. Deleting the merge sort's 145 shared-memory ops and its
barrier buys back ~800 ALU instructions of extra comparison work, and the gap grows as
log N. The measured crossover (-13.7% at `W_PAD = 64`, -1.6% at 128, +10..14% at 256) is
where the two complexity classes cross, not a lucky boundary. So the wide class keeps
the merge sort, and its SASS is byte-identical to what it was before `RowSort` existed.

### Exactness is independent of the choice

**A sort is a PERMUTATION.** The phases downstream read the sorted row only through
(i) the prefix sums `P_k`, `Q_k` and (ii) the value `z_k` at each rank. Both are invariant
under permutation of EQUAL-valued elements — the top-`k` multiset is the same whichever
tied element is placed at rank `k`, so `P_k`/`Q_k` are the same floats and `z_k` is the
same float — hence `k*` is the same and `tau` is the same. `best_col` does differ between
tie orders, but it is consumed only as `chosen >= 0`, i.e. as a boolean. Downstream,
support is `value > tau` on UNSORTED columns, so a tied cluster is jointly in or jointly
out.

**Consequence: no network has to reproduce another's tie-breaking.** The two networks
here, and the merge sort that preceded both, are interchangeable at the bit.

MEASURED, not merely argued: 1560 cases / 5040 output tensors — every padded-width class,
both alphas, both logit and value dtypes, masked and unmasked, across random, tie-cluster,
one-ulp, all-equal and extreme-magnitude row families — digest-compared between a
merge-sort binary and this one. Zero mismatches. The standing gate is
`tests/unit/test_entmax_row_sort.py`, which asserts the tie-sensitive quantity (the
support set) at `rtol=0` and carries its own vacuity and teeth rows.

NOT covered, and stated: NaN keys. `>` is not a strict weak ordering on NaN, so no
sorting network — merge or bitonic — has defined output there. The kernel was already
unspecified on a NaN row (a NaN logit makes `row_max` NaN and every shifted key NaN);
that is unchanged.

## <a id="support-certificate"></a>7.2 The support certificate (derived, NOT shipped)

P74 lever 3 asked whether the sort could be replaced by a PARTIAL selection: sort only a
top-`m` prefix, and certify from the algorithm's own stopping condition that no element
beyond it can enter the support. The certificate exists and is exact. It is recorded here
because it is a property of the solve worth knowing, and because the measurement that
killed it is worth not repeating.

Write `z_1 >= ... >= z_W` for the sorted shifted keys (`z_1 = 0` exactly), `P_k`, `Q_k`
for the prefix sums, and `cand(k)` for the kernel's candidate predicate at rank `k`.

**Lemma A — `cand` is a monotone prefix property.**
For alpha 2, `cand(k) <=> g(k) := 1 + k*z_k - P_k > 0`, and
`g(k+1) - g(k) = k(z_{k+1} - z_k) <= 0`, so `g` is non-increasing.
For alpha 1.5, `cand(k) <=> G(k) := sum_{i<=k}(z_i - z_k)^2 < 4` (the `disc >= 0` clause is
implied), and `G` is non-decreasing because `z_i - z_{k+1} >= z_i - z_k >= 0` for `i <= k`.
Either way `{k : cand(k)} = {1..k*}` exactly, and `k* >= 1` always.

**Lemma B — the certificate is a pure VALUE bound.**
Using `P_k >= z_1 + (k-1) z_k` with `z_1 = 0`: for alpha 2, `g(k) <= 1 + z_k`, so `cand(k)`
requires `z_k > -1`; for alpha 1.5, `G(k) >= z_k^2`, so `cand(k)` requires `z_k > -2`.
Hence with `c = |{ j : z_j > B }|`, `B = -1` / `-2`, no rank beyond `c` can be a candidate,
and running the kernel's own arithmetic over ranks `1..c` yields exactly `k*` and `tau`.

The certificate needs neither the prefix sums nor a convergence test nor a grow-on-demand
loop — only `row_max`, which phase B already has — so `c` is exact and known BEFORE the
sort. Generalized: `P_k >= P_m + (k-m) z_k` gives `cand(k) => z_k > tau_m` for any
`m <= k*`, i.e. The kernel's own `tau_m` is a self-improving cut.

**Why it is not shipped.** MEASURED on production logits at the pinned cell
(`union_routing(1.5)`, widths 64, the untrained router the campaign runs): realized support
is 28.2 of 64, but `c` is **63.7 of 64** — the midpoint spread is under 2.0, so the bound
excludes almost nothing. The bootstrapped cut reaches 29 at best, and only by paying a
second selection stage plus a cross-lane COMPACTION through shared memory — which is the
cost the register network exists to delete. Selection only starts paying at routing gain
>= 4, a sharpness the shipped router does not have. Its win is also entirely
density-conditional, where the bitonic network's cost depends on `W_PAD` alone and never on
the data.

## <a id="logical-warp-mask"></a>8. HAZARD — the logical-warp member mask

```cpp
const unsigned lmask = (LW == WARP_SIZE)
                           ? 0xffffffffu
                           : (((1u << LW) - 1u) << (((threadIdx.x % WARP_SIZE) / LW) * LW));
```

**A full `0xffffffff` mask here DEADLOCKS, and it is not theoretical.** With logical
warps (`LW < 32`) and a ragged final support word, only SOME logical warps of a
physical warp have a live token — the rest take the `continue` — so a broadcast
naming all 32 lanes waits on threads that are somewhere else entirely. The member
mask must name exactly this logical warp.

CUB's own collectives compute this internally; the hand `__shfl_sync` broadcasts of
their lane-0 results did not, which is why the constant is spelled here.

The same expression, and the same reason, appears in the backward kernel.

## <a id="forward-phases"></a>9. The forward kernel, phase by phase

A row past the end is skipped by `continue`, and that is exactly equivalent rather
than merely close: such a row yields `has_valid = 0` → `tau = 0` → `support = false`
→ every store masked off.

**PHASE A — load striped (coalesced), fold, RETIRE both logit tiles.** Both logit
reads happen once; `midpoint = (r + w)/2` and `delta = r - w` (§6) are all that
survive into the rest of the kernel.

**PHASE B — row max/min, sentinel-pad, sort descending.** `midpoint` is shifted by
`row_max` for stability. Padded lanes get a FINITE row-relative sentinel,
`(row_min - row_max) - 1.0f`, which keeps them strictly below every real value
without putting an infinity into the scan; `row_min` is pre-shift, so it is shifted
too.

A sort is a **PERMUTATION** — exact, no arithmetic — so the sorted sequence is
identical whatever algorithm produces it. Feeding striped data is therefore fine:
the output arrangement is what the scan reads, and CUB delivers it BLOCKED (rank
order across lanes), which is exactly what the prefix walk needs.

**PHASE C — `tau`.** `prefix`, `prefix_sq`, `tau_candidates`, `candidate` and
`chosen_col` **all collapse to scalars**; none is ever materialized full width. One
`WarpScan` over the `(sum, sum-of-squares)` pair supplies each lane's carry, and the
per-rank candidate test is

```
alpha = 1.5 :  disc = p^2 - k*(q - 4);  tau_c = (p - sqrt(max(disc,0))) / k
               candidate iff rank < width and disc >= 0 and key > tau_c
alpha = 2.0 :  tau_c = (p - 1) / k
               candidate iff rank < width and key > tau_c
```

with `k = rank + 1`.

**ONE reduction selects the greatest valid `k` AND carries its `tau`**: the column
goes in the high 32 bits of a `unsigned long long`, which makes an integer `Max`
pick it, and lane rank ranges are disjoint so a tie is impossible. `best_col = -1`
maps to key 0, so an all-empty row yields `tau = 0`.

**PHASE D — values, in place over `midpoint`.** `read` is emitted HERE, while
`delta` is still the logit difference and before phase E overwrites it. That
ordering is the reason the read side needs no saved tensor.

**PHASE E — log-weights IN PLACE over `delta`, then the write normalization.** A
max-subtracted `exp` / sum over the support, with `has_mass` and `has_weight` guards
so an empty support produces exact zeros rather than `0/0`.

## <a id="bf16-support-boundary"></a>10. HAZARD — support is defined at the native bf16 storage boundary

The support bit is packed from `bf16_nonzero(read) || bf16_nonzero(write)`, i.e.
from the values AS STORED, **not** from the pre-quantization solve. This makes the
scheduler and the route stream ONE authority.

Union has one scheduling stream for both roles, so it is the UNION of post-cast
nonzeros that is packed: either side may create harmless false-live work, but **a
numerically live stored factor is never skipped**. Testing the pre-cast float
instead flips that guarantee — it silently drops work the consumer will still read —
and nothing in the build complains.

## <a id="epilogue-butterfly"></a>11. The epilogue — packing the 32-token support word

Lanes holding the same column in one physical warp differ only in bits at or above
`LW`, so a `__shfl_xor_sync` butterfly over exactly those bits ORs across the logical
warps IN PLACE — no shared atomics, no zero-init pass. Logical warp 0 of each
physical warp then stages its words, one `__syncthreads` follows, and warp 0 ORs the
`BLOCK_WARPS` staged copies into the output.

## <a id="backward"></a>12. The backward — the α-entmax Jacobian action through the union split

**No sort and no scan: this is a THREE-REDUCTION elementwise pass.** It shares the
forward's block/warp decomposition, width classes, striped
access, stride expression and launch bounds, and it **RECOMPUTES `delta`** rather
than reloading a saved tensor (cheaper than a third logit-sized read).

**SUPPORT IS AN INPUT** here — unpacked from the same frozen `support_words` the
forward emitted — so the two directions cannot disagree about the support set by
construction.

On support the action is `diag(s) - s s^T / (1.s)` — the α-entmax Jacobian action,
with `s = 1` (α = 2) or `s = sqrt(p) = p^(2-alpha)` (α = 1.5) — plus the write side's
second rank-1 (the softmax form inside `d_log_weight`) and the diagonal gate terms.

### Dead directions are exactly zero, by multiplication

Off support: `s = 0`, `read_values = 0` and `write_values = 0` (bf16 of exact
zeros), hence

```
d_log_weight      = write_values * (...)                                   = 0
d_midpoint_logits = s * (...)                                              = 0
d_delta           = d_read*read_values*write_gate - d_log_weight*read_gate = 0
```

so both `0.5*d_ml + d_delta` and `0.5*d_ml - d_delta` are exact zeros. **The gate
asserts the FORM (`rtol=0`), not a magnitude.**

### The tie guard

The VJP divides by the amplitude; at a TIE the amplitude is exactly 0, so the guard
substitutes 1 and the support mask zeroes the term. See the Wolfram sheet.

### Two measured re-read decisions

Both are register-pressure trades. MEASURED under `__launch_bounds__(128, 8)` at
`IPT = 8`:

| staged instead of re-read | measured cost |
|---|---|
| `write_values` / `d_write` across the two passes | 2·`IPT` live registers, **32 B stack frame** (spill) |
| `s` as a third live `IPT`-array | **24 B of spill** |

Re-reading is the cheaper trade on a kernel whose DRAM throughput is only ~30%. The first pass therefore carries NOTHING into the next but two scalars
(`local_dot` and the support mask), and `s` is recomputed from a re-read
`midpoint_values` in the final pass; one extra load per element buys the spill back
to zero.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/entmax/entmax.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="overflow-free-delta-2"></a>
### `overflow_free_delta`

The overflow-free read-write difference: two opposite-sign finite extrema overflow a
raw subtraction while their midpoint stays representable, so the MAGNITUDE is clamped
instead. `*saturated` marks where that clamp fired; the backward zeroes `d_delta`
there -- docs/internals/entmax/entmax.md#overflow-free-delta

<a id="sort-bytes"></a>
### `SORT_BYTES`

HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias
One per-physical-warp slot serves the row sort's scratch and then the support-word
staging. `RowSort::kSlotBytes` is ZERO for the register network, so at `IPT <= 4` the
slot is the staging alone -- docs/internals/entmax/entmax.md#row-sort

<a id="static-assert"></a>
### `static_assert`

These three TempStorages are per-thread rather than __shared__, which is legal only
because CUB resolves all of them to pure-shfl specializations here. MEASURED at
1 byte each; these static_asserts turn a future CUB that changes its mind into a
compile error rather than a race -- docs/internals/entmax/entmax.md#one-row-per-warp

<a id="stream32"></a>
### `stream32`

ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the solve's
stream, and a token-major operand needs it as `(b, h)`. The width is load-bearing --
a 64-bit div/mod here is a register-hungry sequence live across the whole kernel,
and it spilled the family. `heads == 0` means no split at all.
-- docs/internals/entmax/entmax.md#stream-split
THE STREAM SPLIT, and it is UNCONDITIONAL on purpose. `heads >= 1` always -- 1 is
one head, for which this is `sb = stream, sh = 0` -- because guarding it with
`heads ? ... : stream` MEASURED as a family-wide spill: the ternary pins `sb`/`sh`
in registers, where `blockIdx.y` is free to re-read and a plain quotient of it is
cheap to rematerialize. 4 spilling instantiations became 56.
-- docs/internals/entmax/entmax.md#stream-split

<a id="stream32-2"></a>
### `stream32`

ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the solve's
stream, and a token-major operand needs it as `(b, h)`. The width is load-bearing --
a 64-bit div/mod here is a register-hungry sequence live across the whole kernel,
and it spilled the family. `heads == 0` means no split at all.
-- docs/internals/entmax/entmax.md#stream-split
THE STREAM SPLIT, and it is UNCONDITIONAL on purpose. `heads >= 1` always -- 1 is
one head, for which this is `sb = stream, sh = 0` -- because guarding it with
`heads ? ... : stream` MEASURED as a family-wide spill: the ternary pins `sb`/`sh`
in registers, where `blockIdx.y` is free to re-read and a plain quotient of it is
cheap to rematerialize. 4 spilling instantiations became 56.
-- docs/internals/entmax/entmax.md#stream-split

<a id="kwpad"></a>
### `kWPad`

THE LAUNCH IS WRITTEN ONCE, and the `(lane width, items per thread)` pair is
DERIVED from the padded width rather than tabulated per case -- which is what makes
it impossible for one arm's argument list to drift from another's:
docs/internals/dispatch_switch.md#one-launch

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/entmax/entmax.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="strides-2"></a>
### `Strides`

`(stream, stream2, token, col, tile, lane)` -- ONE address expression serves every
operand layout in this family because the caller zeroes the members its layout does
not use, which is why layout is data here -- docs/internals/entmax/entmax.md#strides

THE STREAM AXIS IS TWO STRIDES because a token-major logit plane needs two: `b` and
`h` are separate axes of `[B, T, H, C]` and the solve's stream is `b*H + h`. The
caller splits `blockIdx.y` ONCE per block -- docs/internals/entmax/entmax.md#stream-split

<a id="addr-split"></a>
### `addr_split`

THE SPLIT ADDRESS, for the LOGIT plane alone. `b` and `h` are separate axes of a
token-major `[B, T, H, C]`, so its stream term is two products rather than one.

This is a SECOND expression and not a widening of `addr` because the widening was
MEASURED: making every operand's stream term two products cost the family its
64-register bound -- 4 spilling instantiations became 56, with 180-244 byte spills at
`IPT = 8`. Confining it to the one operand that needs it keeps every other address
rematerializable from a single register, which is what it always was.

<a id="rowsort"></a>
### `RowSort`

THE ROW SORT. Phase B needs one token's row in descending order; `RowSort` is the ONE
entry point, and which network it lowers to is decided by the WIDTH CLASS at compile
time -- the same axis `int_switch` already derives `(LW, IPT)` from.

EXACTNESS IS INDEPENDENT OF THE CHOICE. A sort is a PERMUTATION, and the phases
downstream read the sorted sequence only through its prefix sums and its rank-k value.
Both are invariant to how EQUAL keys are ordered, so `tau` is bit-identical across the
two networks and against the merge sort that preceded them, on every non-NaN row
-- docs/internals/entmax/entmax.md#row-sort

THE SPLIT IS A COMPLEXITY-CLASS FACT, not a tuning knob and not a register wall: a
bitonic network is Theta(N log^2 N) -- 204 compare-exchanges per lane at `W_PAD = 256`
against the merge sort's ~64 -- and the crossover between the two classes sits at
`W_PAD = 128`. Below it the shuffle-only network wins 14-19%; at `IPT = 8` it LOSES
10-14% at EVERY register budget (64 to 174, zero spills throughout; probed 2026-08-17).
-- docs/internals/entmax/entmax.md#row-sort

<a id="sort"></a>
### `sort`

The element's rank is the BLOCKED index `lane*IPT + i`, which is the layout the prefix
walk and its `WarpScan` association require. Partner distances below `IPT` are
intra-register; at or above it the whole `IPT` block moves as one, because the
network's direction bit then lies above every item bit and is uniform across the lane.

<a id="max-batched-levels"></a>
### `MAX_BATCHED_LEVELS`

THE PER-LAUNCH LEVEL TABLE. It travels in the kernel's PARAMETER space (448 B at
this bound, against a 4 KB floor), so batching a plan's levels into one launch costs
no device table, no H2D copy and no indirection. Offsets are COLUMN indices into the
base tensor, which is what `addr` adds them to -- docs/internals/entmax/factor.md#batched-levels

<a id="five"></a>
### `five`

The tuple a tensor's own RANK implies: rank-4 is the tile-major arena (no token
stride of its own), rank-3 the dense plane (no tile/lane stride). Deriving it keeps
one launch site per kernel instead of one per layout.


<a id="five-logits"></a>
### `five_logits`

THE LOGIT PLANE, in whichever of its two layouts the caller holds it: token-major
`[B, T, H, C]` (the packed router GEMM's own output, so taking it here is what
deletes the transpose that used to follow), or the flat `[BH, T, C]` whose head
stride is simply 0 -- docs/internals/entmax/entmax.md#stream-split

<a id="check-logits"></a>
### `check_logits`

TWO DTYPE AXES, ONE CALL: the LOGIT plane's and the route values'. Both are bf16 on
the shipped path -- the router GEMM emits bf16 operands' bf16 output and the consumer
reads bf16 -- and fp32 on the gates that want the solve's arithmetic without either
storage quantization -- docs/internals/entmax/factor.md#logit-dtype

<a id="union-forward"></a>
### `union_forward`

ONE LAUNCH, MANY LEVELS: the entries take BASE tensors plus per-level COLUMN
offsets, so every union level of one width class and one alpha runs in one launch
doing exactly what the per-level loop did -- docs/internals/entmax/entmax.md#launch-shape

`read_values`/`write_values` accept BOTH route-value layouts (rank-3 dense,
rank-4 tile-major) through ONE stride 5-tuple, so layout is data here and never a
codegen axis -- docs/internals/entmax/entmax.md#strides

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/entmax/entmax.cu` condensed at their call site into a short pointer.

<a id="launch-bounds"></a>
### `__launch_bounds__`

---------------------------------------------------------------------------
BACKWARD -- the alpha-entmax Jacobian action through the union split.

No sort and no scan: a THREE-REDUCTION elementwise pass. It shares the forward's
block/warp decomposition, width classes, striped access, stride expression and launch
bounds, and it RECOMPUTES `delta` rather than reloading a saved tensor.

SUPPORT IS AN INPUT here -- unpacked from the same frozen `support_words` the forward
emitted -- so the two directions cannot disagree about the support set by
construction. Off support every output term is an EXACT zero by multiplication, and
the gate asserts that FORM at rtol=0 rather than a magnitude.

The Jacobian derivation, the exact-zero argument, and the two MEASURED spill trades
that made this kernel re-read rather than stage: docs/internals/entmax/entmax.md#backward
---------------------------------------------------------------------------

<a id="logwritegate"></a>
### `log_write_gate`

log(sigmoid(-delta)) == -softplus(delta), finite for every finite input. `log1pf`
rather than `logf(1+x)` because the governing gate is the fp64 oracle's
`logsigmoid` -- docs/internals/entmax/entmax.md#overflow-free-delta

<a id="continue"></a>
### `continue`

Skipping a row past the end is EXACTLY equivalent, not merely close:
docs/internals/entmax/entmax.md#forward-phases

<a id="near-line-161"></a>
### near line 161

A finite row-relative sentinel keeps padded lanes strictly below every real
value without an infinity in the scan. `row_min` is pre-shift, so shift it too.

<a id="near-line-165"></a>
### near line 165

A sort is a PERMUTATION, so striped input is fine and the blocked output is what
the prefix walk wants -- docs/internals/entmax/entmax.md#forward-phases

<a id="packedkey"></a>
### `packed_key`

ONE reduction selects the greatest valid k AND carries its tau: the column in the
high 32 bits makes integer Max pick it, and `best_col = -1` maps to key 0 so an
all-empty row yields tau = 0 -- docs/internals/entmax/entmax.md#forward-phases

<a id="near-line-267"></a>
### near line 267

HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary
Support is the union of POST-CAST nonzeros, never the pre-quantization solve.

<a id="near-line-275"></a>
### near line 275

The butterfly runs over exactly the bits at or above LW, which is what separates
lanes holding the same column: docs/internals/entmax/entmax.md#epilogue-butterfly

<a id="localdot"></a>
### `local_dot`

This pass carries NOTHING into the next one but two scalars (`local_dot` and the
support mask); staging its loads instead is a measured spill
-- docs/internals/entmax/entmax.md#backward

<a id="near-line-368"></a>
### near line 368

`s` is NOT kept: it is recomputed from a re-read `midpoint_values` in the final
pass, for the same measured reason -- docs/internals/entmax/entmax.md#backward

<a id="safe"></a>
### `safe`

the VJP divides by the amplitude, which is exactly 0 at a TIE: substitute 1 and
let the support mask zero the term -- docs/internals/entmax/entmax.md#backward

<a id="nstreams"></a>
### `n_streams`

A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`;
a flat plane declares `heads == 1` and the same product is `size(0)`
-- docs/internals/entmax/entmax.md#stream-split

<a id="nstreams-2"></a>
### `n_streams`

A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`;
a flat plane declares `heads == 1` and the same product is `size(0)`
-- docs/internals/entmax/entmax.md#stream-split

<a id="near-line-33"></a>
### near line 33

`rola_api.cpp` includes this header and is compiled by the HOST compiler, so every
device declaration here is behind `__CUDACC__` -- docs/internals/entmax/entmax.md#warp-row

<a id="wordtokens"></a>
### `WORD_TOKENS`

FROZEN (nine downstream consumers): one CTA owns exactly the 32 tokens of one
support word, so `1 << (token % 32)` needs no global atomic and no zero-init pass
-- docs/internals/entmax/entmax.md#frozen-support-word

<a id="minblockspersm"></a>
### `MIN_BLOCKS_PER_SM`

With `BLOCK_THREADS`, this is `__launch_bounds__(128, 8)`: a 64-register target
stated to ptxas so a regression becomes a compile-time allocation decision instead
of a silent spill, which `tools/ratify.py` then fails the BUILD on.

<a id="addr"></a>
### `addr`

THE FLAT ADDRESS, unchanged: ONE stream register, and every term after it
rematerializes from constant memory. Keeping it that way is a REGISTER decision --
see `addr_split` -- docs/internals/entmax/entmax.md#stream-split

<a id="near-line-86"></a>
### near line 86

The ACCUMULATING store, for the tied backward's second contribution. bf16 device
atomics are sm_80+, which the closed-world arch set already is.

<a id="kslotbytes"></a>
### `kSlotBytes`

Shared bytes per PHYSICAL warp. Zero for the register network, which is the whole
point of it: at `IPT <= 4` the solve's only shared memory is the epilogue's staging.

<a id="near-line-183"></a>
### near line 183

HAZARD logical-warp-mask -- docs/internals/entmax/entmax.md#logical-warp-mask
the member mask for every hand `__shfl_sync` in this family. It names exactly ONE
logical warp; a full `0xffffffff` deadlocks on a ragged support word at `LW < 32`.

<a id="near-line-202"></a>
### near line 202

The union solve's midpoint plane is packed over the UNION levels alone, so it does
not share `value_off`; every other operand does.

<a id="paddedwidth"></a>
### `padded_width`

The padded width is what selects the `(lane width, items per thread)` pair, and every
level in one table must share it -- otherwise the sort and the prefix scan would
associate differently per level and the batch would not be the loop it replaces.

<a id="checklogits"></a>
### `check_logits`

THE LOGIT PLANE IS bf16 ON THE SHIPPED PATH -- the router GEMM runs bf16 operands on
the tensor cores and its output is bf16, so widening it would be the pass the
`LogitT` template exists to delete -- docs/internals/entmax/factor.md#logit-dtype
