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
never kernel-vs-kernel.

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
  Triton;
- **the solve needs ZERO shared memory and ZERO block barriers.** `cub::WarpScan` /
  `cub::WarpReduce` `TempStorage` measures **1 byte** here — the pure-shfl
  specializations — so the 75%-bank-conflict `tl.sort` scratch is ELIMINATED, not
  tuned;
- **exactly ONE `__syncthreads` in the whole kernel** (§8, the epilogue).

CUB's warp primitives are the library half of the spec's "hand-roll semantics, never
primitives" rule.

<a id="static-assert"></a>The per-thread `TempStorage` declarations are only legal because CUB resolves all of
them to the pure-shfl specializations. That is MEASURED, not assumed (1 byte for every `WarpScan` / `WarpReduce`
instantiation this kernel uses), and the `static_assert`s in both kernels turn a future CUB that changes its mind
into a compile error rather than a race.

## <a id="frozen-support-word"></a>3. The frozen support-word layout

<a id="wordtokens"></a>`WORD_TOKENS = 32` is FROZEN (§4.1's per-leaf write bitmap, **nine downstream
consumers**). One CTA therefore owns exactly the 32 tokens of one support word, so
the `1 << (token % 32)` packing needs **no global atomics and no zero-init pass**.

<a id="residency"></a>The launch bound's residency is the solves' plan at the build (`EntmaxPlan<B>`): the blocks an
SM their register target fits in the card's registers -- `__launch_bounds__(128, 8)` on every tabulated card's 64K, a
64-register target (`kForwardRegs`) -- which states the target to ptxas, so a regression becomes a compile-time
allocation decision instead of a silent spill; the ratification manifest (`tools/ratify.py`) then fails the
BUILD on spill growth. The backward's widest class (8 items a thread) holds both terms of
its 8 items (`d_mid`, `d_delta`) across the pair reduction, which sm_80's allocation does
not fit in 64 (an 8-byte spill, measured 2026-10-15; sm_86's did): its target is 72
registers (`backward_regs`), 7 blocks an SM on 64K (`EntmaxPlan<B>::backward_min_blocks`), every other class's 64.

## <a id="strides"></a>4. `Strides` — one address expression, two route-value layouts

<a id="strides-2"></a><a id="five"></a>`read_values` / `write_values` accept BOTH route-value layouts through one stride
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
use, deriving them from the tensor's own RANK (rank-4 is the tile-major arena, rank-3 the dense plane), which keeps
one launch site per kernel instead of one per layout. **Layout is therefore data, not a codegen axis.**

<a id="near-line-202"></a>The union solve's midpoint plane is packed over the UNION levels alone, so it does not
share the levels' `value_off`; every other operand does.

## <a id="stream-split"></a>4.2 The stream axis is TWO strides

A *stream* is one `(batch, head)` pair, and the grid indexes it as one number,
`b*H + h`. The logits do not have to be laid out that way, and the packed router GEMM
does not lay them out that way: it is a single `[B·L, d] x [d, H·C]` sgemm whose output
is `[B, L, H, C]`. Flattening `(B, H)` into one axis of THAT is not a view — it needs a
transpose of the whole plane.

**What that transpose costs, measured** (projection `[2,2048,512] x [8,512,256]`, paired
medians of 51): the `bhlc` einsum plus its reshape is 0.5149 ms, where the `blhc` einsum is
0.4301 ms and returns a CONTIGUOUS VIEW of the one GEMM — no copy at all. **−0.085 ms, and
BIT-IDENTICAL**: a column of the output is a dot product over the full hidden dim, so it
does not depend on where the column is written. The two other candidate reorders
(`unsqueeze`/`expand` matmul) measured +23%.

<a id="stream32"></a><a id="stream32-2"></a><a id="addr"></a><a id="addr-split"></a>So `Strides` carries the
stream axis as a PAIR, `stream` and `stream2`, and each kernel splits `blockIdx.y` once per block, as ONE 32-bit
division on a uniform value (a 64-bit div/mod is a register-hungry sequence live across the whole kernel, and it
spilled the family). The flat `addr` keeps ONE stream register, every other term rematerializing from constant
memory; the logit plane alone takes `addr_split`:

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

<a id="nstreams"></a><a id="nstreams-2"></a><a id="five-logits"></a>The grid's stream extent (`n_streams`) is
`logits.size(0) * heads`, which is `B*H` for the token-major plane and `size(0)` for the flat one, since a flat plane
declares `heads == 1`: the solve takes the logit plane in whichever layout the caller holds it, so the transpose
above never happens.

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

<a id="near-line-33"></a>They sit behind `#ifdef __CUDACC__`. That guard is load-bearing and not defensive:
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
exists once CCCL 3 deleted the originals. Accepted for the CUDA 13 move.

The namespace is NAMED (`rola::entmax::detail`), so moving these out of `entmax.cu`
moved no mangled symbol and changed no generated code — `tools/ratify.py` reported
"the source digest moved but NO ratified body did" across all 380 bodies at the
move, which is the check that says so rather than the claim that it should.

## <a id="launch-shape"></a>5. Launch shape and the width class

<a id="union-forward"></a><a id="max-batched-levels"></a>The grid is `(words, streams, LEVELS)`: one CTA owns one
32-token support word of one stream of one level, and `blockIdx.z` indexes a
[`LevelTable`](factor.md#batched-levels) that arrives in the kernel's parameter space (448 B at
`MAX_BATCHED_LEVELS`, against a 4 KB floor), so batching costs no device table, no H2D copy and no indirection. So
one launch covers every union level
that selects the same template arm — same padded width, same alpha, same output dtype —
and `production.py`'s `_Plan` is what groups a walk into those launches.

<a id="paddedwidth"></a>
<a id="kwpad"></a>**Why the class is in the grouping rule and the width is not.** `LW` and `IPT` are what
`w_pad = next_pow2(width)` selects, and they set how `WarpMergeSort` permutes the row and
how the `WarpScan` associates the prefix sums. Running a width-3 level in a width-64 arm
would therefore round `tau` differently in fp32 and could move a support boundary — it
would not be the loop it replaces. Widths INSIDE a class may differ freely; the kernel
reads each level's own `width` from the table. `build_table` refuses a mixed-class table
rather than padding to the largest.

That this is exactly the loop is gated, at `rtol = 0`:
`test_the_batched_solve_is_the_per_level_loop_bit_for_bit` runs four levels of one class
as one launch and as four, and asserts `torch.equal` on values, second output, midpoints
and support words. The launch itself is written once, the `(lane width, items per thread)` pair derived from the
padded width inside it ([`dispatch_switch.md#one-launch`](../dispatch_switch.md#one-launch)).

<a id="check-logits"></a><a id="checklogits"></a>Two dtype axes meet in one call, the logit plane's and the route
values': bf16 on the shipped path (the router GEMM's bf16 output), fp32 on the gates that want the solve's
arithmetic without either storage quantization ([`factor.md#logit-dtype`](factor.md#logit-dtype)).

## <a id="overflow-free-delta"></a>6. `overflow_free_delta` — the read-write difference

<a id="overflow-free-delta-2"></a>Verbatim from the shipped semantics: two opposite-sign finite extrema overflow a raw
subtraction while their MIDPOINT stays representable, so the MAGNITUDE is clamped
instead. Gate A1 holds this **bit-identical** against plain fp32 torch — it is pure
elementwise, so no reassociation exists.

The `saturated` flag it returns is not cosmetic: the backward zeroes `d_delta`
wherever it fires, because at saturation the difference is no longer a
differentiable function of the two logits.

<a id="logwritegate"></a>`csrc/rola/src/entmax/entmax.cuh`'s `log_write_gate` is the companion:
`log(sigmoid(-delta)) == -softplus(delta)`,
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

<a id="sort-bytes"></a><a id="kslotbytes"></a>`SORT_BYTES` is `RowSort<LW, IPT>::kSlotBytes` (§7.1), shared bytes
per PHYSICAL warp, which is **zero** for the register network. At `W_PAD <= 128` the slot is therefore the
staging alone and the alias is vacuous; at `W_PAD = 256` it is the arrangement above,
unchanged.

## <a id="row-sort"></a>7.1 `RowSort` — the row sort, and why there are two networks

<a id="rowsort"></a>Phase B needs one token's row in descending order. `RowSort<LW, IPT>::sort` is the ONE
entry point; which network it lowers to is decided by the WIDTH CLASS at compile time —
the same axis `int_switch` already derives `(LW, IPT)` from, so this is not a new
dispatch axis, it is the existing one.

| `W_PAD` | `IPT` | network | why |
|---|---|---|---|
| 2 – 128 | 1, 2, 4 | register-resident bitonic, shuffles only, zero shared memory | 14–22% faster |
| 256 | 8 | `cub::WarpMergeSort` | the bitonic's Theta(N log^2 N) work loses 10-14% at every register budget |

### The bitonic network

<a id="sort"></a>The element's rank is the BLOCKED index `lane*IPT + i`, which is the layout the prefix
walk and its `WarpScan` association require (§9). A standard bitonic network compares
indices differing in one bit, so the partner distance `j` splits the stages in two:

- `j < IPT` — partner is another ITEM of the same lane: a register compare-exchange, no
  shuffle at all.
- `j >= IPT` — partner is another lane, `lane ^ (j/IPT)`, and the WHOLE `IPT` block moves
  together. The direction bit `k` is then strictly above every item bit, so `desc` is
  uniform across the lane and no per-item direction is needed.

`W_PAD = 64` is 21 stages, 15 of them cross-lane, with no shared memory; the merge sort stages
the row through shared memory once per merge round.

### Why the split is a complexity-class fact and not a tuning knob

The bitonic does not lose to register spills against the 64-register
`__launch_bounds__(128, 8)`: measured 2026-08-17, it compiles with ZERO spills at 64
registers and at every raised budget to 174, and it loses at all of them (best variant
+10.3%, 86x the probe's noise band; a hybrid with one shared-memory exchange stage loses
more, and its elaborate form converges to `cub::WarpMergeSort` itself).

The honest wall is arithmetic: a bitonic network is Theta(N log^2 N) -- 204
compare-exchanges per lane at `W_PAD = 256` -- against the merge sort's
Theta(N log N) at ~64 per lane. Deleting the merge sort's 145 shared-memory ops and its
barrier buys back ~800 ALU instructions of extra comparison work, and the gap grows as
log N. The measured crossover (-13.7% at `W_PAD = 64`, -1.6% at 128, +10..14% at 256) is
where the two complexity classes cross, not a lucky boundary. So the wide class keeps
the merge sort.

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
are interchangeable at the bit.

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

Could the sort be replaced by a PARTIAL selection? Sort only a
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

<a id="near-line-183"></a>**A full `0xffffffff` mask here DEADLOCKS, and it is not theoretical.** With logical
warps (`LW < 32`) and a ragged final support word, only SOME logical warps of a
physical warp have a live token — the rest take the `continue` — so a broadcast
naming all 32 lanes waits on threads that are somewhere else entirely. The member
mask must name exactly this logical warp.

CUB's own collectives compute this internally; the hand `__shfl_sync` broadcasts of
their lane-0 results did not, which is why the constant is spelled here.

The same expression, and the same reason, appears in the backward kernel.

## <a id="forward-phases"></a>9. The forward kernel, phase by phase

<a id="continue"></a>A row past the end is skipped by `continue`, and that is exactly equivalent rather
than merely close: such a row yields `has_valid = 0` → `tau = 0` → `support = false`
→ every store masked off.

**PHASE A — load striped (coalesced), fold, RETIRE both logit tiles.** Both logit
reads happen once; `midpoint = (r + w)/2` and `delta = r - w` (§6) are all that
survive into the rest of the kernel.

<a id="threshold"></a>PHASES B AND C ARE ONE HEADER FUNCTION, `threshold<LW, IPT, ALPHA15>`: the union forward calls
it on the midpoint row and the independent forward (`factor.cu`) on its masked row -- one statement of the sorted
threshold, taking the row's member bits and their count (a masked row can be empty: its `row_max` is then 0 and its
sentinel -1).

<a id="near-line-161"></a>
<a id="near-line-165"></a>**PHASE B — row max/min, sentinel-pad, sort descending.** `midpoint` is shifted by
`row_max` for stability. Padded lanes get a FINITE row-relative sentinel,
`(row_min - row_max) - 1.0f`, which keeps them strictly below every real value
without putting an infinity into the scan; `row_min` is pre-shift, so it is shifted
too.

A sort is a **PERMUTATION** — exact, no arithmetic — so the sorted sequence is
identical whatever algorithm produces it. Feeding striped data is therefore fine:
the output arrangement is what the scan reads, and CUB delivers it BLOCKED (rank
order across lanes), which is exactly what the prefix walk needs.

**PHASE C — `tau`.** The prefix sums, their squares, the per-rank candidates and the chosen column **all collapse to
scalars**; none is ever materialized full width. One
`WarpScan` over the `(sum, sum-of-squares)` pair supplies each lane's carry, and the
per-rank candidate test is

```
alpha = 1.5 :  disc = p^2 - k*(q - 4);  tau_c = (p - sqrt(max(disc,0))) / k
               candidate iff rank < width and disc >= 0 and key > tau_c
alpha = 2.0 :  tau_c = (p - 1) / k
               candidate iff rank < width and key > tau_c
```

with `k = rank + 1`.

<a id="packedkey"></a>**ONE reduction selects the greatest valid `k` AND carries its `tau`** (`packed_key`): the column
goes in the high 32 bits of a `unsigned long long`, which makes an integer `Max`
pick it, and lane rank ranges are disjoint so a tie is impossible. `best_col = -1`
maps to key 0, so an all-empty row yields `tau = 0`.

**PHASE D — values, in place over `midpoint`.** `read` is emitted HERE, while
`delta` is still the logit difference and before phase E overwrites it. That
ordering is the reason the read side needs no saved tensor.

<a id="union-write"></a>**PHASE E — log-weights IN PLACE over `delta`, then the write normalization**
(`union_write<LW, IPT, kFast>` in the header). A max-subtracted `exp` / sum over the support, with `has_mass` and
`has_weight` guards so an empty support produces exact zeros rather than `0/0`. `kFast` is the fused producer's policy
(`docs/internals/produce/ampere/produce_kernel.md#solve`): the logarithms, exponentials and quotient on the special-function unit,
since its fp32 levels only feed u8 digits; this family keeps the libm forms.

## <a id="bf16-support-boundary"></a>10. HAZARD — support is defined at the native bf16 storage boundary

<a id="near-line-267"></a>The support bit is packed from `bf16_nonzero(read) || bf16_nonzero(write)`, i.e.
from the values AS STORED, **not** from the pre-quantization solve. This makes the
scheduler and the route stream ONE authority.

Union has one scheduling stream for both roles, so it is the UNION of post-cast
nonzeros that is packed: either side may create harmless false-live work, but **a
numerically live stored factor is never skipped**. Testing the pre-cast float
instead flips that guarantee — it silently drops work the consumer will still read —
and nothing in the build complains.

## <a id="epilogue-butterfly"></a>11. The epilogue — packing the 32-token support word

<a id="near-line-275"></a>Lanes holding the same column in one physical warp differ only in bits at or above
`LW`, so a `__shfl_xor_sync` butterfly over exactly those bits ORs across the logical
warps IN PLACE — no shared atomics, no zero-init pass. Logical warp 0 of each
physical warp then stages its words, one `__syncthreads` follows, and warp 0 ORs the
`BLOCK_WARPS` staged copies into the output.

## <a id="backward"></a>12. The backward — the α-entmax Jacobian action through the union split

<a id="launch-bounds"></a>**No sort and no scan: this is a THREE-REDUCTION elementwise pass.** It shares the
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

<a id="safe"></a>The VJP divides by the amplitude; at a TIE the amplitude is exactly 0, so the guard
(`safe`) substitutes 1 and the support mask zeroes the term.

### Two measured re-read decisions

Both are register-pressure trades. MEASURED under `__launch_bounds__(128, 8)` at
`IPT = 8`:

| staged instead of re-read | measured cost |
|---|---|
| `write_values` / `d_write` across the two passes | 2·`IPT` live registers, **32 B stack frame** (spill) |
| `s` as a third live `IPT`-array | **24 B of spill** |

<a id="localdot"></a><a id="near-line-368"></a>Re-reading is the cheaper trade on a kernel whose DRAM throughput
is only ~30%. The first pass therefore carries NOTHING into the next but two scalars
(`local_dot` and the support mask), and `s` is recomputed from a re-read
`midpoint_values` in the final pass; one extra load per element buys the spill back
to zero.

<a id="near-line-86"></a>The tied backward's second contribution lands through the ACCUMULATING store; bf16 device
atomics are sm_80+, which the closed-world architecture set already is.
