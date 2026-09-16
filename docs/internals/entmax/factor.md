# `csrc/rola/src/entmax/factor.{cuh,cu}` — the producer's independent (per-side) solves

Hand CUDA over CUB warp primitives: the four kernels that turn one side's routing
LOGITS into route VALUES, for the two activations a level can carry on its own —
the exact α=1.5/2.0 threshold solve (with §4.1's packed per-leaf support bitmap)
and softmax. `factor.cuh` is the host declaration set, `factor.cu` the
implementation and its dispatch.

Design of record: this document. The union split's own solve, and the warp-row
vocabulary this file shares with it, are [`entmax.md`](entmax.md).

Gates: `tests/integration/test_production_levels.py` (the fp64 oracle, forward AND
VJP, across `tied` / `mixed_per_level` / `dense_only`),
`tests/unit/test_entmax_production.py` (the masked `production_factor` surface, the
support word's bit-31 and tail-word contracts, the tile-major arena),
tests/integration/test_execution_identity_gates.py, deleted with the tiled consumer (which routing family reached
which kernel).

Symbols are `entmax.md`'s: a *stream* is one (batch, head)-like routing stream, a
*token* one row of the solve, `width` the level's branch width with `col` indexing
it, `LW` the logical warp width, `IPT` the items per lane (`W_PAD = LW * IPT`),
`tau` the threshold, *support* the set of columns above it.

## <a id="logit-dtype"></a>
## The logit plane's dtype is a template axis

`LogitT` is `float` or `__nv_bfloat16`, and every kernel in this family widens a logit to
fp32 **in register** before it does any arithmetic. The solve is unchanged; what changed
is the plane it reads.

**Why bf16 is the shipped one.** The router GEMM runs bf16 operands on the tensor cores,
where cuBLAS's output is bf16 — there is no bf16-operand / fp32-output GEMM to ask for.
Measured on GA102 at the pinned cell (`B=2, T=2048, hidden=512, H=8, C=257`): fp32 SIMT
0.4516 ms against bf16 tensor core 0.1638 ms, **2.76x**, and the class ratio on a clean
8192³ GEMM is 69.9 against 23.9 TFLOP/s. Widening the plane back to fp32 before the
solve would cost ~0.055 ms and one launch, which is why the widening is a register
operation inside these kernels instead of a pass over memory.

**What it does NOT relax.** The support-exactness bar binds the SOLVE, and the solve
still runs in fp32 on whatever it is handed. What the bf16 plane changes is the LOGITS,
and that change is gated at the seam: the reference solve is fed the production logits
and the support sets must agree bitwise
(`tests/integration/test_production_levels.py::test_the_solve_seam_is_exact_on_the_production_logits`).
The support difference against a wholly-fp64 projection is purpose-2 accounting and is
reported, not gated.

**The gradient rides the same plane.** `d_logits` is `LogitT` too — autograd requires a
cotangent in the input's dtype, and a fp32 gradient plane feeding a bf16 GEMM backward
would be a widening with nothing on the other end of it. The tied backward's accumulating
store is `add_value`, which is a bf16 device atomic on the closed-world arch set (sm_80+).

<a id="provenance"></a>1. Where these came from, and what that constrains

<!-- ABSENT: _production_fwd_kernel _production_bwd_kernel -->
These four were Triton (`rola/routing/entmax/production.py`'s
`_production_fwd_kernel`, `_production_bwd_kernel` and their softmax pair) until
P73 S5. The port is why `rola/` now imports no Triton, why `pyproject.toml` no
longer depends on it, and why `build.md`'s "not `fla`, not `triton` — the kernels
are CUDA" is true rather than aspirational. Decisively, it is what lets P65 train
through this path: a Triton backward in the training path is the exact thing the
CUDA-engine-first ruling forbids.

**The port is NOT judged against Triton.** The two lowerings associate the prefix
scan differently — Triton scans the padded row with `tl.cumsum`, this one
accumulates serially within a lane and then runs one `WarpScan` across lanes — so
the fp32 sums are not bitwise equal and a kernel-vs-kernel gate would be measuring
the wrong thing (the matching-the-naive antipattern). The governing gate is the
fp64 reference, and it asserts the SUPPORT SET identical, not merely close.

## <a id="the-solve"></a>2. The forward: the union solve's phases B/C/D, fed one tensor

`factor_forward_kernel` is `union_forward_kernel` with the union split removed. Its
phases are that kernel's, unchanged and documented there
([forward-phases](entmax.md#forward-phases)): row max/min, the finite row-relative
sentinel that pads without putting an infinity in the scan, `WarpMergeSort`
descending, the `(Σz, Σz²)` `WarpScan`, the packed-key `Max` reduction that selects
the greatest valid `k` and carries its `tau` in the same 64-bit word, and the
amplitude `((α−1)(z − tau))_+` — squared at α=1.5.

What is NOT there: phase A's midpoint fold and phase E's write normalization. What
IS there and is not in the union kernel: the candidate mask.

### <a id="mask"></a>2.1 The mask, and why it is a per-ROW quantity

`production_factor` is a public surface and takes an optional boolean candidate
mask, laid out like the logits. A masked element is not a candidate: it leaves the
support by construction rather than by a correction applied afterwards.

The consequence that shapes the kernel is that the member count is then a
reduction, not `width`:

* The row max and min are taken over the CANDIDATES, and a fully-masked row has
  neither — `row_max` is defined to 0 there and the sentinel to −1, so the sort and
  the scan still run on finite values;
* The greatest-valid-`k` search runs over `rank < valid_count`, not `rank < width`,
  which is what keeps a masked column from being admitted by a prefix that never
  included it;
* an empty row produces `chosen = −1`, hence `tau = 0`, hence no support and no
  amplitude — the autograd contract
  `test_production_factor_empty_rows_preserve_autograd_contract` pins.

The mask pointer is NULL on every plan-level path (`_launch_factor_backward_into_logits` and the
batched entry), and the count reduction is skipped there — one uniform branch on a
pointer, not a codegen axis, because a mask arm would double the instantiation
matrix to buy a prologue.

### <a id="support"></a>2.2 The support word

Identical to the union solve's, including the two things that are easy to get
wrong: support is the set of POST-CAST bf16 nonzeros and never the pre-quantization
solve ([bf16-support-boundary](entmax.md#bf16-support-boundary)) — which holds even
when this level's values are STORED in fp32, because the word describes what the
consumer will read — and the 32-token word is packed by the epilogue butterfly over
the lanes at or above `LW`
([epilogue-butterfly](entmax.md#epilogue-butterfly)), one CTA owning exactly one
word ([frozen-support-word](entmax.md#frozen-support-word)).

## <a id="backward"></a>3. The backward: a rank-1 correction, support as an INPUT

The α-entmax VJP is `dz = s ⊙ (dp − ⟨s, dp⟩ / Σs)` with `s = p^(2−α)` on support
and 0 off it — `1` at α=2, `√p` at α=1.5. Softmax's is the same shape with `s = p`
and the denominator identically 1, which is why `softmax_backward_kernel` carries no
support word and no `α`.

Two facts make this a five-line kernel rather than a Jacobian:

* **Support is an INPUT**, unpacked from the same frozen word the forward emitted,
  so the two directions cannot disagree about the support set by construction. Off
  support every term is an exact zero by multiplication, which is what
  `assert_off_support_zero` asserts at `rtol=0` — the FORM of the VJP, not a
  magnitude.
* **The two sums are one reduction.** `(Σ s·dp, Σ s)` travel as a `Pair` through a
  single `WarpReduce`, exactly as the union backward's does.

It RE-READS `values` and the cotangents in the second pass rather than staging them
in `IPT`-sized registers. That is the union backward's MEASURED trade
([backward](entmax.md#backward)) and not a stylistic preference: at `IPT = 8` the
staged form is what puts that kernel over its 64-register bound. Re-reading keeps
this family at zero spill across all 96 instantiations, which `tools/ratify.py`
enforces.

`accumulate` selects `atomicAdd` over a store. It is a parameter and not a fact
about the tensor because a TIED level runs two solves into ONE gradient buffer, and
only the second of them may accumulate — a property of the walk, not of the memory.

## <a id="batched-levels"></a>4. One launch across levels

`producer.py` has always claimed "one packed projection and one batched solve", and
until P73 S5 only the first half was true: the solve was a Python `for` loop over
`routing.levels`, one launch and three allocations per level.

The launch shape here makes the claim true. Each entry takes BASE tensors plus
per-level element offsets; `blockIdx.z` indexes a `LevelTable` that travels in the
kernel's parameter space (448 B at `MAX_BATCHED_LEVELS = 16`, against a 4 KB
floor), so batching costs no device table, no H2D copy and no indirection.

**The grouping rule is exactness, not convenience.** Levels may share a launch only
when they share the width CLASS, the α, the operator and the output dtype — that
is, only when they select the same template arm. The class is load-bearing: `LW`
and `IPT` set how the sort permutes and how the prefix scan associates, so running
a width-3 level in a width-64 arm would change the fp32 rounding of `tau` and could
move a support boundary. Widths inside a launch may differ (3 and 4 are one class);
their padded widths may not. What batching changes is the launch SHAPE; the
arithmetic each row performs is the arithmetic the per-level loop performed.

The host `build_table` refuses a mixed-class table rather than padding to the
largest, and the caller (`production.py`) is what groups the plan walk into tables.

## <a id="stored-form"></a>5. The stored form is what the solve WRITES

The routing factors have exactly one consumer -- the chunk operator -- and it reads
`bf16`. So `OutT` is `__nv_bfloat16` on the shipped path and the solve writes the stored
representation directly; there is no fp32 plane and no narrowing pass between them. The
union arm and the independent/softmax arms are both instantiated for it
(`entmax.cu::union_forward`, `factor.cu::check_values`), and the value dtype is read off
the output tensor rather than templated at the host entry, so the choice is one
allocation in `rola/routing/entmax/production.py`.

**Why writing bf16 is not the same as rounding to bf16.** The solve's amplitudes come
from a closed form whose zeros are structural: an entry outside the support is assigned
exactly `0`, and an entry inside it is `((alpha-1)(z-tau))^2` clamped at zero, which is
nonzero whenever the support test passed at fp32. Writing that value narrowed to bf16
preserves both directions of the support test -- `0` stays `0`, and a positive fp32
value below bf16's smallest subnormal cannot arise, because the smallest amplitude the
solve can produce and still call supported is bounded away from it by the threshold
arithmetic. That identity is the self-masking theorem's storage precondition
(`docs/architecture/self-masking-theorem.md`) and it is gated without tolerance by
`tests/integration/test_production_levels.py::test_the_stored_form_preserves_the_support_exactly`
over four stress fixtures x four routing shapes x two alphas.

`stored_dtype=torch.float32` exists for one purpose: a FIDELITY gate that wants the
solve's own arithmetic without the storage quantization on top of it. It is a test-side
argument; no runtime path passes it.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/entmax/factor.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="sort-bytes"></a>
### `SORT_BYTES`

HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias
One per-physical-warp slot serves the row sort's scratch and then the support-word
staging. `RowSort::kSlotBytes` is ZERO for the register network, so at `IPT <= 4` the
slot is the staging alone -- docs/internals/entmax/entmax.md#row-sort

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

<a id="stream32-3"></a>
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

<a id="stream32-4"></a>
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

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/entmax/factor.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/entmax/factor.cu` condensed at their call site into a short pointer.

<a id="launch-bounds"></a>
### `__launch_bounds__`

---------------------------------------------------------------------------
BACKWARD -- the alpha-entmax rank-1 Jacobian action.

No sort and no scan: ONE fused (numerator, denominator) reduction over a support that
is an INPUT, unpacked from the same frozen word the forward emitted, so the two
directions cannot disagree about the support set by construction. Off support every
output term is an exact zero by multiplication.

It RE-READS `values` and the cotangents in the second pass rather than staging them:
the union backward's measured spill trade, and the same `IPT`-sized arrays would apply
here -- docs/internals/entmax/factor.md#backward
---------------------------------------------------------------------------

<a id="validcount"></a>
### `valid_count`

A masked row can be EMPTY, so the member count is a per-row reduction and not
`width`; `production_factor`'s empty-row autograd contract is exactly this case.

<a id="sentinel"></a>
### `sentinel`

A finite row-relative sentinel keeps padded lanes strictly below every real
value without an infinity in the scan. `row_min` is pre-shift, so shift it too.

<a id="packedkey"></a>
### `packed_key`

ONE reduction selects the greatest valid k AND carries its tau: the column in the
high 32 bits makes integer Max pick it, and `best_col = -1` maps to key 0 so an
all-empty row yields tau = 0 -- docs/internals/entmax/entmax.md#forward-phases

<a id="near-line-200"></a>
### near line 200

The second output is the SAME solve, never a second one: a `shares_solve` level
hands read and write one set of amplitudes.

<a id="near-line-204"></a>
### near line 204

HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary
Support is defined at the native bf16 storage boundary, never at the
pre-quantization solve, whatever dtype this level's values are stored in.

<a id="near-line-212"></a>
### near line 212

The butterfly runs over exactly the bits at or above LW, which is what separates
lanes holding the same column: docs/internals/entmax/entmax.md#epilogue-butterfly

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

<a id="nstreams-3"></a>
### `n_streams`

A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`;
a flat plane declares `heads == 1` and the same product is `size(0)`
-- docs/internals/entmax/entmax.md#stream-split

<a id="nstreams-4"></a>
### `n_streams`

A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`;
a flat plane declares `heads == 1` and the same product is `size(0)`
-- docs/internals/entmax/entmax.md#stream-split

<a id="factorforward"></a>
### `factor_forward`

`mask` is a detached boolean candidate mask laid out like `logits`; absent means every
element is a candidate. `values_second` absent means the level stores one side.

<a id="factorbackward"></a>
### `factor_backward`

`accumulate` adds into `d_logits` instead of storing -- what a TIED level's second
side needs, and the reason it is a parameter rather than a fact about the tensor.
