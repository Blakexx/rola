# `csrc/rola/src/decode/decode_fold.cuh` — the decode step's operand pass

Every per-token operand a decode step consumes, widened, normalized where the producer did
not already leave the level on the simplex, and staged in the SHARED MEMORY of the CTA
that is about to consume it.

Symbols: `B` batch, `H` heads, `BH = B * H`, `bh = b * H + h`; `D` levels; `b_l` level
`l`'s width; `d_v` the value width. The operands are the read and write amplitude rows
(`[B, 1, H, b_l]` each, in the producer's own layout), the write gain (`[B, 1, H]`) and the
token's value (`[B, 1, H, d_v]`).

---

## <a id="why-a-header"></a>1. Why the fold is not a launch

The operands are TINY — `2 * sum_l b_l + d_v + 1` values per batch-head, kilobytes at any
shipped topology. The fold was never about moving bytes faster; it is about not being a
launch.

A decode step is launch-bound. Measured on an RTX 3080 Ti at `B = 2`, widths `(16, 16)`,
`d_v = 64`, paged backing: the torch operand chain was eight launches and 15.8 µs of a 39.1
µs device chain, one CUDA kernel replaced them at 29.9 µs, and folding that kernel into the
step took the step to 24.6 µs on 2 launches. Each step of that is the same argument — the
work was always the consuming CTA's own business, and every boundary drawn around it cost
more than the work.

So the fold is a HEADER. The step folds its own operands, the growth probe folds its own
write side, and between them there is no launch, no plane and no ordering constraint.

## <a id="the-identity"></a>2. The fold is an address, not a copy pass

For a CONTIGUOUS operand the fold is not even that: `[B, 1, H, W]` contiguous has element
`(b, 0, h, w)` at `b*H*W + h*W + w`, and `[BH, W]` with `bh = b*H + h` has it at
`(b*H + h)*W + w` — the same offset. So the "layout copies" the torch path issued were
never transpositions; they were `.float()` casts that had to allocate a destination.

A routing level is not generally contiguous, though: the producer hands out VIEWS of one
packed plane, and `.contiguous()` on the way in would be exactly the launch this design
exists to delete. So each operand arrives as a `Span` — its base and its `(b, h, w)`
strides — and the fold is the address the kernel computes, `bh -> (b, h)` by the consumer's
own convention.

What the entry refuses is `T != 1`: the fold is defined on one token and a second one would
be silently dropped.

The widening rounds nothing — bf16 and fp32 both embed exactly in fp32 — so a cast-only
operand reaches the walk with the bytes the torch fold would have produced, and
`tests/integration/test_decode_fold.py` cell 1 states that through the step's own output,
on strided views as well as contiguous ones.

## <a id="the-dtype-arms"></a>3. The dtype arms are runtime, and the step's template axes are not

The fold reads three INDEPENDENT storage families: the routing levels, the write gain and
the value, each `float` or `bfloat16` — independent because the routing bundle and the
value projection are different producers, so a bundle carrying bf16 levels beside an fp32
gain is the ordinary case rather than an oddity.

Making them template axes would multiply the step kernel's instantiation matrix by EIGHT,
for a few hundred bytes of gather in a prologue. So `DecodeParams` carries three flags and
each gather site picks a fully typed loop from them:

```
if (p.levels_bf16) fold_side<__nv_bfloat16>(...); else fold_side<float>(...);
```

The branch is taken once per SIDE, not once per element, and both arms are complete — this
is a dispatch, not a fast path with a fallback. A dtype outside the pair is a REFUSAL at
the host entry, never a silent widening.

Contrast `DECAY` and `D` (`decode.md#instantiation-matrix`), which ARE template axes: each
picks a different loop shape or a different live-register count in the walk's innermost
path, where a runtime branch would cost a register or an instruction on every iteration
rather than a few hundred bytes once per step.

## <a id="declared-order"></a>4. The normalizing sum, and its DECLARED order

A read level the producer did not already leave on the simplex still carries its own mass,
and the decode arm is defined on the simplex, so that level is divided by its row sum
(`RouteFactors.read_needs_normalization` is the predicate, stated once and read by both the
step's `normalize` argument and `RouteFactors.read_simplex`). The write side is never
normalized, whatever the read side's flags say.

<!-- ABSENT: reduce_kernel -->
A sum has an ORDER, and floating-point addition is not associative. The torch reduction
this replaced had whatever order that release's `reduce_kernel` chose — an order no gate can
hold anything to, because it is not written down anywhere. This one is:

> **The declared order.** Lane `j` of the level's warp accumulates
> `row[j], row[j+32], row[j+64], …` into an fp32 register in ASCENDING index order.
> The 32 lane partials are then folded by five `__shfl_down_sync` steps with offsets
> `16, 8, 4, 2, 1`, each adding lane `j + off` into lane `j`. Lane 0 holds the sum.
> Lanes past the row's width contribute an exact `0.0f`.

That is a total order over the additions, reproducible on any device.

<a id="the-gate"></a>**How it is gated now that nothing publishes it.** The fold has no
output a test can read — its destination is shared memory the walk consumes and discards.
So `tests/integration/test_decode_fold.py` gates it through the STEP, on a fixture where
the step's whole arithmetic collapses onto the divisor: `D = 1`, a read level that is
nonzero everywhere, an empty write side, and a state whose rows are zero except at one leaf
`s0`. Then `y[c] = a * e[c] / (a * m + eps)` with `a = row[s0] / row_sum(row)` — one
multiply per column and one divide, so `y` is a function of the divisor's last bits and of
nothing else. The reference computes exactly that from a torch reimplementation of the
declared order.

Two cells are the declaration's teeth: one shows the declared order and a plain ascending
one DISAGREE on a row where summation order is observable, and one puts that row through
the step and shows `y` follows the declared order rather than the ascending one. So the
gate is answering a question about order and not merely about arithmetic in general.

`rola.ops.decode.fold_levels` is the torch fold those references are built on, kept for
that purpose and called from nowhere on the shipped path — the same arrangement as
`_write_atom_bitmap` and for the same reason: a fold checked against its own output proves
nothing.

## <a id="no-allocation"></a>5. Nothing here allocates, and nothing is left behind

The fold's destinations are shared memory, so there is nothing per-step to allocate, retain
or address-pin. That is what makes it capturable by construction: the phase reads no device
value on the host, and a captured graph replays it with no buffer identity to preserve.

`dials` is the one operand the fold does not touch, and the boundary is principled: it is a
per-HEAD constant of the decay arm rather than a per-token operand, and its fp32 cast
carries a refusal (`rola.ops.decay.leaf_rate_dials`) that the host has to raise.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode_fold.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="row-sum"></a>
### `row_sum`

THE DECLARED SUM. Lane `j` accumulates `row[j], row[j + 32], ...` in ascending index
order into an fp32 register, then the five `__shfl_down_sync` steps fold lane `j`
into lane `j - off`. Lanes past the row's width contribute an exact `0.0f`.
docs/internals/decode/decode_fold.md#declared-order

<a id="fold-row-sums"></a>
### `fold_row_sums`

THE ROW SUMS, for a side that carries its own mass. Warp `l < D` owns level `l`'s sum,
which is why `D <= kMaxLevels <= kDecodeWarps` is the condition that makes ONE barrier
after this enough. `s_sum` is `[kMaxLevels]` of the caller's shared memory.
NO BARRIER IS TAKEN HERE -- the caller places it, because a device function that
synchronizes on entry and exit costs a barrier at every seam between two of them.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode_fold.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE DECODE FOLD -- every per-token operand widened, normalized and staged in SHARED
MEMORY by the CTA that is about to consume it.

Three facts a reader must not miss:
  1. THE FOLD IS A GATHER, NOT A COPY PASS. Its inputs are whatever VIEW the producer
     handed out -- a level is ordinarily a slice of one packed plane -- so the fold
     indexes by stride and the `[B, 1, H, W] -> [B * H, W]` fold is the address it
     computes, never a materialized intermediate.
  2. THE SUM'S ORDER IS DECLARED, not incidental: lane-strided ascending partials
     folded by a 5-step `__shfl_down_sync` tree. A reduction whose order is a
     property of whatever the library did that release is a reduction no gate can
     hold to bit-identity.
  3. IT IS A HEADER, NOT A KERNEL. The step folds its own operands and the growth
     probe folds its own write side; between them there is no launch and no plane.

The declared order, the dtype arms and the cast argument:
docs/internals/decode/decode_fold.md

<a id="near-line-53"></a>
### near line 53

ONE SIDE'S `D` LEVELS into the flat `[sum_l b_l]` shared staging the walk indexes by
digit. `normalize` is `nullptr` for a side that never carries its own mass (the write
side, always); otherwise `s_sum` must already hold `fold_row_sums`' answer.
