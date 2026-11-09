# `csrc/rola/src/decode/decode_fold.cuh` — the decode step's operand pass

<a id="note-l3"></a>Every per-token operand a decode step consumes, read from the internal format (docs/api.md
section 2.1) as stored -- no level is normalized, on either side ([`docs/CONSTITUTION.md`
R23](../../CONSTITUTION.md#r23)) -- and staged in the SHARED MEMORY of the CTA that is about to consume it.

Symbols: `B` batch, `H` heads, `BH = B * H`, `bh = b * H + h`; `D` levels; `b_l` level
`l`'s width; `d_v` the value width. The operands are the read and write coefficient rows
(`[B, 1, H, b_l]` each, u8 unit-max digits -- a dense level's f16 -- in the producer's own layout) with their
fp32 row maxima (`[B, 1, H]` each; a producer's write gain is a factor of write level 0's) and the token's values
(`[B, 1, H, d_v]` f16 against its largest channel, and the fp32 token scale).

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

<a id="raw"></a>A digit's RAW value rounds nothing — a u8 digit is its integer, a dense level's f16 coefficient times
255 (exact, its floor stored: [`../facts/liveness_contract.md#floor`](../facts/liveness_contract.md#floor)) — and a
level's amplitude is `raw * (scale / 255)`, one fp32 multiply by a per-row factor (255 is `kRowMaxDigit`, the row
max's digit), on either side. `tests/integration/test_decode_fold.py` cell 1 states
that through the step's own output, on strided views as well as contiguous ones.

<a id="fold-digits"></a>**The digit pass** (`fold_digits`) takes every digit of both sides in one pass, a warp a mask
word. A lane loads its digit and its row's fp32 scale together, one round trip; the warp's ballot of the nonzero test
(after the floor) is the digit mask's word ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask));
and the lane stages `raw * (scale / 255)`. Slots past a level's width are not loaded and ballot a zero.

## <a id="the-dtype-arms"></a>3. The storage types are the format's, and the step's template axes are not

The fold reads two INDEPENDENT storage families: the coefficient rows (u8, a dense level's f16; ruled
2026-10-03/04/07) and the values (f16), with fp32 scales. They are independent because the routing bundle and the
value projection are different producers. A level's row type is the bundle's declaration (its dense bit a side,
`floor_read`/`floor_write`), read once a step; a dtype outside the format is a REFUSAL (the bundle's own, then the
host entry's), never a silent widening.

<a id="fold-value"></a>The value is folded once a step into the CTA's resident row: the
channel's f16 unit times the token's `scale`. `load_value` issues the loads before the digit
pass and `value_of` does the arithmetic after it, so the channel's loads ride the same round
trip.

Contrast `DECAY` and `D` (`decode.md#instantiation-matrix`), which ARE template axes: each
picks a different loop shape or a different live-register count in the walk's innermost
path, where a runtime branch would cost a register or an instruction on every iteration
rather than a few hundred bytes once per step.

## <a id="row-sum"></a>4. The decay's row sums

The decay arm's clock is each leaf's SHARE of the token's write, `prod_l k_l[d_l(s)] / sum_d k_l[d]`
([`docs/CONSTITUTION.md` R23](../../CONSTITUTION.md#r23)), so a `DECAY` step sums each write level's staged row
(`sum_staged_rows`: lane `j` of level `l`'s warp accumulates `row[j], row[j+32], ...` in ascending order, then five
`__shfl_down_sync` steps) and a leaf's clock is its write amplitude times `1 / prod_l sum_l` (`share_scale`; 0 for a
token that writes nothing, its clock still). The read side has no sum: it is read as stored.

## <a id="the-gate"></a>5. How it is gated

The fold has no output a test can read — its destination is shared memory the walk consumes and discards.
So `tests/integration/test_decode_fold.py` gates it through the STEP, on a fixture where
the step's whole arithmetic collapses onto one amplitude: `D = 1`, a read level that is
nonzero everywhere and off the simplex, an empty write side, and a state whose rows are zero
except at one leaf `s0`. Then `y[c] = a * e[c] / (a * m + eps)` with `a = raw[s0] * (scale / 255)`
— one multiply per column and one divide, so `y` is a function of the amplitude's last bits
and of nothing else, and the reference computes exactly that.

`rola.ops.decode.fold_levels` is the torch fold those references are built on, kept for
that purpose and called from nowhere on the shipped path — the same arrangement as
`_write_atom_bitmap` and for the same reason: a fold checked against its own output proves
nothing.

## <a id="no-allocation"></a>6. Nothing here allocates, and nothing is left behind

The fold's destinations are shared memory, so there is nothing per-step to allocate, retain
or address-pin. That is what makes it capturable by construction: the phase reads no device
value on the host, and a captured graph replays it with no buffer identity to preserve.

`dials` is the one operand the fold does not touch, and the boundary is principled: it is a
per-HEAD constant of the decay arm rather than a per-token operand, and its fp32 cast
carries a refusal (`rola.ops.decay.leaf_rate_dials`) that the host has to raise.
