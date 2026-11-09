# `csrc/rola/src/scale/scale.cuh` / `scale.cu` — the scale pass's constants, operand block and translation unit

Mirror doc for the header the scale sources include and the unit that validates a call and launches the three kernels
([`scale_kernel.md`](scale_kernel.md); the host entry and its refusals [`scale_api.md`](scale_api.md)). The pass gives
each token the scale byte of the carry's f16 sum across CTAs ([`carry/carry_api.md#carry-reduce`](../carry/carry_api.md#carry-reduce)):
the power of two every CTA's partials of the token meet at, so its f16 pairs add without leaving f16's range. It
replaces the den pass's bound (ruled 2026-11-02: the den moves into the carry, a write-aligned scale pass gives the
scale; its bound the read-row form, ruled the same day).

THE BOUND, a token `t` of window `w`:

    P(t) = (sum_r x_R(t, r)) (sum_a x_S(t, a) G_w(a)) / carry_factor
    G_w(a) = sum over the windows before w of  w_S(t', a) max_r w_R(t', r) F(t')

`x` and `w` are the stored unit rows (a u8 digit over 255; a dense level's f16 at least the floor 2^-13, as the bundle's
dense zeros read), `F = m s` the facts pass's factor product (`facts/facts.md#columns`: the write gain, the write levels'
row scales and the value scale), the reader's per-token row scales left out as every kernel leaves them, and
`carry_factor` the carry's partials' units (`rola.ops.prefill.carry_factor`). Since `w_R(r) <= max_r w_R` and `s`
bounds every channel of a value, `P` bounds every channel of the carry's `num` for the token; the byte is the carry's
`unit_byte(P, 0)` (`carry/carry.cuh`: the least `u` with `1.02 P <= 2^(u + 15)`), `kUnitByteNone` where `P` is zero.
Level S is the carry's (`carry_level_s`), level 1 where it has none; the bound holds for either choice.
MEASURED (`tools/scale_slack.py`, sm_86, 2026-11-02): its slack over the exact read product reaches 2^14.0 at the layer's
gain 16 (L 64K x 4 heads), median 2^9.2; the carry's f16 sum at it differs from the exact bound's run to run only.

<a id="constants"></a>

## The constants

THE PLAN at a build, `ScalePlan<B>`: its window (`traits_of(B).window`, whole groups; the kernels read it as a
constant, and a write CTA holds the window's token factors), a CTA's `kWarps` = 8 warps (`kThreads` = 256) and a scan thread's `kBatch`
= 8 windows' loads in flight; each kernel takes its build as a type and reads them there. The grains below are the
pass's own, the same at every build. A row is read `kChunk` = 256 digits at a time, `kLaneDigits` = 8 a lane (an 8-byte load of
a u8 row, a 16-byte load of an f16 row); a warp takes its tokens in groups of `kGroup` = 32, `kAhead` = 4 tokens' rows in
flight before their arithmetic (the loads a stage ahead, KERNEL_STANDARDS §29). `kInvDigit` and `kFloor` are the facts
contract's (`facts/liveness_contract.md#floor`).

<a id="params"></a>

## The operand block

`ScaleParams`: `write_s`, `write_r`, `read_s`, `read_r` each side's rows at the two levels as the carry's `Plane`s (base
and byte strides, `[B, T, H, width]`; the read side's heads the query heads); `fm` the facts pass's `(F, m)` a token,
`[BH, L]` float2; `mass` the workspace `[BH, windows, width_S]` fp32 -- each window's own write masses after the first
kernel, their exclusive prefix after the second; `bytes` the output `[BH heads, windows W]` int8 (the window grid whole, as the carry reads it: a partial window's tail `kUnitByteNone`); `grid` its row length; `inv_factor` the reciprocal of
the carry's factor; `L`, `windows`, `heads` (query heads a write head), `write_heads`,
`width_s`, `width_r` the call's extents.

THE TRANSLATION UNIT (`scale.cu`): `scale_forward` checks the planes (`plane_of`: each level's rows u8 or f16, 16-byte
aligned) and the facts pass's `fm`; at the card's build (`run<B>`) it allocates the workspace and the output, then launches `scale_writes` over `(windows, BH)`,
`scale_scan` over `(width_S / 256, BH)` and `scale_reads` over `(windows, BH heads)` with level S's masses in dynamic
shared memory, each kernel's form by its side's two dtypes (a sparse level u8, a dense one f16: four forms a side).

Per architecture: sm_80 and sm_86 the same form (bandwidth-bound: each row read once); sm_90 and sm_100 take each
group's rows by TMA bulk copies into shared memory in place of the lanes' loads, the three units unchanged. Scaling: O(L
(b_S + b_R)) bytes a head on each side, independent of the carry's box; the scan O(windows b_S).
