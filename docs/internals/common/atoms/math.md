# `common/atoms/math.cuh` — the elementwise math

Mirrors `csrc/rola/src/common/atoms/math.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): the special-function unit's roots and powers, byte and halfword selects and packs, the f16 and bf16 pair arithmetic.

- <a id="sqrt-approx"></a>`sqrt_approx`: `sqrt.approx.f32`, one special-function instruction. The IEEE root (`sqrtf`)
  compiles to a range check and a call to a slow-path subroutine, whose calling convention spills a body that holds
  its registers full (the producer's register-resident threshold, at 252 live); the approximate root sits in the same
  error class as the `__fdividef` beside it.

- <a id="exp2-approx"></a>`exp2_approx`: `ex2.approx.ftz.f32`, one special-function instruction: `2^x` within two
  units in the last place, a denormal result flushed to zero, `2^-inf = 0` (the softmax's dead columns).

- <a id="prmt"></a>`prmt`, `pack_byte4`: `prmt.b32` picks four of two words' eight bytes by a selector nibble each;
  `pack_byte4` takes the same byte of four words with three of them (pair selectors `0x0040`, `0x0051`, `0x0062`, then
  `0x5410`), which is how an int8 operand row is formed from four gathered words without shifts or masks.

- <a id="hmul2"></a>`hmul2`, `hfma2`, `pack_half2`: two f16 products or fused products each rounded once to nearest,
  two fp32 to an f16 pair rounded to nearest (`cvt.rn.f16x2.f32`, `lo` in the low half). Every rounding is explicit so
  a reference can mirror it.
- <a id="hmul2-by"></a>`hmul2_by`: an f16 pair times one half, each product rounded once (`__hmul2_rn` on the half
  broadcast): ptxas folds the broadcast into HMUL2's operand selector, its price in
  [budgets.md#prices](../../tools/budgets.md#prices).

The bf16 pair helpers, `hi_bf16` through `or_run`, have no shipped caller.

- <a id="hi-bf16"></a>`hi_bf16`, `hi_bf16x2`: an fp32's top half IS its bf16 truncation, so a register pair becomes an
  operand with one `prmt.b32` (selector `0x7632`) against `pack_bf16x2`'s conversion. It TRUNCATES where
  `pack_bf16x2` rounds to nearest even: one bf16 ulp, biased toward zero, a declared rounding an oracle mirror must
  carry. <a id="lo-bf16x2"></a>`lo_bf16x2` packs the two LOW halves (`prmt` 0x5410), and
  <a id="pack-lo"></a>`pack_lo` / `join_hi_lo` split a word into hi and lo and rejoin it bit for bit.

- <a id="splat-bf16x2"></a>`splat_bf16x2` puts one bf16 in both halves so a scalar is an operand of `mul_bf16x2`;
  <a id="splat-bf16x2-bits"></a>`splat_bf16x2_bits` does it from a raw bit pattern, because a register array must
  never be address-taken (one `&run[i]` puts the whole run in local memory).

- <a id="mul-bf16x2"></a>`mul_bf16x2`: the elementwise product of two packed bf16 pairs, rounded to nearest even. The
  PTX ISA gates `mul.rn.bf16x2` at sm_90 and `fma.rn.bf16x2` at sm_80, so on Ampere the product is one HFMA2 with a
  <a id="knegzeropair"></a>`kNegZeroPair` addend: `a*b + (-0.0)` rounds the exact product once and keeps a signed
  zero, where a `+0.0` addend would turn every `-0` into `+0`. There is no fp32 route: every tabulated row carries
  `bf16_simd_fma`, and an untabulated architecture fails the build.

- <a id="or-run"></a>`or_run<N>` with <a id="kbf16magnitudemask"></a>`kBf16MagnitudeMask` and
  <a id="koctets"></a>`kOctets`: the OR of a contiguous, naturally aligned run at the widest access that covers it;
  the run is entirely zero exactly when the OR has no magnitude bit set, since bf16's only zeros are +0 and -0 and OR
  cannot clear a set bit. `kOctets` names `N / 8` for the unroll-discipline lint.
