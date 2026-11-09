# `common/atoms/mma.cuh` — the MMA atoms and their operands

Mirrors `csrc/rola/src/common/atoms/mma.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): the MMA's shape and unit, its fragment loads and the register transpose, and the bf16, f16 and int8 atoms.

- <a id="load-frag"></a>`load_frag` / `load_frag_t`: four 8x8 b16 tiles from shared memory in the register order the
  MMA consumes; `_t` transposes each 8x8 so one shared tile serves either handedness. THE `memory` CLOBBER IS PART OF
  THE CONTRACT: `ldmatrix` reads shared memory, and without the clobber the compiler may move it across a barrier, so
  a warp could pull rows another warp has not published.

- <a id="mma"></a>`mma(acc, a, b)`: acc += A(16x16) . B(16x8), bf16 operands, fp32 accumulator private to the issuing
  unit, both operands from its registers -- the ampere family's atom, a warp of 32 its unit, `m16n8k16` the shape
  every fragment index is written in. The hopper family's warpgroup atoms (64 x {8..256} x 16) are not written yet;
  they will be this header's members for that family, chosen where they are defined (`docs/KERNEL_STANDARDS.md` section 8).

- <a id="hmma"></a>`hmma`, `hmma_fresh`, `hmma_k8`, `hmma_k8_f16`, `hmma_f16`, `hmma_f16_fresh`: the fp16 family -- `m16n8k16` f16 into fp32 (`hmma_fresh` into a fresh fragment, C = 0; `hmma_k8` the
  `m16n8k8` atom, half the tensor work, the fold's 8-token tail, `hmma_k8_f16` the same into an fp16 accumulator, the
  readout's den; `hmma_f16` into an fp16 accumulator, two f16 pairs a lane --
  `hmma_f16_fresh` the same into a fresh fragment -- rows `lane / 4` and `+ 8`,
  columns `2 (lane % 4)` and `+ 1` -- at twice GeForce's fp32-accumulate rate). Every rounding is explicit so
  `tests/oracle/fold_reference.py` can mirror it; the f16 pair arithmetic beside it is [`math.md#hmul2`](math.md#hmul2).

- <a id="imma-u8"></a>`imma_u8_k32`, `imma_u8_k32_fresh`, `imma_u8s8_k32`: `mma.sync.m16n8k32` over u8 and s8 into
  s32, 4096 multiply-adds. A is four registers a lane (rows `lane / 4` and `+ 8`, four consecutive k a register at
  `4 (lane % 4)` and `+ 16`), B two (column `lane / 4`), the accumulator four s32 -- exactly the register image
  `ldmatrix` gives of two 16-byte rows a lane, so a token-major row of digits is an operand with no shuffle. `u8.u8`
  is digits against digits, `u8.s8` unsigned weights against signed values; `imma_u8_k32_fresh` takes C from the
  zero register so the accumulator is not live before a sum's first step. On sm_86 one issues every 16 cycles a
  scheduler (8 on sm_80),
  four times the bf16 fp32-accumulate rate (probe of 2026-09-28).

- <a id="transpose-frag-b16"></a>`transpose_frag_b16`: an 8x8 b16 tile transposed where it lies
  ([below](#reg-transpose)).

## <a id="reg-transpose"></a>The register transpose

`transpose_frag_b16(x)` transposes an 8x8 b16 tile where it already lies, across
the issuing unit's own registers, for a contraction whose `k` axis is the axis an
accumulator carries on its `m` side (no shipped caller). The contract is about MAPS,
not about data.

For `m16n8k16`, the accumulator spreads its two axes over the lanes as
`(m = t/4 [+8], n = (t%4)*2 + {0,1})` and a `B` operand spreads its two as
`(k = (t%4)*2 + {0,1} [+8], n = t/4)`. The two are the same matrix through a
transposed lane map, so an accumulator can be an operand of a contraction over its
own `m` axis only through this operation.

**MEASURED, not read off the ISA.** A standalone probe loaded a known 16x8 matrix
into a thread's registers using the accumulator map, fed it as `B` against
`A` the 16 x 16 identity, and read the mma's own output back:

| candidate | verdict |
|---|---|
| the accumulator map already IS the `B` map | REFUTED: 112 of 128 elements differ, and the difference is exactly the transpose |
| `movmatrix.sync.aligned.m8n8.trans.b16` | 0 of 128 differ. Two instructions per (16 `d_v` x 8 slot) tile |
| a shared-memory round trip through the padded buffers | not needed, not built |

The capability (`reg_transpose_b16`, an INSTRUCTION fact the atom asserts) has no slower
form beside it, because the alternative is not a slower implementation of the same
operation — it is a different algorithm, with a `BCg x d_v` staging buffer that
does not fit beside the ring.

The SASS gate needs no carve-out for it: `MOVM` is not one of the signatures
`tools/sass_gate.py` reds on (local memory, convergence subroutines, predicated
HMMA, instructions-per-HMMA).

