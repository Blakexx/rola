// csrc/rola/src/common/atoms/math.cuh -- THE ELEMENTWISE MATH: the special-function unit's roots and powers, byte and
// halfword selects and packs, the f16 and bf16 pair arithmetic; a concern of the baseline operations
// (`common/ops.cuh`). see docs/internals/common/atoms/math.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

namespace denseref {
namespace ops {

//: THE SQUARE ROOT ON THE SPECIAL-FUNCTION UNIT (`sqrt.approx.f32`, one `MUFU`): never the IEEE root's slow-path call,
//: whose register convention spills a body holding its registers full. -- see docs/internals/common/atoms/math.md#sqrt-approx
__device__ __forceinline__ float sqrt_approx(float x) {
  float d;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}

//: TWO TO THE POWER `x` ON THE SPECIAL-FUNCTION UNIT (`ex2.approx.ftz.f32`, one `MUFU`): -inf gives 0.
//: -- see docs/internals/common/atoms/math.md#exp2-approx
__device__ __forceinline__ float exp2_approx(float x) {
  float d;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
  return d;
}

//: FOUR BYTES CHOSEN FROM EIGHT by a selector word: the byte gathers of an int8 operand. -- see docs/internals/common/atoms/math.md#prmt
__device__ __forceinline__ uint32_t prmt(uint32_t lo, uint32_t hi, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(lo), "r"(hi), "r"(sel));
  return d;
}

//: THE LARGER OF EACH HALFWORD PAIR on the fp16 order (`HMNMX2`): for halfwords in `[0x0400, 0x7bff]` -- positive
//: normal fp16 -- the integer order, two lanes an instruction where a byte or halfword integer max is emulated.
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: TWO s32 SATURATED TO u16 AND PACKED, `hi` in the upper half (`I2IP.U16.S32.SAT`).
__device__ __forceinline__ uint32_t pack_sat_u16(int32_t hi, int32_t lo) {
  uint32_t d;
  asm("cvt.pack.sat.u16.s32 %0, %1, %2;" : "=r"(d) : "r"(hi), "r"(lo));
  return d;
}

//: the same byte of four words, packed in order: the operand row a lane holds.
__device__ __forceinline__ uint32_t pack_byte4(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
                                               uint32_t pair_sel) {
  return prmt(prmt(v0, v1, pair_sel), prmt(v2, v3, pair_sel), 0x5410u);
}

//: TWO f16 PRODUCTS, each rounded once to nearest (`HMUL2`).
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: TWO f16 PRODUCTS BY ONE HALF, each rounded once, the broadcast in the multiply's operand selector. -- see
//: docs/internals/common/atoms/math.md#hmul2-by
__device__ __forceinline__ uint32_t hmul2_by(uint32_t a, __half m) {
  const __half2 p = __hmul2_rn(*reinterpret_cast<const __half2*>(&a), __half2half2(m));
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: TWO f16 `a b + c`, each rounded once to nearest (`HFMA2`).
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

//: TWO fp32 TO AN f16 PAIR, each rounded to nearest, `lo` in the low half (`F2FP`).
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  static_assert(arch::kCaps.bf16_cvt_packed,
                "UNSUPPORTED ARCHITECTURE: without a packed fp32->bf16x2 convert "
                "the accumulator cannot become an operand in place");
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&p);
}

__device__ __forceinline__ float pair_lo(uint32_t p) { return __uint_as_float(p << 16); }

__device__ __forceinline__ float pair_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }

//: THE ACCUMULATOR'S TOP HALVES, TAKEN AS A bf16 PAIR. -- see docs/internals/common/atoms/math.md#hi-bf16
__device__ __forceinline__ uint16_t hi_bf16(float v) {
  return (uint16_t)(__float_as_uint(v) >> 16);
}

__device__ __forceinline__ uint32_t hi_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x7632;"
      : "=r"(d)
      : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return d;
}

//: the two floats' LOW halves, packed the same way: the residue under the bf16 plane.
__device__ __forceinline__ uint32_t lo_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x5410;"
      : "=r"(d)
      : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return d;
}

//: a halfword into both halves of a packed pair.
__device__ __forceinline__ uint32_t splat_u16(uint32_t x) { return (x & 0xFFFFu) * 0x10001u; }

//: One bf16 into BOTH halves of a packed pair, so that a scalar can be an -- see docs/internals/common/atoms/math.md#splat-bf16x2
__device__ __forceinline__ uint32_t splat_bf16x2(__nv_bfloat16 x) {
  const __nv_bfloat162 p = __bfloat162bfloat162(x);
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: the same splat over a raw bf16 BIT PATTERN. -- see docs/internals/common/atoms/math.md#splat-bf16x2-bits
__device__ __forceinline__ uint32_t splat_bf16x2_bits(unsigned short x) {
  return ((uint32_t)x << 16) | (uint32_t)x;
}

//: one raw bf16 bit pattern as a float, by value, for the same reason.
__device__ __forceinline__ float bf16_bits_to_float(unsigned short x) {
  const uint32_t w = (uint32_t)x << 16;
  return *reinterpret_cast<const float*>(&w);
}

//: THE ARCHETYPAL INSTRUCTION-SELECTED ENTRY. Every implementation satisfies ONE contract: -- see docs/internals/common/atoms/math.md#mul-bf16x2
template <bool Simd = arch::kCaps.bf16_simd_arith, bool Fma = arch::kCaps.bf16_simd_fma>
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  static_assert(Simd <= arch::kCaps.bf16_simd_arith,
                "mul.rn.bf16x2 requires .target sm_90 or higher; this "
                "architecture's row says so and ptxas agrees");
  static_assert(Fma <= arch::kCaps.bf16_simd_fma,
                "fma.rn.bf16x2 requires .target sm_80 or higher; this "
                "architecture's row says so and ptxas agrees");
  if constexpr (Simd) {
    uint32_t d;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
  } else if constexpr (Fma) {
    //: the packed pair of bf16 NEGATIVE ZEROS, which is the sign-preserving -- see docs/internals/common/atoms/math.md#knegzeropair
    constexpr uint32_t kNegZeroPair = 0x80008000u;
    uint32_t d;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(kNegZeroPair));
    return d;
  } else {
    static_assert(Simd || Fma,
                  "a packed bf16 product needs mul.rn.bf16x2 (sm_90) or "
                  "fma.rn.bf16x2 (sm_80); every tabulated row carries one");
    return 0u;
  }
}

//: THE STATE'S LO HALVES, PACKED. The top 16 bits of an fp32 ARE its bf16 -- see docs/internals/common/atoms/math.md#pack-lo
__device__ __forceinline__ uint32_t pack_lo(float a, float b) {
  return (__float_as_uint(a) & 0xFFFFu) | (__float_as_uint(b) << 16);
}

__device__ __forceinline__ float join_hi_lo(uint32_t hi, uint32_t lo, int k) {
  return __uint_as_float(k ? ((hi & 0xFFFF0000u) | (lo >> 16)) : ((hi << 16) | (lo & 0xFFFFu)));
}

//: The magnitude bits of a packed bf16 pair.  A run of bf16 is entirely zero -- see docs/internals/common/atoms/math.md#kbf16magnitudemask
constexpr uint32_t kBf16MagnitudeMask = 0x7FFF7FFFu;

//: THE OR OF A CONTIGUOUS RUN, AT THE WIDEST ACCESS THAT COVERS IT. -- see docs/internals/common/atoms/math.md#or-run
template <int N>
__device__ __forceinline__ uint32_t or_run(const __nv_bfloat16* p) {
  static_assert(N == 1 || N == 2 || N == 4 || (N >= 8 && N % 8 == 0),
                "a run is one access of 2, 4, 8 or 16 bytes, or a whole number "
                "of 16-byte accesses");
  if constexpr (N == 1) {
    return (uint32_t)*reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (N == 2) {
    return *reinterpret_cast<const uint32_t*>(p);
  } else if constexpr (N == 4) {
    const uint2 x = *reinterpret_cast<const uint2*>(p);
    return x.x | x.y;
  } else {
    uint32_t acc = 0;
    //: §6: `N` is this function's template parameter -- see docs/internals/common/atoms/math.md#koctets
    constexpr int kOctets = N / 8;
#pragma unroll
    for (int i = 0; i < kOctets; ++i) {
      const uint4 x = *reinterpret_cast<const uint4*>(p + i * 8);
      acc |= x.x | x.y | x.z | x.w;
    }
    return acc;
  }
}

}  // namespace ops
}  // namespace denseref
