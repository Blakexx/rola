#pragma once

// THE STATE PAGE: 16 leaves as [16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32].
// See docs/internals/common/state_page.md

#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"

namespace rola::state_page {

//: The page's byte geometry: a leaf-channel is the s24 `hi << 8 | lo` at its group's exponent. -- see docs/internals/common/state_page.md#geometry
template <int DV>
struct Blocks {
  static constexpr int kRows = 16;
  static constexpr int kGroups = DV / 32;
  static constexpr uint32_t kHi = 0;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kExp = kLo + (uint32_t)(kRows * DV);
  static constexpr uint32_t kMass = (kExp + (uint32_t)(kRows * kGroups) + 15u) & ~15u;
  static constexpr uint32_t kBytes = kMass + (uint32_t)(kRows * 4);
  static_assert(DV % 32 == 0, "a channel group is 32 channels");
};

//: The s24 mantissa's bound and the smallest group exponent. -- see docs/internals/common/state_page.md#codec
constexpr int kMantissaMax = (1 << 23) - 1;
constexpr int kExpMin = -126;

//: `2^e` for `kExpMin <= e <= 127`, as the float's exponent field.
__device__ __forceinline__ float pow2(int e) { return __int_as_float((e + 127) << 23); }

//: The exponent a group stores at, from its largest magnitude: `frexp`'s exponent less 23, from the float's field. -- see docs/internals/common/state_page.md#codec
__device__ __forceinline__ int group_exponent(float amax) {
  return amax == 0.0f ? 0 : max((__float_as_int(amax) >> 23) - 149, kExpMin);
}

//: `v`'s s24 mantissa at exponent `e`: one rounding, held inside s24.
__device__ __forceinline__ int mantissa(float v, int e) {
  const int m = __float2int_rn(v * pow2(-e));
  return min(max(m, -kMantissaMax), kMantissaMax);
}

//: A lane's word `i` of its eight channels' planes. -- see docs/internals/common/state_page.md#lane-eight
__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

__device__ __forceinline__ uint32_t word(const uint2& v, int i) { return i == 0 ? v.x : v.y; }

//: Channel `c`'s s24 mantissa `hi << 8 | lo`, sign-extended by the permute's sign replication.
__device__ __forceinline__ int mantissa8(const uint4& hi, const uint2& lo, int c) {
  const uint32_t hb = 4u + 2u * static_cast<uint32_t>(c & 1);
  const uint32_t sel =
      static_cast<uint32_t>(c & 3) | (hb << 4) | ((hb + 1u) << 8) | ((8u | (hb + 1u)) << 12);
  return static_cast<int>(denseref::ops::prmt(word(lo, c >> 2), word(hi, c >> 1), sel));
}

//: Channel `c`'s s16 from the hi plane alone.
__device__ __forceinline__ int hi8(const uint4& hi, int c) {
  const int h = static_cast<int>(word(hi, c >> 1));
  return (c & 1) ? (h >> 16) : ((h << 16) >> 16);
}

//: Eight mantissas as their stored hi words: bytes 1-2 of each, two a word.
__device__ __forceinline__ uint4 pack_hi8(const int (&m)[8]) {
  using denseref::ops::prmt;
  return make_uint4(prmt(m[0], m[1], 0x6521u), prmt(m[2], m[3], 0x6521u), prmt(m[4], m[5], 0x6521u),
                    prmt(m[6], m[7], 0x6521u));
}

//: Eight mantissas as their stored lo bytes: byte 0 of each, four a word.
__device__ __forceinline__ uint2 pack_lo8(const int (&m)[8]) {
  using denseref::ops::pack_byte4;
  return make_uint2(pack_byte4(m[0], m[1], m[2], m[3], 0x0040u),
                    pack_byte4(m[4], m[5], m[6], m[7], 0x0040u));
}

//: THE UNPORTED CARRY'S PAGE, split bf16 planes; the carry port deletes it. -- see docs/internals/common/state_page.md#split
template <int DV>
struct SplitBlocks {
  static constexpr int kRows = 16;
  static constexpr int kPairs = kRows * DV / 2;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kMassHi = (uint32_t)(2 * kRows * DV * 2);
  static constexpr uint32_t kMassLo = kMassHi + (uint32_t)(kRows * 2);
  static constexpr uint32_t kBytes = kMassLo + (uint32_t)(kRows * 2);
  static_assert(DV % 2 == 0, "a 32-bit access carries a PAIR of value columns");
};

}  // namespace rola::state_page
