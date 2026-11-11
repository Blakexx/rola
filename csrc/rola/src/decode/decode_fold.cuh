#pragma once

// THE DECODE FOLD -- every per-token operand, read from the internal format as stored and staged in
// SHARED MEMORY by the CTA that is about to consume it.
// see docs/internals/decode/decode_fold.md#note-l3

#include <cuda_bf16.h>

#include <cstdint>

#include "facts/liveness_contract.cuh"
#include "decode.cuh"

namespace rola {
namespace decode {

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

__device__ __forceinline__ float scale_of(const Span& s, int b, int h) {
  return *row_of<float>(s, b, h);
}

//: THE DIGIT PASS: every digit of both sides, a thread's slots' loads issued before any is used, staged and -- see docs/internals/decode/decode_fold.md#fold-digits
template <class Dp, int D>
__device__ __forceinline__ void fold_digits(const DecodeParams& p, int b, int h, float* sa_r,
                                            float* sa_w, uint32_t* mask_r, uint32_t* mask_w) {
  //: a side is at most `kMaxLevels` levels of `kDecodeMaxLevelWidth` digits, so a thread holds at most this many slots.
  constexpr int kSlots = (2 * kMaxLevels * kDecodeMaxLevelWidth) / Dp::kThreads;
  const int lane = threadIdx.x & 31;
  const int slots = p.mask_total << 5;
  float x[kSlots];
  float scale[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = threadIdx.x + j * Dp::kThreads;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int digit = ((word - p.mask_off[l]) << 5) + lane;
    const bool valid = s < 2 * slots && digit < p.level_width[l];
    const Span& rows = write_side ? p.write[l] : p.read[l];
    //: on the byte scale: a u8 digit, at every level
    x[j] = valid ? static_cast<float>(row_of<uint8_t>(rows, b, h)[digit * rows.sw]) : 0.0f;
    scale[j] = valid ? scale_of(write_side ? p.write_scale[l] : p.read_scale[l], b, h) : 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = threadIdx.x + j * Dp::kThreads;
    if (s >= 2 * slots) break;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int digit = ((word - p.mask_off[l]) << 5) + lane;
    const uint32_t live = __ballot_sync(0xffffffffu, x[j] != 0.0f);
    if (lane == 0) (write_side ? mask_w : mask_r)[word] = live;
    if (digit < p.level_width[l])
      (write_side ? sa_w : sa_r)[p.level_amp_offset[l] + digit] =
          x[j] * (scale[j] / facts::kRowMaxDigit);
  }
}

//: A WRITE LEVEL'S ROW SUM, from its staging: the decay's share divides by their product. Warp `first + l` owns level
//: `l`. -- see docs/internals/decode/decode_fold.md#row-sum
__device__ __forceinline__ void sum_staged_rows(const DecodeParams& p, const float* sa_w,
                                                float* s_sum, int first) {
  const int lane = threadIdx.x & 31;
  const int l = (threadIdx.x >> 5) - first;
  if (l >= 0 && l < p.D) {
    const float* row = sa_w + p.level_amp_offset[l];
    float acc = 0.0f;
#pragma unroll 1
    for (int k = lane; k < p.level_width[l]; k += 32) acc += row[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_sum[l] = acc;
  }
}

//: THE DECAY'S SHARE SCALE: `1 / prod_l sum_l` over the write levels' row sums, 0 for a token that writes nothing (its
//: clock still). -- see docs/internals/decode/decode_fold.md#row-sum
__device__ __forceinline__ float share_scale(const DecodeParams& p, const float* s_sum) {
  float total = 1.0f;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) total *= s_sum[l];
  return total > 0.0f ? 1.0f / total : 0.0f;
}

//: Channel `c` of the token's value, LOADED: its bf16 unit and the token's scale, the product left for after the digit pass. -- see docs/internals/decode/decode_fold.md#fold-value
struct ValueUnit {
  __nv_bfloat16 unit;
  float scale;
};

__device__ __forceinline__ ValueUnit load_value(const DecodeParams& p, int b, int h, int c) {
  return ValueUnit{row_of<__nv_bfloat16>(p.v, b, h)[c * p.v.sw], scale_of(p.v_scale, b, h)};
}

//: The channel's value, `unit * scale`.
__device__ __forceinline__ float value_of(const ValueUnit& d) {
  return __bfloat162float(d.unit) * d.scale;
}

}  // namespace decode
}  // namespace rola
