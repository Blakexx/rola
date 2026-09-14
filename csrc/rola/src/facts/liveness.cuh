// csrc/rola/src/facts/liveness.cuh -- THE ONE LIVENESS PASS: the geometry-independent
// kernel that votes the class-1 words of `liveness_contract.cuh`, and its launch.
// Nothing here knows a span, a carve or a compile-time width; every consumer's grain
// is a FOLD over what this writes.
// -- see docs/internals/facts/liveness.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: FOUR WARPS, ONE TOKEN WORD EACH: a row's word is one warp ballot.
//: -- see docs/internals/facts/liveness.md#the-grain
constexpr int kLivenessWarps = 4;
constexpr int kLivenessThreads = kTokBits * kLivenessWarps;

//: SIXTEEN DIGITS PER LANE PER LOAD: two eight-byte-aligned `uint4`s, one full
//: thirty-two-byte sector, which every lawful level's width and row base are aligned
//: to. -- see docs/internals/facts/liveness.md#the-read
constexpr int kVecDigits = 16;

//: The digit mask's widest form: four levels of the widest branch, packed.
constexpr int kMaxDigits = kMaxLevels * 256;
constexpr int kMaxMaskWords = kMaxDigits / 32;

//: THE PASS'S WHOLE OPERAND BLOCK, by value. -- see docs/internals/facts/liveness.md#passplan
struct PassPlan {
  int D;
  int B[kMaxLevels];
  int rows;
  int L;
  int words;
  long side_words;
  uint32_t dense[kSides];
  uint32_t mask[kSides][kMaxMaskWords];
};

//: -- see docs/internals/facts/liveness.md#the-kernel
__global__ __launch_bounds__(kLivenessThreads) void liveness_pass(
    const uint16_t* __restrict__ plane_read, const uint16_t* __restrict__ plane_write,
    uint32_t* __restrict__ out, const PassPlan plan) {
  const int word = blockIdx.x * kLivenessWarps + (int)(threadIdx.x >> 5);
  const int lane = (int)(threadIdx.x & 31u);
  if (word >= plan.words) return;  // a whole lockstep unit leaves together

  const int token = word * kTokBits + lane;
  const bool real = token < plan.L;
  const uint32_t alive = __ballot_sync(0xffffffffu, real);
  const long base = (long)blockIdx.y * plan.L * plan.rows + (long)(real ? token : 0) * plan.rows;

  uint32_t* const dst = out + (long)blockIdx.y * kSides * plan.side_words + word;

#pragma unroll
  for (int side = 0; side < kSides; ++side) {
    const SideStatics statics{plan.dense[side], plan.mask[side]};
    const uint16_t* const column = (side == kRead ? plane_read : plane_write) + base;
    uint32_t* const rows = dst + (long)side * plan.side_words;
    int row = 0;
#pragma unroll 1
    for (int level = 0; level < plan.D; ++level) {
      const int width = plan.B[level];
      if (level_is_dense(statics.dense_levels, level)) {
#pragma unroll 1
        for (int digit = 0; digit < width; ++digit, ++row)
          if (lane == 0)
            rows[(long)row * plan.words] = digit_in_mask(statics.digit_mask, row) ? alive : 0u;
      } else {
#pragma unroll 1
        for (int digit = 0; digit < width; digit += kVecDigits, row += kVecDigits) {
          const uint4* const sector = reinterpret_cast<const uint4*>(column + row);
          const uint4 lo = sector[0];
          const uint4 hi = sector[1];
          const uint32_t halves[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (int k = 0; k < kVecDigits; ++k) {
            const uint32_t half = halves[k >> 1];
            const uint16_t amplitude = (uint16_t)((k & 1) ? (half >> 16) : half);
            const uint32_t vote = __ballot_sync(0xffffffffu, real && amplitude_live(amplitude));
            if (lane == 0) rows[(long)(row + k) * plan.words] = vote;
          }
        }
      }
    }
  }
}

//: -- see docs/internals/facts/liveness.md#the-launch
inline void launch_liveness(const uint16_t* plane_read, const uint16_t* plane_write, uint32_t* out,
                            const PassPlan& plan, int BH, cudaStream_t stream) {
  const dim3 grid((plan.words + kLivenessWarps - 1) / kLivenessWarps, BH);
  liveness_pass<<<grid, kLivenessThreads, 0, stream>>>(plane_read, plane_write, out, plan);
}

}  // namespace rola::facts
