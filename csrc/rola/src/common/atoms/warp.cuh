// csrc/rola/src/common/atoms/warp.cuh -- a concern of the baseline operations (`common/ops.cuh`).
// see docs/internals/common/atoms/warp.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

namespace denseref {
namespace ops {

//: THE n-TH SET BIT of a word (n from 0), branch free -- see docs/internals/common/atoms/warp.md#nth-set-bit
__device__ __forceinline__ int nth_set_bit(uint32_t w, int n) {
  int pos = 0;
#pragma unroll
  for (int width = 16; width >= 1; width >>= 1) {
    const int c = __popc(w & ((1u << width) - 1u));
    const bool hi = n >= c;
    n -= hi ? c : 0;
    pos += hi ? width : 0;
    w = hi ? (w >> width) : w;
  }
  return pos;
}

//: A ONE-TRIP LOOP BOUND ptxas cannot see through -- see docs/internals/common/atoms/warp.md#once
__device__ __forceinline__ int once(bool cond) {
  int n = cond ? 1 : 0;
  asm volatile("" : "+r"(n));
  return n;
}

//: A VALUE MADE OPAQUE at this use: what derives from it is formed here -- see docs/internals/common/atoms/warp.md#opaque
__device__ __forceinline__ int opaque(int v) {
  int out;
  asm volatile("mov.b32 %0, %1;" : "=r"(out) : "r"(v));
  return out;
}

//: THE LOCKSTEP UNIT'S WIDTH, deliberately not the MMA unit's -- see docs/internals/common/atoms/warp.md#kballotlanes
constexpr int kBallotLanes = 32;

//: ONE BIT PER LANE, IN LANE ORDER. -- see docs/internals/common/atoms/warp.md#lane-ballot
__device__ __forceinline__ uint32_t lane_ballot(bool pred) {
  return __ballot_sync(0xFFFFFFFFu, pred);
}

//: THE WARP'S INDEX AS A WARP-UNIFORM VALUE. -- see docs/internals/common/atoms/warp.md#uniform-warp
template <int Warps>
__device__ __forceinline__ int uniform_warp() {
  static_assert(Warps > 0 && (Warps & (Warps - 1)) == 0, "a CTA is a power of two of warps");
  constexpr int kBits = Warps == 1 ? 0 : (Warps == 2 ? 1 : (Warps == 4 ? 2 : 3));
  static_assert(Warps == (1 << kBits), "a CTA is one, two, four or eight warps");
  int w = 0;
#pragma unroll
  for (int b = 0; b < kBits; ++b)
    w |= (__popc(__ballot_sync(0xFFFFFFFFu, ((threadIdx.x >> (5 + b)) & 1u) != 0u)) >> 5) << b;
  return w;
}

//: THE LANES BEFORE THIS ONE, as the hardware's own register rather than a value the body would have to carry
__device__ __forceinline__ uint32_t lanes_below() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

//: THE TWO READINGS OF A VOTE. -- see docs/internals/common/atoms/warp.md#bit-count
__device__ __forceinline__ int bit_count(uint32_t x) { return __popc(x); }

//: `kBallotLanes` for an empty vector: "nowhere" compares past every lane -- see docs/internals/common/atoms/warp.md#firstset
__device__ __forceinline__ int first_set(uint32_t x) {
  const int f = __ffs((int)x);
  return f ? f - 1 : kBallotLanes;
}

//: THE OTHER END OF THE SAME VOTE, for a caller consuming its lanes from the top -- see docs/internals/common/atoms/warp.md#lastset
__device__ __forceinline__ int last_set(uint32_t x) { return 31 - __clz((int)x); }

//: THE UNIT'S OWN RUNNING TOTAL. -- see docs/internals/common/atoms/warp.md#lane-prefix-incl
template <int N>
__device__ __forceinline__ int lane_prefix_incl(int x) {
#pragma unroll
  for (int d = 1; d < N; d <<= 1) {
    const int up = __shfl_up_sync(0xFFFFFFFFu, x, d);
    if ((int)(threadIdx.x & (kBallotLanes - 1)) >= d) x += up;
  }
  return x;
}

//: one lane's value, read by all of them.
__device__ __forceinline__ int lane_broadcast(int x, int src) {
  return __shfl_sync(0xFFFFFFFFu, x, src);
}

}  // namespace ops
}  // namespace denseref
