// csrc/rola/src/scale/scale_kernel.cuh -- THE SCALE PASS's three kernels: each window's write masses over level S
// (`scale_writes`), their exclusive prefix over the windows (`scale_scan`), and each token's byte from its read rows
// against the prefix (`scale_reads`). see docs/internals/scale/scale_kernel.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "scale/scale.cuh"

namespace rola::scale {

namespace ops = ::denseref::ops;

//: A LANE'S 8 DIGITS of a row at `at` as unit values (a u8 digit over 255, a dense f16 at least the floor); a lane past
//: the row's width (`live` false) loads nothing and holds zeros. -- see docs/internals/scale/scale_kernel.md#digits
template <bool F16>
__device__ __forceinline__ void digits_in(float (&x)[kLaneDigits], const uint8_t* at, bool live) {
  if constexpr (F16) {
    const uint4 v = live ? ops::load_vec16(at) : make_uint4(0u, 0u, 0u, 0u);
    const __half2* const h = reinterpret_cast<const __half2*>(&v);
    static_for<4>([&](auto K) {
      const float2 f = __half22float2(h[K]);
      x[2 * K] = live ? fmaxf(f.x, kFloor) : 0.0f;
      x[2 * K + 1] = live ? fmaxf(f.y, kFloor) : 0.0f;
    });
  } else {
    const uint2 v = live ? *reinterpret_cast<const uint2*>(at) : make_uint2(0u, 0u);
    static_for<kLaneDigits>([&](auto K) {
      const uint32_t word = K < 4 ? v.x : v.y;
      x[K] = static_cast<float>((word >> (8 * (K & 3))) & 0xffu) * kInvDigit;
    });
  }
}

//: A WARP'S SUM and LARGEST across its lanes (five butterfly rounds; every lane holds the result).
__device__ __forceinline__ float warp_sum(float v) {
  static_for<5>([&](auto S) { v += __shfl_xor_sync(0xffffffffu, v, 1 << S); });
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  static_for<5>([&](auto S) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1 << S)); });
  return v;
}

//: A ROW's place: token `t` of head `h` in batch `b` (64-bit: a plane spans more than 2^32 bytes).
__device__ __forceinline__ const uint8_t* row_at(const Plane& pl, int b, int h, int t) {
  return pl.base + b * pl.sb + h * pl.sh + static_cast<int64_t>(t) * pl.st;
}

//: THE WINDOW'S WRITE MASSES, a CTA a (write head, window), level S a 256-digit chunk at a time: a warp its token groups
//: of 32, `kAhead` tokens' rows in flight, each token's `w_S c` summed 8 digits a lane -- `c = max_r w_R F` formed in
//: the first chunk beside its level-S loads (a warp max) and kept in `c_tok` for the rest -- the warps' partials joined
//: in shared memory and stored as the window's own masses. -- see docs/internals/scale/scale_kernel.md#writes
template <class Bt, bool SF16, bool RF16>
__global__ __launch_bounds__(ScalePlan<Bt::build>::kThreads) void scale_writes(
    const __grid_constant__ ScaleParams p) {
  constexpr int kWarps = ScalePlan<Bt::build>::kWarps, kWindow = ScalePlan<Bt::build>::kWindow;
  __shared__ float c_tok[kWindow];
  __shared__ __align__(16) float part[kWarps][kChunk];
  const int w = static_cast<int>(blockIdx.x), bh = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int b = bh / p.write_heads, h = bh % p.write_heads;
  const int t0 = w * kWindow, n = min(kWindow, p.L - t0);
  constexpr int kEs = SF16 ? 2 : 1, kEr = RF16 ? 2 : 1;
  float* const out = p.mass + (static_cast<int64_t>(bh) * p.windows + w) * p.width_s;

  //: A CHUNK of level S from digit `c0`; the first forms each token's `c`, the rest read it back
  const auto chunk = [&](int c0, auto first) {
    const int d0 = c0 + kLaneDigits * lane;
    float acc[kLaneDigits];
    static_for<kLaneDigits>([&](auto K) { acc[K] = 0.0f; });
#pragma unroll 1
    for (int g = warp; g * kGroup < n; g += kWarps) {
      float mine = 0.0f;
#pragma unroll 1
      for (int i = 0; i < kGroup; i += kAhead) {
        float x[kAhead][kLaneDigits], c[kAhead];
        static_for<kAhead>([&](auto Q) {
          const int t = min(g * kGroup + i + Q, n - 1);
          digits_in<SF16>(x[Q], row_at(p.write_s, b, h, t0 + t) + kEs * d0, d0 < p.width_s);
        });
        if constexpr (decltype(first)::value) {
          float best[kAhead];
          static_for<kAhead>([&](auto Q) { best[Q] = 0.0f; });
#pragma unroll 1
          for (int r0 = 0; r0 < p.width_r; r0 += kChunk) {
            const int e0 = r0 + kLaneDigits * lane;
            float y[kAhead][kLaneDigits];
            static_for<kAhead>([&](auto Q) {
              const int t = min(g * kGroup + i + Q, n - 1);
              digits_in<RF16>(y[Q], row_at(p.write_r, b, h, t0 + t) + kEr * e0, e0 < p.width_r);
            });
            static_for<kAhead>([&](auto Q) {
              static_for<kLaneDigits>([&](auto K) { best[Q] = fmaxf(best[Q], y[Q][K]); });
            });
          }
          static_for<kAhead>([&](auto Q) {
            const int t = g * kGroup + i + Q;
            const float F = t < n ? p.fm[static_cast<int64_t>(bh) * p.L + t0 + t].x : 0.0f;
            c[Q] = warp_max(best[Q]) * F;
            mine = lane == i + Q ? c[Q] : mine;
          });
        } else {
          static_for<kAhead>([&](auto Q) { c[Q] = c_tok[g * kGroup + i + Q]; });
        }
        static_for<kAhead>([&](auto Q) {
          static_for<kLaneDigits>([&](auto K) { acc[K] = fmaf(x[Q][K], c[Q], acc[K]); });
        });
      }
      if constexpr (decltype(first)::value) c_tok[g * kGroup + lane] = mine;
    }

    static_for<kLaneDigits>([&](auto K) { part[warp][kLaneDigits * lane + K] = acc[K]; });
    __syncthreads();
    float sum = 0.0f;
    static_for<kWarps>([&](auto V) { sum += part[V][threadIdx.x]; });
    if (c0 + static_cast<int>(threadIdx.x) < p.width_s) out[c0 + threadIdx.x] = sum;
    __syncthreads();
  };
  chunk(0, std::true_type{});
#pragma unroll 1
  for (int c0 = kChunk; c0 < p.width_s; c0 += kChunk) chunk(c0, std::false_type{});
}

//: THE PREFIX, a thread a (write head, level-S digit): each window's slot comes to hold the sum of the windows' before
//: it, eight windows' loads in flight. -- see docs/internals/scale/scale_kernel.md#scan
template <class Bt>
__global__ __launch_bounds__(ScalePlan<Bt::build>::kThreads) void scale_scan(
    const __grid_constant__ ScaleParams p) {
  constexpr int kThreads = ScalePlan<Bt::build>::kThreads, kBatch = ScalePlan<Bt::build>::kBatch;
  const int a = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  const int bh = static_cast<int>(blockIdx.y);
  if (a >= p.width_s) return;
  float* const m = p.mass + static_cast<int64_t>(bh) * p.windows * p.width_s + a;
  float run = 0.0f;
#pragma unroll 1
  for (int w0 = 0; w0 < p.windows; w0 += kBatch) {
    float v[kBatch];
    static_for<kBatch>([&](auto Q) {
      v[Q] = w0 + Q < p.windows ? m[static_cast<int64_t>(w0 + Q) * p.width_s] : 0.0f;
    });
    static_for<kBatch>([&](auto Q) {
      if (w0 + Q < p.windows) m[static_cast<int64_t>(w0 + Q) * p.width_s] = run;
      run += v[Q];
    });
  }
}

//: THE BYTES, a CTA a (query head, window): the window's prefix masses in shared memory; each token's level-S row
//: against them and its level-R row's sum, a warp's tokens in groups of 32 (`kAhead` in flight), each total a warp sum;
//: the bound times `inv_factor` is the carry's `unit_byte`, lane `j` storing token `j`'s, on the whole window grid.
//: -- see docs/internals/scale/scale_kernel.md#reads
template <class Bt, bool SF16, bool RF16>
__global__ __launch_bounds__(ScalePlan<Bt::build>::kThreads) void scale_reads(
    const __grid_constant__ ScaleParams p) {
  constexpr int kWarps = ScalePlan<Bt::build>::kWarps, kThreads = ScalePlan<Bt::build>::kThreads;
  constexpr int kWindow = ScalePlan<Bt::build>::kWindow;
  extern __shared__ __align__(16) float mass_s[];
  const int w = static_cast<int>(blockIdx.x), bq = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int qheads = p.write_heads * p.heads, b = bq / qheads, hq = bq % qheads;
  const int bh = bq / p.heads;
  const int t0 = w * kWindow, n = min(kWindow, p.L - t0);
  constexpr int kEs = SF16 ? 2 : 1, kEr = RF16 ? 2 : 1;

  const float* const in = p.mass + (static_cast<int64_t>(bh) * p.windows + w) * p.width_s;
#pragma unroll 1
  for (int a = static_cast<int>(threadIdx.x); a < p.width_s; a += kThreads) mass_s[a] = in[a];
  __syncthreads();

  //: A GROUP of 32 tokens a warp, `kAhead` at a time: level S against the masses, level R's sum
#pragma unroll 1
  for (int g = warp; g * kGroup < n; g += kWarps) {
    int mine = carry::kUnitByteNone;
#pragma unroll 1
    for (int i = 0; i < kGroup; i += kAhead) {
      float dot[kAhead], sum[kAhead];
      static_for<kAhead>([&](auto Q) { dot[Q] = sum[Q] = 0.0f; });
#pragma unroll 1
      for (int c0 = 0; c0 < p.width_s; c0 += kChunk) {
        const int d0 = c0 + kLaneDigits * lane;
        const bool live = d0 < p.width_s;
        float m[kLaneDigits], x[kAhead][kLaneDigits];
        static_for<kAhead>([&](auto Q) {
          const int t = min(g * kGroup + i + Q, n - 1);
          digits_in<SF16>(x[Q], row_at(p.read_s, b, hq, t0 + t) + kEs * d0, live);
        });
        static_for<kLaneDigits>([&](auto K) { m[K] = live ? mass_s[d0 + K] : 0.0f; });
        static_for<kAhead>([&](auto Q) {
          static_for<kLaneDigits>([&](auto K) { dot[Q] = fmaf(x[Q][K], m[K], dot[Q]); });
        });
      }
#pragma unroll 1
      for (int c0 = 0; c0 < p.width_r; c0 += kChunk) {
        const int d0 = c0 + kLaneDigits * lane;
        float x[kAhead][kLaneDigits];
        static_for<kAhead>([&](auto Q) {
          const int t = min(g * kGroup + i + Q, n - 1);
          digits_in<RF16>(x[Q], row_at(p.read_r, b, hq, t0 + t) + kEr * d0, d0 < p.width_r);
        });
        static_for<kAhead>(
            [&](auto Q) { static_for<kLaneDigits>([&](auto K) { sum[Q] += x[Q][K]; }); });
      }
      static_for<kAhead>([&](auto Q) {
        const int byte = carry::unit_byte(warp_sum(sum[Q]) * warp_sum(dot[Q]) * p.inv_factor, 0);
        mine = lane == i + Q ? byte : mine;
      });
    }
    if (g * kGroup + lane < n)
      p.bytes[static_cast<int64_t>(bq) * p.grid + t0 + g * kGroup + lane] =
          static_cast<int8_t>(mine);
  }
  //: a partial window's tail past the call reads nothing
#pragma unroll 1
  for (int t = n + static_cast<int>(threadIdx.x); t < kWindow; t += kThreads)
    p.bytes[static_cast<int64_t>(bq) * p.grid + t0 + t] = static_cast<int8_t>(carry::kUnitByteNone);
}

}  // namespace rola::scale
