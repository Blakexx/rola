// measure/lab/experiments/read_red.cu -- QUESTION: can the sparse readout's exits go straight to `num` by global
// reductions -- each lane's outputs scaled and packed as f16 pairs, `red.global.add.noftz.f16x2`, no shared panel and
// no drain -- and what does that cost beside the readout's MMAs against the exact s16 pairs (the integer panel of
// the time) added into the CTA's
// panel and drained after each pass? METHOD: one CTA an SM, 8 warps; `G` CTAs share a head's `num` rows (the boxes of
// one head add into the same tokens), 1,024 tokens a head, a CTA one 32-channel half of the row. A pass: each warp
// `TILES` tiles, a tile 8 tokens on N (8-aligned, spread over the head's tokens), its read rows by one `ldmatrix.x4`
// from shared memory and 4 bf16 MMAs against a held state (the readout's tokens-on-N tile), then the exit. FORM 0
// the integer panel's: each lane's two tokens' channel pairs (g, g + 8) as exact s16 pairs added into the panel at the token's
// slot (`red.shared.add.u32`), and after the pass a barrier, the drain (a warp its 32 slots, two a request, each word
// read, cleared and added into `num` by `red.global.add.noftz.f16x2`) and a barrier; FORM 1 each pair packed f16 and
// added into `num` at the token's row (`red.global.add.noftz.f16x2`, 4 a lane a tile), nothing after the pass; FORM 2
// the same as fp32 adds (`red.global.add.f32`, 8 a lane a tile); FORM 3 as 1 through the warp's own staging tile in
// shared memory (plain stores, no atomics, no barrier: a `__syncwarp`), each request then two tokens' 64 contiguous
// bytes (lanes 0-15 one token's 16 words, 16-31 the next's) -- half the line requests of 1's fragment order; FORM 4 the same relayout in registers (four `shfl.sync` and a select a request, no shared
// memory); FORM 5 a 64-channel box (4 m-tiles, 8 MMAs a tile) staged as 3, a token's 128 bytes a full line a request; FORM 9 no
// exit (the tiles alone, folded into a sink).
// A sample is the CTA's span over its passes: cycles a pass, and a tile a warp. A rate probe: its sums fold into
// `num` and a sink, unchecked. see docs/lab/read_red.md
#include <cuda_fp16.h>

#include "lab.cuh"

namespace read_red {

constexpr int kWarps = 8, kThreads = kWarps * 32, kTokens = 1024, kSlots = 256, kWords = 16,
              kPasses = 256;
//: the read rows a tile reads (64 tiles' 512 bytes, cycled), the panel (a slot a token's 16 words) after them, then
//: each warp's staging tile (8 tokens' 16 words)
constexpr int kRows = 64 * 512, kPanelBytes = kSlots * kWords * 4, kStaging = 8 * 32 * 4;
constexpr int kSmem = kRows + kPanelBytes + kWarps * kStaging;

__device__ __forceinline__ void red_h2(__half2* at, uint32_t v) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void red_f32(float* at, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(at), "f"(v) : "memory");
}

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: the exact pair (`read_image::s16_exact`): two outputs times the coefficient as one u32 of two s16 halves
__device__ __forceinline__ uint32_t s16_exact(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

__device__ __forceinline__ uint32_t f16_pair(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

template <int FORM, int TILES, int G>
__global__ void __launch_bounds__(kThreads, 1) exits(__half2* num, float* num32, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), b = static_cast<int>(blockIdx.x);
  const int head = b / G, half = b & 1, g = lane >> 2, tq = lane & 3;
  //: the read rows bf16 1/64, the panel zero
  for (int k = static_cast<int>(threadIdx.x); k < kSmem / 4; k += kThreads)
    reinterpret_cast<uint32_t*>(smem)[k] = k < kRows / 4 ? 0x3C803C80u : 0u;
  __syncthreads();
  const uint32_t sm = lab::smem(smem), pad = sm + kRows;
  constexpr int kM = FORM == 5 ? 4 : 2;
  uint32_t f[kM * 8];
#pragma unroll
  for (int i = 0; i < kM * 8; ++i) f[i] = lab::held(0x3F803F80u + static_cast<uint32_t>(i & 1));
  const float c0 = __uint_as_float(lab::held(0x3F800000u)),
              c1 = __uint_as_float(lab::held(0x3F7FFFFFu));
  __half2* const hrow = num + static_cast<size_t>(head) * kTokens * 32 + half * 16;
  float* const frow = num32 + static_cast<size_t>(head) * kTokens * 64 + half * 32;
  lab::WarpSpan clock;
  clock.start();
  uint32_t fold = 0;
#pragma unroll 1
  for (int p = 0; p < kPasses; ++p) {
#pragma unroll 1
    for (int i = 0; i < TILES; ++i) {
      const int n = p * TILES + i;
      const int t0 = ((b * 131 + warp * 37 + n * 61) * 8) & (kTokens - 1);
      uint32_t bb[4];
      lab::ldsm(bb, sm + ((n * kWarps + warp) & 63) * 512 + (static_cast<uint32_t>(lane) << 4));
      float acc[kM][4] = {};
#pragma unroll
      for (int m = 0; m < kM; ++m)
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1],
                                 f[(m * 2 + kb) * 4 + 2], f[(m * 2 + kb) * 4 + 3]};
          lab::mma_bf16_f32(acc[m], a, bb[2 * kb], bb[2 * kb + 1]);
        }
      const int tok0 = t0 + 2 * tq, tok1 = tok0 + 1;
      if constexpr (FORM == 0) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          panel_red(pad + static_cast<uint32_t>(((tok0 & (kSlots - 1)) * kWords + 8 * m + g) * 4),
                    s16_exact(acc[m][0], acc[m][2], c0));
          panel_red(pad + static_cast<uint32_t>(((tok1 & (kSlots - 1)) * kWords + 8 * m + g) * 4),
                    s16_exact(acc[m][1], acc[m][3], c1));
        }
      } else if constexpr (FORM == 1) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          red_h2(hrow + static_cast<size_t>(tok0) * 32 + 8 * m + g,
                 f16_pair(acc[m][0] * c0, acc[m][2] * c0));
          red_h2(hrow + static_cast<size_t>(tok1) * 32 + 8 * m + g,
                 f16_pair(acc[m][1] * c1, acc[m][3] * c1));
        }
      } else if constexpr (FORM == 3) {
        const uint32_t st = pad + kPanelBytes + static_cast<uint32_t>(warp * kStaging);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(
                           st + static_cast<uint32_t>(((2 * tq) * kWords + 8 * m + g) * 4)),
                       "r"(f16_pair(acc[m][0] * c0, acc[m][2] * c0)));
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(
                           st + static_cast<uint32_t>(((2 * tq + 1) * kWords + 8 * m + g) * 4)),
                       "r"(f16_pair(acc[m][1] * c1, acc[m][3] * c1)));
        }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int tok = 2 * k + (lane >> 4), word = lane & 15;
          uint32_t v;
          asm volatile("ld.shared.u32 %0, [%1];"
                       : "=r"(v)
                       : "r"(st + static_cast<uint32_t>((tok * kWords + word) * 4)));
          red_h2(hrow + static_cast<size_t>(t0 + tok) * 32 + word, v);
        }
        __syncwarp();
      } else if constexpr (FORM == 4) {
        //: lane L's word of request k: token 2k + (L >> 4), pair (m, g') = ((L >> 3) & 1, L & 7), held by lane 4 g' + k as
        //: its (m, token parity) word
        const uint32_t w00 = f16_pair(acc[0][0] * c0, acc[0][2] * c0),
                       w01 = f16_pair(acc[0][1] * c1, acc[0][3] * c1);
        const uint32_t w10 = f16_pair(acc[1][0] * c0, acc[1][2] * c0),
                       w11 = f16_pair(acc[1][1] * c1, acc[1][3] * c1);
        const int mm = (lane >> 3) & 1, jj = lane >> 4, word = lane & 15;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int src = 4 * (lane & 7) + k;
          const uint32_t a00 = __shfl_sync(0xffffffffu, w00, src),
                         a01 = __shfl_sync(0xffffffffu, w01, src);
          const uint32_t a10 = __shfl_sync(0xffffffffu, w10, src),
                         a11 = __shfl_sync(0xffffffffu, w11, src);
          const uint32_t v = mm ? (jj ? a11 : a10) : (jj ? a01 : a00);
          red_h2(hrow + static_cast<size_t>(t0 + 2 * k + jj) * 32 + word, v);
        }
      } else if constexpr (FORM == 5) {
        //: the 64-channel box: a token's 32 words (128 bytes) a request, staged through the warp's tile
        const uint32_t st = pad + kPanelBytes + static_cast<uint32_t>(warp * kStaging);
        __half2* const row64 = num + static_cast<size_t>(head) * kTokens * 32;
#pragma unroll
        for (int m = 0; m < kM; ++m) {
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(
                           st + static_cast<uint32_t>(((2 * tq) * 32 + 8 * m + g) * 4)),
                       "r"(f16_pair(acc[m][0] * c0, acc[m][2] * c0)));
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(
                           st + static_cast<uint32_t>(((2 * tq + 1) * 32 + 8 * m + g) * 4)),
                       "r"(f16_pair(acc[m][1] * c1, acc[m][3] * c1)));
        }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          uint32_t v;
          asm volatile("ld.shared.u32 %0, [%1];"
                       : "=r"(v)
                       : "r"(st + static_cast<uint32_t>((k * 32 + lane) * 4)));
          red_h2(row64 + static_cast<size_t>(t0 + k) * 32 + lane, v);
        }
        __syncwarp();
      } else if constexpr (FORM == 2) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float* const r0 = frow + static_cast<size_t>(tok0) * 64 + 16 * m + g;
          float* const r1 = frow + static_cast<size_t>(tok1) * 64 + 16 * m + g;
          red_f32(r0, acc[m][0] * c0);
          red_f32(r0 + 8, acc[m][2] * c0);
          red_f32(r1, acc[m][1] * c1);
          red_f32(r1 + 8, acc[m][3] * c1);
        }
      } else {
#pragma unroll
        for (int m = 0; m < kM; ++m)
#pragma unroll
          for (int k = 0; k < 4; ++k) fold ^= __float_as_uint(acc[m][k]);
      }
    }
    if constexpr (FORM == 0) {
      //: THE DRAIN: a warp its 32 slots, two a request (lanes 0-15 one slot's words, 16-31 the next's), each word
      //: read, cleared and added at the slot's token
      __syncthreads();
#pragma unroll 4
      for (int k = 0; k < 16; ++k) {
        const int slot = warp * 32 + 2 * k + (lane >> 4), word = lane & 15;
        const uint32_t at = pad + static_cast<uint32_t>((slot * kWords + word) * 4);
        uint32_t v;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(at));
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(at), "r"(0u));
        red_h2(hrow + static_cast<size_t>((p * 256 + slot) & (kTokens - 1)) * 32 + word, v);
      }
      __syncthreads();
    }
  }
  clock.stop(t);
  lab::sink(out, fold);
}

template <int FORM, int TILES, int G>
void variant(const lab::Bench& b, __half2* num, float* num32) {
  const std::string name = lab::kernel("exits", {FORM, TILES, G});
  b.expect(name, "HMMA", ">=", 4);
  if (FORM >= 1 && FORM <= 5) {
    b.expect(name, "RED", ">=", 1);
    //: the exits' global reductions inside the tile's step: the form under test
    b.expect(name, "STEP_GLOBAL", ">=", 0);
  }
  //: the pass loop's tile loop and the drain sit between one tile's MMAs and the next's: the forms under test
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a rate probe: its exits fold into num and a sink");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(exits<FORM, TILES, G>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSmem));
  lab::Samples tiles;
  const lab::Samples pass = b.variant(
      [&] { exits<FORM, TILES, G><<<b.sms, kThreads, kSmem>>>(num, num32, b.times, b.sinks); },
      [&] {
        std::vector<long long> h(static_cast<size_t>(b.sms) * kWarps * 2);
        LAB_CHECK(
            cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double c = 0;
        for (int sm = 0; sm < b.sms; ++sm) {
          long long lo = h[2 * sm * kWarps], hi = h[2 * sm * kWarps + 1];
          for (int w = 1; w < kWarps; ++w) {
            lo = std::min(lo, h[2 * (sm * kWarps + w)]);
            hi = std::max(hi, h[2 * (sm * kWarps + w) + 1]);
          }
          c += double(hi - lo) / kPasses;
        }
        tiles.v.push_back(c / b.sms / TILES);
        return c / b.sms;
      });
  const char* form =
      FORM == 0   ? "exact s16 pairs into the panel, drained a pass"
      : FORM == 1 ? "f16 pairs straight to num (red.global.add.noftz.f16x2)"
      : FORM == 2 ? "fp32 straight to num (red.global.add.f32)"
      : FORM == 3 ? "f16 pairs to num through the warp's staging tile, 64 bytes a token a request"
      : FORM == 4 ? "f16 pairs to num relaid out by shuffles, 64 bytes a token a request"
      : FORM == 5 ? "a 64-channel box, staged, a token's 128 bytes (a line) a request"
                  : "no exit (the tiles alone)";
  const std::string row = std::string(form) + ", " + std::to_string(TILES)
                          + " tiles a warp a pass, " + std::to_string(G) + " CTAs a head";
  b.unchecked(row, "a rate probe: its exits fold into num and a sink");
  b.record(row, "cycles a pass a CTA", pass);
  b.record(row + ": cycles a tile a warp", "cycles", tiles);
}

}  // namespace read_red

int main(int argc, char** argv) {
  using namespace read_red;
  lab::Bench b(argc, argv);
  __half2* num = nullptr;
  float* num32 = nullptr;
  if (!b.expect_only) {
    const size_t heads = static_cast<size_t>(b.sms) + 1;
    LAB_CHECK(cudaMalloc(&num, heads * kTokens * 32 * sizeof(__half2)));
    LAB_CHECK(cudaMalloc(&num32, heads * kTokens * 64 * sizeof(float)));
    LAB_CHECK(cudaMemset(num, 0, heads * kTokens * 32 * sizeof(__half2)));
    LAB_CHECK(cudaMemset(num32, 0, heads * kTokens * 64 * sizeof(float)));
  }
  //: the sparse readout's ~3.3 tiles a warp a window (alt-k4) and a denser 16; a head's boxes sharing its rows, and none
  variant<9, 4, 16>(b, num, num32);
  variant<0, 4, 16>(b, num, num32);
  variant<1, 4, 16>(b, num, num32);
  variant<2, 4, 16>(b, num, num32);
  variant<3, 4, 16>(b, num, num32);
  variant<4, 4, 16>(b, num, num32);
  variant<5, 4, 16>(b, num, num32);
  variant<1, 4, 1>(b, num, num32);
  variant<9, 16, 16>(b, num, num32);
  variant<0, 16, 16>(b, num, num32);
  variant<1, 16, 16>(b, num, num32);
  variant<2, 16, 16>(b, num, num32);
  variant<3, 16, 16>(b, num, num32);
  variant<4, 16, 16>(b, num, num32);
  variant<5, 16, 16>(b, num, num32);
}
