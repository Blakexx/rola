// measure/lab/experiments/publish_rounds.cu -- QUESTION: on the even carve the fold's warps own 4 level-R digits x 32
// level-S x 32 channels (fp32, 128 registers) and the readout's 4 level-S x 32 level-R x 32 channels (f16 B fragments,
// 64 registers), so each window's publish is a cross-warp transpose of 32K values. The write slots fill the shared
// memory a whole publish (64 KB) would need; in four rounds of 16 KB (one level-S quarter, two readout warps' slices
// each), single or double buffered under mbarriers, what does it cost against publishing it whole? METHOD: 8 warps; a
// window = fold steps (32 fp32-accumulate HMMAs each), the publish, the readout's tiles (32 fp16-accumulate HMMAs each
// from the pulled slice); stores of f16 pairs into `[S][R][channel]` (an S block 16 bytes past 2 KB, channel chunks
// swizzled by `(R >> 1) & 3`), pulls by `ldmatrix.x4.trans`. The block's span a window.
// see docs/internals/tools/lab.md#publish-rounds
#include "lab.cuh"

namespace publish_rounds {

constexpr int kThreads = 256, kWarps = 8, kLists = 4, kS = 32, kR = 32, kChannels = 32;
constexpr int kBlock = kR * kChannels * 2 + 16, kRoundS = 8, kRounds = kS / kRoundS;

//: MODE 0 no publish; 1 whole, barriers (stores, barrier, pulls, barrier); 2 whole, an mbarrier (stores, arrive, wait,
//: pulls; the buffer's reuse ordered by the next window's stored barrier); 3 rounds, one 16 KB buffer (each round
//: stored, signalled, pulled by its two warps, signalled, before the next); 4 rounds, two buffers (a round stored once
//: the round two before is pulled)
constexpr int kModes = 5;
const char* kModeName[kModes] = {"no publish", "whole, barriers", "whole, mbarrier",
                                 "rounds, one buffer", "rounds, two buffers"};

__host__ __device__ constexpr int buffer_bytes(int mode) {
  return mode == 0   ? 16
         : mode <= 2 ? kS * kBlock
         : mode == 3 ? kRoundS * kBlock
                     : 2 * kRoundS * kBlock;
}

__host__ __device__ constexpr int fold_steps(int size) { return size == 0 ? 32 : 8; }

__host__ __device__ constexpr int read_tiles(int size) { return size == 0 ? 32 : 8; }

const char* kSizeName[2] = {"dense window", "sparse window"};

__device__ __forceinline__ int pub_at(int s, int r, int chunk) {
  return s * kBlock + r * (kChannels * 2) + 16 * (chunk ^ ((r >> 1) & 3));
}

//: A FOLD WARP'S SHARE OF ROUND `k` (level-S rows 8 k .. 8 k + 7 = m-tile k >> 1, half k & 1): its four lists' values
//: there as f16 pairs, at S row `s - s0` of the buffer
__device__ __forceinline__ void store_round(uint8_t* buf, const float (&acc)[kLists][2][4][4],
                                            int k, int s0, int lane) {
  const int g = lane >> 2, tq = lane & 3, w = lab::warp_id(), mt = k >> 1, h = k & 1;
  const int s = 16 * mt + 8 * h + g;
#pragma unroll
  for (int j = 0; j < kLists; ++j)
#pragma unroll
    for (int n = 0; n < 4; ++n)
      *reinterpret_cast<uint32_t*>(buf + pub_at(s - s0, kLists * w + j, n) + 4 * tq) =
          lab::pack_half2(acc[j][mt][n][2 * h], acc[j][mt][n][2 * h + 1]);
}

template <int M>
__device__ __forceinline__ void store_rounds(uint8_t* buf, const float (&acc)[kLists][2][4][4],
                                             int lane) {
#pragma unroll
  for (int k = 0; k < kRounds; ++k) store_round(buf, acc, k, 0, lane);
}

//: A READOUT WARP'S PULL: its four level-S digits (`4 v ..`) as B fragments (K = level R, N = channels), from S row
//: `s - s0`
__device__ __forceinline__ void pull(uint32_t (&b)[4][2][4][2], const uint8_t* buf, int s0,
                                     int lane) {
  const int q = lane >> 3, r = lane & 7, v = lab::warp_id();
#pragma unroll
  for (int d = 0; d < 4; ++d)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk)
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        uint32_t x[4];
        lab::ldsm_t(
            x,
            lab::smem(buf + pub_at(4 * v + d - s0, 16 * kk + 8 * (q & 1) + r, 2 * nh + (q >> 1))));
        b[d][kk][2 * nh][0] = x[0];
        b[d][kk][2 * nh][1] = x[1];
        b[d][kk][2 * nh + 1][0] = x[2];
        b[d][kk][2 * nh + 1][1] = x[3];
      }
}

template <int MODE, int SIZE>
__global__ void __launch_bounds__(kThreads, 1) window(int windows, long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  __shared__ uint64_t stored[2], pulled[2];
  if (threadIdx.x < 2) {
    lab::mbar_init(lab::smem(&stored[threadIdx.x]), kThreads);
    lab::mbar_init(lab::smem(&pulled[threadIdx.x]), 64);
  }
  __syncthreads();
  const int lane = lab::lane_id(), w = lab::warp_id();
  float acc[kLists][2][4][4] = {};
  uint32_t b[4][2][4][2] = {}, outp[4][2] = {};
  uint32_t fa[4], fb[4][2], ra[4];
  for (int i = 0; i < 4; ++i) {
    fa[i] = 0x2c002c00u ^ (0x00010001u * (lane + i));
    ra[i] = 0x2c002c00u ^ (0x00010001u * (lane + 2 * i));
  }
  for (int n = 0; n < 4; ++n) {
    fb[n][0] = 0x2c002c00u ^ (lane + n);
    fb[n][1] = fb[n][0] ^ 0x00100010u;
  }
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < windows; ++x) {
#pragma unroll 1
    for (int s = 0; s < fold_steps(SIZE); ++s) {
      fa[0] ^= 0x00010001u;
#pragma unroll
      for (int j = 0; j < kLists; ++j)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int n = 0; n < 4; ++n) lab::mma_f16_f32(acc[j][mt][n], fa, fb[n][0], fb[n][1]);
    }
    if constexpr (MODE == 1) {
      store_rounds<MODE>(raw, acc, lane);
      __syncthreads();
      pull(b, raw, 0, lane);
      __syncthreads();
    }
    if constexpr (MODE == 2) {
      //: the buffer is rewritten only after every warp pulled last window: the pulled barrier (all 256 arrive)
      if (x >= 1) lab::mbar_wait(lab::smem(&pulled[0]), static_cast<uint32_t>(x - 1) & 1u);
      store_rounds<MODE>(raw, acc, lane);
      lab::mbar_arrive(lab::smem(&stored[0]));
      lab::mbar_wait(lab::smem(&stored[0]), static_cast<uint32_t>(x) & 1u);
      pull(b, raw, 0, lane);
      if (lane < 8) lab::mbar_arrive(lab::smem(&pulled[0]));
    }
    if constexpr (MODE == 3 || MODE == 4) {
      //: round k into buffer `k % buffers`; its two readout warps (2 k, 2 k + 1) pull it; the buffer's next round waits
      //: for that pull. The phase counts run over windows: round k of window x is the (x rounds + k)-th use
      constexpr int kBuffers = MODE == 3 ? 1 : 2;
#pragma unroll
      for (int k = 0; k < kRounds; ++k) {
        const int use = x * kRounds + k, buf = use % kBuffers, nth = use / kBuffers;
        uint8_t* const at = raw + buf * kRoundS * kBlock;
        if (use >= kBuffers)
          lab::mbar_wait(lab::smem(&pulled[buf]), static_cast<uint32_t>(nth - 1) & 1u);
        store_round(at, acc, k, kRoundS * k, lane);
        lab::mbar_arrive(lab::smem(&stored[buf]));
        if ((w >> 1) == k) {
          lab::mbar_wait(lab::smem(&stored[buf]), static_cast<uint32_t>(nth) & 1u);
          pull(b, at, kRoundS * k, lane);
          lab::mbar_arrive(lab::smem(&pulled[buf]));
        }
      }
    }
#pragma unroll 1
    for (int tile = 0; tile < read_tiles(SIZE); ++tile) {
      ra[0] ^= 0x00010001u;
#pragma unroll
      for (int d = 0; d < 4; ++d)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int n = 0; n < 4; ++n) lab::mma_f16_f16(outp[n], ra, b[d][kk][n][0], b[d][kk][n][1]);
    }
    if constexpr (MODE == 0) acc[0][0][0][0] += 1.0f;
  }
  clock.stop(t);
  uint32_t f = 0;
  for (int j = 0; j < kLists; ++j)
    for (int mt = 0; mt < 2; ++mt)
      for (int n = 0; n < 4; ++n)
        for (int i = 0; i < 4; ++i) f += lab::bits(acc[j][mt][n][i]);
  for (int n = 0; n < 4; ++n) f += outp[n][0] ^ outp[n][1];
  for (int d = 0; d < 4; ++d)
    for (int kk = 0; kk < 2; ++kk) f += b[d][kk][0][0] ^ b[d][kk][3][1];
  lab::sink(out, f);
}

constexpr int kWindows = 200;

template <int MODE, int SIZE>
void variant(const lab::Bench& b) {
  const std::string name = lab::kernel("window", {MODE, SIZE});
  b.expect(name, "HMMA.16816.F32", ">=", 32);
  b.expect(name, "HMMA.16816.F16", ">=", 16);
  if (MODE != 0) b.expect(name, "LDSM", ">=", 16);
  if (MODE != 0) b.expect(name, "STS", ">=", 64);
  if (b.expect_only) return;
  const int bytes = buffer_bytes(MODE);
  LAB_CHECK(
      cudaFuncSetAttribute(window<MODE, SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s =
      b.variant([&] { window<MODE, SIZE><<<b.sms, kThreads, bytes>>>(kWindows, b.times, b.sinks); },
                [&] { return b.span(kWarps) / kWindows; });
  b.record(std::string(kModeName[MODE]) + " / " + kSizeName[SIZE] + " / "
               + std::to_string(bytes / 1024) + " KB",
           "cycles a window", s);
}

template <int SIZE>
void sizes(const lab::Bench& b) {
  variant<0, SIZE>(b);
  variant<1, SIZE>(b);
  variant<2, SIZE>(b);
  variant<3, SIZE>(b);
  variant<4, SIZE>(b);
}

}  // namespace publish_rounds

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  publish_rounds::sizes<0>(b);
  publish_rounds::sizes<1>(b);
}
