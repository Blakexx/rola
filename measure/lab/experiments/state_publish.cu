// measure/lab/experiments/state_publish.cu -- QUESTION: each window the owner CTA publishes its fold state to its
// readout: the fold's warps own 8 R-lists x all 16 S-digits (fp32 C fragments, 128 registers a thread), the readout's
// warps own 2 S-digits x all 64 R-leaves (f16 A fragments, 64 registers), so the publish is a cross-warp transpose of
// 1024 leaves x 32 channels through shared memory. What does it cost alone, how much of a dense and a sparse window
// does it expose, and does it hide under the boundary's other work (the read pool's fill, the lists' integer chains and
// barriers)? METHOD: 8 warps; a window = fold steps (32 fp32-accumulate HMMAs each into the state) + the boundary + the
// readout's tiles (16 fp16-accumulate HMMAs each from the pulled slice); the publish stores the state as f16 into
// `[S][R][channel]` with chunk `c ^ ((R >> 1) & 3) ^ (S & 3)`, one barrier, each warp pulls its slice by
// `ldmatrix.x4.trans`. The block's span a window; `tools/lab.py ncu` reads the bank conflicts.
// see docs/internals/tools/lab.md#state-publish
#include "lab.cuh"

namespace state_publish {

constexpr int kThreads = 256, kLists = 8, kS = 16, kR = 64, kChannels = 32;
constexpr int kPublishBytes = kS * kR * kChannels * 2, kReadRows = 128, kReadRowBytes = 96;
constexpr int kSourceRows = 1 << 18;

//: MODE 0 the publish alone; 1 the window without it; 2 the window, the publish between fold and readout; 3 the window
//: with the boundary's work and no publish; 4 the boundary's work then the publish; 5 the two interleaved; 6 the
//: publish by an mbarrier -- stores, arrive, the readout's start-up (the read pool's fill and the integer chain, no
//: barriers), wait, pull. The padded variants (PAD 1) give each S-digit's block 16 more bytes, so the stores of a warp
//: (8 S-digits at once) land on distinct banks
constexpr int kModes = 7;
const char* kModeName[kModes] = {"publish alone",
                                 "fold + readout",
                                 "fold + publish + readout",
                                 "fold + boundary + readout",
                                 "fold + boundary, then publish + readout",
                                 "fold + boundary interleaved with publish + readout",
                                 "fold + publish by mbarrier over the read start-up + readout"};

//: SIZE 0 a dense window (32 fold steps, 64 read tiles a warp); 1 a sparse one (8 and 16)
__host__ __device__ constexpr int fold_steps(int size) { return size == 0 ? 32 : 8; }

__host__ __device__ constexpr int read_tiles(int size) { return size == 0 ? 64 : 16; }

const char* kSizeName[2] = {"dense window", "sparse window"};

template <int PAD>
__device__ __forceinline__ int pub_at(int s, int r, int chunk) {
  //: padded, an S-digit's block starts 16 bytes on from the last, so the S-digits a warp stores at once spread over the
  //: banks and the swizzle needs only the reads' term
  return s * (kR * kChannels * 2 + 16 * PAD) + r * (kChannels * 2)
         + 16 * (chunk ^ ((r >> 1) & 3) ^ (PAD ? 0 : (s & 3)));
}

//: THE PUBLISH: the warp's state to f16 at `[S][R][channel]` (lane (g, tq) holds S = g and g + 8, R = 8 warp + list,
//: channels 8 n + 2 tq, + 1)
template <int PAD>
__device__ __forceinline__ void publish_store(uint8_t* pub, const float (&acc)[kLists][4][4],
                                              int lane) {
  const int g = lane >> 2, tq = lane & 3, w = lab::warp_id();
#pragma unroll
  for (int j = 0; j < kLists; ++j)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int r = kLists * w + j;
      *reinterpret_cast<uint32_t*>(pub + pub_at<PAD>(g, r, n) + 4 * tq) =
          lab::pack_half2(acc[j][n][0], acc[j][n][1]);
      *reinterpret_cast<uint32_t*>(pub + pub_at<PAD>(g + 8, r, n) + 4 * tq) =
          lab::pack_half2(acc[j][n][2], acc[j][n][3]);
    }
}

//: THE PULL: the warp's two S-digits as A fragments (M = channels, K = R), two m-tiles by four k-steps a digit
template <int PAD>
__device__ __forceinline__ void publish_pull(uint32_t (&a)[2][2][4][4], const uint8_t* pub,
                                             int lane) {
  const int q = lane >> 3, r = lane & 7, w = lab::warp_id();
#pragma unroll
  for (int d = 0; d < 2; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        lab::ldsm_t(
            a[d][mt][kk],
            lab::smem(pub + pub_at<PAD>(2 * w + d, 16 * kk + 8 * (q >> 1) + r, 2 * mt + (q & 1))));
}

//: THE BOUNDARY'S OWN WORK: the read pool's fill (128 rows of 96 bytes, scattered) and a lists-like integer chain with
//: two barriers; `between(i)` runs in the chain's gaps (the interleaved publish's stores)
template <bool kBarriers, class Between>
__device__ __forceinline__ uint32_t boundary(uint8_t* pool, const uint8_t* src, int window, int tid,
                                             Between&& between) {
  const uint32_t base = lab::smem(pool);
#pragma unroll 1
  for (int i = tid; i < kReadRows * (kReadRowBytes / 16); i += kThreads) {
    const int row = i / (kReadRowBytes / 16), chunk = i % (kReadRowBytes / 16);
    const int64_t token =
        (static_cast<int64_t>(blockIdx.x) * 977 + window * 131 + row * 37) % kSourceRows;
    lab::cp_async16(base + row * kReadRowBytes + 16 * chunk,
                    src + token * kReadRowBytes + 16 * chunk);
  }
  uint32_t x = tid * 2654435761u + window;
#pragma unroll 1
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int k = 0; k < 24; ++k) x = (x ^ (x >> 7)) * 0x9E3779B1u + k;
    between(i);
    if (kBarriers && i == 1) __syncthreads();
  }
  lab::cp_wait_all();
  if (kBarriers) __syncthreads();
  return x;
}

template <int MODE, int SIZE, int PAD>
__global__ void __launch_bounds__(kThreads, 1) window(int windows, const uint8_t* src, long long* t,
                                                      int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  __shared__ uint64_t pubbar;
  uint8_t* const pub = raw;
  uint8_t* const pool = raw + kPublishBytes + 256;
  if (threadIdx.x == 0) lab::mbar_init(lab::smem(&pubbar), kThreads);
  __syncthreads();
  const int lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  float acc[kLists][4][4] = {};
  uint32_t a[2][2][4][4] = {}, outp[2][2][2] = {};
  uint32_t fa[4], fb[4][2], rb[4][2];
  for (int i = 0; i < 4; ++i) fa[i] = 0x2c002c00u ^ (0x00010001u * (lane + i));
  for (int n = 0; n < 4; ++n) {
    fb[n][0] = 0x2c002c00u ^ (lane + n);
    fb[n][1] = fb[n][0] ^ 0x00100010u;
    rb[n][0] = 0x2c002c00u ^ (lane + 3 * n);
    rb[n][1] = rb[n][0] ^ 0x00200020u;
  }
  uint32_t chain = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    if constexpr (MODE != 0) {
#pragma unroll 1
      for (int s = 0; s < fold_steps(SIZE); ++s) {
        fa[0] ^= 0x00010001u;
#pragma unroll
        for (int j = 0; j < kLists; ++j)
#pragma unroll
          for (int n = 0; n < 4; ++n) lab::mma_f16_f32(acc[j][n], fa, fb[n][0], fb[n][1]);
      }
    }
    if constexpr (MODE == 6) {
      publish_store<PAD>(pub, acc, lane);
      lab::mbar_arrive(lab::smem(&pubbar));
      chain += boundary<false>(pool, src, w, tid, [](int) {});
      lab::mbar_wait(lab::smem(&pubbar), static_cast<uint32_t>(w) & 1u);
      publish_pull<PAD>(a, pub, lane);
    }
    if constexpr (MODE >= 3 && MODE <= 5) {
      chain += boundary<true>(pool, src, w, tid, [&](int i) {
        if constexpr (MODE == 5) {
          if (i == 0) publish_store<PAD>(pub, acc, lane);
        }
      });
    }
    if constexpr (MODE == 0 || MODE == 2 || MODE == 4) publish_store<PAD>(pub, acc, lane);
    if constexpr (MODE != 1 && MODE != 3 && MODE != 6) {
      __syncthreads();
      publish_pull<PAD>(a, pub, lane);
    }
    if constexpr (MODE != 0) {
#pragma unroll 1
      for (int tile = 0; tile < read_tiles(SIZE); ++tile) {
        rb[0][0] ^= 0x00010001u;
#pragma unroll
        for (int d = 0; d < 2; ++d)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              lab::mma_f16_f16(outp[d][mt], a[d][mt][kk], rb[kk][0], rb[kk][1]);
      }
    }
    //: the buffer is rewritten next window only after every warp has pulled: a barrier (the readout's tiles lie
    //: between under the other modes, but nothing else orders them)
    if constexpr (MODE != 1 && MODE != 3) __syncthreads();
    if constexpr (MODE == 0) acc[0][0][0] += 1.0f;
  }
  clock.stop(t);
  uint32_t fold = chain;
  for (int j = 0; j < kLists; ++j)
    for (int n = 0; n < 4; ++n)
      for (int i = 0; i < 4; ++i) fold ^= lab::bits(acc[j][n][i]);
  for (int d = 0; d < 2; ++d)
    for (int mt = 0; mt < 2; ++mt) {
      fold ^= outp[d][mt][0] ^ outp[d][mt][1];
      for (int kk = 0; kk < 4; ++kk) fold ^= a[d][mt][kk][0] ^ a[d][mt][kk][3];
    }
  lab::sink(out, fold);
}

constexpr int kWindows = 200;

template <int MODE, int SIZE, int PAD>
void variant(const lab::Bench& b, const uint8_t* src) {
  const std::string name = lab::kernel("window", {MODE, SIZE, PAD});
  if (MODE != 0) b.expect(name, "HMMA.16816.F32", ">=", 32);
  if (MODE != 0) b.expect(name, "HMMA.16816.F16", ">=", 16);
  if (MODE != 1 && MODE != 3) b.expect(name, "LDSM", ">=", 16);
  if (MODE != 1 && MODE != 3) b.expect(name, "STS", ">=", 64);
  if (b.expect_only) return;
  const int bytes = kPublishBytes + 256 + kReadRows * kReadRowBytes;
  LAB_CHECK(cudaFuncSetAttribute(window<MODE, SIZE, PAD>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] { window<MODE, SIZE, PAD><<<b.sms, kThreads, bytes>>>(kWindows, src, b.times, b.sinks); },
      [&] { return b.span(8) / kWindows; });
  b.record(std::string(kModeName[MODE]) + (MODE == 0 ? "" : std::string(" / ") + kSizeName[SIZE])
               + (PAD ? " / padded" : ""),
           "cycles a window", s);
}

template <int SIZE>
void for_size(const lab::Bench& b, const uint8_t* src) {
  variant<1, SIZE, 0>(b, src);
  variant<2, SIZE, 0>(b, src);
  variant<2, SIZE, 1>(b, src);
  variant<3, SIZE, 0>(b, src);
  variant<4, SIZE, 0>(b, src);
  variant<5, SIZE, 0>(b, src);
  variant<6, SIZE, 1>(b, src);
}

}  // namespace state_publish

int main(int argc, char** argv) {
  using namespace state_publish;
  lab::Bench b(argc, argv);
  uint8_t* src = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&src, static_cast<size_t>(kSourceRows) * kReadRowBytes));
    LAB_CHECK(cudaMemset(src, 0x3c, static_cast<size_t>(kSourceRows) * kReadRowBytes));
  }
  variant<0, 0, 0>(b, src);
  variant<0, 0, 1>(b, src);
  for_size<0>(b, src);
  for_size<1>(b, src);
}
