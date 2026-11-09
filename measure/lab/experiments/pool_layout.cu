// measure/lab/experiments/pool_layout.cu -- QUESTION: the fold's pool rows are 160 bytes, so the 8 rows of one ldmatrix
// matrix (one 16-byte column) land on 4 bank groups -- a 2-way conflict -- unless half the rows move that column by 16
// bytes, while cp.async folds a lane pair's 32-byte global sector into one L2 request only when the shared destination
// keeps the sector's 32-byte phase. Does swapping the two 16-byte halves inside each aligned 32-byte pair (on rows
// with bit 2 set) keep the fill's sectors and make the steps' reads conflict-free? METHOD: the owner part's fill (a
// quad a token: the 64-byte run and the 64-byte values in 16-byte copies, four 4-byte factors from four planes, a
// 16-byte dense chunk) into 160-byte rows plain, 160-byte rows with the halves swapped, or 176-byte rows (odd rows 16
// bytes off phase); and the steps' reads of a pool row (two ldmatrix.x4.trans over the values, four coefficient bytes a
// lane). Timing: the fill's span a pass, the reads' span a step; `tools/lab.py ncu` reads the fill's L2 sectors and the
// reads' bank conflicts. see docs/internals/tools/lab.md#pool-layout
#include "lab.cuh"

namespace pool_layout {

constexpr int kRows = 128, kThreads = 256, kWindow = 512, kSteps = kRows / 16, kLists = 8;

//: LAYOUT 0 160-byte rows; 1 160-byte rows, a row with bit 2 set swapping the 16-byte halves of each 32-byte pair; 2
//: 176-byte rows
constexpr int kLayouts = 3;
const char* kLayoutName[kLayouts] = {"160 B rows", "160 B rows, halves swapped", "176 B rows"};
constexpr int kPoolBytes = kRows * 176;

//: the byte offset of 16-byte chunk `c` of row `r`
template <int LAYOUT>
__device__ __forceinline__ int chunk_at(int r, int c) {
  if constexpr (LAYOUT == 0) return r * 160 + 16 * c;
  if constexpr (LAYOUT == 1) return r * 160 + 16 * (c ^ ((r >> 2) & 1));
  return r * 176 + 16 * c;
}

//: SOURCE 0 400 windows (33 MB: DRAM), 1 24 windows (2 MB: L2)
constexpr int kWindowsOf[2] = {400, 24};
const char* kSourceName[2] = {"33 MB source", "2 MB source"};

struct Source {
  const uint8_t *run, *values, *factor, *dense;
  int tokens;
};

template <int LAYOUT, int SOURCE>
__global__ void __launch_bounds__(kThreads, 1) fill(int passes, Source src, long long* t,
                                                    int* out) {
  __shared__ __align__(128) uint8_t pool[kPoolBytes];
  const int tid = static_cast<int>(threadIdx.x), q = tid & 3;
  const uint32_t base = lab::smem(pool);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    const int64_t win = static_cast<int64_t>((blockIdx.x * 7 + x) % kWindowsOf[SOURCE]) * kWindow;
#pragma unroll 1
    for (int r = tid >> 2; r < kRows; r += kThreads / 4) {
      const int64_t tg = win + ((r * 37 + 128 * (x & 3)) & (kWindow - 1));
      lab::cp_async16(base + chunk_at<LAYOUT>(r, q), src.run + tg * 64 + 16 * q);
      lab::cp_async16(base + chunk_at<LAYOUT>(r, 4 + q), src.values + tg * 64 + 16 * q);
      lab::cp_async4(base + chunk_at<LAYOUT>(r, 8) + 4 * q,
                     src.factor + (q * static_cast<int64_t>(src.tokens) + tg) * 4);
      if (q == 0) lab::cp_async16(base + chunk_at<LAYOUT>(r, 9), src.dense + tg * 16);
    }
    lab::cp_wait_all();
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, pool[tid]);
}

template <int LAYOUT>
__global__ void __launch_bounds__(kThreads, 1) read(int steps, long long* t, int* out) {
  __shared__ __align__(128) uint8_t pool[kPoolBytes];
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id();
  for (int i = tid; i < kPoolBytes; i += kThreads) pool[i] = static_cast<uint8_t>(i * 7);
  __syncthreads();
  const uint32_t base = lab::smem(pool);
  const int r = lane & 7, q = lane >> 3, tq = lane & 3,
            byte = lab::warp_id() * kLists + (lane >> 2);
  uint32_t fold = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int s = 0; s < steps; ++s) {
    const int e0 = (s & (kSteps - 1)) * 16, rb = e0 + 8 * (q & 1) + r;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t x[4];
      lab::ldsm_t(x, base + chunk_at<LAYOUT>(rb, 4 + 2 * h + (q >> 1)));
      fold ^= x[0] ^ x[1] ^ x[2] ^ x[3];
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int row = e0 + 2 * tq + (k & 1) + 8 * (k >> 1);
      fold += pool[chunk_at<LAYOUT>(row, byte >> 4) + (byte & 15)];
    }
  }
  clock.stop(t);
  lab::sink(out, fold);
}

constexpr int kPasses = 400, kReadSteps = 20000;

template <int LAYOUT, int SOURCE>
void fill_variant(const lab::Bench& b, const Source& src) {
  const std::string name = lab::kernel("fill", {LAYOUT, SOURCE});
  b.expect(name, "LDGSTS", ">=", 4);
  if (b.expect_only) return;
  Source s = src;
  s.tokens = kWindowsOf[0] * kWindow;
  const lab::Samples v =
      b.variant([&] { fill<LAYOUT, SOURCE><<<b.sms, kThreads>>>(kPasses, s, b.times, b.sinks); },
                [&] { return b.span(8) / kPasses; });
  b.record(std::string("fill / ") + kLayoutName[LAYOUT] + " / " + kSourceName[SOURCE],
           "cycles a pass", v);
}

template <int LAYOUT>
void read_variant(const lab::Bench& b) {
  const std::string name = lab::kernel("read", {LAYOUT});
  b.expect(name, "LDSM", ">=", 2);
  if (b.expect_only) return;
  const lab::Samples v =
      b.variant([&] { read<LAYOUT><<<b.sms, kThreads>>>(kReadSteps, b.times, b.sinks); },
                [&] { return b.span(8) / kReadSteps; });
  b.record(std::string("read / ") + kLayoutName[LAYOUT], "cycles a step (8 warps)", v);
}

}  // namespace pool_layout

int main(int argc, char** argv) {
  using namespace pool_layout;
  lab::Bench b(argc, argv);
  Source src{};
  if (!b.expect_only) {
    const int tokens = kWindowsOf[0] * kWindow;
    uint8_t* buf = nullptr;
    LAB_CHECK(cudaMalloc(&buf, static_cast<size_t>(tokens) * 160));
    LAB_CHECK(cudaMemset(buf, 0x3c, static_cast<size_t>(tokens) * 160));
    src =
        Source{buf, buf + static_cast<size_t>(tokens) * 64, buf + static_cast<size_t>(tokens) * 128,
               buf + static_cast<size_t>(tokens) * 144, tokens};
  }
  fill_variant<0, 0>(b, src);
  fill_variant<1, 0>(b, src);
  fill_variant<2, 0>(b, src);
  fill_variant<0, 1>(b, src);
  fill_variant<1, 1>(b, src);
  fill_variant<2, 1>(b, src);
  read_variant<0>(b);
  read_variant<1>(b);
  read_variant<2>(b);
}
