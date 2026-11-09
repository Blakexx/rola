// measure/lab/experiments/row_span.cu -- QUESTION: how fast does the facts pass's row stream read DRAM when a work
// item owns a wider span of a level's digits? A level's rows are [T, H, width] u8 -- a token's head's row `width`
// contiguous bytes, heads side by side -- and today an item owns 32 digits, one 32-byte piece of each token's row.
// METHOD: the facts pass's grid and ring (two blocks of eight warps an SM, a warp's ring four 1 KB stages, three in
// flight, one cp.async quantum of 16 bytes a lane twice a stage), items taken in order (heads, then the level-side,
// then the span innermost, as the facts pass deals its units); every item 64 KB, eight stages a warp, a stage
// `1024 / S` tokens of an S-digit span, each token's S bytes one run. Read only: each stage's bytes XORed into a sink.
// Against a flat stream of the same bytes and grid (each stage 1 KB contiguous): the rate of each span over it. The
// same at one block of sixteen warps an SM, and one of eight.
// see docs/internals/tools/lab.md#row-span
#include "lab.cuh"

namespace row_span {

constexpr int kStages = 4, kStageBytes = 1024, kItemStages = 8;
constexpr int kTokens = 65536, kHeads = 4, kWidth = 256, kLevelSides = 2;
constexpr long long kLevelBytes = static_cast<long long>(kTokens) * kHeads * kWidth;
constexpr long long kBytes = kLevelBytes * kLevelSides;
template <int WARPS>
constexpr int kItems = static_cast<int>(kBytes / (WARPS * kItemStages * kStageBytes));

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

//: STAGE `k` OF A WARP'S ITEM: its two quanta a lane. SPAN 0 is the flat stream (the stage 1 KB contiguous); else an
//: S-digit span: quantum q token q / (S / 16) of the stage's tokens, at its row's span bytes. APART: the span
//: outermost in the item order (a row's spans read a whole sweep apart), else innermost
template <int SPAN, int WARPS, bool APART = false>
__device__ __forceinline__ void issue(const uint8_t* rows, uint32_t ring, int item, int warp, int k, int lane) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int q = lane + 32 * h;
    const uint8_t* src;
    if constexpr (SPAN == 0) {
      src = rows + (static_cast<long long>(item) * WARPS + warp) * kItemStages * kStageBytes + k * kStageBytes
            + 16 * q;
    } else {
      constexpr int kQuanta = SPAN / 16, kStageTokens = kStageBytes / SPAN, kSpans = kWidth / SPAN;
      constexpr int kItemTokens = WARPS * kItemStages * kStageTokens;
      //: item -> (token range, head, level-side, span), the span innermost
      constexpr int kRest = kItems<WARPS> / kSpans;
      const int span = APART ? item / kRest : item % kSpans, rest = APART ? item % kRest : item / kSpans;
      const int ls = rest % kLevelSides, head = rest / kLevelSides % kHeads, range = rest / kLevelSides / kHeads;
      const int t = range * kItemTokens + (warp * kItemStages + k) * kStageTokens + q / kQuanta;
      src = rows + ls * kLevelBytes + (static_cast<long long>(t) * kHeads + head) * kWidth + span * SPAN
            + 16 * (q % kQuanta);
    }
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(ring + (k % kStages) * kStageBytes + 16 * q),
                 "l"(src));
  }
}

template <int SPAN, int WARPS, int BLOCKS, bool APART = false>
__global__ void __launch_bounds__(WARPS * 32, BLOCKS) stream(const uint8_t* rows, long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = lab::lane_id(), warp = lab::warp_id();
  const uint32_t ring = lab::smem(smem + warp * kStages * kStageBytes);
  uint32_t sink = 0u;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int item = static_cast<int>(blockIdx.x); item < kItems<WARPS>; item += static_cast<int>(gridDim.x)) {
#pragma unroll
    for (int k = 0; k < kStages - 1; ++k) {
      issue<SPAN, WARPS, APART>(rows, ring, item, warp, k, lane);
      commit();
    }
#pragma unroll
    for (int k = 0; k < kItemStages; ++k) {
      if (k + kStages - 1 < kItemStages) issue<SPAN, WARPS, APART>(rows, ring, item, warp, k + kStages - 1, lane);
      commit();
      wait<kStages - 1>();
      __syncwarp();
      const uint4 a = *reinterpret_cast<const uint4*>(smem + warp * kStages * kStageBytes
                                                      + (k % kStages) * kStageBytes + 32 * lane);
      const uint4 b = *reinterpret_cast<const uint4*>(smem + warp * kStages * kStageBytes
                                                      + (k % kStages) * kStageBytes + 32 * lane + 16);
      sink ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
      __syncwarp();
    }
  }
  clock.stop(t);
  lab::sink(out, sink);
}

template <int SPAN, int WARPS, int BLOCKS, bool APART = false>
void variant(const lab::Bench& b, const uint8_t* rows, double& flat) {
  std::string name = lab::kernel("stream", {SPAN, WARPS, BLOCKS});
  name = name.substr(0, name.size() - 3) + ", \\(bool\\)" + (APART ? "1" : "0") + ">\\(";
  b.expect(name, "LDGSTS", ">=", 2);
  if (b.expect_only) return;
  const int blocks = BLOCKS * b.sms, bytes = WARPS * kStages * kStageBytes;
  LAB_CHECK(cudaFuncSetAttribute(stream<SPAN, WARPS, BLOCKS, APART>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes));
  const lab::Samples s = b.variant(
      [&] { stream<SPAN, WARPS, BLOCKS, APART><<<blocks, WARPS * 32, bytes>>>(rows, b.times, b.sinks); },
      [&] { return static_cast<double>(kBytes) / b.span(WARPS, blocks); });
  const std::string label = (SPAN == 0 ? std::string("flat stream") : std::to_string(SPAN) + "-digit span")
                            + " / " + std::to_string(BLOCKS) + " x " + std::to_string(WARPS) + " warps an SM"
                            + (APART ? " / spans apart" : "");
  b.record(label, "bytes a cycle", s);
  if (SPAN == 0) flat = s.median();
  lab::Samples over;
  for (double v : s.v) over.v.push_back(v / flat);
  b.record(label + " / flat", "x", over);
}

}  // namespace row_span

int main(int argc, char** argv) {
  using namespace row_span;
  lab::Bench b(argc, argv);
  uint8_t* rows = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&rows, kBytes));
    LAB_CHECK(cudaMemset(rows, 0x5a, kBytes));
  }
  double flat = 0.0;
  variant<0, 8, 2>(b, rows, flat);
  variant<32, 8, 2>(b, rows, flat);
  variant<64, 8, 2>(b, rows, flat);
  variant<128, 8, 2>(b, rows, flat);
  variant<256, 8, 2>(b, rows, flat);
  variant<0, 16, 1>(b, rows, flat);
  variant<32, 16, 1>(b, rows, flat);
  variant<64, 16, 1>(b, rows, flat);
  variant<128, 16, 1>(b, rows, flat);
  variant<128, 16, 1, true>(b, rows, flat);
  variant<64, 16, 1, true>(b, rows, flat);
  variant<256, 16, 1>(b, rows, flat);
  variant<0, 8, 1>(b, rows, flat);
  variant<128, 8, 1>(b, rows, flat);
  variant<256, 8, 1>(b, rows, flat);
}
