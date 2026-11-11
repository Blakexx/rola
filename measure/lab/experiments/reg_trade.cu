// measure/lab/experiments/reg_trade.cu -- QUESTION: two warpgroups of a CTA trade registers with `setmaxnreg`, one
// down and one up; when does the compiler honour it, what does the trade cost, and does state held in registers
// across it survive? METHOD: a CTA of two warpgroups, launch bounds of two CTAs an SM (128 registers a thread at
// entry). Both fold one state (an m64n16 wgmma) into registers; traded, warpgroup 0 lowers itself to 24 and writes
// shared memory while warpgroup 1 raises itself to 232 -- what the other released -- and reads that state out into
// five accumulator sets (160 fp32 a thread); untraded, both read out into the three sets the entry count holds.
// The trade is clocked alone; every accumulator is checked against the host's.
// see docs/internals/tools/lab.md#reg-trade
#include "lab.cuh"

namespace reg_trade {

constexpr int kM = 64, kK = 16, kFoldN = 16, kReadN = 64, kGroupThreads = 128, kThreads = 2 * kGroupThreads;
//: the accumulator sets a readout holds: what the entry count leaves untraded, and what the raised count holds
constexpr int kSetsCapped = 3, kSetsRaised = 5, kSetsMax = kSetsRaised;
//: the trade, balanced about the entry count: 104 a thread released, 104 acquired
constexpr int kEntry = 128, kLow = 24, kHigh = 232;
static_assert(kEntry - kLow == kHigh - kEntry, "a raise takes what the other warpgroup released");

constexpr int fold_a(int m, int k) { return (m * 3 + k * 5) % 5 - 2; }
constexpr int fold_b(int k, int n) { return (k * 7 + n * 3) % 5 - 2; }
constexpr int read_b(int k, int n) { return (k + 2 * n) % 3 - 1; }

constexpr int tile_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }
constexpr int tile_bytes(int lines) { return 256 * (lines / 8); }
constexpr int kFoldA = 0, kFoldB = kFoldA + tile_bytes(kM), kReadB = kFoldB + tile_bytes(kFoldN),
              kScratch = kReadB + tile_bytes(kReadN), kSmem = kScratch + 4 * kGroupThreads;

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

//: a thread's outputs: its sets' accumulators
constexpr int kOut = 32 * kSetsMax;

//: THE READOUT: the state truncated into register A, `kSets` accumulator sets a step, stored at the end
template <int kSets>
__device__ __forceinline__ void readout(float (&s)[8], uint64_t rb, float* out, int steps) {
  uint32_t a[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) a[w] = lab::bf16x2_trunc(s[2 * w], s[2 * w + 1]);
  float d[kSets][32] = {};
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    lab::wg_fence();
#pragma unroll
    for (int k = 0; k < kSets; ++k) lab::wg_bf16_n64_rs(d[k], a, rb);
    lab::wg_commit();
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int k = 0; k < kSets; ++k)
#pragma unroll
    for (int i = 0; i < 32; ++i) lab::wg_hold(d[k][i]);
  float* mine = out + (static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x) * kOut;
#pragma unroll
  for (int k = 0; k < kSets; ++k)
#pragma unroll
    for (int i = 0; i < 32; ++i) mine[32 * k + i] = d[k][i];
}

//: THE LOWERED WARPGROUP'S WORK: a word of shared memory a step
__device__ __forceinline__ void lowered(uint8_t* smem, int* sinks, int steps) {
  auto* scratch = reinterpret_cast<uint32_t*>(smem + kScratch);
  const int tid = threadIdx.x & (kGroupThreads - 1);
  uint32_t x = tid;
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    scratch[tid] = x;
    x = scratch[(tid + 1) & (kGroupThreads - 1)] + it;
  }
  lab::sink(sinks, x);
}

template <int TRADED>
__global__ void __launch_bounds__(kThreads, 2) trade(long long* times, long long* trade_times, int* sinks, float* out, int steps) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, group = tid >> 7;
  auto* tile = reinterpret_cast<uint16_t*>(smem);
#pragma unroll 1
  for (int i = tid; i < kM * kK; i += kThreads) tile[(kFoldA + tile_at(i / kK, i % kK)) / 2] = bf16_of(fold_a(i / kK, i % kK));
#pragma unroll 1
  for (int i = tid; i < kFoldN * kK; i += kThreads)
    tile[(kFoldB + tile_at(i / kK, i % kK)) / 2] = bf16_of(fold_b(i % kK, i / kK));
#pragma unroll 1
  for (int i = tid; i < kReadN * kK; i += kThreads)
    tile[(kReadB + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_b(i % kK, i / kK));
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  const uint64_t fa = lab::wg_desc(base + kFoldA, 8, 16), fb = lab::wg_desc(base + kFoldB, 8, 16),
                 rb = lab::wg_desc(base + kReadB, 8, 16);

  //: THE STATE, folded before the trade and read after it
  float s[8] = {};
  lab::wg_fence();
  lab::wg_bf16_n16_ss(s, fa, fb);
  lab::wg_commit();
  lab::wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 8; ++i) lab::wg_hold(s[i]);

  //: THE TRADE AND EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH: a join after a `setmaxnreg` compiles what follows it at
  //: the lower count
  lab::WarpClock swap;
  lab::WarpSpan clock;
  swap.start();
  if (TRADED && group == 0) {
    lab::regs_down<kLow>();
    swap.stop(trade_times);
    clock.start();
    lowered(smem, sinks, steps);
    clock.stop(times);
  } else {
    if constexpr (TRADED) lab::regs_up<kHigh>();
    swap.stop(trade_times);
    clock.start();
    readout<TRADED ? kSetsRaised : kSetsCapped>(s, rb, out, steps);
    clock.stop(times);
  }
}

}  // namespace reg_trade

template <int TRADED>
void run(lab::Bench& b, long long* trade_times, float* out) {
  using namespace reg_trade;
  constexpr int kCheckSteps = 16, kSteps = 2048;
  constexpr int kSets = TRADED ? kSetsRaised : kSetsCapped;
  const std::string name = TRADED ? "traded: one warpgroup at 24, one at 232, five sets"
                                  : "untraded: both at the entry count, three sets";

  LAB_CHECK(cudaMemset(out, 0, sizeof(float) * b.sms * kThreads * kOut));
  trade<TRADED><<<b.sms, kThreads>>>(b.times, trade_times, b.sinks, out, kCheckSteps);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * kThreads * kOut);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  //: the host's: the state `s = A B` once, truncated to bf16 (exact: small integers), read out `steps` times a set
  std::vector<double> s(kM * kFoldN, 0.0), d(kM * kReadN, 0.0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kFoldN; ++n)
      for (int k = 0; k < kK; ++k) s[m * kFoldN + n] += fold_a(m, k) * fold_b(k, n);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kReadN; ++n)
      for (int k = 0; k < kK; ++k) d[m * kReadN + n] += kCheckSteps * s[m * kFoldN + k] * read_b(k, n);
  long long wrong = 0, checked = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = TRADED ? kGroupThreads : 0; tid < kThreads; ++tid) {
      const float* mine = got.data() + (static_cast<size_t>(blk) * kThreads + tid) * kOut;
      const int g = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, t = tid & 3;
      for (int i = 0; i < 32 * kSets; ++i, ++checked) {
        const int j = i % 32, m = g + 8 * ((j >> 1) & 1), n = 8 * (j >> 2) + 2 * t + (j & 1);
        wrong += mine[i] != static_cast<float>(d[m * kReadN + n]);
      }
    }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(checked) + " accumulators differ from the host's at "
              + std::to_string(kCheckSteps) + " steps");

  lab::Samples swap;
  const lab::Samples cycles = b.variant([&] { trade<TRADED><<<b.sms, kThreads>>>(b.times, trade_times, b.sinks, out, kSteps); }, [&] {
    std::vector<long long> h(static_cast<size_t>(b.sms) * (kThreads >> 5));
    LAB_CHECK(cudaMemcpy(h.data(), trade_times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    swap.v.push_back(lab::Bench::mean(h, kThreads >> 5, 0, kThreads >> 5));
    //: the readout's warps alone: warpgroup 1's span over its steps and sets
    std::vector<long long> t(static_cast<size_t>(b.sms) * (kThreads >> 5) * 2);
    LAB_CHECK(cudaMemcpy(t.data(), b.times, t.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    double sum = 0;
    for (int blk = 0; blk < b.sms; ++blk) {
      long long lo = t[2 * (blk * 8 + 4)], hi = t[2 * (blk * 8 + 4) + 1];
      for (int w = 5; w < 8; ++w) {
        lo = std::min(lo, t[2 * (blk * 8 + w)]);
        hi = std::max(hi, t[2 * (blk * 8 + w) + 1]);
      }
      sum += double(hi - lo);
    }
    return sum / b.sms / kSteps / kSets;
  });
  b.record(name + ", the readout", "cycles a wgmma", cycles);
  b.record(name + ", the trade", "cycles a warp", swap);
}

int main(int argc, char** argv) {
  using namespace reg_trade;
  lab::Bench b(argc, argv);
  for (int f = 0; f < 2; ++f) {
    const std::string k = lab::kernel("trade", {f});
    b.expect(k, "HGMMA", "==", 1 + (f ? kSetsRaised : kSetsCapped));
    b.expect(k, "USETMAXREG", "==", f ? 2 : 0);
    b.exempt(k, "a register-trade probe: the readout is its load, no kernel's step");
  }
  if (b.expect_only) return 0;
  float* out = nullptr;
  long long* trade_times = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kThreads * kOut));
  LAB_CHECK(cudaMalloc(&trade_times, sizeof(long long) * b.sms * (kThreads >> 5)));
  run<0>(b, trade_times, out);
  run<1>(b, trade_times, out);
  return 0;
}
