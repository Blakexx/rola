// measure/lab/experiments/bulk_drain.cu -- QUESTION: a CTA's fp32 output panel leaves shared memory for a global
// sum other CTAs add into too; what does a token cost by one asynchronous bulk reduction a panel, by one a token's
// span, and by the warps' own reductions a value straight from registers? METHOD: a CTA an SM, 4 warps; a round
// writes a panel of 64 tokens x 64 channels fp32 (16 KB) and drains it -- one `cp.reduce.async.bulk` of the panel
// waited before the next round, the same over two panels with one drain in flight, one a token (64 threads issue
// theirs), or no panel and a `red.global.add.f32` a value; the panel's stores alone are the base. Every CTA adds
// into ONE global panel (the contended sum) or its own; the sums are checked exactly.
// see docs/internals/tools/lab.md#bulk-drain
#include "lab.cuh"

namespace bulk_drain {

constexpr int kForms = 5;
enum Form : int {
  kPanelOnly = 0,  // the panel's stores and the round's barriers, no drain
  kBulkPanel,      // one bulk add of the panel, waited
  kBulkTwoPanels,  // two panels in turn, one bulk add in flight
  kBulkToken,      // one bulk add a token's span, each from its own thread
  kWarpReds,       // no panel: a reduction a value from registers
};
const char* kFormName[kForms] = {"the panel's stores alone", "one bulk add a panel", "one bulk add a panel, one in flight",
                                 "one bulk add a token", "a reduction a value, no panel"};

constexpr int kTokens = 64, kChannels = 64, kThreads = 128, kPanel = kTokens * kChannels,
              kMine = kPanel / kThreads, kPanelBytes = kPanel * 4;

//: a panel element's value, a small integer: every sum exact in fp32
constexpr int value_of(int i) { return i % 7 + 1; }

template <int FORM>
__global__ void drain(long long* times, int* sinks, float* out, int rounds, int own) {
  __shared__ alignas(128) float panels[2 * kPanel];
  const int tid = threadIdx.x, warp = tid >> 5;
  float* const dst = out + (own ? static_cast<long long>(blockIdx.x) * kPanel : 0);
  const uint32_t base = lab::smem(panels);
  float v[kMine];
#pragma unroll
  for (int i = 0; i < kMine; ++i) v[i] = static_cast<float>(value_of(i * kThreads + tid));

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    constexpr int kPanels = FORM == kBulkTwoPanels ? 2 : 1;
    const int at = (r % kPanels) * kPanel;
    if constexpr (FORM == kWarpReds) {
      //: THE REDUCTIONS: this thread's values, each to its global word
#pragma unroll
      for (int i = 0; i < kMine; ++i) lab::red_add_f32(dst + i * kThreads + tid, v[i]);
    } else {
      //: THE PANEL: this thread's values into the round's panel -- word `128 i + tid`, a warp's 32 lanes on 32 banks
      //: -- then the edge the bulk engine reads them behind
#pragma unroll
      for (int i = 0; i < kMine; ++i) panels[at + i * kThreads + tid] = v[i];
      if constexpr (FORM != kPanelOnly) lab::bulk_fence();
      __syncthreads();
      //: THE BASE'S READER: a neighbour's word a round, so the panel's stores are live with no drain
      if constexpr (FORM == kPanelOnly) seen ^= lab::bits(panels[at + (tid + 1) % kThreads]) + r;
      //: THE DRAIN
      if constexpr (FORM == kBulkPanel || FORM == kBulkTwoPanels) {
        if (tid == 0) {
          lab::bulk_add_f32(dst, base + 4 * at, kPanelBytes);
          lab::bulk_commit();
          if constexpr (FORM == kBulkPanel) lab::bulk_wait<0>();
          if constexpr (FORM == kBulkTwoPanels) lab::bulk_wait<1>();
        }
      }
      if constexpr (FORM == kBulkToken) {
        if (tid < kTokens) {
          lab::bulk_add_f32(dst + tid * kChannels, base + 4 * (at + tid * kChannels), kChannels * 4);
          lab::bulk_commit();
          lab::bulk_wait<0>();
        }
      }
      __syncthreads();
    }
  }
  if constexpr (FORM == kBulkTwoPanels) {
    if (tid == 0) lab::bulk_wait<0>();
  }
  clock.stop(times);
  lab::sink(sinks, seen);
}

}  // namespace bulk_drain

template <int FORM>
void run(lab::Bench& b, float* out, int own) {
  using namespace bulk_drain;
  constexpr int kCheckRounds = 16, kRounds = 2048;
  const std::string name = std::string(kFormName[FORM]) + (own ? ", each CTA its own sum" : ", one sum for every CTA");
  const size_t words = static_cast<size_t>(own ? b.sms : 1) * kPanel;

  if (FORM != kPanelOnly) {
    LAB_CHECK(cudaMemset(out, 0, words * sizeof(float)));
    drain<FORM><<<b.sms, kThreads>>>(b.times, b.sinks, out, kCheckRounds, own);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(words);
    LAB_CHECK(cudaMemcpy(got.data(), out, words * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (size_t i = 0; i < words; ++i)
      wrong += got[i] != static_cast<float>(value_of(static_cast<int>(i % kPanel)) * kCheckRounds * (own ? 1 : b.sms));
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(words) + " sums differ from the exact ones at "
                + std::to_string(kCheckRounds) + " rounds");
  } else {
    b.unchecked(name, "the base: it writes shared memory alone");
  }

  const lab::Samples cycles = b.variant([&] { drain<FORM><<<b.sms, kThreads>>>(b.times, b.sinks, out, kRounds, own); },
                                        [&] { return b.span(kThreads >> 5) / kRounds / kTokens; });
  b.record(name, "cycles a token", cycles);
}

int main(int argc, char** argv) {
  using namespace bulk_drain;
  lab::Bench b(argc, argv);
  for (int f = 0; f < kForms; ++f) {
    const std::string k = lab::kernel("drain", {f});
    b.expect(k, "UBLKRED", "==", f == kBulkPanel || f == kBulkTwoPanels || f == kBulkToken ? 1 : 0);
    b.expect(k, "REDG", "==", f == kWarpReds ? kMine : 0);
    b.expect(k, "STS", f == kWarpReds ? "==" : ">=", f == kWarpReds ? 0 : kMine);
  }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kPanel));
  for (int own = 0; own <= 1; ++own) {
    run<kPanelOnly>(b, out, own);
    run<kBulkPanel>(b, out, own);
    run<kBulkTwoPanels>(b, out, own);
    run<kBulkToken>(b, out, own);
    run<kWarpReds>(b, out, own);
  }
  return 0;
}
