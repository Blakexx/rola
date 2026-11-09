// measure/lab/experiments/mma_pairing.cu -- QUESTION: two warps a scheduler each run a fold-like step of 32
// fp16-accumulate HMMAs (8 lists x 4 n-tiles); what does one more atom of each kind cost, and how stable is the pair
// from launch to launch? METHOD: every warp of a 256-thread block runs the step `kIters` times; a sample is the mean
// warp's cycles a step in one launch, and the samples' spread is the phase lock. see docs/internals/tools/lab.md#mma-pairing
#include "lab.cuh"

namespace mma_pairing {

constexpr int kModes = 4;
const char* kModeName[kModes] = {"32 fp16-acc", "+ 1 fresh fp32-acc + 4 FADD", "+ 1 fp16-acc",
                                 "+ 1 chained fp32-acc"};
//: each mode's floor, cycles a step a warp at two warps a scheduler (fp16-acc 16, fp32-acc 32 cycles a scheduler)
const double kFloor[kModes] = {1024, 1088, 1056, 1088};

template <int MODE>
__global__ void __launch_bounds__(256, 1) pairing(int iters, long long* t, int* out) {
  const int lane = lab::lane_id();
  uint32_t part[8][4][2] = {}, extra[2] = {};
  float m[4] = {}, chained[4] = {};
  uint32_t a[4], b[4][2];
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (0x00010001u * (lane + i));
  for (int n = 0; n < 4; ++n) {
    b[n][0] = 0x3c003c00u ^ (lane + n);
    b[n][1] = b[n][0] ^ 0x00100010u;
  }
  lab::WarpClock clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    if constexpr (MODE == 1) {
      float d[4];
      lab::mma_f16_f32_fresh(d, a, b[0][0], b[0][1]);
      for (int i = 0; i < 4; ++i) m[i] += d[i];
    }
    if constexpr (MODE == 2) lab::mma_f16_f16(extra, a, b[0][0], b[0][1]);
    if constexpr (MODE == 3) lab::mma_f16_f32(chained, a, b[0][0], b[0][1]);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int n = 0; n < 4; ++n) lab::mma_f16_f16(part[j][n], a, b[n][0], b[n][1]);
  }
  clock.stop(t);
  uint32_t fold = extra[0] ^ lab::bits(m[0]) ^ lab::bits(chained[0]);
  for (int j = 0; j < 8; ++j)
    for (int n = 0; n < 4; ++n) fold ^= part[j][n][0];
  lab::sink(out, fold);
}

constexpr int kIters = 4000;

template <int MODE>
void mode(const lab::Bench& b) {
  const std::string name = lab::kernel("pairing", {MODE});
  b.expect(name, "HMMA.16816.F16", ">=", MODE == 2 ? 33 : 32);
  if (MODE == 1 || MODE == 3) b.expect(name, "HMMA.16816.F32", ">=", 1);
  if (b.expect_only) return;
  const lab::Samples s = b.variant([&] { pairing<MODE><<<b.sms, 256>>>(kIters, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(8), 8, 0, 8) / kIters; });
  b.record(kModeName[MODE], "cycles a step", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / kFloor[MODE]);
  b.record(std::string(kModeName[MODE]) + " / floor", "x", over);
}

}  // namespace mma_pairing

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  mma_pairing::mode<0>(b);
  mma_pairing::mode<1>(b);
  mma_pairing::mode<2>(b);
  mma_pairing::mode<3>(b);
}
