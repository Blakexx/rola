// measure/lab/experiments/wgmma_rate.cu -- QUESTION: what is one m64n64k16 wgmma's own time on the tensor pipe,
// apart from its commit group's round trip, and do two warpgroups of a CTA share that pipe? METHOD: one or two
// warpgroups a CTA, a CTA an SM; a step issues K wgmmas into K accumulator sets in ONE commit group and waits for
// it, K = 1, 2, 4, 6, with A in registers or in shared memory and every operand descriptor held in a register (so
// the step holds no descriptor arithmetic). The slope of a step's cycles over K is a wgmma's own time; the intercept
// is the group's round trip. Every accumulator is checked against the host's.
// see docs/internals/tools/lab.md#wgmma-rate
#include "lab.cuh"

namespace wgmma_rate {

constexpr int kM = 64, kK = 16, kN = 64, kGroupThreads = 128, kMaxSets = 6;

constexpr int read_a(int m, int k) { return (m + k) % 7 - 3; }
constexpr int read_b(int k, int n) { return (k + 2 * n) % 3 - 1; }
constexpr int tile_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }
constexpr int kTileA = 0, kTileB = 2048, kSmem = 4096;

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

template <int kSets, int SHARED>
__global__ void __launch_bounds__(2 * kGroupThreads, 1) rate(long long* times, float* out, int steps) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, lane = tid & 31;
  auto* tile = reinterpret_cast<uint16_t*>(smem);
#pragma unroll 1
  for (int i = tid; i < kM * kK; i += blockDim.x) tile[(kTileA + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_a(i / kK, i % kK));
#pragma unroll 1
  for (int i = tid; i < kN * kK; i += blockDim.x) tile[(kTileB + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_b(i % kK, i / kK));
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  //: THE DESCRIPTORS AND REGISTER A, held: nothing of them is derived inside the step
  const uint64_t da = lab::held(lab::wg_desc(base + kTileA, 8, 16)), db = lab::held(lab::wg_desc(base + kTileB, 8, 16));
  uint32_t a[4];
  const int g = 16 * ((tid >> 5) & 3) + lane / 4, kt = lane & 3;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int m = g + 8 * (w & 1), k = 8 * (w >> 1) + 2 * kt;
    a[w] = lab::held(bf16_of(read_a(m, k)) | static_cast<uint32_t>(bf16_of(read_a(m, k + 1))) << 16);
  }
  float d[kSets][32] = {};

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    lab::wg_fence();
#pragma unroll
    for (int k = 0; k < kSets; ++k) {
      if constexpr (SHARED) lab::wg_bf16_n64_ss(d[k], da, db);
      else lab::wg_bf16_n64_rs(d[k], a, db);
    }
    lab::wg_commit();
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int k = 0; k < kSets; ++k)
#pragma unroll
    for (int i = 0; i < 32; ++i) lab::wg_hold(d[k][i]);
  clock.stop(times);
  float* mine = out + (static_cast<long long>(blockIdx.x) * blockDim.x + tid) * 32 * kMaxSets;
#pragma unroll
  for (int k = 0; k < kSets; ++k)
#pragma unroll
    for (int i = 0; i < 32; ++i) mine[32 * k + i] = d[k][i];
}

}  // namespace wgmma_rate

template <int kSets, int SHARED>
void run(lab::Bench& b, float* out, int groups) {
  using namespace wgmma_rate;
  constexpr int kCheckSteps = 16, kSteps = 4096;
  const int threads = kGroupThreads * groups, warps = threads / 32;
  const std::string name = std::to_string(kSets) + (kSets == 1 ? " wgmma" : " wgmmas") + " a group, A "
                           + (SHARED ? "shared" : "registers") + (groups == 1 ? ", one warpgroup" : ", two warpgroups");
  rate<kSets, SHARED><<<b.sms, threads>>>(b.times, out, kCheckSteps);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * threads * 32 * kMaxSets);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<double> d(kM * kN, 0.0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n)
      for (int k = 0; k < kK; ++k) d[m * kN + n] += kCheckSteps * read_a(m, k) * read_b(k, n);
  long long wrong = 0, checked = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = 0; tid < threads; ++tid) {
      const float* mine = got.data() + (static_cast<size_t>(blk) * threads + tid) * 32 * kMaxSets;
      const int g = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, t = tid & 3;
      for (int i = 0; i < 32 * kSets; ++i, ++checked) {
        const int j = i % 32, m = g + 8 * ((j >> 1) & 1), n = 8 * (j >> 2) + 2 * t + (j & 1);
        wrong += mine[i] != static_cast<float>(d[m * kN + n]);
      }
    }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(checked) + " accumulators differ from the host's at "
              + std::to_string(kCheckSteps) + " steps");
  const lab::Samples cycles = b.variant([&] { rate<kSets, SHARED><<<b.sms, threads>>>(b.times, out, kSteps); },
                                        [&] { return b.span(warps) / kSteps; });
  b.record(name, "cycles a step", cycles);
}

int main(int argc, char** argv) {
  using namespace wgmma_rate;
  lab::Bench b(argc, argv);
  for (int sets : {1, 2, 4, 6})
    for (int shared = 0; shared < 2; ++shared) {
      const std::string k = lab::kernel("rate", {sets, shared});
      b.expect(k, "HGMMA", "==", sets);
      b.expect(k, "ULOP3", "<=", 4);
      b.exempt(k, "a rate probe: the tensor pipe's own time a wgmma, no kernel's step");
    }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * 2 * kGroupThreads * 32 * kMaxSets));
  for (int groups = 1; groups <= 2; ++groups)
    for (int shared = 0; shared < 2; ++shared) {
      if (shared) {
        run<1, 1>(b, out, groups);
        run<2, 1>(b, out, groups);
        run<4, 1>(b, out, groups);
        run<6, 1>(b, out, groups);
      } else {
        run<1, 0>(b, out, groups);
        run<2, 0>(b, out, groups);
        run<4, 0>(b, out, groups);
        run<6, 0>(b, out, groups);
      }
    }
  return 0;
}
