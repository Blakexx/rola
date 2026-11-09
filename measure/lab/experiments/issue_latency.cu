// measure/lab/experiments/issue_latency.cu -- QUESTION: how many cycles after an instruction issues can a dependent
// one issue -- a shared load to its use, a tensor MMA to the next MMA on its accumulator, an FFMA to its use (the
// instrument's probe)? METHOD: one warp a scheduler (4 a CTA, a CTA an SM); each repetition reads the SM clock, runs a
// chain of K links, each link's operand the one before's result (its first tied to the clock read by a run-time zero),
// and reads the clock again (in-order issue: after the K-th link has issued). The latency is the slope between K = 32
// and K = 64, the clock reads' own cost cancelled. see docs/internals/tools/lab.md#issue-latency
#include "lab.cuh"

namespace issue_latency {

constexpr int kReps = 256, kWarps = 4, kModes = 5;
const char* kModeName[kModes] = {"FFMA", "LDS.32 (address from the load)", "LDSM.x4 (address from the load)",
                                 "fp32-accumulate HMMA.16816 on one accumulator",
                                 "fp16-accumulate HMMA.16816 on one accumulator"};
const char* kOpcode[kModes] = {"FFMA", "LDS", "LDSM", "HMMA", "HMMA"};

__device__ __forceinline__ uint32_t clock32() {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory");
  return t;
}

//: MODE as `kModeName`; the 16-byte row holds its own shared address in every word, so a load from it returns the
//: next link's address
template <int MODE, int K>
__global__ void __launch_bounds__(kWarps * 32, 1) chain(int reps, const uint32_t* in, long long* t, int* out) {
  __shared__ alignas(16) uint32_t row[4];
  const uint32_t self = static_cast<uint32_t>(__cvta_generic_to_shared(row));
  if (threadIdx.x < 4) row[threadIdx.x] = self;
  __syncthreads();
  const uint32_t zero = static_cast<uint32_t>(reps) >> 31;
  const float one = __uint_as_float(in[0]), nil = __uint_as_float(in[1]);
  uint32_t a[4], b0 = in[4 + (threadIdx.x & 7)], b1 = in[12 + (threadIdx.x & 7)];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = in[i + 2];
  uint32_t fold = 0;
  long long cycles = 0;
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    const uint32_t t0 = clock32();
    if constexpr (MODE == 0) {
      float f = __uint_as_float(lab::bits(one) ^ (t0 & zero));
#pragma unroll
      for (int k = 0; k < K; ++k) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(one), "f"(nil));
      cycles += clock32() - t0;
      fold ^= lab::bits(f);
    } else if constexpr (MODE == 1) {
      uint32_t x = self ^ (t0 & zero);
#pragma unroll
      for (int k = 0; k < K; ++k) asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x)::"memory");
      cycles += clock32() - t0;
      fold ^= x;
    } else if constexpr (MODE == 2) {
      uint32_t x = self ^ (t0 & zero), y[3];
#pragma unroll
      for (int k = 0; k < K; ++k)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(x), "=r"(y[0]), "=r"(y[1]), "=r"(y[2])
                     : "r"(x)
                     : "memory");
      cycles += clock32() - t0;
      fold ^= x ^ y[0] ^ y[1] ^ y[2];
    } else if constexpr (MODE == 3) {
      float acc[4] = {__uint_as_float(t0 & zero), 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < K; ++k) lab::mma_f16_f32(acc, a, b0, b1);
      cycles += clock32() - t0;
      fold ^= lab::bits(acc[0]) ^ lab::bits(acc[3]);
    } else {
      uint32_t acc[2] = {t0 & zero, 0u};
#pragma unroll
      for (int k = 0; k < K; ++k) lab::mma_f16_f16(acc, a, b0, b1);
      cycles += clock32() - t0;
      fold ^= acc[0] ^ acc[1];
    }
    a[1] ^= fold & zero;
  }
  if (lab::lane_id() == 0) t[blockIdx.x * kWarps + lab::warp_id()] = cycles;
  lab::sink(out, fold);
}

template <int MODE, int K>
lab::Samples point(const lab::Bench& b, const uint32_t* in) {
  const std::string name = lab::kernel("chain", {MODE, K});
  b.expect(name, kOpcode[MODE], ">=", K);
  if (MODE >= 3) b.exempt(name, "a latency probe: one dependent chain, no step to budget");
  if (b.expect_only) return {};
  const std::string variant = std::string(kModeName[MODE]) + " / " + std::to_string(K) + " links";
  b.unchecked(variant, "a latency probe: its chain's value is a fixed point folded into a sink");
  const lab::Samples s = b.variant([&] { chain<MODE, K><<<b.sms, kWarps * 32>>>(kReps, in, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(kWarps), kWarps, 0, kWarps) / kReps; });
  b.record(variant, "cycles to issue", s);
  return s;
}

//: the latency, a sample a repetition: the slope between the 32- and 64-link chains
template <int MODE>
void latency(const lab::Bench& b, const uint32_t* in) {
  const lab::Samples s32 = point<MODE, 32>(b, in), s64 = point<MODE, 64>(b, in);
  if (b.expect_only) return;
  const std::string variant = std::string(kModeName[MODE]) + " / latency";
  b.unchecked(variant, "derived: the slope between the two chains' rows");
  lab::Samples slope;
  for (size_t r = 0; r < s32.v.size(); ++r) slope.v.push_back((s64.v[r] - s32.v[r]) / 32.0);
  b.record(variant, "cycles a link", slope);
}

template <int... M>
void sweep(const lab::Bench& b, const uint32_t* in, std::integer_sequence<int, M...>) {
  (latency<M>(b, in), ...);
}

}  // namespace issue_latency

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  uint32_t* in = nullptr;
  if (!b.expect_only) {
    uint32_t host[20];
    host[0] = 0x3f800000u;
    host[1] = 0u;
    for (int i = 2; i < 20; ++i) host[i] = 0x3c00u;
    LAB_CHECK(cudaMalloc(&in, sizeof(host)));
    LAB_CHECK(cudaMemcpy(in, host, sizeof(host), cudaMemcpyHostToDevice));
  }
  issue_latency::sweep(b, in, std::make_integer_sequence<int, issue_latency::kModes>{});
}
