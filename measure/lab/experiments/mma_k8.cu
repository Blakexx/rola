// measure/lab/experiments/mma_k8.cu -- QUESTION: does an m16n8k8 MMA cost half an m16n8k16 on the tensor pipe, so
// a fold step of 8 tokens costs half a 16-token step? METHOD: every warp of a 256-thread block (two a scheduler) runs
// 32 independent MMAs a loop into 32 accumulators, fp32 accumulate (the fold's) and fp16 accumulate, at k16 and k8;
// a sample is the mean warp's cycles an MMA in one launch. see docs/internals/tools/lab.md#mma-k8
#include "lab.cuh"

namespace mma_k8 {

constexpr int kModes = 4;
const char* kModeName[kModes] = {"m16n8k16, fp32 accumulate", "m16n8k8, fp32 accumulate",
                                 "m16n8k16, fp16 accumulate", "m16n8k8, fp16 accumulate"};
constexpr int kMmas = 32, kIters = 2000;

__device__ __forceinline__ void k8_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ void k8_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(b));
}

template <int MODE>
__global__ void __launch_bounds__(256, 1) mmas(int iters, long long* t, int* out) {
  const int lane = lab::lane_id();
  float f[kMmas][4] = {};
  uint32_t h[kMmas][2] = {};
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (0x00010001u * (lane + i));
  b[0] = 0x3c003c00u ^ lane;
  b[1] = b[0] ^ 0x00100010u;
  lab::WarpClock clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < kMmas; ++j) {
      if constexpr (MODE == 0) lab::mma_f16_f32(f[j], a, b[0], b[1]);
      if constexpr (MODE == 1) k8_f32(f[j], a, b[0]);
      if constexpr (MODE == 2) lab::mma_f16_f16(h[j], a, b[0], b[1]);
      if constexpr (MODE == 3) k8_f16(h[j], a, b[0]);
    }
  }
  clock.stop(t);
  uint32_t fold = 0u;
  for (int j = 0; j < kMmas; ++j) fold ^= lab::bits(f[j][0]) ^ h[j][0];
  lab::sink(out, fold);
}

template <int MODE>
void mode(const lab::Bench& b) {
  const std::string name = lab::kernel("mmas", {MODE});
  const char* op[kModes] = {"HMMA.16816.F32", "HMMA.1688.F32", "HMMA.16816.F16", "HMMA.1688.F16"};
  b.expect(name, op[MODE], ">=", kMmas);
  if (b.expect_only) return;
  const lab::Samples s = b.variant([&] { mmas<MODE><<<b.sms, 256>>>(kIters, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(8), 8, 0, 8) / (kIters * kMmas); });
  b.record(kModeName[MODE], "cycles an MMA a warp (two warps a scheduler)", s);
}

}  // namespace mma_k8

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  mma_k8::mode<0>(b);
  mma_k8::mode<1>(b);
  mma_k8::mode<2>(b);
  mma_k8::mode<3>(b);
}
