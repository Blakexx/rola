// measure/lab/experiments/mma_queue.cu -- QUESTION: how many independent MMAs can a warp issue back to back before it
// stalls on the tensor pipe -- is there a queue a warp can fill and then go on to other work while it drains?
// METHOD: one warp a scheduler (4 a CTA, a CTA an SM); each repetition reads the SM clock, issues K MMAs into K
// fresh accumulators (their A operand tied to the clock read by a run-time zero, so none issues before it), reads the
// clock again (in-order issue: after the K-th MMA has issued), then folds the outputs (waiting for them, so the next
// repetition starts on an empty pipe). Cycles from clock to clock against K: flat while the pipe takes them, then
// one MMA's pipe time a step.
// see docs/internals/tools/lab.md#mma-queue
#include "lab.cuh"

namespace mma_queue {

constexpr int kReps = 256, kWarps = 4;

__device__ __forceinline__ uint32_t clock32() {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory");
  return t;
}

//: KIND 0 fp16-accumulate, 1 fp32-accumulate
template <int K, int KIND>
__global__ void __launch_bounds__(kWarps * 32, 1) burst(int reps, const uint32_t* in, long long* t, int* out) {
  const uint32_t zero = static_cast<uint32_t>(reps) >> 31;
  uint32_t a[4], b0 = in[4 + (threadIdx.x & 7)], b1 = in[12 + (threadIdx.x & 7)];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = in[i];
  uint32_t fold = 0;
  long long cycles = 0;
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    const uint32_t t0 = clock32();
    const uint32_t a_t[4] = {a[0] ^ (t0 & zero), a[1], a[2], a[3]};
    if constexpr (KIND == 0) {
      uint32_t acc[K][2] = {};
#pragma unroll
      for (int k = 0; k < K; ++k) lab::mma_f16_f16(acc[k], a_t, b0, b1);
      cycles += clock32() - t0;
#pragma unroll
      for (int k = 0; k < K; ++k) fold ^= acc[k][0] ^ acc[k][1];
    } else {
      float acc[K][4] = {};
#pragma unroll
      for (int k = 0; k < K; ++k) lab::mma_f16_f32(acc[k], a_t, b0, b1);
      cycles += clock32() - t0;
#pragma unroll
      for (int k = 0; k < K; ++k) fold ^= lab::bits(acc[k][0]) ^ lab::bits(acc[k][3]);
    }
    a[1] ^= fold & zero;
  }
  if (lab::lane_id() == 0) t[blockIdx.x * kWarps + lab::warp_id()] = cycles;
  lab::sink(out, fold);
}

const char* kKindName[2] = {"fp16-accumulate HMMA", "fp32-accumulate HMMA"};

template <int K, int KIND>
void point(const lab::Bench& b, const uint32_t* in) {
  const std::string name = lab::kernel("burst", {K, KIND});
  b.expect(name, "HMMA", "==", K);
  if (b.expect_only) return;
  const lab::Samples s = b.variant([&] { burst<K, KIND><<<b.sms, kWarps * 32>>>(kReps, in, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(kWarps), kWarps, 0, kWarps) / kReps; });
  b.record(std::string(kKindName[KIND]) + " / " + std::to_string(K) + " back to back", "cycles to issue", s);
}

template <int KIND, int... K>
void sweep(const lab::Bench& b, const uint32_t* in, std::integer_sequence<int, K...>) {
  (point<K + 1, KIND>(b, in), ...);
}

}  // namespace mma_queue

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  uint32_t* in = nullptr;
  if (!b.expect_only) {
    const uint32_t host[20] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c00u, 0x3c00u, 0x3c00u,
                               0x3c00u,     0x3c00u,     0x3c00u,     0x3c00u,     0x3c00u, 0x3c00u, 0x3c00u,
                               0x3c00u,     0x3c00u,     0x3c00u,     0x3c00u,     0x3c00u, 0x3c00u};
    LAB_CHECK(cudaMalloc(&in, sizeof(host)));
    LAB_CHECK(cudaMemcpy(in, host, sizeof(host), cudaMemcpyHostToDevice));
  }
  mma_queue::sweep<0>(b, in, std::make_integer_sequence<int, 12>{});
  mma_queue::sweep<1>(b, in, std::make_integer_sequence<int, 12>{});
  mma_queue::point<16, 0>(b, in);
  mma_queue::point<16, 1>(b, in);
}
