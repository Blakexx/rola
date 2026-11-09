// measure/lab/experiments/tensor_tax.cu -- QUESTION: one warp on a scheduler runs an MMA chain while another issues
// N instructions of one class; how many cycles does each instruction add to the chain? METHOD: warps 0-3 (one a
// scheduler) run 8 MMAs a loop into 8 accumulators; warps 4-7 (the same schedulers) issue the class in 8 independent
// chains (each instruction's second operand the next chain's value, so ptxas can fold nothing), sized to ~40% of the
// chain alone so every instruction lands inside it; tax = (chain with - alone) / N. No role branch: every warp runs both
// loops with its other role's trip count zero. A collective cannot be timed this way: role-dependent control flow of
// any form turns it into a CALL (`collective_branch`).
// see docs/internals/tools/lab.md#tensor-tax
#include "lab.cuh"

#include <utility>

namespace tensor_tax {

constexpr int kKinds = 4;
const char* kKindName[kKinds] = {"hmma f16->f32", "hmma f16->f16", "hmma bf16->f32",
                                 "imma u8.s8 k32"};
const char* kKindOp[kKinds] = {"HMMA.16816.F32", "HMMA.16816.F16", "HMMA.16816.F32.BF16",
                               "IMMA.16832"};

constexpr int kOps = 17;
const char* kOpName[kOps + 1] = {
    "none",       "fmul",     "ffma",      "fadd",  "hmul2",
    "hfma2",      "hadd2",    "bf16 fma2", "hmax2", "add + xor (iadd3 or imad.iadd, lop3)",
    "imad",       "lop3",     "prmt",      "fmnmx", "lds",
    "f32->f16x2", "u32->f32", "f16->f32"};
const char* kOpcode[kOps + 1] = {"",      "FMUL",  "FFMA",       "FADD",   "HMUL2",
                                 "HFMA2", "HADD2", "HFMA2.BF16", "HMNMX2", "IADD3|IMAD.IADD",
                                 "IMAD",  "LOP3",  "PRMT",       "FMNMX",  "LDS",
                                 "F2FP",  "I2F",   "HADD2.F32"};

template <int KIND>
__device__ __forceinline__ void mma(float (&d)[4], uint32_t (&h)[2], int (&s)[4],
                                    const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  if constexpr (KIND == 0) lab::mma_f16_f32(d, a, b0, b1);
  if constexpr (KIND == 1) lab::mma_f16_f16(h, a, b0, b1);
  if constexpr (KIND == 2) lab::mma_bf16_f32(d, a, b0, b1);
  if constexpr (KIND == 3) lab::mma_u8s8(s, a, b0, b1);
}

template <int OP>
__device__ __forceinline__ void op(uint32_t& x, uint32_t k, uint32_t k2, uint32_t smem) {
  //: `k`, `k2` are the next chains' values: loop-variant operands, so no two instructions combine. Integer adds are
  //: exactly associative and ptxas rewrites any add recurrence, so the add is timed as an add-then-xor pair
  if constexpr (OP == 1) asm volatile("mul.rn.f32 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 2) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 3) asm volatile("add.rn.f32 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 4) asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 5) asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 6) asm volatile("add.rn.f16x2 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 7) asm volatile("fma.rn.bf16x2 %0, %0, %1, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 8) asm volatile("max.f16x2 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 9) x = (x + k) ^ k2;
  if constexpr (OP == 10) asm volatile("mad.lo.u32 %0, %0, %1, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 11) asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(x) : "r"(k));
  if constexpr (OP == 12) asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(x) : "r"(k));
  if constexpr (OP == 13) asm volatile("min.f32 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 14)
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem + ((x & 63u) << 2)));
  if constexpr (OP == 15) asm volatile("cvt.rn.f16x2.f32 %0, %0, %1;" : "+r"(x) : "r"(k));
  if constexpr (OP == 16) asm volatile("cvt.rn.f32.u32 %0, %0;" : "+r"(x));
  if constexpr (OP == 17) {
    const uint16_t lo = static_cast<uint16_t>(x);
    asm volatile("cvt.f32.f16 %0, %1;" : "=r"(x) : "h"(lo));
  }
}

template <int KIND, int OP>
__global__ void __launch_bounds__(256, 1) tax(int mma_iters, int x_iters, long long* t, int* out) {
  __shared__ uint32_t buf[64];
  const int lane = lab::lane_id();
  if (threadIdx.x < 64) buf[threadIdx.x] = threadIdx.x * 2654435761u;
  __syncthreads();
  const bool chain = lab::warp_id() < 4;
  const int mi = chain ? mma_iters : 0, xi = chain || OP == 0 ? 0 : x_iters;
  lab::WarpClock clock;
  float d[8][4] = {};
  uint32_t h[8][2] = {};
  int s[8][4] = {};
  uint32_t a[4];
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (0x00010001u * (lane + i));
  const uint32_t b0 = 0x3c003c00u ^ lane, b1 = b0 ^ 0x00100010u;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < mi; ++it)
#pragma unroll
    for (int j = 0; j < 8; ++j) mma<KIND>(d[j], h[j], s[j], a, b0, b1);
  const long long t_mma = clock64() - clock.t0;
  uint32_t x[8];
  for (int j = 0; j < 8; ++j) x[j] = 0x3c003c01u + lane * 7u + j;
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
  clock.start();
#pragma unroll 1
  for (int it = 0; it < xi; ++it)
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) op<OP>(x[j], x[(j + 1) & 7], x[(j + 2) & 7], smem);
  const long long t_x = clock64() - clock.t0;
  if (lane == 0) t[blockIdx.x * 8 + lab::warp_id()] = chain ? t_mma : t_x;
  uint32_t fold = 0;
  for (int j = 0; j < 8; ++j) fold ^= lab::bits(d[j][0]) ^ h[j][0] ^ lab::bits(s[j][0]) ^ x[j];
  lab::sink(out, fold);
}

constexpr int kMmaIters = 2000;

template <int KIND, int OP>
lab::Samples launch(const lab::Bench& b, int mma_iters, int x_iters, int first) {
  return b.variant([&] { tax<KIND, OP><<<b.sms, 256>>>(mma_iters, x_iters, b.times, b.sinks); },
                   [&] { return lab::Bench::mean(b.cycles(8), 8, first, 4); });
}

template <int KIND, int OP>
void one(const lab::Bench& b, double alone) {
  b.expect(lab::kernel("tax", {KIND, OP}), kOpcode[OP], ">=", 32);
  if (OP == 9) b.expect(lab::kernel("tax", {KIND, OP}), "LOP3", ">=", 32);
  if (b.expect_only) return;
  const double rate = launch<KIND, OP>(b, 0, 1000, 4).median() / 1000.0;
  const int x_iters = std::max(1, static_cast<int>(0.4 * alone / rate));
  const lab::Samples with = launch<KIND, OP>(b, kMmaIters, x_iters, 0);
  lab::Samples taxed;
  for (double c : with.v) taxed.v.push_back((c - alone) / (32.0 * x_iters));
  b.record(std::string(kKindName[KIND]) + " / " + kOpName[OP], "cycles an instruction", taxed);
  if (KIND == 0)
    b.record(std::string("alone / ") + kOpName[OP], "cycles an instruction", {{rate / 32.0}});
}

template <int KIND, int... OPS>
void kind(const lab::Bench& b, std::integer_sequence<int, OPS...>) {
  b.expect(lab::kernel("tax", {KIND, 0}), kKindOp[KIND], ">=", 8);
  double alone = 0;
  if (!b.expect_only) {
    const lab::Samples chain = launch<KIND, 0>(b, kMmaIters, 0, 0);
    alone = chain.median();
    lab::Samples per;
    for (double c : chain.v) per.v.push_back(c / (8.0 * kMmaIters));
    b.record(std::string(kKindName[KIND]) + " / alone", "cycles an MMA", per);
  }
  (one<KIND, OPS + 1>(b, alone), ...);
}

}  // namespace tensor_tax

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  const auto ops = std::make_integer_sequence<int, tensor_tax::kOps>{};
  tensor_tax::kind<0>(b, ops);
  tensor_tax::kind<1>(b, ops);
  tensor_tax::kind<2>(b, ops);
  tensor_tax::kind<3>(b, ops);
}
