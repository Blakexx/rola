// measure/lab/experiments/wgmma_regs.cu -- QUESTION: a fold's fp32 accumulators (an m64n16 wgmma's output) are the
// next wgmma's register operand A once each pair is narrowed to bf16 in place; what does the narrowing cost a step,
// by truncation and by rounding, beside A read from shared memory? METHOD: one or two warpgroups a CTA, a CTA an SM;
// a step is a fold (m64n16k16, both operands shared) and a readout (m64n64k16) whose A is shared memory, registers
// loaded once, or the fold's accumulators narrowed each step (PRMT: truncated; cvt: rounded), the two MMAs of a step
// in one commit group; each part alone gives the sum to compare against. Every form's outputs are checked against
// a host recurrence of the same integers, so the register layouts are proven, not assumed.
// see docs/internals/tools/lab.md#wgmma-regs
#include "lab.cuh"

#include <cmath>

namespace wgmma_regs {

constexpr int kForms = 7;
enum Form : int {
  kFold = 0,       // the fold alone
  kReadShared,     // the readout alone, A shared
  kReadRegs,       // the readout alone, A registers loaded once
  kStepShared,     // fold + readout, A shared
  kStepRegs,       // fold + readout, A registers loaded once
  kStepTruncated,  // fold + readout, A the fold's accumulators truncated
  kStepRounded,    // fold + readout, A the fold's accumulators rounded
};
const char* kFormName[kForms] = {"fold alone",
                                 "readout alone, A shared",
                                 "readout alone, A registers",
                                 "fold + readout, A shared",
                                 "fold + readout, A registers",
                                 "fold + readout, A the state truncated",
                                 "fold + readout, A the state rounded"};

constexpr int kM = 64, kK = 16, kFoldN = 16, kReadN = 64, kGroupThreads = 128;

//: THE OPERANDS, small integers so every sum is exact in fp32 at the checked step count: the fold's A (values) and B
//: (coefficients), the readout's B (read weights) and its fixed A
constexpr int fold_a(int m, int k) { return (m * 3 + k * 5) % 5 - 2; }
constexpr int fold_b(int k, int n) { return (k * 7 + n * 3) % 5 - 2; }
constexpr int read_b(int k, int n) { return (k + 2 * n) % 3 - 1; }
constexpr int read_a(int m, int k) { return (m + k) % 7 - 3; }

//: A K-MAJOR TILE'S BYTE: line `n` (a row of A, a column of B) K `k`, bricks of 8 lines, a brick's K halves 128 bytes
//: apart and bricks 256 (`lab::wg_desc(a, 8, 16)`)
constexpr int tile_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }
constexpr int tile_bytes(int lines) { return 256 * (lines / 8); }

constexpr int kFoldA = 0, kFoldB = kFoldA + tile_bytes(kM), kReadB = kFoldB + tile_bytes(kFoldN),
              kReadA = kReadB + tile_bytes(kReadN), kSmem = kReadA + tile_bytes(kM);

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

//: a thread's outputs: the readout's 32 accumulators, then the state's 8
constexpr int kOut = 40;

template <int FORM>
__global__ void step(long long* times, float* out, int steps) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  auto* tile = reinterpret_cast<uint16_t*>(smem);
  //: THE TILES, written once by the CTA
#pragma unroll 1
  for (int i = tid; i < kM * kK; i += blockDim.x) {
    const int m = i / kK, k = i % kK;
    tile[(kFoldA + tile_at(m, k)) / 2] = bf16_of(fold_a(m, k));
    tile[(kReadA + tile_at(m, k)) / 2] = bf16_of(read_a(m, k));
  }
#pragma unroll 1
  for (int i = tid; i < kFoldN * kK; i += blockDim.x)
    tile[(kFoldB + tile_at(i / kK, i % kK)) / 2] = bf16_of(fold_b(i % kK, i / kK));
#pragma unroll 1
  for (int i = tid; i < kReadN * kK; i += blockDim.x)
    tile[(kReadB + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_b(i % kK, i / kK));
  __syncthreads();

  const uint32_t base = lab::smem(smem);
  const uint64_t fa = lab::wg_desc(base + kFoldA, 8, 16), fb = lab::wg_desc(base + kFoldB, 8, 16),
                 rb = lab::wg_desc(base + kReadB, 8, 16), ra = lab::wg_desc(base + kReadA, 8, 16);
  float s[8] = {}, d[32] = {};
  uint32_t a[4] = {};
  //: THE FIXED REGISTER A: this thread's 8 values of the readout's A, word `j + 2q` its rows `g + 8j` at K
  //: `8q + 2t`, `8q + 2t + 1`
  if constexpr (FORM == kReadRegs || FORM == kStepRegs) {
    const int g = 16 * (warp & 3) + lane / 4, t = lane & 3;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int m = g + 8 * (w & 1), k = 8 * (w >> 1) + 2 * t;
      a[w] = bf16_of(read_a(m, k)) | static_cast<uint32_t>(bf16_of(read_a(m, k + 1))) << 16;
    }
  }
  constexpr bool kFolds = FORM == kFold || FORM >= kStepShared;
  constexpr bool kNarrows = FORM == kStepTruncated || FORM == kStepRounded;

  lab::WarpSpan clock;
  clock.start();
  //: THE FIRST STATE, so a narrowed step's A is one fold old
  if constexpr (kNarrows) {
    lab::wg_fence();
    lab::wg_bf16_n16_ss(s, fa, fb);
    lab::wg_commit();
  }
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    if constexpr (kNarrows) {
      lab::wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 8; ++i) lab::wg_hold(s[i]);
#pragma unroll
      for (int w = 0; w < 4; ++w)
        a[w] = FORM == kStepTruncated ? lab::bf16x2_trunc(s[2 * w], s[2 * w + 1])
                                      : lab::bf16x2_round(s[2 * w], s[2 * w + 1]);
    }
    lab::wg_fence();
    if constexpr (kFolds) lab::wg_bf16_n16_ss(s, fa, fb);
    if constexpr (FORM == kReadShared || FORM == kStepShared) lab::wg_bf16_n64_ss(d, ra, rb);
    if constexpr (FORM == kReadRegs || FORM == kStepRegs || kNarrows) lab::wg_bf16_n64_rs(d, a, rb);
    lab::wg_commit();
    if constexpr (!kNarrows) lab::wg_wait<0>();
  }
  lab::wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 8; ++i) lab::wg_hold(s[i]);
#pragma unroll
  for (int i = 0; i < 32; ++i) lab::wg_hold(d[i]);
  clock.stop(times + 2 * (blockIdx.x * (blockDim.x >> 5) + warp));

  float* mine = out + (static_cast<long long>(blockIdx.x) * blockDim.x + tid) * kOut;
#pragma unroll
  for (int i = 0; i < 32; ++i) mine[i] = d[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) mine[32 + i] = s[i];
}

//: bf16 of an fp32, truncated or rounded to nearest even, as the device narrows it
float narrowed(float x, bool rounded) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if (rounded) u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  float y;
  std::memcpy(&y, &u, 4);
  return y;
}

//: THE HOST RECURRENCE of one form after `steps`: the readout `d [64 x 64]` and the state `s [64 x 16]`
void reference(int form, int steps, std::vector<double>& d, std::vector<double>& s) {
  d.assign(kM * kReadN, 0.0);
  s.assign(kM * kFoldN, 0.0);
  std::vector<double> f(kM * kFoldN, 0.0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kFoldN; ++n)
      for (int k = 0; k < kK; ++k) f[m * kFoldN + n] += fold_a(m, k) * fold_b(k, n);
  const bool folds = form == kFold || form >= kStepShared, narrows = form >= kStepTruncated;
  if (narrows) s = f;
  for (int it = 0; it < steps; ++it) {
    if (form != kFold)
      for (int m = 0; m < kM; ++m)
        for (int n = 0; n < kReadN; ++n)
          for (int k = 0; k < kK; ++k) {
            const double av = narrows ? narrowed(static_cast<float>(s[m * kFoldN + k]), form == kStepRounded)
                                      : read_a(m, k);
            d[m * kReadN + n] += av * read_b(k, n);
          }
    if (folds)
      for (size_t i = 0; i < s.size(); ++i) s[i] += f[i];
  }
}

}  // namespace wgmma_regs

template <int FORM>
void run(lab::Bench& b, float* out, int groups) {
  using namespace wgmma_regs;
  constexpr int kCheckSteps = 24, kSteps = 4096;
  const int threads = kGroupThreads * groups, warps = threads / 32;
  const std::string name = std::string(kFormName[FORM]) + (groups == 1 ? ", one warpgroup" : ", two warpgroups");

  //: THE CHECK: a short launch whose every accumulator is the host's, by the register layout CUTLASS documents
  step<FORM><<<b.sms, threads>>>(b.times, out, kCheckSteps);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * threads * kOut);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<double> d, s;
  reference(FORM, kCheckSteps, d, s);
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = 0; tid < threads; ++tid) {
      const float* mine = got.data() + (static_cast<size_t>(blk) * threads + tid) * kOut;
      const int g = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, t = tid & 3;
      for (int i = 0; i < 32; ++i) {
        const int m = g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1);
        wrong += mine[i] != static_cast<float>(d[m * kReadN + n]);
      }
      for (int i = 0; i < 8; ++i) {
        const int m = g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1);
        wrong += mine[32 + i] != static_cast<float>(s[m * kFoldN + n]);
      }
    }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(got.size()) + " accumulators differ from the host's at "
              + std::to_string(kCheckSteps) + " steps");

  const lab::Samples cycles = b.variant([&] { step<FORM><<<b.sms, threads>>>(b.times, out, kSteps); },
                                        [&] { return b.span(warps) / kSteps; });
  b.record(name, "cycles a step", cycles);
}

int main(int argc, char** argv) {
  using namespace wgmma_regs;
  lab::Bench b(argc, argv);
  for (int f = 0; f < kForms; ++f) {
    const std::string k = lab::kernel("step", {f});
    const int mmas = (f == kFold || f >= kStepShared ? 1 : 0) + (f != kFold ? 1 : 0)
                     + (f >= kStepTruncated ? 1 : 0);
    b.expect(k, "HGMMA", "==", mmas);
    b.expect(k, "PRMT", f == kStepTruncated ? ">=" : "==", f == kStepTruncated ? 4 : 0);
    b.expect(k, "F2FP", "==", f == kStepRounded ? 4 : 0);
    b.exempt(k, "a rate probe: the step's parts and their sum, no kernel's step");
  }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * 2 * kGroupThreads * kOut));
  for (int groups = 1; groups <= 2; ++groups) {
    run<kFold>(b, out, groups);
    run<kReadShared>(b, out, groups);
    run<kReadRegs>(b, out, groups);
    run<kStepShared>(b, out, groups);
    run<kStepRegs>(b, out, groups);
    run<kStepTruncated>(b, out, groups);
    run<kStepRounded>(b, out, groups);
  }
  return 0;
}
