// measure/lab/experiments/mma_overlap.cu -- QUESTION: a warp's step is 512 tensor cycles of MMAs plus the chain that
// makes their operands (ldmatrix, coefficient loads and math, shuffles, a packed product a list); which schedule hides
// the chain best -- inline (operands at the top of the step) or staged (the next step's operands under this step's
// MMAs) -- and how does that depend on the warps a scheduler, the MMA kind and how the warps synchronize (free-running,
// a CTA barrier every pass of 8 steps, or a two-stage mbarrier that lets them drift a pass apart)? METHOD: one block a
// SM of 1-4 warps a scheduler, the fold's operand chain over shared tiles, 8 accumulators (so every warp count fits its
// registers), a sample the block's span a step over its floor (512 cycles x warps a scheduler).
// see docs/internals/tools/lab.md#mma-overlap
#include "lab.cuh"

namespace mma_overlap {

constexpr int kLists = 8, kRows = 128, kStep = 16, kPassSteps = kRows / kStep;

//: KIND 0 fp16-acc HMMA (16 cycles), 1 fp32-acc HMMA (32), 2 u8.s8 IMMA k32 (16): a step is 512 tensor cycles, 32
//: atoms (4 n-tiles a list) or 16 (2 a list, fp32-acc)
constexpr int kKinds = 3;
const char* kKindName[kKinds] = {"fp16-acc", "fp32-acc", "int8"};
const char* kKindOp[kKinds] = {"HMMA.16816.F16", "HMMA.16816.F32", "IMMA.16832"};

__host__ __device__ constexpr int tiles_of(int kind) { return kind == 1 ? 2 : 4; }

//: SCHED 0 inline, 1 staged; SYNC 0 free-running, 1 CTA barrier a pass, 2 two-stage mbarrier a pass (polled), 3 the
//: same with a 64-ns sleep between polls
const char* kSchedName[2] = {"inline", "staged"};
const char* kSyncName[4] = {"free", "barrier", "two-stage mbarrier",
                            "two-stage mbarrier, sleeping"};

struct Tiles {
  uint16_t dense[kRows][24];
  uint16_t vals[kRows][40];
  uint8_t write[kRows][128];
  float2 fm[kRows];
};

struct Operands {
  uint32_t a[4], b[4][2], c[2];
};

__device__ __forceinline__ void operands(Operands& o, const Tiles& t, int e0, int lane, float pu) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3,
            byte = lab::warp_id() * kLists + (lane >> 2);
  const int ra = e0 + 8 * (q >> 1) + r, rb = e0 + 8 * (q & 1) + r;
  lab::ldsm_t(o.a, lab::smem(&t.dense[ra][8 * (q & 1)]));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t x[4];
    lab::ldsm_t(x, lab::smem(&t.vals[rb][16 * h + 8 * (q >> 1)]));
    o.b[2 * h][0] = x[0];
    o.b[2 * h][1] = x[1];
    o.b[2 * h + 1][0] = x[2];
    o.b[2 * h + 1][1] = x[3];
  }
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int row = e0 + 2 * tq + (k & 1) + 8 * (k >> 1);
    const float b = fmaxf(static_cast<float>(t.write[row][byte]), 2.0f);
    v[k] = __fmul_rn(__fmul_rn(b, t.fm[row].x), pu);
  }
  o.c[0] = lab::pack_half2(v[0], v[1]);
  o.c[1] = lab::pack_half2(v[2], v[3]);
}

template <int KIND>
struct Acc {
  float f[8][4];
  uint32_t h[8][2];
  int s[8][4];
};

template <int KIND>
__device__ __forceinline__ void atom(Acc<KIND>& acc, int i, const uint32_t (&a)[4], uint32_t b0,
                                     uint32_t b1) {
  if constexpr (KIND == 0) lab::mma_f16_f16(acc.h[i], a, b0, b1);
  if constexpr (KIND == 1) lab::mma_f16_f32(acc.f[i], a, b0, b1);
  if constexpr (KIND == 2) lab::mma_u8s8(acc.s[i], a, b0, b1);
}

template <int KIND>
__device__ __forceinline__ void step(Acc<KIND>& acc, const Operands& o, int lane) {
  const int tq = lane & 3;
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    const uint32_t c01 = __shfl_sync(0xffffffffu, o.c[0], 4 * j + tq);
    const uint32_t c89 = __shfl_sync(0xffffffffu, o.c[1], 4 * j + tq);
    const uint32_t aj[4] = {lab::hmul2(o.a[0], c01), lab::hmul2(o.a[1], c01),
                            lab::hmul2(o.a[2], c89), lab::hmul2(o.a[3], c89)};
#pragma unroll
    for (int n = 0; n < tiles_of(KIND); ++n) {
      //: two n-tiles a list (fp32-acc) alternate between the step's two halves, so every B is read
      const int tile = tiles_of(KIND) == 4 ? n : n + 2 * (j & 1);
      atom<KIND>(acc, (j * tiles_of(KIND) + n) & 7, aj, o.b[tile][0], o.b[tile][1]);
    }
  }
}

template <int SYNC>
__device__ __forceinline__ void pass_end(uint64_t* bars, int pass, int lane) {
  if constexpr (SYNC == 1) __syncthreads();
  if constexpr (SYNC == 2) {
    //: arrive for this pass; wait only for the pass before it, so a warp may run a pass ahead (two stages)
    if (lane == 0) lab::mbar_arrive(lab::smem(&bars[pass & 1]));
    if (pass >= 1) lab::mbar_wait(lab::smem(&bars[(pass - 1) & 1]), ((pass - 1) >> 1) & 1);
  }
  if constexpr (SYNC == 3) {
    if (lane == 0) lab::mbar_arrive(lab::smem(&bars[pass & 1]));
    if (pass >= 1)
      while (!lab::mbar_test(lab::smem(&bars[(pass - 1) & 1]), ((pass - 1) >> 1) & 1))
        __nanosleep(64);
  }
}

template <int KIND, int SCHED, int SYNC, int WPS>
__global__ void __launch_bounds__(128 * WPS, 1) overlap(int passes, int steps, long long* t,
                                                        int* out) {
  __shared__ Tiles tiles;
  __shared__ uint64_t bars[2];
  const int lane = lab::lane_id();
  for (int i = threadIdx.x; i < kRows; i += blockDim.x) {
    for (int c = 0; c < 24; ++c)
      tiles.dense[i][c] = static_cast<uint16_t>(0x2c00 + ((i * 7 + c) & 255));
    for (int c = 0; c < 40; ++c)
      tiles.vals[i][c] = static_cast<uint16_t>(0x2c00 + ((i * 5 + c) & 255));
    for (int c = 0; c < 128; ++c) tiles.write[i][c] = static_cast<uint8_t>(i * 3 + c);
    tiles.fm[i] = make_float2(1e-3f * (1 + (i & 7)), 1e-3f * (1 + (i & 3)));
  }
  if (threadIdx.x < 2) lab::mbar_init(lab::smem(&bars[threadIdx.x]), blockDim.x / 32);
  __syncthreads();
  const float pu = 0.5f + 1e-3f * lane;
  Acc<KIND> acc = {};
  Operands cur;
  if constexpr (SCHED == 1) operands(cur, tiles, 0, lane, pu);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int p = 0; p < passes; ++p) {
    if constexpr (SCHED == 1) {
      //: two steps a trip, so the stages alternate between two register sets without copies
#pragma unroll 2
      for (int s = 0; s < steps; ++s) {
        Operands next;
        operands(next, tiles, ((s + 1) % kPassSteps) * kStep, lane, pu);
        step<KIND>(acc, cur, lane);
        cur = next;
      }
    } else {
#pragma unroll 1
      for (int s = 0; s < steps; ++s) {
        operands(cur, tiles, s * kStep, lane, pu);
        step<KIND>(acc, cur, lane);
      }
    }
    pass_end<SYNC>(bars, p, lane);
  }
  clock.stop(t);
  uint32_t fold = 0;
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 4; ++k)
      fold ^= lab::bits(acc.f[i][k]) ^ (k < 2 ? acc.h[i][k] : 0u) ^ lab::bits(acc.s[i][k]);
  lab::sink(out, fold);
}

constexpr int kPasses = 200;

template <int KIND, int SCHED, int SYNC, int WPS>
void variant(const lab::Bench& b) {
  const std::string name = lab::kernel("overlap", {KIND, SCHED, SYNC, WPS});
  b.expect(name, kKindOp[KIND], ">=", 8 * tiles_of(KIND));
  b.expect(name, "LDSM", ">=", 3);
  b.expect(name, "SHFL", ">=", 16);
  b.expect(name, "HMUL2", ">=", 32);
  if (SYNC == 1) b.expect(name, "BAR", ">=", 1);
  if (b.expect_only) return;
  const int warps = 4 * WPS;
  const lab::Samples s = b.variant(
      [&] {
        overlap<KIND, SCHED, SYNC, WPS>
            <<<b.sms, 32 * warps>>>(kPasses, kPassSteps, b.times, b.sinks);
      },
      [&] { return b.span(warps) / (kPasses * kPassSteps * 512.0 * WPS); });
  b.record(std::string(kKindName[KIND]) + " / " + std::to_string(WPS) + " a scheduler / "
               + kSchedName[SCHED] + " / " + kSyncName[SYNC],
           "x floor", s);
}

template <int KIND, int WPS>
void warps(const lab::Bench& b) {
  variant<KIND, 0, 0, WPS>(b);
  variant<KIND, 1, 0, WPS>(b);
  variant<KIND, 0, 1, WPS>(b);
  variant<KIND, 1, 1, WPS>(b);
  variant<KIND, 0, 2, WPS>(b);
  variant<KIND, 1, 2, WPS>(b);
  variant<KIND, 0, 3, WPS>(b);
  variant<KIND, 1, 3, WPS>(b);
}

template <int KIND>
void kind(const lab::Bench& b) {
  warps<KIND, 1>(b);
  warps<KIND, 2>(b);
  warps<KIND, 3>(b);
  warps<KIND, 4>(b);
}

}  // namespace mma_overlap

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  mma_overlap::kind<0>(b);
  mma_overlap::kind<1>(b);
  mma_overlap::kind<2>(b);
}
