// measure/lab/experiments/pass_pipeline.cu -- QUESTION: the fold's pass is a fill (its 128 token rows, scattered in
// global memory, copied by cp.async into a pool), a restage (pool rows to the f16 tiles the steps read) and 8 steps of
// MMAs, and the fp16 arm joins its partials every two passes. Serial between CTA barriers it pays the copy's latency,
// the restage and the realignment; where should each piece go in a two-stage pipeline (fill two passes ahead, restage
// one ahead, mbarriers for landed / staged / drained) -- the restage in a burst at the pass's start, a warp a step or
// mid-pass; the join in a burst, a list a step, or staggered between a scheduler's two warps? METHOD: the owner part's
// fill pattern (a quad a token: the 64-byte run and the values' 64-byte run in 16-byte copies, four 4-byte factors from
// four planes, a 16-byte dense chunk), its restage (two threads a row) and `fold_step`'s step at the arm's registers,
// over a 33 MB source (under the WSL translation reach); the block's span a pass over the steps' floor.
// see docs/internals/tools/lab.md#pass-pipeline
#include "lab.cuh"

namespace pass_pipeline {

constexpr int kLists = 8, kRows = 128, kStep = 16, kSteps = kRows / kStep, kThreads = 256;
constexpr int kRowBytes = 160, kWindow = 512, kWindows = 400, kTokens = kWindows * kWindow;
constexpr int kArmFloat = 0, kArmHalf = 1;

//: MODE 0 serial (fill, barrier, restage, barrier, steps, barrier: the owner part); 1 pipelined, the restage a burst at
//: the pass's start; 2 pipelined, warp w restaging its rows after its step w; 3 the steps alone; 4 pipelined, every warp
//: restaging after its step 3 (half a pass of slack each way). Under 2 and 4 the wait for the tile slot's last readers
//: sits just before the restage, not at the pass's start
constexpr int kModes = 5;
const char* kModeName[kModes] = {"serial", "pipelined, restage burst",
                                 "pipelined, restage a warp a step", "steps alone",
                                 "pipelined, restage mid-pass"};

//: the tiles a pass's steps read; the coefficient run moves here too, since the pool slot is refilled two passes ahead
struct Tiles {
  uint16_t dense[kRows][24];
  uint16_t vals[kRows][40];
  uint8_t run[kRows][64];
  float2 fm[kRows];
};

struct alignas(16) Shared {
  uint8_t pool[2][kRows][kRowBytes];
  Tiles ft[2];
  uint64_t landed[2], staged[2], drained[2];
};

//: the source planes, a token's row in each: the coefficient run (64 B), the values (64 B), four factors (4 B, a plane
//: each, `kTokens` x 4 bytes apart -- one base, since a runtime index into an array of pointers puts it in local
//: memory), the dense chunk (16 B)
struct Source {
  const uint8_t *run, *values, *factor, *dense;
};

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: PASS `x`'s rows into pool slot `k`, a quad a token
__device__ __forceinline__ void fill(Shared& sh, int k, const Source& src, int x, int tid) {
  const int q = tid & 3;
  const int64_t win = static_cast<int64_t>((blockIdx.x * 7 + x) % kWindows) * kWindow;
#pragma unroll 1
  for (int r = tid >> 2; r < kRows; r += kThreads / 4) {
    const int64_t tg = win + ((r * 37 + 128 * (x & 3)) & (kWindow - 1));
    uint8_t* const dst = sh.pool[k][r];
    lab::cp_async16(lab::smem(dst + 16 * q), src.run + tg * 64 + 16 * q);
    lab::cp_async16(lab::smem(dst + 64 + 16 * q), src.values + tg * 64 + 16 * q);
    lab::cp_async4(lab::smem(dst + 128 + 4 * q),
                   src.factor + (q * static_cast<int64_t>(kTokens) + tg) * 4);
    if (q == 0) lab::cp_async16(lab::smem(dst + 144), src.dense + tg * 16);
  }
}

//: ITEM `i` OF A RESTAGE (row `i >> 1`): the values copied (odd) or the coefficient run copied, the dense digits to
//: f16 and the factors' products
__device__ __forceinline__ void restage_item(Tiles& ft, const Shared& sh, int k, int i) {
  const int r = i >> 1;
  const uint8_t* const row = sh.pool[k][r];
  if (i & 1) {
    const uint4* const s = reinterpret_cast<const uint4*>(row + 64);
    uint4* const d = reinterpret_cast<uint4*>(&ft.vals[r][0]);
#pragma unroll
    for (int j = 0; j < 4; ++j) d[j] = s[j];
    return;
  }
  const uint4* const s = reinterpret_cast<const uint4*>(row);
  uint4* const c = reinterpret_cast<uint4*>(&ft.run[r][0]);
#pragma unroll
  for (int j = 0; j < 4; ++j) c[j] = s[j];
  const float* const f = reinterpret_cast<const float*>(row + 128);
  const float m = __fmul_rn(__fmul_rn(f[0], f[2]), f[3]);
  ft.fm[r] = make_float2(__fmul_rn(m, f[1]), m);
  const uint4 w4 = *reinterpret_cast<const uint4*>(row + 144);
  const uint32_t w[4] = {w4.x, w4.y, w4.z, w4.w};
  uint32_t h[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    h[2 * j] =
        hmax2(hfma2(prmt(w[j], 0x64006400u, 0x4140u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
    h[2 * j + 1] =
        hmax2(hfma2(prmt(w[j], 0x64006400u, 0x4342u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
  }
  uint4* const d = reinterpret_cast<uint4*>(&ft.dense[r][0]);
  d[0] = make_uint4(h[0], h[1], h[2], h[3]);
  d[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

struct Operands {
  uint32_t a[4], b[4][2], cv[2], cm[2];
};

//: `fold_step`'s step, as built: the operands, both coefficients, the fresh mass atom, 8 lists by shuffles
template <int ARM>
struct Acc {
  float acc[kLists][4][4];
  uint32_t part[ARM == kArmHalf ? kLists : 1][4][2];
};

__device__ __forceinline__ void operands(Operands& o, const Tiles& t, int e0, int lane, float pu,
                                         float pv) {
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
  float v[4], m[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int row = e0 + 2 * tq + (k & 1) + 8 * (k >> 1);
    const float b = fmaxf(static_cast<float>(t.run[row][byte]), 2.0f);
    const float2 f = t.fm[row];
    v[k] = __fmul_rn(__fmul_rn(b, f.x), pu);
    m[k] = __fmul_rn(__fmul_rn(b, f.y), pv);
  }
  o.cv[0] = lab::pack_half2(v[0], v[1]);
  o.cv[1] = lab::pack_half2(v[2], v[3]);
  o.cm[0] = lab::pack_half2(m[0], m[1]);
  o.cm[1] = lab::pack_half2(m[2], m[3]);
}

template <int ARM>
__device__ __forceinline__ void step(Acc<ARM>& st, float (&m)[4], const Operands& o, int lane) {
  const int tq = lane & 3;
  float d[4];
  lab::mma_f16_f32_fresh(d, o.a, o.cm[0], o.cm[1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) m[i] += d[i];
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    const uint32_t c01 = __shfl_sync(0xffffffffu, o.cv[0], 4 * j + tq);
    const uint32_t c89 = __shfl_sync(0xffffffffu, o.cv[1], 4 * j + tq);
    const uint32_t aj[4] = {lab::hmul2(o.a[0], c01), lab::hmul2(o.a[1], c01),
                            lab::hmul2(o.a[2], c89), lab::hmul2(o.a[3], c89)};
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      if constexpr (ARM == kArmFloat)
        lab::mma_f16_f32(st.acc[j][n], aj, o.b[n][0], o.b[n][1]);
      else
        lab::mma_f16_f16(st.part[j][n], aj, o.b[n][0], o.b[n][1]);
    }
  }
}

//: LIST `j`'S fp16 PARTIALS JOINED into its fp32 accumulators and cleared (`j` a constant once the caller unrolls)
template <int ARM>
__device__ __forceinline__ void join_list(Acc<ARM>& st, int j) {
  if constexpr (ARM == kArmHalf) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&st.part[j][n][0]));
      const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&st.part[j][n][1]));
      st.acc[j][n][0] += lo.x;
      st.acc[j][n][1] += lo.y;
      st.acc[j][n][2] += hi.x;
      st.acc[j][n][3] += hi.y;
      st.part[j][n][0] = st.part[j][n][1] = 0u;
    }
  }
}

template <int ARM>
__device__ __forceinline__ void join(Acc<ARM>& st) {
#pragma unroll
  for (int j = 0; j < kLists; ++j) join_list(st, j);
}

//: JOIN 0 every list after every second pass (as built); 1 list `s` at the start of step `s` of every second pass, so
//: each list still joins every 256 tokens -- by a switch on the step, since unrolling the step loop spills the fp16 arm;
//: 2 the burst staggered: warps 4-7 (each sharing a scheduler with warp w - 4) join after the other passes
constexpr int kJoins = 3;
const char* kJoinName[kJoins] = {"join burst", "join a list a step", "join burst, staggered"};

template <int ARM>
__device__ __forceinline__ void join_step(Acc<ARM>& st, int s) {
  switch (s) {
    case 0:
      join_list(st, 0);
      break;
    case 1:
      join_list(st, 1);
      break;
    case 2:
      join_list(st, 2);
      break;
    case 3:
      join_list(st, 3);
      break;
    case 4:
      join_list(st, 4);
      break;
    case 5:
      join_list(st, 5);
      break;
    case 6:
      join_list(st, 6);
      break;
    default:
      join_list(st, 7);
      break;
  }
}

__device__ __forceinline__ uint32_t parity(int x) { return static_cast<uint32_t>(x >> 1) & 1u; }

//: A PASS'S STEPS over tile slot `ft`; the warp's restage share of the next pass (its 32 items) after its step `warp`
//: (MODE 2) or step 3 (MODE 4), once the slot's readers of pass `p - 1` are done; under JOIN 1 list `s` joined at step
//: `s` of an odd pass
template <int ARM, int MODE, int JOIN>
__device__ __forceinline__ void steps(Acc<ARM>& st, float (&m)[4], Shared& sh, const Tiles& ft,
                                      int kn, int lane, float pu, float pv, int p) {
  const bool odd = (p & 1) != 0;
  const int warp = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
#pragma unroll 1
  for (int s = 0; s < kSteps; ++s) {
    if constexpr (JOIN == 1) {
      if (odd) join_step(st, s);
    }
    Operands o;
    operands(o, ft, s * kStep, lane, pu, pv);
    step<ARM>(st, m, o, lane);
    if constexpr (MODE == 2 || MODE == 4) {
      if (s == (MODE == 2 ? warp : 3)) {
        if (p >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(p - 1));
        restage_item(sh.ft[kn], sh, kn, tid);
        lab::mbar_arrive(lab::smem(&sh.staged[kn]));
      }
    }
  }
}

template <int ARM, int MODE, int JOIN>
__global__ void __launch_bounds__(kThreads, 1) pass(int passes, Source src, long long* t,
                                                    int* out) {
  extern __shared__ __align__(16) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  if (tid < 2) {
    lab::mbar_init(lab::smem(&sh.landed[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.staged[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.drained[tid]), kThreads);
  }
  for (int i = tid; i < 2 * kRows; i += kThreads) {
    restage_item(sh.ft[0], sh, 0, i);
    restage_item(sh.ft[1], sh, 1, i);
  }
  __syncthreads();
  const float pu = 0.5f + 1e-3f * lane, pv = 0.25f + 1e-3f * lane;
  Acc<ARM> st = {};
  float m[4] = {}, mass = 0.0f;
  if constexpr (MODE == 1 || MODE == 2 || MODE == 4) {
    fill(sh, 0, src, 0, tid);
    lab::mbar_arrive_when_landed(lab::smem(&sh.landed[0]));
    fill(sh, 1, src, 1, tid);
    lab::mbar_arrive_when_landed(lab::smem(&sh.landed[1]));
    lab::mbar_wait(lab::smem(&sh.landed[0]), 0);
    restage_item(sh.ft[0], sh, 0, tid);
    lab::mbar_arrive(lab::smem(&sh.staged[0]));
  }
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int p = 0; p < passes; ++p) {
    if constexpr (MODE == 0) {
      fill(sh, 0, src, p, tid);
      lab::cp_commit();
      lab::cp_wait<0>();
      __syncthreads();
      restage_item(sh.ft[0], sh, 0, tid);
      __syncthreads();
      steps<ARM, 0, JOIN>(st, m, sh, sh.ft[0], 0, lane, pu, pv, p);
      __syncthreads();
    }
    if constexpr (MODE == 1 || MODE == 2 || MODE == 4) {
      const int k = p & 1, kn = (p + 1) & 1;
      lab::mbar_wait(lab::smem(&sh.staged[k]), parity(p));
      fill(sh, k, src, p + 2, tid);
      lab::mbar_arrive_when_landed(lab::smem(&sh.landed[k]));
      lab::mbar_wait(lab::smem(&sh.landed[kn]), parity(p + 1));
      if constexpr (MODE == 1) {
        if (p >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(p - 1));
        restage_item(sh.ft[kn], sh, kn, tid);
        lab::mbar_arrive(lab::smem(&sh.staged[kn]));
      }
      steps<ARM, MODE, JOIN>(st, m, sh, sh.ft[k], kn, lane, pu, pv, p);
      lab::mbar_arrive(lab::smem(&sh.drained[k]));
    }
    if constexpr (MODE == 3) steps<ARM, 3, JOIN>(st, m, sh, sh.ft[0], 0, lane, pu, pv, p);
    if ((JOIN == 0 && (p & 1)) || (JOIN == 2 && ((p + (lab::warp_id() >> 2)) & 1))) join(st);
    mass += m[0] + m[1] + m[2] + m[3];
    m[0] = m[1] = m[2] = m[3] = 0.0f;
  }
  clock.stop(t);
  lab::cp_wait_all();
  uint32_t fold = lab::bits(mass);
  for (int j = 0; j < kLists; ++j)
    for (int n = 0; n < 4; ++n)
      for (int i = 0; i < 4; ++i) fold ^= lab::bits(st.acc[j][n][i]);
  lab::sink(out, fold);
}

constexpr int kPasses = 200;
const double kFloor[2] = {2112, 1056};
const char* kArmName[2] = {"fp32 arm", "fp16 arm"};

template <int ARM, int MODE, int JOIN>
void variant(const lab::Bench& b, const Source& src) {
  const std::string name = lab::kernel("pass", {ARM, MODE, JOIN});
  b.expect(name, ARM == kArmHalf ? "HMMA.16816.F16" : "HMMA.16816.F32", ">=", 32);
  b.expect(name, "LDGSTS", MODE == 3 ? "==" : ">=", MODE == 3 ? 0 : 4);
  if (MODE == 0) b.expect(name, "BAR", ">=", 3);
  if (ARM == kArmHalf && JOIN == 1 && MODE != 2 && MODE != 4) b.expect(name, "BSSY", ">=", 1);
  if (b.expect_only) return;
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(pass<ARM, MODE, JOIN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes));
  const lab::Samples s = b.variant(
      [&] { pass<ARM, MODE, JOIN><<<b.sms, kThreads, bytes>>>(kPasses, src, b.times, b.sinks); },
      [&] { return b.span(8) / (kPasses * kSteps * kFloor[ARM]); });
  b.record(std::string(kArmName[ARM]) + " / " + kModeName[MODE]
               + (ARM == kArmHalf ? std::string(" / ") + kJoinName[JOIN] : ""),
           "x floor", s);
}

//: the fp16 arm under every join; the fp32 arm has none to schedule
template <int ARM>
void arm(const lab::Bench& b, const Source& src) {
  variant<ARM, 0, 0>(b, src);
  variant<ARM, 1, 0>(b, src);
  variant<ARM, 2, 0>(b, src);
  variant<ARM, 3, 0>(b, src);
  variant<ARM, 4, 0>(b, src);
  if constexpr (ARM == kArmHalf) {
    variant<ARM, 0, 1>(b, src);
    variant<ARM, 1, 1>(b, src);
    variant<ARM, 2, 1>(b, src);
    variant<ARM, 3, 1>(b, src);
    variant<ARM, 4, 1>(b, src);
    variant<ARM, 0, 2>(b, src);
    variant<ARM, 1, 2>(b, src);
    variant<ARM, 2, 2>(b, src);
    variant<ARM, 3, 2>(b, src);
    variant<ARM, 4, 2>(b, src);
  }
}

}  // namespace pass_pipeline

int main(int argc, char** argv) {
  using namespace pass_pipeline;
  lab::Bench b(argc, argv);
  Source src{};
  if (!b.expect_only) {
    uint8_t* buf = nullptr;
    const size_t bytes = static_cast<size_t>(kTokens) * kRowBytes;
    LAB_CHECK(cudaMalloc(&buf, bytes));
    LAB_CHECK(cudaMemset(buf, 0x3c, bytes));
    src.run = buf;
    src.values = buf + static_cast<size_t>(kTokens) * 64;
    src.factor = buf + static_cast<size_t>(kTokens) * 128;
    src.dense = buf + static_cast<size_t>(kTokens) * 144;
  }
  arm<kArmHalf>(b, src);
  arm<kArmFloat>(b, src);
}
