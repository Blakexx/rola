// measure/lab/experiments/fold_step.cu -- QUESTION: the fp16 fold's dense step (32 fp16-accumulate HMMAs, one fresh
// fp32 mass HMMA, its operands by ldmatrix, its coefficients from list bytes, 16 shuffles, 32 HMUL2) runs 1.23x its
// floor in the owner part; which of its parts costs that, and does a register stage (the next step's operands formed
// under this step's MMAs) recover it at the arm's real register pressure? METHOD: the step transplanted from
// `owner_fold.cuh` (dense side, two warps a scheduler, the arm's accumulators live, the fp16 partials joined every 16
// steps), over restaged rows that stay in shared memory; one variant a removal (a class replaced by a LOP3 against an
// immediate, so nothing downstream folds) or the staged form; cycles a step a warp over the steps alone.
// see docs/internals/tools/lab.md#fold-step
#include "lab.cuh"

namespace fold_step {

constexpr int kLists = 8, kRows = 128, kStep = 16;
constexpr int kArmFloat = 0, kArmHalf = 1;

//: MODE 0 as built; 1 staged (the next step's operands and coefficients formed under this step's MMAs); 2 SHFL ->
//: LOP3; 3 HMUL2 -> LOP3; 4 no coefficient loads or math; 5 no operand ldmatrix; 6 the MMAs alone; 7 the operand
//: loads staged alone; 8 the coefficients staged alone; 9 as built with a CTA barrier every pass (8 steps), as the
//: owner part synchronizes
constexpr int kModes = 10;
const char* kModeName[kModes] = {
    "as built",         "staged",     "shuffles -> lop3", "hmul2 -> lop3",       "no coefficient",
    "no operand loads", "mmas alone", "loads staged",     "coefficients staged", "barrier a pass"};

struct Tiles {
  uint16_t dense[kRows][24];
  uint16_t vals[kRows][40];
  uint8_t write[kRows][64];
  float2 fm[kRows];
};

//: one step's operands: the dense digits (A), the values (B, four n-tiles), the lane's value and mass coefficients
struct Operands {
  uint32_t a[4], b[4][2], cv[2], cm[2];
};

//: the step's A and B by `ldmatrix.trans` (none when the variant removes the operand loads)
template <int MODE>
__device__ __forceinline__ void fetch(Operands& o, const Tiles& t, int e0, int lane) {
  if constexpr (MODE != 5 && MODE != 6) {
    const int r = lane & 7, q = lane >> 3;
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
  }
}

//: the lane's value and mass coefficients at its four tokens, or (removed) a loop-variant flip without loads, so the
//: shuffles and HMUL2 that read them stay in the loop
template <int MODE>
__device__ __forceinline__ void coef(Operands& o, const Tiles& t, int e0, int lane, float pu,
                                     float pv) {
  if constexpr (MODE != 4 && MODE != 6) {
    const int tq = lane & 3, byte = lab::warp_id() * kLists + (lane >> 2);
    float v[4], m[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int row = e0 + 2 * tq + (k & 1) + 8 * (k >> 1);
      const float b = fmaxf(static_cast<float>(t.write[row][byte]), 2.0f);
      const float2 f = t.fm[row];
      v[k] = __fmul_rn(__fmul_rn(b, f.x), pu);
      m[k] = __fmul_rn(__fmul_rn(b, f.y), pv);
    }
    o.cv[0] = lab::pack_half2(v[0], v[1]);
    o.cv[1] = lab::pack_half2(v[2], v[3]);
    o.cm[0] = lab::pack_half2(m[0], m[1]);
    o.cm[1] = lab::pack_half2(m[2], m[3]);
  } else {
    const uint32_t flip = static_cast<uint32_t>(e0) & 1u;
    o.cv[0] ^= flip;
    o.cv[1] ^= flip;
    o.cm[0] ^= flip;
  }
}

template <int ARM>
struct Acc {
  float acc[kLists][4][4];
  uint32_t part[ARM == kArmHalf ? kLists : 1][4][2];
};

template <int ARM, int MODE>
__device__ __forceinline__ void step(Acc<ARM>& st, float (&m)[4], const Operands& o, int lane) {
  const int tq = lane & 3;
  float d[4];
  lab::mma_f16_f32_fresh(d, o.a, o.cm[0], o.cm[1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) m[i] += d[i];
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    uint32_t c01, c89, aj[4];
    if constexpr (MODE == 2) {
      c01 = o.cv[0] ^ (0x00010001u * j);
      c89 = o.cv[1] ^ (0x00010001u * j);
    } else {
      c01 = __shfl_sync(0xffffffffu, o.cv[0], 4 * j + tq);
      c89 = __shfl_sync(0xffffffffu, o.cv[1], 4 * j + tq);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t c = i < 2 ? c01 : c89;
      aj[i] = MODE == 6 ? o.a[i] : MODE == 3 ? o.a[i] ^ c : lab::hmul2(o.a[i], c);
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      if constexpr (ARM == kArmFloat)
        lab::mma_f16_f32(st.acc[j][n], aj, o.b[n][0], o.b[n][1]);
      else
        lab::mma_f16_f16(st.part[j][n], aj, o.b[n][0], o.b[n][1]);
    }
  }
}

template <int ARM>
__device__ __forceinline__ void join(Acc<ARM>& st) {
  if constexpr (ARM == kArmHalf) {
#pragma unroll
    for (int j = 0; j < kLists; ++j)
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

template <int ARM, int MODE>
__global__ void __launch_bounds__(256, 1) fold(int windows, int steps, long long* t, int* out) {
  __shared__ Tiles tiles;
  const int lane = lab::lane_id();
  for (int i = threadIdx.x; i < kRows; i += blockDim.x) {
    for (int c = 0; c < 24; ++c)
      tiles.dense[i][c] = static_cast<uint16_t>(0x2c00 + ((i * 7 + c) & 255));
    for (int c = 0; c < 40; ++c)
      tiles.vals[i][c] = static_cast<uint16_t>(0x2c00 + ((i * 5 + c) & 255));
    for (int c = 0; c < 64; ++c) tiles.write[i][c] = static_cast<uint8_t>(i * 3 + c);
    tiles.fm[i] = make_float2(1e-3f * (1 + (i & 7)), 1e-3f * (1 + (i & 3)));
  }
  __syncthreads();
  const float pu = 0.5f + 1e-3f * lane, pv = 0.25f + 1e-3f * lane;
  Acc<ARM> st = {};
  float m[4] = {}, mass = 0.0f;
  Operands cur;
  for (int i = 0; i < 4; ++i) cur.a[i] = 0x2c002c00u ^ (0x00010001u * (lane + i));
  for (int n = 0; n < 4; ++n) cur.b[n][0] = cur.b[n][1] = 0x2c002c00u ^ (lane + n);
  cur.cv[0] = cur.cm[0] = 0x2c002c00u ^ lane;
  cur.cv[1] = cur.cm[1] = 0x2c102c10u ^ lane;
  long long busy = 0;
  if constexpr (MODE == 1 || MODE == 7) fetch<0>(cur, tiles, 0, lane);
  if constexpr (MODE == 1 || MODE == 8) coef<0>(cur, tiles, 0, lane, pu, pv);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    const long long t0 = clock64();
    if constexpr (MODE == 1 || MODE == 7 || MODE == 8) {
      //: two steps a trip, so the stages alternate between two register sets without copies
#pragma unroll 2
      for (int s = 0; s < steps; ++s) {
        const int e0 = (s & 7) * kStep, e1 = ((s + 1) & 7) * kStep;
        Operands next = cur;
        if constexpr (MODE != 8) fetch<0>(next, tiles, e1, lane);
        if constexpr (MODE != 7) coef<0>(next, tiles, e1, lane, pu, pv);
        if constexpr (MODE == 7) coef<0>(cur, tiles, e0, lane, pu, pv);
        if constexpr (MODE == 8) fetch<0>(cur, tiles, e0, lane);
        step<ARM, 0>(st, m, cur, lane);
        cur = next;
      }
    } else {
#pragma unroll 1
      for (int s = 0; s < steps; ++s) {
        fetch<MODE>(cur, tiles, (s & 7) * kStep, lane);
        coef<MODE>(cur, tiles, (s & 7) * kStep, lane, pu, pv);
        step<ARM, MODE>(st, m, cur, lane);
        if (MODE == 9 && (s & 7) == 7) __syncthreads();
      }
    }
    busy += clock64() - t0;
    join(st);
    mass += m[0] + m[1] + m[2] + m[3];
    m[0] = m[1] = m[2] = m[3] = 0.0f;
  }
  if (lane == 0) t[blockIdx.x * 8 + lab::warp_id()] = busy;
  uint32_t fold = lab::bits(mass);
  for (int j = 0; j < kLists; ++j)
    for (int n = 0; n < 4; ++n)
      for (int i = 0; i < 4; ++i) fold ^= lab::bits(st.acc[j][n][i]);
  lab::sink(out, fold);
}

constexpr int kWindows = 200, kSteps = 16;
//: the floor, cycles a step a warp at two warps a scheduler: 32 fp16-acc atoms and the fresh fp32 one at 16 cycles
//: each (`mma_pairing`), or 33 fp32-acc atoms at 32
const double kFloor[2] = {2112, 1056};
const char* kArmName[2] = {"fp32 arm", "fp16 arm"};

template <int ARM, int MODE>
void variant(const lab::Bench& b) {
  const std::string name = lab::kernel("fold", {ARM, MODE});
  b.expect(name, ARM == kArmHalf ? "HMMA.16816.F16" : "HMMA.16816.F32", ">=", 32);
  b.expect(name, "HMMA.16816.F32", ">=", 1);
  if (MODE != 5 && MODE != 6) b.expect(name, "LDSM", ">=", 3);
  if (MODE != 2 && MODE != 6) b.expect(name, "SHFL", ">=", 16);
  if (MODE != 3 && MODE != 6) b.expect(name, "HMUL2", ">=", 32);
  if (MODE != 4 && MODE != 6) b.expect(name, "FMUL", ">=", 16);
  if (b.expect_only) return;
  const lab::Samples s =
      b.variant([&] { fold<ARM, MODE><<<b.sms, 256>>>(kWindows, kSteps, b.times, b.sinks); },
                [&] { return lab::Bench::mean(b.cycles(8), 8, 0, 8) / (kWindows * kSteps); });
  const std::string label = std::string(kArmName[ARM]) + " / " + kModeName[MODE];
  b.record(label, "cycles a step", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / kFloor[ARM]);
  b.record(label + " / floor", "x", over);
}

//: every variant, but no staged one on the fp16 arm: each spills beside its 254 registers (full stage 12 LDL/STL,
//: loads alone 36, coefficients alone 27; CUDA 13.0)
template <int ARM>
void arm(const lab::Bench& b) {
  variant<ARM, 0>(b);
  variant<ARM, 2>(b);
  variant<ARM, 3>(b);
  variant<ARM, 4>(b);
  variant<ARM, 5>(b);
  variant<ARM, 6>(b);
  variant<ARM, 9>(b);
  if constexpr (ARM == kArmFloat) {
    variant<ARM, 1>(b);
    variant<ARM, 7>(b);
    variant<ARM, 8>(b);
  }
}

}  // namespace fold_step

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  fold_step::arm<fold_step::kArmHalf>(b);
  fold_step::arm<fold_step::kArmFloat>(b);
}
