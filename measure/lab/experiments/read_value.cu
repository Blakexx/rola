// measure/lab/experiments/read_value.cu -- QUESTION: what does the value-carve readout of a listed side cost a window
// against its floor, when the CTA's 8 warps walk the same digits in lockstep -- each warp its 8 channels of a 64-channel
// box over every leaf, a tile one digit's entries (tokens on M) gathered from the pool, its slice of the digit from the
// image, output scaling, the tile added into the f16 panel by a load, an add and a store (no atomics), the den's own
// tile rotated over the warps -- at alt-k4's ~1 tile a (digit, warp) a window? METHOD: one CTA an SM, 8 warps. A pass
// is a window: `DIGITS` digits of 32 lists each `ENTRIES` entries (distinct slots of 128, a fixed pattern), for each
// digit each warp loads its slice (32 lists x 8 channels, one `ldmatrix.x4.trans`), then per tile of 16 entries the
// rows (two `ldmatrix.x4` from the swizzled pool), two m16n8k16 MMAs (fp32 accumulate, or fp16 at `ACC16`), the
// entries' coefficients on the accumulator, and the panel's load-add-store at each of the lane's two tokens; the den's
// tile (two more MMAs on the den image's slice) by warp `tile mod 8`, its word a token a warp in shared memory; a
// barrier ends the pass (the window's FREE). A sample is the CTA's span over its passes. A rate probe: the panel and a
// sink take the results, unchecked. see docs/lab/read_value.md
#include <cuda_fp16.h>

#include "lab.cuh"

namespace read_value {

constexpr int kWarps = 8, kThreads = kWarps * 32, kLists = 32, kSlots = 128, kPasses = 128;
//: a slice (32 lists x 8 channels f16), a pool row (32 lists f16), a panel row (64 channels f16)
constexpr int kSlice = kLists * 8 * 2, kRow = kLists * 2, kPanelRow = 64 * 2;

template <int DIGITS, int ENTRIES>
struct Layout {
  static constexpr int image = 0, den = image + DIGITS * kWarps * kSlice,
                       pool = den + DIGITS * kSlice, entries = pool + kSlots * kRow,
                       panel = entries + DIGITS * ENTRIES * 8, dens = panel + kSlots * kPanelRow,
                       bytes = dens + kSlots * kWarps * 4;
};

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: a pool row's 16-byte chunk `c` for `slot` (the carry's 64-byte-row swizzle: the residue's high bits)
__device__ __forceinline__ uint32_t pool_at(uint32_t pool, uint32_t slot, uint32_t c) {
  return pool + slot * kRow + ((c ^ ((slot >> 1) & 3u)) << 4);
}

//: a panel row's word for this warp's channel pair `tq` (the warp's 16 bytes XORed by the slot's residue)
__device__ __forceinline__ uint32_t panel_at(uint32_t panel, uint32_t slot, uint32_t warp,
                                             uint32_t tq) {
  return panel + slot * kPanelRow + (((warp ^ (slot & 7u)) << 4) | (tq << 2));
}

template <int DIGITS, int ENTRIES, int ACC16, int DEN>
__global__ void __launch_bounds__(kThreads, 1) readout(long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<DIGITS, ENTRIES>;
  const uint32_t warp = lab::warp_id(), lane = lab::lane_id(), g = lane >> 2, tq = lane & 3;
  //: the image and pool f16 1/64, the entries' slots distinct within a digit, the coefficient 0.5, the panel zero
  for (int k = static_cast<int>(threadIdx.x); k < L::bytes / 4; k += kThreads) {
    uint32_t v = 0;
    if (k * 4 < L::entries) v = 0x24002400u;
    if (k * 4 >= L::entries && k * 4 < L::panel && (k & 1) == 0) {
      const int e = (k * 4 - L::entries) / 8, d = e / ENTRIES, i = e % ENTRIES;
      v = static_cast<uint32_t>((i * 37 + d * 11) & (kSlots - 1)) | (0x3800u << 16);
    }
    reinterpret_cast<uint32_t*>(smem)[k] = v;
  }
  __syncthreads();
  const uint32_t sm = lab::smem(smem), image = sm + L::image, den = sm + L::den,
                 pool = sm + L::pool, ents = sm + L::entries, panel = sm + L::panel,
                 dens = sm + L::dens;
  constexpr int kTiles = (ENTRIES + 15) / 16;
  lab::WarpSpan clock;
  clock.start();
  uint32_t fold = 0;
#pragma unroll 1
  for (int p = 0; p < kPasses; ++p) {
#pragma unroll 1
    for (int d = 0; d < DIGITS; ++d) {
      //: THE DIGIT: this warp's slice, and the den's where one of the digit's tiles is this warp's
      uint32_t bs[4], bd[4] = {};
      lab::ldsm_t(bs, image + static_cast<uint32_t>((d * kWarps + static_cast<int>(warp)) * kSlice)
                          + (lane << 4));
      const bool den_here = DEN && ((static_cast<uint32_t>(d * kTiles) - warp) & 7u) < kTiles;
      if (den_here) lab::ldsm_t(bd, den + static_cast<uint32_t>(d * kSlice) + (lane << 4));
#pragma unroll
      for (int j = 0; j < kTiles; ++j) {
        //: THE TILE: its rows by entry, two MMAs, the coefficients, the panel's load-add-store
        const int base = d * ENTRIES + j * 16, rows = ENTRIES - j * 16 < 16 ? ENTRIES - j * 16 : 16;
        const uint32_t mine = lane & 15u;
        const uint32_t er = lds32(
            ents
            + static_cast<uint32_t>((base + (mine < static_cast<uint32_t>(rows) ? mine : 0u)) * 8));
        const uint32_t slot = er & 0xFFFFu, half = lane >> 4;
        uint32_t a0[4], a1[4];
        lab::ldsm(a0, pool_at(pool, slot, half));
        lab::ldsm(a1, pool_at(pool, slot, 2u + half));
        const uint32_t e0 = lds32(ents + static_cast<uint32_t>((base + static_cast<int>(g)) * 8));
        const uint32_t e1 =
            lds32(ents
                  + static_cast<uint32_t>(
                      (base + (g + 8u < static_cast<uint32_t>(rows) ? g + 8u : g)) * 8));
        const float c0 = __half2float(__ushort_as_half(static_cast<unsigned short>(e0 >> 16)));
        const float c1 = __half2float(__ushort_as_half(static_cast<unsigned short>(e1 >> 16)));
        const uint32_t s0 = e0 & 0xFFFFu, s1 = e1 & 0xFFFFu;
        uint32_t w0, w1;
        if constexpr (ACC16) {
          uint32_t acc[2] = {0u, 0u};
          lab::mma_f16_f16(acc, a0, bs[0], bs[1]);
          lab::mma_f16_f16(acc, a1, bs[2], bs[3]);
          const __half2 h0 = __half2half2(__float2half_rn(c0)),
                        h1 = __half2half2(__float2half_rn(c1));
          w0 = lab::hmul2(acc[0], *reinterpret_cast<const uint32_t*>(&h0));
          w1 = lab::hmul2(acc[1], *reinterpret_cast<const uint32_t*>(&h1));
        } else {
          float acc[4] = {};
          lab::mma_f16_f32(acc, a0, bs[0], bs[1]);
          lab::mma_f16_f32(acc, a1, bs[2], bs[3]);
          w0 = lab::pack_half2(acc[0] * c0, acc[1] * c0);
          w1 = lab::pack_half2(acc[2] * c1, acc[3] * c1);
        }
        const uint32_t p0 = panel_at(panel, s0, warp, tq), p1 = panel_at(panel, s1, warp, tq);
        const uint32_t o0 = lds32(p0);
        sts32(p0, hadd2(o0, w0));
        if (g + 8u < static_cast<uint32_t>(rows)) {
          const uint32_t o1 = lds32(p1);
          sts32(p1, hadd2(o1, w1));
        }
        fold ^= o0;
        if (DEN && ((static_cast<uint32_t>(d * kTiles + j) & 7u) == warp)) {
          //: THE DEN'S TILE: two MMAs on the den slice, the token's word in this warp's den column
          float dacc[4] = {};
          lab::mma_f16_f32(dacc, a0, bd[0], bd[1]);
          lab::mma_f16_f32(dacc, a1, bd[2], bd[3]);
          if (tq == 0) {
            const uint32_t q0 = dens + (s0 * kWarps + warp) * 4u,
                           q1 = dens + (s1 * kWarps + warp) * 4u;
            sts32(q0, __float_as_uint(__uint_as_float(lds32(q0)) + dacc[0] * c0));
            sts32(q1, __float_as_uint(__uint_as_float(lds32(q1)) + dacc[2] * c1));
          }
        }
      }
    }
    //: THE WINDOW'S END: every warp's readout done (FREE)
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, fold);
}

//: A TILE'S OPERANDS, loaded a tile ahead: the digit's slices, the rows, the lane's two tokens' entries
struct Stage {
  uint32_t bs[4], bd[4], a0[4], a1[4], e0, e1;
};

template <int DIGITS, int ENTRIES, int DEN>
__device__ __forceinline__ void stage_in(Stage& s, int idx, uint32_t image, uint32_t den,
                                         uint32_t pool, uint32_t ents, uint32_t warp,
                                         uint32_t lane) {
  constexpr int kTiles = (ENTRIES + 15) / 16;
  const int d = idx / kTiles, j = idx % kTiles, base = d * ENTRIES + j * 16;
  const int rows = ENTRIES - j * 16 < 16 ? ENTRIES - j * 16 : 16;
  const uint32_t g = lane >> 2, mine = lane & 15u;
  lab::ldsm_t(s.bs, image + static_cast<uint32_t>((d * kWarps + static_cast<int>(warp)) * kSlice)
                        + (lane << 4));
  if (DEN && ((static_cast<uint32_t>(idx) & 7u) == warp))
    lab::ldsm_t(s.bd, den + static_cast<uint32_t>(d * kSlice) + (lane << 4));
  const uint32_t er = lds32(
      ents + static_cast<uint32_t>((base + (mine < static_cast<uint32_t>(rows) ? mine : 0u)) * 8));
  lab::ldsm(s.a0, pool_at(pool, er & 0xFFFFu, lane >> 4));
  lab::ldsm(s.a1, pool_at(pool, er & 0xFFFFu, 2u + (lane >> 4)));
  s.e0 = lds32(ents + static_cast<uint32_t>((base + static_cast<int>(g)) * 8));
  s.e1 = lds32(
      ents
      + static_cast<uint32_t>((base + (g + 8u < static_cast<uint32_t>(rows) ? g + 8u : g)) * 8));
}

template <int ENTRIES, int DEN, int SPLIT>
__device__ __forceinline__ uint32_t stage_out(const Stage& s, int idx, uint32_t panel,
                                              uint32_t dens, uint32_t warp, uint32_t lane) {
  constexpr int kTiles = (ENTRIES + 15) / 16;
  const int j = idx % kTiles, rows = ENTRIES - j * 16 < 16 ? ENTRIES - j * 16 : 16;
  const uint32_t g = lane >> 2, tq = lane & 3;
  const float c0 = __half2float(__ushort_as_half(static_cast<unsigned short>(s.e0 >> 16)));
  const float c1 = __half2float(__ushort_as_half(static_cast<unsigned short>(s.e1 >> 16)));
  const uint32_t s0 = s.e0 & 0xFFFFu, s1 = s.e1 & 0xFFFFu;
  float acc[4] = {};
  if constexpr (SPLIT) {
    //: two independent accumulators, one a k-step, summed: no MMA waits on the other
    float acc1[4] = {};
    lab::mma_f16_f32(acc, s.a0, s.bs[0], s.bs[1]);
    lab::mma_f16_f32(acc1, s.a1, s.bs[2], s.bs[3]);
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc1[k];
  } else {
    lab::mma_f16_f32(acc, s.a0, s.bs[0], s.bs[1]);
    lab::mma_f16_f32(acc, s.a1, s.bs[2], s.bs[3]);
  }
  const uint32_t p0 = panel_at(panel, s0, warp, tq), p1 = panel_at(panel, s1, warp, tq);
  const uint32_t o0 = lds32(p0);
  sts32(p0, hadd2(o0, lab::pack_half2(acc[0] * c0, acc[1] * c0)));
  if (g + 8u < static_cast<uint32_t>(rows))
    sts32(p1, hadd2(lds32(p1), lab::pack_half2(acc[2] * c1, acc[3] * c1)));
  if (DEN && ((static_cast<uint32_t>(idx) & 7u) == warp)) {
    float dacc[4] = {};
    lab::mma_f16_f32(dacc, s.a0, s.bd[0], s.bd[1]);
    lab::mma_f16_f32(dacc, s.a1, s.bd[2], s.bd[3]);
    if (tq == 0) {
      const uint32_t q0 = dens + (s0 * kWarps + warp) * 4u, q1 = dens + (s1 * kWarps + warp) * 4u;
      sts32(q0, __float_as_uint(__uint_as_float(lds32(q0)) + dacc[0] * c0));
      sts32(q1, __float_as_uint(__uint_as_float(lds32(q1)) + dacc[2] * c1));
    }
  }
  return o0;
}

//: THE PIPELINED FORM: the window's (digit, tile) sequence flattened, two stages, each tile's operands loaded while
//: the tile before it computes
template <int DIGITS, int ENTRIES, int DEN, int SPLIT>
__global__ void __launch_bounds__(kThreads, 1) readout_pipe(long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<DIGITS, ENTRIES>;
  const uint32_t warp = lab::warp_id(), lane = lab::lane_id();
  for (int k = static_cast<int>(threadIdx.x); k < L::bytes / 4; k += kThreads) {
    uint32_t v = 0;
    if (k * 4 < L::entries) v = 0x24002400u;
    if (k * 4 >= L::entries && k * 4 < L::panel && (k & 1) == 0) {
      const int e = (k * 4 - L::entries) / 8, d = e / ENTRIES, i = e % ENTRIES;
      v = static_cast<uint32_t>((i * 37 + d * 11) & (kSlots - 1)) | (0x3800u << 16);
    }
    reinterpret_cast<uint32_t*>(smem)[k] = v;
  }
  __syncthreads();
  const uint32_t sm = lab::smem(smem), image = sm + L::image, den = sm + L::den,
                 pool = sm + L::pool, ents = sm + L::entries, panel = sm + L::panel,
                 dens = sm + L::dens;
  constexpr int kTotal = DIGITS * ((ENTRIES + 15) / 16);
  static_assert(kTotal % 2 == 0, "the two stages alternate over an even count");
  lab::WarpSpan clock;
  clock.start();
  uint32_t fold = 0;
  Stage sa, sb;
#pragma unroll 1
  for (int p = 0; p < kPasses; ++p) {
    stage_in<DIGITS, ENTRIES, DEN>(sa, 0, image, den, pool, ents, warp, lane);
#pragma unroll 1
    for (int idx = 0; idx < kTotal; idx += 2) {
      //: STAGE B IN, STAGE A OUT; then A IN (the next pair's first), B OUT
      stage_in<DIGITS, ENTRIES, DEN>(sb, idx + 1, image, den, pool, ents, warp, lane);
      fold ^= stage_out<ENTRIES, DEN, SPLIT>(sa, idx, panel, dens, warp, lane);
      if (idx + 2 < kTotal)
        stage_in<DIGITS, ENTRIES, DEN>(sa, idx + 2, image, den, pool, ents, warp, lane);
      fold ^= stage_out<ENTRIES, DEN, SPLIT>(sb, idx + 1, panel, dens, warp, lane);
    }
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, fold);
}

template <int DIGITS, int ENTRIES, int DEN, int SPLIT>
void variant_pipe(const lab::Bench& b) {
  using L = Layout<DIGITS, ENTRIES>;
  const std::string name = lab::kernel("readout_pipe", {DIGITS, ENTRIES, DEN, SPLIT});
  b.expect(name, "HMMA", ">=", 2);
  b.expect(name, "LDSM", ">=", 3);
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a rate probe: its results fold into the panel and a sink");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(readout_pipe<DIGITS, ENTRIES, DEN, SPLIT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, L::bytes));
  constexpr int kTiles = (ENTRIES + 15) / 16;
  lab::Samples tiles;
  const lab::Samples pass = b.variant(
      [&] {
        readout_pipe<DIGITS, ENTRIES, DEN, SPLIT><<<b.sms, kThreads, L::bytes>>>(b.times, b.sinks);
      },
      [&] {
        std::vector<long long> h(static_cast<size_t>(b.sms) * kWarps * 2);
        LAB_CHECK(
            cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double c = 0;
        for (int s = 0; s < b.sms; ++s) {
          long long lo = h[2 * s * kWarps], hi = h[2 * s * kWarps + 1];
          for (int w = 1; w < kWarps; ++w) {
            lo = std::min(lo, h[2 * (s * kWarps + w)]);
            hi = std::max(hi, h[2 * (s * kWarps + w) + 1]);
          }
          c += double(hi - lo) / kPasses;
        }
        tiles.v.push_back(c / b.sms / (DIGITS * kTiles));
        return c / b.sms;
      });
  const std::string row = std::to_string(DIGITS) + " digits x " + std::to_string(ENTRIES)
                          + " entries a window, fp32 accumulate"
                          + (DEN ? ", the den's tile" : ", no den") + ", pipelined a tile ahead"
                          + (SPLIT ? ", k-steps split" : "");
  b.unchecked(row, "a rate probe: its results fold into the panel and a sink");
  b.record(row, "cycles a window a CTA", pass);
  b.record(row + ": cycles a tile (all 8 warps)", "cycles", tiles);
}

template <int DIGITS, int ENTRIES, int ACC16, int DEN>
void variant(const lab::Bench& b) {
  using L = Layout<DIGITS, ENTRIES>;
  const std::string name = lab::kernel("readout", {DIGITS, ENTRIES, ACC16, DEN});
  b.expect(name, "HMMA", ">=", 2);
  b.expect(name, "LDSM", ">=", 3);
  //: the digit loop, the den's turn and a partial tile's second token are warp-uniform branches: the form under test
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a rate probe: its results fold into the panel and a sink");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(readout<DIGITS, ENTRIES, ACC16, DEN>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, L::bytes));
  lab::Samples tiles;
  constexpr int kTiles = (ENTRIES + 15) / 16;
  const lab::Samples pass = b.variant(
      [&] {
        readout<DIGITS, ENTRIES, ACC16, DEN><<<b.sms, kThreads, L::bytes>>>(b.times, b.sinks);
      },
      [&] {
        std::vector<long long> h(static_cast<size_t>(b.sms) * kWarps * 2);
        LAB_CHECK(
            cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double c = 0;
        for (int s = 0; s < b.sms; ++s) {
          long long lo = h[2 * s * kWarps], hi = h[2 * s * kWarps + 1];
          for (int w = 1; w < kWarps; ++w) {
            lo = std::min(lo, h[2 * (s * kWarps + w)]);
            hi = std::max(hi, h[2 * (s * kWarps + w) + 1]);
          }
          c += double(hi - lo) / kPasses;
        }
        tiles.v.push_back(c / b.sms / (DIGITS * kTiles));
        return c / b.sms;
      });
  const std::string row = std::to_string(DIGITS) + " digits x " + std::to_string(ENTRIES)
                          + " entries a window, " + (ACC16 ? "fp16" : "fp32") + " accumulate"
                          + (DEN ? ", the den's tile" : ", no den");
  b.unchecked(row, "a rate probe: its results fold into the panel and a sink");
  b.record(row, "cycles a window a CTA", pass);
  b.record(row + ": cycles a tile (all 8 warps)", "cycles", tiles);
}

}  // namespace read_value

int main(int argc, char** argv) {
  using namespace read_value;
  lab::Bench b(argc, argv);
  //: alt-k4's ~16 entries a digit a window at W 1,024 (8 at 512), alt-k16's ~32-64; half a 16-digit box (sm_86's area)
  variant<8, 16, 0, 1>(b);
  variant<8, 16, 1, 1>(b);
  variant<8, 16, 0, 0>(b);
  variant<8, 8, 0, 1>(b);
  variant<8, 32, 0, 1>(b);
  variant<8, 64, 0, 1>(b);
  variant_pipe<8, 16, 1, 0>(b);
  variant_pipe<8, 16, 0, 0>(b);
  variant_pipe<8, 8, 1, 0>(b);
  variant_pipe<8, 32, 1, 0>(b);
  variant_pipe<8, 16, 1, 1>(b);
  variant_pipe<8, 8, 1, 1>(b);
}
