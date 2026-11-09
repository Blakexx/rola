// measure/lab/experiments/warp_hiding.cu -- QUESTION: the carry's window is latency-bound at two warps a scheduler
// (its register-resident state allows no more): would more warps hide that latency, or does a pipe bind the step's
// instruction mix? METHOD: each of the carry's step chains, rebuilt from the primitives with the production
// dependency structure on synthetic shared rows, run by 4, 8 and 16 warps a CTA (1, 2, 4 a scheduler; one CTA an SM
// by 80 KB of dynamic shared memory), every warp the same loop: the READOUT tile pair (an entry's shared load, the
// unit's byte, four shuffles, one `ldmatrix`, four fp16-accumulate MMAs into a fresh fragment, two exact pairs, four
// shared atomics -- `read_chunk`'s `Pairs` body), the FOLD's 8-token step (an entry, a shuffle and a packed multiply,
// a shuffle, two `ldmatrix.trans`, four packed multiplies, eight k8 fp32-accumulate MMAs into live accumulators,
// alternating two lists -- `fold_chunk`'s `step8`), and the read FILL (a token's shared load, a 64-bit address, two
// predicated 16-byte `cp.async`, four slots a thread ahead -- `fill`), and the HAND-OFF of a warp's lists (each
// list's digit maxima over its 32 fp32 accumulators a lane: absolute maxima in chains of eight, the quad's two
// shuffles, the leaf byte stored, one shared atomic a lane; then every value scaled, packed to f16 and stored 16 bytes
// a row -- `post_keys` and `store_batches`), 4 lists a warp at 4 and 8 warps and 2 lists a warp at 16 (the state's
// 32 registers a list: a 16-warp CTA holds the same box as an 8-warp one). A warp's cycles a step is the chain under the
// others' load; the warps a scheduler divided by it is the scheduler's throughput: flat cycles a step across warp
// counts means the latency hides (the state's two warps are the bound), cycles growing with the warps means a pipe
// binds (the mix is). see docs/internals/tools/lab.md#warp-hiding
#include "lab.cuh"

namespace warp_hiding {

constexpr int kSlots = 256, kRow = 128, kEntries = 1024, kSteps = 2000;
constexpr int kPool = 0, kPanel = kPool + kSlots * kRow, kWords = kPanel + kSlots * 64;
constexpr int kUnits = kWords + kEntries * 8, kBytes = 80 * 1024;
constexpr int kForms = 4;
const char* kForm[kForms] = {"readout tile pair", "fold step8", "read fill", "hand-off (maxima + store)"};

__device__ __forceinline__ uint32_t chunk_at(uint32_t base, uint32_t slot, uint32_t c) {
  return base + kRow * slot + 16u * (c ^ (slot & 7u));
}

__device__ __forceinline__ void mma_f16_f16_fresh(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0u));
}

__device__ __forceinline__ void mma_k8_f32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void atoms_add(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t s16_pair(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

__device__ __forceinline__ void copy16(uint32_t dst, const void* src, bool issue, bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16, %3;\n}\n" ::"r"(dst),
      "l"(src), "r"(static_cast<int>(issue)), "r"(live ? 16 : 0));
}

//: the synthetic tenants: rows of f16 near 1 (lane-dependent), the panel zero, entries `e`: its slot `e & 255` with
//: its place `e & 511` above, coefficient f16 near 1 with its panel row `64 (e & 255)` above; units a byte a place
__device__ void init(uint8_t* smem, int tid, int threads) {
  for (int i = tid; i < kSlots * kRow / 4; i += threads)
    reinterpret_cast<uint32_t*>(smem + kPool)[i] = 0x3c003c00u ^ (0x00010001u * (i & 7));
  for (int i = tid; i < kSlots * 64 / 4; i += threads) reinterpret_cast<uint32_t*>(smem + kPanel)[i] = 0u;
  for (int e = tid; e < kEntries; e += threads) {
    const uint32_t slot = e & (kSlots - 1), place = e & 511;
    reinterpret_cast<uint2*>(smem + kWords)[e] =
        make_uint2(slot | place << 16, 0x3c00u | ((64u * slot) << 16));
  }
  for (int i = tid; i < 512; i += threads) smem[kUnits + i] = static_cast<uint8_t>(i & 3);
}

template <int WARPS, int FORM>
__global__ void __launch_bounds__(32 * WARPS, 1) hide(int steps, const uint8_t* rows, long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = lab::lane_id(), warp = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  const int g = lane >> 2, tq = lane & 3;
  init(smem, tid, 32 * WARPS);
  __syncthreads();
  const uint32_t pool = lab::smem(smem + kPool), panel = lab::smem(smem + kPanel), words = lab::smem(smem + kWords),
                 units = lab::smem(smem + kUnits);
  uint32_t fold = 0;
  lab::WarpClock clock;

  if constexpr (FORM == 0) {
    //: the two digits' state fragments, held as `state_in` holds them
    uint32_t f[2][16];
    for (int q = 0; q < 2; ++q)
      for (int k = 0; k < 4; ++k) {
        uint32_t a[4];
        lab::ldsm_t(a, chunk_at(pool, (8 * q + k + warp) & (kSlots - 1), k));
        for (int i = 0; i < 4; ++i) f[q][4 * k + i] = a[i];
      }
    clock.start();
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      uint32_t acc[2][2][2];
      uint32_t e0[2], e1[2];
      float c0[2], c1[2];
      uint32_t b[2][4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint2 v;
        const uint32_t at = words + 8u * ((16 * s + 8 * q + (lane & 7) + 64 * warp) & (kEntries - 1));
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(at));
        int u;
        asm volatile("ld.shared.s8 %0, [%1];" : "=r"(u) : "r"(units + (v.x >> 16)));
        const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(v.y)));
        const float cu = __uint_as_float(__float_as_uint(c) + (static_cast<uint32_t>(6 - u) << 23));
        e0[q] = __shfl_sync(0xffffffffu, v.y, 2 * tq);
        e1[q] = __shfl_sync(0xffffffffu, v.y, 2 * tq + 1);
        c0[q] = __shfl_sync(0xffffffffu, cu, 2 * tq);
        c1[q] = __shfl_sync(0xffffffffu, cu, 2 * tq + 1);
        lab::ldsm(b[q], chunk_at(pool, v.x & 0xffffu, static_cast<uint32_t>(lane >> 3)));
      }
#pragma unroll
      for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const uint32_t a0[4] = {f[q][8 * m], f[q][8 * m + 1], f[q][8 * m + 2], f[q][8 * m + 3]};
          const uint32_t a1[4] = {f[q][8 * m + 4], f[q][8 * m + 5], f[q][8 * m + 6], f[q][8 * m + 7]};
          mma_f16_f16_fresh(acc[q][m], a0, b[q][0], b[q][1]);
          lab::mma_f16_f16(acc[q][m], a1, b[q][2], b[q][3]);
        }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t p0 = panel + (e0[q] >> 16) + 4u * g, p1 = panel + (e1[q] >> 16) + 4u * g;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&acc[q][m][0]));
          const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&acc[q][m][1]));
          atoms_add(p0 ^ (32u * m), s16_pair(x.x, y.x, c0[q]));
          atoms_add(p1 ^ (32u * m), s16_pair(x.y, y.y, c1[q]));
        }
      }
    }
    clock.stop(t);
    fold = reinterpret_cast<uint32_t*>(smem + kPanel)[tid & 1023];
  } else if constexpr (FORM == 1) {
    float acc[2][2][4][4];
    for (int j = 0; j < 2; ++j)
      for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n)
          for (int i = 0; i < 4; ++i) acc[j][m][n][i] = 0.0f;
    const uint32_t cf = 0x3c003c00u;
    const int r = lane & 7, q = lane >> 3, t4 = lane & 3;
    clock.start();
#pragma unroll 1
    for (int s = 0; s < steps; s += 2) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint2 v;
        const uint32_t at = words + 8u * ((8 * (s + j) + (lane & 7) + 64 * warp) & (kEntries - 1));
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(at));
        const uint32_t slot = v.x & 0xffffu;
        const uint32_t mine = v.y & 0xffffu;
        const uint32_t cpair = lab::hmul2(mine | (__shfl_sync(0xffffffffu, mine, lane ^ 1) << 16), cf);
        const uint32_t o = __shfl_sync(0xffffffffu, slot, r), c01 = __shfl_sync(0xffffffffu, cpair, 2 * t4);
        uint32_t a[4], b[4];
        lab::ldsm_t(a, chunk_at(pool, o, static_cast<uint32_t>(q)));
        lab::ldsm_t(b, chunk_at(pool, o, 4u + static_cast<uint32_t>(q)));
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = lab::hmul2(a[k], c01);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n) mma_k8_f32(acc[j][m][n], a[2 * m], a[2 * m + 1], b[n]);
      }
    }
    clock.stop(t);
    for (int j = 0; j < 2; ++j)
      for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n) fold ^= lab::bits(acc[j][m][n][0]) ^ lab::bits(acc[j][m][n][3]);
  } else if constexpr (FORM == 2) {
    //: the read fill: a quad a slot, lane `r` chunk `r` of the slot's 64-byte slice; a thread's slots four at a time,
    //: their tokens loaded before any copy; a predicated copy a slot (every slot live here)
    const int quads = 8 * WARPS, rq = lane & 3, qd = tid >> 2;
    const uint8_t* const base = rows + 16 * rq;
    clock.start();
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      int u[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int slot = (qd + quads * k + 1024 * s) & 511;
        uint32_t tok;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(tok) : "r"(words + 8u * static_cast<uint32_t>(slot) + 2u));
        u[k] = static_cast<int>(tok & 0x1ffu);
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int slot = (qd + quads * k) & (kSlots - 1);
        const uint8_t* const src = base + static_cast<int64_t>(u[k]) * 64 + static_cast<int64_t>(s & 7) * 32768;
        copy16(chunk_at(pool, static_cast<uint32_t>(slot), static_cast<uint32_t>(rq)), src, true, true);
      }
      if ((s & 7) == 7) lab::cp_wait_all();
    }
    lab::cp_wait_all();
    clock.stop(t);
    fold = reinterpret_cast<uint32_t*>(smem + kPool)[tid & 4095];
  } else {
    //: the hand-off: `kLists` lists' accumulators a lane (rows g, g + 8 of two m-tiles, four n-tiles), the digit
    //: maxima then the stores, the keys' atomics and the stores' rows at the lane's addresses
    constexpr int kLists = WARPS == 16 ? 2 : 4;
    //: the state: values ptxas cannot fold (a shared load each), held in place every step so the maxima are live
    float acc[kLists][2][4][4];
    for (int j = 0; j < kLists; ++j)
      for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n)
          for (int i = 0; i < 4; ++i)
            acc[j][m][n][i] = __uint_as_float(0x3f800000u | (reinterpret_cast<const uint32_t*>(
                                                                 smem + kPool)[(lane + 97 * (16 * j + 4 * m + n) + i) & 4095]
                                                             & 0x007fffffu));
    const int t4 = lane & 3;
    const uint32_t keys = units, leaves = panel + 2048u * static_cast<uint32_t>(warp & 7), image = pool;
    clock.start();
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      //: the maxima: a list's 8 values a (M, H) to one key, the quad's reduce-scatter, the leaf byte, the digit key
      int key[2][2];
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          int leaf[kLists];
#pragma unroll
          for (int j = 0; j < kLists; ++j) {
            float x = fmaxf(fabsf(acc[j][m][0][2 * h]), fabsf(acc[j][m][0][2 * h + 1]));
#pragma unroll
            for (int n = 1; n < 4; ++n)
              x = fmaxf(x, fmaxf(fabsf(acc[j][m][n][2 * h]), fabsf(acc[j][m][n][2 * h + 1])));
            leaf[j] = __float_as_int(x) + (j << 20) + s;
          }
          int mine = leaf[t4 & (kLists - 1)];
          mine = max(mine, __shfl_xor_sync(0xffffffffu, mine, 2));
          mine = max(mine, __shfl_xor_sync(0xffffffffu, mine, 1));
          int k = max(mine, __shfl_xor_sync(0xffffffffu, mine, 1));
          key[m][h] = max(k, __shfl_xor_sync(0xffffffffu, k, 2));
          const int byte = min(max(((mine + 0x7fffff) >> 23) - 127, -127), 127);
          asm volatile("st.shared.u8 [%0], %1;" ::"r"(leaves + 32u * (16 * m + 8 * h + g) + static_cast<uint32_t>(t4)),
                       "r"(byte)
                       : "memory");
        }
      const int k = t4 & 2 ? (t4 & 1 ? key[1][1] : key[1][0]) : (t4 & 1 ? key[0][1] : key[0][0]);
      asm volatile("red.shared.max.s32 [%0], %1;" ::"r"(keys + 4u * ((s & 1) * 32 + lane)), "r"(k) : "memory");
      //: the store: each list's rows scaled to the digit's power of two, packed to f16, one 16-byte store a row
      const float sc = __int_as_float((120 + (s & 7)) << 23);
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < kLists; ++j) {
            uint32_t v[4];
#pragma unroll
            for (int n = 0; n < 4; ++n)
              v[n] = lab::pack_half2(acc[j][m][n][2 * h] * sc, acc[j][m][n][2 * h + 1] * sc);
            const uint32_t to = image + 2048u * (16 * m + 8 * h) + 64u * static_cast<uint32_t>(g)
                                + 16u * static_cast<uint32_t>(t4 ^ (g & 3)) + 512u * static_cast<uint32_t>(j);
            asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(to), "r"(v[0]), "r"(v[1]), "r"(v[2]),
                         "r"(v[3])
                         : "memory");
          }
#pragma unroll
      for (int j = 0; j < kLists; ++j)
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][m][n][i] = __uint_as_float(lab::held(lab::bits(acc[j][m][n][i])));
    }
    clock.stop(t);
    fold = reinterpret_cast<uint32_t*>(smem + kPool)[tid & 4095] ^ reinterpret_cast<uint32_t*>(smem + kUnits)[lane];
  }
  lab::sink(out, fold);
}

template <int WARPS, int FORM>
void variant(const lab::Bench& b, const uint8_t* rows) {
  const std::string name = lab::kernel("hide", {WARPS, FORM});
  if (FORM == 0) {
    b.expect(name, "HMMA.16816.F16", ">=", 8);
    b.expect(name, "LDSM", ">=", 2);
    b.expect(name, "ATOMS", ">=", 8);
    b.exempt(name, "a latency probe of the readout tile's chain at 1, 2 and 4 warps a scheduler");
  } else if (FORM == 1) {
    b.expect(name, "HMMA.1688.F32", ">=", 16);
    b.expect(name, "LDSM", ">=", 4);
    b.exempt(name, "a latency probe of the fold step's chain at 1, 2 and 4 warps a scheduler");
  } else if (FORM == 2) {
    b.expect(name, "LDGSTS", ">=", 4);
    b.units(name, "slot", static_cast<long long>(kSteps) * 4 * 8 * WARPS);
  } else {
    b.expect(name, "FMNMX", ">=", 56);
    b.expect(name, "SHFL", ">=", 16);
    b.expect(name, "STS", ">=", 8);
    b.units(name, "list hand-off", static_cast<long long>(kSteps) * (WARPS == 16 ? 2 : 4) * WARPS);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(hide<WARPS, FORM>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const std::string label = std::string(kForm[FORM]) + ", " + std::to_string(WARPS / 4) + " warps a scheduler";
  const lab::Samples s = b.variant(
      [&] { hide<WARPS, FORM><<<b.sms, 32 * WARPS, kBytes>>>(kSteps, rows, b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(WARPS), WARPS, 0, WARPS) / kSteps; });
  b.unchecked(label, "a timing probe: the outputs fold into the sink");
  b.record(label + ": cycles a step a warp", "cycles", s);
  lab::Samples thr;
  for (double c : s.v) thr.v.push_back(1000.0 * (WARPS / 4.0) / c);
  b.record(label + ": steps a 1K cycles a scheduler", "steps", thr);
}

}  // namespace warp_hiding

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  uint8_t* rows = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&rows, 8 * 32768 + 65536));
    LAB_CHECK(cudaMemset(rows, 0x3c, 8 * 32768 + 65536));
  }
  warp_hiding::variant<4, 0>(b, rows);
  warp_hiding::variant<8, 0>(b, rows);
  warp_hiding::variant<16, 0>(b, rows);
  warp_hiding::variant<4, 1>(b, rows);
  warp_hiding::variant<8, 1>(b, rows);
  warp_hiding::variant<16, 1>(b, rows);
  warp_hiding::variant<4, 2>(b, rows);
  warp_hiding::variant<8, 2>(b, rows);
  warp_hiding::variant<16, 2>(b, rows);
  warp_hiding::variant<4, 3>(b, rows);
  warp_hiding::variant<8, 3>(b, rows);
  warp_hiding::variant<16, 3>(b, rows);
}
