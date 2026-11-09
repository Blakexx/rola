// measure/lab/experiments/pass_fixed.cu -- QUESTION: what does each piece of the fold's fixed work cost alone, and what
// do its levers buy -- the fill (four factor planes, or the factor products precomputed into one 8-byte plane, or those
// and the dense digits precomputed as f16; 64-bit plane addresses or 32-bit offsets from a window base), the restage
// (the dense digits to f16 and the factor products, or the digits alone, or nothing), and the window's lists (each
// warp's entries by a serial loop over every set bit, as built, or word-parallel by ballot and popc)? METHOD: one block a
// SM of 8 warps; the fill a pass of 128 scattered token rows from a 33 MB or 2 MB source, waited and synchronized each
// pass; the restage a pass's 256 items; the lists each warp's 16 union words at a density (1 in 4, 1 in 16) to its
// entries; the block's span a pass or a window. see docs/internals/tools/lab.md#pass-fixed
#include "lab.cuh"

namespace pass_fixed {

constexpr int kThreads = 256, kWarps = 8, kRows = 128, kWindow = 512, kWords = 16;
constexpr int kWindows = 400, kTokens = kWindows * kWindow, kPoolRow = 192;

__device__ __forceinline__ int pool_at(int r, int c) {
  return r * kPoolRow + 16 * (c ^ ((r >> 2) & 1));
}

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}

//: the source planes: the coefficient run (64 B), the values (64 B), four factor planes (4 B each), the factor
//: products (8 B), the dense digits (16 B as u8, 32 B as f16)
struct Source {
  const uint8_t *run, *values, *factor, *fm, *dense8, *dense16;
};

//: FILL 0 as built (four factor planes, the u8 dense chunk); 1 the factor products precomputed (one 8-byte plane);
//: 2 those and the dense digits precomputed as f16 (two 16-byte copies); 3 as 2 with 32-bit offsets from the window's
//: plane bases
constexpr int kFills = 4;
const char* kFillName[kFills] = {
    "fill as built (4 factor planes, u8 dense)", "fill, factor products precomputed",
    "fill, products and f16 dense precomputed", "fill, precomputed, 32-bit offsets"};
//: SOURCE 0 400 windows (33 MB of rows: DRAM), 1 24 windows (L2)
constexpr int kSourceWindows[2] = {400, 24};
const char* kSourceName[2] = {"33 MB source", "2 MB source"};

template <int FILL, int SOURCE>
__global__ void __launch_bounds__(kThreads, 1) fill(int passes, Source src, long long* t,
                                                    int* out) {
  __shared__ __align__(128) uint8_t pool[kRows * kPoolRow];
  const int tid = static_cast<int>(threadIdx.x), q = tid & 3;
  const uint32_t base = lab::smem(pool);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    const int win = (static_cast<int>(blockIdx.x) * 7 + x / 4) % kSourceWindows[SOURCE], lp = x % 4;
    if constexpr (FILL <= 2) {
      const int64_t w0 = static_cast<int64_t>(win) * kWindow;
#pragma unroll 1
      for (int r = tid >> 2; r < kRows; r += kThreads / 4) {
        const int64_t tg = w0 + (((kRows * lp + r) * 37) & (kWindow - 1));
        lab::cp_async16(base + pool_at(r, q), src.run + tg * 64 + 16 * q);
        lab::cp_async16(base + pool_at(r, 4 + q), src.values + tg * 64 + 16 * q);
        if constexpr (FILL == 0) {
          lab::cp_async4(base + pool_at(r, 8) + 4 * q,
                         src.factor + (q * static_cast<int64_t>(kTokens) + tg) * 4);
          if (q == 0) lab::cp_async16(base + pool_at(r, 9), src.dense8 + tg * 16);
        } else if constexpr (FILL == 1) {
          if (q == 0) cp_async8(base + pool_at(r, 8), src.fm + tg * 8);
          if (q == 1) lab::cp_async16(base + pool_at(r, 9), src.dense8 + tg * 16);
        } else {
          if (q == 0) cp_async8(base + pool_at(r, 8), src.fm + tg * 8);
          if (q >= 2)
            lab::cp_async16(base + pool_at(r, 8 + q), src.dense16 + tg * 32 + 16 * (q - 2));
        }
      }
    } else {
      //: the window's plane bases once, a row a 32-bit byte offset from them
      const int64_t w0 = static_cast<int64_t>(win) * kWindow;
      const uint8_t* const run = src.run + w0 * 64 + 16 * q;
      const uint8_t* const values = src.values + w0 * 64 + 16 * q;
      const uint8_t* const fm = src.fm + w0 * 8;
      const uint8_t* const dense = src.dense16 + w0 * 32 + 16 * (q & 1);
#pragma unroll 1
      for (int r = tid >> 2; r < kRows; r += kThreads / 4) {
        const uint32_t t = ((kRows * lp + r) * 37) & (kWindow - 1);
        lab::cp_async16(base + pool_at(r, q), run + t * 64u);
        lab::cp_async16(base + pool_at(r, 4 + q), values + t * 64u);
        if (q == 0) cp_async8(base + pool_at(r, 8), fm + t * 8u);
        if (q >= 2) lab::cp_async16(base + pool_at(r, 8 + q), dense + t * 32u);
      }
    }
    lab::cp_wait_all();
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, pool[tid]);
}

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

//: RESTAGE 0 as built (the dense digits to f16, the factors' products); 1 the digits alone (products precomputed)
constexpr int kRestages = 2;
const char* kRestageName[kRestages] = {"restage as built (digits to f16, factor products)",
                                       "restage, digits alone (products precomputed)"};

template <int RESTAGE>
__global__ void __launch_bounds__(kThreads, 1) restage(int passes, long long* t, int* out) {
  __shared__ __align__(128) uint8_t pool[kRows * kPoolRow];
  __shared__ __align__(16) uint8_t dense[kRows * 32];
  __shared__ float2 fm[kRows];
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kRows * kPoolRow; i += kThreads) pool[i] = static_cast<uint8_t>(i * 7);
  __syncthreads();
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    const int r = tid >> 1, half = tid & 1;
    const uint32_t* const d = reinterpret_cast<const uint32_t*>(pool + pool_at(r, 9)) + 2 * half;
    uint32_t h[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      h[2 * j] =
          hmax2(hfma2(prmt(d[j], 0x64006400u, 0x4140u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
      h[2 * j + 1] =
          hmax2(hfma2(prmt(d[j], 0x64006400u, 0x4342u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
    }
    *reinterpret_cast<uint4*>(dense + r * 32 + 16 * (half ^ ((r >> 2) & 1))) =
        make_uint4(h[0], h[1], h[2], h[3]);
    if constexpr (RESTAGE == 0) {
      if (half == 0) {
        const float* const f = reinterpret_cast<const float*>(pool + pool_at(r, 8));
        const float m = __fmul_rn(__fmul_rn(f[0], f[2]), f[3]);
        fm[r] = make_float2(__fmul_rn(m, f[1]), m);
      }
    }
    __syncthreads();
    pool[(x * 131 + tid) % (kRows * kPoolRow)] ^= dense[(tid * 7 + x) & (kRows * 32 - 1)];
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, dense[tid] ^ lab::bits(fm[tid & (kRows - 1)].x));
}

//: LISTS 0 each warp's entries by a serial loop over every set bit of its words (lane k word k, as built); 1
//: word-parallel: for each word, each lane its bit, its rank by popc, its slot `base + popc(union below)`
constexpr int kListForms = 2;
const char* kListName[kListForms] = {"entries, a serial loop over set bits",
                                     "entries, word-parallel by popc"};
constexpr int kDensityOf[2] = {4, 16};

template <int LISTS>
__global__ void __launch_bounds__(kThreads, 1) lists(int windows, const uint32_t* words,
                                                     long long* t, int* out) {
  __shared__ uint32_t w[kWarps][kWords];
  __shared__ uint32_t uni[kWords];
  __shared__ uint16_t base[kWords];
  __shared__ uint16_t entries[kWarps][kWindow];
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  if (tid < kWords) {
    uni[tid] = ~0u;
    base[tid] = static_cast<uint16_t>(32 * tid);
  }
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < windows; ++x) {
    const int win = (static_cast<int>(blockIdx.x) * 7 + x) % kWindows;
    if (tid < kWarps * kWords)
      (&w[0][0])[tid] = words[static_cast<int64_t>(win) * kWarps * kWords + tid];
    __syncthreads();
    if constexpr (LISTS == 0) {
      const uint32_t mine = lane < kWords ? w[warp][lane] : 0u;
      int at = __popc(mine);
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, at, d);
        if (lane >= d) at += up;
      }
      at -= __popc(mine);
      if (lane < kWords) {
        uint32_t bits = mine;
        const uint32_t u = uni[lane];
        const int b0 = base[lane];
#pragma unroll 1
        for (int i = 0; bits != 0u; ++i) {
          const int b = __ffs(bits) - 1;
          bits &= bits - 1u;
          entries[warp][at + i] = static_cast<uint16_t>(b0 + __popc(u & ((1u << b) - 1u)));
        }
      }
    } else {
      const uint32_t below = (1u << lane) - 1u;
      int at = 0;
#pragma unroll 4
      for (int k = 0; k < kWords; ++k) {
        const uint32_t m = w[warp][k];
        if ((m >> lane) & 1u)
          entries[warp][at + __popc(m & below)] =
              static_cast<uint16_t>(base[k] + __popc(uni[k] & below));
        at += __popc(m);
      }
    }
    __syncthreads();
  }
  clock.stop(t);
  lab::sink(out, entries[warp][lane * 3]);
}

constexpr int kPasses = 400, kListWindows = 400;

template <int FILL, int SOURCE>
void fill_variant(const lab::Bench& b, const Source& src) {
  b.expect(lab::kernel("fill", {FILL, SOURCE}), "LDGSTS", ">=", 3);
  if (b.expect_only) return;
  const lab::Samples s =
      b.variant([&] { fill<FILL, SOURCE><<<b.sms, kThreads>>>(kPasses, src, b.times, b.sinks); },
                [&] { return b.span(kWarps) / kPasses; });
  b.record(std::string(kFillName[FILL]) + " / " + kSourceName[SOURCE], "cycles a pass", s);
}

template <int RESTAGE>
void restage_variant(const lab::Bench& b) {
  b.expect(lab::kernel("restage", {RESTAGE}), "HFMA2", ">=", 4);
  if (b.expect_only) return;
  const lab::Samples s =
      b.variant([&] { restage<RESTAGE><<<b.sms, kThreads>>>(kPasses, b.times, b.sinks); },
                [&] { return b.span(kWarps) / kPasses; });
  b.record(kRestageName[RESTAGE], "cycles a pass (with two barriers)", s);
}

template <int LISTS>
void lists_variant(const lab::Bench& b, const uint32_t* words, int density) {
  b.expect(lab::kernel("lists", {LISTS}), "POPC", ">=", 1);
  if (b.expect_only) return;
  const lab::Samples s =
      b.variant([&] { lists<LISTS><<<b.sms, kThreads>>>(kListWindows, words, b.times, b.sinks); },
                [&] { return b.span(kWarps) / kListWindows; });
  b.record(std::string(kListName[LISTS]) + " / 1 in " + std::to_string(density), "cycles a window",
           s);
}

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

}  // namespace pass_fixed

int main(int argc, char** argv) {
  using namespace pass_fixed;
  lab::Bench b(argc, argv);
  Source src{};
  uint32_t* words[2] = {nullptr, nullptr};
  if (!b.expect_only) {
    uint8_t* buf = nullptr;
    const size_t per = 64 + 64 + 16 + 8 + 16 + 32;
    LAB_CHECK(cudaMalloc(&buf, static_cast<size_t>(kTokens) * per));
    LAB_CHECK(cudaMemset(buf, 0x3c, static_cast<size_t>(kTokens) * per));
    size_t at = 0;
    auto plane = [&](size_t bytes) {
      const uint8_t* p = buf + at;
      at += static_cast<size_t>(kTokens) * bytes;
      return p;
    };
    src.run = plane(64);
    src.values = plane(64);
    src.factor = plane(16);
    src.fm = plane(8);
    src.dense8 = plane(16);
    src.dense16 = plane(32);
    for (int d = 0; d < 2; ++d) {
      std::vector<uint32_t> host(static_cast<size_t>(kWindows) * kWarps * kWords, 0u);
      for (int wi = 0; wi < kWindows; ++wi)
        for (int w = 0; w < kWarps; ++w)
          for (int tk = 0; tk < kWindow; ++tk)
            if (hash3(wi, w, tk) % kDensityOf[d] == 0)
              host[(static_cast<size_t>(wi) * kWarps + w) * kWords + (tk >> 5)] |= 1u << (tk & 31);
      LAB_CHECK(cudaMalloc(&words[d], host.size() * 4));
      LAB_CHECK(cudaMemcpy(words[d], host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    }
  }
  fill_variant<0, 0>(b, src);
  fill_variant<1, 0>(b, src);
  fill_variant<2, 0>(b, src);
  fill_variant<3, 0>(b, src);
  fill_variant<0, 1>(b, src);
  fill_variant<1, 1>(b, src);
  fill_variant<2, 1>(b, src);
  fill_variant<3, 1>(b, src);
  restage_variant<0>(b);
  restage_variant<1>(b);
  lists_variant<0>(b, words[0], 4);
  lists_variant<1>(b, words[0], 4);
  lists_variant<0>(b, words[1], 16);
  lists_variant<1>(b, words[1], 16);
}
