// measure/lab/experiments/sparse_pass.cu -- QUESTION: in a sparse window each warp's fold has a different number of
// steps a pass (0-8: its lists' union in the pass, by 16), so a schedule tuned on dense passes may stall or couple the
// warps; which pass pipeline holds across densities -- the values restaged into a tile (two pool slots, fill two passes
// ahead), or read in place from the swizzled pool (two pool slots, or three so the fill still runs two ahead) -- and
// what do the fold side's entries cost derived in the step from the warp's union words (`base + popc`) against read
// from a list a window ahead? METHOD: each warp's live tokens drawn on the host at a density (all, 1 in 4, 1 in 16),
// the CTA's window union every token (four passes a window), the owner part's fill pattern from a 33 MB source through
// a token permutation, the restage (dense digits to f16, factor products; values and run in the tile forms), the fold
// step at the arm's registers, the fp16 join staggered, the window's lists loaded two passes ahead; the block's span a
// pass over the pass's tensor floor (the busiest scheduler's two warps' steps). see docs/internals/tools/lab.md#sparse-pass
#include "lab.cuh"

#include <algorithm>
#include <vector>

namespace sparse_pass {

constexpr int kLists = 8, kRows = 128, kStep = 16, kThreads = 256, kWarps = 8, kWords = 16;
constexpr int kWindow = 512, kPassesPerWindow = kWindow / kRows, kWindows = 400,
              kTokens = kWindows * kWindow;
constexpr int kArmFloat = 0, kArmHalf = 1;

//: FORM 0 serial, value tile, listed (the owner part); 1 pipelined, value tile, two pool slots, derived; 2 pipelined,
//: values in place, two pool slots, derived; 3 the same with three pool slots; 4 three pool slots, listed; 5 and 6 as
//: 3 and 4 with LOAD-AHEAD: a warp takes up to three steps' operands into registers at a time, the pass's restage share
//: after the loads of the chunk holding its mid-pass step, and signals its slots drained after its last loads, before
//: those steps' MMAs; 7 as 5 with the factor products precomputed (an 8-byte plane, read in place; the restage converts
//: the digits alone); 8 as 7 with the dense digits precomputed as f16 too, read in place by `ldmatrix` -- no restage, no
//: tile, no staged signal. 7 and 8 use 192-byte pool rows, chunk `c ^ ((r >> 1) & 3)`
//: 9 as 8 with the pass's rows built once at the pass's start (word-parallel over its four words into the warp's
//: rows buffer), the steps reading them as listed entries
constexpr int kForms = 10;
const char* kFormName[kForms] = {
    "serial, value tile, listed",
    "pipelined, value tile, derived",
    "pipelined, in place, 2 pool slots, derived",
    "pipelined, in place, 3 pool slots, derived",
    "pipelined, in place, 3 pool slots, listed",
    "load-ahead, in place, 3 pool slots, derived",
    "load-ahead, in place, 3 pool slots, listed",
    "load-ahead, derived, factor products precomputed",
    "load-ahead, derived, products and f16 dense precomputed (no restage)",
    "no restage, rows built a pass at its start"};

template <int FORM>
struct Form {
  static constexpr bool kTile = FORM <= 1, kListed = FORM == 0 || FORM == 4 || FORM == 6;
  static constexpr bool kAhead = FORM >= 5;
  static constexpr int kPools = FORM == 0 ? 1 : FORM >= 3 ? 3 : 2;
  static constexpr bool kFmPlane = FORM >= 7, kDense16 = FORM >= 8, kPassRows = FORM == 9;
  static constexpr int kRow = FORM >= 7 ? 192 : 160;
};

//: DENSITY: a warp's tokens live 1 in DENSITY
constexpr int kDensities = 3;
constexpr int kDensityOf[kDensities] = {1, 4, 16};

//: THE LAYOUTS, swizzled (lab `pool_layout`): a pool row 160 bytes, a row with bit 2 set swapping the 16-byte halves
//: of each 32-byte pair; the dense tile 32 bytes a row, the same swap; the value and run tiles 64 bytes a row, chunk
//: `c ^ ((r >> 1) & 3)`
template <int FORM>
__device__ __forceinline__ int pool_at(int r, int c) {
  if constexpr (Form<FORM>::kRow == 192) return r * 192 + 16 * (c ^ ((r >> 1) & 3));
  return r * 160 + 16 * (c ^ ((r >> 2) & 1));
}

__device__ __forceinline__ int dense_at(int r, int c) { return r * 32 + 16 * (c ^ ((r >> 2) & 1)); }

__device__ __forceinline__ int tile_at(int r, int c) { return r * 64 + 16 * (c ^ ((r >> 1) & 3)); }

//: A WINDOW'S LISTS as the host lays them out: each warp's union words, its live count a pass, its tokens in order
struct GLists {
  uint32_t words[kWarps][kWords];
  uint16_t counts[kWarps][kPassesPerWindow];
  uint16_t entries[kWarps][kWindow];
};

constexpr int kListsHead =
    sizeof(uint32_t) * kWarps * kWords + sizeof(uint16_t) * kWarps * kPassesPerWindow;

template <int FORM>
struct alignas(128) Shared {
  using F = Form<FORM>;
  uint8_t pool[F::kPools][kRows * F::kRow];

  struct alignas(16) Tile {
    uint8_t dense[F::kDense16 ? 16 : kRows * 32];
    float2 fm[F::kFmPlane ? 1 : kRows];
    uint8_t vals[F::kTile ? kRows * 64 : 16];
    uint8_t run[F::kTile ? kRows * 64 : 16];
  } tile[2];

  struct Lists {
    uint32_t words[kWarps][kWords];
    uint16_t counts[kWarps][kPassesPerWindow];
    uint16_t entries[kWarps][F::kListed ? kWindow : 8];
  } lists[2];

  uint16_t scratch[kWarps][32];
  uint16_t pass_rows[kWarps][F::kPassRows ? kRows : 1];
  uint32_t uni[kWords];
  uint16_t base[kWords];
  uint64_t landed[3], staged[2], drained[2], lists_landed[2];
};

struct Source {
  const uint8_t *run, *values, *factor, *dense, *fm, *dense16;
  const GLists* lists;
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

__device__ __forceinline__ uint32_t parity(int x) { return static_cast<uint32_t>(x >> 1) & 1u; }

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ int window_of(int x) {
  return (static_cast<int>(blockIdx.x) * 7 + x / kPassesPerWindow) % kWindows;
}

//: PASS `x`'s rows into a pool slot, a quad a token; a token's global row through a permutation of its window
template <int FORM>
__device__ __forceinline__ void fill(uint8_t* pool, const Source& src, int x, int tid) {
  const int q = tid & 3, lp = x % kPassesPerWindow;
  const int64_t win = static_cast<int64_t>(window_of(x)) * kWindow;
  const uint32_t base = lab::smem(pool);
#pragma unroll 1
  for (int r = tid >> 2; r < kRows; r += kThreads / 4) {
    const int64_t tg = win + (((kRows * lp + r) * 37) & (kWindow - 1));
    lab::cp_async16(base + pool_at<FORM>(r, q), src.run + tg * 64 + 16 * q);
    lab::cp_async16(base + pool_at<FORM>(r, 4 + q), src.values + tg * 64 + 16 * q);
    if constexpr (Form<FORM>::kFmPlane) {
      if (q == 0) cp_async8(base + pool_at<FORM>(r, 10), src.fm + tg * 8);
      if constexpr (Form<FORM>::kDense16) {
        if (q >= 2)
          lab::cp_async16(base + pool_at<FORM>(r, 8 + (q - 2)),
                          src.dense16 + tg * 32 + 16 * (q - 2));
      } else {
        if (q == 1) lab::cp_async16(base + pool_at<FORM>(r, 8), src.dense + tg * 16);
      }
    } else {
      lab::cp_async4(base + pool_at<FORM>(r, 8) + 4 * q,
                     src.factor + (q * static_cast<int64_t>(kTokens) + tg) * 4);
      if (q == 0) lab::cp_async16(base + pool_at<FORM>(r, 9), src.dense + tg * 16);
    }
  }
}

//: WINDOW `wi`'s lists into a buffer (the listed forms take the entries too)
template <int FORM>
__device__ __forceinline__ void load_lists(typename Shared<FORM>::Lists& l, const Source& src,
                                           int x, int tid) {
  constexpr int kBytes = Form<FORM>::kListed ? static_cast<int>(sizeof(GLists)) : kListsHead;
  const uint8_t* g = reinterpret_cast<const uint8_t*>(src.lists + window_of(x));
  const uint32_t s = lab::smem(&l);
#pragma unroll 1
  for (int i = 16 * tid; i < kBytes; i += 16 * kThreads) lab::cp_async16(s + i, g + i);
}

//: ITEM `i` OF A RESTAGE (row `i >> 1`): digits 0-7 to f16 and the factors' products (even), digits 8-15 (odd); the
//: tile forms copy the values and the run too (odd)
template <int FORM>
__device__ __forceinline__ void restage_item(typename Shared<FORM>::Tile& t, const uint8_t* pool,
                                             int i) {
  const int r = i >> 1, half = i & 1;
  const uint32_t* const d =
      reinterpret_cast<const uint32_t*>(pool + pool_at<FORM>(r, Form<FORM>::kFmPlane ? 8 : 9))
      + 2 * half;
  uint32_t h[4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    h[2 * j] =
        hmax2(hfma2(prmt(d[j], 0x64006400u, 0x4140u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
    h[2 * j + 1] =
        hmax2(hfma2(prmt(d[j], 0x64006400u, 0x4342u), 0x3c003c00u, 0xe400e400u), 0x07f807f8u);
  }
  *reinterpret_cast<uint4*>(t.dense + dense_at(r, half)) = make_uint4(h[0], h[1], h[2], h[3]);
  if (Form<FORM>::kFmPlane) {
  } else if (half == 0) {
    const float* const f = reinterpret_cast<const float*>(pool + pool_at<FORM>(r, 8));
    const float m = __fmul_rn(__fmul_rn(f[0], f[2]), f[3]);
    t.fm[r] = make_float2(__fmul_rn(m, f[1]), m);
  } else if constexpr (Form<FORM>::kTile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      *reinterpret_cast<uint4*>(t.vals + tile_at(r, j)) =
          *reinterpret_cast<const uint4*>(pool + pool_at<FORM>(r, 4 + j));
      *reinterpret_cast<uint4*>(t.run + tile_at(r, j)) =
          *reinterpret_cast<const uint4*>(pool + pool_at<FORM>(r, j));
    }
  }
}

//: A STEP'S ROWS: the lane's A and B `ldmatrix` rows and its four c tokens with their membership
struct Rows {
  int a, b, c[4];
  bool in[4];
};

//: THE LISTED FORM'S ROWS: entries `[e0, e0 + 16)` of the warp's window list, past `end` row 0 and not in
__device__ __forceinline__ Rows listed_rows(const uint16_t* entries, int e0, int end, int first,
                                            int lane) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  Rows w;
  auto row = [&](int e, bool& in) {
    in = e < end;
    return in ? static_cast<int>(entries[e]) - first : 0;
  };
  bool unused;
  w.a = row(e0 + 8 * (q >> 1) + r, unused);
  w.b = row(e0 + 8 * (q & 1) + r, unused);
#pragma unroll
  for (int k = 0; k < 4; ++k) w.c[k] = row(e0 + 2 * tq + (k & 1) + 8 * (k >> 1), w.in[k]);
  return w;
}

//: A PASS'S ROWS BUILT AT ITS START: for each of the pass's four words, each lane its bit, ranked by `popc`, its slot
//: `base + popc(union below)` less the pass's first into the warp's buffer; the count
__device__ __forceinline__ int pass_rows(const uint32_t* words, const uint32_t* uni,
                                         const uint16_t* base, uint16_t* out, int lp, int first,
                                         int lane) {
  const uint32_t below = (1u << lane) - 1u;
  int at = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int word = 4 * lp + k;
    const uint32_t m = words[word];
    if ((m >> lane) & 1u)
      out[at + __popc(m & below)] =
          static_cast<uint16_t>(base[word] + __popc(uni[word] & below) - first);
    at += __popc(m);
  }
  __syncwarp();
  return at;
}

//: THE DERIVED FORM'S ROWS (option 2): the warp union's next 16 set bits from token `cursor`, before `end`, gathered a
//: word at a time -- each lane its bit, ranked by `popc`, storing its token's slot `base + popc(union below)` -- into
//: the warp's scratch, the 17th's token as the next cursor; a row is a slot less the pass's first.
__device__ __forceinline__ Rows derived_rows(const uint32_t* words, const uint32_t* uni,
                                             const uint16_t* base, uint16_t* scratch, int& cursor,
                                             int end, int first, int lane) {
  __syncwarp();
  int got = 0, k = cursor >> 5;
  const int kend = end >> 5;
  const uint32_t below = (1u << lane) - 1u;
  uint32_t m = k < kend ? words[k] & (~0u << (cursor & 31)) : 0u;
#pragma unroll 1
  while (k < kend) {
    const int rank = got + __popc(m & below);
    if (((m >> lane) & 1u) && rank <= 16)
      scratch[rank] =
          static_cast<uint16_t>(rank == 16 ? 32 * k + lane : base[k] + __popc(uni[k] & below));
    got += __popc(m);
    if (got > 16) break;
    ++k;
    m = k < kend ? words[k] : 0u;
  }
  __syncwarp();
  const int count = min(got, 16);
  cursor = got > 16 ? static_cast<int>(scratch[16]) : end;
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  Rows w;
  auto row = [&](int e, bool& in) {
    in = e < count;
    return in ? static_cast<int>(scratch[e]) - first : 0;
  };
  bool unused;
  w.a = row(8 * (q >> 1) + r, unused);
  w.b = row(8 * (q & 1) + r, unused);
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) w.c[k2] = row(2 * tq + (k2 & 1) + 8 * (k2 >> 1), w.in[k2]);
  return w;
}

struct Operands {
  uint32_t a[4], b[4][2], cv[2], cm[2];
};

template <int FORM, class Lap>
__device__ __forceinline__ void operands(Operands& o, const typename Shared<FORM>::Tile& t,
                                         const uint8_t* pool, const Rows& w, int lane, float pu,
                                         float pv, Lap&& lap) {
  const int q = lane >> 3, byte = lab::warp_id() * kLists + (lane >> 2);
  lab::ldsm_t(o.a, Form<FORM>::kDense16 ? lab::smem(pool + pool_at<FORM>(w.a, 8 + (q & 1)))
                                        : lab::smem(t.dense + dense_at(w.a, q & 1)));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t x[4];
    const int c = 2 * h + (q >> 1);
    lab::ldsm_t(x, Form<FORM>::kTile ? lab::smem(t.vals + tile_at(w.b, c))
                                     : lab::smem(pool + pool_at<FORM>(w.b, 4 + c)));
    o.b[2 * h][0] = x[0];
    o.b[2 * h][1] = x[1];
    o.b[2 * h + 1][0] = x[2];
    o.b[2 * h + 1][1] = x[3];
  }
  lap(2);
  float v[4], m[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int row = w.c[k];
    const uint8_t b8 = Form<FORM>::kTile ? t.run[tile_at(row, byte >> 4) + (byte & 15)]
                                         : pool[pool_at<FORM>(row, byte >> 4) + (byte & 15)];
    const float b = fmaxf(static_cast<float>(b8), 2.0f);
    const float2 f = Form<FORM>::kFmPlane
                         ? *reinterpret_cast<const float2*>(pool + pool_at<FORM>(row, 10))
                         : t.fm[row];
    v[k] = w.in[k] ? __fmul_rn(__fmul_rn(b, f.x), pu) : 0.0f;
    m[k] = w.in[k] ? __fmul_rn(__fmul_rn(b, f.y), pv) : 0.0f;
  }
  o.cv[0] = lab::pack_half2(v[0], v[1]);
  o.cv[1] = lab::pack_half2(v[2], v[3]);
  o.cm[0] = lab::pack_half2(m[0], m[1]);
  o.cm[1] = lab::pack_half2(m[2], m[3]);
  lap(3);
}

template <int ARM>
struct Acc {
  float acc[kLists][4][4];
  uint32_t part[ARM == kArmHalf ? kLists : 1][4][2];
};

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

//: THE PHASE LEDGER (LEDGER 1, the load-ahead forms): a warp's cycles a pass in its start-of-pass waits, its rows and
//: loads, its mid-pass work (the slot waits, the fill, the restage), its MMAs' issue, and its pass end
constexpr int kPhases = 7;
const char* kPhaseName[kPhases] = {"start waits",
                                   "rows",
                                   "operand ldmatrix",
                                   "coefficient (loads, math)",
                                   "mid-pass (slot waits, fill, restage)",
                                   "MMA issue",
                                   "pass end (join, sums)"};

template <int ARM, int FORM, int LEDGER>
__global__ void __launch_bounds__(kThreads, 1) pass(int passes, Source src, long long* t, int* out,
                                                    long long* led) {
  using F = Form<FORM>;
  extern __shared__ __align__(128) unsigned char raw[];
  Shared<FORM>& sh = *reinterpret_cast<Shared<FORM>*>(raw);
  const int lane = lab::lane_id(), warp = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  if (tid < 3) lab::mbar_init(lab::smem(&sh.landed[tid]), kThreads);
  if (tid < 2) {
    lab::mbar_init(lab::smem(&sh.staged[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.drained[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.lists_landed[tid]), kThreads);
  }
  //: the CTA's window union: every token, so a token's slot is itself
  if (tid < kWords) {
    sh.uni[tid] = ~0u;
    sh.base[tid] = static_cast<uint16_t>(32 * tid);
  }
  __syncthreads();
  const float pu = 0.5f + 1e-3f * lane, pv = 0.25f + 1e-3f * lane;
  Acc<ARM> st = {};
  float m[4] = {}, mass = 0.0f;

  //: the prologue: window 0's lists, the first passes' fills, pass 0's restage
  load_lists<FORM>(sh.lists[0], src, 0, tid);
  lab::mbar_arrive_when_landed(lab::smem(&sh.lists_landed[0]));
  if constexpr (FORM != 0) {
    fill<FORM>(sh.pool[0], src, 0, tid);
    lab::mbar_arrive_when_landed(lab::smem(&sh.landed[0]));
    if constexpr (FORM != 2) {
      fill<FORM>(sh.pool[1], src, 1, tid);
      lab::mbar_arrive_when_landed(lab::smem(&sh.landed[1]));
    }
    if constexpr (!F::kDense16) {
      lab::mbar_wait(lab::smem(&sh.landed[0]), 0);
      restage_item<FORM>(sh.tile[0], sh.pool[0], tid);
      lab::mbar_arrive(lab::smem(&sh.staged[0]));
    }
  }
  lab::WarpSpan clock;
  clock.start();
  long long ledger[kPhases] = {}, last = clock64();
  auto lap = [&](int ph) {
    if constexpr (LEDGER != 0) {
      const long long now = clock64();
      ledger[ph] += now - last;
      last = now;
    }
  };
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    const int lp = x % kPassesPerWindow, wi = x / kPassesPerWindow, lb = wi & 1, first = kRows * lp;
    const int k = x & 1, kn = (x + 1) & 1;
    const int slot = F::kPools == 3 ? x % 3 : F::kPools == 2 ? k : 0;
    if (lp == 0) lab::mbar_wait(lab::smem(&sh.lists_landed[lb]), parity(wi));
    if (lp == 2 && FORM != 0) {
      load_lists<FORM>(sh.lists[lb ^ 1], src, x + kPassesPerWindow, tid);
      lab::mbar_arrive_when_landed(lab::smem(&sh.lists_landed[lb ^ 1]));
    }
    if constexpr (FORM == 0) {
      fill<FORM>(sh.pool[0], src, x, tid);
      lab::cp_wait_all();
      __syncthreads();
      restage_item<FORM>(sh.tile[0], sh.pool[0], tid);
      __syncthreads();
    } else {
      if constexpr (F::kDense16)
        lab::mbar_wait(lab::smem(&sh.landed[x % 3]), static_cast<uint32_t>(x / 3) & 1u);
      else
        lab::mbar_wait(lab::smem(&sh.staged[k]), parity(x));
      if constexpr (FORM == 1) {
        fill<FORM>(sh.pool[k], src, x + 2, tid);
        lab::mbar_arrive_when_landed(lab::smem(&sh.landed[k]));
      }
    }
    lap(0);
    const typename Shared<FORM>::Lists& L = sh.lists[lb];
    //: the warp's steps this pass and its cursor
    int count;
    if constexpr (F::kListed) {
      count = L.counts[warp][lp];
    } else if constexpr (F::kPassRows) {
      count = pass_rows(L.words[warp], sh.uni, sh.base, sh.pass_rows[warp], lp, first, lane);
    } else {
      const int c = lane < 4 ? __popc(L.words[warp][4 * lp + lane]) : 0;
      count = __reduce_add_sync(0xffffffffu, c);
    }
    int e0 = 0;
    if constexpr (F::kListed) {
      for (int j = 0; j < lp; ++j) e0 += L.counts[warp][j];
    }
    int cursor = first;
    const int n = (count + kStep - 1) / kStep, mid = n == 0 ? -1 : min(3, n - 1);
    auto middle = [&]() {
      if constexpr (FORM == 1) {
        lab::mbar_wait(lab::smem(&sh.landed[kn]), parity(x + 1));
        if (x >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(x - 1));
        restage_item<FORM>(sh.tile[kn], sh.pool[kn], tid);
        lab::mbar_arrive(lab::smem(&sh.staged[kn]));
      }
      if constexpr (FORM == 2) {
        if (x >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(x - 1));
        fill<FORM>(sh.pool[kn], src, x + 1, tid);
        lab::mbar_arrive_when_landed(lab::smem(&sh.landed[kn]));
      }
      if constexpr (F::kDense16) {
        if (x >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(x - 1));
        fill<FORM>(sh.pool[(x + 2) % 3], src, x + 2, tid);
        lab::mbar_arrive_when_landed(lab::smem(&sh.landed[(x + 2) % 3]));
      } else if constexpr (FORM >= 3) {
        if (x >= 1) lab::mbar_wait(lab::smem(&sh.drained[kn]), parity(x - 1));
        fill<FORM>(sh.pool[(x + 2) % 3], src, x + 2, tid);
        lab::mbar_arrive_when_landed(lab::smem(&sh.landed[(x + 2) % 3]));
        lab::mbar_wait(lab::smem(&sh.landed[(x + 1) % 3]), static_cast<uint32_t>((x + 1) / 3) & 1u);
        restage_item<FORM>(sh.tile[kn], sh.pool[(x + 1) % 3], tid);
        lab::mbar_arrive(lab::smem(&sh.staged[kn]));
      }
    };
    const typename Shared<FORM>::Tile& T = sh.tile[FORM == 0 ? 0 : k];
    const uint8_t* const P = sh.pool[slot];
    auto load = [&](Operands& o, int s) {
      Rows w;
      if constexpr (F::kListed)
        w = listed_rows(L.entries[warp], e0 + kStep * s, e0 + count, first, lane);
      else if constexpr (F::kPassRows)
        w = listed_rows(sh.pass_rows[warp], kStep * s, count, 0, lane);
      else
        w = derived_rows(L.words[warp], sh.uni, sh.base, sh.scratch[warp], cursor, first + kRows,
                         first, lane);
      lap(1);
      operands<FORM>(o, T, P, w, lane, pu, pv, lap);
    };
    if constexpr (F::kAhead) {
      //: chunks of three steps: the chunk's operands, the slots drained after the pass's last loads, the restage share
      //: after the loads of the chunk holding the mid-pass step, then the chunk's MMAs
#pragma unroll 1
      for (int s0 = 0; s0 == 0 || s0 < n; s0 += 3) {
        Operands o0, o1, o2;
        if (s0 < n) load(o0, s0);
        if (s0 + 1 < n) load(o1, s0 + 1);
        if (s0 + 2 < n) load(o2, s0 + 2);
        if (s0 + 3 >= n) lab::mbar_arrive(lab::smem(&sh.drained[k]));
        if (s0 <= max(mid, 0) && max(mid, 0) < s0 + 3) middle();
        lap(4);
        if (s0 < n) step<ARM>(st, m, o0, lane);
        if (s0 + 1 < n) step<ARM>(st, m, o1, lane);
        if (s0 + 2 < n) step<ARM>(st, m, o2, lane);
        lap(5);
      }
    } else {
      if (FORM != 0 && mid < 0) middle();
#pragma unroll 1
      for (int s = 0; s < n; ++s) {
        Operands o;
        load(o, s);
        step<ARM>(st, m, o, lane);
        if (FORM != 0 && s == mid) middle();
      }
    }
    if constexpr (FORM == 0) {
      __syncthreads();
    } else {
      if constexpr (FORM == 2) {
        lab::mbar_wait(lab::smem(&sh.landed[kn]), parity(x + 1));
        restage_item<FORM>(sh.tile[kn], sh.pool[kn], tid);
        lab::mbar_arrive(lab::smem(&sh.staged[kn]));
      }
      if constexpr (!F::kAhead) lab::mbar_arrive(lab::smem(&sh.drained[k]));
    }
    if constexpr (FORM == 0) {
      if (lp == kPassesPerWindow - 1) {
        load_lists<FORM>(sh.lists[lb ^ 1], src, x + 1, tid);
        lab::mbar_arrive_when_landed(lab::smem(&sh.lists_landed[lb ^ 1]));
      }
    }
    if ((x + (warp >> 2)) & 1) join(st);
    mass += m[0] + m[1] + m[2] + m[3];
    m[0] = m[1] = m[2] = m[3] = 0.0f;
    lap(6);
  }
  clock.stop(t);
  if constexpr (LEDGER != 0) {
    if (lane == 0)
      for (int ph = 0; ph < kPhases; ++ph)
        led[(blockIdx.x * kWarps + warp) * kPhases + ph] = ledger[ph];
  }
  lab::cp_wait_all();
  uint32_t fold = lab::bits(mass);
  for (int j = 0; j < kLists; ++j)
    for (int n2 = 0; n2 < 4; ++n2)
      for (int i = 0; i < 4; ++i) fold ^= lab::bits(st.acc[j][n2][i]);
  lab::sink(out, fold);
}

constexpr int kPasses = 200;
const char* kArmName[2] = {"fp32 arm", "fp16 arm"};
//: a warp-step's tensor cycles alone: 33 atoms at 32 (fp32 arm) or 16 (fp16 arm; the fresh fp32 one counts 16 in the mix)
constexpr double kWarpStep[2] = {33 * 32, 33 * 16};

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

//: each window's lists at a density, and the mean pass floor (the busiest scheduler's two warps' steps) over a launch
std::vector<GLists> make_lists(int density, int sms, double& steps_floor) {
  std::vector<GLists> out(kWindows);
  for (int wi = 0; wi < kWindows; ++wi) {
    GLists& g = out[wi];
    for (int w = 0; w < kWarps; ++w) {
      int n = 0;
      for (int p = 0; p < kPassesPerWindow; ++p) g.counts[w][p] = 0;
      for (int k = 0; k < kWords; ++k) g.words[w][k] = 0;
      for (int t = 0; t < kWindow; ++t) {
        if (hash3(wi, w, t) % density != 0) continue;
        g.words[w][t >> 5] |= 1u << (t & 31);
        ++g.counts[w][t / kRows];
        g.entries[w][n++] = static_cast<uint16_t>(t);
      }
      for (; n < kWindow; ++n) g.entries[w][n] = 0;
    }
  }
  double total = 0;
  for (int b = 0; b < sms; ++b)
    for (int x = 0; x < kPasses; ++x) {
      const GLists& g = out[(b * 7 + x / kPassesPerWindow) % kWindows];
      int busiest = 0;
      for (int s = 0; s < 4; ++s) {
        const int lp = x % kPassesPerWindow;
        busiest = std::max(busiest, (g.counts[s][lp] + 15) / 16 + (g.counts[s + 4][lp] + 15) / 16);
      }
      total += busiest;
    }
  steps_floor = total / (static_cast<double>(sms) * kPasses);
  return out;
}

template <int ARM, int DENSITY, int FORM>
void variant(const lab::Bench& b, Source src, const GLists* lists, double steps_floor) {
  const std::string name = lab::kernel("pass", {ARM, FORM, 0});
  b.expect(name, ARM == kArmHalf ? "HMMA.16816.F16" : "HMMA.16816.F32", ">=", 32);
  b.expect(name, "LDGSTS", ">=", 4);
  if (FORM == 0) b.expect(name, "BAR", ">=", 3);
  if (ARM == 0 && (FORM == 6 || FORM == 9)) b.expect(name, "BSSY", ">=", 1);
  if (b.expect_only) return;
  src.lists = lists;
  const int bytes = static_cast<int>(sizeof(Shared<FORM>));
  LAB_CHECK(
      cudaFuncSetAttribute(pass<ARM, FORM, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] {
        pass<ARM, FORM, 0><<<b.sms, kThreads, bytes>>>(kPasses, src, b.times, b.sinks, nullptr);
      },
      [&] { return b.span(8) / kPasses; });
  const std::string label = std::string(kArmName[ARM]) + " / 1 in "
                            + std::to_string(kDensityOf[DENSITY]) + " / " + kFormName[FORM];
  b.record(label, "cycles a pass", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / (steps_floor * kWarpStep[ARM]));
  b.record(label + " / floor", "x", over);
}

//: a load-ahead form's phase ledger: each phase's mean cycles a warp a pass
template <int ARM, int DENSITY, int FORM>
void ledger(const lab::Bench& b, Source src, const GLists* lists) {
  const std::string name = lab::kernel("pass", {ARM, FORM, 1});
  b.expect(name, "HMMA.16816.F32", ">=", 32);
  if (ARM == 0 && (FORM == 6 || FORM == 9)) b.expect(name, "BSSY", ">=", 1);
  if (b.expect_only) return;
  src.lists = lists;
  const int bytes = static_cast<int>(sizeof(Shared<FORM>));
  long long* led = nullptr;
  const size_t n = static_cast<size_t>(b.sms) * kWarps * kPhases;
  LAB_CHECK(cudaMalloc(&led, n * sizeof(long long)));
  LAB_CHECK(
      cudaFuncSetAttribute(pass<ARM, FORM, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  std::vector<lab::Samples> per(kPhases);
  b.variant(
      [&] { pass<ARM, FORM, 1><<<b.sms, kThreads, bytes>>>(kPasses, src, b.times, b.sinks, led); },
      [&] {
        std::vector<long long> h(n);
        LAB_CHECK(cudaMemcpy(h.data(), led, n * sizeof(long long), cudaMemcpyDeviceToHost));
        for (int ph = 0; ph < kPhases; ++ph) {
          double sum = 0;
          for (size_t i = ph; i < n; i += kPhases) sum += double(h[i]);
          per[ph].v.push_back(sum / (double(b.sms) * kWarps * kPasses));
        }
        return 0.0;
      });
  for (int ph = 0; ph < kPhases; ++ph)
    b.record(std::string("ledger / ") + kArmName[ARM] + " / 1 in "
                 + std::to_string(kDensityOf[DENSITY]) + " / " + kFormName[FORM] + " / "
                 + kPhaseName[ph],
             "cycles a warp a pass", per[ph]);
  LAB_CHECK(cudaFree(led));
}

template <int ARM, int DENSITY>
void density(const lab::Bench& b, const Source& src) {
  double steps_floor = 0;
  GLists* lists = nullptr;
  if (!b.expect_only) {
    const std::vector<GLists> host = make_lists(kDensityOf[DENSITY], b.sms, steps_floor);
    LAB_CHECK(cudaMalloc(&lists, sizeof(GLists) * kWindows));
    LAB_CHECK(cudaMemcpy(lists, host.data(), sizeof(GLists) * kWindows, cudaMemcpyHostToDevice));
    b.record(std::string("steps a pass, busiest scheduler / 1 in ")
                 + std::to_string(kDensityOf[DENSITY]),
             "steps", {{steps_floor}});
  }
  variant<ARM, DENSITY, 0>(b, src, lists, steps_floor);
  variant<ARM, DENSITY, 1>(b, src, lists, steps_floor);
  variant<ARM, DENSITY, 2>(b, src, lists, steps_floor);
  variant<ARM, DENSITY, 3>(b, src, lists, steps_floor);
  variant<ARM, DENSITY, 4>(b, src, lists, steps_floor);
  //: the load-ahead forms need the registers only the fp32 arm has
  if constexpr (ARM == kArmFloat) {
    variant<ARM, DENSITY, 5>(b, src, lists, steps_floor);
    variant<ARM, DENSITY, 6>(b, src, lists, steps_floor);
    variant<ARM, DENSITY, 7>(b, src, lists, steps_floor);
    variant<ARM, DENSITY, 8>(b, src, lists, steps_floor);
    variant<ARM, DENSITY, 9>(b, src, lists, steps_floor);
    ledger<ARM, DENSITY, 5>(b, src, lists);
    ledger<ARM, DENSITY, 9>(b, src, lists);
    ledger<ARM, DENSITY, 6>(b, src, lists);
    ledger<ARM, DENSITY, 8>(b, src, lists);
  }
  if (lists) LAB_CHECK(cudaFree(lists));
}

template <int ARM>
void arm(const lab::Bench& b, const Source& src) {
  density<ARM, 0>(b, src);
  density<ARM, 1>(b, src);
  density<ARM, 2>(b, src);
}

}  // namespace sparse_pass

int main(int argc, char** argv) {
  using namespace sparse_pass;
  lab::Bench b(argc, argv);
  Source src{};
  if (!b.expect_only) {
    uint8_t* buf = nullptr;
    const size_t bytes = static_cast<size_t>(kTokens) * 200;
    LAB_CHECK(cudaMalloc(&buf, bytes));
    LAB_CHECK(cudaMemset(buf, 0x3c, bytes));
    src.run = buf;
    src.values = buf + static_cast<size_t>(kTokens) * 64;
    src.factor = buf + static_cast<size_t>(kTokens) * 128;
    src.dense = buf + static_cast<size_t>(kTokens) * 144;
    src.fm = buf + static_cast<size_t>(kTokens) * 160;
    src.dense16 = buf + static_cast<size_t>(kTokens) * 168;
  }
  arm<kArmHalf>(b, src);
  arm<kArmFloat>(b, src);
}
