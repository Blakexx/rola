// csrc/rola/src/carry/ampere/fold.cuh -- THE FOLD: a warp's write lists over a window's chunks into the fp32 state -- each
// list in its running unit, steps of 16 tokens and an 8-token tail, a step's A the slots' dense slice times the
// entries' c, its B their values, fp32-accumulate MMAs. see docs/internals/carry/ampere/fold.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "carry/ampere/parts.cuh"

namespace rola::carry {

//: A DENSE WRITE LEVEL'S BYTE SCALE: its f16 digits carry 1/255 of the byte scale c is formed on.
constexpr float kByteScale = 1.0f / 255.0f;

//: A WARP'S LIST `j`: the box's list `warp + 8 j`, as the image's stores lay the lists out.
__device__ __forceinline__ constexpr int fold_list(int warp, int j) { return warp + kWarps * j; }

//: 2^e as an f16 pair: normal from 2^-14, subnormal to 2^-24, zero below.
__device__ __forceinline__ uint32_t pow2_half2(int e) {
  const uint32_t h = e >= -14   ? static_cast<uint32_t>(e + 15) << 10
                     : e >= -24 ? 1u << (e + 24)
                                : 0u;
  return h | (h << 16);
}

//: A UNIT'S NEXT WINDOW: a live list's exponent, held within the bound's headroom; the state's rescale returned and
//: the bound moved (`next_units` in tests/oracle/fold_reference.py). -- see docs/internals/carry/ampere/fold.md#units
template <class P>
__device__ __forceinline__ float next_unit(int k, UnitBook& book) {
  constexpr float kWindowAdd = P::kWindow * 16.0f;
  const int held = book.unit + ceil_exp(book.bound) - kUnitHeadroom;
  const int next = max(k <= kExpDead ? book.unit : max(k, held), kUnitMin);
  const int d = book.unit - next;
  const float f = d < -126 ? 0.0f : pow2(d);
  book.bound = __fmaf_rn(book.bound, f, kWindowAdd);
  book.unit = next;
  return f;
}

//: A WRITE SIDE WITH NO LIST LEVEL'S WINDOW EXPONENT, every list's, from the window's shift of F (`gains_shift`: its
//: largest F in [2^-4, 2^-3) after it): F under 2^(-3 - shift) and a dense byte under 2^8 -- a bound, the dense level's
//: floor holding each list's largest within 2^13 of it. -- see docs/internals/carry/ampere/fold.md#units
__device__ __forceinline__ int dense_exp(int shift) { return 5 - shift; }

//: A WINDOW'S UNITS, before the warp's steps: each list's exponent moves its book (lane group `j` list `j`); its state's
//: rescale `f` (applied beside the list's first MMAs) and its c factor `2^(E - unit)` posted to the list's scale (the
//: `kScale` tenant, read by the window's fold chunks). A side with no list level keeps its unit.
//: -- see docs/internals/carry/ampere/fold.md#units
struct ListScale {
  float f;
  uint32_t cf;
};

template <class P>
__device__ __forceinline__ void fold_units(uint8_t* smem, const Window& ww, int w, int warp,
                                           int lane) {
  constexpr int kLanes = 32 / P::kFoldLists;
  const int j = lane / kLanes;
  const int16_t* exps =
      reinterpret_cast<const int16_t*>(smem + P::kWriteExp) + parity_of(w) * kMaxExtent * 2;
  const int k = ww.listed ? exps[2 * fold_list(warp, j) + 1] : dense_exp(exps[1]);
  UnitBook* const book = reinterpret_cast<UnitBook*>(smem + P::kBook) + fold_list(warp, j);
  UnitBook b = *book;
  const float fl = next_unit<P>(k, b);
  const int cexp = max(k, kUnitMin) - b.unit;
  __syncwarp();
  if (lane % kLanes == 0) {
    *book = b;
    reinterpret_cast<ListScale*>(smem + P::kScale)[fold_list(warp, j)] =
        ListScale{fl, pow2_half2(cexp)};
  }
  __syncwarp();
}

//: AN ENTRY'S LANE VALUES: its slot's row offset in the union (its swizzle folded in; the chunk's pool is taken back
//: by its first slot `first`) and its c paired with the next lane's, moved to the list's unit; where the list carries
//: the den (`Den`), its c_m (the record's high half) paired the same way -- one shuffle of the record's word for both.
//: A pad (past the run) is the chunk's first row with c = 0. -- see docs/internals/carry/ampere/fold.md#den
template <class P, bool Den>
__device__ __forceinline__ void entry_of(uint2 rec, bool live, int first, uint32_t cf,
                                         uint32_t& off, uint32_t& cpair, uint32_t& dpair) {
  const uint32_t s = live ? rec.x & 0xffffu : static_cast<uint32_t>(first);
  off = s * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(s, 0);
  const uint32_t mine = live ? rec.y : 0u, next = __shfl_xor_sync(0xffffffffu, mine, 1);
  cpair = ops::hmul2(__byte_perm(mine, next, 0x5410), cf);
  if constexpr (Den) dpair = __byte_perm(mine, next, 0x7632);
}

//: A LIST'S DEN COLUMN over its steps: two m-tiles, the column in every one of B's (each lane its quad's rows).
//: -- see docs/internals/carry/ampere/fold.md#den
template <class P>
using DenColumn = float[P::kFoldMTiles][4];

//: A 16-TOKEN STEP into list `J`'s accumulators: its slots' dense slices (A, `ldmatrix.trans`, times c) and values
//: (B), eight fp32-accumulate MMAs; where the list carries the den (`Den`), its column's two more on the raw slices
//: (B the entries' c_m, every column). Lanes `0..15` hold its entries. -- see docs/internals/carry/ampere/fold.md#step
//: @burst
template <class P, int J, bool Den>
__device__ __forceinline__ void step16(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                       uint32_t cpair, uint32_t dpair, DenColumn<P>& den,
                                       int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t oa = __shfl_sync(0xffffffffu, off, 8 * (q >> 1) + r);
  const uint32_t ob = __shfl_sync(0xffffffffu, off, 8 * (q & 1) + r);
  const uint32_t c01 = __shfl_sync(0xffffffffu, cpair, 2 * t),
                 c89 = __shfl_sync(0xffffffffu, cpair, 2 * t + 8);
  uint32_t raw[P::kFoldMTiles][4], a[P::kFoldMTiles][4], b[P::kFoldNTiles][2];
  static_for<P::kFoldMTiles>([&](auto M) {
    ops::load_frag_t(raw[M], chunk_addr<P::kWriteRow>(pool + oa, 2 * M + (q & 1)));
    a[M][0] = ops::hmul2(raw[M][0], c01);
    a[M][1] = ops::hmul2(raw[M][1], c01);
    a[M][2] = ops::hmul2(raw[M][2], c89);
    a[M][3] = ops::hmul2(raw[M][3], c89);
  });
  static_for<P::kFoldNTiles / 2>([&](auto H) {
    uint32_t x[4];
    ops::load_frag_t(x, chunk_addr<P::kWriteRow>(pool + ob, P::kValuesAt / 16 + 2 * H + (q >> 1)));
    b[2 * H][0] = x[0];
    b[2 * H][1] = x[1];
    b[2 * H + 1][0] = x[2];
    b[2 * H + 1][1] = x[3];
  });
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<P::kFoldNTiles>([&](auto N) { ops::hmma(st.acc[J][M][N], a[M], b[N][0], b[N][1]); });
  });
  if constexpr (Den) {
    const uint32_t d01 = __shfl_sync(0xffffffffu, dpair, 2 * t),
                   d89 = __shfl_sync(0xffffffffu, dpair, 2 * t + 8);
    static_for<P::kFoldMTiles>([&](auto M) { ops::hmma(den[M], raw[M], d01, d89); });
  }
}

//: AN 8-TOKEN TAIL into list `J`'s accumulators: m16n8k8 MMAs, half a step's tensor work; the den's column on the raw
//: slices where the list carries it. Lanes `0..7` hold its entries. -- see docs/internals/carry/ampere/fold.md#step
//: @burst
template <class P, int J, bool Den>
__device__ __forceinline__ void step8(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                      uint32_t cpair, uint32_t dpair, DenColumn<P>& den, int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t o = __shfl_sync(0xffffffffu, off, r), c01 = __shfl_sync(0xffffffffu, cpair, 2 * t);
  //: the slice's rows a matrix each (`ldmatrix.x4` at two m-tiles, `.x2` at one)
  uint32_t raw[2 * P::kFoldMTiles], a[2 * P::kFoldMTiles], b[P::kFoldNTiles];
  if constexpr (P::kFoldMTiles == 2)
    ops::load_frag_t(raw, chunk_addr<P::kWriteRow>(pool + o, q));
  else
    ops::load_frag_t2(raw, chunk_addr<P::kWriteRow>(pool + o, q & 1));
  static_for<P::kFoldNTiles / 4>([&](auto Q) {
    uint32_t x[4];
    ops::load_frag_t(x, chunk_addr<P::kWriteRow>(pool + o, P::kValuesAt / 16 + 4 * Q + q));
    static_for<4>([&](auto k) { b[4 * Q + k] = x[k]; });
  });
  constexpr int kRaw = 2 * P::kFoldMTiles;
#pragma unroll
  for (int k = 0; k < kRaw; ++k) a[k] = ops::hmul2(raw[k], c01);
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<P::kFoldNTiles>(
        [&](auto N) { ops::hmma_k8(st.acc[J][M][N], a[2 * M], a[2 * M + 1], b[N]); });
  });
  if constexpr (Den) {
    const uint32_t d01 = __shfl_sync(0xffffffffu, dpair, 2 * t);
    static_for<P::kFoldMTiles>(
        [&](auto M) { ops::hmma_k8(den[M], raw[2 * M], raw[2 * M + 1], d01); });
  }
}

//: A LIST'S STEPS over its `n` entries: 16-token steps while more than 8 remain, then an 8-token tail, `fetch` the
//: entry at an index; `off`, `cpair` and `dpair` arrive holding the first step's lanes (entry `lane & 15`), and each
//: step fetches the next's in its own block, so its loads issue under its MMAs. -- see docs/internals/carry/ampere/fold.md#step
template <class P, int J, bool Den, class Fetch>
__device__ __forceinline__ void steps(State<P>& st, uint32_t cf, ops::SmemAddr pool, int first,
                                      int n, int lane, uint32_t off, uint32_t cpair, uint32_t dpair,
                                      DenColumn<P>& den, Fetch fetch) {
  int s = 0;
  for (int k = __any_sync(0xffffffffu, n > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      step16<P, J, Den>(st, pool, off, cpair, dpair, den, lane);
      s += 16;
      const int e = s + (lane & 15);
      const uint2 rec = fetch(min(e, n - 1));
      entry_of<P, Den>(rec, e < n, first, cf, off, cpair, dpair);
    } while (__any_sync(0xffffffffu, n - s > 8));
  for (int tail = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; tail > 0; --tail)
    step8<P, J, Den>(st, pool, off, cpair, dpair, den, lane);
}

//: A FOLD CHUNK'S FILL, the copies issued: its rows (slices then values) at `buf`, its entries after them (a warp its
//: own lists' runs, `run` the chunk's). -- see docs/internals/carry/ampere/parts.md#fill
template <class P>
__device__ __forceinline__ void fold_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& ww, const Place& c, int w, const Run& run,
                                          int first, int slots, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P, P::kFoldFactors, true>(addr.slices[kWrite], addr.values, static_cast<int>(p.values.st),
                                 union_of<P>(p, smem, kWrite, w), ww, first, slots, buf);
  const ops::SmemAddr ents = buf + static_cast<uint32_t>(fold_ents<P>(slots));
  if (ww.listed) {
    entries_fill<P, P::kLists>(p, p.write_lists, ww, c, w, run, ents, warp, lane);
  } else {
    //: a side with no list level: each slot's token's (F, m), 8 bytes, in the entries' place
    const float2* const src = p.factors + static_cast<int64_t>(c.bh) * p.L + ww.t0 + first;
#pragma unroll 1
    for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < slots; k += kThreads)
      ops::stage_run<8>(ents + 8u * static_cast<uint32_t>(k), src + k);
  }
}

//: A FOLD CHUNK, a warp's: each of its lists' run of the chunk (`run`, the chunk's) from the buffer's entries at
//: `ents`, in 16-token steps and an 8-token tail over the rows at `rows`, the list's scale read from its tenant and its
//: rescale applied before the window's first chunk (`rescale`) where its unit moved; a side with no list level every
//: slot of the chunk, c from the slot's row. -- see docs/internals/carry/ampere/fold.md#chunk
//: THE FOLD'S LAPS, an instrument: `laps(m)` at the chunk's marks (0 its lists' first steps' lanes read, `1 + 2 J` list
//: `J`'s rescale guarded, `2 + 2 J` its steps done); the kernel's are none. -- see docs/internals/carry/ampere/fold.md#laps
struct NoLaps {
  __device__ __forceinline__ void operator()(int) const {}
};

//: THE DEN LISTS a channel box carries, a mask over a warp's lists `j`: where `j boxes / 4` is the box.
//: -- see docs/internals/carry/ampere/fold.md#den
template <class P>
__device__ __forceinline__ uint32_t den_lists(int channel0, int dv) {
  const int box = channel0 / P::kChannels, boxes = dv / P::kChannels;
  uint32_t mask = 0;
  constexpr int kLists = P::kFoldLists;
#pragma unroll
  for (int j = 0; j < kLists; ++j) mask |= (j * boxes / kLists == box ? 1u : 0u) << j;
  return mask;
}

//: A LIST'S DEN COLUMN INTO THE DEN'S REGISTERS (`State::den`, `den_place`), each row times `2^(K - kCoefTop)`: list `J`
//: in the lanes whose pairs hold it, from their m-tile's column (every lane holds its quad's rows).
//: -- see docs/internals/carry/ampere/fold.md#den
template <int J, class P>
__device__ __forceinline__ void den_add(State<P>& st, const DenColumn<P>& den, int k, int lane) {
  static_assert(P::kFoldMTiles <= 2,
                "one or two m-tiles: a lane run's m-tile its index's `kRun` bit");
  constexpr int kPairs = P::kDenLists, kRun = P::kFoldLists / kPairs;
  const int t = lane & 3;
  const bool mine = t % kRun == J / kPairs, high = P::kFoldMTiles == 2 && (t & kRun) != 0;
  const float f = pow2(k - kCoefTop);
  static_for<2>([&](auto B) {
    const float x = high ? den[P::kFoldMTiles - 1][2 * B] : den[0][2 * B];
    st.den[J % kPairs + kPairs * B] += mine ? x * f : 0.0f;
  });
}

//: A DENSE SIDE'S STEPS (no list level), one list: every slot of the chunk ONCE for all the warp's lists -- a 16-slot
//: step's slices and values loaded once, each list's A the slices times its c, four lists' eight MMAs, and where the warp
//: does its pair's den (`Den`, warps `w < 4` of a box carrying it) the pair's eight lists' tile, two more MMAs on the raw
//: slices (B each slot's c_m); an 8-slot tail after. -- see docs/internals/carry/ampere/fold.md#dense
//: @burst-exempt a loop over the chunk's slots (its votes and `unroll 1`), not yet `Burst` units
template <class P, bool Den, class Laps>
__device__ __forceinline__ void steps_dense(State<P>& st, bool rescale, const uint8_t* smem,
                                            ops::SmemAddr pool, ops::SmemAddr rows,
                                            ops::SmemAddr ents, int first, int slots, float p2,
                                            float p2m, float (&den)[P::kFoldMTiles][4], int warp,
                                            int lane, Laps laps) {
  constexpr int kLists = P::kFoldLists, kTiles = P::kFoldMTiles;
  static_assert(kLists == 4 || kLists == 2, "two lists' c a packed word, one or two words a slot");
  //: THE DEN TILE'S COLUMNS, four a lane half (the MMA's N, 8): a pair of warps' lists at four a warp, at two a quad's
  //: (`w`, `w + 2` in a half, `w + 4`, `w + 6` in the other); `kLoads` coefficients a lane
  constexpr int kCols = 4, kLoads = Den ? kCols : kLists;
  ListScale sc[kLists];
  static_for<kLists>([&](auto J) {
    sc[J] = reinterpret_cast<const ListScale*>(smem + P::kScale)[fold_list(warp, J)];
  });
  //: each slot's weight factor `255 2^(4 - k)` (exact: a power of two times 255), the den's c_m = f16(b f16(m 255 2^(4 -
  //: k))) rounded twice as ruled
  const float w255 = 255.0f * p2m;
  //: slot `s + (lane & 15)`'s row offset, its c paired with the next lane's a list each (lanes `0..15`), and (`Den`) its
  //: c_m pairs a list, lanes `0..15` the warp's lists and `16..31` its partner's (`w + 4`)
  const int owner = Den && lane >= 16 ? warp + kWarps / 2 : warp;
  const auto list_of = [&](int j) {
    return kLists == 4 ? fold_list(owner, j) : fold_list(owner + 2 * (j >> 1), j & 1);
  };
  const auto entry = [&](int s, uint32_t& off, uint32_t(&cp)[kLists], uint32_t(&cm)[kCols]) {
    const int e = s + (lane & 15);
    const bool live = e < slots;
    const uint32_t k = static_cast<uint32_t>(max(min(e, slots - 1), 0));
    const uint32_t row0 = rows + k * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(k, 0);
    float c[kLists];
    uint32_t bh[kLoads];
    float fk, mk;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(fk), "=f"(mk) : "r"(ents + 8u * k));
    static_for<kLoads>([&](auto J) {
      const uint32_t list = static_cast<uint32_t>(list_of(J));
      asm volatile("ld.shared.u16 %0, [%1];"
                   : "=r"(bh[J])
                   : "r"(chunk_addr<P::kWriteRow>(row0, (2u * P::kRows + 2u * list) >> 4)
                         + (2u * list & 15u)));
      constexpr int kJ = decltype(J)::value;
      if constexpr (kJ < kLists) {
        const float b = __half2float(__ushort_as_half(static_cast<unsigned short>(bh[J])));
        c[J] = __fmul_rn(__fmul_rn(__fmul_rn(b, 255.0f), fk), p2);
      }
    });
    if constexpr (kLists == 4) {
      const __half2 h01 = __floats2half2_rn(c[0], c[1]), h23 = __floats2half2_rn(c[2], c[3]);
      const uint32_t w01 = live ? *reinterpret_cast<const uint32_t*>(&h01) : 0u,
                     w23 = live ? *reinterpret_cast<const uint32_t*>(&h23) : 0u;
      const uint32_t n01 = __shfl_xor_sync(0xffffffffu, w01, 1),
                     n23 = __shfl_xor_sync(0xffffffffu, w23, 1);
      cp[0] = ops::hmul2(__byte_perm(w01, n01, 0x5410), sc[0].cf);
      cp[1] = ops::hmul2(__byte_perm(w01, n01, 0x7632), sc[1].cf);
      cp[2] = ops::hmul2(__byte_perm(w23, n23, 0x5410), sc[2].cf);
      cp[3] = ops::hmul2(__byte_perm(w23, n23, 0x7632), sc[3].cf);
    } else {
      const __half2 h01 = __floats2half2_rn(c[0], c[1]);
      const uint32_t w01 = live ? *reinterpret_cast<const uint32_t*>(&h01) : 0u;
      const uint32_t n01 = __shfl_xor_sync(0xffffffffu, w01, 1);
      cp[0] = ops::hmul2(__byte_perm(w01, n01, 0x5410), sc[0].cf);
      cp[1] = ops::hmul2(__byte_perm(w01, n01, 0x7632), sc[1].cf);
    }
    if constexpr (Den) {
      const __half2 wm = __float2half2_rn(live ? mk * w255 : 0.0f);
      const uint32_t w2 = *reinterpret_cast<const uint32_t*>(&wm);
      const uint32_t m01 = ops::hmul2(__byte_perm(bh[0], bh[1], 0x5410), w2),
                     m23 = ops::hmul2(__byte_perm(bh[2], bh[3], 0x5410), w2);
      const uint32_t n01 = __shfl_xor_sync(0xffffffffu, m01, 1),
                     n23 = __shfl_xor_sync(0xffffffffu, m23, 1);
      cm[0] = __byte_perm(m01, n01, 0x5410);
      cm[1] = __byte_perm(m01, n01, 0x7632);
      cm[2] = __byte_perm(m23, n23, 0x5410);
      cm[3] = __byte_perm(m23, n23, 0x7632);
    }
    const uint32_t u = live ? static_cast<uint32_t>(first) + k : static_cast<uint32_t>(first);
    off = u * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(u, 0);
  };
  //: THE DEN'S B (`Den`): lane `(g, t)` column `g`'s c_m at the slot pair from `from` -- the warp's list `g` (`g < 4`),
  //: else its partner's list `g - 4`
  const auto den_b = [&](const uint32_t(&cm)[kCols], int from) {
    const int g = lane >> 2, src = from + 16 * (g >> 2);
    uint32_t x[kCols];
    static_for<kCols>([&](auto J) { x[J] = __shfl_sync(0xffffffffu, cm[J], src); });
    const int j = g & 3;
    return j == 0 ? x[0] : j == 1 ? x[1] : j == 2 ? x[2] : x[3];
  };

  laps(0);
  static_for<kLists>([&](auto J) {
    const float f = rescale ? sc[J].f : 1.0f;
    //: a list whose unit stood multiplies nothing (one trip where it moved; `f` is the warp's)
    for (int k = __any_sync(0xffffffffu, f != 1.0f) ? 1 : 0; k > 0; --k)
      static_for<kTiles>([&](auto M) {
        static_for<P::kFoldNTiles>(
            [&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f; }); });
      });
  });
  laps(1);

  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  uint32_t off, cp[kLists], cm[kCols];
  entry(0, off, cp, cm);
  int s = 0;
  for (int k = __any_sync(0xffffffffu, slots > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      const uint32_t oa = __shfl_sync(0xffffffffu, off, 8 * (q >> 1) + r);
      const uint32_t ob = __shfl_sync(0xffffffffu, off, 8 * (q & 1) + r);
      uint32_t raw[kTiles][4], b[P::kFoldNTiles][2];
      static_for<kTiles>([&](auto M) {
        ops::load_frag_t(raw[M], chunk_addr<P::kWriteRow>(pool + oa, 2 * M + (q & 1)));
      });
      static_for<P::kFoldNTiles / 2>([&](auto H) {
        uint32_t x[4];
        ops::load_frag_t(x,
                         chunk_addr<P::kWriteRow>(pool + ob, P::kValuesAt / 16 + 2 * H + (q >> 1)));
        b[2 * H][0] = x[0];
        b[2 * H][1] = x[1];
        b[2 * H + 1][0] = x[2];
        b[2 * H + 1][1] = x[3];
      });
      static_for<kLists>([&](auto J) {
        const uint32_t c01 = __shfl_sync(0xffffffffu, cp[J], 2 * t),
                       c89 = __shfl_sync(0xffffffffu, cp[J], 2 * t + 8);
        static_for<kTiles>([&](auto M) {
          const uint32_t a[4] = {ops::hmul2(raw[M][0], c01), ops::hmul2(raw[M][1], c01),
                                 ops::hmul2(raw[M][2], c89), ops::hmul2(raw[M][3], c89)};
          static_for<P::kFoldNTiles>(
              [&](auto N) { ops::hmma(st.acc[J][M][N], a, b[N][0], b[N][1]); });
        });
      });
      if constexpr (Den) {
        const uint32_t d01 = den_b(cm, 2 * t), d89 = den_b(cm, 2 * t + 8);
        static_for<kTiles>([&](auto M) { ops::hmma(den[M], raw[M], d01, d89); });
      }
      s += 16;
      entry(s, off, cp, cm);
    } while (__any_sync(0xffffffffu, slots - s > 8));
  for (int tail = __any_sync(0xffffffffu, slots - s > 0) ? 1 : 0; tail > 0; --tail) {
    const uint32_t o = __shfl_sync(0xffffffffu, off, r);
    uint32_t raw[4], b[P::kFoldNTiles];
    ops::load_frag_t(raw, chunk_addr<P::kWriteRow>(pool + o, q));
    static_for<P::kFoldNTiles / 4>([&](auto Q) {
      uint32_t x[4];
      ops::load_frag_t(x, chunk_addr<P::kWriteRow>(pool + o, P::kValuesAt / 16 + 4 * Q + q));
      static_for<4>([&](auto k) { b[4 * Q + k] = x[k]; });
    });
    static_for<kLists>([&](auto J) {
      const uint32_t c01 = __shfl_sync(0xffffffffu, cp[J], 2 * t);
      uint32_t a[4];
      static_for<4>([&](auto K) { a[K] = ops::hmul2(raw[K], c01); });
      static_for<kTiles>([&](auto M) {
        static_for<P::kFoldNTiles>(
            [&](auto N) { ops::hmma_k8(st.acc[J][M][N], a[2 * M], a[2 * M + 1], b[N]); });
      });
    });
    if constexpr (Den) {
      const uint32_t d01 = den_b(cm, 2 * t);
      static_for<kTiles>([&](auto M) { ops::hmma_k8(den[M], raw[2 * M], raw[2 * M + 1], d01); });
    }
  }
  static_for<2 * kLists - 1>([&](auto L) { laps(2 + L); });
}

//: A DENSE FOLD'S DEN TILES INTO THE DEN'S REGISTERS (`den_place`, the list owner's), at `2^(kden - kCoefTop)` where the
//: box carries the list: warp `w < 4`'s own columns by a shuffle, its partner `w + 4`'s through shared memory (2 KB over
//: the den image, free while folding: its readers done at the window's last drain; a barrier).
//: -- see docs/internals/carry/ampere/fold.md#dense
//: THE TWO-LIST FORM: a doer `w < 2` holds a quad's tile (warps `w`, `w + 2`, `w + 4`, `w + 6`); every doer lane's tile
//: goes through shared memory (its own columns too), each owner lane takes its two words.
template <class P>
__device__ __forceinline__ void den_quads(State<P>& st, const uint8_t* smem,
                                          const float (&den)[P::kFoldMTiles][4], int kden,
                                          uint32_t dens, int warp, int lane) {
  static_assert(P::kFoldLists == 2 && P::kFoldMTiles == 2 && P::kDenRegs == 2,
                "two lists a warp of two m-tiles, two den words a lane");
  const int g = lane >> 2, t = lane & 3;
  const DenPlace<P> at = den_place<P>(warp, lane);
  const float f = pow2(kden - kCoefTop);
  const ops::SmemAddr ex = ops::smem_addr(smem + P::kDenImage);
  //: a doer lane's m-tile `M` at `(doer, M, g, t)` -- 2 doers x 2 m-tiles x 32 lanes x 16 bytes
  const auto slot = [&](int doer, int M, int tq) {
    return ex + 16u * static_cast<uint32_t>(((doer * 2 + M) * 8 + g) * 4 + tq);
  };
  if (warp < 2) {
    static_for<2>([&](auto M) {
      ops::store_shared_vec16(slot(warp, M, t),
                              make_uint4(__float_as_uint(den[M][0]), __float_as_uint(den[M][1]),
                                         __float_as_uint(den[M][2]), __float_as_uint(den[M][3])));
    });
  }
  ops::rendezvous();
  //: owner `w`'s columns are its doer's (`w & 1`) lane `w >> 1` of the quad: its list `t & 1` an element of each row
  const uint4 v = ops::load_shared_vec16(slot(warp & 1, t >> 1, warp >> 1));
  const bool odd = (t & 1) != 0;
  const float x[2] = {__uint_as_float(odd ? v.y : v.x), __uint_as_float(odd ? v.w : v.z)};
  static_for<2>([&](auto I) {
    const bool carried = ((dens >> (at.list(I) >> 3)) & 1u) != 0u;
    st.den[I] += carried ? x[I] * f : 0.0f;
  });
}

template <class P>
__device__ __forceinline__ void den_pairs4(State<P>& st, const uint8_t* smem,
                                           const float (&den)[P::kFoldMTiles][4], int kden,
                                           uint32_t dens, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const bool doer = warp < kWarps / 2;
  static_assert(P::kFoldMTiles == 2 && P::kDenRegs == 4,
                "the exchange serves four lists a warp of two m-tiles, four den words a lane");
  const DenPlace<P> at = den_place<P>(warp, lane);
  const float f = pow2(kden - kCoefTop);
  static_assert(kMaxExtent * kMaxExtent * 2 >= 128 * 16,
                "the exchange's 128 16-byte slots fit the den image");
  const ops::SmemAddr ex = ops::smem_addr(smem + P::kDenImage);
  //: word `i` of lane `(g, t)` is element `i` of digit tile `t >> 1` in lane `(g, t & 1)` (own) or `(g, 2 + (t & 1))`
  //: (the partner's)
  const auto slot = [&](int pair, int half, int tile) {
    return ex + 16u * static_cast<uint32_t>(((pair * 8 + g) * 2 + half) * 2 + tile);
  };
  float up[4];
  static_for<4>([&](auto I) { up[I] = __shfl_xor_sync(0xffffffffu, den[1][I], 2); });
  if (doer) {
    static_for<4>([&](auto I) {
      const bool carried = ((dens >> (at.list(I) >> 3)) & 1u) != 0u;
      st.den[I] += carried ? ((t & 2) ? up[I] : den[0][I]) * f : 0.0f;
    });
    if (t & 2) {
      static_for<2>([&](auto M) {
        ops::store_shared_vec16(slot(warp, t - 2, M),
                                make_uint4(__float_as_uint(den[M][0]), __float_as_uint(den[M][1]),
                                           __float_as_uint(den[M][2]), __float_as_uint(den[M][3])));
      });
    }
  }
  ops::rendezvous();
  if (!doer) {
    const uint4 v = ops::load_shared_vec16(slot(warp - kWarps / 2, t & 1, t >> 1));
    const float x[4] = {__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z),
                        __uint_as_float(v.w)};
    static_for<4>([&](auto I) {
      const bool carried = ((dens >> (at.list(I) >> 3)) & 1u) != 0u;
      st.den[I] += carried ? x[I] * f : 0.0f;
    });
  }
}

//: THE ONE-M-TILE FORM (four lists a warp of one m-tile, a lane's list `w + 8 t` at rows `g`, `g + 8`): column `c` of
//: the tile is lane `(g, c >> 1)`'s elements `c & 1` and `2 + (c & 1)` -- a doer `w < 4` its own column `t` by a quad
//: shuffle, its partner `w + 4`'s columns `4 + t` through shared memory (1 KB over the den image; a barrier).
template <class P>
__device__ __forceinline__ void den_pairs1(State<P>& st, const uint8_t* smem,
                                           const float (&den)[P::kFoldMTiles][4], int kden,
                                           uint32_t dens, int warp, int lane) {
  static_assert(P::kFoldLists == 4 && P::kFoldMTiles == 1 && P::kDenRegs == 2,
                "four lists a warp of one m-tile, two den words a lane");
  const int g = lane >> 2, t = lane & 3;
  const bool doer = warp < kWarps / 2, odd = (t & 1) != 0;
  const DenPlace<P> at = den_place<P>(warp, lane);
  const float f = pow2(kden - kCoefTop);
  const ops::SmemAddr ex = ops::smem_addr(smem + P::kDenImage);
  //: a doer lane `(g, 2 + h)`'s tile at `(doer, g, h)`: 4 doers x 8 rows x 2 x 16 bytes
  const auto slot = [&](int d, int h) {
    return ex + 16u * static_cast<uint32_t>((d * 8 + g) * 2 + h);
  };
  float own[4];
  static_for<4>([&](auto I) { own[I] = __shfl_sync(0xffffffffu, den[0][I], 4 * g + (t >> 1)); });
  const auto add = [&](float x0, float x1) {
    static_for<2>([&](auto I) {
      const bool carried = ((dens >> (at.list(I) >> 3)) & 1u) != 0u;
      st.den[I] += carried ? (decltype(I)::value == 0 ? x0 : x1) * f : 0.0f;
    });
  };
  if (doer) {
    add(odd ? own[1] : own[0], odd ? own[3] : own[2]);
    if (t & 2)
      ops::store_shared_vec16(slot(warp, t - 2),
                              make_uint4(__float_as_uint(den[0][0]), __float_as_uint(den[0][1]),
                                         __float_as_uint(den[0][2]), __float_as_uint(den[0][3])));
  }
  ops::rendezvous();
  if (!doer) {
    const uint4 v = ops::load_shared_vec16(slot(warp - kWarps / 2, t >> 1));
    add(__uint_as_float(odd ? v.y : v.x), __uint_as_float(odd ? v.w : v.z));
  }
}

template <class P>
__device__ __forceinline__ void den_pairs(State<P>& st, const uint8_t* smem,
                                          const float (&den)[P::kFoldMTiles][4], int kden,
                                          uint32_t dens, int warp, int lane) {
  if constexpr (P::kFoldLists == 2)
    den_quads<P>(st, smem, den, kden, dens, warp, lane);
  else if constexpr (P::kFoldMTiles == 1)
    den_pairs1<P>(st, smem, den, kden, dens, warp, lane);
  else
    den_pairs4<P>(st, smem, den, kden, dens, warp, lane);
}

template <class P, class Laps = NoLaps>
__device__ __forceinline__ void fold_chunk(State<P>& st, bool rescale, const uint8_t* smem,
                                           const Window& ww, int w, const Run& run, int first,
                                           int slots, ops::SmemAddr rows, ops::SmemAddr ents,
                                           int warp, int lane, uint32_t dens, Laps laps = {}) {
  const ops::SmemAddr pool = rows - static_cast<uint32_t>(first * P::kWriteRow);
  //: a side with no list level: 2^(4 - E) of the window's exponents of F and m (`dense_exp`), the facts records' scale
  const int16_t* const exps =
      reinterpret_cast<const int16_t*>(smem + P::kWriteExp) + parity_of(w) * kMaxExtent * 2;
  const float p2 = ww.listed ? 1.0f : ldexpf(1.0f, 4 - dense_exp(exps[1])),
              p2m = ww.listed ? 1.0f : ldexpf(1.0f, 4 - dense_exp(exps[0]));
  //: a listed side's lists carry the den in their steps; a dense side's rides its one list's (`steps_dense`)
  const uint32_t list_dens = dens;
  //: THE WARP'S LISTS: every list's scale, count, base and first step's lanes before any list's MMAs (four independent
  //: chains), then each list's rescale and steps (the den's column in them); `fetch(base, k)` a list's entry `k`
  const auto lists = [&](const int(&n)[P::kFoldLists], const int(&base)[P::kFoldLists],
                         auto fetch) {
    constexpr int kLists = P::kFoldLists;
    ListScale sc[kLists];
    uint32_t off[kLists], cpair[kLists], dpair[kLists];
    static_for<kLists>([&](auto J) {
      sc[J] = reinterpret_cast<const ListScale*>(smem + P::kScale)[fold_list(warp, J)];
    });
    static_for<kLists>([&](auto J) {
      const int e = lane & 15;
      const uint2 rec = fetch(base[J], max(min(e, n[J] - 1), 0));
      entry_of<P, true>(rec, e < n[J], first, sc[J].cf, off[J], cpair[J], dpair[J]);
    });

    laps(0);
    static_for<kLists>([&](auto J) {
      const float f = rescale ? sc[J].f : 1.0f;
      //: a list whose unit stood multiplies nothing (one trip where it moved; `f` is the warp's)
      for (int k = __any_sync(0xffffffffu, f != 1.0f) ? 1 : 0; k > 0; --k)
        static_for<P::kFoldMTiles>([&](auto M) {
          static_for<P::kFoldNTiles>(
              [&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f; }); });
        });
      laps(1 + 2 * J);
      //: a list whose den the box carries (a uniform branch): its column, then into the den's registers at its exponent
      const auto list_fetch = [&](int k) { return fetch(base[J], k); };
      if (((list_dens >> J) & 1u) != 0u) {
        DenColumn<P> den{};
        steps<P, J, true>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J], dpair[J], den,
                          list_fetch);
        den_add<J>(st, den,
                   ww.listed ? max(static_cast<int>(exps[2 * fold_list(warp, J)]), kUnitMin)
                             : dense_exp(exps[0]),
                   lane);
      } else {
        DenColumn<P> none;
        steps<P, J, false>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J], 0u, none,
                           list_fetch);
      }
      laps(2 + 2 * J);
    });
  };

  //: the arm's fold side is listed (one dense level) or dense (two), and the window's plan agrees
  if (ww.listed != (P::kFoldFactors == 1)) __trap();
  if constexpr (P::kFoldFactors == 1) {
    int n[P::kFoldLists], base[P::kFoldLists];
    //: a list's count and place, each under 2^16, one shuffle
    const uint32_t packed = static_cast<uint32_t>(run.n) << 16 | static_cast<uint32_t>(run.to);
    static_for<P::kFoldLists>([&](auto J) {
      const uint32_t v = __shfl_sync(0xffffffffu, packed, fold_list(warp, J));
      n[J] = static_cast<int>(v >> 16);
      base[J] = static_cast<int>(v & 0xffffu);
    });
    lists(n, base, [&](int to, int k) {
      uint2 e;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(e.x), "=r"(e.y)
                   : "r"(ents + 8u * static_cast<uint32_t>(to + k)));
      return e;
    });
  } else {
    //: the den's MMAs only where the box carries its lists', a pair's in its lower warp: two instantiations
    float den[P::kFoldMTiles][4] = {};
    //: a vote, so ptxas sees a warp-uniform choice (a branch on the warp's index guards every collective under it)
    if (__any_sync(0xffffffffu, dens != 0u && warp < kWarps * P::kFoldLists / 8)) {
      steps_dense<P, true>(st, rescale, smem, pool, rows, ents, first, slots, p2, p2m, den, warp,
                           lane, laps);
    } else {
      steps_dense<P, false>(st, rescale, smem, pool, rows, ents, first, slots, p2, p2m, den, warp,
                            lane, laps);
    }
    for (int once = ops::once(dens != 0u); once > 0; --once)
      den_pairs<P>(st, smem, den, dense_exp(exps[0]), dens, warp, lane);
  }
}

//: THE PROBE (an instrument; nothing unbound): the state at the call's end, absolute, fp32 `[BH, N, d_v]` by global
//: leaf in level order. -- see docs/internals/carry/ampere/fold.md#probe
__device__ __forceinline__ int64_t level_stride(const CarryParams& p, int level) {
  int64_t s = 1;
#pragma unroll 1
  for (int l = level + 1; l < p.D; ++l) s *= p.widths[l];
  return s;
}

template <class P>
__device__ __forceinline__ void fold_probe(const CarryParams& p, const State<P>& st,
                                           const uint8_t* smem, const Window& win, const Place& c,
                                           int warp, int lane) {
  if (p.probe == nullptr && p.probe_den == nullptr) return;
  const int64_t step = level_stride(p, -1);
  int64_t base = 0;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) base += origin_of(p, l) * level_stride(p, l);
  const int64_t list_stride = level_stride(p, win.listed ? p.write.list : p.write.operand[1]);
  const int64_t row_stride = level_stride(p, p.write.operand[0]);
  const int g = lane >> 2, tq = lane & 3;
  const UnitBook* book = reinterpret_cast<const UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) {
    const int list = fold_list(warp, J);
    const float scale = pow2(book[list].unit - kCoefTop) * kByteScale;
    const int64_t at_list = base + list * list_stride;
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<2>([&](auto Hi) {
        const int row = 16 * M + g + 8 * Hi;
        const int64_t leaf = at_list + row * row_stride;
        if (p.probe != nullptr) {
          float* const out =
              p.probe + (static_cast<int64_t>(c.bh) * step + leaf) * p.dv + c.channel0;
          static_for<P::kFoldNTiles>([&](auto N) {
            *reinterpret_cast<float2*>(out + 8 * N + 2 * tq) =
                make_float2(st.acc[J][M][N][2 * Hi] * scale, st.acc[J][M][N][2 * Hi + 1] * scale);
          });
        }
      });
    });
  });
  //: the den's words at their places (`den_place`), by the box carrying the list (`den_lists`: its group `list >> 3`)
  if (p.probe_den == nullptr) return;
  const DenPlace<P> at = den_place<P>(warp, lane);
  static_for<P::kDenRegs>([&](auto I) {
    if (((den_lists<P>(c.channel0, p.dv) >> (at.list(I) >> 3)) & 1u) != 0u)
      p.probe_den[static_cast<int64_t>(c.bh) * step + base + at.list(I) * list_stride
                  + at.digit(I) * row_stride] = st.den[I] * kByteScale;
  });
}

}  // namespace rola::carry
