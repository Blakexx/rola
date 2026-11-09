// csrc/rola/src/carry/ampere/parts.cuh -- THE CARRY'S COMMON PARTS: the fold's state and the den's register places, the CTA's
// place, a window's lists and facts in, a chunk's fills (its rows, its entries, its cleared panel) and runs.
// see docs/internals/carry/ampere/parts.md
#pragma once

#include <cuda_fp16.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#include "carry/ampere/plan.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"

namespace rola::carry {

namespace ops = ::denseref::ops;

//: THE UNITS' CONSTANTS: c's top 2^4, a state held under 2^100 of its unit, the lowest unit, a list never live,
//: the facts pass's dead exponent. -- see docs/internals/carry/ampere/fold.md#units
constexpr int kCoefTop = 4;
constexpr int kUnitHeadroom = 100;
constexpr int kUnitMin = kCoefTop - 127;
constexpr int kUnitNone = -1000;
constexpr int kExpDead = -126;

//: A LIST'S UNIT ACROSS WINDOWS, in shared memory: its exponent and its state's bound in units.
struct UnitBook {
  int unit;
  float bound;
};

__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

//: A POWER OF TWO over the whole float range: normal from 2^-126, a subnormal's bit down to 2^-149, zero below.
__device__ __forceinline__ float pow2_wide(int e) {
  return e >= -126 ? pow2(min(e, 127)) : e >= -149 ? __uint_as_float(1u << (e + 149)) : 0.0f;
}

//: the smallest `e` with `x <= 2^e`, `kUnitNone` for zero
__device__ __forceinline__ int ceil_exp(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits == 0u ? kUnitNone : static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: A WARP'S STATE, the fold's accumulators: per list, its digits' m-tiles by the channels' n-tiles, fp32 in the list's
//: unit; and the den's, the v+1 channel absolute, `kDenRegs` words a lane (`den_place`). -- see
//: docs/internals/carry/ampere/fold.md#state
template <class P>
struct State {
  float acc[P::kFoldLists][P::kFoldMTiles][P::kFoldNTiles][4];
  float den[P::kDenRegs];

  //: the state's `k`-th register, `k` compile-time
  template <int K>
  __device__ __forceinline__ float at() const {
    constexpr int kTile = 4 * P::kFoldNTiles;
    return acc[K / (P::kFoldMTiles * kTile)][K / kTile % P::kFoldMTiles][K / 4 % P::kFoldNTiles]
              [K % 4];
  }
};

//: THE CTA'S PLACE: its batch-head and its box's first channel (a level's first digit is `origin_of`).
//: -- see docs/internals/carry/ampere/parts.md#place
struct Place {
  int bh, channel0;
};

__device__ __forceinline__ int origin_of(const CarryParams& p, int level) {
  return (int)(blockIdx.x / (unsigned)p.stride[level]) % (p.widths[level] / p.extents[level])
         * p.extents[level];
}

__device__ __forceinline__ Place place_of(const CarryParams& p) {
  const int bh = (int)blockIdx.y;
  return Place{bh, (int)(blockIdx.x % (unsigned)(p.dv / p.channels)) * p.channels};
}

//: A SIDE'S OPERAND SLICES for this CTA: a level's slice at the box's digits, `bytes` a token's (a power of two), at
//: `st` a token.
struct Slices {
  const uint8_t* at[kMaxLevels];
  int st[kMaxLevels], sh[kMaxLevels];
  int bytes[kMaxLevels];
};

//: A SIDE'S WINDOW STATICS, the same every window: its union's group, its box's first list digit and its lists, and
//: whether it lists.
struct Statics {
  int group, digit0, lists, listed;
};

//: THE CTA'S ADDRESSES, worked out once before the windows, so the window loop divides by nothing: each side's statics
//: and operand slices, the values at the box's channels. -- see docs/internals/carry/ampere/parts.md#place
struct Addr {
  Statics side[2];
  Slices slices[2];
  const uint8_t* values;
  int channel0;
};

static_assert(sizeof(Addr) <= Tenants<512>::kAddrBytes, "the addresses fit their tenant");

template <class P>
__device__ __forceinline__ void addr_in(Addr& out, const CarryParams& p, const Place& c) {
  //: the read side's rows at the write head's first query head, `h g`
  const int b = c.bh / p.H, h = c.bh - b * p.H;
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    const Side& s = k == 0 ? p.read : p.write;
    Statics& st = out.side[k];
    st.listed = s.list >= 0;
    st.digit0 = st.listed ? origin_of(p, s.list) : 0;
    //: the facts pass's list group holding the box's digits (`plan.md#list-groups`)
    st.group = st.digit0 / (k == 0 ? P::kReadListGroup : P::kWriteListGroup);
    st.lists = st.listed ? p.extents[s.list] : kMaxExtent;
#pragma unroll 1
    for (int i = 0; i < s.operands; ++i) {
      const int level = s.operand[i];
      const Plane& pl = s.rows[i];
      out.slices[k].at[i] = pl.base + b * pl.sb + h * (k == 0 ? p.tables.heads : 1) * pl.sh
                            + (int64_t)origin_of(p, level) * s.elem[i];
      out.slices[k].st[i] = static_cast<int>(pl.st);
      out.slices[k].sh[i] = static_cast<int>(pl.sh);
      out.slices[k].bytes[i] = p.extents[level] * s.elem[i];
    }
  }
  out.values = p.values.base + b * p.values.sb + h * p.values.sh + c.channel0 * 2;
  out.channel0 = c.channel0;
}

//: a window's parity, opaque (`ops::opaque`), so an offset by it is formed at each use rather than held across a
//: window's steps
__device__ __forceinline__ int parity_of(int w) {
  int parity;
  asm volatile("and.b32 %0, %1, 1;" : "=r"(parity) : "r"(w));
  return parity;
}

//: A SIDE'S WINDOW: its union (the list level's 32-digit group holding the box's digits) and its box lists' first
//: digit, or every window token a slot where the side has no list level.
struct Window {
  int tokens, t0, group, digit0, lists;
  bool listed;
};

template <class P>
__device__ __forceinline__ Window window_of(const CarryParams& p, const uint8_t* smem, int side,
                                            int w) {
  //: read at each use (volatile): held live across the window loop they would cost the state registers
  const volatile Statics& s = reinterpret_cast<const volatile Addr*>(smem + P::kAddr)->side[side];
  Window out;
  out.t0 = ops::opaque(w) * P::kWindow;
  out.tokens = min(P::kWindow, p.L - out.t0);
  out.listed = s.listed != 0;
  out.digit0 = s.digit0;
  out.group = s.group;
  out.lists = s.lists;
  return out;
}

__device__ __forceinline__ int64_t list_row(const SideLists& l, const Place& c, int windows, int w,
                                            int digit) {
  return ((int64_t)c.bh * windows + w) * l.width + digit;
}

//: THE SIDES, as the tenants index them: the read side 0, the write side 1.
enum SideIndex : int { kRead = 0, kWrite = 1 };

//: A SIDE'S TENANTS for window `w` (two windows' a side, by parity): its union tokens and its group's boundaries in
//: prefix form -- the read side's in the call's read tables, a row of them its query heads' -- and its columns a row.
//: -- see docs/internals/carry/ampere/plan.md#tenants
template <class P>
__device__ __forceinline__ int cols_of(const CarryParams& p, int side) {
  return P::kListAt * (side == kRead ? p.tables.heads : 1);
}

template <class P>
__device__ __forceinline__ const uint16_t* union_of(const CarryParams& p, const uint8_t* smem,
                                                    int side, int w) {
  return side == kRead
             ? reinterpret_cast<const uint16_t*>(smem + p.tables.union_at)
                   + parity_of(w) * p.tables.heads * P::kWindow
             : reinterpret_cast<const uint16_t*>(smem + P::kUnionW) + parity_of(w) * P::kWindow;
}

template <class P>
__device__ __forceinline__ const uint16_t* pre_of(const CarryParams& p, const uint8_t* smem,
                                                  int side, int w) {
  return reinterpret_cast<const uint16_t*>(smem + (side == kRead ? p.tables.pre_at : P::kPreW))
         + parity_of(w) * kPreRows * cols_of<P>(p, side);
}

//: A CHUNK'S COLUMNS in the boundaries, union slots `first .. first + slots - 1` (`first` a multiple of 64, the end a
//: multiple of 64 or the union's `total`): the one before its first slot (-1 for none) and the one at its end, of a
//: row's `cols`.
__device__ __forceinline__ int2 chunk_cols(int first, int slots, int total, int cols) {
  return make_int2(first / kListBound - 1,
                   first + slots >= total ? cols - 1 : (first + slots) / kListBound - 1);
}

//: LISTS IN, a window ahead: a side's union tokens and count and its group's boundaries in prefix form, a row its query
//: heads' columns. -- see docs/internals/carry/ampere/parts.md#lists-in
template <class P>
__device__ __forceinline__ void lists_in(const CarryParams& p, const SideLists& l,
                                         const Window& win, const Place& c, int w, uint8_t* smem,
                                         int side) {
  if (!win.listed) return;
  const int cols = cols_of<P>(p, side), slots = P::kWindow * l.heads,
            rows = (side == kRead ? P::kReadListGroup : P::kWriteListGroup) + 1;
  const int64_t group = ((int64_t)c.bh * p.windows + w) * l.groups + win.group;
  const int union_copies = slots * 2 / ops::kStageBytes, row_copies = cols * 2 / ops::kStageBytes;
  const ops::SmemAddr unions = ops::smem_addr(union_of<P>(p, smem, side, w)),
                      pres = ops::smem_addr(pre_of<P>(p, smem, side, w));
  //: the thread opaque, so the copies' sources are formed at each staging, never held across the windows
#pragma unroll 1
  for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < union_copies + rows * row_copies;
       k += kThreads) {
    if (k < union_copies) {
      ops::stage_tile(unions + 16 * k, l.union_tok + group * slots + 8 * k);
    } else {
      const int r = k - union_copies;
      ops::stage_tile(pres + 16 * r, l.pre + group * rows * cols + 8 * r);
    }
  }
  if (threadIdx.x == kThreads - 1)
    ops::stage_run<4>(ops::smem_addr(smem + P::kCount) + 4 * (2 * side + (w & 1)),
                      l.union_count + group);
}

//: FACTS IN, a window ahead: the box's write lists' exponent pairs and each of the read side's query heads' tokens' scale
//: bytes, the den pass's (query head `j` of write head `bh` the call's `bh g + j`): the f16 sum's scale.
template <class P>
__device__ __forceinline__ void facts_in(const CarryParams& p, const Window& ww, const Place& c,
                                         int w, uint8_t* smem) {
  //: the box's lists' exponent pairs, 4 bytes a list (never past its row: a 16-list box's level may end there)
  constexpr int kUnitCopies = P::kWindow / ops::kStageBytes,
                kExpCopies = P::kLists * 4 / ops::kStageBytes;
  const int heads = p.tables.heads;
  const ops::SmemAddr scale =
      ops::smem_addr(smem + p.tables.scale_at) + (w & 1) * heads * P::kWindow;
#pragma unroll 1
  for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < heads * kUnitCopies + kExpCopies;
       k += kThreads) {
    if (k < heads * kUnitCopies) {
      const int j = k / kUnitCopies;
      ops::stage_tile(scale + 16 * k,
                      p.scale + ((int64_t)(c.bh * heads + j) * p.windows + w) * P::kWindow
                          + 16 * (k - j * kUnitCopies));
    } else if (ww.listed) {
      const int e = k - heads * kUnitCopies;
      ops::stage_tile(
          ops::smem_addr(smem + P::kWriteExp) + (w & 1) * kMaxExtent * 4 + 16 * e,
          p.write_exp + list_row(p.write_lists, c, p.windows, w, ww.digit0) * 2 + 8 * e);
    } else if (k == heads * kUnitCopies) {
      //: a write side with no list level: the window's shifts of (m, F), in the exponents' place
      ops::stage_run<4>(ops::smem_addr(smem + P::kWriteExp) + (w & 1) * kMaxExtent * 4,
                        p.factor_shift + (static_cast<int64_t>(c.bh) * p.windows + w) * 2);
    }
  }
}

//: A POOL ROW'S SWIZZLE at a compiled row (`pool_chunk`'s): the row's own start alternates halves of the bank line.
//: -- see docs/internals/carry/ampere/parts.md#swizzle
template <int Row>
__device__ __forceinline__ constexpr uint32_t chunk_at(uint32_t s, uint32_t k) {
  static_assert(Row % 64 == 32 || (Row % 64 == 0 && (Row == 128 || (Row / 64) % 2 == 1)),
                "a row of 128 bytes, an odd number of 64, or an odd number of 32");
  //: an odd number of 32 bytes (160, 224): slots `s`, `s + 4` a 128-byte line apart, the chunk's low bit flipped by
  //: `(s >> 2) & 1`
  if constexpr (Row % 64 == 32)
    return k ^ ((s >> 2) & 1u);
  else
    return pool_chunk(Row, s, k);
}

//: LOGICAL CHUNK `k`'s ADDRESS in a row whose chunk 0 sits at `row0` (its swizzle folded in): an XOR inside the
//: swizzle's group, the group's own offset added.
template <int Row>
__device__ __forceinline__ uint32_t chunk_addr(uint32_t row0, uint32_t k) {
  if constexpr (Row == 128) return row0 ^ (16u * k);
  if constexpr (Row % 64 == 32) return (row0 ^ (16u * (k & 1u))) + 16u * (k & ~1u);
  return (row0 ^ (16u * (k & 3u))) + 16u * (k & ~3u);
}

//: ONE ASYNC RUN of 16, 8 or 4 bytes, the run a slice's width gives; zero-filled for a gap slot (`live` false).
__device__ __forceinline__ void stage_bytes(ops::SmemAddr dst, const void* src, int run,
                                            bool live) {
  // R9-CASES: a copy run, 16, 8 or 4 bytes (the host refuses anything under 4)
  if (run == ops::kStageBytes) {
    ops::stage_run_if<16>(dst, src, live);
  } else if (run == 8) {
    ops::stage_run_if<8>(dst, src, live);
  } else if (run == 4) {
    ops::stage_run_if<4>(dst, src, live);
  } else {
    __trap();
  }
}

//: AN ASYNC 16-BYTE COPY at `dst + At`, issued where `At` is under `room`, its first `size` bytes read.
template <uint32_t At>
__device__ __forceinline__ void copy16(ops::SmemAddr dst, const void* src, int room, int size) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.gt.s32 p, %2, %4;\n@p cp.async.cg.shared.global [%0+%4], [%1], 16, %3;\n}\n" ::
          "r"(dst),
      "l"(src), "r"(room), "r"(size), "n"(At));
}

//: A 16-BYTE FILL of word `v` at `dst + 16 At`, stored where `At` is under `room`.
template <int At>
__device__ __forceinline__ void fill16(ops::SmemAddr dst, int room, uint32_t v) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.gt.s32 p, %1, %2;\n@p st.shared.v4.u32 [%0+%3], {%4, %4, %4, %4};\n}\n" ::
          "r"(dst),
      "r"(room), "n"(At), "n"(16 * At), "r"(v)
      : "memory");
}

//: A ROW'S SOURCE: `base` plus two products, each one wide multiply-add (the operands unsigned, under 2^32).
__device__ __forceinline__ const uint8_t* row_at(const uint8_t* base, int a, int b, int c, int d) {
  const uint64_t x = reinterpret_cast<uint64_t>(base)
                     + static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b);
  return reinterpret_cast<const uint8_t*>(
      x + static_cast<uint64_t>(static_cast<uint32_t>(c)) * static_cast<uint32_t>(d));
}

//: A CHUNK'S ROWS: `slots` union slots from `first`, each its token's row (its slices, then `Values` its values),
//: swizzled, by `cp.async`: thread `4 q + r` chunk `r` of slots `q + 64 k` of each 256, a trip one block -- its tokens
//: (unclamped), every source, then the copies on the trip's room; a gap slot zero-filled. -- see
//: docs/internals/carry/ampere/parts.md#fill
template <class P, int Factors, bool Values>
__device__ __forceinline__ void fill(const volatile Slices& sl, const uint8_t* values,
                                     int values_st, const uint16_t* tok, const Window& win,
                                     int first, int slots, ops::SmemAddr pool) {
  static_assert(2 * P::kRows <= 4 * ops::kStageBytes && 2 * P::kK <= 4 * ops::kStageBytes
                    && 2 * P::kLists <= 4 * ops::kStageBytes,
                "a fill's quad copies a slice, a 16-byte chunk a lane");
  constexpr int kRow = Values ? P::kWriteRow : P::kReadRow;
  //: a slice's first chunk in the row (the write side's rows, then a dense side's lists) and a lane's value chunks
  constexpr int kSlice1 = 2 * P::kRows / ops::kStageBytes,
                kValues = P::kValuesAt / ops::kStageBytes;
  constexpr int kValueCopies = Values ? 2 * P::kChannels / (4 * ops::kStageBytes) : 0;
  constexpr int kQuads = kThreads / 4, kAhead = 4;
  //: the thread opaque, so its quad lane and first slot are formed at each fill, never held across the windows
  const int tid = ops::opaque(static_cast<int>(threadIdx.x)), r = tid & 3, q = tid >> 2;
  const uint8_t* base[Factors];
  int st[Factors], sh[Factors], bytes[Factors], size[Factors];
  bool active[Factors], narrow = false;
  static_for<Factors>([&](auto i) {
    bytes[i] = sl.bytes[i];
    st[i] = sl.st[i];
    sh[i] = Values ? 0 : sl.sh[i];
    base[i] = sl.at[i] + static_cast<int64_t>(win.t0) * st[i] + ops::kStageBytes * r;
    active[i] = ops::kStageBytes * r < bytes[i];
    size[i] = active[i] ? ops::kStageBytes : 0;
    narrow |= bytes[i] < ops::kStageBytes;
  });
  const uint8_t* const vbase =
      Values ? values + static_cast<int64_t>(win.t0) * values_st + ops::kStageBytes * r : nullptr;

  const bool listed = win.listed;
  const int16_t* const toks = reinterpret_cast<const int16_t*>(tok) + first + q;
  //: the thread's rows' places at slot `q`: each slice's chunk and the values' (`chunk_at`)
  const ops::SmemAddr row = pool + static_cast<uint32_t>(q * kRow);
  uint32_t at[Factors + kValueCopies];
  static_for<Factors + kValueCopies>([&](auto i) {
    constexpr int I = decltype(i)::value;
    constexpr int k0 = I < Factors ? (I == 0 ? 0 : kSlice1) : kValues + 4 * (I - Factors);
    at[i] = row + 16u * chunk_at<kRow>(static_cast<uint32_t>(q), static_cast<uint32_t>(k0 + r));
  });

  //: a narrow slice (8 or 4 bytes) its quad's first lane's whole run; a trip of 256 slots, the first straight-line
  const auto rows = [&](auto kNarrow) {
    const auto trip_at = [&](int s0) {
      //: the trip's four tokens, loaded on either side and selected (an unlisted side's slot is its token)
      int u[kAhead];
      static_for<kAhead>([&](auto k) {
        const int v = toks[s0 + kQuads * k];
        u[k] = listed ? v : first + q + s0 + kQuads * k;
      });
      const int room = (slots - q - s0) * kRow;
      const uint32_t trip = static_cast<uint32_t>(s0 * kRow);

      //: every source first: a slot's query head `j` (the read side's) and token `t`, its gap mask
      const uint8_t* src[kAhead][Factors + 1];
      int gap[kAhead];
      static_for<kAhead>([&](auto k) {
        const int t = u[k] & (P::kWindow - 1),
                  j = Values ? 0
                             : (u[k] >> window_bits(P::kWindow)) & (window_heads(P::kWindow) - 1);
        gap[k] = u[k] >> 31;
        static_for<Factors>([&](auto i) { src[k][i] = row_at(base[i], t, st[i], j, sh[i]); });
        if constexpr (Values) src[k][Factors] = row_at(vbase, t, values_st, 0, 0);
      });

      static_for<kAhead>([&](auto k) {
        constexpr uint32_t kAt = static_cast<uint32_t>(decltype(k)::value * kQuads * kRow);
        static_for<Factors>([&](auto i) {
          if constexpr (decltype(kNarrow)::value) {
            const ops::SmemAddr dst = at[i] + trip + kAt;
            const bool issue = static_cast<int>(kAt) < room, live = gap[k] == 0;
            if (issue && !active[i] && r == 0) stage_bytes(dst, src[k][i], bytes[i], live);
            if (issue && active[i]) ops::stage_run_if<16>(dst, src[k][i], live);
          } else {
            //: a slice's chunks in the box's row: a lane past them issues nothing (a zero-fill would land on the
            //: next part of the row)
            constexpr int kIn =
                (decltype(i)::value == 0 ? (Values ? 2 * P::kRows : 2 * P::kK) : 2 * P::kLists)
                / ops::kStageBytes;
            if constexpr (kIn >= 4)
              copy16<kAt>(at[i] + trip, src[k][i], room, size[i] & ~gap[k]);
            else
              copy16<kAt>(at[i] + trip, src[k][i], r < kIn ? room : 0, size[i] & ~gap[k]);
          }
        });
        static_for<kValueCopies>([&](auto v) {
          copy16<kAt>(at[Factors + v] + trip, src[k][Factors] + 4 * ops::kStageBytes * v, room,
                      ops::kStageBytes & ~gap[k]);
        });
      });
    };

    trip_at(0);
#pragma unroll 1
    for (int s0 = kAhead * kQuads; s0 < slots - q; s0 += kAhead * kQuads) trip_at(s0);
  };
  if (narrow) {
    rows(std::true_type{});
  } else {
    rows(std::false_type{});
  }
}

//: A CHUNK'S TOKEN-GROUP SPLIT, its slots `0 .. slots - 1`: the readout's group 0 below it, group 1 from it, a multiple
//: of 64 so a list's run splits where the prefix rows hold a boundary. -- see docs/internals/carry/ampere/readout.md#warps
__host__ __device__ __forceinline__ constexpr int group_split(int slots) {
  return slots <= kListBound ? slots
                             : kListBound * ((slots + 2 * kListBound - 1) / (2 * kListBound));
}

//: A CHUNK'S RUNS, lane `l` list `l`'s: its first entry of the chunk, its count, its count below the chunk's token-group
//: split, and its place in a buffer's entries (the lists' runs in list order), from its prefix rows modulo 2^16.
//: -- see docs/internals/carry/ampere/parts.md#entries
struct Run {
  int at, n, n0, to;
};

template <class P>
__device__ __forceinline__ Run chunk_runs(const CarryParams& p, const uint8_t* smem,
                                          const Window& win, int side, int w, int first, int slots,
                                          int total, int lane) {
  if (!win.listed || lane >= win.lists) return Run{0, 0, 0, 0};
  const int cols = cols_of<P>(p, side);
  const int2 col = chunk_cols(first, slots, total, cols);
  //: the lane's two rows (lists `l`, `l + 1`): the chunk's end in both, its start where it is not the first, the split
  //: where it falls inside the chunk
  const int group = side == kRead ? P::kReadListGroup : P::kWriteListGroup;
  const uint16_t* const row =
      pre_of<P>(p, smem, side, w) + ((win.digit0 & (group - 1)) + lane) * cols;
  const int a1 = row[col.y], b1 = row[cols + col.y];
  const bool start = col.x >= 0;
  const int a0 = start ? row[max(col.x, 0)] : 0, b0 = start ? row[cols + max(col.x, 0)] : 0;
  const int split = group_split(slots), cm = (first + split) / kListBound - 1;
  const int n = ((b1 - a1) - (b0 - a0)) & 0xffff;
  const int n0 = split >= slots ? n : ((row[cols + cm] - row[cm]) - (b0 - a0)) & 0xffff;
  return Run{(b0 - a0) & 0xffff, n, n0, (a1 - a0) & 0xffff};
}

//: A CHUNK'S ENTRIES, a warp's share: its own lists' runs (`warp + 8 j`, of the side's `Lists`) to their places at
//: `dst`, 8 lanes a run at 4 lists a warp or 16 at 2, two copies a trip, the trips the warp's longest run's.
//: -- see docs/internals/carry/ampere/parts.md#entries
template <class P, int Lists>
__device__ __forceinline__ void entries_fill(const CarryParams& p, const SideLists& l,
                                             const Window& win, const Place& c, int w,
                                             const Run& run, ops::SmemAddr dst, int warp,
                                             int lane) {
  static_assert(Lists == 4 * kWarps || Lists == 2 * kWarps, "a warp's four lists or two");
  constexpr int kLanes = 32 * kWarps / Lists;
  //: a list's row `kListCap` entries a query head on (opaque: formed at each fill, never held across windows)
  const int cap = P::kListCap * ops::opaque(l.heads);
  const int list = warp + kWarps * (lane / kLanes), i = lane % kLanes;
  const int at = __shfl_sync(0xffffffffu, run.at, list), n = __shfl_sync(0xffffffffu, run.n, list),
            to = __shfl_sync(0xffffffffu, run.to, list);
  const int trips =
      (static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<unsigned>(n))) + kLanes - 1)
      / kLanes;
  const uint2* src = l.entries + list_row(l, c, p.windows, ops::opaque(w), win.digit0) * cap
                     + (list * cap + at + i);
  ops::SmemAddr d = dst + 8u * static_cast<uint32_t>(to + i);
  const int left = n - i;
  //: the first two trips straight-line, any more in a loop
  const auto two = [&](int t) {
    ops::stage_run_lanes<8>(d, src, kLanes * t < left);
    ops::stage_run_lanes<8>(d + 8u * kLanes, src + kLanes, kLanes * (t + 1) < left);
    src += 2 * kLanes;
    d += 16u * kLanes;
  };
  two(0);
#pragma unroll 1
  for (int t = 2; t < trips; t += 2) two(t);
}

//: A CHUNK'S BUFFER, laid out at its own slots: a read buffer's rows, its num panel (a slot's channels at the reduce
//: dtype: `panel_row_bytes`), its den words (`kDenWords` fp32 a slot, one a readout channel group), its entries; a fold
//: buffer's rows, its entries. -- see docs/internals/carry/ampere/schedule.md#buffers
template <class P>
__host__ __device__ __forceinline__ constexpr int panel_row_bytes(int reduce) {
  return reduce == kReduceF32 ? P::kChannels * 4 : P::kChannels * 2;
}

template <class P>
__host__ __device__ constexpr int read_panel(int slots) {
  return align_of(slots * P::kReadRow, 128);
}

template <class P>
__host__ __device__ constexpr int read_dens(int slots, int reduce) {
  return read_panel<P>(slots) + align_of(slots * panel_row_bytes<P>(reduce), 128);
}

template <class P>
__host__ __device__ constexpr int read_ents(int slots, int reduce) {
  return read_dens<P>(slots, reduce) + align_of(slots * P::kDenRowBytes, 128);
}

template <class P>
__host__ __device__ constexpr int fold_ents(int slots) {
  return align_of(slots * P::kWriteRow, 128);
}

//: A READ CHUNK'S FILL, the copies issued: its rows at `buf`, its panel and den words cleared, its entries (a warp its
//: own digits' runs, `run` the chunk's). -- see docs/internals/carry/ampere/parts.md#fill
template <class P>
__device__ __forceinline__ void read_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& rw, const Place& c, int w, const Run& run,
                                          int first, int slots, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P, P::kReadFactors, false>(addr.slices[kRead], nullptr, 0, union_of<P>(p, smem, kRead, w),
                                  rw, first, slots, buf);
  //: the panel and the den words cleared by stores, four a thread a trip, the first straight-line
  constexpr int kAhead = 4;
  const int tid = ops::opaque(static_cast<int>(threadIdx.x));
  const auto clear = [&](int from, int to, uint32_t v) {
    const int quanta = (to - from) / 16 - tid;
    const ops::SmemAddr zero = buf + static_cast<uint32_t>(from) + 16u * static_cast<uint32_t>(tid);
    const auto four = [&](int k) {
      const ops::SmemAddr at = zero + 16u * static_cast<uint32_t>(k);
      static_for<kAhead>([&](auto j) { fill16<decltype(j)::value * kThreads>(at, quanta - k, v); });
    };
    four(0);
#pragma unroll 1
    for (int k = kAhead * kThreads; k < quanta; k += kAhead * kThreads) four(k);
  };
  clear(read_panel<P>(slots), read_ents<P>(slots, p.reduce), 0u);
  if (rw.listed)
    entries_fill<P, P::kDigits>(p, p.read_lists, rw, c, w, run,
                                buf + static_cast<uint32_t>(read_ents<P>(slots, p.reduce)), warp,
                                lane);
}

//: A LANE'S DEN WORDS' PLACES, each of the box's lists x rows once across the CTA, the list owner's on either side: lane
//: `(g, t)` its quad's (m-tile, list) pairs `kDenLists t ..` (in m-tile-major order), word `i` pair `i % kDenLists` at
//: row `g + 8 (i / kDenLists)` of it.
template <class P>
struct DenPlace {
  int list0, digit0;

  __device__ __forceinline__ int list(int i) const { return list0 + kWarps * (i % P::kDenLists); }

  __device__ __forceinline__ int digit(int i) const { return digit0 + 8 * (i / P::kDenLists); }
};

template <class P>
__device__ __forceinline__ DenPlace<P> den_place(int warp, int lane) {
  constexpr int kRun = P::kFoldLists / P::kDenLists;  // a lane run sharing an m-tile
  const int g = lane >> 2, t = lane & 3;
  return DenPlace<P>{warp + kWarps * P::kDenLists * (t % kRun), g + 16 * (t / kRun)};
}

}  // namespace rola::carry
