// csrc/rola/src/carry/ampere/readout.cuh -- THE READOUT, the value carve over a listed read side: a warp its token group's
// slots and its channel group's 8 channels over every digit of the box. see docs/internals/carry/ampere/readout.md
#pragma once

#include "carry/ampere/handoff.cuh"
#include "carry/ampere/panel.cuh"

namespace rola::carry {

//: A WARP'S PLACE IN THE READOUT: its token group (a chunk's half of the slots, `group_split`) and its channel group
//: (8 channels), the plan's (`P::kTokenGroups`, `P::kChannelGroups`). -- see docs/internals/carry/ampere/readout.md#warps
template <class P>
constexpr bool readout_box() {
  static_assert(P::kChannelGroups * P::kTokenGroups == kWarps
                    && (P::kChannelGroups == 4 || P::kChannelGroups == 8)
                    && P::kDenWords == P::kChannelGroups,
                "4 channel groups by 2 token groups or 8 by 1, a den word a channel group");
  return true;
}

//: AN f16 PAIR ADDED INTO A WORD, round to nearest.
__device__ __forceinline__ uint32_t add_h2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: A TILE'S TWO MMAs over the 32 lists into `acc`, accumulating in fp16 (`F16`) or fp32.
//: @burst
template <bool F16>
__device__ __forceinline__ void tile_mma(float (&acc)[4], const uint32_t (&a0)[4],
                                         const uint32_t (&a1)[4], const uint32_t (&b)[4]) {
  if constexpr (F16) {
    uint32_t h[2];
    ops::hmma_f16_fresh(h, a0, b[0], b[1]);
    ops::hmma_f16(h, a1, b[2], b[3]);
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[0])),
                 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
    acc[0] = lo.x;
    acc[1] = lo.y;
    acc[2] = hi.x;
    acc[3] = hi.y;
  } else {
    ops::hmma_fresh(acc, a0, b[0], b[1]);
    ops::hmma(acc, a1, b[2], b[3]);
  }
}

__device__ __forceinline__ float coef_of(uint32_t ey) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(ey & 0xffffu)));
}

//: A TOKEN'S SCALE `2^(-scale - 1)` from its byte, its exponent field built directly (no branch in the step).
__device__ __forceinline__ float token_scale(int sb) {
  return __uint_as_float(static_cast<uint32_t>(min(max(-sb - 1, -126), 127) + 127) << 23);
}

//: AN ENTRY'S KEY from its pool row's 16-byte chunk `c` (`4 s` and the row's swizzle): at 32 channels the row's byte
//: offset, the f16 panel row's base; at 64 the slot's f16 panel base in 16-byte units, `8 s + (s & 7)`.
//: -- see docs/internals/carry/ampere/readout.md#factors
template <class P>
__device__ __forceinline__ uint32_t entry_key(uint32_t c) {
  if constexpr (P::kChannels == 32)
    return c << 4;
  else
    return (c & ~3u) << 1 | ((c >> 2) & 7u);
}

//: A READ CHUNK'S ENTRY FACTORS, once a chunk: the warp's own digits' entries rewritten in place to their rows' byte
//: offsets, coefficients and num factors. -- see docs/internals/carry/ampere/readout.md#factors
template <class P>
__device__ __forceinline__ void read_factors(const CarryParams& p, const uint8_t* smem,
                                             const Run& run, int w, int first, ops::SmemAddr ents,
                                             int warp, int lane) {
  constexpr int kMine = P::kDigits / kWarps;
  static_assert(
      P::kReadRow == 64 && P::kWindow * P::kReadRow <= 65536,
      "a 64-byte pool row, its offset the f16 panel row's base (`panel_word`), under 2^16");
  const uint32_t back = 4u * static_cast<uint32_t>(first);
  __syncwarp();
  const ops::SmemAddr scale = ops::smem_addr(smem + p.tables.scale_at)
                              + static_cast<uint32_t>((w & 1) * p.tables.heads * P::kWindow);
  const ops::SmemAddr exps = exps_of<P>(smem, warp);
  const bool f32 = p.reduce == kReduceF32;

  //: the warp's digits' runs and exponents, then their entries 32 a trip, the four digits' chains side by side
  int n[kMine], to[kMine], most = 0;
  float dpow[kMine];
  static_for<kMine>([&](auto J) {
    const int d = warp + kWarps * decltype(J)::value;
    n[J] = __shfl_sync(0xffffffffu, run.n, d);
    to[J] = __shfl_sync(0xffffffffu, run.to, d);
    most = max(most, n[J]);
    dpow[J] = pow2_wide(
        static_cast<int>(ops::load_shared_u32(exps + 4u * static_cast<uint32_t>(exp_slot(d))))
        - kImageTop - kCoefTop);
  });
#pragma unroll 1
  for (int k = lane; k - lane < most; k += 32) {
    ops::SmemAddr at[kMine];
    uint2 e[kMine];
    bool live[kMine];
    static_for<kMine>([&](auto J) {
      live[J] = k < n[J];
      at[J] = ents + 8u * static_cast<uint32_t>(to[J] + k);
      e[J] = ops::load_shared_u64_if(at[J], live[J]);
    });
    static_for<kMine>([&](auto J) {
      //: a dead lane's place is 0, a byte of the table: read, never used
      const float c = coef_of(e[J].y);
      const float ts =
          token_scale(static_cast<int8_t>(ops::load_shared_u8(scale + (e[J].x >> 16))));
      const float f = f32 ? c * dpow[J] : c * dpow[J] * ts;
      ops::store_shared_u64_if(at[J],
                               entry_key<P>((e[J].x - back) & 0xfffu) | (e[J].y & 0xffffu) << 16,
                               __float_as_uint(f), live[J]);
    });
  }
}

//: A HALF: up to 8 entries of what remains of a digit's run, one word -- digit, place << 5, count << 21 (0 past the
//: last). -- see docs/internals/carry/ampere/readout.md#chunk
constexpr int kHalfAt = 5, kHalfCount = 21;

__device__ __forceinline__ int half_digit(uint32_t h) { return static_cast<int>(h & 31u); }

__device__ __forceinline__ int half_count(uint32_t h) { return static_cast<int>(h >> kHalfCount); }

__device__ __forceinline__ uint32_t half_place(uint32_t h) {
  return (h >> kHalfAt) & ((1u << (kHalfCount - kHalfAt)) - 1u);
}

//: A TILE: two halves, rows 0-7 the first's entries and rows 8-15 the second's, and its operands.
//: -- see docs/internals/carry/ampere/readout.md#chunk
struct Tile {
  uint32_t a0[4], a1[4], b[2][4], h[2];
  uint2 t[2];
  float k[2];
};

//: THE PASS'S PLACE: the warp's channel group and lanes, its token group's run a lane a digit (`run`: a half's word),
//: the buffers; the lane's slice row in a plane (`slice`) and the digit term's bits (`dbits`).
struct ReadLanes {
  ops::SmemAddr rows, ents, image, den_image, den_fac, panel, dens;
  uint32_t slice, dbits, word4, chan4, run;
  int cg, lane, g, tq;
};

//: DIGIT `d`'s SLICE ROW for the lane: one product, a mask and an XOR. -- see docs/internals/carry/ampere/readout.md#chunk
template <class P>
__device__ __forceinline__ uint32_t slice_at(const ReadLanes& r, int d) {
  static_assert(P::kPlaneBytes == P::kChannelGroups * P::kSliceBlock
                    && (P::kPlaneBytes & (P::kPlaneBytes - 1)) == 0
                    && 16 * P::kDigits <= P::kPlaneBytes,
                "a plane a power of two above the lane rows and the swizzle's bits");
  return r.image + (r.slice ^ ((static_cast<uint32_t>(d) * (P::kPlaneBytes + 16u)) & r.dbits));
}

//: AN ENTRY of a half, the `i`-th (the half's first where past its count: read, never added).
__device__ __forceinline__ ops::SmemAddr half_entry(const ReadLanes& r, uint32_t h, int i) {
  return r.ents + 8u * (half_place(h) + (i < half_count(h) ? i : 0));
}

template <class P, bool Den>
__device__ __forceinline__ void tile_in(Tile& t, const ReadLanes& r, uint32_t h0, uint32_t h1) {
  t.h[0] = h0;
  t.h[1] = h1;
  const uint32_t x = ops::load_shared_u32(half_entry(r, (r.lane & 8) ? h1 : h0, r.lane & 7));
  const ops::SmemAddr row0 = r.rows + (P::kChannels == 32 ? x & 0xffffu : (x << 3) & 0xfff0u);
  ops::load_frag(t.a0, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(r.lane >> 4)));
  ops::load_frag(t.a1, chunk_addr<P::kReadRow>(row0, 2u + static_cast<uint32_t>(r.lane >> 4)));
}

//: A TILE'S LATE OPERANDS, each half's slice (`Den`: den factor) and token entry, before the next tile's rows (not a tile
//: ahead: registers). -- see docs/internals/carry/ampere/readout.md#chunk
template <class P, bool Den>
__device__ __forceinline__ void tile_late(Tile& t, const ReadLanes& r) {
  const int d0 = half_digit(t.h[0]), d1 = half_digit(t.h[1]);
  ops::load_frag_t(t.b[0], slice_at<P>(r, d0));
  ops::load_frag_t(t.b[1], slice_at<P>(r, d1));
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    if constexpr (Den)
      t.k[j] = ops::load_shared_f32(r.den_fac + 4u * static_cast<uint32_t>(j ? d1 : d0));
    t.t[j] = ops::load_shared_u64(half_entry(r, t.h[j], r.g));
  }
}

//: A TILE'S MMAs (each half's slice; `Den`: the warp's den block once) and the lane's two tokens' adds, one half then the
//: other. -- see docs/internals/carry/ampere/readout.md#chunk
template <class P, bool F32, bool F16, bool Den>
__device__ __forceinline__ void tile_out(const Tile& t, const ReadLanes& r, const uint32_t (&bd)[4],
                                         bool valid) {
  float acc[2][4], dacc[4];
#pragma unroll
  for (int j = 0; j < 2; ++j) tile_mma<F16>(acc[j], t.a0, t.a1, t.b[j]);
  if constexpr (Den) tile_mma<F16>(dacc, t.a0, t.a1, bd);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    //: half `j`'s token is row `g + 8 j`, its pair `acc[j][2 j ..]`
    const uint32_t key = t.t[j].x & 0xffffu;
    const float f = __uint_as_float(t.t[j].y);
    const int d = half_digit(t.h[j]);
    const bool live = valid && r.g < half_count(t.h[j]);
    //: the den column the digit's (`den_col`; its low bit picks in the lane's pair), kept by the lanes holding it
    const bool mine = Den && live && r.tq == (den_col<P>(d) >> 1);
    ops::SmemAddr at, dat = 0u;
    uint2 old64 = {0u, 0u};
    uint32_t old32 = 0u;
    float dold = 0.0f;
    //: at 64 channels `word4` and `chan4` are the group (the panel's base past the lane's pair)
    if constexpr (F32) {
      if constexpr (P::kChannels == 32) {
        const uint32_t sl = key >> 6;
        at = r.panel + ((128u * sl + 16u * (sl & 7u)) ^ r.chan4);
      } else {
        at = r.panel + 32u * (key ^ r.chan4);
      }
      old64 = ops::load_shared_u64_if(at, live);
    } else {
      at = P::kChannels == 32 ? r.panel + (key ^ r.word4) : r.panel + 16u * (key ^ r.word4);
      old32 = ops::load_shared_u32_if(at, live);
    }
    if constexpr (Den) {
      dat = r.dens + (P::kChannels == 32 ? (key >> 2) & ~15u : (key << 2) & ~31u)
            + 4u * static_cast<uint32_t>(r.cg);
      dold = ops::load_shared_f32_if(dat, mine);
    }

    if constexpr (F32) {
      ops::store_shared_f32x2_if(at, __uint_as_float(old64.x) + acc[j][2 * j] * f,
                                 __uint_as_float(old64.y) + acc[j][2 * j + 1] * f, live);
    } else {
      const __half2 v = __floats2half2_rn(acc[j][2 * j] * f, acc[j][2 * j + 1] * f);
      ops::store_shared_u32_if(at, add_h2(old32, *reinterpret_cast<const uint32_t*>(&v)), live);
    }
    if constexpr (Den) {
      const float dn = (den_col<P>(d) & 1) ? dacc[2 * j + 1] : dacc[2 * j];
      ops::store_shared_f32_if(dat, dold + dn * (coef_of(t.t[j].x >> 16) * t.k[j]), mine);
    }
    //: the first half's adds before the second's loads: their tokens may be one slot
    if (j == 0) __syncwarp();
  }
}

//: A PASS over the halves of the digits `live` names, two a tile, two tiles a trip, a tile's rows a tile ahead, the trip's
//: one exit at its end (§20). -- see docs/internals/carry/ampere/readout.md#chunk
//: @burst-exempt a loop over the pass's tiles (`unroll 1`, its exit a shuffle), not yet `Burst` units
template <class P, bool F32, bool F16, bool Den>
__device__ __forceinline__ void read_pass(const ReadLanes& r, uint32_t live) {
  if (live == 0u) return;
  //: lane `d`'s next: the run of the next digit `live` names above `d`, else none (its own, count 0), once a pass
  const uint32_t above = live & ~((2u << r.lane) - 1u);
  const uint32_t up = __shfl_sync(0xffffffffu, r.run, __ffs(above | 1u << 31) - 1);
  const uint32_t next_h = above != 0u ? up : r.run & ((1u << kHalfCount) - 1u);
  //: the half after `h`: its digit's next 8 (the run advanced by 8), else lane `d`'s next (one shuffle)
  const auto succ = [&](uint32_t h) {
    const uint32_t n = __shfl_sync(0xffffffffu, next_h, half_digit(h));
    return 8 < half_count(h) ? h + (8u << kHalfAt) - (8u << kHalfCount) : n;
  };
  //: the tile after `t`: the two halves after its second; whether it has any
  const auto next = [&](const Tile& t, Tile& n) {
    const uint32_t h0 = succ(t.h[1]);
    tile_in<P, Den>(n, r, h0, succ(h0));
    return half_count(h0) != 0;
  };
  //: the warp's den block, every own digit's (`den_row`), once a pass
  uint32_t bd[4];
  if constexpr (Den)
    ops::load_frag_t(bd, r.den_image + den_row<P>(r.lane, r.cg / kDenBlockGroups<P>));
  Tile a, b;
  const uint32_t h0 = __shfl_sync(0xffffffffu, r.run, __ffs(live) - 1);
  tile_in<P, Den>(a, r, h0, succ(h0));
  bool go = true;
#pragma unroll 1
  while (go) {
    tile_late<P, Den>(a, r);
    const bool go_b = next(a, b);
    tile_out<P, F32, F16, Den>(a, r, bd, true);
    __syncwarp();
    tile_late<P, Den>(b, r);
    const bool go_a = next(b, a);
    tile_out<P, F32, F16, Den>(b, r, bd, go_b);
    __syncwarp();
    //: warp-uniform by construction, shuffled from lane 0 so ptxas sees it so (no divergence checks on the shuffles)
    go = __shfl_sync(0xffffffffu, go_b && go_a, 0);
  }
}

//: A WARP'S RUN WORD for digit `lane`, its token group's part as a half's word, once a chunk.
//: -- see docs/internals/carry/ampere/readout.md#chunk
template <class P>
__device__ __forceinline__ uint32_t read_run(const Run& run, int warp, int lane) {
  const int tg = P::kTokenGroups == 1 ? 0 : warp / P::kChannelGroups;
  const int count = P::kTokenGroups == 1 ? run.n
                    : tg                 ? run.n - run.n0
                                         : run.n0,
            start = run.to + (tg ? run.n0 : 0);
  return static_cast<uint32_t>(lane) | static_cast<uint32_t>(start) << kHalfAt
         | static_cast<uint32_t>(count) << kHalfCount;
}

//: THE READOUT of a chunk's slots over digits `lo .. hi - 1` of the image into the num panel and the den words, reduce
//: `F32`, accumulate `F16`. -- see docs/internals/carry/ampere/readout.md#chunk
template <class P, bool F32, bool F16>
__device__ __forceinline__ void read_chunk(const uint8_t* smem, uint32_t run, int lo, int hi,
                                           ops::SmemAddr image, uint32_t mask, ops::SmemAddr rows,
                                           ops::SmemAddr panel, ops::SmemAddr dens,
                                           ops::SmemAddr ents, int warp, int lane) {
  static_assert(P::kReadFactors == 1, "the value-carve readout of a listed read side");
  static_assert(P::kWindow < (1 << (32 - kHalfCount))
                    && P::kDigits * P::kWindow <= (1 << (kHalfCount - kHalfAt)),
                "a half's count and place in its word");
  static_assert(readout_box<P>());
  const int cg = warp % P::kChannelGroups, tq = lane & 3;
  constexpr bool kNarrow = P::kChannels == 32;
  //: at 64 channels the lane's pair past the panel's base, the group its key's XOR
  const ops::SmemAddr base = kNarrow ? panel : panel + static_cast<uint32_t>((F32 ? 8 : 4) * tq);
  const ReadLanes r{
      rows,
      ents,
      image,
      ops::smem_addr(smem + P::kDenImage),
      ops::smem_addr(smem + P::kDenFactor),
      base,
      dens,
      slice_row<P>(cg, lane, 0),
      mask * P::kPlaneBytes | 0x70u,
      kNarrow ? static_cast<uint32_t>(4 * (4 * cg + tq)) : static_cast<uint32_t>(cg),
      kNarrow ? static_cast<uint32_t>(4 * (8 * cg + 2 * tq)) : static_cast<uint32_t>(cg),
      run,
      cg,
      lane,
      lane >> 2,
      tq};
  const uint32_t live = __ballot_sync(0xffffffffu, half_count(run) > 0 && lane >= lo && lane < hi);
  //: the digits whose den the warp's channel group reads (`d mod 4` at 32 channels; at 64 its `kDenGroupDigits`
  //: contiguous ones, `den_block`) with their den, then the others: each digit once
  constexpr int kGroup = kDenGroupDigits<P>;
  constexpr uint32_t kOwn = kNarrow ? 0x11111111u : (1u << kGroup) - 1u;
  const uint32_t own = live & (kOwn << (kNarrow ? cg : kGroup * cg));
  read_pass<P, F32, F16, true>(r, own);
  read_pass<P, F32, F16, false>(r, live & ~own);
}

}  // namespace rola::carry
