// csrc/rola/src/carry/ampere/panel.cuh -- THE PANEL, a read chunk's num and den words, and THE DRAIN of both to HBM by the
// cross-CTA adds.
// see docs/internals/carry/ampere/panel.md
#pragma once

#include "carry/ampere/parts.cuh"

namespace rola::carry {

namespace ops = ::denseref::ops;

//: THE CROSS-CTA ADDS, unpredicated: f16 pairs (no flush to zero), one fp32.
__device__ __forceinline__ void red_add_h2(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void red_add_f32(float* at, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(at), "f"(v) : "memory");
}

//: A TOKEN'S ROW in `den` (and in `num`, in units of `dv`): its units index `u` (a query head's token in the window)
//: as the head's row and the window's token.
template <class P>
__device__ __forceinline__ uint32_t token_row(uint32_t u, uint32_t L) {
  return (u >> window_bits(P::kWindow) & static_cast<uint32_t>(window_heads(P::kWindow) - 1)) * L
         + (u & (P::kWindow - 1));
}

//: A WORD `off` words past `base` (`off` under 2^32), its own add: never re-associated into the base's sum.
template <class T>
__device__ __forceinline__ T* word_at(T* base, uint32_t off) {
  static_assert(sizeof(T) == 4, "a 32-bit word");
  uint64_t a;
  asm("mad.wide.u32 %0, %1, 4, %2;" : "=l"(a) : "r"(off), "l"(reinterpret_cast<uint64_t>(base)));
  return reinterpret_cast<T*>(a);
}

//: SLOT `s`'s NUM WORD `w` in the panel, bytes, rotated by the slot (an XOR): a 64-byte row's (32 f16 channels) by the
//: pool row's swizzle, a 128-byte row's by `16 (s & 7)`, a 256-byte row's by `32 (s & 7)`.
//: -- see docs/internals/carry/ampere/panel.md#words
template <int RowBytes>
__host__ __device__ __forceinline__ constexpr uint32_t panel_word(uint32_t s, uint32_t w) {
  static_assert(RowBytes == 64 || RowBytes == 128 || RowBytes == 256,
                "a 64-, 128- or 256-byte row");
  if constexpr (RowBytes == 64)
    return (64u * s + 16u * ((s >> 1) & 3u)) ^ (4u * w);
  else
    return (RowBytes * s + RowBytes / 8 * (s & 7u)) ^ (4u * w);
}

//: THE DRAIN: a chunk's panel to `num` and its den words to `den`, after every warp's adds; warp `w` slots `w + 8 i`, two
//: a trip, no branch around an add. -- see docs/internals/carry/ampere/panel.md#drain
template <class P, bool F32>
__device__ __forceinline__ void drain(const CarryParams& p, const uint16_t* union_tok, int first,
                                      int slots, ops::SmemAddr panel, ops::SmemAddr dens,
                                      int64_t head0, int channel0, int warp, int lane) {
  const uint32_t L = static_cast<uint32_t>(p.L), dv = static_cast<uint32_t>(p.dv);
  //: the chunk's slots that hold a token: the union's gaps are its tail inside its last step, so the chunk's last 8 (a
  //: dense read side's union is its window's slots, no gap: a slot its token, the fill's)
  constexpr bool kDense = P::kReadFactors == 2;
  const bool gap =
      !kDense && lane < 8 && lane < slots && (union_tok[first + slots - 1 - lane] & kGap) != 0u;
  const int n = slots - __popc(__ballot_sync(0xffffffffu, gap));

  //: a slot's row (`token_row`: a word offset under 2^32, the host's bound)
  const auto row_of = [&](int s) {
    return token_row<P>(kDense ? static_cast<uint16_t>(first + s) : union_tok[first + s], L);
  };
  //: a slot past `n` adds zero at the trip's first slot's row: a token's, never one row every CTA shares
  constexpr int kWords = P::kChannels / (F32 ? 1 : 2), kRowBytes = 4 * kWords;
  if constexpr (F32 || kWords == 32) {
    //: a warp a slot, a lane its word (and the word 32 past it at 64 fp32 channels)
    using T = std::conditional_t<F32, float, uint32_t>;
    T* const out = reinterpret_cast<T*>(p.num) + head0 * (p.dv / (F32 ? 1 : 2))
                   + channel0 / (F32 ? 1 : 2) + lane;
    const uint32_t row = static_cast<uint32_t>(p.dv / (F32 ? 1 : 2));
    const auto add = [&](int at, bool live) {
      static_for<kWords / 32>([&](auto k) {
        constexpr uint32_t kW = 32u * decltype(k)::value;
        if constexpr (F32) {
          const float v = ops::load_shared_f32(
              panel + panel_word<kRowBytes>(static_cast<uint32_t>(at), lane + kW));
          red_add_f32(word_at(out + kW, row_of(at) * row), live ? v : 0.0f);
        } else {
          const uint32_t v = ops::load_shared_u32(
              panel + panel_word<kRowBytes>(static_cast<uint32_t>(at), lane + kW));
          red_add_h2(word_at(out + kW, row_of(at) * row), live ? v : 0u);
        }
      });
    };
#pragma unroll 1
    for (int s = warp; s < n; s += 2 * kWarps) {
      add(s, true);
      add(s + kWarps < n ? s + kWarps : s, s + kWarps < n);
    }
  } else {
    const int half = lane >> 4, word = lane & 15;
    uint32_t* const out = p.num + head0 * (p.dv / 2) + channel0 / 2 + word;
    const auto add = [&](int at, bool live) {
      const uint32_t v =
          ops::load_shared_u32(panel + panel_word<64>(static_cast<uint32_t>(at), word));
      red_add_h2(word_at(out, row_of(at) * (dv / 2)), live ? v : 0u);
    };
#pragma unroll 1
    for (int pair = warp; 2 * pair < n; pair += 2 * kWarps) {
      const int s0 = 2 * pair + half, s1 = s0 + 2 * kWarps;
      add(s0 < n ? s0 : 2 * pair, s0 < n);
      add(s1 < n ? s1 : 2 * pair, s1 < n);
    }
  }

  //: THE DEN: a thread a slot, its words (a channel group's each) summed into one add
#pragma unroll 1
  for (int s = warp * 32 + lane; s < n; s += kThreads) {
    float sum[P::kDenWords / 4];
    static_for<P::kDenWords / 4>([&](auto k) {
      const uint4 d = ops::load_shared_vec16(dens + P::kDenRowBytes * static_cast<uint32_t>(s)
                                             + 16u * decltype(k)::value);
      sum[k] = (__uint_as_float(d.x) + __uint_as_float(d.y))
               + (__uint_as_float(d.z) + __uint_as_float(d.w));
    });
    float total = sum[0];
    static_for<P::kDenWords / 4 - 1>([&](auto k) { total += sum[k + 1]; });
    red_add_f32(word_at(p.den + head0, row_of(s)), total);
  }
}

}  // namespace rola::carry
