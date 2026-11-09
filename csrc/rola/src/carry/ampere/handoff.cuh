// csrc/rola/src/carry/ampere/handoff.cuh -- THE HAND-OFF, the fold's state to the readout's image: the digits' keys and
// exponents, the image's slices, a batch's stores and the den's tables, each in the sides' two forms.
// see docs/internals/carry/ampere/handoff.md
#pragma once

#include "carry/ampere/parts.cuh"

namespace rola::carry {

//: A WARP'S LISTS' UNITS, list `j` the box's list `warp + 8 j`.
template <class P>
__device__ __forceinline__ void list_units(int (&u)[P::kFoldLists], const uint8_t* smem, int warp) {
  //: read at each use (volatile): merged across a window's stores they would hold registers across it
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) { u[J] = book[warp + kWarps * J].unit; });
}

//: THE BOX'S TOP UNIT, its lists' largest, one reduction a warp (the book read at its use; a lane past the box's
//: lists a never-live unit).
template <class P>
__device__ __forceinline__ int box_top(const uint8_t* smem, int lane) {
  static_assert(P::kLists <= 32, "a lane a list");
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  if constexpr (P::kLists == 32)
    return __reduce_max_sync(0xffffffffu, book[lane].unit);
  else
    return __reduce_max_sync(0xffffffffu, lane < P::kLists ? book[lane].unit : kUnitNone);
}

//: THE KEYS' SPAN: a list more than this many units under the box's top keys as if this far (a live unit lies in
//: `kUnitMin .. 128`, a span under it; a never-live list's state is zero).
constexpr int kKeySpan = 255;

//: digit `d`'s place in `kExps` (lane order: `16 M + 8 H + g` at `4 g + 2 M + H`)
__host__ __device__ __forceinline__ constexpr int exp_slot(int d) { return 4 * (d & 7) + (d >> 3); }

//: THE DIGIT KEYS: a digit's largest |value| at its list's unit as an integer key, at window `w`'s parity at
//: `exp_slot`; `post_keys` returns the box's top. -- see docs/internals/carry/ampere/handoff.md#maxima
//: THE ALTERNATING FORM: a digit is a fold row across every warp's lists, its key one shared atomic max a lane.
template <class P>
__device__ __forceinline__ void post_row_keys(const State<P>& st, uint8_t* smem, int w, int top,
                                              int warp, int lane) {
  static_assert(P::kFoldLists == 4, "a quad's lane `t` keys its list `t`'s leaves");
  const int t = lane & 3;
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  const int shift = max(book[warp + kWarps * t].unit - top, -kKeySpan) * (1 << 23);
  int key[P::kFoldMTiles][2];
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<2>([&](auto H) {
      int bits[P::kFoldLists];
      static_for<P::kFoldLists>([&](auto J) {
        float x = fmaxf(fabsf(st.acc[J][M][0][2 * H]), fabsf(st.acc[J][M][0][2 * H + 1]));
        static_for<P::kFoldNTiles - 1>([&](auto N) {
          x = fmaxf(
              x, fmaxf(fabsf(st.acc[J][M][N + 1][2 * H]), fabsf(st.acc[J][M][N + 1][2 * H + 1])));
        });
        bits[J] = __float_as_int(x);
      });
      //: the quad's reduce-scatter of the bits: lane `t` ends with list `t`'s (lists `2 b + 0, 1` kept across `xor 2`,
      //: then one of them across `xor 1`), keyed; the digit's key their max across the quad
      const bool hi = t & 2, odd = t & 1;
      int keep0 = hi ? bits[2] : bits[0], keep1 = hi ? bits[3] : bits[1];
      keep0 = max(keep0, __shfl_xor_sync(0xffffffffu, hi ? bits[0] : bits[2], 2));
      keep1 = max(keep1, __shfl_xor_sync(0xffffffffu, hi ? bits[1] : bits[3], 2));
      const int mine_bits =
          max(odd ? keep1 : keep0, __shfl_xor_sync(0xffffffffu, odd ? keep0 : keep1, 1));
      const int mine_leaf = mine_bits == 0 ? INT_MIN : mine_bits + shift;
      int k = max(mine_leaf, __shfl_xor_sync(0xffffffffu, mine_leaf, 1));
      key[M][H] = max(k, __shfl_xor_sync(0xffffffffu, k, 2));
    });
  });

  //: lane `t` its row `2 M + H = t` (`exp_slot`: `4 g + t`), a lane's rows two a m-tile
  int* const keys = reinterpret_cast<int*>(smem + P::kKeys) + (w & 1) * kMaxExtent;
  if constexpr (P::kFoldMTiles == 2) {
    const int mine = t & 2 ? (t & 1 ? key[1][1] : key[1][0]) : (t & 1 ? key[0][1] : key[0][0]);
    atomicMax(keys + lane, mine);
  } else {
    if (t < 2) atomicMax(keys + lane, t & 1 ? key[0][1] : key[0][0]);
  }
}

//: THE SAME SPLIT'S KEYS: a digit is a fold list, every row of it this warp's -- its key the list's largest |value|
//: over the warp at its unit, lane `j` storing list `warp + 8 j`'s at its place (`exp_slot`), the one writer.
template <class P>
__device__ __forceinline__ void post_list_keys(const State<P>& st, uint8_t* smem, int w, int top,
                                               int warp, int lane) {
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  int mine = INT_MIN;
  static_for<P::kFoldLists>([&](auto J) {
    float x = 0.0f;
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<P::kFoldNTiles>(
          [&](auto N) { static_for<4>([&](auto E) { x = fmaxf(x, fabsf(st.acc[J][M][N][E])); }); });
    });
    const int bits = __reduce_max_sync(0xffffffffu, __float_as_int(x));
    const int shift = max(book[warp + kWarps * J].unit - top, -kKeySpan) * (1 << 23);
    mine = lane == J ? (bits == 0 ? INT_MIN : bits + shift) : mine;
  });
  const ops::SmemAddr keys =
      ops::smem_addr(smem + P::kKeys) + 4u * static_cast<uint32_t>((w & 1) * kMaxExtent);
  ops::store_shared_u32_if(keys + 4u * static_cast<uint32_t>(exp_slot(warp + kWarps * (lane & 3))),
                           static_cast<uint32_t>(mine), lane < P::kFoldLists);
}

template <class P>
__device__ __forceinline__ int post_keys(const State<P>& st, uint8_t* smem, int w, int warp,
                                         int lane) {
  const int top = box_top<P>(smem, lane);
  if constexpr (P::kSameSplit)
    post_list_keys<P>(st, smem, w, top, warp, lane);
  else
    post_row_keys<P>(st, smem, w, top, warp, lane);
  return top;
}

//: THE IMAGE'S TOP: a digit's largest value placed under 2^10, so a 32-list sum fits an fp16 accumulator.
//: -- see docs/internals/carry/ampere/handoff.md#maxima
constexpr int kImageTop = 10;

//: THE IMAGE, the value-carve readout's: digit `d`'s plane a slice a readout channel group `cg` -- 32 lists x 8 channels
//: f16, list `l`'s row 16 bytes at `l ^ (d & 7)`, so the hand-off's 4-byte stores and the readout's `ldmatrix.trans`
//: both meet 32 banks. -- see docs/internals/carry/ampere/handoff.md#image
template <class P>
__host__ __device__ __forceinline__ constexpr uint32_t slice_row(int cg, int l, int d) {
  return static_cast<uint32_t>(cg * P::kSliceBlock + 16 * (l ^ (d & 7)));
}

//: THE DEN IMAGE: blocks of 8 digit columns (the den MMA's N), K rows x 8 digits f16, digit `d` the den of channel group
//: `d mod 4` at 4 groups, of `d / kDenGroupDigits` at 8 (a group's digits one image half's): block `den_block`, column
//: `den_col`; K row `l`'s 16 bytes at `l` XOR a constant of its 8-row group and the block (the reader's rows a 128-byte
//: group; the writer's lanes on 16 banks, on 8 at the same split). -- see docs/internals/carry/ampere/handoff.md#den
template <class P>
constexpr int kDenGroupDigits = P::kDigits / P::kChannelGroups;

template <class P>
constexpr int kDenBlockGroups = 8 / kDenGroupDigits<P>;

template <class P>
__host__ __device__ __forceinline__ constexpr int den_block(int d) {
  if constexpr (P::kChannelGroups == 4)
    return d & 3;
  else
    return d >> 3;
}

template <class P>
__host__ __device__ __forceinline__ constexpr int den_col(int d) {
  if constexpr (P::kChannelGroups == 4)
    return d >> 2;
  else
    return d & 7;
}

template <class P>
__host__ __device__ __forceinline__ constexpr uint32_t den_row(int l, int block) {
  static_assert((P::kK == 32 || (P::kK == 16 && P::kChannelGroups == 8))
                    && kDenGroupDigits<P> * kDenBlockGroups<P> == 8
                    && (P::kChannelGroups == 4 || P::kChannelGroups == 8),
                "32 K rows (16 at 8 groups), a block 8 digit columns, 4 or 8 channel groups");
  if constexpr (P::kChannelGroups == 4)
    return static_cast<uint32_t>((block & 3) * P::kSliceBlock
                                 + 16 * (l ^ (((l >> 4) & 1) << 2 | (block & 3))));
  else
    return static_cast<uint32_t>(
        block * P::kSliceBlock
        + 16 * (l ^ (((l >> 4) & 1) << 2 | ((l >> 3) & 1) << 1 | (block & 1))));
}

//: THE DIGITS' EXPONENTS from the keys at window `w`'s parity, lane `i` digit `i`'s (`exp_slot`): the smallest `e`
//: with its largest |state| at most `2^e`, in the warp's own row of `kExps` (`exps_of`); warp 0 resets the other
//: parity's keys. -- see docs/internals/carry/ampere/handoff.md#maxima
template <class P>
__device__ __forceinline__ ops::SmemAddr exps_of(const uint8_t* smem, int warp) {
  return ops::smem_addr(smem + P::kExps) + 4u * static_cast<uint32_t>(kMaxExtent * warp);
}

template <class P>
__device__ __forceinline__ void post_exps(uint8_t* smem, int w, int top, int warp, int lane) {
  static_assert(P::kDigits <= kMaxExtent,
                "lane `i` slot `i` (past the digits unwritten), a parity's keys one bit apart");
  int* const keys = reinterpret_cast<int*>(smem + P::kKeys);
  const int at = (w & 1) * kMaxExtent + lane;
  const int key = keys[at];
  if (warp == 0) keys[at ^ kMaxExtent] = INT_MIN;
  reinterpret_cast<int*>(smem + P::kExps)[kMaxExtent * warp + lane] =
      max(((key + 0x7fffff) >> 23) + top - 127, kImageTop - 127);
  __syncwarp();
}

//: THE DEN'S TABLES, a window's, from the den's registers (the state before the window's writes): each digit's largest
//: by an integer max (`den_tables_max`), then each digit's factor and the den image (`den_tables_out`, past a barrier).
//: -- see docs/internals/carry/ampere/handoff.md#den
//: A DIGIT'S DEN EXPONENT `dE` from its largest den (`kDenMax`), `kImageTop - 127` where it has none.
__device__ __forceinline__ int den_exp(float top) {
  return top > 0.0f ? ceil_exp(top) : kImageTop - 127;
}

//: A DEN WORD'S READOUT DIGIT and its K, `I` compile-time: its fold row and list (the alternating form), its list and
//: row (the same split); `den_word_scale` the one of the lane's two digits word `I` is at.
template <class P, int I>
__device__ __forceinline__ int den_digit(const DenPlace<P>& at) {
  return P::kSameSplit ? at.list(I) : at.digit(I);
}

template <class P, int I>
__device__ __forceinline__ int den_k(const DenPlace<P>& at) {
  return P::kSameSplit ? at.digit(I) : at.list(I);
}

//: a lane's den digits: its pairs' lists (the same split) or its two rows, word `i`'s the `den_word_scale`-th
template <class P>
constexpr int kDenDigits = P::kSameSplit ? P::kDenLists : 2;

template <class P>
__host__ __device__ constexpr int den_word_scale(int i) {
  return P::kSameSplit ? i % P::kDenLists : i / P::kDenLists;
}

//: AN INTEGER MAX INTO A SHARED WORD, fire and forget (a non-negative float's bits order as integers).
__device__ __forceinline__ void red_max_u32(ops::SmemAddr at, uint32_t v) {
  asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

template <class P>
__device__ __forceinline__ void den_tables_max(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace<P> at = den_place<P>(warp, lane);
  const ops::SmemAddr tops = ops::smem_addr(smem + P::kDenMax);
  static_for<P::kDenRegs>([&](auto I) {
    red_max_u32(tops + 4u * static_cast<uint32_t>(den_digit<P, decltype(I)::value>(at)),
                __float_as_uint(st.den[I]));
  });
}

template <class P>
__device__ __forceinline__ void den_tables_out(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace<P> at = den_place<P>(warp, lane);
  const float* const tops = reinterpret_cast<const float*>(smem + P::kDenMax);
  //: a word at its digit's scale, at its place in the den image (`den_row`)
  float scale[kDenDigits<P>];
  static_for<kDenDigits<P>>([&](auto B) {
    constexpr int I = P::kSameSplit ? decltype(B)::value : P::kDenLists * decltype(B)::value;
    scale[B] = pow2(kImageTop - den_exp(tops[den_digit<P, I>(at)]));
  });
  static_for<P::kDenRegs>([&](auto I) {
    constexpr int i = decltype(I)::value;
    const int d = den_digit<P, i>(at);
    *reinterpret_cast<__half*>(smem + P::kDenImage + den_row<P>(den_k<P, i>(at), den_block<P>(d))
                               + 2 * den_col<P>(d)) =
        __float2half_rn(st.den[I] * scale[den_word_scale<P>(I)]);
  });
  ops::store_shared_f32_if(ops::smem_addr(smem + P::kDenFactor) + 4u * static_cast<uint32_t>(lane),
                           pow2_wide(den_exp(tops[lane]) - kImageTop), warp == 0);
}

//: A LIST'S SCALE for a digit, `2^(a - e)` held to a normal float's exponents: two integer multiply-adds (the FMA pipe;
//: `bits` loaded, so ptxas cannot fold them into an ALU shift) and two clamps.
__device__ __forceinline__ float list_scale(int a, int e, uint2 bits) {
  int x;
  asm("mad.lo.s32 %0, %1, -1, %2;" : "=r"(x) : "r"(e), "r"(a));
  x = min(max(x, -126), 127);
  uint32_t f;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(f) : "r"(x), "r"(bits.x), "r"(bits.y));
  return __uint_as_float(f);
}

//: A LANE'S HAND-OFF OPERANDS: its base (opaque: formed at each call, not hoisted across the windows), four exponents
//: of its warp's row from `16 i`, the bits `list_scale` takes, its lists' units at the image's top.
struct StoreFrame {
  uint32_t at;
  int4 ex;
  uint2 bits;
};

template <class P>
__device__ __forceinline__ StoreFrame store_frame(const uint8_t* smem, uint32_t base, int i,
                                                  int (&a)[P::kFoldLists], int warp) {
  StoreFrame f;
  asm volatile("mov.u32 %0, %1;" : "=r"(f.at) : "r"(base));
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(f.ex.x), "=r"(f.ex.y), "=r"(f.ex.z), "=r"(f.ex.w)
               : "r"(exps_of<P>(smem, warp) + 16u * static_cast<uint32_t>(i))
               : "memory");
  const volatile uint32_t* const bp = reinterpret_cast<const volatile uint32_t*>(smem + P::kBits);
  f.bits = make_uint2(bp[0], bp[1]);
  list_units<P>(a, smem, warp);
  static_for<P::kFoldLists>([&](auto J) { a[J] += kImageTop; });
  return f;
}

//: A BATCH'S STORES: the warp's state of batches `lo .. hi - 1` (8 digits each) into the image (plane `d & Mask`), f16
//: at each digit's power of two (`list_scale`), a lane's channel pair one 4-byte store at its held base and an
//: immediate. -- see docs/internals/carry/ampere/handoff.md#store
//: THE ALTERNATING FORM: a digit is a fold row, batch `2 M + H` its rows `16 M + 8 H + g`, list `warp + 8 J` the
//: digit's K row `(warp ^ g) + 8 J` (digit `g`'s swizzle).
template <class P, uint32_t Mask>
__device__ __forceinline__ void store_row_batches(const State<P>& st, uint8_t* smem,
                                                  ops::SmemAddr image, int lo, int hi, int warp,
                                                  int lane) {
  const int g = lane >> 2, t = lane & 3;
  int a[P::kFoldLists];
  const StoreFrame fr =
      store_frame<P>(smem,
                     image + static_cast<uint32_t>(g) * P::kPlaneBytes + slice_row<P>(0, warp, g)
                         + 4u * static_cast<uint32_t>(t),
                     g, a, warp);
  const uint32_t at = fr.at;
  const uint2 bits = fr.bits;
  const int es[2][2] = {{fr.ex.x, fr.ex.y}, {fr.ex.z, fr.ex.w}};
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<2>([&](auto H) {
      constexpr int b = 2 * M + H;
      if (b >= lo && b < hi) {
        static_for<P::kFoldLists>([&](auto J) {
          const float scale = list_scale(a[J], es[M][H], bits);
          static_for<P::kFoldNTiles>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            const uint32_t to =
                at + ((16u * M + 8u * H) & Mask) * P::kPlaneBytes + P::kSliceBlock * N + 128u * J;
            ops::store_shared_u32(to, *reinterpret_cast<const uint32_t*>(&h));
          });
        });
      }
    });
  });
}

//: THE SAME SPLIT: a digit is a fold list, batch `J` list `warp + 8 J`, its rows the digit's K -- row `16 M + 8 H + g`
//: at `16 M + 8 H + (g ^ warp)` (the list's swizzle, `warp`).
template <class P, uint32_t Mask>
__device__ __forceinline__ void store_list_batches(const State<P>& st, uint8_t* smem,
                                                   ops::SmemAddr image, int lo, int hi, int warp,
                                                   int lane) {
  const int g = lane >> 2, t = lane & 3;
  int a[P::kFoldLists];
  const StoreFrame fr =
      store_frame<P>(smem,
                     image + static_cast<uint32_t>(warp) * P::kPlaneBytes + slice_row<P>(0, g, warp)
                         + 4u * static_cast<uint32_t>(t),
                     warp, a, warp);
  const int es[4] = {fr.ex.x, fr.ex.y, fr.ex.z, fr.ex.w};
  static_for<P::kFoldLists>([&](auto J) {
    if (J >= lo && J < hi) {
      const float scale = list_scale(a[J], es[J], fr.bits);
      static_for<P::kFoldMTiles>([&](auto M) {
        static_for<2>([&](auto H) {
          static_for<P::kFoldNTiles>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            const uint32_t to = fr.at + ((8u * J) & Mask) * P::kPlaneBytes + P::kSliceBlock * N
                                + 16u * (16u * M + 8u * H);
            ops::store_shared_u32(to, *reinterpret_cast<const uint32_t*>(&h));
          });
        });
      });
    }
  });
}

template <class P, uint32_t Mask>
__device__ __forceinline__ void store_batches(const State<P>& st, uint8_t* smem,
                                              ops::SmemAddr image, int lo, int hi, int warp,
                                              int lane) {
  if constexpr (P::kSameSplit)
    store_list_batches<P, Mask>(st, smem, image, lo, hi, warp, lane);
  else
    store_row_batches<P, Mask>(st, smem, image, lo, hi, warp, lane);
}

}  // namespace rola::carry
