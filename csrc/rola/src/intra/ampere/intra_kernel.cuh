// csrc/rola/src/intra/ampere/intra_kernel.cuh -- THE INTRA KERNEL: a CTA a (query head, window, piece of a block's sweep),
// the block's reader rows resident in shared memory, the writer tiles streamed through three stage buffers, each refilled
// by its last reader, under the slab gate the facts pass's unions give; each level's Gram as f16 MMAs on the rows as
// stored (a u8 digit widened in registers), the epilogue into the value product's f16 weights, the value product on the
// f16 values, and the block's rows reduced into the output planes at the end; a call that is not a whole number of
// windows runs the arms with every row past its end bounded (`kEven` false).
// see docs/internals/intra/ampere/intra_kernel.md
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/static_for.cuh"
#include "common/structure_switch.cuh"
#include "facts/liveness_contract.cuh"
#include "intra/intra.cuh"

namespace rola::intra::detail {

namespace ops = denseref::ops;
namespace arch = ::denseref::arch;

using Clock = PhaseClock<kIntraPhases, true>;

//: THE SHARED TENANTS at (W, block rows): three writer stages, the tile's values and writer factors, the warps' row
//: sums, the slab gate's table and the pipeline's mbarriers; the block's reader rows follow, `pitch` bytes a row.
//: -- see docs/internals/intra/ampere/intra_kernel.md#smem
template <int W, int Rows, int Warps>
struct alignas(128) SmemAt {
  uint8_t stage[kStageBuffers][kTile][kStageBytes];
  uint8_t values[kTile][kValueBytes];
  float2 fm[kTile];
  float den[Warps][kWarpRows];
  uint32_t gate[W / kTile][Rows / kTile];
  uint64_t values_full;
  int values_released[2];
  int4 words[kMaxStages];
  long long clock[Warps][Clock::kRow];
};

//: THE WALK at build `B`: its window, its block of reader rows -- the most the card's area holds at the widest rows,
//: a multiple of the tile and at most a row group a warp -- a warp 16 of them, a CTA's warps two a scheduler.
//: -- see docs/internals/intra/intra.md#block
template <arch::Build B>
struct Walk {
  static constexpr int kWindow = arch::traits_of(B).window;
  static constexpr int kWarps = 8;
  //: a reader row at the widest levels: every level its widest, f16 a side
  static constexpr int kWidestPitch = kMaxLevels * kMaxLevelWidth * 2;
  template <int Rows>
  static constexpr bool kFits = sizeof(SmemAt<kWindow, Rows, kWarps>) + Rows * kWidestPitch
                                <= arch::traits_of(B).caps.smem_per_cta_max;
  static constexpr int kBlockRows = kFits<kWarps * kWarpRows> ? kWarps * kWarpRows : kTile;
  static constexpr int kWindowTiles = kWindow / kTile;
  static constexpr int kRowTiles = kBlockRows / kTile;
  static constexpr int kRowWarps = kBlockRows / kWarpRows;
  static constexpr int kColWarps = kWarps / kRowWarps;
  static constexpr int kWarpCols = kTile / kColWarps;
  static constexpr int kThreads = kWarps * 32;
  //: a warp's n-tiles (8 writers each), their pairs (an ldmatrix.x4's two), a tile's channel pairs and union quads
  static constexpr int kNT = kWarpCols / 8;
  static constexpr int kNTPairs = kNT / 2;
  static constexpr int kChannelPairs = kValueWidth / 16;
  static constexpr int kWindowQuads = kWindowTiles / 4;
  using Pieces = PiecesAt<kWindow, kBlockRows>;
  using Smem = SmemAt<kWindow, kBlockRows, kWarps>;
  static_assert(kWindow > 0 && kFits<kBlockRows> && kWarps <= kLedgerWarps,
                "a built card's window chosen, its block in its area, its warps in the ledger");
};

//: THE WINDOW'S PIECES where a CTA reads its own: one uniform constant load
template <class Wk>
__constant__ typename Wk::Pieces kPieceTable = typename Wk::Pieces{};

//: A LEVEL'S K ORDER: f16 on both sides the MMA's own (ldmatrix both); with a u8 side the u8 fragment's, a lane's four
//: consecutive digits a row (slot 2t, 2t + 1 <- digits 4t, 4t + 1; slot 2t + 8, 2t + 9 <- 4t + 2, 4t + 3), the f16
//: side read in it. -- see docs/internals/intra/ampere/intra_kernel.md#gram
template <int F>
constexpr bool kNatural = form_read_f16(F) && form_write_f16(F);

//: THE INTRA AT A WALK: its device code over the walk `Wk`'s sizes. -- see docs/internals/intra/ampere/intra_kernel.md#walk
template <class Wk>
struct IntraAt {
  static constexpr int kWindow = Wk::kWindow, kBlockRows = Wk::kBlockRows,
                       kWindowTiles = Wk::kWindowTiles;
  static constexpr int kRowTiles = Wk::kRowTiles, kWarpCols = Wk::kWarpCols,
                       kColWarps = Wk::kColWarps;
  static constexpr int kRowWarps = Wk::kRowWarps, kWarps = Wk::kWarps, kThreads = Wk::kThreads;
  static constexpr int kNT = Wk::kNT, kNTPairs = Wk::kNTPairs, kChannelPairs = Wk::kChannelPairs;
  static constexpr int kWindowQuads = Wk::kWindowQuads;
  using Pieces = typename Wk::Pieces;
  using Smem = typename Wk::Smem;

  static __device__ __forceinline__ uint8_t* reader_rows(Smem& sm) {
    return reinterpret_cast<uint8_t*>(&sm + 1);
  }

  //: A ROW'S 16-BYTE CHUNK `c` lies at chunk `c ^ swizzle(row)`: eight rows at one chunk in eight bank groups (ldmatrix),
  //: four rows' 32-byte pairs in four (the pair reads). -- see docs/internals/intra/ampere/intra_kernel.md#swizzle
  static __device__ __forceinline__ int swizzle(int row) {
    return ((row & 3) << 1) | ((row >> 2) & 1);
  }

  static __device__ __forceinline__ const uint8_t* row_at(const RowView& v, int b, int t, int h) {
    return static_cast<const uint8_t*>(v.base) + b * v.sb + h * v.sh
           + static_cast<int64_t>(t) * v.st;
  }

  //: `rows` token strides of `st` bytes: one unsigned 32 x 32 -> 64 multiply (the seam bounds a stride to 31 bits)
  static __device__ __forceinline__ uint64_t rows_of(int rows, int64_t st) {
    return static_cast<uint64_t>(static_cast<uint32_t>(rows)) * static_cast<uint32_t>(st);
  }

  //: A u8 WORD'S FOUR DIGITS as two f16 pairs, (0, 1) and (2, 3), exact: at fp16 accumulate `d / 16` (the byte under
  //: 0x54 is `64 + d / 16`, less 64), at fp32 the f16 subnormal `d 2^-24` (the byte alone).
  //: -- see docs/internals/intra/ampere/intra_kernel.md#widening
  template <bool kF16>
  static __device__ __forceinline__ void widen(uint32_t x, uint32_t& lo, uint32_t& hi) {
    if constexpr (kF16) {
      constexpr uint32_t kExponent = 0x54545454u, kOne = 0x3C003C00u, kLess64 = 0xD400D400u;
      lo = ops::hfma2(ops::prmt(x, kExponent, 0x4140u), kOne, kLess64);
      hi = ops::hfma2(ops::prmt(x, kExponent, 0x4342u), kOne, kLess64);
    } else {
      lo = ops::prmt(x, 0u, 0x4140u);
      hi = ops::prmt(x, 0u, 0x4342u);
    }
  }

  //: THE WARP'S STATE ACROSS A PIECE: a level's Gram (f16 pairs or fp32; s32 at a u8 x u8 level, in the same registers),
  //: the running level product and the output
  template <bool kF16, int = 0>
  struct Acc;

  template <int X>
  struct Acc<true, X> {
    union {
      uint32_t gram[kNT][2];
      int32_t igram[kNT][4];
    };

    float product[kNT][4];
    float out[8][4];
  };

  template <int X>
  struct Acc<false, X> {
    union {
      float gram[kNT][4];
      int32_t igram[kNT][4];
    };

    float product[kNT][4];
    float out[8][4];
  };

  //: @burst
  template <bool kF16>
  static __device__ __forceinline__ void gram_mma(Acc<kF16>& acc, int n, const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
    if constexpr (kF16)
      ops::hmma_f16(acc.gram[n], a, b0, b1);
    else
      ops::hmma(acc.gram[n], a, b0, b1);
  }

  //: an n-tile's Gram as fp32, rows g (0, 1) and g + 8 (2, 3), from the level form `F`'s accumulator
  template <int F, bool kF16>
  static __device__ __forceinline__ void gram_values(const Acc<kF16>& acc, int n, float (&v)[4]) {
    if constexpr (F == kFormU8U8) {
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = __int2float_rn(acc.igram[n][i]);
    } else if constexpr (kF16) {
      const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&acc.gram[n][0]));
      const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&acc.gram[n][1]));
      v[0] = lo.x, v[1] = lo.y, v[2] = hi.x, v[3] = hi.y;
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = acc.gram[n][i];
    }
  }

  template <int F, bool kF16>
  static __device__ __forceinline__ void zero_gram(Acc<kF16>& acc) {
    if constexpr (F == kFormU8U8) {
#pragma unroll
      for (int n = 0; n < kNT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc.igram[n][i] = 0;
    } else {
      constexpr int kWords = kF16 ? 2 : 4;
#pragma unroll
      for (int n = 0; n < kNT; ++n)
#pragma unroll
        for (int i = 0; i < kWords; ++i) acc.gram[n][i] = 0;
    }
  }

  //: A LANE'S FIXED PLACES in the operand loads: its reader rows and the writer tokens it addresses, with their swizzles.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#gram
  struct Lanes {
    uint32_t a_row, f_row, f_row8, b_row, g_row;
    int a_hi, a_swz, f_h, f_swz;
    uint32_t f_off, b_q16, g_q16;

    __device__ __forceinline__ Lanes(Smem& sm, const IntraParams& p, int group, int col0,
                                     int lane) {
      const int g = lane >> 2, tq = lane & 3;
      const uint32_t rows = ops::smem_addr(reader_rows(sm));
      a_row = rows + static_cast<uint32_t>((group * kWarpRows + (lane & 15)) * p.pitch);
      a_hi = lane >> 4;
      a_swz = swizzle(lane & 15);
      f_row = rows + static_cast<uint32_t>((group * kWarpRows + g) * p.pitch);
      f_row8 = f_row + static_cast<uint32_t>(8 * p.pitch);
      f_h = tq >> 1;
      f_swz = swizzle(g);
      f_off = static_cast<uint32_t>(8 * (tq & 1));
      const int brow = (lane & 7) + ((lane >> 4) << 3);
      b_row = static_cast<uint32_t>((col0 + brow) * kStageBytes);
      b_q16 = static_cast<uint32_t>((((lane >> 3) & 1) ^ swizzle(lane & 7)) << 4);
      g_row = static_cast<uint32_t>((col0 + g) * kStageBytes + 8 * (tq & 1));
      g_q16 = static_cast<uint32_t>(((tq >> 1) ^ swizzle(g)) << 4);
    }
  };

  //: ONE SLAB'S OPERANDS as loaded (two k-steps of 16 digits): the warp's 16 readers' A words and the stage's 64 writer
  //: tokens' B words, a u8 operand still four digits a word.
  template <int F>
  struct SlabFrags {
    uint32_t a[form_read_f16(F) ? 8 : 4];
    uint32_t b[form_write_f16(F) ? 4 * kNT : 2 * kNT];
  };

  //: A SLAB'S LOADS, each landing in the registers its MMA reads: ldmatrix for a u8 operand and for both of a natural
  //: level; for the f16 reader of a u8 writer a 64-bit read a row (g, g + 8) a k-step, for the f16 writer of a u8 reader
  //: one an n-tile -- a half-warp's pair reads one wavefront. -- see docs/internals/intra/ampere/intra_kernel.md#gram
  template <int F>
  static __device__ __forceinline__ void load_slab(const Lanes& ln, int rc, uint32_t buf,
                                                   SlabFrags<F>& fr, int S) {
    if constexpr (!form_read_f16(F)) {
      ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.a[0]),
                     ln.a_row + static_cast<uint32_t>(((rc + 2 * S + ln.a_hi) ^ ln.a_swz) << 4));
    } else if constexpr (kNatural<F>) {
#pragma unroll
      for (int k = 0; k < 2; ++k)
        ops::load_frag(
            *reinterpret_cast<uint32_t(*)[4]>(&fr.a[4 * k]),
            ln.a_row + static_cast<uint32_t>(((rc + 2 * (2 * S + k) + ln.a_hi) ^ ln.a_swz) << 4));
    } else {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint32_t at =
            static_cast<uint32_t>(((rc + 2 * (2 * S + k) + ln.f_h) ^ ln.f_swz) << 4) + ln.f_off;
        const uint2 lo = ops::load_shared_u64(ln.f_row + at),
                    hi = ops::load_shared_u64(ln.f_row8 + at);
        fr.a[4 * k] = lo.x, fr.a[4 * k + 2] = lo.y;
        fr.a[4 * k + 1] = hi.x, fr.a[4 * k + 3] = hi.y;
      }
    }

    if constexpr (!form_write_f16(F)) {
#pragma unroll
      for (int j = 0; j < kNTPairs; ++j)
        ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.b[4 * j]),
                       buf + ln.b_row + (ln.b_q16 ^ (S << 5)) + j * 16 * kStageBytes);
    } else if constexpr (kNatural<F>) {
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int j = 0; j < kNTPairs; ++j)
          ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.b[2 * kNT * k + 4 * j]),
                         buf + ln.b_row + (ln.b_q16 ^ ((2 * S + k) << 5)) + j * 16 * kStageBytes);
    } else {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint32_t at = buf + ln.g_row + (ln.g_q16 ^ ((2 * S + k) << 5));
#pragma unroll
        for (int n = 0; n < kNT; ++n) {
          const uint2 b = ops::load_shared_u64(at + n * 8 * kStageBytes);
          fr.b[2 * (kNT * k + n)] = b.x, fr.b[2 * (kNT * k + n) + 1] = b.y;
        }
      }
    }
  }

  //: A SLAB'S MMAs: an IMMA an n-tile at a u8 x u8 level; otherwise 16, a k-step's eight n-tiles apart before the next
  //: k-step's, u8 words widened as they are used
  //: @burst
  template <int F, bool kF16>
  static __device__ __forceinline__ void mma_slab(const SlabFrags<F>& fr, Acc<kF16>& acc) {
    //: A u8 x u8 LEVEL: one IMMA an n-tile, the slab's 32 digits its K, exact into s32 -- no widening
    if constexpr (F == kFormU8U8) {
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        const int w = 4 * (n >> 1) + 2 * (n & 1);
        ops::imma_u8_k32(acc.igram[n], fr.a, fr.b[w], fr.b[w + 1]);
      }
      return;
    }

    //: THE FLOOR TRICK, an f16 reader against a u8 writer: the writer's digits the free subnormals `d 2^-24`, the
    //: reader's floored f16 (>= 2^-13 of its row's max) times 2^15 into the A registers in order, a product `a d 2^-9`
    constexpr bool kFloored = F == kFormF16U8;
    constexpr uint32_t kTwo15 = 0x78007800u;
    uint32_t a[2][4];
    if constexpr (!form_read_f16(F)) {
      widen<kF16>(fr.a[0], a[0][0], a[0][2]);
      widen<kF16>(fr.a[1], a[0][1], a[0][3]);
      widen<kF16>(fr.a[2], a[1][0], a[1][2]);
      widen<kF16>(fr.a[3], a[1][1], a[1][3]);
    } else {
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          a[k][i] = kFloored ? ops::hmul2(fr.a[4 * k + i], kTwo15) : fr.a[4 * k + i];
    }
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        if constexpr (!form_write_f16(F)) {
          uint32_t lo, hi;
          widen<kF16 && !kFloored>(fr.b[4 * (n >> 1) + 2 * (n & 1) + k], lo, hi);
          gram_mma(acc, n, a[k], lo, hi);
        } else if constexpr (kNatural<F>) {
          //: ldmatrix's quads: an n-tile pair's (k 0-7, k 8-15) words for its first then its second n-tile
          const int w = 2 * kNT * k + 4 * (n >> 1) + 2 * (n & 1);
          gram_mma(acc, n, a[k], fr.b[w], fr.b[w + 1]);
        } else {
          gram_mma(acc, n, a[k], fr.b[2 * (kNT * k + n)], fr.b[2 * (kNT * k + n) + 1]);
        }
      }
  }

  //: THE GRAM OF ONE STAGE, every slab of it live: one straight-line block, each slab's loads issued before the slab
  //: ahead of it multiplies, the ring's refill a half after each half of the slabs (`refill(half)`).
  //: -- see docs/internals/intra/ampere/intra_kernel.md#gram
  template <int F, bool kF16, typename Refill>
  static __device__ __forceinline__ void gram_dense(const Lanes& ln, int rc, uint32_t buf,
                                                    Acc<kF16>& acc, const Refill& refill) {
    constexpr int kSlabs = form_stage_digits(F) / kSlabDigits;
    SlabFrags<F> fr[2];
    load_slab<F>(ln, rc, buf, fr[0], 0);
    rola::static_for<kSlabs>([&](auto S) {
      constexpr int kS = decltype(S)::value;
      if constexpr (kS + 1 < kSlabs) load_slab<F>(ln, rc, buf, fr[(kS + 1) & 1], kS + 1);
      mma_slab<F, kF16>(fr[kS & 1], acc);
      constexpr int kHalf = kSlabs / 2;
      if constexpr ((kS + 1) % kHalf == 0)
        refill(std::integral_constant<int, (kS + 1) / kHalf - 1>{});
    });
  }

  //: THE GRAM OF ONE STAGE under the warp's slab bits: a loop over its live slabs two at a time -- each slab's loads and
  //: widening under the other's MMAs, the step branch-free -- and the odd slab peeled after it (KERNEL_STANDARDS
  //: section 27.5). -- see docs/internals/intra/ampere/intra_kernel.md#gram
  template <int F, bool kF16>
  static __device__ __forceinline__ void gram_slabs(const Lanes& ln, int rc, uint32_t buf,
                                                    uint32_t slabs, Acc<kF16>& acc) {
    uint32_t m = slabs;
#pragma unroll 1
    for (int n = __popc(slabs) >> 1; n > 0; --n) {
      const int a = __ffs(m) - 1;
      m &= m - 1u;
      const int b = __ffs(m) - 1;
      m &= m - 1u;
      SlabFrags<F> fa, fb;
      load_slab<F>(ln, rc, buf, fa, a);
      load_slab<F>(ln, rc, buf, fb, b);
      mma_slab<F, kF16>(fa, acc);
      mma_slab<F, kF16>(fb, acc);
    }
    if (m != 0u) {
      SlabFrags<F> fa;
      load_slab<F>(ln, rc, buf, fa, __ffs(m) - 1);
      mma_slab<F, kF16>(fa, acc);
    }
  }

  //: A LEVEL'S GRAM INTO THE RUNNING PRODUCT: the first level lifted to digit units (an exact power of two), each later
  //: one multiplied in.
  template <int F, bool kF16>
  static __device__ __forceinline__ void fold_level(Acc<kF16>& acc, bool first, float lift) {
    if (first) {
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        float v[4];
        gram_values<F, kF16>(acc, n, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc.product[n][i] = v[i] * lift;
      }
    } else {
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        float v[4];
        gram_values<F, kF16>(acc, n, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc.product[n][i] *= v[i];
      }
    }
  }

  //: THE EPILOGUE: the level product times each writer's factors -- `m` into the row sums `den`, `m s` into the value
  //: product's weights -- zero above the diagonal on a diagonal tile. -- see docs/internals/intra/ampere/intra_kernel.md#epilogue
  template <bool kF16>
  static __device__ __forceinline__ void epilogue(Smem& sm, float unit, bool diagonal, int warp,
                                                  int group, int col0, int lane, Acc<kF16>& acc) {
    const int g = lane >> 2, tq = lane & 3;
    const int row0 = (group % (kRowWarps / kRowTiles)) * kWarpRows + g;
    float den[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      const uint4 w = ops::load_shared_v4(ops::smem_addr(&sm.fm[col0 + 8 * n + 2 * tq]));
      const float fs[2] = {__uint_as_float(w.x), __uint_as_float(w.z)};
      const float m[2] = {__uint_as_float(w.y), __uint_as_float(w.w)};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int col = col0 + 8 * n + 2 * tq + (i & 1), row = row0 + 8 * (i >> 1);
        const float pr = diagonal && col > row ? 0.0f : acc.product[n][i];
        den[i >> 1] += pr * m[i & 1];
        acc.product[n][i] = pr * fs[i & 1];
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      den[r] += __shfl_xor_sync(0xffffffffu, den[r], 1);
      den[r] += __shfl_xor_sync(0xffffffffu, den[r], 2);
    }
    if (tq == 0) {
      sm.den[warp][g] += den[0] * unit;
      sm.den[warp][g + 8] += den[1] * unit;
    }
  }

  //: THE VALUE PRODUCT: each reader's weights at its own power of two (its largest in [1, 2)) as f16, packed from the
  //: accumulator's registers into the A fragments; the values by ldmatrix.trans; a channel pair's sums back at the
  //: reader's power of two into the output. -- see docs/internals/intra/ampere/intra_kernel.md#value-product
  //: @burst
  template <bool kF16>
  static __device__ __forceinline__ void value_product(const Smem& sm, float unit, int col0,
                                                       int lane, Acc<kF16>& acc) {
    constexpr int kSteps = kNTPairs;
    float m[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < kNT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) m[i >> 1] = fmaxf(m[i >> 1], acc.product[n][i]);
    float scale[2], undo[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
      m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
      const uint32_t e = max((__float_as_uint(m[r]) >> 23) & 0xffu, 1u);
      scale[r] = __uint_as_float((254u - e) << 23);
      undo[r] = __uint_as_float(e << 23) * unit;
    }

    uint32_t a[kSteps][4];
#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float(&c)[4] = acc.product[2 * kk + half];
        a[kk][2 * half] = ops::pack_half2(c[0] * scale[0], c[1] * scale[0]);
        a[kk][2 * half + 1] = ops::pack_half2(c[2] * scale[1], c[3] * scale[1]);
      }

    const uint32_t values = ops::smem_addr(&sm.values[0][0]);
    const int token = col0 + (lane & 15);
#pragma unroll
    for (int jc = 0; jc < kChannelPairs; ++jc) {
      std::conditional_t<kF16, uint32_t[2], float[4]> sums[2] = {};
#pragma unroll
      for (int kk = 0; kk < kSteps; ++kk) {
        const int t = 16 * kk + token;
        uint32_t v[4];
        ops::load_frag_t(v,
                         values + t * kValueBytes + (((2 * jc + (lane >> 4)) ^ swizzle(t)) << 4));
        if constexpr (kF16) {
          ops::hmma_f16(sums[0], a[kk], v[0], v[1]);
          ops::hmma_f16(sums[1], a[kk], v[2], v[3]);
        } else {
          ops::hmma(sums[0], a[kk], v[0], v[1]);
          ops::hmma(sums[1], a[kk], v[2], v[3]);
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float s[4];
        if constexpr (kF16) {
          const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sums[e][0]));
          const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sums[e][1]));
          s[0] = lo.x, s[1] = lo.y, s[2] = hi.x, s[3] = hi.y;
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) s[i] = sums[e][i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) acc.out[2 * jc + e][i] += s[i] * undo[i >> 1];
      }
    }
  }

  //: A WARP'S COPIES: one warp issues a tile's values, a lane's place fixed (row `lane / 8 + 4 i`, chunk
  //: `lane % 8`), the source a row's 128 contiguous bytes eight lanes at a time.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#copies
  struct Copier {
    static constexpr int kRowsPerCopy = 4, kCopies = kTile / kRowsPerCopy;
    uint32_t dst;
    int j, k, lane, b, h, bhw, t_window, lw;

    __device__ __forceinline__ Copier(int b_, int h_, int bhw_, int t_window_, int lw_, int lane_)
        : j(lane_ >> 3),
          k(lane_ & 7),
          lane(lane_),
          b(b_),
          h(h_),
          bhw(bhw_),
          t_window(t_window_),
          lw(lw_) {
      dst = static_cast<uint32_t>(j * kStageBytes + ((k ^ (j << 1)) << 4));
    }

    //: tile `tile`'s values and writer factors, or immediate arrivals when no warp's pair with it is live; a partial
    //: call's (`!kEven`) tile the call ends inside, or any tile of an odd-length call, bounded
    template <bool kEven>
    __device__ __forceinline__ void values(const Smem& sm, const IntraParams& p, int tile,
                                           bool issue) const {
      if (!issue) {
        ops::mbar_arrive(ops::smem_addr(&sm.values_full));
        return;
      }
      const uint8_t* const row = row_at(p.values, b, t_window + tile * kTile + j, h) + k * kChunk;
      const uint32_t base = ops::smem_addr(&sm.values[0][0]) + dst;
      if constexpr (kEven) {
#pragma unroll
        for (int i = 0; i < kCopies; ++i)
          ops::stage_tile((base + i * kRowsPerCopy * kValueBytes) ^ ((i & 1) << 4),
                          row + rows_of(i * kRowsPerCopy, p.values.st));
        ops::stage_tile(ops::smem_addr(&sm.fm[2 * lane]), p.fm + static_cast<int64_t>(bhw) * p.L
                                                              + t_window + tile * kTile + 2 * lane);
      } else {
        //: a row past the call's last token zero-fills (finite, its pairs above the diagonal); a writer's factors a run
        //: of their own (a head's row of them starts on 8 bytes where the call's length is odd)
        const int left = lw - tile * kTile;
        const float2* const fm =
            p.fm + static_cast<int64_t>(bhw) * p.L + t_window + tile * kTile + 2 * lane;
#pragma unroll
        for (int i = 0; i < kCopies; ++i)
          ops::stage_tile_if((base + i * kRowsPerCopy * kValueBytes) ^ ((i & 1) << 4),
                             row + rows_of(i * kRowsPerCopy, p.values.st),
                             j + i * kRowsPerCopy < left);
#pragma unroll
        for (int e = 0; e < 2; ++e)
          ops::stage_run_if<sizeof(float2)>(ops::smem_addr(&sm.fm[2 * lane + e]), fm + e,
                                            2 * lane + e < left);
      }
      ops::mbar_arrive_when_landed(ops::smem_addr(&sm.values_full));
    }
  };

  //: A THREAD'S COPY BASES: each level's write rows of the call's (batch, write head) at the window's first token and
  //: read rows at the block's first reader, formed once -- a stage's copy adds its row's token strides and its digits.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  struct Bases {
    const uint8_t* write[kMaxLevels];
    const uint8_t* read[kMaxLevels];
    int64_t write_st[kMaxLevels], read_st[kMaxLevels];
    //: a partial call's bounds: the window's tokens and the block's readers before its end (a row past them zero-fills,
    //: read from nowhere), and whether the call ends inside the block
    int lw, lr;
    bool whole;
  };

  static __device__ __forceinline__ Bases bases_of(const IntraParams& p, int b, int hw, int h,
                                                   int t_window, int t_block) {
    Bases out;
    out.lw = min(kWindow, p.L - t_window);
    out.lr = min(kBlockRows, p.L - t_block);
    out.whole = out.lr == kBlockRows;
#pragma unroll
    for (int l = 0; l < kMaxLevels; ++l) {
      out.write[l] = row_at(p.write[l], b, t_window, hw);
      out.read[l] = row_at(p.read[l], b, t_block, h);
      out.write_st[l] = p.write[l].st;
      out.read_st[l] = p.read[l].st;
    }
    return out;
  }

  //: A STAGE'S WORD, what a step reads of it, built once a CTA into shared memory (`build_words`): the write side's
  //: first byte and byte limit; the stage's level (bits 0-7), its reader chunk (bits 8-15) and its reader part's chunks a
  //: row as a power of two (bits 16-23: the stage's digits at the read side's width, 4, 8 or 16); its slab bits' first
  //: index and mask; the read side's first byte and limit. -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  static __device__ __forceinline__ void build_words(Smem& sm, const IntraParams& p, int tid) {
    if (tid < p.stages) {
      const Stage& g = p.stage[tid];
      const int form = p.form[g.level];
      const int we = form_write_f16(form) ? 2 : 1, re = form_read_f16(form) ? 2 : 1;
      const int chunks = form_stage_digits(form) * re / kChunk, lg = chunks == 16  ? 4
                                                                     : chunks == 8 ? 3
                                                                                   : 2;
      sm.words[tid] = make_int4(
          (g.digit0 * we) | (p.width[g.level] * we) << 16, g.level | g.reader_chunk << 8 | lg << 16,
          g.slab0 | static_cast<int>(g.mask << 8), (g.digit0 * re) | (p.width[g.level] * re) << 16);
    }
  }

  //: A THREAD'S SHARE OF A STAGE: tile `tile`'s stage `s` into buffer `buf`, its two rows `tid / 8` and `+ 32` (halves 0
  //: and 1; `kHalves` the ones copied) at chunk `tid % 8` (swizzled by the row), predicated on `issue` -- a stage no
  //: warp reads, or past the piece, reads nothing and zero-fills a buffer every warp has released.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  template <int D, bool kEven, int kHalf0 = 0, int kHalves = 2>
  static __device__ __forceinline__ void stage_share(const Smem& sm, const Bases& at,
                                                     const int4& word, int buf, int tile,
                                                     bool issue, int tid) {
    static_assert(kTile * 8 == 2 * kThreads, "a stage's rows are two copies a thread");
    const int level = D == 1 ? 0 : word.y & 0xff, k = tid & 7;
    const int byte = (word.x & 0xffff) + k * kChunk;
    const bool live = issue & (byte < (word.x >> 16));
    const uint8_t* const base =
        (level == 0 ? at.write[0] : at.write[kMaxLevels - 1]) + (live ? byte : 0);
    const int64_t st = level == 0 ? at.write_st[0] : at.write_st[kMaxLevels - 1];
    const uint32_t rows = ops::smem_addr(&sm.stage[buf][0][0]);
    constexpr int kHalfEnd = kHalf0 + kHalves;
#pragma unroll
    for (int i = kHalf0; i < kHalfEnd; ++i) {
      const int row = (tid >> 3) + 32 * i;
      if constexpr (kEven) {
        ops::stage_tile_if(
            rows + static_cast<uint32_t>(row * kStageBytes + ((k ^ swizzle(row)) << 4)),
            base + rows_of(tile * kTile + row, st), live);
      } else {
        ops::stage_tile_if(
            rows + static_cast<uint32_t>(row * kStageBytes + ((k ^ swizzle(row)) << 4)),
            base + rows_of(tile * kTile + row, st), live & (tile * kTile + row < at.lw));
      }
    }
  }

  //: THE BLOCK'S READER ROWS for stage `s`, once: the stage's digits of each reader row (4, 8 or 16 chunks, its word's) into
  //: the row's slot for it, a thread's chunk `tid % 16` of rows `tid / 16 + (kThreads / 16) i`, in the copy group of the
  //: piece's first tile's stage `s`, so the first Gram waits for one part, not the block's rows; each copy issued only
  //: where `issue` (a stage the call has) and the part has its chunk, its digits past the level's width zero-filled.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  template <int D, bool kEven>
  static __device__ __forceinline__ void reader_part_any(Smem& sm, const IntraParams& p,
                                                         const Bases& at, int s, bool issue,
                                                         int tid) {
    constexpr int kMaxChunks = 16, kPass = kThreads / kMaxChunks, kRowsEach = kBlockRows / kPass;
    static_assert(kBlockRows % kPass == 0, "a widest part is whole passes of the block's threads");
    const int4 word = sm.words[s];
    const int level = D == 1 ? 0 : word.y & 0xff, chunk = (word.y >> 8) & 0xff,
              lg = (word.y >> 16) & 0xff;
    const int k = tid & (kMaxChunks - 1), r0 = tid / kMaxChunks;
    const int byte = (word.w & 0xffff) + k * kChunk;
    const bool live = byte < (word.w >> 16), has = issue & (k < (1 << lg));
    const int64_t st = level == 0 ? at.read_st[0] : at.read_st[kMaxLevels - 1];
    //: a thread's rows are `kPass` apart and share a swizzle (`kPass` a multiple of 8): its copies step by fixed strides
    static_assert(kPass % 8 == 0, "a thread's reader rows share their swizzle");
    const uint8_t* src =
        (level == 0 ? at.read[0] : at.read[kMaxLevels - 1]) + rows_of(r0, st) + (live ? byte : 0);
    const uint64_t src_step = rows_of(kPass, st);
    uint32_t dst =
        ops::smem_addr(reader_rows(sm) + r0 * p.pitch + (((chunk + k) ^ swizzle(r0)) << 4));
    const uint32_t dst_step = static_cast<uint32_t>(kPass * p.pitch);
#pragma unroll
    for (int i = 0; i < kRowsEach; ++i) {
      if constexpr (kEven) {
        ops::stage_tile_when(dst, src, has, live);
      } else {
        ops::stage_tile_when(dst, src, has, live & (r0 + kPass * i < at.lr));
      }
      src += src_step;
      dst += dst_step;
    }
  }

  //: A READER PART OF FORM `F`, its chunks a row known: a thread's chunk `tid % chunks` of rows `tid / chunks + (kThreads /
  //: chunks) i`, every copy live where `issue` -- the first tile's steps, whose part is always their own level's.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  template <int D, int F>
  static __device__ __forceinline__ void reader_part(Smem& sm, const IntraParams& p,
                                                     const Bases& at, int s, bool issue, int tid) {
    constexpr int kChunks = form_stage_digits(F) * (form_read_f16(F) ? 2 : 1) / kChunk,
                  kPass = kThreads / kChunks;
    constexpr int kRowsEach = kBlockRows / kPass;
    static_assert(kPass % 8 == 0 && kBlockRows % kPass == 0,
                  "a part's rows share a swizzle, whole passes a thread");
    const int4 word = sm.words[s];
    const int level = D == 1 ? 0 : word.y & 0xff, chunk = (word.y >> 8) & 0xff;
    const int k = tid % kChunks, r0 = tid / kChunks;
    const int byte = (word.w & 0xffff) + k * kChunk;
    const bool live = byte < (word.w >> 16);
    const int64_t st = level == 0 ? at.read_st[0] : at.read_st[kMaxLevels - 1];
    const uint8_t* src =
        (level == 0 ? at.read[0] : at.read[kMaxLevels - 1]) + rows_of(r0, st) + (live ? byte : 0);
    const uint64_t src_step = rows_of(kPass, st);
    uint32_t dst =
        ops::smem_addr(reader_rows(sm) + r0 * p.pitch + (((chunk + k) ^ swizzle(r0)) << 4));
    const uint32_t dst_step = static_cast<uint32_t>(kPass * p.pitch);
#pragma unroll
    for (int i = 0; i < kRowsEach; ++i) {
      ops::stage_tile_when(dst, src, issue, live);
      src += src_step;
      dst += dst_step;
    }
  }

  //: A TILE UNION's address in the facts pass's grouped table: a write head's `group + 1` slots, its query heads' read
  //: sides then its write side.
  template <bool kEven>
  static __device__ __forceinline__ const uint32_t* union_at(const IntraParams& p, int bhw,
                                                             int slot, int group, int tile) {
    //: the facts pass's tiles a row, the last one partial where a partial call ends inside it
    const int tiles = kEven ? p.L / kTile : (p.L + kTile - 1) / kTile;
    return p.unions
           + ((static_cast<int64_t>(bhw) * (p.group + 1) + slot) * p.groups + group) * tiles + tile;
  }

  //: THE GATE TABLE: per (window tile, row tile) the slabs whose reader and writer tile unions meet, a lane a slab,
  //: cleared whole when a level has none (the pair's product is exactly zero) or the pair lies above the diagonal; a
  //: dense level answers all-ones unread; a partial call's (`!kEven`) pairs with a tile past its last (`tiles` the
  //: window's before it) cleared, the union table read only before it. One warp.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#slab-gate
  template <int D, bool kEven>
  static __device__ __forceinline__ void gate_table(Smem& sm, const IntraParams& p, int bhw,
                                                    int slot, int window, int block, int tiles,
                                                    int lane) {
    int level = 0;
#pragma unroll
    for (int l = 1; l < D; ++l)
      if (lane >= p.slab0[l]) level = l;
    const int group = p.group0[level] + lane - p.slab0[level];
    const bool slab = lane < p.slabs;
    const uint32_t all = slab ? ~0u : 0u;
    const int tile0 = window * kWindowTiles;

    uint32_t rows[kRowTiles], columns[kWindowTiles];
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) rows[rt] = all;
#pragma unroll
    for (int t = 0; t < kWindowTiles; ++t) columns[t] = all;
    if (slab && !facts::level_is_dense(p.dense[facts::kRead], level)) {
      const uint32_t* const r = union_at<kEven>(p, bhw, slot, group, tile0 + block * kRowTiles);
#pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt)
        rows[rt] = kEven || block * kRowTiles + rt < tiles ? r[rt] : 0u;
    }
    if (slab && !facts::level_is_dense(p.dense[facts::kWrite], level)) {
      const uint32_t* const w = union_at<kEven>(p, bhw, p.group, group, tile0);
      if constexpr (kEven) {
#pragma unroll
        for (int q = 0; q < kWindowQuads; ++q) {
          const uint4 x = ops::load_vec16(w + 4 * q);
          columns[4 * q] = x.x, columns[4 * q + 1] = x.y, columns[4 * q + 2] = x.z,
                      columns[4 * q + 3] = x.w;
        }
      } else {
        //: a word a tile: a head's row of unions need not start on 16 bytes
#pragma unroll
        for (int t = 0; t < kWindowTiles; ++t) columns[t] = t < tiles ? w[t] : 0u;
      }
    }
    uint32_t level_slabs[D];
#pragma unroll
    for (int l = 0; l < D; ++l) level_slabs[l] = p.level_slabs[l];
#pragma unroll
    for (int t = 0; t < kWindowTiles; ++t)
#pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt) {
        uint32_t mask = __ballot_sync(0xffffffffu, (rows[rt] & columns[t]) != 0u);
#pragma unroll
        for (int l = 0; l < D; ++l)
          if ((mask & level_slabs[l]) == 0u) mask = 0u;
        if (lane == 0)
          sm.gate[t][rt] =
              t <= block * kRowTiles + rt && (kEven || block * kRowTiles + rt < tiles) ? mask : 0u;
      }
  }

  //: a stage's slabs of a gate word, from its stage word
  static __device__ __forceinline__ uint32_t word_slabs(const int4& word, uint32_t gate) {
    return (gate >> (word.z & 0xff)) & (static_cast<uint32_t>(word.z) >> 8);
  }

  //: any row tile's gate word for writer tile `tile`
  static __device__ __forceinline__ uint32_t tile_gate(const Smem& sm, int tile) {
    uint32_t any = 0u;
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) any |= sm.gate[tile][rt];
    return any;
  }

  //: THE DEPOSIT: the warp's 16 rows of partial sums -- its writer columns' -- reduced into the output planes through a
  //: staging tile of its own in the dead reader rows, a row's channels two lines of consecutive reductions; its rows'
  //: partial sums into `den`. -- see docs/internals/intra/ampere/intra_kernel.md#deposit
  template <bool kEven>
  static __device__ __forceinline__ void deposit(Smem& sm, const IntraParams& p, int bh, int t_row,
                                                 int warp, int lane, const float (&out)[8][4]) {
    constexpr int kPitch = kValueWidth + 8;
    static_assert(kWarpRows * kPitch * 4 == kStagingBytes, "the staging tile is the walk's");
    //: the reader rows are dead once every warp's last Gram is done
    __syncthreads();
    float* const staging = reinterpret_cast<float*>(reader_rows(sm) + warp * kStagingBytes);
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        ops::store_shared_f32x2(ops::smem_addr(&staging[(g + 8 * r) * kPitch + 8 * n + 2 * tq]),
                                out[n][2 * r], out[n][2 * r + 1]);
    __syncwarp();
    float* const num = p.num + (static_cast<int64_t>(bh) * p.L + t_row) * kValueWidth;
    //: a partial call's rows past its end are never written
    const int rows = kEven ? kWarpRows : min(kWarpRows, p.L - t_row);
#pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      if (kEven || r < rows) {
        ops::fan_in_add(num + r * kValueWidth + lane, staging[r * kPitch + lane]);
        ops::fan_in_add(num + r * kValueWidth + 32 + lane, staging[r * kPitch + 32 + lane]);
      }
    }
    if (lane < rows)
      ops::fan_in_add(p.den + static_cast<int64_t>(bh) * p.L + t_row + lane, sm.den[warp][lane]);
  }

  //: THE RING'S PLACE: the stage's buffer and the buffer before it (the refill's), and the stage the refill copies --
  //: `kStageBuffers - 1` past the current, as a (tile, stage) and as the piece's stage index -- stepped once a stage.
  struct Ring {
    int buf = 0, prev = kStageBuffers - 1, tile, s;
    //: the refill's stage word and its tile's gate word, read a step ahead of the refill that uses them
    int4 word;
    uint32_t gate;

    __device__ __forceinline__ Ring(int t, int s_) : tile(t), s(s_) {}

    __device__ __forceinline__ void read(const Smem& sm) {
      word = sm.words[s];
      gate = tile_gate(sm, min(tile, kWindowTiles - 1));
    }

    __device__ __forceinline__ void step(const Smem& sm, const IntraParams& p) {
      prev = buf;
      buf = buf + 1 == kStageBuffers ? 0 : buf + 1;
      const bool next = s + 1 == p.stages;
      s = next ? 0 : s + 1;
      tile = next ? tile + 1 : tile;
      read(sm);
    }
  };

  //: the piece's stage `ahead` stages past (tile, s): its tile and stage, uniform
  static __device__ __forceinline__ int2 stage_after(const IntraParams& p, int tile, int s,
                                                     int ahead) {
    int ns = s + ahead, ntile = tile;
#pragma unroll 1
    while (ns >= p.stages) ns -= p.stages, ++ntile;
    return make_int2(ntile, ns);
  }

  //: HOW A STAGE'S GRAM RUNS: none (a dead pair's stage is awaited and released), straight-line, or a loop over its slabs
  enum StageGram : int { kGramNone, kGramDense, kGramSlabs };

  //: ONE STAGE OF A TILE, the same block for every warp (KERNEL_STANDARDS section 4's form on sm_80): the thread's copy
  //: group of the stage landed, one CTA barrier -- every thread's copies landed and every warp done with the stage
  //: before -- then the stage's Gram under the level's form `F` with the refill under it (the stage `kStageBuffers - 1`
  //: on into the buffer just released, a half after each half of a straight-line Gram's slabs, after the Gram
  //: otherwise), the reader part of that stage after it while the piece's first tile is read.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  template <int D, int F, bool kF16, int Gram, bool kReader, bool kEven>
  static __device__ __forceinline__ void tile_stage(Smem& sm, const IntraParams& p, const Lanes& ln,
                                                    const Bases& at, const int4& word,
                                                    uint32_t mine, int t1, int tid, Ring& ring,
                                                    Acc<kF16>& acc, int rs, bool rissue) {
    ops::stage_wait<kStageBuffers - 2>();
    __syncthreads();
    const int refill_tile = min(ring.tile, kWindowTiles - 1);
    const bool refill_live = (ring.tile < t1) & (word_slabs(ring.word, ring.gate) != 0u);
    const auto refill = [&](auto Half) {
      stage_share<D, kEven, decltype(Half)::value, 1>(sm, at, ring.word, ring.prev, refill_tile,
                                                      refill_live, tid);
    };
    const int chunk = (word.y >> 8) & 0xff;
    const uint32_t rows = ops::smem_addr(&sm.stage[ring.buf][0][0]);
    if constexpr (Gram == kGramDense) {
      gram_dense<F, kF16>(ln, chunk, rows, acc, refill);
    } else {
      if constexpr (Gram == kGramSlabs)
        gram_slabs<F, kF16>(ln, chunk, rows, word_slabs(word, mine), acc);
      stage_share<D, kEven>(sm, at, ring.word, ring.prev, refill_tile, refill_live, tid);
    }
    if constexpr (kReader) reader_part<D, F>(sm, p, at, rs, rissue, tid);
    ops::stage_commit();
    ring.step(sm, p);
  }

  //: A TILE'S STAGES, level by level under each level's form -- one uniform switch a level, its Gram opened before them
  //: and folded after -- straight-line where the warp reads every slab of the level, by its slabs where it reads some,
  //: none for a dead pair, chosen once a level. The first tile's steps carry the reader part two stages on where it is
  //: the level's own (its first two parts ride the prologue's groups; a block a partial call ends inside lands every
  //: part there). -- see docs/internals/intra/ampere/intra_kernel.md#pipeline
  template <int D, bool kF16, bool kReader, bool kEven>
  static __device__ __forceinline__ void tile_stages(Smem& sm, const IntraParams& p,
                                                     const Lanes& ln, const Bases& at, int tile,
                                                     int t1, int row_tile, int tid, Ring& ring,
                                                     Acc<kF16>& acc) {
    const uint32_t mine = sm.gate[tile][row_tile % kRowTiles];
    rola::static_for<D>([&](auto Level) {
      constexpr int kL = decltype(Level)::value;
      rola::uniform_switch<kForms>(p.form[kL], [&](auto F) {
        constexpr int kF = decltype(F)::value, kSlabs = form_stage_digits(kF) / kSlabDigits;
        zero_gram<kF, kF16>(acc);
        const uint32_t level = p.level_slabs[kL];
        const int s0 = p.level_stage0[kL], s1 = s0 + p.level_stages[kL];
        const auto stages = [&](auto Gram) {
          int4 word = sm.words[s0];
#pragma unroll 1
          for (int s = s0; s < s1; ++s) {
            const int4 now = word;
            word = sm.words[min(s + 1, s1 - 1)];
            tile_stage<D, kF, kF16, decltype(Gram)::value, kReader, kEven>(
                sm, p, ln, at, now, mine, t1, tid, ring, acc, min(s + 2, s1 - 1),
                kEven ? s + 2 < s1 : (s + 2 < s1) & at.whole);
          }
        };
        if (mine == 0u) {
          stages(std::integral_constant<int, kGramNone>{});
        } else if ((mine & level) == level && __popc(level) == p.level_stages[kL] * kSlabs) {
          stages(std::integral_constant<int, kGramDense>{});
        } else {
          stages(std::integral_constant<int, kGramSlabs>{});
        }
        fold_level<kF, kF16>(acc, kL == 0, p.lift);
      });
    });
  }

  //: A TILE'S VALUES: awaited after its last stage; the epilogue and the value product for a live pair, the values
  //: released and the next tile's copied by the last warp to release them.
  template <bool kF16, bool kEven>
  static __device__ __forceinline__ void tile_values(Smem& sm, const IntraParams& p,
                                                     const Copier& copy, int tile, int t0, int t1,
                                                     int row_tile, int warp, int group, int col0,
                                                     int lane, Acc<kF16>& acc, Clock& clock) {
    const bool leader = lane == 0;
    const int r = tile - t0;
    ops::mbar_wait(ops::smem_addr(&sm.values_full), r & 1);
    clock.lap(kPhaseWait);
    if (sm.gate[tile][row_tile % kRowTiles] != 0u) {
      epilogue(sm, p.unit, tile == row_tile, warp, group, col0, lane, acc);
      clock.lap(kPhaseEpilogue);
      value_product(sm, p.unit, col0, lane, acc);
      clock.lap(kPhaseValue);
    }
    //: the last warp to release the tile's values copies the next tile's. Every warp awaits a tile's values before it
    //: releases them, and tile `t + 2`'s land only after tile `t + 1`'s last release, so tiles `t` and `t + 2` share a
    //: counter by parity and never interleave on it
    __syncwarp();
    int last = 0;
    if (leader) {
      __threadfence_block();
      last = atomicAdd(&sm.values_released[r & 1], 1) == (r / 2 + 1) * kWarps - 1;
    }
    if (__shfl_sync(0xffffffffu, last, 0) && tile + 1 < t1) {
      __threadfence_block();
      copy.template values<kEven>(sm, p, tile + 1, tile_gate(sm, tile + 1) != 0u);
    }
  }

  //: THE GRID: a CTA a (window, piece) along x -- a window's longest piece first -- and a query head along y, an arm a
  //: depth `D` and an accumulation. The block's readers land once, a part a stage, tile 0's first stage beside the first
  //: part; the piece's writer tiles stream through the stage buffers; its partial sums reduce into the output planes. An
  //: arm `kEven` false serves a call whose last window is partial: a piece runs its tiles before the call's end, one with
  //: none exits, every row past the end bounded; those arms compile in `intra_partial.cu`.
  //: -- see docs/internals/intra/ampere/intra_kernel.md#the-grid
  template <int D, bool kF16, bool kEven>
  static __device__ __forceinline__ void run(const IntraParams& p) {
    extern __shared__ __align__(128) unsigned char raw[];
    Smem& sm = *reinterpret_cast<Smem*>(raw);
    if (blockDim.x != kThreads) __trap();
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int bh = static_cast<int>(blockIdx.y);
    const int window = static_cast<int>(blockIdx.x) / Pieces::count();
    const Piece entry = kPieceTable<Wk>.table[static_cast<int>(blockIdx.x) % Pieces::count()];
    //: a partial call's window tiles before its end: a piece whose readers or writers all lie past them has nothing to do
    const int tiles =
        kEven ? kWindowTiles : (min(kWindow, p.L - window * kWindow) + kTile - 1) / kTile;
    const Piece piece = kEven ? entry : Piece{entry.block, entry.t0, min(entry.t1, tiles)};
    const int block = piece.block;
    if constexpr (!kEven) {
      if (piece.t0 >= piece.t1 || block * kRowTiles >= tiles) return;
    }
    const int b = bh / p.H, h = bh - b * p.H;
    const int hw = h / p.group, slot = h - hw * p.group;
    const int bhw = b * (p.H / p.group) + hw;
    const int t_window = window * kWindow;
    const int t_block = t_window + block * kBlockRows;
    //: the warp's reader group and its writer columns in a tile
    const int group = warp / kColWarps, col0 = (warp % kColWarps) * kWarpCols;
    const int row_tile = block * kRowTiles + group / (kRowWarps / kRowTiles);
    Clock clock(&sm.clock[warp][0], lane, p.ledger != nullptr);
    if (tid == 0) {
      sm.values_released[0] = sm.values_released[1] = 0;
      ops::mbar_init(ops::smem_addr(&sm.values_full), 32);
    }
    build_words(sm, p, tid);
    __syncthreads();

    Acc<kF16> acc;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc.out[n][i] = 0.0f;
    if (lane < kWarpRows) sm.den[warp][lane] = 0.0f;

    const Copier copy(b, hw, bhw, t_window, min(kWindow, p.L - t_window), lane);
    const Lanes ln(sm, p, group, col0, lane);
    //: the piece's first `kStageBuffers - 1` stages, each a copy group with its reader part, copied before the gate
    //: exists; the reader parts after them ride the ring's refills (`tile_stage`)
    const Bases at = bases_of(p, b, hw, h, t_window, t_block);
#pragma unroll 1
    for (int i = 0; i + 1 < kStageBuffers; ++i) {
      const int2 first = stage_after(p, piece.t0, 0, i);
      stage_share<D, kEven>(sm, at, sm.words[first.y], i, min(first.x, kWindowTiles - 1),
                            first.x < piece.t1, tid);
      //: the reader parts no step carries -- each level's first two, every part of a block a partial call ends
      //: inside -- in the first two groups, a part `s` in group min(s, 1)
#pragma unroll 1
      for (int s = 0; s < p.stages; ++s) {
        const bool front = s - p.level_stage0[sm.words[s].y & 0xff] < 2 || (!kEven && !at.whole);
#pragma unroll 1
        for (int k = ops::once(front && min(s, kStageBuffers - 2) == i); k > 0; --k)
          reader_part_any<D, kEven>(sm, p, at, s, true, tid);
      }
      ops::stage_commit();
    }
    if (warp == 0) copy.template values<kEven>(sm, p, piece.t0, true);
    clock.lap(kPhaseReader);
    if (warp == 0) gate_table<D, kEven>(sm, p, bhw, slot, window, block, tiles, lane);
    __syncthreads();
    clock.lap(kPhaseGate);

    const int2 refill = stage_after(p, piece.t0, 0, kStageBuffers - 1);
    Ring ring(refill.x, refill.y);
    ring.read(sm);
    //: the piece's first tile carries the reader parts the ring's first refills did not
    tile_stages<D, kF16, true, kEven>(sm, p, ln, at, piece.t0, piece.t1, row_tile, tid, ring, acc);
    clock.lap(kPhaseGram);
    tile_values<kF16, kEven>(sm, p, copy, piece.t0, piece.t0, piece.t1, row_tile, warp, group, col0,
                             lane, acc, clock);
#pragma unroll 1
    for (int tile = piece.t0 + 1; tile < piece.t1; ++tile) {
      tile_stages<D, kF16, false, kEven>(sm, p, ln, at, tile, piece.t1, row_tile, tid, ring, acc);
      clock.lap(kPhaseGram);
      tile_values<kF16, kEven>(sm, p, copy, tile, piece.t0, piece.t1, row_tile, warp, group, col0,
                               lane, acc, clock);
    }
    deposit<kEven>(sm, p, bh, t_block + group * kWarpRows, warp, lane, acc.out);
    clock.lap(kPhaseDeposit);
    clock.publish(p.ledger, 0, kWarps, warp);
  }
};

//: THE KERNEL at build `Bt::build` (a type: docs/internals/common/hardware.md#build-tag)
template <class Bt, int D, bool kF16, bool kEven>
__global__ __launch_bounds__(Walk<Bt::build>::kThreads, 1) void intra_kernel(
    const __grid_constant__ IntraParams p) {
  IntraAt<Walk<Bt::build>>::template run<D, kF16, kEven>(p);
}

}  // namespace rola::intra::detail
