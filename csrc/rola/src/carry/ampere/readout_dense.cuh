// csrc/rola/src/carry/ampere/readout_dense.cuh -- THE READOUT OVER A DENSE READ SIDE: every token of a chunk reads every digit
// of the box, a warp its 8 channels over every leaf into one fp32 accumulator a token, out through the panel.
// see docs/internals/carry/ampere/readout_dense.md
#pragma once

#include "carry/ampere/readout.cuh"

namespace rola::carry {

//: A DENSE READ ROW: the token's K rows (the MMA's A), then its digits' coefficients from `kCoefAt`, rewritten by the
//: factors; a slot's two exit words (`DenseExit`) where a listed side's entries are. -- see readout_dense.md#row
template <class P>
constexpr int kCoefAt = 2 * P::kK;

//: A TOKEN BATCH a warp: 64 tokens, four tiles' accumulators; a TRANSIT GROUP 4 digits' slices (the fold's 128 state
//: registers live beside them). -- see docs/internals/carry/ampere/readout_dense.md#nest
constexpr int kDenseTiles = 4, kDenseBatch = 16 * kDenseTiles, kTransit = 4;

//: A DENSE ROW'S 16-BYTE CHUNK `k` of slot `s`, the row's swizzle folded in.
template <class P>
__device__ __forceinline__ ops::SmemAddr dense_chunk(ops::SmemAddr rows, int s, int k) {
  return rows + static_cast<uint32_t>(s * P::kReadRow)
         + 16u * chunk_at<P::kReadRow>(static_cast<uint32_t>(s), static_cast<uint32_t>(k));
}

//: A POWER OF TWO by selects (`pow2_wide`'s range, no branch): normal from 2^-126, a subnormal's bit to 2^-149, zero
//: below, at most 2^127.
__device__ __forceinline__ float pow2_sel(int e) {
  const uint32_t normal = static_cast<uint32_t>(min(max(e, -126), 127) + 127) << 23;
  const uint32_t sub = 1u << min(max(e + 149, 0), 31);
  return __uint_as_float(e >= -126 ? normal : e >= -149 ? sub : 0u);
}

//: THE DENSE FACTORS, once a chunk, four lanes a token (8 digits a lane, one 16-byte load) and 64 tokens a step over
//: the CTA: `y = a 2^(e_d)` exactly, the token's exponent `e_t = ceil_exp(max |y|)` (= the largest `e_d + ceil_exp(|a|)`;
//: two shuffles, no branch), its coefficients rewritten `f16(y 2^(-e_t))`, its exit words -- the num's
//: `2^(e_t - kImageTop - kCoefTop)` (the state's c top undone; at the f16 panel with its scale byte's
//: `2^(-scale - 1)`, one power of two) and the den's `2^(e_t)`; zero for a token reading nothing. -- see docs/internals/carry/ampere/readout_dense.md#factors
template <class P>
__device__ __forceinline__ void dense_factors(const CarryParams& p, const uint8_t* smem, int w,
                                              int first, int slots, ops::SmemAddr rows,
                                              ops::SmemAddr exits, int warp, int lane) {
  static_assert(P::kReadFactors == 2 && P::kDigits == 32, "a dense read side, four lanes a token");
  const ops::SmemAddr scale =
      ops::smem_addr(smem + p.tables.scale_at)
      + static_cast<uint32_t>((w & 1) * p.tables.heads * P::kWindow + first);
  const ops::SmemAddr exps = exps_of<P>(smem, warp);
  const bool f32 = p.reduce == kReduceF32;

  //: the lane's quarter of the digits and its token in the warp's eight; each of its digits' `2^(e_d)`
  const int q = lane & 3, t = lane >> 2;
  float pd[8];
  static_for<8>([&](auto D) {
    pd[D] = pow2_sel(static_cast<int>(ops::load_shared_u32(
        exps + 4u * static_cast<uint32_t>(exp_slot(8 * q + decltype(D)::value)))));
  });

  const int steps = __shfl_sync(0xffffffffu, (slots + 8 * kWarps - 1) / (8 * kWarps), 0);
#pragma unroll 1
  for (int j = 0; j < steps; ++j) {
    //: a slot past the chunk is its last: the same load, the same words stored again
    const int s = min(8 * kWarps * j + 8 * warp + t, slots - 1);
    const ops::SmemAddr c = dense_chunk<P>(rows, s, kCoefAt<P> / 16 + q);
    const uint4 raw = ops::load_shared_vec16(c);
    const __half* h = reinterpret_cast<const __half*>(&raw);
    float y[8];
    uint32_t m = 0u;
    static_for<8>([&](auto D) {
      y[D] = __half2float(h[D]) * pd[D];
      m = max(m, __float_as_uint(fabsf(y[D])));
    });

    m = max(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = max(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const bool any = m != 0u;
    const int et = static_cast<int>(m >> 23) - 127 + static_cast<int>((m & 0x7fffffu) != 0u);
    const float back = pow2_sel(any ? -et : 0);
    uint32_t out[4];
    static_for<4>([&](auto I) {
      const __half2 v = __floats2half2_rn(y[2 * I] * back, y[2 * I + 1] * back);
      out[I] = *reinterpret_cast<const uint32_t*>(&v);
    });
    ops::store_shared_vec16(c, make_uint4(out[0], out[1], out[2], out[3]));

    const int sb = static_cast<int8_t>(ops::load_shared_u8(scale + static_cast<uint32_t>(s)));
    const int en = et - kImageTop - kCoefTop + (f32 ? 0 : min(max(-sb - 1, -126), 127));
    const ops::SmemAddr x = exits + 8u * static_cast<uint32_t>(s);
    ops::store_shared_f32_if(x, any ? pow2_sel(en) : 0.0f, q == 0);
    ops::store_shared_f32_if(x + 4u, any ? pow2_sel(et) : 0.0f, q == 0);
  }
}

//: A TRANSIT GROUP'S SLICES: digits `d0 .. d0 + 3` (K 16 x the warp's 8 channels, an MMA's B each), two digits an
//: `ldmatrix.x4.trans` (lanes 0-15 the first's 16 rows, 16-31 the second's). -- see readout_dense.md#nest
template <class P>
__device__ __forceinline__ void dense_slices(uint32_t (&b)[kTransit][2], ops::SmemAddr image,
                                             uint32_t mask, int cg, int d0, int lane) {
  static_for<kTransit / 2>([&](auto J) {
    const int d = d0 + 2 * decltype(J)::value + (lane >> 4);
    uint32_t r[4];
    ops::load_frag_t(r, image + (static_cast<uint32_t>(d) & mask) * P::kPlaneBytes
                            + slice_row<P>(cg, lane & 15, d));
    b[2 * J][0] = r[0];
    b[2 * J][1] = r[1];
    b[2 * J + 1][0] = r[2];
    b[2 * J + 1][1] = r[3];
  });
}

//: A TILE'S ROWS SCALED by a digit's coefficients: the lane's rows `g` (registers 0, 2) and `g + 8` (1, 3) times their
//: tokens' `c`.
__device__ __forceinline__ void scaled(uint32_t (&y)[4], const uint32_t (&a)[4], __half cg0,
                                       __half cg8) {
  y[0] = ops::hmul2_by(a[0], cg0);
  y[1] = ops::hmul2_by(a[1], cg8);
  y[2] = ops::hmul2_by(a[2], cg0);
  y[3] = ops::hmul2_by(a[3], cg8);
}

//: THE DENSE READOUT of a chunk's slots over digits `lo .. hi - 1` of the image (the whole, or a half) into the panel:
//: a warp its 8 channels, a batch its four tiles' fp32 accumulators, a transit group's slices held over the batch's
//: tiles, a tile's rows loaded once and scaled a digit, each token's accumulator times its exit word added into its
//: panel word (`F32` the panel's dtype). -- see docs/internals/carry/ampere/readout_dense.md#nest
//: @burst-exempt a loop over the chunk's slots (`unroll 1`), not yet `Burst` units
template <class P, bool F32>
__device__ __forceinline__ void read_dense(int slots, int lo, int hi, ops::SmemAddr image,
                                           uint32_t mask, ops::SmemAddr rows, ops::SmemAddr panel,
                                           ops::SmemAddr exits, int warp, int lane) {
  static_assert(P::kReadFactors == 2 && P::kTokenGroups == 1 && P::kK == 16 && P::kChannels == 64,
                "a dense read side at K 16, one token group of 8 channel groups");
  const int cg = warp, g = lane >> 2, tq = lane & 3;
  //: the lane's panel word: its pair's channels `8 cg + 2 tq` (f16) or its two words (fp32)
  const uint32_t word =
      F32 ? static_cast<uint32_t>(8 * cg + 2 * tq) : static_cast<uint32_t>(4 * cg + tq);

#pragma unroll 1
  for (int b0 = 0; b0 < slots; b0 += kDenseBatch) {
    float acc[kDenseTiles][4];
    static_for<kDenseTiles>([&](auto T) { static_for<4>([&](auto I) { acc[T][I] = 0.0f; }); });
#pragma unroll 1
    for (int d0 = lo; d0 < hi; d0 += kTransit) {
      uint32_t b[kTransit][2];
      dense_slices<P>(b, image, mask, cg, d0, lane);
      //: the group's coefficients, 8 bytes a row in its 16-byte chunk
      const int cd = kCoefAt<P> / 16 + d0 / 8;
      const uint32_t ch = 2u * static_cast<uint32_t>(d0 % 8);
      static_for<kDenseTiles>([&](auto T) {
        const int s = b0 + 16 * decltype(T)::value;
        //: the tile's rows (a slot past the chunk its last: read, never added) and its two rows' coefficients
        uint32_t a[4];
        ops::load_frag(a, dense_chunk<P>(rows, min(s + (lane & 15), slots - 1), lane >> 4));
        const uint2 c0 = ops::load_shared_u64(dense_chunk<P>(rows, min(s + g, slots - 1), cd) + ch);
        const uint2 c8 =
            ops::load_shared_u64(dense_chunk<P>(rows, min(s + g + 8, slots - 1), cd) + ch);
        const __half* h0 = reinterpret_cast<const __half*>(&c0);
        const __half* h8 = reinterpret_cast<const __half*>(&c8);
        static_for<kTransit>([&](auto D) {
          uint32_t y[4];
          scaled(y, a, h0[D], h8[D]);
          ops::hmma(acc[T], y, b[D][0], b[D][1]);
        });
      });
    }

    //: THE BATCH OUT: each row's accumulator pair times its token's exit word, into its panel word
    static_for<kDenseTiles>([&](auto T) {
      static_for<2>([&](auto H) {
        const int s = b0 + 16 * decltype(T)::value + g + 8 * decltype(H)::value;
        const bool live = s < slots;
        const float f = ops::load_shared_f32_if(exits + 8u * static_cast<uint32_t>(s), live);
        const ops::SmemAddr at =
            panel + panel_word<F32 ? 256 : 128>(static_cast<uint32_t>(s), word);
        const float v0 = acc[T][2 * H] * f, v1 = acc[T][2 * H + 1] * f;
        if constexpr (F32) {
          const uint2 old = ops::load_shared_u64_if(at, live);
          ops::store_shared_f32x2_if(at, __uint_as_float(old.x) + v0, __uint_as_float(old.y) + v1,
                                     live);
        } else {
          const uint32_t old = ops::load_shared_u32_if(at, live);
          const __half2 v = __floats2half2_rn(v0, v1);
          ops::store_shared_u32_if(at, add_h2(old, *reinterpret_cast<const uint32_t*>(&v)), live);
        }
      });
    });
  }
}

//: THE DENSE DEN, once a chunk over the whole den image: warp `w` the tiles `i = w mod 8`, a den MMA a block of 8
//: digits on the unscaled K rows, its 8 columns times `c[t, d]` and the digit's `2^(dE - kImageTop - e_d)`, the
//: token's `2^(e_t)`, into the warp's den word. -- see docs/internals/carry/ampere/readout_dense.md#den
//: @burst-exempt a loop over the chunk's slots (`unroll 1`), not yet `Burst` units
template <class P>
__device__ __forceinline__ void dense_den(const uint8_t* smem, int slots, ops::SmemAddr rows,
                                          ops::SmemAddr dens, ops::SmemAddr exits, int warp,
                                          int lane) {
  constexpr int kBlocks = P::kDigits / kBatchPlanes;
  const int g = lane >> 2, tq = lane & 3;
  const ops::SmemAddr den_image = ops::smem_addr(smem + P::kDenImage);
  const ops::SmemAddr dfac = ops::smem_addr(smem + P::kDenFactor), exps = exps_of<P>(smem, warp);
  //: the lane's columns' factors, every block's: the den factor times `2^(-e_d)` (the coefficient's scaling undone)
  float gd[kBlocks][2];
  static_for<kBlocks>([&](auto B) {
    static_for<2>([&](auto I) {
      const int d = kBatchPlanes * decltype(B)::value + 2 * tq + decltype(I)::value;
      gd[B][I] = ops::load_shared_f32(dfac + 4u * static_cast<uint32_t>(d))
                 * pow2_sel(-static_cast<int>(
                     ops::load_shared_u32(exps + 4u * static_cast<uint32_t>(exp_slot(d)))));
    });
  });

#pragma unroll 1
  for (int s = 16 * warp; s < slots; s += 16 * kWarps) {
    uint32_t a[4];
    ops::load_frag(a, dense_chunk<P>(rows, min(s + (lane & 15), slots - 1), lane >> 4));
    float sum[2] = {0.0f, 0.0f};
    static_for<kBlocks>([&](auto B) {
      constexpr int kB = decltype(B)::value;
      uint32_t bd[2];
      ops::load_frag_t2(bd, den_image + den_row<P>(lane & 15, kB));
      float d[4];
      ops::hmma_fresh(d, a, bd[0], bd[1]);
      const int cd = kCoefAt<P> / 16 + kB;
      static_for<2>([&](auto H) {
        constexpr int kH = decltype(H)::value;
        const uint32_t c2 =
            ops::load_shared_u32(dense_chunk<P>(rows, min(s + g + 8 * kH, slots - 1), cd)
                                 + 4u * static_cast<uint32_t>(tq));
        const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&c2));
        sum[kH] += d[2 * kH] * c.x * gd[B][0] + d[2 * kH + 1] * c.y * gd[B][1];
      });
    });

    //: the quad's columns summed; lane `tq == 0` the row's den, at the token's `2^(e_t)`
    static_for<2>([&](auto H) {
      constexpr int kH = decltype(H)::value;
      sum[kH] += __shfl_xor_sync(0xffffffffu, sum[kH], 1);
      sum[kH] += __shfl_xor_sync(0xffffffffu, sum[kH], 2);
      const int t = s + g + 8 * kH;
      const bool live = tq == 0 && t < slots;
      const float e = ops::load_shared_f32_if(exits + 8u * static_cast<uint32_t>(t) + 4u, live);
      ops::store_shared_f32_if(
          dens + P::kDenRowBytes * static_cast<uint32_t>(t) + 4u * static_cast<uint32_t>(warp),
          sum[kH] * e, live);
    });
  }
}

}  // namespace rola::carry
