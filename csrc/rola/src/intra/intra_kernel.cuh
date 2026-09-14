// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"
#include "intra/intra.cuh"

namespace rola {
namespace intra {
namespace detail {

using denseref::ops::fan_in_add;
using denseref::ops::kBf16MagnitudeMask;
using denseref::ops::load_frag;
using denseref::ops::load_frag_t;
using denseref::ops::load_shared_u32;
using denseref::ops::mma;
using denseref::ops::pack_bf16x2;
using denseref::ops::smem_addr;
using denseref::ops::stage_commit;
using denseref::ops::stage_tile;
using denseref::ops::stage_wait;
using denseref::ops::store_shared_f32x2;
using denseref::ops::store_shared_u32;
using denseref::ops::warp_sync;

//: THE WINDOW IS AN ARM, so the tile counts it induces are functions of it and -- see docs/internals/intra/intra_kernel.md#tilesof
constexpr int tiles_of(int window) { return window / kTile; }

//: THE OUTPUT BLOCK IS C = `kTile * row_tiles_of(W)` READER TOKENS. -- see docs/internals/intra/intra_kernel.md#note-l31
constexpr int row_tiles_of(int window) { return tiles_of(window) >= 2 ? 2 : 1; }

constexpr int rows_of(int window) { return row_tiles_of(window) * kTile; }

//: ONE WARP PER `kWarpRows` OF THE BLOCK: a warp holds the Gram of ONE tile -- see docs/internals/intra/intra_kernel.md#kwarpspertile
constexpr int threads_of(int window) { return 32 * (rows_of(window) / kWarpRows); }

constexpr int kWarpsPerTile = kTile / kWarpRows;

//: A LEVEL'S STAGED WIDTH is its digit count rounded up to the MMA's -- see docs/internals/intra/intra_kernel.md#kstagecap
constexpr int level_digits(int width) { return width < kSlabDigits ? kSlabDigits : width; }

constexpr int slabs_per_level(int width) { return level_digits(width) / kSlabDigits; }

constexpr int mask_words(int width) { return (level_digits(width) + 31) / 32; }

//: THE PIPELINE STAGE IS AT MOST AS WIDE AS THE LEVEL. A stage is one cp.async -- see docs/internals/intra/intra_kernel.md#kstagecap-2
constexpr int kStageCap = 64;

constexpr int stage_digits(int width) {
  return level_digits(width) < kStageCap ? level_digits(width) : kStageCap;
}

constexpr int slabs_per_stage(int width) { return stage_digits(width) / kSlabDigits; }

constexpr int stages_per_level(int width) { return level_digits(width) / stage_digits(width); }

//: The operand rows are padded so the eight row addresses ldmatrix reads per -- see docs/internals/intra/intra_kernel.md#operandpitch
constexpr int operand_pitch(int width) { return stage_digits(width) + 8; }

constexpr int kValuePitch = kValueWidth + 8;

//: THE DEPOSIT'S STAGING ROW: one 32-channel segment, the row's mass, and the pad that
//: keeps a fragment's 64-bit pair stores bank free -- `kWarpRows` rows a warp, laid in the
//: operand slabs, which are dead once a block's last tile has been consumed.
//: -- see docs/internals/intra/intra_kernel.md#deposit
constexpr int kStageStride = 32 + 8;
constexpr int kStageFloats = kWarpRows * kStageStride;

//: `sread`/`swrite`: the producer's frozen support word for the plane of the same -- see docs/internals/intra/intra_kernel.md#intraparams
struct IntraParams {
  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* v;
  const int32_t* sread;
  const int32_t* swrite;
  float* o;
  float* den;
  int L;
  int row_width;
  int token_words;
  int heads;
  int tokens;
  uint32_t modes;
};

//: THE OPERAND SLABS ARE PER-TILE-PAIR AND DO NOT GROW WITH THE WINDOW: only -- see docs/internals/intra/intra_kernel.md#alignas
template <int D, int B, int W>
struct alignas(16) IntraSmem {
  __nv_bfloat16 rslab[2][rows_of(W)][operand_pitch(B)];
  __nv_bfloat16 wslab[2][kTile][operand_pitch(B)];
  __nv_bfloat16 vslab[kTile][kValuePitch];
  __nv_bfloat16 gain[kTile];
  uint32_t mask[tiles_of(W)][D][2][mask_words(B)];
};

__device__ __forceinline__ uint32_t nonzero_pair(uint32_t w) {
  const uint32_t m = w & kBf16MagnitudeMask;
  return ((m & 0xFFFFu) != 0u ? 1u : 0u) | ((m & 0xFFFF0000u) != 0u ? 2u : 0u);
}

template <int N>
__device__ __forceinline__ uint32_t digit_union_word(const __nv_bfloat16* row) {
  static_assert(N % 8 == 0 && N <= 32, "a certificate word covers 8, 16 or 32 digits");
  uint32_t bits = 0;
  //: §6: `N` is this function's template parameter (renamed from lowercase `n` -- see docs/internals/intra/intra_kernel.md#koctets
  constexpr int kOctets = N / 8;
#pragma unroll
  for (int q = 0; q < kOctets; ++q) {
    const uint4 x = *reinterpret_cast<const uint4*>(row + q * 8);
    bits |= nonzero_pair(x.x) << (q * 8 + 0);
    bits |= nonzero_pair(x.y) << (q * 8 + 2);
    bits |= nonzero_pair(x.z) << (q * 8 + 4);
    bits |= nonzero_pair(x.w) << (q * 8 + 6);
  }
  return bits;
}

//: One bit per digit, unioned over a tile's 64 tokens -- the exact zero -- see docs/internals/intra/intra_kernel.md#build-masks
template <int D, int B, int W>
__device__ void build_masks(IntraSmem<D, B, W>& sm, const IntraParams& p, int bh, int wbase,
                            int tid) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kThreads = threads_of(W);
  constexpr int kWords = mask_words(B);
  constexpr int kWarps = kThreads / 32;
  constexpr int kUnits = kTiles * kWords;
  //: no zero-fill and no leading barrier: every DECLARED (tile, level, side, word) -- see docs/internals/intra/intra_kernel.md#lane
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int word0 = (wbase >> 5) + bh * p.token_words;
#pragma unroll
  for (int l = 0; l < D; ++l) {
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
      if (((p.modes >> (2 * l)) & want) == 0u) continue;
      const int32_t* words = (side == 0) ? p.sread : p.swrite;
      for (int u = warp; u < kUnits; u += kWarps) {
        const int w = u % kWords;
        const int ti = u / kWords;
        const int digit = w * 32 + lane;
        const long long base = (long long)(word0 + ti * (kTile / 32)) * p.row_width + l * B + digit;
        const bool live =
            digit < B && ((uint32_t)words[base] | (uint32_t)words[base + p.row_width]) != 0u;
        const uint32_t bits = __ballot_sync(0xFFFFFFFFu, live);
        if (lane == 0) sm.mask[ti][l][side][w] = bits;
      }
    }
  }
  __syncthreads();
}

template <int D, int B, int W>
__device__ __forceinline__ uint32_t slab_bits(const IntraSmem<D, B, W>& sm, uint32_t modes, int ti,
                                              int l, int side, int slab) {
  const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
  if (((modes >> (2 * l)) & want) == 0u) return 0xFFFFu;
  return (sm.mask[ti][l][side][slab >> 1] >> ((slab & 1) * 16)) & 0xFFFFu;
}

template <int D, int B, int W>
__device__ __forceinline__ bool slab_live(const IntraSmem<D, B, W>& sm, uint32_t modes, int rt,
                                          int ct, int l, int slab) {
  return (slab_bits(sm, modes, rt, l, 0, slab) & slab_bits(sm, modes, ct, l, 1, slab)) != 0u;
}

template <int D, int B, int W>
__device__ __forceinline__ bool stage_live(const IntraSmem<D, B, W>& sm, uint32_t modes, int rt,
                                           int ct, int l, int stage) {
  constexpr int kSubs = slabs_per_stage(B);
  bool any = false;
#pragma unroll
  for (int sub = 0; sub < kSubs; ++sub)
    any = any || slab_live<D, B, W>(sm, modes, rt, ct, l, stage * kSubs + sub);
  return any;
}

//: One bit per row tile of the block: the pairs of this stage with work in them. -- see docs/internals/intra/intra_kernel.md#near-line-185
template <int D, int B, int W>
__device__ __forceinline__ uint32_t live_rows(const IntraSmem<D, B, W>& sm, uint32_t modes, int rt0,
                                              int ct, int l, int stage) {
  uint32_t bits = 0u;
  //: §6: row_tiles_of(W) is compile-time (W is this function's template parameter) -- see docs/internals/intra/intra_kernel.md#krowtiles
  constexpr int kRowTiles = row_tiles_of(W);
#pragma unroll
  for (int r = 0; r < kRowTiles; ++r)
    if (rt0 + r >= ct && stage_live<D, B, W>(sm, modes, rt0 + r, ct, l, stage)) bits |= 1u << r;
  return bits;
}

template <int D, int B, int W>
__device__ __forceinline__ void stage_slabs(IntraSmem<D, B, W>& sm, const IntraParams& p, int bh,
                                            int wbase, int rt0, int ct, int l, int stage, int buf,
                                            int tid, uint32_t rows) {
  constexpr int kStage = stage_digits(B);
  const int digit = l * B + stage * kStage;
  const long long base = (long long)(bh * p.L + wbase) * p.row_width + digit;

  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColDigits = kStage / kColThreads;
  static_assert(kColDigits >= 8, "a staging thread's share of a stage is one cp.async granule");
  //: §6: kColDigits/8 and kStage/16 are compile-time but the heuristic only reads a -- see docs/internals/intra/intra_kernel.md#kcoldigitoctets
  constexpr int kColDigitOctets = kColDigits / 8;
  constexpr int kStageHexadecs = kStage / 16;
  const int ctoken = tid / kColThreads;
  const int cpart = (tid % kColThreads) * kColDigits;
  //: a level's REAL digits only.  Where `B` is narrower than the MMA's contraction -- see docs/internals/intra/intra_kernel.md#near-line-218
  if (cpart < B) {
    const __nv_bfloat16* wsrc =
        p.pwrite + base + (long long)(ct * kTile + ctoken) * p.row_width + cpart;
#pragma unroll
    for (int c = 0; c < kColDigitOctets; ++c)
      stage_tile(smem_addr(&sm.wslab[buf][ctoken][cpart + c * 8]), wsrc + c * 8);
  }

  const int rtoken = (tid >> 1) & (kTile - 1);
  const int rtile = tid / (2 * kTile);
  const int rpart = (tid & 1) * (kStage / 2);
  if (((rows >> rtile) & 1u) && rpart < B) {
    const __nv_bfloat16* rsrc =
        p.pread + base + (long long)((rt0 + rtile) * kTile + rtoken) * p.row_width + rpart;
#pragma unroll
    for (int c = 0; c < kStageHexadecs; ++c)
      stage_tile(smem_addr(&sm.rslab[buf][rtile * kTile + rtoken][rpart + c * 8]), rsrc + c * 8);
  }
}

template <int D, int B, int W>
__device__ __forceinline__ void stage_values(IntraSmem<D, B, W>& sm, const IntraParams& p, int bh,
                                             int wbase, int ct, int tid) {
  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColChannels = kValueWidth / kColThreads;
  //: §6: kColChannels/8 is compile-time but the heuristic only reads a bare -- see docs/internals/intra/intra_kernel.md#kcolchanneloctets
  constexpr int kColChannelOctets = kColChannels / 8;
  const int b = bh / p.heads;
  const int h = bh - b * p.heads;
  const int t0 = wbase + ct * kTile;
  const int token = tid / kColThreads;
  const int part = (tid % kColThreads) * kColChannels;
  const __nv_bfloat16* src =
      p.v + ((long long)(b * p.tokens + t0 + token) * p.heads + h) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < kColChannelOctets; ++c)
    stage_tile(smem_addr(&sm.vslab[token][part + c * 8]), src + c * 8);
  if (tid < kTile / 8) {
    stage_tile(smem_addr(&sm.gain[tid * 8]), p.gwrite + (long long)bh * p.L + t0 + tid * 8);
  }
}

template <int D, int B, int W>
__global__ __launch_bounds__(threads_of(W)) void intra_kernel(IntraParams p) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kRowTiles = row_tiles_of(W);
  constexpr int kStages = stages_per_level(B);
  constexpr int kSubs = slabs_per_stage(B);
  static_assert(kTiles * kTile == W, "the window is a whole number of output tiles");
  static_assert(kTiles % kRowTiles == 0, "the window is a whole number of output blocks");
  static_assert((B & (B - 1)) == 0 && B >= 8, "one uniform power-of-two level width");
  extern __shared__ __align__(16) char raw[];
  IntraSmem<D, B, W>& sm = *reinterpret_cast<IntraSmem<D, B, W>*>(raw);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bh = blockIdx.y;
  const int wbase = blockIdx.x * W;

  //: THE PAD IS ZEROED ONCE, AND ONLY WHERE THERE IS ONE. A level narrower than -- see docs/internals/intra/intra_kernel.md#kfill
  if constexpr (level_digits(B) > B) {
    constexpr int kFill = (int)(sizeof(sm.rslab) + sizeof(sm.wslab)) / (int)sizeof(uint4);
    uint4* z = reinterpret_cast<uint4*>(&sm.rslab[0][0][0]);
    for (int i = tid; i < kFill; i += threads_of(W)) z[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  build_masks<D, B, W>(sm, p, bh, wbase, tid);

  //: The warp's row tile within the block, and its rows within that tile.
  const int mine = (warp / kWarpsPerTile) & (kRowTiles - 1);
  const int arow = warp * kWarpRows + (lane & 15);
  const int trow = (warp % kWarpsPerTile) * kWarpRows;
  const int acol = (lane >> 4) * 8;
  const int row_lo = lane >> 2;
  const int col_lo = (lane & 3) * 2;

  for (int rt0 = 0; rt0 < kTiles; rt0 += kRowTiles) {
    const int rt = rt0 + mine;
    float oacc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    float dacc[2] = {0.f, 0.f};

#pragma unroll 1
    for (int ct = 0; ct < rt0 + kRowTiles; ++ct) {
      __syncthreads();
      stage_values<D, B, W>(sm, p, bh, wbase, ct, tid);
      uint32_t rows = live_rows<D, B, W>(sm, p.modes, rt0, ct, 0, 0);
      if (rows) stage_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, 0, 0, 0, tid, rows);
      stage_commit();

      float gram[D][8][4];
#pragma unroll
      for (int l = 0; l < D; ++l)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) gram[l][n][i] = 0.f;

#pragma unroll
      for (int l = 0; l < D; ++l) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
          const int step = l * kStages + stage;
          const int nl = (stage + 1 == kStages) ? l + 1 : l;
          const int ns = (stage + 1 == kStages) ? 0 : stage + 1;
          uint32_t next_rows = 0u;
          if (nl < D) {
            next_rows = live_rows<D, B, W>(sm, p.modes, rt0, ct, nl, ns);
            if (next_rows)
              stage_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, nl, ns, (step + 1) & 1, tid,
                                   next_rows);
          }
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          if ((rows >> mine) & 1u) {
            const int buf = step & 1;
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) {
              const int slab = stage * kSubs + sub;
              if (!slab_live<D, B, W>(sm, p.modes, rt, ct, l, slab)) continue;
              const int k0 = sub * kSlabDigits + acol;
              uint32_t af[4];
              load_frag(af, smem_addr(&sm.rslab[buf][arow][k0]));
#pragma unroll
              for (int jn = 0; jn < 4; ++jn) {
                uint32_t bf[4];
                load_frag(bf, smem_addr(&sm.wslab[buf][jn * 16 + (lane & 15)][k0]));
                const uint32_t b0[2] = {bf[0], bf[2]};
                const uint32_t b1[2] = {bf[1], bf[3]};
                mma(gram[l][2 * jn], af, b0);
                mma(gram[l][2 * jn + 1], af, b1);
              }
            }
          }
          rows = next_rows;
          __syncthreads();
        }
      }

      if (rt >= ct) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float a = gram[0][n][i];
#pragma unroll
            for (int l = 1; l < D; ++l) a *= gram[l][n][i];
            const int cl = n * 8 + col_lo + (i & 1);
            a *= __bfloat162float(sm.gain[cl]);
            if (rt == ct && cl > trow + row_lo + (i >> 1) * 8) a = 0.f;
            gram[0][n][i] = a;
          }
          dacc[0] += gram[0][n][0] + gram[0][n][1];
          dacc[1] += gram[0][n][2] + gram[0][n][3];
        }

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint32_t av[4] = {pack_bf16x2(gram[0][2 * kb][0], gram[0][2 * kb][1]),
                                  pack_bf16x2(gram[0][2 * kb][2], gram[0][2 * kb][3]),
                                  pack_bf16x2(gram[0][2 * kb + 1][0], gram[0][2 * kb + 1][1]),
                                  pack_bf16x2(gram[0][2 * kb + 1][2], gram[0][2 * kb + 1][3])};
#pragma unroll
          for (int nv = 0; nv < 4; ++nv) {
            uint32_t bf[4];
            load_frag_t(bf, smem_addr(&sm.vslab[kb * 16 + (lane & 15)][nv * 16 + acol]));
            const uint32_t b0[2] = {bf[0], bf[1]};
            const uint32_t b1[2] = {bf[2], bf[3]};
            mma(oacc[nv * 2], av, b0);
            mma(oacc[nv * 2 + 1], av, b1);
          }
        }
      }
    }

    //: THE DEPOSIT, at the line law: the block's rows through the warp's staging tile, a
    //: 32-channel segment at a time, so each reduction is one row's 32 consecutive channels
    //: -- one 128-byte line -- and the sixteen masses one line more. The slabs are dead here:
    //: the block's last stage ended on an edge after every warp's last MMA, and the next
    //: block's fill begins on one.
    static_assert(threads_of(W) / 32 * kStageFloats * (int)sizeof(float)
                      <= (int)(sizeof(IntraSmem<D, B, W>::rslab) + sizeof(IntraSmem<D, B, W>::wslab)
                               + sizeof(IntraSmem<D, B, W>::vslab)),
                  "the warps' staging tiles fit the operand slabs they alias");
    const uint32_t stage =
        smem_addr(&sm.rslab[0][0][0]) + (uint32_t)(warp * kStageFloats * (int)sizeof(float));
#pragma unroll
    for (int s = 1; s <= 2; s *= 2) {
      dacc[0] += __shfl_xor_sync(0xFFFFFFFFu, dacc[0], s);
      dacc[1] += __shfl_xor_sync(0xFFFFFFFFu, dacc[1], s);
    }
    const int grow = wbase + rt * kTile + trow;
    float* const obase = p.o + (long long)(bh * p.L + grow) * kValueWidth + lane;
    constexpr int kSegs = kValueWidth / 32;
#pragma unroll
    for (int seg = 0; seg < kSegs; ++seg) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const uint32_t col = (uint32_t)((n * 8 + col_lo) * 4);
        store_shared_f32x2(stage + (uint32_t)(row_lo * kStageStride * 4) + col,
                           oacc[seg * 4 + n][0], oacc[seg * 4 + n][1]);
        store_shared_f32x2(stage + (uint32_t)((row_lo + 8) * kStageStride * 4) + col,
                           oacc[seg * 4 + n][2], oacc[seg * 4 + n][3]);
      }
      if (seg == 0 && (lane & 3) == 0) {
        store_shared_u32(stage + (uint32_t)((row_lo * kStageStride + 32) * 4),
                         __float_as_uint(dacc[0]));
        store_shared_u32(stage + (uint32_t)(((row_lo + 8) * kStageStride + 32) * 4),
                         __float_as_uint(dacc[1]));
      }
      warp_sync();
#pragma unroll
      for (int r = 0; r < kWarpRows; ++r)
        fan_in_add(
            obase + r * kValueWidth + 32 * seg,
            __uint_as_float(load_shared_u32(stage + (uint32_t)((r * kStageStride + lane) * 4))));
      if (seg == 0 && lane < kWarpRows)
        fan_in_add(
            p.den + (long long)bh * p.L + grow + lane,
            __uint_as_float(load_shared_u32(stage + (uint32_t)((lane * kStageStride + 32) * 4))));
      warp_sync();
    }
  }
}

}  // namespace detail
}  // namespace intra
}  // namespace rola
