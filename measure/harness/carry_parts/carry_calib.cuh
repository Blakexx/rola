// measure/harness/carry_parts/carry_calib.cuh -- THE CALIBRATION KERNELS (KERNEL_STANDARDS §22 (10)): each
// isolates one cost the carry kernel's parts are made of and runs it back to back on every warp of every
// CTA, timed by the host under the locked clock -- an HMMA of the kernel's own atom, a burst of shared
// loads or stores, an asynchronous copy (bank-free or aliased), a global reduction, a CTA barrier, a
// shared-memory barrier, a warp sync; and the SETTLING ROWS, an HMMA burst with the loads or the
// reductions of the kernel's parts interleaved, which say whether those overlap the pipe or share
// it. Built into the part harness's module; never shipped.
// See docs/internals/carry/calibration.md
#pragma once

#include "common/burst.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"

namespace rola::carry::calib {

namespace ops = denseref::ops;

enum CalibMode : int {
  kHmma = 0,
  kSharedLoad = 1,
  kSharedStore = 2,
  kAsyncCopy = 3,
  kAsyncCopyAliased = 4,
  kGlobalReduce = 5,
  kCtaBarrier = 6,
  kShmBarrier = 7,
  kWarpSync = 8,
  kSharedMatrix = 9,
  kHmmaMatrix = 10,
  kHmmaMatrixFree = 11,
  kHmmaReduce = 12,
  kHmmaReduceDiv = 13,
  kHmmaWide = 14,
  kHmmaWideChain = 15,
  kAsyncCopyStrided = 16,
  kAsyncCopy4 = 17,
  kAsyncCopyRows = 18,
  kAsyncCopyLines = 19,
  kSharedMatrixRows = 20,
  kHmmaWideAlu = 21,
  kHmmaQueue = 22,
  kHmmaWideChainHooked = 23,
  kHmmaWideChainHooked2 = 24,
  kHmmaLatency = 25,
  kHmmaWideOperands = 26,
  kAsyncCopyZfillSink = 27,
  kAsyncCopyZfillSpread = 28,
  kAsyncCopyMixedSink = 29,
  kAsyncCopy4ZfillSink = 30,
  kAsyncCopyLanes = 31,
  kAsyncCopy4Lanes = 32,
  kLdsChain = 33,
  kShflChain = 34,
  kLdsmChain = 35,
  kBranchTaken = 36,
  kBranchDivergent = 37,
  kIcache = 38,
  kAsyncCopyLatency = 39,
  kPairPhase = 40
};

//: A LINE REPEATED, for the instruction-cache rows: `ROLA_REP<n>(x)` is `n` copies of the string `x`.
#define ROLA_REP2(x) x x
#define ROLA_REP4(x) ROLA_REP2(x) ROLA_REP2(x)
#define ROLA_REP8(x) ROLA_REP4(x) ROLA_REP4(x)
#define ROLA_REP16(x) ROLA_REP8(x) ROLA_REP8(x)
#define ROLA_REP32(x) ROLA_REP16(x) ROLA_REP16(x)
#define ROLA_REP64(x) ROLA_REP32(x) ROLA_REP32(x)
#define ROLA_REP128(x) ROLA_REP64(x) ROLA_REP64(x)
#define ROLA_REP256(x) ROLA_REP128(x) ROLA_REP128(x)
#define ROLA_REP512(x) ROLA_REP256(x) ROLA_REP256(x)
#define ROLA_REP1024(x) ROLA_REP512(x) ROLA_REP512(x)
#define ROLA_REP2048(x) ROLA_REP1024(x) ROLA_REP1024(x)

constexpr int kCalibSmemBytes = 96416;

struct CalibParams {
  int iters;
  float* out;       //: the reduction target, `[owners][16 rows][256 lanes]` floats
  const char* src;  //: the copy source, `[owners][256 lanes][128]` bytes
};

//: ONE CALIBRATION: `Mode` run `iters` times by every warp, `Burst` operations a unit where a unit has
//: several; each warp's results reach a stored sink so nothing is dead.
template <int Warps, int Mode, int Burst>
__global__ __launch_bounds__(Warps * 32, 1) void calib_kernel(
    __grid_constant__ const CalibParams c) {
  extern __shared__ __align__(16) char smem_raw[];
  const ops::SmemAddr base = ops::smem_addr(smem_raw);
  const int owner = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<Warps>();
  const int lane = tid & 31;
  //: a warp's own shared lines: `Burst` lines of 128 bytes each, a line a bank line.
  const ops::SmemAddr lines = base + (uint32_t)(warp * 128 * 64);

  if constexpr (Mode == kHmma) {
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t ab[4] = {w, w, w, w};
    const uint32_t bf[4] = {w, w, w, w};
    float y[8][4], d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<8>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      ops::mma(d, ab, bf);
    }
    float sink = d[0] + d[1] + d[2] + d[3];
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines, __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaWideChainHooked) {
    //: THE GRANULAR FORK: the fragment's burst (eighteen HMMAs) with the NEXT unit's gather chain
    //: cut in four pieces, each pinned into a three-HMMA window by two dependencies -- its input
    //: after an HMMA's result (`rola::burst::after` on the accumulator), its output hooked into
    //: a later HMMA's B -- so the schedule is HMMA, piece, HMMA, piece and no piece's latency
    //: sits outside a shadow. The chain and its operands are `kHmmaWideChain`'s.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    rola::static_for<4>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4), w);
    });
    __syncwarp();
    const uint32_t bf[4] = {w, w, w, w};
    uint32_t ab[4] = {w, w, w, w};
    float y[18][4];
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    uint32_t entry = (uint32_t)lane;
    const auto hmma = [&](int j, uint32_t hook) {
      uint32_t b[4] = {bf[0], bf[1], bf[2], bf[3]};
      b[0] = rola::burst::after(b[0], hook);
      ops::mma(y[j], ab, b);
    };
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      //: piece 1: the next entry's rows -- a shuffle, two loads (issued at the top: loads first)
      const uint32_t re =
          (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, (lane & 7) + 8 * (lane >> 4));
      uint32_t a[4], sp[4];
      ops::load_frag_t(a, lines + (uint32_t)((re & 1u) * 128));
      ops::load_frag_t(sp, lines + (uint32_t)(256 + (re & 1u) * 128));
      ops::mma(y[0], ab, bf);
      ops::mma(y[1], ab, bf + 2);
      ops::mma(y[2], ab, bf);
      hmma(3, a[0] ^ sp[0] ^ re);  //: the loads landed under HMMAs 0-2
      //: piece 2: the gain, after HMMA 3's result
      const uint32_t gin = rola::burst::after(entry, __float_as_uint(y[3][0]));
      const uint32_t g = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)gin, 2 * (lane & 3));
      sp[0] = ops::mul_bf16x2(sp[0], g);
      sp[2] = ops::mul_bf16x2(sp[2], g);
      ops::mma(y[4], ab, bf);
      ops::mma(y[5], ab, bf + 2);
      hmma(6, sp[0] ^ sp[2]);
      //: piece 3: the scaled pair to the lanes, after HMMA 6's result
      const uint32_t s0in = rola::burst::after(sp[0], __float_as_uint(y[6][0]));
      const uint32_t s0 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)s0in, lane & 3);
      const uint32_t s1 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[2], lane & 3);
      ops::mma(y[7], ab, bf + 2);
      ops::mma(y[8], ab, bf);
      hmma(9, s0 ^ s1);
      //: piece 4: the next A, after HMMA 9's result
      a[0] = rola::burst::after(a[0], __float_as_uint(y[9][0]));
      uint32_t nab[4];
      nab[0] = ops::mul_bf16x2(a[0], s0);
      nab[1] = ops::mul_bf16x2(a[1], s0);
      nab[2] = ops::mul_bf16x2(a[2], s1);
      nab[3] = ops::mul_bf16x2(a[3], s1);
      ops::mma(y[10], ab, bf);
      ops::mma(y[11], ab, bf + 2);
      hmma(12, nab[0] ^ nab[3]);
      rola::static_for<5>([&](auto Jc) {
        constexpr int j = 13 + decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      rola::static_for<4>([&](auto Ec) { ab[decltype(Ec)::value] = nab[decltype(Ec)::value]; });
      entry += 1u;
    }
    float sink = 0.0f;
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(4 * 128 + lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaWideOperands) {
    //: THE FRAGMENT'S BURST WITH ITS OPERAND PATTERN: eighteen HMMAs a unit into eighteen
    //: accumulators, A one of two register sets a box, B a DIFFERENT register pair every n-tile
    //: (the fold's sixteen V registers) and a ones pair for the mass -- all resident, no loads:
    //: the unit's rate when no operand is reused across HMMAs, against `hmma_wide`'s constants.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    uint32_t ab[2][4], v[16], bm[2] = {w, w};
    rola::static_for<2>([&](auto Bc) {
      rola::static_for<4>([&](auto Ec) {
        ab[decltype(Bc)::value][decltype(Ec)::value] =
            w ^ (uint32_t)(decltype(Bc)::value * 4 + decltype(Ec)::value);
      });
    });
    rola::static_for<16>(
        [&](auto Ic) { v[decltype(Ic)::value] = w ^ (uint32_t)(16 + decltype(Ic)::value); });
    float y[2][9][4];
    rola::static_for<2>([&](auto Bc) {
      rola::static_for<9>([&](auto Jc) {
        rola::static_for<4>([&](auto Ec) {
          y[decltype(Bc)::value][decltype(Jc)::value][decltype(Ec)::value] = 0.0f;
        });
      });
    });
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<2>([&](auto Bc) {
        constexpr int bi = decltype(Bc)::value;
        rola::static_for<8>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          ops::mma(y[bi][j], ab[bi], v + 2 * j);
        });
        ops::mma(y[bi][8], ab[bi], bm);
      });
    }
    float sink = 0.0f;
    rola::static_for<2>([&](auto Bc) {
      rola::static_for<9>([&](auto Jc) {
        rola::static_for<4>([&](auto Ec) {
          sink += y[decltype(Bc)::value][decltype(Jc)::value][decltype(Ec)::value];
        });
      });
    });
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaLatency) {
    //: THE HMMA'S COMPLETION LATENCY: eighteen HMMAs a unit into ONE accumulator, each dependent
    //: on the last -- cycles an HMMA is the latency, not the pipe's rate.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t bf[4] = {w, w, w, w};
    const uint32_t ab[4] = {w, w, w, w};
    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<18>([&](auto Jc) { ops::mma(y, ab, bf + 2 * (decltype(Jc)::value & 1)); });
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), __float_as_uint(y[0] + y[1] + y[2] + y[3]));
  }

  if constexpr (Mode == kHmmaWideChainHooked2) {
    //: THE GRANULAR FORK, second pinning: each piece's INPUT after an HMMA's result two ahead of
    //: its window, and its HOOK VALUE (the XOR the hooked HMMA's B waits on) after the result of
    //: the HMMA two before the hooked one -- so ptxas can neither hoist a piece to the top nor
    //: compute a hook while its loads are in flight (the first pinning left a 30-cycle wait at
    //: the loads' XOR, placed right behind them).
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    rola::static_for<4>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4), w);
    });
    __syncwarp();
    const uint32_t bf[4] = {w, w, w, w};
    uint32_t ab[4] = {w, w, w, w};
    float y[18][4];
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    uint32_t entry = (uint32_t)lane;
    const auto hmma = [&](int j, uint32_t hook) {
      const uint32_t b[4] = {rola::burst::after(bf[0], hook), bf[1], bf[2], bf[3]};
      ops::mma(y[j], ab, b);
    };
    const auto res = [&](int j) { return __float_as_uint(y[j][0]); };
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      const uint32_t re =
          (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, (lane & 7) + 8 * (lane >> 4));
      uint32_t a[4], sp[4];
      ops::load_frag_t(a, lines + (uint32_t)((re & 1u) * 128));
      ops::load_frag_t(sp, lines + (uint32_t)(256 + (re & 1u) * 128));
      ops::mma(y[0], ab, bf);
      ops::mma(y[1], ab, bf + 2);
      ops::mma(y[2], ab, bf);
      hmma(3, rola::burst::after(a[0] ^ sp[0] ^ re, res(1)));
      const uint32_t gin = rola::burst::after(entry, res(2));
      const uint32_t g = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)gin, 2 * (lane & 3));
      sp[0] = ops::mul_bf16x2(sp[0], g);
      sp[2] = ops::mul_bf16x2(sp[2], g);
      ops::mma(y[4], ab, bf);
      ops::mma(y[5], ab, bf + 2);
      hmma(6, rola::burst::after(sp[0] ^ sp[2], res(4)));
      const uint32_t s0in = rola::burst::after(sp[0], res(5));
      const uint32_t s0 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)s0in, lane & 3);
      const uint32_t s1 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[2], lane & 3);
      ops::mma(y[7], ab, bf + 2);
      ops::mma(y[8], ab, bf);
      hmma(9, rola::burst::after(s0 ^ s1, res(7)));
      a[0] = rola::burst::after(a[0], res(8));
      uint32_t nab[4];
      nab[0] = ops::mul_bf16x2(a[0], s0);
      nab[1] = ops::mul_bf16x2(a[1], s0);
      nab[2] = ops::mul_bf16x2(a[2], s1);
      nab[3] = ops::mul_bf16x2(a[3], s1);
      ops::mma(y[10], ab, bf);
      ops::mma(y[11], ab, bf + 2);
      hmma(12, rola::burst::after(nab[0] ^ nab[3], res(10)));
      rola::static_for<5>([&](auto Jc) {
        constexpr int j = 13 + decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      rola::static_for<4>([&](auto Ec) { ab[decltype(Ec)::value] = nab[decltype(Ec)::value]; });
      entry += 1u;
    }
    float sink = 0.0f;
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(4 * 128 + lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaQueue) {
    //: THE TENSOR PIPE'S QUEUE DEPTH: eighteen HMMAs a unit, then ONE dependent chain of `Burst` fp32
    //: adds (~5 cycles a link) that nothing can interleave with the HMMAs. A warp that can post q HMMAs
    //: ahead of the pipe runs q x 32.5 cycles of the chain under them; the exposed part of the chain,
    //: read against the burst's pipe time, is the depth.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t bf[4] = {w, w, w, w};
    const uint32_t ab[4] = {w, w, w, w};
    float y[18][4];
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    float m = 0.0f;
    const float d = __uint_as_float(w & 0xFFFF0000u);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<18>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      //: the chain as volatile asm: ptxas keeps it AFTER the burst's HMMAs (which are volatile asm too), so
      //: what is hidden is hidden by HMMAs already posted, never by interleaving
      rola::static_for<Burst>([&](auto Kc) {
        asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                     : "+f"(m)
                     : "f"(d), "f"(__int_as_float((int)i + decltype(Kc)::value)));
      });
    }
    float sink = m;
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaWideAlu) {
    //: THE FRAGMENT'S BURST WITH ALU WORK BESIDE IT: eighteen HMMAs a unit into eighteen accumulators
    //: and `Burst` fp32 adds a unit in four independent chains (the mass-on-FMA form's unpack-and-add,
    //: 32 a fragment), the adds placed after the HMMAs as ptxas placed them there: what an ALU
    //: instruction costs a warp whose HMMAs hold the pipe.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t bf[4] = {w, w, w, w};
    const uint32_t ab[4] = {w, w, w, w};
    float y[18][4];
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float d = __uint_as_float(w & 0xFFFF0000u);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<18>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        m[k & 3] += d + __int_as_float((int)i + k);
      });
    }
    float sink = m[0] + m[1] + m[2] + m[3];
    rola::static_for<18>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaWide || Mode == kHmmaWideChain) {
    //: THE FRAGMENT'S BURST: `Burst` HMMAs a unit into `Burst` distinct accumulators (the fold's
    //: eighteen: two boxes' eight n-tiles and a mass each), A from a register; under
    //: `kHmmaWideChain` each unit's A comes off the fold's gather chain first: a shuffle, an
    //: `ldmatrix`, a packed multiply by a shuffled pair, a second multiply.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    rola::static_for<4>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4), w);
    });
    __syncwarp();
    const uint32_t bf[4] = {w, w, w, w};
    uint32_t ab[4] = {w, w, w, w};
    float y[Burst][4];
    rola::static_for<Burst>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    uint32_t entry = (uint32_t)lane;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if constexpr (Mode == kHmmaWideChain) {
        const uint32_t re =
            (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, (lane & 7) + 8 * (lane >> 4));
        uint32_t a[4], sp[4];
        ops::load_frag_t(a, lines + (uint32_t)((re & 1u) * 128));
        ops::load_frag_t(sp, lines + (uint32_t)(256 + (re & 1u) * 128));
        const uint32_t g = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, 2 * (lane & 3));
        sp[0] = ops::mul_bf16x2(sp[0], g);
        sp[2] = ops::mul_bf16x2(sp[2], g);
        const uint32_t s0 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[0], lane & 3);
        const uint32_t s1 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[2], lane & 3);
        ab[0] = ops::mul_bf16x2(a[0], s0);
        ab[1] = ops::mul_bf16x2(a[1], s0);
        ab[2] = ops::mul_bf16x2(a[2], s1);
        ab[3] = ops::mul_bf16x2(a[3], s1);
        entry += 1u;
      }
      rola::static_for<Burst>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
    }
    float sink = 0.0f;
    rola::static_for<Burst>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(4 * 128 + lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kSharedLoad || Mode == kSharedStore) {
    rola::static_for<Burst>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4),
                            (uint32_t)tid);
    });
    __syncwarp();
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        const ops::SmemAddr at = lines + (uint32_t)(k * 128 + lane * 4);
        if constexpr (Mode == kSharedLoad)
          acc ^= ops::load_shared_u32(at);
        else
          ops::store_shared_u32(at, (uint32_t)i ^ (uint32_t)k);
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(Burst * 128 + lane * 4), acc);
  }

  if constexpr (Mode == kSharedMatrixRows) {
    //: `Burst` `ldmatrix.trans` loads a unit AS THE KERNEL ISSUES THEM: a lane its own row of a
    //: 128-byte-row tile, the chunk swizzled by the row, so each of the four matrices reads eight
    //: distinct bank groups -- four wavefronts a load (the broadcast row below is one).
    rola::static_for<8>([&](auto Rc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Rc)::value * 128 + lane * 4),
                            (uint32_t)tid);
    });
    __syncwarp();
    const uint32_t at = (uint32_t)((lane & 7) * 128 + (((lane >> 3) ^ (lane & 7)) & 7) * 16);
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        uint32_t r[4];
        ops::load_frag_t(r, lines + (uint32_t)((decltype(Kc)::value & 7) * 1024) + at);
        acc ^= r[0] ^ r[3];
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(8 * 1024 + lane * 4), acc);
  }

  if constexpr (Mode == kSharedMatrix) {
    //: `Burst` two-n-tile `ldmatrix.trans` loads a unit off ONE line each (every lane the same
    //: address: a broadcast, one wavefront a load): the load's latency and issue, not the
    //: kernel's bandwidth -- `kSharedMatrixRows` is the kernel's pattern.
    rola::static_for<Burst>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4),
                            (uint32_t)tid);
    });
    __syncwarp();
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        uint32_t r[4];
        ops::load_frag_t(r, lines + (uint32_t)(decltype(Kc)::value * 128));
        acc ^= r[0] ^ r[3];
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(Burst * 128 + lane * 4), acc);
  }

  if constexpr (Mode == kHmmaMatrix || Mode == kHmmaMatrixFree) {
    //: THE LOADS UNDER THE BURST: `Burst` two-n-tile `ldmatrix.trans` loads a unit ahead of nine
    //: HMMAs. Under `kHmmaMatrix` the HMMAs take their B from the loads in turn (the readout's box,
    //: the fold's fragment: the dependency the kernel has); under `kHmmaMatrixFree` the loads'
    //: results go to a sink and the HMMAs take a register B, so only the pipes' sharing is timed.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    rola::static_for<Burst>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4), w);
    });
    __syncwarp();
    const uint32_t ab[4] = {w, w, w, w};
    const uint32_t bf[4] = {w, w, w, w};
    float y[8][4], d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      uint32_t r[Burst][4];
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        ops::load_frag_t(r[k], lines + (uint32_t)(k * 128));
      });
      if constexpr (Mode == kHmmaMatrix) {
        rola::static_for<8>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          ops::mma(y[j], ab, r[j % Burst] + 2 * (j & 1));
        });
        ops::mma(d, ab, r[0]);
      } else {
        rola::static_for<Burst>([&](auto Kc) {
          constexpr int k = decltype(Kc)::value;
          acc ^= r[k][0] ^ r[k][3];
        });
        rola::static_for<8>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          ops::mma(y[j], ab, bf + 2 * (j & 1));
        });
        ops::mma(d, ab, bf);
      }
    }
    float sink = d[0] + d[1] + d[2] + d[3] + __uint_as_float(acc);
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(Burst * 128 + lane * 4), __float_as_uint(sink));
  }

  if constexpr (Mode == kHmmaReduce || Mode == kHmmaReduceDiv) {
    //: THE REDUCTIONS UNDER THE BURST: `Burst` global f32 reductions a unit, spread between its
    //: nine HMMAs (the drain's fire-and-forget adds beside the readout's boxes): does their issue
    //: hold the HMMAs back? Coalesced (a warp's red one 128-byte line) or DIVERGENT in the
    //: accumulator's own shape (lane `r, q` at row `r`, floats `2 q`: eight rows' sectors a red).
    static_assert(Burst >= 1 && Burst <= 8 && (8 % Burst) == 0,
                  "a reduction every 8 / Burst HMMAs");
    float* const at = ops::pin_address(
        c.out + (long)owner * 128 * 256
        + (Mode == kHmmaReduce ? (long)tid
                               : (long)(warp * 8 + (lane >> 2)) * 256 + (lane & 3) * 2));
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t ab[4] = {w, w, w, w};
    const uint32_t bf[4] = {w, w, w, w};
    float y[8][4], d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<8>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
        if constexpr (j % (8 / Burst) == 0)
          ops::red_global_add_f32(at + (long)((i + j) & 15) * (Mode == kHmmaReduce ? 256 : 8),
                                  1.0f);
      });
      ops::mma(d, ab, bf);
    }
    float sink = d[0] + d[1] + d[2] + d[3];
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines, __float_as_uint(sink));
  }

  if constexpr (Mode == kAsyncCopy || Mode == kAsyncCopyAliased) {
    //: `Burst` 16-byte runs a unit into the warp's lines: bank-free (each run its own line and lane
    //: offset) or ALIASED (every run's lane at the same 128-byte stride, one bank group).
    const char* const row = c.src + (long)owner * 256 * 16 + (long)tid * 16;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        const uint32_t off =
            Mode == kAsyncCopy ? (uint32_t)(k * 128 + (lane & 7) * 16) : (uint32_t)(k * 128);
        ops::stage_run<16>(lines + off + (uint32_t)((lane >> 3) * 128 * Burst), row);
      });
      ops::stage_commit();
      ops::stage_wait<0>();
    }
  }

  if constexpr (Mode == kAsyncCopyStrided || Mode == kAsyncCopy4 || Mode == kAsyncCopyRows
                || Mode == kAsyncCopyLines) {
    //: the source side of a landing, `[owners][256 lanes][128]` bytes, a lane a 128-byte line:
    //: STRIDED, the bank-free destination fed from a line a lane (sixteen-byte runs); FOUR, four-byte
    //: runs a lane, one contiguous line in and out; ROWS, the pool fill's V pattern (a lane a token row
    //: at `j`, its half at `h`, two adjacent chunks of sixteen rows a run, the channel-row swizzle);
    //: LINES, the same rows landed a whole line at a time (eight lanes a row, four rows a run).
    const char* const block = c.src + (long)owner * 256 * 128;
    const int j = lane & 15, h = lane >> 4;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        if constexpr (Mode == kAsyncCopyStrided) {
          const uint32_t off = (uint32_t)(k * 128 + (lane & 7) * 16 + (lane >> 3) * 128 * Burst);
          ops::stage_run<16>(lines + off, block + (long)tid * 128 + k * 16);
        }
        if constexpr (Mode == kAsyncCopy4) {
          ops::stage_run<4>(lines + (uint32_t)(k * 128 + lane * 4),
                            block + (long)k * 128 + lane * 4);
        }
        if constexpr (Mode == kAsyncCopyRows) {
          const int chunk = 2 * k + h;
          ops::stage_run<16>(lines + (uint32_t)(j * 128 + ((chunk ^ (j & 7)) * 16)),
                             block + (long)(warp * 16 + j) * 128 + chunk * 16);
        }
        if constexpr (Mode == kAsyncCopyLines) {
          const int r = 4 * k + (lane >> 3), chunk = lane & 7;
          ops::stage_run<16>(lines + (uint32_t)(r * 128 + ((chunk ^ (r & 7)) * 16)),
                             block + (long)(warp * 16 + r) * 128 + chunk * 16);
        }
      });
      ops::stage_commit();
      ops::stage_wait<0>();
    }
  }

  if constexpr (Mode == kAsyncCopyZfillSink || Mode == kAsyncCopyZfillSpread
                || Mode == kAsyncCopyMixedSink || Mode == kAsyncCopy4ZfillSink
                || Mode == kAsyncCopyLanes || Mode == kAsyncCopy4Lanes) {
    //: the fill's DEAD LANES: a zero-size copy lands sixteen zero bytes and reads nothing. SINK,
    //: every lane's landing one slot (the pool's zero row, as the fill lands them); SPREAD, a slot
    //: a lane; MIXED, four lanes live from their own rows and twenty-eight dead to the one slot (a
    //: sparse round); FOUR, sixteen lanes' zero-size four-byte copies to one word (a dead gain pair).
    const char* const block = c.src + (long)owner * 256 * 128;
    const int j = lane & 15, h = lane >> 4;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        if constexpr (Mode == kAsyncCopyZfillSink) {
          ops::stage_run_if<16>(lines + (uint32_t)(k * 16), block + k * 16, false);
        }
        if constexpr (Mode == kAsyncCopyZfillSpread) {
          ops::stage_run_if<16>(lines + (uint32_t)(k * 512 + lane * 16), block + k * 16, false);
        }
        if constexpr (Mode == kAsyncCopyMixedSink) {
          const bool in = lane < 4;
          const int chunk = 2 * k + h, row = in ? j : 16;
          ops::stage_run_if<16>(lines + (uint32_t)(row * 128 + ((chunk ^ (row & 7)) * 16)),
                                block + (long)(warp * 16 + (in ? j : 0)) * 128 + chunk * 16, in);
        }
        if constexpr (Mode == kAsyncCopy4ZfillSink) {
          if (h == 0) ops::stage_run_if<4>(lines + (uint32_t)(k * 4), block + k * 4, false);
        }
        //: LANES: the MIXED round with its dead lanes predicated off instead of zero-size (the
        //: fill's dead lanes land on a zero row that is zero already); FOUR LANES, two of sixteen
        //: lanes' four-byte copies live, the rest off.
        if constexpr (Mode == kAsyncCopyLanes) {
          const bool in = lane < 4;
          const int chunk = 2 * k + h, row = in ? j : 16;
          ops::stage_run_lanes<16>(lines + (uint32_t)(row * 128 + ((chunk ^ (row & 7)) * 16)),
                                   block + (long)(warp * 16 + (in ? j : 0)) * 128 + chunk * 16, in);
        }
        if constexpr (Mode == kAsyncCopy4Lanes) {
          if (h == 0)
            ops::stage_run_lanes<4>(lines + (uint32_t)(k * 128 + lane * 4),
                                    block + (long)k * 128 + lane * 4, lane < 2);
        }
      });
      ops::stage_commit();
      ops::stage_wait<0>();
    }
  }

  if constexpr (Mode == kLdsChain || Mode == kLdsmChain) {
    //: DEPENDENT SHARED LOADS: each load's address is the last load's result plus the lane's own offset; the
    //: words read are zero, so the address never moves and every load waits on the one before (the latency).
    //: LDS a lane its own word; LDSM a lane its own sixteen-byte row, one matrix a load.
    for (int i = lane; i < 64 * 32; i += 32) ops::store_shared_u32(lines + (uint32_t)(i * 4), 0u);
    __syncwarp();
    uint32_t v = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto) {
        if constexpr (Mode == kLdsChain) {
          v = ops::load_shared_u32(lines + (uint32_t)(lane * 4) + v);
        } else {
          uint32_t r;
          asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                       : "=r"(r)
                       : "r"(lines + (uint32_t)((lane & 7) * 16) + v));
          v = r;
        }
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(64 * 32 * 4 + lane * 4), v);
  }

  if constexpr (Mode == kShflChain) {
    //: DEPENDENT SHUFFLES: each shuffle's value is the last one's result.
    uint32_t v = (uint32_t)lane;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto) { v = (uint32_t)__shfl_xor_sync(0xFFFFFFFFu, (int)v, 1); });
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), v);
  }

  if constexpr (Mode == kBranchTaken) {
    //: A TAKEN UNIFORM BRANCH: `Burst` a unit, each over a block of sixteen dependent adds it skips (too large for
    //: ptxas to predicate); the predicate is a runtime value the compiler cannot know.
    const uint32_t one = c.iters > 0 ? 1u : 0u;
    uint32_t acc = (uint32_t)lane;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto) {
        asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %1, 0;\n@p bra.uni SKIP%=;\n" ROLA_REP16(
                         "add.u32 %0, %0, %1;\n") "SKIP%=:\n}\n"
                     : "+r"(acc)
                     : "r"(one));
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), acc);
  }

  if constexpr (Mode == kBranchDivergent) {
    //: A DIVERGENT BRANCH AND ITS RECONVERGENCE: odd and even lanes take different blocks of eight dependent adds,
    //: `Burst` a unit; the cost over two blocks' adds is the divergence's.
    uint32_t acc = (uint32_t)lane;
    const uint32_t odd = (uint32_t)lane & 1u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto) {
        if (odd) {
          asm volatile(ROLA_REP8("add.u32 %0, %0, %1;\n") : "+r"(acc) : "r"(odd));
        } else {
          asm volatile(ROLA_REP8("xor.b32 %0, %0, %1;\n") : "+r"(acc) : "r"(odd + 3u));
        }
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), acc);
  }

  if constexpr (Mode == kIcache) {
    //: THE INSTRUCTION CACHE: a loop body of `Burst` multiply-adds over four independent accumulators (issue-bound:
    //: IMAD's latency is covered), so the body's code is `Burst` x 16 bytes; cycles an instruction rise where the
    //: body outgrows a cache level.
    uint32_t a0 = (uint32_t)lane, a1 = a0 + 1u, a2 = a0 + 2u, a3 = a0 + 3u;
    const uint32_t m = (uint32_t)c.iters | 1u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if constexpr (Burst == 256) {
        asm volatile(
            ROLA_REP64(
                "mad.lo.u32 %0, %0, %4, %4;\nmad.lo.u32 %1, %1, %4, %4;\nmad.lo.u32 %2, %2, %4, %4;\nmad.lo.u32 %3, %3, %4, %4;\n")
            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
            : "r"(m));
      } else if constexpr (Burst == 1024) {
        asm volatile(
            ROLA_REP256(
                "mad.lo.u32 %0, %0, %4, %4;\nmad.lo.u32 %1, %1, %4, %4;\nmad.lo.u32 %2, %2, %4, %4;\nmad.lo.u32 %3, %3, %4, %4;\n")
            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
            : "r"(m));
      } else if constexpr (Burst == 2048) {
        asm volatile(
            ROLA_REP512(
                "mad.lo.u32 %0, %0, %4, %4;\nmad.lo.u32 %1, %1, %4, %4;\nmad.lo.u32 %2, %2, %4, %4;\nmad.lo.u32 %3, %3, %4, %4;\n")
            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
            : "r"(m));
      } else if constexpr (Burst == 4096) {
        asm volatile(
            ROLA_REP1024(
                "mad.lo.u32 %0, %0, %4, %4;\nmad.lo.u32 %1, %1, %4, %4;\nmad.lo.u32 %2, %2, %4, %4;\nmad.lo.u32 %3, %3, %4, %4;\n")
            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
            : "r"(m));
      } else if constexpr (Burst == 8192) {
        asm volatile(
            ROLA_REP2048(
                "mad.lo.u32 %0, %0, %4, %4;\nmad.lo.u32 %1, %1, %4, %4;\nmad.lo.u32 %2, %2, %4, %4;\nmad.lo.u32 %3, %3, %4, %4;\n")
            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
            : "r"(m));
      }
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), a0 ^ a1 ^ a2 ^ a3);
  }

  if constexpr (Mode == kAsyncCopyLatency) {
    //: A COPY GROUP'S ROUND TRIP: one sixteen-byte `cp.async.cg` a lane (the warp's 512 bytes, four lines, from L2:
    //: the source is 2.6 MB), committed and waited on before the next -- cycles a group is the landing latency.
    const char* const block = c.src + (long)owner * 256 * 128;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      ops::stage_run<16>(lines + (uint32_t)(lane * 16),
                         block + (long)((i * 8 + warp) & 255) * 128 + (lane & 7) * 16);
      ops::stage_commit();
      ops::stage_wait<0>();
    }
  }

  if constexpr (Mode == kPairPhase) {
    //: THE SCHEDULER'S ARBITRATION: each warp loops a burst of `Burst` HMMAs into distinct accumulators, then 48
    //: dependent multiply-adds (its own latency, no pipe). A fair scheduler keeps a pair in step and idles the pipe
    //: through both stretches (a period of two bursts plus one stretch); a greedy one gives a warp its whole burst and
    //: hides the partner's stretch under it (two bursts, the partner a burst behind). Lane 0 stamps each iteration's
    //: start into `c.out` (64 a warp) so the host reads each pair's settled offset.
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t bf[4] = {w, w, w, w};
    const uint32_t ab[4] = {w, w, w, w};
    float y[Burst][4];
    rola::static_for<Burst>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
    uint32_t x = (uint32_t)lane;
    const uint32_t m = (uint32_t)lane | 1u;
    unsigned int* const stamps = reinterpret_cast<unsigned int*>(c.out + (long)owner * 16 * 256);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if (lane == 0 && i < 64) stamps[warp * 64 + i] = (unsigned int)clock64();
      rola::static_for<Burst>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      asm volatile(ROLA_REP32("mad.lo.u32 %0, %0, %1, %1;\n")
                       ROLA_REP16("mad.lo.u32 %0, %0, %1, %1;\n")
                   : "+r"(x)
                   : "r"(m));
    }
    float sink = 0.0f;
    rola::static_for<Burst>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), __float_as_uint(sink) ^ x);
  }

  if constexpr (Mode == kGlobalReduce) {
    float* const at = ops::pin_address(c.out + (long)owner * 16 * 256 + tid);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      //: a row an iteration of sixteen, so no reduction repeats the one before it at its address.
      rola::static_for<Burst>([&](auto Kc) {
        ops::red_global_add_f32(at + (long)((i + decltype(Kc)::value) & 15) * 256, 1.0f);
      });
    }
  }

  if constexpr (Mode == kCtaBarrier) {
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      acc += (uint32_t)i;
      ops::rendezvous();
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), acc);
  }

  if constexpr (Mode == kShmBarrier) {
    const ops::SmemAddr bar = base + (uint32_t)(Warps * 128 * 64 + warp * 8);
    if (lane == 0) ops::mbar_init(bar, 1u);
    __syncwarp();
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if (lane == 0) ops::mbar_arrive(bar);
      ops::mbar_wait(bar, (uint32_t)i & 1u);
    }
  }

  if constexpr (Mode == kWarpSync) {
    const ops::SmemAddr at = lines + (uint32_t)(lane * 4);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if (lane == 0) ops::store_shared_u32(lines, (uint32_t)i);
      __syncwarp();
      ops::store_shared_u32(at, ops::load_shared_u32(lines));
    }
  }
}

//: THE INSTANTIATED CALIBRATIONS: (warps, mode, burst). The host picks one by value.
#define ROLA_CALIB_SET_X(F)          \
  F(8, kHmma, 1)                     \
  F(4, kHmma, 1)                     \
  F(8, kSharedLoad, 4)               \
  F(8, kSharedLoad, 16)              \
  F(8, kSharedLoad, 64)              \
  F(8, kSharedStore, 4)              \
  F(8, kSharedStore, 16)             \
  F(8, kSharedStore, 64)             \
  F(8, kSharedMatrix, 4)             \
  F(8, kSharedMatrix, 16)            \
  F(8, kSharedMatrixRows, 4)         \
  F(8, kSharedMatrixRows, 16)        \
  F(4, kSharedMatrixRows, 4)         \
  F(8, kHmmaMatrix, 1)               \
  F(8, kHmmaMatrix, 4)               \
  F(4, kHmmaMatrix, 4)               \
  F(8, kHmmaMatrixFree, 4)           \
  F(8, kHmmaReduce, 1)               \
  F(8, kHmmaReduce, 2)               \
  F(8, kHmmaReduce, 4)               \
  F(8, kHmmaReduce, 8)               \
  F(8, kHmmaReduceDiv, 4)            \
  F(8, kHmmaReduceDiv, 8)            \
  F(4, kHmmaWide, 18)                \
  F(8, kHmmaWide, 18)                \
  F(4, kHmmaWideChain, 18)           \
  F(8, kHmmaWideChain, 18)           \
  F(4, kHmmaWideAlu, 32)             \
  F(8, kHmmaWideAlu, 32)             \
  F(4, kHmmaWideAlu, 64)             \
  F(8, kHmmaWideAlu, 64)             \
  F(4, kHmmaQueue, 16)               \
  F(4, kHmmaQueue, 40)               \
  F(4, kHmmaQueue, 80)               \
  F(8, kHmmaQueue, 40)               \
  F(4, kHmmaWideChainHooked, 18)     \
  F(8, kHmmaWideChainHooked, 18)     \
  F(4, kHmmaWideChainHooked2, 18)    \
  F(8, kHmmaWideChainHooked2, 18)    \
  F(4, kHmmaLatency, 18)             \
  F(4, kHmmaWideOperands, 18)        \
  F(8, kHmmaWideOperands, 18)        \
  F(8, kAsyncCopy, 4)                \
  F(8, kAsyncCopyAliased, 4)         \
  F(8, kAsyncCopyStrided, 4)         \
  F(8, kAsyncCopy4, 4)               \
  F(8, kAsyncCopyRows, 4)            \
  F(8, kAsyncCopyLines, 4)           \
  F(8, kAsyncCopyZfillSink, 4)       \
  F(8, kAsyncCopyZfillSpread, 4)     \
  F(8, kAsyncCopyMixedSink, 4)       \
  F(8, kAsyncCopy4ZfillSink, 4)      \
  F(8, kAsyncCopyLanes, 4)           \
  F(8, kAsyncCopy4Lanes, 4)          \
  F(8, kAsyncCopyRows, 16)           \
  F(8, kAsyncCopyMixedSink, 16)      \
  F(8, kAsyncCopyLanes, 16)          \
  F(8, kAsyncCopy4, 16)              \
  F(8, kAsyncCopy4ZfillSink, 16)     \
  F(8, kAsyncCopy4Lanes, 16)         \
  F(4, kLdsChain, 16)                \
  F(8, kLdsChain, 16)                \
  F(4, kShflChain, 16)               \
  F(4, kLdsmChain, 16)               \
  F(4, kBranchTaken, 16)             \
  F(8, kBranchTaken, 16)             \
  F(4, kBranchDivergent, 16)         \
  F(8, kIcache, 256)                 \
  F(8, kIcache, 1024)                \
  F(8, kIcache, 2048)                \
  F(8, kIcache, 4096)                \
  F(8, kIcache, 8192)                \
  F(4, kIcache, 1024)                \
  F(4, kIcache, 8192)                \
  F(4, kAsyncCopyLatency, 1)         \
  F(8, kAsyncCopyLatency, 1)         \
  F(8, kPairPhase, 36)               \
  F(8, kPairPhase, 18)               \
  F(4, kPairPhase, 36)               \
  F(8, kGlobalReduce, 1)             \
  F(8, kCtaBarrier, 1)               \
  F(8, kShmBarrier, 1)               \
  F(8, kWarpSync, 1)

//: ONE LAUNCH of calibration (warps, mode, burst) over `owners` CTAs, `iters` units a warp; returns once
//: the stream has finished it. The caller times it.
inline void calibrate(int64_t warps, int64_t mode, int64_t burst, int64_t iters, int64_t owners,
                      Tensor& out, const Tensor& src) {
  STD_TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == Dtype::Float && out.numel() >= owners * 128 * 256,
      "calibrate: out is a CUDA float32 tensor of owners x 128 x 256");
  STD_TORCH_CHECK(
      src.is_cuda() && src.scalar_type() == Dtype::Byte && src.numel() >= owners * 256 * 128,
      "calibrate: src is a CUDA uint8 tensor of owners x 256 x 128");
  const CalibParams c{(int)iters, out.mutable_data_ptr<float>(),
                      reinterpret_cast<const char*>(src.mutable_data_ptr<uint8_t>())};
  bool found = false;
#define ROLA_CALIB_LAUNCH(W_, M_, B_)                                                               \
  if (!found && warps == (W_) && mode == (M_) && burst == (B_)) {                                   \
    auto* kern = calib_kernel<(W_), (M_), (B_)>;                                                    \
    STD_TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kCalibSmemBytes) \
                    == cudaSuccess,                                                                   \
                "calibrate: smem attribute");                                                         \
    kern<<<dim3((unsigned)owners, 1u), (W_) * 32, kCalibSmemBytes, current_stream()>>>(c); \
    found = true;                                                                                     \
  }
  ROLA_CALIB_SET_X(ROLA_CALIB_LAUNCH)
#undef ROLA_CALIB_LAUNCH
  STD_TORCH_CHECK(found, "calibrate: no calibration (warps ", warps, ", mode ", mode, ", burst ",
                  burst, ")");
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "calibrate: the launch failed");
}

}  // namespace rola::carry::calib
