// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// measure/lab/experiments/color_union.cu -- THE FACTS PASS'S COLORING, PRICED: a box-window's union colored by
// residue mod 8 (its pool rows' bank groups) so each digit's tokens spread over the banks (KERNEL_STANDARDS section
// 30), one warp a box-window from its bit tile in shared memory to each union token's slot. Two forms: the SERIAL
// greedy (`color_union.inputs.py`'s `color`, checked exactly) and the BATCHED one (its `batched`, claims racing on
// shared atomics, checked by invariants). see docs/internals/tools/lab.md#color-union
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../lab.cuh"

namespace color_union {

//: THE GRID: a fixed 64 blocks of 12 warps (one an SM: the batched form's 8.25 KB a warp fills sm_86's 99 KB), each
//: warp `kPerWarp` box-windows, so a launch's units are known before it runs
constexpr int kWarps = 12, kThreads = 32 * kWarps, kBlocks = 64, kPerWarp = 4;
constexpr long long kUnits = 1LL * kBlocks * kWarps * kPerWarp;

//: A WARP'S SHARED MEMORY: two tiles (the next box's copy in flight), the single tokens' words, each digit's counts a
//: residue (8 u16), forced counts (8 u16) and ties word, each digit's take (8 u16), each token's residue; the
//: multi-digit tokens' list (u16) lives where the plans go, dead before they are written
constexpr int kTileBytes = 32 * 16 * 4;
constexpr int kSingles = 2 * kTileBytes;
constexpr int kCounts = kSingles + 64;
constexpr int kForced = kCounts + 32 * 16;
constexpr int kList = kForced;
constexpr int kTies = kForced + 32 * 16;
constexpr int kTake = kTies + 32 * 4;
constexpr int kRes = kTake + 32 * 16;
constexpr int kWarpBytes = kRes + 512;
constexpr int kBytes = kWarps * kWarpBytes;
static_assert(kRes - kList >= 512 * 2, "the list holds every token");
static_assert(kBytes <= 99 * 1024, "the block fits sm_86's 99 KB");

//: THE TILE, the facts pass's: digit `d`'s word `j` at `16 d + (j ^ (d / 2 % 16))`, so 32 lanes reading one word of
//: every digit, or one digit's word each of 16 lanes, meet 32 banks
__device__ __forceinline__ int tile_at(int d, int j) { return 16 * d + (j ^ ((d >> 1) & 15)); }

template <int CELL>
__global__ __launch_bounds__(kThreads, 1) void color(const uint32_t* __restrict__ tiles, int boxes,
                                                     uint16_t* __restrict__ slots, uint16_t* __restrict__ extents,
                                                     long long* times, int* sinks) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  uint8_t* const base = smem + warp * kWarpBytes;
  uint32_t* const singles = reinterpret_cast<uint32_t*>(base + kSingles);
  uint32_t* const row = reinterpret_cast<uint32_t*>(base + kCounts) + 4 * lane;
  uint16_t* const list = reinterpret_cast<uint16_t*>(base + kList);
  uint16_t* const forced = reinterpret_cast<uint16_t*>(base + kForced);
  uint32_t* const ties = reinterpret_cast<uint32_t*>(base + kTies);
  uint16_t* const take = reinterpret_cast<uint16_t*>(base + kTake);
  uint8_t* const res = base + kRes;
  const int first = static_cast<int>(blockIdx.x) * kWarps + warp, stride = static_cast<int>(gridDim.x) * kWarps;
  //: a box's tile into buffer `buf`: lane l copies words l + 32 i (digit (l + 32 i) / 16), coalesced and swizzled
  const auto fetch = [&](int box, int buf) {
    const uint32_t* src = tiles + static_cast<int64_t>(box) * 512;
    const uint32_t dst = lab::smem(base + buf * kTileBytes);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int k = lane + 32 * i;
      lab::cp_async4(dst + 4u * tile_at(k >> 4, k & 15), src + k);
    }
    lab::cp_commit();
  };
  const int r = lane & 7, j = lane & 15, h = lane >> 4;
  //: field r of a pair of count words into bytes 2-3 (`__byte_perm`), the pair (r < 4 ? 0-1 : 2-3) picked after
  const uint32_t field = (2u * (r & 3)) << 8 | (2u * (r & 3) + 1) << 12;
  const int step = stride % boxes;
  int box = first % boxes, next = box + step - (box + step >= boxes ? boxes : 0);
  uint32_t fold = 0;
  lab::WarpSpan span;
  fetch(box, 0);
  span.start();
#pragma unroll 1
  for (int it = 0; it < kPerWarp; ++it) {
    fetch(next, (it + 1) & 1);
    lab::cp_wait<1>();
    __syncwarp();
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(base + (it & 1) * kTileBytes);

    //: THE UNION AND ITS MULTI-DIGIT TOKENS: lane (j, h) folds word j of digits 16 h .., half 1 in the order that keeps
    //: the halves on opposite banks, then the halves meet
    uint32_t once = 0, twice = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t x = tile[tile_at(16 * h + (i ^ h), j)];
      twice |= once & x;
      once |= x;
    }
    const uint32_t o = __shfl_xor_sync(~0u, once, 16), w2 = __shfl_xor_sync(~0u, twice, 16);
    const uint32_t uword = once | o, mword = twice | w2 | (once & o);
    if (lane < 16) singles[j] = uword & ~mword;
    //: the multi-digit tokens listed in order: word j's at its prefix
    {
      uint32_t x = lane < 16 ? mword : 0u, at = __popc(x);
#pragma unroll
      for (int s = 1; s < 16; s <<= 1) {
        const uint32_t y = __shfl_up_sync(~0u, at, s, 16);
        at += j >= s ? y : 0u;
      }
      at -= __popc(x);
#pragma unroll 1
      while (x) {
        list[at++] = static_cast<uint16_t>(32 * j + __ffs(x) - 1);
        x &= x - 1;
      }
    }
    const uint32_t multi = __reduce_add_sync(~0u, lane < 16 ? __popc(mword) : 0u);
    __syncwarp();

    //: THE MULTI-DIGIT TOKENS IN ORDER: lane d's counts a residue in registers (two u16 a word); a token's digits sum
    //: theirs, and it takes the residue they use least, then the least loaded (`lr`, residue lane & 7's load x 8 +
    //: itself), then the lowest; the next token's membership loads a step ahead
    uint32_t lr = r, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t t = list[0] & 511u;
    uint32_t in = (tile[tile_at(lane, t >> 5)] >> (t & 31)) & 1u;
#pragma unroll 1
    for (uint32_t i = 0; i < multi; ++i) {
      const uint32_t tn = list[i + 1] & 511u;
      const uint32_t inn = (tile[tile_at(lane, tn >> 5)] >> (tn & 31)) & 1u;
      const uint32_t m = 0u - in;
      const uint32_t s0 = __reduce_add_sync(~0u, c0 & m), s1 = __reduce_add_sync(~0u, c1 & m);
      const uint32_t s2 = __reduce_add_sync(~0u, c2 & m), s3 = __reduce_add_sync(~0u, c3 & m);
      const uint32_t p = r < 4 ? __byte_perm(s0, s1, field) : __byte_perm(s2, s3, field);
      const uint32_t best = __reduce_min_sync(~0u, (p & 0xffff0000u) | lr) & 7u;
      const uint32_t inc = in << ((best & 1) << 4), pair = best >> 1;
      c0 += pair == 0 ? inc : 0u;
      c1 += pair == 1 ? inc : 0u;
      c2 += pair == 2 ? inc : 0u;
      c3 += pair == 3 ? inc : 0u;
      lr += static_cast<uint32_t>(r) == best ? 8u : 0u;
      res[t] = static_cast<uint8_t>(best);
      t = tn;
      in = inn;
    }
    *reinterpret_cast<uint4*>(row) = make_uint4(c0, c1, c2, c3);
    uint32_t load = lr >> 3;
    __syncwarp();

    //: EACH DIGIT'S PLAN (lane d): its single tokens' words and count, then the water-fill over its counts -- the level
    //: every residue is filled to (`forced`) and the singles left for that level's residues (`ties`)
    uint32_t sw[16], k = 0;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
      sw[q] = tile[tile_at(lane, q)] & singles[q];
      k += __popc(sw[q]);
    }
    const uint4 cw = *reinterpret_cast<const uint4*>(row);
    const uint32_t cc[8] = {cw.x & 0xffffu, cw.x >> 16, cw.y & 0xffffu, cw.y >> 16,
                            cw.z & 0xffffu, cw.z >> 16, cw.w & 0xffffu, cw.w >> 16};
    uint32_t level = cc[0], placed = 0;
#pragma unroll
    for (int q = 1; q < 8; ++q) level = min(level, cc[q]);
#pragma unroll 1
    for (;;) {
      uint32_t a = 0, next = ~0u;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        a += cc[q] <= level;
        next = cc[q] > level ? min(next, cc[q]) : next;
      }
      if (next != ~0u && placed + a * (next - level) <= k) {
        placed += a * (next - level);
        level = next;
        continue;
      }
      const uint32_t q = (k - placed) / a;
      level += q;
      placed += q * a;
      break;
    }
    uint32_t active = 0, fw[4];
#pragma unroll
    for (int q = 0; q < 8; q += 2) {
      active |= (cc[q] <= level ? 1u : 0u) << q | (cc[q + 1] <= level ? 1u : 0u) << (q + 1);
      fw[q / 2] = (level > cc[q] ? level - cc[q] : 0u) | (level > cc[q + 1] ? level - cc[q + 1] : 0u) << 16;
    }
    *reinterpret_cast<uint4*>(forced + 8 * lane) = make_uint4(fw[0], fw[1], fw[2], fw[3]);
    ties[lane] = k == 0 ? 0u : (k - placed) | active << 16 | 1u << 24;
    __syncwarp();

    //: THE TIES, digit by digit: a level's singles go to its residues the least loaded first (lanes as residues)
#pragma unroll 1
    for (int d = 0; d < 32; ++d) {
      const uint32_t tr = ties[d];
      if (tr == 0) continue;
      uint32_t n = forced[8 * d + r];
      const uint32_t rem = tr & 0xffffu;
      if (rem != 0) {
        const bool act = (tr >> (16 + r)) & 1u;
        const uint32_t key = act ? (load << 3 | r) : 0xffffu;
        uint32_t rank = 0;
#pragma unroll
        for (int i = 1; i < 8; ++i) rank += __shfl_sync(~0u, key, (lane & ~7) | ((lane + i) & 7)) < key;
        n += act && rank < rem;
      }
      load += n;
      if (lane < 8) take[8 * d + r] = static_cast<uint16_t>(n);
    }
    __syncwarp();

    //: EACH DIGIT'S SINGLES (lane d), in token order, dealt to its take's residues in residue order
    {
      const uint4 tk = *reinterpret_cast<const uint4*>(take + 8 * lane);
      const uint32_t b1 = tk.x & 0xffffu, b2 = b1 + (tk.x >> 16), b3 = b2 + (tk.y & 0xffffu), b4 = b3 + (tk.y >> 16);
      const uint32_t b5 = b4 + (tk.z & 0xffffu), b6 = b5 + (tk.z >> 16), b7 = b6 + (tk.w & 0xffffu);
      uint32_t idx = 0;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        uint32_t x = sw[q];
#pragma unroll 1
        while (x) {
          const int bit = __ffs(x) - 1;
          x &= x - 1;
          const uint32_t rr = (b1 <= idx) + (b2 <= idx) + (b3 <= idx) + (b4 <= idx) + (b5 <= idx) + (b6 <= idx) +
                              (b7 <= idx);
          res[32 * q + bit] = static_cast<uint8_t>(rr);
          ++idx;
        }
      }
    }
    __syncwarp();

    //: THE SLOTS: a token's residue plus 8 x its rank among the residue's tokens, 32 tokens a word; lane r keeps
    //: residue r's running count
    uint16_t* const out = slots + static_cast<int64_t>(box) * 512;
    uint32_t count = 0;
#pragma unroll 4
    for (int q = 0; q < 16; ++q) {
      const uint32_t u = __shfl_sync(~0u, uword, q);
      const bool in = (u >> lane) & 1u;
      const uint32_t rr = res[32 * q + lane] & 7u;
      const uint32_t v0 = __ballot_sync(~0u, in && (rr & 1u)), v1 = __ballot_sync(~0u, in && (rr & 2u));
      const uint32_t v2 = __ballot_sync(~0u, in && (rr & 4u));
      const uint32_t same = u & (rr & 1u ? v0 : ~v0) & (rr & 2u ? v1 : ~v1) & (rr & 4u ? v2 : ~v2);
      const uint32_t at = __shfl_sync(~0u, count, static_cast<int>(rr));
      out[32 * q + lane] = in ? static_cast<uint16_t>(rr + 8 * (at + __popc(same & ((1u << lane) - 1)))) : 0xffffu;
      count += __popc(u & (r & 1 ? v0 : ~v0) & (r & 2 ? v1 : ~v1) & (r & 4 ? v2 : ~v2));
    }
    const uint32_t widest = __reduce_max_sync(~0u, count);
    if (lane == 0) extents[box] = static_cast<uint16_t>(8 * widest);
    fold ^= widest + count + load;
    box = next;
    next += step - (next + step >= boxes ? boxes : 0);
    __syncwarp();
  }
  lab::cp_wait<0>();
  span.stop(times);
  lab::sink(sinks, fold);
}

//: THE BATCHED FORM'S SHARED MEMORY a warp: two tiles, each token's digits (u32), the queue of multi-digit tokens
//: (u16, a ring; then the single tokens' words), each digit's counts (8 u16), full colors and capacity, each token's
//: color
constexpr int kMasks = 2 * kTileBytes;
constexpr int kQueue = kMasks + 512 * 4;
constexpr int kBCounts = kQueue + 512 * 2;
constexpr int kFull = kBCounts + 32 * 16;
constexpr int kCap = kFull + 32 * 4;
constexpr int kBRes = kCap + 32 * 4;
constexpr int kBWarpBytes = kBRes + 512;
constexpr int kBBytes = kWarps * kBWarpBytes;
static_assert(kBBytes <= 99 * 1024, "the batched block fits sm_86's 99 KB");

//: a 32 x 32 bit transpose across the warp: lane d's bit i to lane i's bit d. A stage of stride s swaps the
//: off-diagonal blocks of lanes d and d ^ s: each lane keeps its half `keep` (its blocks' mask, lane-constant) and
//: sends the other half rotated into place, `rot` left by s (bit s set) or right by s, its wrapped bits masked off
struct Transpose {
  uint32_t keep[5], rot[5];

  __device__ __forceinline__ explicit Transpose(int lane) {
    constexpr uint32_t kHalf[5] = {0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu, 0x33333333u, 0x55555555u};
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const int s = 16 >> k;
      keep[k] = lab::held(lane & s ? ~kHalf[k] : kHalf[k]);
      rot[k] = lab::held(lane & s ? s : 32 - s);
    }
  }

  __device__ __forceinline__ uint32_t operator()(uint32_t x) const {
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const uint32_t send = __funnelshift_l(x, x, rot[k]) & keep[k];
      x = (x & keep[k]) | __shfl_xor_sync(~0u, send, 16 >> k);
    }
    return x;
  }
};

//: THE BATCHED FORM (`color_union.inputs.py`'s `batched`, claims resolved by shared atomics in arrival order): the
//: multi-digit tokens 32 at a time (a lane each), each the least-used color under every digit's capacity, a claim over
//: one withdrawn and requeued; each digit's singles water-fill its room; the slots in the parity layout -- a token's
//: position by its color's parity, its color's other two bits the row's swizzle (`out`: position << 2 | swizzle)
template <int CELL>
__global__ __launch_bounds__(kThreads, 1) void batched(const uint32_t* __restrict__ tiles, int boxes,
                                                       uint16_t* __restrict__ out, uint16_t* __restrict__ extents,
                                                       long long* times, int* sinks) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  uint8_t* const base = smem + warp * kBWarpBytes;
  uint32_t* const masks = reinterpret_cast<uint32_t*>(base + kMasks);
  uint16_t* const queue = reinterpret_cast<uint16_t*>(base + kQueue);
  uint32_t* const singles = reinterpret_cast<uint32_t*>(base + kQueue);
  uint32_t* const counts = reinterpret_cast<uint32_t*>(base + kBCounts);
  uint32_t* const fullw = reinterpret_cast<uint32_t*>(base + kFull);
  uint32_t* const capw = reinterpret_cast<uint32_t*>(base + kCap);
  uint8_t* const res = base + kBRes;
  const int first = static_cast<int>(blockIdx.x) * kWarps + warp, stride = static_cast<int>(gridDim.x) * kWarps;
  const auto fetch = [&](int box, int buf) {
    const uint32_t* src = tiles + static_cast<int64_t>(box) * 512;
    const uint32_t dst = lab::smem(base + buf * kTileBytes);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int k = lane + 32 * i;
      lab::cp_async4(dst + 4u * tile_at(k >> 4, k & 15), src + k);
    }
    lab::cp_commit();
  };
  const int j = lane & 15, h = lane >> 4;
  const uint32_t below = (1u << lane) - 1;
  const Transpose transpose(lane);
  const int step = stride % boxes;
  int box = first % boxes, next = box + step - (box + step >= boxes ? boxes : 0);
  uint32_t fold = 0;
  lab::WarpSpan span;
  fetch(box, 0);
  span.start();
#pragma unroll 1
  for (int it = 0; it < kPerWarp; ++it) {
    fetch(next, (it + 1) & 1);
    lab::cp_wait<1>();
    __syncwarp();
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(base + (it & 1) * kTileBytes);

    //: THE UNION, ITS TOKENS IN TWO OR MORE DIGITS AND IN THREE OR MORE: lane (j, h) counts word j of digits 16 h ..,
    //: then the halves meet; lane d its digit's capacity a color, ceil(n / 8)
    uint32_t once = 0, twice = 0, thrice = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t x = tile[tile_at(16 * h + (i ^ h), j)];
      thrice |= twice & x;
      twice |= once & x;
      once |= x;
    }
    const uint32_t o1 = __shfl_xor_sync(~0u, once, 16), o2 = __shfl_xor_sync(~0u, twice, 16);
    const uint32_t o3 = __shfl_xor_sync(~0u, thrice, 16);
    const uint32_t uword = once | o1, mword = twice | o2 | (once & o1);
    const uint32_t tword = thrice | o3 | (twice & o1) | (once & o2);
    {
      uint32_t n = 0;
#pragma unroll
      for (int q = 0; q < 16; ++q) n += __popc(tile[tile_at(lane, q)]);
      capw[lane] = (n + 7) >> 3;
      fullw[lane] = 0;
      *reinterpret_cast<uint4*>(counts + 4 * lane) = make_uint4(0, 0, 0, 0);
    }
    //: the queue: tokens in three or more digits, then in two, each in token order
    uint32_t multi;
    {
      uint32_t x3 = lane < 16 ? tword : 0u, x2 = lane < 16 ? mword & ~tword : 0u;
      uint32_t a3 = __popc(x3), a2 = __popc(x2);
#pragma unroll
      for (int s = 1; s < 16; s <<= 1) {
        const uint32_t y3 = __shfl_up_sync(~0u, a3, s, 16), y2 = __shfl_up_sync(~0u, a2, s, 16);
        a3 += j >= s ? y3 : 0u;
        a2 += j >= s ? y2 : 0u;
      }
      const uint32_t n3 = __shfl_sync(~0u, a3, 15), n2 = __shfl_sync(~0u, a2, 15);
      uint32_t at3 = a3 - __popc(x3), at2 = n3 + a2 - __popc(x2);
#pragma unroll 1
      for (; x3; x3 &= x3 - 1) queue[at3++] = static_cast<uint16_t>(32 * j + __ffs(x3) - 1);
#pragma unroll 1
      for (; x2; x2 &= x2 - 1) queue[at2++] = static_cast<uint16_t>(32 * j + __ffs(x2) - 1);
      multi = n3 + n2;
    }
    //: each token's digits, a word's 32 tokens by one transpose, the words with a multi-digit token only
#pragma unroll 1
    for (int q = 0; q < (multi ? 16 : 0); ++q) {
      if (__shfl_sync(~0u, mword, q) == 0) continue;
      masks[32 * q + lane] = transpose(tile[tile_at(lane, q)]);
    }
    __syncwarp();

    //: THE MULTI-DIGIT TOKENS, 32 A BATCH: a lane's token sums its digits' counts and ORs their full colors, takes the
    //: least-used color allowed (any, when none is: blocked), and claims it on every digit by a shared atomic; a claim
    //: that finds a digit full is withdrawn and requeued; a batch that keeps nothing runs its first token alone next
    uint32_t head = 0, tail = multi, even = 0, odd = 0;
    bool solo = false;
#pragma unroll 1
    while (head < tail) {
      const uint32_t take = solo ? 1u : min(32u, tail - head);
      const bool live = static_cast<uint32_t>(lane) < take;
      const uint32_t t = live ? queue[(head + lane) & 511u] : 0u;
      head += take;
      const uint32_t m = live ? masks[t] : 0u;
      uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, full = 0;
#pragma unroll 1
      for (uint32_t x = m; x; x &= x - 1) {
        const int d = __ffs(x) - 1;
        const uint4 v = *reinterpret_cast<const uint4*>(counts + 4 * d);
        s0 += v.x;
        s1 += v.y;
        s2 += v.z;
        s3 += v.w;
        full |= fullw[d];
      }
      const uint32_t allowed = ~full & 0xffu, pool = allowed ? allowed : 0xffu;
      const uint32_t f[8] = {s0 & 0xffffu, s0 >> 16, s1 & 0xffffu, s1 >> 16,
                             s2 & 0xffffu, s2 >> 16, s3 & 0xffffu, s3 >> 16};
      uint32_t best = ~0u;
#pragma unroll
      for (int c = 0; c < 8; ++c) best = (pool >> c) & 1u ? min(best, f[c] << 3 | ((c - t) & 7u)) : best;
      const uint32_t c = ((best & 7u) + t) & 7u, shift = 16 * (c & 1), inc = 1u << shift;
      bool over = false;
#pragma unroll 1
      for (uint32_t x = m; x; x &= x - 1) {
        const int d = __ffs(x) - 1;
        const uint32_t old = atomicAdd(counts + 4 * d + (c >> 1), inc);
        over |= ((old >> shift) & 0xffffu) >= capw[d];
      }
      over = over && allowed != 0;
      if (over) {
#pragma unroll 1
        for (uint32_t x = m; x; x &= x - 1) atomicSub(counts + 4 * (__ffs(x) - 1) + (c >> 1), inc);
      }
      __syncwarp();
      const bool won = live && !over;
      if (won) {
#pragma unroll 1
        for (uint32_t x = m; x; x &= x - 1) {
          const int d = __ffs(x) - 1;
          if (((counts[4 * d + (c >> 1)] >> shift) & 0xffffu) >= capw[d]) atomicOr(fullw + d, 1u << c);
        }
        res[t] = static_cast<uint8_t>(c);
      }
      const uint32_t lost = __ballot_sync(~0u, live && over), kept = __ballot_sync(~0u, won);
      const uint32_t kept_even = __ballot_sync(~0u, won && !(c & 1));
      even += __popc(kept_even);
      odd += __popc(kept) - __popc(kept_even);
      if (live && over) queue[(tail + __popc(lost & below)) & 511u] = static_cast<uint16_t>(t);
      tail += __popc(lost);
      solo = kept == 0;
      __syncwarp();
    }

    //: EACH DIGIT'S SINGLES (lane d) BY FIRST FIT: the colors in order from an even or an odd start, each to its room
    //: (any fill within the room is conflict-free). Every digit starts even; digits whose odd start moves the parity
    //: count the needed way flip, in digit order, until the running change reaches half the imbalance
    if (lane < 16) singles[lane] = uword & ~mword;
    __syncwarp();
    uint32_t k = 0;
#pragma unroll
    for (int q = 0; q < 16; ++q) k += __popc(tile[tile_at(lane, q)] & singles[q]);
    const uint4 cw = *reinterpret_cast<const uint4*>(counts + 4 * lane);
    const int cp = static_cast<int>(capw[lane]);
    const int room[8] = {max(0, cp - static_cast<int>(cw.x & 0xffffu)), max(0, cp - static_cast<int>(cw.x >> 16)),
                         max(0, cp - static_cast<int>(cw.y & 0xffffu)), max(0, cp - static_cast<int>(cw.y >> 16)),
                         max(0, cp - static_cast<int>(cw.z & 0xffffu)), max(0, cp - static_cast<int>(cw.z >> 16)),
                         max(0, cp - static_cast<int>(cw.w & 0xffffu)), max(0, cp - static_cast<int>(cw.w >> 16))};
    uint32_t n0[8], n1[8];
    {
      int p0 = 0, p1 = 0;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        n0[c] = static_cast<uint32_t>(min(room[c], max(0, static_cast<int>(k) - p0)));
        p0 += room[c];
      }
#pragma unroll
      for (int i = 1; i <= 8; ++i) {
        const int c = i & 7;
        n1[c] = static_cast<uint32_t>(min(room[c], max(0, static_cast<int>(k) - p1)));
        p1 += room[c];
      }
    }
    const int e0 = static_cast<int>(n0[0] + n0[2] + n0[4] + n0[6]);
    const int e1 = static_cast<int>(n1[0] + n1[2] + n1[4] + n1[6]);
    const int total = static_cast<int>(even + odd) + __reduce_add_sync(~0u, k);
    const int need = static_cast<int>(even) + __reduce_add_sync(~0u, static_cast<uint32_t>(e0)) - (total + 1) / 2;
    const int delta = e1 - e0;
    uint32_t gain = need > 0 ? static_cast<uint32_t>(max(0, -delta)) : static_cast<uint32_t>(max(0, delta));
    uint32_t upto = gain;
#pragma unroll
    for (int sft = 1; sft < 32; sft <<= 1) {
      const uint32_t y = __shfl_up_sync(~0u, upto, sft);
      upto += lane >= sft ? y : 0u;
    }
    const bool flip = gain > 0 && static_cast<int>(upto) <= abs(need);
    //: the deal: the singles in token order take the colors in order, each its count
    {
      uint32_t n[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) n[c] = flip ? n1[c] : n0[c];
      const uint32_t b1 = n[0], b2 = b1 + n[1], b3 = b2 + n[2], b4 = b3 + n[3], b5 = b4 + n[4], b6 = b5 + n[5];
      const uint32_t b7 = b6 + n[6];
      //: one walk over the digit's singles: the cursor's word reloaded when it empties
      int q = 0;
      uint32_t x = tile[tile_at(lane, 0)] & singles[0];
#pragma unroll 1
      for (uint32_t idx = 0; idx < k; ++idx) {
#pragma unroll 1
        while (x == 0) {
          ++q;
          x = tile[tile_at(lane, q)] & singles[q];
        }
        const uint32_t rr = (b1 <= idx) + (b2 <= idx) + (b3 <= idx) + (b4 <= idx) + (b5 <= idx) + (b6 <= idx) +
                            (b7 <= idx);
        res[32 * q + __ffs(x) - 1] = static_cast<uint8_t>(rr);
        x &= x - 1;
      }
    }
    __syncwarp();

    //: THE SLOTS, THE PARITY LAYOUT: a token's position is 2 x its rank among its parity's tokens + the parity
    uint16_t* const dst = out + static_cast<int64_t>(box) * 512;
    uint32_t be = 0, bo = 0;
#pragma unroll 4
    for (int q = 0; q < 16; ++q) {
      const uint32_t u = __shfl_sync(~0u, uword, q);
      const bool in = (u >> lane) & 1u;
      const uint32_t c = res[32 * q + lane] & 7u;
      const uint32_t od = __ballot_sync(~0u, in && (c & 1u)), ev = u & ~od;
      const uint32_t rank = (c & 1u ? bo : be) + __popc((c & 1u ? od : ev) & below);
      const uint32_t slot = (2 * rank + (c & 1u)) << 2 | c >> 1;
      dst[32 * q + lane] = static_cast<uint16_t>(in ? slot : 0xffffu);
      be += __popc(ev);
      bo += __popc(od);
    }
    if (lane == 0) extents[box] = static_cast<uint16_t>(2 * max(be, bo));
    fold ^= be + bo + even;
    box = next;
    next += step - (next + step >= boxes ? boxes : 0);
    __syncwarp();
  }
  lab::cp_wait<0>();
  span.stop(times);
  lab::sink(sinks, fold);
}

//: a cell's tiles and expected slots, host and device
struct Cell {
  int boxes = 0;
  uint32_t* tiles = nullptr;
  uint16_t *slots = nullptr, *extents = nullptr;
  std::vector<uint32_t> host;
  std::vector<uint16_t> expected;
};

Cell load_cell(const std::string& dir) {
  Cell c;
  FILE* f = std::fopen((dir + "/tiles.bin").c_str(), "rb");
  int head[4] = {};
  if (!f || std::fread(head, 4, 4, f) != 4 || head[0] != 0x524F4C43) {
    std::fprintf(stderr, "color_union: %s/tiles.bin is not this lab's\n", dir.c_str());
    std::exit(1);
  }
  c.boxes = head[1];
  c.host.resize(static_cast<size_t>(c.boxes) * 512);
  if (std::fread(c.host.data(), 4, c.host.size(), f) != c.host.size()) std::exit(1);
  std::fclose(f);
  c.expected.resize(static_cast<size_t>(c.boxes) * 513);
  f = std::fopen((dir + "/expected.bin").c_str(), "rb");
  if (!f || std::fread(c.expected.data(), 2, c.expected.size(), f) != c.expected.size()) std::exit(1);
  std::fclose(f);
  LAB_CHECK(cudaMalloc(&c.tiles, c.host.size() * 4));
  LAB_CHECK(cudaMemcpy(c.tiles, c.host.data(), c.host.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&c.slots, static_cast<size_t>(c.boxes) * 512 * 2));
  LAB_CHECK(cudaMalloc(&c.extents, static_cast<size_t>(c.boxes) * 2));
  return c;
}

//: THE BUDGET LINES a box-window (KERNEL_STANDARDS section 29), by class. The serial form's was stated before its code
//: at ~45 instructions a multi-digit token and ~1.15K fixed and RESTATED from its counts at ~64 and ~2.6K (the misses
//: itemized at docs/internals/tools/lab.md#color-union). The batched form's, stated before its code at ~150 a batch
//: of 32, ~430 for the transposes and ~1K fixed (~1.8K, ~3.7K, ~1.1K at the cells), RESTATED from its counts at
//: 3.8K, 7.8K and 1.6K (the misses itemized there too).
struct Line {
  const char* cell;
  double shuffle, load, store, atomic, select, integer, control, global;
};

constexpr Line kSerial[3] = {{"alt-k4", 430, 203, 113, 0, 1325, 3650, 680, 38},
                             {"alt-k16", 1815, 760, 405, 0, 4370, 13420, 1525, 38},
                             {"thinned 1/16", 125, 102, 27, 0, 500, 1390, 410, 38}};
constexpr Line kBatched[3] = {{"alt-k4", 128, 259, 60, 25, 400, 2415, 560, 40},
                              {"alt-k16", 135, 475, 93, 110, 900, 4590, 1470, 40},
                              {"thinned 1/16", 40, 126, 30, 2, 250, 1120, 150, 40}};

void declare(const lab::Bench& b, const std::string& name, const Line& l) {
  b.units(name, "box-window", kUnits);
  b.budget(name, "shuffle", l.shuffle);
  b.budget(name, "shared load", l.load);
  b.budget(name, "shared store", l.store);
  if (l.atomic > 0) b.budget(name, "shared atomic", l.atomic);
  b.budget(name, "select/compare", l.select);
  b.budget(name, "integer/address", l.integer);
  b.budget(name, "control", l.control);
  b.budget(name, "global", l.global);
}

//: a launch's two timings: an SM's cycles a box-window (the block's span over its warps' box-windows) and a warp's
template <class Launch>
void time(const lab::Bench& b, Launch launch, const std::string& label) {
  b.record("an SM's cycles a box-window, 12 warps / " + label, "cycles",
           b.variant(launch, [&] { return b.span(kWarps, kBlocks) / (kWarps * kPerWarp); }));
  b.record("a warp's cycles a box-window / " + label, "cycles", b.variant(launch, [&] {
             std::vector<long long> t(2 * kBlocks * kWarps);
             LAB_CHECK(cudaMemcpy(t.data(), b.times, t.size() * sizeof(long long), cudaMemcpyDeviceToHost));
             double sum = 0;
             for (int i = 0; i < kBlocks * kWarps; ++i) sum += double(t[2 * i + 1] - t[2 * i]);
             return sum / (kBlocks * kWarps * kPerWarp);
           }));
}

std::vector<uint16_t> outputs(const Cell& c) {
  std::vector<uint16_t> got(static_cast<size_t>(c.boxes) * 513);
  LAB_CHECK(cudaMemcpy(got.data(), c.slots, static_cast<size_t>(c.boxes) * 512 * 2, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(got.data() + static_cast<size_t>(c.boxes) * 512, c.extents, static_cast<size_t>(c.boxes) * 2,
                       cudaMemcpyDeviceToHost));
  return got;
}

template <int CELL>
void serial(const lab::Bench& b, const Cell& c) {
  const std::string name = lab::kernel("color_union::color", {CELL});
  b.expect(name, "REDUX", ">=", 6);
  b.expect(name, "LDGSTS", ">=", 16);
  //: its divergent token walks and conditional stores (KERNEL_STANDARDS 29.13), a debt the facts pass's form pays off
  b.expect(name, "LOOP_CONVERGENCE", "<=", 77);
  declare(b, name, kSerial[CELL]);
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(color<CELL>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] {
    color<CELL><<<kBlocks, kThreads, kBytes>>>(c.tiles, c.boxes, c.slots, c.extents, b.times, b.sinks);
  };
  const std::string label = std::string("serial / ") + kSerial[CELL].cell;
  time(b, launch, label);
  LAB_CHECK(cudaMemset(c.slots, 0xAB, static_cast<size_t>(c.boxes) * 512 * 2));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  const std::vector<uint16_t> got = outputs(c);
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != c.expected[i];
  const std::string detail = std::to_string(c.boxes) + " box-windows' slots and extents, " + std::to_string(bad) +
                             " of " + std::to_string(got.size()) + " differ from the mirror";
  b.check("a warp's cycles a box-window / " + label, bad == 0, detail);
  b.check("an SM's cycles a box-window, 12 warps / " + label, bad == 0, detail);
}

//: the batched form's outputs against its invariants (claims race, so no mirror is exact): the union placed, positions
//: dense in each parity, the extent 2 x the larger parity; and its tokens over a digit's capacity, counted
template <int CELL>
void batched_variant(const lab::Bench& b, const Cell& c) {
  const std::string name = lab::kernel("color_union::batched", {CELL});
  b.expect(name, "ATOMS", ">=", 2);
  b.expect(name, "LDGSTS", ">=", 16);
  b.expect(name, "LOOP_CONVERGENCE", "<=", 52);
  declare(b, name, kBatched[CELL]);
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(batched<CELL>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBBytes));
  const auto launch = [&] {
    batched<CELL><<<kBlocks, kThreads, kBBytes>>>(c.tiles, c.boxes, c.slots, c.extents, b.times, b.sinks);
  };
  const std::string label = std::string("batched / ") + kBatched[CELL].cell;
  time(b, launch, label);
  LAB_CHECK(cudaMemset(c.slots, 0xAB, static_cast<size_t>(c.boxes) * 512 * 2));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  const std::vector<uint16_t> got = outputs(c);
  size_t bad = 0, over = 0, tokens = 0;
  for (int x = 0; x < c.boxes; ++x) {
    const uint32_t* rows = c.host.data() + static_cast<size_t>(x) * 512;
    const uint16_t* o = got.data() + static_cast<size_t>(x) * 512;
    std::vector<int> seen(1024, 0);
    int even = 0, odd = 0;
    for (int t = 0; t < 512; ++t) {
      uint32_t in = 0;
      for (int d = 0; d < 32; ++d) in |= rows[16 * d + t / 32] >> (t % 32) & 1u;
      bad += in != (o[t] != 0xffff);
      if (o[t] == 0xffff) continue;
      ++tokens;
      const int pos = o[t] >> 2;
      bad += pos >= 1024 || seen[pos]++;
      (pos & 1 ? odd : even) += 1;
    }
    for (int pos = 0; pos < 2 * std::max(even, odd); ++pos)
      bad += seen[pos] != ((pos & 1 ? odd : even) > pos / 2);
    bad += got[static_cast<size_t>(c.boxes) * 512 + x] != 2 * std::max(even, odd);
    for (int d = 0; d < 32; ++d) {
      int n = 0, count[8] = {};
      for (int t = 0; t < 512; ++t) {
        if (!(rows[16 * d + t / 32] >> (t % 32) & 1u)) continue;
        ++n;
        ++count[(o[t] >> 2 & 1) | (o[t] & 3) << 1];
      }
      for (int k = 0; k < 8; ++k) over += std::max(0, count[k] - (n + 7) / 8);
    }
  }
  const std::string detail = std::to_string(c.boxes) + " box-windows: " + std::to_string(bad) +
                             " invariant failures (union, dense parity positions, extent); " + std::to_string(over) +
                             " tokens over a digit's capacity in " + std::to_string(tokens);
  b.check("a warp's cycles a box-window / " + label, bad == 0, detail);
  b.check("an SM's cycles a box-window, 12 warps / " + label, bad == 0, detail);
  lab::Samples v;
  v.v.push_back(double(over) / c.boxes);
  b.record("tokens over a digit's capacity a box-window / " + label, "tokens", v);
  b.unchecked("tokens over a digit's capacity a box-window / " + label, "a count of the checked outputs");
}

}  // namespace color_union

int main(int argc, char** argv) {
  using namespace color_union;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Cell cells[3];
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "color_union: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    if (b.sms * lab::Bench::kMaxWarps < 2 * kBlocks * kWarps) {
      std::fprintf(stderr, "color_union: %d SMs hold too few timer slots for %d blocks\n", b.sms, kBlocks);
      return 1;
    }
    cells[0] = load_cell(dir + "/alt-k4");
    cells[1] = load_cell(dir + "/alt-k16");
    cells[2] = load_cell(dir + "/thin");
  }
  serial<0>(b, cells[0]);
  serial<1>(b, cells[1]);
  serial<2>(b, cells[2]);
  batched_variant<0>(b, cells[0]);
  batched_variant<1>(b, cells[1]);
  batched_variant<2>(b, cells[2]);
}
