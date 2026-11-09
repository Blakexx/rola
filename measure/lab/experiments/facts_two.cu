// measure/lab/experiments/facts_two.cu -- QUESTION: does the facts pass as two plain kernels -- a STREAM kernel that
// reads every row once and a BUILD kernel that ranks from its intermediates, the call chunked and the two overlapped on
// two CUDA streams -- come near its DRAM floor at k 4 and k 16, and what do its intermediates cost? METHOD: rows
// [level-side, T, H, 256] u8 (level-side 0 the write list level, 1 the read list level, 2-3 voted only), each token's
// head live in k random digits, as `facts_block`. STREAM: a CTA of eight warps a UNIT (window, level-side, head,
// 32-digit group): its 512 tokens' 32-byte runs by cp.async into a 16 KB window, the vote (bit words to a shared tile),
// the tile unions, and on a list level-side the bit words out (one 2 KB run), the union words and the digits' stash
// bases (META), and the group's STASH: each live entry's coefficient at its rank in its digit's token-ordered list, the
// digits' lists packed by prefix count (write: the f16 pair (c_v, c_m) at 2^(4 - E), E from the list's own maxima over
// the unit's whole window; read: f16 byte / 256), and the write lists' exponents. BUILD: a warp a quarter of a unit,
// lane (d, q) digit 8 wq + d's words 4 q .. 4 q + 3: the union ranks from the meta, each lane's walks of its own bit
// words pass by pass (counts by residue in two registers, joined across the digit's four lanes by shuffles) writing
// the lists' 8-byte (slot, 0, c_v, c_m) or 4-byte (slot, c_r) entries and the pass starts. OVERLAP: chunk i's build on a
// second stream beside chunk i + 1's stream kernel, intermediates in two buffers, ordered by events only. Checked
// against a naive per-list reference (union ranks, places, exponents, coefficients) and by the live count; timed
// against the required-bytes floor and the pair's own floor at a flat read's rate.
// see docs/internals/tools/lab.md#facts-two
#include "lab.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace facts_two {

constexpr int kTokens = 65536, kHeads = 4, kWidth = 256, kLevelSides = 4, kWindow = 512;
constexpr int kWindows = kTokens / kWindow, kGroups = kWidth / 32;
constexpr long long kLevelBytes = static_cast<long long>(kTokens) * kHeads * kWidth;
constexpr long long kBytes = kLevelBytes * kLevelSides;
constexpr int kPassSlots = 128, kMaxPasses = 4, kListCap = kWindow, kListAt = 8, kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127, kTileStride = 17, kStashCap = 32 * kWindow;
constexpr int kEntryMeta = 36;  // a list unit's digit bases, then its entry count
constexpr int kStreamThreads = 256, kBuildWarps = 8, kMaxChunk = 128;
constexpr float kListFloor = 1.17549435e-38f;
constexpr int kListUnits = 2 * kHeads * kGroups;  // list units a window (two list level-sides)
constexpr int kStreamUnits = kLevelSides * kHeads * kGroups;

//: the stream kernel's shared memory: the 16 KB window, the bit tile, the write side's gains, the digits' totals
constexpr int kSmWindow = 16384, kSmTile = kSmWindow, kSmGains = kSmTile + 32 * kTileStride * 4;
constexpr int kSmTotals = kSmGains + kWindow * 8, kSmBytes = kSmTotals + 32 * 4 + 16 * 4;
constexpr int kSmBytesP = 2 * kSmWindow + 32 * kTileStride * 4 + 32 * 4 + 16 * 4 + 2 * kWindow * 8;  // the persistent form

struct Params {
  const uint8_t* rows;
  const float2* fm;
  uint32_t* entries;      // [2][kMaxChunk * kListUnits][kEntryMeta]: each digit's first entry, then the unit's count
  uint16_t* slots;        // [2][kMaxChunk * kListUnits][kStashCap]: each entry's slot | digit << 9
  uint32_t* stash0;       // [2][kMaxChunk * kListUnits][kStashCap] (c_v | c_m << 16)
  uint16_t* stash1;       // the same, f16 read coefficients
  uint32_t* tile_unions;  // [level-side][window][head][group][8]
  uint32_t* list_exp;     // [window][head][digit] (K_m, K_v) int16 pair
  uint16_t* at;           // [list level-side][window][head][digit][8]
  uint2* rec0;            // [window][head][digit][kListCap]
  uint32_t* rec1;
  uint16_t* union_tok;    // [list level-side][window][head][group][kWindow]
  uint16_t* union_count;  // [list level-side][window][head][group]
};

struct Chunk {
  int win0, windows, buffer;
};

__device__ __forceinline__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

__global__ void fill(uint8_t* rows, float2* fm, int k) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<long long>(kTokens) * kHeads)
    fm[i] = make_float2(1e-3f * (1 + mix(static_cast<uint32_t>(i), 1u, 2u) % 1000u),
                        1e-3f * (1 + mix(static_cast<uint32_t>(i), 3u, 4u) % 1000u));
  if (i >= static_cast<long long>(kLevelSides) * kTokens * kHeads) return;
  for (int j = 0; j < k; ++j) {
    const uint32_t h = mix(static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0x5eedu);
    rows[i * kWidth + (h & 255u)] = static_cast<uint8_t>(1u + (h >> 8) % 255u);
  }
}

__global__ void count_live(const uint8_t* rows, long long first, long long bytes, unsigned long long* live) {
  unsigned long long n = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    n += rows[first + i] != 0;
  atomicAdd(live, n);
}

__global__ void sum_lists(const uint16_t* at, long long lists, const uint16_t* counts, long long groups,
                          unsigned long long* total) {
  unsigned long long n = 0, u = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < lists;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    n += at[i * kListAt + kMaxPasses];
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < groups;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    u += counts[i];
  atomicAdd(total, n);
  atomicAdd(total + 1, u);
}

//: THE FLAT READ, the DRAM floor's rate: the rows as contiguous 1 KB stages a warp, three in flight, sunk
__global__ void __launch_bounds__(512, 1) flat(const uint8_t* rows, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kStages = 4, kStage = 1024, kWarps = 16;
  const int lane = lab::lane_id(), warp = lab::warp_id();
  const uint32_t ring = lab::smem(smem + warp * kStages * kStage);
  const long long stages = kBytes / kStage, per = stages / (gridDim.x * kWarps);
  const long long first = (static_cast<long long>(blockIdx.x) * kWarps + warp) * per;
  uint32_t sink = 0u;
  const auto copy = [&](long long k) {
    if (k < per)
      for (int h = 0; h < 2; ++h) {
        const int q = 16 * (lane + 32 * h);
        lab::cp_async16(ring + static_cast<uint32_t>(k % kStages) * kStage + q, rows + (first + k) * kStage + q);
      }
    lab::cp_commit();
  };
  for (int k = 0; k < kStages - 1; ++k) copy(k);
#pragma unroll 1
  for (long long k = 0; k < per; ++k) {
    copy(k + kStages - 1);
    lab::cp_wait<kStages - 1>();
    __syncwarp();
    sink ^= *reinterpret_cast<const uint32_t*>(smem + warp * kStages * kStage + (k % kStages) * kStage + 4 * lane);
    __syncwarp();
  }
  lab::sink(out, sink);
}

__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

__device__ __forceinline__ uint32_t nonzero_bytes(uint32_t x) {
  return (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
}

__device__ __forceinline__ int ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(static_cast<uint32_t>(127 + e) << 23); }

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}

//: the place of rank `j` of residue `r` in a pass whose residues hold the byte counts `lo` (0-3) and `hi` (4-7), each
//: at most 16: every residue's entries in rounds below `j`, then round `j`'s residues below `r`
__device__ __forceinline__ int round_robin(uint32_t lo, uint32_t hi, int j, int r) {
  const uint32_t off = 0x7f7f7f7fu - 0x01010101u * static_cast<uint32_t>(j);
  const uint32_t gl = (lo + off) & 0x80808080u, gh = (hi + off) & 0x80808080u;
  const uint32_t ml = prmt<0xba98u>(gl, 0u), mh = prmt<0xba98u>(gh, 0u);
  const uint32_t le = __dp4a(lo & ~ml, 0x01010101u, __dp4a(hi & ~mh, 0x01010101u, 0u));
  const uint32_t below = r >= 4 ? gl | (gh & ((1u << (8 * (r - 4))) - 1u)) >> 4 : gl & ((1u << (8 * r)) - 1u);
  return static_cast<int>(le) + j * __popc(gl | gh >> 4) + __popc(below);
}

//: the stream kernel's unit: the group innermost, then head, level-side, window
struct StreamUnit {
  int group, head, ls, window_in_chunk;
};

__device__ __forceinline__ StreamUnit decode_stream(int u) { return {u & 7, (u >> 3) & 3, (u >> 5) & 3, u >> 7}; }

//: THE PLACEMENT, one warp's: the entries [e0, e0 + n) of a unit (the flat list in (digit, token) order, each entry its
//: slot and digit and its coefficient) whose digits lie in [digit0, digit0 + nd), nd at most 32. Round r takes the
//: entries 32 r .. 32 r + 31, a lane an entry. An entry's key is (digit, pass, residue); `match.any` groups the round's
//: lanes by key, the lowest lane of a key adds its group to the key's byte of a shared counter (one atomic a key), and
//: the group's lanes rank themselves in lane order -- which is token order. Round-robin places need a key group's
//: total, so the pass is made twice: counting (the totals, the pass starts), then ranking and writing. The tables are
//: the warp's: `total` and `run` [nd][pass][lo, hi], `start` [nd][pass].
__device__ __forceinline__ void place(const Params& p, long long slice, int lsl, long long list0, int e0, int count,
                                      int digit0, int nd, uint32_t* total, uint32_t* run, uint32_t* start_of) {
  const int lane = lab::lane_id();
  const int n = __shfl_sync(0xffffffffu, count, 0);  // the warp's one count, uniform to the compiler
  const uint16_t* const slots = p.slots + slice * kStashCap + e0;
  const uint32_t* const stash0 = p.stash0 + slice * kStashCap + e0;
  const uint16_t* const stash1 = p.stash1 + slice * kStashCap + e0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (32 * i + lane < nd * 8) {
      total[32 * i + lane] = 0u;
      run[32 * i + lane] = 0u;
    }
  }
  __syncwarp();
#pragma unroll 1
  for (int base = 0; base < n; base += 4 * 32) {
    uint32_t td[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int e = base + 32 * r + lane;
      const uint32_t v = slots[min(e, n - 1)];
      td[r] = e < n ? v : 0xffffu;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const bool valid = td[r] != 0xffffu;
      const int digit = static_cast<int>(td[r] >> 9) - digit0, slot = static_cast<int>(td[r] & 511u);
      const int pass = slot / kPassSlots, res = slot & 7;
      const uint32_t key = valid ? (digit << 5 | pass << 3 | res) : 0xffffu;
      const uint32_t mask = __match_any_sync(0xffffffffu, key);
      if (valid && lane == __ffs(mask) - 1)
        atomicAdd(&total[digit * 8 + pass * 2 + (res >> 2)], static_cast<uint32_t>(__popc(mask)) << (8 * (res & 3)));
    }
  }
  __syncwarp();
  //: the lists' pass starts and totals: lane a digit
  {
    uint32_t starts[kMaxPasses];
    int start = 0;
#pragma unroll
    for (int pass = 0; pass < kMaxPasses; ++pass) {
      starts[pass] = static_cast<uint32_t>(start);
      if (lane < nd) {
        start_of[lane * 4 + pass] = static_cast<uint32_t>(start);
        start += static_cast<int>(__dp4a(total[lane * 8 + 2 * pass], 0x01010101u,
                                         __dp4a(total[lane * 8 + 2 * pass + 1], 0x01010101u, 0u)));
      }
    }
    const uint32_t sum = static_cast<uint32_t>(start);
    if (lane < nd) {
      uint4* const at = reinterpret_cast<uint4*>(
          p.at + (static_cast<long long>(lsl) * kWindows * kHeads * kWidth + list0 + digit0 + lane) * kListAt);
      *at = make_uint4(starts[0] | starts[1] << 16, starts[2] | starts[3] << 16, sum | sum << 16, sum | sum << 16);
    }
  }
  __syncwarp();

#pragma unroll 1
  for (int base = 0; base < n; base += 4 * 32) {
    uint32_t td[4], coef[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int e = base + 32 * r + lane, at = min(e, n - 1);
      const uint32_t v = slots[at];
      td[r] = e < n ? v : 0xffffu;
      coef[r] = lsl == 0 ? stash0[at] : stash1[at];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const bool valid = td[r] != 0xffffu;
      const int digit = static_cast<int>(td[r] >> 9) - digit0, slot = static_cast<int>(td[r] & 511u);
      const int pass = slot / kPassSlots, res = slot & 7;
      const uint32_t key = valid ? (digit << 5 | pass << 3 | res) : 0xffffu;
      const uint32_t mask = __match_any_sync(0xffffffffu, key);
      const int leader = __ffs(mask) - 1;
      uint32_t old = 0u;
      if (valid && lane == leader)
        old = atomicAdd(&run[digit * 8 + pass * 2 + (res >> 2)], static_cast<uint32_t>(__popc(mask)) << (8 * (res & 3)));
      old = __shfl_sync(0xffffffffu, old, leader);
      if (valid) {
        const int rank = static_cast<int>((old >> (8 * (res & 3))) & 0xffu) + __popc(mask & ((1u << lane) - 1u));
        const long long at = (list0 + digit0 + digit) * kListCap + start_of[digit * 4 + pass]
                             + round_robin(total[digit * 8 + pass * 2], total[digit * 8 + pass * 2 + 1], rank, res);
        if (lsl == 0)
          p.rec0[at] = make_uint2(static_cast<uint32_t>(slot), coef[r]);
        else
          p.rec1[at] = static_cast<uint32_t>(slot) | coef[r] << 16;
      }
    }
  }
}

//: THE STREAM KERNEL. A CTA of eight warps a unit (window, level-side, head, 32-digit group); warp w the window's tokens
//: 64 w .. 64 w + 63 (two 32-token blocks, each 1 KB of the window at `ring_row`). `issue_unit` starts a unit's copies;
//: `process_unit` is everything after they land.
__device__ __forceinline__ void issue_unit(const Params& p, const Chunk& c, int u, uint8_t* win, float2* gains) {
  const StreamUnit su = decode_stream(u);
  const int window = c.win0 + su.window_in_chunk, ls = su.ls, head = su.head, group = su.group;
  const int warp = lab::warp_id(), lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  const uint8_t* const rows = p.rows + ls * kLevelBytes
                              + (static_cast<long long>(window) * kWindow * kHeads + head) * kWidth + group * 32;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int quantum = lane + 32 * i, token = quantum >> 1, part = quantum & 1;
      lab::cp_async16(lab::smem(win + j * 1024 + ring_row(token) * 32 + 16 * part),
                      rows + static_cast<long long>(32 * j + token) * kHeads * kWidth + 16 * part);
    }
  }
  if (ls == 0) {
    const float2* const g = p.fm + static_cast<long long>(window) * kWindow * kHeads + head;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      cp_async8(lab::smem(gains + tid + 256 * i), g + static_cast<long long>(tid + 256 * i) * kHeads);
  }
}

template <bool kFuse>
__device__ __forceinline__ void process_unit(const Params& p, const Chunk& c, int u, uint8_t* win, float2* gains,
                                             uint32_t* tile, int* totals, uint32_t* unions) {
  const StreamUnit su = decode_stream(u);
  const int window = c.win0 + su.window_in_chunk, ls = su.ls, head = su.head, group = su.group;
  const int warp = lab::warp_id(), lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  const bool listed = ls < 2;
  const int unit_list = ((su.window_in_chunk * 2 + ls) * kHeads + head) * kGroups + group;

  //: the vote: lane (tk, dq) token 8 tk + i's digits 4 dq .. as a word, each byte's nonzero bit at bit i; a 4 x 4 byte
  //: transpose across the token blocks leaves lane (tk, dq) digit 4 dq + tk's 32 token bits
  const int tk = lane >> 3, dq = lane & 7, digit = 4 * dq + tk;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    const uint8_t* const region = win + j * 1024;
    uint32_t flags = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      flags |= nonzero_bytes(*reinterpret_cast<const uint32_t*>(region + (4 * i + tk) * 32 + 4 * dq)) >> (7 - i);
    uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
    flags = (tk & 2) ? prmt<0x3276u>(flags, y) : prmt<0x5410u>(flags, y);
    y = __shfl_xor_sync(0xffffffffu, flags, 8);
    flags = (tk & 1) ? prmt<0x3715u>(flags, y) : prmt<0x6240u>(flags, y);
    tile[digit * kTileStride + j] = flags;
  }
  __syncthreads();

  //: the tile unions (the intra's), one ballot a 64-token tile: lane a digit
  {
    const uint32_t any = __ballot_sync(
        0xffffffffu, (tile[lane * kTileStride + 2 * warp] | tile[lane * kTileStride + 2 * warp + 1]) != 0u);
    if (lane == 0)
      p.tile_unions[(((static_cast<long long>(ls) * kWindows + window) * kHeads + head) * kGroups + group) * 8 + warp] =
          any;
  }
  if (!listed) return;

  //: the digits' counts (half a warp a digit, a lane a word) and the union's words
  const int hw = lane >> 4, jl = lane & 15;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int d = 4 * warp + 2 * r + hw;
    int n = __popc(tile[d * kTileStride + jl]);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) n += __shfl_xor_sync(0xffffffffu, n, off);
    if (jl == 0) totals[d] = n;
  }
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const uint32_t u = __reduce_or_sync(0xffffffffu, tile[lane * kTileStride + 2 * warp + jj]);
    if (lane == 0) unions[2 * warp + jj] = u;
  }
  __syncthreads();
  int inc = totals[lane];
  const int mine = inc;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, off);
    inc += lane >= off ? y : 0;
  }
  const int excl = inc - mine;
  //: the union's ranks: lane j of each half-warp its word j's base; the union's tokens and count out
  const long long slice = static_cast<long long>(c.buffer) * kMaxChunk * kListUnits + unit_list;
  const uint32_t uj = unions[jl];
  int ubj = __popc(uj);
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, ubj, off, 16);
    ubj += jl >= off ? y : 0;
  }
  const int union_total = __shfl_sync(0xffffffffu, ubj, 15);
  ubj -= __popc(uj);
  {
    const long long ug = (static_cast<long long>(ls) * kWindows + window) * kHeads * kGroups + head * kGroups + group;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int j = 2 * warp + jj;
      const uint32_t u = unions[j];
      const int base = __shfl_sync(0xffffffffu, ubj, j);
      if ((u >> lane) & 1u)
        p.union_tok[ug * kWindow + base + __popc(u & ((1u << lane) - 1u))] = static_cast<uint16_t>(32 * j + lane);
    }
    const int entries = __shfl_sync(0xffffffffu, inc, 31);
    if (warp == 0) {
      p.entries[slice * kEntryMeta + lane] = static_cast<uint32_t>(excl);
      if (lane == 0) {
        p.entries[slice * kEntryMeta + 32] = static_cast<uint32_t>(entries);
        p.union_count[ug] = static_cast<uint16_t>(union_total);
      }
    }
  }

  const long long region = slice * kStashCap;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int d = 4 * warp + 2 * r + hw;
    const int dbase = __shfl_sync(0xffffffffu, excl, d);
    const uint32_t w = tile[d * kTileStride + jl];
    int incl = __popc(w);
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, off, 16);
      incl += jl >= off ? y : 0;
    }
    int pos = dbase + incl - __popc(w);
    const uint8_t* const col = win + jl * 1024 + d;
    if (ls == 0) {
      float pm = 0.0f, pf = 0.0f;
#pragma unroll 1
      for (uint32_t b = w; b != 0u; b &= b - 1u) {
        const int t = __ffs(b) - 1;
        const float byte = static_cast<float>(col[ring_row(t) * 32]);
        const float2 g = gains[32 * jl + t];
        pm = fmaxf(pm, byte * g.x);
        pf = fmaxf(pf, byte * g.y);
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
        pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
        pf = fmaxf(pf, __shfl_xor_sync(0xffffffffu, pf, off));
      }
      const int km = ceil_exponent(pm), kv = ceil_exponent(pf);
      const float sm = pow2(kCoefTop - max(km, kUnitMin)), sv = pow2(kCoefTop - max(kv, kUnitMin));
      if (jl == 0)
        p.list_exp[(static_cast<long long>(window) * kHeads + head) * kWidth + group * 32 + d] =
            static_cast<uint32_t>(static_cast<uint16_t>(km)) | static_cast<uint32_t>(static_cast<uint16_t>(kv)) << 16;
#pragma unroll 1
      for (uint32_t b = w; b != 0u; b &= b - 1u) {
        const int t = __ffs(b) - 1;
        const float byte = static_cast<float>(col[ring_row(t) * 32]);
        const float2 g = gains[32 * jl + t];
        const uint32_t cv = __half_as_ushort(__float2half_rn(byte * g.y * sv));
        const uint32_t cm = __half_as_ushort(__float2half_rn(byte * g.x * sm));
        p.stash0[region + pos] = cv | cm << 16;
        p.slots[region + pos] = static_cast<uint16_t>(ubj + __popc(uj & ((1u << t) - 1u)) | d << 9);
        ++pos;
      }
    } else {
#pragma unroll 1
      for (uint32_t b = w; b != 0u; b &= b - 1u) {
        const int t = __ffs(b) - 1;
        const float byte = static_cast<float>(col[ring_row(t) * 32]);
        p.stash1[region + pos] = __half_as_ushort(__float2half_rn(byte * (1.0f / 256.0f)));
        p.slots[region + pos] = static_cast<uint16_t>(ubj + __popc(uj & ((1u << t) - 1u)) | d << 9);
        ++pos;
      }
    }
  }
  //: FUSED: the unit's own warps place the entries just stashed, a warp its four digits' (the window is dead: its first
  //: bytes are the warps' tables)
  if constexpr (kFuse) {
    __syncthreads();
    uint32_t* const tab = reinterpret_cast<uint32_t*>(win) + warp * 80;
    const int e0 = __shfl_sync(0xffffffffu, excl, 4 * warp);
    const int next = __shfl_sync(0xffffffffu, excl, (4 * warp + 4) & 31);
    const int e1 = warp == 7 ? __shfl_sync(0xffffffffu, inc, 31) : next;
    place(p, slice, ls, (static_cast<long long>(window) * kHeads + head) * kWidth + group * 32, e0, e1 - e0, 4 * warp, 4,
          tab, tab + 32, tab + 64);
  }
}


template <bool kFuse>
__global__ void __launch_bounds__(kStreamThreads) stream(Params p, Chunk c) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const win = smem;
  uint32_t* const tile = reinterpret_cast<uint32_t*>(smem + kSmTile);
  float2* const gains = reinterpret_cast<float2*>(smem + kSmGains);
  int* const totals = reinterpret_cast<int*>(smem + kSmTotals);
  uint32_t* const unions = reinterpret_cast<uint32_t*>(smem + kSmTotals + 32 * 4);
  issue_unit(p, c, static_cast<int>(blockIdx.x), win, gains);
  lab::cp_commit();
  lab::cp_wait_all();
  __syncthreads();
  process_unit<kFuse>(p, c, static_cast<int>(blockIdx.x), win, gains, tile, totals, unions);
}

//: THE PERSISTENT STREAM KERNEL: a CTA loops over units a grid apart, the next unit's copies in flight in the other
//: window while this unit is voted
template <bool kFuse>
__global__ void __launch_bounds__(kStreamThreads) stream_p(Params p, Chunk c) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint32_t* const tile = reinterpret_cast<uint32_t*>(smem + 2 * kSmWindow);
  int* const totals = reinterpret_cast<int*>(smem + 2 * kSmWindow + 32 * kTileStride * 4);
  uint32_t* const unions = reinterpret_cast<uint32_t*>(smem + 2 * kSmWindow + 32 * kTileStride * 4 + 32 * 4);
  float2* const gain_base = reinterpret_cast<float2*>(smem + 2 * kSmWindow + 32 * kTileStride * 4 + 32 * 4 + 16 * 4);
  const int units = c.windows * kStreamUnits;
  int buf = 0;
  if (static_cast<int>(blockIdx.x) < units) issue_unit(p, c, static_cast<int>(blockIdx.x), smem, gain_base);
  lab::cp_commit();
#pragma unroll 1
  for (int u = static_cast<int>(blockIdx.x); u < units; u += static_cast<int>(gridDim.x)) {
    const int next = u + static_cast<int>(gridDim.x);
    if (next < units) issue_unit(p, c, next, smem + (buf ^ 1) * kSmWindow, gain_base + (buf ^ 1) * kWindow);
    lab::cp_commit();
    lab::cp_wait<1>();
    __syncthreads();
    process_unit<kFuse>(p, c, u, smem + buf * kSmWindow, gain_base + buf * kWindow, tile, totals, unions);
    __syncthreads();
    buf ^= 1;
  }
}

//: THE BUILD KERNEL. A warp an eighth of a unit (window, list level-side, head, 32-digit group),
//: its four digits' entries -- a contiguous run of the flat list, from the stream kernel's digit bases.
struct BuildSmem {
  uint32_t total[4 * 8];
  uint32_t run[4 * 8];
  uint32_t start[4 * 4];
};

__global__ void __launch_bounds__(kBuildWarps * 32) build(Params p, Chunk c) {
  extern __shared__ __align__(128) uint8_t smem[];
  BuildSmem& s = reinterpret_cast<BuildSmem*>(smem)[lab::warp_id()];
#pragma unroll 1
  for (int task = static_cast<int>(blockIdx.x) * kBuildWarps + lab::warp_id(); task < c.windows * kListUnits * 8;
       task += static_cast<int>(gridDim.x) * kBuildWarps) {
    const int unit = task >> 3, eighth = task & 7;
    const int group = unit & 7, head = (unit >> 3) & 3, lsl = (unit >> 5) & 1, window = c.win0 + (unit >> 6);
    const long long slice = static_cast<long long>(c.buffer) * kMaxChunk * kListUnits + unit;
    const uint32_t* const meta = p.entries + slice * kEntryMeta;
    const int e0 = static_cast<int>(meta[4 * eighth]);
    const int e1 = static_cast<int>(meta[4 * eighth + 4]);  // the last eighth's end is the count at 32
    place(p, slice, lsl, (static_cast<long long>(window) * kHeads + head) * kWidth + group * 32, e0, e1 - e0, 4 * eighth,
          4, s.total, s.run, s.start);
  }
}

//: THE REFERENCE: one thread a list of the first `windows` windows, naive and serial -- the union's ranks by scanning
//: the group's 32 digits a token, the places by the round-robin definition, the exponents and coefficients with the
//: same operations as the stream kernel -- compared with every defined output of the pair
__device__ __forceinline__ uint32_t byte_of(const uint8_t* rows, int ls, int window, int t, int head, int digit) {
  return rows[ls * kLevelBytes + ((static_cast<long long>(window) * kWindow + t) * kHeads + head) * kWidth + digit];
}

__global__ void reference(Params p, int windows, unsigned long long* bad) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 2LL * windows * kHeads * kWidth) return;
  const int digit = static_cast<int>(i % kWidth), head = static_cast<int>(i / kWidth % kHeads);
  const int window = static_cast<int>(i / kWidth / kHeads % windows);
  const int ls = static_cast<int>(i / kWidth / kHeads / windows);
  const int group = digit / 32;
  const long long list = (static_cast<long long>(window) * kHeads + head) * kWidth + digit;
  const long long ug = (static_cast<long long>(ls) * kWindows + window) * kHeads * kGroups + head * kGroups + group;
  //: counts a (pass, residue), in eight words by if-chains (no local memory)
  uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  float pm = 0.0f, pf = 0.0f;
  int slot = 0, entries = 0, badc = 0;
  for (int walk = 0; walk < 2; ++walk) {
    slot = 0;
    entries = 0;
    int ucount_expected = 0;
    for (int t = 0; t < kWindow; ++t) {
      bool any = false;
      for (int d = 0; d < 32; ++d) any |= byte_of(p.rows, ls, window, t, head, group * 32 + d) != 0u;
      const uint32_t byte = byte_of(p.rows, ls, window, t, head, digit);
      if (any) {
        if (walk == 0 && digit % 32 == 0) {
          if (p.union_tok[ug * kWindow + ucount_expected] != t) ++badc;
        }
        ++ucount_expected;
      }
      if (byte != 0u) {
        const int pass = slot / kPassSlots, r = slot & 7;
        const uint32_t inc = 1u << (8 * (r & 3));
        const bool hi = r >= 4;
        if (walk == 0) {
          if (pass == 0) { if (hi) t1 += inc; else t0 += inc; }
          if (pass == 1) { if (hi) t3 += inc; else t2 += inc; }
          if (pass == 2) { if (hi) t5 += inc; else t4 += inc; }
          if (pass == 3) { if (hi) t7 += inc; else t6 += inc; }
          if (ls == 0) {
            const float2 g = p.fm[(static_cast<long long>(window) * kWindow + t) * kHeads + head];
            pm = fmaxf(pm, static_cast<float>(byte) * g.x);
            pf = fmaxf(pf, static_cast<float>(byte) * g.y);
          }
        } else {
          uint32_t rn, tl, th;
          uint32_t start = 0;
          const uint32_t a0 = __dp4a(t0, 0x01010101u, __dp4a(t1, 0x01010101u, 0u));
          const uint32_t a1 = __dp4a(t2, 0x01010101u, __dp4a(t3, 0x01010101u, 0u));
          const uint32_t a2 = __dp4a(t4, 0x01010101u, __dp4a(t5, 0x01010101u, 0u));
          if (pass == 0) { tl = t0; th = t1; rn = hi ? (r1 >> 8 * (r & 3)) : (r0 >> 8 * (r & 3)); }
          else if (pass == 1) { tl = t2; th = t3; start = a0; rn = hi ? (r3 >> 8 * (r & 3)) : (r2 >> 8 * (r & 3)); }
          else if (pass == 2) { tl = t4; th = t5; start = a0 + a1; rn = hi ? (r5 >> 8 * (r & 3)) : (r4 >> 8 * (r & 3)); }
          else { tl = t6; th = t7; start = a0 + a1 + a2; rn = hi ? (r7 >> 8 * (r & 3)) : (r6 >> 8 * (r & 3)); }
          const int rank = static_cast<int>(rn & 0xffu);
          if (pass == 0) { if (hi) r1 += inc; else r0 += inc; }
          if (pass == 1) { if (hi) r3 += inc; else r2 += inc; }
          if (pass == 2) { if (hi) r5 += inc; else r4 += inc; }
          if (pass == 3) { if (hi) r7 += inc; else r6 += inc; }
          //: the place by definition: residues' entries in rounds below `rank`, then this round's residues below r
          int place = static_cast<int>(start);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint32_t cnt = ((q < 4 ? tl : th) >> (8 * (q & 3))) & 0xffu;
            place += static_cast<int>(min(cnt, static_cast<uint32_t>(rank)));
            if (q < r && cnt > static_cast<uint32_t>(rank)) ++place;
          }
          const long long at = list * kListCap + place;
          if (ls == 0) {
            const int km = ceil_exponent(pm), kv = ceil_exponent(pf);
            const float sm = pow2(kCoefTop - max(km, kUnitMin)), sv = pow2(kCoefTop - max(kv, kUnitMin));
            const float2 g = p.fm[(static_cast<long long>(window) * kWindow + t) * kHeads + head];
            const uint32_t cv = __half_as_ushort(__float2half_rn(static_cast<float>(byte) * g.y * sv));
            const uint32_t cm = __half_as_ushort(__float2half_rn(static_cast<float>(byte) * g.x * sm));
            const uint2 got = p.rec0[at];
            if (got.x != static_cast<uint32_t>(slot) || got.y != (cv | cm << 16)) ++badc;
          } else {
            const uint32_t coef = __half_as_ushort(__float2half_rn(static_cast<float>(byte) * (1.0f / 256.0f)));
            if (p.rec1[at] != (static_cast<uint32_t>(slot) | coef << 16)) ++badc;
          }
        }
        ++entries;
      }
      if (any) ++slot;
    }
    if (walk == 0 && digit % 32 == 0 && p.union_count[ug] != slot) ++badc;
  }
  const uint16_t* const at = p.at + (static_cast<long long>(ls) * kWindows * kHeads * kWidth + list) * kListAt;
  const uint32_t a0 = __dp4a(t0, 0x01010101u, __dp4a(t1, 0x01010101u, 0u));
  const uint32_t a1 = __dp4a(t2, 0x01010101u, __dp4a(t3, 0x01010101u, 0u));
  const uint32_t a2 = __dp4a(t4, 0x01010101u, __dp4a(t5, 0x01010101u, 0u));
  if (at[0] != 0 || at[1] != a0 || at[2] != a0 + a1 || at[3] != a0 + a1 + a2 || at[4] != entries || at[5] != entries ||
      at[6] != entries || at[7] != entries)
    atomicAdd(bad + 1, 1ull);
  if (ls == 0) {
    const uint32_t want = static_cast<uint32_t>(static_cast<uint16_t>(ceil_exponent(pm))) |
                          static_cast<uint32_t>(static_cast<uint16_t>(ceil_exponent(pf))) << 16;
    if (p.list_exp[list] != want) ++badc;
  }
  if (badc) atomicAdd(bad, static_cast<unsigned long long>(badc));
}

// -------------------------------------------------------------------------------------------------------- host

struct Rig {
  cudaStream_t sa, sb;
  cudaEvent_t t0, t1, ev_stream[64], ev_build[64];
  cudaEvent_t tl[2][64][2];  // [stream, build][chunk][begin, end], recorded when FACTS_TWO_TIMELINE is set
};

int g_stream_ctas = 0;  // 0: a CTA a unit; else the persistent form with this many CTAs

template <bool kFuse>
void launch_stream(const Params& p, const Chunk& c, cudaStream_t s) {
  if (g_stream_ctas > 0)
    stream_p<kFuse><<<std::min(c.windows * kStreamUnits, g_stream_ctas), kStreamThreads, kSmBytesP, s>>>(p, c);
  else
    stream<kFuse><<<c.windows * kStreamUnits, kStreamThreads, kSmBytes, s>>>(p, c);
}

int g_build_ctas = 1 << 30;  // the build grid's cap: a CTA loops over units, so a small grid leaves SM room for the stream

void launch_build(const Params& p, const Chunk& c, cudaStream_t s) {
  const int units = c.windows * kListUnits;
  build<<<std::min((8 * units + kBuildWarps - 1) / kBuildWarps, g_build_ctas), kBuildWarps * 32,
          sizeof(BuildSmem) * kBuildWarps, s>>>(p, c);
}

//: the call as chunks of `cw` windows; MODE 0 stream alone, 1 build alone, 2 the pair on two streams, 3 the pair on one
double run_pair(Rig& r, const Params& p, int cw, int mode, int limit = kWindows) {
  //: cw == 0: a GEOMETRIC schedule, big chunks while the stream runs and small ones at the end, so the last build is short
  std::vector<Chunk> plan;
  if (cw == 0) {
    int at = 0;
    for (int w : {32, 32, 32, 16, 8, 4, 2, 2}) {
      plan.push_back(Chunk{at, w, static_cast<int>(plan.size()) & 1});
      at += w;
    }
  } else {
    for (int i = 0; i < std::min(limit, kWindows) / cw; ++i) plan.push_back(Chunk{i * cw, cw, i & 1});
  }
  const int chunks = static_cast<int>(plan.size());
  LAB_CHECK(cudaEventRecord(r.t0, r.sa));
  LAB_CHECK(cudaStreamWaitEvent(r.sb, r.t0));
  for (int i = 0; i < chunks; ++i) {
    const Chunk c = plan[i];
    if (mode == 3) {
      launch_stream<false>(p, c, r.sa);
      launch_build(p, c, r.sa);
      continue;
    }
    if (mode == 4) {
      launch_stream<true>(p, Chunk{c.win0, c.windows, 0}, r.sa);
      continue;
    }
    const bool timeline = mode == 2 && cw == 32 && std::getenv("FACTS_TWO_TIMELINE") != nullptr;
    if (mode != 1) {
      if (mode == 2 && i >= 2) LAB_CHECK(cudaStreamWaitEvent(r.sa, r.ev_build[i - 2]));
      if (timeline) LAB_CHECK(cudaEventRecord(r.tl[0][i][0], r.sa));
      launch_stream<false>(p, c, r.sa);
      if (timeline) LAB_CHECK(cudaEventRecord(r.tl[0][i][1], r.sa));
      LAB_CHECK(cudaEventRecord(r.ev_stream[i], r.sa));
    }
    if (mode != 0) {
      if (mode == 2) LAB_CHECK(cudaStreamWaitEvent(r.sb, r.ev_stream[i]));
      if (timeline) LAB_CHECK(cudaEventRecord(r.tl[1][i][0], r.sb));
      launch_build(p, c, r.sb);
      if (timeline) LAB_CHECK(cudaEventRecord(r.tl[1][i][1], r.sb));
      LAB_CHECK(cudaEventRecord(r.ev_build[i], r.sb));
    }
  }
  if (mode == 1 || mode == 2) LAB_CHECK(cudaStreamWaitEvent(r.sa, r.ev_build[chunks - 1]));
  LAB_CHECK(cudaEventRecord(r.t1, r.sa));
  LAB_CHECK(cudaDeviceSynchronize());
  float ms = 0.0f;
  LAB_CHECK(cudaEventElapsedTime(&ms, r.t0, r.t1));
  if (mode == 2 && cw == 32 && std::getenv("FACTS_TWO_TIMELINE") != nullptr) {
    for (int i = 0; i < chunks; ++i) {
      float a = 0, b = 0, c0 = 0, d = 0;
      LAB_CHECK(cudaEventElapsedTime(&a, r.t0, r.tl[0][i][0]));
      LAB_CHECK(cudaEventElapsedTime(&b, r.t0, r.tl[0][i][1]));
      LAB_CHECK(cudaEventElapsedTime(&c0, r.t0, r.tl[1][i][0]));
      LAB_CHECK(cudaEventElapsedTime(&d, r.t0, r.tl[1][i][1]));
      std::fprintf(stderr, "  chunk %d: stream %.0f..%.0f us, build %.0f..%.0f us\n", i, 1000 * a, 1000 * b, 1000 * c0, 1000 * d);
    }
  }
  return 1000.0 * ms;
}

}  // namespace facts_two

int main(int argc, char** argv) {
  using namespace facts_two;
  lab::Bench b(argc, argv);
  b.expect("facts_two::stream<\\(bool\\)0>\\(", "LDGSTS", ">=", 2);
  b.expect("facts_two::stream_p<\\(bool\\)0>\\(", "LDGSTS", ">=", 2);
  b.expect("facts_two::stream<\\(bool\\)1>\\(", "MATCH", ">=", 2);
  b.expect("facts_two::build\\(", "MATCH", ">=", 2);
  if (b.expect_only) return 0;

  bool smoke = false;
  int profile = 0;  // --profile N: after the check, one pair at N windows a chunk, no timing variants (for ncu)
  for (int i = 1; i < argc; ++i) {
    smoke |= !std::strcmp(argv[i], "--smoke");
    if (!std::strcmp(argv[i], "--profile") && i + 1 < argc) profile = std::atoi(argv[i + 1]);
  }
  if (const char* cap = std::getenv("FACTS_TWO_BUILD_CTAS")) g_build_ctas = std::atoi(cap);
  if (const char* cap = std::getenv("FACTS_TWO_STREAM_CTAS")) g_stream_ctas = std::atoi(cap);
  Params p{};
  uint8_t* rows = nullptr;
  float2* fm = nullptr;
  LAB_CHECK(cudaMalloc(&rows, kBytes));
  LAB_CHECK(cudaMalloc(&fm, sizeof(float2) * kTokens * kHeads));
  p.rows = rows;
  p.fm = fm;
  const long long list_units = 2LL * kMaxChunk * kListUnits;  // two buffers
  LAB_CHECK(cudaMalloc(&p.entries, list_units * kEntryMeta * 4));
  LAB_CHECK(cudaMalloc(&p.slots, list_units * kStashCap * 2));
  LAB_CHECK(cudaMalloc(&p.stash0, list_units * kStashCap * 4));
  LAB_CHECK(cudaMalloc(&p.stash1, list_units * kStashCap * 2));
  LAB_CHECK(cudaMalloc(&p.tile_unions, sizeof(uint32_t) * kLevelSides * kWindows * kHeads * kGroups * 8));
  const long long lists = static_cast<long long>(kWindows) * kHeads * kWidth;
  LAB_CHECK(cudaMalloc(&p.list_exp, 4 * lists));
  LAB_CHECK(cudaMalloc(&p.at, 2 * lists * kListAt * 2));
  LAB_CHECK(cudaMalloc(&p.rec0, lists * kListCap * 8));
  LAB_CHECK(cudaMalloc(&p.rec1, lists * kListCap * 4));
  const long long ugroups = 2LL * kWindows * kHeads * kGroups;
  LAB_CHECK(cudaMalloc(&p.union_tok, ugroups * kWindow * 2));
  LAB_CHECK(cudaMalloc(&p.union_count, ugroups * 2));
  unsigned long long* scratch = nullptr;
  LAB_CHECK(cudaMalloc(&scratch, 8 * sizeof(unsigned long long)));
  int* sink = nullptr;
  LAB_CHECK(cudaMalloc(&sink, 4 * 1024 * 1024));

  LAB_CHECK(cudaFuncSetAttribute(stream<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmBytes));
  LAB_CHECK(cudaFuncSetAttribute(stream<false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  LAB_CHECK(cudaFuncSetAttribute(stream<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmBytes));
  LAB_CHECK(cudaFuncSetAttribute(stream<true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  LAB_CHECK(cudaFuncSetAttribute(stream_p<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmBytesP));
  LAB_CHECK(cudaFuncSetAttribute(stream_p<false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  LAB_CHECK(cudaFuncSetAttribute(stream_p<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmBytesP));
  LAB_CHECK(cudaFuncSetAttribute(stream_p<true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  LAB_CHECK(cudaFuncSetAttribute(build, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sizeof(BuildSmem)) * kBuildWarps));
  LAB_CHECK(cudaFuncSetAttribute(build, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  Rig rig{};
  int least = 0, greatest = 0;
  LAB_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  LAB_CHECK(cudaStreamCreateWithPriority(&rig.sa, cudaStreamNonBlocking, least));
  LAB_CHECK(cudaStreamCreateWithPriority(&rig.sb, cudaStreamNonBlocking, greatest));
  LAB_CHECK(cudaEventCreate(&rig.t0));
  LAB_CHECK(cudaEventCreate(&rig.t1));
  for (int i = 0; i < 64; ++i) {
    LAB_CHECK(cudaEventCreate(&rig.ev_stream[i]));
    LAB_CHECK(cudaEventCreate(&rig.ev_build[i]));
    for (int a = 0; a < 2; ++a)
      for (int e = 0; e < 2; ++e) LAB_CHECK(cudaEventCreate(&rig.tl[a][i][e]));
  }

  //: the flat read's rate
  double flat_us = 0.0;
  {
    const int bytes = 16 * 4 * 1024;
    LAB_CHECK(cudaFuncSetAttribute(flat, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const lab::Samples s = b.variant(
        [&] {
          LAB_CHECK(cudaEventRecord(rig.t0));
          flat<<<b.sms, 512, bytes>>>(rows, sink);
          LAB_CHECK(cudaEventRecord(rig.t1));
        },
        [&] {
          float ms = 0.0f;
          LAB_CHECK(cudaEventElapsedTime(&ms, rig.t0, rig.t1));
          return 1000.0 * ms;
        });
    b.record("flat read of all the rows", "us", s);
    flat_us = s.median();
  }
  const double flat_rate = static_cast<double>(kBytes) / flat_us;  // bytes a us

  for (int k : {4, 16}) {
    LAB_CHECK(cudaMemset(rows, 0, kBytes));
    const long long heads = static_cast<long long>(kLevelSides) * kTokens * kHeads;
    fill<<<static_cast<int>((heads + 255) / 256), 256>>>(rows, fm, k);
    LAB_CHECK(cudaDeviceSynchronize());
    LAB_CHECK(cudaMemset(scratch, 0, 8 * sizeof(unsigned long long)));
    count_live<<<4 * b.sms, 256>>>(rows, 0, kLevelBytes, scratch);
    count_live<<<4 * b.sms, 256>>>(rows, kLevelBytes, kLevelBytes, scratch + 1);
    unsigned long long live[2];
    LAB_CHECK(cudaMemcpy(live, scratch, sizeof(live), cudaMemcpyDeviceToHost));

    if (profile > 0) {  // for ncu: the live counts above, then the pair and the fused kernel once at this chunk size
      std::fprintf(stderr, "facts_two profile: k %d, %llu live write entries, %llu read\n", k, live[0], live[1]);
      run_pair(rig, p, profile, 2);
      run_pair(rig, p, profile, 4);
      continue;
    }
    //: correctness first: the pair on two streams, then the reference over the first windows and the live count
    LAB_CHECK(cudaMemset(p.at, 0, 2 * lists * kListAt * 2));
    LAB_CHECK(cudaDeviceSynchronize());
    run_pair(rig, p, smoke ? 4 : 8, 2, smoke ? 8 : kWindows);
    LAB_CHECK(cudaMemset(scratch, 0, 8 * sizeof(unsigned long long)));
    constexpr int kChecked = 8;
    reference<<<static_cast<int>((2LL * kChecked * kHeads * kWidth + 63) / 64), 64>>>(p, kChecked, scratch);
    sum_lists<<<4 * b.sms, 256>>>(p.at, 2 * lists, p.union_count, ugroups, scratch + 1);
    unsigned long long h[4];
    LAB_CHECK(cudaMemcpy(h, scratch, sizeof(h), cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaDeviceSynchronize());
    //: the fused unit kernel, its first windows against the same reference (outputs cleared first)
    LAB_CHECK(cudaMemset(p.at, 0, 2 * lists * kListAt * 2));
    LAB_CHECK(cudaMemset(p.rec0, 0, lists * kListCap * 8));
    LAB_CHECK(cudaMemset(p.rec1, 0, lists * kListCap * 4));
    LAB_CHECK(cudaDeviceSynchronize());
    run_pair(rig, p, 4, 4, 8);
    LAB_CHECK(cudaMemset(scratch + 4, 0, 2 * sizeof(unsigned long long)));
    reference<<<static_cast<int>((2LL * kChecked * kHeads * kWidth + 63) / 64), 64>>>(p, kChecked, scratch + 4);
    unsigned long long fused_bad = 0;
    LAB_CHECK(cudaMemcpy(&fused_bad, scratch + 4, sizeof(fused_bad), cudaMemcpyDeviceToHost));
    {
      unsigned long long dbg[2];
      LAB_CHECK(cudaMemcpy(dbg, scratch + 4, sizeof(dbg), cudaMemcpyDeviceToHost));
      std::fprintf(stderr, "facts_two: fused: %llu entry/union/exponent mismatches, %llu pass-start mismatches\n", dbg[0], dbg[1]);
    }
    std::fprintf(stderr, "facts_two: k %d: the fused kernel differs from the reference in %llu places\n", k, fused_bad);
    if (fused_bad != 0) return 1;
    if (smoke) {
      std::fprintf(stderr, "facts_two smoke: k %d: %llu differ from the reference\n", k, h[0]);
      if (h[0] != 0) return 1;
      continue;
    }
    if (profile == 0 && !std::getenv("FACTS_TWO_NOCHECK") && (h[0] != 0 || h[1] != live[0] + live[1])) {
      std::fprintf(stderr, "facts_two: k %d: %llu entries differ from the reference, %llu listed against %llu live\n", k,
                   h[0], h[1], live[0] + live[1]);
      return 1;
    }
    std::fprintf(stderr,
                 "facts_two: k %d: reference agrees over %d windows; %llu live write entries, %llu read, %llu union tokens\n",
                 k, kChecked, live[0], live[1], h[2]);
    if (profile > 0) {
      run_pair(rig, p, profile, 2);
      if (k == 4) continue;
      return 0;
    }

    //: the floors: bytes at the flat read's rate. REQUIRED: the rows, the entries (8 + 4 bytes), the pass starts
    //: (16 bytes a list), the union tokens and counts, the exponents, the tile unions
    const double rows_b = static_cast<double>(kBytes);
    const double outs = live[0] * 8.0 + live[1] * 4.0 + 2.0 * lists * 16 + h[2] * 2.0 + ugroups * 2.0 + lists * 4.0 +
                        static_cast<double>(kLevelSides) * kWindows * kHeads * kGroups * 8 * 4;
    const double inter = 2.0 * (live[0] * 6.0 + live[1] * 4.0);  // the stash: coefficient and slot, written and read
    const double req_us = (rows_b + outs) / flat_rate, pair_us = (rows_b + outs + inter) / flat_rate;
    const std::string tag = "k " + std::to_string(k) + " / ";
    for (int cw : {8, 16, 32, 64, 128}) {
      const std::string chunk = cw == 0 ? std::string(" / chunk geometric") : " / chunk " + std::to_string(cw);
      const auto timed = [&](int mode) {
        double us = 0.0;
        return b.variant([&] { us = run_pair(rig, p, cw, mode); }, [&] { return us; });
      };
      const lab::Samples a = timed(0), bb = timed(1), both = timed(2), one = timed(3), fused = timed(4);
      b.record(tag + "stream alone" + chunk, "us", a);
      b.record(tag + "build alone" + chunk, "us", bb);
      b.record(tag + "pair on two streams" + chunk, "us", both);
      b.record(tag + "pair on one stream" + chunk, "us", one);
      b.record(tag + "fused unit kernel" + chunk, "us", fused);
      lab::Samples x1, x2;
      for (double v : both.v) {
        x1.v.push_back(v / req_us);
        x2.v.push_back(v / pair_us);
      }
      b.record(tag + "pair x required-bytes floor" + chunk, "x", x1);
      b.record(tag + "pair x pair floor" + chunk, "x", x2);
    }
    lab::Samples fl;
    fl.v.push_back(req_us);
    b.record(tag + "required-bytes floor", "us", fl);
    fl.v[0] = pair_us;
    b.record(tag + "pair floor", "us", fl);
  }
  return 0;
}
