// measure/lab/experiments/facts_block.cu -- QUESTION: what shape lets the facts pass's build hide under its row
// stream, at or under 1.15x its DRAM floor? METHOD: rows [level-side, T, H, 256] u8, each token's head live in k random
// digits; level-side 0 the write list level (its build forms 8-byte records whose coefficient needs the byte, the
// token's (F, m) pair and the window's maxima of b x gains), 1 the read list level (2-byte slot lists from its bits),
// 2-3 voted only. One block of sixteen warps an SM, warp w the window's w-th 32 tokens. The build is the facts pass's
// walk build (four warps a 32-digit group, each lane an equal share of its warp's entries, two walks and a segmented
// scan of ranks, round-robin places). FORMS: BULK, today's structure -- an item (window, level-side, head, 64 digits)
// at a time over two whole-window buffers, three barriers an item, each warp's bit words stored from the vote, the
// write level's maxima over every byte in the vote; TILED, the same with the bits leaving a shared tile as 64-byte
// runs after the barrier and the maxima in the build's first walk (over live entries); SPLIT, a unit (window, head,
// 128 digits) as the write level's two 64-digit halves in one held window and the other levels as 128-digit items
// through a per-warp ring of 8-token stages refilled as each is voted, one barrier an item, mbarrier waits sleeping
// between polls. Each form vote only and with the build, streamed and resident (no stream: the compute's own rate);
// time against the DRAM floor: the rows, the gains, the bits and union words, the lists and their pass starts, at a
// flat read's rate measured in the same run. Every build's lists checked against the live count and, by checksum, bit
// for bit against the first form's.
// see docs/internals/tools/lab.md#facts-block
#include "lab.cuh"

#include <cuda_fp16.h>

#include <map>
#include <type_traits>
#include <utility>

namespace facts_block {

constexpr int kWarps = 16, kThreads = kWarps * 32, kWindow = 512, kBlocks = kWindow / 32;
constexpr int kTokens = 65536, kHeads = 4, kWidth = 256, kLevelSides = 4;
constexpr long long kLevelBytes = static_cast<long long>(kTokens) * kHeads * kWidth;
constexpr long long kBytes = kLevelBytes * kLevelSides;
constexpr int kWindows = kTokens / kWindow;
constexpr int kPassSlots = 128, kMaxPasses = 4, kListCap = kWindow, kListAt = 8, kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127, kTileStride = 17, kParts = 4, kWords = kBlocks / kParts, kDigits = 32 / kParts;
constexpr int kKeys = kDigits * kMaxPasses;
constexpr float kListFloor = 1.17549435e-38f;

template <int W>
struct Shape {
  static constexpr int kGroups = W / 32, kSpans = kWidth / W;
  static constexpr int kItems = kWindows * kLevelSides * kHeads * kSpans, kRoleItems = kWindows * kHeads * kSpans;
  static constexpr int kItemBytes = kWindow * W, kBlockBytes = 32 * W;
};

struct Params {
  const uint8_t* rows;
  const __half2* fm;
  uint32_t* bits;
  uint32_t* unions;
  uint16_t* at;
  uint2* records;
  uint16_t* slots;
};

__device__ __forceinline__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

__global__ void fill(uint8_t* rows, __half2* fm, int k) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<long long>(kTokens) * kHeads)
    fm[i] = __floats2half2_rn(1e-3f * (1 + mix(static_cast<uint32_t>(i), 1u, 2u) % 1000u),
                              1e-3f * (1 + mix(static_cast<uint32_t>(i), 3u, 4u) % 1000u));
  if (i >= static_cast<long long>(kLevelSides) * kTokens * kHeads) return;
  for (int j = 0; j < k; ++j) {
    const uint32_t h = mix(static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0x5eedu);
    rows[i * kWidth + (h & 255u)] = static_cast<uint8_t>(1u + (h >> 8) % 255u);
  }
}

__global__ void count_live(const uint8_t* rows, unsigned long long* live) {
  unsigned long long n = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * kLevelBytes;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    n += rows[i] != 0;
  atomicAdd(live, n);
}

__global__ void count_lists(const uint16_t* at, long long lists, unsigned long long* total) {
  unsigned long long n = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < lists;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    n += at[i * kListAt + kMaxPasses];
  atomicAdd(total, n);
}

//: THE OUTPUTS' PLACES, independent of an item's width: a list level's 32-digit group (its first list in `at` is 32
//: times it, its union words 16 times it), and a group's first record list (per (window, head); the level-side is
//: the record kind)
__host__ __device__ __forceinline__ long long group_of(int ls, int window, int head, int group32) {
  return ((static_cast<long long>(ls) * kWindows + window) * kHeads + head) * (kWidth / 32) + group32;
}

__host__ __device__ __forceinline__ long long records_of(int window, int head, int group32) {
  return ((static_cast<long long>(window) * kHeads + head) * (kWidth / 32) + group32) * 32;
}

//: THE LISTS' CHECKSUM, every form's to agree bit for bit: each list's pass starts and length, each entry with its
//: place (write entries slot and coefficients, read entries slot), over the list levels of every (window, head, span)
__global__ void checksum(Params p, unsigned long long* out) {
  const long long lists = 2LL * kWindows * kHeads * kWidth;
  unsigned long long acc = 0;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < lists;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int digit = static_cast<int>(i % kWidth), head = static_cast<int>(i / kWidth % kHeads);
    const int window = static_cast<int>(i / kWidth / kHeads % kWindows);
    const int ls = static_cast<int>(i / kWidth / kHeads / kWindows);
    const uint16_t* const at = p.at + (group_of(ls, window, head, digit / 32) * 32 + digit % 32) * kListAt;
    const int n = at[kMaxPasses];
    const long long list = records_of(window, head, digit / 32) + digit % 32;
    acc += mix(at[0] | static_cast<uint32_t>(at[1]) << 16, at[2] | static_cast<uint32_t>(at[3]) << 16,
               static_cast<uint32_t>(n) ^ static_cast<uint32_t>(i) * 2654435761u);
    for (int e = 0; e < n; ++e) {
      const uint2 v = ls == 0 ? p.records[list * kListCap + e] : make_uint2(p.slots[list * kListCap + e], 0u);
      acc += mix(v.x, v.y ^ static_cast<uint32_t>(e), static_cast<uint32_t>(i));
    }
  }
  atomicAdd(out, acc);
}

__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }


template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

//: THE FLAT READ, the DRAM floor's rate: the same bytes as contiguous 1 KB stages a warp, three in flight, sunk
__global__ void __launch_bounds__(kThreads, 1) flat(const uint8_t* rows, long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kStages = 4, kStage = 1024;
  const int lane = lab::lane_id(), warp = lab::warp_id();
  const uint32_t ring = lab::smem(smem + warp * kStages * kStage);
  const long long stages = kBytes / kStage, per = stages / (gridDim.x * kWarps);
  const long long first = (static_cast<long long>(blockIdx.x) * kWarps + warp) * per;
  uint32_t sink = 0u;
  lab::WarpSpan clock;
  clock.start();
  const auto copy = [&](long long k) {
    if (k < per)
      for (int h = 0; h < 2; ++h) {
        const int q = 16 * (lane + 32 * h);
        const uint32_t dst = ring + static_cast<uint32_t>(k % kStages) * kStage + q;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                     "l"(rows + (first + k) * kStage + q));
      }
    commit();
  };
  for (int k = 0; k < kStages - 1; ++k) copy(k);
#pragma unroll 1
  for (long long k = 0; k < per; ++k) {
    copy(k + kStages - 1);
    wait<kStages - 1>();
    __syncwarp();
    sink ^= *reinterpret_cast<const uint32_t*>(smem + warp * kStages * kStage + (k % kStages) * kStage + 4 * lane);
    __syncwarp();
  }
  clock.stop(t);
  lab::sink(out, sink);
}


__device__ __forceinline__ int ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(static_cast<uint32_t>(127 + e) << 23); }

//: the place of rank `j` of residue `r` in a list dealt round-robin over residues (counts `lo`/`hi` bytewise, each at
//: most 16, `j` below 16): every residue's entries in rounds below j, then the residues before r still in round j.
//: Bytewise c > j from the carry of c + 127 - j into bit 7; sum of min(c, j) = sum of the counts at or below j + j per
//: count above
__device__ __forceinline__ int round_robin(uint32_t lo, uint32_t hi, int j, int r) {
  const uint32_t off = 0x7f7f7f7fu - 0x01010101u * static_cast<uint32_t>(j);
  const uint32_t gl = (lo + off) & 0x80808080u, gh = (hi + off) & 0x80808080u;
  const uint32_t ml = prmt<0xba98u>(gl, 0u), mh = prmt<0xba98u>(gh, 0u);
  const uint32_t le = __dp4a(lo & ~ml, 0x01010101u, __dp4a(hi & ~mh, 0x01010101u, 0u));
  const uint32_t below = r >= 4 ? gl | (gh & ((1u << (8 * (r - 4))) - 1u)) >> 4 : gl & ((1u << (8 * r)) - 1u);
  return static_cast<int>(le) + j * __popc(gl | gh >> 4) + __popc(below);
}

__device__ __forceinline__ int nth_bit(uint32_t w, int m) {
  int pos = 0;
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    const int c = __popc(w & ((1u << s) - 1u));
    const bool up = m >= c;
    m = up ? m - c : m;
    pos = up ? pos + s : pos;
    w = up ? w >> s : w;
  }
  return pos;
}

//: an item's place: the span innermost, then the head, the level-side, the window
template <int W>
__device__ __forceinline__ void decode(int item, int& span, int& head, int& ls, int& window) {
  span = item % Shape<W>::kSpans;
  head = (item / Shape<W>::kSpans) % kHeads;
  ls = (item / Shape<W>::kSpans / kHeads) % kLevelSides;
  window = item / Shape<W>::kSpans / kHeads / kLevelSides;
}

//: warp `w`'s copies of an item: its 32 tokens' W-byte runs, W / 16 consecutive lanes a run, each 32-byte quarter of a
//: run into its 32-digit sub-word's rows (the facts pass's word layout); on the write list level its tokens' gains
template <int W>
__device__ __forceinline__ void issue(const Params& p, int item, uint32_t buffer, uint32_t gains, bool all, int warp,
                                      int lane) {
  int span, head, ls, window;
  decode<W>(item, span, head, ls, window);
  if (ls == 0 || all) {
    const __half2* const g = p.fm + static_cast<long long>(window * kWindow + 32 * warp + lane) * kHeads + head;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(gains + (32 * warp + lane) * 4u), "l"(g));
  }
  constexpr int kQuanta = W / 16;
  const long long token = window * kWindow + 32 * warp;
  const uint8_t* const src = p.rows + ls * kLevelBytes + (token * kHeads + head) * kWidth + span * W;
  const uint32_t dst = buffer + warp * Shape<W>::kBlockBytes;
#pragma unroll
  for (int q = 0; q < kQuanta; ++q) {
    const int quantum = 32 * q + lane, token = quantum / kQuanta, part = quantum % kQuanta;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + (part >> 1) * 1024u + ring_row(token) * 32u
                                                                    + 16u * (part & 1)),
                 "l"(src + static_cast<long long>(token) * kHeads * kWidth + 16 * part));
  }
}

//: ONE 32-DIGIT GROUP'S LISTS by its four warps (`part`), the facts pass's balanced build (`build_lists`, P = 4): lane
//: `(dl, q)` digit dl's words 4q..4q+3; a write entry's byte from the item's buffer
template <bool kCoef, int W, bool kWalkMax = false>
__device__ void build(const Params& p, const uint32_t* tile, const uint32_t* live, const uint8_t* buffer,
                      const __half2* gains,
                      uint32_t* top, uint4* table, uint16_t* marks, int group, int part, int lane, long long gid,
                      long long rec) {
  constexpr int kLanesAWord = 32 / kWords;
  uint32_t* const cell_mask = reinterpret_cast<uint32_t*>(marks);
  uint16_t* const bases = marks + 16;
  const uint32_t un = lane < kBlocks ? live[lane] : 0u;
  uint32_t ub = __popc(un);
#pragma unroll
  for (int off = 1; off < kBlocks; off <<= 1) {
    const uint32_t x = __shfl_up_sync(0xffffffffu, ub, off);
    ub += lane >= off ? x : 0u;
  }
  ub -= __popc(un);
  if (lane < kBlocks) bases[lane] = static_cast<uint16_t>(ub);
  if (part == 0 && lane < kBlocks)
    p.unions[gid * kBlocks + lane] = un;

  const int q = lane % kParts, dl = lane / kParts;
  uint32_t pre = 0u, nonempty = 0u;
  int n = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pre |= static_cast<uint32_t>(n) << (8 * i);
    const int c = __popc(tile[(part * kDigits + dl) * kTileStride + q * kWords + i]);
    n += c;
    nonempty |= c != 0 ? 1u << i : 0u;
  }
#pragma unroll
  for (int x = 0; x < kWords; ++x) {
    const uint32_t v = __reduce_or_sync(0xffffffffu, lane / kLanesAWord == x ? nonempty << (lane % kLanesAWord * kWords)
                                                                           : 0u);
    if (lane == 0) cell_mask[x] = v;
  }
  int first = n;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, first, off);
    first += lane >= off ? x : 0;
  }
  const int total = __shfl_sync(0xffffffffu, first, 31);
  first -= n;
  table[lane] = make_uint4(0u, 0u, 0u, 0u);
  __syncwarp();

  const int share = lane * total >> 5, count = ((lane + 1) * total >> 5) - share;
  int c0 = 0;
  uint32_t bits0 = 0u, slot0 = 0u;
  if (total > 0) {
    const int g = min(share, total - 1);
    int o = 0;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const int x = __shfl_sync(0xffffffffu, first, o + s);
      o = x <= g ? o + s : o;
    }
    int m = g - __shfl_sync(0xffffffffu, first, o);
    const uint32_t sums = __shfl_sync(0xffffffffu, pre, o);
    const int w = __popc((((static_cast<uint32_t>(m) * 0x01010101u) | 0x80808080u) - sums) & 0x80808080u) - 1;
    m -= static_cast<int>((sums >> (8 * w)) & 0xffu);
    c0 = o * kWords + w;
    const uint32_t word = tile[(part * kDigits + (c0 >> 4)) * kTileStride + (c0 & 15)];
    const int t0 = nth_bit(word, m);
    bits0 = word & (0xffffffffu << t0);
    slot0 = bases[c0 & 15] + __popc(live[c0 & 15] & ((1u << t0) - 1u));
  }
  __syncwarp();
  const int key0 = (c0 >> 4) * kMaxPasses + static_cast<int>(slot0) / kPassSlots;
  const auto walk = [&](auto&& visit) {
    int c = c0;
    uint32_t bits = bits0, u = live[c0 & 15], b = bases[c0 & 15];
#pragma unroll 1
    for (int e = 0; e < count; ++e) {
      if (bits == 0u) {
        int x = c + 1;
        uint32_t mw = cell_mask[x >> 5] & (0xffffffffu << (x & 31));
        while (mw == 0u) {
          x = (x | 31) + 1;
          mw = cell_mask[x >> 5];
        }
        c = (x & ~31) + __ffs(mw) - 1;
        bits = tile[(part * kDigits + (c >> 4)) * kTileStride + (c & 15)];
        u = live[c & 15];
        b = bases[c & 15];
      }
      const int t = __ffs(bits) - 1;
      bits &= bits - 1u;
      visit(c, t, b + __popc(u & ((1u << t) - 1u)));
    }
  };

  int cur = key0, cur_dl = c0 >> 4;
  bool many = false;
  uint32_t run_lo = 0u, run_hi = 0u;
  float run_m = 0.0f, run_f = 0.0f;
  const auto flush = [&] {
    atomicMax(top + 2 * (32 * group + part * kDigits + cur_dl), __float_as_uint(run_m));
    atomicMax(top + 2 * (32 * group + part * kDigits + cur_dl) + 1, __float_as_uint(run_f));
  };
  walk([&](int c, int t, uint32_t slot) {
    if constexpr (kCoef && kWalkMax) {
      const int dl = c >> 4, j = c & 15;
      const float byte = static_cast<float>(
          buffer[j * Shape<W>::kBlockBytes + group * 1024 + ring_row(t) * 32 + part * kDigits + dl]);
      const float2 g = __half22float2(gains[32 * j + t]);
      if (dl != cur_dl) {
        flush();
        run_m = run_f = 0.0f;
        cur_dl = dl;
      }
      run_m = fmaxf(run_m, byte * g.x);
      run_f = fmaxf(run_f, byte * g.y);
    }
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint32_t inc = 1u << (8 * (r & 3));
    atomicAdd(reinterpret_cast<uint32_t*>(table + key) + (r >> 2), inc);
    const bool same = key == cur;
    many |= !same;
    cur = key;
    run_lo = (same ? run_lo : 0u) + (r < 4 ? inc : 0u);
    run_hi = (same ? run_hi : 0u) + (r < 4 ? 0u : inc);
  });
  if constexpr (kCoef && kWalkMax) {
    if (count > 0) flush();
  }
  const int cur_before = __shfl_up_sync(0xffffffffu, cur, 1);
  uint32_t v_lo = run_lo, v_hi = run_hi;
  int head_flag = many || lane == 0 || cur_before != cur;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t xl = __shfl_up_sync(0xffffffffu, v_lo, off), xh = __shfl_up_sync(0xffffffffu, v_hi, off);
    const int xf = __shfl_up_sync(0xffffffffu, head_flag, off);
    if (lane >= off) {
      v_lo += head_flag ? 0u : xl;
      v_hi += head_flag ? 0u : xh;
      head_flag |= xf;
    }
  }
  const bool carried = lane > 0 && cur_before == key0;
  const uint32_t before_lo = __shfl_up_sync(0xffffffffu, v_lo, 1), before_hi = __shfl_up_sync(0xffffffffu, v_hi, 1);
  __syncwarp();
  const long long lists = gid * 32;
  {
    const int key = lane, pass = key % kMaxPasses, d = part * kDigits + key / kMaxPasses;
    const uint4 c = table[key];
    const int s = static_cast<int>(__dp4a(c.x, 0x01010101u, __dp4a(c.y, 0x01010101u, 0u)));
    int incl = s;
#pragma unroll
    for (int off = 1; off < kMaxPasses; off <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, incl, off);
      incl += pass >= off ? x : 0;
    }
    const int end = __shfl_sync(0xffffffffu, incl, lane | (kMaxPasses - 1));
    uint16_t* const at = p.at + (lists + d) * kListAt;
    at[pass] = static_cast<uint16_t>(incl - s);
    at[kMaxPasses + pass] = static_cast<uint16_t>(end);
    uint32_t scales = 0u;
    if constexpr (kCoef) {
      const int em = kCoefTop - max(ceil_exponent(__uint_as_float(top[2 * (32 * group + d)])), kUnitMin);
      const int ev = kCoefTop - max(ceil_exponent(__uint_as_float(top[2 * (32 * group + d) + 1])), kUnitMin);
      scales = static_cast<uint32_t>(em + 128) | static_cast<uint32_t>(ev + 128) << 8;
    }
    table[key].z = static_cast<uint32_t>(incl - s);
    table[key].w = scales;
  }
  __syncwarp();

  cur = key0;
  run_lo = carried ? before_lo : 0u;
  run_hi = carried ? before_hi : 0u;
  const long long role_lists = rec;
  walk([&](int c, int t, uint32_t slot) {
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint4 tc = table[key];
    const bool same = key == cur;
    cur = key;
    run_lo = same ? run_lo : 0u;
    run_hi = same ? run_hi : 0u;
    const int rank = static_cast<int>(((r < 4 ? run_lo : run_hi) >> (8 * (r & 3))) & 0xffu);
    const uint32_t inc = 1u << (8 * (r & 3));
    run_lo += r < 4 ? inc : 0u;
    run_hi += r < 4 ? 0u : inc;
    const int d = part * kDigits + (c >> 4), j = c & 15;
    const long long at = (role_lists + d) * kListCap + static_cast<int>(tc.z) + round_robin(tc.x, tc.y, rank, r);
    if constexpr (kCoef) {
      const float byte = static_cast<float>(
          buffer[j * Shape<W>::kBlockBytes + group * 1024 + ring_row(t) * 32 + d]);
      const float2 g = __half22float2(gains[32 * j + t]);
      const float sm = pow2(static_cast<int>(tc.w & 0xffu) - 128), sv = pow2(static_cast<int>(tc.w >> 8) - 128);
      const uint32_t cv = __half_as_ushort(__float2half_rn(byte * g.y * sv));
      const uint32_t cm = __half_as_ushort(__float2half_rn(byte * g.x * sm));
      p.records[at] = make_uint2(slot, cv | cm << 16);
    } else {
      p.slots[at] = static_cast<uint16_t>(slot);
    }
  });
}

//: warp `w`'s vote of an item's 32 tokens: each 32-digit sub-word's words into the tile, each group's union word, and
//: on the write list level each digit's maxima of b x gains into the block's (float bits, maxed as integers)
template <int W, bool kMaxima>
__device__ __forceinline__ void vote(const Params& p, const uint8_t* buffer, const __half2* gains, uint32_t* tile,
                                     uint32_t* live, uint32_t* top, int item, int warp, int lane) {
  int span, head, ls, window;
  decode<W>(item, span, head, ls, window);
  const int tk = lane >> 3, dq = lane & 7;
  float2 g[8];
  if constexpr (kMaxima) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      g[i] = __half22float2(gains[32 * warp + 8 * tk + i]);
  }
#pragma unroll
  for (int s = 0; s < Shape<W>::kGroups; ++s) {
    const uint8_t* const slot = buffer + warp * Shape<W>::kBlockBytes + s * 1024;
    uint32_t flags = 0u;
    float mx[4] = {}, fx[4] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t x = *reinterpret_cast<const uint32_t*>(slot + (4 * i + tk) * 32 + 4 * dq);
      flags |= ((((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u) >> (7 - i);
      if constexpr (kMaxima) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const float bf = static_cast<float>((x >> (8 * jj)) & 0xffu);
          mx[jj] = fmaxf(mx[jj], bf * g[i].x);
          fx[jj] = fmaxf(fx[jj], bf * g[i].y);
        }
      }
    }
    uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
    flags = (tk & 2) ? prmt<0x3276u>(flags, y) : prmt<0x5410u>(flags, y);
    y = __shfl_xor_sync(0xffffffffu, flags, 8);
    flags = (tk & 1) ? prmt<0x3715u>(flags, y) : prmt<0x6240u>(flags, y);
    const int digit = 4 * dq + tk;
    tile[(32 * s + digit) * kTileStride + warp] = flags;
    const uint32_t any = __reduce_or_sync(0xffffffffu, flags);
    if (lane == 0) live[s * kBlocks + warp] = any;
    if constexpr (kMaxima) {
      float m = 0.0f, f = 0.0f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        float a = mx[jj], c = fx[jj];
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
        c = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 8));
        c = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 16));
        m = jj == tk ? a : m;
        f = jj == tk ? c : f;
      }
      atomicMax(top + 2 * (32 * s + digit), __float_as_uint(m));
      atomicMax(top + 2 * (32 * s + digit) + 1, __float_as_uint(f));
    }
  }
  //: the bits out, the warp's word of each of the span's rows
  uint32_t* const out = p.bits + ((static_cast<long long>(ls) * kHeads + head) * kWidth + span * W) * (kTokens / 32)
                        + window * kBlocks + warp;
#pragma unroll
  for (int s = 0; s < Shape<W>::kGroups; ++s)
    out[static_cast<long long>(32 * s + lane) * (kTokens / 32)] = tile[(32 * s + lane) * kTileStride + warp];
}

//: THE TILED VOTE of warp `w`'s 32 tokens of an item into the item's bit tile column, its union words out and into
//: `un`: every word of the item's groups loaded first, then each group's flags and transposition; the flags (lane =
//: digit) and union words returned
template <int W>
__device__ __forceinline__ void vote3(const Params& p, const uint8_t* buffer, uint32_t* tile, uint32_t* un, int item,
                                      int warp, int lane, uint32_t (&flags)[Shape<W>::kGroups],
                                      uint32_t (&any)[Shape<W>::kGroups]) {
  constexpr int G = Shape<W>::kGroups;
  const int tk = lane >> 3, dq = lane & 7, digit = 4 * dq + tk;
  uint32_t words[G][8];
#pragma unroll
  for (int s = 0; s < G; ++s)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      words[s][i] = *reinterpret_cast<const uint32_t*>(buffer + warp * Shape<W>::kBlockBytes + s * 1024
                                                         + (4 * i + tk) * 32 + 4 * dq);
#pragma unroll
  for (int s = 0; s < G; ++s) {
    flags[s] = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t x = words[s][i];
      flags[s] |= ((((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u) >> (7 - i);
    }
  }
#pragma unroll
  for (int s = 0; s < G; ++s) {
    const uint32_t y = __shfl_xor_sync(0xffffffffu, flags[s], 16);
    flags[s] = (tk & 2) ? prmt<0x3276u>(flags[s], y) : prmt<0x5410u>(flags[s], y);
  }
#pragma unroll
  for (int s = 0; s < G; ++s) {
    const uint32_t y = __shfl_xor_sync(0xffffffffu, flags[s], 8);
    flags[s] = (tk & 1) ? prmt<0x3715u>(flags[s], y) : prmt<0x6240u>(flags[s], y);
  }
  //: lane `digit` holds digit `digit`'s word; the walks below want lane `l` to hold digit `l`'s
#pragma unroll
  for (int s = 0; s < G; ++s) {
    any[s] = __reduce_or_sync(0xffffffffu, flags[s]);
    tile[(32 * s + digit) * kTileStride + warp] = flags[s];
    flags[s] = __shfl_sync(0xffffffffu, flags[s], 8 * (lane & 3) + (lane >> 2));
    if (lane == 0) {
      int span, head, ls, window;
      decode<W>(item, span, head, ls, window);
      p.unions[group_of(ls, window, head, span * G + s) * kBlocks + warp] = any[s];
      un[s * kBlocks + warp] = any[s];
    }
  }
}

//: an item's bits out of its tile, each digit's window of words one 64-byte run: warp `w` digits 4w.., two a store
template <int W>
__device__ __forceinline__ void bits_out(const Params& p, const uint32_t* tile, int item, int warp, int lane) {
  int span, head, ls, window;
  decode<W>(item, span, head, ls, window);
  uint32_t* const out = p.bits + ((static_cast<long long>(ls) * kHeads + head) * kWidth + span * W) * (kTokens / 32)
                        + window * kBlocks + (lane & 15);
#pragma unroll
  for (int r = 0; r < W / kWarps; r += 2) {
    const int digit = W / kWarps * warp + r + (lane >> 4);
    out[static_cast<long long>(digit) * (kTokens / 32)] = tile[digit * kTileStride + (lane & 15)];
  }
}

//: THE BULK FORM: an item at a time, three barriers an item. MODE bit 0 the build, bit 1 resident: the first two items'
//: copies only, every later item's vote and build over them (the compute's rate without the stream). TILED: the vote
//: into the bit tile with the bits out of it as 64-byte runs after the barrier, the maxima of b x gains in the build's
//: first walk (over live entries); else each warp's bit words out from the vote and the maxima over every byte
template <int W, int MODE, int K, bool TILED>
__global__ void __launch_bounds__(kThreads, 1) facts(Params p, long long* t) {
  using S = Shape<W>;
  constexpr bool kBuild = MODE & 1, kResident = MODE & 2;
  static_assert(K > 0);
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const buffers = smem;
  __half2* const gains = reinterpret_cast<__half2*>(smem + 2 * S::kItemBytes);
  uint32_t* const tile = reinterpret_cast<uint32_t*>(gains + 2 * kWindow);
  uint32_t* const live = tile + W * kTileStride;
  uint32_t* const top = live + S::kGroups * kBlocks;
  uint4* const tables = reinterpret_cast<uint4*>(top + 2 * 2 * W);
  uint16_t* const marks = reinterpret_cast<uint16_t*>(tables + kWarps * kKeys);
  const int lane = lab::lane_id(), warp = lab::warp_id();
  lab::WarpSpan clock;
  clock.start();
  for (int i = threadIdx.x; i < 2 * 2 * W; i += kThreads) top[i] = 0u;
  __syncthreads();
  int item = static_cast<int>(blockIdx.x), n = 0;
  if (item < S::kItems) {
    issue<W>(p, item, lab::smem(buffers), lab::smem(gains), kResident, warp, lane);
    commit();
  }
#pragma unroll 1
  while (item < S::kItems) {
    const int next = item + static_cast<int>(gridDim.x);
    if (next < S::kItems && (!kResident || n == 0))
      issue<W>(p, next, lab::smem(buffers + ((n + 1) & 1) * S::kItemBytes), lab::smem(gains + ((n + 1) & 1) * kWindow),
               kResident, warp, lane);
    commit();
    wait<1>();
    __syncwarp();
    const uint8_t* const buffer = buffers + (n & 1) * S::kItemBytes;
    const __half2* const gains_n = gains + (n & 1) * kWindow;
    uint32_t* const top_n = top + (n & 1) * 2 * W;
    int span, head, ls, window;
    decode<W>(item, span, head, ls, window);
    if constexpr (TILED) {
      uint32_t flags[S::kGroups], any[S::kGroups];
      vote3<W>(p, buffer, tile, live, item, warp, lane, flags, any);
    } else if (ls == 0) {
      vote<W, true>(p, buffer, gains_n, tile, live, top_n, item, warp, lane);
    } else {
      vote<W, false>(p, buffer, gains_n, tile, live, top_n, item, warp, lane);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 2 * W; i += kThreads) top[((n + 1) & 1) * 2 * W + i] = 0u;
    if constexpr (TILED) bits_out<W>(p, tile, item, warp, lane);
    if (kBuild && ls < 2 && warp < kParts * S::kGroups) {
      const int group = warp / kParts, part = warp % kParts;
      const long long gid = group_of(ls, window, head, span * S::kGroups + group);
      const long long rec = records_of(window, head, span * S::kGroups + group);
      if (ls == 0)
        build<true, W, TILED>(p, tile + 32 * group * kTileStride, live + group * kBlocks, buffer, gains_n, top_n,
                       tables + warp * kKeys, marks + warp * 64, group, part, lane, gid, rec);
      else
        build<false, W>(p, tile + 32 * group * kTileStride, live + group * kBlocks, buffer, gains_n, top_n,
                        tables + warp * kKeys, marks + warp * 64, group, part, lane, gid, rec);
    }
    __syncthreads();
    item = next;
    ++n;
  }
  clock.stop(t);
}

__device__ __forceinline__ uint32_t prmt_dyn(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

constexpr int kRingDigits = 128, kStageTokens = 8, kStageBytes = kStageTokens * kRingDigits, kRingDepth = 2;
constexpr int kRingOnlyDepth = 4;

//: an mbarrier wait that sleeps between polls: on sm_86 a wait is a shared word polled by the warp, and a tight poll
//: takes issue slots from the warps doing work
__device__ __forceinline__ void wait_landed(uint32_t bar, uint32_t parity) {
  while (!lab::mbar_test(bar, parity)) __nanosleep(64);
}
constexpr int kStagesAnItem = 32 / kStageTokens, kRingItems = 3, kSpans128 = kWidth / kRingDigits;
constexpr int kUnits = kWindows * kHeads * kSpans128;

//: THE SPLIT FORM'S SHARED MEMORY: the write level's window (64 digits, held through its build) with its gains, a
//: ring of 8-token x 128-digit stages a warp, the write level's tile and two rotating ring tiles, their union words,
//: the maxima, the builders' tables
struct SplitLayout {
  static constexpr int kX = 0, kGains = kX + Shape<64>::kItemBytes, kRing = kGains + kWindow * 4;
  static constexpr int kTile0 = kRing + kWarps * kRingDepth * kStageBytes;
  static constexpr int kTiles = kTile0 + 64 * kTileStride * 4;
  static constexpr int kLive0 = kTiles + 2 * kRingDigits * kTileStride * 4;
  static constexpr int kLives = kLive0 + 2 * kBlocks * 4, kTop = kLives + 2 * 4 * kBlocks * 4;
  static constexpr int kTables = kTop + 2 * 64 * 4, kMarks = kTables + kWarps * kKeys * 16;
  static constexpr int kBytes = kMarks + kWarps * 64 * 2;
};

//: THE SPLIT FORM: a block's units (window, head, 128-digit span) in turn. Only the write list level needs its bytes
//: after the vote (its build forms each entry's coefficient from the byte and the window's maxima), so it alone is a
//: held 64-digit window, two a unit; the read list level and the voted-only levels stream 128-byte runs a token
//: through a per-warp ring of 8-token stages, each stage refilled as soon as it is voted. A unit: write half 0, read,
//: voted, write half 1, voted -- a barrier after each vote; the write halves built by warps 0-7, the read level by
//: all sixteen (four a group); the held window refilled once its builders are done (waits sleep between polls: an
//: sm_86 mbarrier wait is a polled shared word, and a tight poll takes the builders' issue slots)
template <int MODE, int K>
__global__ void __launch_bounds__(kThreads, 1) split(Params p, long long* t) {
  using L = SplitLayout;
  constexpr bool kBuild = MODE & 1, kResident = MODE & 2, kRingOnly = MODE & 4;
  constexpr int kRingDepth = kRingOnly ? kRingOnlyDepth : ::facts_block::kRingDepth;
  static_assert(kWarps * kRingDepth * kStageBytes <= L::kTile0);
  static_assert(K > 0 && L::kBytes <= 101376 - 512 && !(kBuild && kRingOnly));
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const x = smem + L::kX;
  __half2* const gains = reinterpret_cast<__half2*>(smem + L::kGains);
  uint8_t* const ring = smem + (kRingOnly ? 0 : L::kRing);
  uint32_t* const tile0 = reinterpret_cast<uint32_t*>(smem + L::kTile0);
  uint32_t* const tiles = reinterpret_cast<uint32_t*>(smem + L::kTiles);
  uint32_t* const live0 = reinterpret_cast<uint32_t*>(smem + L::kLive0);
  uint32_t* const lives = reinterpret_cast<uint32_t*>(smem + L::kLives);
  uint32_t* const top = reinterpret_cast<uint32_t*>(smem + L::kTop);
  uint4* const tables = reinterpret_cast<uint4*>(smem + L::kTables);
  uint16_t* const marks = reinterpret_cast<uint16_t*>(smem + L::kMarks);
  __shared__ __align__(8) unsigned long long full_x[kWarps], full_ring[kWarps][kRingDepth], built;
  const int lane = lab::lane_id(), warp = lab::warp_id(), stride = static_cast<int>(gridDim.x);
  lab::WarpSpan clock;
  clock.start();
  if (lane == 0) {
    lab::mbar_init(lab::smem(&full_x[warp]), 32);
    for (int r = 0; r < kRingDepth; ++r) lab::mbar_init(lab::smem(&full_ring[warp][r]), 32);
  }
  if (threadIdx.x == 0) lab::mbar_init(lab::smem(&built), kWarps / 2);
  for (int i = threadIdx.x; i < 2 * 64; i += kThreads) top[i] = 0u;
  __syncthreads();
  const int units = (kUnits - static_cast<int>(blockIdx.x) + stride - 1) / stride;
  const auto unit_of = [&](int i, int& window, int& head, int& span) {
    const int q = static_cast<int>(blockIdx.x) + i * stride;
    span = q % kSpans128;
    head = q / kSpans128 % kHeads;
    window = q / kSpans128 / kHeads;
  };
  //: the write level's halves, f = 2 i + half: copies into the held window (with the gains), its barrier
  const auto fill_x = [&](int f) {
    if (kRingOnly) return;
    if (f / 2 < units) {
      int window, head, span;
      unit_of(f / 2, window, head, span);
      issue<64>(p, ((window * kLevelSides) * kHeads + head) * Shape<64>::kSpans + 2 * span + f % 2, lab::smem(x),
                lab::smem(gains), false, warp, lane);
    }
    lab::mbar_arrive_when_landed(lab::smem(&full_x[warp]));
  };
  //: ring stage n = (3 i + level - 1) x 4 + stage: this warp's 8 tokens' 128-byte runs, 8 lanes a run
  const auto fill_ring = [&](int n) {
    const int slot = n % kRingDepth, i = n / (kStagesAnItem * kRingItems);
    if (i < units) {
      int window, head, span;
      unit_of(i, window, head, span);
      const int ls = 1 + n / kStagesAnItem % kRingItems, st = n % kStagesAnItem;
      const uint32_t dst = lab::smem(ring + (warp * kRingDepth + slot) * kStageBytes);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = (lane >> 3) + 4 * h;
        const long long token = static_cast<long long>(window) * kWindow + 32 * warp + kStageTokens * st + row;
        const uint8_t* const src =
            p.rows + ls * kLevelBytes + (token * kHeads + head) * kWidth + span * kRingDigits + 16 * (lane & 7);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + row * kRingDigits + 16 * (lane & 7)),
                     "l"(src));
      }
    }
    lab::mbar_arrive_when_landed(lab::smem(&full_ring[warp][slot]));
  };
  int n_ring = 0, n_x = 0;
  //: a ring item's vote: lane (g, dq) digits 4 dq.. of group g, each stage's 8 tokens a byte of flags a digit
  const auto ring_vote = [&](int ls, int r, int window, int head, int span) {
    uint32_t v[kStagesAnItem];
#pragma unroll
    for (int st = 0; st < kStagesAnItem; ++st) {
      const int slot = n_ring % kRingDepth;
      if (!kResident || n_ring < kRingDepth)
        wait_landed(lab::smem(&full_ring[warp][slot]), static_cast<uint32_t>(n_ring / kRingDepth & 1));
      __syncwarp();
      const uint8_t* const stage = ring + (warp * kRingDepth + slot) * kStageBytes + 4 * lane;
      uint32_t acc = 0u;
#pragma unroll
      for (int i = 0; i < kStageTokens; ++i) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(stage + i * kRingDigits);
        acc |= ((((w & 0x7f7f7f7fu) + 0x7f7f7f7fu) | w) & 0x80808080u) >> (7 - i);
      }
      v[st] = acc;
      __syncwarp();
      if (!kResident) fill_ring(n_ring + kRingDepth);
      ++n_ring;
    }
    const int g = lane >> 3, dq = lane & 7;
    uint32_t* const tile = tiles + (r & 1) * kRingDigits * kTileStride;
    uint32_t any = 0u;
#pragma unroll
    for (int dd = 0; dd < 4; ++dd) {
      const uint32_t sel = static_cast<uint32_t>(dd) | static_cast<uint32_t>(4 + dd) << 4;
      const uint32_t f = prmt_dyn(prmt_dyn(v[0], v[1], sel), prmt_dyn(v[2], v[3], sel), 0x5410u);
      tile[(32 * g + 4 * dq + dd) * kTileStride + warp] = f;
      any |= f;
    }
    any |= __shfl_xor_sync(0xffffffffu, any, 1);
    any |= __shfl_xor_sync(0xffffffffu, any, 2);
    any |= __shfl_xor_sync(0xffffffffu, any, 4);
    if (dq == 0) {
      lives[(r & 1) * 4 * kBlocks + g * kBlocks + warp] = any;
      p.unions[group_of(ls, window, head, span * 4 + g) * kBlocks + warp] = any;
    }
  };
  const auto x_vote = [&](int item) {
    if (kRingOnly) return;
    if (!kResident || n_x < 1) wait_landed(lab::smem(&full_x[warp]), static_cast<uint32_t>(n_x & 1));
    ++n_x;
    __syncwarp();
    uint32_t flags[2], any[2];
    vote3<64>(p, x, tile0, live0, item, warp, lane, flags, any);
  };
  const auto build_x = [&](int half, int window, int head, int span) {
    if (kBuild && warp < kWarps / 2) {
      const int group = warp / kParts, part = warp % kParts, g32 = 2 * (2 * span + half) + group;
      build<true, 64, true>(p, tile0 + 32 * group * kTileStride, live0 + group * kBlocks, x, gains, top,
                            tables + warp * kKeys, marks + warp * 64, group, part, lane, group_of(0, window, head, g32),
                            records_of(window, head, g32));
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(lab::smem(&built));
    }
  };
  fill_x(0);
  for (int n = 0; n < kRingDepth; ++n) fill_ring(n);
  int n_built = 0;
#pragma unroll 1
  for (int i = 0; i < units; ++i) {
    int window, head, span;
    unit_of(i, window, head, span);
    const int x_item = ((window * kLevelSides) * kHeads + head) * Shape<64>::kSpans + 2 * span;
    const auto ring_item = [&](int ls) { return ((window * kLevelSides + ls) * kHeads + head) * kSpans128 + span; };
    //: write half 0; its window refilled with half 1 once its builders are done (no build: at once)
    x_vote(x_item);
    if (!kBuild && !kResident) fill_x(2 * i + 1);
    __syncthreads();
    bits_out<64>(p, tile0, x_item, warp, lane);
    build_x(0, window, head, span);
    //: the read level
    ring_vote(1, 3 * i, window, head, span);
    if (kBuild) wait_landed(lab::smem(&built), static_cast<uint32_t>(n_built++ & 1));
    if (kBuild && !kResident) fill_x(2 * i + 1);
    __syncthreads();
    for (int z = threadIdx.x; z < 2 * 64; z += kThreads) top[z] = 0u;
    bits_out<kRingDigits>(p, tiles + (3 * i & 1) * kRingDigits * kTileStride, ring_item(1), warp, lane);
    if (kBuild) {
      const int group = warp / kParts, part = warp % kParts, g32 = 4 * span + group;
      build<false, kRingDigits>(p, tiles + ((3 * i & 1) * kRingDigits + 32 * group) * kTileStride,
                                lives + (3 * i & 1) * 4 * kBlocks + group * kBlocks, nullptr, nullptr, top,
                                tables + warp * kKeys, marks + warp * 64, group, part, lane,
                                group_of(1, window, head, g32), records_of(window, head, g32));
    }
    //: a voted-only level
    ring_vote(2, 3 * i + 1, window, head, span);
    __syncthreads();
    bits_out<kRingDigits>(p, tiles + ((3 * i + 1) & 1) * kRingDigits * kTileStride, ring_item(2), warp, lane);
    //: write half 1; its window refilled with the next unit's half 0
    x_vote(x_item + 1);
    if (!kBuild && !kResident) fill_x(2 * i + 2);
    __syncthreads();
    bits_out<64>(p, tile0, x_item + 1, warp, lane);
    build_x(1, window, head, span);
    //: the other voted-only level
    ring_vote(3, 3 * i + 2, window, head, span);
    if (kBuild) wait_landed(lab::smem(&built), static_cast<uint32_t>(n_built++ & 1));
    if (kBuild && !kResident) fill_x(2 * i + 2);
    __syncthreads();
    for (int z = threadIdx.x; z < 2 * 64; z += kThreads) top[z] = 0u;
    bits_out<kRingDigits>(p, tiles + ((3 * i + 2) & 1) * kRingDigits * kTileStride, ring_item(3), warp, lane);
  }
  clock.stop(t);
}

double flat_rate = 0.0;

void flat_variant(const lab::Bench& b, const uint8_t* rows) {
  b.expect("::flat\\(", "LDGSTS", ">=", 1);
  if (b.expect_only) return;
  const int bytes = kWarps * 4 * 1024;
  LAB_CHECK(cudaFuncSetAttribute(flat, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant([&] { flat<<<b.sms, kThreads, bytes>>>(rows, b.times, b.sinks); },
                                   [&] { return static_cast<double>(kBytes) / b.span(kWarps); });
  b.record("flat read", "bytes a cycle", s);
  flat_rate = s.median();
}

template <int FORM, int W, int MODE, int K>
void variant(const lab::Bench& b, Params p, unsigned long long* scratch) {
  using S = Shape<W>;
  std::string name = FORM == 2 ? lab::kernel("split", {MODE, K}) : lab::kernel("facts", {W, MODE, K});
  if (FORM != 2) name = name.substr(0, name.size() - 3) + ", \\(bool\\)" + std::to_string(FORM) + ">\\(";
  b.expect(name, "LDGSTS", ">=", W / 16);
  if (b.expect_only) return;
  LAB_CHECK(cudaMemset(const_cast<uint8_t*>(p.rows), 0, kBytes));
  const long long heads = static_cast<long long>(kLevelSides) * kTokens * kHeads;
  fill<<<static_cast<int>((heads + 255) / 256), 256>>>(const_cast<uint8_t*>(p.rows), const_cast<__half2*>(p.fm), K);
  LAB_CHECK(cudaDeviceSynchronize());
  const int bytes = FORM == 2 ? (MODE & 4 ? SplitLayout::kTables : SplitLayout::kBytes)
                              : 2 * S::kItemBytes + 2 * kWindow * 4
                                    + (W * kTileStride + S::kGroups * kBlocks + 4 * W) * 4 + kWarps * kKeys * 16
                                    + kWarps * 64 * 2;
  const auto kernel = FORM == 2 ? split<MODE, K> : facts<W, MODE, K, FORM == 1>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (MODE == 1) {
    const long long role_lists = static_cast<long long>(Shape<32>::kRoleItems) * 32 * kListCap;
    LAB_CHECK(cudaMemset(p.at, 0, sizeof(uint16_t) * Shape<32>::kItems * 32 * kListAt));
    LAB_CHECK(cudaMemset(p.records, 0, sizeof(uint2) * role_lists));
    LAB_CHECK(cudaMemset(p.slots, 0, sizeof(uint16_t) * role_lists));
  }
  const bool ring_only = FORM == 2 && (MODE & 4);
  const double read = ring_only ? kBytes * 0.75 : static_cast<double>(kBytes);
  const lab::Samples s = b.variant([&] { kernel<<<b.sms, kThreads, bytes>>>(p, b.times); },
                                   [&] { return read / b.span(kWarps); });
  //: the DRAM floor's bytes: the rows, the write level's gains, the bits and union words out, and with the build the
  //: lists (8-byte write records, 2-byte read slots, their pass starts), at the flat read's rate
  double moved = read + read / 8 + (ring_only ? 3.0 : 4.0) * kWindows * kHeads * kWidth / 32 * kBlocks * 4;
  if (!ring_only) moved += static_cast<double>(kWindows) * kHeads * kWindow * 4 * 4;
  if (MODE & 1) {
    LAB_CHECK(cudaMemset(scratch, 0, sizeof(unsigned long long)));
    count_live<<<4 * b.sms, 256>>>(p.rows, scratch);
    unsigned long long live = 0;
    LAB_CHECK(cudaMemcpy(&live, scratch, sizeof(live), cudaMemcpyDeviceToHost));
    moved += live / 2.0 * (8 + 2) + 2.0 * kWindows * kHeads * kWidth * kListAt * 2;
  }
  lab::Samples floor;
  for (double v : s.v) floor.v.push_back(read / v / (moved / flat_rate));
  const std::string form = FORM == 2 ? "split / " : FORM ? "tiled / " : "bulk / ";
  b.record(form + (MODE & 1 ? "vote + build" : "vote") + (MODE & 2 ? " resident" : "") + (MODE & 4 ? " ring only" : "")
               + (FORM == 2 ? std::string() : " / " + std::to_string(W) + " digits") + " / k " + std::to_string(K),
           "bytes a cycle", s);
  b.record(form + (MODE & 1 ? "vote + build" : "vote") + (MODE & 2 ? " resident" : "") + (MODE & 4 ? " ring only" : "")
               + (FORM == 2 ? std::string() : " / " + std::to_string(W) + " digits") + " / k " + std::to_string(K)
               + " / DRAM floor",
           "x", floor);
  if (MODE == 1) {
    LAB_CHECK(cudaMemset(scratch, 0, 3 * sizeof(unsigned long long)));
    count_live<<<4 * b.sms, 256>>>(p.rows, scratch);
    count_lists<<<4 * b.sms, 256>>>(p.at, 2LL * kWindows * kHeads * kWidth, scratch + 1);
    checksum<<<4 * b.sms, 256>>>(p, scratch + 2);
    unsigned long long h[3];
    LAB_CHECK(cudaMemcpy(h, scratch, sizeof(h), cudaMemcpyDeviceToHost));
    if (h[0] != h[1]) {
      std::fprintf(stderr, "facts_block: %llu live entries on the list levels, %llu listed\n", h[0], h[1]);
      std::exit(1);
    }
    static std::map<std::pair<int, int>, unsigned long long> first;
    const auto [it, fresh] = first.emplace(std::make_pair(W, K), h[2]);
    if (!fresh && it->second != h[2]) {
      std::fprintf(stderr, "facts_block: form %d's lists at %d digits, k %d differ from the first form's\n", FORM, W,
                   K);
      std::exit(1);
    }
  }
}

}  // namespace facts_block

int main(int argc, char** argv) {
  using namespace facts_block;
  lab::Bench b(argc, argv);
  Params p{};
  unsigned long long* scratch = nullptr;
  if (!b.expect_only) {
    uint8_t* rows = nullptr;
    __half2* fm = nullptr;
    LAB_CHECK(cudaMalloc(&rows, kBytes));
    LAB_CHECK(cudaMalloc(&fm, sizeof(__half2) * kTokens * kHeads));
    p.rows = rows;
    p.fm = fm;
    const long long items = Shape<32>::kItems, role_items = Shape<32>::kRoleItems;
    LAB_CHECK(cudaMalloc(&p.bits, kBytes / 8));
    LAB_CHECK(cudaMalloc(&p.unions, sizeof(uint32_t) * items * kBlocks));
    LAB_CHECK(cudaMalloc(&p.at, sizeof(uint16_t) * items * 32 * kListAt));
    LAB_CHECK(cudaMemset(p.at, 0, sizeof(uint16_t) * items * 32 * kListAt));
    LAB_CHECK(cudaMalloc(&p.records, sizeof(uint2) * role_items * 32 * kListCap));
    LAB_CHECK(cudaMalloc(&p.slots, sizeof(uint16_t) * role_items * 32 * kListCap));
    LAB_CHECK(cudaMalloc(&scratch, 3 * sizeof(unsigned long long)));
  }
  flat_variant(b, p.rows);
  variant<0, 64, 0, 4>(b, p, scratch);
  variant<0, 64, 2, 4>(b, p, scratch);
  variant<0, 64, 1, 4>(b, p, scratch);
  variant<0, 64, 3, 4>(b, p, scratch);
  variant<0, 64, 1, 16>(b, p, scratch);
  variant<0, 64, 3, 16>(b, p, scratch);
  variant<1, 64, 0, 4>(b, p, scratch);
  variant<1, 64, 2, 4>(b, p, scratch);
  variant<1, 64, 1, 4>(b, p, scratch);
  variant<1, 64, 3, 4>(b, p, scratch);
  variant<1, 64, 1, 16>(b, p, scratch);
  variant<1, 64, 3, 16>(b, p, scratch);
  variant<2, 64, 0, 4>(b, p, scratch);
  variant<2, 64, 4, 4>(b, p, scratch);
  variant<2, 64, 2, 4>(b, p, scratch);
  variant<2, 64, 1, 4>(b, p, scratch);
  variant<2, 64, 3, 4>(b, p, scratch);
  variant<2, 64, 1, 16>(b, p, scratch);
  variant<2, 64, 3, 16>(b, p, scratch);
}
