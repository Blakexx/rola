// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once, a CTA a unit (a window's side,
// level and 32-digit group): both sides' class-1 words and tile unions, the window's columns, and each side's lists
// ranked and placed, an entry a slot and a byte; and the host entry that validates the bundle's views, plans a
// window's units and launches them.
// see docs/internals/facts/facts.md

#include "common/torch_seam.cuh"

#include <algorithm>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "facts/facts.cuh"
#include "facts/facts_api.cuh"

namespace rola {
namespace facts {
namespace detail {

namespace ops = ::denseref::ops;
using Clock = PhaseClock<kFactsPhases, true>;

//: THE FACTS PASS AT A PLAN: its device code over the plan `Fp`'s window. -- see docs/internals/facts/facts.md#window
template <class Fp>
struct FactsAt {
  static constexpr int kWindowTokens = Fp::kWindowTokens, kWindowWords = Fp::kWindowWords;
  static constexpr int kWindowTiles = Fp::kWindowTiles, kWarpBlocks = Fp::kWarpBlocks;
  static constexpr int kThreadTokens = Fp::kThreadTokens, kLaneWords = Fp::kLaneWords;
  static constexpr int kListCap = Fp::kListCap, kListAt = Fp::kListAt,
                       kLaneBounds = Fp::kLaneBounds;
  template <int G>
  using FactsSmem = typename Fp::template FactsSmem<G>;

  static __host__ __device__ constexpr int tile_at(int d, int j) { return Fp::tile_at(d, j); }

  //: A WINDOW BYTE: token `t` of its 32-token block at row `4 (t % 8) + t / 8` of the block's 1 KB, so the four tokens a
  //: vote lane group reads together sit in four consecutive rows; the group's digit `d` its byte in the row.
  //: -- see docs/internals/facts/facts.md#load
  static __device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

  static __device__ __forceinline__ int window_at(int t, int d) {
    return (t >> 5) * 1024 + ring_row(t & 31) * 32 + d;
  }

  //: A BYTE PERMUTE WITH ITS SELECTOR IMMEDIATE.
  template <uint32_t Sel>
  static __device__ __forceinline__ uint32_t prmt_imm(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
    return d;
  }

  //: bit 7 of each byte set iff the byte is nonzero (three operations for four digits)
  static __device__ __forceinline__ uint32_t nonzero_bytes(uint32_t x) {
    return (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
  }

  //: A UNIT'S PLACE: its code's fields and its (write head's batch-head, window). -- see docs/internals/facts/facts.md#the-grid
  struct Unit {
    int kind, side, level, group, head, b, h, bh, window;
  };

  static __device__ __forceinline__ Unit unit_of(const FactsParams& p) {
    const uint32_t code = p.unit[blockIdx.x];
    Unit x;
    x.kind = unit_kind(code);
    x.side = unit_side(code);
    x.level = unit_level(code);
    x.group = unit_group(code);
    x.head = unit_head(code);
    x.window = static_cast<int>(blockIdx.y);
    x.bh = static_cast<int>(blockIdx.z);
    x.b = x.bh / p.H;
    x.h = x.bh - x.b * p.H;
    return x;
  }

  static __device__ __forceinline__ bool digit_in(const FactsParams& p, int side, int level,
                                                  int digit) {
    const int row = p.row_base[level] + digit;
    return digit < p.B[level] && mask_bit(p.mask[side][row >> 5], row);
  }

  //: A UNIT'S LIST GROUPS: the side's `G` (32 in a one-group pass), whether group `q` holds a digit, group 0's index
  //: among the side's (group `q` `q` past it). -- see docs/internals/facts/facts.md#list-groups
  template <int kGroups>
  static __device__ __forceinline__ int list_group_of(const FactsParams& p, const Unit& x) {
    return kGroups > 1 ? p.list_group[x.side] : kGroupDigits;
  }

  template <int kGroups>
  static __device__ __forceinline__ bool list_group_in(const FactsParams& p, const Unit& x, int q) {
    const int at = list_group_of<kGroups>(p, x) * q;
    return at < kGroupDigits && kGroupDigits * x.group + at < p.B[x.level];
  }

  //: LIST GROUP `q`'S HALF of a pair of counts packed as 16-bit halves (a group's slots a window, `max_group` heads' at most); a
  //: one-group unit holds its count whole
  template <int kGroups>
  static __device__ __forceinline__ int half_of(uint32_t pair, int q) {
    return static_cast<int>(kGroups > 1 ? (pair >> (16 * q)) & 0xffffu : pair);
  }

  template <int kGroups>
  static __device__ __forceinline__ int64_t list_group_index(const FactsParams& p, const Unit& x) {
    const int G = list_group_of<kGroups>(p, x), lg = 31 - __clz(G);
    return (static_cast<int64_t>(x.bh) * p.windows + x.window) * ((p.B[x.level] + G - 1) >> lg)
           + ((kGroupDigits * x.group) >> lg);
  }

  //: THE WINDOW'S COLUMN of token `i`: `m` = the write scales' product in level order (the write side's shared scale)
  //: and `F = m s`, a token past `L` zero. -- see docs/internals/facts/facts.md#columns
  static __device__ __forceinline__ float2 load_column(const FactsParams& p, const Unit& x, int i) {
    const int t = x.window * kWindowTokens + i;
    if (t >= p.L) return make_float2(0.0f, 0.0f);
    float m = 1.0f;
#pragma unroll
    for (int l = 0; l < kMaxLevels; ++l)
      if (l < p.D)
        m *= __ldg(p.scale_write[l].base + x.b * p.scale_write[l].sb + x.h * p.scale_write[l].sh
                   + t * p.scale_write[l].st);
    const float s =
        __ldg(p.v_scale.base + x.b * p.v_scale.sb + x.h * p.v_scale.sh + t * p.v_scale.st);
    return make_float2(m, m * s);
  }

  //: THE WINDOW'S COPIES: token `t`'s 32 bytes of the group, two 16-byte quanta, a lane pair a token, past the rows'
  //: width or `L` zero-filled; a read unit's query head `j` of its write head's group. -- see docs/internals/facts/facts.md#load
  static __device__ __forceinline__ void issue_window(const FactsParams& p, const Unit& x,
                                                      ops::SmemAddr win, int tid, int j) {
    const RowView v = p.rows[x.side][x.level];
    const int head = x.side == kRead ? x.h * p.g + j : x.h;
    const int digit0 = kGroupDigits * x.group, part = tid & 1, i = tid >> 1;
    constexpr int kCopies = kWindowTokens * 2 / kFactsThreads, kStep = kFactsThreads / 2;
    const int t = x.window * kWindowTokens + i;
    const uint8_t* const src =
        v.base + x.b * v.sb + head * v.sh + static_cast<int64_t>(t) * v.st + digit0 + 16 * part;
    const int64_t step = kStep * v.st;
    const uint32_t run = static_cast<uint32_t>(max(0, min(16, v.width - digit0 - 16 * part)));
    const ops::SmemAddr dst = win + window_at(i, 16 * part);
#pragma unroll
    for (int k = 0; k < kCopies; ++k)
      ops::stage_tile_n(dst + window_at(kStep * k, 0), src + k * step,
                        t + kStep * k < p.L ? run : 0u);
  }

  //: THE COLUMNS OUT, the window's one unit's: each token's `(F, m)`. -- see docs/internals/facts/facts.md#columns
  static __device__ __forceinline__ void columns_out(const FactsParams& p, const Unit& x,
                                                     const float2 (&col)[kThreadTokens], int tid) {
#pragma unroll
    for (int k = 0; k < kThreadTokens; ++k) {
      const int t = x.window * kWindowTokens + tid + kFactsThreads * k;
      if (t < p.L) p.fm[static_cast<int64_t>(x.bh) * p.L + t] = make_float2(col[k].y, col[k].x);
    }
  }

  //: THE VOTE, a warp's two blocks: lane `(tk, dq)` token `8 tk + i`'s digits `4 dq ..` as a word, each byte's nonzero
  //: bit at bit `i`; a 4 x 4 byte transpose across the token blocks leaves digit `4 dq + tk`'s 32 token bits, masked;
  //: the warp's OR of each word over each of the unit's `kGroups` list groups' digits its union words.
  //: -- see docs/internals/facts/facts.md#vote
  template <int kGroups>
  static __device__ __forceinline__ void vote(const FactsParams& p, const Unit& x,
                                              const uint8_t* win, uint32_t* tile, uint32_t* uword,
                                              int warp, int lane) {
    const int tk = lane >> 3, dq = lane & 7, d = 4 * dq + tk;
    const bool in = digit_in(p, x.side, x.level, kGroupDigits * x.group + d),
               high = kGroups > 1 && d >= p.list_group[x.side];
#pragma unroll
    for (int jj = 0; jj < kWarpBlocks; ++jj) {
      const int j = kWarpBlocks * warp + jj;
      const uint8_t* const region = win + j * 1024;
      uint32_t flags = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        flags |=
            nonzero_bytes(*reinterpret_cast<const uint32_t*>(region + (4 * i + tk) * 32 + 4 * dq))
            >> (7 - i);
      uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
      flags = (tk & 2) ? prmt_imm<0x3276u>(flags, y) : prmt_imm<0x5410u>(flags, y);
      y = __shfl_xor_sync(0xffffffffu, flags, 8);
      flags = (tk & 1) ? prmt_imm<0x3715u>(flags, y) : prmt_imm<0x6240u>(flags, y);
      flags = in ? flags : 0u;
      tile[tile_at(d, j)] = flags;
      const uint32_t u0 = __reduce_or_sync(0xffffffffu, high ? 0u : flags);
      if (lane == 0) uword[j] = u0;
      if constexpr (kGroups > 1) {
        const uint32_t u1 = __reduce_or_sync(0xffffffffu, high ? flags : 0u);
        if (lane == 0) uword[kWindowWords + j] = u1;
      }
    }
  }

  //: THE BITS OUT at the words' `slot` (a query head's read side, or the write side's last): each digit's window of words
  //: one 64-byte run (a half-warp a digit); warp `w` tile `w`'s union, a digit a lane, one ballot.
  //: -- see docs/internals/facts/facts.md#tiles
  static __device__ __forceinline__ void bits_out(const FactsParams& p, const Unit& x,
                                                  const uint32_t* tile, int slot, int tid, int warp,
                                                  int lane) {
    const int word0 = x.window * kWindowWords, d = tid / kWindowWords, j = tid % kWindowWords;
    const int digits = p.B[x.level] - kGroupDigits * x.group;
    const bool word_in = j < p.token_words - word0;
    uint32_t* const out = p.words
                          + ((static_cast<int64_t>(x.bh) * (p.g + 1) + slot) * p.rows_total
                             + p.row_base[x.level] + kGroupDigits * x.group + d)
                                * p.token_words
                          + word0 + j;
    constexpr int kHalf = kFactsThreads / kWindowWords, kRowPasses = kGroupDigits / kHalf;
#pragma unroll
    for (int k = 0; k < kRowPasses; ++k)
      if (word_in & (d + kHalf * k < digits))
        out[static_cast<int64_t>(kHalf * k) * p.token_words] = tile[tile_at(d + kHalf * k, j)];
    constexpr int kWarpTiles = kWindowTiles / kFactsWarps;
#pragma unroll
    for (int tt = 0; tt < kWarpTiles; ++tt) {
      const int ti = kWarpTiles * warp + tt;
      const uint32_t any = __ballot_sync(
          0xffffffffu, (tile[tile_at(lane, 2 * ti)] | tile[tile_at(lane, 2 * ti + 1)]) != 0u);
      const int tile_index = x.window * kWindowTiles + ti;
      if (lane == 0 && tile_index < p.tiles)
        p.unions[((static_cast<int64_t>(x.bh) * (p.g + 1) + slot) * p.groups + p.group_base[x.level]
                  + x.group)
                     * p.tiles
                 + tile_index] = any;
    }
  }

  //: THE RANKS of query head `j`'s rows in each list group `q`, a warp's two words: each union row at its SLOT, half `q`
  //: of `at` past the heads before it; the token at the slot of each of the head's eight boundaries from `k0` its bound.
  //: The groups' counts ride one scan as 16-bit halves (`half_of`). Each list lane gets its own group's union words and
  //: the union's tokens before them; `n` the head's unions, packed. -- see docs/internals/facts/facts.md#ranks
  struct Ranks {
    uint32_t u[kLaneWords];
    int pre[kLaneWords];
    uint32_t n;
  };

  //: a side's rows a window: a query head's tokens for each of the group's heads on the read side
  static __device__ __forceinline__ int heads_of(const FactsParams& p, const Unit& x) {
    return x.side == kRead ? p.g : 1;
  }

  template <int kGroups>
  static __device__ __forceinline__ Ranks ranks(const FactsParams& p, const Unit& x, uint8_t* smem,
                                                int tid, int warp, int lane, uint32_t at, int j) {
    const uint32_t* const uword =
        reinterpret_cast<const uint32_t*>(smem + FactsSmem<kGroups>::kUword);
    int* const bound = reinterpret_cast<int*>(smem + FactsSmem<kGroups>::kBound);
    const ListsOut out = p.lists[x.side];
    const int slots = heads_of(p, x) * kWindowTokens;
    //: each group's union past its heads before
    uint16_t* union_tok[kGroups];
#pragma unroll
    for (int q = 0; q < kGroups; ++q)
      union_tok[q] =
          out.union_tok + (list_group_index<kGroups>(p, x) + q) * slots + half_of<kGroups>(at, q);
    const int i = lane & (kWindowWords - 1);
    const uint32_t ui[kUnitGroups] = {uword[i], kGroups > 1 ? uword[kWindowWords + i] : 0u};
    const uint32_t own =
        static_cast<uint32_t>(__popc(ui[0])) | static_cast<uint32_t>(__popc(ui[1])) << 16;
    uint32_t incl = own;
#pragma unroll
    for (int off = 1; off < kWindowWords; off <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, incl, off, kWindowWords);
      incl += i >= off ? y : 0u;
    }
    const uint32_t excl = incl - own, n = __shfl_sync(0xffffffffu, incl, kWindowWords - 1);

    const uint32_t below = (1u << lane) - 1u;
#pragma unroll
    for (int jj = 0; jj < kWarpBlocks; ++jj) {
      const int jw = kWarpBlocks * warp + jj, t = kTokBits * jw + lane;
      uint32_t us[kGroups];
      us[0] = __shfl_sync(0xffffffffu, ui[0], jw);
      const uint32_t pre = __shfl_sync(0xffffffffu, excl, jw);
      if constexpr (kGroups > 1) us[kGroups - 1] = __shfl_sync(0xffffffffu, ui[1], jw);
#pragma unroll
      for (int q = 0; q < kGroups; ++q) {
        const uint32_t u = us[q];
        if ((u >> lane) & 1u) {
          const int a = half_of<kGroups>(at, q),
                    slot = half_of<kGroups>(pre, q) + __popc(u & below);
          union_tok[q][slot] = static_cast<uint16_t>(kWindowTokens * j + t);
          const int c = (a + slot) / kListBound - 1 - a / kListBound;
          if ((a + slot) % kListBound == 0 && c >= 0) bound[kListAt * q + c] = t;
        }
      }
    }
    Ranks rk;
    rk.n = n;
    //: a list group's bounds past its union
    if (tid < kGroups * kListAt) {
      const int q = tid / kListAt, c = tid % kListAt, a = half_of<kGroups>(at, q);
      if (kListBound * (a / kListBound + c + 1) - a >= half_of<kGroups>(n, q))
        bound[tid] = kWindowTokens;
    }
    //: the lane's list group: its list's digit `warp + 8 (lane / 8)` at or past the side's group
    const int s = lane & 7;
    const bool high = kGroups > 1 && warp + kFactsWarps * (lane >> 3) >= p.list_group[x.side];
#pragma unroll
    for (int q = 0; q < kLaneWords; ++q) {
      const uint32_t e = __shfl_sync(0xffffffffu, excl, kLaneWords * s + q);
      rk.u[q] = __shfl_sync(0xffffffffu, ui[0], kLaneWords * s + q);
      if constexpr (kGroups > 1) {
        const uint32_t hi = __shfl_sync(0xffffffffu, ui[1], kLaneWords * s + q);
        rk.u[q] = high ? hi : rk.u[q];
      }
      rk.pre[q] = half_of<kGroups>(e, high);
    }
    return rk;
  }

  //: THE UNIONS' TAILS, after their heads: every slot past each list group's union `kGap`, and its size.
  template <int kGroups>
  static __device__ __forceinline__ void union_tail(const FactsParams& p, const Unit& x,
                                                    uint32_t at, int tid) {
    const int slots = heads_of(p, x) * kWindowTokens;
    const ListsOut out = p.lists[x.side];
    const int64_t lists_at = list_group_index<kGroups>(p, x);
#pragma unroll
    for (int q = 0; q < kGroups; ++q) {
      if (kGroups > 1 && !list_group_in<kGroups>(p, x, q)) continue;
      const int64_t group_index = lists_at + q;
      const int n = half_of<kGroups>(at, q);
#pragma unroll 1
      for (int k = n + tid; k < slots; k += kFactsThreads)
        out.union_tok[group_index * slots + k] = kGap;
      if (tid == 0) out.union_count[group_index] = n;
    }
  }

  //: A LIST LANE'S TOKENS BEFORE token `b`, lane `s`'s words `kLaneWords s ..`.
  static __device__ __forceinline__ int tokens_before(const uint32_t (&w)[kLaneWords], int s,
                                                      int b) {
    int n = 0;
#pragma unroll
    for (int jj = 0; jj < kLaneWords; ++jj) {
      const int r = b - kTokBits * (kLaneWords * s + jj);
      n += __popc(w[jj] & (r <= 0 ? 0u : (r >= kTokBits ? ~0u : (1u << r) - 1u)));
    }
    return n;
  }

  //: THE LISTS of query head `j`'s rows, a warp's 4 lists `w + 8 j`, 8 lanes a list, lane `s` its words `2s`, `2s + 1`:
  //: each lane's tokens before each of the head's boundaries' bounds, summed over the list's lanes into its boundaries
  //: past `placed` (the list's entries in the heads before) and scanned into the lane's first place; a walk places and
  //: writes each entry -- its slot past its list group's heads before, and its byte.
  //: -- see docs/internals/facts/facts.md#lists-out
  template <int kGroups>
  static __device__ __forceinline__ void lists_out(const FactsParams& p, const Unit& x,
                                                   uint8_t* smem, const Ranks& r, int warp,
                                                   int lane, uint32_t ats, int& placed) {
    const uint8_t* const win = smem + FactsSmem<kGroups>::kWin;
    const uint32_t* const tile =
        reinterpret_cast<const uint32_t*>(smem + FactsSmem<kGroups>::kTile);
    const int dl = lane >> 3, s = lane & 7, d = warp + kFactsWarps * dl;
    const bool high = kGroups > 1 && d >= p.list_group[x.side];
    const int at = half_of<kGroups>(ats, high);
    const int* const bound =
        reinterpret_cast<const int*>(smem + FactsSmem<kGroups>::kBound) + (high ? kListAt : 0);
    const int width = p.B[x.level], digit = kGroupDigits * x.group + d;
    const bool listed = digit < width;
    const ListsOut out = p.lists[x.side];
    const int64_t list = (static_cast<int64_t>(x.bh) * p.windows + x.window) * width + digit;
    uint32_t* const entries = out.entries + list * heads_of(p, x) * kListCap;
    uint32_t words[kLaneWords];
#pragma unroll
    for (int q = 0; q < kLaneWords; ++q) words[q] = tile[tile_at(d, kLaneWords * s + q)];

    //: the lane's tokens before the head's eight boundaries as u16 pairs, then its total, scanned over the list's lanes
    constexpr int kBoundPairs = kListAt / 2, kPairs = kBoundPairs + 1;
    uint32_t own[kPairs], inc[kPairs];
#pragma unroll
    for (int k = 0; k < kBoundPairs; ++k) {
      own[k] = static_cast<uint32_t>(tokens_before(words, s, bound[2 * k]))
               | static_cast<uint32_t>(tokens_before(words, s, bound[2 * k + 1])) << 16;
      inc[k] = own[k];
    }
    own[kBoundPairs] = 0u;
#pragma unroll
    for (int q = 0; q < kLaneWords; ++q)
      own[kBoundPairs] += static_cast<uint32_t>(__popc(words[q]));
    inc[kBoundPairs] = own[kBoundPairs];
#pragma unroll
    for (int off = 1; off < 8; off <<= 1)
#pragma unroll
      for (int k = 0; k < kPairs; ++k) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, inc[k], off, 8);
        inc[k] += s >= off ? y : 0u;
      }
    //: a head's boundary past the list's entries before it; column `kListAt` the list's entries through the head
    const int head_total = static_cast<int>(__shfl_sync(0xffffffffu, inc[kBoundPairs], 7, 8));
    uint16_t* const table =
        reinterpret_cast<uint16_t*>(smem + FactsSmem<kGroups>::kTable) + d * (kListAt + 1);
    //: lane `s` writes boundaries `kLaneBounds s ..` from the list's last lane, each pair shuffled and selected at once
    //: (an array of them is indexed through local memory); its first place is its tokens before
#pragma unroll
    for (int q = 0; q < kLaneBounds; ++q) {
      const int b = kLaneBounds * s + q;
      uint32_t mine = __shfl_sync(0xffffffffu, inc[0], 7, 8);
#pragma unroll
      for (int k = 1; k < kBoundPairs; ++k) {
        const uint32_t all = __shfl_sync(0xffffffffu, inc[k], 7, 8);
        mine = b >> 1 == k ? all : mine;
      }
      table[b] = listed ? static_cast<uint16_t>(placed + (b & 1 ? mine >> 16 : mine & 0xffffu))
                        : uint16_t{0};
    }
    if (s == 7) table[kListAt] = listed ? static_cast<uint16_t>(placed + head_total) : uint16_t{0};
    const int first = placed + static_cast<int>(inc[kBoundPairs] - own[kBoundPairs]);
    placed += head_total;

    //: the entries: the lane's `i`-th at its first place plus `i`
    int pos = first;
#pragma unroll
    for (int jj = 0; jj < kLaneWords; ++jj) {
      uint32_t xw = words[jj];
      for (int k = __any_sync(0xffffffffu, xw != 0u) != 0; k > 0; --k)
#pragma unroll 1
        do {
          if (xw != 0u) {
            const int bit = __ffs(xw) - 1, t = kTokBits * (kLaneWords * s + jj) + bit;
            xw &= xw - 1u;
            const uint32_t sl =
                static_cast<uint32_t>(at + r.pre[jj] + __popc(r.u[jj] & ((1u << bit) - 1u)));
            if (listed) entries[pos] = sl | static_cast<uint32_t>(win[window_at(t, d)]) << 16;
            ++pos;
          }
        } while (__any_sync(0xffffffffu, xw != 0u));
    }
  }

  //: THE BOUNDARIES OUT, a head's in list group `q`, warp `q`'s: each of its columns `k0 ..` (a later head's own
  //: overwrite the ones past this head's rows; the last head's entries through it fill every column past its eight) in
  //: PREFIX FORM over the group's lists, modulo 2^16 (a reader takes differences within a chunk), and its exact sum, the
  //: group's total. -- see docs/internals/facts/facts.md#lists-out
  template <int kGroups>
  static __device__ __forceinline__ void bounds_out(const FactsParams& p, const Unit& x,
                                                    uint8_t* smem, int lane, int q, int k0,
                                                    bool last) {
    const int G = list_group_of<kGroups>(p, x), columns = heads_of(p, x) * kListAt;
    const uint16_t* const table =
        reinterpret_cast<const uint16_t*>(smem + FactsSmem<kGroups>::kTable)
        + q * G * (kListAt + 1);
    const int64_t group_index = list_group_index<kGroups>(p, x) + q;
    uint16_t* const pre = p.lists[x.side].pre + group_index * (G + 1) * columns;
    int32_t* const totals = p.lists[x.side].totals + group_index * columns;
    //: lane `c` the head's column `k0 + c`, lane 8 every column past its eight on the last head
    const int c = min(lane, kListAt);
    const int lo = lane < kListAt ? k0 + lane : k0 + kListAt;
    const int hi =
        lane < kListAt ? min(k0 + lane + 1, columns) : (last && lane == kListAt ? columns : lo);
    int sum = 0;
#pragma unroll 1
    for (int l = 0; l <= G; ++l) {
#pragma unroll 1
      for (int k = lo; k < hi; ++k) pre[l * columns + k] = static_cast<uint16_t>(sum);
      if (l < G) sum += table[l * (kListAt + 1) + c];
    }
#pragma unroll 1
    for (int k = lo; k < hi; ++k) totals[k] = sum;
  }

  //: THE GRID: a CTA a unit, a window's units innermost. Every unit loads its window (a columns unit nothing) and a write
  //: list unit its columns; a vote or list unit votes and writes its bits; a list unit ranks its union and writes its
  //: lists -- a read list unit its write head's query heads in turn, each past the ones before. `kGroups` a unit's list
  //: groups at most: 1 when every side's is its whole unit. -- see docs/internals/facts/facts.md#the-grid
  template <int kGroups>
  static __device__ __forceinline__ void run(const FactsParams& p) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ long long s_clock[kFactsWarps][Clock::kRow];
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    Clock clock(&s_clock[warp][0], lane, p.ledger != nullptr);
    const Unit x = unit_of(p);
    uint8_t* const win = smem + FactsSmem<kGroups>::kWin;
    uint32_t* const tile = reinterpret_cast<uint32_t*>(smem + FactsSmem<kGroups>::kTile);
    const bool listed = x.kind == kUnitList;
    const int heads = listed ? heads_of(p, x) : 1;

    //: THE COLUMNS UNIT: each token's `(F, m)` from its scales, and nothing else
    if (x.kind == kUnitColumns) {
      float2 col[kThreadTokens];
#pragma unroll
      for (int k = 0; k < kThreadTokens; ++k) col[k] = load_column(p, x, tid + kFactsThreads * k);
      clock.lap(kPhaseLoad);
      columns_out(p, x, col, tid);
      clock.publish(p.ledger, 0, kFactsWarps, warp);
      return;
    }

    //: THE LOAD: the window's copies
    issue_window(p, x, ops::smem_addr(win), tid, x.head);
    ops::stage_commit();
    ops::stage_wait<0>();
    __syncthreads();
    clock.lap(kPhaseLoad);

    //: each list group's slots before the head, packed (`half_of`)
    uint32_t at = 0u;
    int placed = 0;
#pragma unroll 1
    for (int j = 0; j < heads; ++j) {
      const int head = listed ? j : x.head;
      if (j > 0) {
        //: the next head's window, once every warp is done with this one's
        __syncthreads();
        issue_window(p, x, ops::smem_addr(win), tid, head);
        ops::stage_commit();
        ops::stage_wait<0>();
        __syncthreads();
        clock.lap(kPhaseLoad);
      }
      //: THE VOTE AND THE BITS OUT
      vote<kGroups>(p, x, win, tile, reinterpret_cast<uint32_t*>(smem + FactsSmem<kGroups>::kUword),
                    warp, lane);
      __syncthreads();
      bits_out(p, x, tile, x.side == kRead ? head : p.g, tid, warp, lane);
      clock.lap(kPhaseVote);
      if (!listed) break;

      //: THE RANKS AND THE LISTS, each list group's
      const Ranks r = ranks<kGroups>(p, x, smem, tid, warp, lane, at, head);
      __syncthreads();
      clock.lap(kPhaseRanks);
      lists_out<kGroups>(p, x, smem, r, warp, lane, at, placed);
      __syncthreads();
      if (kGroups > 1 ? warp < kGroups && list_group_in<kGroups>(p, x, warp) : warp == 0)
        bounds_out<kGroups>(p, x, smem, lane, kGroups > 1 ? warp : 0,
                            half_of<kGroups>(at, warp) / kListBound, j == heads - 1);
      at += r.n;
      clock.lap(kPhaseLists);
    }
    if (listed) union_tail<kGroups>(p, x, at, tid);
    clock.publish(p.ledger, 0, kFactsWarps, warp);
  }
};

//: THE KERNEL at build `Bt::build` (a type: docs/internals/common/hardware.md#build-tag)
template <class Bt, int kGroups>
__global__ __launch_bounds__(kFactsThreads, FactsPlan<Bt::build>::kFactsCtasMin) void facts_pass(
    const FactsParams p) {
  FactsAt<FactsPlan<Bt::build>>::template run<kGroups>(p);
}

//: A level's rows as the kernel reads them: `[B, T, H, width]` u8, digits contiguous; the view's strides in bytes.
RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Byte, name, " is a CUDA u8 tensor, got ",
                  t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H, width] = [", B, ", ", T, ", H, width], got ", shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(3) == 1 || t.size(3) == 1, name, "'s digits are contiguous");
  //: THE ROW ALIGNMENT LAW: the first row and every stride of a dimension longer than one on 32 bytes
  const bool aligned = reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kRowAlign == 0
                       && (t.size(0) == 1 || t.stride(0) % kRowAlign == 0)
                       && (t.size(1) == 1 || t.stride(1) % kRowAlign == 0)
                       && (t.size(2) == 1 || t.stride(2) % kRowAlign == 0);
  STD_TORCH_CHECK(aligned, name,
                  " breaks the row alignment law: its rows start on and step by multiples of ",
                  kRowAlign, " bytes");
  return RowView{static_cast<const uint8_t*>(t.const_data_ptr()), t.stride(0), t.stride(1),
                 t.stride(2), static_cast<int>(t.size(3))};
}

ColumnView column_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Float, name,
                  " is a CUDA fp32 tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 3 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H] = [", B, ", ", T, ", ", H, "], got ", shape(t.sizes()));
  return ColumnView{t.const_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

//: THE PLAN, a window's units: each side's sparse levels in order, a unit a group -- a list unit at the side's list
//: level (the read side's its write head's query heads in turn), a vote unit elsewhere (the read side's one a query
//: head) -- and the window's columns unit.
//: -- see docs/internals/facts/facts.md#the-plan
void plan_units(FactsParams& p, const uint32_t (&dense)[kSides]) {
  int n = 0;
  for (int side = 0; side < kSides; ++side)
    for (int level = 0; level < p.D; ++level) {
      if (level_is_dense(dense[side], level)) continue;
      const bool list = level == p.list_level[side];
      const int heads = side == kRead && !list ? p.g : 1;
      for (int g = 0; g < (p.B[level] + kGroupDigits - 1) / kGroupDigits; ++g)
        for (int head = 0; head < heads; ++head)
          p.unit[n++] = unit_code(list ? kUnitList : kUnitVote, side, level, g, head);
    }
  p.unit[n++] = unit_code(kUnitColumns, kWrite, 0, 0, 0);
  p.per_window = n;
}

//: THE BOUND PHASE LEDGER, or null: the next launches book into it.
long long* g_ledger = nullptr;

//: THE PLAN'S NUMBERS the host sizes the call by, at a card's build. -- see docs/internals/facts/facts.md#window
struct HostPlan {
  int window, window_words, list_cap, list_at;
};

HostPlan plan_of(const Card& card) {
  HostPlan out{};
  build_switch(card, [&](auto b) {
    using Plan = FactsPlan<decltype(b)::build>;
    out = HostPlan{Plan::kWindowTokens, Plan::kWindowWords, Plan::kListCap, Plan::kListAt};
  });
  return out;
}

}  // namespace detail

//: -- see docs/internals/facts/facts_api.md#rola-facts-ledger-bind
void rola_facts_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 2
                      && ledger->size(0) == kFactsWarps && ledger->size(1) == kFactsPhases
                      && ledger->is_contiguous(),
                  "the facts pass's phase ledger is a contiguous CUDA int64 tensor [", kFactsWarps,
                  "][", kFactsPhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
}

//: -- see docs/internals/facts/facts_api.md#rola-facts-forward
std::vector<Tensor> rola_facts_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                                       std::vector<Tensor> scale_write, Tensor v_scale,
                                       std::vector<int64_t> widths, int64_t dense_read,
                                       std::vector<int64_t> mask_read, int64_t dense_write,
                                       std::vector<int64_t> mask_write, int64_t write_list_level,
                                       int64_t read_list_level, int64_t read_list_group,
                                       int64_t write_list_group) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the routing depth is one to four levels, got ", D);
  STD_TORCH_CHECK(static_cast<int>(read.size()) == D && static_cast<int>(write.size()) == D
                      && static_cast<int>(scale_write.size()) == D,
                  "each side holds one row set a level and the write side one scale a level: ", D);
  const Tensor& first = read[0];
  const int device = first.get_device_index();
  const Card& card = card_of(device);
  const detail::HostPlan s = detail::plan_of(card);
  const int W = s.window;
  STD_TORCH_CHECK(first.dim() == 4 && write[0].dim() == 4,
                  "a level's rows are [B, T, H, width], got ", shape(first.sizes()));
  //: GROUPED QUERIES: the read side's H query heads over the write side's H_w write heads, g = H / H_w each
  const int64_t B = first.size(0), T = first.size(1), Hq = first.size(2), H = write[0].size(2);
  STD_TORCH_CHECK(T >= 1, "a call carries at least one token");
  STD_TORCH_CHECK(H >= 1 && Hq % H == 0 && Hq / H <= max_group(W), "the read side's ", Hq,
                  " query heads are a whole number, at most ", max_group(W),
                  ", of the write side's ", H, " write heads");

  FactsParams p{};
  p.D = D;
  p.H = static_cast<int>(H);
  p.g = static_cast<int>(Hq / H);
  p.L = static_cast<int>(T);
  p.token_words = token_words(p.L);
  p.windows = (p.token_words + s.window_words - 1) / s.window_words;
  const std::vector<Tensor>* sides[kSides] = {&read, &write};
  const int64_t dense_in[kSides] = {dense_read, dense_write};
  const std::vector<int64_t>* masks[kSides] = {&mask_read, &mask_write};
  uint32_t dense[kSides] = {};
  for (int level = 0; level < D; ++level) {
    STD_TORCH_CHECK(widths[level] >= 1 && widths[level] <= kMaxWidth, "level ", level, " is ",
                    widths[level], " digits wide; a level is 1 to ", kMaxWidth);
    p.B[level] = static_cast<int>(widths[level]);
    p.row_base[level] = p.rows_total;
    p.rows_total += p.B[level];
  }
  const int need = mask_words(p.rows_total);
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(dense_in[side] >= 0 && dense_in[side] < (1 << D), "side ", side,
                    "'s dense-level bitmask ", dense_in[side], " names a level outside the depth ",
                    D);
    STD_TORCH_CHECK(static_cast<int>(masks[side]->size()) == need, "side ", side, " carries ",
                    masks[side]->size(), " digit-mask words; ", p.rows_total, " rows need ", need);
    dense[side] = static_cast<uint32_t>(dense_in[side]);
    for (int w = 0; w < need; ++w) p.mask[side][w] = static_cast<uint32_t>((*masks[side])[w]);
    for (int level = 0; level < D; ++level) {
      if (level_is_dense(dense[side], level)) continue;
      const Tensor& rows = (*sides[side])[level];
      p.rows[side][level] =
          detail::row_view(rows, B, T, side == kRead ? Hq : H, side ? "write[l]" : "read[l]");
      //: a digit the mask names is read, so it lies inside the level's row
      const int64_t row_width = rows.size(3);
      for (int digit = 0; digit < p.B[level]; ++digit) {
        const int row = p.row_base[level] + digit;
        STD_TORCH_CHECK(digit < row_width || !mask_bit(p.mask[side][row >> 5], row), "side ", side,
                        " level ", level, "'s mask names digit ", digit, " past its ", row_width,
                        "-digit rows");
      }
    }
  }
  const int64_t list_levels[kSides] = {read_list_level, write_list_level};
  const int64_t list_groups[kSides] = {read_list_group, write_list_group};
  for (int side = 0; side < kSides; ++side) {
    const int64_t level = list_levels[side];
    STD_TORCH_CHECK(level >= -1 && level < D && (level < 0 || !level_is_dense(dense[side], level)),
                    "side ", side, "'s list level is a sparse level of the depth ", D,
                    " or -1 for none, got ", level);
    p.list_level[side] = static_cast<int>(level);
    STD_TORCH_CHECK(list_groups[side] == kListGroupMin || list_groups[side] == kGroupDigits,
                    "side ", side, "'s list group is ", kListGroupMin, " or ", kGroupDigits,
                    " digits, got ", list_groups[side]);
    p.list_group[side] = static_cast<int>(list_groups[side]);
  }
  for (int level = 0; level < D; ++level)
    p.scale_write[level] = detail::column_view(scale_write[level], B, T, H, "scale_write[l]");
  p.v_scale = detail::column_view(v_scale, B, T, H, "v_scale");
  LivenessLayout layout{D, {}, p.L};
  for (int level = 0; level < D; ++level) layout.B[level] = p.B[level];
  for (int level = 0; level < D; ++level) p.group_base[level] = layout.group_base(level);
  p.groups = layout.groups();
  p.tiles = layout.tiles();
  //: THE OUTPUTS, a write head's: the words and unions (its query heads' read slots, then its write side's), the
  //: factor products, then each side's lists at its list level
  const int64_t BH = B * H;
  const auto out = [&](std::initializer_list<int64_t> dims, Dtype dtype) {
    return empty_cuda(Shape(std::vector<int64_t>(dims)), dtype, device);
  };
  std::vector<Tensor> outs = {out({BH, p.g + 1, p.rows_total, p.token_words}, Dtype::Int),
                              out({BH, p.g + 1, p.groups, p.tiles}, Dtype::Int),
                              out({BH, T, 2}, Dtype::Float)};
  p.words = reinterpret_cast<uint32_t*>(outs[0].mutable_data_ptr<int32_t>());
  p.unions = reinterpret_cast<uint32_t*>(outs[1].mutable_data_ptr<int32_t>());
  p.fm = static_cast<float2*>(outs[2].mutable_data_ptr());
  //: THE LISTS OUT, a side's at its list level (none: width 0): entries, boundaries, the union by slot, its size
  for (const int side : {kWrite, kRead}) {
    const int64_t width = p.list_level[side] >= 0 ? p.B[p.list_level[side]] : 0,
                  G = p.list_group[side], groups = (width + G - 1) / G,
                  heads = side == kRead ? p.g : 1;
    const Tensor entries = out({BH, p.windows, width, heads * s.list_cap}, Dtype::Int);
    const Tensor pre = out({BH, p.windows, groups, G + 1, heads * s.list_at}, Dtype::Short);
    const Tensor totals = out({BH, p.windows, groups, heads * s.list_at}, Dtype::Int);
    const Tensor union_tok = out({BH, p.windows, groups, heads * s.list_cap}, Dtype::Short);
    const Tensor union_count = out({BH, p.windows, groups}, Dtype::Int);
    if (width > 0) {
      ListsOut& o = p.lists[side];
      o.entries = static_cast<uint32_t*>(entries.mutable_data_ptr());
      o.pre = static_cast<uint16_t*>(pre.mutable_data_ptr());
      o.totals = static_cast<int32_t*>(totals.mutable_data_ptr());
      o.union_tok = static_cast<uint16_t*>(union_tok.mutable_data_ptr());
      o.union_count = static_cast<int32_t*>(union_count.mutable_data_ptr());
    }
    outs.insert(outs.end(), {entries, pre, totals, union_tok, union_count});
  }
  detail::plan_units(p, dense);
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  //: THE GRID: a window's units by x, the windows by y, the batch-heads by z
  STD_TORCH_CHECK(p.windows <= 65535 && B * H <= 65535,
                  "the grid is (units a window, windows, batch-heads): ", p.windows,
                  " windows and ", B * H, " batch-heads, each at most 65535");
  //: THE PASS a unit's list groups need -- one group a unit unless a side splits its units in halves -- and its tenants'
  //: bytes at the device's window
  const bool halves = p.list_group[kRead] < kGroupDigits || p.list_group[kWrite] < kGroupDigits;
  build_switch(card, [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB)) {
      using Plan = FactsPlan<kB>;
      using Bt = ::denseref::arch::BuildTag<kB>;
      const auto bytes_of = [](bool split) {
        return split ? Plan::template FactsSmem<kUnitGroups>::kBytes
                     : Plan::template FactsSmem<1>::kBytes;
      };
      const auto pass_of = [](bool split) {
        return split ? detail::facts_pass<Bt, kUnitGroups> : detail::facts_pass<Bt, 1>;
      };
      const int smem_bytes = bytes_of(halves);
      //: ONCE: each pass's carveout at its most shared memory, so the SM holds as many units as it can
      static bool configured = false;
      if (!configured) {
        for (const bool split : {false, true}) {
          const int bytes = bytes_of(split);
          cudaFuncAttributes attributes{};
          ROLA_CUDA_CHECK(cudaFuncGetAttributes(&attributes, pass_of(split)));
          int most = 0;
          ROLA_CUDA_CHECK(
              cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
          STD_TORCH_CHECK(static_cast<int>(attributes.sharedSizeBytes) + bytes <= most,
                          "the facts pass's ", bytes, " bytes of dynamic and ",
                          attributes.sharedSizeBytes,
                          " of static shared memory exceed this device's ", most, " a block");
          ROLA_CUDA_CHECK(cudaFuncSetAttribute(pass_of(split),
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
          ROLA_CUDA_CHECK(cudaFuncSetAttribute(pass_of(split),
                                               cudaFuncAttributePreferredSharedMemoryCarveout,
                                               cudaSharedmemCarveoutMaxShared));
        }
        configured = true;
      }
      const dim3 grid(static_cast<unsigned>(p.per_window), static_cast<unsigned>(p.windows),
                      static_cast<unsigned>(B * H));
      pass_of(halves)<<<grid, kFactsThreads, smem_bytes, current_stream()>>>(p);
      ROLA_CUDA_LAUNCH_CHECK();
    }
  });
  return outs;
}

}  // namespace facts
}  // namespace rola
