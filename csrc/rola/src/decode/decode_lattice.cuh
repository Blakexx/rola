#pragma once

// THE T=1 FACTOR TABLES -- the per-level objects the step's lattice walk is
// enumerated from, and nothing else.
// see docs/internals/decode/decode_lattice.md#note-l3

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {

using LatticeScan = cub::BlockScan<int, kDecodeThreads>;

//: THE FOUR PER-LEVEL DIGIT SETS the terms are built from, as the index of one -- see docs/internals/decode/decode_lattice.md#kkindw
constexpr int kKindW = 0;
constexpr int kKindR = 1;
constexpr int kKindRW = 2;   // R_l & W_l
constexpr int kKindRmW = 3;  // R_l & ~W_l
constexpr int kKinds = 4;

//: `D + 2` term slots: the write product, one per level, and -- LAST -- the -- see docs/internals/decode/decode_lattice.md#kmaxterms
constexpr int kMaxTerms = kMaxLevels + 2;

//: THE TERM TABLE, as arithmetic rather than as a table: term `0` is the write product, -- see docs/internals/decode/decode_lattice.md#termkind
__device__ __forceinline__ int term_kind(int term, int l, int D) {
  if (term == 0 || term > D) return kKindW;  // term 0 and the candidate slot are `prod W`
  const int split = term - 1;
  return l < split ? kKindRW : (l == split ? kKindRmW : kKindR);
}

//: A KIND'S BITS, from the two sides' bits. One expression, used on single digits and on -- see docs/internals/decode/decode_lattice.md#kindbits
__device__ __forceinline__ uint32_t kind_bits(uint32_t r, uint32_t w, int kind) {
  switch (kind) {
    case kKindW:
      return w;
    case kKindR:
      return r;
    case kKindRW:
      return r & w;
    default:
      return r & ~w;
  }
}

//: One side's DIGIT MASK, one bit per digit of every level, packed level by level. One -- see docs/internals/decode/decode_lattice.md#near-line-53
template <int D>
__device__ __forceinline__ void build_digit_mask(const DecodeParams& p, const float* amp,
                                                 uint32_t* mask) {
#pragma unroll 1
  for (int idx = threadIdx.x; idx < p.mask_total; idx += kDecodeThreads) {
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (idx >= p.mask_off[q]) l = q;
    }
    const int base = (idx - p.mask_off[l]) << 5;
    const int width = p.level_width[l];
    const float* row = amp + p.level_amp_offset[l];
    uint32_t v = 0u;
    const int take = (width - base) < 32 ? (width - base) : 32;
#pragma unroll 1
    for (int b = 0; b < take; ++b) {
      if (row[base + b] != 0.0f) v |= 1u << b;
    }
    mask[idx] = v;
  }
}

//: Level `l`'s digits `[lo, lo + count)` as one word. `count <= kDecodeMaxSpan` is the -- see docs/internals/decode/decode_lattice.md#levelslice
__device__ __forceinline__ uint32_t level_slice(const DecodeParams& p, const uint32_t* mask, int l,
                                                int lo, int count) {
  return extract_bits(mask + p.mask_off[l], p.mask_words[l], lo, count);
}

__device__ __forceinline__ int ceil_log2_dev(int n) {
  int bits = 0;
  while ((1 << bits) < n) ++bits;
  return bits;
}

//: ONE DIGIT'S MEMBERSHIP, as ONE load and a shift. `level_slice` reads two words and -- see docs/internals/decode/decode_lattice.md#maskbit
__device__ __forceinline__ bool mask_bit(const DecodeParams& p, const uint32_t* mask, int l,
                                         int bit) {
  return ((mask[p.mask_off[l] + (bit >> 5)] >> (static_cast<uint32_t>(bit) & 31u)) & 1u) != 0u;
}

//: THE OWNER LIVENESS BITMASK: per KIND and level, ONE BIT PER OWNER -- see docs/internals/decode/decode_lattice.md#build-owner-mask
template <int D>
__device__ __forceinline__ void build_owner_mask(const DecodeParams& p, const uint32_t* mask_r,
                                                 const uint32_t* mask_w, uint32_t* omask,
                                                 int32_t* n_o, int32_t* n_d) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int word = tid >> 5;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int span = p.lat_s[l];
    uint32_t r = 0u;
    uint32_t w = 0u;
    if (tid < p.lat_g[l]) {
      const int lo = tid * span;
      r = level_slice(p, mask_r, l, lo, span);
      w = level_slice(p, mask_w, l, lo, span);
    }
    //: HAZARD ballot-uniformity -- docs/internals/decode/decode_lattice.md#ballot-uniformity
    const bool keep = lane == 0 && word < p.omask_words[l];
#pragma unroll
    for (int k = 0; k < kKinds; ++k) {
      const uint32_t bits = __ballot_sync(0xffffffffu, kind_bits(r, w, k) != 0u);
      if (keep) {
        omask[k * p.omask_total + p.omask_off[l] + word] = bits;
        atomicAdd(&n_o[k * kMaxLevels + l], __popc(bits));
      }
    }
    if (l < D - 1 && tid < p.mask_words[l]) {
      atomicAdd(&n_d[l], __popc(mask_w[p.mask_off[l] + tid]));
    }
  }
}

//: THE COMPACTION, AND IT COSTS ONE `POPC` PER WORD. -- see docs/internals/decode/decode_lattice.md#compact-owner-lists
template <int D>
__device__ __forceinline__ void compact_owner_lists(const DecodeParams& p, const uint32_t* omask,
                                                    const uint32_t* mask_w, int32_t* olist,
                                                    int32_t* dlist) {
  const int tid = threadIdx.x;
  const int w_hi = tid >> 5;
  const uint32_t below = (1u << (tid & 31)) - 1u;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    if (tid < p.lat_g[l]) {
#pragma unroll
      for (int k = 0; k < kKinds; ++k) {
        const uint32_t* wds = omask + k * p.omask_total + p.omask_off[l];
        if (((wds[w_hi] >> (tid & 31)) & 1u) != 0u) {
          int rank = __popc(wds[w_hi] & below);
#pragma unroll 1
          for (int w = 0; w < w_hi; ++w) rank += __popc(wds[w]);
          olist[k * p.olist_total + p.olist_off[l] + rank] = tid;
        }
      }
    }
    if (l < D - 1 && tid < p.level_width[l] && mask_bit(p, mask_w, l, tid)) {
      const uint32_t* wds = mask_w + p.mask_off[l];
      int rank = __popc(wds[w_hi] & below);
#pragma unroll 1
      for (int w = 0; w < w_hi; ++w) rank += __popc(wds[w]);
      dlist[p.dlist_off[l] + rank] = tid;
    }
  }
}

//: A LEVEL'S RANK FIELD, PACKED INTO ONE REGISTER, and the packing is a -- see docs/internals/decode/decode_lattice.md#kfieldwidthmax
constexpr int kFieldWidthMax = 10;
static_assert(kDecodeMaxLevelWidth <= (1 << kFieldWidthMax),
              "a level's realized owner or digit count must fit the packed field's width");
static_assert(5 + 5 + kFieldWidthMax <= 32,
              "the packed field's three sub-fields must fit one 32-bit word");

__device__ __forceinline__ int field_pack(int shift, int width, int count) {
  return shift | (width << 5) | (count << 10);
}

__device__ __forceinline__ int field_shift(int f) { return f & 31; }

__device__ __forceinline__ int field_width(int f) { return (f >> 5) & 31; }

__device__ __forceinline__ int field_count(int f) { return f >> 10; }

//: THE LEVEL'S INDEX OUT OF A TERM'S RANK: the padded field, extracted by shifts alone.
__device__ __forceinline__ int field_index(int f, int rank) {
  return (rank >> field_shift(f)) & ((1 << field_width(f)) - 1);
}

//: THE TERM LAYOUT: each term's own rank space, DERIVED INTO REGISTERS BY -- see docs/internals/decode/decode_lattice.md#term-layout
template <int D>
__device__ __forceinline__ int term_layout(const DecodeParams& p, const int32_t* n_o,
                                           const int32_t* n_d, int term, int32_t* fld) {
  int acc = 0;
  bool empty = false;
#pragma unroll
  for (int l = D - 1; l >= 0; --l) {
    const int count = (term >= 1 && term <= D)   ? n_o[term_kind(term, l, D) * kMaxLevels + l]
                      : (term == 0 && l < D - 1) ? n_d[l]
                                                 : n_o[kKindW * kMaxLevels + l];
    const int width = ceil_log2_dev(count);
    fld[l] = field_pack(acc, width, count);
    acc += width;
    empty = empty || count == 0;
  }
  if (empty) return 0;
  return (term == 0) ? (1 << acc) : (1 << (acc + p.lat_outer_bits));
}

//: THE WHOLE TABLE AND THE RETIREMENT BOUND, published once per CTA. -- see docs/internals/decode/decode_lattice.md#publish-term-layout
template <int D>
__device__ __forceinline__ void publish_term_layout(const DecodeParams& p, const int32_t* n_o,
                                                    const int32_t* n_d, int32_t* fld,
                                                    int32_t* units, int32_t* umax) {
  const int t = threadIdx.x;
  if (t <= D + 1) {
    const int u = term_layout<D>(p, n_o, n_d, t, fld + t * kMaxLevels);
    units[t] = u;
    if (t <= D) atomicMax(umax, u);
  }
  __syncthreads();
}

//: ONE UNIT of the walk: the `(owner, r_0 .. r_{D-2})` pair a rank names -- an -- see docs/internals/decode/decode_lattice.md#unit
template <int D>
struct Unit {
  //: NO `owner` FIELD. The owner's mixed-radix index is a resolver LOCAL that dies into -- see docs/internals/decode/decode_lattice.md#base
  int base;  // the unit's first leaf, as a LATTICE index
  bool valid;
  bool live_r;  // every level but the innermost carries read amplitude at its fixed digit
  bool live_w;
  int dlast;  // level D-1's staging offset: its run base
  float ar;   // the fixed levels' amplitude product, per side, and the decay rate's
  float aw;
  float rate;
  uint32_t cr;  // level D-1's whole run, as one slice, per side
  uint32_t cw;
};

//: THE UNIT INDEX, decomposed by shifts alone within its TERM. -- see docs/internals/decode/decode_lattice.md#resolve-unit
template <int D, bool DECAY>
__device__ __forceinline__ Unit<D> resolve_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                const uint32_t* mask_w, const float* sa_r,
                                                const float* sa_w, const float* sd,
                                                const int32_t* olist, const int32_t* fld, int term,
                                                int local) {
  Unit<D> u;
  int owner = 0;
  u.valid = true;
  u.live_r = true;
  u.live_w = true;
  u.ar = 1.0f;
  u.aw = 1.0f;
  u.rate = 1.0f;
  bool live_kind = true;
  uint32_t c_kind = 0u;
  const int outer = local & ((1 << p.lat_outer_bits) - 1);
  const int rank = local >> p.lat_outer_bits;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int kind = term_kind(term, l, D);
    int i = field_index(fld[l], rank);
    if (i >= field_count(fld[l])) {
      u.valid = false;
      i = 0;
    }
    const int o = olist[kind * p.olist_total + p.olist_off[l] + i];
    owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    if (l == D - 1) {
      u.dlast = p.level_amp_offset[l] + run_base;
      u.cr = level_slice(p, mask_r, l, run_base, p.lat_s[l]);
      u.cw = level_slice(p, mask_w, l, run_base, p.lat_s[l]);
      c_kind = kind_bits(u.cr, u.cw, kind);
    } else {
      const int digit = run_base + ((outer >> p.lat_oshift[l]) & (p.lat_s[l] - 1));
      const int off = p.level_amp_offset[l] + digit;
      const uint32_t rb = mask_bit(p, mask_r, l, digit) ? 1u : 0u;
      const uint32_t wb = mask_bit(p, mask_w, l, digit) ? 1u : 0u;
      u.live_r = u.live_r && rb != 0u;
      u.live_w = u.live_w && wb != 0u;
      live_kind = live_kind && kind_bits(rb, wb, kind) != 0u;
      u.ar *= sa_r[off];
      u.aw *= sa_w[off];
      if (DECAY) u.rate *= sd[off];
    }
  }
  //: THE CONDITIONING the walk's one expression rests on, and the whole of the term's -- see docs/internals/decode/decode_lattice.md#alive
  const bool alive = u.valid && live_kind;
  if (term == 0 || term > D) {
    if (!alive) {
      u.cr = 0u;
      u.cw = 0u;
    }
  } else {
    u.cr = alive ? c_kind : 0u;
    u.cw = 0u;
    u.live_w = false;
    u.live_r = true;
  }
  u.base = (owner << p.lat_local_bits) + (outer << p.lat_inner_bits);
  return u;
}

//: THE CANDIDATE ATOM'S UNIT, and it is deliberately NOT a `Unit<D>`. -- see docs/internals/decode/decode_lattice.md#candidateunit
struct CandidateUnit {
  int base;
  bool valid;
  bool live_w;
  uint32_t cw;
};

template <int D>
__device__ __forceinline__ CandidateUnit resolve_candidate_unit(const DecodeParams& p,
                                                                const uint32_t* mask_w,
                                                                const int32_t* olist,
                                                                const int32_t* fld, int unit) {
  CandidateUnit u;
  int owner = 0;
  u.valid = true;
  u.live_w = true;
  const int outer = unit & ((1 << p.lat_outer_bits) - 1);
  const int rank = unit >> p.lat_outer_bits;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    int i = field_index(fld[l], rank);
    if (i >= field_count(fld[l])) {
      u.valid = false;
      i = 0;
    }
    const int o = olist[kKindW * p.olist_total + p.olist_off[l] + i];
    owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    if (l == D - 1) {
      u.cw = level_slice(p, mask_w, l, run_base, p.lat_s[l]);
    } else {
      const int digit = run_base + ((outer >> p.lat_oshift[l]) & (p.lat_s[l] - 1));
      u.live_w = u.live_w && mask_bit(p, mask_w, l, digit);
    }
  }
  u.base = (owner << p.lat_local_bits) + (outer << p.lat_inner_bits);
  return u;
}

//: THE WRITE PRODUCT'S UNIT, resolved from LIVE DIGITS. -- see docs/internals/decode/decode_lattice.md#resolve-write-unit
template <int D, bool DECAY>
__device__ __forceinline__ Unit<D> resolve_write_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                      const uint32_t* mask_w, const float* sa_r,
                                                      const float* sa_w, const float* sd,
                                                      const int32_t* olist, const int32_t* dlist,
                                                      const int32_t* fld, int unit) {
  //: THE WARP-LOCALITY ROTATION, and it is the second DRAM decision in this -- see docs/internals/decode/decode_lattice.md#krotlevel
  constexpr int kRotLevel = (D > 1) ? D - 2 : 0;
  int rank = unit;
  if (D > 1) {
    const int w = field_width(fld[kRotLevel]);
    const int lo_bits = w < kUnitBlockWarpsBits ? w : kUnitBlockWarpsBits;
    const int m = field_shift(fld[kRotLevel]);
    const int lo = unit & ((1 << lo_bits) - 1);
    const int mid = (unit >> lo_bits) & ((1 << m) - 1);
    const int rest = unit >> (lo_bits + m);
    rank = mid | (lo << m) | (rest << (m + lo_bits));
  }

  Unit<D> u;
  int owner = 0;
  u.valid = true;
  u.live_r = true;
  u.live_w = true;
  u.ar = 1.0f;
  u.aw = 1.0f;
  u.rate = 1.0f;
  int runs = 0;
  //: §6: `D` is this function's template parameter, so `D - 1` is compile-time; -- see docs/internals/decode/decode_lattice.md#kdm1
  constexpr int kDm1 = D - 1;
#pragma unroll
  for (int l = 0; l < kDm1; ++l) {
    int i = field_index(fld[l], rank);
    if (i >= field_count(fld[l])) {
      u.valid = false;
      i = 0;
    }
    const int digit = dlist[p.dlist_off[l] + i];
    owner += (digit >> p.lat_sbits[l]) * p.lat_gsuf[l];
    runs += (digit & (p.lat_s[l] - 1)) << p.lat_run_shift[l];
    const int off = p.level_amp_offset[l] + digit;
    u.live_r = u.live_r && mask_bit(p, mask_r, l, digit);
    u.ar *= sa_r[off];
    u.aw *= sa_w[off];
    if (DECAY) u.rate *= sd[off];
  }
  {
    constexpr int l = D - 1;
    int i = field_index(fld[l], rank);
    if (i >= field_count(fld[l])) {
      u.valid = false;
      i = 0;
    }
    const int o = olist[kKindW * p.olist_total + p.olist_off[l] + i];
    owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    u.dlast = p.level_amp_offset[l] + run_base;
    u.cr = level_slice(p, mask_r, l, run_base, p.lat_s[l]);
    u.cw = level_slice(p, mask_w, l, run_base, p.lat_s[l]);
  }
  //: THE SAME CONDITIONING `resolve_unit` applies to the write term, for the same reason: -- see docs/internals/decode/decode_lattice.md#near-line-421
  if (!u.valid) {
    u.cr = 0u;
    u.cw = 0u;
  }
  u.base = (owner << p.lat_local_bits) + runs;
  return u;
}

}  // namespace decode
}  // namespace rola
