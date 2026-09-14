// csrc/rola/src/common/geom.cuh -- THE ADDRESSING BLOCK: one derivation, computed
// once per launch on the host, consumed by every kernel of the family.
//
// THE AXIS LAW: the REGISTER SHAPE -- leaves per warp, `DV`, and
// the warp sub-box's spans and slot layout -- is compile-time `f(D, DV)`; the widths
// `B_l`, the per-side orders and the carve OFFSETS they fix are runtime and CTA-uniform.
// Symbols, the two span laws, the refusal set and the transit's row map:
// see docs/internals/common/geom.md
#pragma once

#include <cstdint>

#include "common/arch_caps.cuh"
#include "common/static_for.cuh"

namespace rola::carry {

//: the deepest topology the block is built for: every per-level field's array bound.
constexpr int kGeomMaxLevels = 4;

//: THE PAGE GRANULE, here because the owner span law is CONSTRAINED by it.
static constexpr int kAtomLeaves = 16;

//: the page rectangle's bit width: the trailing canonical bits one page holds.
constexpr int kAtomLeavesBits = 4;
static_assert((1 << kAtomLeavesBits) == kAtomLeaves,
              "the page rectangle's bit width and its leaf count are one fact");

//: THE BIT COUNT OF A POWER OF TWO, and -1 otherwise -- a REFUSAL, never a rounding.
constexpr int geom_ilog2(int x) {
  int b = 0;
  if (x <= 0) return -1;
  while ((1 << b) < x) ++b;
  return (1 << b) == x ? b : -1;
}

//: THE ARCH'S PRIMARY CONSTANTS, IN DERIVATION ORDER. -- see docs/internals/common/geom.md#arch-constants

//: THE STATE'S SHARE of the SM's register file. -- see docs/internals/common/geom.md#state-fraction
constexpr int kStateFileNumerator = 1, kStateFileDenominator = 4;

//: fp32 state elements one SM's register file holds under that share.
constexpr int state_elems_per_sm(const denseref::arch::Caps& c) {
  return c.regs_per_sm * kStateFileNumerator / kStateFileDenominator;
}

constexpr int leaves_per_sm(int dv) { return state_elems_per_sm(denseref::arch::kCaps) / dv; }

constexpr int warps_per_sm = 8;

//: THE CLUSTER LEVEL: CTAs of one cluster partition the cluster's box, and every cluster
//: term folds away at one. -- see docs/internals/common/geom.md#cluster
constexpr int kClusterCtas = 1;

constexpr int leaves_per_warp(int dv) { return leaves_per_sm(dv) / warps_per_sm; }

//: A BALANCED BIT SPLIT over `D` levels, remainder INNERMOST-FIRST.
constexpr int balanced_bits(int bits, int D, int l) {
  return bits / D + (l >= D - bits % D ? 1 : 0);
}

//: THE OWNER BOX: `log2 BC` balanced, the ATOM held WHOLE innermost. -- see docs/internals/common/geom.md#span-rule
constexpr int owner_span_bits(int D, int bc, int l) {
  const int total = geom_ilog2(bc), floor_in = geom_ilog2(kAtomLeaves);
  int b[kGeomMaxLevels] = {0, 0, 0, 0}, sum = 0;
  for (int i = 0; i < D; ++i) {
    b[i] = balanced_bits(total, D, i);
    if (i == D - 1 && b[i] < floor_in) b[i] = floor_in;
    sum += b[i];
  }
  while (sum > total) {
    int pick = -1;
    for (int i = 0; i < D; ++i) {
      const int fl = (i == D - 1) ? floor_in : 0;
      if (b[i] > fl && (pick < 0 || b[i] > b[pick])) pick = i;
    }
    if (pick < 0) break;
    --b[pick];
    --sum;
  }
  return b[l];
}

constexpr int owner_span(int D, int bc, int l) { return 1 << owner_span_bits(D, bc, l); }

//: the owner-local canonical index's bit offset for level `l`.
constexpr int owner_run_shift(int D, int bc, int l) {
  int sh = 0;
  for (int i = l + 1; i < D; ++i) sh += owner_span_bits(D, bc, i);
  return sh;
}

//: THE CARVE IS TOP-DOWN (Blake, 2026-08-29): the owner box's spans are divided level
//: by level IN THE SIDE'S OWN ORDER, outermost first, until the box count reaches the
//: side's stream count; what remains is that side's WARP SUB-BOX.
//: -- see docs/internals/common/geom.md#warp-box
constexpr void carve_sub_bits(int D, int bc, int ns, const int* outer_order, int* bits) {
  for (int l = 0; l < D; ++l) bits[l] = owner_span_bits(D, bc, l);
  int want = geom_ilog2(ns);
  for (int p = 0; p < D && want > 0; ++p) {
    const int l = outer_order[p];
    //: THE INNERMOST DIGITS ARE THE VECTOR WIDTH: the order passes over that level rather
    //: than dividing it below the page rectangle.
    //: -- see docs/internals/common/geom.md#innermost-invariant
    const int floor_l = (l == D - 1) ? kAtomLeavesBits : 0;
    const int room = bits[l] - floor_l;
    const int take = room < want ? room : want;
    if (take > 0) {
      bits[l] -= take;
      want -= take;
    }
  }
}

//: the largest set the generator can return: `D * (D - 1)` shapes at depth four.
constexpr int kGeomMaxSubBoxes = 12;

//: THE COMPILE-TIME SET OF WARP SUB-BOX SHAPES, GENERATED from the owner spans and the
//: stream count by running the carve over EVERY level order and de-duplicating. A side
//: selects one member at launch from its own order; nothing here is hand-written.
//: -- see docs/internals/common/geom.md#warp-box-set
struct SubBoxSet {
  int count;
  int bits[kGeomMaxSubBoxes][kGeomMaxLevels];
};

constexpr SubBoxSet sub_boxes(int D, int box_leaves, int workers) {
  SubBoxSet set{};
  set.count = 0;
  int fact = 1;
  for (int i = 2; i <= D; ++i) fact *= i;
  for (int k = 0; k < fact; ++k) {
    int pool[kGeomMaxLevels] = {0, 1, 2, 3}, order[kGeomMaxLevels] = {0, 0, 0, 0};
    int rem = k, avail = D;
    for (int i = 0; i < D; ++i) {
      int f = 1;
      for (int j = 2; j <= avail - 1; ++j) f *= j;
      const int idx = rem / f;
      rem %= f;
      order[i] = pool[idx];
      for (int j = idx; j < avail - 1; ++j) pool[j] = pool[j + 1];
      --avail;
    }
    int b[kGeomMaxLevels] = {0, 0, 0, 0};
    carve_sub_bits(D, box_leaves, workers, order, b);
    bool seen = false;
    for (int a = 0; a < set.count; ++a) {
      bool eq = true;
      for (int l = 0; l < D; ++l)
        if (set.bits[a][l] != b[l]) eq = false;
      if (eq) seen = true;
    }
    if (!seen && set.count < kGeomMaxSubBoxes) {
      for (int l = 0; l < D; ++l) set.bits[set.count][l] = b[l];
      ++set.count;
    }
  }
  return set;
}

//: one member's slot layout: the sub-box's bits packed INNERMOST-FIRST, so the
//: canonical innermost level sits at offset zero and seeds the accumulator.
constexpr int sb_shift(const SubBoxSet& set, int D, int a, int l) {
  int sh = 0;
  for (int i = l + 1; i < D; ++i) sh += set.bits[a][i];
  return sh;
}

//: THE SET BY VALUE, for device code: a static constexpr member of a class is ODR-USED
//: when a device body names it, so every compile-time consumer below re-derives the set
//: from the same three numbers instead. -- see docs/internals/common/geom.md#warp-box-set
constexpr int sb_span_bits(int D, int bc, int ns, int a, int l) {
  const SubBoxSet set = sub_boxes(D, bc, ns);
  return set.bits[a][l];
}

constexpr int sb_shift_of(int D, int bc, int ns, int a, int l) {
  const SubBoxSet set = sub_boxes(D, bc, ns);
  int sh = 0;
  for (int i = l + 1; i < D; ++i) sh += set.bits[a][i];
  return sh;
}

//: A SLOT'S REGION ROW under one member: each level's slot digit placed at that level's
//: own bit offset in the owner-local canonical index. GF(2)-LINEAR in the slot's bits, so
//: a lane-dependent part and a compile-time part compose by OR -- which is what keeps
//: every transit displacement an immediate. -- see docs/internals/common/geom.md#row-map
constexpr int sb_scatter_of(int D, int bc, int ns, int a, int P) {
  const SubBoxSet set = sub_boxes(D, bc, ns);
  int row = 0;
  for (int l = 0; l < D; ++l)
    row += ((P >> sb_shift_of(D, bc, ns, a, l)) & ((1 << set.bits[a][l]) - 1))
           << owner_run_shift(D, bc, l);
  return row;
}

//: THE REGION'S SWIZZLE of a read-order index: every three-bit group folded into the low
//: three, GF(2)-linear, a bijection. -- see docs/internals/common/geom.md#row-swizzle
__host__ __device__ constexpr int sb_swizzle(int row) {
  return row ^ (((row >> 3) ^ (row >> 6) ^ (row >> 9)) & 7);
}

//: the member a carved shape IS, or `-1` -- the exhaustiveness question, asked of the
//: generated set itself and never of a hand-written list.
constexpr int sb_index(const SubBoxSet& set, int D, const int* bits) {
  for (int a = 0; a < set.count; ++a) {
    bool eq = true;
    for (int l = 0; l < D; ++l)
      if (set.bits[a][l] != bits[l]) eq = false;
    if (eq) return a;
  }
  return -1;
}

//: ONE SIDE'S CARVE, indexed BY LEVEL except `level_at`, which inverts the order.
struct SideGeom {
  int level_at[kGeomMaxLevels];    //: rank -> level (rank 0 is innermost)
  int rank[kGeomMaxLevels];        //: level -> rank
  int grid[kGeomMaxLevels];        //: level -> s(l) / sv(l), the stream grid
  int grid_div[kGeomMaxLevels];    //: level -> the stream index's weight at this level
  int row_prefix[kGeomMaxLevels];  //: level -> its first ballot row
  int carves;                      //: bit `l` set when this side carves level `l`
  int row_g[kGeomMaxLevels];       //: level -> ballot rows at that level, 0 when spanned whole
  int row_span[kGeomMaxLevels];    //: level -> the run one ballot row ORs, the rows pass's own
  int rows;                        //: this side's ballot rows in total
  int leaf;                        //: prod_l sv(l) = BC / streams
  int streams;                     //: prod_l grid(l)
  //: bit counts, never values: no consumer of this block emits an integer divide.
  int grid_div_bits[kGeomMaxLevels];
  //: THIS SIDE'S WARP SUB-BOX: which member of the generated set its order selects,
  //: and that member's spans and slot layout. -- see docs/internals/common/geom.md#warp-box-set
  int assign;
  int sub_bits[kGeomMaxLevels];   //: level -> log2 sv(l)
  int sub_shift[kGeomMaxLevels];  //: level -> the slot index's bit offset
  int grid_bits[kGeomMaxLevels];  //: level -> log2 grid(l) = log2 s(l) - log2 sv(l)
};

//: THE BLOCK ITSELF -- what the host hands the kernel, by value, once per launch.
struct CarryGeomRT {
  int D;
  int bc;
  int width[kGeomMaxLevels];      //: B_l
  int span[kGeomMaxLevels];       //: s(l), the compile-time owner law's own values
  int grid[kGeomMaxLevels];       //: g(l) = B_l / s(l)
  int col_base[kGeomMaxLevels];   //: the prefix sum of the widths
  int run_shift[kGeomMaxLevels];  //: sum_{l' > l} log2 s(l')
  int owner_div[kGeomMaxLevels];  //: prod_{l' > l} g(l')
  int weight[kGeomMaxLevels];     //: prod_{l' > l} B_l', the canonical mixed radix
  int wtot;                       //: sum_l B_l -- the packed amplitude row's stride
  int owners;                     //: prod_l g(l)
  int leaves;                     //: prod_l B_l
  int local_bits;                 //: log2 BC
  int min_width;                  //: min_l B_l -- a CENSUS FACT, never a refusal
  SideGeom r;
  SideGeom w;
  //: the former `kTied`, now a REPORTED FACT -- see docs/internals/common/geom.md#row-map
  int identity_map;
  int span_bits[kGeomMaxLevels];       //: log2 s(l)
  int grid_bits[kGeomMaxLevels];       //: log2 g(l)
  int owner_div_bits[kGeomMaxLevels];  //: log2 owner_div(l)
};

//: THE MODE WORD'S ENCODING, as the seam spells it: two bits per level.
constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;

constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

namespace geom_detail {

//: THE ORDER BY MODE, unchanged in law, on a runtime word.
inline int mode_group(uint32_t MO, int l, uint32_t side) {
  const bool rd = mode_declares(MO, l, kSideRead), wr = mode_declares(MO, l, kSideWrite);
  if (rd && wr) return 2;                     // both-sparse
  return mode_declares(MO, l, side) ? 1 : 0;  // sparse : dense
}

inline bool mode_before(uint32_t MO, uint32_t side, int a, int b) {
  const int ga = mode_group(MO, a, side), gb = mode_group(MO, b, side);
  return ga != gb ? ga < gb : a > b;
}

}  // namespace geom_detail

//: THE CARVE ORDER, a runtime input: each side's levels ranked, rank 0 INNERMOST, and the
//: structural default the level modes give it.
//: -- see docs/internals/common/geom.md#carve-order
struct CarveOrder {
  int level_at[2][kGeomMaxLevels];
};

inline void carve_order_of_modes(int D, uint32_t MO, CarveOrder& o) {
  const uint32_t sides[2] = {kSideRead, kSideWrite};
  for (int s = 0; s < 2; ++s) {
    for (int l = 0; l < kGeomMaxLevels; ++l) o.level_at[s][l] = l;
    for (int l = 0; l < D; ++l) {
      int r = 0;
      for (int i = 0; i < D; ++i)
        if (geom_detail::mode_before(MO, sides[s], i, l)) ++r;
      o.level_at[s][r] = l;
    }
  }
}

//: ONE SIDE'S CARVE; `NS == 1` is the S = 1 point of it. -- see docs/internals/common/geom.md#carve
inline const char* derive_side(int D, const int* width, const int* span, int bc,
                               const int* level_at, int NS, SideGeom& g) {
  if (NS < 1 || (NS & (NS - 1)) != 0) return "a side's stream count is a power of two";
  int named = 0;
  for (int r = 0; r < D; ++r) {
    const int l = level_at[r];
    if (l < 0 || l >= D || ((named >> l) & 1) != 0)
      return "a side's carve order must rank each of its levels exactly once";
    named |= 1 << l;
    g.rank[l] = r;
    g.level_at[r] = l;
  }

  //: THE SIDE'S OWN CARVE: its order, walked OUTERMOST FIRST.
  int outer[kGeomMaxLevels] = {0, 0, 0, 0}, sub_bits[kGeomMaxLevels] = {0, 0, 0, 0};
  for (int p = 0; p < D; ++p) outer[p] = g.level_at[D - 1 - p];
  carve_sub_bits(D, bc, NS, outer, sub_bits);
  int carved = 0;
  for (int l = 0; l < D; ++l) carved += owner_span_bits(D, bc, l) - sub_bits[l];
  if (carved != geom_ilog2(NS))
    return "this carve order cannot reach the side's stream count without dividing the "
           "innermost level below the page rectangle";
  const SubBoxSet set = sub_boxes(D, bc, NS);
  g.assign = sb_index(set, D, sub_bits);
  if (g.assign < 0) return "this side's carved sub-box is not a member of the generated set";
  int sub[kGeomMaxLevels] = {1, 1, 1, 1};
  for (int l = 0; l < D; ++l) {
    g.sub_bits[l] = sub_bits[l];
    g.sub_shift[l] = sb_shift(set, D, g.assign, l);
    sub[l] = 1 << sub_bits[l];
  }
  g.streams = 1;
  g.leaf = 1;
  g.carves = 0;
  for (int l = 0; l < D; ++l) {
    g.grid[l] = span[l] / sub[l];
    g.row_span[l] = g.grid[l] == 0 ? 0 : span[l] / g.grid[l];
    if (g.grid[l] < 1 || g.grid[l] * g.row_span[l] != span[l])
      return "the warp sub-box does not divide the owner's box at some level";
    g.grid_bits[l] = geom_ilog2(g.grid[l]);
    if (g.grid_bits[l] < 0) return "a side's stream grid must be a power of two";
    g.streams *= g.grid[l];
    g.leaf *= g.row_span[l];
    if (g.grid[l] > 1) g.carves |= 1 << l;
  }
  if (g.streams != NS) return "the stream grid does not partition the owner's box";
  //: THE STREAM INDEX IS READ IN THE SIDE'S OWN ORDER: rank 0 varies fastest.
  int gd = 1;
  for (int r = 0; r < D; ++r) {
    const int l = g.level_at[r];
    g.grid_div[l] = gd;
    g.grid_div_bits[l] = geom_ilog2(gd);
    if (g.grid_div_bits[l] < 0) return "a side's stream grid must be a power of two";
    gd *= g.grid[l];
  }
  //: THE BALLOT ROWS at this side's own grain: a level spanned whole votes none.
  int pfx = 0;
  for (int l = 0; l < D; ++l) {
    g.row_g[l] = g.row_span[l] < width[l] ? width[l] / g.row_span[l] : 0;
    g.row_prefix[l] = pfx;
    pfx += g.row_g[l];
  }
  g.rows = pfx;
  for (int l = D; l < kGeomMaxLevels; ++l) {
    g.rank[l] = l;
    g.level_at[l] = l;
    g.grid[l] = 1;
    g.grid_div[l] = 1;
    g.grid_div_bits[l] = 0;
    g.grid_bits[l] = 0;
    g.sub_bits[l] = 0;
    g.sub_shift[l] = 0;
    g.row_span[l] = width[0];
    g.row_g[l] = 0;
    g.row_prefix[l] = pfx;
  }
  return nullptr;
}

//: THE ONE DERIVATION; returns nullptr or the REFUSAL's own words. -- see docs/internals/common/geom.md#refusals
inline const char* derive_carry_geom(int D, const int* width, int bc, int dv,
                                     const CarveOrder& order, int NSR, int NSW, CarryGeomRT& g) {
  if (D < 1 || D > kGeomMaxLevels) return "the addressing block is built to depth four";
  if (geom_ilog2(bc) < 0) return "BC must be a power of two";
  if (geom_ilog2(dv) < 0 || dv % 16 != 0) return "the value width is a power of two, sixteen up";
  g.D = D;
  g.bc = bc;
  if (leaves_per_warp(dv) < 1) return "the value width admits no leaves per warp";

  for (int l = 0; l < D; ++l) {
    g.width[l] = width[l];
    g.span[l] = owner_span(D, bc, l);
    if (geom_ilog2(g.width[l]) < 0) return "every level width must be a power of two";
    //: THE ONE WIDTH LAW THIS SHAPE OWNS -- there is no narrower case to fall back to.
    if (g.width[l] < g.span[l]) return "a level is narrower than the owner box's span there";
  }
  long leaves = 1, owners = 1;
  int wtot = 0, local_bits = 0, min_width = width[0];
  for (int l = 0; l < D; ++l) {
    g.grid[l] = g.width[l] / g.span[l];
    g.col_base[l] = wtot;
    wtot += g.width[l];
    leaves *= g.width[l];
    owners *= g.grid[l];
    local_bits += owner_span_bits(D, bc, l);
    if (g.width[l] < min_width) min_width = g.width[l];
  }
  if (local_bits != geom_ilog2(bc)) return "the owner box's spans do not multiply to BC";
  if (leaves > (long)0x40000000) return "the topology's leaf count does not fit the id space";
  g.wtot = wtot;
  g.owners = (int)owners;
  g.leaves = (int)leaves;
  g.local_bits = local_bits;
  g.min_width = min_width;
  int wgt = 1, odiv = 1;
  for (int l = D - 1; l >= 0; --l) {
    g.weight[l] = wgt;
    g.owner_div[l] = odiv;
    g.owner_div_bits[l] = geom_ilog2(odiv);
    g.grid_bits[l] = geom_ilog2(g.grid[l]);
    g.span_bits[l] = owner_span_bits(D, bc, l);
    g.run_shift[l] = owner_run_shift(D, bc, l);
    if (g.owner_div_bits[l] < 0 || g.grid_bits[l] < 0)
      return "the owner grid must be a power of two";
    wgt *= g.width[l];
    odiv *= g.grid[l];
  }
  for (int l = D; l < kGeomMaxLevels; ++l) {
    g.width[l] = width[0];
    g.span[l] = width[0];
    g.grid[l] = 1;
    g.col_base[l] = wtot;
    g.weight[l] = 1;
    g.owner_div[l] = 1;
    g.owner_div_bits[l] = 0;
    g.grid_bits[l] = 0;
    g.span_bits[l] = geom_ilog2(width[0]);
    g.run_shift[l] = 0;
  }
  if (const char* e = derive_side(D, g.width, g.span, bc, order.level_at[0], NSR, g.r)) return e;
  if (const char* e = derive_side(D, g.width, g.span, bc, order.level_at[1], NSW, g.w)) return e;
  //: a level nobody carves fixes no digit, so its index weight is not part of the map.
  int identical = g.r.assign == g.w.assign ? 1 : 0;
  for (int l = 0; l < D; ++l)
    if (g.r.grid[l] > 1 && g.r.grid_div_bits[l] != g.w.grid_div_bits[l]) identical = 0;
  g.identity_map = identical;
  return nullptr;
}

//: THE TRANSIT'S ROW MAP: the region is indexed by the OWNER-LOCAL CANONICAL INDEX, so
//: each side reaches a leaf through its OWN base word and its own member's fixed shifts.
//: -- see docs/internals/common/geom.md#row-map
//: A SIDE'S SLOT AS THE OWNER-LOCAL CANONICAL INDEX -- one law, three readers.
//: -- see docs/internals/common/geom.md#local-row
template <int D, int BC>
__host__ __device__ inline int geom_local_row(const SideGeom& s, int strm, int P) {
  int row = 0;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int off = ((strm >> s.grid_div_bits[l]) & (s.grid[l] - 1)) << s.sub_bits[l];
    const int dig = (P >> s.sub_shift[l]) & ((1 << s.sub_bits[l]) - 1);
    row += (off + dig) << owner_run_shift(D, BC, l);
  });
  return row;
}

//: THE READ-ORDER INDEX of an owner-local canonical row: the region's row law, both sides.
//: -- see docs/internals/carry/carry_kernel.md#read-order
__host__ __device__ inline int geom_read_index(const CarryGeomRT& g, const SideGeom& r, int local) {
  int index = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l) {
    int sh = 0;
#pragma unroll 1
    for (int m = 0; m < g.D; ++m)
      if (r.rank[m] < r.rank[l]) sh += geom_ilog2(g.span[m]);
    index += ((local >> g.run_shift[l]) & (g.span[l] - 1)) << sh;
  }
  return index;
}

__host__ __device__ inline int geom_xch_row(const CarryGeomRT& g, const SideGeom& s, int strm,
                                            int P) {
  int row = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l) {
    const int off = ((strm >> s.grid_div_bits[l]) & (s.grid[l] - 1)) << s.sub_bits[l];
    const int dig = (P >> s.sub_shift[l]) & ((1 << s.sub_bits[l]) - 1);
    row += (off + dig) << g.run_shift[l];
  }
  return sb_swizzle(geom_read_index(g, g.r, row));
}

//: THE ABSOLUTE DIGIT of level `l` for owner `O`'s local slot.
//: -- see docs/internals/common/geom.md#digit
template <int D, int BC>
__host__ __device__ inline int geom_digit(const CarryGeomRT& g, int O, int local, int l) {
  return (((O >> g.owner_div_bits[l]) & (g.grid[l] - 1)) << owner_span_bits(D, BC, l))
         + ((local >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1));
}

//: THE CANONICAL LEAF of owner `O`'s local slot, and the atom that holds it.
//: THE SAME MAP WITH THE DEPTH AND THE OWNER LAW COMPILE-TIME (KERNEL_STANDARDS.md §R12: no runtime trip
//: count in a register-shaped loop). The kernel knows `D` and `BC` as template
//: arguments, so the spans and the run shifts are IMMEDIATES here and only the
//: width-derived terms stay runtime. -- see docs/internals/common/geom.md#canon-leaf
template <int D, int BC, int L>
__host__ __device__ inline int geom_canon_acc(const CarryGeomRT& g, int O, int local) {
  if constexpr (L >= D) {
    return 0;
  } else {
    const int d = (((O >> g.owner_div_bits[L]) & (g.grid[L] - 1)) << owner_span_bits(D, BC, L))
                  + ((local >> owner_run_shift(D, BC, L)) & (owner_span(D, BC, L) - 1));
    return d * g.weight[L] + geom_canon_acc<D, BC, L + 1>(g, O, local);
  }
}

template <int D, int BC>
__host__ __device__ inline int geom_canon_leaf_t(const CarryGeomRT& g, int O, int local) {
  return geom_canon_acc<D, BC, 0>(g, O, local);
}

__host__ __device__ inline int geom_canon_leaf(const CarryGeomRT& g, int O, int local) {
  int acc = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l) {
    const int d = (((O >> g.owner_div_bits[l]) & (g.grid[l] - 1)) << g.span_bits[l])
                  + ((local >> g.run_shift[l]) & (g.span[l] - 1));
    acc += d * g.weight[l];
  }
  return acc;
}

}  // namespace rola::carry
