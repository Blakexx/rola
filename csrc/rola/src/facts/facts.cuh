// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`), the window's columns, and each side's lists in the carry's final form, a CTA a unit.
// see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "carry/carry.cuh"
#include "common/hardware.cuh"
#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE CTA: eight warps a UNIT. -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 8;
constexpr int kFactsThreads = kFactsWarps * 32;

//: THE ROW ALIGNMENT LAW's grain (`rola.routing.quant.ROW_ALIGN`). -- see docs/internals/facts/liveness_contract.md#alignment
constexpr int kRowAlign = 32;

constexpr int kMaxWidth = 256;
constexpr int kMaxGroups = kMaxWidth / kGroupDigits;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;
//: THE QUERY HEADS a write head's read side merges: the parameter block's bound, and a window's at W (every head's rows
//: at a 16-bit pool offset). -- see docs/internals/facts/facts.md#window
constexpr int kMaxGroup = 16;

__host__ __device__ constexpr int max_group(int w) { return 8192 / w; }

//: A UNIT, coded (kind, side, level, group, head in the group). -- see docs/internals/facts/facts.md#the-plan
enum UnitKind : int { kUnitVote = 0, kUnitList = 1, kUnitColumns = 2 };

constexpr int kMaxUnits = kMaxLevels * kMaxGroups * (kMaxGroup + 1) + 1;

__host__ __device__ constexpr uint32_t unit_code(int kind, int side, int level, int group,
                                                 int head) {
  return static_cast<uint32_t>(kind | side << 2 | level << 3 | group << 5 | head << 8);
}

__host__ __device__ constexpr int unit_kind(uint32_t c) { return c & 3; }

__host__ __device__ constexpr int unit_side(uint32_t c) { return (c >> 2) & 1; }

__host__ __device__ constexpr int unit_level(uint32_t c) { return (c >> 3) & 3; }

__host__ __device__ constexpr int unit_group(uint32_t c) { return (c >> 5) & 7; }

__host__ __device__ constexpr int unit_head(uint32_t c) { return (c >> 8) & 15; }

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32. -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

//: THE LISTS OUT, the carry's final form (`carry/carry.cuh` owns its addresses), a write entry's coefficients at
//: 2^(kCoefTop - E) with E floored at `kUnitMin`. -- see docs/internals/facts/facts.md#lists-out
constexpr int kListBound = carry::kListBound;
constexpr int kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127;
static_assert(kFactsWarps == carry::kWarps && kGroupDigits == carry::kMaxExtent,
              "the lists are the carry's: its box lists, a boundary every 64 slots");

//: THE STEP: a union's extent is its size in whole steps of the carry's 8 entries.
constexpr int kStep = 8;

//: A SIDE'S LIST GROUP: a union's digits, a unit's group or each half. -- see docs/internals/facts/facts.md#list-groups
constexpr int kListGroupMin = 16;
constexpr int kUnitGroups = kGroupDigits / kListGroupMin;
static_assert(kUnitGroups == 2, "a list unit's groups are its liveness group or each half");

enum FactsPhase : int { kPhaseLoad, kPhaseVote, kPhaseRanks, kPhaseLists, kFactsPhases };

//: THE CTA'S SHARED MEMORY, by offset, a pass of `G` list groups a unit: the window, the bit tile, the columns, each
//: group's union words and bounds, the warps' maxima, a head's boundaries. -- see docs/internals/facts/facts.md#shared-memory
template <int W, int G>
struct FactsSmemAt {
  static constexpr int kWin = 0;
  static constexpr int kTile = kWin + W * kGroupDigits;
  static constexpr int kCols = kTile + kGroupDigits * (W / kTokBits) * 4;
  static constexpr int kUword = kCols + W * 8;
  static constexpr int kBound = kUword + G * (W / kTokBits) * 4;
  static constexpr int kWmax = kBound + G * (W / kListBound) * 4;
  static constexpr int kTable = kWmax + kFactsWarps * 8;
  static constexpr int kBytes = kTable + kGroupDigits * (W / kListBound + 1) * 2;
};

template <int W, int G>
constexpr bool tenants_aligned() {
  using S = FactsSmemAt<W, G>;
  return S::kTile % 16 == 0 && S::kCols % 16 == 0 && S::kUword % 16 == 0 && S::kWmax % 8 == 0
         && S::kTable % 16 == 0;
}

//: THE PASS'S PLAN at build `B`: its window and the counts the stages map onto it (a warp's 32-token blocks, a
//: thread's tokens, a list lane's words and boundaries), its residency and its tenants.
//: -- see docs/internals/facts/facts.md#window
template <::denseref::arch::Build B>
struct FactsPlan {
  static constexpr int kWindowTokens = ::denseref::arch::traits_of(B).window;
  //: four CTAs an SM, two at a window of 1,024 (a list lane's twice the words and boundaries in registers)
  static constexpr int kFactsCtasMin = kWindowTokens >= 1024 ? 2 : 4;
  static constexpr int kWindowWords = kWindowTokens / kTokBits;
  static constexpr int kWindowTiles = kWindowTokens / kTileTokens;
  static constexpr int kWarpBlocks = kWindowWords / kFactsWarps;
  static constexpr int kThreadTokens = kWindowTokens / kFactsThreads;
  static constexpr int kLaneWords = kWindowWords / 8;
  static constexpr int kListCap = kWindowTokens;
  static constexpr int kListAt = kWindowTokens / kListBound;
  static constexpr int kLaneBounds = kListAt / 8;
  template <int G>
  using FactsSmem = FactsSmemAt<kWindowTokens, G>;

  //: A BIT TILE: the group's 32 digits x the window's 16 words, row `d` word `j` at `16 d + (j ^ (d / 2 % 16))`.
  //: -- see docs/internals/facts/facts.md#tiles
  static __host__ __device__ constexpr int tile_at(int d, int j) {
    return kWindowWords * d + (j ^ (kWindowWords >= 32 ? (d & 31) : ((d >> 1) & 15)));
  }

  static_assert(
      max_group(kWindowTokens) * kWindowTokens < 65536,
      "a list group's slots a window, every head's, fit the 16-bit half its count rides in");
  static_assert(tenants_aligned<kWindowTokens, 1>()
                    && tenants_aligned<kWindowTokens, kUnitGroups>(),
                "each tenant on its access's alignment");
};

//: A SIDE'S LISTS OUT at its list level: entries, prefix rows (mod 2^16), exact totals, the union by slot, its extent.
//: -- see docs/internals/facts/facts_api.md#rola-facts-forward
struct ListsOut {
  uint2* entries;
  uint16_t* pre;
  int32_t* totals;
  uint16_t* union_tok;
  int32_t* union_count;
};

//: A level's rows by batch, token, head strides in BYTES and width in digits; an fp32 column by element strides.
struct RowView {
  const uint8_t* base;
  int64_t sb, st, sh;
  int width;
};

struct ColumnView {
  const float* base;
  int64_t sb, st, sh;
};

//: THE PASS'S OPERAND BLOCK, by value. -- see docs/internals/facts/facts.md#factsparams
struct FactsParams {
  RowView rows[kSides][kMaxLevels];
  ColumnView scale_write[kMaxLevels];
  ColumnView v_scale;
  uint32_t* words;
  uint32_t* unions;
  uint32_t* list_exp;  // [BH, windows, B_list], a digit's (K_m, K_v) int16 pair
  float2* fm;  // [BH, L], a token's (F, m): the fold's factor products (facts.md#factor-products)
  uint32_t*
      gains16;  // [BH, L], a token's (m, F) as an f16 pair at its window's scales (facts.md#gains)
  int16_t* gains_exp;  // [BH, windows, 2], those scales' exponents
  ListsOut lists[kSides];
  long long* ledger;
  int D, H, g, L, token_words, windows, rows_total, groups, tiles, per_window, read_row;
  int B[kMaxLevels];
  int row_base[kMaxLevels];
  int group_base[kMaxLevels];
  int list_level[kSides];
  int list_group[kSides];
  uint32_t unit[kMaxUnits];
  uint32_t mask[kSides][kMaxMaskWords];
};

}  // namespace rola::facts
