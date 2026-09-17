// csrc/rola/src/carry/box.cuh -- the CTA box's compile-time plan: the carve the arm fixes,
// the pool and the snapshot's geometry, and the shared-memory ledger's offsets.
// See docs/internals/carry/box.md and docs/internals/carry/smem_ledger.md
#pragma once
#include "carry/layout.cuh"
#include "carry/params.cuh"
#include "common/arch_caps.cuh"
#include "common/constants.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/state_page.cuh"

namespace rola::carry {

constexpr int round_up16(int bytes) { return (bytes + 15) / 16 * 16; }

constexpr int round_up128(int bytes) { return (bytes + 127) / 128 * 128; }

//: THE BOX, THE STATE SLICE AND THE LEDGER for one arm. -- see docs/internals/carry/smem_ledger.md
template <int D_, int DV_, int WARPS_>
struct BoxPlan {
  static constexpr int D = D_;
  static constexpr int kDv = DV_;
  static constexpr int kWarps = WARPS_;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kClusterCtas = rola::carry::kClusterCtas;

  //: THE BOX, BC leaves; a warp OWNS `kDealt` boxes of it, every channel. -- carry_kernel.md#state
  static constexpr int kLeavesPerWarp = leaves_per_warp(kDv);
  static constexpr int kBC = kLeavesPerWarp * kWarps;
  static constexpr int kAtoms = kBC / kAtomLeaves;
  static constexpr int kBoxes = kBC / denseref::ops::kMmaK;
  static constexpr int kTile = denseref::ops::kMmaM;
  static constexpr int kTiles = kWindow / kTile;
  static constexpr int kRounds = kWindow / 32;
  static constexpr int kDealt = kBoxes / kWarps;
  static constexpr int kNT = kDv / denseref::ops::kMmaN;
  static constexpr int kVRowBytes = kDv * 2;
  static constexpr int kVChunks = kVRowBytes / 16;

  //: THE LIVENESS WORDS staged a window ahead, both sides.
  static constexpr int kBoxRows = span_total(D, kBC);
  static constexpr int kWordsBytes = round_up16(kBoxRows * kRounds * 4);
  static_assert(kBoxRows % 4 == 0, "a round\'s digit words are whole 16-byte loads");

  //: THE HEAD'S PRODUCTS, two by window parity. -- carry_kernel.md#order
  static constexpr int kBuckets = kBoxes + 1;
  static constexpr int kOrderBytes = kWindow * 2;
  static constexpr int kTileMaskBytes = kTiles * 2;
  static constexpr int kRCountBytes = round_up16(kBuckets * kRounds * 2);
  static constexpr int kMaskBytes = kWindow * 2;
  static constexpr int kWarpWordsBytes = kWarps * kRounds * 4;
  static constexpr int kUnionBytes = kRounds * 4;
  static constexpr int kPrefixBytes = round_up16((kRounds + 1) * 2);
  static constexpr int kLiveBytes = 16;

  //: THE POOL: union-live tokens by rank, a token's V row, runs and gain pair; `kPoolSlots`
  //: slots under full/empty barriers; a ZERO ROW a slot. -- smem_ledger.md#pool
  static constexpr int kRunBytes = 32;
  static constexpr int kPoolSlots = 2;
  static constexpr int kPoolRowBytes = kVRowBytes + 2 * kRunBytes + 4;
  //: the slot count: the largest multiple of 16 tokens the ledger's remainder holds.
  static constexpr int kPoolTokMax = 128;

  //: THE SNAPSHOT: the state's high halves at the window start, the readout's B.
  //: -- carry_kernel.md#snapshot
  static constexpr int kSnapshotBytes = kBC * kVRowBytes;

  //: THE READOUT'S PRIVATE RING and DRAIN STAGE, a warp's. -- carry_kernel.md#readout
  static constexpr int kRunTileBytes = kTile * kRunBytes;
  static constexpr int kRRSlots = 2;
  static constexpr int kRRSlotBytes = 2 * kRunTileBytes;
  //: rows padded a chunk apart: bank free at immediate offsets. -- smem_ledger.md
  static constexpr int kDrainRows = 4;
  static constexpr int kDrainRowBytes = kDv * 4 + 32;
  static constexpr int kDrainBytes = kDrainRows * kDrainRowBytes;
  static constexpr int kWarpReadBytes = kRRSlots * kRRSlotBytes + kDrainBytes;

  //: THE MASS ROW, fp32 in read order; THE ROW MAP, a write leaf's read row.
  static constexpr int kMassRowBytes = kBC * 4;
  static constexpr int kRowMapBytes = kBC * 2;
  //: THE PHASE LEDGER's accumulators; the readout's tile counter; THE WALK RING, a warp's.
  static constexpr int kLedgerBytes = kWarps * kPhases * 8;
  static constexpr int kWalkEntries = 64;
  static constexpr int kWalkBytes = kWarps * kWalkEntries * 4;

  //: WHAT THE ARM'S ORDERS CAN DO (compile-time variants of the streams). -- carry_kernel.md#layout
  static constexpr bool kMayStraddle = [] {
    for (int l = 0; l < D; ++l)
      if (owner_span_bits(D, kBC, l) < kAtomLeavesBits) return true;
    return false;
  }();
  static constexpr bool kMayCompose = D > 2 || kMayStraddle;

  //: THE LEDGER. -- see docs/internals/carry/smem_ledger.md
  static constexpr int kWordsOffset = 0;
  static constexpr int kOrderOffset = kWordsOffset + 2 * kWordsBytes;
  static constexpr int kTileMaskOffset = kOrderOffset + 2 * kOrderBytes;
  static constexpr int kRCountOffset = kTileMaskOffset + 2 * kTileMaskBytes;
  static constexpr int kMaskOffset = kRCountOffset + kRCountBytes;
  static constexpr int kWarpWordsOffset = kMaskOffset + kMaskBytes;
  static constexpr int kUnionOffset = kWarpWordsOffset + 2 * kWarpWordsBytes;
  static constexpr int kPrefixOffset = kUnionOffset + 2 * kUnionBytes;
  static constexpr int kLiveOffset = kPrefixOffset + 2 * kPrefixBytes;
  static constexpr int kSnapshotOffset = round_up128(kLiveOffset + kLiveBytes);
  static constexpr int kMassRowOffset = kSnapshotOffset + kSnapshotBytes;
  static constexpr int kRowMapOffset = kMassRowOffset + kMassRowBytes;
  static constexpr int kSlotOffset = kRowMapOffset + kRowMapBytes;
  static constexpr int kLedgerOffset = kSlotOffset + round_up16(kAtoms * 4);
  static constexpr int kCounterOffset = kLedgerOffset + kLedgerBytes;
  //: a warp's TAKE WORD: the tile its lane 0 took, read back by every lane (a shared
  //: load is warp-uniform to the compiler where a shuffle's result is not).
  static constexpr int kTakeOffset = kCounterOffset + 16;
  static constexpr int kPoolBarOffset = round_up16(kTakeOffset + kWarps * 4);
  static constexpr int kRingBarOffset = kPoolBarOffset + 2 * kPoolSlots * 8;
  static constexpr int kWalkOffset = round_up16(kRingBarOffset + kWarps * kRRSlots * 8);
  static constexpr int kRegionOffset = round_up128(kWalkOffset + kWalkBytes);
  //: the region: the readout's blocks (a warp's ring slots and drain stage), then the
  //: pool's slots -- both streams run at once, so neither aliases the other.
  static constexpr int kReadOffset = kRegionOffset;
  static constexpr int kReadBytes = kWarps * kWarpReadBytes;
  static constexpr int kPoolOffset = kReadOffset + kReadBytes;
  static constexpr int kRemainder = (int)denseref::arch::kCaps.smem_per_cta_max - kPoolOffset;
  //: a slot holds `kPoolTok` tokens and the ZERO ROW as row `kPoolTok`.
  static constexpr int kPoolTokFit = (kRemainder / kPoolRowBytes / kPoolSlots - 1) / 16 * 16;
  static constexpr int kPoolTok = kPoolTokFit < kPoolTokMax ? kPoolTokFit : kPoolTokMax;
  static constexpr int kPoolRows = kPoolTok + 1;
  static constexpr int kPoolSlotBytes = round_up16(kPoolRows * kPoolRowBytes);
  static constexpr int kPoolVOffset = 0;
  static constexpr int kPoolInnerOffset = kPoolRows * kVRowBytes;
  static constexpr int kPoolOuterOffset = kPoolInnerOffset + kPoolRows * kRunBytes;
  static constexpr int kPoolGainOffset = kPoolOuterOffset + kPoolRows * kRunBytes;
  static constexpr int kPoolBytes = kPoolSlots * kPoolSlotBytes;
  static constexpr int kRegionBytes = kReadBytes + kPoolBytes;
  //: the fill's share of a chunk, a warp.
  static constexpr int kFillPerWarp = kPoolTok / kWarps;
  static constexpr int kSmemBytes = kRegionOffset + kRegionBytes;

  using Page = rola::state_page::Blocks<DV_>;

  static_assert(kBoxes <= 16, "a side's box mask is a halfword");
  static_assert(kBuckets <= 32, "the count scan is one lane a bucket");
  static_assert(kDealt >= 1 && kDealt * kWarps == kBoxes, "the boxes deal evenly to the warps");
  static_assert(kDealt * kNT * 4 <= 128, "a warp's state is at most 128 registers a lane");
  static_assert(D >= 2 && D <= 4, "the digit walk is spelled for two to four levels");
  static_assert(kWindow % 32 == 0 && kTiles * kTile == kWindow && kTiles == 32 && kRounds == 16,
                "the window is sixteen rounds and thirty-two tiles, one a lane");
  static_assert(owner_span_bits(D, kBC, D - 1) >= kAtomLeavesBits,
                "the innermost level holds a page whole");
  static_assert(kVChunks >= 4 && kVChunks % 4 == 0, "a V row is a multiple of four 16-byte chunks");
  static_assert(kPoolTok >= 32 && kPoolTok % 16 == 0, "a pool slot holds whole tiles");
  static_assert(kFillPerWarp * kWarps == kPoolTok && kFillPerWarp <= 16,
                "the fill deals a chunk evenly, at most sixteen tokens a warp");
  static_assert(kRRSlots >= 2, "the readout's ring issues a tile ahead");
  static_assert(kRegionBytes >= kSnapshotBytes,
                "the sweeps stage the low halves across the pool region");
  static_assert(kPoolTok >= 16, "a pool slot holds at least one fragment of tokens");
  static_assert(denseref::design::smem_fits((size_t)kSmemBytes),
                "the carry box's shared-memory ledger exceeds this architecture's per-CTA "
                "maximum");
  static_assert(kBC % kAtomLeaves == 0, "a CTA box is a whole number of pages");
};

//: THE FRAME'S EDGES, named -- barrier ids. -- see docs/internals/carry/carry_kernel.md#edges
enum class Edge : int { kCounts = 1, kOrder = 2 };

}  // namespace rola::carry
