// csrc/rola/src/intra/intra.cuh -- THE INTRA's constants and operand block: the within-window pairs, each level's Gram
// as f16 MMAs over the bundle's rows as stored (u8 digits widened in registers, dense f16 rows as they are), the value
// product over the f16 values, reduced into the output planes.
// see docs/internals/intra/intra.md
#pragma once

#include <cstdint>

#include "common/window.cuh"

namespace rola::intra {

//: THE WALK: tiles, a warp's 16 readers, a piece's tiles. -- see docs/internals/intra/intra.md#constants
constexpr int kTile = 64;
constexpr int kWarpRows = 16;
constexpr int kPieceTiles = 4;
constexpr int kValueWidth = 64;
constexpr int kValueBytes = 2 * kValueWidth;

constexpr int kMaxLevels = 2;
constexpr int kMaxLevelWidth = 256;
constexpr int kDigitGranule = 16;

//: A COPY CHUNK, a STAGE (128 bytes of a tile's rows: 128 u8 or 64 f16 digits) and a SLAB (32 digits, the gate's grain).
//: -- see docs/internals/intra/intra.md#constants
constexpr int kChunk = 16;
constexpr int kStageBytes = 128;
constexpr int kSlabDigits = 32;
constexpr int kStageBuffers = 3;
constexpr int kLevelStages = 2 * kMaxLevelWidth / kStageBytes;
constexpr int kMaxStages = kMaxLevels * kLevelStages;
//: the deposit stages a warp's 16 rows of 64 sums at a 72-float pitch in the dead reader rows
constexpr int kStagingBytes = 16 * 72 * 4;

//: A LEVEL'S FORM: bit 0 its read rows f16, bit 1 its write rows f16 (u8 where clear).
enum LevelForm : int { kFormU8U8, kFormF16U8, kFormU8F16, kFormF16F16, kForms };

__host__ __device__ constexpr bool form_read_f16(int form) { return (form & 1) != 0; }

__host__ __device__ constexpr bool form_write_f16(int form) { return (form & 2) != 0; }

__host__ __device__ constexpr int form_stage_digits(int form) {
  return form_write_f16(form) ? 64 : 128;
}

enum IntraPhase : int {
  kPhaseReader,
  kPhaseGate,
  kPhaseWait,
  kPhaseGram,
  kPhaseEpilogue,
  kPhaseValue,
  kPhaseDeposit,
  kIntraPhases
};

//: the ledger's rows: the most warps a card's CTA has
constexpr int kLedgerWarps = 8;

struct Piece {
  int block, t0, t1;
};

//: A WINDOW'S PIECES at (W, block rows): each block's writer tiles in sweeps of at most `kPieceTiles`, longest first and
//: the heavier block first among equals. -- see docs/internals/intra/intra.md#pieces
template <int W, int Rows>
struct PiecesAt {
  static constexpr int kWindowTiles = W / kTile, kRowTiles = Rows / kTile;
  static constexpr int kBlocks = kWindowTiles / kRowTiles;

  static constexpr int count() {
    int n = 0;
    for (int b = 0; b < kBlocks; ++b) n += ((b + 1) * kRowTiles + kPieceTiles - 1) / kPieceTiles;
    return n;
  }

  Piece table[count()]{};

  constexpr PiecesAt() {
    int n = 0;
    for (int len = kPieceTiles; len > 0; --len)
      for (int b = kBlocks - 1; b >= 0; --b)
        for (int t0 = 0; t0 < (b + 1) * kRowTiles; t0 += kPieceTiles) {
          const int end = (b + 1) * kRowTiles;
          const int t1 = t0 + kPieceTiles < end ? t0 + kPieceTiles : end;
          if (t1 - t0 == len) table[n++] = Piece{b, t0, t1};
        }
  }
};

//: One level's rows or the value plane where the bundle holds them: the batch, token and head strides in bytes.
struct RowView {
  const void* base;
  int64_t sb, st, sh;
};

//: A STAGE: a level's 128-byte slice of the writer rows, its gate bits `mask << slab0` and its reader slot's first
//: chunk. -- see docs/internals/intra/intra.md#stages
struct Stage {
  int index, level, digit0, slab0;
  uint32_t mask;
  int reader_chunk;
};

//: THE OPERAND BLOCK, a grid constant: the views, the call's widths and forms and the stage table they give.
//: -- see docs/internals/intra/intra.md#intraparams
struct IntraParams {
  RowView read[kMaxLevels];
  RowView write[kMaxLevels];
  RowView values;
  const float2* fm;
  const uint32_t* unions;
  float* num;
  float* den;
  long long* ledger;
  Stage stage[kMaxStages];
  int width[kMaxLevels], form[kMaxLevels], slab0[kMaxLevels], group0[kMaxLevels];
  int level_stage0[kMaxLevels], level_stages[kMaxLevels];
  uint32_t level_slabs[kMaxLevels];
  int stages, slabs, groups, pitch;
  int H, group, L, windows;
  float lift, unit;
  uint32_t dense[2];
};

}  // namespace rola::intra
