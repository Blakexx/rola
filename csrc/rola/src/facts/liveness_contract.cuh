// csrc/rola/src/facts/liveness_contract.cuh -- THE CLASS-1 LIVENESS WORDS: the layout the facts pass writes and
// every fold reads. No kernel and no launch signature lives here; the pass and the host folds address these bytes
// THROUGH this type, and rola/engine/facts/liveness.py is its Python mirror.
// -- see docs/internals/facts/liveness_contract.md
#pragma once

#include <cstdint>

namespace rola::facts {

constexpr int kMaxLevels = 4;

//: TOKENS ARE THE MINOR AXIS, thirty-two to a word; a TILE is two words, the grain of the tile unions.
constexpr int kTokBits = 32;
constexpr int kTileTokens = 64;
constexpr int kGroupDigits = 32;

enum Side : int { kRead = 0, kWrite = 1, kSides = 2 };

//: A BUNDLE DIGIT'S NONZERO TEST: a u8 digit.
constexpr bool digit_live(uint8_t digit) { return digit != 0u; }

//: THE DIGITS against the row max. -- see docs/internals/facts/liveness_contract.md#digits
constexpr float kRowMaxDigit = 255.0f;

constexpr int token_words(int L) { return (L + kTokBits - 1) / kTokBits; }

constexpr int mask_words(int bits) { return (bits + 31) / 32; }

//: THE CLASS-1 TABLE. -- see docs/internals/facts/liveness_contract.md#layout
struct LivenessLayout {
  int D;
  int B[kMaxLevels];
  int L;

  constexpr int row_base(int level) const {
    int base = 0;
    for (int l = 0; l < level; ++l) base += B[l];
    return base;
  }

  constexpr int rows() const { return row_base(D); }

  constexpr int row_of(int level, int digit) const { return row_base(level) + digit; }

  constexpr int words() const { return token_words(L); }

  constexpr long side_words() const { return (long)rows() * words(); }

  constexpr long side_bytes() const { return side_words() * 4; }

  constexpr long call_words() const { return (long)kSides * side_words(); }

  constexpr long word_index(int bh, int side, int row, int token) const {
    return (long)bh * call_words() + (long)side * side_words() + (long)row * words()
           + token / kTokBits;
  }

  constexpr uint32_t token_bit(int token) const { return 1u << (token % kTokBits); }

  //: THE TILE UNIONS: per (side, 32-digit group of a level, 64-token tile) the group's digits live anywhere in the
  //: tile, one word. -- see docs/internals/facts/liveness_contract.md#tile-unions
  constexpr int group_base(int level) const {
    int base = 0;
    for (int l = 0; l < level; ++l) base += (B[l] + kGroupDigits - 1) / kGroupDigits;
    return base;
  }

  constexpr int groups() const { return group_base(D); }

  constexpr int tiles() const { return (L + kTileTokens - 1) / kTileTokens; }

  constexpr long union_index(int bh, int side, int group, int tile) const;
};

//: A TILE UNION's address from the table's extents: the layout's own, or a kernel's compile-time ones.
constexpr long union_address(int bh, int side, int group, int tile, int groups, int tiles) {
  return ((long)bh * kSides + side) * groups * tiles + (long)group * tiles + tile;
}

//: A SIDE'S STATIC OPERANDS, which carry no logical width (CONSTITUTION.md §R13's lint): a dense-level bitmask and
//: the digit mask's words. -- see docs/internals/facts/liveness_contract.md#statics
constexpr bool level_is_dense(uint32_t dense_levels, int level) {
  return (dense_levels >> level) & 1u;
}

constexpr long LivenessLayout::union_index(int bh, int side, int group, int tile) const {
  return union_address(bh, side, group, tile, groups(), tiles());
}

//: `row`'s bit, from the mask word that holds it (`mask[row / 32]`).
constexpr bool mask_bit(uint32_t word, int row) { return (word >> (row % 32)) & 1u; }

}  // namespace rola::facts
