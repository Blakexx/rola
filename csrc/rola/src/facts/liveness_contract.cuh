// csrc/rola/src/facts/liveness_contract.cuh -- THE CLASS-1 LIVENESS WORDS: the layout
// the one geometry-independent pass writes and every fold reads. No kernel and no
// launch signature lives here; the pass and the host folds address these bytes
// THROUGH this type, and rola/engine/facts/liveness.py is its Python mirror.
// -- see docs/internals/facts/liveness_contract.md
#pragma once

#include <cstdint>

namespace rola::facts {

constexpr int kMaxLevels = 4;

//: TOKENS ARE THE MINOR AXIS, thirty-two to a word -- one warp ballot per row.
constexpr int kTokBits = 32;

enum Side : int { kRead = 0, kWrite = 1, kSides = 2 };

constexpr uint16_t kBf16MagnitudeMask = 0x7fffu;

constexpr bool amplitude_live(uint16_t bits) { return (bits & kBf16MagnitudeMask) != 0u; }

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
};

//: THE PASS'S PER-SIDE STATIC OPERANDS, which carry no logical width (KERNEL_STANDARDS.md §R13's lint).
//: -- see docs/internals/facts/liveness_contract.md#statics
struct SideStatics {
  uint32_t dense_levels;
  const uint32_t* digit_mask;
};

constexpr bool level_is_dense(uint32_t dense_levels, int level) {
  return (dense_levels >> level) & 1u;
}

constexpr bool digit_in_mask(const uint32_t* digit_mask, int row) {
  return (digit_mask[row / 32] >> (row % 32)) & 1u;
}

}  // namespace rola::facts
