#pragma once

// THE STATE PAGE'S SPLIT bf16 PLANES: the one gather/scatter every body's state
// I/O goes through. A page is [16 x DV hi][16 x DV lo][16 mass hi][16 mass lo] --
// the same fp32 bits an fp32-row page held, re-laid out so a hi row is one line
// and an atom a call only READS moves half of them.
// See docs/internals/common/state_page.md

#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"

namespace rola::state_page {

using denseref::ops::gather_ptr;

//: The page's leaf rectangle, and the block offsets it implies, in BYTES. -- see docs/internals/common/state_page.md#geometry
template <int DV>
struct Blocks {
  static constexpr int kRows = 16;
  static constexpr int kPairs = kRows * DV / 2;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kMassHi = (uint32_t)(2 * kRows * DV * 2);
  static constexpr uint32_t kMassLo = kMassHi + (uint32_t)(kRows * 2);
  static constexpr uint32_t kBytes = kMassLo + (uint32_t)(kRows * 2);
  static_assert(DV % 2 == 0, "a 32-bit access carries a PAIR of value columns");
  static_assert(kBytes == (uint32_t)(kRows * (DV + 1) * (int)sizeof(float)),
                "the split layout is a re-layout of the fp32 page, not a resize");
};

//: The page's SMEM image, `[16][LDO]` fp32 with the mass in column `DV`. -- see docs/internals/common/state_page.md#load
template <int DV, int LDO, int THREADS>
__device__ __forceinline__ void load(float* so, const void* plane, uint32_t pageOff, int tid,
                                     bool with_lo) {
  using G = Blocks<DV>;
#pragma unroll 1
  for (int p = tid; p < G::kPairs; p += THREADS) {
    const int row = p / (DV / 2), col = 2 * (p % (DV / 2));
    const uint32_t off = pageOff + (uint32_t)((row * DV + col) * 2);
    const uint32_t h = *reinterpret_cast<const uint32_t*>(gather_ptr(plane, off));
    uint32_t e0 = h << 16, e1 = h & 0xFFFF0000u;
    if (with_lo) {
      const uint32_t l = *reinterpret_cast<const uint32_t*>(gather_ptr(plane, off + G::kLo));
      e0 = __byte_perm(l, h, 0x5410);
      e1 = __byte_perm(l, h, 0x7632);
    }
    so[row * LDO + col] = __uint_as_float(e0);
    so[row * LDO + col + 1] = __uint_as_float(e1);
  }
#pragma unroll 1
  for (int r = tid; r < G::kRows; r += THREADS) {
    const uint32_t o = pageOff + (uint32_t)(r * 2);
    uint32_t w = (uint32_t)(*reinterpret_cast<const uint16_t*>(gather_ptr(plane, o + G::kMassHi)))
                 << 16;
    if (with_lo)
      w |= (uint32_t)(*reinterpret_cast<const uint16_t*>(gather_ptr(plane, o + G::kMassLo)));
    so[r * LDO + DV] = __uint_as_float(w);
  }
}

//: The mirror of `load`, and it always writes BOTH planes: a page is stored -- see docs/internals/common/state_page.md#store
template <int DV, int LDO, int THREADS>
__device__ __forceinline__ void store(const float* so, void* plane, uint32_t pageOff, int tid) {
  using G = Blocks<DV>;
#pragma unroll 1
  for (int p = tid; p < G::kPairs; p += THREADS) {
    const int row = p / (DV / 2), col = 2 * (p % (DV / 2));
    const uint32_t off = pageOff + (uint32_t)((row * DV + col) * 2);
    const uint32_t v0 = __float_as_uint(so[row * LDO + col]);
    const uint32_t v1 = __float_as_uint(so[row * LDO + col + 1]);
    *reinterpret_cast<uint32_t*>(gather_ptr(plane, off)) = __byte_perm(v0, v1, 0x7632);
    *reinterpret_cast<uint32_t*>(gather_ptr(plane, off + G::kLo)) = __byte_perm(v0, v1, 0x5410);
  }
#pragma unroll 1
  for (int r = tid; r < G::kRows; r += THREADS) {
    const uint32_t o = pageOff + (uint32_t)(r * 2);
    const uint32_t w = __float_as_uint(so[r * LDO + DV]);
    *reinterpret_cast<uint16_t*>(gather_ptr(plane, o + G::kMassHi)) = (uint16_t)(w >> 16);
    *reinterpret_cast<uint16_t*>(gather_ptr(plane, o + G::kMassLo)) = (uint16_t)(w & 0xFFFFu);
  }
}

}  // namespace rola::state_page
