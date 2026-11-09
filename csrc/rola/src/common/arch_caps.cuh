#pragma once

// THE CARD'S FACTS at the compiling target: the rows `common/hardware.cuh` generates from common/hardware.json, the
// compiling architecture and its row. see docs/internals/common/arch_caps.md

#include <cuda_runtime.h>

#include "common/hardware.cuh"

namespace denseref {
namespace arch {

// THE COMPILING TARGET; inert in the host pass, pinned to the smallest tabulated config.
#ifdef __CUDA_ARCH__
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 860;
#endif

static_assert(tabulated(kArch),
              "untabulated compute capability: add its row to caps_of() rather "
              "than letting an operation guess which implementation it has");

constexpr Caps kCaps = caps_of(kArch);

//: THE FAMILIES WITH ATOMS: a compiling target of a family none of whose atoms are written is refused by name
static_assert(
    traits_of(Build{kArch, 0}).family == Family::kAmpere,
    "no atom of this architecture's family is written (the hopper family's are next): build for an ampere "
    "architecture (--//:archs)");

constexpr int caps_digest(const Caps& c) {  // see docs/internals/common/arch_caps.md#capsdigest
  return c.cc * 1000 + (int)c.bf16_cvt_packed * 16 + (int)c.bf16_simd_arith * 32
         + (int)c.reg_transpose_b16 * 2048 + (int)c.bf16_simd_fma * 4096;
}

//: the compiling target's digest, the carry's device stamp reads it
constexpr int kCapsDigest = caps_digest(kCaps);

}  // namespace arch
}  // namespace denseref
