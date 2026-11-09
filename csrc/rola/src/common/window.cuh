// csrc/rola/src/common/window.cuh -- a union slot's token bits and query heads at W. see docs/internals/common/window.md
#pragma once

namespace rola {

__host__ __device__ constexpr int window_bits(int w) { return w == 1024 ? 10 : w == 512 ? 9 : 0; }

__host__ __device__ constexpr int window_heads(int w) { return 1 << (15 - window_bits(w)); }

}  // namespace rola
