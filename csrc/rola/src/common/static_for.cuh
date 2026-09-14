// csrc/rola/src/common/static_for.cuh -- see docs/internals/common/static_for.md
#pragma once

#include <type_traits>

namespace rola {

//: `f(I)` for I = 0 .. N-1, I compile-time. -- see docs/internals/common/static_for.md#static-for
template <int N, class F>
__host__ __device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

}  // namespace rola
