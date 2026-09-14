// csrc/rola/src/common/structure_switch.cuh -- KERNEL_STANDARDS.md §R9's four enforcement layers, once.
// See docs/internals/common/structure_switch.md
#pragma once

#include <type_traits>

#include "common/static_for.cuh"

namespace rola {

//: ONE CTA-UNIFORM BRANCH OVER A GENERATED SET. -- see docs/internals/common/structure_switch.md#uniform-switch
template <int NA, class F>
__device__ __forceinline__ void uniform_switch(int value, F&& body) {
  static_assert(NA >= 1, "a generated structure set is never empty");
  bool hit = false;
  static_for<NA>([&](auto Ac) {
    if (value == decltype(Ac)::value) {
      body(Ac);
      hit = true;
    }
  });
  if (!hit) __trap();
}

}  // namespace rola
