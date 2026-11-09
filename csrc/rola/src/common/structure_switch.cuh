// csrc/rola/src/common/structure_switch.cuh -- CONSTITUTION.md §R9's four enforcement layers, once.
// See docs/internals/common/structure_switch.md
#pragma once

#include <type_traits>

namespace rola {

namespace detail {

//: The members from `A` on, one force-inlined frame each: the dispatch holds no lambda of its own to be outlined.
template <int A, int NA, class F>
__device__ __forceinline__ void uniform_switch_from(int value, F& body) {
  if constexpr (A < NA) {
    if (value == A) {
      body(std::integral_constant<int, A>{});
      return;
    }
    uniform_switch_from<A + 1, NA>(value, body);
  } else {
    __trap();
  }
}

}  // namespace detail

//: ONE CTA-UNIFORM BRANCH OVER A GENERATED SET. -- see docs/internals/common/structure_switch.md#uniform-switch
template <int NA, class F>
__device__ __forceinline__ void uniform_switch(int value, F&& body) {
  static_assert(NA >= 1, "a generated structure set is never empty");
  detail::uniform_switch_from<0, NA>(value, body);
}

}  // namespace rola
