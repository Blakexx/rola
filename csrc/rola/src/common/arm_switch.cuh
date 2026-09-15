// csrc/rola/src/common/arm_switch.cuh -- the HOST dispatch over the shipped arm set,
// the host twin of common/structure_switch.cuh's `uniform_switch`. KERNEL_STANDARDS.md §R9's four layers,
// host side: see docs/internals/common/arm_switch.md
#pragma once

#include <torch/headeronly/util/Exception.h>

#include <string>
#include <type_traits>
#include <utility>

#include "carry_selection.inc"

namespace rola {

//: ONE ARM'S CONSTANTS, handed to the body as compile-time values.
template <int INDEX, int D_, int DV_, int WARPS_>
struct Arm {
  static constexpr int index = INDEX;
  static constexpr int D = D_;
  static constexpr int DV = DV_;
  static constexpr int warps_per_cta = WARPS_;
};

//: THE CARRY FAMILY'S ARM SET, GENERATED -- see docs/internals/common/arm_switch.md#carry-arm-set
struct CarryArmSet {
  //: what THIS BINARY carries, and what the declaration declares. Both, because the
  //: refusal tells an unbuilt arm from an undeclared one.
  static constexpr int count = ROLA_CARRY_ARM_COUNT;
  static constexpr int declared = ROLA_CARRY_ARM_DECLARED;
  static constexpr const char* family = "carry";
  static constexpr const char* key = "(D, DV, warps_per_cta)";

  template <class Body>
  static int visit(int D, int dv, int warps_per_cta, Body&& body) {
    int hits = 0;
#define ROLA_ARM_VISIT(I, D_, DV_, W_)                                            \
  if (D == (D_) && dv == (DV_) && warps_per_cta == (W_)) {                        \
    ++hits;                                                                       \
    if (hits == 1) body(Arm<I, D_, DV_, W_>{});                                   \
  }
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_VISIT)
#undef ROLA_ARM_VISIT
    return hits;
  }

  //: THE BUILT SET, SPELLED, so a refusal is one a caller can act on.
  static std::string members() {
    std::string out;
#define ROLA_ARM_NAME(I, D_, DV_, W_)                                             \
  out += (out.empty() ? "" : ", ");                                               \
  out += "(" #D_ ", " #DV_ ", " #W_ ")";
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_NAME)
#undef ROLA_ARM_NAME
    return out.empty() ? std::string("none") : out;
  }

  //: WHICH FIELD HAS NO MEMBER AT ALL -- see docs/internals/common/arm_switch.md#offending-field
  static std::string offending_field(int D, int dv, int warps_per_cta) {
    bool d_seen = false, dv_seen = false, w_seen = false;
#define ROLA_ARM_FIELD(I, D_, DV_, W_)                                            \
  d_seen = d_seen || D == (D_);                                                   \
  dv_seen = dv_seen || dv == (DV_);                                               \
  w_seen = w_seen || warps_per_cta == (W_);
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_FIELD)
#undef ROLA_ARM_FIELD
    if (!d_seen) return "D";
    if (!dv_seen) return "DV";
    if (!w_seen) return "warps_per_cta";
    return "the combination";
  }
};

//: ONE HOST BRANCH OVER A GENERATED ARM SET; an empty set refuses every call by name.
//: -- see docs/internals/common/arm_switch.md#arm-switch
template <class ArmSet, class Body>
inline void arm_switch(int D, int dv, int warps_per_cta, Body&& body) {
  static_assert(ArmSet::count >= 0, "an arm set carries a member count");
  static_assert(ArmSet::count <= ArmSet::declared,
                "a build carries a subset of the declared arms; a set larger than the "
                "declaration means the selection and the declaration disagree");
  const int hits = ArmSet::visit(D, dv, warps_per_cta, std::forward<Body>(body));
  STD_TORCH_CHECK(hits <= 1, "the ", ArmSet::family, " arm set covers ", ArmSet::key, " = (", D,
                  ", ", dv, ", ", warps_per_cta, ") ", hits,
                  " times; an arm set is a set and each key selects one body");
  STD_TORCH_CHECK(hits == 1, "this build carries no ", ArmSet::family, " arm for ", ArmSet::key,
                  " = (", D, ", ", dv, ", ", warps_per_cta, "); the field that misses is ",
                  ArmSet::offending_field(D, dv, warps_per_cta), ". It carries ", ArmSet::count,
                  " of ", ArmSet::declared, " declared arms: ", ArmSet::members());
}

}  // namespace rola
