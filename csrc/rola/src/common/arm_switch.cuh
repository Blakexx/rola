// csrc/rola/src/common/arm_switch.cuh -- the HOST dispatch over the shipped arm set,
// the host twin of common/structure_switch.cuh's `uniform_switch`. CONSTITUTION.md §R9's four layers,
// host side: see docs/internals/common/arm_switch.md
#pragma once

#include <torch/headeronly/util/Exception.h>

#include <string>
#include <type_traits>
#include <utility>

#include "carry_selection.inc"
#include "common/arm.cuh"

namespace rola {

//: THE CARRY FAMILY'S ARM SET, GENERATED -- see docs/internals/common/arm_switch.md#carry-arm-set
struct CarryArmSet {
  //: what THIS BINARY carries, and what the declaration declares. Both, because the
  //: refusal tells an unbuilt arm from an undeclared one.
  static constexpr int count = ROLA_CARRY_ARM_COUNT;
  static constexpr int declared = ROLA_CARRY_ARM_DECLARED;
  static constexpr const char* family = "carry";
  static constexpr const char* key =
      "(fold_rows, fold_factors, read_rows, read_factors, channels, same_split)";

  template <class Body>
  static int visit(const CarryKey& k, Body&& body) {
    int hits = 0;
#define ROLA_ARM_VISIT(I, FR_, FF_, RR_, RF_, CH_, SS_)                                                       \
  if (k.fold_rows == (FR_) && k.fold_factors == (FF_) && k.read_rows == (RR_) && k.read_factors == (RF_) \
      && k.channels == (CH_) && k.same_split == (SS_)) {                                                  \
    ++hits;                                                                                               \
    if (hits == 1) body(Arm<I, FR_, FF_, RR_, RF_, CH_, SS_>{});                                          \
  }
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_VISIT)
#undef ROLA_ARM_VISIT
    return hits;
  }

  //: THE BUILT SET, SPELLED, so a refusal is one a caller can act on.
  static std::string members() {
    std::string out;
#define ROLA_ARM_NAME(I, FR_, FF_, RR_, RF_, CH_, SS_) \
  out += (out.empty() ? "" : ", ");                    \
  out += "(" #FR_ ", " #FF_ ", " #RR_ ", " #RF_ ", " #CH_ ", " #SS_ ")";
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_NAME)
#undef ROLA_ARM_NAME
    return out.empty() ? std::string("none") : out;
  }

  //: WHICH FIELD HAS NO MEMBER AT ALL -- see docs/internals/common/arm_switch.md#offending-field
  static std::string offending_field(const CarryKey& k) {
    bool seen[6] = {false, false, false, false, false, false};
#define ROLA_ARM_FIELD(I, FR_, FF_, RR_, RF_, CH_, SS_) \
  seen[0] = seen[0] || k.fold_rows == (FR_);           \
  seen[1] = seen[1] || k.fold_factors == (FF_);        \
  seen[2] = seen[2] || k.read_rows == (RR_);           \
  seen[3] = seen[3] || k.read_factors == (RF_);        \
  seen[4] = seen[4] || k.channels == (CH_);            \
  seen[5] = seen[5] || k.same_split == (SS_);
    ROLA_CARRY_ARM_SET_X(ROLA_ARM_FIELD)
#undef ROLA_ARM_FIELD
    static constexpr const char* names[6] = {"fold_rows",    "fold_factors", "read_rows",
                                             "read_factors", "channels",     "same_split"};
    for (int i = 0; i < 6; ++i)
      if (!seen[i]) return names[i];
    return "the combination";
  }
};

//: ONE HOST BRANCH OVER A GENERATED ARM SET; an empty set refuses every call by name.
//: -- see docs/internals/common/arm_switch.md#arm-switch
template <class ArmSet, class Body>
inline void arm_switch(const CarryKey& k, Body&& body) {
  static_assert(ArmSet::count >= 0, "an arm set carries a member count");
  static_assert(ArmSet::count <= ArmSet::declared,
                "a build carries a subset of the declared arms; a set larger than the "
                "declaration means the selection and the declaration disagree");
  const int hits = ArmSet::visit(k, std::forward<Body>(body));
  STD_TORCH_CHECK(hits <= 1, "the ", ArmSet::family, " arm set covers ", ArmSet::key, " = (",
                  k.fold_rows, ", ", k.fold_factors, ", ", k.read_rows, ", ", k.read_factors, ", ",
                  k.channels, ", ", k.same_split, ") ", hits,
                  " times; an arm set is a set and each key selects one body");
  STD_TORCH_CHECK(hits == 1, "this build carries no ", ArmSet::family, " arm for ", ArmSet::key,
                  " = (", k.fold_rows, ", ", k.fold_factors, ", ", k.read_rows, ", ",
                  k.read_factors, ", ", k.channels, ", ", k.same_split,
                  "); the field that misses is ", ArmSet::offending_field(k), ". It carries ",
                  ArmSet::count, " of ", ArmSet::declared, " declared arms: ", ArmSet::members());
}

}  // namespace rola
