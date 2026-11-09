// csrc/rola/src/common/arm.cuh -- ONE CARRY ARM'S KEY AND CONSTANTS, torch-free: the per-arm units and the host
// dispatch (common/arm_switch.cuh) share them. see docs/internals/common/arm.md
#pragma once

namespace rola {

//: A CARRY ARM'S KEY: the carve's operation-plan pair (tools/gen_shards.py's `ARM_KEY`, in its order).
struct CarryKey {
  int fold_rows, fold_factors, read_rows, read_factors, channels, same_split;
};

template <int INDEX, int FOLD_ROWS, int FOLD_FACTORS, int READ_ROWS, int READ_FACTORS, int CHANNELS,
          int SAME_SPLIT>
struct Arm {
  static constexpr int index = INDEX;
  static constexpr int fold_rows = FOLD_ROWS;
  static constexpr int fold_factors = FOLD_FACTORS;
  static constexpr int read_rows = READ_ROWS;
  static constexpr int read_factors = READ_FACTORS;
  static constexpr int channels = CHANNELS;
  static constexpr int same_split = SAME_SPLIT;
};

}  // namespace rola
