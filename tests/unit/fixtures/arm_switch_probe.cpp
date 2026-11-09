// The host arm switch, exercised against a synthetic selection header.
// See docs/internals/common/arm_switch.md and tests/unit/test_arm_switch.py

#include <cstdio>
#include <string>

#include <c10/util/Exception.h>

#include "common/arm_switch.cuh"

namespace {

void probe(const char* label, rola::CarryKey k) {
  int index = -1, rows = -1, factors = -1, channels = -1, split = -1;
  try {
    rola::arm_switch<rola::CarryArmSet>(k, [&](auto arm) {
      index = decltype(arm)::index;
      rows = decltype(arm)::fold_rows;
      factors = decltype(arm)::read_factors;
      channels = decltype(arm)::channels;
      split = decltype(arm)::same_split;
    });
  } catch (const std::exception& e) {
    std::printf("%s\trefused\t%s\n", label, std::string(e.what()).c_str());
    return;
  }
  std::printf("%s\tdispatched\t%d\t%d\t%d\t%d\t%d\n", label, index, rows, factors, channels, split);
}

}  // namespace

int main() {
  std::printf("count\t%d\t%d\n", rola::CarryArmSet::count, rola::CarryArmSet::declared);
  std::printf("members\t%s\n", rola::CarryArmSet::members().c_str());
  probe("built_first", {32, 1, 32, 1, 32, 0});
  probe("built_last", {128, 2, 128, 2, 64, 1});
  probe("declared_unbuilt", {32, 1, 32, 1, 32, 1});
  probe("undeclared_rows", {16, 1, 32, 1, 32, 0});
  probe("undeclared_channels", {32, 1, 32, 1, 16, 0});
  probe("undeclared_factors", {32, 1, 32, 3, 32, 0});
  probe("undeclared_split", {32, 1, 32, 1, 32, 2});
  return 0;
}
