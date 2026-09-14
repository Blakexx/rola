// The host arm switch, exercised against a synthetic selection header.
// See docs/internals/common/arm_switch.md and tests/unit/test_arm_switch.py

#include <cstdio>
#include <string>

#include <c10/util/Exception.h>

#include "common/arm_switch.cuh"

namespace {

void probe(const char* label, int D, int dv, int warps) {
  int index = -1, seen_d = -1, seen_dv = -1, seen_w = -1;
  try {
    rola::arm_switch<rola::CarryArmSet>(D, dv, warps, [&](auto arm) {
      index = decltype(arm)::index;
      seen_d = decltype(arm)::D;
      seen_dv = decltype(arm)::DV;
      seen_w = decltype(arm)::warps_per_cta;
    });
  } catch (const std::exception& e) {
    std::printf("%s\trefused\t%s\n", label, std::string(e.what()).c_str());
    return;
  }
  std::printf("%s\tdispatched\t%d\t%d\t%d\t%d\n", label, index, seen_d, seen_dv, seen_w);
}

}  // namespace

int main() {
  std::printf("count\t%d\t%d\n", rola::CarryArmSet::count, rola::CarryArmSet::declared);
  std::printf("members\t%s\n", rola::CarryArmSet::members().c_str());
  probe("built_first", 1, 64, 8);
  probe("built_last", 2, 128, 4);
  probe("declared_unbuilt", 2, 64, 8);
  probe("undeclared_depth", 3, 64, 8);
  probe("undeclared_width", 2, 256, 8);
  probe("undeclared_warps", 2, 128, 2);
  return 0;
}
