// The compliant twin of drift_host_bad.cpp, for tools/lint/drift_guards.py.

#include "common/arm_switch.cuh"

namespace fixture {

void launch(int D, int dv, int warps_per_cta) {
  rola::arm_switch<rola::CarryArmSet>(D, dv, warps_per_cta, [&](auto arm) {
    (void)decltype(arm)::index;
  });
}

}  // namespace fixture
