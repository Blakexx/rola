// csrc/rola/src/common/cost.cuh -- THE COST OF A FORM'S LOAD: its MMAs on its busiest scheduler or its shared-memory
// traffic on its SM, the larger, in quanta of 1/256 cycle at a product's rates. see docs/internals/common/cost.md
#pragma once

#include "common/hardware.cuh"

namespace denseref {
namespace arch {

//: a product's rates in quanta (resolved on the host, passed at launch): see docs/internals/common/cost.md#quanta
struct Quanta {
  long long mma[kMmaKinds];
  long long wavefront, atomic, atomic_wavefront;
};

//: a form's MMAs on its busiest scheduler and shared-memory traffic on its SM: see docs/internals/common/cost.md#cost
struct Load {
  long long mma[kMmaKinds];
  long long wavefronts, atomics, atomic_wavefronts;
};

constexpr long long cost(const Quanta& q, const Load& l) {
  long long tensor = 0;
  for (int k = 0; k < kMmaKinds; ++k) tensor += l.mma[k] * q.mma[k];
  const long long atomics = l.atomics * q.atomic > l.atomic_wavefronts * q.atomic_wavefront
                                ? l.atomics * q.atomic
                                : l.atomic_wavefronts * q.atomic_wavefront;
  const long long smem = l.wavefronts * q.wavefront + atomics;
  return tensor > smem ? tensor : smem;
}

//: the parity cases tests/unit/test_hardware.py prices too (the RTX 3080 Ti's quanta)
static_assert(cost(Quanta{{8226, 4130, 8226, 4130}, 256, 513, 256}, Load{{16, 0, 0, 0}, 64, 16, 24})
              == 131616);
static_assert(cost(Quanta{{8226, 4130, 8226, 4130}, 256, 513, 256}, Load{{1, 0, 0, 0}, 64, 16, 24})
              == 24592);

}  // namespace arch
}  // namespace denseref
