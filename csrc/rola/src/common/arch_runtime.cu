// csrc/rola/src/common/arch_runtime.cu -- the runtime half of the closed-world arch rule (R4), the one
// definition every launching entry calls first. See docs/internals/common/arch_runtime.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <algorithm>

#include "common/arch_caps.cuh"
#include "common/arch_runtime.cuh"
#include "common/select.cuh"

namespace rola {

//: the current device's card admitted (`common/select.cu`) -- see docs/internals/common/arch_runtime.md#check-arch-table
void check_arch_table() { (void)current_card(); }

int64_t segment_tokens(int64_t budget, int64_t state, int64_t per_token, int64_t batch,
                       int64_t length, int64_t device) {
  const int64_t window = card_of(static_cast<int>(device)).traits.window;
  const int64_t windows = (budget - state) / (window * batch * per_token);
  return windows < 1 ? 0 : std::min(length, windows * window);
}

}  // namespace rola
