// csrc/rola/src/common/arch_runtime.cu -- the runtime half of the closed-world arch rule (R4), the one
// definition every launching entry calls first. See docs/internals/common/arch_runtime.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

#include "common/arch_caps.cuh"
#include "common/arch_runtime.cuh"
#include "common/select.cuh"

namespace rola {

//: the current device's card admitted (`common/select.cu`) -- see docs/internals/common/arch_runtime.md#check-arch-table
void check_arch_table() { (void)current_card(); }

//: -- see docs/internals/common/arch_runtime.md#mma-quanta
const long long* mma_quanta(int device) {
  static std::mutex mu;
  static std::map<int, const long long*> resolved;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = resolved.find(device);
  if (at != resolved.end()) return at->second;
  const Card& card = card_of(device);
  const char* slug = card.product >= 0 ? denseref::arch::kProducts[card.product].slug : nullptr;
  const long long* out = nullptr;
  for (const auto& row : denseref::arch::kMmaRows) {
    if (row.cc != card.cc) continue;
    if (row.product == nullptr && out == nullptr) out = row.mma;
    if (row.product != nullptr && slug != nullptr && std::strcmp(row.product, slug) == 0) {
      out = row.mma;
      break;
    }
  }
  resolved[device] = out;
  return out;
}

//: -- see docs/internals/common/arch_runtime.md#segment-tokens
int64_t segment_tokens(int64_t budget, int64_t state, int64_t per_token, int64_t batch,
                       int64_t length, int64_t device) {
  const int64_t window = card_of(static_cast<int>(device)).traits.window;
  const int64_t windows = (budget - state) / (window * batch * per_token);
  return windows < 1 ? 0 : std::min(length, windows * window);
}

//: -- see docs/internals/common/arch_runtime.md#readout-f16
bool readout_f16(int device) {
  const long long* q = mma_quanta(device);
  return q != nullptr && q[denseref::arch::kHmmaF16F32] > q[denseref::arch::kHmmaF16F16];
}

}  // namespace rola
