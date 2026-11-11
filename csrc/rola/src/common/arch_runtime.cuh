// the closed-world arch rule's runtime half -- see docs/internals/common/arch_runtime.md
#pragma once

#include <cstdint>

namespace rola {

void check_arch_table();

int64_t segment_tokens(int64_t budget, int64_t state, int64_t per_token, int64_t batch,
                       int64_t length,
                       int64_t device);  // arch_runtime.md#segment-tokens

}  // namespace rola
