// csrc/rola/src/facts/liveness_api.cuh -- the liveness pass's HOST entry for the pybind
// TU. See docs/internals/facts/liveness_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <vector>

namespace rola::facts {

Tensor liveness_words(const Tensor& read_plane, const Tensor& write_plane,
                      std::vector<int64_t> widths, int64_t dense_read,
                      std::vector<int64_t> mask_read, int64_t dense_write,
                      std::vector<int64_t> mask_write);  // liveness_api.md#liveness-words

}  // namespace rola::facts
