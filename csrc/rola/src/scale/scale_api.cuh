// csrc/rola/src/scale/scale_api.cuh -- the scale pass's HOST entry. see docs/internals/scale/scale_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <vector>

namespace rola::scale {

//: THREE LAUNCHES: the write masses, their prefix, the bytes. -- see docs/internals/scale/scale_api.md#scale-forward
Tensor scale_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows, Tensor fm,
                     std::vector<int64_t> widths, int64_t level_s, int64_t heads,
                     double inv_factor);

}  // namespace rola::scale
