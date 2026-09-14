// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- host declarations; the four
// kernels and the host dispatch are `factor.cu`. ONE ENTRY, MANY LEVELS: a plan's
// levels sharing a width class, alpha and output dtype run in ONE launch
// (`blockIdx.z` indexes the table). Full solve, batching rule and mask contract:
// see docs/internals/entmax/factor.md
#pragma once

#include <cstdint>
#include <vector>

#include <torch/types.h>

namespace rola {
namespace entmax {

//: `mask` is a detached boolean candidate mask laid out like `logits`; absent means every -- see docs/internals/entmax/factor.md#factorforward
void factor_forward(const torch::Tensor& logits, const c10::optional<torch::Tensor>& mask,
                    torch::Tensor& values, const c10::optional<torch::Tensor>& values_second,
                    torch::Tensor& support_words, const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha);

//: `accumulate` adds into `d_logits` instead of storing -- what a TIED level's second -- see docs/internals/entmax/factor.md#factorbackward
void factor_backward(const torch::Tensor& values, const torch::Tensor& support_words,
                     const c10::optional<torch::Tensor>& mask, const torch::Tensor& d_values,
                     const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, bool accumulate, int64_t heads,
                     double alpha);

void softmax_forward(const torch::Tensor& logits, torch::Tensor& values,
                     const c10::optional<torch::Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                     int64_t heads);

void softmax_backward(const torch::Tensor& values, const torch::Tensor& d_values,
                      const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      bool accumulate, int64_t heads);

}  // namespace entmax
}  // namespace rola
