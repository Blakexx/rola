// csrc/rola/src/produce/produce_api.cuh -- the producer's HOST entry for the pybind TU. See
// docs/internals/produce/produce_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>

namespace rola::produce {

void rola_produce_ledger_bind(
    std::optional<Tensor> ledger);  // produce_api.md#rola-produce-ledger-bind

void rola_produce_forward(Tensor x, Tensor route_W, std::optional<Tensor> route_bias,
                          Tensor gain_bias, Tensor v_W, Tensor jobs, Tensor amp_read,
                          Tensor amp_write, std::optional<Tensor> dense_read,
                          std::optional<Tensor> dense_write, Tensor scale_read, Tensor scale_write,
                          Tensor v, Tensor v_scale, std::optional<Tensor> logit_probe,
                          std::optional<Tensor> level_probe, Tensor shapes, int64_t levels,
                          int64_t dense_read_mask,
                          int64_t dense_write_mask);  // produce_api.md#rola-produce-forward

}  // namespace rola::produce
