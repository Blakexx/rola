// the extension's one meeting point with torch, through torch's STABLE ABI only -- see docs/internals/common/torch_seam.md
#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <initializer_list>
#include <string>

namespace rola {

using Tensor = torch::stable::Tensor;
using Dtype = torch::headeronly::ScalarType;
using DeviceGuard = torch::stable::accelerator::DeviceGuard;
using Shape = torch::headeronly::IntHeaderOnlyArrayRef;

//: A shape as a refusal prints it: `[2, 3, 4]`.
inline std::string shape(Shape sizes) {
  std::string out = "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(sizes[i]);
  }
  return out + "]";
}

//: A CUDA runtime call's status, refused with the runtime's own name for it.
#define ROLA_CUDA_CHECK(expr)                                                              \
  do {                                                                                     \
    const cudaError_t rola_cuda_status_ = (expr);                                          \
    STD_TORCH_CHECK(rola_cuda_status_ == cudaSuccess, "CUDA call failed: ",                \
                    cudaGetErrorName(rola_cuda_status_), ": ", cudaGetErrorString(rola_cuda_status_)); \
  } while (0)

//: A kernel launch's status: the launch is asynchronous, so this reports what the launch itself refused.
#define ROLA_CUDA_LAUNCH_CHECK() ROLA_CUDA_CHECK(cudaGetLastError())

inline int32_t current_device() {
  int device = 0;
  ROLA_CUDA_CHECK(cudaGetDevice(&device));
  return (int32_t)device;
}

//: The stream torch is issuing work on for `device` (the current device by default).
inline cudaStream_t current_stream(int32_t device = -1) {
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(device < 0 ? current_device() : device)
          .nativeHandle());
}

inline cudaDeviceProp current_device_properties() {
  cudaDeviceProp props;
  ROLA_CUDA_CHECK(cudaGetDeviceProperties(&props, current_device()));
  return props;
}

inline torch::stable::Device cuda_device(int32_t index = -1) {
  return torch::stable::Device(torch::headeronly::DeviceType::CUDA,
                               index < 0 ? current_device() : index);
}

//: An uninitialized tensor of `dtype` on CUDA device `device` (the current device by default).
inline Tensor empty_cuda(Shape sizes, Dtype dtype, int32_t device = -1) {
  return torch::stable::empty(sizes, dtype, std::nullopt, cuda_device(device));
}

inline Tensor zeros_cuda(Shape sizes, Dtype dtype, int32_t device = -1) {
  Tensor out = empty_cuda(sizes, dtype, device);
  torch::stable::zero_(out);
  return out;
}

inline Tensor to_cpu(const Tensor& t) {
  return torch::stable::to(t, torch::stable::Device(torch::headeronly::DeviceType::CPU));
}

//: The one value a one-element tensor holds, read on the host.
template <class T>
inline T item(const Tensor& t) {
  STD_TORCH_CHECK(t.numel() == 1, "a scalar read wants one element, got ", t.numel());
  return to_cpu(t).const_data_ptr<T>()[0];
}

}  // namespace rola
