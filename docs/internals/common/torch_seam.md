# `common/torch_seam.cuh` — the extension's one meeting point with torch

Mirrors `csrc/rola/src/common/torch_seam.cuh`.

Every kernel's host code reaches torch through this header and nothing else, and this header reaches torch through the
**stable ABI** only: `torch/csrc/stable/` (the `Tensor`, `DeviceGuard`, `empty`, `to`, `zero_` and `from_blob` of
torch's C shim) and `torch/headeronly/` (the dtype enum, `IntHeaderOnlyArrayRef`, `STD_TORCH_CHECK`). No TU under
`csrc/rola` includes `torch/extension.h`, ATen or c10, so the binary does not depend on torch's C++ ABI and one build
loads under every torch from `tools/build_flags.py`'s `TORCH_MIN` on ([`../rola_api.md`](../rola_api.md),
[`docs/build.md`](../../build.md#stable-abi)). A header that needs only the refusal macro
(`common/arm_switch.cuh`, `dispatch_switch.cuh`) includes `torch/headeronly/util/Exception.h` itself, so a host-only
probe of it compiles without the CUDA runtime headers.

| name | what it is |
|---|---|
| `Tensor`, `Dtype`, `Shape`, `DeviceGuard` | the stable tensor, the header-only dtype enum and array reference, and the stable device guard |
| `STD_TORCH_CHECK(cond, ...)` | the refusal: raises with the arguments streamed into the message (`shape()` prints a shape) |
| `ROLA_CUDA_CHECK(expr)`, `ROLA_CUDA_LAUNCH_CHECK()` | a CUDA runtime call's status, and the last launch's, refused with the runtime's own error name and string |
| `current_stream(device)` | the native CUDA stream torch is issuing work on (`accelerator::getCurrentStream(...).nativeHandle()`, torch 2.13) |
| `current_device()`, `current_device_properties()` | the CUDA runtime's own answers (there is no stable accessor for device properties) |
| `empty_cuda`, `zeros_cuda`, `to_cpu`, `item<T>` | tensor creation on a CUDA device, a host copy, and one element read on the host |

The stable accessors check more than the C++ ones did: `mutable_data_ptr<T>()` refuses a tensor whose dtype is not `T`,
which is a host comparison per call. `size(dim)` takes a non-negative dimension, so a trailing axis is
`size(dim() - 1)`.
