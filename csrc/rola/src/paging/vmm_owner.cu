#include "vmm_owner.cuh"

#include "common/torch_seam.cuh"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

namespace rola::paging {
namespace {

std::string driver_error(CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::ostringstream stream;
  stream << (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name);
  if (description != nullptr) stream << ": " << description;
  return stream.str();
}

CUmemAllocationProp allocation_prop(int device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

}  // namespace

void VmmOwner::initialize_driver() {
  static std::once_flag once;
  static CUresult init_result = CUDA_SUCCESS;
  std::call_once(once, [] { init_result = cuInit(0); });
  check(init_result, "cuInit");
}

void VmmOwner::check(CUresult result, const char* operation) {
  STD_TORCH_CHECK(result == CUDA_SUCCESS, "CUDA VMM ", operation,
                  " failed: ", driver_error(result));
}

std::size_t VmmOwner::round_up(std::size_t value, std::size_t alignment) {
  STD_TORCH_CHECK(
      alignment != 0 && value <= std::numeric_limits<std::size_t>::max() - (alignment - 1),
      "CUDA VMM size overflows size_t");
  return ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t VmmOwner::checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
  STD_TORCH_CHECK(rhs == 0 || lhs <= std::numeric_limits<std::uint64_t>::max() / rhs, "CUDA VMM ",
                  what, " overflows uint64");
  return lhs * rhs;
}

std::size_t VmmOwner::checked_size(std::uint64_t value, const char* what) {
  STD_TORCH_CHECK(value <= std::numeric_limits<std::size_t>::max(), "CUDA VMM ", what,
                  " exceeds host size_t");
  return static_cast<std::size_t>(value);
}

int VmmOwner::normalize_device(int device) {
  if (device >= 0) return device;
  int current = -1;
  STD_TORCH_CHECK(cudaGetDevice(&current) == cudaSuccess && current >= 0,
                  "CUDA VMM could not resolve the current CUDA device");
  return current;
}

VmmProbe VmmOwner::probe(int device) {
  initialize_driver();
  VmmProbe result;
  if (device < 0) {
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess || device < 0) {
      result.reason = status == cudaSuccess ? "CUDA returned an invalid current device"
                                            : cudaGetErrorString(status);
      return result;
    }
  }
  result.device = device;

  CUdevice cu_device;
  CUresult status = cuDeviceGet(&cu_device, device);
  if (status != CUDA_SUCCESS) {
    result.reason = driver_error(status);
    return result;
  }
  int supported = 0;
  status = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                cu_device);
  if (status != CUDA_SUCCESS || supported == 0) {
    result.reason = status == CUDA_SUCCESS
                        ? "CUDA driver reports virtual memory management unsupported"
                        : driver_error(status);
    return result;
  }

  DeviceGuard guard(device);
  auto prop = allocation_prop(device);
  std::size_t granularity = 0;
  status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (status != CUDA_SUCCESS || granularity == 0) {
    result.reason = status == CUDA_SUCCESS ? "driver returned zero allocation granularity"
                                           : driver_error(status);
    return result;
  }
  result.allocation_granularity = granularity;

  // A probe is not complete until the driver can reserve, map, grant access,
  // unmap, and release one granularity-sized allocation in this context.
  CUdeviceptr address = 0;
  CUmemGenericAllocationHandle handle = 0;
  bool mapped = false;
  status = cuMemAddressReserve(&address, granularity, 0, 0, 0);
  if (status == CUDA_SUCCESS) {
    status = cuMemCreate(&handle, granularity, &prop, 0);
  }
  if (status == CUDA_SUCCESS) {
    status = cuMemMap(address, granularity, 0, handle, 0);
    mapped = status == CUDA_SUCCESS;
  }
  if (status == CUDA_SUCCESS) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = cuMemSetAccess(address, granularity, &access, 1);
  }
  if (address != 0) {
    if (mapped) {
      // Mapping may have succeeded even when access setup failed. The driver
      // cleanup must follow the actual map state, not the final status.
      cuMemUnmap(address, granularity);
    }
    cuMemAddressFree(address, granularity);
  }
  if (handle != 0) cuMemRelease(handle);
  if (status != CUDA_SUCCESS) {
    result.reason = driver_error(status);
    return result;
  }
  result.supported = true;
  result.reason = "CUDA driver VMM reserve/map/access probe passed";
  return result;
}

std::shared_ptr<VmmOwner> VmmOwner::create(int device, std::int64_t dense_limit_pages,
                                           std::int64_t page_rows, std::int64_t page_cols,
                                           std::int64_t target_chunk_bytes) {
  STD_TORCH_CHECK(dense_limit_pages > 0 && page_rows > 0 && page_cols > 0,
                  "CUDA VMM page geometry must be positive");
  const auto normalized_device = normalize_device(device);
  const auto capability = probe(normalized_device);
  STD_TORCH_CHECK(capability.supported, "the RoLA page arena's VMM backing requires CUDA virtual ",
                  "memory management on device ", capability.device, "; ", capability.reason);
  auto owner = std::shared_ptr<VmmOwner>(
      new VmmOwner(capability.device, dense_limit_pages, page_rows, page_cols,
                   capability.allocation_granularity, target_chunk_bytes));
  owner->reserve_plane();
  owner->map_initial_chunk();
  return owner;
}

VmmOwner::VmmOwner(int device, std::int64_t dense_limit_pages, std::int64_t page_rows,
                   std::int64_t page_cols, std::uint64_t allocation_granularity,
                   std::int64_t target_chunk_bytes)
    : device_(device),
      dense_limit_pages_(dense_limit_pages),
      page_rows_(page_rows),
      page_cols_(page_cols),
      allocation_granularity_(allocation_granularity),
      view_tracker_(std::make_shared<ViewTracker>()) {
  STD_TORCH_CHECK(target_chunk_bytes > 0, "CUDA VMM target_chunk_bytes must be positive");
  const auto elements = checked_mul(static_cast<std::uint64_t>(page_rows),
                                    static_cast<std::uint64_t>(page_cols), "page element count");
  const auto bytes = checked_mul(elements, sizeof(float), "page byte count");
  page_bytes_ = checked_size(bytes, "page byte count");
  const auto chunk_pages =
      std::max<std::uint64_t>(1, static_cast<std::uint64_t>(target_chunk_bytes) / bytes);
  chunk_pages_ = static_cast<std::int64_t>(
      std::min<std::uint64_t>(static_cast<std::uint64_t>(dense_limit_pages), chunk_pages));
}

void VmmOwner::reserve_plane() {
  DeviceGuard guard(device_);
  const auto payload =
      checked_mul(static_cast<std::uint64_t>(page_bytes_),
                  static_cast<std::uint64_t>(dense_limit_pages_), "virtual extent");
  virtual_bytes_ = round_up(checked_size(payload, "virtual extent"), allocation_granularity_);
  check(cuMemAddressReserve(&address_, virtual_bytes_, 0, 0, 0), "cuMemAddressReserve");
}

void VmmOwner::map_initial_chunk() {
  grow(std::min<std::int64_t>(dense_limit_pages_, chunk_pages_));
}

void VmmOwner::grow(std::int64_t required_pages) {
  STD_TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  STD_TORCH_CHECK(required_pages >= 0 && required_pages <= dense_limit_pages_,
                  "CUDA VMM required page count exceeds the reserved maximum");
  if (required_pages <= mapped_capacity_pages()) return;
  DeviceGuard guard(device_);
  const auto current_pages = mapped_capacity_pages();
  const auto max_chunks = (dense_limit_pages_ + chunk_pages_ - 1) / chunk_pages_;
  const auto required_chunks = (required_pages + chunk_pages_ - 1) / chunk_pages_;
  const auto target_chunks =
      (current_pages == 0 || required_chunks >= max_chunks) ? required_chunks : required_chunks + 1;
  const auto target_pages =
      std::min<std::int64_t>(dense_limit_pages_, target_chunks * chunk_pages_);
  auto target_bytes =
      round_up(checked_size(checked_mul(static_cast<std::uint64_t>(target_pages),
                                        static_cast<std::uint64_t>(page_bytes_), "growth extent"),
                            "growth extent"),
               allocation_granularity_);
  if (target_bytes <= mapped_bytes_) return;
  STD_TORCH_CHECK(target_bytes <= virtual_bytes_,
                  "CUDA VMM growth exceeds reserved virtual extent");

  const auto bytes = target_bytes - mapped_bytes_;
  auto prop = allocation_prop(device_);
  CUmemGenericAllocationHandle handle = 0;
  bool mapped = false;
  check(cuMemCreate(&handle, bytes, &prop, 0), "cuMemCreate");
  try {
    STD_TORCH_CHECK(mapped_bytes_ <= std::numeric_limits<CUdeviceptr>::max() - address_,
                    "CUDA VMM mapped address overflows CUdeviceptr");
    const auto address = address_ + mapped_bytes_;
    check(cuMemMap(address, bytes, 0, handle, 0), "cuMemMap");
    mapped = true;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(cuMemSetAccess(address, bytes, &access, 1), "cuMemSetAccess");
    chunks_.push_back({address, bytes, handle});
    mapped_bytes_ = target_bytes;
    committed_bytes_ = static_cast<std::uint64_t>(mapped_bytes_);
  } catch (...) {
    if (mapped) cuMemUnmap(address_ + mapped_bytes_, bytes);
    cuMemRelease(handle);
    throw;
  }
}

void VmmOwner::remove_latest_chunk() {
  STD_TORCH_CHECK(!chunks_.empty(), "CUDA VMM rollback has no committed growth chunk");
  const auto chunk = chunks_.back();
  STD_TORCH_CHECK(chunk.address + chunk.bytes == address_ + mapped_bytes_,
                  "CUDA VMM rollback chunk does not end at the mapped extent");
  check(cuMemUnmap(chunk.address, chunk.bytes), "rollback cuMemUnmap");
  check(cuMemRelease(chunk.handle), "rollback cuMemRelease");
  chunks_.pop_back();
  mapped_bytes_ -= chunk.bytes;
}

void VmmOwner::rollback_to(std::int64_t target_pages) {
  STD_TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  STD_TORCH_CHECK(target_pages >= 0 && target_pages <= mapped_capacity_pages(),
                  "CUDA VMM rollback capacity is outside the mapped extent");
  DeviceGuard guard(device_);
  STD_TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
                  "CUDA VMM rollback could not quiesce the device");
  if (target_pages == mapped_capacity_pages()) return;

  const auto target =
      round_up(checked_size(checked_mul(static_cast<std::uint64_t>(target_pages),
                                        static_cast<std::uint64_t>(page_bytes_), "rollback extent"),
                            "rollback extent"),
               allocation_granularity_);
  STD_TORCH_CHECK(target <= mapped_bytes_, "CUDA VMM rollback target exceeds mapped extent");
  // A target inside a chunk rounds UP to that chunk's end, which would silently keep more
  // than was asked for. Naming it here is what makes the refusal describe the caller's
  // mistake rather than the consistency check it would otherwise trip later.
  STD_TORCH_CHECK(page_bytes_ != 0
                      && std::min<std::int64_t>(static_cast<std::int64_t>(target / page_bytes_),
                                                dense_limit_pages_)
                             == target_pages,
                  "CUDA VMM rollback target is not a committed chunk boundary: ", target_pages,
                  " pages rounds up to a mapping of ", target / page_bytes_, " pages");
  std::size_t boundary = 0;
  bool found = target == 0;
  for (const auto& chunk : chunks_) {
    STD_TORCH_CHECK(boundary <= std::numeric_limits<std::size_t>::max() - chunk.bytes,
                    "CUDA VMM rollback boundary overflows size_t");
    boundary += chunk.bytes;
    found = found || boundary == target;
  }
  STD_TORCH_CHECK(found, "CUDA VMM rollback target is not a committed chunk boundary");
  STD_TORCH_CHECK(!chunks_.empty() && target >= chunks_.front().bytes,
                  "CUDA VMM rollback cannot remove the initial headroom chunk");

  while (mapped_bytes_ > target) remove_latest_chunk();
  STD_TORCH_CHECK(mapped_bytes_ == target, "CUDA VMM rollback did not restore the plane exactly");
  STD_TORCH_CHECK(mapped_capacity_pages() == target_pages,
                  "CUDA VMM rollback did not restore the requested prior capacity");
  committed_bytes_ = static_cast<std::uint64_t>(mapped_bytes_);
}

std::int64_t VmmOwner::mapped_capacity_pages() const {
  const auto pages = page_bytes_ == 0 ? 0 : mapped_bytes_ / page_bytes_;
  return static_cast<std::int64_t>(
      std::min<std::size_t>(pages, static_cast<std::size_t>(dense_limit_pages_)));
}

Tensor VmmOwner::base() {
  STD_TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  view_tracker_->live_views.fetch_add(1, std::memory_order_relaxed);
  auto tracker = view_tracker_;
  const std::vector<std::int64_t> sizes{dense_limit_pages_, page_rows_, page_cols_};
  const std::vector<std::int64_t> strides{page_rows_ * page_cols_, page_cols_, 1};
  // Keep the driver allocation alive for every tensor view. This prevents a
  // caller holding a slice from outliving the native owner and observing an
  // unmapped raw pointer.
  auto owner_lifetime = shared_from_this();
  try {
    return torch::stable::from_blob(reinterpret_cast<void*>(address_), sizes, strides,
                                    cuda_device(device_), Dtype::Float,
                                    [tracker, owner_lifetime](void*) {
                                      tracker->live_views.fetch_sub(1, std::memory_order_relaxed);
                                      (void)owner_lifetime;
                                    });
  } catch (...) {
    view_tracker_->live_views.fetch_sub(1, std::memory_order_relaxed);
    throw;
  }
}

void VmmOwner::reset() {
  STD_TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  // Reset is intentionally logical only. Mappings and their stable addresses survive.
}

void VmmOwner::release_plane() noexcept {
  if (address_ == 0) return;
  for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
    if (it->address != 0 && it->bytes != 0) cuMemUnmap(it->address, it->bytes);
    if (it->handle != 0) cuMemRelease(it->handle);
  }
  chunks_.clear();
  mapped_bytes_ = 0;
  cuMemAddressFree(address_, virtual_bytes_);
  address_ = 0;
  committed_bytes_ = 0;
  virtual_bytes_ = 0;
}

void VmmOwner::close() {
  if (closed_) return;
  STD_TORCH_CHECK(view_tracker_->live_views.load(std::memory_order_acquire) == 0,
                  "cannot close CUDA VMM owner while exported tensor views are live");
  DeviceGuard guard(device_);
  STD_TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
                  "CUDA VMM close could not quiesce the device");
  release_plane();
  closed_ = true;
}

bool VmmOwner::quiesce() noexcept {
  try {
    DeviceGuard guard(device_);
    return cudaDeviceSynchronize() == cudaSuccess;
  } catch (...) {
    return false;
  }
}

VmmOwner::~VmmOwner() noexcept {
  if (!closed_) {
    // Never unmap while work may still reference the reserved ranges. If the
    // context is already unavailable, leaking the mapping is safer than
    // tearing it down under in-flight CUDA work; process teardown reclaims it.
    if (quiesce()) {
      try {
        DeviceGuard guard(device_);
        release_plane();
      } catch (...) {
        // Destructors cannot report driver failures.
      }
    }
  }
}

std::shared_ptr<VmmOwner> create_vmm_owner(int device, std::int64_t dense_limit_pages,
                                           std::int64_t page_rows, std::int64_t page_cols,
                                           std::int64_t target_chunk_bytes) {
  return VmmOwner::create(device, dense_limit_pages, page_rows, page_cols, target_chunk_bytes);
}

}  // namespace rola::paging
