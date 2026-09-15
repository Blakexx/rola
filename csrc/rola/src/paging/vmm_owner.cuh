#pragma once

#include "common/torch_seam.cuh"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rola::paging {

struct VmmProbe {
  int device = -1;
  bool supported = false;
  std::uint64_t allocation_granularity = 0;
  std::string reason;
};

// Owns only the CUDA-driver backing: one virtual reservation at the dense limit and a
// mapped prefix that grows. Which atom lives on which page, when a page is admitted and
// what it holds are PageArena's, so the dense and VMM backings share one ABI.
class VmmOwner final : public std::enable_shared_from_this<VmmOwner> {
 public:
  static VmmProbe probe(int device);

  static std::shared_ptr<VmmOwner> create(int device, std::int64_t dense_limit_pages,
                                          std::int64_t page_rows, std::int64_t page_cols,
                                          std::int64_t target_chunk_bytes);

  ~VmmOwner() noexcept;

  VmmOwner(const VmmOwner&) = delete;
  VmmOwner& operator=(const VmmOwner&) = delete;

  // A contiguous logical view over the WHOLE reserved range. Only the prefix reported by
  // mapped_capacity_pages is mapped and may be touched.
  Tensor base();

  void grow(std::int64_t required_pages);
  void rollback_to(std::int64_t target_pages);
  void reset();
  void close();

  bool closed() const { return closed_; }

  std::int64_t mapped_capacity_pages() const;

  std::int64_t dense_limit_pages() const { return dense_limit_pages_; }

  std::uint64_t committed_bytes() const { return committed_bytes_; }

  std::uint64_t virtual_bytes() const { return virtual_bytes_; }

  std::uint64_t allocation_granularity() const { return allocation_granularity_; }

  std::int64_t chunk_pages() const { return chunk_pages_; }

  std::string backing_kind() const { return "cuda_driver_vmm"; }

 private:
  struct Chunk {
    CUdeviceptr address = 0;
    std::size_t bytes = 0;
    CUmemGenericAllocationHandle handle = 0;
  };

  struct ViewTracker {
    std::atomic<std::int64_t> live_views{0};
  };

  VmmOwner(int device, std::int64_t dense_limit_pages, std::int64_t page_rows,
           std::int64_t page_cols, std::uint64_t allocation_granularity,
           std::int64_t target_chunk_bytes);

  static std::size_t round_up(std::size_t value, std::size_t alignment);
  static std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what);
  static std::size_t checked_size(std::uint64_t value, const char* what);
  static int normalize_device(int device);
  static void check(CUresult result, const char* operation);
  static void initialize_driver();

  bool quiesce() noexcept;

  void reserve_plane();
  void map_initial_chunk();
  void remove_latest_chunk();
  void release_plane() noexcept;

  int device_;
  std::int64_t dense_limit_pages_;
  std::int64_t page_rows_;
  std::int64_t page_cols_;
  std::uint64_t allocation_granularity_;
  std::size_t page_bytes_;
  std::int64_t chunk_pages_;
  std::uint64_t virtual_bytes_ = 0;
  std::uint64_t committed_bytes_ = 0;
  CUdeviceptr address_ = 0;
  std::size_t mapped_bytes_ = 0;
  std::vector<Chunk> chunks_;
  std::shared_ptr<ViewTracker> view_tracker_;
  bool closed_ = false;
};

std::shared_ptr<VmmOwner> create_vmm_owner(int device, std::int64_t dense_limit_pages,
                                           std::int64_t page_rows, std::int64_t page_cols,
                                           std::int64_t target_chunk_bytes);

}  // namespace rola::paging
