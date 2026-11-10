// csrc/rola/src/common/select.cu -- a device's card: the closed-world admission, its product by name and the build it
// runs, refused by name where this binary carries none. see docs/internals/common/select.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "common/select.cuh"

namespace rola {

namespace {

//: A DEVICE NAME'S PRODUCT SLUG: `measure/hardware.py`'s `slugify_gpu`. -- see docs/internals/common/select.md#slug
std::string slug_of(const char* name) {
  std::string s = name;
  for (const char* word : {"NVIDIA", "GeForce", "Tesla", "Quadro"}) {
    for (size_t at = s.find(word); at != std::string::npos; at = s.find(word))
      s.erase(at, std::strlen(word));
  }
  s = s.substr(0, s.find('-'));
  std::string out;
  for (const char ch : s) {
    if (std::isalnum(static_cast<unsigned char>(ch))) out += static_cast<char>(std::tolower(ch));
  }
  return out;
}

constexpr bool carried(Build b) {
#define ROLA_BUILD_CARRIED(CC_, V_) (b.cc == (CC_) && b.variant == (V_)) ||
  return ROLA_BUILDS(ROLA_BUILD_CARRIED) false;
#undef ROLA_BUILD_CARRIED
}

std::string carried_list() {
  std::string out;
#define ROLA_BUILD_NAME(CC_, V_) out += (out.empty() ? "" : ", ") + std::string("sm_") + std::to_string((CC_) / 10) \
                                        + (V_ ? " variant " + std::to_string(V_) : "");
  ROLA_BUILDS(ROLA_BUILD_NAME)
#undef ROLA_BUILD_NAME
  return out;
}

//: THE CLOSED-WORLD ADMISSION of one device: its row, the driver's numbers against the row, its build carried.
Card admit(int device) {
  cudaDeviceProp prop{};
  ROLA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int cc = prop.major * 100 + prop.minor * 10;
  STD_TORCH_CHECK(
      denseref::arch::tabulated(cc), "the capability table has no row for compute capability ",
      prop.major, ".", prop.minor,
      "; add it to hardware.json rather than letting an operation guess which implementation it has. "
      "This binary is closed-world: it runs the architectures it was measured on and refuses the rest.");
  const denseref::arch::Caps want = denseref::arch::caps_of(cc);

  int smem_per_sm = 0, smem_per_cta_max = 0;
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_cta_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  STD_TORCH_CHECK(
      want.smem_per_sm == smem_per_sm && want.smem_per_cta_max == smem_per_cta_max
          && want.regs_per_sm == prop.regsPerMultiprocessor
          && want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor,
      "the capability table disagrees with the driver at compute capability ", prop.major, ".",
      prop.minor, ": table {smem/SM ", want.smem_per_sm, ", smem/CTA ", want.smem_per_cta_max,
      ", regs/SM ", want.regs_per_sm, ", threads/SM ", want.max_threads_per_sm,
      "} vs driver {smem/SM ", smem_per_sm, ", smem/CTA ", smem_per_cta_max, ", regs/SM ",
      prop.regsPerMultiprocessor, ", threads/SM ", prop.maxThreadsPerMultiProcessor,
      "}. Every launch shape compiled into this binary was derived from the table, so they are wrong "
      "for this device.");

  const std::string slug = slug_of(prop.name);
  int product = -1, variant = 0;
  for (int i = 0; i < (int)(sizeof denseref::arch::kProducts / sizeof denseref::arch::kProducts[0]);
       ++i) {
    const auto& row = denseref::arch::kProducts[i];
    if (row.cc == cc && slug == row.slug) product = i, variant = row.variant;
  }

  const Build build{cc, variant};
  STD_TORCH_CHECK(carried(build), "this binary carries no build for ", prop.name, " (sm_", cc / 10,
                  variant ? " variant " + std::to_string(variant) : std::string(), "); it carries ",
                  carried_list(), ". Build it with its architecture in --//:archs.");
  return Card{device, cc, product, prop.multiProcessorCount, build, traits_of(build)};
}

}  // namespace

//: -- see docs/internals/common/select.md#card-of
const Card& card_of(int device) {
  static std::mutex mu;
  static std::map<int, Card> cards;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = cards.find(device);
  if (at != cards.end()) return at->second;
  return cards.emplace(device, admit(device)).first->second;
}

const Card& current_card() {
  int dev = 0;
  ROLA_CUDA_CHECK(cudaGetDevice(&dev));
  return card_of(dev);
}

}  // namespace rola
