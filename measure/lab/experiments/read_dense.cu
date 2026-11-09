// measure/lab/experiments/read_dense.cu -- QUESTION: what does the value-carve readout of a DENSE read side cost a
// warp-window alone -- production's `dense_factors`, `dense_den` and `read_dense` over a chunk whose every token reads
// every digit of the box, nothing else in the window -- against its tensor floor (fp32 accumulate: one accumulator a
// token over the box's 512 leaves)? METHOD: one CTA an SM, the (16, 2, 16, 2, 64, 1) arm's plan (32 digits x K 16 x
// 64 channels); each window a 128-slot read chunk of dense rows (K rows and 32 digits' coefficients, f16), the image
// in halves as sm_86's area runs it, the factors and the den, then both halves' readouts with a rendezvous between.
// see docs/lab/read_dense.md
#include <cstdio>
#include <vector>

#include "carry/ampere/carry_kernel.cuh"
#include "common/arm.cuh"
#include "carry/ampere/readout_dense.cuh"
#include "lab.cuh"

namespace read_dense {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
using A = rola::Arm<0, 16, 2, 16, 2, 64, 1>;
using P = rc::Plan<A, lab::kLabBuild>;

//: THE LAUNCH'S SHARED BYTES, the lab build's plan
inline int smem_of(const cudaDeviceProp&) { return P::kSmemBytes; }

constexpr int kWindows = 64, kSlots = 128;

__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

//: MODE 0: an f16 panel (sm_86's production form); 1 an fp32 panel; 2 the control (the span without the readout); 3 the
//: section: the factors and the den alone; 4 the factors alone; 5 the den alone
template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void read(long long* t, int* out) {
  constexpr bool kF32 = MODE == 1, kEmpty = MODE == 2, kFactorsOnly = MODE >= 3,
                 kFactors = MODE != 5, kDen = MODE != 4;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  rc::CarryParams p{};
  p.tables = rc::tables_of<P::kWindow>(1, P::kSmemBytes);
  p.reduce = kF32 ? rc::kReduceF32 : rc::kReduceF16;
  const ops::SmemAddr image = ops::smem_addr(smem + P::kAreaAt);
  const ops::SmemAddr rows = image + static_cast<uint32_t>(P::kImageRing);
  const ops::SmemAddr panel = rows + static_cast<uint32_t>(rc::read_panel<P>(kSlots)),
                      dens = rows + static_cast<uint32_t>(rc::read_dens<P>(kSlots, p.reduce)),
                      exits = rows + static_cast<uint32_t>(rc::read_ents<P>(kSlots, p.reduce));
  //: the image half and the den image small f16 values; each digit's exponent and den factor; every token's scale byte
  for (int i = tid; i < P::kImageRing / 4; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kAreaAt)[i] =
        0x24002400u ^ (mix(static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kMaxExtent * rc::kMaxExtent / 2; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kDenImage)[i] =
        0x24002400u ^ (mix(~static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kWarps * rc::kMaxExtent; i += rc::kThreads)
    reinterpret_cast<int*>(smem + P::kExps)[i] = 4;
  if (tid < rc::kMaxExtent) reinterpret_cast<float*>(smem + P::kDenFactor)[tid] = 1.0f;
  for (int i = tid; i < 2 * P::kWindow; i += rc::kThreads) (smem + p.tables.scale_at)[i] = 2;
  __syncthreads();

  long long busy = 0;
  float fold = 0.0f;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: the chunk's dense rows: small f16 K rows and coefficients, drawn a window, as the read fill copies them; the
    //: panel and the den words cleared, as the fill clears them
    for (int i = tid; i < kSlots * P::kReadRow / 4; i += rc::kThreads)
      reinterpret_cast<uint32_t*>(smem + (rows - ops::smem_addr(smem)))[i] =
          0x30003000u
          ^ (mix(static_cast<uint32_t>(i) * 31u + static_cast<uint32_t>(w)) & 0x07ff07ffu);
    for (int i = tid; i < (rc::read_ents<P>(kSlots, p.reduce) - rc::read_panel<P>(kSlots)) / 4;
         i += rc::kThreads)
      reinterpret_cast<uint32_t*>(smem + (panel - ops::smem_addr(smem)))[i] = 0u;
    //: two barriers before the clock (read_alone's measured lining-up)
    __syncthreads();
    __syncthreads();
    const long long t0 = clock64();
    if constexpr (!kEmpty && kFactors)
      rc::dense_factors<P>(p, smem, w, 0, kSlots, rows, exits, warp, lane);
    if constexpr (!kEmpty) ops::rendezvous();
    if constexpr (!kEmpty && kDen) rc::dense_den<P>(smem, kSlots, rows, dens, exits, warp, lane);
    if constexpr (!kEmpty && !kFactorsOnly) {
      rc::read_dense<P, kF32>(kSlots, 0, P::kHalfPlanes, image, P::kHalfPlanes - 1u, rows, panel,
                              exits, warp, lane);
      ops::rendezvous();
      rc::read_dense<P, kF32>(kSlots, P::kHalfPlanes, 2 * P::kHalfPlanes, image,
                              P::kHalfPlanes - 1u, rows, panel, exits, warp, lane);
    }
    const long long t1 = clock64();
    busy += t1 - t0;
    //: the sink read between barriers: every warp's panel words and den words written before it
    __syncthreads();
    fold += reinterpret_cast<const float*>(smem + (panel - ops::smem_addr(smem)))[tid]
            + reinterpret_cast<const float*>(smem + (dens - ops::smem_addr(smem)))[tid];
    __syncthreads();
  }
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = busy;
  lab::sink(out, lab::bits(fold));
}

template <int MODE>
void variant(const lab::Bench& b, const char* label) {
  const std::string name = lab::kernel("read", {MODE});
  if constexpr (MODE == 0 || MODE == 1) b.expect(name, "HMMA.16816.F32", ">=", 8);
  b.exempt(name,
           "an isolation of the production readout; its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  cudaDeviceProp dev{};
  LAB_CHECK(cudaGetDeviceProperties(&dev, 0));
  const int smem = smem_of(dev);
  LAB_CHECK(cudaFuncSetAttribute(read<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const lab::Samples s = b.variant(
      [&] { read<MODE><<<b.sms, rc::kThreads, smem>>>(b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(label,
              "a timing isolation; the readout's arithmetic is the oracle's (test_carry_readout)");
  b.record(label, "cycles a warp-window", s);
  if constexpr (MODE >= 2) return;
  //: THE TENSOR FLOOR, a warp-window: a scheduler's two warps' MMAs -- a tile a digit (32 x 8 tiles a warp) and the
  //: den's (a tile a block over the warps) -- at fp32 accumulate (32 cycles a scheduler on sm_86, 8 on sm_80)
  cudaDeviceProp prop{};
  LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
  const double mma = prop.major == 8 && prop.minor == 6 ? 32.0 : 8.0;
  const double tiles = kSlots / 16.0,
               mmas = tiles * P::kDigits + tiles * (P::kDigits / 8) / rc::kWarps;
  const std::string nm = std::string(label) + ": tensor floor";
  b.unchecked(nm, "computed on the host from the chunk's shape");
  std::printf(
      "{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"cycles a warp-window\", \"samples\": [%.1f]}\n",
      nm.c_str(), 2.0 * mmas * mma);
}

}  // namespace read_dense

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  read_dense::variant<0>(b, "f16 panel (sm_86's form)");
  read_dense::variant<1>(b, "fp32 panel");
  read_dense::variant<2>(b, "control: the timed span without the readout");
  read_dense::variant<3>(b, "section: the factors and the den alone");
  read_dense::variant<4>(b, "section: the factors alone");
  read_dense::variant<5>(b, "section: the den alone");
  return 0;
}
