// THE PARTIAL CALL'S ARMS -- `intra_kernel<D, kF16, false>` at each depth and accumulation and their launch, a
// translation unit of their own so the whole-window arms' unit compiles as it did.
// see docs/internals/intra/ampere/intra_partial.md

#include "intra/ampere/intra_kernel.cuh"
#include "intra/ampere/intra_partial.cuh"
#include "common/select.cuh"

namespace rola::intra::detail {

template <arch::Build B, int D, bool kF16>
cudaError_t launch_partial(const IntraParams& p, int pieces, int threads, int64_t smem, int64_t bh,
                           cudaStream_t stream) {
  if constexpr (in_image(B)) {
    //: the attribute call is host time on the launch path: raised only when a call's footprint passes the last one set
    static int64_t configured = 0;
    if (smem > configured) {
      const cudaError_t e =
          cudaFuncSetAttribute(intra_kernel<arch::BuildTag<B>, D, kF16, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      if (e != cudaSuccess) return e;
      configured = smem;
    }
    const dim3 grid(static_cast<unsigned>(p.windows * pieces), static_cast<unsigned>(bh));
    intra_kernel<arch::BuildTag<B>, D, kF16, false>
        <<<grid, static_cast<unsigned>(threads), static_cast<size_t>(smem), stream>>>(p);
    return cudaGetLastError();
  } else {
    return cudaErrorInvalidDeviceFunction;
  }
}

//: the partial arms at every build this binary carries (`builds.inc`)
#define ROLA_INTRA_PARTIAL(CC_, V_)                                                                         \
  template cudaError_t launch_partial<arch::Build{CC_, V_}, 1, false>(const IntraParams&, int, int, int64_t, \
                                                                      int64_t, cudaStream_t);                \
  template cudaError_t launch_partial<arch::Build{CC_, V_}, 1, true>(const IntraParams&, int, int, int64_t,  \
                                                                     int64_t, cudaStream_t);                 \
  template cudaError_t launch_partial<arch::Build{CC_, V_}, 2, false>(const IntraParams&, int, int, int64_t, \
                                                                      int64_t, cudaStream_t);                \
  template cudaError_t launch_partial<arch::Build{CC_, V_}, 2, true>(const IntraParams&, int, int, int64_t,  \
                                                                     int64_t, cudaStream_t);
ROLA_BUILDS(ROLA_INTRA_PARTIAL)
#undef ROLA_INTRA_PARTIAL

}  // namespace rola::intra::detail
