// measure/lab/experiments/handoff_alone.cu -- QUESTION: what does production's hand-off cost a window alone on sm_86 --
// the digits' keys and the den's maxima (`post_keys`, `den_tables_max`), their exponents and the den's tables
// (`post_exps`, `den_tables_out`), the image stored in halves (`store_batches`), with the kernel's rendezvous between
// -- against the hand-off lab's lean forms (docs/lab/handoff.md)? METHOD: one CTA an SM with arm 0's plan tenants; the
// box's state in the fold's registers (values spanning octaves by list and digit, made opaque each window), each
// list's unit in the book (one list never live); 64 windows, a window's span two barriers past its setup.
// see docs/lab/handoff_alone.md
#include <climits>
#include <cstdio>

#include "carry/ampere/carry_kernel.cuh"
#include "common/arm.cuh"
#include "lab.cuh"

namespace handoff_alone {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm, lab::kLabBuild>;
constexpr int kWindows = 64;

__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

//: A STATE VALUE: list `l`, digit `d`, channel `c` -- a sign, a mantissa and an octave by list and digit (-8 .. 7); digit
//: `d` zero in the lists `l` with `(l + d) % 11 == 0`, so some keys meet a zero row
__device__ __forceinline__ float value(int l, int d, int c) {
  const uint32_t h = mix(static_cast<uint32_t>((l * 32 + d) * 32 + c));
  if ((l + d) % 11 == 0) return 0.0f;
  const int e = static_cast<int>(mix(static_cast<uint32_t>(l * 32 + d)) % 16u) - 8;
  const float m = 1.0f + static_cast<float>(h & 0xffffu) / 65536.0f;
  return (h >> 31 ? -m : m) * __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

//: MODE 0 the hand-off as sm_86 runs it a read chunk (the keys and the den's maxima, a rendezvous, the exponents and the
//: den's tables, half 0's stores, a rendezvous, half 1's); 1 the keys and the den's maxima; 2 those, the rendezvous and
//: the exponents and the den's tables; 3 the stores (both halves, a rendezvous between); 4 the empty span; 5 the stores
//: at the kernel's call: one call site, the halves' batch bounds at run time
template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void handoff(long long* t, int* out) {
  constexpr bool kKeys = MODE <= 2, kExps = MODE == 0 || MODE == 2,
                 kStores = MODE == 0 || MODE == 3, kCall = MODE == 5;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane >> 2,
            t4 = lane & 3;
  //: the tenants as the kernel sets them: a list's unit (list 5 never live), the keys cleared, the scales' bits
  if (tid < rc::kMaxExtent)
    reinterpret_cast<rc::UnitBook*>(smem + P::kBook)[tid] = rc::UnitBook{
        tid == 5 ? rc::kUnitNone : static_cast<int>(mix(static_cast<uint32_t>(tid)) % 13u) - 6,
        0.0f};
  if (tid < 2 * rc::kMaxExtent) reinterpret_cast<int*>(smem + P::kKeys)[tid] = INT_MIN;
  if (tid == 0) *reinterpret_cast<uint2*>(smem + P::kBits) = make_uint2(1u << 23, 127u << 23);
  //: the warp's state: list `warp + 8 J`, digit `16 M + g + 8 (I >> 1)`, channel `8 N + 2 t + (I & 1)`; its den words
  rc::State<P> st;
  rola::static_for<P::kFoldLists>([&](auto J) {
    rola::static_for<P::kFoldMTiles>([&](auto M) {
      rola::static_for<P::kFoldNTiles>([&](auto N) {
        rola::static_for<4>([&](auto I) {
          st.acc[J][M][N][I] =
              value(warp + rc::kWarps * J, 16 * M + g + 8 * (I >> 1), 8 * N + 2 * t4 + (I & 1));
        });
      });
    });
  });
  rola::static_for<P::kDenRegs>([&](auto I) {
    st.den[I] = 1.0f + static_cast<float>(mix(static_cast<uint32_t>(tid * 4 + I)) & 0xffu);
  });
  const ops::SmemAddr image = ops::smem_addr(smem + P::kAreaAt);
  if (tid < rc::kMaxExtent) reinterpret_cast<uint32_t*>(smem + P::kDenMax)[tid] = 0u;
  __syncthreads();
  //: the stores alone take the exponents of window 0, formed before the clock
  if constexpr (!kKeys) {
    const int top = rc::post_keys<P>(st, smem, 0, warp, lane);
    rc::den_tables_max<P>(smem, st, warp, lane);
    ops::rendezvous();
    rc::post_exps<P>(smem, 0, top, warp, lane);
    rc::den_tables_out<P>(smem, st, warp, lane);
    __syncthreads();
  }
  long long busy = 0;
  int sink = 0;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: the state opaque, so nothing of it is hoisted out of the window
    rola::static_for<P::kFoldLists>([&](auto J) {
      rola::static_for<P::kFoldMTiles>([&](auto M) {
        rola::static_for<P::kFoldNTiles>([&](auto N) {
          rola::static_for<4>([&](auto I) { asm volatile("" : "+f"(st.acc[J][M][N][I])); });
        });
      });
    });
    rola::static_for<P::kDenRegs>([&](auto I) { asm volatile("" : "+f"(st.den[I])); });
    //: the den's maxima cleared, as the kernel's window does
    if (tid < rc::kMaxExtent) reinterpret_cast<uint32_t*>(smem + P::kDenMax)[tid] = 0u;
    //: two barriers before the clock (`read_alone`: after one, the warps are not lined up)
    __syncthreads();
    __syncthreads();
    const long long t0 = clock64();
    int top = 0;
    if constexpr (kKeys) {
      top = rc::post_keys<P>(st, smem, w, warp, lane);
      rc::den_tables_max<P>(smem, st, warp, lane);
    }
    if constexpr (kExps) {
      ops::rendezvous();
      rc::post_exps<P>(smem, w, top, warp, lane);
      rc::den_tables_out<P>(smem, st, warp, lane);
    }
    if constexpr (kStores) {
      rc::store_batches<P, P::kHalfPlanes - 1>(st, smem, image, 0, 2, warp, lane);
      ops::rendezvous();
      rc::store_batches<P, P::kHalfPlanes - 1>(st, smem, image, 2, 4, warp, lane);
    }
    if constexpr (kCall) {
#pragma unroll 1
      for (int h = 0; h < ops::opaque(2); ++h) {
        if (h > 0) ops::rendezvous();
        rc::store_batches<P, P::kHalfPlanes - 1>(st, smem, image, 2 * h, 2 * h + 2, warp, lane);
      }
    }
    const long long t1 = clock64();
    busy += t1 - t0;
    //: the sink read between barriers: the exponents and the image written before it
    __syncthreads();
    sink += reinterpret_cast<const int*>(smem + P::kExps)[tid]
            + reinterpret_cast<const int*>(smem + P::kAreaAt)[tid * 31];
    __syncthreads();
  }
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = busy;
  lab::sink(out, sink);
}

template <int MODE>
void variant(const lab::Bench& b, const char* label) {
  const std::string name = lab::kernel("handoff", {MODE});
  b.exempt(
      name,
      "an isolation of the production hand-off (no MMA); its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(handoff<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 P::kSmemBytes));
  const lab::Samples s = b.variant(
      [&] { handoff<MODE><<<b.sms, rc::kThreads, P::kSmemBytes>>>(b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(
      label,
      "a timing isolation; the hand-off's image and exponents are the oracle's (test_carry_readout)");
  b.record(label, "cycles a warp-window", s);
}

}  // namespace handoff_alone

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  handoff_alone::variant<0>(b, "the hand-off as sm_86 runs it a read chunk");
  handoff_alone::variant<1>(b, "section: the keys and the den's maxima");
  handoff_alone::variant<2>(b,
                            "section: the keys, a rendezvous, the exponents and the den's tables");
  handoff_alone::variant<3>(b, "section: the stores, both halves, a rendezvous between");
  handoff_alone::variant<4>(b, "control: the empty span");
  handoff_alone::variant<5>(
      b, "section: the stores at the kernel's call (the halves' bounds at run time)");
  return 0;
}
