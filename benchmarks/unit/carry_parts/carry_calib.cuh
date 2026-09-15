// benchmarks/unit/carry_parts/carry_calib.cuh -- THE CALIBRATION KERNELS (KERNEL_STANDARDS §22 (10)): each
// isolates one cost the carry kernel's parts are made of and runs it back to back on every warp of every
// CTA, timed by the host under the locked clock -- an HMMA of the kernel's own atom, a burst of shared
// loads or stores, an asynchronous copy (bank-free or aliased), a global reduction, a CTA barrier, a
// shared-memory barrier, a warp sync. Built into the part harness's module; never shipped.
// See docs/internals/carry/calibration.md
#pragma once

#include "common/ops.cuh"
#include "common/static_for.cuh"

namespace rola::carry::calib {

namespace ops = denseref::ops;

enum CalibMode : int {
  kHmma = 0,
  kSharedLoad = 1,
  kSharedStore = 2,
  kAsyncCopy = 3,
  kAsyncCopyAliased = 4,
  kGlobalReduce = 5,
  kCtaBarrier = 6,
  kShmBarrier = 7,
  kWarpSync = 8,
  kSharedMatrix = 9
};

constexpr int kCalibSmemBytes = 96416;

struct CalibParams {
  int iters;
  float* out;       //: the reduction target, `[owners][16 rows][256 lanes]` floats
  const char* src;  //: the copy source, `[owners][256 lanes][16]` bytes
};

//: ONE CALIBRATION: `Mode` run `iters` times by every warp, `Burst` operations a unit where a unit has
//: several; each warp's results reach a stored sink so nothing is dead.
template <int Warps, int Mode, int Burst>
__global__ __launch_bounds__(Warps * 32, 1) void calib_kernel(
    __grid_constant__ const CalibParams c) {
  extern __shared__ __align__(16) char smem_raw[];
  const ops::SmemAddr base = ops::smem_addr(smem_raw);
  const int owner = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<Warps>();
  const int lane = tid & 31;
  //: a warp's own shared lines: `Burst` lines of 128 bytes each, a line a bank line.
  const ops::SmemAddr lines = base + (uint32_t)(warp * 128 * 64);

  if constexpr (Mode == kHmma) {
    const uint32_t w = 0x3F003E80u ^ ((uint32_t)lane & 1u);
    const uint32_t ab[4] = {w, w, w, w};
    const uint32_t bf[4] = {w, w, w, w};
    float y[8][4], d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
    });
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<8>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(y[j], ab, bf + 2 * (j & 1));
      });
      ops::mma(d, ab, bf);
    }
    float sink = d[0] + d[1] + d[2] + d[3];
    rola::static_for<8>([&](auto Jc) {
      rola::static_for<4>([&](auto Ec) { sink += y[decltype(Jc)::value][decltype(Ec)::value]; });
    });
    ops::store_shared_u32(lines, __float_as_uint(sink));
  }

  if constexpr (Mode == kSharedLoad || Mode == kSharedStore) {
    rola::static_for<Burst>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4),
                            (uint32_t)tid);
    });
    __syncwarp();
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        const ops::SmemAddr at = lines + (uint32_t)(k * 128 + lane * 4);
        if constexpr (Mode == kSharedLoad)
          acc ^= ops::load_shared_u32(at);
        else
          ops::store_shared_u32(at, (uint32_t)i ^ (uint32_t)k);
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(Burst * 128 + lane * 4), acc);
  }

  if constexpr (Mode == kSharedMatrix) {
    //: `Burst` two-n-tile `ldmatrix.trans` loads a unit, each its own line: the readout's B and the
    //: fold's operand loads, whose short-scoreboard stalls hold back the HMMAs that consume them.
    rola::static_for<Burst>([&](auto Kc) {
      ops::store_shared_u32(lines + (uint32_t)(decltype(Kc)::value * 128 + lane * 4),
                            (uint32_t)tid);
    });
    __syncwarp();
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        uint32_t r[4];
        ops::load_frag_t(r, lines + (uint32_t)(decltype(Kc)::value * 128));
        acc ^= r[0] ^ r[3];
      });
    }
    ops::store_shared_u32(lines + (uint32_t)(Burst * 128 + lane * 4), acc);
  }

  if constexpr (Mode == kAsyncCopy || Mode == kAsyncCopyAliased) {
    //: `Burst` 16-byte runs a unit into the warp's lines: bank-free (each run its own line and lane
    //: offset) or ALIASED (every run's lane at the same 128-byte stride, one bank group).
    const char* const row = c.src + (long)owner * 256 * 16 + (long)tid * 16;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      rola::static_for<Burst>([&](auto Kc) {
        constexpr int k = decltype(Kc)::value;
        const uint32_t off =
            Mode == kAsyncCopy ? (uint32_t)(k * 128 + (lane & 7) * 16) : (uint32_t)(k * 128);
        ops::stage_run<16>(lines + off + (uint32_t)((lane >> 3) * 128 * Burst), row);
      });
      ops::stage_commit();
      ops::stage_wait<0>();
    }
  }

  if constexpr (Mode == kGlobalReduce) {
    float* const at = ops::pin_address(c.out + (long)owner * 16 * 256 + tid);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      //: a row an iteration of sixteen, so no reduction repeats the one before it at its address.
      rola::static_for<Burst>([&](auto Kc) {
        ops::red_global_add_f32(at + (long)((i + decltype(Kc)::value) & 15) * 256, 1.0f);
      });
    }
  }

  if constexpr (Mode == kCtaBarrier) {
    uint32_t acc = 0u;
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      acc += (uint32_t)i;
      ops::rendezvous();
    }
    ops::store_shared_u32(lines + (uint32_t)(lane * 4), acc);
  }

  if constexpr (Mode == kShmBarrier) {
    const ops::SmemAddr bar = base + (uint32_t)(Warps * 128 * 64 + warp * 8);
    if (lane == 0) ops::mbar_init(bar, 1u);
    __syncwarp();
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if (lane == 0) ops::mbar_arrive(bar);
      ops::mbar_wait(bar, (uint32_t)i & 1u);
    }
  }

  if constexpr (Mode == kWarpSync) {
    const ops::SmemAddr at = lines + (uint32_t)(lane * 4);
#pragma unroll 1
    for (int i = 0; i < c.iters; ++i) {
      if (lane == 0) ops::store_shared_u32(lines, (uint32_t)i);
      __syncwarp();
      ops::store_shared_u32(at, ops::load_shared_u32(lines));
    }
  }
}

//: THE INSTANTIATED CALIBRATIONS: (warps, mode, burst). The host picks one by value.
#define ROLA_CALIB_SET_X(F)          \
  F(8, kHmma, 1)                     \
  F(4, kHmma, 1)                     \
  F(8, kSharedLoad, 4)               \
  F(8, kSharedLoad, 16)              \
  F(8, kSharedLoad, 64)              \
  F(8, kSharedStore, 4)              \
  F(8, kSharedStore, 16)             \
  F(8, kSharedStore, 64)             \
  F(8, kSharedMatrix, 4)             \
  F(8, kSharedMatrix, 16)            \
  F(8, kAsyncCopy, 4)                \
  F(8, kAsyncCopyAliased, 4)         \
  F(8, kGlobalReduce, 1)             \
  F(8, kCtaBarrier, 1)               \
  F(8, kShmBarrier, 1)               \
  F(8, kWarpSync, 1)

//: ONE LAUNCH of calibration (warps, mode, burst) over `owners` CTAs, `iters` units a warp; returns once
//: the stream has finished it. The caller times it.
inline void calibrate(int64_t warps, int64_t mode, int64_t burst, int64_t iters, int64_t owners,
                      Tensor& out, const Tensor& src) {
  STD_TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == Dtype::Float && out.numel() >= owners * 16 * 256,
      "calibrate: out is a CUDA float32 tensor of owners x 16 x 256");
  STD_TORCH_CHECK(
      src.is_cuda() && src.scalar_type() == Dtype::Byte && src.numel() >= owners * 256 * 16,
      "calibrate: src is a CUDA uint8 tensor of owners x 256 x 16");
  const CalibParams c{(int)iters, out.mutable_data_ptr<float>(),
                      reinterpret_cast<const char*>(src.mutable_data_ptr<uint8_t>())};
  bool found = false;
#define ROLA_CALIB_LAUNCH(W_, M_, B_)                                                               \
  if (!found && warps == (W_) && mode == (M_) && burst == (B_)) {                                   \
    auto* kern = calib_kernel<(W_), (M_), (B_)>;                                                    \
    STD_TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kCalibSmemBytes) \
                    == cudaSuccess,                                                                   \
                "calibrate: smem attribute");                                                         \
    kern<<<dim3((unsigned)owners, 1u), (W_) * 32, kCalibSmemBytes, current_stream()>>>(c); \
    found = true;                                                                                     \
  }
  ROLA_CALIB_SET_X(ROLA_CALIB_LAUNCH)
#undef ROLA_CALIB_LAUNCH
  STD_TORCH_CHECK(found, "calibrate: no calibration (warps ", warps, ", mode ", mode, ", burst ",
                  burst, ")");
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "calibrate: the launch failed");
}

}  // namespace rola::carry::calib
