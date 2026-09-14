// csrc/rola/src/carry/carry.cu -- THE CARRY DISPATCH: the kernel boundary's refusals, the
// addressing block's one derivation, and the arm the launch resolves to.
//
// This translation unit holds no kernel body of its own beyond the build stamp: an arm's
// body lives in its generated translation unit, so an arm-subset build is a file subset.
// See docs/internals/carry/carry.md
#include "carry/carry_api.cuh"

#include <c10/cuda/CUDAStream.h>

#include "carry/box.cuh"
#include "carry/carry_arm_abi.cuh"
#include "carry/carry_host.cuh"
#include "carry/params.cuh"
#include "common/arch_runtime.cuh"
#include "common/arm_switch.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"

namespace rola::carry {

ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_DECLARE)

namespace {

//: THE PHASE LEDGER BINDING: a device int64 tensor `[ctas][warps][kPhases]` the next launches
//: add their per-warp phase cycles into; unbound (nullptr) in production. The launch checks
//: the bound tensor covers its grid. -- carry_kernel.md#phase-ledger
long long* g_phase_ledger = nullptr;
int64_t g_phase_ledger_ctas = 0;

}  // namespace

void carry_forward(const at::Tensor& read, const at::Tensor& write, const at::Tensor& gain,
                   const at::Tensor& v, at::Tensor& num, at::Tensor& den,
                   const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                   int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
                   const std::vector<int64_t>& schedule, const at::Tensor& liveness,
                   const at::Tensor& activity, const c10::optional<at::Tensor>& state_in,
                   const c10::optional<at::Tensor>& state_out,
                   const c10::optional<at::Tensor>& page_table) {
  CarryCall call =
      derive_carry_call(read, write, gain, v, num, den, widths, dv, page_bits, warps_per_cta,
                        carve_order, schedule, liveness, activity, state_in, state_out, page_table);
  CarryParams& p = call.p;
  const int64_t BH = call.BH;
  const int D = (int)widths.size();
  p.ledger = g_phase_ledger;
  if (p.ledger != nullptr)
    TORCH_CHECK(g_phase_ledger_ctas >= (int64_t)p.g.owners * BH, "the bound phase ledger covers ",
                g_phase_ledger_ctas, " CTAs, the launch needs ", (int64_t)p.g.owners * BH);

  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = cudaSuccess;
  bool launched = false;
  arm_switch<CarryArmSet>(D, (int)dv, (int)warps_per_cta, [&](auto A) {
    using Arm = decltype(A);
#define ROLA_CARRY_ARM_CALL(I_, D_, DV_, W_)                                          \
  if (Arm::index == (I_)) {                                                            \
    launched = true;                                                                   \
    status = carry_arm_forward_##I_(p, p.g.owners, (int)BH, stream);                   \
  }
    ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_CALL)
#undef ROLA_CARRY_ARM_CALL
  });
  TORCH_CHECK(launched, "the carry arm set resolved a member that carries no launch");
  TORCH_CHECK(status == cudaSuccess, "the carry launch failed: ", cudaGetErrorString(status));
}

void carry_ledger_bind(const std::optional<at::Tensor>& ledger) {
  if (!ledger) {
    g_phase_ledger = nullptr;
    g_phase_ledger_ctas = 0;
    return;
  }
  TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == at::kLong && ledger->dim() == 3
                  && ledger->size(2) == kPhases && ledger->is_contiguous(),
              "the phase ledger is a contiguous CUDA int64 tensor [ctas][warps][", kPhases, "]");
  g_phase_ledger = reinterpret_cast<long long*>(ledger->data_ptr<int64_t>());
  g_phase_ledger_ctas = ledger->size(0);
}

namespace stamp {

//: THE DEVICE-SIDE FACT: the only answer a STALE binary cannot produce.
__global__ void carry_stamp_kernel(int* out) {
  out[0] = kAtomLeaves * 1000 + warps_per_sm;
  out[1] = denseref::design::kCapsDigest;
  out[2] = state_elems_per_sm(denseref::arch::kCaps);
  out[3] = kClusterCtas;
}

}  // namespace stamp

int64_t carry_build_stamp() {
  check_arch_table();
  auto buf = at::empty({4}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  stamp::carry_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(buf.data_ptr<int32_t>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the carry build stamp did not run");
  const auto host = buf.to(at::kCPU);
  const int32_t* f = host.data_ptr<int32_t>();
  //: THE FOUR FACTS AS ONE NUMBER, the way every other family's stamp reads: a mixed
  //: radix wide enough that no field can carry into its neighbour.
  return ((((int64_t)f[0] * 1000003 + f[1]) * 1000003) + f[2]) * 16 + f[3];
}

std::vector<std::vector<int64_t>> carry_census() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_CARRY_ARM_CENSUS(I_, D_, DV_, W_)                                          \
  {                                                                                     \
    const ArmCensus c = carry_arm_census_##I_();                                        \
    out.push_back({(int64_t)(I_), (int64_t)(D_), (int64_t)(DV_), (int64_t)(W_), c.regs, \
                   c.local_bytes, c.smem_bytes, c.max_threads, c.binary_version,        \
                   c.threads, c.box_leaves, c.atoms, c.boxes, c.state_elems,            \
                   c.read_bytes, c.pool_bytes});                                       \
  }
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_CENSUS)
#undef ROLA_CARRY_ARM_CENSUS
  return out;
}

std::vector<std::vector<int64_t>> carry_arms() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_CARRY_ARM_ROW(I_, D_, DV_, W_) out.push_back({(int64_t)(D_), (int64_t)(DV_), (int64_t)(W_)});
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_ROW)
#undef ROLA_CARRY_ARM_ROW
  return out;
}

}  // namespace rola::carry
