// csrc/rola/src/carry/carry.cu -- THE CARRY DISPATCH: the kernel boundary's refusals, the
// addressing block's one derivation, and the arm the launch resolves to.
//
// This translation unit holds no kernel body of its own beyond the build stamp: an arm's
// body lives in its generated translation unit, so an arm-subset build is a file subset.
// See docs/internals/carry/carry.md
#include "carry/carry_api.cuh"

#include "common/torch_seam.cuh"

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
//: THE PHASE TRACE BINDING: a device int64 tensor `[ctas][warps][cap]` the next launches stamp
//: their first `ctas` CTAs' events into; unbound in production. -- carry_kernel.md#phase-trace
long long* g_phase_trace = nullptr;
int64_t g_phase_trace_ctas = 0;
int64_t g_phase_trace_cap = 0;

}  // namespace

void carry_forward(const Tensor& read, const Tensor& write, const Tensor& gain, const Tensor& v,
                   Tensor& num, Tensor& den, const std::vector<int64_t>& widths, int64_t dv,
                   int64_t page_bits, int64_t warps_per_cta,
                   const std::vector<int64_t>& carve_order, const std::vector<int64_t>& schedule,
                   const Tensor& liveness, const Tensor& activity,
                   const std::optional<Tensor>& state_in, const std::optional<Tensor>& state_out,
                   const std::optional<Tensor>& page_table) {
  check_arch_table();
  CarryCall call =
      derive_carry_call(read, write, gain, v, num, den, widths, dv, page_bits, warps_per_cta,
                        carve_order, schedule, liveness, activity, state_in, state_out, page_table);
  CarryParams& p = call.p;
  const int64_t BH = call.BH;
  const int D = (int)widths.size();
  p.ledger = g_phase_ledger;
  if (p.ledger != nullptr)
    STD_TORCH_CHECK(g_phase_ledger_ctas >= (int64_t)p.g.owners * BH,
                    "the bound phase ledger covers ", g_phase_ledger_ctas,
                    " CTAs, the launch needs ", (int64_t)p.g.owners * BH);
  p.trace = g_phase_trace;
  p.trace_ctas = (int)g_phase_trace_ctas;
  p.trace_cap = (int)g_phase_trace_cap;

  const auto stream = current_stream();
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
  STD_TORCH_CHECK(launched, "the carry arm set resolved a member that carries no launch");
  STD_TORCH_CHECK(status == cudaSuccess, "the carry launch failed: ", cudaGetErrorString(status));
}

void carry_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger) {
    g_phase_ledger = nullptr;
    g_phase_ledger_ctas = 0;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(2) == kPhases && ledger->is_contiguous(),
                  "the phase ledger is a contiguous CUDA int64 tensor [ctas][warps][", kPhases,
                  "]");
  g_phase_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  g_phase_ledger_ctas = ledger->size(0);
}

void carry_trace_bind(const std::optional<Tensor>& trace, int64_t warps_per_cta) {
  if (!trace) {
    g_phase_trace = nullptr;
    g_phase_trace_ctas = 0;
    g_phase_trace_cap = 0;
    return;
  }
  STD_TORCH_CHECK(trace->is_cuda() && trace->scalar_type() == Dtype::Long && trace->dim() == 3
                      && trace->size(1) == warps_per_cta && trace->is_contiguous(),
                  "the phase trace is a contiguous CUDA int64 tensor [ctas][", warps_per_cta,
                  " warps][cap]");
  g_phase_trace = reinterpret_cast<long long*>(trace->mutable_data_ptr<int64_t>());
  g_phase_trace_ctas = trace->size(0);
  g_phase_trace_cap = trace->size(2);
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
  auto buf = empty_cuda({4}, Dtype::Int);
  stamp::carry_stamp_kernel<<<1, 1, 0, current_stream()>>>(buf.mutable_data_ptr<int32_t>());
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the carry build stamp did not run");
  const auto host = to_cpu(buf);
  const int32_t* f = host.const_data_ptr<int32_t>();
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
