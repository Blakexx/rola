// THE PRODUCER'S HOST ENTRY -- the operands validated and the build's kernel launched (`ampere/produce_kernel.cuh`).
// see docs/internals/produce/produce.md

#include "common/torch_seam.cuh"

#include <cstring>

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "produce/ampere/produce_kernel.cuh"
#include "produce/produce_api.cuh"

namespace rola {
namespace produce {
namespace detail {

void check_bf16(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::BFloat16, name,
                  " is a CUDA bf16 tensor, got ", t.scalar_type());
}

void check_f32(const Tensor& t, const char* name, int64_t numel) {
  STD_TORCH_CHECK(
      t.is_cuda() && t.scalar_type() == Dtype::Float && t.is_contiguous() && t.numel() == numel,
      name, " is a contiguous CUDA fp32 tensor of ", numel, " elements, got ", t.numel());
}

//: THE BOUND PHASE LEDGER, or null: the next launches book into it.
long long* g_ledger = nullptr;
int64_t g_ledger_jobs = 0;

}  // namespace detail

//: -- see docs/internals/produce/produce_api.md#rola-produce-ledger-bind
void rola_produce_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    detail::g_ledger_jobs = 0;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == kProduceWarps && ledger->size(2) == kProducePhases
                      && ledger->is_contiguous(),
                  "the producer's phase ledger is a contiguous CUDA int64 tensor [jobs][",
                  kProduceWarps, "][", kProducePhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  detail::g_ledger_jobs = ledger->size(0);
}

//: -- see docs/internals/produce/produce_api.md#rola-produce-forward
void rola_produce_forward(Tensor x, Tensor route_W, std::optional<Tensor> route_bias,
                          Tensor gain_bias, Tensor v_W, Tensor jobs, Tensor amp_read,
                          Tensor amp_write, std::optional<Tensor> dense_read,
                          std::optional<Tensor> dense_write, Tensor scale_read, Tensor scale_write,
                          Tensor v, Tensor v_scale, std::optional<Tensor> logit_probe,
                          std::optional<Tensor> level_probe, Tensor shapes, int64_t levels,
                          int64_t dense_read_mask, int64_t dense_write_mask) {
  check_arch_table();
  detail::check_bf16(x, "x");
  detail::check_bf16(route_W, "route_W");
  detail::check_bf16(v_W, "v_W");
  STD_TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x is [BT, hidden], contiguous, got ",
                  shape(x.sizes()));
  STD_TORCH_CHECK(route_W.dim() == 3 && route_W.stride(1) == 1,
                  "route_W is [H, hidden, C] with its hidden axis contiguous, got ",
                  shape(route_W.sizes()));
  STD_TORCH_CHECK(v_W.dim() == 2 && v_W.stride(1) == 1,
                  "v_W is [H * d_v, hidden] with its hidden axis contiguous, got ",
                  shape(v_W.sizes()));
  const int64_t BT = x.size(0), hidden = x.size(1), H = route_W.size(0), C = route_W.size(2);
  if (route_bias.has_value()) detail::check_f32(*route_bias, "route_bias", H * C);
  STD_TORCH_CHECK(gain_bias.is_cuda() && gain_bias.scalar_type() == Dtype::Float
                      && gain_bias.is_contiguous() && gain_bias.numel() % H == 0,
                  "gain_bias is a contiguous CUDA fp32 [H, G]");
  //: every hidden run is read in 16-byte chunks: eight elements, aligned
  STD_TORCH_CHECK(hidden % 8 == 0 && route_W.stride(0) % 8 == 0 && route_W.stride(2) % 8 == 0
                      && v_W.stride(0) % 8 == 0,
                  "the hidden size and the weights' strides are multiples of 8, got hidden ",
                  hidden);
  for (const Tensor* t : {&x, &route_W, &v_W}) {
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 == 0,
                    "x and the weights start 16-byte aligned");
  }
  //: the copy sources are 32-bit element offsets (produce.md#stage): x's tokens, a head's routing columns, a value
  //: job's channels
  STD_TORCH_CHECK(
      x.numel() < (int64_t{1} << 32) && route_W.size(2) * route_W.stride(2) < (int64_t{1} << 32)
          && v_W.numel() < (int64_t{1} << 32),
      "x, a head's routing columns and the value weights each hold under 2^32 elements");
  STD_TORCH_CHECK(route_W.size(1) == hidden && v_W.size(1) == hidden, "route_W and v_W read x's ",
                  hidden, " hidden channels");
  STD_TORCH_CHECK(v_W.size(0) % H == 0, "v_W's ", v_W.size(0), " channels are not ", H,
                  " heads' worth");
  const int64_t d_v = v_W.size(0) / H;
  STD_TORCH_CHECK(jobs.is_cuda() && jobs.scalar_type() == Dtype::Int && jobs.is_contiguous()
                      && jobs.dim() == 2 && jobs.size(1) == kJobFields,
                  "jobs is the plan's [n, ", kJobFields, "] int32 job table on the device");
  STD_TORCH_CHECK(levels >= 1 && levels <= 4, "the routing depth is one to four levels, got ",
                  levels);
  for (auto [t, name] : {std::pair{&amp_read, "amp_read"}, {&amp_write, "amp_write"}}) {
    STD_TORCH_CHECK(t->scalar_type() == Dtype::Byte && t->is_contiguous() && t->dim() == 4
                        && t->size(1) == H && t->size(0) * t->size(2) == BT,
                    name, " is head-major [B, H, T, rows] u8, contiguous");
  }
  STD_TORCH_CHECK(amp_read.size(2) == amp_write.size(2), "both sides' rows hold the same tokens");
  //: A SIDE'S DENSE PLANE: its dense levels' f16 rows, present exactly when its mask names a level
  const std::optional<Tensor>* dense[2] = {&dense_read, &dense_write};
  const int64_t masks[2] = {dense_read_mask, dense_write_mask};
  for (int side = 0; side < 2; ++side) {
    STD_TORCH_CHECK(masks[side] >= 0 && masks[side] < (int64_t{1} << levels)
                        && dense[side]->has_value() == (masks[side] != 0),
                    "side ", side, "'s dense levels' f16 rows come with their mask (", masks[side],
                    ")");
    if (dense[side]->has_value()) {
      const Tensor& t = **dense[side];
      STD_TORCH_CHECK(t.scalar_type() == Dtype::Half && t.is_contiguous() && t.dim() == 4
                          && t.sizes().slice(0, 3) == amp_read.sizes().slice(0, 3),
                      "side ", side,
                      "'s dense plane is head-major [B, H, T, rows] f16, contiguous, the rows' "
                      "tokens");
    }
  }
  for (auto [t, name] : {std::pair{&scale_read, "scale_read"}, {&scale_write, "scale_write"}}) {
    detail::check_f32(*t, name, BT * H * levels);
  }
  detail::check_f32(v_scale, "v_scale", BT * H);
  STD_TORCH_CHECK(v.scalar_type() == Dtype::Half && v.is_contiguous() && v.numel() == BT * H * d_v,
                  "v is [BT, H, d_v] f16");
  const int64_t tiles = (BT + kTokenTile - 1) / kTokenTile;
  STD_TORCH_CHECK(!shapes.is_cuda() && shapes.scalar_type() == Dtype::Short
                      && shapes.is_contiguous() && shapes.numel() == jobs.size(0),
                  "shapes is the plan's host int16 shape word a job");
  STD_TORCH_CHECK(jobs.size(0) <= kMaxJobs, "a launch runs at most ", kMaxJobs, " jobs, got ",
                  jobs.size(0));
  for (auto [t, name] : {std::pair{&logit_probe, "logit_probe"}, {&level_probe, "level_probe"}}) {
    if (t->has_value()) detail::check_f32(**t, name, jobs.size(0) * BT * kMaxColumns);
  }

  ProduceParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  p.route_W = reinterpret_cast<const __nv_bfloat16*>(route_W.const_data_ptr());
  p.w_head = route_W.stride(0);
  p.w_col = route_W.stride(2);
  p.route_bias = route_bias.has_value() ? route_bias->const_data_ptr<float>() : nullptr;
  p.gain_bias = gain_bias.const_data_ptr<float>();
  p.v_W = reinterpret_cast<const __nv_bfloat16*>(v_W.const_data_ptr());
  p.v_row = v_W.stride(0);
  p.jobs = jobs.const_data_ptr<int32_t>();
  p.amp_read = reinterpret_cast<uint8_t*>(amp_read.mutable_data_ptr());
  p.amp_write = reinterpret_cast<uint8_t*>(amp_write.mutable_data_ptr());
  for (int side = 0; side < 2; ++side) {
    p.dense[side] = dense[side]->has_value()
                        ? reinterpret_cast<__half*>((*dense[side])->mutable_data_ptr())
                        : nullptr;
    p.dense_rows[side] = dense[side]->has_value() ? static_cast<int>((*dense[side])->size(3)) : 0;
    p.dense_mask[side] = static_cast<int>(masks[side]);
  }
  p.scale_read = scale_read.mutable_data_ptr<float>();
  p.scale_write = scale_write.mutable_data_ptr<float>();
  p.v = reinterpret_cast<__half*>(v.mutable_data_ptr());
  p.v_scale = v_scale.mutable_data_ptr<float>();
  p.logit_probe = logit_probe.has_value() ? logit_probe->mutable_data_ptr<float>() : nullptr;
  p.level_probe = level_probe.has_value() ? level_probe->mutable_data_ptr<float>() : nullptr;
  STD_TORCH_CHECK(detail::g_ledger == nullptr || detail::g_ledger_jobs >= jobs.size(0),
                  "the bound phase ledger holds ", detail::g_ledger_jobs,
                  " jobs; this launch runs ", jobs.size(0));
  std::memcpy(p.shapes, shapes.const_data_ptr(),
              static_cast<size_t>(shapes.numel()) * sizeof(int16_t));
  p.ledger = detail::g_ledger;
  p.BT = static_cast<int>(BT);
  p.T = static_cast<int>(amp_read.size(2));
  //: T's fast division (CUTLASS's FastDivmod form): with l = ceil(log2 T), mul = ceil(2^(31 + l) / T) < 2^32 and a
  //: token's sequence umulhi(token, mul) >> (l - 1), exact below 2^31
  if (p.T > 1) {
    int l = 0;
    while ((int64_t{1} << l) < p.T) ++l;
    p.t_mul = static_cast<uint32_t>(((uint64_t{1} << (31 + l)) + p.T - 1) / p.T);
    p.t_shift = l - 1;
  }
  p.hidden = static_cast<int>(hidden);
  p.H = static_cast<int>(H);
  p.C = static_cast<int>(C);
  p.G = static_cast<int>(gain_bias.numel() / H);
  p.rows_read = static_cast<int>(amp_read.size(3));
  p.rows_write = static_cast<int>(amp_write.size(3));
  p.D = static_cast<int>(levels);
  p.d_v = static_cast<int>(d_v);

  const DeviceGuard guard(x.get_device_index());
  //: ONCE: the attribute call is host time on the launch path; the carveout to shared, so two blocks' stages fit an SM
  static bool configured = false;
  if (!configured) {
    const int optin = card_of(x.get_device_index()).traits.caps.smem_per_cta_max;
    STD_TORCH_CHECK(kProduceSmem <= optin, "the producer's ", kProduceSmem,
                    "-byte stages exceed this device's ", optin, "-byte opt-in limit");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(
        detail::produce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kProduceSmem));
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::produce_kernel,
                                         cudaFuncAttributePreferredSharedMemoryCarveout,
                                         cudaSharedmemCarveoutMaxShared));
    configured = true;
  }
  detail::produce_kernel<<<dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(jobs.size(0))),
                           kProduceThreads, kProduceSmem, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace produce
}  // namespace rola
