// csrc/rola/src/carry/carry.cu -- THE CARRY DISPATCH: the kernel boundary's refusals, the operand block from the
// carve, and the arm its key resolves to. An arm's body lives in its generated translation unit, so an arm-subset
// build is a file subset. see docs/internals/carry/carry.md
#include "carry/carry_api.cuh"

#include "common/torch_seam.cuh"

#include "carry/carry_arm_abi.cuh"
#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/hardware.cuh"
#include "common/arm_switch.cuh"
#include "common/arch_caps.cuh"

#include <cstring>

namespace rola::carry {

ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_DECLARE)

namespace {

//: THE PHASE LEDGER BINDING: a device int64 tensor `[ctas][warps][kCarryPhases]` the next launches add their per-warp
//: phase cycles into; unbound (nullptr) in production. -- carry_kernel.md#phases
long long* g_ledger = nullptr;
int64_t g_ledger_ctas = 0;
//: THE PROBE BINDING: a device fp32 tensor `[BH, N, d_v]` the next launches write their final state into, absolute;
//: unbound in production. -- see docs/internals/carry/ampere/fold.md#probe
float* g_probe = nullptr;
int64_t g_probe_values = 0;
float* g_probe_den = nullptr;
int64_t g_probe_den_values = 0;
//: THE CAPTURE BINDING: a host u8 tensor of `sizeof(CarryCapture)` bytes the next launch copies its operand block and
//: grid into; unbound in production. -- see docs/internals/carry/carry_api.md#carry-capture-bind
uint8_t* g_capture = nullptr;

bool pow2(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

//: A LEVEL'S ROWS as the kernel reads them: `[B, T, H, width]`, digits contiguous, u8 or f16, on the row alignment law.
Plane plane_of(const Tensor& t, int64_t B, int64_t T, int64_t H, int64_t width, const char* name,
               int level) {
  STD_TORCH_CHECK(t.is_cuda() && (t.scalar_type() == Dtype::Byte || t.scalar_type() == Dtype::Half),
                  name, " level ", level, " is a CUDA u8 or f16 tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(
      t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H && t.size(3) == width,
      name, " level ", level, " is [B, T, H, width] = [", B, ", ", T, ", ", H, ", ", width,
      "], got ", shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(3) == 1, name, " level ", level, "'s digits are contiguous");
  const int64_t size = t.element_size();
  return Plane{static_cast<const uint8_t*>(t.const_data_ptr()), t.stride(0) * size,
               t.stride(1) * size, t.stride(2) * size};
}

//: ONE SIDE OF THE CARVE: its list level and operand levels cover every level; a listed side's list level spans the
//: arm's list group (`group`: the box's lists or digits), its operand levels in level order; a side with no list level names two levels, its row
//: level then its coefficient level; an operand slice at least a 4-byte copy. -- see docs/internals/carry/carry.md#refusals
Side side_of(const std::vector<Tensor>& rows, int64_t list, const std::vector<int64_t>& operand,
             const std::vector<int64_t>& widths, const std::vector<int64_t>& extents, int64_t B,
             int64_t T, int64_t H, int W, int64_t group, const char* name) {
  const int D = (int)widths.size();
  Side s{};
  s.list = (int)list;
  s.operands = (int)operand.size();
  STD_TORCH_CHECK((int)rows.size() == D, name, " holds ", rows.size(),
                  " levels' rows; the carve has ", D);
  STD_TORCH_CHECK(list >= -1 && list < D, name, "'s list level is -1 or a level, got ", list);
  STD_TORCH_CHECK(list < 0 || extents[list] == group, name, "'s list level spans ",
                  list < 0 ? 0 : extents[list],
                  " digits of the box; a listed level spans the arm's list group of ", group);
  STD_TORCH_CHECK(
      s.operands >= 1 && (int)operand.size() + (list >= 0 ? 1 : 0) == D
          && (list >= 0 || s.operands == 2),
      name,
      " lists at most one level and forms its operand from the others, every level once, or with no list level "
      "names a row level and a coefficient level; this side lists ",
      list, " and forms ", operand.size(), " of ", D);
  for (int i = 0; i < s.operands; ++i) {
    const int64_t l = operand[i];
    STD_TORCH_CHECK(
        l >= 0 && l < D && l != list && (list < 0 || i == 0 || l > operand[i - 1])
            && (list >= 0 || i == 0 || l != operand[0]),
        name,
        "'s operand levels are distinct levels, not the list level, in level order where listed");
    s.operand[i] = (int)l;
    s.rows[i] = plane_of(rows[l], B, T, H, widths[l], name, (int)l);
    s.elem[i] = (int)rows[l].element_size();
    STD_TORCH_CHECK(extents[l] * s.elem[i] >= 4 && extents[l] * s.elem[i] <= kSliceBytes, name,
                    " level ", l, "'s slice is ", extents[l] * s.elem[i],
                    " bytes; a slice is a 4-byte copy to ", kSliceBytes);
    //: a window's row offsets are 32-bit (`fill`): its tokens' and query heads' strides under 2^31 together
    STD_TORCH_CHECK(
        s.rows[i].st > 0 && (W - 1) * s.rows[i].st + (H - 1) * s.rows[i].sh < (int64_t{1} << 31),
        name, " level ", l, "'s token stride ", s.rows[i].st, " and head stride ", s.rows[i].sh,
        " bytes pass a window's 32-bit row offsets");
  }
  return s;
}

//: ONE SIDE'S LISTS as the facts pass writes them in their final form: 8-byte entries, the boundaries' prefix rows,
//: the groups' totals, the union's tokens and count; empty where the side has no list level.
//: -- see docs/internals/carry/carry.md#lists
SideLists lists_of(const std::vector<Tensor>& t, int64_t BH, int64_t windows, int64_t width,
                   int64_t heads, int W, int64_t group, const char* name) {
  STD_TORCH_CHECK(t.size() == 5, name, " is (entries, pre, totals, union, count), got ", t.size(),
                  " tensors");
  if (width == 0) return SideLists{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 1};
  const int64_t groups = (width + group - 1) / group;
  const std::vector<std::vector<int64_t>> want = {
      {BH, windows, width, W * heads, 4},
      {BH, windows, groups, group + 1, list_at(W) * heads},
      {BH, windows, groups, list_at(W) * heads},
      {BH, windows, groups, W * heads},
      {BH, windows, groups}};
  for (int k = 0; k < 5; ++k) {
    const bool wide = k == 2 || k == 4;
    STD_TORCH_CHECK(t[k].is_cuda() && t[k].is_contiguous()
                        && t[k].scalar_type() == (wide ? Dtype::Int : Dtype::Short)
                        && t[k].sizes() == Shape(want[k]),
                    name, " tensor ", k, " is a contiguous CUDA ", wide ? "int32 " : "int16 ",
                    shape(Shape(want[k])), ", got ", shape(t[k].sizes()));
  }
  return SideLists{reinterpret_cast<const uint16_t*>(t[3].const_data_ptr()),
                   t[4].const_data_ptr<int32_t>(),
                   static_cast<const uint2*>(t[0].const_data_ptr()),
                   reinterpret_cast<const uint16_t*>(t[1].const_data_ptr()),
                   t[2].const_data_ptr<int32_t>(),
                   (int)width,
                   (int)groups,
                   (int)heads};
}

}  // namespace

Tensor carry_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows, Tensor values,
                     std::vector<Tensor> write_lists, Tensor write_exp, Tensor factors,
                     Tensor factor_shift, std::vector<Tensor> read_lists, Tensor scale, Tensor num,
                     Tensor den, Tensor check, std::vector<int64_t> widths,
                     std::vector<int64_t> extents, int64_t channels, int64_t write_list,
                     std::vector<int64_t> write_operand, int64_t read_list,
                     std::vector<int64_t> read_operand, std::vector<int64_t> key,
                     int64_t accumulate, int64_t reduce, int64_t forced) {
  check_arch_table();
  const int D = (int)widths.size();
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels && (int)extents.size() == D, "the carve names ", D,
                  " widths and ", extents.size(), " extents; the carry is built to depth ",
                  kMaxLevels);
  STD_TORCH_CHECK(values.is_cuda() && values.scalar_type() == Dtype::Half && values.dim() == 4
                      && values.stride(3) == 1,
                  "values is the CUDA f16 [B, T, H, d_v] plane, channels contiguous");
  const int64_t B = values.size(0), L = values.size(1), H = values.size(2), dv = values.size(3),
                BH = B * H;
  //: THE CALL'S CARD, its values' device's, which the launch runs on: its window is the card's build's
  const int device = values.get_device_index();
  const Card& card = card_of(device);
  const DeviceGuard guard(device);
  const int W = card.traits.window;
  const int64_t windows = (L + W - 1) / W;
  STD_TORCH_CHECK(L >= 1 && L * dv < 0x40000000, "the call's tokens and channels index in 32 bits");
  STD_TORCH_CHECK(
      pow2(channels) && channels >= 32 && channels <= 128 && dv % channels == 0,
      "the box's channel extent is a power of two from 32 to 128 dividing d_v; the carve names ",
      channels, " of ", dv);
  int64_t leaves = 1, boxes = dv / channels;
  int stride[kMaxLevels] = {};
  for (int l = D - 1; l >= 0; --l) {
    STD_TORCH_CHECK(
        pow2(widths[l]) && widths[l] >= 16 && pow2(extents[l]) && widths[l] % extents[l] == 0,
        "level ", l, " is ", widths[l], " digits, a power of two from 16, carved by ", extents[l],
        ", a power of two dividing it");
    stride[l] = (int)boxes;
    boxes *= widths[l] / extents[l];
    leaves *= extents[l];
  }
  STD_TORCH_CHECK(leaves * channels == kStateValues, "the box holds ", leaves, " leaves x ",
                  channels, " channels; a CTA's state is ", kStateValues, " values");
  STD_TORCH_CHECK(boxes <= 0x7FFFFFFF && BH <= 65535,
                  "the grid is (boxes, batch-heads) within the launch limits");

  CarryParams p{};
  p.D = D;
  p.H = (int)H;
  p.L = (int)L;
  p.windows = (int)windows;
  p.dv = (int)dv;
  p.channels = (int)channels;
  for (int l = 0; l < D; ++l) {
    p.widths[l] = (int)widths[l];
    p.extents[l] = (int)extents[l];
    p.stride[l] = stride[l];
  }
  //: THE GROUP: the read side's query heads a write head, by its rows; its lists merge them, so it lists
  STD_TORCH_CHECK(
      !read_rows.empty() && read_rows[0].dim() == 4 && read_rows[0].size(2) % H == 0,
      "the read side's rows hold a whole number of query heads a write head; values hold ", H,
      " write heads");
  const int64_t heads = read_rows[0].size(2) / H;
  STD_TORCH_CHECK(heads == 1 || read_list >= 0, "a grouped read side lists at a level: ", heads,
                  " query heads a write head read every window token of each with no list level");
  STD_TORCH_CHECK(reduce == kReduceF16 || reduce == kReduceF32,
                  "the cross-CTA sum is f16 pairs (0) or fp32 (1), got ", reduce);
  STD_TORCH_CHECK(heads * L * dv < (int64_t{1} << 32),
                  "a write head's num rows index in 32-bit values (the drain's offsets): ", heads,
                  " query heads x ", L, " tokens x ", dv, " channels");
  //: THE ARM'S LIST GROUPS, its plan's (`plan.md#list-groups`): the readout's digits a read union, the fold's lists a
  //: write union, the facts pass's groups the lists were ranked in
  STD_TORCH_CHECK(key.size() == 6 && key[0] >= 16 && key[4] >= 32, "the arm key is ",
                  CarryArmSet::key, ", its rows from 16 and channels from 32; got ", key.size(),
                  " fields");
  const int64_t arm_lists = kStateValues / (key[0] * key[4]),
                read_group = key[5] ? arm_lists : key[0], write_group = arm_lists;
  p.write = side_of(write_rows, write_list, write_operand, widths, extents, B, L, H, W, write_group,
                    "the write side");
  p.read = side_of(read_rows, read_list, read_operand, widths, extents, B, L, H * heads, W,
                   read_group, "the read side");
  p.values = Plane{static_cast<const uint8_t*>(values.const_data_ptr()), values.stride(0) * 2,
                   values.stride(1) * 2, values.stride(2) * 2};
  STD_TORCH_CHECK(p.values.st > 0 && p.values.st < (int64_t{1} << 31) / W,
                  "values' token stride is ", p.values.st, " bytes; a window's offsets are 32-bit");
  p.write_lists = lists_of(write_lists, BH, windows, write_list < 0 ? 0 : widths[write_list], 1, W,
                           write_group, "the write lists");
  p.read_lists = lists_of(read_lists, BH, windows, read_list < 0 ? 0 : widths[read_list], heads, W,
                          read_group, "the read lists");
  STD_TORCH_CHECK(
      heads <= window_heads(W) && heads * W <= 255 * kListBound, "a write head has at most ",
      window_heads(W), " query heads at W = ", W,
      " (the union slot's head field), its read union at most 255 boundaries (the plan's chunk"
      " fields); got ",
      heads);
  p.tables.heads = static_cast<int>(heads);
  p.tables.plan_words = plan_words(static_cast<int>(heads), W);
  if (write_list >= 0) {
    STD_TORCH_CHECK(
        write_exp.is_cuda() && write_exp.is_contiguous() && write_exp.scalar_type() == Dtype::Short
            && write_exp.sizes() == Shape({BH, windows, widths[write_list], 2}),
        "write_exp is the facts pass's contiguous CUDA int16 [BH, windows, width, 2] list "
        "exponents at the write list level, got ",
        shape(write_exp.sizes()));
    p.write_exp = write_exp.const_data_ptr<int16_t>();
  } else {
    STD_TORCH_CHECK(
        factors.is_cuda() && factors.is_contiguous() && factors.scalar_type() == Dtype::Float
            && factors.sizes() == Shape({BH, L, 2}),
        "factors is the facts pass's contiguous CUDA fp32 [BH, L, 2] (F, m) a token, got ",
        shape(factors.sizes()));
    STD_TORCH_CHECK(
        factor_shift.is_cuda() && factor_shift.is_contiguous()
            && factor_shift.scalar_type() == Dtype::Short
            && factor_shift.sizes() == Shape({BH, windows, 2}),
        "factor_shift is the facts pass's contiguous CUDA int16 [BH, windows, 2] window "
        "shifts of (m, F), got ",
        shape(factor_shift.sizes()));
    p.factors = reinterpret_cast<const float2*>(factors.const_data_ptr<float>());
    p.factor_shift = factor_shift.const_data_ptr<int16_t>();
  }
  STD_TORCH_CHECK(
      scale.is_cuda() && scale.is_contiguous() && scale.scalar_type() == Dtype::Char
          && scale.sizes() == Shape({BH * heads, windows * W}),
      "scale is the contiguous CUDA int8 [B H, windows x W] den-pass exponent bytes a query head, got ",
      shape(scale.sizes()));
  if (reduce == kReduceF16) {
    STD_TORCH_CHECK(
        num.is_cuda() && num.is_contiguous() && num.scalar_type() == Dtype::Int
            && num.sizes() == Shape({BH * heads, L, dv / 2}),
        "num at reduce f16 is the contiguous CUDA int32 [B H, L, d_v / 2] f16 pairs a query head, got ",
        shape(num.sizes()));
  } else {
    STD_TORCH_CHECK(
        num.is_cuda() && num.is_contiguous() && num.scalar_type() == Dtype::Float
            && num.sizes() == Shape({BH * heads, L, dv}),
        "num at reduce fp32 is the contiguous CUDA fp32 [B H, L, d_v] a query head, got ",
        shape(num.sizes()));
  }
  STD_TORCH_CHECK(den.is_cuda() && den.is_contiguous() && den.scalar_type() == Dtype::Float
                      && den.sizes() == Shape({BH * heads, L}),
                  "den is the contiguous CUDA fp32 [B H, L] a query head the caller zeroed, got ",
                  shape(den.sizes()));
  STD_TORCH_CHECK(check.is_cuda() && check.scalar_type() == Dtype::Float && check.numel() == 1,
                  "check is one CUDA fp32 value");
  p.scale = scale.const_data_ptr<int8_t>();
  p.reduce = static_cast<int>(reduce);
  STD_TORCH_CHECK(accumulate >= -1 && accumulate <= 1,
                  "the readout accumulates in fp32 (0), fp16 (1) or the device's rule (-1), got ",
                  accumulate);
  p.readout_f16 = accumulate < 0 ? (readout_f16(device) ? 1 : 0) : static_cast<int>(accumulate);
  STD_TORCH_CHECK(
      forced >= -1 && forced < (1 << 14) && (forced < 0 || (forced & 3) != 3),
      "the forced plan form is -1 (none) or mode | read chunk << 2 | fold chunk << 10 (mode 0 "
      "the rule's, 1 whole, 2 halves), got ",
      forced);
  p.forced = static_cast<int>(forced);
  p.num = static_cast<uint32_t*>(num.mutable_data_ptr());
  p.den = den.mutable_data_ptr<float>();
  p.check = check.mutable_data_ptr<float>();
  p.probe = g_probe;
  p.probe_den = g_probe_den;
  if (p.probe != nullptr)
    STD_TORCH_CHECK(g_probe_values == BH * leaves * boxes / (dv / channels) * dv,
                    "the bound probe holds ", g_probe_values, " values, the call's state ",
                    BH * leaves * boxes / (dv / channels) * dv);
  if (p.probe_den != nullptr)
    STD_TORCH_CHECK(g_probe_den_values == BH * leaves * boxes / (dv / channels),
                    "the bound den probe holds ", g_probe_den_values,
                    " values, the call's den state ", BH * leaves * boxes / (dv / channels));
  //: THE PLANS, a window's a (batch-head, box class), the plan pass's to write and the carry's to stage
  const int64_t classes = (p.read_lists.groups > 0 ? p.read_lists.groups : 1)
                          * (p.write_lists.groups > 0 ? p.write_lists.groups : 1);
  const int64_t words = p.tables.plan_words;
  Tensor plans = empty_cuda({BH, windows, classes, words}, Dtype::Int, device);
  p.plans = reinterpret_cast<uint32_t*>(plans.mutable_data_ptr<int32_t>());
  p.ledger = g_ledger;
  if (p.ledger != nullptr)
    STD_TORCH_CHECK(g_ledger_ctas >= boxes * BH, "the bound phase ledger covers ", g_ledger_ctas,
                    " CTAs, the launch needs ", boxes * BH);

  const CarryKey k{(int)key[0], (int)key[1], (int)key[2], (int)key[3], (int)key[4], (int)key[5]};
  STD_TORCH_CHECK(k.channels == channels && k.fold_factors == p.write.operands
                      && k.read_factors == p.read.operands,
                  "the arm key's channels and factors are the carve's: ", channels, ", ",
                  p.write.operands, ", ", p.read.operands);
  STD_TORCH_CHECK(!(accumulate == 1 && k.read_factors == 2),
                  "a dense read side's readout accumulates fp32 alone: its fp16 form is not built");
  //: the hand-off's form is the sides' levels: the fold's lists (a dense fold's coefficients) the readout's digits
  const int64_t fold_outer = write_list >= 0 ? write_list
                                             : (write_operand.size() > 1 ? write_operand[1] : -1),
                read_outer =
                    read_list >= 0 ? read_list : (read_operand.size() > 1 ? read_operand[1] : -1);
  STD_TORCH_CHECK(k.same_split == (fold_outer == read_outer ? 1 : 0),
                  "the arm key's hand-off form (same_split ", k.same_split,
                  ") is not the carve's: the fold's lists at level ", fold_outer,
                  ", the readout's digits at level ", read_outer);
  if (g_capture != nullptr) {
    const CarryCapture c{(int64_t)sizeof(CarryCapture), boxes, BH, p};
    std::memcpy(g_capture, &c, sizeof c);
  }
  const auto stream = current_stream();
  cudaError_t status = cudaSuccess;
  bool launched = false;
  arm_switch<CarryArmSet>(k, [&](auto A) {
    using Arm = decltype(A);
#define ROLA_CARRY_ARM_CALL(I_, FR_, FF_, RR_, RF_, CH_, SS_)                   \
  if (Arm::index == (I_)) {                                                     \
    launched = true;                                                            \
    status = carry_arm_forward_##I_(p, (int)boxes, (int)BH, stream);            \
  }
    ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_CALL)
#undef ROLA_CARRY_ARM_CALL
  });
  STD_TORCH_CHECK(launched, "the carry arm set resolved a member that carries no launch");
  STD_TORCH_CHECK(status == cudaSuccess, "the carry launch failed: ", cudaGetErrorString(status));
  return plans;
}

//: fp16 where the device's fp32-accumulate MMA is slower than its fp16-accumulate one, by its rates; fp32 where they
//: are equal or no rates are tabulated (its SM's full rate). -- see docs/internals/carry/carry_api.md#carry-readout-accumulate
int64_t carry_readout_accumulate(int64_t device) {
  check_arch_table();
  return readout_f16(static_cast<int>(device)) ? 1 : 0;
}

void carry_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger) {
    g_ledger = nullptr;
    g_ledger_ctas = 0;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == kWarps && ledger->size(2) == kCarryPhases
                      && ledger->is_contiguous(),
                  "the phase ledger is a contiguous CUDA int64 tensor [ctas][", kWarps, "][",
                  kCarryPhases, "]");
  g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  g_ledger_ctas = ledger->size(0);
}

void carry_probe_bind(const std::optional<Tensor>& probe, const std::optional<Tensor>& den) {
  g_probe = nullptr;
  g_probe_values = 0;
  g_probe_den = nullptr;
  g_probe_den_values = 0;
  if (probe) {
    STD_TORCH_CHECK(
        probe->is_cuda() && probe->scalar_type() == Dtype::Float && probe->is_contiguous(),
        "the probe is a contiguous CUDA fp32 tensor [BH, N, d_v]");
    g_probe = probe->mutable_data_ptr<float>();
    g_probe_values = probe->numel();
  }
  if (den) {
    STD_TORCH_CHECK(den->is_cuda() && den->scalar_type() == Dtype::Float && den->is_contiguous(),
                    "the den probe is a contiguous CUDA fp32 tensor [BH, N]");
    g_probe_den = den->mutable_data_ptr<float>();
    g_probe_den_values = den->numel();
  }
}

void carry_capture_bind(const std::optional<Tensor>& out) {
  if (!out) {
    g_capture = nullptr;
    return;
  }
  STD_TORCH_CHECK(!out->is_cuda() && out->scalar_type() == Dtype::Byte && out->is_contiguous()
                      && out->numel() == (int64_t)sizeof(CarryCapture),
                  "the capture is a contiguous host u8 tensor of ", sizeof(CarryCapture), " bytes");
  g_capture = static_cast<uint8_t*>(out->mutable_data_ptr());
}

int64_t carry_capture_bytes() { return (int64_t)sizeof(CarryCapture); }

namespace stamp {

//: THE DEVICE-SIDE FACT at build `Bt::build`: the only answer a stale binary cannot produce.
template <class Bt>
__global__ void carry_stamp_kernel(int* out) {
  out[0] = Bt::traits.window;
  out[1] = denseref::arch::kCapsDigest;
  out[2] = kStateValues;
  out[3] = kListBound;
}

}  // namespace stamp

int64_t carry_build_stamp() {
  check_arch_table();
  auto buf = empty_cuda({4}, Dtype::Int);
  build_switch(current_card(), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB))
      stamp::carry_stamp_kernel<::denseref::arch::BuildTag<kB>>
          <<<1, 1, 0, current_stream()>>>(buf.mutable_data_ptr<int32_t>());
  });
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the carry build stamp did not run");
  const auto host = to_cpu(buf);
  const int32_t* f = host.const_data_ptr<int32_t>();
  //: THE FOUR FACTS AS ONE NUMBER: a mixed radix wide enough that no field carries into its neighbour.
  return ((((int64_t)f[0] * 1000003 + f[1]) * 1000003) + f[2]) * 1024 + f[3];
}

std::vector<std::vector<int64_t>> carry_census() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_CARRY_ARM_CENSUS(I_, FR_, FF_, RR_, RF_, CH_, SS_)                                  \
  {                                                                                                   \
    const ArmCensus c = carry_arm_census_##I_();                                                      \
    out.push_back({(int64_t)(I_), FR_, FF_, RR_, RF_, CH_, SS_, c.regs, c.local_bytes,         \
                   c.form.smem_bytes, c.max_threads, c.binary_version, c.form.area,                   \
                   c.form.read_row, c.form.write_row, c.form.read_list_group,                         \
                   c.form.write_list_group});                                                         \
  }
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_CENSUS)
#undef ROLA_CARRY_ARM_CENSUS
  return out;
}

std::vector<std::vector<int64_t>> carry_arms() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_CARRY_ARM_ROW(I_, FR_, FF_, RR_, RF_, CH_, SS_) out.push_back({FR_, FF_, RR_, RF_, CH_, SS_});
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_ROW)
#undef ROLA_CARRY_ARM_ROW
  return out;
}

}  // namespace rola::carry
