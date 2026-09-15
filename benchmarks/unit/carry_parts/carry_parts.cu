// benchmarks/unit/carry_parts/carry_parts.cu -- THE PART HARNESS'S DRIVERS: one kernel a
// component, each the kernel's own prologue plus that component alone, on the call the
// kernel would run (`derive_carry_call`), its products written out for the harness to check
// against a reference and its time against the model's floor (KERNEL_STANDARDS §19). A
// bench instrument, built by the harness from the repo's headers; never shipped.
// See docs/internals/carry/carry_kernel.md#part-harness
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "common/torch_seam.cuh"

#include "carry/carry_host.cuh"
#include "carry/carry_kernel.cuh"
#include "carry_calib.cuh"

namespace rola::carry::parts {

//: THE LAYOUT VARIANTS the kernel dispatches among, for a part to do the same.
template <class F>
__device__ __forceinline__ void run_head(bool rc, bool rs, bool wc, bool ws, F&& f) {
  using T = std::true_type;
  using Fa = std::false_type;
  if (!rc && !ws && !rs && !wc)
    f(Fa{}, Fa{}, Fa{}, Fa{});
  else if (rs && ws)
    f(T{}, T{}, T{}, T{});
  else if (rs) {
    if (wc)
      f(T{}, T{}, T{}, Fa{});
    else
      f(T{}, T{}, Fa{}, Fa{});
  } else if (ws) {
    if (rc)
      f(T{}, Fa{}, T{}, T{});
    else
      f(Fa{}, Fa{}, T{}, T{});
  } else if (rc && wc)
    f(T{}, Fa{}, T{}, Fa{});
  else if (rc)
    f(T{}, Fa{}, Fa{}, Fa{});
  else
    f(Fa{}, Fa{}, T{}, Fa{});
}

//: THE POOL'S PROLOGUE as `window_loop` runs it: each slot's barriers, and its zero row
//: (row `kPoolTok`) zeroed. A fill with no idle lane never writes that row, so a driver that
//: skips this reads leftover SMEM there.
template <class BP>
__device__ __forceinline__ void pool_prologue(const Smem<BP>& sm, int tid) {
  if (tid < BP::kPoolSlots) {
    ops::mbar_init(sm.full_bar(tid), BP::kThreads);
    ops::mbar_init(sm.empty_bar(tid), BP::kWarps);
  }
  constexpr int kZeroParts = BP::kVChunks + 4;
#pragma unroll 1
  for (int i = tid; i < BP::kPoolSlots * kZeroParts; i += BP::kThreads) {
    const int s = i / kZeroParts, part = i % kZeroParts;
    if (part < BP::kVChunks)
      ops::store_shared_vec16(sm.pool(s) + pool_v_off<BP>(BP::kPoolTok, part),
                              make_uint4(0u, 0u, 0u, 0u));
    else if (part < BP::kVChunks + 2)
      ops::store_shared_vec16(sm.pool(s) + pool_inner_off<BP>(BP::kPoolTok, part - BP::kVChunks),
                              make_uint4(0u, 0u, 0u, 0u));
    else if (part < BP::kVChunks + 4)
      ops::store_shared_vec16(
          sm.pool(s) + pool_outer_off<BP>(BP::kPoolTok, part - BP::kVChunks - 2),
          make_uint4(0u, 0u, 0u, 0u));
    if (part == 0) ops::store_shared_u32(sm.pool(s) + pool_gain_off<BP>(BP::kPoolTok), 0u);
  }
}

//: THE HEAD PART: every window's head, `reps` times over (the products are idempotent);
//: the last pass's read order, tile masks, live counts, warp words, union words and prefix
//: written to `[bh][owner][window][...]`.
template <int D, int DV, int WARPS>
__global__ __launch_bounds__(BoxPlan<D, DV, WARPS>::kThreads, 1) void head_part_kernel(
    __grid_constant__ const CarryParams p, uint16_t* out_order, uint16_t* out_tilemask,
    int32_t* out_live, uint32_t* out_warpwords, uint32_t* out_union, uint16_t* out_prefix,
    int reps) {
  using BP = BoxPlan<D, DV, WARPS>;
  constexpr int BC = BP::kBC;
  extern __shared__ __align__(16) char smem_raw[];
  const Smem<BP> sm{smem_raw};
  const int owner = (int)blockIdx.x, bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<BP::kWarps>(), lane = tid & 31;
  const SideLayout(&lay)[2] = p.lay;
  int abase[D];
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
  });
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  const SideLayout& lr = lay[rola::facts::kRead];
  const SideLayout& lw = lay[rola::facts::kWrite];
  const bool RC = lr.single_inner < 0 || lr.single_outer < 0, RS = lr.straddle >= 0;
  const bool WC = lw.single_inner < 0 || lw.single_outer < 0, WS = lw.straddle >= 0;
  PhaseClock pc(sm.ledger(warp), lane);
#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;
    const int nWords = (wLen + 31) / 32;
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kRead, abase, warp, lane, bh, t0, nWords);
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kWrite, abase, warp, lane, bh, t0, nWords);
    ops::stage_wait<0>();
    ops::rendezvous();
    int live[2] = {0, 0};
#pragma unroll 1
    for (int r = 0; r < reps; ++r) {
      run_head(RC, RS, WC, WS, [&](auto rc, auto rs, auto wc, auto ws) {
        head<BP, D, BC, decltype(rc)::value, decltype(rs)::value, decltype(wc)::value,
             decltype(ws)::value>(p, sm, lay, win & 1, wLen, warp, lane, live, pc);
      });
      ops::rendezvous();
    }
    const long cta = (long)bh * gridDim.x + owner;
    const long w = cta * nWindows + win;
    for (int t = tid; t < kWindow; t += BP::kThreads)
      out_order[w * kWindow + t] = sm.order(win & 1)[t];
    if (tid < BP::kTiles) out_tilemask[w * BP::kTiles + tid] = sm.tilemask(win & 1)[tid];
    if (tid < 2) out_live[w * 2 + tid] = live[tid];
    if (tid < BP::kWarps * BP::kRounds)
      out_warpwords[w * BP::kWarps * BP::kRounds + tid] =
          sm.warpwords(win & 1, tid / BP::kRounds)[tid % BP::kRounds];
    if (tid < BP::kRounds) out_union[w * BP::kRounds + tid] = sm.unionwords(win & 1)[tid];
    if (tid < BP::kRounds + 1) out_prefix[w * (BP::kRounds + 1) + tid] = sm.prefix(win & 1)[tid];
    ops::rendezvous();
  }
}

//: THE FILL PART: every window's head, then every pool chunk filled, finished and waited
//: in turn, the slots of the first `kOutOwners` owners copied out as
//: `[owner][window][chunk][kPoolSlotBytes]`; `reps` refills each chunk for timing.
constexpr int kOutOwners = 2;
constexpr int kMaxChunks = (kWindow + 31) / 32;

template <int D, int DV, int WARPS>
__global__ __launch_bounds__(BoxPlan<D, DV, WARPS>::kThreads, 1) void fill_part_kernel(
    __grid_constant__ const CarryParams p, uint8_t* out_slots, int reps) {
  using BP = BoxPlan<D, DV, WARPS>;
  constexpr int BC = BP::kBC;
  extern __shared__ __align__(16) char smem_raw[];
  const Smem<BP> sm{smem_raw};
  const int owner = (int)blockIdx.x, bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<BP::kWarps>(), lane = tid & 31;
  const SideLayout(&lay)[2] = p.lay;
  int abase[D];
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
  });
  const int2 wbases = run_bases<D>(lay[rola::facts::kWrite], abase);
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  const SideLayout& lr = lay[rola::facts::kRead];
  const SideLayout& lw = lay[rola::facts::kWrite];
  const bool RC = lr.single_inner < 0 || lr.single_outer < 0, RS = lr.straddle >= 0;
  const bool WC = lw.single_inner < 0 || lw.single_outer < 0, WS = lw.straddle >= 0;
  pool_prologue<BP>(sm, tid);
  ops::rendezvous();
  PhaseClock pc(sm.ledger(warp), lane);
  int ordinal = 0;
#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;
    const int nWords = (wLen + 31) / 32;
    const uint32_t rowbase = (uint32_t)bh * (uint32_t)p.L + (uint32_t)t0;
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kRead, abase, warp, lane, bh, t0, nWords);
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kWrite, abase, warp, lane, bh, t0, nWords);
    ops::stage_wait<0>();
    ops::rendezvous();
    int live[2] = {0, 0};
    run_head(RC, RS, WC, WS, [&](auto rc, auto rs, auto wc, auto ws) {
      head<BP, D, BC, decltype(rc)::value, decltype(rs)::value, decltype(wc)::value,
           decltype(ws)::value>(p, sm, lay, win & 1, wLen, warp, lane, live, pc);
    });
    ops::rendezvous();
    const int chunks = (live[rola::facts::kWrite] + BP::kPoolTok - 1) / BP::kPoolTok;
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
#pragma unroll 1
      for (int r = 0; r < reps; ++r) {
        const int s = ordinal % BP::kPoolSlots;
        if (ordinal >= BP::kPoolSlots)
          ops::mbar_wait(sm.empty_bar(s), (uint32_t)(ordinal / BP::kPoolSlots - 1) & 1u);
        pool_fill<BP, D, BC>(p, sm, wbases, win & 1, c, s, rowbase, warp, lane);
        ops::mbar_wait(sm.full_bar(s), (uint32_t)(ordinal / BP::kPoolSlots) & 1u);
        if (r == reps - 1 && owner < kOutOwners) {
          uint8_t* const o =
              out_slots + ((((long)owner * nWindows + win) * kMaxChunks + c)) * BP::kPoolSlotBytes;
          for (int b = tid * 16; b < BP::kPoolSlotBytes; b += BP::kThreads * 16)
            *reinterpret_cast<uint4*>(o + b) = ops::load_shared_v4(sm.pool(s) + (uint32_t)b);
        }
        __syncwarp();
        if (lane == 0) ops::mbar_arrive(sm.empty_bar(s));
        ++ordinal;
      }
    }
    ops::rendezvous();
  }
}

//: THE FOLD PART: every window's head, its first two chunks filled, then the fold `reps`
//: times over (each time refilling and re-walking the window: the state accumulates
//: `reps` times); the state written out as `[bh][owner][warp][lane][kDealt][kNT][4]` and
//: the masses as `[bh][owner][warp][lane][kDealt][4]`.
template <int D, int DV, int WARPS>
__global__ __launch_bounds__(BoxPlan<D, DV, WARPS>::kThreads, 1) void fold_part_kernel(
    __grid_constant__ const CarryParams p, float* out_state, float* out_mass, int reps) {
  using BP = BoxPlan<D, DV, WARPS>;
  constexpr int BC = BP::kBC;
  extern __shared__ __align__(16) char smem_raw[];
  const Smem<BP> sm{smem_raw};
  const int owner = (int)blockIdx.x, bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<BP::kWarps>(), lane = tid & 31;
  const SideLayout(&lay)[2] = p.lay;
  int abase[D];
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
  });
  const int2 wbases = run_bases<D>(lay[rola::facts::kWrite], abase);
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  const SideLayout& lr = lay[rola::facts::kRead];
  const SideLayout& lw = lay[rola::facts::kWrite];
  const bool RC = lr.single_inner < 0 || lr.single_outer < 0, RS = lr.straddle >= 0;
  const bool WC = lw.single_inner < 0 || lw.single_outer < 0, WS = lw.straddle >= 0;
  pool_prologue<BP>(sm, tid);
  ops::rendezvous();
  PhaseClock pc(sm.ledger(warp), lane);
  State<BP> st;
  rola::static_for<BP::kDealt>([&](auto Bi) {
    rola::static_for<BP::kNT>([&](auto Jc) {
      rola::static_for<4>([&](auto Kc) {
        st.c[decltype(Bi)::value][decltype(Jc)::value][decltype(Kc)::value] = 0.0f;
      });
    });
    rola::static_for<4>([&](auto Kc) { st.m[decltype(Bi)::value][decltype(Kc)::value] = 0.0f; });
  });
  PoolCursor<BP> cur{0u};
#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;
    const int nWords = (wLen + 31) / 32;
    const uint32_t rowbase = (uint32_t)bh * (uint32_t)p.L + (uint32_t)t0;
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kRead, abase, warp, lane, bh, t0, nWords);
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kWrite, abase, warp, lane, bh, t0, nWords);
    ops::stage_wait<0>();
    ops::rendezvous();
    int live[2] = {0, 0};
    run_head(RC, RS, WC, WS, [&](auto rc, auto rs, auto wc, auto ws) {
      head<BP, D, BC, decltype(rc)::value, decltype(rs)::value, decltype(wc)::value,
           decltype(ws)::value>(p, sm, lay, win & 1, wLen, warp, lane, live, pc);
    });
    ops::rendezvous();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) {
      fold<BP, D, BC, false, false>(p, sm, lay[rola::facts::kWrite], wbases, win & 1,
                                    live[rola::facts::kWrite], rowbase, warp, lane, st, cur, pc);
    }
    ops::rendezvous();
  }
  const long cta = (long)bh * gridDim.x + owner;
  float* const so =
      out_state + ((cta * BP::kWarps + warp) * 32 + lane) * (BP::kDealt * BP::kNT * 4);
  float* const mo = out_mass + ((cta * BP::kWarps + warp) * 32 + lane) * (BP::kDealt * 4);
  rola::static_for<BP::kDealt>([&](auto Bi) {
    constexpr int bi = decltype(Bi)::value;
    rola::static_for<BP::kNT>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      rola::static_for<4>([&](auto Kc) {
        constexpr int e = decltype(Kc)::value;
        so[(bi * BP::kNT + j) * 4 + e] = st.c[bi][j][e];
      });
    });
    rola::static_for<4>(
        [&](auto Kc) { mo[bi * 4 + decltype(Kc)::value] = st.m[bi][decltype(Kc)::value]; });
  });
}

//: the flagship arm is the part harness's shape; the arm set proper is the launch's.
constexpr int kD = 2, kDV = 64, kW = 8;
using FlagPlan = BoxPlan<kD, kDV, kW>;

std::vector<Tensor> head_part(const Tensor& read, const Tensor& write, const Tensor& gain,
                              const Tensor& v, Tensor& num, Tensor& den,
                              const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                              int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
                              const std::vector<int64_t>& schedule, const Tensor& liveness,
                              const Tensor& activity, int64_t reps) {
  CarryCall call = derive_carry_call(read, write, gain, v, num, den, widths, dv, page_bits,
                                     warps_per_cta, carve_order, schedule, liveness, activity,
                                     std::nullopt, std::nullopt, std::nullopt);
  STD_TORCH_CHECK((int)widths.size() == kD && dv == kDV && warps_per_cta == kW,
                  "the part harness is built for the flagship arm (D=2, DV=64, 8 warps)");
  const CarryParams& p = call.p;
  const int64_t BH = call.BH, owners = p.g.owners;
  const int64_t nWindows = (p.L + kWindow - 1) / kWindow;
  const int32_t device = read.get_device_index();
  Tensor order = empty_cuda({BH, owners, nWindows, kWindow}, Dtype::Short, device);
  Tensor tilemask = empty_cuda({BH, owners, nWindows, FlagPlan::kTiles}, Dtype::Short, device);
  Tensor live = empty_cuda({BH, owners, nWindows, 2}, Dtype::Int, device);
  Tensor warpwords = empty_cuda({BH, owners, nWindows, kW, FlagPlan::kRounds}, Dtype::Int, device);
  Tensor unionw = empty_cuda({BH, owners, nWindows, FlagPlan::kRounds}, Dtype::Int, device);
  Tensor prefix = empty_cuda({BH, owners, nWindows, FlagPlan::kRounds + 1}, Dtype::Short, device);
  auto* kern = head_part_kernel<kD, kDV, kW>;
  static bool attr = false;
  if (!attr) {
    STD_TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         FlagPlan::kSmemBytes)
                        == cudaSuccess,
                    "smem attribute");
    attr = true;
  }
  kern<<<dim3((unsigned)owners, (unsigned)BH), FlagPlan::kThreads, FlagPlan::kSmemBytes,
         current_stream()>>>(p, reinterpret_cast<uint16_t*>(order.mutable_data_ptr<int16_t>()),
                             reinterpret_cast<uint16_t*>(tilemask.mutable_data_ptr<int16_t>()),
                             live.mutable_data_ptr<int32_t>(),
                             reinterpret_cast<uint32_t*>(warpwords.mutable_data_ptr<int32_t>()),
                             reinterpret_cast<uint32_t*>(unionw.mutable_data_ptr<int32_t>()),
                             reinterpret_cast<uint16_t*>(prefix.mutable_data_ptr<int16_t>()),
                             (int)reps);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the head part did not launch");
  return {order, tilemask, live, warpwords, unionw, prefix};
}

Tensor fill_part(const Tensor& read, const Tensor& write, const Tensor& gain, const Tensor& v,
                 Tensor& num, Tensor& den, const std::vector<int64_t>& widths, int64_t dv,
                 int64_t page_bits, int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
                 const std::vector<int64_t>& schedule, const Tensor& liveness,
                 const Tensor& activity, int64_t reps) {
  CarryCall call = derive_carry_call(read, write, gain, v, num, den, widths, dv, page_bits,
                                     warps_per_cta, carve_order, schedule, liveness, activity,
                                     std::nullopt, std::nullopt, std::nullopt);
  STD_TORCH_CHECK((int)widths.size() == kD && dv == kDV && warps_per_cta == kW,
                  "the part harness is built for the flagship arm (D=2, DV=64, 8 warps)");
  const CarryParams& p = call.p;
  const int64_t BH = call.BH, owners = p.g.owners;
  const int64_t nWindows = (p.L + kWindow - 1) / kWindow;
  STD_TORCH_CHECK(BH == 1, "the fill part copies out owners of one batch row");
  Tensor slots = zeros_cuda({kOutOwners, nWindows, kMaxChunks, FlagPlan::kPoolSlotBytes},
                            Dtype::Byte, read.get_device_index());
  auto* kern = fill_part_kernel<kD, kDV, kW>;
  static bool attr = false;
  if (!attr) {
    STD_TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         FlagPlan::kSmemBytes)
                        == cudaSuccess,
                    "smem attribute");
    attr = true;
  }
  kern<<<dim3((unsigned)owners, (unsigned)BH), FlagPlan::kThreads, FlagPlan::kSmemBytes,
         current_stream()>>>(p, slots.mutable_data_ptr<uint8_t>(), (int)reps);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the fill part did not launch");
  return slots;
}

std::vector<Tensor> fold_part(const Tensor& read, const Tensor& write, const Tensor& gain,
                              const Tensor& v, Tensor& num, Tensor& den,
                              const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                              int64_t warps_per_cta, const std::vector<int64_t>& carve_order,
                              const std::vector<int64_t>& schedule, const Tensor& liveness,
                              const Tensor& activity, int64_t reps) {
  CarryCall call = derive_carry_call(read, write, gain, v, num, den, widths, dv, page_bits,
                                     warps_per_cta, carve_order, schedule, liveness, activity,
                                     std::nullopt, std::nullopt, std::nullopt);
  STD_TORCH_CHECK((int)widths.size() == kD && dv == kDV && warps_per_cta == kW,
                  "the part harness is built for the flagship arm (D=2, DV=64, 8 warps)");
  const CarryParams& p = call.p;
  const int64_t BH = call.BH, owners = p.g.owners;
  Tensor state = zeros_cuda({BH, owners, kW, 32, FlagPlan::kDealt, FlagPlan::kNT, 4}, Dtype::Float,
                            read.get_device_index());
  Tensor mass =
      zeros_cuda({BH, owners, kW, 32, FlagPlan::kDealt, 4}, Dtype::Float, read.get_device_index());
  auto* kern = fold_part_kernel<kD, kDV, kW>;
  static bool attr = false;
  if (!attr) {
    STD_TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         FlagPlan::kSmemBytes)
                        == cudaSuccess,
                    "smem attribute");
    attr = true;
  }
  kern<<<dim3((unsigned)owners, (unsigned)BH), FlagPlan::kThreads, FlagPlan::kSmemBytes,
         current_stream()>>>(p, state.mutable_data_ptr<float>(), mass.mutable_data_ptr<float>(),
                             (int)reps);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the fold part did not launch");
  return {state, mass};
}

}  // namespace rola::carry::parts

namespace {

std::vector<int64_t> pool_geometry() {
  using P = rola::carry::parts::FlagPlan;
  return {P::kPoolTok,         P::kPoolRows,    P::kPoolSlotBytes,
          P::kVRowBytes,       P::kPoolVOffset, P::kPoolInnerOffset,
          P::kPoolOuterOffset, P::kVChunks,     P::kPoolGainOffset};
}

int64_t smem_bytes() { return (int64_t)rola::carry::parts::FlagPlan::kSmemBytes; }

}  // namespace

//: THE PART HARNESS'S OPERATORS (`torch.ops.rola_parts`), through the stable ABI like the extension's own.
STABLE_TORCH_LIBRARY(rola_parts, m) {
  m.def(
      "head_part(Tensor read, Tensor write, Tensor gain, Tensor v, Tensor(a!) num, Tensor(b!) den, int[] widths, "
      "int dv, int page_bits, int warps_per_cta, int[] carve_order, int[] schedule, Tensor liveness, Tensor activity, "
      "int reps) -> Tensor[]");
  m.def(
      "fill_part(Tensor read, Tensor write, Tensor gain, Tensor v, Tensor(a!) num, Tensor(b!) den, int[] widths, "
      "int dv, int page_bits, int warps_per_cta, int[] carve_order, int[] schedule, Tensor liveness, Tensor activity, "
      "int reps) -> Tensor");
  m.def(
      "fold_part(Tensor read, Tensor write, Tensor gain, Tensor v, Tensor(a!) num, Tensor(b!) den, int[] widths, "
      "int dv, int page_bits, int warps_per_cta, int[] carve_order, int[] schedule, Tensor liveness, Tensor activity, "
      "int reps) -> Tensor[]");
  m.def("pool_geometry() -> int[]");
  m.def("smem_bytes() -> int");
  m.def(
      "calibrate(int warps, int mode, int burst, int iters, int owners, Tensor(a!) out, Tensor src) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(rola_parts, CompositeExplicitAutograd, m) {
  m.impl("head_part", TORCH_BOX(&rola::carry::parts::head_part));
  m.impl("fill_part", TORCH_BOX(&rola::carry::parts::fill_part));
  m.impl("fold_part", TORCH_BOX(&rola::carry::parts::fold_part));
  m.impl("pool_geometry", TORCH_BOX(&pool_geometry));
  m.impl("smem_bytes", TORCH_BOX(&smem_bytes));
  m.impl("calibrate", TORCH_BOX(&rola::carry::calib::calibrate));
}

extern "C" PyObject* PyInit__C_parts(void) {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                               "_C_parts",
                               "the part harness's operators, torch.ops.rola_parts",
                               -1,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
  return PyModule_Create(&module);
}
