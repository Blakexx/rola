// measure/lab/experiments/fold_alone.cu -- QUESTION: what does the carry's fold phase cost a warp-window ALONE --
// production's `fold_chunk` over lists drawn as nl64k-alt-k4's are, nothing else in the window -- against its tensor
// floor and against the composed kernel's fold phase? METHOD: one CTA an SM, the arm-0 plan's tenants; each window a
// box's 32 lists of Binomial(16, 1/2) entries over a 216-slot union (the carve model's alt-k4 steps: 40% of the lists a
// 16-token step, the rest an 8-token one), ~45% of the lists rescaled; a warp writes its own lists' entries and scales,
// then (as the carry runs it) a rendezvous, then the timed fold. see docs/lab/fold_alone.md
#include <cstdio>
#include <vector>

#include "carry/ampere/carry_kernel.cuh"
#include "common/arm.cuh"
#include "lab.cuh"

namespace fold_alone {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm, lab::kLabBuild>;
constexpr int kWindows = 64, kSlots = 216, kMovedOf256 = 115;

//: the draw's hash (both sides compute it: the kernel its lists, the host the floor)
__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t draw(int block, int w, int list) {
  return mix(static_cast<uint32_t>(block) * 7919u + static_cast<uint32_t>(w) * 104729u
             + static_cast<uint32_t>(list));
}

//: THE TRACE (MODE 3): each mark's cycles since the last, summed a warp over the windows, lane 0's
constexpr int kMarks = 9;

struct Trace {
  long long (&sums)[kMarks];
  long long& last;

  __device__ __forceinline__ void operator()(int m) const {
    const long long now = clock64();
    sums[m] += now - last;
    last = now;
  }
};

//: THE TIMELINE (MODE 6): each lap's absolute clock, block 0's warps, windows `kTimelineFrom ..` (after a warm-up)
constexpr int kTimelineFrom = 8, kTimelineWindows = 8;

struct Stamps {
  long long (&at)[kMarks];

  __device__ __forceinline__ void operator()(int m) const { at[m] = clock64(); }
};

//: MODE 0 as the carry runs it (a rendezvous before each window's fold); 1 the windows back to back; 2 no list
//: rescaled; 3 MODE 0 with the fold's laps traced (its own reading: a clock read is a point ptxas does not schedule
//: across); 4 MODE 0 on conflict-free slots (entry `e` of a list on a slot of residue `e mod 8`, the coloring's best
//: case); 5 MODE 4 back to back; 6 MODE 0 with each lap's clock stamped (the timeline); 7 MODE 0 timed from one
//: fold's end to the next's, so work moved out of the fold's own span still shows -- each with the den column on every
//: list (a one-box carry's); 8 MODE 0 with it on half the lists (each of two boxes'); 9 MODE 0 with none
template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void fold(long long* t, int* out, long long* marks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  const ops::SmemAddr rows = ops::smem_addr(smem + P::kAreaAt);
  const ops::SmemAddr ents = rows + static_cast<uint32_t>(rc::fold_ents<P>(kSlots));
  //: the union's rows, small f16 values in every slot's slices and values
  for (int i = tid; i < kSlots * P::kWriteRow / 4; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kAreaAt)[i] =
        0x24002400u ^ (mix(static_cast<uint32_t>(i)) & 0x03ff03ffu);
  const rc::Window ww{P::kWindow, 0, 0, 0, P::kLists, true};
  rc::State<P> st;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) (&st.acc[0][0][0][0])[k] = 0.0f;
  __syncthreads();

  long long busy = 0, whole = 0, end = 0;
  //: the trace's sums, registers (constant marks, every call inlined)
  long long sums[kMarks] = {}, last = 0;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: lane `l` list `l`'s count and its place (the counts before it), as `chunk_runs` gives them
    const int n = __popc(draw(blockIdx.x, w, lane) & 0xffffu);
    int to = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, to, o);
      if (lane >= o) to += v;
    }
    const rc::Run run{0, n, to - n};
    //: the warp's own lists' entries (a slot and c = ~1) and scales
    rola::static_for<P::kFoldLists>([&](auto J) {
      const int list = rc::fold_list(warp, J);
      const int nl = __shfl_sync(0xffffffffu, run.n, list),
                tl = __shfl_sync(0xffffffffu, run.to, list);
      for (int e = lane; e < nl; e += 32) {
        const uint32_t h = mix(draw(blockIdx.x, w, list) + 977u * static_cast<uint32_t>(e));
        const uint32_t slot =
            MODE >= 4 ? 8u * (h % (kSlots / 8)) + static_cast<uint32_t>(e & 7) : h % kSlots;
        reinterpret_cast<uint2*>(smem + (ents - ops::smem_addr(smem)))[tl + e] =
            make_uint2(slot, (0x3c00u ^ ((h >> 20) & 0xffu)) * 0x10001u);
      }
      if (lane == 0) {
        const bool moved =
            MODE != 2 && (mix(draw(blockIdx.x, w, list) ^ 0x5bd1e995u) & 0xffu) < kMovedOf256;
        reinterpret_cast<rc::ListScale*>(smem + P::kScale)[list] =
            rc::ListScale{moved ? 0.5f : 1.0f, rc::pow2_half2(0)};
      }
    });
    __syncwarp();
    if constexpr (MODE != 1 && MODE != 5) ops::rendezvous();
    const uint32_t dens = MODE == 8 ? 0x3u : MODE == 9 ? 0u : 0xfu;
    const long long t0 = clock64();
    if constexpr (MODE == 6) {
      long long at[kMarks];
      rc::fold_chunk<P>(st, true, smem, ww, w, run, 0, kSlots, rows, ents, warp, lane, dens,
                        Stamps{at});
      const int k = w - kTimelineFrom;
      if (blockIdx.x == 0 && lane == 0 && k >= 0 && k < kTimelineWindows) {
        long long* row = marks + (k * rc::kWarps + warp) * (kMarks + 1);
        row[0] = t0;
#pragma unroll
        for (int m = 0; m < kMarks; ++m) row[1 + m] = at[m];
      }
    } else if constexpr (MODE == 3) {
      last = t0;
      rc::fold_chunk<P>(st, true, smem, ww, w, run, 0, kSlots, rows, ents, warp, lane, dens,
                        Trace{sums, last});
    } else {
      rc::fold_chunk<P>(st, true, smem, ww, w, run, 0, kSlots, rows, ents, warp, lane, dens);
    }
    const long long t1 = clock64();
    busy += t1 - t0;
    if (w > 0) whole += t1 - end;
    end = t1;
  }
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = MODE == 7 ? whole : busy;
  if (MODE == 3 && lane == 0)
#pragma unroll
    for (int m = 0; m < kMarks; ++m) marks[(blockIdx.x * rc::kWarps + warp) * kMarks + m] = sums[m];
  uint32_t fold = 0;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) fold ^= lab::bits((&st.acc[0][0][0][0])[k]);
  lab::sink(out, fold);
}

//: THE TENSOR FLOOR a warp-window: each scheduler's two warps' MMA cycles (a 16-token step 8 x 32, an 8-token one
//: 8 x 16), the mean over the schedulers and the busiest's
struct Floor {
  double mean = 0, busiest = 0;
};

Floor floor_of(int blocks) {
  Floor f;
  for (int b = 0; b < blocks; ++b)
    for (int w = 0; w < kWindows; ++w) {
      double top = 0;
      for (int s = 0; s < 4; ++s) {
        double c = 0;
        for (int warp : {s, s + 4})
          for (int j = 0; j < P::kFoldLists; ++j) {
            const int n = __builtin_popcount(draw(b, w, warp + rc::kWarps * j) & 0xffffu);
            c += n > 8 ? 256.0 : n > 0 ? 128.0 : 0.0;
          }
        f.mean += c / 4;
        top = c > top ? c : top;
      }
      f.busiest += top;
    }
  f.mean /= double(blocks) * kWindows;
  f.busiest /= double(blocks) * kWindows;
  return f;
}

long long* g_marks = nullptr;

template <int MODE>
void variant(const lab::Bench& b, const char* label) {
  const std::string name = lab::kernel("fold", {MODE});
  b.expect(name, "HMMA.16816.F32", ">=", 8);
  b.expect(name, "HMMA.1688.F32", ">=", 8);
  b.exempt(name, "an isolation of the production fold; its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  LAB_CHECK(
      cudaFuncSetAttribute(fold<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::kSmemBytes));
  const lab::Samples s = b.variant(
      [&] { fold<MODE><<<b.sms, rc::kThreads, P::kSmemBytes>>>(b.times, b.sinks, g_marks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(label, "a timing isolation; the fold's arithmetic is the oracle's (test_carry_fold)");
  b.record(label, "cycles a warp-window", s);
  if constexpr (MODE == 6) {
    //: per scheduler (warps s, s + 4), each list's steps' span from the rendezvous's release (the earlier warp's start),
    //: averaged over the windows; and how much of the two warps' step spans overlap
    std::vector<long long> h(static_cast<size_t>(kTimelineWindows) * rc::kWarps * (kMarks + 1));
    LAB_CHECK(cudaMemcpy(h.data(), g_marks, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    const auto at = [&](int k, int warp, int m) {
      return h[(k * rc::kWarps + warp) * (kMarks + 1) + m];
    };
    for (int sch = 0; sch < 4; ++sch) {
      double span[2][kMarks + 1] = {}, overlap = 0, busy = 0;
      for (int k = 0; k < kTimelineWindows; ++k) {
        const long long rel = std::min(at(k, sch, 0), at(k, sch + 4, 0));
        for (int x = 0; x < 2; ++x)
          for (int m = 0; m <= kMarks; ++m)
            span[x][m] += double(at(k, sch + 4 * x, m) - rel) / kTimelineWindows;
        for (int j = 0; j < P::kFoldLists; ++j)
          for (int jj = 0; jj < P::kFoldLists; ++jj) {
            const long long a0 = at(k, sch, 2 + 2 * j), a1 = at(k, sch, 3 + 2 * j);
            const long long b0 = at(k, sch + 4, 2 + 2 * jj), b1 = at(k, sch + 4, 3 + 2 * jj);
            overlap +=
                double(std::max(0LL, std::min(a1, b1) - std::max(a0, b0))) / kTimelineWindows;
          }
        for (int x = 0; x < 2; ++x)
          for (int j = 0; j < P::kFoldLists; ++j)
            busy += double(at(k, sch + 4 * x, 3 + 2 * j) - at(k, sch + 4 * x, 2 + 2 * j))
                    / kTimelineWindows;
      }
      std::printf(
          "{\"lab\": \"note\", \"text\": \"scheduler %d (warps %d, %d), cycles from the release: ",
          sch, sch, sch + 4);
      for (int x = 0; x < 2; ++x) {
        std::printf("warp %d: start %.0f, lanes %.0f", sch + 4 * x, span[x][0], span[x][1]);
        for (int j = 0; j < P::kFoldLists; ++j)
          std::printf(", list %d %.0f-%.0f", j, span[x][2 + 2 * j], span[x][3 + 2 * j]);
        std::printf("; ");
      }
      std::printf("the two warps' step spans overlap %.0f of %.0f\"}\n", overlap, busy / 2);
    }
  }
  if constexpr (MODE == 3) {
    std::vector<long long> h(static_cast<size_t>(b.sms) * rc::kWarps * kMarks);
    LAB_CHECK(cudaMemcpy(h.data(), g_marks, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    const char* names[kMarks] = {"lists' first lanes", "list 0 guard", "list 0 steps",
                                 "list 1 guard",       "list 1 steps", "list 2 guard",
                                 "list 2 steps",       "list 3 guard", "list 3 steps"};
    for (int m = 0; m < kMarks; ++m) {
      double sum = 0;
      for (size_t i = 0; i < static_cast<size_t>(b.sms) * rc::kWarps; ++i)
        sum += double(h[i * kMarks + m]);
      lab::Samples one;
      one.v.push_back(sum / (double(b.sms) * rc::kWarps * kWindows));
      const std::string nm = std::string("trace: ") + names[m];
      b.unchecked(nm, "a lap of the traced variant");
      b.record(nm, "cycles a warp-window", one);
    }
  }
}

}  // namespace fold_alone

int main(int argc, char** argv) {
  using namespace fold_alone;
  lab::Bench b(argc, argv);
  if (!b.expect_only)
    LAB_CHECK(cudaMalloc(&g_marks, sizeof(long long) * b.sms * rc::kWarps * kMarks));
  fold_alone::variant<0>(b, "composed: a rendezvous a window");
  fold_alone::variant<1>(b, "windows back to back");
  fold_alone::variant<2>(b, "no list rescaled");
  fold_alone::variant<3>(b, "composed, traced");
  fold_alone::variant<4>(b, "composed, conflict-free slots");
  fold_alone::variant<5>(b, "back to back, conflict-free slots");
  fold_alone::variant<6>(b, "composed, timeline");
  fold_alone::variant<7>(b,
                         "composed, the whole window (the harness's writes and the wait included)");
  fold_alone::variant<8>(b, "composed, the den column on half the lists (two channel boxes)");
  fold_alone::variant<9>(b, "composed, no den column (a box carrying none)");
  if (!b.expect_only) {
    const Floor f = floor_of(b.sms);
    b.unchecked("tensor floor (mean scheduler)", "computed on the host from the same draw");
    b.unchecked("tensor floor (busiest scheduler)", "computed on the host from the same draw");
    std::printf(
        "{\"lab\": \"result\", \"name\": \"tensor floor (mean scheduler)\", \"unit\": \"cycles a warp-window\", "
        "\"samples\": [%.1f]}\n",
        f.mean);
    std::printf(
        "{\"lab\": \"result\", \"name\": \"tensor floor (busiest scheduler)\", \"unit\": \"cycles a "
        "warp-window\", \"samples\": [%.1f]}\n",
        f.busiest);
  }
  return 0;
}
