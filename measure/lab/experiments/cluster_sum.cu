// measure/lab/experiments/cluster_sum.cu -- QUESTION: a token's output is a sum of partials that the CTAs of a
// cluster hold; what does a token cost a CTA when the cluster sums them in one CTA's shared memory before one drain
// to global memory, by each of five forms, beside every CTA draining its own -- and how much of it is the round's
// fixed cost? And does one collector hold a cluster's partials, or must the collecting be spread? METHOD: clusters
// of 8 CTAs, 4 warps each; a round is a partial of TOKENS x 64 channels a CTA, from registers, at 16 and at 64
// tokens: two round sizes separate a form's cost a round from its cost a token. `direct`: every CTA adds its partial
// to global memory. `slots`: every CTA's warps store its partial into its own slot of the collector's shared
// memory; the collector adds a token's 8 slots in registers and adds the sum to global memory. `atomic`: the warps
// add fp32 into the collector's one panel. `bulk slots`: the slot form with the partial staged in the CTA's own
// shared memory and sent by one bulk copy a collector. `bulk integers`: staged as 64-bit integers and sent by one
// bulk reduction into the collector's one panel; `warp integers`: the warps' own 64-bit adds into it. `hand-over`:
// the arrival forms' barriers with no partial. One collector (rank 0) or eight (a token's collector its index
// modulo 8). Barriers a round order them (the bulk forms' bytes complete the collector's), the cluster's own its
// start and end; a thread that wrote into a mate, or read and cleared a panel, fences at cluster scope first.
// see docs/internals/tools/lab.md#cluster-sum
#include "lab.cuh"

namespace cluster_sum {

constexpr int kCtas = 8, kThreads = 128, kChannels = 64, kPeriod = 4;
//: a thread's collectors repeat every kPeriod of its values
static_assert(kPeriod * kThreads / kChannels % kCtas == 0);
template <int TOKENS>
constexpr int kPanel = TOKENS * kChannels;
template <int TOKENS>
constexpr int kMine = kPanel<TOKENS> / kThreads;
enum Form : int { kDirect = 0, kSlots, kAtomic, kBulkSlots, kBulkInteger, kWarpInteger, kHandOver, kForms };
const char* kFormName[kForms] = {"every CTA drains its own", "fp32 slots stored by the warps",
                                 "fp32 atomic adds into one panel", "fp32 slots by a bulk copy",
                                 "64-bit integers by a bulk reduction", "64-bit integer adds by the warps",
                                 "the round's barriers alone, no partial"};
constexpr float kScale = 1048576.0f;

//: a CTA's value at panel element `e`, a small integer: every sum exact in fp32
constexpr int value_of(int rank, int e) { return e % 7 + 1 + rank; }

//: THE COLLECTING, spread or not: the CTA that collects element `e`, the element's index there, the elements a
//: collector holds
template <int SPREAD>
constexpr int collector_of(int e) { return SPREAD ? (e / kChannels) % kCtas : 0; }
template <int SPREAD>
constexpr int local_of(int e) { return SPREAD ? (e / kChannels / kCtas) * kChannels + e % kChannels : e; }
template <int SPREAD, int TOKENS>
constexpr int kLocal = SPREAD ? kPanel<TOKENS> / kCtas : kPanel<TOKENS>;
template <int SPREAD>
constexpr int global_of(int rank, int l) {
  return SPREAD ? (kCtas * (l / kChannels) + rank) * kChannels + l % kChannels : l;
}

//: the shared memory: the collector's panel (a slot a CTA for the slot forms; 8 bytes a value for the integers),
//: a bulk form's staged partial (a collector's elements together), then two barriers
template <int FORM>
constexpr bool kSlotted = FORM == kSlots || FORM == kBulkSlots;
template <int FORM>
constexpr bool kBulk = FORM == kBulkSlots || FORM == kBulkInteger;
template <int FORM>
constexpr int kElement = FORM == kBulkInteger || FORM == kWarpInteger ? 8 : 4;
template <int FORM, int SPREAD, int TOKENS>
constexpr int kPanelBytes = kLocal<SPREAD, TOKENS> * kElement<FORM> * (kSlotted<FORM> ? kCtas : 1);
template <int FORM, int TOKENS>
constexpr int kStageBytes = kBulk<FORM> ? kPanel<TOKENS> * kElement<FORM> : 0;
template <int FORM, int SPREAD, int TOKENS>
constexpr int kSmemBytes = kPanelBytes<FORM, SPREAD, TOKENS> + kStageBytes<FORM, TOKENS> + 16;

template <int FORM, int SPREAD, int TOKENS>
__global__ void __cluster_dims__(kCtas, 1, 1) sum(long long* times, float* out, int rounds) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t rank = lab::cluster_rank();
  constexpr int kLocal = cluster_sum::kLocal<SPREAD, TOKENS>, kMine = cluster_sum::kMine<TOKENS>;
  constexpr int kPanelBytes = cluster_sum::kPanelBytes<FORM, SPREAD, TOKENS>;
  constexpr int kStageAt = kPanelBytes, kBars = kStageAt + kStageBytes<FORM, TOKENS>;
  constexpr int kCollectors = SPREAD ? kCtas : 1, kShare = kLocal * kElement<FORM>;
  const uint32_t base = lab::smem(smem), full = base + kBars, done = full + 8;
  float* const dst = out + static_cast<long long>(blockIdx.x / kCtas) * kPanel<TOKENS>;
  const bool collects = SPREAD || rank == 0;

  //: THE PANEL, zero; the barriers: `full` an arrival a CTA (a bulk form: the collector's one, its bytes the
  //: copies'), `done` an arrival a collector
#pragma unroll 1
  for (int i = tid; i < kPanelBytes / 4; i += kThreads) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  if (tid == 0) {
    lab::mbar_init(full, kBulk<FORM> ? 1 : kCtas);
    lab::mbar_init(done, kCollectors);
    lab::cluster_fence_init();
  }
  if constexpr (kBulk<FORM>) lab::bulk_fence();
  __syncthreads();
  //: a bulk form's collector states the round's bytes before any mate can send them
  if constexpr (kBulk<FORM>) {
    if (collects && tid == 0) lab::mbar_expect(full, kCtas * kShare);
  }
  //: every CTA of the cluster has its panel and barriers before any reaches into another's
  lab::cluster_sync();
  float v[kMine];
#pragma unroll
  for (int i = 0; i < kMine; ++i) v[i] = static_cast<float>(value_of(rank, i * kThreads + tid));
  //: THE MATES' ADDRESSES, mapped once: the panels this thread's values go to, the collector it hands over to (its
  //: panel and barrier), every CTA's `done`
  uint32_t there[kPeriod], dones[kCtas];
#pragma unroll
  for (int j = 0; j < kPeriod; ++j) there[j] = lab::cluster_addr(base, collector_of<SPREAD>(j * kThreads + tid));
#pragma unroll
  for (int c = 0; c < kCtas; ++c) dones[c] = lab::cluster_addr(done, c);
  const uint32_t to = SPREAD ? tid % kCtas : 0;
  const uint32_t bar = lab::cluster_addr(full, to), panel = lab::cluster_addr(base, to);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    //: THE PARTIAL OUT: this thread's values, each to its collector (a bulk form: to this CTA's stage)
#pragma unroll
    for (int i = 0; i < kMine; ++i) {
      const int e = i * kThreads + tid;
      if constexpr (FORM == kHandOver) break;
      if constexpr (FORM == kDirect) {
        lab::red_add_f32(dst + e, v[i]);
      } else {
        const int c = collector_of<SPREAD>(e), l = local_of<SPREAD>(e);
        const uint32_t at = there[i % kPeriod];
        if constexpr (FORM == kSlots) lab::cluster_store_f32(at + 4 * (rank * kLocal + l), v[i]);
        if constexpr (FORM == kAtomic) lab::cluster_add_f32(at + 4 * l, v[i]);
        if constexpr (FORM == kWarpInteger) lab::cluster_add_u64(at + 8 * l, static_cast<long long>(v[i] * kScale));
        if constexpr (FORM == kBulkSlots) reinterpret_cast<float*>(smem + kStageAt)[c * kLocal + l] = v[i];
        if constexpr (FORM == kBulkInteger)
          reinterpret_cast<long long*>(smem + kStageAt)[c * kLocal + l] = static_cast<long long>(v[i] * kScale);
      }
    }
    if constexpr (FORM != kDirect) {
      //: every thread's writes ordered before the hand-over: a bulk form's stage to the bulk engine, another form's
      //: stores and adds to the cluster
      if constexpr (kBulk<FORM>) lab::bulk_fence();
      if constexpr (FORM == kSlots || FORM == kAtomic || FORM == kWarpInteger) lab::cluster_fence();
      __syncthreads();
      //: THE HAND-OVER: a thread a collector -- its share of the stage by one bulk copy, or an arrival
      if (tid < kCollectors) {
        if constexpr (FORM == kBulkSlots)
          lab::cluster_bulk_copy(panel + rank * kShare, base + kStageAt + tid * kShare, kShare, bar);
        if constexpr (FORM == kBulkInteger)
          lab::cluster_bulk_add_u64(panel, base + kStageAt + tid * kShare, kShare, bar);
        if constexpr (!kBulk<FORM>) lab::cluster_arrive(bar);
      }
      //: THE COLLECTOR: its elements summed, added to global memory, the panel cleared for the next round
      if (collects) {
        lab::cluster_wait(full, r & 1);
#pragma unroll 1
        for (int l = tid; l < kLocal; l += kThreads) {
          if constexpr (FORM == kHandOver) break;
          float sum = 0.0f;
          if constexpr (kSlotted<FORM>) {
#pragma unroll
            for (int s = 0; s < kCtas; ++s) sum += reinterpret_cast<const float*>(smem)[s * kLocal + l];
          }
          if constexpr (FORM == kAtomic) {
            sum = reinterpret_cast<const float*>(smem)[l];
            reinterpret_cast<float*>(smem)[l] = 0.0f;
          }
          if constexpr (FORM == kBulkInteger || FORM == kWarpInteger) {
            sum = static_cast<float>(reinterpret_cast<const long long*>(smem)[l]) / kScale;
            reinterpret_cast<long long*>(smem)[l] = 0;
          }
          lab::red_add_f32(dst + global_of<SPREAD>(rank, l), sum);
        }
        if constexpr (kBulk<FORM>) lab::bulk_fence();
        //: the collector's reads and clears ordered for the cluster before its mates' next round
        lab::cluster_fence();
        __syncthreads();
        if (tid == 0) {
          if constexpr (kBulk<FORM>) lab::mbar_expect(full, kCtas * kShare);
#pragma unroll
          for (int s = 0; s < kCtas; ++s) lab::cluster_arrive(dones[s]);
        }
      }
      lab::cluster_wait(done, r & 1);
    }
  }
  clock.stop(times);
  //: no CTA leaves while a mate may still arrive at its barrier
  lab::cluster_sync();
}

}  // namespace cluster_sum

template <int FORM, int SPREAD, int TOKENS>
void run(lab::Bench& b, float* out) {
  using namespace cluster_sum;
  constexpr int kCheckRounds = 16, kRounds = 1024, kPanel = cluster_sum::kPanel<TOKENS>;
  const int clusters = b.sms / kCtas, blocks = clusters * kCtas;
  const std::string name = std::string(kFormName[FORM])
                           + (FORM == kDirect ? "" : SPREAD ? ", eight collectors" : ", one collector")
                           + ", rounds of " + std::to_string(TOKENS) + " tokens";
  constexpr int bytes = kSmemBytes<FORM, SPREAD, TOKENS>;
  const auto kernel = sum<FORM, SPREAD, TOKENS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const size_t words = static_cast<size_t>(clusters) * kPanel;
  if (FORM == kHandOver) {
    b.unchecked(name, "an isolation: the arrival forms' barriers with no partial sent or summed");
  } else {
    LAB_CHECK(cudaMemset(out, 0, words * sizeof(float)));
    kernel<<<blocks, kThreads, bytes>>>(b.times, out, kCheckRounds);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(words);
    LAB_CHECK(cudaMemcpy(got.data(), out, words * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (size_t i = 0; i < words; ++i) {
      const int e = static_cast<int>(i % kPanel);
      int want = 0;
      for (int rank = 0; rank < kCtas; ++rank) want += value_of(rank, e);
      wrong += got[i] != static_cast<float>(want * kCheckRounds);
    }
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(words) + " sums differ from the exact ones at "
                + std::to_string(kCheckRounds) + " rounds");
  }
  const lab::Samples cycles = b.variant([&] { kernel<<<blocks, kThreads, bytes>>>(b.times, out, kRounds); },
                                        [&] { return b.span(kThreads >> 5, blocks) / kRounds / TOKENS; });
  b.record(name, "cycles a token a CTA", cycles);
}

template <int TOKENS>
void expect(lab::Bench& b) {
  using namespace cluster_sum;
  constexpr int kMine = cluster_sum::kMine<TOKENS>;
  b.expect(lab::kernel("sum", {kDirect, 0, TOKENS}), "REDG", "==", kMine);
  for (int spread = 0; spread < 2; ++spread) {
    b.expect(lab::kernel("sum", {kSlots, spread, TOKENS}), "ST.E", "==", kMine);
    b.expect(lab::kernel("sum", {kAtomic, spread, TOKENS}), "ATOM.E.ADD.F32", "==", kMine);
    b.expect(lab::kernel("sum", {kWarpInteger, spread, TOKENS}), "ATOM.E.ADD.64", "==", kMine);
    b.expect(lab::kernel("sum", {kBulkSlots, spread, TOKENS}), "UBLKCP.S.S", "==", 1);
    b.expect(lab::kernel("sum", {kBulkInteger, spread, TOKENS}), "UBLKRED.S.S.ADD.U64", "==", 1);
  }
}

template <int TOKENS>
void rounds(lab::Bench& b, float* out) {
  using namespace cluster_sum;
  run<kDirect, 0, TOKENS>(b, out);
  run<kSlots, 0, TOKENS>(b, out);
  run<kSlots, 1, TOKENS>(b, out);
  run<kAtomic, 0, TOKENS>(b, out);
  run<kAtomic, 1, TOKENS>(b, out);
  run<kBulkSlots, 0, TOKENS>(b, out);
  run<kBulkSlots, 1, TOKENS>(b, out);
  run<kBulkInteger, 0, TOKENS>(b, out);
  run<kBulkInteger, 1, TOKENS>(b, out);
  run<kWarpInteger, 0, TOKENS>(b, out);
  run<kWarpInteger, 1, TOKENS>(b, out);
  run<kHandOver, 0, TOKENS>(b, out);
  run<kHandOver, 1, TOKENS>(b, out);
}

int main(int argc, char** argv) {
  using namespace cluster_sum;
  lab::Bench b(argc, argv);
  expect<16>(b);
  expect<64>(b);
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * (b.sms / kCtas) * kPanel<64>));
  rounds<16>(b, out);
  rounds<64>(b, out);
  return 0;
}
