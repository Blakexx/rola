// measure/lab/experiments/image_ring.cu -- QUESTION: the carry's state image streamed from the fold's warps to the
// readout's through a flagged ring in shared memory (L2): what do its stores, packs and handshakes add to a
// free-running window, by round size and ring depth, against the same window reading a static image? METHOD:
// `read_image.cu`'s tokens-on-N readout (its inputs, steps and exact check) with a warp's 4 lists x 32 digits x 32
// channels of fp32 state held in registers as the fold leaves them (opaque to ptxas each window). A round is one
// 8-digit group (a fold lane's g is its digit) x U 8-channel chunks: each warp packs its lists' values to bf16 and
// stores them with full-warp 4-byte stores, then arrives on the slot's `landed` mbarrier; a readout warp waits on it,
// pulls its digit's slice into its A fragments by `ldmatrix.trans` (the loads a static image costs) and arrives on
// `drained`; a slot is rewritten once all 8 warps drained it. A slot's 16-byte units are swizzled by `g ^ s(list)` so
// the producers' stores (one list, 8 digits) and the consumers' loads (one digit, 8 lists) are both conflict-free
// (KERNEL_STANDARDS 30). A window publishes its first D rounds, then consumes each round, runs a digit's tiles once
// its group has landed, and publishes the round D on. Modes: 0 the static image, 1 the publish without handshakes (an
// isolation: its outputs race), 2 the ring on mbarriers, 3 two slots under one named barrier a round: a warp consumes
// round r and publishes r + 1 before it, so passing it means every warp has landed r + 1 and drained r - 1, whose
// slot r + 1 reuses; 4 and 5 exchange then compute: the rounds' stores and loads between named barriers with nothing
// uneven between them, a warp's 4 digits' fragments held in registers (or 2, the window in halves), the tiles after;
// 6 as 4 with the next window's fold (its stand-in) interleaved with the tiles, as the exchanged copy frees it;
// 7 the static image with a barrier a window (the rendezvous's skew alone), 8 the exchange alone (an isolation).
// see docs/internals/tools/lab.md#image-ring
#include "read_image.cuh"

namespace image_ring {

using namespace read_image;

//: the ring's mbarriers, two a slot (landed, drained), past the readout lab's layout
constexpr int kBarriers = kBytes, kRingBytes = kBarriers + 2 * 8 * 8;
static_assert(kPlanes * kPlaneBytes >= 32 * 1024, "the ring's slots live in the image's region");

//: a slot's 16-byte unit for list `l`, chunk `c` and consumer block `g`, U chunks a list: `(l U + c) ^ s(l) ^ g`, s
//: the swizzle that spreads a block's 8 consecutive lists over 8 bank groups
template <int U>
__host__ __device__ constexpr int unit_of(int l, int c, int g) {
  return ((l * U + c) ^ ((l / (8 / U)) % U) ^ g) & (32 * U - 1);
}

//: two values to a bf16 pair, the lower channel low
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

//: THE FOLD'S STAND-IN between windows: one m16n8k8 MMA into each of a warp's 32 accumulator tiles (~0.7x the
//: fold's tensor work at alt-k4), its operands zeros from memory, so the state's values stay what the check expects
//: but ptxas cannot see them, and each window's publish packs what the fold left
__device__ __forceinline__ void fold_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

//: a wait that gives up after 2^22 polls: a deadlock ends the kernel and fails its check rather than holding the GPU
__device__ __forceinline__ void wait(uint32_t a, uint32_t parity) {
  for (uint32_t n = 0; !lab::mbar_test(a, parity) && n < (1u << 22); ++n) {
  }
}

template <int U, int D, int MODE>
__global__ __launch_bounds__(kThreads, 1) void ring(Lists ls, const uint32_t* zeros, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kSlot = 4096 * U, kRounds = 16 / U, kParts = 4 / U, kFirst = D < kRounds ? D : kRounds;
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = ls.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kRingBytes / 16; k += kThreads) {
    const uint32_t v = k < kPool / 16 ? (MODE == 0 || MODE == 7 ? 0x3d80bd80u : 0u) : k < kPads / 16 ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  if (MODE == 2 && threadIdx.x < 2 * D) lab::mbar_init(sm + kBarriers + 8 * threadIdx.x, kWarps);
  __syncthreads();
  const Lane c = lane_of<1>(sm, warp, lane);
  //: THE STATE as the fold leaves it, a list's rows by the channels' n-tiles: even channels -0.0625, odd +0.0625
  const uint32_t zero = zeros[0];
  float acc[4][2][4][4];
#pragma unroll
  for (int k = 0; k < 128; ++k) (&acc[0][0][0][0])[k] = __uint_as_float(zero | (k & 1 ? 0x3d800000u : 0xbd800000u));
  //: a producer lane's store address for its list J (chunk 0, slot 0), a consumer lane's load address (its list,
  //: chunk 0, slot 0), and lane 0's arrivals
  const int g = lane >> 2, t = lane & 3;
  uint32_t put[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    put[j] = lab::held(sm + kImage + 512 * U * g + 16 * unit_of<U>(4 * warp + j, 0, g) + 4 * t);
  const uint32_t get = lab::held(sm + kImage + 512 * U * warp + 16 * unit_of<U>(lane, 0, warp));
  const uint32_t first = lab::held(lane == 0);
  float sink = 0.0f;
  const uint8_t* src = ls.heads + (static_cast<int64_t>(box) * kWarps + warp) * kHeadBytes + 16 * lane;
  const int64_t src_step = static_cast<int64_t>(boxes) * kWarps * kHeadBytes;
  const uint32_t hb0 = sm + kHeads + warp * kHeadBytes, hb_step = kWarps * kHeadBytes;
  const uint32_t to0 = lab::held(hb0 + 16 * lane), he0 = lab::held(hb0 + c.entry);
  const auto heads_in = [&](uint32_t to) {
#pragma unroll
    for (int k = 0; k < kHead / 16; ++k) lab::cp_async16(to + 512 * k, src + 512 * k);
  };
  heads_in(to0);
  lab::cp_commit();
  //: round `r` of window `w`: its slot, the parity of its slot's use, and the barriers
  const auto landed = [&](int r) { return sm + kBarriers + 16 * (r % D); };
  const auto drained = [&](int r) { return sm + kBarriers + 16 * (r % D) + 8; };
  const auto publish = [&](int w, int r) {
    const int q = w * kRounds + r, k = r / kParts, part = r % kParts;
    if constexpr (MODE == 2)
      for (int k2 = q >= D ? 1 : 0; k2 > 0; --k2) wait(drained(r), (q / D - 1) & 1);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int n = U * part + u;
        const uint32_t v = pack_bf16(acc[j][k >> 1][n][2 * (k & 1)], acc[j][k >> 1][n][2 * (k & 1) + 1]);
        asm volatile("st.shared.u32 [%0], %1;" ::"r"((put[j] ^ (16u * u)) + kSlot * (r % D)), "r"(v) : "memory");
      }
    if constexpr (MODE == 2) {
      __syncwarp();
      for (int k2 = first; k2 > 0; --k2) lab::mbar_arrive(landed(r));
    }
  };
  const auto consume = [&](int w, int r, uint32_t (&f)[16]) {
    const int q = w * kRounds + r, part = r % kParts;
    if constexpr (MODE == 2) wait(landed(r), (q / D) & 1);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int n = U * part + u, m = n >> 1, h = n & 1;
      uint32_t x[4];
      lab::ldsm_t(x, (get ^ (16u * u)) + kSlot * (r % D));
      f[(m * 2) * 4 + h] = x[0];
      f[(m * 2) * 4 + h + 2] = x[1];
      f[(m * 2 + 1) * 4 + h] = x[2];
      f[(m * 2 + 1) * 4 + h + 2] = x[3];
    }
    if constexpr (MODE == 2) {
      __syncwarp();
      for (int k2 = first; k2 > 0; --k2) lab::mbar_arrive(drained(r));
    }
  };
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    const uint32_t flip = (w & 1) * hb_step;
    __syncwarp();
    src += src_step;
    heads_in(to0 + (hb_step - flip));
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    const uint32_t he = he0 + flip;
    const auto listed = [&](int i, int e) {
      const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
      return ls.entries[row * kListCap + e];
    };
    if constexpr (MODE == 1 || MODE == 2) {
#pragma unroll
      for (int r = 0; r < kFirst; ++r) publish(w, r);
    }
    if constexpr (MODE == 3) {
      publish(w, 0);
      lab::bar_sync(1, kThreads);
    }
    //: digit i's tiles: the first staged, the second on a vote, any past the staged from its list
    const auto digit = [&](int i, const uint32_t(&f)[16]) {
      const uint2 e = lds64(he + kHead * 8 * i);
      const int steps = static_cast<int>((e.x >> 16) & 0xffu);
      tile_n<1>(sm, e, c, f, sink);
      for (int k = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k > 0; --k)
        tile_n<1>(sm, lds64(he + kHead * 8 * i + 64), c, f, sink);
      for (int k = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k > 0; --k)
        do tile_n<1>(sm, listed(i, 8 * s + (lane & 7)), c, f, sink);
        while (__any_sync(0xffffffffu, ++s < steps));
    };
    //: a quarter of the fold stand-in (8 MMAs), interleaved with a digit's tiles in mode 6
    const auto fold_quarter = [&](int q) {
#pragma unroll
      for (int k = 8 * q; k < 8 * q + 8; ++k)
        fold_k8(*reinterpret_cast<float(*)[4]>(&acc[k / 8][k / 4 % 2][k % 4][0]), zero, zero, zero);
    };
    if constexpr (MODE == 8) {
      uint32_t f[4][16];
      lab::bar_sync(1, kThreads);
      publish(w, 0);
      publish(w, 1);
      lab::bar_sync(1, kThreads);
      consume(w, 0, f[0]);
      consume(w, 1, f[1]);
      lab::bar_sync(1, kThreads);
      publish(w, 2);
      publish(w, 3);
      lab::bar_sync(1, kThreads);
      consume(w, 2, f[2]);
      consume(w, 3, f[3]);
#pragma unroll
      for (int k = 0; k < 64; ++k) sink += __uint_as_float((&f[0][0])[k]);
    } else if constexpr (MODE == 4 || MODE == 5 || MODE == 6) {
      //: EXCHANGE THEN COMPUTE: every warp stores and loads alike between the named barriers (no skew in them),
      //: holding its digits' fragments; the uneven tiles run after, from registers. The window's first barrier
      //: frees the slots the last window's rounds 2 and 3 held
      constexpr int kHeld = MODE == 5 ? 2 : 4;
      uint32_t f[kHeld][16];
      lab::bar_sync(1, kThreads);
      publish(w, 0);
      publish(w, 1);
      lab::bar_sync(1, kThreads);
      consume(w, 0, f[0]);
      consume(w, 1, f[1]);
      if constexpr (MODE == 5) {
        digit(0, f[0]);
        digit(1, f[1]);
      }
      lab::bar_sync(1, kThreads);
      publish(w, 2);
      publish(w, 3);
      lab::bar_sync(1, kThreads);
      consume(w, 2, f[kHeld - 2]);
      consume(w, 3, f[kHeld - 1]);
#pragma unroll
      for (int i = 4 - kHeld; i < 4; ++i) {
        digit(i, f[i - (4 - kHeld)]);
        if constexpr (MODE == 6) fold_quarter(i);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t f[16];
        if constexpr (MODE == 0 || MODE == 7) {
          if (MODE == 7 && i == 0) lab::bar_sync(1, kThreads);
          state_in<1>(f, c, plane_off(i));
        } else if constexpr (MODE == 3) {
#pragma unroll
          for (int p = 0; p < kParts; ++p) {
            consume(w, kParts * i + p, f);
            if (kParts * i + p + 1 < kRounds) {
              publish(w, kParts * i + p + 1);
              lab::bar_sync(1, kThreads);
            }
          }
        } else {
#pragma unroll
          for (int p = 0; p < kParts; ++p) {
            consume(w, kParts * i + p, f);
            if (kParts * i + p + D < kRounds) publish(w, kParts * i + p + D);
          }
        }
        digit(i, f);
      }
    }
    if constexpr (MODE != 6 && MODE != 8) {
#pragma unroll
      for (int q = 0; q < 4; ++q) fold_quarter(q);
    }
  }
  clock.stop(times);
  lab::cp_wait_all();
  //: the state into the sink, after the timed loop: the stand-in's MMAs are live in every mode
#pragma unroll
  for (int k = 0; k < 128; ++k) sink += (&acc[0][0][0][0])[k];
  lab::sink(sinks, lab::bits(sink));
  __syncthreads();
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * kWords / 4; k += kThreads)
    reinterpret_cast<uint4*>(ls.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kWords / 4 + k] =
        chunks[(kPanel + (k >> 2) * kPanelPitch * 4) / 16 + (k & 3)];
}

//: THE BUDGETS (KERNEL_STANDARDS sections 29-30), per MMA at alt-k4: a warp-window's ~22.5 tokens-on-N readout MMAs
//: at `read_image`'s free-running counts, the fold stand-in's 32 MMAs (nothing else), and the ring's R = 16 / U
//: rounds, over the 54.5 MMAs. A warp's round: 4U packs
//: (`F2FP`) and 4U stores (an XOR each past a list's first chunk); its U `ldmatrix.trans` are the static image's state
//: loads; two waits (a `test_wait` poll, its predicate, the bound's count and compare, and the branch, once each when
//: the round is ready), two syncwarps, lane 0's two arrivals (shared atomics), the slot's parity. A window: 64 packs
//: and 64 stores whatever U. Per MMA, R / 22.5 rounds: pack 2.84; shared store 2.84; shared atomic + 2 a round;
//: select + 2; control + 4; integer + 2 a round and the XORs (64 (U - 1) / U a window). Wavefronts: the stores 512 a
//: CTA-window (2.84 an MMA), and a poll and an arrival each one (4 a round).
struct Budget {
  double pack, shared_store, shared_atomic, select, control, integer, wavefronts;
};

constexpr double kReadMmas = 22.5, kMmas = kReadMmas + 32.0;

template <int U>
constexpr Budget budget_of() {
  const double r = 16.0 / U;
  return {64 / kMmas, 64 / kMmas, (kReadMmas + 2 * r) / kMmas, (0.5 * kReadMmas + 2 * r) / kMmas,
          (0.6 * kReadMmas + 4 * r) / kMmas, (3.75 * kReadMmas + 2 * r + 64.0 * (U - 1) / U) / kMmas,
          (6.4 * kReadMmas + 64 + 4 * r) / kMmas};
}

template <int U, int D, int MODE>
void variant(const lab::Bench& b, const Lists& ls, const uint32_t* zeros, const Reference& ref) {
  const std::string name = lab::kernel("image_ring::ring", {U, D, MODE});
  //: the exchange alone runs no MMA
  if (MODE != 8) b.expect(name, "HMMA.16816.F32.BF16", ">=", 4);
  b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (MODE == 0) b.exempt(name, "the baseline: read_image's budgeted tokens-on-N step from a static image");
  if (MODE == 1) b.exempt(name, "an isolation: the publish's stores and packs without the handshakes");
  if (MODE >= 3) b.exempt(name, "a probe of a hand-off's cost against the mbarrier ring's (budgeted)");
  if (MODE == 2) {
    constexpr Budget u = budget_of<U>();
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", 1.4 * kReadMmas / kMmas);
    b.budget(name, "shared store", u.shared_store);
    b.budget(name, "shared atomic", u.shared_atomic);
    b.budget(name, "shuffle", 0.8 * kReadMmas / kMmas);
    b.budget(name, "pack", u.pack);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "float", 2.0 * kReadMmas / kMmas);
    b.budget(name, "select/compare", u.select);
    b.budget(name, "integer/address", u.integer);
    b.budget(name, "control", u.control);
    b.budget(name, "global", 0.06 * kReadMmas / kMmas);
    b.budget(name, "other", 0.0);
    b.budget(name, "wavefronts", u.wavefronts);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(ring<U, D, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kRingBytes));
  const auto launch = [&] { ring<U, D, MODE><<<b.sms, kThreads, kRingBytes>>>(ls, zeros, b.times, b.sinks); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps) / ls.windows; });
  const double floor = floor_of(*ref.h, 1, false, true);
  const std::string rounds = std::to_string(U * 8) + "-channel rounds (" + std::to_string(4 * U) + " KB)";
  const std::string label = MODE == 0   ? std::string("a static image")
                            : MODE == 1 ? "the publish without handshakes, " + rounds
                            : MODE == 2 ? "the ring, " + rounds + " x " + std::to_string(D) + " slots, mbarriers"
                            : MODE == 3 ? "the ring, " + rounds + " x 2 slots, a named barrier a round"
                            : MODE == 7 ? std::string("a static image, a barrier a window")
                            : MODE == 8 ? std::string("the exchange alone, 2 slots of 16 KB")
                            : MODE == 6 ? std::string("exchange, then the tiles and the fold interleaved, 4 held")
                            : MODE == 4 ? std::string("exchange then compute, 4 digits held, 2 slots of 16 KB")
                                        : std::string("exchange then compute in halves, 2 digits held, 2 slots");
  b.record(label, "cycles a CTA-window", s);
  lab::Samples own;
  for (double v : s.v) own.v.push_back(v / floor);
  b.record(label + " / the free floor", "x", own);
  if constexpr (MODE == 1 || MODE == 8) {
    b.unchecked(label, MODE == 1 ? "an isolation: the stores race the loads without handshakes"
                                 : "an isolation: the exchange's loads fold into a sink");
  } else {
    LAB_CHECK(cudaMemset(ls.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    const Verdict v = verify<1, 1, 0>(ref, ls);
    b.check(label, v.bad == 0 && v.words > 0, v.detail);
  }
}

}  // namespace image_ring

int main(int argc, char** argv) {
  using namespace image_ring;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists lm{}, ln{};
  HostLists host;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "image_ring: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    load(dir, b.sms, host, lm, ln);
  }
  uint32_t* zeros = nullptr;
  LAB_CHECK(cudaMalloc(&zeros, 64));
  LAB_CHECK(cudaMemset(zeros, 0, 64));
  const Reference ref{&host, b.sms};
  variant<1, 2, 0>(b, ln, zeros, ref);
  variant<1, 2, 1>(b, ln, zeros, ref);
  variant<4, 2, 1>(b, ln, zeros, ref);
  variant<1, 2, 2>(b, ln, zeros, ref);
  variant<1, 4, 2>(b, ln, zeros, ref);
  variant<1, 8, 2>(b, ln, zeros, ref);
  variant<2, 2, 2>(b, ln, zeros, ref);
  variant<2, 4, 2>(b, ln, zeros, ref);
  variant<4, 2, 2>(b, ln, zeros, ref);
  variant<1, 2, 3>(b, ln, zeros, ref);
  variant<2, 2, 3>(b, ln, zeros, ref);
  variant<4, 2, 3>(b, ln, zeros, ref);
  variant<4, 2, 4>(b, ln, zeros, ref);
  variant<4, 2, 5>(b, ln, zeros, ref);
  variant<4, 2, 6>(b, ln, zeros, ref);
  variant<4, 2, 7>(b, ln, zeros, ref);
  variant<4, 2, 8>(b, ln, zeros, ref);
}
