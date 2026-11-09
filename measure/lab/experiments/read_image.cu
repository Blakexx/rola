// measure/lab/experiments/read_image.cu -- QUESTION: the carry's readout step in its ruled final form, the state from
// a bf16 image in shared memory: tokens on the MMA's M (16-token steps, the state the B operand) or on its N (8-token
// tiles, the state the A operand) -- what does each cost against its tensor floor and its instruction budget at
// nl64k-alt-k4's read lists and at a dense cell, and what do the panel's adds and the pass's drain add? METHOD: 8
// warps, one CTA an SM a box of 32 lists x 32 read digits x 32 channels; a warp reads digits w, w + 8, w + 16, w + 24,
// every digit's first step in one straight-line block, its state fragments (16 registers) from the image by
// `ldmatrix.trans`; read rows (32 lists of bf16, 64 bytes, swizzled) gathered from the pass's pool by the entries the
// facts pass emits in their final form (`read_image.inputs.py`): the pool row's offset, the coefficient, the panel
// row's offset with the lane's token folded in, so a pad lands on its residue's dummy row with no test -- the slots
// colored and the tokens placed so every gather and add is free of bank conflicts (KERNEL_STANDARDS 30); bf16 MMAs with
// fp32 accumulate; each output pair times its coefficient to the exact integer hi x 2^16 + lo on the FMA pipe, added
// by `red.shared` into its token's panel row; the drain at the pass's end (`red.global.add`, each word cleared). A
// warp's heads (its digits' first entries) staged a window ahead by `cp.async`. Outputs checked exactly, the
// even channels negative (a low half below zero carries into the high half) and two coefficients.
// see docs/internals/tools/lab.md#read-image
#include "read_image.cuh"

namespace read_image {

template <int FORM, int OUT, int DENSE, int BAR>
__global__ __launch_bounds__(kThreads, 1) void read(Lists ls, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = ls.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625, the pool's lists +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kBytes / 16; k += kThreads) {
    const uint32_t v = k < kPool / 16 ? 0x3d80bd80u : k < kPads / 16 ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const Lane c = lane_of<FORM>(sm, warp, lane);
  float sink = 0.0f;
  //: the warp's heads, a window ahead: a lane's 16-byte chunks of the entries (its first at `src`, `to` in the window's
  //: buffer)
  const uint8_t* src = ls.heads + (static_cast<int64_t>(box) * kWarps + warp) * kHeadBytes + 16 * lane;
  const int64_t src_step = static_cast<int64_t>(boxes) * kWarps * kHeadBytes;
  const uint32_t hb0 = sm + kHeads + warp * kHeadBytes, hb_step = kWarps * kHeadBytes;
  const uint32_t to0 = lab::held(hb0 + 16 * lane), he0 = lab::held(hb0 + c.entry);
  const auto heads_in = [&](uint32_t to) {
#pragma unroll
    for (int k = 0; k < kHead / 16; ++k) lab::cp_async16(to + 512 * k, src + 512 * k);
  };
  const uint16_t* count_at = ls.count + box;
  if constexpr (!DENSE) heads_in(to0);
  lab::cp_commit();
  int next = DENSE ? 2 * kSlots : *count_at;
  //: the drain of pass `p` (`rows` slots): each warp's slots, two a request, each word cleared
  const auto drain = [&](int w, int p, int rows) {
    uint32_t* words = reinterpret_cast<uint32_t*>(smem);
#pragma unroll 1
    for (int k = 0; k < (min(rows - kSlots / kWarps * warp, kSlots / kWarps) + 1) / 2; ++k) {
      const int slot = kSlots / kWarps * warp + 2 * k + (lane >> 4), word = lane & 15;
      uint32_t* cell = words + kPanel / 4 + slot * kPanelPitch + word;
      uint32_t* target = ls.out + ((static_cast<int64_t>(w) * 2 + p) * kSlots + slot) * kWords + word;
      asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target), "r"(*cell) : "memory");
      *cell = 0u;
    }
    __syncthreads();
  };
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    const int count = next;
    if constexpr (DENSE) {
      const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
      const uint32_t a0 = (sm + kPool + in_plane(FORM == 0 ? 8 * (q & 1) + r : r, 0)) ^ c.chunk;
      const uint32_t p0 = sm + kPanel + (FORM == 0 ? 80u * g + 4u * t : 160u * t + 4u * g);
#pragma unroll 1
      for (int p = 0; p < 2; ++p) {
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
          uint32_t f[16];
          state_in<FORM>(f, c, plane_off(i));
#pragma unroll
          for (int s = 0; s < kSlots / (FORM == 0 ? 16 : 8); ++s) {
            if constexpr (FORM == 0) step_dense<OUT>(a0, p0, f, sink, s);
            if constexpr (FORM == 1) tile_dense<OUT>(a0, p0, f, sink, s);
          }
        }
        if constexpr (BAR == 1) __syncthreads();
        if constexpr (OUT == 2) drain(w, p, kSlots);
      }
      continue;
    }
    const uint32_t flip = (w & 1) * hb_step;
    //: the next window's heads and count, unconditionally: the last window's next is the inputs' zero pad. The buffer
    //: the copies overwrite was read last window by every lane
    __syncwarp();
    src += src_step;
    count_at += ls.groups;
    heads_in(to0 + (hb_step - flip));
    next = *count_at;
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    const uint32_t he = he0 + flip;
    //: a digit's entry `e` past the staged ones, from its list in global memory (zero past its count, in its row since
    //: its steps end by entry 512): 0.3% of digits at alt-k4, the facts pass's past-16 debt
    const auto listed = [&](int i, int e) {
      const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
      return ls.entries[row * kListCap + e];
    };
    //: a digit's steps (or tiles) and whether any of the warp's run past the staged entries, from the high half of the
    //: lane's first entry's pool word: each branch on them a vote, so ptxas sees it warp-uniform (no convergence
    //: barrier around the collectives it guards)
    uint32_t tag[4] = {};
    const auto steps = [&](int i) { return static_cast<int>((tag[i] >> 16) & 0xffu); };
    const auto past = [&] { return __any_sync(0xffffffffu, ((tag[0] | tag[1] | tag[2] | tag[3]) >> 24) & 1u); };
    constexpr int kStaged = kHead / (FORM == 0 ? 16 : 8);
    if constexpr (FORM == 0) {
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        state_in<0>(f[i], c, plane_off(i));
        const uint2 e = lds64(he + kHead * 8 * i);
        tag[i] = e.x;
        step_m<OUT>(sm, e, c, f[i], sink);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int s = 1; s < kStaged; ++s)
          for (int k = __any_sync(0xffffffffu, steps(i) > s) ? 1 : 0; k > 0; --k)
            step_m<OUT>(sm, lds64(he + kHead * 8 * i + 128 * s), c, f[i], sink);
      for (int rare = past() ? 1 : 0; rare > 0; --rare)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          for (int k = __any_sync(0xffffffffu, steps(i) > kStaged) ? 1 : 0, s = kStaged; k > 0; --k)
            do step_m<OUT>(sm, listed(i, 16 * s + (lane & 15)), c, f[i], sink);
            while (__any_sync(0xffffffffu, ++s < steps(i)));
    } else {
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        state_in<1>(f[i], c, plane_off(i));
        const uint2 e = lds64(he + kHead * 8 * i);
        tag[i] = e.x;
        tile_n<OUT>(sm, e, c, f[i], sink);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int s = 1; s < kStaged; ++s)
          for (int k = __any_sync(0xffffffffu, steps(i) > s) ? 1 : 0; k > 0; --k)
            tile_n<OUT>(sm, lds64(he + kHead * 8 * i + 64 * s), c, f[i], sink);
      for (int rare = past() ? 1 : 0; rare > 0; --rare)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          for (int k = __any_sync(0xffffffffu, steps(i) > kStaged) ? 1 : 0, s = kStaged; k > 0; --k)
            do tile_n<OUT>(sm, listed(i, 8 * s + (lane & 7)), c, f[i], sink);
            while (__any_sync(0xffffffffu, ++s < steps(i)));
    }
    if constexpr (BAR == 1) __syncthreads();
    if constexpr (OUT == 2) drain(w, 0, count);
  }
  clock.stop(times);
  lab::cp_wait_all();
  lab::sink(sinks, lab::bits(sink));
  //: the check's read of the panel, after the timed loop: every word a CTA's adds left in it
  if constexpr (OUT >= 1) {
    __syncthreads();
    for (int k = static_cast<int>(threadIdx.x); k < kSlots * kWords / 4; k += kThreads)
      reinterpret_cast<uint4*>(ls.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kWords / 4 + k] =
          chunks[(kPanel + (k >> 2) * kPanelPitch * 4) / 16 + (k & 3)];
  }
}

//: THE BUDGETS (KERNEL_STANDARDS sections 29-30), each class's instructions per MMA and the shared-memory wavefronts
//: per MMA, from the operation count at alt-k4: a read digit ~8 tokens a window (one 16-position step, 8 MMAs; ~1.41
//: 8-token tiles, ~5.6 MMAs), a warp 4 digits. TOKENS ON M -- a step: 2 read-row loads and an entry load, 3 shuffles,
//: 16 FFMAs, 8 adds; integer 14 (the read rows' mask-XOR and XOR, 2 coefficients' shifts, 2 panel bases, 8 packs). A
//: digit: 4 state loads (immediates). A window (32 MMAs a warp): the heads' copy (global) and the 3 never-executed
//: `LDS` ptxas sets before an `LDGSTS`; integer ~14 (the source and count pointers, the buffer's parity, the tags' OR
//: and test, the loop counter); select ~6 (the loop's test, the past-staged vote and its predicate); control ~8 (2
//: alignment NOPs, commit, wait, the loop's exit CALL and branch, the past-staged branch; the pass barrier). Per MMA:
//: shared load 0.97, shuffle 0.38, float 2.0, integer 2.2, select 0.19, control 0.25 (0.28 with the barrier), global
//: 0.04. TOKENS ON N -- a tile (4 MMAs): a read-row load and an entry load, 3 shuffles, 8 FFMAs, 4 adds, integer 9 (a
//: mask-XOR, 2 shifts, 2 bases, 4 packs) and ~1.3 of ptxas's never-executed `UIADD3` fillers between its dependent
//: MMAs (two accumulators, each two MMAs deep); a digit (~5.6 MMAs): 4 state loads, its tag's extraction (2), the
//: second tile's vote and branch, after every digit's first tile in one straight-line block (as M's first steps); a
//: window (~22.6 MMAs): as on M. Per MMA: shared load 1.36, shuffle 0.75, float 2.0,
//: integer 3.7, select 0.45, control 0.55 (0.6), global 0.05. DENSE (a digit 16 steps or 32 tiles a pass): a step 2
//: read-row loads, 16 FFMAs, 8 packs, 8 adds (a tile 1, 8, 4, 4), its addresses immediates; a digit 4 state loads.
//: Per MMA: shared load 0.28, float 2.0, integer 1.05 (N: its fillers too), control 0.05. AT W = 1024 (`read_w1024.cu`:
//: ~16 tokens a digit, 32 entries staged, two 16-byte copies a lane): M's digit adds a vote and its branch for its
//: second step (~12 MMAs a digit), N's three for its tiles past the first (~9.7): select 0.35 and 0.95, global 0.07.
//: Each adds the kernel's setup and the check's copy-out over its ~1000 MMAs a warp (~0.1 an MMA, mostly integer);
//: "other" (an opcode in no class) is budgeted zero.
//: WAVEFRONTS, conflict-free by the entries' coloring and placement: a step's 2 read-row loads 8, its 8 adds 8, its
//: entry load 2 (an 8-byte load serves a half-warp a wavefront); a tile's 4, 4 and 2; a digit's state 16; a window's
//: heads copy 8 a 512 bytes (`LDGSTS` counts twice a copy's 16 bytes); the init and copy-out ~0.1 an MMA; and the
//: counter's remainder no operation is named for, ~0.6 a shuffle. Per MMA at alt-k4: M 2.25 + 1.99 + 0.25 + 0.1 +
//: 0.22 = 4.8, N 2.5 + 2.84 + 0.36 + 0.1 + 0.45 = 6.25; dense 2.0 + 0.13; at W = 1024 M 2.25 + 1.4 + 0.35 + 0.1 + 0.22
//: = 4.3, N 2.5 + 1.65 + 0.42 + 0.1 + 0.45 = 5.1.
struct Budget {
  double shared_load, shuffle, float_, integer, select, control, global, wavefronts;
};

constexpr bool kWide = kSlots == 512;
constexpr Budget kBudget[4] = {
    {1.0, 0.4, 2.0, 2.5, kWide ? 0.35 : 0.2, 0.3, kWide ? 0.07 : 0.05, kWide ? 4.45 : 4.95},
    {1.4, 0.8, 2.0, 3.75, kWide ? 0.95 : 0.5, 0.6, kWide ? 0.08 : 0.06, kWide ? 5.25 : 6.4},
    {0.3, 0.0, 2.0, 1.15, 0.03, 0.08, 0.01, 2.25},
    {0.3, 0.0, 2.0, 1.5, 0.03, 0.08, 0.01, 2.25}};

template <int FORM, int OUT, int DENSE, int BAR = 1>
void variant(const lab::Bench& b, const Lists& ls, const char* lists, const Reference& ref) {
  static_assert(BAR == 1 || OUT < 2, "the drain needs the pass's barrier");
  const std::string name = lab::kernel("read_image::read", {FORM, OUT, DENSE, BAR});
  b.expect(name, "HMMA.16816.F32.BF16", ">=", FORM == 0 ? 8 : 4);
  //: a digit past its staged entries reads them from global memory inside its steps (the facts pass's past-16 debt)
  if (!DENSE) b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (OUT == 0) b.exempt(name, "an isolation: the MMAs alone");
  if (OUT == 2) b.exempt(name, "a budgeted step (its OUT 1 variant) plus the pass's drain");
  if (OUT == 1) {
    const Budget& u = kBudget[DENSE ? 2 + FORM : FORM];
    b.mma(name, "hmma_bf16_f32", 16);
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", u.shared_load);
    b.budget(name, "shared store", 0.03);
    b.budget(name, "shared atomic", 1.0);
    b.budget(name, "shuffle", u.shuffle);
    b.budget(name, "pack", 0.0);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "float", u.float_);
    b.budget(name, "select/compare", u.select);
    b.budget(name, "integer/address", u.integer);
    b.budget(name, "control", u.control);
    b.budget(name, "global", u.global);
    b.budget(name, "other", 0.0);
    b.budget(name, "wavefronts", u.wavefronts);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(read<FORM, OUT, DENSE, BAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] { read<FORM, OUT, DENSE, BAR><<<b.sms, kThreads, kBytes>>>(ls, b.times, b.sinks); };
  const double floor = floor_of(*ref.h, FORM, DENSE, BAR == 0), floor_m = floor_of(*ref.h, 0, DENSE, BAR == 0);
  const lab::Samples s = b.variant(launch,
                                   [&] { return b.span(kWarps) / ls.windows; });
  const char* forms[2] = {"tokens on M", "tokens on N"};
  const char* outs[3] = {"the MMAs alone", "+ panel adds", "+ panel adds + drain"};
  const std::string label = std::string(DENSE ? "dense / " : "alt-k4 / ") + forms[FORM] + " / " + outs[OUT] + lists +
                            (BAR ? "" : " / no pass barrier");
  b.record(label, "cycles a CTA-window", s);
  lab::Samples own, m;
  for (double v : s.v) {
    own.v.push_back(v / floor);
    m.v.push_back(v / floor_m);
  }
  b.record(label + " / its floor", "x", own);
  b.record(label + " / tokens-on-M floor", "x", m);
  if constexpr (OUT == 0) {
    b.unchecked(label, "an isolation: the MMAs alone, their outputs folded into a sink");
  } else {
    LAB_CHECK(cudaMemset(ls.out, 0, static_cast<size_t>(ls.windows) * 2 * kSlots * kWords * 4));
    LAB_CHECK(cudaMemset(ls.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    const Verdict v = verify<FORM, OUT, DENSE>(ref, ls);
    b.check(label, v.bad == 0 && v.words > 0, v.detail);
  }
}

}  // namespace read_image

int main(int argc, char** argv) {
  using namespace read_image;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists lm{}, ln{};
  HostLists host;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_image: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    load(dir, b.sms, host, lm, ln);
    std::printf("{\"lab\": \"note\", \"floor_m\": %.1f, \"floor_n\": %.1f, \"free_m\": %.1f, \"free_n\": %.1f, "
                "\"dense_m\": %.1f}\n", floor_of(host, 0, false, false), floor_of(host, 1, false, false),
                floor_of(host, 0, false, true), floor_of(host, 1, false, true), floor_of(host, 0, true, false));
  }
  const Reference ref{&host, b.sms};
  const char* lists = kSlots == 512 ? " / W 1024, a 512-slot pass" : "";
  variant<0, 0, 0>(b, lm, lists, ref);
  variant<0, 1, 0>(b, lm, lists, ref);
  variant<0, 1, 0, 0>(b, lm, lists, ref);
  variant<0, 2, 0>(b, lm, lists, ref);
  variant<1, 0, 0>(b, ln, lists, ref);
  variant<1, 1, 0>(b, ln, lists, ref);
  variant<1, 1, 0, 0>(b, ln, lists, ref);
  variant<1, 2, 0>(b, ln, lists, ref);
#if READ_IMAGE_SLOTS == 256
  variant<0, 1, 1>(b, lm, "", ref);
  variant<0, 2, 1>(b, lm, "", ref);
  variant<1, 1, 1>(b, ln, "", ref);
  variant<1, 2, 1>(b, ln, "", ref);
#endif
}
