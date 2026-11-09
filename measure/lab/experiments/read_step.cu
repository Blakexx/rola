// measure/lab/experiments/read_step.cu -- QUESTION: the readout's step on the even carve (a warp owns 4 level-S read
// digits x 32 level-R digits x 32 channels): per digit (each digit over its own read tokens, gathered) or merged (the
// warp's digits summed over the union in registers), what does a step cost beside its tensor floor, what do its output
// adds into the shared panel add, and does draining the panel to global memory cost anything once it is double
// buffered and spread through the next pass, against draining it in a burst at the pass's end, or against adding each
// step's outputs straight to global rows with no panel? METHOD: 8 warps, two a
// scheduler, the fold's 128 fp32 accumulators live beside the published slice (B fragments: K = level R, permuted so a
// lane's four k rows are one token's four consecutive digits, N = channels); a step's A = 16 tokens' level-R bytes by
// one `ldmatrix.x4` (rows = tokens: 96-byte read rows, level R's 32 bytes then level S's, the 16-byte halves swapped on
// rows with bit 2 set) and a PRMT + HFMA2 a register to f16 with the token's level-S coefficient; 8
// fp16-accumulate MMAs and 2 for den (the mass as a fifth n-tile); the outputs to s16 pairs (two channels a word) added
// by `red.shared.add` into a 128-slot panel (17 words a slot); the drain two slots a warp request
// (`red.global.add.u32`, scattered over 16 MB), each word cleared as it goes. The block's span a pass against its
// tensor floor (the busiest scheduler's two warps' MMAs at 16 cycles). see docs/internals/tools/lab.md#read-step
#include "lab.cuh"

#include <utility>

namespace read_step {

template <class F, int... I>
__device__ __forceinline__ void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}

//: `f(std::integral_constant<int, i>)` for i in [0, N), unrolled
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

constexpr int kThreads = 256, kWarps = 8, kSlots = 128, kRowBytes = 96, kPanelPitch = 17;
//: the conflict-free panel (mode 12): 20 words a row, so slots of distinct residue mod 8 land on disjoint 4-bank groups,
//: and 8 dummy rows past the slots for the pads' adds of zero
constexpr int kFreePitch = 20, kPanelWords = (kSlots + 8) * kFreePitch;
//: modes 22-23's f16 rows: a slot's 32 level-R halves, 64 bytes, the 16-byte chunks swizzled by (slot >> 1) & 3 so 8
//: slots of distinct residue mod 8 read by ldmatrix fall on 8 distinct bank groups
constexpr int kHalfRowBytes = 64;
constexpr int kDigits = 4, kMaxSteps = 8, kStep = 16, kTargets = 1 << 18;
constexpr uint16_t kPad = 0xffffu;
constexpr uint32_t kHalfMagic = 0x64646464u;
//: a warp's drain requests a pass: its 16 slots, two a request
constexpr int kShare = kSlots / kWarps / 2;

//: MODE 0 the steps alone (outputs folded into the sink); 1 + the panel adds; 2 + the drain in a burst at the pass's end
//: (one panel, a CTA barrier either side); 3 + the drain spread: two panels, a warp draining its share of the last
//: pass's panel a request after each step, under mbarriers (`complete` when every warp's adds are in, `drained` when
//: every warp's share is out). Modes 2 and 3 drain 17-word rows (a slot's 16 channel words and den) to a 17-word target
//: pitch, den a request of its own; 4 and 5 are 2 and 3 with the channel words alone to 64-byte rows (den carried
//: elsewhere), so a request is a token's half line each; 6 no den (the den pass carries it), each step's adds issued
//: after the next step's MMAs; 7 is 6 with the aligned burst drain; 8 (per digit) is 6 with each step's operands loaded
//: before the step ahead's MMAs, across digits; 9 is 8 with the token's level-S coefficient applied after the MMAs, in
//: the conversion (A the raw level-R bytes); 10 is 9 converting on the conversion unit (an f16 multiply by the token's
//: scale, then F2I) in place of the FMA pipe; 11 is 9 with each entry carrying its byte offsets (its row with the
//: swizzle folded in, its panel row, a pad bit), as the facts pass could emit them; 12 is 11 into the conflict-free
//: panel (20-word rows, each step's tokens of distinct slot residue mod 8, pads to their own dummy rows); 13 and 14 are
//: cost isolations of 12, their values wrong: 13 adds the f16 words unconverted, 14 adds nothing (the outputs folded);
//: 15 converts by as many dependent integer operations, 16 issues the conversion on values independent of the outputs
//: and adds the words raw, 17 converts half the values, 18 is 16 with 15's integer operations; 19 is 12 with each MMA
//: followed by one of the last step's output words, held in that order by a dependency (written in the source order
//: alone, ptxas compiled it to 12's SASS); 20 holds it both ways; 21 is 12 with the two warps of a scheduler taking
//: turns at their steps' MMAs (named barriers), each converting and adding while the other's MMAs run; 22 is 12 with
//: the level-R side native f16 (A loaded as it is, no unpack); 23 is 22 accumulating in fp32 (no conversion to fp32;
//: half rate on GeForce, so its time here is not the full-rate cards'); 24 is 12 with its panel drained in a burst at the
//: pass's end (every slot, two a warp request); 25 is 12 with each output word added straight to its token's row in
//: global memory (`red.global.add`, a pad's to a dummy row), no panel and no drain
constexpr int kModes = 26;
const char* kModeName[kModes] = {
    "steps alone",
    "+ panel adds",
    "+ drain, burst at the pass end",
    "+ drain, double panel, spread",
    "+ drain, burst, aligned rows, no den",
    "+ drain, spread, aligned rows, no den",
    "no den, adds a step behind",
    "no den, adds a step behind, + aligned burst drain",
    "no den, operands a step ahead, adds a step behind",
    "no den, operands ahead, adds behind, coefficient after the MMAs",
    "the same, converted by F2I",
    "the same as 9, entries as byte offsets",
    "the same as 11, conflict-free panel",
    "isolation: 12 without the conversion",
    "isolation: 12 without the adds",
    "isolation: 12 converting by as many dependent LOP3/PRMT",
    "isolation: 12 with the conversion independent of the outputs",
    "isolation: 12 converting half the values",
    "isolation: 16 with the integer operations of 15",
    "the same as 12, an MMA then an output word, held by a dependency",
    "the same, held both ways",
    "the same as 12, the scheduler's two warps taking turns at the MMAs",
    "the same as 12, the level-R side f16 (no unpack)",
    "the same as 22, accumulating in fp32 (half rate here)",
    "the same as 12, + its panel drained in a burst",
    "the same as 12, the adds to global rows (no panel, no drain)"};

//: FORM 0 per digit (a step = one digit's 16 tokens, its outputs added); 1 merged (a step = 16 union tokens, every
//: digit's MMAs into one accumulator, one add)
const char* kFormName[2] = {"per digit", "merged"};

struct alignas(128) Shared {
  uint8_t rows[kSlots * kRowBytes];
  uint32_t panel[2][kPanelWords];
  uint16_t entries[kWarps][kDigits][kMaxSteps][kStep];
  uint32_t offsets[kWarps][kDigits][kMaxSteps][kStep];
  uint32_t offsets_free[kWarps][kDigits][kMaxSteps][kStep];
  alignas(128) uint8_t rows_f16[kSlots * kHalfRowBytes];
  uint16_t offsets_f16[kWarps][kDigits][kMaxSteps][kStep];
  uint64_t complete[2], drained[2];
};

__device__ __forceinline__ int at(int r, int c) {
  return r * kRowBytes + 16 * (c ^ ((r >> 2) & 1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ void red_shared(uint32_t a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ void red_global(uint32_t* p, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

//: an f16 pair at the panel's unit to two s16 in a word
__device__ __forceinline__ uint32_t to_s16x2(uint32_t h) {
  const __half2 x = __hmul2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(64.0f));
  const int lo = __half2short_rn(__low2half(x)), hi = __half2short_rn(__high2half(x));
  return (static_cast<uint32_t>(lo) & 0xffffu) | (static_cast<uint32_t>(hi) << 16);
}

//: the same on the FMA pipe, not the quarter-rate conversion unit: each half to fp32, x 64 + 1.5 x 2^23 (the low 16
//: bits then the rounded integer, two's complement), the two low halves packed
__device__ __forceinline__ uint32_t to_s16x2_fma(uint32_t h) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return __byte_perm(__float_as_uint(fmaf(f.x, 64.0f, 12582912.0f)),
                     __float_as_uint(fmaf(f.y, 64.0f, 12582912.0f)), 0x5410);
}

struct State {
  uint32_t b[kDigits][2][4][2];
  uint32_t mass[kDigits][2][2];
};

//: A STEP'S A for 16 tokens (slots `ra` of the lane's ldmatrix row) at a digit: the level-R bytes to f16 at the
//: token's level-S coefficient (tokens g and g + 8 for the lane's rows)
__device__ __forceinline__ void step_a(uint32_t (&a)[2][4], const Shared& sh, int ra, int khalf,
                                       int rg, int rg8, int digit, bool in_g, bool in_g8) {
  uint32_t x[4];
  lab::ldsm(x, lab::smem(sh.rows + at(ra, khalf)));
  //: loaded whatever the token (a pad's slot is 0, a real row), the value masked: no branch around the load
  const float sg = (1.0f + sh.rows[at(rg, 2) + digit]) * (in_g ? 1.0f / 65536.0f : 0.0f);
  const float s8 = (1.0f + sh.rows[at(rg8, 2) + digit]) * (in_g8 ? 1.0f / 65536.0f : 0.0f);
  const uint32_t scale_g = lab::pack_half2(sg, sg), scale_g8 = lab::pack_half2(s8, s8);
  const uint32_t bias_g = lab::pack_half2(-1024.0f * sg, -1024.0f * sg);
  const uint32_t bias_g8 = lab::pack_half2(-1024.0f * s8, -1024.0f * s8);
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    a[kk][0] = hfma2(prmt(x[2 * kk], kHalfMagic, 0x4140u), scale_g, bias_g);
    a[kk][1] = hfma2(prmt(x[2 * kk + 1], kHalfMagic, 0x4140u), scale_g8, bias_g8);
    a[kk][2] = hfma2(prmt(x[2 * kk], kHalfMagic, 0x4342u), scale_g, bias_g);
    a[kk][3] = hfma2(prmt(x[2 * kk + 1], kHalfMagic, 0x4342u), scale_g8, bias_g8);
  }
}

//: A STEP'S A AS THE RAW LEVEL-R BYTES (each byte b as f16 1024 + b by PRMT, less 1024), and the lane's two tokens'
//: level-S coefficients times the panel's unit as the conversion's scales, a pad's scale zero
__device__ __forceinline__ void step_a_raw(uint32_t (&a)[2][4], float& sg, float& s8,
                                           const Shared& sh, int ra, int khalf, int rg, int rg8,
                                           int digit, bool in_g, bool in_g8) {
  uint32_t x[4];
  lab::ldsm(x, lab::smem(sh.rows + at(ra, khalf)));
  const uint32_t less = lab::pack_half2(-1024.0f, -1024.0f);
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    a[kk][0] = hadd2(prmt(x[2 * kk], kHalfMagic, 0x4140u), less);
    a[kk][1] = hadd2(prmt(x[2 * kk + 1], kHalfMagic, 0x4140u), less);
    a[kk][2] = hadd2(prmt(x[2 * kk], kHalfMagic, 0x4342u), less);
    a[kk][3] = hadd2(prmt(x[2 * kk + 1], kHalfMagic, 0x4342u), less);
  }
  sg = (1.0f + sh.rows[at(rg, 2) + digit]) * (in_g ? 64.0f / 65536.0f : 0.0f);
  s8 = (1.0f + sh.rows[at(rg8, 2) + digit]) * (in_g8 ? 64.0f / 65536.0f : 0.0f);
}

//: A STEP'S A AND SCALES FROM ITS ENTRIES' OFFSETS (`ea` the lane's ldmatrix row, `eg`/`e8` its two tokens): an
//: address an XOR or an add
template <int DIGIT>
__device__ __forceinline__ void step_a_offsets(uint32_t (&a)[2][4], float& sg, float& s8,
                                               const Shared& sh, uint32_t ea, uint32_t eg,
                                               uint32_t e8, int khalf) {
  uint32_t x[4];
  lab::ldsm(x, lab::smem(sh.rows + ((ea & 0xffffu) ^ (static_cast<uint32_t>(khalf) << 4))));
  const uint32_t less = lab::pack_half2(-1024.0f, -1024.0f);
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    a[kk][0] = hadd2(prmt(x[2 * kk], kHalfMagic, 0x4140u), less);
    a[kk][1] = hadd2(prmt(x[2 * kk + 1], kHalfMagic, 0x4140u), less);
    a[kk][2] = hadd2(prmt(x[2 * kk], kHalfMagic, 0x4342u), less);
    a[kk][3] = hadd2(prmt(x[2 * kk + 1], kHalfMagic, 0x4342u), less);
  }
  sg = (1.0f + sh.rows[(eg & 0xffffu) + 32 + DIGIT]) * ((eg >> 31) ? 0.0f : 64.0f / 65536.0f);
  s8 = (1.0f + sh.rows[(e8 & 0xffffu) + 32 + DIGIT]) * ((e8 >> 31) ? 0.0f : 64.0f / 65536.0f);
}

//: A STEP'S A FROM THE F16 ROWS (`ea` the lane's row with its swizzle key, from `offsets_f16`; two ldmatrix, no
//: unpack) and its scales as `step_a_offsets`'s
template <int DIGIT>
__device__ __forceinline__ void step_a_f16(uint32_t (&a)[2][4], float& sg, float& s8,
                                           const Shared& sh, uint32_t ea, uint32_t eg, uint32_t e8,
                                           int khalf) {
  const uint32_t row = lab::smem(sh.rows_f16) + ea;
#pragma unroll
  for (int kk = 0; kk < 2; ++kk)
    lab::ldsm(a[kk], row ^ (16u * static_cast<uint32_t>(2 * kk + khalf)));
  sg = (1.0f + sh.rows[(eg & 0xffffu) + 32 + DIGIT]) * ((eg >> 31) ? 0.0f : 64.0f / 65536.0f);
  s8 = (1.0f + sh.rows[(e8 & 0xffffu) + 32 + DIGIT]) * ((e8 >> 31) ? 0.0f : 64.0f / 65536.0f);
}

//: the step's rows: the lane's ldmatrix row (`8 (q & 1) + r` of the step's 16, k half `q >> 1`) and its two output tokens
__device__ __forceinline__ int slot_of(const Shared& sh, int w, int d, int s, int e, bool& in) {
  const uint16_t x = sh.entries[w][d][s][e];
  in = x != kPad;
  return in ? x : 0;
}

template <int D0, int DIGITS, bool kDen>
__device__ __forceinline__ void mmas(uint32_t (&out)[4][2], uint32_t (&den)[2], const State& st,
                                     const uint32_t (&a)[2][4]) {
  constexpr int d0 = D0;
#pragma unroll
  for (int dd = 0; dd < DIGITS; ++dd)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int n = 0; n < 4; ++n)
        lab::mma_f16_f16(out[n], a[kk], st.b[d0 + dd][kk][n][0], st.b[d0 + dd][kk][n][1]);
      if constexpr (kDen)
        lab::mma_f16_f16(den, a[kk], st.mass[d0 + dd][kk][0], st.mass[d0 + dd][kk][1]);
    }
}

//: THE STEP'S OUTPUTS into panel `p`: lane (g, tq) holds tokens g and g + 8, channels 8 n + 2 tq, + 1 -- a word each;
//: den by the tq = 0 lanes
template <bool kDen>
__device__ __forceinline__ void panel_add(Shared& sh, int p, const uint32_t (&out)[4][2],
                                          const uint32_t (&den)[2], int slot_g, int slot_g8,
                                          bool in_g, bool in_g8, int lane) {
  const int tq = lane & 3;
  const uint32_t base = lab::smem(sh.panel[p]);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if (in_g) red_shared(base + 4 * (slot_g * kPanelPitch + 4 * n + tq), to_s16x2(out[n][0]));
    if (in_g8) red_shared(base + 4 * (slot_g8 * kPanelPitch + 4 * n + tq), to_s16x2(out[n][1]));
  }
  if (kDen && tq == 0) {
    if (in_g) red_shared(base + 4 * (slot_g * kPanelPitch + 16), to_s16x2(den[0]));
    if (in_g8) red_shared(base + 4 * (slot_g8 * kPanelPitch + 16), to_s16x2(den[1]));
  }
}

//: THE STEP'S OUTPUTS into panel `p` without den, every lane adding: a pad token's value masked to zero at slot 0
//: (no branch)
__device__ __forceinline__ void panel_add_masked(Shared& sh, int p, const uint32_t (&out)[4][2],
                                                 int slot_g, int slot_g8, bool in_g, bool in_g8,
                                                 int lane) {
  const int tq = lane & 3;
  const uint32_t base = lab::smem(sh.panel[p]);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    red_shared(base + 4 * (slot_g * kPanelPitch + 4 * n + tq),
               to_s16x2_fma(out[n][0]) & (0u - in_g));
    red_shared(base + 4 * (slot_g8 * kPanelPitch + 4 * n + tq),
               to_s16x2_fma(out[n][1]) & (0u - in_g8));
  }
}

//: THE STEP'S OUTPUTS into panel `p` at each token's scale (its coefficient times the unit; zero for a pad, so its
//: adds are of zero): each half to fp32, x scale + 1.5 x 2^23, the low halves packed
template <bool kF2i>
__device__ __forceinline__ void panel_add_scaled(Shared& sh, int p, const uint32_t (&out)[4][2],
                                                 int slot_g, int slot_g8, float sg, float s8,
                                                 int lane) {
  const int tq = lane & 3;
  const uint32_t base = lab::smem(sh.panel[p]);
  const auto cvt = [](uint32_t h, float sc) {
    if constexpr (kF2i) {
      const __half2 x = __hmul2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(sc));
      const int lo = __half2short_rn(__low2half(x)), hi = __half2short_rn(__high2half(x));
      return (static_cast<uint32_t>(lo) & 0xffffu) | (static_cast<uint32_t>(hi) << 16);
    } else {
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
      return __byte_perm(__float_as_uint(fmaf(f.x, sc, 12582912.0f)),
                         __float_as_uint(fmaf(f.y, sc, 12582912.0f)), 0x5410);
    }
  };
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    red_shared(base + 4 * (slot_g * kPanelPitch + 4 * n + tq), cvt(out[n][0], sg));
    red_shared(base + 4 * (slot_g8 * kPanelPitch + 4 * n + tq), cvt(out[n][1], s8));
  }
}

__device__ __forceinline__ uint32_t lop3v(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t prmtv(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: a token's row in panel `p` from its entry's offset, at the lane's channel pair
__device__ __forceinline__ uint32_t panel_row(Shared& sh, int p, uint32_t e, int lane) {
  return lab::smem(sh.panel[p]) + 4u * static_cast<uint32_t>(lane & 3) + ((e >> 16) & 0x7fffu);
}

//: AN OUTPUT WORD (an f16 pair, n-tile `n`) AS ITS S16 PAIR at scale `sc`: VAR 0 on the FMA pipe; the cost isolations
//: (values wrong): 1 the word raw; 2 as many dependent integer operations (LOP3, PRMT: the cheapest beside an MMA
//: chain); 3 the word XORed with the conversion computed on values independent of the outputs; 4 n-tiles 0-1
//: converted, 2-3 raw; 5 is 3 with 2's integer operations
template <int VAR>
__device__ __forceinline__ uint32_t convert(uint32_t h, float sc, int n) {
  const auto fma_cvt = [](uint32_t x, float s) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&x));
    return __byte_perm(__float_as_uint(fmaf(f.x, s, 12582912.0f)),
                       __float_as_uint(fmaf(f.y, s, 12582912.0f)), 0x5410);
  };
  const auto int_cvt = [](uint32_t x, uint32_t b) {
    uint32_t t = lop3v(x, b, 0x55555555u);
    t = prmtv(t, x);
    t = lop3v(t, b, 0x0f0f0f0fu);
    t = lop3v(t, x, 0x33333333u);
    return prmtv(t, b);
  };
  const uint32_t b = __float_as_uint(sc), off = b ^ (0x3c003c00u + 0x00010001u * n);
  if constexpr (VAR == 0) return fma_cvt(h, sc);
  if constexpr (VAR == 1) return h ^ b;
  if constexpr (VAR == 2) return int_cvt(h, b);
  if constexpr (VAR == 3) return h ^ fma_cvt(off, sc);
  if constexpr (VAR == 5) return h ^ int_cvt(off, b);
  return n < 2 ? fma_cvt(h, sc) : h ^ b;
}

//: THE STEP'S OUTPUTS into panel `p` at the tokens' panel rows from their entries' offsets
template <int VAR>
__device__ __forceinline__ void panel_add_offsets(Shared& sh, int p, const uint32_t (&out)[4][2],
                                                  uint32_t eg, uint32_t e8, float sg, float s8,
                                                  int lane) {
  const uint32_t rg = panel_row(sh, p, eg, lane), r8 = panel_row(sh, p, e8, lane);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    red_shared(rg + 16 * n, convert<VAR>(out[n][0], sg, n));
    red_shared(r8 + 16 * n, convert<VAR>(out[n][1], s8, n));
  }
}

//: THE STEP'S FP32 OUTPUTS into panel `p` (mode 23): lane (g, tq) holds tokens g (c0, c1) and g + 8 (c2, c3)
__device__ __forceinline__ void panel_add_f32(Shared& sh, int p, const float (&out)[4][4],
                                              uint32_t eg, uint32_t e8, float sg, float s8,
                                              int lane) {
  const uint32_t rg = panel_row(sh, p, eg, lane), r8 = panel_row(sh, p, e8, lane);
  const auto pack = [](float x, float y, float sc) {
    return __byte_perm(__float_as_uint(fmaf(x, sc, 12582912.0f)),
                       __float_as_uint(fmaf(y, sc, 12582912.0f)), 0x5410);
  };
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    red_shared(rg + 16 * n, pack(out[n][0], out[n][1], sg));
    red_shared(r8 + 16 * n, pack(out[n][2], out[n][3], s8));
  }
}

//: ONE DRAIN REQUEST: slots `2 i` and `2 i + 1` of the warp's share of panel `p`, a lane a word (lanes 0-15 the first,
//: 16-31 the second; den by lanes 0 and 16 after), to the slots' scattered targets, each word cleared
template <bool kAligned>
__device__ __forceinline__ void drain_request(Shared& sh, int p, uint32_t* target, int pass, int i,
                                              int lane) {
  const int w = lab::warp_id(), slot = (kSlots / kWarps) * w + 2 * i + (lane >> 4),
            word = lane & 15;
  const uint32_t token =
      (static_cast<uint32_t>(blockIdx.x) * 7919u + pass * 131u + slot * 37u) % kTargets;
  uint32_t* const at = &sh.panel[p][slot * kPanelPitch + word];
  red_global(target + token * (kAligned ? 16u : 17u) + word, *at);
  *at = 0u;
  if (!kAligned && word == 0) {
    uint32_t* const d = &sh.panel[p][slot * kPanelPitch + 16];
    red_global(target + token * 17u + 16, *d);
    *d = 0u;
  }
}

//: A TOKEN'S ROW in global memory (modes 24-25), from its panel row's offset and the pass: 4,096 rows every CTA shares,
//: L2-resident, as a window's tokens are to its boxes
constexpr uint32_t kSharedRows = 4096u;

__device__ __forceinline__ uint32_t token_of(int pass, uint32_t panel_offset) {
  return (panel_offset + static_cast<uint32_t>(pass) * 131u) & (kSharedRows - 1u);
}

//: ONE DRAIN REQUEST of the conflict-free panel (mode 24): slots `2 i` and `2 i + 1` of the warp's share, a lane a
//: word, to their 16-word rows, each word cleared
__device__ __forceinline__ void drain_free_request(Shared& sh, uint32_t* target, int pass, int i,
                                                   int lane) {
  const int w = lab::warp_id(), slot = (kSlots / kWarps) * w + 2 * i + (lane >> 4),
            word = lane & 15;
  uint32_t* const at = &sh.panel[0][slot * kFreePitch + word];
  red_global(target + token_of(pass, static_cast<uint32_t>(slot) * kFreePitch * 4u) * 16u + word,
             *at);
  *at = 0u;
}

//: THE STEP'S OUTPUTS STRAIGHT TO GLOBAL ROWS (mode 25): each token's row from its entry's panel offset, the lane's
//: channel pair, as `panel_add_offsets` would add them to the panel
__device__ __forceinline__ void global_add_offsets(uint32_t* target, int pass,
                                                   const uint32_t (&out)[4][2], uint32_t eg,
                                                   uint32_t e8, float sg, float s8, int lane) {
  const uint32_t lg = token_of(pass, (eg >> 16) & 0x7fffu) * 16u + static_cast<uint32_t>(lane & 3),
                 l8 = token_of(pass, (e8 >> 16) & 0x7fffu) * 16u + static_cast<uint32_t>(lane & 3);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    red_global(target + lg + 4 * n, convert<0>(out[n][0], sg, n));
    red_global(target + l8 + 4 * n, convert<0>(out[n][1], s8, n));
  }
}

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

template <int MODE, int FORM, int STEPS, int LIVE>
__global__ void __launch_bounds__(kThreads, 1) read(int passes, const uint16_t* entries,
                                                    uint32_t* target, const uint32_t* init,
                                                    long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int lane = lab::lane_id(), w = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  const int r = lane & 7, q = lane >> 3, g = lane >> 2;
#pragma unroll 1
  for (int i = tid; i < kSlots * kRowBytes / 4; i += kThreads)
    reinterpret_cast<uint32_t*>(sh.rows)[i] = 0x01010101u * static_cast<uint32_t>(1 + (i & 63));
#pragma unroll 1
  for (int i = tid; i < 2 * kPanelWords; i += kThreads) (&sh.panel[0][0])[i] = 0u;
#pragma unroll 1
  for (int i = tid; i < kWarps * kDigits * kMaxSteps * kStep; i += kThreads) {
    (&sh.entries[0][0][0][0])[i] = entries[i];
    //: an entry's offsets: its row's byte offset at k half 0 (the swizzle folded in), its panel row's, a pad bit
    const uint32_t slot = entries[i] == kPad ? 0u : entries[i];
    (&sh.offsets[0][0][0][0])[i] = static_cast<uint32_t>(at(static_cast<int>(slot), 0))
                                   | (slot * kPanelPitch * 4u) << 16
                                   | (entries[i] == kPad ? 1u << 31 : 0u);
    //: the conflict-free panel's: a pad's adds go to dummy row 128 + its position mod 8
    const uint32_t row = entries[i] == kPad ? kSlots + (i % kStep & 7) : slot;
    (&sh.offsets_free[0][0][0][0])[i] = static_cast<uint32_t>(at(static_cast<int>(slot), 0))
                                        | (row * kFreePitch * 4u) << 16
                                        | (entries[i] == kPad ? 1u << 31 : 0u);
    (&sh.offsets_f16[0][0][0][0])[i] =
        static_cast<uint16_t>(slot * kHalfRowBytes | 16u * ((slot >> 1) & 3u));
  }
#pragma unroll 1
  for (int i = tid; i < kSlots * kHalfRowBytes / 2; i += kThreads)
    reinterpret_cast<__half*>(sh.rows_f16)[i] = __float2half(static_cast<float>(1 + (i & 63)));
  if (tid < 2) {
    lab::mbar_init(lab::smem(&sh.complete[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.drained[tid]), kThreads);
  }
  __syncthreads();
  //: the fold's accumulators, live across the readout as in the owner kernel, and the published slice -- both loaded
  //: from memory, so neither is rematerialized inside the loop
  const uint32_t* const mine = init + (static_cast<size_t>(blockIdx.x) * kThreads + tid) * 208;
  float fold[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) fold[i] = __uint_as_float(mine[i]);
  State st;
#pragma unroll
  for (int d = 0; d < kDigits; ++d)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        st.b[d][kk][n][0] = mine[128 + ((d * 2 + kk) * 4 + n) * 2];
        st.b[d][kk][n][1] = mine[128 + ((d * 2 + kk) * 4 + n) * 2 + 1];
      }
      st.mass[d][kk][0] = mine[192 + (d * 2 + kk) * 2];
      st.mass[d][kk][1] = mine[192 + (d * 2 + kk) * 2 + 1];
    }
  uint32_t fold_out = 0u;
  //: mode 21's turns: warps w and w + 4 share a scheduler and take turns issuing a step's MMAs, w first
  [[maybe_unused]] const int turn = 1 + 2 * (w & 3) + (w >> 2),
                             yield = 1 + 2 * (w & 3) + 1 - (w >> 2);
  if (MODE == 21 && w >= 4) lab::bar_arrive(yield, 64);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    constexpr bool kSpread = MODE == 3 || MODE == 5,
                   kBurst = MODE == 2 || MODE == 4 || MODE == 7 || MODE == 24, kAligned = MODE >= 4,
                   kDen = MODE < 6, kBehind = MODE >= 6;
    const int p = kSpread ? (x & 1) : 0, other = p ^ 1;
    if constexpr (kSpread) {
      if (x >= 2)
        lab::mbar_wait(lab::smem(&sh.drained[p]), static_cast<uint32_t>((x - 2) >> 1) & 1u);
    }
    int drained_requests = 0;
    [[maybe_unused]] uint32_t held[4][2] = {};
    [[maybe_unused]] int held_g = 0, held_g8 = 0;
    [[maybe_unused]] bool held_in_g = false, held_in_g8 = false, holding = false;
    const auto spread = [&]() {
      if constexpr (kSpread) {
        if (x >= 1 && drained_requests < kShare) {
          if (drained_requests == 0)
            lab::mbar_wait(lab::smem(&sh.complete[other]),
                           static_cast<uint32_t>((x - 1) >> 1) & 1u);
          drain_request<kAligned>(sh, other, target, x - 1, drained_requests, lane);
          ++drained_requests;
        }
      }
    };
    const auto one_step = [&](auto digit0, int s, auto digits) {
      constexpr int d0 = decltype(digit0)::value, kDigitsHere = decltype(digits)::value;
      bool in_a, in_g, in_g8;
      const int ra = slot_of(sh, w, d0, s, 8 * (q & 1) + r, in_a);
      const int rg = slot_of(sh, w, d0, s, g, in_g), rg8 = slot_of(sh, w, d0, s, g + 8, in_g8);
      uint32_t a[2][4];
      step_a(a, sh, ra, q >> 1, rg, rg8, d0, in_g, in_g8);
      uint32_t o[4][2] = {}, den[2] = {};
      mmas<d0, kDigitsHere, kDen>(o, den, st, a);
      if constexpr (MODE == 0) {
#pragma unroll
        for (int n = 0; n < 4; ++n) fold_out += o[n][0] ^ o[n][1];
        fold_out += den[0] ^ den[1];
      } else if constexpr (kBehind) {
        //: the last step's adds after this step's MMAs, this step's held for the next
        if (holding)
          panel_add<false>(sh, p, held, den, held_g, held_g8, held_in_g, held_in_g8, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          held[n][0] = o[n][0];
          held[n][1] = o[n][1];
        }
        held_g = rg;
        held_g8 = rg8;
        held_in_g = in_g;
        held_in_g8 = in_g8;
        holding = true;
      } else {
        panel_add<kDen>(sh, p, o, den, rg, rg8, in_g, in_g8, lane);
      }
      spread();
    };
    if constexpr (MODE >= 11) {
      //: mode 9's sequence from the entries' offsets
      constexpr int kItems = kDigits * STEPS;
      const uint32_t* const off =
          MODE >= 12 ? &sh.offsets_free[w][0][0][0] : &sh.offsets[w][0][0][0];
      const auto entry = [&](int item, int e) {
        return off[((item / STEPS) * kMaxSteps + item % STEPS) * kStep + e];
      };
      uint32_t a[2][2][4], o[2][4][2], eg[3] = {}, e8[3] = {};
      [[maybe_unused]] float o32[2][4][4];
      float sg[3] = {}, s8[3] = {};
      constexpr bool kHalfRows = MODE == 22 || MODE == 23;
      //: the step's A and scales, from the u8 rows or (22, 23) the f16 rows
      const auto load = [&](auto item, uint32_t(&an)[2][4], auto slot) {
        constexpr int it = decltype(item)::value, m = decltype(slot)::value;
        eg[m] = entry(it, g);
        e8[m] = entry(it, g + 8);
        if constexpr (kHalfRows)
          step_a_f16<it / STEPS>(
              an, sg[m], s8[m], sh,
              (&sh.offsets_f16[w][0][0][0])[((it / STEPS) * kMaxSteps + it % STEPS) * kStep
                                            + 8 * (q & 1) + r],
              eg[m], e8[m], q >> 1);
        else
          step_a_offsets<it / STEPS>(an, sg[m], s8[m], sh, entry(it, 8 * (q & 1) + r), eg[m], e8[m],
                                     q >> 1);
      };
      //: mode 19's order words, zero at run time (`passes` > 0) but not to ptxas
      [[maybe_unused]] uint32_t order[8] = {};
      [[maybe_unused]] const uint32_t zero = static_cast<uint32_t>(passes) >> 31;
      load(std::integral_constant<int, 0>{}, a[0], std::integral_constant<int, 0>{});
      static_for<kItems + 1>([&](auto I) {
        constexpr int i = decltype(I)::value, c = i & 1, n = c ^ 1, mn = (i + 1) % 3,
                      mp = (i + 2) % 3;
        if constexpr (i + 1 < kItems)
          load(std::integral_constant<int, i + 1>{}, a[n], std::integral_constant<int, mn>{});
        if constexpr (MODE == 19 || MODE == 20) {
          //: MMA j, then the last step's output word j; MMA j reads word j - 1 masked to zero, which keeps it there, and
          //: in 20 word j reads MMA j - 2's output so masked as well
          static_for<8>([&](auto J) {
            constexpr int j = decltype(J)::value;
            [[maybe_unused]] constexpr int kk = j >> 2, t = j & 3;
            if constexpr (i < kItems) {
              const uint32_t d = order[(j + 7) & 7];
              if constexpr (kk == 0) {
                o[c][t][0] = d;
                o[c][t][1] = 0u;
              } else {
                o[c][t][0] ^= d;
              }
              lab::mma_f16_f16(o[c][t], a[c][kk], st.b[i / STEPS][kk][t][0],
                               st.b[i / STEPS][kk][t][1]);
            }
            if constexpr (i > 0) {
              uint32_t h = o[n][j >> 1][j & 1];
              if constexpr (MODE == 20 && i < kItems && j >= 2) h ^= o[c][(j - 2) & 3][0] & zero;
              if constexpr (MODE == 20 && i < kItems && j < 2) h ^= o[n][(j + 6) & 3][1] & zero;
              const uint32_t word = convert<0>(h, (j & 1) ? s8[mp] : sg[mp], j >> 1);
              red_shared(panel_row(sh, p, (j & 1) ? e8[mp] : eg[mp], lane) + 16 * (j >> 1), word);
              order[j] = word & zero;
            }
          });
        } else if constexpr (MODE == 23 && i < kItems) {
#pragma unroll
          for (int t = 0; t < 4; ++t)
            o32[c][t][0] = o32[c][t][1] = o32[c][t][2] = o32[c][t][3] = 0.0f;
#pragma unroll
          for (int kk = 0; kk < 2; ++kk)
#pragma unroll
            for (int t = 0; t < 4; ++t)
              lab::mma_f16_f32(o32[c][t], a[c][kk], st.b[i / STEPS][kk][t][0],
                               st.b[i / STEPS][kk][t][1]);
        } else if constexpr (i < kItems) {
#pragma unroll
          for (int x4 = 0; x4 < 4; ++x4) o[c][x4][0] = o[c][x4][1] = 0u;
          uint32_t den[2] = {};
          if constexpr (MODE == 21) lab::bar_sync(turn, 64);
          mmas<i / STEPS, 1, false>(o[c], den, st, a[c]);
          if constexpr (MODE == 21) lab::bar_arrive(yield, 64);
        }
        if constexpr (i > 0 && MODE == 14) {
#pragma unroll
          for (int x4 = 0; x4 < 4; ++x4) fold_out += o[n][x4][0] ^ o[n][x4][1];
        } else if constexpr (i > 0 && MODE == 23) {
          panel_add_f32(sh, p, o32[n], eg[mp], e8[mp], sg[mp], s8[mp], lane);
        } else if constexpr (i > 0 && MODE == 25) {
          global_add_offsets(target, x, o[n], eg[mp], e8[mp], sg[mp], s8[mp], lane);
        } else if constexpr (i > 0 && MODE != 19 && MODE != 20) {
          constexpr int kVar = MODE == 13   ? 1
                               : MODE == 15 ? 2
                               : MODE == 16 ? 3
                               : MODE == 17 ? 4
                               : MODE == 18 ? 5
                                            : 0;
          panel_add_offsets<kVar>(sh, p, o[n], eg[mp], e8[mp], sg[mp], s8[mp], lane);
        }
      });
    } else if constexpr (MODE == 9 || MODE == 10) {
      //: mode 8's sequence with A the raw bytes and each token's coefficient applied in the conversion
      const auto load = [&](int dd, int ss, uint32_t(&an)[2][4], int& rg, int& rg8, float& sg,
                            float& s8) {
        bool ia, ig, ig8;
        const int ra = slot_of(sh, w, dd, ss, 8 * (q & 1) + r, ia);
        rg = slot_of(sh, w, dd, ss, g, ig);
        rg8 = slot_of(sh, w, dd, ss, g + 8, ig8);
        step_a_raw(an, sg, s8, sh, ra, q >> 1, rg, rg8, dd, ig, ig8);
      };
      constexpr int kItems = kDigits * STEPS;
      uint32_t a[2][2][4], o[2][4][2];
      int rg[3] = {}, rg8[3] = {};
      float sg[3] = {}, s8[3] = {};
      load(0, 0, a[0], rg[0], rg8[0], sg[0], s8[0]);
      static_for<kItems + 1>([&](auto I) {
        constexpr int i = decltype(I)::value, c = i & 1, n = c ^ 1, mn = (i + 1) % 3,
                      mp = (i + 2) % 3;
        if constexpr (i + 1 < kItems)
          load((i + 1) / STEPS, (i + 1) % STEPS, a[n], rg[mn], rg8[mn], sg[mn], s8[mn]);
        if constexpr (i < kItems) {
          uint32_t den[2] = {};
#pragma unroll
          for (int x4 = 0; x4 < 4; ++x4) o[c][x4][0] = o[c][x4][1] = 0u;
          mmas<i / STEPS, 1, false>(o[c], den, st, a[c]);
        }
        if constexpr (i > 0)
          panel_add_scaled<MODE == 10>(sh, p, o[n], rg[mp], rg8[mp], sg[mp], s8[mp], lane);
      });
    } else if constexpr (MODE == 8) {
      //: the warp's digit steps as one compile-time sequence, unrolled: rings of operands and outputs (two slots) and of
      //: token slots (three: the next step's are loaded while the last step's still add) at fixed registers -- no
      //: copies, no branches; the next step's operands, this step's MMAs, the last step's adds
      const auto load = [&](int dd, int ss, uint32_t(&an)[2][4], int& rg, int& rg8, bool& ig,
                            bool& ig8) {
        bool ia;
        const int ra = slot_of(sh, w, dd, ss, 8 * (q & 1) + r, ia);
        rg = slot_of(sh, w, dd, ss, g, ig);
        rg8 = slot_of(sh, w, dd, ss, g + 8, ig8);
        step_a(an, sh, ra, q >> 1, rg, rg8, dd, ig, ig8);
      };
      constexpr int kItems = kDigits * STEPS;
      uint32_t a[2][2][4], o[2][4][2];
      int rg[3] = {}, rg8[3] = {};
      bool ig[3] = {}, ig8[3] = {};
      load(0, 0, a[0], rg[0], rg8[0], ig[0], ig8[0]);  // item i's slots in ring position i mod 3
      static_for<kItems + 1>([&](auto I) {
        constexpr int i = decltype(I)::value, c = i & 1, n = c ^ 1, mn = (i + 1) % 3,
                      mp = (i + 2) % 3;
        if constexpr (i + 1 < kItems)
          load((i + 1) / STEPS, (i + 1) % STEPS, a[n], rg[mn], rg8[mn], ig[mn], ig8[mn]);
        if constexpr (i < kItems) {
          uint32_t den[2] = {};
#pragma unroll
          for (int x4 = 0; x4 < 4; ++x4) o[c][x4][0] = o[c][x4][1] = 0u;
          mmas<i / STEPS, 1, false>(o[c], den, st, a[c]);
        }
        if constexpr (i > 0) panel_add_masked(sh, p, o[n], rg[mp], rg8[mp], ig[mp], ig8[mp], lane);
      });
    } else if constexpr (FORM == 0) {
      static_for<kDigits>([&](auto D) {
#pragma unroll 1
        for (int s = 0; s < STEPS; ++s) one_step(D, s, std::integral_constant<int, 1>{});
      });
    } else {
#pragma unroll 1
      for (int s = 0; s < STEPS; ++s)
        one_step(std::integral_constant<int, 0>{}, s, std::integral_constant<int, kDigits>{});
    }
    if constexpr (kBehind) {
      const uint32_t none[2] = {};
      if (holding)
        panel_add<false>(sh, p, held, none, held_g, held_g8, held_in_g, held_in_g8, lane);
    }
    if constexpr (kSpread) {
#pragma unroll 1
      while (x >= 1 && drained_requests < kShare) spread();
      if (x >= 1) lab::mbar_arrive(lab::smem(&sh.drained[other]));
      lab::mbar_arrive(lab::smem(&sh.complete[p]));
    }
    if constexpr (kBurst) {
      __syncthreads();
#pragma unroll 1
      for (int i = 0; i < kShare; ++i) {
        if constexpr (MODE == 24) {
          drain_free_request(sh, target, x, i, lane);
        } else {
          drain_request<kAligned>(sh, 0, target, x, i, lane);
        }
      }
      __syncthreads();
    }
  }
  if constexpr (MODE == 3 || MODE == 5) {
    //: the last pass's panel, exposed
    lab::mbar_wait(lab::smem(&sh.complete[(passes - 1) & 1]),
                   static_cast<uint32_t>((passes - 1) >> 1) & 1u);
#pragma unroll 1
    for (int i = 0; i < kShare; ++i)
      drain_request<MODE >= 4>(sh, (passes - 1) & 1, target, passes - 1, i, lane);
  }
  //: the second warp's last turn back, so every named barrier ends complete
  if (MODE == 21 && w < 4) lab::bar_sync(turn, 64);
  clock.stop(t);
  uint32_t f = fold_out;
#pragma unroll
  for (int i = 0; i < 128; ++i) f += lab::bits(fold[i]);
  lab::sink(out, f);
}

//: THE ENTRIES: per warp, digit and step, `LIVE` slots then pads -- per digit distinct gathered slots (a hashed start,
//: then 37 apart over the 128: a list never holds a token twice); merged
//: the step's own 16 (digit 0's entries, the union)
std::vector<uint16_t> make_entries(int form, int live) {
  std::vector<uint16_t> e(kWarps * kDigits * kMaxSteps * kStep);
  for (int w = 0; w < kWarps; ++w)
    for (int d = 0; d < kDigits; ++d)
      for (int s = 0; s < kMaxSteps; ++s)
        for (int i = 0; i < kStep; ++i) {
          const size_t at = ((static_cast<size_t>(w) * kDigits + d) * kMaxSteps + s) * kStep + i;
          if (i >= live)
            e[at] = kPad;
          else if (form == 0)
            e[at] = static_cast<uint16_t>((hash3(w, d * kMaxSteps + s, 0) + 37u * i) % kSlots);
          else
            e[at] = static_cast<uint16_t>((s * kStep + i) % kSlots);
        }
  return e;
}

constexpr int kPasses = 400;

template <int MODE, int FORM, int STEPS, int LIVE>
void variant(const lab::Bench& b, uint32_t* target, const uint32_t* init) {
  const std::string name = lab::kernel("read", {MODE, FORM, STEPS, LIVE});
  b.expect(name, MODE == 23 ? "HMMA.16816.F32" : "HMMA.16816.F16", ">=", MODE >= 6 ? 8 : 10);
  if (MODE == 23) b.expect(name, "HADD2.F32", "==", 0);
  b.expect(name, "LDSM", ">=", 1);
  if (MODE != 0 && MODE != 14 && MODE != 25) b.expect(name, "ATOMS", ">=", 8);
  if (MODE == 25) b.expect(name, "ATOMS", "==", 0);
  if ((MODE >= 2 && MODE != 6 && MODE < 8) || MODE == 24) b.expect(name, "RED", ">=", 1);
  if (MODE == 25) b.expect(name, "RED", ">=", 8);
  //: every form here converts in its step as the readout of its day did (the rule came after), and 25 adds to global
  //: memory in its step by design
  b.expect(name, "STEP_XU", ">=", 0);
  if (MODE == 25) b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (MODE >= 1 && MODE <= 7) b.expect(name, "BSSY", ">=", 1);
  if (b.expect_only) return;
  const std::vector<uint16_t> h = make_entries(FORM, LIVE);
  uint16_t* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, h.size() * sizeof(uint16_t)));
  LAB_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(read<MODE, FORM, STEPS, LIVE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] {
        read<MODE, FORM, STEPS, LIVE>
            <<<b.sms, kThreads, bytes>>>(kPasses, d, target, init, b.times, b.sinks);
      },
      [&] { return b.span(kWarps) / kPasses; });
  //: a warp's MMAs a pass at 16 cycles, two warps a scheduler: per digit 4 digits x STEPS steps x 10 (8 without den);
  //: merged STEPS x 40 (32)
  const double per = MODE >= 6 ? 8.0 : 10.0;
  const double mmas = FORM == 0 ? kDigits * STEPS * per : STEPS * 4 * per;
  const double floor = 2.0 * mmas * 16;
  const std::string label = std::string(kFormName[FORM]) + " / " + std::to_string(STEPS)
                            + (FORM == 0 ? " steps a digit" : " steps") + " / "
                            + std::to_string(LIVE) + " live / " + kModeName[MODE];
  b.record(label, "cycles a pass", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / floor);
  b.record(label + " / floor", "x", over);
  LAB_CHECK(cudaFree(d));
}

template <int FORM, int STEPS, int LIVE>
void modes(const lab::Bench& b, uint32_t* target, const uint32_t* init) {
  variant<0, FORM, STEPS, LIVE>(b, target, init);
  variant<1, FORM, STEPS, LIVE>(b, target, init);
  variant<2, FORM, STEPS, LIVE>(b, target, init);
  variant<3, FORM, STEPS, LIVE>(b, target, init);
  variant<4, FORM, STEPS, LIVE>(b, target, init);
  variant<5, FORM, STEPS, LIVE>(b, target, init);
  variant<6, FORM, STEPS, LIVE>(b, target, init);
  variant<7, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<8, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<9, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<10, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<11, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<12, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<13, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<14, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<15, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<16, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<17, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<18, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<19, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<20, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<21, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<22, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<23, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<24, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<25, FORM, STEPS, LIVE>(b, target, init);
}

}  // namespace read_step

int main(int argc, char** argv) {
  using namespace read_step;
  lab::Bench b(argc, argv);
  uint32_t *target = nullptr, *init = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&target, static_cast<size_t>(kTargets) * 17 * sizeof(uint32_t)));
    LAB_CHECK(cudaMemset(target, 0, static_cast<size_t>(kTargets) * 17 * sizeof(uint32_t)));
    const size_t words = static_cast<size_t>(b.sms) * kThreads * 208;
    std::vector<uint32_t> h(words);
    for (size_t i = 0; i < words; ++i)
      h[i] = (i % 208) < 128 ? 0x3a800000u + (i & 1023) : 0x2c002c00u ^ (i & 255);
    LAB_CHECK(cudaMalloc(&init, words * sizeof(uint32_t)));
    LAB_CHECK(cudaMemcpy(init, h.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
  //: dense: merged (one add a step for 4 digits) against per digit (an add a digit step)
  modes<1, 8, 16>(b, target, init);
  modes<0, 8, 16>(b, target, init);
  //: sparse per digit, gathered: full steps, and half full (alt-k4's ~8 tokens a digit a window)
  modes<0, 2, 16>(b, target, init);
  modes<0, 1, 8>(b, target, init);
}
