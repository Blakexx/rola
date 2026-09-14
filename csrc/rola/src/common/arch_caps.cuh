#pragma once

// LAYER 1 -- THE ARCHITECTURE CAPABILITY FACTS: the only file that knows an
// architecture exists. Entry kinds (REQUIRED/SELECTING/RESIDENCY/FACT) and the
// no-fallback-chain rule: see docs/internals/common/arch_caps.md

#include <cuda_runtime.h>

namespace denseref {
namespace arch {

struct Caps {
  int cc;  // compute capability x10, e.g. 860

  // REQUIRED (a clause of the baseline structure; see arch_caps.md)
  bool async_copy;       // global->shared staging whose completion is a signal
  bool tile_load;        // 8x8 b16 quads landed directly in MMA register order
  bool mma_bf16_f32;     // tensor MAC, bf16 operands, fp32 accumulator
  bool bf16_cvt_packed;  // two fp32 -> one b32 of two bf16, one instruction
  bool
      reg_transpose_b16;  // an 8x8 b16 tile spread across the MMA unit's own registers is transposed IN PLACE. see docs/internals/common/arch_caps.md#reg-transpose-b16

  // RESIDENCY (where an accumulator lives / an operand sources from; design.cuh reads these)
  bool
      mma_a_from_regs;  // the A operand may be supplied from the issuing unit's OWN registers. see docs/internals/common/arch_caps.md#mma-a-from-regs
  bool
      mma_b_from_regs;  // the same for B. A warpgroup MMA sources B ONLY from a shared-memory descriptor. see docs/internals/common/arch_caps.md#mma-b-from-regs
  bool
      accum_in_tmem;  // the MMA's accumulator lives in a dedicated tensor memory rather than in the unit's register ... see docs/internals/common/arch_caps.md#accum-in-tmem
  bool
      mma_a_from_tmem;  // the A operand may be sourced from that same tensor memory, which is what lets the invariant ... see docs/internals/common/arch_caps.md#mma-a-from-tmem

  // SELECTING (implements a baseline op more efficiently; not required)
  bool
      bf16_simd_arith;  // elementwise bf16 arithmetic on a packed pair (`mul.rn.bf16x2`). see docs/internals/common/arch_caps.md#bf16-simd-arith
  bool
      bf16_simd_fma;  // FUSED elementwise bf16 arithmetic on a packed pair (`fma.rn.bf16x2`). see docs/internals/common/arch_caps.md#bf16-simd-fma
  bool
      async_bulk_copy;  // TMA. Would implement `stage_tile`'s contract with one descriptor issue instead of one per ... see docs/internals/common/arch_caps.md#async-bulk-copy
  bool
      warpgroup_mma;  // `wgmma`. Present on sm_90 and the reason `mma_b_from_regs` is false there. see docs/internals/common/arch_caps.md#warpgroup-mma
  bool
      dyn_reg_realloc;  // `setmaxnreg`: a warpgroup may GIVE UP or CLAIM registers at run time, so two groups of the ... see docs/internals/common/arch_caps.md#dyn-reg-realloc

  // FACT (a device number, no judgment)
  int mma_unit_threads;  // threads that jointly issue one MMA and jointly own its accumulator. 32 (warp) or 128 ... see docs/internals/common/arch_caps.md#mma-unit-threads
  int smem_per_sm;       // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
  int smem_per_cta_max;  // opt-in dynamic maximum, one CTA
  int smem_granularity;  // shared-memory allocation quantum, bytes
  int regs_per_sm;
  int regs_per_lane_max;
  int reg_alloc_unit;  // registers allocated per warp, in units of this
  int max_threads_per_sm;
  int max_ctas_per_sm;
  int driver_smem_reserve;  // per-CTA reserve the driver takes off the top; the residency condition is on smem_per_cta + ... see docs/internals/common/arch_caps.md#driver-smem-reserve
  int tensor_f32_accum_pct;  // fp32-accumulate tensor rate as a percentage of the fp16-accumulate rate. 50 on consumer ... see docs/internals/common/arch_caps.md#tensor-f32-accum-pct
};

//: The RATIFIED SET.  A compute capability that is not a row here fails the -- see docs/internals/common/arch_caps.md#tabulated
constexpr bool tabulated(int cc) {
  return cc == 800 || cc == 860 || cc == 870 || cc == 890 || cc == 900;
}

// Field order matches the struct's REQUIRED|RESIDENCY|SELECTING|FACT layout above.
// See docs/internals/common/arch_caps.md#note-l125
constexpr Caps caps_of(int cc) {
  return cc == 800   ? Caps{800,   true,  true, true,  true,  true,  true, true,   false,
                          false, false, true, false, false, false, 32,   167936, 166912,
                          128,   65536, 255,  256,   2048,  32,    1024, 100}
         : cc == 870 ? Caps{870,   true,  true, true,  true,  true,  true, true,   false,
                            false, false, true, false, false, false, 32,   167936, 166912,
                            128,   65536, 255,  256,   1536,  16,    1024, 50}
         : cc == 890 ? Caps{890,   true,  true, true,  true,  true,  true, true,   false,
                            false, false, true, false, false, false, 32,   102400, 101376,
                            128,   65536, 255,  256,   1536,  24,    1024, 50}
         : cc == 900
             // sm_90 TABULATED, NOT BUILT; see docs/internals/common/arch_caps.md#caps-of-note-l147
             ? Caps{900,   true,  true, true, true, true, true, false,  false,
                    false, true,  true, true, true, true, 128,  233472, 232448,
                    128,   65536, 255,  256,  2048, 32,   1024, 100}
             : Caps{
                   860,   true,  true, true,  true,  true,  true, true,   false,  // sm_86: local dev (GA102)
                   false, false, true, false, false, false, 32,   102400, 101376, 128,
                   65536, 255,   256,  1536,  16,    1024,  50};
}

constexpr bool supported(const Caps& c) {  // conjunction of REQUIRED; see arch_caps.md#note-l163
  return c.async_copy && c.tile_load && c.mma_bf16_f32 && c.bf16_cvt_packed && c.reg_transpose_b16;
}

// DERIVATION -- the quantitative half; only consumers of the FACT rows, no arch branch.

constexpr int round_up(int x, int q) { return (x + q - 1) / q * q; }

//: Registers a CTA actually reserves.  Allocation is per warp in `reg_alloc_unit` -- see docs/internals/common/arch_caps.md#regspercta
constexpr int regs_per_cta(const Caps& c, int threads, int regs_per_lane) {
  return threads / 32 * round_up(regs_per_lane * 32, c.reg_alloc_unit);
}

constexpr int ctas_by_smem(const Caps& c, int smem_per_cta) {
  return c.smem_per_sm / (round_up(smem_per_cta, c.smem_granularity) + c.driver_smem_reserve);
}

constexpr int ctas_by_regs(const Caps& c, int threads, int regs_per_lane) {
  return c.regs_per_sm / regs_per_cta(c, threads, regs_per_lane);
}

constexpr int ctas_by_threads(const Caps& c, int threads) { return c.max_threads_per_sm / threads; }

//: Resident CTAs per SM: the binding constraint among shared memory, registers, -- see docs/internals/common/arch_caps.md#min4
constexpr int min4(int a, int b, int d, int e) {
  return a < b ? (a < d ? (a < e ? a : e) : (d < e ? d : e))
               : (b < d ? (b < e ? b : e) : (d < e ? d : e));
}

constexpr int residency(const Caps& c, int smem_per_cta, int threads, int regs_per_lane) {
  return min4(ctas_by_smem(c, smem_per_cta), ctas_by_regs(c, threads, regs_per_lane),
              ctas_by_threads(c, threads), c.max_ctas_per_sm);
}

//: THE PER-ARCH REGISTER BUDGET. The largest per-lane register count that still -- see docs/internals/common/arch_caps.md#note-l203
constexpr int reg_budget(const Caps& c, int threads, int target_ctas) {
  return (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit) / 32
                 < c.regs_per_lane_max
             ? (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit)
                   / 32
             : c.regs_per_lane_max;
}

// THE COMPILING TARGET; inert in the host pass, pinned to the smallest tabulated config.
#ifdef __CUDA_ARCH__
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 860;
#endif

static_assert(tabulated(kArch),
              "untabulated compute capability: add its row to caps_of() rather "
              "than letting an operation guess which implementation it has");

constexpr Caps kCaps = caps_of(kArch);

static_assert(supported(kCaps),
              "UNSUPPORTED ARCHITECTURE: a required capability "
              "clause is false; see the Caps row");

// `mma_unit_threads` is NOT asserted here -- an operation fact, not an arch one; lives in ops.cuh.

constexpr int caps_digest(const Caps& c) {  // see docs/internals/common/arch_caps.md#capsdigest
  return c.cc * 1000 + (int)c.async_copy * 1 + (int)c.tile_load * 2 + (int)c.mma_bf16_f32 * 4
         + (int)c.mma_b_from_regs * 8 + (int)c.bf16_cvt_packed * 16 + (int)c.bf16_simd_arith * 32
         + (int)c.async_bulk_copy * 64 + (int)c.warpgroup_mma * 128 + (int)c.mma_a_from_regs * 256
         + (int)c.accum_in_tmem * 512 + (int)c.mma_a_from_tmem * 1024
         + (int)c.reg_transpose_b16 * 2048 + (int)c.bf16_simd_fma * 4096 + c.mma_unit_threads * 2;
}

}  // namespace arch
}  // namespace denseref
