// THE T=1 DECODE STEP -- ONE KERNEL: fold, factor, walk. Token-stationary,
// state-streaming, lattice-walked. Owns the step kernel, its 16-way instantiation matrix
// and the decode path's host entry.
//
// The ratified shape, literally: the token's routing weights and value are RESIDENT
// (SMEM/registers), the STATE streams past, and a work queue hands the lattice's unit
// rank space to resident CTAs.
//
// Three facts a reader must not miss:
//   1. There is one token, so the contraction index is the LEAF and the output index is
//      the COLUMN: a lane holds eight channels of a leaf and no MMA exists here.
//   2. THE CTAs VISIT HEADS. A head's work is known only after its prologue, so the grid is
//      one wave of resident CTAs that claim a head's positions, add their partials into its
//      sums, and steal the head with the most work left; the state stays byte-reproducible
//      (its positions partition the written leaves) and `y` is reproducible to its sums'
//      order: docs/internals/decode/decode.md#queue
//   3. THE PROLOGUE IS PER VISIT AND THAT IS THE DESIGN: the factor tables never leave
//      shared memory, and the walk takes no barrier:
//      docs/internals/decode/decode.md#per-cta-prologue,
//      docs/internals/decode/decode.md#no-barrier
//
// The GEMV/GEMM argument, the walk's cost accounting against `N`, the barrier ledger, and
// the derivation that collapses the causal intra Gram into UPDATE-THEN-READ:
// docs/internals/decode/decode.md

#include "common/torch_seam.cuh"
#include <cuda_bf16.h>

#include <algorithm>

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/phase_clock.cuh"
#include "common/state_page.cuh"
#include "decode.cuh"
#include "decode_selection.inc"
#include "decode_api.cuh"
#include "decode_fold.cuh"
#include "decode_lattice.cuh"
#include "decode_leaf.cuh"
#include "dispatch_switch.cuh"

namespace rola {
namespace decode {
namespace detail {

//: NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a hash that -- see docs/internals/decode/decode.md#near-line-51
namespace stampdet {

__device__ int64_t g_decode_stamp_probe;

//: THE STEP'S OWN FOOTPRINT, recompiled with this translation unit, so a test that reads -- see docs/internals/decode/decode.md#decodestampkernel
template <class Bt>
__global__ void decode_stamp_kernel(int64_t* out) {
  //: THE ENUMERATION'S SHAPE AND THE LAUNCH'S WIDTH ARE PART OF THE FOOTPRINT, -- see docs/internals/decode/decode.md#decode-stamp-kernel-note-l63
  *out = (static_cast<int64_t>(kKinds * kMaxTerms * DecodePlan<Bt::build>::kWarps + kUnitBlockWarps
                               + kFieldWidthMax)
              * 4096
          + static_cast<int64_t>(sizeof(DecodeParams)))
             * 1000003
         + static_cast<int64_t>(sizeof(Unit<4>)) * 101 + static_cast<int64_t>(kDecodeMaxSpan);
  g_decode_stamp_probe = *out;
}

}  // namespace stampdet

//: THE VERDICT'S LAST STEP, by the warp that finishes a head: the finished-head count, and the last head's -- see docs/internals/decode/decode.md#seen
__device__ __forceinline__ void finish_verdict(const DecodeParams& p, int lane) {
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = (atomicAdd(p.growth_ctr, 1) == p.BH - 1) ? 1 : 0;
  last = __shfl_sync(0xffffffffu, last, 0);
  if (last == 0) return;
  __threadfence();
  int seen = 0;
#pragma unroll 1
  for (int i = lane; i < p.BH; i += 32) seen |= __ldcg(p.growth + i);
  const bool any_needy = __any_sync(0xffffffffu, seen != 0);
  if (!any_needy) {
#pragma unroll 1
    for (int i = lane; i < p.BH; i += 32) p.done[i] = 0;
  }
  if (lane == 0) {
    *p.growth_any = any_needy ? 1 : 0;
    *p.growth_ctr = 0;
  }
}

//: A VISIT'S PARTIAL INTO ITS HEAD, by warp 0: the CTA's fold summed a column a lane and added once, the -- see docs/internals/decode/decode.md#queue-finish
template <class Dp, int DV>
__device__ __forceinline__ void add_visit(const DecodeParams& p, const float* fold, int bh,
                                          int blocks, bool paged, int* finished, int lane) {
  constexpr int cols = DV + 1;
  int positions_walked = 0;
#pragma unroll 1
  for (int wq = 0; wq < Dp::kWarps; ++wq) {
    positions_walked += __float_as_int(fold[wq * (cols + 3) + DV + 1]);
  }
  if (positions_walked == 0) return;
  float* const sums = p.yacc + static_cast<int64_t>(bh) * cols;
#pragma unroll 1
  for (int c = lane; c <= DV; c += 32) {
    float part = 0.0f;
#pragma unroll 1
    for (int wq = 0; wq < Dp::kWarps; ++wq) part += fold[wq * (cols + 3) + c];
    atomicAdd(sums + c, part);
  }
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = atomicAdd(finished, positions_walked) + positions_walked == blocks ? 1 : 0;
  last = __shfl_sync(0xffffffffu, last, 0);
  if (last == 0) return;
  //: THE HEAD'S LAST POSITIONS FINISH IT: y from the sums, which the finisher zeroes for the next step.
  __threadfence();
  const float den = __ldcg(sums + DV) + p.eps;
#pragma unroll 1
  for (int c = lane; c < DV; c += 32) {
    p.y[static_cast<int64_t>(bh) * DV + c] = __ldcg(sums + c) / den;
    sums[c] = 0.0f;
  }
  if (lane == 0) sums[DV] = 0.0f;
  if (!paged) return;
  //: THE DONE FLAG is set by the head's finisher alone: every visit that read it has claimed its last position.
  if (lane == 0) {
    p.growth[bh] = 0;
    p.done[bh] = 1;
  }
  finish_verdict(p, lane);
}

//: THE STEP at build `Bt::build` (a type: docs/internals/common/hardware.md#build-tag), at its plan's threads.
template <class Bt, int DV, int D, bool DECAY>
__global__
__launch_bounds__(DecodePlan<Bt::build>::kThreads, DecodePlan<Bt::build>::kBlocksPerSm) void
decode_step_kernel(DecodeParams p) {
  using Dp = DecodePlan<Bt::build>;
  using Scan = LatticeScan<Dp::kThreads>;
  //: The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a -- see docs/internals/decode/decode.md#cols
  constexpr int cols = DV + 1;
  //: THE STATE PAGE: s24 planes at their group exponents, and the mass.
  using Page = rola::state_page::Blocks<DV>;
  using Lanes = LeafLanes<DV>;

  extern __shared__ __align__(16) char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int total_rows = p.total_rows;

  // ---- SMEM ledger ----------------------------------------------------------
  // sa_r, sa_w [total_rows] fp32 both sides' amplitude rows ...
  // see docs/internals/decode/decode.md#decode-step-kernel-note-l95
  float* sa_r = reinterpret_cast<float*>(smem_raw);
  float* sa_w = sa_r + total_rows;
  float* sd = sa_w + total_rows;
  float* v_tok = sd + (DECAY ? total_rows : 0);
  float* fold = v_tok + DV;
  uint32_t* mask_r = reinterpret_cast<uint32_t*>(fold + Dp::kWarps * (cols + 3));
  uint32_t* mask_w = mask_r + p.mask_total;
  int32_t* olist = reinterpret_cast<int32_t*>(mask_w + p.mask_total);
  int32_t* dlist = olist + kKinds * p.olist_total;

  __shared__ float s_sum[kMaxLevels];
  __shared__ typename Scan::TempStorage scan_temp;
  __shared__ int s_running;
  __shared__ int s_n_o[kKinds * kMaxLevels];
  __shared__ int s_n_d[kMaxLevels];
  //: THE TERM TABLE, ONE PACKED WORD PER (term, level): the field's position, its -- see docs/internals/decode/decode.md#decode-step-kernel-note-l137
  __shared__ int s_fld[kMaxTerms * kMaxLevels];
  //: One unit count per TERM. The walk's outer loop is the term, so each term's rank space -- see docs/internals/decode/decode.md#near-line-122
  __shared__ int s_units[kMaxTerms];
  //: THE HEAD'S ADMISSION as this CTA holds it: whether it admits, the outcome, the position count. -- see docs/internals/decode/decode.md#queue-admission
  __shared__ int s_admit;
  __shared__ int s_blocks;
  //: The head the CTA visits next, and whether it steals it, as warp 0 chose. -- see docs/internals/decode/decode.md#queue-next
  __shared__ int s_next;
  __shared__ int s_last;
  //: A HEAD'S TERMS' FIRST POSITIONS, and past the last its position count. -- see docs/internals/decode/decode.md#queue-positions
  __shared__ int s_first[kMaxTerms + 1];
  __shared__ long long s_clock[Dp::kWarps][PhaseClock<kDecodePhases>::kRow];

  PhaseClock<kDecodePhases> clock(&s_clock[warp][0], lane);

  //: THE QUEUE, per batch-head: positions claimed, positions finished, the position count (-1 until known), the -- see docs/internals/decode/decode.md#queue
  auto q_claim = [&](int i) -> int& { return p.queue[i]; };
  auto q_finished = [&](int i) -> int& { return p.queue[p.BH + i]; };
  auto q_blocks = [&](int i) -> int& { return p.queue[2 * p.BH + i]; };
  auto q_state = [&](int i) -> int& { return p.queue[3 * p.BH + i]; };
  int* const q_cursor = p.queue + 4 * p.BH;
  int* const q_exit = q_cursor + 1;

  //: THE LANE'S GROUP AND CHANNELS: group `grp` walks its own unit; the lane holds channels -- see docs/internals/decode/decode.md#leaf-lanes
  const int grp = lane / Lanes::kLanes;
  const int ch0 = 8 * (lane % Lanes::kLanes);
  const bool row_lead = ch0 == 0;
  const bool exp_lead = (ch0 & 31) == 0;
  const bool paged = p.page_tbl != nullptr;
  //: A CTA'S FIRST HEAD is its index modulo the heads; a head past the grid comes from the cursor. -- see docs/internals/decode/decode.md#queue-next
  int bh = static_cast<int>(blockIdx.x) % p.BH;
  bool stolen = false;

#pragma unroll 1
  while (bh >= 0) {
    const int b = bh / p.H;
    const int h = bh - b * p.H;

    // ---- THE PROLOGUE: three phases, three barriers ------------------------------
    //: PHASE A, EVERY OPERAND'S LOAD AT ONCE: a value channel, the dials and the digits of both sides -- see docs/internals/decode/decode.md#prologue
    const ValueUnit v_c = tid < DV ? load_value(p, b, h, tid) : ValueUnit{__float2half(0.0f), 0.0f};
    if (DECAY) {
#pragma unroll 1
      for (int i = tid; i < total_rows; i += Dp::kThreads) {
        int l = 0;
#pragma unroll
        for (int q = 1; q < D; ++q) {
          if (i >= p.level_amp_offset[q]) l = q;
        }
        sd[i] = p.dials[static_cast<int64_t>(h) * total_rows + p.level_row_offset[l]
                        + (i - p.level_amp_offset[l])];
      }
    }
    fold_digits<Dp, D>(p, b, h, sa_r, sa_w, mask_r, mask_w);
    if (tid < DV) v_tok[tid] = value_of(v_c);
    __syncthreads();
    clock.lap(kPhFold);

    //: PHASE B, FROM THE DIGIT MASKS, one warp a job: the owner and digit lists with their counts, and a decay step's write row sums -- see docs/internals/decode/decode.md#prologue
    build_lists<D>(p, mask_r, mask_w, olist, dlist, s_n_o, s_n_d);
    if (DECAY) sum_staged_rows(p, sa_w, s_sum, 2 * D - 1);
    __syncthreads();
    //: PHASE C: the decay's share scale and the term layout (whose barrier ends it).
    const float share = DECAY ? share_scale(p, s_sum) : 0.0f;
    publish_term_layout<D>(p, s_n_o, s_n_d, s_fld, s_units);
    clock.lap(kPhFactor);

    //: THE HEAD'S POSITIONS: a warp's share of each term in turn -- a unit of a whole-atom write term, a batch -- see docs/internals/decode/decode.md#queue-positions
    if (tid == 0) {
      int at = 0;
#pragma unroll
      for (int term = 0; term <= D; ++term) {
        s_first[term] = at;
        at +=
            (term == 0 && p.write_atoms) ? s_units[0] : ceil_div_int(s_units[term], Lanes::kGroups);
      }
      s_first[D + 1] = at;
    }

    // ---- THE ADMISSION: one decision a head, published -------------------------------
    //: Three outcomes PER BATCH-HEAD: advance, no-op, or re-read without -- see docs/internals/decode/decode.md#paged
    const bool deposit = !(paged && p.done[bh] != 0);
    if (tid == 0) {
      //: an unpaged head has nothing to decide; a paged one's first arrival owns the decision.
      int role = kQueueDecided;
      if (paged && deposit && !stolen)
        role = atomicCAS(&q_state(bh), 0, 1) == 0 ? kQueueOwner : kQueueFollower;
      if (paged && !deposit) role = kQueueDecided;
      s_admit = role;
    }
    __syncthreads();
    const bool owner = (s_admit & kQueueOwner) != 0;
    const bool follower = (s_admit & kQueueFollower) != 0 || (paged && deposit && stolen);
    bool needy = false;
    if (owner) {
      const int inner_bits = p.lat_inner_bits;
      //: THE CANDIDATE ATOMS, enumerated from the same lattice product the walk uses, -- see docs/internals/decode/decode.md#win-bits
      const int win_bits = inner_bits < kAtomShift ? inner_bits : kAtomShift;
      const uint32_t win_mask = (1u << (1 << win_bits)) - 1u;
      const int n_units = 1 << p.lat_units_per_atom_shift;
      //: THE CANDIDATE SLOT'S ROW of the same packed term table -- the ADMISSION's rank space, -- see docs/internals/decode/decode.md#candfld
      const int32_t* cand_fld = s_fld + (D + 1) * kMaxLevels;
      auto candidate_atom = [&](int i) -> int {
        const int a = i & ((1 << p.atoms_per_unit_shift) - 1);
        const int unit0 = (i >> p.atoms_per_unit_shift) << p.lat_units_per_atom_shift;
        int atom = -1;
#pragma unroll 1
        for (int q = 0; q < n_units; ++q) {
          const CandidateUnit u = resolve_candidate_unit<D>(p, mask_w, olist, cand_fld, unit0 + q);
          if (!u.valid) return -1;
          if (atom < 0) atom = (u.base + (a << kAtomShift)) >> kAtomShift;
          if (u.live_w
              && ((u.cw >> ((a << kAtomShift) & ((1 << inner_bits) - 1))) & win_mask) != 0u) {
            return atom;
          }
        }
        return -1;
      };
      //: FROM THE CANDIDATE SLOT, not from the walk's write term: the two enumerate the same -- see docs/internals/decode/decode.md#ncandidates
      const int n_candidates = (s_units[D + 1] >> p.lat_units_per_atom_shift)
                               << p.atoms_per_unit_shift;
      const int64_t row_base = static_cast<int64_t>(bh) * p.atoms_per_bh;
      int32_t* const page_row = p.page_tbl + row_base;
      //: THE TABLE TEST: how many atoms this batch-head's write side reaches that the -- see docs/internals/decode/decode.md#a
      int found = 0;
#pragma unroll 1
      for (int i = tid; i < n_candidates; i += Dp::kThreads) {
        const int a = candidate_atom(i);
        if (a >= 0 && page_row[a] < 0) ++found;
      }
      int excl = 0;
      int demand = 0;
      Scan(scan_temp).ExclusiveSum(found, excl, demand);
      __syncthreads();
      if (demand > 0) {
        const int cursor = (p.pool_cap > 0) ? p.pool_cursor[bh] : 0;
        const bool claimed = demand <= p.pool_cap - cursor;
        needy = !claimed;
        if (claimed) {
          //: THE CLAIM, owned by the head's owner: the touched-unmapped atoms in ascending atom id take the next -- see docs/internals/decode/decode.md#queue-claim-pool
          const int32_t* slots = p.pool_slots + static_cast<int64_t>(bh) * p.pool_cap;
          if (tid == 0) s_running = 0;
          __syncthreads();
#pragma unroll 1
          for (int base = 0; base < n_candidates; base += Dp::kThreads) {
            const int i = base + tid;
            const int a = (i < n_candidates) ? candidate_atom(i) : -1;
            const bool want = a >= 0 && page_row[a] < 0;
            int rank = 0;
            int aggregate = 0;
            Scan(scan_temp).ExclusiveSum(want ? 1 : 0, rank, aggregate);
            if (want) page_row[a] = slots[cursor + s_running + rank];
            __syncthreads();
            if (tid == 0) s_running += aggregate;
            __syncthreads();
          }
          if (tid == 0) p.pool_cursor[bh] = cursor + demand;
        }
      }
      //: THE CONDENSATION, published by the head's owner: the write-atom set a host admission plans from. -- see docs/internals/decode/decode.md#decode-step-kernel-note-l355
      bool* atom_row = p.atom_bits + row_base;
#pragma unroll 1
      for (int a = tid; a < p.atoms_per_bh; a += Dp::kThreads) atom_row[a] = false;
      __syncthreads();
#pragma unroll 1
      for (int i = tid; i < n_candidates; i += Dp::kThreads) {
        const int a = candidate_atom(i);
        if (a >= 0) atom_row[a] = true;
      }
      //: THE PUBLICATION: the count, every thread's fence, then the one swap that makes it all readable. A needy head has no positions.
      if (tid == 0) {
        s_blocks = needy ? 0 : (s_first[D + 1] > 0 ? s_first[D + 1] : 1);
        q_blocks(bh) = s_blocks;
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicExch(&q_state(bh), kQueueReady | (needy ? kQueueNeedy : 0));
    } else if (follower) {
      //: A FOLLOWER WAITS FOR THE OWNER, which is running: it won the swap. -- see docs/internals/decode/decode.md#queue-admission
      if (tid == 0) {
        while ((atomicAdd(&q_state(bh), 0) & kQueueReady) == 0) __nanosleep(64);
      }
    } else if (tid == 0) {
      //: nothing to decide: the head is unpaged, or a replay re-reads it; every visitor derives the same count.
      s_blocks = s_first[D + 1] > 0 ? s_first[D + 1] : 1;
      q_blocks(bh) = s_blocks;
    }
    __syncthreads();
    int blocks = s_blocks;
    if (follower) {
      //: EVERY THREAD READS THE OWNER'S PUBLICATION ITSELF, after its own fence (KERNEL_STANDARDS §12).
      __threadfence();
      needy = (__ldcg(&q_state(bh)) & kQueueNeedy) != 0;
      blocks = __ldcg(&q_blocks(bh));
    }
    clock.lap(kPhAdmit);

    //: THE WARP'S OWN PARTIAL for this head, over every position it claims.
    float acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0f;
    float acc_m = 0.0f;
    int walked = 0;

    //: THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is -- see docs/internals/decode/decode.md#decode-step-kernel-note-l397
    const int32_t* page_row =
        !paged ? nullptr : p.page_tbl + static_cast<int64_t>(bh) * p.atoms_per_bh;
    const int64_t dense_bh_base = static_cast<int64_t>(bh) * p.atoms_per_bh;
    int cached_atom = -1;
    int64_t atom_base = 0;
    bool atom_present = false;

    // ---- THE WALK: a warp claims a position, walks it, claims again -------------------
    //: @barrier-free the walk: a lane group's read and write-back are one statement in program order; see docs/internals/decode/decode.md#no-barrier
    //: A NEEDY BATCH-HEAD RUNS NOTHING: its count is zero, so no warp claims. -- see docs/internals/decode/decode.md#queue-admission
    //: THE NEXT CLAIM IS ISSUED BEFORE THE CURRENT POSITION IS WALKED, so its round trip hides behind the walk. -- see docs/internals/decode/decode.md#queue-claim
    int claim = 0;
    if (lane == 0) claim = atomicAdd(&q_claim(bh), 1);
#pragma unroll 1
    while (true) {
      const int pos = __shfl_sync(0xffffffffu, claim, 0);
      if (pos >= blocks) break;
      if (lane == 0) claim = atomicAdd(&q_claim(bh), 1);
      ++walked;
      //: THE POSITION'S TERM: the last term whose first position it reaches (an empty head's one position walks nothing).
      int term = -1;
#pragma unroll
      for (int t = 0; t <= D; ++t) {
        if (pos >= s_first[t] && pos < s_first[t + 1]) term = t;
      }
      if (term < 0) continue;
      const int local = pos - s_first[term];
      //: THE TERM'S ROW OF THE PACKED TABLE: every level's -- see docs/internals/decode/decode.md#units-2
      const int units = s_units[term];
      const int32_t* fld = s_fld + term * kMaxLevels;
      //: TERM 0 IS THE WRITE PRODUCT: every leaf it walks is written, and no other term writes. -- see docs/internals/decode/decode_lattice.md#the-terms
      const bool writes = term == 0;

      if (writes && p.write_atoms) {
        //: THE WRITE TERM BY WHOLE ATOMS: a warp a unit, its lane groups a quarter of the atom's leaves each, every -- see docs/internals/decode/decode.md#whole-atoms
        constexpr int kRows = kAtomLeaves / Lanes::kGroups;
        const Unit<D> u = resolve_write_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd, olist,
                                                       dlist, fld, local);
        if (u.cw == 0u) continue;
        const uint32_t br = u.live_r ? u.cr : 0u;
#pragma unroll 1
        for (int a = 0; a < (1 << p.atoms_per_unit_shift); ++a) {
          const int atom = (u.base >> kAtomShift) + a;
          bool present = true;
          int64_t base = (dense_bh_base + atom) * static_cast<int64_t>(Page::kBytes);
          if (page_row != nullptr) {
            const int slot = page_row[atom];
            present = slot >= 0;
            base = static_cast<int64_t>(slot) * Page::kBytes;
          }
          uint8_t* const page = p.state + base;
          LeafAddr addr[kRows];
          LeafSlice slice[kRows];
#pragma unroll
          for (int k = 0; k < kRows; ++k) {
            addr[k] = leaf_addr<DV>(page, k * Lanes::kGroups + grp, ch0);
            slice[k] = load_leaf(addr[k], present, true, row_lead);
          }
#pragma unroll
          for (int k = 0; k < kRows; ++k) {
            const int t = (a << kAtomShift) + k * Lanes::kGroups + grp;
            //: ONLY THE WRITE SUPPORT IS THIS TERM'S: a leaf outside it belongs to a read term, which reads it.
            const bool mine = present && ((u.cw >> t) & 1u) != 0u;
            const bool in_r = ((br >> t) & 1u) != 0u;
            const float ar = (mine && in_r) ? u.ar * sa_r[u.dlast + t] : 0.0f;
            if (deposit) {
              const float aw = mine ? u.aw * sa_w[u.dlast + t] : 0.0f;
              const float rate = (DECAY && mine) ? u.rate * sd[u.dlast + t] : 0.0f;
              deposit_leaf<DECAY>(slice[k], addr[k], ar, aw, rate, share, v_tok, ch0, mine,
                                  row_lead, exp_lead, acc, acc_m);
            } else {
              read_full(slice[k], ar, row_lead, acc, acc_m);
            }
          }
        }
        continue;
      }

      const int unit = local * Lanes::kGroups + grp;
      const bool has_unit = unit < units;
      //: TWO RESOLVERS, ONE WALK. -- see docs/internals/decode/decode.md#u
      const int named = has_unit ? unit : 0;
      const Unit<D> u = writes ? resolve_write_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd,
                                                              olist, dlist, fld, named)
                               : resolve_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd, olist,
                                                        fld, term, named);
      const uint32_t br = (has_unit && u.live_r) ? u.cr : 0u;
      const uint32_t bw = (has_unit && u.live_w) ? u.cw : 0u;
      //: THE ENUMERATED RUN IS THIS TERM'S. Term 0 walks the write run; every other -- see docs/internals/decode/decode.md#decode-step-kernel-note-l474
      uint32_t bits = writes ? bw : br;

      //: THE SLOT LOOP IS THE INNERMOST LEVEL'S RUN, in ASCENDING lattice order, one leaf a group -- see docs/internals/decode/decode.md#t
#pragma unroll 1
      while (__any_sync(0xffffffffu, bits != 0u)) {
        const bool live = bits != 0u;
        const int t = live ? __ffs(static_cast<int>(bits)) - 1 : 0;
        bits &= bits - 1u;
        const int off_last = u.dlast + t;
        const bool in_r = ((br >> t) & 1u) != 0u;

        const int leaf = u.base + t;
        const int atom = leaf >> kAtomShift;
        if (live && atom != cached_atom) {
          cached_atom = atom;
          if (page_row != nullptr) {
            const int slot = page_row[atom];
            atom_present = slot >= 0;
            atom_base = static_cast<int64_t>(slot) * Page::kBytes;
          } else {
            atom_present = true;
            atom_base = (dense_bh_base + atom) * (int64_t)Page::kBytes;
          }
        }
        //: AN ABSENT ATOM READS AS ZEROS AND IS NOT WRITTEN. Exact, not defensive: the -- see docs/internals/decode/decode.md#decode-step-kernel-note-l516
        const bool present = live && atom_present;
        const float ar = (present && in_r) ? u.ar * sa_r[off_last] : 0.0f;
        //: A LEAF THIS STEP ONLY READS PULLS THE hi PLANE ALONE; a WRITTEN one both planes -- see docs/internals/decode/decode.md#int8-state
        const LeafAddr addr = leaf_addr<DV>(p.state + atom_base, leaf & kAtomMask, ch0);
        const LeafSlice slice = load_leaf(addr, present, writes, row_lead);
        if (!writes) {
          read_hi(slice, ar, row_lead, acc, acc_m);
        } else if (!deposit) {
          read_full(slice, ar, row_lead, acc, acc_m);
        } else {
          //: UPDATE FIRST, THEN READ THE STORED ROW -- AND THAT ORDER IS THE SEMANTICS, -- see docs/internals/decode/decode.md#decode-step-kernel-note-l532
          const float aw = present ? u.aw * sa_w[off_last] : 0.0f;
          const float rate = DECAY ? u.rate * sd[off_last] : 0.0f;
          deposit_leaf<DECAY>(slice, addr, ar, aw, rate, share, v_tok, ch0, present, row_lead,
                              exp_lead, acc, acc_m);
        }
      }
    }
    //: @barrier-free-end
    clock.lap(kPhWalk);

    // ---- THE READOUT: the CTA folds its warps' partials, adds them into the head's sums once, and the ------
    // ---- visit that finishes the head's positions finishes the head ------------------------------------------
    //: THE GROUPS' PARTIALS, summed across the warp by a fixed butterfly: every lane of a channel slice ends -- see docs/internals/decode/decode.md#leaf-lanes
#pragma unroll
    for (int off = Lanes::kLanes; off < 32; off <<= 1) {
#pragma unroll
      for (int c = 0; c < 8; ++c) acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], off);
      acc_m += __shfl_xor_sync(0xffffffffu, acc_m, off);
    }
    //: THE CTA'S FOLD: a warp's row of partials and its position count, summed by warp 0 -- one set of -- see docs/internals/decode/decode.md#queue-finish
    float* const my_row = fold + warp * (cols + 3);
    if (grp == 0) {
#pragma unroll
      for (int c = 0; c < 8; ++c) my_row[ch0 + c] = acc[c];
    }
    if (lane == 0) {
      my_row[DV] = acc_m;
      my_row[DV + 1] = __int_as_float(walked);
    }
    __syncthreads();
    if (warp == 0) {
      add_visit<Dp, DV>(p, fold, bh, blocks, paged, &q_finished(bh), lane);
      //: A NEEDY HEAD'S OWNER FINISHES IT: no positions, no y, its growth flag set. -- see docs/internals/decode/decode.md#queue-finish
      if (owner && needy) {
        if (lane == 0) p.growth[bh] = 1;
        finish_verdict(p, lane);
      }
    }
    clock.lap(kPhCombine);

    // ---- THE NEXT HEAD: an unstarted one, else the one with the most unclaimed positions --------------
    //: every warp is done with this head's tables before the next head's prologue overwrites them.
    __syncthreads();
    if (warp == 0) {
      int next = -1;
      const int started = p.ctas < p.BH ? p.ctas : p.BH;
      if (lane == 0 && started < p.BH) {
        const int c = atomicAdd(q_cursor, 1) + started;
        if (c < p.BH) next = c;
      }
      next = __shfl_sync(0xffffffffu, next, 0);
      int steal = 0;
      if (next < 0) {
        //: STEALING: the head with the most positions left, among those whose count is published. -- see docs/internals/decode/decode.md#queue-next
        int best = -1;
        int most = 0;
#pragma unroll 1
        for (int base = 0; base < p.BH; base += 32) {
          const int cand = base + lane;
          int left = 0;
          if (cand < p.BH) {
            const int count = __ldcg(&q_blocks(cand));
            if (count > 0) left = count - __ldcg(&q_claim(cand));
          }
          int top = left;
          int who = cand;
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            const int t2 = __shfl_xor_sync(0xffffffffu, top, off);
            const int w2 = __shfl_xor_sync(0xffffffffu, who, off);
            if (t2 > top || (t2 == top && w2 < who)) {
              top = t2;
              who = w2;
            }
          }
          if (top > most) {
            most = top;
            best = who;
          }
        }
        //: A STEAL PAYS A PROLOGUE, so it takes a head only with a position left for every warp.
        if (most >= Dp::kWarps) {
          next = best;
          steal = 1;
        }
      }
      if (lane == 0) {
        s_next = next;
        s_last = steal;
      }
    }
    __syncthreads();
    bh = s_next;
    stolen = s_last != 0;
  }

  // ---- THE EXIT: the last CTA out resets the queue for the next step ----------------------
  //: No memset launch stands between two steps, so the step stays capturable. -- see docs/internals/decode/decode.md#queue
  clock.publish(p.ledger, static_cast<long>(blockIdx.x), Dp::kWarps, warp);
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(q_exit, 1) == p.ctas - 1 ? 1 : 0;
  __syncthreads();
  if (s_last != 0) {
    __threadfence();
#pragma unroll 1
    for (int i = tid; i < p.BH; i += Dp::kThreads) {
      q_claim(i) = 0;
      q_finished(i) = 0;
      q_blocks(i) = -1;
      q_state(i) = 0;
    }
    if (tid == 0) {
      *q_cursor = 0;
      *q_exit = 0;
    }
  }
}

}  // namespace detail

size_t decode_step_smem_bytes(int warps, int total_rows, int d_v, bool decay, int mask_total,
                              int olist_total, int dlist_total) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 3 : 2);
  const size_t fold = static_cast<size_t>(warps) * (d_v + 4);
  const size_t tables = 2 * static_cast<size_t>(mask_total)
                        + kKinds * static_cast<size_t>(olist_total)
                        + static_cast<size_t>(dlist_total);
  return (amps + static_cast<size_t>(d_v) + fold + tables) * sizeof(float);
}

//: THE BUILT SET, READ BACK OFF THE SAME LIST: `[d_v, D, decay]` per arm, in the order -- see docs/internals/decode/decode.md#roladecodearms
std::vector<std::vector<int64_t>> rola_decode_arms() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_DECODE_ROW(I_, DV_, D_, DECAY_) out.push_back({DV_, D_, DECAY_});
  ROLA_DECODE_ARM_SET_X(ROLA_DECODE_ROW)
#undef ROLA_DECODE_ROW
  return out;
}

//: THE BOUND PHASE LEDGER, or null: the next steps book into it; see decode.md#phase-ledger
long long* g_phase_ledger = nullptr;
int64_t g_phase_ledger_ctas = 0;

void rola_decode_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger) {
    g_phase_ledger = nullptr;
    g_phase_ledger_ctas = 0;
    return;
  }
  const int warps = detail::step_warps(card_of(ledger->get_device_index()));
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == warps && ledger->size(2) == kDecodePhases
                      && ledger->is_contiguous(),
                  "the decode phase ledger is a contiguous CUDA int64 tensor [ctas][", warps, "][",
                  kDecodePhases, "]");
  g_phase_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  g_phase_ledger_ctas = ledger->size(0);
}

void launch_decode_step(const Card& card, const DecodeParams& p, int d_v, bool decay,
                        int64_t smem_budget) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16`. What is NOT an axis -- -- see docs/internals/decode/decode.md#torchcheck
  STD_TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const size_t smem = decode_step_smem_bytes(detail::step_warps(card), p.total_rows, d_v, decay,
                                             p.mask_total, p.olist_total, p.dlist_total);
  STD_TORCH_CHECK(static_cast<int64_t>(smem) <= smem_budget,
                  "the decode step's staged operands need ", smem,
                  " B of shared memory against a per-block budget of ", smem_budget,
                  " B; sum_l width_l = ", p.total_rows,
                  ". This is a declared capacity, refused loudly rather than taken as a "
                  "slower path.");
  auto stream = current_stream();
  //: ONE LAUNCH SITE. Each axis arrives as an `integral_constant`, so the three -- see docs/internals/decode/decode.md#launch
  auto launch = [&](auto DV, auto LEVELS, auto DECAY) {
    constexpr int kDv = decltype(DV)::value, kLevels = decltype(LEVELS)::value;
    constexpr bool kDecay = decltype(DECAY)::value;
    build_switch(card, [&](auto b) {
      constexpr ::denseref::arch::Build kB = decltype(b)::build;
      if constexpr (in_image(kB))
        detail::decode_step_kernel<::denseref::arch::BuildTag<kB>, kDv, kLevels, kDecay>
            <<<dim3(p.ctas), DecodePlan<kB>::kThreads, smem, stream>>>(p);
    });
  };
  bool matched = false;
#define ROLA_DECODE_TRY(I_, DV_, D_, DECAY_)                                                  \
  if (!matched && d_v == (DV_) && p.D == (D_) && decay == ((DECAY_) != 0)) {               \
    matched = true;                                                                        \
    launch(std::integral_constant<int, (DV_)>{}, std::integral_constant<int, (D_)>{},      \
           std::integral_constant<bool, (DECAY_) != 0>{});                                 \
  }
  ROLA_DECODE_ARM_SET_X(ROLA_DECODE_TRY)
#undef ROLA_DECODE_TRY
  STD_TORCH_CHECK(matched, "this binary does not carry the decode arm (d_v = ", d_v, ", D = ", p.D,
                  ", decay = ", decay,
                  "). A full build carries every declared arm; an ITERATION build "
                  "(--//:decode_arms) carries a subset and is not shippable -- "
                  "rola.ops.decode.arms() lists what this one has.");
  ROLA_CUDA_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

namespace detail {

void check_f32(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Float, name, " must be float32, got ", t.scalar_type());
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//: an fp32 operand the step reads by address, strided as the producer left it.
void check_f32_span(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Float, name, " must be float32, got ", t.scalar_type());
}

void check_i32(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Int, name, " must be int32, got ", t.scalar_type());
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//: The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. `T != 1` is refused -- the fold -- see docs/internals/decode/decode.md#spanof
Span span_of(const Tensor& t, int64_t B, int64_t H, int64_t width, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.dim() == 4 && t.size(1) == 1, name, " must be [B, 1, H, W], got ",
                  shape(t.sizes()));
  STD_TORCH_CHECK(t.size(0) == B && t.size(2) == H, name, " is ", shape(t.sizes()),
                  " against the value's B = ", B, ", H = ", H);
  STD_TORCH_CHECK(width == 0 || t.size(3) == width, name, " is ", t.size(3), " wide against ",
                  width);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), t.stride(3)};
}

Span scalar_span_of(const Tensor& t, int64_t B, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.dim() == 3 && t.size(1) == 1, name, " must be [B, 1, H], got ",
                  shape(t.sizes()));
  STD_TORCH_CHECK(t.size(0) == B && t.size(2) == H, name, " is ", shape(t.sizes()),
                  " against the value's B = ", B, ", H = ", H);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), 0};
}

int ilog2_exact(int64_t x, const char* what) {
  STD_TORCH_CHECK(x >= 1, what, " must be positive, got ", x);
  int e = 0;
  while ((int64_t{1} << e) < x) ++e;
  STD_TORCH_CHECK((int64_t{1} << e) == x, what, " must be a power of two, got ", x);
  return e;
}

//: THE STEP'S PLAN at `card`'s build, for the host: its warps and its residency.
StepPlan step_plan(const Card& card) {
  StepPlan out{};
  build_switch(card, [&](auto b) {
    using Plan = DecodePlan<decltype(b)::build>;
    out = StepPlan{Plan::kWarps, Plan::kBlocksPerSm};
  });
  return out;
}

int step_warps(const Card& card) { return step_plan(card).warps; }

//: THE STEP'S RESIDENT GRID on `card`: one wave of the declared residency, at most `bh` heads' unit supply.
//: -- see docs/internals/decode/decode.md#the-split-policy
int grid_of(const Card& card, int64_t unit_ceiling, int64_t bh) {
  const int64_t supply =
      std::max<int64_t>(1, (unit_ceiling + kDecodeUnitsPerCta - 1) / kDecodeUnitsPerCta);
  return static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(int64_t{card.sms} * step_plan(card).blocks_per_sm, bh * supply)));
}

//: THE (k, m) BOX, FILLED AND REFUSED HERE. The derivation is `rola/ops/lattice.py`'s -- see docs/internals/decode/decode.md#fill-lattice
void fill_lattice(DecodeParams& p, int64_t k, int64_t m, int64_t N) {
  const int kappa = ilog2_exact(k, "the lattice span k");
  const int j = ilog2_exact(m, "the lattice capacity m");
  STD_TORCH_CHECK(k <= kDecodeMaxSpan, "the lattice span k = ", k, " exceeds the decode path's ",
                  kDecodeMaxSpan);
  p.lat_k = static_cast<int>(k);

  const int D = p.D;
  //: the lattice's section 2.5 as the atom-rectangle law generalizes it: the -- see docs/internals/decode/decode.md#floor-inner
  const int floor_inner = kAtomShift > kappa ? kAtomShift - kappa : 0;
  int ceil_inner = 0;
  while ((p.lat_k << (ceil_inner + 1)) <= p.level_width[D - 1]
         && p.level_width[D - 1] % (p.lat_k << (ceil_inner + 1)) == 0)
    ++ceil_inner;
  const int share = j / D + ((j % D) ? 1 : 0);
  int a_inner = share > floor_inner ? share : floor_inner;
  if (a_inner > ceil_inner) a_inner = ceil_inner;
  if (a_inner > j) a_inner = j;
  const int j_outer = j - a_inner;

  int64_t bc = 1;
  int64_t owners = 1;
  int local_bits = 0;
  for (int l = 0; l < D; ++l) {
    const int a = (l == D - 1 || D == 1)
                      ? a_inner
                      : j_outer / (D - 1) + ((l >= (D - 1) - (j_outer % (D - 1))) ? 1 : 0);
    p.lat_s[l] = p.lat_k << a;
    STD_TORCH_CHECK(p.lat_s[l] <= kDecodeMaxSpan, "level ", l, "'s owner span k*m_l = ", p.lat_s[l],
                    " exceeds the decode path's ", kDecodeMaxSpan);
    STD_TORCH_CHECK(p.level_width[l] % p.lat_s[l] == 0, "level ", l, " of width ", p.level_width[l],
                    " is not a whole number of owner spans k*m_l = ", p.lat_s[l],
                    " (an owner sits inside a level)");
    p.lat_g[l] = p.level_width[l] / p.lat_s[l];
    bc *= p.lat_s[l];
    owners *= p.lat_g[l];
    local_bits += kappa + a;
  }
  STD_TORCH_CHECK(owners * bc == N, "the lattice partitions ", owners * bc, " leaves against ", N);
  p.lat_local_bits = local_bits;

  //: `lattice run_shift`: level `l`'s run offset sits at `log2 prod_{l\' > l} s_l\'` in the -- see docs/internals/decode/decode.md#gsuf
  int gsuf = 1;
  int run_shift = 0;
  for (int l = D - 1; l >= 0; --l) {
    p.lat_gsuf[l] = gsuf;
    p.lat_run_shift[l] = run_shift;
    gsuf *= p.lat_g[l];
    run_shift += ilog2_exact(p.lat_s[l], "the level's owner span");
  }

  //: THE UNIT, and the atom it is reconciled with. -- see docs/internals/decode/decode.md#fill-lattice-note-l960
  p.lat_inner_bits = ilog2_exact(p.lat_s[D - 1], "the innermost level's owner span");
  p.lat_outer_bits = local_bits - p.lat_inner_bits;
  for (int l = 0; l < D; ++l) {
    p.lat_oshift[l] = l < D - 1 ? p.lat_run_shift[l] - p.lat_inner_bits : 0;
  }
  p.atoms_per_unit_shift = p.lat_inner_bits > kAtomShift ? p.lat_inner_bits - kAtomShift : 0;
  p.lat_units_per_atom_shift = p.lat_inner_bits < kAtomShift ? kAtomShift - p.lat_inner_bits : 0;

  int mask_total = 0;
  int olist_total = 0;
  int dlist_total = 0;
  for (int l = 0; l < D; ++l) {
    p.lat_sbits[l] = ilog2_exact(p.lat_s[l], "the level's owner span");
    p.mask_off[l] = mask_total;
    p.mask_words[l] = ceil_div_int(p.level_width[l], 32);
    mask_total += p.mask_words[l];
    p.olist_off[l] = olist_total;
    olist_total += p.lat_g[l];
    p.dlist_off[l] = dlist_total;
    if (l < D - 1) dlist_total += p.level_width[l];
  }
  p.mask_total = mask_total;
  p.olist_total = olist_total;
  p.dlist_total = dlist_total;
}

//: The per-level geometry every decode entry validates the same way, filled into `p` and -- see docs/internals/decode/decode.md#filllevels
int fill_levels(DecodeParams& p, const std::vector<Tensor>& levels,
                const std::vector<int64_t>& level_widths, int64_t B, int64_t H, int64_t N) {
  int total_rows = 0;
  int64_t expect_n = 1;
  for (int l = 0; l < p.D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    STD_TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                    " exceeds the decode path's capacity ", kDecodeMaxLevelWidth,
                    " (see kProducerMaxBranchWidth)");
    p.write[l] = span_of(levels[l], B, H, width, "write[l]");
    p.level_width[l] = width;
    p.level_amp_offset[l] = total_rows;
    total_rows += width;
    expect_n *= width;
  }
  STD_TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
                  " disagrees with the state's leaf axis ", N);
  return total_rows;
}

int64_t smem_budget_of(int64_t device_index) {
  int smem_per_block = 0;
  ROLA_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                                         static_cast<int>(device_index)));
  return smem_per_block;
}

}  // namespace detail

int64_t rola_decode_capacity() { return kDecodeMaxLevelWidth; }

int64_t rola_decode_ledger_warps() {
  check_arch_table();
  return detail::step_warps(current_card());
}

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately -- see docs/internals/decode/decode.md#rola-decode-producer-width-mirror
int64_t rola_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

//: THE BUILD STAMP: a path or hash check cannot catch a stale `.so` at the right path, -- see docs/internals/decode/decode.md#roladecodebuildstamp
int64_t rola_decode_build_stamp() {
  auto out = empty_cuda({1}, Dtype::Long);
  build_switch(current_card(), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB))
      detail::stampdet::decode_stamp_kernel<::denseref::arch::BuildTag<kB>>
          <<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  });
  ROLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

Tensor rola_decode_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                           std::vector<Tensor> read_scale, std::vector<Tensor> write_scale,
                           int64_t floor_read, int64_t floor_write, std::optional<Tensor> dials,
                           Tensor v, Tensor v_scale, Tensor state, Tensor yacc, Tensor queue,
                           Tensor growth, Tensor growth_any, Tensor growth_ctr, Tensor done,
                           Tensor atom_bits, std::vector<int64_t> level_widths,
                           std::vector<int64_t> level_row_offsets, int64_t lattice_k,
                           int64_t lattice_m, bool write_dense, int64_t ctas, double eps,
                           std::optional<Tensor> page_table, std::optional<Tensor> pool_slots,
                           std::optional<Tensor> pool_cursor) {
  check_arch_table();
  const int D = static_cast<int>(read.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  STD_TORCH_CHECK(static_cast<int>(write.size()) == D && static_cast<int>(level_widths.size()) == D
                      && static_cast<int>(level_row_offsets.size()) == D,
                  "read, write and the two per-level integer vectors must each have D entries");

  STD_TORCH_CHECK(v.dim() == 4 && v.size(1) == 1, "v must be [B, 1, H, d_v], got ",
                  shape(v.sizes()));
  const int B = static_cast<int>(v.size(0));
  const int H = static_cast<int>(v.size(2));
  const int BH = B * H;
  const int d_v = static_cast<int>(v.size(3));
  const int cols = d_v + 1;  // the mass column is unconditional (raw is dead)

  STD_TORCH_CHECK(state.is_cuda() && state.scalar_type() == Dtype::Byte && state.is_contiguous(),
                  "state is the int8 state format's pages, a contiguous CUDA uint8 tensor; got ",
                  state.scalar_type());
  const int64_t page_bytes =
      d_v == 32 ? state_page::Blocks<32>::kBytes : state_page::Blocks<64>::kBytes;
  detail::check_f32(yacc, "yacc");
  detail::check_i32(queue, "queue");

  //: THE TWO BACKINGS, validated where they differ and nowhere else. -- see docs/internals/decode/decode.md#note-l1083
  int64_t N = 0;
  int atoms_per_bh = 0;
  int32_t* page_ptr = nullptr;
  if (page_table.has_value()) {
    Tensor& tbl = *page_table;
    detail::check_i32(tbl, "page_table");
    STD_TORCH_CHECK(tbl.dim() == 2 && tbl.size(0) == BH, "page_table must be [BH, N / ",
                    kAtomLeaves, "] int32");
    atoms_per_bh = static_cast<int>(tbl.size(1));
    N = static_cast<int64_t>(atoms_per_bh) * kAtomLeaves;
    STD_TORCH_CHECK(state.dim() == 2 && state.size(1) == page_bytes,
                    "under a page table the state is the arena's plane, [slots, ", page_bytes,
                    "] bytes");
    page_ptr = tbl.mutable_data_ptr<int32_t>();
  } else {
    STD_TORCH_CHECK(state.dim() == 3 && state.size(0) == BH && state.size(2) == page_bytes,
                    "a dense state is [BH, N / ", kAtomLeaves, ", ", page_bytes, "] bytes");
    N = 1;
    for (auto w : level_widths) N *= w;
    STD_TORCH_CHECK(N % kAtomLeaves == 0,
                    "THERE IS NO RAGGED STATE: N = prod_l B_l must be a whole number of ",
                    kAtomLeaves,
                    "-leaf pages, and a lawful routing makes it one (every B_l a "
                    "power of two at or above 16). PREFILL AND DECODE SHARE ONE ADMISSION "
                    "LAW; got N = ",
                    N);
    atoms_per_bh = static_cast<int>(N / kAtomLeaves);
    STD_TORCH_CHECK(state.size(1) == atoms_per_bh, "a dense state holds N / ", kAtomLeaves, " = ",
                    atoms_per_bh, " pages a batch-head; got ", state.size(1));
  }

  DecodeParams p{};
  p.D = D;
  const int total_rows = detail::fill_levels(p, write, level_widths, B, H, N);
  detail::fill_lattice(p, lattice_k, lattice_m, N);
  //: THE COEFFICIENT ROWS: u8, a dense level's f16 (the floor stored). -- see docs/internals/decode/decode.md#coefficient-rows
  STD_TORCH_CHECK(
      read_scale.size() == static_cast<size_t>(D) && write_scale.size() == static_cast<size_t>(D),
      "one row scale a level a side");
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::span_of(read[l], B, H, p.level_width[l], "read[l]");
    const bool read_half = ((floor_read >> l) & 1) != 0, write_half = ((floor_write >> l) & 1) != 0;
    STD_TORCH_CHECK(read[l].scalar_type() == (read_half ? Dtype::Half : Dtype::Byte)
                        && write[l].scalar_type() == (write_half ? Dtype::Half : Dtype::Byte),
                    "the coefficient rows are u8, a dense level's f16; level ", l, " is ",
                    read[l].scalar_type(), " and ", write[l].scalar_type());
    detail::check_f32_span(read_scale[l], "read_scale[l]");
    detail::check_f32_span(write_scale[l], "write_scale[l]");
    p.read_scale[l] = detail::scalar_span_of(read_scale[l], B, H, "read_scale[l]");
    p.write_scale[l] = detail::scalar_span_of(write_scale[l], B, H, "write_scale[l]");
    p.level_row_offset[l] = static_cast<int>(level_row_offsets[l]);
  }
  //: whole atoms need the innermost run to cover one: a span of at least an atom (decode.md#whole-atoms).
  p.write_atoms = (write_dense && p.lat_inner_bits >= kAtomShift) ? 1 : 0;
  STD_TORCH_CHECK(
      floor_read >= 0 && floor_read < (1 << D) && floor_write >= 0 && floor_write < (1 << D),
      "the floor bitmasks name levels inside the depth ", D, ": ", floor_read, ", ", floor_write);
  p.floor_read = static_cast<uint32_t>(floor_read);
  p.floor_write = static_cast<uint32_t>(floor_write);
  //: THE VALUE: f16 against the token's largest channel, the fp32 token scale.
  STD_TORCH_CHECK(v.scalar_type() == Dtype::Half, "v is f16, got ", v.scalar_type());
  p.v = detail::span_of(v, B, H, d_v, "v");
  detail::check_f32_span(v_scale, "v_scale");
  p.v_scale = detail::scalar_span_of(v_scale, B, H, "v_scale");

  STD_TORCH_CHECK(ctas == -1 || ctas >= 1,
                  "ctas is -1 (the card's grid) or a gate's forced count of at least 1, got ",
                  ctas);
  STD_TORCH_CHECK(yacc.dim() == 2 && yacc.size(0) == BH && yacc.size(1) == cols,
                  "yacc must be [BH, cols]");
  STD_TORCH_CHECK(queue.dim() == 1 && queue.size(0) == 4 * static_cast<int64_t>(BH) + 2,
                  "queue must be [4 BH + 2] int32");

  //: THE ADMISSION BUFFERS, checked whichever backing this is: the step's own -- see docs/internals/decode/decode.md#note-l1131
  detail::check_i32(growth, "growth");
  detail::check_i32(growth_any, "growth_any");
  detail::check_i32(growth_ctr, "growth_ctr");
  detail::check_i32(done, "done");
  STD_TORCH_CHECK(growth.dim() == 1 && growth.size(0) == BH, "growth must be [BH] int32");
  STD_TORCH_CHECK(done.dim() == 1 && done.size(0) == BH, "done must be [BH] int32");
  STD_TORCH_CHECK(growth_any.dim() == 1 && growth_any.size(0) == 1, "growth_any must be [1] int32");
  STD_TORCH_CHECK(growth_ctr.dim() == 1 && growth_ctr.size(0) == 1, "growth_ctr must be [1] int32");
  STD_TORCH_CHECK(
      atom_bits.is_cuda() && atom_bits.scalar_type() == Dtype::Bool && atom_bits.is_contiguous(),
      "atom_bits must be a contiguous CUDA bool tensor, got ", atom_bits.scalar_type());
  STD_TORCH_CHECK(
      atom_bits.dim() == 2 && atom_bits.size(0) == BH && atom_bits.size(1) == N / kAtomLeaves,
      "atom_bits must be [BH, N / ", kAtomLeaves, "] bool, got ", shape(atom_bits.sizes()));

  //: THE ATOM'S PLACE IN THE LATTICE, required of a PAGED backing only, and now -- see docs/internals/decode/decode.md#torch-check
  STD_TORCH_CHECK(page_ptr == nullptr || p.lat_local_bits >= kAtomShift,
                  "a paged decode needs one walk unit's leaves to hold a whole number of ",
                  kAtomLeaves, "-leaf atoms, and this (k, m) gives an owner only ",
                  (1 << p.lat_local_bits), " (the innermost span must cover an atom)");

  const bool decay = dials.has_value();
  const int64_t budget = detail::smem_budget_of(v.get_device_index());

  if (decay) {
    detail::check_f32(*dials, "dials");
    STD_TORCH_CHECK(dials->dim() == 2 && dials->size(0) == H && dials->size(1) == total_rows,
                    "dials must be [H, sum_l width_l]");
    p.dials = dials->mutable_data_ptr<float>();
  }

  //: `empty`, not `zeros`: every element is written, so a zero-fill is a launch the -- see docs/internals/decode/decode.md#y
  auto y = empty_cuda({BH, d_v}, Dtype::Float, v.get_device_index());
  p.state = state.mutable_data_ptr<uint8_t>();
  p.page_tbl = page_ptr;
  p.atoms_per_bh = atoms_per_bh;
  p.y = y.mutable_data_ptr<float>();
  p.yacc = yacc.mutable_data_ptr<float>();
  p.queue = queue.mutable_data_ptr<int32_t>();
  p.growth = growth.mutable_data_ptr<int32_t>();
  p.growth_any = growth_any.mutable_data_ptr<int32_t>();
  p.growth_ctr = growth_ctr.mutable_data_ptr<int32_t>();
  p.done = done.mutable_data_ptr<int32_t>();
  p.atom_bits = atom_bits.mutable_data_ptr<bool>();

  //: THE SLACK POOL, present or absent as ONE decision -- two buffers and a capacity, -- see docs/internals/decode/decode.md#pooled
  const bool pooled = pool_slots.has_value();
  STD_TORCH_CHECK(pooled == pool_cursor.has_value(),
                  "the slack pool is its two buffers together: pass pool_slots and pool_cursor, "
                  "or neither");
  if (pooled) {
    STD_TORCH_CHECK(page_table.has_value(),
                    "a slack pool admits into a page table and this state is dense-backed");
    detail::check_i32(*pool_slots, "pool_slots");
    detail::check_i32(*pool_cursor, "pool_cursor");
    STD_TORCH_CHECK(pool_slots->dim() == 2 && pool_slots->size(0) == BH,
                    "pool_slots must be [BH, pool_cap] int32, got ", shape(pool_slots->sizes()));
    STD_TORCH_CHECK(pool_cursor->dim() == 1 && pool_cursor->size(0) == BH,
                    "pool_cursor must be [BH] int32");
    p.pool_slots = pool_slots->mutable_data_ptr<int32_t>();
    p.pool_cursor = pool_cursor->mutable_data_ptr<int32_t>();
    p.pool_cap = static_cast<int>(pool_slots->size(1));
  }
  p.N = N;
  p.BH = BH;
  p.H = H;
  //: the step's grid: the card's (a unit a run of the innermost level's span), or a gate's forced count
  p.ctas = ctas < 0 ? detail::grid_of(card_of(v.get_device_index()), N / p.lat_s[D - 1], BH)
                    : static_cast<int>(ctas);
  p.total_rows = total_rows;
  p.eps = static_cast<float>(eps);
  p.ledger = g_phase_ledger;
  if (p.ledger != nullptr)
    STD_TORCH_CHECK(g_phase_ledger_ctas >= p.ctas, "the bound decode phase ledger covers ",
                    g_phase_ledger_ctas, " CTAs; this step launches ", p.ctas);

  launch_decode_step(card_of(v.get_device_index()), p, d_v, decay, budget);
  return y;
}

}  // namespace decode
}  // namespace rola
