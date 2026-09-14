#pragma once

// LAYER 2a -- THE DERIVED SELECTIONS: what the design LOOKS LIKE on this arch.
// Every entry is COMPUTED from `arch_caps.cuh`'s facts, never looked up; the
// kernel reads this layer for its design and calls `ops.cuh` to execute it.
// See docs/internals/common/design.md

#include "common/arch_caps.cuh"

namespace denseref {
namespace design {

// THE COMPILE-TIME ANNOUNCEMENT: a diagnostic that does NOT fail the build; see design.md#announce
template <bool Suboptimal>
struct Announce {
  __host__ __device__ constexpr Announce() {}
};

template <>
struct Announce<true> {
  [[deprecated(
      "KNOWINGLY SUBOPTIMAL SELECTION: this architecture's capability row "
      "offers a better implementation of a baseline operation than the one "
      "this baseline builds. The build is correct and the gap is stated; see "
      "the announcement site in ops.cuh for which operation.")]] __host__
  __device__ constexpr Announce() {}
};

// STATE RESIDENCY -- where the state lives; see docs/internals/common/design.md#note-l36
enum class StateResidency { REGISTERS, TMEM, SHARED };

constexpr unsigned residency_bit(StateResidency r) { return 1u << (unsigned)r; }

// see docs/internals/common/design.md#satisfiableresidencies -- the mask meeting the requirement
constexpr unsigned satisfiable_residencies(const arch::Caps& c) {
  return (c.accum_in_tmem && c.mma_a_from_tmem ? residency_bit(StateResidency::TMEM) : 0u)
         | (!c.accum_in_tmem && c.mma_a_from_regs ? residency_bit(StateResidency::REGISTERS) : 0u);
}

// TMEM preferred where it exists; see docs/internals/common/design.md#near-line-51
constexpr StateResidency kResidencyOrder[2] = {StateResidency::TMEM, StateResidency::REGISTERS};

// see docs/internals/common/design.md#selectresidency -- best residency meeting the requirement
constexpr StateResidency select_residency(unsigned satisfiable, unsigned built) {
  return (satisfiable & built & residency_bit(kResidencyOrder[0]))   ? kResidencyOrder[0]
         : (satisfiable & built & residency_bit(kResidencyOrder[1])) ? kResidencyOrder[1]
                                                                     : StateResidency::SHARED;
}

// see docs/internals/common/design.md#residencysuboptimal
constexpr bool residency_suboptimal(unsigned satisfiable, unsigned built) {
  return (satisfiable & residency_bit(kResidencyOrder[0]))
         && select_residency(satisfiable, built) == kResidencyOrder[1];
}

constexpr unsigned kSatisfiableResidencies = satisfiable_residencies(arch::kCaps);

// A part with no satisfying residency at all is UNSUPPORTED (distinct from NOT BUILT, ops.cuh's).
static_assert(kSatisfiableResidencies != 0u,
              "UNSUPPORTED ARCHITECTURE: on this part the MMA's accumulator "
              "cannot be reused as an operand without a memory round trip, in "
              "any storage class. The fold accumulator IS the readout's "
              "operand; a design that stores and reloads it is a different "
              "algorithm, not a slower arm of this one.");

//: THE RESIDENCIES THIS BASELINE BUILDS. -- see docs/internals/common/design.md#kbuiltresidencies
constexpr unsigned kBuiltResidencies = residency_bit(StateResidency::REGISTERS);

constexpr StateResidency kStateResidency =
    select_residency(kSatisfiableResidencies, kBuiltResidencies);

static_assert(kStateResidency != StateResidency::SHARED,
              "NOT BUILT FOR THIS ARCHITECTURE: the residency this part meets "
              "the accumulator-as-operand requirement in has no implementation "
              "here (a `tcgen05` accumulator needs the TMEM path, which is not "
              "written). Every structural clause holds; this is a capability "
              "gap, not a refused structure.");

//: Announced, not refused: the built path is correct on such a part, it is -- see docs/internals/common/design.md#kresidencysuboptimal
constexpr bool kResidencySuboptimal =
    residency_suboptimal(kSatisfiableResidencies, kBuiltResidencies);
inline constexpr Announce<kResidencySuboptimal> kResidencyAnnouncement{};

// WARP SPECIALIZATION -- whether a unit may be dedicated to production; see design.md#kspecializationpays
constexpr bool specialization_pays(const arch::Caps& c) { return c.dyn_reg_realloc; }

constexpr bool kSpecializationPays = specialization_pays(arch::kCaps);

constexpr bool kBuiltSpecialization =
    false;  // NOT BUILT, announced not refused; see design.md#kbuiltspecialization
inline constexpr Announce<kSpecializationPays && !kBuiltSpecialization>
    kSpecializationAnnouncement{};

// STAGES -- the ring's depth, derived; see docs/internals/common/design.md#kstagesmin
constexpr int kStagesMin = 2;  // the wait/issue order needs two groups
constexpr int kStagesMax = 8;  // the search bound, not a design constant

//: The largest per-lane register count is a table quantity; this is its inverse -- see docs/internals/common/design.md#ctasbyregfloor
constexpr int ctas_by_reg_floor(const arch::Caps& c, int threads, int reg_floor, int n) {
  return n <= 1 ? 1
                : (arch::reg_budget(c, threads, n) >= reg_floor
                       ? n
                       : ctas_by_reg_floor(c, threads, reg_floor, n - 1));
}

//: The residency ceiling that is NOT a function of this arm's shared memory: -- see docs/internals/common/design.md#ceilingwithoutsmem
constexpr int ceiling_without_smem(const arch::Caps& c, int threads, int reg_floor) {
  return ctas_by_reg_floor(c, threads, reg_floor,
                           arch::ctas_by_threads(c, threads) < c.max_ctas_per_sm
                               ? arch::ctas_by_threads(c, threads)
                               : c.max_ctas_per_sm);
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach.  Shared -- see docs/internals/common/design.md#ctatarget
constexpr int cta_target(const arch::Caps& c, int smem_bytes, int threads, int reg_floor) {
  return arch::ctas_by_smem(c, smem_bytes) < ceiling_without_smem(c, threads, reg_floor)
             ? arch::ctas_by_smem(c, smem_bytes)
             : ceiling_without_smem(c, threads, reg_floor);
}

//: A ring of `s` slots on top of `fixed` bytes of unringed buffers.
constexpr int ring_bytes(int slot_bytes, int fixed_bytes, int s) {
  return s * slot_bytes + fixed_bytes;
}

constexpr bool stages_fit(const arch::Caps& c, int slot_bytes, int fixed_bytes, int s) {
  return ring_bytes(slot_bytes, fixed_bytes, s) <= c.smem_per_cta_max;
}

//: The residency reachable at ANY depth. The ring can only shrink it, so this -- see docs/internals/common/design.md#note-l217
constexpr int best_residency(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                             int reg_floor, int s) {
  return s >= kStagesMax || stages_fit(c, slot_bytes, fixed_bytes, s)
             ? cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads, reg_floor)
             : best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, s + 1);
}

//: The SHALLOWEST ring that reaches `want` CTAs, searched upward from the -- see docs/internals/common/design.md#stagesbyresidency
constexpr int stages_by_residency(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                                  int reg_floor, int want, int s) {
  return s >= kStagesMax
             ? s
             : (stages_fit(c, slot_bytes, fixed_bytes, s)
                        && cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads, reg_floor)
                               >= want
                    ? s
                    : stages_by_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, want,
                                          s + 1));
}

constexpr int stages_for(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                         int reg_floor) {
  return stages_by_residency(
      c, slot_bytes, fixed_bytes, threads, reg_floor,
      best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, kStagesMin), kStagesMin);
}

//: The compiling target's answers, so a kernel writes a geometry and never a -- see docs/internals/common/design.md#stagesfor
constexpr int stages_for(int slot_bytes, int fixed_bytes, int threads, int reg_floor) {
  return stages_for(arch::kCaps, slot_bytes, fixed_bytes, threads, reg_floor);
}

constexpr int cta_target(int smem_bytes, int threads, int reg_floor) {
  return cta_target(arch::kCaps, smem_bytes, threads, reg_floor);
}

//: THE ONE HARD RESOURCE REFUSAL, from the table rather than from a literal: an -- see docs/internals/common/design.md#smemfits
constexpr bool smem_fits(size_t bytes) { return bytes <= (size_t)arch::kCaps.smem_per_cta_max; }

//: The digest of the capability row THE DEVICE CODE WAS COMPILED AGAINST, -- see docs/internals/common/design.md#kcapsdigest
constexpr int kCapsDigest = arch::caps_digest(arch::kCaps);

}  // namespace design
}  // namespace denseref
