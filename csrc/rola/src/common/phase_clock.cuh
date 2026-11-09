// csrc/rola/src/common/phase_clock.cuh -- THE PHASE LEDGER AND TRACE, one warp's: cycles per phase
// accumulated by lane 0 into the warp's shared row and added to the launch's ledger at the end; when a
// trace is bound, a stamp per lap and per stream step. Shared by every kernel family that measures itself.
// See docs/internals/carry/ampere/carry_kernel.md#phase-ledger
#pragma once

#include <cuda_runtime.h>

#include "common/ops.cuh"

namespace rola {

//: `Gated`: an unbound clock's laps issue nothing. -- see docs/internals/common/phase_clock.md#on
template <int Phases, bool Gated = false>
struct PhaseClock {
  //: A warp's shared row: its `Phases` accumulators, then the last lap's mark, so the clock holds no register across a phase.
  static constexpr int kRow = Phases + 1;
  long long* acc;
  long long* trace;
  int n, cap;
  bool on, lead;

  __device__ __forceinline__ PhaseClock(long long* row, int lane, bool on_ = true)
      : acc(row), trace(nullptr), n(0), cap(0), on(on_), lead(on_ && lane == 0) {
    constexpr int kCount = Phases;
    if (lead) {
#pragma unroll
      for (int ph = 0; ph < kCount; ++ph) acc[ph] = 0;
      acc[kCount] = denseref::ops::cycles();
    }
  }

  __device__ __forceinline__ void bind_trace(long long* row, int cap_) {
    trace = row;
    cap = cap_;
  }

  __device__ __forceinline__ void record(long long now, int ev) {
    if (trace != nullptr && n < cap) {
      trace[n] = (now << 8) | (long long)ev;
      ++n;
    }
  }

  //: an activity begins on this warp: a stamp when the trace is bound.
  __device__ __forceinline__ void stamp(int ev) {
    if (lead) record(denseref::ops::cycles(), ev);
  }

  //: the cycles since the last mark go to phase `ph`.
  //: The mark's load is the instruction a deferred barrier issues before it blocks, so a lap right after
  //: `__syncthreads` books the release, not the arrival -- see docs/internals/common/phase_clock.md#deferred-barrier
  __device__ __forceinline__ void lap(int ph) {
    if constexpr (Gated) {
      for (int k = denseref::ops::once(on); k > 0; --k) lap_lead(ph);
    } else {
      lap_lead(ph);
    }
  }

  __device__ __forceinline__ void lap_lead(int ph) {
    if (lead) {
      const long long mark = acc[Phases];
      const long long now = denseref::ops::cycles_after(mark);
      acc[ph] += now - mark;
      acc[Phases] = now;
      record(now, ph);
    }
  }

  //: the warp's row into the launch's ledger, `[cta][warp][Phases]`.
  __device__ __forceinline__ void publish(long long* ledger, long cta, int warps, int warp) const {
    if (ledger == nullptr || !lead) return;
    constexpr int kCount = Phases;
    long long* const row = ledger + (cta * warps + warp) * kCount;
#pragma unroll
    for (int ph = 0; ph < kCount; ++ph)
      atomicAdd(reinterpret_cast<unsigned long long*>(row + ph), (unsigned long long)acc[ph]);
  }
};

}  // namespace rola
