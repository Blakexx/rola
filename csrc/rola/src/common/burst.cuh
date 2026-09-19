// csrc/rola/src/common/burst.cuh -- THE BURST: the one loop shape that issues MMAs. A counted loop
// over units, two operand sets alternating, each unit's gather issued under the burst before it.
// See docs/internals/common/burst.md
#pragma once

namespace rola::burst {

__device__ __forceinline__ uint32_t after(uint32_t x, uint32_t y) {
  uint32_t r;
  asm volatile("prmt.b32 %0, %1, %2, 0x3210;\n" : "=r"(r) : "r"(x), "r"(y));
  return r;
}

template <class Ops>
struct Burst {
  //: units `first .. first + count - 1`, two sets alternating; `mma(cur, next, unit)` bursts
  //: `cur` while gathering `unit` into `next`, `last` with nothing to gather. -- burst.md#fork
  //: @burst-exempt the primitive itself: the loop and its guards on the uniform count
  template <class Gather, class Mma, class Last>
  __device__ __forceinline__ static void run(int first, int count, Gather&& gather, Mma&& mma,
                                             Last&& last) {
    if (count <= 0) return;
    Ops a, b;
    gather(first, a);
    int i = first;
    const int end = first + count;
#pragma unroll 1
    for (; i + 2 < end; i += 2) {
      mma(a, b, i + 1);
      mma(b, a, i + 2);
    }
    if (i + 1 < end) {
      mma(a, b, i + 1);
      last(b, a);
    } else {
      last(a, b);
    }
  }
};

}  // namespace rola::burst
