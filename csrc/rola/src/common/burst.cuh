// csrc/rola/src/common/burst.cuh -- THE BURST: the one loop shape that issues MMAs. A counted loop
// over units, two operand sets alternating, each unit's gather issued under the burst before it.
// See docs/internals/common/burst.md
#pragma once

namespace rola::burst {

//: `x` whole, with a data dependency on `y`. -- burst.md#run
__device__ __forceinline__ uint32_t after(uint32_t x, uint32_t y) {
  uint32_t r;
  asm volatile("prmt.b32 %0, %1, %2, 0x3210;\n" : "=r"(r) : "r"(x), "r"(y));
  return r;
}

//: units `first .. first + count - 1`, two sets alternating. -- burst.md#run
template <class Ops>
struct Burst {
  template <class Gather, class Mma>
  __device__ __forceinline__ static void run(int first, int count, Gather&& gather, Mma&& mma) {
    if (count <= 0) return;
    Ops a, b;
    gather(first, a);
    int i = first;
    const int end = first + count;
#pragma unroll 1
    for (; i + 1 < end; i += 2) {
      gather(i + 1, b);
      mma(a, b);
      if (i + 2 < end) gather(i + 2, a);
      mma(b, a);
    }
    if (i < end) mma(a, b);
  }
};

}  // namespace rola::burst
