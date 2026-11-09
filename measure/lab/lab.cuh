// measure/lab/lab.cuh -- THE LAB'S KIT: what every experiment shares -- a warp's cycles over a region, a sink the
// compiler cannot drop, the atoms as PTX, and the host runner that launches a variant, repeats it, prints one JSON line
// a result and declares the SASS each kernel must contain (checked by `tools/lab.py` before any number is kept).
// see docs/internals/tools/lab.md
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/hardware.cuh"

namespace lab {

//: THE BUILD an experiment compiles at: its one architecture's own, the same in both passes (`tools/lab.py` passes
//: `ROLA_LAB_CC`; docs/internals/tools/lab.md#build)
constexpr denseref::arch::Build kLabBuild{ROLA_LAB_CC, 0};

// ------------------------------------------------------------------------------------------------- device

__device__ __forceinline__ int warp_id() { return static_cast<int>(threadIdx.x) >> 5; }

__device__ __forceinline__ int lane_id() { return static_cast<int>(threadIdx.x) & 31; }

//: A WARP'S CYCLES over a region, lane 0's `clock64`, stored at `(block, warp)`.
struct WarpClock {
  long long t0 = 0;

  __device__ __forceinline__ void start() { t0 = clock64(); }

  __device__ __forceinline__ void stop(long long* out) {
    const long long dt = clock64() - t0;
    if (lane_id() == 0) out[blockIdx.x * (blockDim.x >> 5) + warp_id()] = dt;
  }
};

//: A WARP'S SPAN, its start and end `clock64` (the SM's counter), stored at `2 (block, warp)`: a block's span -- its
//: last warp's end less its first warp's start -- is the throughput when warps finish apart, where the mean warp's
//: cycles undercount.
struct WarpSpan {
  long long t0 = 0;

  __device__ __forceinline__ void start() { t0 = clock64(); }

  __device__ __forceinline__ void stop(long long* out) {
    const long long t1 = clock64();
    const int at = 2 * (blockIdx.x * (blockDim.x >> 5) + warp_id());
    if (lane_id() == 0) {
      out[at] = t0;
      out[at + 1] = t1;
    }
  }
};

//: THE SINK: a fold of everything a thread computed, stored unconditionally. A result tested against a value the
//: compiler can fold lets ptxas delete the work behind it -- the failure the expectations exist to catch.
__device__ __forceinline__ void sink(int* out, uint32_t v) {
  out[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<int>(v);
}

__device__ __forceinline__ uint32_t bits(float x) { return __float_as_uint(x); }

__device__ __forceinline__ uint32_t bits(uint32_t x) { return x; }

__device__ __forceinline__ uint32_t bits(int x) { return static_cast<uint32_t>(x); }

//: THE ATOMS, one PTX instruction each, so a variant names exactly what it issues.
__device__ __forceinline__ void mma_f16_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16_f32_fresh(float (&d)[4], const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
}

__device__ __forceinline__ void mma_f16_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                             uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_u8u8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//: THE OPERAND PIECES the kernels use: `ldmatrix.x4` and its `.trans`, a packed f16 product, two fp32 to an f16 pair.
__device__ __forceinline__ void ldsm(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ void ldsm_t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

//: `v` HELD in a register across the loops that use it: an empty asm hides its derivation, so ptxas cannot
//: rematerialize it (a lane's constants otherwise come back from `S2R` and their arithmetic at every use)
__device__ __forceinline__ uint32_t held(uint32_t v) {
  asm volatile("" : "+r"(v));
  return v;
}

//: THE ASYNCHRONOUS COPIES (`cp.async`, sm_80+) as `ops.cuh` spells them: 16 bytes bypassing L1, or 4 through it; a
//: group committed and waited, or an mbarrier arrival when all of the thread's prior copies land.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

//: THE SHARED-MEMORY BARRIER (`mbarrier`, sm_80+), as `ops.cuh` spells it: `count` arrivals complete a phase, a phase's
//: parity flips at each completion, a wait polls until the phase of `parity` has completed.
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_when_landed(uint32_t a) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT_%=:\nmbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n@!p bra LAB_WAIT_%=;\n}" ::
          "r"(a),
      "r"(parity)
      : "memory");
}

//: A NAMED BARRIER (ids 1-15; 0 is `__syncthreads`'s): `count` threads, a multiple of 32, complete it; `bar_sync`
//: arrives and waits, `bar_arrive` arrives and goes on
__device__ __forceinline__ void bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

//: the same, asked once: has the phase of parity `parity` completed?
__device__ __forceinline__ bool mbar_test(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(a), "r"(parity)
      : "memory");
  return ok != 0u;
}

// --------------------------------------------------------------------------------------------------- host

//: A KERNEL'S DEMANGLED NAME AS AN EXPECTATION'S PATTERN: `kernel("tax", {3, 17})` matches
//: `ns::tax<(int)3, (int)17>(...)`, the spelling `cu++filt` gives integer template arguments.
inline std::string kernel(const std::string& name, std::initializer_list<int> args) {
  std::string out = name + "<";
  int i = 0;
  for (int a : args) out += std::string(i++ ? ", " : "") + "\\(int\\)" + std::to_string(a);
  return out + ">\\(";
}

#define LAB_CHECK(call)                                                                       \
  do {                                                                                        \
    const cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "lab: %s failed: %s\n", #call, cudaGetErrorString(e_));            \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

//: A RESULT'S SAMPLES, one a repetition.
struct Samples {
  std::vector<double> v;

  double median() const {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s.empty() ? 0.0 : s[s.size() / 2];
  }
};

//: THE RUNNER. `--expect` prints the declared SASS and runs nothing; otherwise each `variant` warms up once, repeats
//: `--reps` times, and `record` prints a result as one JSON line `tools/lab.py` reads.
class Bench {
 public:
  static constexpr int kMaxWarps = 32;
  int sms = 0, reps = 7, blocks = 0;
  bool expect_only = false;
  long long* times = nullptr;
  int* sinks = nullptr;

  Bench(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--expect")) expect_only = true;
      if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
      if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = std::atoi(argv[++i]);
    }
    if (expect_only) return;
    LAB_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    //: `--blocks N`: the grid an experiment launches over `sms` held to N CTAs (a sanitizer's run at one CTA)
    if (blocks > 0 && blocks < sms) sms = blocks;
    LAB_CHECK(cudaMalloc(&times, sizeof(long long) * sms * kMaxWarps));
    LAB_CHECK(cudaMalloc(&sinks, sizeof(int) * sms * kMaxWarps * 32));
    cudaDeviceProp prop{};
    LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"lab\": \"device\", \"name\": \"%s\", \"cc\": %d%d, \"sms\": %d}\n", prop.name,
                prop.major, prop.minor, sms);
  }

  //: a kernel whose demangled name matches `kernel` must hold `op` `count` times (`op` an opcode prefix, `cmp` one of
  //: ">=", "==", "<=")
  void expect(const std::string& kernel, const std::string& op, const char* cmp, int count) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf(
        "{\"lab\": \"expect\", \"kernel\": \"%s\", \"op\": \"%s\", \"cmp\": \"%s\", \"count\": %d}\n",
        escaped.c_str(), op.c_str(), cmp, count);
  }

  //: THE INSTRUCTION BUDGET (KERNEL_STANDARDS section 29): a kernel whose demangled name matches `kernel` executes at
  //: most `max` instructions of opcode class `cls` (`tools/sass.py`'s `OPCODE_CLASSES`) per tensor-pipe instruction,
  //: stated from the design's operation count before its code; `tools/lab.py run` reads the executed counts (ncu) and
  //: refuses to record a variant over it
  void budget(const std::string& kernel, const char* cls, double max) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"budget\", \"kernel\": \"%s\", \"class\": \"%s\", \"max\": %.3f}\n", escaped.c_str(),
                cls, max);
  }

  //: AN MMA KERNEL WITHOUT A BUDGET, on purpose (KERNEL_STANDARDS section 29): a rate probe or an isolation, its reason
  void exempt(const std::string& kernel, const char* reason) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"exempt\", \"kernel\": \"%s\", \"reason\": \"%s\"}\n", escaped.c_str(), reason);
  }

  //: A VARIANT'S OUTPUTS CHECKED against a host reference (KERNEL_STANDARDS section 29): `tools/lab.py run` records no
  //: variant whose check failed, nor one with neither a check nor `unchecked`; `detail` says what was compared
  void check(const std::string& variant, bool ok, const std::string& detail) const {
    std::printf("{\"lab\": \"check\", \"variant\": \"%s\", \"ok\": %s, \"detail\": \"%s\"}\n", variant.c_str(),
                ok ? "true" : "false", detail.c_str());
  }

  //: a variant whose outputs are not checked, and why (an isolation whose outputs fold into a sink)
  void unchecked(const std::string& variant, const char* reason) const {
    std::printf("{\"lab\": \"unchecked\", \"variant\": \"%s\", \"reason\": \"%s\"}\n", variant.c_str(), reason);
  }

  //: THE MMA a budgeted kernel issues (`common/hardware.json`'s kind, e.g. "hmma_bf16_f32", and its K: 16 or 8), so
  //: `tools/lab.py run` prices its counts on every card: tensor, shared-memory and issue cycles an MMA and their max
  void mma(const std::string& kernel, const char* kind, int k) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"mma\", \"kernel\": \"%s\", \"kind\": \"%s\", \"k\": %d}\n", escaped.c_str(), kind, k);
  }

  //: THE UNIT a kernel without MMAs is budgeted per: its first launch does `n` of them (a box-window colored, say), so
  //: `tools/lab.py run` divides its executed counts by `n` where an MMA kernel's divide by its MMAs
  void units(const std::string& kernel, const char* unit, long long n) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"units\", \"kernel\": \"%s\", \"unit\": \"%s\", \"n\": %lld}\n", escaped.c_str(),
                unit, n);
  }

  //: the per-(block, warp) cycles of the last launch, `warps` a block
  std::vector<long long> cycles(int warps) const {
    std::vector<long long> h(static_cast<size_t>(sms) * warps);
    LAB_CHECK(cudaMemcpy(h.data(), times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    return h;
  }

  //: the mean over blocks of the block's span (`WarpSpan`), `warps` a block, `blocks` launched (one an SM if omitted)
  double span(int warps, int blocks = 0) const {
    const int n = blocks > 0 ? blocks : sms;
    std::vector<long long> h(static_cast<size_t>(n) * warps * 2);
    LAB_CHECK(cudaMemcpy(h.data(), times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int b = 0; b < n; ++b) {
      long long lo = h[2 * b * warps], hi = h[2 * b * warps + 1];
      for (int w = 1; w < warps; ++w) {
        lo = std::min(lo, h[2 * (b * warps + w)]);
        hi = std::max(hi, h[2 * (b * warps + w) + 1]);
      }
      s += double(hi - lo);
    }
    return s / n;
  }

  //: the mean over blocks of warps `[first, first + count)`'s cycles
  static double mean(const std::vector<long long>& t, int warps, int first, int count) {
    double s = 0;
    const size_t blocks = t.size() / warps;
    for (size_t b = 0; b < blocks; ++b)
      for (int w = first; w < first + count; ++w) s += double(t[b * warps + w]);
    return s / double(blocks * count);
  }

  //: one variant's samples: `launch()` runs it once, `measure()` reads the value from its cycles
  template <class Launch, class Measure>
  Samples variant(Launch launch, Measure measure) const {
    Samples out;
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
      launch();
      LAB_CHECK(cudaDeviceSynchronize());
      out.v.push_back(measure());
    }
    return out;
  }

  //: a result, as one JSON line
  void record(const std::string& name, const char* unit, const Samples& s) const {
    std::string vals;
    for (size_t i = 0; i < s.v.size(); ++i) vals += (i ? ", " : "") + std::to_string(s.v[i]);
    std::printf("{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"%s\", \"samples\": [%s]}\n",
                name.c_str(), unit, vals.c_str());
  }
};

}  // namespace lab
