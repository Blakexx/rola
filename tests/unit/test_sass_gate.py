"""The SASS gate (`tools/sass_gate.py`) on compiled probes: a clean kernel, one whose runtime-indexed local array must
go to local memory, one whose shuffle sits in a loop ptxas cannot prove warp-uniform (a convergence subroutine), and
one whose division calls its slow path (a call that is not one), and two whose MMA loops convert on the conversion unit
or read global memory (§29's step lints) beside one that converts on the FMA pipe and stages by `cp.async` -- the
gate's reds are what it reads, proven on a cubin that has them.

See KERNEL_STANDARDS §20, §29.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import dev_config  # noqa: E402
import sass_gate  # noqa: E402

PROBE = r"""
__global__ void clean_probe(float* y, const float* x) { y[threadIdx.x] = 2.0f * x[threadIdx.x]; }

__global__ void local_probe(float* y, const int* i) {
  float a[512];
  for (int k = 0; k < 512; ++k) a[k] = k * y[k];
  a[i[threadIdx.x + 32]] = y[threadIdx.x];
  y[threadIdx.x] = a[i[threadIdx.x]];
}

__global__ void converge_probe(float* y, const int* n) {
  float v = y[threadIdx.x];
  for (int i = 0; i < n[threadIdx.x]; ++i) v += __shfl_xor_sync(0xffffffffu, v, 1);
  y[threadIdx.x] = v;
}

__global__ void divide_probe(float* y, const float* x) { y[threadIdx.x] = x[threadIdx.x] / y[threadIdx.x + 32]; }

__device__ __forceinline__ void mma(float (&d)[4], unsigned a, unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b));
}

//: a step loop: an MMA, its outputs converted to s16 and stored; `XU` converts on the conversion unit (F2I), else on
//: the FMA pipe; `GLOBAL` reads its operand from global memory inside the step, else from shared memory
template <bool XU, bool GLOBAL>
__device__ void step_loop(int* y, const unsigned* x, int n) {
  __shared__ unsigned s[256];
  s[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  float d[4] = {};
  int out = 0;
  for (int i = 0; i < n; ++i) {
    const unsigned a = GLOBAL ? x[i * 32 + threadIdx.x] : s[(i + threadIdx.x) & 255];
    mma(d, a, a ^ i);
    out += XU ? __float2int_rn(d[0]) : static_cast<int>(__float_as_uint(fmaf(d[0], 1.0f, 12582912.0f)) & 0xffff);
  }
  y[threadIdx.x] = out;
}

__global__ void step_xu_probe(int* y, const unsigned* x, int n) { step_loop<true, false>(y, x, n); }
__global__ void step_global_probe(int* y, const unsigned* x, int n) { step_loop<false, true>(y, x, n); }
__global__ void step_clean_probe(int* y, const unsigned* x, int n) { step_loop<false, false>(y, x, n); }
"""


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    nvcc = dev_config.cuda_bin("nvcc")
    if nvcc is None or not Path(nvcc).exists():
        pytest.skip("no nvcc in this environment's toolkit")
    root = tmp_path_factory.mktemp("sass_gate")
    (root / "probe.cu").write_text(PROBE)
    subprocess.run([nvcc, "-cubin", "-arch=sm_86", "-O3", "-o", str(root / "probe.cubin"), str(root / "probe.cu")],
                   check=True, capture_output=True)
    return root / "probe.cubin"


def test_the_gate_passes_the_clean_kernel_and_reds_local_memory(cubin):
    ok, report = sass_gate.gate(cubin, None, None)
    #: the bare probe by its mangled name: `clean_probe` alone also matches the step probe `step_clean_probe`
    clean = sass_gate.STATS["_Z11clean_probePfPKf"]
    local = next(v for k, v in sass_gate.STATS.items() if "local_probe" in k)
    assert not clean["reds"], report
    assert local["stack"] > 0 and local["ldl"] > 0, report
    assert any("stack frame" in r for r in local["reds"]) and not ok


def test_the_step_lints_red_the_conversion_unit_and_global_memory_in_an_mma_loop(cubin):
    """§29: F2I inside an MMA loop and a global load inside one are red; the FMA-pipe conversion over shared memory is
    clean of both. The probes' step cover is §32's reading, not this test's subject: one dependent MMA a step leaves
    the clean probe under the line."""
    ok, report = sass_gate.gate(cubin, None, None)
    xu = next(v for k, v in sass_gate.STATS.items() if "step_xu_probe" in k)
    memory = next(v for k, v in sass_gate.STATS.items() if "step_global_probe" in k)
    clean = next(v for k, v in sass_gate.STATS.items() if "step_clean_probe" in k)
    assert xu["step_xu"] > 0 and any("conversion unit" in r for r in xu["reds"]), report
    assert memory["step_global"] > 0 and any("global memory" in r for r in memory["reds"]), report
    assert clean["step_xu"] == 0 and clean["step_global"] == 0, report
    assert not any("conversion unit" in r or "global memory" in r for r in clean["reds"]), report
    assert not ok


def test_a_narrowing_that_matches_nothing_is_red(cubin, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sass_gate.py", str(cubin), "--kernel", "no_such_kernel"])
    assert sass_gate.main() == 1
    assert "no function matched" in capsys.readouterr().out


def test_a_guarded_convergence_fallback_and_a_slow_path_call_are_not_red(cubin):
    ok, report = sass_gate.gate(cubin, None, None)
    converge = next(v for k, v in sass_gate.STATS.items() if "converge_probe" in k)
    divide = next(v for k, v in sass_gate.STATS.items() if "divide_probe" in k)
    assert converge["calls"] == 0 and converge["guarded"] == 1 and not converge["reds"], report
    assert divide["calls"] == 0 and divide["guarded"] == 0 and not divide["reds"], report


#: a convergence subroutine and a call into it, as a disassembly lists them: `(address, opcode, operands)`
_SUBROUTINE = [(0x100, "WARPSYNC", "0xffffffff", ""), (0x110, "SHFL.BFLY", "PT, R0, R1, 0x1, 0x1f", ""),
               (0x120, "RET.REL.NODEC", "R2 0x0", "")]


def test_the_classifier_reds_an_unguarded_call_and_counts_a_guarded_one():
    """This toolchain's ptxas guards every call it emits (the probe above), so the red is proven on the lines."""
    hot = [(0x0, "CALL.REL.NOINC", "0x100", ""), (0x10, "EXIT", "", "")] + _SUBROUTINE
    via_div = [(0x0, "BRA.DIV", "~URZ, 0x40", ""), (0x10, "SHFL.BFLY", "PT, R0, R1, 0x1, 0x1f", ""),
               (0x20, "EXIT", "", ""), (0x40, "CALL.REL.NOINC", "0x100", ""), (0x50, "BRA", "0x20", "")] + _SUBROUTINE
    past_conv = [(0x0, "BRA.CONV", "~URZ, 0x20", ""), (0x10, "CALL.REL.NOINC", "0x100", ""),
                 (0x20, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(hot, {}) == (1, 0)
    assert sass_gate.convergence_calls(via_div, {}) == (0, 1)
    assert sass_gate.convergence_calls(past_conv, {}) == (0, 1)
    never = [(0x0, "BRA", "0x20", "@!PT"), (0x10, "BRA", "0x30", ""), (0x20, "CALL.REL.NOINC", "0x100", ""),
             (0x30, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(never, {}) == (0, 1)
    reached = [(0x0, "BRA", "0x20", "@!PT"), (0x10, "MOV", "R0, R1", ""), (0x20, "CALL.REL.NOINC", "0x100", ""),
               (0x30, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(reached, {}) == (1, 0)


def test_a_loop_without_mmas_inside_an_mma_loop_is_its_body_and_hides_nothing():
    """The step cover's loops: an MMA loop holding a compare-and-swap spin (sm_80/86's shared fp32 add) is still the
    innermost MMA loop, the spin read as one trip of its body; a loop holding an inner MMA loop is not one. Until
    2026-11-01 any backward branch inside hid the loop, and the den's and the intra's steps were never read."""
    spin = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F32", "R8, R4, R6, R8", ""),
            (0x20, "LDS", "R9, [R3]", ""), (0x30, "FADD", "R10, R9, R8", ""),
            (0x40, "ATOMS.CAST.SPIN", "R10, [R3], R9, R10", ""), (0x50, "ISETP.EQ.U32.AND", "P3, PT, R10, 0x1, PT", ""),
            (0x60, "BRA", "0x20", "@!P3"), (0x70, "BRA", "0x0", "@P1"), (0x80, "EXIT", "", "")]
    assert sass_gate.mma_loops(spin, {}) == [(0, 7)]
    nested = [(0x0, "MOV", "R1, R2", ""), (0x10, "HMMA.16816.F32", "R8, R4, R6, R8", ""),
              (0x20, "BRA", "0x10", "@P1"), (0x30, "BRA", "0x0", "@P2"), (0x40, "EXIT", "", "")]
    assert sass_gate.mma_loops(nested, {}) == [(1, 2)]
    #: a shuffle's divergence fallback, a cold stub past the exit that branches back into the step it left
    stub = [(0x0, "HMMA.16816.F32", "R8, R4, R6, R8", ""), (0x10, "BRA.DIV", "~URZ, 0x60", ""),
            (0x20, "SHFL.BFLY", "PT, R1, R2, 0x1, 0x1f", ""), (0x30, "BRA", "0x0", "@P1"), (0x40, "STG.E", "[R2.64], R1", ""),
            (0x50, "EXIT", "", ""), (0x60, "MOV", "R1, R2", ""), (0x70, "BRA", "0x20", "")]
    assert sass_gate.mma_loops(stub, {}) == [(0, 3)]
    #: a long one -- ptxas copies the rest of the step into the fallback -- is cold to its return branch, which lands
    #: before the step's second MMA (until 2026-11-02 a stub past 32 lines read as a loop from there to its end)
    step = [(0x0, "HMMA.16816.F32", "R8, R4, R6, R8", ""), (0x10, "BRA.DIV", "~URZ, 0x70", ""),
            (0x20, "SHFL.BFLY", "PT, R1, R2, 0x1, 0x1f", ""), (0x30, "HMMA.16816.F32", "R8, R4, R1, R8", ""),
            (0x40, "BRA", "0x0", "@P1"), (0x50, "STG.E", "[R2.64], R1", ""), (0x60, "EXIT", "", "")]
    long_stub = step + [(0x70 + 0x10 * i, "MOV", "R1, R2", "") for i in range(64)] + [(0x470, "BRA", "0x20", "")]
    assert sass_gate.mma_loops(long_stub, {}) == [(0, 4)]


def test_a_branch_inside_an_mma_step_is_counted_and_one_outside_it_is_not():
    """KERNEL_STANDARDS 27 on the lines: a loop (a backward branch) holding an HMMA counts its convergence barriers and
    forward conditional branches; an outer loop's branches around an inner MMA loop do not count, nor a loop with no
    MMA."""
    step = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
            (0x20, "BSSY", "0x50", ""), (0x30, "BRA", "0x50", "@!P0"), (0x40, "ATOMS.ADD", "RZ, [R3], R8", ""),
            (0x50, "BSYNC", "", ""), (0x60, "BRA", "0x0", "@P1"), (0x70, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(step, {}) == 2
    clean = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
             (0x20, "ATOMS.ADD", "RZ, [R3], R8", ""), (0x30, "BRA", "0x0", "@P1"), (0x40, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(clean, {}) == 0
    nested = [(0x0, "BRA", "0x40", "@P2"), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
              (0x20, "BRA", "0x10", "@P1"), (0x30, "BAR.SYNC", "0x0", ""), (0x40, "BRA", "0x0", "@P3"),
              (0x50, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(nested, {}) == 0
    no_mma = [(0x0, "BSSY", "0x30", ""), (0x10, "BRA", "0x30", "@!P0"), (0x20, "LDS", "R4, [R2]", ""),
              (0x30, "BRA", "0x0", "@P1"), (0x40, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(no_mma, {}) == 0


def test_a_loops_exit_laid_out_as_its_latch_is_the_steps_end():
    """The loop's exit test out of the loop, followed only by register copies and an unconditional back-edge (ptxas's
    layout where the exit is not the fall-through, the readout's dtype copies before their join) ends the step's block
    at its end: not counted. An exit with other work between it and the back-edge, a branch back into the loop, or a
    conditional back-edge still count."""
    latch = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
             (0x20, "STS", "[R3], R8", ""), (0x30, "BRA", "0x60", "@!P1"), (0x40, "BRA", "0x0", ""),
             (0x50, "EXIT", "", ""), (0x60, "EXIT", "", "")]
    assert sass_gate.mma_loops(latch, {}) == [(0, 4)]
    assert sass_gate.mma_loop_branches(latch, {}) == 0
    copied = latch[:4] + [(0x38, "IMAD.MOV.U32", "R4, RZ, RZ, R5", ""), (0x3c, "MOV", "R6, R7", "")] + latch[4:]
    assert sass_gate.mma_loop_branches(copied, {}) == 0
    worked = latch[:4] + [(0x38, "IADD3", "R4, R5, 0x1, RZ", "")] + latch[4:]
    assert sass_gate.mma_loop_branches(worked, {}) == 1
    guarded = latch[:4] + [(0x38, "IMAD.MOV.U32", "R4, RZ, RZ, R5", "@P2")] + latch[4:]
    assert sass_gate.mma_loop_branches(guarded, {}) == 1
    inside = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
              (0x20, "BRA", "0x30", "@!P1"), (0x30, "BRA", "0x0", "")]
    assert sass_gate.mma_loop_branches(inside, {}) == 1
    conditional = latch[:4] + [(0x40, "BRA", "0x0", "@P2")] + latch[5:]
    assert sass_gate.mma_loop_branches(conditional, {}) == 1


def test_an_instrument_declaration_may_name_its_architecture():
    """`sm_80 <pattern>` keeps a signature on that architecture's SASS alone, a bare pattern on every one
    (docs/internals/tools/sass_gate.md#instrument)."""
    name = "_ZN4rola5carry5calib12calib_kernelILi4ELi12ELi36EEEvNS1_11CalibParamsE"
    assert sass_gate._declares(r"calib_kernelILi\d+ELi12ELi", name, "sm_86")
    assert sass_gate._declares(r"sm_80 calib_kernelILi\d+ELi12ELi", name, "sm_80")
    assert not sass_gate._declares(r"sm_80 calib_kernelILi\d+ELi12ELi", name, "sm_86")
    assert not sass_gate._declares(r"calib_kernelILi\d+ELi13ELi", name, "sm_80")
