#!/usr/bin/env python3
"""SASS-SHAPE CENSUS for arch-selected primitives.

**WHAT THIS CATCHES.** `csrc/rola/src/common/ops.cuh` names a primitive by its
CONTRACT (KERNEL_STANDARDS section 8: "feature selection lives where the
primitive is defined ... never as `#if` at call sites") and picks a NATIVE
instruction per architecture behind an `arch::Caps` flag. The failure mode this
gate exists to catch is the `mul_bf16x2`-before-K35 shape: a primitive quietly
falls back to a WIDER-TYPE EMULATION (unpack to fp32, compute, repack) on some
built arch instead of the one native instruction the ISA actually offers
there -- and nothing before this gate notices, because the emulation is
functionally correct and only costs SASS shape. `mul_bf16x2` passed the fp64
oracle for months carrying exactly this defect; it was found by the K4
instruction ledger, not by an audit. This file is the audit, made permanent:
an emulation FAILS THE DAY IT IS WRITTEN rather than the day someone happens to
ledger that kernel again.

**HOW.** For each `(primitive, arch)` cell below, compile a MINIMAL translation
unit that `#include`s `ops.cuh` and calls the primitive from one `__global__`
kernel, disassemble the resulting `cubin` with `cuobjdump -sass`, and count
opcodes by FAMILY (the mnemonic up to its first `.` modifier -- `HFMA2.BF16_V2`
and `HFMA2.MMA` are both family `HFMA2`). Each cell asserts an exact count for
one or more families; the K42 audit table in
`prototypes/k35_final_build/FINDINGS.md`'s `## K42` entry is this file's
derivation record -- read it for WHY each cell's expected shape is what it is,
this file only asserts the shape.

**BUILT ARCHES ONLY.** `denseref::arch::caps_of`'s tabulated rows include
sm_90, but sm_90 is TABULATED AND NOT BUILT (no `mma`/`load_frag` for a
128-thread warpgroup unit) -- compiling a probe for it would exercise ops.cuh
in a configuration this tree never ships, so it is a documented SKIP, not a
fabricated compile. The built set is read from `tools/build_flags.py`'s own
arch list so this gate and the real build can never disagree about which
arches "built" means.

**WIRING.** Run this after `tools/ratify.py`'s own census (`--census`), same
convention: a compile-only gate, no GPU, no `flock`. `docs/build.md`'s
"Standards lint" section is where the pre-commit-wired checks are listed;
add this file's invocation there alongside `lint_standards.py` and
`run_clang_tidy.sh` when it graduates from stage gate to a wired check.

Usage (no GPU required -- compile + disassemble only):

    python tools/lint/primitives_census.py
    python tools/lint/primitives_census.py -v      # print every cell, not just failures
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                    check=True).stdout.strip()
)
sys.path.insert(0, str(REPO_ROOT / "tools"))
from ratify import cuda_bin  # noqa: E402

OPS_INCLUDE_DIR = REPO_ROOT / "csrc" / "rola" / "src"

#: BUILT ARCHES -- the non-900 rows `arch_caps.cuh`'s `caps_of()` tabulates
#: (800, 860, 870, 890; see that file's `tabulated()`). This is a COMPILE-only
#: gate (no GPU, no ratified manifest needed): `nvcc`/`ptxas` accept any of
#: these as a `-arch=sm_XX` target on this toolchain regardless of which ones
#: `tools/ratify.py`'s `DEFAULT_ARCHES` / `tools/manifests/<toolchain>/*.json` currently
#: ratify for THIS host's default build (80, 86 today) -- the census's job is
#: auditing the CAPABILITY TABLE's claims, not mirroring one host's build
#: config, so it compiles every tabulated-and-built row.
BUILT_ARCHES: list[str] = ["80", "86", "87", "89"]

#: sm_90 is TABULATED (arch_caps.cuh's `caps_of`) but NOT BUILT (no 128-thread
#: warpgroup `mma`/`load_frag` -- ops.cuh's own consistency `static_assert`
#: refuses it at `kMmaUnitThreads`). Documented skip, never silently dropped.
TABULATED_NOT_BUILT = ["90"]

#: One probe kernel per audited primitive. Each calls the REAL `ops.cuh`
#: entry point -- never a hand-rolled restatement of what it should compile
#: to -- so a regression in the shipped primitive is exactly what this trips.
PROBE_SOURCE = """
#include <cuda_bf16.h>
#include "common/ops.cuh"

extern "C" __global__ void census_mul_bf16x2(uint32_t* out, uint32_t a, uint32_t b) {
  out[0] = denseref::ops::mul_bf16x2(a, b);
}

extern "C" __global__ void census_pack_bf16x2(uint32_t* out, float a, float b) {
  out[0] = denseref::ops::pack_bf16x2(a, b);
}

extern "C" __global__ void census_hi_bf16x2(uint32_t* out, float a, float b) {
  out[0] = denseref::ops::hi_bf16x2(a, b);
}

extern "C" __global__ void census_transpose_frag_b16(uint32_t* out, uint32_t x) {
  out[0] = denseref::ops::transpose_frag_b16(x);
}
"""

#: THE CENSUS TABLE.  Each row: (primitive, function symbol, family -> exact
#: count).  A family absent from a row's dict is NOT asserted (e.g. `mul_bf16x2`
#: does not assert `PRMT`, because `hi_bf16x2`'s presence there is a different
#: primitive's contract) -- only families that WOULD appear under emulation, or
#: that the native route is defined by, are asserted, per the card's own
#: worked example ("exactly 1 HFMA2 and 0 FMUL").
CENSUS: dict[str, dict] = {
    "mul_bf16x2": {
        "symbol": "census_mul_bf16x2",
        #: FMA route (sm_80/86/87/89, K35 R-C): one `fma.rn.bf16x2` with a
        #: `-0.0` addend lowers to one HFMA2; the WIDER-TYPE EMULATION this
        #: gate exists to catch would show FMUL (fp32 multiply) plus PRMT/pack
        #: traffic instead.
        "families": {"HFMA2": 1, "FMUL": 0, "F2FP": 0},
    },
    "pack_bf16x2": {
        "symbol": "census_pack_bf16x2",
        #: `__floats2bfloat162_rn`: one packed fp32->bf16x2 convert (F2FP
        #: family on every tabulated arch -- the operand order differs
        #: between sm_80's PACK_AB and sm_90's F32.PACK_AB, both same family).
        "families": {"F2FP": 1, "FMUL": 0, "FADD": 0},
    },
    "hi_bf16x2": {
        "symbol": "census_hi_bf16x2",
        #: byte-select PRMT, unconditioned by any Caps flag -- available on
        #: every built arch. An emulation route here would show FMUL/F2FP
        #: traffic where one PRMT suffices.
        "families": {"PRMT": 1, "FMUL": 0, "F2FP": 0},
    },
    "transpose_frag_b16": {
        "symbol": "census_transpose_frag_b16",
        #: `movmatrix.sync.aligned.m8n8.trans.b16`, native since sm_75 -- every
        #: tabulated arch here is sm_80+. The wider-type-emulation shape this
        #: would take instead is a shared-memory round trip (STS + barrier +
        #: LDS.TRANS or `ldmatrix.trans`), so STS/LDS presence is the tell.
        "families": {"MOVM": 1, "STS": 0, "LDS": 0},
    },
}

_OPCODE_RE = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_]*)")


def opcode_families(sass_text: str) -> dict[str, int]:
    """Count SASS opcodes by FAMILY: the mnemonic up to (excluding) its first
    `.` modifier -- the capturing group already stops there (`[A-Z][A-Z0-9_]*`
    contains no `.`), so `HFMA2.BF16_V2` and `HFMA2.MMA` both count as `HFMA2`.
    """
    counts: dict[str, int] = {}
    for line in sass_text.splitlines():
        m = _OPCODE_RE.search(line)
        if not m:
            continue
        family = m.group(1)
        if family in ("NOP", "BRA", "EXIT"):
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def compile_and_disassemble(arch: str, tmpdir: Path) -> dict[str, int]:
    src = tmpdir / f"probe_sm{arch}.cu"
    cubin = tmpdir / f"probe_sm{arch}.cubin"
    src.write_text(PROBE_SOURCE)
    nvcc = cuda_bin("nvcc")
    cmd = [nvcc, f"-arch=sm_{arch}", "-cubin", "-I", str(OPS_INCLUDE_DIR), "-o", str(cubin),
           str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"primitives_census: probe FAILED to compile for sm_{arch}\n"
                          f"  command: {' '.join(cmd)}\n{proc.stderr}")
    cuobjdump = cuda_bin("cuobjdump")
    sass = subprocess.run([cuobjdump, "-sass", str(cubin)], capture_output=True, text=True,
                          check=True).stdout
    #: split by function so a count never leaks across probe kernels sharing one cubin.
    per_function: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if current is not None:
                per_function[current] = "\n".join(buf)
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        per_function[current] = "\n".join(buf)
    return {name: opcode_families(text) for name, text in per_function.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every cell's measured opcode families, not just failures")
    args = ap.parse_args()

    ok = True
    with tempfile.TemporaryDirectory(prefix="rola_primitives_census_") as td:
        tmpdir = Path(td)
        for arch in BUILT_ARCHES:
            print(f"=== sm_{arch} ===")
            by_function = compile_and_disassemble(arch, tmpdir)
            for prim, row in CENSUS.items():
                symbol = row["symbol"]
                measured = by_function.get(symbol)
                if measured is None:
                    ok = False
                    print(f"  FAIL  {prim}: symbol {symbol} not found in disassembly "
                          f"(compiled but produced no code? check the probe TU)")
                    continue
                mismatches = []
                for family, expected in row["families"].items():
                    got = measured.get(family, 0)
                    if got != expected:
                        mismatches.append((family, expected, got))
                if mismatches:
                    ok = False
                    print(f"  FAIL  {prim} (sm_{arch}): SASS shape moved")
                    for family, expected, got in mismatches:
                        print(f"          {family}: expected {expected}, measured {got}")
                    print(f"          full measured opcode counts: {measured}")
                elif args.verbose:
                    wanted = {k: measured.get(k, 0) for k in row["families"]}
                    print(f"  pass  {prim} (sm_{arch}): {wanted}")
        for arch in TABULATED_NOT_BUILT:
            print(f"=== sm_{arch}: tabulated, NOT built -- documented skip, no compile issued ===")

    print(f"\n{'PASS' if ok else 'FAIL'}  primitives census "
          f"({len(CENSUS)} primitive(s) x {len(BUILT_ARCHES)} built arch(es))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
