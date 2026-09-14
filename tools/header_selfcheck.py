"""EVERY HEADER COMPILES ALONE — the gate for a name a header uses but does not include.

**WHAT THIS CATCHES.** A header that references a name it never included compiles
anyway, silently, whenever some *other* file in the translation unit happened to
include that name's home first. Nothing in the build says so, the layering it claims
in its own file header is false, and the break surfaces later — when a new caller
includes the header first, or when the include that was carrying it is removed for an
unrelated reason. It is the standard failure mode of splitting one file into layers,
and this project met it the first time it did: an extracted layout header used a
constant from its caller's header and compiled only because that caller had
included the constant's home four lines earlier.

**HOW.** One generated translation unit per header, containing that header twice and
nothing else. Twice, because the second inclusion is what proves the include guard:
a header without `#pragma once` passes the single-inclusion form and fails here.
Compiled with the SAME flag list the extension is built with (`tools/build_flags.py`,
the one source), so a header that needs a define the build supplies is not reported as
broken and a header that needs one the build does NOT supply is.

    python tools/header_selfcheck.py            # every header under csrc/rola
    python tools/header_selfcheck.py --list     # what it would compile, no nvcc

ONE ARCH, and that is deliberate: this asks a question about NAMES, not about codegen,
so the per-arch expansion the ratification gate pays for buys nothing here. Neither
does `--ptxas-options=-v`: there are no entry points in these translation units.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "csrc" / "rola"

#: The generated instantiation shards are not headers and the vendored subtree is not
#: ours to hold to this standard.
SKIP_DIRS = ("third_party", "instantiations")


def headers() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.cuh")
                  if not set(p.parts) & set(SKIP_DIRS))


def compile_flags(arch: int) -> list[str]:
    import torch
    from torch.utils.cpp_extension import include_paths

    flags = ["-O0", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    flags += build_flags.torch_defines()
    flags += [f"-I{p}" for p in include_paths(device_type="cuda")]
    flags += [f"-I{SRC / 'src'}", f"-I{ROOT / 'csrc' / 'third_party' / 'cutlass' / 'include'}"]
    import sysconfig
    flags.append(f"-I{sysconfig.get_paths()['include']}")
    del torch
    return flags


def check(header: Path, flags: list[str], tmp: Path) -> tuple[Path, str]:
    """Compile a TU that is this header, twice, and nothing else."""
    tu = tmp / (header.stem + "_selfcheck.cu")
    #: The absolute path, so the result does not depend on which `-I` resolves it.
    tu.write_text(f'#include "{header}"\n#include "{header}"\n')
    proc = subprocess.run(
        ["nvcc", "-c", str(tu), "-o", str(tu.with_suffix(".o")), *flags],
        capture_output=True, text=True)
    return header, "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", type=int, default=86, help="one sm_XX; names, not codegen")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print the header list and stop")
    args = ap.parse_args()

    hs = headers()
    if args.list:
        for h in hs:
            print(h.relative_to(ROOT))
        return 0
    if not hs:
        print("FAIL  no headers found; the glob is broken", file=sys.stderr)
        return 1

    flags = compile_flags(args.arch)
    failures: list[tuple[Path, str]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for header, err in pool.map(lambda h: check(h, flags, tmp), hs):
                if err:
                    failures.append((header, err))
                    print(f"FAIL  {header.relative_to(ROOT)} does not compile alone")
                else:
                    print(f"ok    {header.relative_to(ROOT)}")

    if failures:
        print()
        for header, err in failures:
            print(f"--- {header.relative_to(ROOT)}")
            print(err[:2000])
        print(f"\nFAIL  {len(failures)} of {len(hs)} header(s) depend on an include "
              f"they do not make themselves")
        return 1
    print(f"\nPASS  all {len(hs)} headers compile standalone (and twice over)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
