#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ARM GATE FOR THE GRAPH'S LIBRARY: the built arm table against the declared shipped set, and the coverage sweep
(`tools/gen_shards.py`'s `check_built_arm_table` and `sweep_binary`) -- the binary asked
what it carries, loaded by the environment's interpreter against the toolkit's driver stub, so the answer is the
binary's on any machine, GPU or none. Run by `bazel/defs.bzl`'s `rola_extension`; writes its verdict to `--out`.
Docs: docs/build-graph.md#gates.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_shards  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", type=Path, required=True, help="the built extension")
    ap.add_argument("--header", type=Path, required=True, help="the carry selection header the build compiled against")
    ap.add_argument("--python", required=True, help="the environment's interpreter, torch beside it")
    ap.add_argument("--driver-stub", type=Path, required=True, help="the toolkit's libcuda.so stub")
    ap.add_argument("--carry-arms", default="", help="the build's carry arm spec")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    want = gen_shards.family_arms(gen_shards.CARRY, a.carry_arms, "--//:carry_arms")
    bad = gen_shards.check_built_arm_table(a.header.read_text(), want)
    if bad:
        raise SystemExit("the built arm table:\n" + "\n".join(bad))
    with tempfile.TemporaryDirectory(prefix="rola_arm_gate_") as tmp:
        #: the stub under its soname, the one the binary's NEEDED names
        (Path(tmp) / "libcuda.so.1").symlink_to(a.driver_stub.absolute())
        swept = gen_shards.sweep_binary(str(Path(a.python).absolute()), a.so.absolute(), want, cwd=Path(tmp),
                                        env={"LD_LIBRARY_PATH": tmp, "PATH": "/usr/bin:/bin"})
    a.out.write_text(f"arm table OK  {len(want)} built of {len(gen_shards.CARRY_ARMS)} declared; {swept} arm(s) swept\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
