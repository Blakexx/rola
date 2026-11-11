#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BUILD'S GENERATED HEADERS for one build's parameters: the carry and decode arm selections and the
builds the binary carries -- `tools/gen_shards.py`'s and `measure/hardware.py`'s. Run by the graph's
header action (`bazel/defs.bzl`'s `rola_headers`).
Docs: docs/build-graph.md#compile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import gen_shards  # noqa: E402

from measure import hardware  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--archs", required=True, help="the build's architectures, comma-separated (`80,86`)")
    ap.add_argument("--carry-arms", default="", help="carry arm indices, comma-separated; empty is every arm")
    ap.add_argument("--decode-arms", default="", help="decode arm indices, comma-separated; empty is every arm")
    a = ap.parse_args()
    carry = gen_shards.family_arms(gen_shards.CARRY, a.carry_arms, "--//:carry_arms")
    decode = gen_shards.family_arms(gen_shards.DECODE, a.decode_arms, "--//:decode_arms")
    texts = {gen_shards.CARRY_SELECTION_INC: gen_shards.carry_selection_header(carry),
             gen_shards.DECODE.header: gen_shards.decode_selection_header(decode),
             gen_shards.BUILDS_INC: hardware.builds_header([x for x in a.archs.split(",") if x])}
    for name, text in texts.items():
        (a.out_dir / name).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
