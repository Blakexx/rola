#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRAPH'S FRONT DOOR, this workspace's stub: bazelisk runs `bazel/bazel` -- this file, the wrapper `.bazeliskrc`
names -- in place of the Bazel it downloaded, and it runs rola-devtools' front door (`rola_devtools.graph.front_door`)
for this checkout, from the rola-devtools checkout `tools/bootstrap.py` finds: the machine's caches, every build under
the host budget, and that checkout as the override of MODULE.bazel's rola-devtools pin.

Runs under the machine's `python3` before any venv exists. Docs: docs/build-graph.md#front-door.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bootstrap  # noqa: E402 -- the path insert above must precede it

sys.path.insert(1, str(bootstrap.devtools()))
from rola_devtools.graph import front_door  # noqa: E402

if __name__ == "__main__":
    sys.exit(front_door.main(sys.argv[1:], ROOT))
