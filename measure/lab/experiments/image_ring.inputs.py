# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE IMAGE RING LAB'S INPUTS: `read_image.inputs.py`'s files, the same lists in the same form, for `image_ring.cu`
(docs/internals/tools/lab.md#image-ring)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(directory: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base.write(directory)
