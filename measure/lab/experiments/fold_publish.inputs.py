# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HAND-OFF LAB'S INPUTS: `read_image.inputs.py`'s files for a 16-channel box (1-KB image planes, 48-byte panel
rows) with each box-window's digits also dealt balanced over the warps, for `fold_publish.cu`
(docs/internals/tools/lab.md#fold-publish)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(directory: Path) -> list[str]:
    import torch

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count = base.lists(base.TOKENS // 512)
    out = base.emit(directory, slots, at, count, 256, 32, 16, plane_bytes=1024, pitch=48, channels=16, balanced=True)
    torch.cuda.empty_cache()
    return out
