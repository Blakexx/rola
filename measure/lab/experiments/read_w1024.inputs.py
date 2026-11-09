# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE W = 1024 READOUT LAB'S INPUTS: nl64k-alt-k4's read lists (one head, 32 windows of 512 tokens) merged in pairs
into 16 windows of 1024 tokens, for `read_w1024.cu` (docs/internals/tools/lab.md#read-w1024): a box's union ranks of the
second window follow the first's (a box's union a window holds at most 512 tokens, asserted), a list's entries are the
two windows' concatenated. The files are `read_image.inputs.py`'s in its formats, for a 512-slot pass over 4 image
planes with a digit's first 32 entries staged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(directory: Path) -> list[str]:
    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count = base.lists(base.TOKENS // 512)
    windows, width, cap = slots.shape
    groups = count.shape[-1]
    merged = np.zeros((windows // 2, width, cap), np.int64)
    at2 = np.zeros((windows // 2, width, 8), np.int64)
    count2 = count[0::2] + count[1::2]
    for v in range(windows // 2):
        for d in range(width):
            first = count[2 * v, d * groups // width]
            a = slots[2 * v, d, at[2 * v, d, 0]:at[2 * v, d, -1]]
            b = slots[2 * v + 1, d, at[2 * v + 1, d, 0]:at[2 * v + 1, d, -1]] + first
            merged[v, d, :len(a) + len(b)] = np.concatenate([a, b])
            at2[v, d, 1:] = len(a) + len(b)
    out = base.emit(directory, merged, at2, count2, 512, 4, 32)
    torch.cuda.empty_cache()
    return out
