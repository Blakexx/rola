# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BOX LAB'S INPUTS for `digit_box32.cu` (32 lists x 32 digits; docs/internals/tools/lab.md#digit-box):
nl64k-alt-k4's read lists in two cells -- `alt/` as drawn, `thin/` each entry kept with probability 1/16 (a box of
the same state whose lists average under one write a window, as at ~1M leaves for the same fan-out) -- each in
`read_image.inputs.py`'s files for the box's readout, and in `fold/` for its fold (the 32-digit box's lists, 64-byte
write rows)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: the thinned cell's chance of keeping an entry, and its seed
KEEP, SEED = 1 / 16, 16


def write(directory: Path) -> list[str]:
    import torch

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, _ = base.lists(base.TOKENS // 512)
    files = []
    for cell, keep in (("alt", 1.0), ("thin", KEEP)):
        (directory / cell / "fold").mkdir(parents=True, exist_ok=True)
        fs, fa, fc = base.select(slots, at, 32, keep, SEED)
        files += [f"{cell}/fold/{f}" for f in base.emit(directory / cell / "fold", fs, fa, fc, 256, 16, 16)]
        rs, ra, rc = base.select(slots, at, 32, keep, SEED + 1)
        out = base.emit(directory / cell, rs, ra, rc, 256, 16, 16)
        files += [f"{cell}/{f}" for f in out]
    torch.cuda.empty_cache()
    return files
