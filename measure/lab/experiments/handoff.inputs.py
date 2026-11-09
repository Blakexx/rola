# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HAND-OFF'S INPUTS: a box's fold state for `handoff.cu` (docs/lab/handoff.md) -- 32 lists x 32 digits x 32
channels, fp32 in each list's unit -- the lists' units, and the image and the digits' exponents the hand-off must form
from them, by `csrc/rola/src/carry/ampere/parts.cuh`'s rules: a digit's exponent the smallest `e` with every list's value at
its unit at most 2^e (`ceil_exp` of the list's largest |value| plus its unit, the largest over the lists, at least
`IMAGE_TOP - 127`), and each value stored f16 (round to nearest) times 2^(unit + IMAGE_TOP - e), the exponent held to
[-126, 127], at `unit_of`'s place.

The values are signed integers times powers of two that vary by list and digit (each digit's largest list a different
one, some digits zero in some lists, one digit zero in all), so every exponent and scale the rules form is exercised.
Files (little-endian):
- `state.bin`: f32 `[32 lists][32 digits][32 channels]`.
- `units.bin`: int32 `[32 lists]`.
- `image.bin`: the 64 KB image: 32 planes (a digit's), each 2 KB of 16-byte units, `unit_of(list, chunk, digit)`'s.
- `image16.bin`: the same values in the 16-byte image: chunk `t` of a list holds channels `8 n + 2 t + b` at position
  `2 n + b`, at `unit16_of(list, t, digit)`.
- `exps.bin`: int32 `[32 digits]`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LISTS = DIGITS = CHANNELS = 32
IMAGE_TOP = 10
UNIT_NONE = -1000
PLANE_BYTES = LISTS * CHANNELS * 2


def ceil_exp(x: float) -> int:
    """`parts.cuh`'s: the smallest `e` with `x <= 2^e` from the fp32 bits, `UNIT_NONE` for zero."""
    bits = int(np.float32(x).view(np.uint32))
    return UNIT_NONE if bits == 0 else (bits >> 23) - 127 + ((bits & 0x7FFFFF) != 0)


def unit_of(lst: int, chunk: int, digit: int) -> int:
    """`parts.cuh`'s 16-byte unit of digit `digit`'s plane for list `lst`, chunk `chunk` (8 channels)."""
    return ((4 * lst + (chunk ^ ((lst >> 1) & 3))) ^ (digit & 7)) & (PLANE_BYTES // 16 - 1)


def unit16_of(lst: int, chunk: int, digit: int) -> int:
    """`handoff.cu`'s 16-byte image unit: the digit's low bit moved to bit 2 of its swizzle."""
    swizzle = ((digit & 1) << 2) | ((digit >> 1) & 3)
    return ((4 * lst + (chunk ^ ((lst >> 1) & 3))) ^ swizzle) & (PLANE_BYTES // 16 - 1)


def write(directory: Path) -> list[str]:
    rng = np.random.default_rng(20261018)
    units = rng.integers(-40, 41, size=LISTS).astype(np.int32)
    units[5] = UNIT_NONE  # a list never live: its state zero
    shift = rng.integers(-30, 31, size=(LISTS, DIGITS))
    state = rng.integers(-2047, 2048, size=(LISTS, DIGITS, CHANNELS)).astype(np.float64)
    state *= np.exp2(shift)[..., None]
    state[5] = 0.0
    state[rng.random((LISTS, DIGITS)) < 0.1] = 0.0  # a digit zero in a list
    state[:, 17] = 0.0  # a digit zero in every list
    state = state.astype(np.float32)

    exps = np.zeros(DIGITS, np.int32)
    for d in range(DIGITS):
        e = IMAGE_TOP - 127
        for lst in range(LISTS):
            e = max(e, ceil_exp(float(np.abs(state[lst, d]).max())) + int(units[lst]))
        exps[d] = e
    image = np.zeros((DIGITS, PLANE_BYTES // 2), np.float16)
    image16 = np.zeros((DIGITS, PLANE_BYTES // 2), np.float16)
    for d in range(DIGITS):
        for lst in range(LISTS):
            k = min(max(int(units[lst]) + IMAGE_TOP - int(exps[d]), -126), 127)
            vals = (state[lst, d].astype(np.float32) * np.float32(2.0 ** k)).astype(np.float16)
            for chunk in range(4):
                at = 8 * unit_of(lst, chunk, d)
                image[d, at:at + 8] = vals[8 * chunk:8 * chunk + 8]
                at16 = 8 * unit16_of(lst, chunk, d)
                for n in range(4):
                    image16[d, at16 + 2 * n:at16 + 2 * n + 2] = vals[8 * n + 2 * chunk:8 * n + 2 * chunk + 2]
    (directory / "state.bin").write_bytes(state.tobytes())
    (directory / "units.bin").write_bytes(units.tobytes())
    (directory / "image.bin").write_bytes(image.tobytes())
    (directory / "image16.bin").write_bytes(image16.tobytes())
    (directory / "exps.bin").write_bytes(exps.tobytes())
    print(f"handoff: exponents {exps.min()} .. {exps.max()}, units {units[units > UNIT_NONE].min()} .. {units.max()}")
    return ["state.bin", "units.bin", "image.bin", "image16.bin", "exps.bin"]


if __name__ == "__main__":
    write(Path(sys.argv[1]))
