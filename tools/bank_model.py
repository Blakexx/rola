#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BANK MODEL: the shared-memory wavefronts a warp instruction costs, from the byte addresses its lanes touch
(KERNEL_STANDARDS section 30). Shared memory is 32 banks of 4 bytes and serves 128 bytes a wavefront; an instruction
costs, for each 128-byte phase of its lanes, the most distinct bank-rows any one bank holds (lanes reading the same
word share it). A design states its accesses in these terms before code, and a producer that picks addresses (the
facts pass's slots) runs this on its output so a conflict is red before hardware. The model is checked against ncu's
wavefront count once a layout (docs/internals/tools/bank_model.md).

    wavefronts(addresses, width)      a 4-, 8- or 16-byte load, store or atomic: one address a lane
    ldmatrix(rows)                    an `ldmatrix`: each 8x8 matrix's 8 row addresses (16 bytes each)
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

#: banks, the bytes a bank serves a row, the bytes a wavefront serves
BANKS, WORD, WAVEFRONT = 32, 4, 128


def _phase(addresses: Iterable[int], width: int) -> int:
    """One phase's wavefronts: the most distinct bank-rows in one bank, over the words each access covers."""
    rows: dict[int, set[int]] = defaultdict(set)
    for a in addresses:
        for w in range(a // WORD, (a + width) // WORD):
            rows[w % BANKS].add(w // BANKS)
    return max((len(r) for r in rows.values()), default=0)


def wavefronts(addresses: Sequence[int | None], width: int) -> int:
    """A warp's `width`-byte accesses, one byte address a lane (None for an inactive lane): served in phases of
    `128 / width` lanes (all 32 for 4 bytes, 16 for 8, 8 for 16), each phase its conflict degree."""
    if width not in (4, 8, 16):
        raise ValueError(f"an access is 4, 8 or 16 bytes, not {width}")
    if any(a is not None and a % width for a in addresses):
        raise ValueError("an access must be aligned to its width")
    per = WAVEFRONT // width
    return sum(_phase([a for a in addresses[p:p + per] if a is not None], width)
               for p in range(0, len(addresses), per))


def ldmatrix(matrices: Sequence[Sequence[int]]) -> int:
    """An `ldmatrix` of len(matrices) 8x8 matrices, each given by its 8 row addresses (16 bytes a row)."""
    if any(len(m) != 8 for m in matrices):
        raise ValueError("an ldmatrix matrix has 8 rows")
    return sum(_phase(m, 16) for m in matrices)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("width", type=int, choices=(4, 8, 16))
    p.add_argument("address", type=lambda s: int(s, 0), nargs="+", help="a lane's byte address, in lane order")
    args = p.parse_args()
    print(wavefronts(args.address, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
