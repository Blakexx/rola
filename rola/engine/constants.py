# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carry family's file constants, launch table and S-from-SMEM rule: the Python
mirror of ``csrc/rola/src/common/constants.cuh``.

The two modules declare the SAME numbers -- ``WINDOW``, ``SEGMENT_TOKENS``, the
``warps_per_cta`` set, the per-arch launch table and the S-from-SMEM rule -- because a
caller building a launch on the host and a kernel reading its own prologue must agree
on all of them without either reading the other's source. ``tests/unit/
test_carry_constants.py`` is the agreement test: it parses the C++ header's literals
and asserts they equal these.

Everything here is either a literal or arithmetic on literals and ``leaves_per_warp``
(the register shape a value width selects, mirrored on both sides so `DV` picks the
same answer everywhere) -- never runtime addressing (level widths, row maps), which
the descriptor and ``rola.ops.carry`` own.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LAUNCH_TABLE",
    "SEGMENT_TOKENS",
    "WARPS_PER_CTA",
    "WARPS_PER_CTA_DIAGNOSTIC",
    "WARPS_PER_CTA_SHIPPED",
    "WARPS_PER_SM",
    "WINDOW",
    "LaunchRow",
    "box_leaves",
    "derive_stream_count",
    "is_warps_per_cta",
    "launch_row",
    "leaves_per_warp",
]

#: The kernel's window, in tokens: a short sequence runs a partial window inside the
#: kernel rather than taking a different value here. The inter/intra decomposition is
#: defined on this grid, so the fp64 reference mirrors the same number.
WINDOW = 512

#: The stream's buffering grain, in tokens: the slice a producer fills, seals and hands
#: to the consumers, and the k step one deposit MMA contracts. Mirrors
#: ``common/constants.cuh``'s ``kSegmentTokens``.
SEGMENT_TOKENS = 16

#: The register file's warp count per multiprocessor: eight warps fill it exactly at
#: 256 registers/lane. Mirrors ``common/geom.cuh``'s ``warps_per_sm``.
WARPS_PER_SM = 8

#: ``warps_per_cta``'s two-value set: 8 is the shipped shape (one CTA per SM, the
#: register file filled exactly); 4 is the two-CTA latency diagnostic and ships
#: nowhere.
WARPS_PER_CTA_SHIPPED = 8
WARPS_PER_CTA_DIAGNOSTIC = 4
WARPS_PER_CTA = (WARPS_PER_CTA_SHIPPED, WARPS_PER_CTA_DIAGNOSTIC)


def is_warps_per_cta(warps_per_cta: int) -> bool:
    """Whether ``warps_per_cta`` is a launch shape this family builds."""
    return warps_per_cta in WARPS_PER_CTA


#: The register budget ``leaves_per_warp`` is derived from (``common/geom.cuh``'s
#: ``leaves_per_sm(dv) = state_elems_per_sm / dv``): the same DERIVATION, mirrored so the
#: two sides' arithmetic is textually comparable, not merely numerically equal by
#: coincidence. The numerator is the state's declared share of one SM's register file --
#: ``regs_per_sm * STATE_FILE_NUMERATOR / STATE_FILE_DENOMINATOR`` -- and 65536 registers
#: is the value every tabulated architecture's capability row carries.
STATE_FILE_NUMERATOR, STATE_FILE_DENOMINATOR = 1, 4
_REGS_PER_SM = 65536
_LEAVES_PER_SM_NUMERATOR = _REGS_PER_SM * STATE_FILE_NUMERATOR // STATE_FILE_DENOMINATOR


def leaves_per_warp(dv: int) -> int:
    """The warp sub-box's leaf count at this value width -- register shape, derived.

    32 at ``DV <= 64`` (two m-tiles, the ldmatrix granule), 16 at ``DV = 128`` (leaves
    per warp halves). Uniform across sm_80/86/90 today because the register budget the
    numerator encodes does not vary by architecture; a part that changes it gets its
    own numerator here, not a branch.
    """
    return (_LEAVES_PER_SM_NUMERATOR // int(dv)) // WARPS_PER_SM


def box_leaves(dv: int, warps_per_cta: int) -> int:
    """``BC``: the CTA's box, ``leaves_per_warp * warps_per_cta``. Never an axis."""
    return leaves_per_warp(dv) * int(warps_per_cta)


@dataclass(frozen=True, slots=True)
class LaunchRow:
    """One row of the per-arch launch table: ``(cc, ctas_per_sm, warps_per_cta)``."""

    cc: int
    ctas_per_sm: int
    warps_per_cta: int


#: The per-arch launch table, as data the seam reads: sm_86 runs two CTAs of four
#: warps; every other tabulated architecture runs one CTA of eight warps.
LAUNCH_TABLE = (
    LaunchRow(800, 1, WARPS_PER_CTA_SHIPPED),
    LaunchRow(860, 1, WARPS_PER_CTA_SHIPPED),
    LaunchRow(870, 1, WARPS_PER_CTA_SHIPPED),
    LaunchRow(890, 1, WARPS_PER_CTA_SHIPPED),
    LaunchRow(900, 1, WARPS_PER_CTA_SHIPPED),
)
_LAUNCH_TABLE_BY_CC = {row.cc: row for row in LAUNCH_TABLE}


def launch_row(cc: int) -> LaunchRow:
    """The launch shape for compute capability ``cc`` x10 (e.g. 860), or a refusal.

    Mirrors the C++ ``launch_row``: an untabulated ``cc`` is never guessed at.
    """
    try:
        return _LAUNCH_TABLE_BY_CC[int(cc)]
    except KeyError:
        raise ValueError(
            f"cc={cc} has no launch-table row; the tabulated set is "
            f"{sorted(_LAUNCH_TABLE_BY_CC)} (docs/internals/common/constants.md"
            f"#launch-table)."
        ) from None


def derive_stream_count(per_stream_bytes: int, residency_smem_bytes: int,
                        warps_per_cta: int) -> int:
    """The largest stream count a stated SMEM layout admits.

    The largest power of two at or below ``warps_per_cta`` whose streams' segments
    fit in ``residency_smem_bytes`` at ``per_stream_bytes`` each, or ``0`` if even one
    stream's segment does not fit -- a refusal, never a silent fallback to ``S = 0``
    meaning "run anyway".
    """
    s = warps_per_cta
    while s >= 1:
        if s * per_stream_bytes <= residency_smem_bytes:
            return s
        s >>= 1
    return 0
