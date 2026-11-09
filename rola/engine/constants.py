# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The leaf geometry's register shape, the Python mirror of ``csrc/rola/src/common/geom.cuh``'s: the leaves a warp
holds at a value width and the tall owner set's constants. ``tests/unit/test_carry_constants.py`` is the agreement
test: it parses the C++ header's literals and asserts they equal these.
"""
from __future__ import annotations

__all__ = [
    "BLOCK_LEAVES",
    "CTA_CHANNELS",
    "TALL_BOX_LEAVES",
    "TALL_DENSE_SPAN_MAX",
    "TALL_STATE_FILE_DENOMINATOR",
    "TALL_STATE_FILE_NUMERATOR",
    "WARPS_PER_SM",
    "channel_groups",
    "leaves_per_warp",
    "tall_leaves_per_warp",
]

#: The register file's warp count per multiprocessor: eight warps fill it exactly at
#: 256 registers/lane. Mirrors ``common/geom.cuh``'s ``warps_per_sm``.
WARPS_PER_SM = 8

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


#: THE TALL OWNER SET: the state takes HALF the register file and a CTA holds
#: ``CTA_CHANNELS`` of ``DV`` (one channel group), so an owner set is
#: ``65536 / 2 / 32 = 1024`` leaves and 128 a warp. Mirrors ``common/geom.cuh``'s
#: ``kCtaChannels`` / ``kTallStateFile*`` / ``tall_leaves_per_warp`` (geom.md#tall-owner-set).
CTA_CHANNELS = 32
TALL_STATE_FILE_NUMERATOR, TALL_STATE_FILE_DENOMINATOR = 1, 2


def tall_leaves_per_warp() -> int:
    """128: ``regs_per_sm * 1/2 / CTA_CHANNELS / WARPS_PER_SM``, the same derivation as C++."""
    return (_REGS_PER_SM * TALL_STATE_FILE_NUMERATOR // TALL_STATE_FILE_DENOMINATOR
            // CTA_CHANNELS) // WARPS_PER_SM


TALL_BOX_LEAVES = tall_leaves_per_warp() * WARPS_PER_SM

#: THE TALL OWNER LAW'S SHAPE CONSTANTS (ruled 2026-09-27), mirroring ``kTallDenseSpanBitsMax`` /
#: ``kBlockLeavesBits``: a read-dense level's span is at most 128 digits (the drain / fold balance) and
#: the IMMA's K block is 32 leaves of one read-sparse digit (geom.md#tall-owner-set).
TALL_DENSE_SPAN_MAX = 1 << 7
BLOCK_LEAVES = 1 << 5


def channel_groups(dv: int) -> int:
    """The CTAs one leaf set takes: one per ``CTA_CHANNELS`` of ``DV``."""
    return int(dv) // CTA_CHANNELS
