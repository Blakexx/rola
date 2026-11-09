"""THE LEAF GEOMETRY'S REGISTER SHAPE, in two places that must agree: ``csrc/rola/src/common/geom.cuh``'s constants and
their Python mirror, ``rola.engine.constants`` -- the header's literals parsed out of the source text and checked equal,
and the derived counts checked against the numbers they are stated as.

No device, no build: this reads source text and calls pure Python.
"""
from __future__ import annotations

import re
from pathlib import Path

from rola.engine import constants as py

ROOT = Path(__file__).resolve().parents[2]


def test_the_leaves_per_sm_numerator_agrees():
    """``leaves_per_sm(dv) = 16384 / dv`` lives in ``common/geom.cuh`` (not this
    header); the Python mirror restates the same literal rather than re-deriving it,
    so this pins that restatement against the numbers K50 item 1 states directly.
    """
    assert py._LEAVES_PER_SM_NUMERATOR == 16384
    assert py.leaves_per_warp(64) == 32
    assert py.leaves_per_warp(128) == 16


# --------------------------------------------------------------------------- the tall owner set
GEOM_HEADER = ROOT / "csrc" / "rola" / "src" / "common" / "geom.cuh"


def _geom_constexpr_int(name: str) -> int:
    m = re.search(rf"constexpr int {re.escape(name)}\s*=\s*(-?\d+)\s*;", GEOM_HEADER.read_text())
    assert m, f"no `constexpr int {name} = ...;` found in {GEOM_HEADER}"
    return int(m.group(1))


def test_the_tall_owner_sets_constants_agree():
    assert _geom_constexpr_int("kCtaChannels") == py.CTA_CHANNELS
    assert _geom_constexpr_int("kTallStateFileNumerator") == py.TALL_STATE_FILE_NUMERATOR
    assert _geom_constexpr_int("kTallStateFileDenominator") == py.TALL_STATE_FILE_DENOMINATOR


def test_the_tall_owner_set_is_the_half_share_over_the_channel_group():
    """`tall_leaves_per_warp = regs_per_sm * 1/2 / kCtaChannels / warps_per_sm`, mirrored."""
    regs = 65536
    want = regs * py.TALL_STATE_FILE_NUMERATOR // py.TALL_STATE_FILE_DENOMINATOR // py.CTA_CHANNELS // py.WARPS_PER_SM
    assert py.tall_leaves_per_warp() == want == 128
    assert py.TALL_BOX_LEAVES == 1024
