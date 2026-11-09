# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CODEGEN-EQUIVALENCE INSTRUMENT'S RENAME RULE (`tools/sass_bodies.py`, docs/internals/common/hardware.md#builds-renamed):
a kernel's own-image `Build` argument is dropped from its signature, another image's is refused, and the source stamp
is reported, never a verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import sass_bodies  # noqa: E402

#: `rola::k<Arm<3>>(int)` and `rola::k<BuildTag<Build{860, 0}>, Arm<3>>(int)`, as nvcc 13.0 mangles them
OLD = "sm_86:_ZN4rola1kINS_3ArmILi3EEEEEvi"
NEW = "sm_86:_ZN4rola1kIN8denseref4arch8BuildTagIXtlNS2_5BuildELi860ELi0EEEEENS_3ArmILi3EEEEEvi"
STAMP = "sm_86:_ZN4rola8stampdet17csrc_stamp_kernelEPl"


def test_an_own_image_build_argument_is_dropped():
    assert sass_bodies.by_signature({NEW: "h"}, True) == sass_bodies.by_signature({OLD: "h"}, True)


def test_another_images_build_argument_is_refused():
    with pytest.raises(SystemExit, match="in the sm_80 image"):
        sass_bodies.by_signature({NEW.replace("sm_86:", "sm_80:"): "h"}, True)


def test_the_source_stamp_is_never_a_verdict():
    a = sass_bodies.by_signature({OLD: "h", STAMP: "s1"}, True)
    b = sass_bodies.by_signature({NEW: "h", STAMP: "s2"}, True)
    assert sass_bodies.compare(a, b, "a", "b")
    assert not sass_bodies.compare(a, {**b, "sm_86:rola::k<rola::Arm<3> >(int)": "moved"}, "a", "b")
