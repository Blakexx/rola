# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The producer/decode capacity relation, pinned across the language boundary.

``csrc/rola/src/decode/decode.cuh`` carries a ``static_assert`` that the decode path's
level-width capacity is at least the producer's ``MAX_BRANCH_WIDTH``. That assertion is
only half the gate: **a C++ ``static_assert`` cannot read a Python constant**, so the
C++ side compares against its own mirrored copy of the number, and a change to
``rola/routing/topology.py`` would move the producer's real limit while leaving the
mirror — and therefore the assertion — silently satisfied.

This file is the other half: it compares the SHIPPED BINARY's capacity against the
SHIPPED PYTHON's constant. Together the two make the relation checkable rather than
merely written down, and a topology the producer can emit but the decode path would
over-read on becomes a red test instead of an out-of-bounds read.
"""

from __future__ import annotations

import pytest
import torch

from rola.routing.topology import MAX_BRANCH_WIDTH

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_decode_capacity_covers_the_producer_max_branch_width():
    from rola.ops._ext import extension

    capacity = extension().rola_decode_capacity()
    assert capacity >= MAX_BRANCH_WIDTH, (
        f"the decode path's level-width capacity is {capacity} but the producer emits "
        f"levels up to MAX_BRANCH_WIDTH={MAX_BRANCH_WIDTH}. A topology the producer can "
        "build would over-read the decode kernel's staged amplitude row. Raise "
        "kDecodeMaxLevelWidth in csrc/rola/src/decode/decode.cuh (and re-check the GEMV's SMEM "
        "ledger, which scales with sum_l width_l).")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_the_cxx_mirror_of_max_branch_width_has_not_drifted():
    """The mirror is EQUAL, not merely bounded.

    The ``static_assert`` would still pass if the producer's limit dropped and the
    mirror stayed high, which is harmless, and if the producer's limit ROSE while the
    mirror stayed put, which is not. Pinning equality catches the second case at the
    only place both numbers are visible at once.
    """
    from rola.ops._ext import extension

    assert extension().rola_decode_producer_width_mirror() == MAX_BRANCH_WIDTH, (
        "csrc/rola/src/decode/decode.cuh's kProducerMaxBranchWidth mirror has drifted from "
        f"rola/routing/topology.py's MAX_BRANCH_WIDTH={MAX_BRANCH_WIDTH}; update the "
        "mirror in the same commit that moves the producer's limit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_the_binary_under_test_is_this_tree():
    """A device-side fact, because nothing else catches a stale extension.

    The stamp is computed BY A KERNEL out of the loaded fatbin's own view of the
    ENUMERATION SHAPE (``kKinds * kMaxTerms``), ``sizeof(DecodeParams)``,
    ``sizeof(Unit<4>)`` and ``kDecodeMaxSpan`` — facts a path check, a hash check and an
    import check are all blind to. The SPAN ceiling is the half with teeth here: it is the
    same number ``rola.ops.lattice`` refuses candidate boxes against, so a host that
    outgrew the device's box fails here. The shape field is what moves the stamp when the
    walk's index space is redefined without either struct changing size.
    """
    from rola.ops._ext import extension
    from rola.ops.lattice import _MAX_SPAN

    stamp = extension().rola_decode_build_stamp()
    assert stamp > 0, "the decode build stamp is not a device-side fact"
    high, remainder = divmod(stamp, 1000003)
    shape, params_bytes = divmod(high, 4096)
    unit_bytes, max_span = divmod(remainder, 101)
    assert max_span == _MAX_SPAN, (
        f"the binary's kDecodeMaxSpan is {max_span} and the host solver refuses "
        f"candidates above {_MAX_SPAN}; one of the two is stale")
    assert 0 < params_bytes < 4096 and 0 < unit_bytes < 4096, (
        f"the stamp does not decode to plausible struct sizes ({params_bytes}, "
        f"{unit_bytes} bytes); the loaded binary is not this tree's")
    assert 0 < shape < 4096, (
        f"the stamp's enumeration-shape field is {shape}; the loaded binary is not "
        "this tree's")
