"""THE PRODUCER'S DENSITY SWEEP: the logit gain is the knob that moves realized routing's density, as it does in
training, so a cell set that claims to cover the distribution has to show the knob moving it.

The constructions are RoLA's `producer-sweep-g*` (`benchmarks.cells.layer.CONSTRUCTIONS`): one entmax constructor at
four logit gains on one central layer input. Each is built by `benchmarks.cells.layer.build`, and its REALIZED write
density -- the exact-nonzero fraction of the write side's amplitudes -- must fall monotonically as the gain rises, across
at least a factor of four.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import by_name
from benchmarks.cells.layer import CONSTRUCTIONS, build

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is built on the GPU")]


def _write_density(routes) -> float:
    live = sum(int((level != 0).sum()) for level in routes.write)
    return live / sum(level.numel() for level in routes.write)


def test_the_logit_gain_sweeps_the_realized_density():
    sweep = [c for name, c in CONSTRUCTIONS.items() if name.startswith("producer-sweep-")]
    assert len(sweep) == 4 and [c.logit_gain for c in sweep] == sorted(c.logit_gain for c in sweep), (
        "the sweep is four constructions listed by rising gain")
    assert len({c.cells for c in sweep}) == 1, "the sweep moves the gain alone, on one input"
    densities = [_write_density(build(by_name(c.cells[0]), c)["routes"]) for c in sweep]
    assert all(a >= b for a, b in zip(densities, densities[1:])), (
        f"realized write densities {densities} do not fall as the gain rises; the gain is not the density knob")
    assert densities[0] >= 4.0 * max(densities[-1], 1e-9), (
        f"realized write densities {densities} span less than 4x; the sweep does not move the distribution")
