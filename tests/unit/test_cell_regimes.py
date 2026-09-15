"""`benchmarks/cells/regimes.py`: the checks that prove a carry cell's draw sits in its declared regime have teeth,
and the registry covers the whole box.

No GPU: the checks are tensor arithmetic over small planted views, and the registry is read, not realized (every
cell's draw is proven where it is realized, on the device, by `benchmarks.cells.realize`).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from benchmarks.cells import _validate, carry_cells
from benchmarks.cells.regimes import REGIME_AXES, View, check
from rola.ops.carry import WINDOW

#: THE CORNERS, frozen: a named region of the distribution, once a cell, never leaves the registry.
CORNERS = ("corner-tied-k4", "corner-concentrated", "corner-window-flip", "corner-onehot", "corner-anti",
           "corner-cold-read", "corner-read-only")

T, WIDTH = 64, 8


def _simplex(p=1.0, seed=0, shape=(1, T, 1, WIDTH)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, dtype=torch.float64, generator=gen)
    if p < 1.0:
        x = x * (torch.rand(shape, dtype=torch.float64, generator=gen) < p)
        x[..., 0] = x[..., 0] + (x.sum(-1) == 0)
    return x / x.sum(-1, keepdim=True)


def _one_hot(seed=0, digit=None):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, WIDTH, (1, T, 1, 1), generator=gen) if digit is None else torch.full((1, T, 1, 1), digit)
    return torch.zeros(1, T, 1, WIDTH, dtype=torch.float64).scatter_(-1, idx, 1.0)


def _half(second, seed):
    x = _simplex(seed=seed)
    x[..., WIDTH // 2:] *= float(second)
    x[..., :WIDTH // 2] *= float(not second)
    return x / x.sum(-1, keepdim=True)


def _view(read, write, tokens=T, entry=None):
    return View(SimpleNamespace(tokens=tokens), (read,), (write,), entry)


DENSE, OTHER = _simplex(seed=1), _simplex(seed=2)
SPARSE = _simplex(0.4, seed=3)
CONSTANT = _simplex(seed=4)[:, :1].expand(1, T, 1, WIDTH)
ENTRY = torch.ones(1, WIDTH, 2)

#: For every (axis, value): a view that demonstrates it, and one that does not.
CASES = {
    ("density", "dense"): (_view(DENSE, OTHER), _view(SPARSE, OTHER)),
    ("density", "sparse"): (_view(SPARSE, OTHER), _view(DENSE, OTHER)),
    ("coherence", "iid"): (_view(DENSE, OTHER), _view(CONSTANT, OTHER)),
    ("coherence", "coherent"): (_view(CONSTANT, OTHER), _view(DENSE, OTHER)),
    ("rw_correlation", "tied"): (_view(DENSE, DENSE), _view(DENSE, OTHER)),
    ("rw_correlation", "independent"): (_view(DENSE, OTHER), _view(DENSE, DENSE)),
    ("rw_correlation", "anti"): (_view(_half(False, 5), _half(True, 6)), _view(DENSE, OTHER)),
    ("mass", "spread"): (_view(DENSE, OTHER), _view(_one_hot(1), _one_hot(2))),
    ("mass", "concentrated"): (_view(_one_hot(1), _one_hot(2)), _view(DENSE, OTHER)),
    ("mass", "cold_read"): (_view(DENSE, _one_hot(digit=0), entry=ENTRY), _view(DENSE, OTHER, entry=ENTRY)),
    ("tail", "divisible"): (_view(DENSE, OTHER, tokens=2 * WINDOW), _view(DENSE, OTHER, tokens=WINDOW + 1)),
    ("tail", "ragged"): (_view(DENSE, OTHER, tokens=WINDOW + 1), _view(DENSE, OTHER, tokens=2 * WINDOW)),
    ("support", "singleton"): (_view(DENSE, _one_hot(2)), _view(DENSE, OTHER)),
    ("support", "partial"): (_view(SPARSE, OTHER), _view(DENSE, OTHER)),
    ("support", "full"): (_view(DENSE, OTHER), _view(SPARSE, OTHER)),
}


def test_every_value_of_every_axis_has_a_case_and_nothing_else_does():
    assert set(CASES) == {(axis, value) for axis, values in REGIME_AXES.items() for value in values}
    with pytest.raises(KeyError):
        check("temperature", "hot", CASES[("density", "dense")][0])


@pytest.mark.parametrize("axis,value", sorted(CASES), ids=[f"{a}={v}" for a, v in sorted(CASES)])
def test_a_check_accepts_its_regime_and_refuses_another(axis, value):
    inside, outside = CASES[(axis, value)]
    assert check(axis, value, inside) is None
    assert check(axis, value, outside) is not None, f"{axis}={value} accepted a view outside it: the check has no teeth"


def test_a_cold_read_without_an_entry_state_is_refused():
    assert check("mass", "cold_read", _view(DENSE, _one_hot(digit=0))) is not None


def test_a_record_states_every_axis_or_none_at_all():
    record = {"name": "x", "widths": [16, 16], "dv": 64, "tokens": 16, "warps_per_cta": 8,
              "draw": "dense", "k_tok": None, "cohort": None, "support": 1.0, "backing": "dense",
              "state": "fresh", "tier": "oracle", "regime": None}
    with pytest.raises(ValueError, match="states its regime"):
        _validate(record)
    partial = dict(record, regime={"density": "dense"})
    with pytest.raises(ValueError, match="not every axis"):
        _validate(partial)
    with pytest.raises(ValueError, match="declares no regime"):
        _validate(dict(record, draw="dead", regime={axis: values[0] for axis, values in REGIME_AXES.items()}))


def test_the_corners_stay_and_the_registry_covers_the_whole_box():
    cells = carry_cells()
    names = {cell.name for cell in cells}
    assert set(CORNERS) <= names, f"a frozen corner left the registry: {set(CORNERS) - names}"
    covered = {pair for cell in cells if cell.regime for pair in cell.regime}
    missing = {(axis, value) for axis, values in REGIME_AXES.items() for value in values} - covered
    assert not missing, f"no registry cell demonstrates {sorted(missing)}"
