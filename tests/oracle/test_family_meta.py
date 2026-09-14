"""META — the non-vacuity machinery itself is under test, and it has teeth.

The family scheme's self-policing rule is only as strong as its checkers: a
checker that passes on a fixture OUTSIDE its regime would let a family drift
out of its claimed cell silently, which is exactly the vacuity failure the rule
generalizes. So every ``(axis, value)`` checker is fed a fixture that VIOLATES
its regime and must refuse it; the checker table is proven complete against
``REGIME_AXES`` in both directions; and the adversarial corners are FROZEN — a
corner, once named, may never leave the family list.

Pure CPU: the checkers are tensor arithmetic, and teeth must be checkable in
the CPU CI job where the census runs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.oracle import generators as g
from tests.oracle.families import FAMILY_BY_NAME, REGIME_AXES


def _simplex(shape, p, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, dtype=torch.float64, generator=gen)
    if p < 1.0:
        x = x * (torch.rand(shape, dtype=torch.float64, generator=gen) < p)
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return x / x.sum(-1, keepdim=True)


def _one_hot(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    digits = torch.randint(0, shape[-1], shape[:-1], generator=gen)
    return torch.zeros(shape, dtype=torch.float64).scatter_(-1, digits.unsqueeze(-1), 1.0)


def _view(read, write, *, widths=(8,), T=32, BT=16, P=1, initial_state=None):
    return SimpleNamespace(read_levels=read, write_levels=write, widths=widths,
                           T=T, BT=BT, P=P, initial_state=initial_state)


_S = (1, 32, 1, 8)
_DENSE = lambda seed=0: (_simplex(_S, 1.0, seed),)              # noqa: E731
_SPARSE = lambda seed=0: (_simplex(_S, 0.4, seed),)             # noqa: E731
_CONSTANT = lambda: (_simplex((1, 1, 1, 8), 1.0).expand(_S),)   # noqa: E731


#: One VIOLATING view per checker. The checker must REFUSE it; a checker that
#: accepts its violation has no teeth and the family scheme is decorative.
VIOLATIONS = {
    ("density", "dense"): _view(_SPARSE(), _DENSE(1)),
    ("density", "sparse"): _view(_DENSE(), _DENSE(1)),
    ("coherence", "iid"): _view(_CONSTANT(), _DENSE(1)),
    ("coherence", "coherent"): _view(_DENSE(), _DENSE(1)),
    ("rw_correlation", "tied"): _view(_DENSE(), _DENSE(1)),
    ("rw_correlation", "independent"): _view(_DENSE(), _DENSE()),      # tied
    ("rw_correlation", "anti"): _view(_DENSE(), _DENSE(1)),            # overlapping
    ("mass", "spread"): _view(_one_hot(_S), _one_hot(_S, 1)),
    ("mass", "concentrated"): _view(_DENSE(), _DENSE(1)),
    ("mass", "cold_read"): _view(_DENSE(), _DENSE(1)),                 # everything written
    ("tail", "divisible"): _view(_DENSE(), _DENSE(1), T=33),
    ("tail", "ragged"): _view(_DENSE(), _DENSE(1), T=32),
    ("support", "singleton"): _view(_DENSE(), _DENSE(1)),
    ("support", "partial"): _view(_one_hot(_S), _one_hot(_S, 1)),
    ("support", "full"): _view(_SPARSE(), _SPARSE(1)),
}


def test_the_checker_table_is_complete_in_both_directions():
    declared = {(axis, value) for axis, values in REGIME_AXES.items() for value in values}
    assert set(g.CHECKERS) == declared, (
        f"missing checkers: {declared - set(g.CHECKERS)}; "
        f"orphan checkers: {set(g.CHECKERS) - declared}")
    assert set(VIOLATIONS) == declared - {("density", "swept")}, (
        "every checker needs a violation fixture here (swept is proven by "
        "assert_swept, whose teeth are below)")


@pytest.mark.parametrize("key", sorted(VIOLATIONS), ids=lambda k: f"{k[0]}={k[1]}")
def test_every_checker_refuses_its_violation(key):
    with pytest.raises(AssertionError):
        g.CHECKERS[key](VIOLATIONS[key])


def test_every_checker_accepts_a_fixture_inside_its_regime():
    """The other half of teeth: a checker that refuses EVERYTHING is equally
    useless. The committed families are the in-regime witnesses, and they run
    on CUDA in test_families.py; here the cheap CPU half — each checker passes
    on at least one hand-built in-regime view — keeps the pair honest without a
    device."""
    m0 = torch.zeros(1, 1, 8, 1)
    passing = {
        ("density", "dense"): _view(_DENSE(), _DENSE(1)),
        ("density", "sparse"): _view(_SPARSE(), _SPARSE(1)),
        ("density", "swept"): _view(_DENSE(), _DENSE(1)),   # no-op checker
        ("coherence", "iid"): _view(_DENSE(), _DENSE(1)),
        ("coherence", "coherent"): _view(_CONSTANT(), _DENSE(1)),
        ("rw_correlation", "tied"): _view(_DENSE(), _DENSE()),
        ("rw_correlation", "independent"): _view(_DENSE(), _DENSE(1)),
        ("rw_correlation", "anti"): _view(
            (_simplex(_S, 1.0) * torch.tensor([1.0] * 4 + [0.0] * 4),),
            (_simplex(_S, 1.0, 1) * torch.tensor([0.0] * 4 + [1.0] * 4),)),
        ("mass", "spread"): _view(_DENSE(), _DENSE(1)),
        ("mass", "concentrated"): _view(_one_hot(_S), _one_hot(_S, 1)),
        ("mass", "cold_read"): _view(
            _DENSE(), (_simplex(_S, 1.0, 1) * torch.tensor([1.0] * 4 + [0.0] * 4),),
            initial_state=m0),
        ("tail", "divisible"): _view(_DENSE(), _DENSE(1), T=32),
        ("tail", "ragged"): _view(_DENSE(), _DENSE(1), T=33),
        ("support", "singleton"): _view(_one_hot(_S), _one_hot(_S, 1)),
        ("support", "partial"): _view(_SPARSE(), _SPARSE(1)),
        ("support", "full"): _view(_DENSE(), _DENSE(1)),
    }
    # The anti and cold-read views above are masked products rather than
    # renormalized simplices; both checkers read SUPPORTS only, so that is
    # fine by design.
    assert set(passing) == set(g.CHECKERS)
    for key, view in passing.items():
        g.CHECKERS[key](view)


def test_assert_swept_refuses_a_flat_and_a_nonmonotone_sweep():
    with pytest.raises(AssertionError):
        g.assert_swept([0.5, 0.45, 0.4, 0.38])          # < 4x span
    with pytest.raises(AssertionError):
        g.assert_swept([1.0, 0.2, 0.5, 0.1])            # non-monotone
    g.assert_swept([1.0, 0.8, 0.3, 0.1])                # a real sweep passes


def test_corner_checks_have_teeth():
    # all-read-only with TWO written leaves is refused
    two = torch.zeros(_S, dtype=torch.float64)
    two[..., :16, :, 0] = 1.0
    two[..., 16:, :, 1] = 1.0
    with pytest.raises(AssertionError):
        g.CORNER_CHECKS["corner/all-read-only"](_view(_DENSE(), (two,)))
    # chunk-boundary with a MID-TILE flip is refused
    mask = torch.ones(_S, dtype=torch.float64)
    mask[:, 8:, :, 4:] = 0.0    # flips at token 8, not a multiple of BT=16
    flipped = _simplex(_S, 1.0) * mask
    flipped = flipped / flipped.sum(-1, keepdim=True)
    bad = _view(_DENSE(), (flipped,))
    bad._flip_tokens = (8, 16)
    with pytest.raises(AssertionError):
        g.CORNER_CHECKS["corner/chunk-boundary"](bad)


#: THE CORNERS ARE FROZEN PERMANENT. This week's bug list, kept forever:
#: deleting one is deleting a regression gate. A new bug-discovered axis ADDS a
#: name here and a Family beside it — names only ever accumulate.
FROZEN_CORNERS = frozenset({
    "corner/cold-read",
    "corner/all-read-only",
    "corner/chunk-boundary",
})


def test_the_adversarial_corners_are_permanent():
    missing = FROZEN_CORNERS - set(FAMILY_BY_NAME)
    assert not missing, (
        f"adversarial corner(s) {sorted(missing)} have been removed from the "
        "family list. Corners are FROZEN PERMANENT (ratified design, "
        "2026-08-05): they are this week's bug list kept forever, and deleting "
        "one is deleting a regression gate.")
    for name in FROZEN_CORNERS:
        assert FAMILY_BY_NAME[name].source == "adversarial"
