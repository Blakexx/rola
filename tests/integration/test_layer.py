"""The routing vocabulary's own gate.

The vocabulary IS the tag vocabulary: one level config per routing level, each carrying
its width and its per-duty activation OBJECTS -- ``IndependentRouting(width, read,
write)``, ``TiedRouting(width, op)`` and ``UnionRouting(width, alpha)``
(``rola.routing.types``), with ``dense_routing``/``union_routing``/``split_routing``/
``tied_routing`` as the spellings and ``uniform`` as the way to say one of them D times.
Every row here holds one refusal that vocabulary owes a caller: a value that is not an
activation, a width that cannot be a width, a depth past the built bound.
"""

from __future__ import annotations

import inspect

import pytest

from rola.routing.types import IndependentRouting, UnionRouting, entmax, softmax

# ---------------------------------------------------------------------------
# `IndependentRouting`/`UnionRouting` validation -- the tag vocabulary's own gate.
# ---------------------------------------------------------------------------


def test_independent_routing_has_only_public_fields():
    assert set(inspect.signature(IndependentRouting).parameters) == {
        'width', 'read', 'write'}
    with pytest.raises(TypeError):
        IndependentRouting(probability='softmax')


def test_union_routing_has_only_public_fields():
    assert set(inspect.signature(UnionRouting).parameters) == {'width', 'alpha'}
    with pytest.raises(TypeError):
        UnionRouting(probability='softmax')


@pytest.mark.parametrize('side', ['read', 'write'])
@pytest.mark.parametrize('value', ['softmax', 'entmax', 'SPARSE', 1, None])
def test_independent_routing_rejects_non_activation_side_values(side, value):
    """The old config took `read='sparse'`/`write='dense'` strings and rejected an
    unrecognized one with `ValueError`. The new vocabulary takes activation OBJECTS
    (`softmax()`/`entmax(alpha)`) -- a bare string, int or `None` has no `.tag` to
    look up in the registry, so the failure is `TypeError` ("not an activation"),
    not `ValueError` ("not a recognized string")."""
    kwargs = dict(width=4, read=softmax(), write=softmax())
    kwargs[side] = value
    with pytest.raises(TypeError):
        IndependentRouting(**kwargs)


@pytest.mark.parametrize('value', [True, 1, 1.8, float('inf'), float('nan')])
def test_union_routing_validates_alpha_like_entmax(value):
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        UnionRouting(width=4, alpha=value)
    with pytest.raises(error):
        entmax(value)
