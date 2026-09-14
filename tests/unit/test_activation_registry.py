# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The activation registry.

The point of the registry is that activation properties live in ONE table and every
rule reads columns from it. So most of what is gated here is not "does softmax have
the right flags" -- it is the *structural* claims: that the table is the only source,
that an unknown tag cannot run, that rules are stated against columns rather than
against activation names, and that the docs' matrix is generated rather than typed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import rola
import rola.expert
from rola.routing.activations import (
    ACTIVATION_COLUMNS,
    ACTIVATION_REGISTRY,
    ActivationProperties,
    UnknownActivation,
    capability_table,
    properties_for,
    ratified_entmax_alphas,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
    entmax,
    softmax,
)

# --- the table itself --------------------------------------------------------


def test_v0_1_ratifies_exactly_the_two_declared_activation_families():
    """v0.1 ships exactly two ratified families, plus the opaque row.

    A fourth row appearing without a version bump and a docs change is the thing this
    catches. ``("opaque", None)`` is the row a level filled by a FOREIGN producer reads,
    and it is a row rather than a special case precisely so that every rule quantifying
    over cells covers such a level without knowing it is one.
    """
    assert set(ACTIVATION_REGISTRY) == {
        ("softmax", None), ("entmax", 1.5), ("entmax", 2.0), ("opaque", None)}
    assert all(row.kerneled for row in ACTIVATION_REGISTRY.values())


def test_every_row_declares_every_column():
    for tag, row in ACTIVATION_REGISTRY.items():
        assert isinstance(row, ActivationProperties), tag
        for column in ACTIVATION_COLUMNS:
            assert isinstance(getattr(row, column), bool), f"{tag}.{column}"


def test_the_declared_properties_are_the_ones_the_maps_actually_have():
    """The table is only worth reading if it is true. Checked against the reference
    implementations, not against the docstrings that describe them."""
    logits = torch.tensor([[3.0, 0.5, -2.0, -4.0]], dtype=torch.float64)
    from rola.routing.entmax.reference import entmax as entmax_reference

    dense = torch.softmax(logits, dim=-1)
    assert torch.allclose(dense.sum(-1), torch.ones(1, dtype=torch.float64))
    assert (dense > 0).all(), "softmax declares exact_zeros=False"
    assert ACTIVATION_REGISTRY[("softmax", None)].normalized
    assert not ACTIVATION_REGISTRY[("softmax", None)].exact_zeros

    for alpha in ratified_entmax_alphas():
        p = entmax_reference(logits, alpha=alpha)
        assert torch.allclose(p.sum(-1), torch.ones(1, dtype=torch.float64), atol=1e-9)
        assert (p == 0).any(), f"entmax({alpha}) declares exact_zeros=True"
        assert ACTIVATION_REGISTRY[("entmax", alpha)].exact_zeros
        assert ACTIVATION_REGISTRY[("entmax", alpha)].normalized


# --- closed world ------------------------------------------------------------

def test_an_unknown_tag_is_a_hard_panic_at_config_construction():
    """Closed-world lookup. A variant either has a row -- and therefore participates
    in every rule immediately -- or nothing runs. There is no default row."""
    with pytest.raises(UnknownActivation, match="CLOSED-WORLD"):
        properties_for(("entmax", 1.25), owner="test")
    with pytest.raises(UnknownActivation, match="CLOSED-WORLD"):
        properties_for(("sigmoid_top_k", 8), owner="test")
    with pytest.raises(UnknownActivation):
        EntmaxActivation(1.25)
    with pytest.raises(UnknownActivation):
        UnionRouting(width=4, alpha=3.0)


def test_the_admissible_alphas_are_read_off_the_registry_not_listed_twice():
    assert ratified_entmax_alphas() == (1.5, 2.0)
    assert EntmaxActivation(2).alpha == 2.0
    assert entmax(1.5).tag == ("entmax", 1.5)
    assert softmax().tag == ("softmax", None)


def test_a_registry_row_added_at_runtime_is_immediately_constructible():
    """The extension story, exercised rather than asserted: one row, and the variant
    exists everywhere. If this needed a second edit somewhere, the registry would not
    be the single source of truth."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=True,
        support_differentiable=False, kerneled=False)
    try:
        activation = EntmaxActivation(1.75)
        assert activation.properties.exact_zeros
        assert not activation.properties.kerneled
        assert 1.75 in ratified_entmax_alphas()
    finally:
        del ACTIVATION_REGISTRY[tag]


# --- users never write property metadata -------------------------------------

def test_the_public_constructors_take_no_property_metadata():
    """Plain tagged constructors only. A caller that could supply
    `normalized=` could supply it wrong, and every validator downstream would then be
    checking a claim instead of a fact."""
    assert [f.name for f in SoftmaxActivation.__dataclass_fields__.values()] == []
    assert [f.name for f in EntmaxActivation.__dataclass_fields__.values()] == ["alpha"]
    for column in ACTIVATION_COLUMNS:
        with pytest.raises(TypeError):
            EntmaxActivation(1.5, **{column: True})


# --- rules read columns, never names -----------------------------------------

def _topology(read, write, *, width=4, tied=True):
    if tied:
        # Callers that ask for `tied=True` always pass the same activation for both
        # duties (the only case a tied level can express).
        return Topology(levels=(TiedRouting(width=width, op=read),))
    return Topology(levels=(IndependentRouting(width=width, read=read, write=write),))


def test_per_level_and_per_duty_configuration_is_addressable_as_a_grid():
    """Per-level AND per-duty configuration is what makes level-split pure
    CONFIGURATION (M-3) rather than a code path."""
    topology = _topology(softmax(), entmax(1.5), tied=False)
    assert topology.duty_cells() == ((0, "read"), (0, "write"))
    assert topology.activation(0, "read").tag == ("softmax", None)
    assert topology.activation(0, "write").tag == ("entmax", 1.5)
    assert topology.activation_properties(0, "write").exact_zeros
    with pytest.raises(ValueError, match="duty must be"):
        topology.activation(0, "both")


def test_mass_decay_demands_a_normalized_write_side_loudly():
    """THE property-violation test. `normalized=False` on the write side is refused
    with an error naming the COLUMN that failed and why it matters -- never silently
    accepted, and never phrased as "softmax or entmax is required"."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=False, simplex_output=False, exact_zeros=False,
        support_differentiable=True, kerneled=False)
    try:
        topology = _topology(softmax(), EntmaxActivation(1.75), tied=False)
        decay = LeafMassDecay(dials=(torch.full((1, 4), 0.5),))
        with pytest.raises(ValueError, match="normalized"):
            decay.validate_against(topology)
    finally:
        del ACTIVATION_REGISTRY[tag]


def test_an_activation_with_a_row_is_configurable_without_widening_an_isinstance_list():
    """Admission is a registry lookup, so a variant with a row is usable at once.
    A duck-typed activation stands in for a future family the shipped classes cannot
    express -- if this needed a second edit in `types.py`, the registry would not be
    the single source of truth it claims to be."""

    class _Future:
        tag = ("future_family", None)

    ACTIVATION_REGISTRY[("future_family", None)] = ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=False,
        support_differentiable=True, kerneled=False)
    try:
        routing = IndependentRouting(width=4, read=_Future(), write=_Future())
        assert routing.activation("read").tag == ("future_family", None)
    finally:
        del ACTIVATION_REGISTRY[("future_family", None)]

    with pytest.raises(UnknownActivation):
        IndependentRouting(width=4, read=_Future(), write=_Future())


def test_mass_decay_accepts_every_ratified_activation():
    """Non-vacuity for the rule above: it must not reject the shipped world."""
    for activation in (softmax(), entmax(1.5), entmax(2.0)):
        topology = _topology(activation, activation)
        LeafMassDecay(dials=(torch.full((1, 4), 0.5),)).validate_against(topology)


def test_union_routing_demands_exact_zeros_for_the_intersection_guarantee():
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=False,
        support_differentiable=True, kerneled=False)
    try:
        with pytest.raises(ValueError, match="exact_zeros"):
            UnionRouting(width=4, alpha=1.75)
    finally:
        del ACTIVATION_REGISTRY[tag]


def test_two_exact_zero_sides_still_require_one_shared_solve():
    with pytest.raises(ValueError, match="exact_zeros"):
        IndependentRouting(width=4, read=entmax(1.5), write=entmax(1.5))


def test_tied_routing_admits_any_registered_activation_without_widening_an_isinstance_list():
    """Same non-vacuity as `IndependentRouting`'s duck-typed row above: a `TiedRouting`
    admits a future family the moment it has a registry row, checked by lookup rather
    than by an isinstance list against the two shipped classes."""

    class _Future:
        tag = ("future_family", None)

    ACTIVATION_REGISTRY[("future_family", None)] = ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=False,
        support_differentiable=True, kerneled=False)
    try:
        tied = TiedRouting(width=4, op=_Future())
        assert tied.activation("read").tag == tied.activation("write").tag == ("future_family", None)
    finally:
        del ACTIVATION_REGISTRY[("future_family", None)]

    with pytest.raises(UnknownActivation):
        TiedRouting(width=4, op=_Future())


def test_mass_decay_accepts_tied_routing_and_still_demands_a_normalized_write_side():
    """`LeafMassDecay` reads `topology.activation(level, 'write')` generically -- a
    `TiedRouting` level needs no special case, on either side of the rule."""
    topology = Topology(levels=(TiedRouting(width=4, op=entmax(1.5)),))
    LeafMassDecay(dials=(torch.full((1, 4), 0.5),)).validate_against(topology)

    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=False, simplex_output=False, exact_zeros=False,
        support_differentiable=True, kerneled=False)
    try:
        unnormalized = Topology(levels=(TiedRouting(width=4, op=EntmaxActivation(1.75)),))
        with pytest.raises(ValueError, match="normalized"):
            LeafMassDecay(dials=(torch.full((1, 4), 0.5),)).validate_against(unnormalized)
    finally:
        del ACTIVATION_REGISTRY[tag]


# --- the not-kerneled surface ------------------------------------------------

class _StubBundle:
    def __init__(self, topology):
        self.topology = topology
        self.device = torch.device("cuda")


def test_an_unratified_activation_is_refused_and_the_refusal_names_it():
    """Nobody measured a kernel for it, so there is no arm and the envelope says
    so BY NAME. The flag is a property of the configuration, so the answer is
    available before the call and without a device -- and it wins over every clause
    that would otherwise admit the call."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=True,
        support_differentiable=False, kerneled=False)
    try:
        topology = _topology(entmax(1.75), entmax(1.75))
        assert not topology.kerneled
        assert topology.unratified_activations == (
            (0, "read", ("entmax", 1.75)), (0, "write", ("entmax", 1.75)))

        refusal = rola.expert.envelope_refusal(
            _StubBundle(topology), torch.zeros(1, 1, 1, 64))
        assert "unratified activations" in refusal and "entmax" in refusal
    finally:
        del ACTIVATION_REGISTRY[tag]


#: RETIRED with the tiled consumer: `test_the_private_kernel_path_refuses_an_unratified_
#: activation_rather_than_running_it`. Its subject was the retired tiled arm's own
#: activation check -- a refusal at its private entry, so that a caller who
#: bypassed the dispatch could not launch something nobody measured. The chunk arm
#: has no such entry to bypass (`chunk.execute` has exactly one caller and it is inside
#: the autograd node, asserted structurally in `tests/unit/test_interface.py`), and
#: the path the check guarded is closed at BOTH ends by rows that already exist:
#: the GATEWAY refuses to produce an unratified activation's routing at all
#: (`rola/routing/entmax/production.py` raises -- MEASURED here while re-anchoring:
#: registering `("entmax", 1.75)` does not make the producer able to emit it), and
#: `rola_op` refuses any bundle the producer did not derive
#: (`tests/unit/test_api_contract.py`), so no bundle carrying an unratified
#: activation can reach any arm. The DISPATCH half -- an unratified activation goes
#: to the reference arm and is named in `Dispatch.not_kerneled` -- is the row
#: directly above this one.


def test_every_shipped_configuration_is_kerneled():
    for activation in (softmax(), entmax(1.5), entmax(2.0)):
        assert _topology(activation, activation).kerneled


# --- no case list may exist outside the registry -----------------------------

_ALLOWED_TO_NAME_ACTIVATIONS = {
    # The registry itself, and the two tagged constructors that key into it.
    "routing/activations.py",
    "routing/types.py",
    # The producer and the reference lower a tag to a kernel/solve; that dispatch IS
    # a mapping from tag to implementation and cannot be a column.
    "routing/producer.py",
    "routing/reference.py",
    "routing/entmax/production.py",
    "routing/entmax/reference.py",
    # The layer names activations where it validates the decay source against them.
    "layer.py",
}


def test_no_module_outside_the_registry_branches_on_an_activation_type():
    """Rules state preconditions against COLUMNS, never against tag lists.
    A module that does `isinstance(a, EntmaxActivation)` to decide a POLICY is a
    second, invisible copy of the table.

    The exemptions are the modules that lower a tag to an implementation -- that is a
    dispatch, not a rule, and it is exactly what a registry column cannot express.
    """
    package = pathlib.Path(rola.__path__[0])
    names = {"SoftmaxActivation", "EntmaxActivation"}
    offenders = []
    for path in sorted(package.rglob("*.py")):
        relative = str(path.relative_to(package))
        if relative in _ALLOWED_TO_NAME_ACTIVATIONS:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in names:
                offenders.append(f"{relative}:{node.lineno}:{node.id}")
    assert not offenders, (
        "these modules name an activation class outside the registry: " + str(offenders))


# --- the docs' matrix is generated -------------------------------------------

def test_the_capability_matrix_is_generated_from_the_registry():
    table = capability_table()
    assert table.splitlines()[0] == "| activation | " + " | ".join(ACTIVATION_COLUMNS) + " |"
    assert "| entmax(alpha=1.5) | yes | yes | yes | no | yes |" in table
    assert "| softmax | yes | yes | no | yes | yes |" in table
    assert len(table.splitlines()) == len(ACTIVATION_REGISTRY) + 2


def test_a_new_column_reaches_the_matrix_without_being_typed_anywhere():
    assert tuple(ActivationProperties.__annotations__) == ACTIVATION_COLUMNS
