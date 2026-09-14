# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The activation registry: one table, and every rule reads it.

A routing activation used to be a class whose
properties were prose in a docstring and a hand-written `can_zero`. Now it is a TAG
with a ROW, and the row is the only place the properties live.

**The shape, and why it is this shape.**

* *Users* write plain tagged constructors -- `softmax()`, `entmax(1.5)` -- and never
  see or supply property metadata. Property metadata supplied by a caller is
  metadata that can be supplied WRONG, and a validator reading it would then be
  validating a claim rather than a fact.
* *The package* keeps :data:`ACTIVATION_REGISTRY`, `tag -> ActivationProperties`,
  as the single source of truth. It is a plain serializable table, centrally
  auditable, and it is deliberately the same closed-world pattern as the arch table
  and the spill manifest: a thing either has a ratified row or it does not run.
* *Every validator reads COLUMNS, never tags.* A rule says "mass decay requires a
  `normalized` write side", not "mass decay requires softmax or entmax". **No case
  list over activations may exist anywhere outside this module** -- that is the
  invariant that makes adding a variant a one-row change instead of a hunt.
* *An unknown tag is a hard panic at config construction*, not a default and not a
  warning. A closed-world lookup is what makes "every rule applies to every variant"
  true by construction rather than by review.

**The columns.**

| column | question it answers |
|---|---|
| `normalized` | is the STORED level on the simplex (mass exactly 1)? |
| `simplex_output` | does the map's OWN output already land there, before the fold? |
| `exact_zeros` | does it produce exact structural zeros, not merely small values? |
| `support_differentiable` | does gradient flow through the support boundary? |
| `kerneled` | has the CUDA consumer been RATIFIED for this activation? |

`normalized` and `simplex_output` are asked on opposite sides of the producer's mass
fold and are NOT duplicates. `normalized` is a promise about what the BUNDLE carries
and is satisfiable BY the fold, so an opaque cell answers it `True`. `simplex_output`
is a claim about the PRODUCER's own arithmetic, and it is what licenses the fold to be
skipped, so an opaque cell answers it `False` -- reading `normalized` for that
decision would be circular (the fold would be skipped on the strength of a promise the
fold is what keeps).

`kerneled` is the kernel-dispatch column and it is a statement about EVIDENCE, not
about capability: `launch_consumer` remains the single authority on supported shapes
and a `False` here means "nobody measured it", which is why a config naming
such an activation runs on the reference path carrying an explicit `not_kerneled`
flag. That is a LABELED RESEARCH SURFACE, and it is a different thing from a kernel
fallback -- the kernel engineering standard's no-fallbacks rule is about the kernel
declining work it was asked to do, which never happens here.

**Extending it** is one row plus one constructor, and it is a MINOR version bump
The docs' capability matrix is GENERATED from this table by
:func:`capability_table`, so it cannot describe a table that does not exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

__all__ = [
    "ACTIVATION_REGISTRY",
    "ActivationProperties",
    "ActivationTag",
    "UnknownActivation",
    "capability_table",
    "properties_for",
    "ratified_entmax_alphas",
]


class UnknownActivation(ValueError):
    """Raised at config construction for a tag with no registry row."""


@dataclass(frozen=True, slots=True)
class ActivationProperties:
    """One registry row. Five booleans; no behaviour, so it stays serializable."""

    normalized: bool
    simplex_output: bool
    exact_zeros: bool
    support_differentiable: bool
    kerneled: bool


#: `(family, parameter)`. The parameter is `None` for a family with no parameter,
#: so that every tag has the same arity and the table can be keyed uniformly.
ActivationTag = tuple[str, float | None]


#: THE TABLE. v0.1 ratifies exactly two families and three tags.
#:
#: * `softmax` -- dense. Lands on the simplex, never exactly zero, and differentiable
#:   everywhere (there is no support boundary to be non-differentiable at).
#: * `entmax` at `alpha = 1.5` and `alpha = 2` (the latter is sparsemax). Both land
#:   on the simplex and both produce EXACT structural zeros -- that exactness is the
#:   whole reason the production solve is an exact entmax and not an epsilon-floored
#:   approximation (sparsity comes from entmax, not from a floor). Gradient does not flow across
#:   the support boundary: an off-support coordinate has zero gradient, which is a
#:   property of the map and not an implementation shortcut.
#:
#: `simplex_output` is TRUE for both families because each is a closed-form projection
#: onto the simplex: the sum is 1 to the floating-point floor by construction, not by a
#: subsequent renormalization. MEASURED across the shipped census (D in 1..4, fp32 and
#: fp64): `max|sum - 1|` is 2.4e-7 in fp32 and 4.4e-16 in fp64 on every cell whose
#: output is one solve's whole output.
ACTIVATION_REGISTRY: dict[ActivationTag, ActivationProperties] = {
    ("softmax", None): ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=False,
        support_differentiable=True, kerneled=True),
    ("entmax", 1.5): ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=True,
        support_differentiable=False, kerneled=True),
    ("entmax", 2.0): ActivationProperties(
        normalized=True, simplex_output=True, exact_zeros=True,
        support_differentiable=False, kerneled=True),
    ("opaque", None): ActivationProperties(
        normalized=True, simplex_output=False, exact_zeros=True,
        support_differentiable=False, kerneled=True),
}

#: THE ROW FOR A CELL WE DID NOT SOLVE. A producer is duck-typed -- anything callable
#: returning a :class:`~rola.routing.factors.RouteFactors` -- so a level may be filled
#: by a map this table has never heard of. Such a cell is `opaque`, and its row states
#: the honest position of every column:
#:
#: * `normalized` is TRUE because the fold normalizes what it stores: it factors every
#:   level into `(mass, simplex)` and moves the mass onto the side gain, so the stored
#:   level is on the simplex whatever the map returned.
#: * `simplex_output` is FALSE, and it is the column that keeps the line above honest:
#:   a foreign producer's own output is exactly the thing this package cannot vouch
#:   for, so an opaque cell always pays the generic fold.
#: * `exact_zeros` is TRUE, which is the CONSERVATIVE reading: it means "may produce
#:   zeros", so nothing may assume this cell's support is the full rectangle.
#: * `support_differentiable` is FALSE, the assumption that never over-promises.
#: * `kerneled` is TRUE, and it is the one that needs saying. The column means "the
#:   CONSUMER has been ratified for this cell", and the consumer eats tensors: it
#:   cannot tell what produced them (no producer identity reaches `ConsumerParams`).
#:   A foreign producer's own solve is its own module's business, so there is no
#:   unratified kernel of ours to route around.

#: The column names, derived from the dataclass so a new column cannot be added to
#: the rows and forgotten by the generated capability matrix.
ACTIVATION_COLUMNS: tuple[str, ...] = tuple(ActivationProperties.__annotations__)


def properties_for(tag: ActivationTag, *, owner: str) -> ActivationProperties:
    """The row for `tag`, or a hard panic naming what would have to be added.

    `owner` is the config object being constructed, so that the panic says which
    configuration is unbuildable rather than merely that a lookup missed.
    """
    try:
        return ACTIVATION_REGISTRY[tag]
    except (KeyError, TypeError):
        raise UnknownActivation(
            f"{owner}: no ratified activation {tag!r}. The registry is CLOSED-WORLD "
            f"(rola/routing/activations.py): known tags are "
            f"{sorted(ACTIVATION_REGISTRY, key=repr)}. A new variant needs a registry "
            "row -- at which point it participates in every validation rule "
            "immediately -- and adding one is a MINOR version bump. It does not get "
            "to run without one.") from None


def ratified_entmax_alphas() -> tuple[float, ...]:
    """The entmax `alpha` values with a row, READ OFF the registry.

    Exists so that `EntmaxActivation`'s validation is a registry lookup rather than a
    second list of alphas that has to be kept in step with the first one.
    """
    return tuple(sorted(
        parameter for family, parameter in ACTIVATION_REGISTRY
        if family == "entmax" and parameter is not None))


def validate_alpha(alpha: Any, *, owner: str) -> float:
    """A real scalar with an `entmax` row, or a panic. The ONLY alpha check."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"{owner}.alpha must be a real scalar, got {alpha!r}")
    alpha = float(alpha)
    properties_for(("entmax", alpha), owner=owner)
    return alpha


def capability_table() -> str:
    """The docs' capability matrix, GENERATED from the registry.

    Generated, not maintained. A hand-written matrix is a second
    copy of the table that goes stale the first time a row changes and nobody
    notices, because nothing checks prose.
    """
    header = ["activation", *ACTIVATION_COLUMNS]
    rows = [header, ["---"] * len(header)]
    for (family, parameter), properties in sorted(ACTIVATION_REGISTRY.items(), key=repr):
        name = family if parameter is None else f"{family}(alpha={parameter:g})"
        values = asdict(properties)
        rows.append([name, *("yes" if values[column] else "no" for column in ACTIVATION_COLUMNS)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)
