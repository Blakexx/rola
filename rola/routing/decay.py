# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decay-source slot: where the recurrence's forgetting rates come from.

Decay is a recurrence dial, so it is part of the reduction and belongs on the public
surface. WHERE its per-digit rates come from is not: a constant the user picked, one
learned scalar, one learned rate per level, or one per state are four answers to the
same question, and the recurrence cannot tell them apart. So the source is a SLOT with
the same shape as the feature-map slot -- one small contract, interchangeable
implementations, ours shipped, foreign ones duck-typed and validated at construction.

**The contract.** A decay source declares the shape it emits for
(``widths``, ``num_heads``) and returns a :class:`~rola.routing.types.LeafMassDecay`
whose ``dials[l]`` is ``[H, width_l]`` with entries in ``[0, 1)``. That is the whole
interface; :func:`validate_decay_source` checks it against the topology at layer
construction, so a mismatched source fails there and never inside a forward.

**Why this is a slot and not a preference.** The taxonomy the queue records: GUARANTEES
live in the plan (measured, hard, free), CHOICES live in slots (user or learned,
interchangeable), PREFERENCES live in optional losses (soft, opt-in, never load
bearing). Decay's rate is a CHOICE. Setting :class:`ConstantDecay` high is a
performance lever with no new mechanism behind it: fast forgetting shows up in the
realized routing as sparser write support, through exactly the door a learned source
uses.

**The clock this feeds is write-conditioned** (mass decay, never time decay). A leaf
whose write amplitude is exactly zero ticks its clock by zero, so ``keep = (1 -
rate)**0 = 1`` exactly and its state is left bit-unchanged. Several decisions elsewhere
rest on that, so a decay source that tried to decay on wall-clock time rather than on
writes would not be a new implementation of this slot -- it would be a different
recurrence, and it is vetoed as one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rola.routing.types import LeafMassDecay, Topology

__all__ = [
    "ConstantDecay",
    "DecaySource",
    "LearnedDecay",
    "validate_decay_source",
]


def _symmetric_rate(target_leaf_rate: float, depth: int) -> float:
    """Every level's equal share of a target product rate: ``rho ** (1/D)``.

    The leaf rate is the product over levels, so the symmetric split is the one
    initialization that reproduces the requested ``rho`` exactly at ``D`` levels
    without preferring a level.
    """
    if not 0.0 < target_leaf_rate < 1.0:
        raise ValueError(
            f"target_leaf_rate must lie in (0, 1) -- it is a product of per-level rates and "
            f"rate 1 makes `keep` reachable at zero -- got {target_leaf_rate}")
    return target_leaf_rate ** (1.0 / depth)


class DecaySource(nn.Module):
    """The slot's contract. Subclass it, or duck-type it."""

    #: The branch width of each level this source emits a dial for.
    widths: tuple[int, ...] = ()
    #: The head count its dials are sized for.
    num_heads: int = 0

    def forward(self) -> LeafMassDecay:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward() -> LeafMassDecay")


class ConstantDecay(DecaySource):
    """A rate the user picked, held fixed.

    No parameters and no gradient: the dials are buffers, so ``.to()``/``.half()`` move
    them and an optimizer never sees them. This is the performance lever -- a high
    constant rate makes the state forget fast, which shows up in the realized
    routing rather than being declared.
    """

    def __init__(self, target_leaf_rate: float, *, widths: tuple[int, ...], num_heads: int) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.num_heads = int(num_heads)
        rate = _symmetric_rate(float(target_leaf_rate), len(self.widths))
        for index, width in enumerate(self.widths):
            self.register_buffer(
                f"dial_{index}", torch.full((self.num_heads, width), rate, dtype=torch.float32))

    def forward(self) -> LeafMassDecay:
        return LeafMassDecay(dials=tuple(
            getattr(self, f"dial_{index}") for index in range(len(self.widths))))


class LearnedDecay(DecaySource):
    """Rates learned through a sigmoid, at one of two resolutions.

    ``scope='state'`` learns one rate per head per digit (the finest the recurrence can
    use), ``'global'`` one per head. Every scope emits the same ``[H, width_l]`` dials --
    broadcasting is where the coarser scope differs, not the contract -- so the
    recurrence is identical across them and the choice is purely how many degrees of
    freedom the model gets.

    These are the two IDENTIFIABLE resolutions: under ``keep = (1 - prod_l delta_l)**c``
    (one write-conditioned clock per leaf), a rate only ever enters
    the recurrence through that per-leaf product, so per-digit (``'state'``) and
    per-head (``'global'``) are the two degrees of freedom the product structure lets
    the recurrence actually distinguish.

    The dials are held in fp32 regardless of the module's dtype: the kernel reads an
    fp32 rate, and a dial rounded through bf16 would quantize the forgetting schedule
    to a handful of distinct rates.
    """

    _SCOPES = ("state", "global")

    def __init__(
        self,
        target_leaf_rate: float = 2.0 ** -8,
        *,
        widths: tuple[int, ...],
        num_heads: int,
        scope: str = "state",
    ) -> None:
        super().__init__()
        if scope not in self._SCOPES:
            raise ValueError(f"scope must be one of {self._SCOPES}, got {scope!r}")
        self.widths = tuple(widths)
        self.num_heads = int(num_heads)
        self.scope = scope
        rate = _symmetric_rate(float(target_leaf_rate), len(self.widths))
        theta = math.log(rate) - math.log1p(-rate)
        if scope == "global":
            shapes = [(self.num_heads, 1)]
        else:
            shapes = [(self.num_heads, width) for width in self.widths]
        self.theta = nn.ParameterList([
            nn.Parameter(torch.full(shape, theta, dtype=torch.float32)) for shape in shapes])

    def _apply(self, fn, recurse=True):
        """Move the dials without letting a module-wide cast reduce their precision."""
        theta = self._modules.get("theta")
        if theta is None:
            return super()._apply(fn, recurse=recurse)
        del self._modules["theta"]
        try:
            result = super()._apply(fn, recurse=recurse)
            with torch.no_grad():
                for param in theta:
                    param.data = fn(param.data).float()
                    if param.grad is not None:
                        param.grad = fn(param.grad).float()
            return result
        finally:
            self._modules["theta"] = theta

    def forward(self) -> LeafMassDecay:
        dials = []
        for index, width in enumerate(self.widths):
            theta = self.theta[0] if self.scope == "global" else self.theta[index]
            dials.append(torch.sigmoid(theta).expand(self.num_heads, width))
        return LeafMassDecay(dials=tuple(dials))


def validate_decay_source(source, topology: Topology, num_heads: int) -> None:
    """Check a decay source against the model it will drive, at construction.

    The shape mismatch this catches -- a source built for other widths, or for another
    head count -- is otherwise discovered by the kernel's own panel check on the first
    forward, several frames away from the line that chose the source.
    """
    widths = getattr(source, "widths", None)
    if tuple(widths or ()) != topology.widths:
        raise ValueError(
            f"{type(source).__name__} emits dials for widths {tuple(widths or ())}, but this "
            f"model's routing levels are {topology.widths}. A decay source is sized to the "
            "topology it drives; build it from `routes.widths`.")
    if int(getattr(source, "num_heads", 0)) != num_heads:
        raise ValueError(
            f"{type(source).__name__} emits dials for {getattr(source, 'num_heads', 0)} heads "
            f"against this model's {num_heads}")
