# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGIME AXES: where in the routing distribution a carry cell's draw sits, declared and proven.

Routing is a moving distribution -- the gain knob alone sweeps the candidate fraction from about 0.96 to 0.09 in
training -- so a cell set drawn at one density is an artifact. Every carry cell therefore states a value on each axis
below (`carry_cells.json`'s `regime`), and `realize` PROVES the draw sits there before any reader sees it: a cell that
cannot demonstrate its declared regime refuses to realize, so the oracle and the benches can never run a cell that is
somewhere other than where it says. Independent per-token sampling manufactures anti-structure (sides that never
coincide, supports that never persist), which is why the coherent, tied, anti and cold corners are cells of their own
rather than something a random draw is hoped to reach.

Every check reads the per-level amplitudes and never a leaf product, so proving a 65536-token, 65536-leaf cell costs a
pass over its `L x sum_l B_l` digits. Symbols: ``L`` tokens, ``B_l`` level ``l``'s digit count, ``W`` the carry
kernel's fixed window.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: Every axis, its values, and why it is an axis (a bug class or a fact about realized routing).
REGIME_AXES: Mapping[str, tuple[str, ...]] = {
    #: exact zeros anywhere, or none: the dense fast path must be the same function as the sparse one.
    "density": ("dense", "sparse"),
    #: whether consecutive tokens route alike. Realized routing is coherent over spans, and coherence is what keeps
    #: whole windows of an owner hot or cold.
    "coherence": ("iid", "coherent"),
    #: the two sides' supports: bitwise the same (a read/write index swap is invisible), overlapping, or disjoint at
    #: every token (the swap is maximally visible).
    "rw_correlation": ("tied", "independent", "anti"),
    #: how a row's amplitude is spread; `cold_read` reads leaves no token of the sequence writes, against an entry
    #: state that makes those reads observable.
    "mass": ("spread", "concentrated", "cold_read"),
    #: `L` a whole number of windows, or a partial last window.
    "tail": ("divisible", "ragged"),
    #: a token's live digits per level: exactly one on some side at every level, some, or all everywhere.
    "support": ("singleton", "partial", "full"),
}

#: The measured bands, with the gap between each pair on purpose: a draw between them demonstrates neither value.
#: MEASURED 2026-09-14 over the registry: iid draws read 0.99-1.06, cohort draws' SUPPORT reads far below 0.5 while
#: their values read 0.50-0.54; spread draws' width-normalized concentration reads 0.004-0.37.
IID_ABOVE, COHERENT_BELOW = 0.7, 0.5
SPREAD_AT_MOST, CONCENTRATED_AT_LEAST = 0.65, 0.75


@dataclass(frozen=True)
class View:
    """What the checks read: the cell, its two sides' levels as fp64 (the amplitudes the kernel reads), and its entry
    state (None when the cell binds none)."""

    spec: object
    read: tuple
    write: tuple
    entry: object = None

    @classmethod
    def of(cls, drawn) -> View:
        read, write, _, _ = drawn.doubles()
        return cls(drawn.spec, read, write, drawn.entry)


def _sides(view):
    return (("read", view.read), ("write", view.write))


def _change_ratio(planes, constant: float | None) -> float | None:
    """Mean change between consecutive tokens over the mean change between shuffled ones: about 1 when tokens are
    drawn independently, far below it when they persist; ``constant`` when nothing varies at all."""
    import torch

    adjacent = shuffled = 0.0
    for plane in planes:
        perm = torch.randperm(plane.shape[1], generator=torch.Generator().manual_seed(7)).to(plane.device)
        adjacent += float((plane[:, 1:] - plane[:, :-1]).abs().mean())
        shuffled += float((plane - plane[:, perm]).abs().mean())
    return adjacent / shuffled if shuffled > 0 else constant


def coherence(view) -> dict[str, float | None]:
    """Per side, the change ratio of the amplitudes and of the support (which digits are live). A side whose
    amplitudes never change persists perfectly (0); a support that never changes says nothing (None), because a full
    support never changes either."""
    out = {}
    for side, levels in _sides(view):
        out[f"{side} values"] = _change_ratio(levels, constant=0.0)
        out[f"{side} support"] = _change_ratio([(level != 0).double() for level in levels], constant=None)
    return out


def concentration(view) -> float:
    """The mean width-normalized largest share of a row, over both sides and every level: 0 for a uniform row, 1 for
    a one-hot row, comparable across widths."""
    import torch

    shares = []
    for _, levels in _sides(view):
        for level in levels:
            width = level.shape[-1]
            shares.append((level.max(dim=-1).values.reshape(-1) - 1.0 / width) / (1.0 - 1.0 / width))
    return float(torch.cat(shares).mean())


def _token_overlap(view):
    """``[B, L, H]``: whether the token reads some leaf it also writes (every level shares a live digit)."""
    import torch

    (_, read), (_, write) = _sides(view)
    return torch.stack([((r != 0) & (w != 0)).any(dim=-1) for r, w in zip(read, write)]).all(dim=0)


def check(axis: str, value: str, view) -> str | None:
    """None when the draw demonstrates ``(axis, value)``; otherwise what it shows instead."""
    import torch

    from rola.ops.carry import WINDOW

    sides = _sides(view)
    if axis == "density":
        zeros = any(bool((level == 0).any()) for _, levels in sides for level in levels)
        return None if zeros == (value == "sparse") else f"exact zeros {'absent' if value == 'sparse' else 'present'}"
    if axis == "coherence":
        ratios = coherence(view)
        known = [r for r in ratios.values() if r is not None]
        if value == "coherent":
            return None if any(r < COHERENT_BELOW for r in known) else f"no side persists: {ratios}"
        return None if all(r > IID_ABOVE for r in known) else f"a side persists: {ratios}"
    if axis == "rw_correlation":
        (_, read), (_, write) = sides
        tied = all(torch.equal(r, w) for r, w in zip(read, write))
        overlap = bool(_token_overlap(view).any())
        if value == "tied":
            return None if tied else "the sides differ"
        if value == "independent":
            return None if (not tied and overlap) else ("the sides are tied" if tied else "no token overlaps")
        return None if not overlap else "some token reads a leaf it writes"
    if axis == "mass":
        if value == "cold_read":
            (_, read), (_, write) = sides
            cold = any(bool(((r != 0) & ~(w != 0).any(dim=(0, 1, 2))).any()) for r, w in zip(read, write))
            entry = view.entry
            if not cold:
                return "every read digit is written somewhere in the sequence"
            return None if entry is not None and bool(entry.any()) else "no entry state to read cold leaves from"
        mean = concentration(view)
        if value == "spread":
            return None if mean <= SPREAD_AT_MOST else f"concentration {mean:.2f}"
        return None if mean >= CONCENTRATED_AT_LEAST else f"concentration {mean:.2f}"
    if axis == "tail":
        ragged = view.spec.tokens % WINDOW != 0
        return None if ragged == (value == "ragged") else f"L = {view.spec.tokens} under W = {WINDOW}"
    if axis == "support":
        counts = {side: [(level != 0).sum(dim=-1) for level in levels] for side, levels in sides}
        widths = [level.shape[-1] for level in sides[0][1]]
        if value == "singleton":
            one = any(all(bool((c == 1).all()) for c in cs) for cs in counts.values())
            return None if one else "no side has exactly one live digit on every level"
        if value == "partial":
            some = any(bool(((c > 1) & (c < w)).any()) for cs in counts.values() for c, w in zip(cs, widths))
            return None if some else "every row is one digit or full"
        full = all(bool((c == w).all()) for cs in counts.values() for c, w in zip(cs, widths))
        return None if full else "a row is thinned"
    raise KeyError(f"no regime axis {axis!r}")


def prove(view) -> None:
    """Refuse a draw that does not sit where its cell declares, naming every axis it misses."""
    view = View.of(view)
    missed = [f"{axis}={value} ({why})" for axis, value in view.spec.regime
              if (why := check(axis, value, view)) is not None]
    if missed:
        raise ValueError(f"{view.spec.name}: the draw is not in its declared regime: {'; '.join(missed)}")


__all__ = ["REGIME_AXES", "View", "check", "coherence", "concentration", "prove"]
