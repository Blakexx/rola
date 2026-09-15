# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAYER CELLS: the registry's whole-layer half.

A KERNEL cell (`benchmarks/cells/carry_cells.json`) declares a DRAW -- amplitudes put
directly on the simplex -- and is what the carry family's oracle and probe run. A LAYER
cell declares a CONSTRUCTOR: a producer, a routing template and a gain, from which the
amplitudes are PRODUCED. The two answer different questions and neither can stand in for
the other, so both live in this one registry package under one set of names.

**A CELL IS ITS CONSTRUCTOR AND ITS SEED.** Every field of :class:`LayerCellSpec` is a
construction parameter, :func:`build` is deterministic from them, and documents cite a
cell BY NAME and never by a description of its density. Two numbers carrying the same
cell name are commensurable; two carrying different names are not -- which is why a name
here is never reused and never renamed, whatever vocabulary it was minted in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REGISTRY = _HERE / "layer_cells.json"

_DTYPES = {"bf16": "bfloat16", "fp32": "float32"}


@dataclass(frozen=True, slots=True)
class LayerCellSpec:
    """One layer fixture's constructor. Nothing derived is stored."""

    name: str
    widths: tuple[int, ...]
    B: int
    tokens: int
    H: int
    hidden: int
    dv: int
    alpha: float | None
    logit_gain: float
    dtype: str
    seed: int
    decode_steps: int
    note: str

    @property
    def hidden_size(self) -> int:
        return self.hidden or self.H * self.dv

    @property
    def N(self) -> int:
        return prod(self.widths)

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, _DTYPES[self.dtype])


def layer_cell(name: str, **params) -> LayerCellSpec:
    """A layer cell's data provider (`layer_cells.json` names it): the record's constructor, checked."""
    cell = LayerCellSpec(name=name, widths=tuple(params["widths"]), B=params["B"], tokens=params["tokens"],
                         H=params["H"], hidden=params["hidden"], dv=params["dv"], alpha=params["alpha"],
                         logit_gain=params["logit_gain"], dtype=params["dtype"], seed=params["seed"],
                         decode_steps=params["decode_steps"], note=params["note"])
    if cell.dtype not in _DTYPES:
        raise ValueError(f"{cell.name}: dtype {cell.dtype!r} is not one of {sorted(_DTYPES)}")
    return cell


def layer_cells() -> tuple[LayerCellSpec, ...]:
    cells = tuple(layer_cell(**r) for r in json.loads(_REGISTRY.read_text())["cells"])
    names = [c.name for c in cells]
    if len(set(names)) != len(names):
        raise ValueError("two layer cells share a name; a name is a published identity")
    return cells


def by_name(name: str) -> LayerCellSpec:
    for cell in layer_cells():
        if cell.name == name:
            return cell
    raise KeyError(f"no layer cell named {name!r} in {_REGISTRY}")


def build(spec: LayerCellSpec) -> dict:
    """The fixture, built deterministically from this cell alone.

    The module seed and the input seed are separate draws: parameters come from
    `manual_seed(seed)` immediately before construction (the convention every layer bench
    and layer test uses), inputs from an explicit device generator at `seed + 500`, so
    changing `tokens` does not move the parameters.

    IT BUILDS NO CALL PLAN. What a subject times is the launch it names, and the packing
    a particular op seam does for it is that subject's own business; a fixture that also
    ran a call's planner would tie every layer cell to whichever seam happened to be
    buildable, which is exactly how this fixture stopped building at all.
    """
    import torch

    from rola import RoLA, RouteProducer, dense_routing, union_routing

    torch.manual_seed(spec.seed)
    template = (union_routing(1, spec.alpha) if spec.alpha is not None
                else dense_routing(1))
    levels = [template.at(w) for w in spec.widths]
    producer = RouteProducer(levels, hidden_size=spec.hidden_size, num_heads=spec.H)
    layer = RoLA(producer, d_v=spec.dv, decay=None, layer_idx=0)
    layer = layer.to("cuda", spec.torch_dtype).eval()
    if spec.logit_gain != 1.0:
        with torch.no_grad():
            producer.route_W.mul_(spec.logit_gain)
    for param in layer.parameters():
        param.requires_grad_(False)

    gen = torch.Generator(device="cuda").manual_seed(spec.seed + 500)
    total = spec.tokens + spec.decode_steps
    x = torch.randn((spec.B, total, spec.hidden_size), device="cuda",
                    dtype=spec.torch_dtype, generator=gen)
    v = torch.randn((spec.B, total, spec.H, spec.dv), device="cuda",
                    dtype=spec.torch_dtype, generator=gen)
    with torch.no_grad():
        routes = layer.routes(x[:, :spec.tokens])
    return {"cell": spec, "layer": layer, "producer": layer.routes, "x": x, "v": v,
            "routes": routes,
            "v_prefill": v[:, :spec.tokens].contiguous(),
            "v_kernel": v[:, :spec.tokens].to(torch.bfloat16).contiguous()}
