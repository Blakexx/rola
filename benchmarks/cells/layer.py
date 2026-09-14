# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAYER CELLS: the registry's whole-layer half, and their committed statistics.

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

:func:`check_manifest` is the standing gate over `layer_manifest.json`. Every realized
statistic there is INTEGER-DERIVED -- support counts and live-atom counts, never a float
reduction -- so a drift in torch's RNG, in the entmax solver or in the producer
announces itself as a diff rather than as a tolerance question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REGISTRY = _HERE / "layer_cells.json"
MANIFEST = _HERE / "layer_manifest.json"

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


def realized(spec: LayerCellSpec, fx: dict | None = None) -> dict:
    """This cell's INTEGER statistics, as the fixture actually realizes them.

    Support counts rather than amplitude sums: a count is bit-exact across runs, so the
    manifest diffs on a routing change and never on a reduction order.
    """
    from rola.engine.facts import planes
    from rola.ops.padding import pad_routes
    from rola.routing.types import padded_level_width

    fx = build(spec) if fx is None else fx
    #: THE STATISTICS ARE TAKEN AT THE KERNEL-FACING WIDTHS. A level narrower than the
    #: descriptor's floor has no state format of its own and no liveness pass will read
    #: it; padding is the one lawful shape it has, and it is the shape the layer itself
    #: now runs. The pad is the identity on a topology already at its widths, so those
    #: cells' counts are untouched by this.
    padded = tuple(padded_level_width(width) for width in spec.widths)
    read = planes.pack_side(pad_routes(fx["routes"].read, padded))
    write = planes.pack_side(pad_routes(fx["routes"].write, padded))
    live = planes.written_atoms(planes.atom_bits(write, padded))
    return {
        "N": prod(padded),
        "padded_widths": list(padded),
        "BH": spec.B * spec.H,
        "read_nonzero": int((read != 0).sum()),
        "write_nonzero": int((write != 0).sum()),
        "amplitudes_total": int(write.numel()),
        "live_atoms": int(live.sum()),
        "atoms_total": int(live.numel()),
    }


def cell_manifest(spec: LayerCellSpec, fx: dict | None = None) -> dict:
    out = {"name": spec.name, "widths": list(spec.widths), "B": spec.B, "T": spec.tokens,
           "H": spec.H, "d_v": spec.dv, "hidden": spec.hidden_size, "alpha": spec.alpha,
           "logit_gain": spec.logit_gain, "dtype": spec.dtype, "seed": spec.seed,
           "decode_steps": spec.decode_steps}
    out.update(realized(spec, fx))
    out["write_support_frac"] = round(out["write_nonzero"] / out["amplitudes_total"], 6)
    out["live_atom_frac"] = round(out["live_atoms"] / out["atoms_total"], 6)
    return out


def build_manifest(names=None) -> dict:
    cells = {c.name: c for c in layer_cells()}
    return {n: cell_manifest(cells[n]) for n in (names or cells)}


def check_manifest(names=None) -> bool:
    """A DRIFTED cell is a cell whose numbers are no longer comparable to the ones
    already published under its name, so this is a hard check and not a report."""
    if not MANIFEST.exists():
        raise SystemExit(f"no committed manifest at {MANIFEST}; record one with "
                         f"`python -c \"from cells.layer import write_manifest; "
                         f"write_manifest()\"`")
    want = json.loads(MANIFEST.read_text())
    got = build_manifest(names)
    bad = []
    for name in sorted(got):
        if want.get(name) != got[name]:
            bad.append(name)
            print(f"  DRIFT {name}\n    committed {want.get(name)}\n    realized  {got[name]}")
    print(f"{len(got) - len(bad)}/{len(got)} layer cells match the committed manifest")
    return not bad


def write_manifest() -> None:
    MANIFEST.write_text(json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n")
