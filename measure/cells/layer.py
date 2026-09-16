# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RoLA'S LAYER CONSTRUCTIONS: what a RoLA layer is built as on a central layer cell.

A central layer cell (`rola_devtools.cells.layer`) is only an input: hidden states and values drawn from a seed. How a
RoLA layer is built to read it -- the routing widths, the entmax alpha (none selects dense softmax routing) and the logit
gain the producer's projection is scaled by -- is RoLA's, so it lives here as a named CONSTRUCTION with the input cells
it was declared for. The names are the layer cells' names before the inputs moved to rola-devtools (each central
layer cell's note names them), so a construction reads as the cell a published number cites.

**A CONSTRUCTION ON A CELL IS DETERMINISTIC.** `build` seeds the parameters from the cell's own seed immediately before
construction and takes the inputs from the cell's draw, which never moves with the construction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Construction:
    name: str
    widths: tuple[int, ...]
    alpha: float | None
    logit_gain: float
    #: the central layer cells this construction is declared for
    cells: tuple[str, ...]
    note: str


def _c(name, widths, alpha, gain, cell, note):
    return Construction(name, tuple(widths), alpha, gain, (cell,), note)


CONSTRUCTIONS = {c.name: c for c in (
    _c("chunk-p73-pinned", (64, 64), 1.5, 1.0, "layer-B2-T2048-H8-h512-dv64-bf16-s0",
       "the inventory constructor every published step number was taken on"),
    _c("chunk-sparse-gain8", (64, 64), 1.5, 8.0, "layer-B2-T512-H8-h512-dv64-bf16-s1", "the pinned constructor at gain 8"),
    _c("chunk-deadblocks-T64", (64, 64), 1.5, 16.0, "layer-B2-T64-H8-h512-dv64-bf16-s5", "dead blocks for the plan to skip"),
    _c("chunk-dense-D2", (64, 64), None, 1.0, "layer-B2-T512-H8-h512-dv64-bf16-s2", "softmax: the dense end"),
    _c("chunk-dense-NL4096", (64, 64), None, 1.0, "layer-B2-T4096-H8-h512-dv64-bf16-s6", "dense at T = N = 4096"),
    _c("chunk-bwd-smalln", (16, 16), 1.5, 1.0, "layer-B2-T2048-H8-h512-dv64-bf16-s7", "N = 256 at T = 2048"),
    _c("chunk-bwd-dense-N64", (8, 8), None, 1.0, "layer-B2-T2048-H8-h512-dv64-bf16-s8", "N = 64, dense: one block"),
    _c("chunk-bwd-splitk", (256, 256), 1.5, 1.0, "layer-B1-T512-H2-h512-dv64-bf16-s9", "N = 65536 at BC = 128"),
    _c("chunk-deep-D4-w8", (8, 8, 8, 8), 1.5, 8.0, "layer-B2-T512-H8-h512-dv64-bf16-s3", "D = 4 at N = 4096"),
    _c("chunk-decode-w16", (16, 16), 1.5, 1.0, "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8", "prefill + 8 carried steps"),
    _c("chunk-decode-la-w256", (256,), None, 1.0, "layer-B2-T128-H2-h128-dv64-fp32-s10-dec8",
       "D = 1 flat dense: the LA control for the routed decode"),
    #: THE PRODUCER SWEEP: one entmax constructor at rising logit gains; the realized density must fall across it
    *(_c(f"producer-sweep-g{gain:g}", (64, 64), 1.5, gain, "layer-B1-T128-H4-h64-dv64-fp32-s201",
         "the producer sweep: the logit gain moves the realized density") for gain in (0.25, 1.0, 4.0, 16.0)),
)}


def build(cell, construction: Construction) -> dict:
    """The fixture: the layer built by `construction` in the cell's dtype, parameters seeded from the cell's seed, and
    the cell's inputs and the routes the layer produces for its prefill.

    IT BUILDS NO CALL PLAN. What a subject times is the launch it names, and the packing a particular op seam does for
    it is that subject's own business.
    """
    import torch
    from rola_devtools.cells.layer import realize

    from rola import RoLA, RouteProducer, dense_routing, union_routing

    if cell.name not in construction.cells:
        raise ValueError(f"{construction.name} is declared for {list(construction.cells)}, not {cell.name}")
    torch.manual_seed(cell.seed)
    template = union_routing(1, construction.alpha) if construction.alpha is not None else dense_routing(1)
    producer = RouteProducer([template.at(w) for w in construction.widths], hidden_size=cell.hidden_size,
                             num_heads=cell.H)
    layer = RoLA(producer, d_v=cell.dv, decay=None, layer_idx=0).to("cuda", cell.torch_dtype).eval()
    if construction.logit_gain != 1.0:
        with torch.no_grad():
            producer.route_W.mul_(construction.logit_gain)
    for param in layer.parameters():
        param.requires_grad_(False)
    drawn = realize(cell)
    x, v = drawn.x, drawn.v
    with torch.no_grad():
        routes = layer.routes(x[:, :cell.tokens])
    return {"cell": cell, "construction": construction, "layer": layer, "producer": layer.routes, "x": x, "v": v,
            "routes": routes, "v_prefill": v[:, :cell.tokens].contiguous(),
            "v_kernel": v[:, :cell.tokens].to(torch.bfloat16).contiguous()}


__all__ = ["CONSTRUCTIONS", "Construction", "build"]
