"""Oracle-anchored support gate for the packed-router projection GEMM.

The projection-invariant gate
found that scoping ``torch.backends.cuda.matmul.allow_tf32`` around the
projection einsum (effd23c, reverted) flips 14-95 of the alpha=1.5 entmax
support entries per cell on fresh seeds, against the SAME fp32 code with the
flag off -- and every flip is a NEW divergence from the fp64 oracle's support
that the fp32-without-TF32 path did not have. No existing test caught this:
the tier-1/layer oracle gates feed both the fp64 and CUDA backends from the
SAME fp32 producer, so a projection error is common-mode with the thing it is
checked against and cancels out. This test breaks that common mode on
purpose -- it recomputes the routing logits independently in fp64
(``packed_router_logits_reference``, a different arithmetic domain the TF32
flag cannot reach) and compares SUPPORT MEMBERSHIP, not values, with
``rtol=0``: the product invariant is that a token's leaf is either live or
not, exactly, not "close".

Fixtures are Gate 2's probe fixtures (seeds 5150/6161/7272, parity topology:
hidden=256, H=4, d_v=64, branches=(16,16), read=softmax dense, write=entmax
alpha=1.5), not the projection/entmax authors' own fixtures, by the same
"independent fixtures" discipline Gate 2 itself used. L=2048 alone already
showed 14/18/26 flips under effd23c and is kept small to hold this to a unit
test's budget; the full L=2048/8192 sweep lives in Gate 2's findings doc.

Any future precision change to `_packed_router_logits`'s fp32 GEMM (TF32,
reduced-precision accumulation, bf16x3, ...) must keep this test green, i.e.
must not move the production support set away from the fp64 oracle's.
"""

import pytest
import torch
from conftest import build_layer, resolved_routing_from_topology

from rola import entmax, softmax
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.projection import packed_router_logits, packed_router_logits_reference
from rola.routing.types import IndependentRouting

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_SEEDS = (5150, 6161, 7272)
_LENGTH = 2048
_HIDDEN = 256
_HEADS = 4
_HEAD_V_DIM = 64
_BRANCHES = (16, 16)


def _support(read_logits, write_logits, routing):
    reads, writes = production_routing_factor_levels(read_logits, write_logits, routing)
    return [factor > 0 for factor in reads], [factor > 0 for factor in writes]


@pytest.mark.cuda
@pytest.mark.parametrize("seed", _SEEDS)
def test_packed_router_logits_matches_fp64_oracle_support(seed):
    """The production fp32 projection's entmax support must equal the fp64
    oracle projection's support EXACTLY -- zero flips, not a tolerance.

    This is the gate effd23c shipped without: `_assert_matches_oracle`-style
    tier-1 checks compare the CUDA and fp64 backends fed by the SAME fp32
    producer logits, so a projection-level precision change never reaches
    them. Recomputing the oracle logits here, from the same `h`/`route_W`,
    through the fp64 projection path (`packed_router_logits_reference`) is
    what makes this non-common-mode with the producer under test.
    """
    torch.manual_seed(seed)
    # Old: `level_routing_configs=LevelRoutingConfig(read="dense", write="sparse",
    # alpha=1.5)` -- a mixed per-side tag, so neither `dense_routing()`
    # nor `union_routing()` (the symmetric convenience constructors) applies; built
    # directly as the `IndependentRouting` the old mapping produced.
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_HEADS, d_v=_HEAD_V_DIM, widths=_BRANCHES,
        routing=IndependentRouting(width=1, read=softmax(), write=entmax(1.5)),
        decay=None,
    ).to("cuda", torch.float32).eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    producer = layer.routes
    routing = resolved_routing_from_topology(layer.topology)
    h = torch.randn(1, _LENGTH, _HIDDEN, device="cuda", dtype=torch.float32)

    #: THE ROUTING COLUMNS ALONE. `route_W`'s trailing span is the side gain's, and the
    #: support this gate counts flips in is a threshold on routing logits only.
    route_W = producer.route_W[:, :, :routing.packed_router_width]
    route_bias = (None if producer.route_bias is None
                  else producer.route_bias[:, :routing.packed_router_width])
    read_logits, write_logits = packed_router_logits(
        h, route_W, routing, route_bias=route_bias)
    read64, write64 = packed_router_logits_reference(
        h.double(), route_W.double(), routing,
        route_bias=None if route_bias is None else route_bias.double())

    production_support = _support(read_logits, write_logits, routing)
    oracle_support = _support(read64.float(), write64.float(), routing)

    total = 0
    flips = 0
    for side_production, side_oracle in zip(production_support, oracle_support):
        for factor_production, factor_oracle in zip(side_production, side_oracle):
            total += factor_production.numel()
            flips += (factor_production ^ factor_oracle).sum().item()

    assert total > 0
    assert flips == 0, (
        f"seed={seed}: {flips}/{total} production-vs-fp64-oracle entmax support "
        "flips -- the projection's fp32 GEMM must produce the SAME support set as "
        "the fp64 oracle, exactly (see gate2-invariants.md item 2)")
