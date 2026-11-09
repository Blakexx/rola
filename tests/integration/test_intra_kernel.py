"""Gate for the INTRA (csrc/rola/src/intra/): the within-window pairs of the bundle against the fp64 intra on the same
stored operands, per slot.

The reference is `tests/oracle/int_reference.py`'s ``intra_bundle_reference`` -- the bundle's rows, factors and values
as stored, nothing rounded -- itself gated against the fp64 intra of the bundle's own dequantized levels
(`tests/oracle/test_int_reference.py`). The kernel's roundings are declared and bounded per slot
(docs/internals/intra/ampere/intra_kernel.md#numerics): at fp16 accumulate each k-step's sum rounds once to f16 (``U16`` of
the partial sum, which never exceeds the final: every term is non-negative), so a level's Gram is within ``k_l U16``
of its own value over its ``k_l`` k-steps; the value product's weights round to f16 (``U16`` each, or half an f16
subnormal below the reader's power of two) and, at fp16 accumulate, its four k-steps' sums. Each bound is relative
to the slot's ENVELOPE, the same intra on ``|v|`` (``den`` against itself), plus fp32 accumulation.

Symbols: ``L`` tokens, ``H`` query heads, ``H_w`` write heads, ``BH`` batch x query heads, ``W`` the window.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from rola.engine.facts import liveness as lv
from rola.ops import facts as facts_ops
from rola.ops import intra as intra_ops
from rola.ops.window import window_of
from rola.routing.factors import RouteFactors
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values
from tests.oracle import int_reference as iref
from tests.oracle import tolerances
from tests.oracle.fixtures import with_gain

#: the lists' form the facts pass is called in (sm_86's even carry arm); this file reads its unions and factors only
_LISTS = facts_ops.ListsForm(read_row=64, read_list_group=32, write_list_group=32)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the intra is a CUDA kernel")

#: f16's and fp32's unit roundoff
U16, U32 = 2.0 ** -11, 2.0 ** -24
#: the value product's k-steps over a tile's 64 writers
VALUE_STEPS = 4
ACCUMULATE = ("fp32", "fp16")
#: ONE DENSE LEVEL A SIDE AT LEAST (the format's rule): the side's per-level dense flags at depth 2
SIDE_PATTERNS = ((True, False), (False, True), (True, True))


def _bundle(L, H, keep, *, widths=(256, 256), read_dense=(False, True), write_dense=(True, False), Hw=None, B=1,
            seed=0):
    """A bundle in the format at ``widths``: a sparse level's rows u8 with ``keep`` of their digits live, a dense
    level's f16 (the floor taken), gains in [0.1, 1.1), Gaussian values; ``Hw`` write heads (``H`` by default)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hw = H if Hw is None else Hw

    def levels(heads, dense):
        out = []
        for width, is_dense in zip(widths, dense, strict=True):
            x = torch.rand((B, L, heads, width), device="cuda", generator=gen)
            if not is_dense:
                x = x * (torch.rand((B, L, heads, width), device="cuda", generator=gen) < keep)
            out.append(x)
        return out

    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    gain = torch.rand((B, L, Hw), device="cuda", generator=gen) + 0.1
    sparse = (tuple(not d for d in read_dense), tuple(not d for d in write_dense))
    routes = RouteFactors.from_levels(topology, levels(H, read_dense), with_gain(levels(Hw, write_dense), gain), sparse=sparse)
    values = Values.from_projection(torch.randn((B, L, Hw, 64), device="cuda", generator=gen))
    return routes, values


def _clustered(L, H, seed=0):
    """A bundle the slab gate harvests: at level 0 (u8 both sides) a reader in window tile ``j`` holds digits only in
    slab ``j`` and a writer only in slabs ``j`` and ``j + 1``, so a tile pair meets on the diagonal and one below it,
    the second on one slab of eight; level 1 is dense on both sides."""
    routes, values = _bundle(L, H, 0.3, read_dense=(False, True), write_dense=(False, True), seed=seed)
    tile = (torch.arange(L, device="cuda") % window_of()) // 64
    slab = torch.arange(256, device="cuda") // 32

    def keep(rows, slabs):
        return rows * torch.stack([slab == s % 8 for s in slabs]).any(0)[None, :, None, :].to(torch.uint8)

    read = (keep(routes.amp_read[0], [tile[:, None]]), routes.amp_read[1])
    write = (keep(routes.amp_write[0], [tile[:, None], tile[:, None] + 1]), routes.amp_write[1])
    return RouteFactors(topology=routes.topology, amp_read=read, amp_write=write, scale_read=routes.scale_read,
                        scale_write=routes.scale_write, sparse=routes.sparse), values


def _run(routes, values, *, accumulate="fp32", num=None, den=None, gated=True):
    """The facts pass and the intra; ``gated=False`` declares every level dense to the intra, so it reads no unions
    and gates nothing."""
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    facts = facts_ops.facts(routes, values, layout, *statics, _LISTS)
    if not gated:
        statics = tuple(lv.side_statics(layout, (True,) * layout.D, routes.topology.widths) for _ in range(2))
    BH, L = routes.batch * routes.heads, routes.tokens
    num = torch.zeros(BH, L, 64, device="cuda") if num is None else num
    den = torch.zeros(BH, L, device="cuda") if den is None else den
    intra_ops.intra_forward(routes, values, facts, layout, *statics, num, den, accumulate=accumulate)
    return num, den


def _same(a, b):
    """Equal to fp32 reassociation: a block's pieces reduce into one row in either order (a few ulps of the row's
    largest term), where a race or a skipped contribution is off by orders of magnitude."""
    return bool(((a - b).abs() <= 2.0 ** -18 * b.abs().amax() + 1e-12).all())


def _flat(x, routes):
    """``[B, T, H, ...]`` -> ``[BH, T, ...]``, the kernel's planes."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    return x.transpose(1, 2).reshape(B * H, T, *x.shape[3:])


def _bounds(routes, accumulate):
    """``(num, den)``: each plane's relative bound against its envelope (docs/internals/intra/ampere/intra_kernel.md#numerics)."""
    steps = sum(2 * math.ceil(w / 32) for w in routes.topology.widths)
    tiles = window_of() // 64
    fp32 = (tiles + 64) * U32
    if accumulate == "fp16":
        return (steps + 1 + VALUE_STEPS) * U16 + fp32, steps * U16 + fp32
    return U16 + steps * 64 * U32 + fp32, steps * 64 * U32 + fp32


def _record(routes, accumulate, plane, ratio):
    """``pytest --oracle-margins=<file>``: one JSON line a plane -- the worst error over its allowance, read beside the
    oracle comparisons' (`tests/oracle/fixtures.py`'s rows)."""
    if not tolerances.MARGINS_FILE:
        return
    forms = [f"{'f16' if r.dtype == torch.float16 else 'u8'}x{'f16' if w.dtype == torch.float16 else 'u8'}"
             for r, w in zip(routes.amp_read, routes.amp_write)]
    row = {"output": f"intra {plane}", "what": f"{accumulate} widths {list(routes.topology.widths)} forms {forms}",
           "slots": ratio.numel(), "failed": int((ratio > 1).sum()), "worst": float(ratio.max())}
    with open(tolerances.MARGINS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def _check(routes, values, accumulate="fp32"):
    num, den = _run(routes, values, accumulate=accumulate)
    want_num, want_den = iref.intra_bundle_reference(routes, values, window_of())
    envelope, _ = iref.intra_bundle_reference(routes, Values(v=values.v.abs(), scale=values.scale), window_of())
    mass, _ = iref.intra_bundle_reference(routes, Values(v=torch.ones_like(values.v), scale=values.scale), window_of())
    want_num, want_den = _flat(want_num, routes), _flat(want_den, routes)
    envelope, mass = _flat(envelope, routes), _flat(mass, routes)
    num_bound, den_bound = _bounds(routes, accumulate)
    #: a weight more than f16's normal range below its reader's power of two rounds absolutely, within a subnormal of
    #: the tile's largest weight: at most a tile's 64 of them a term of the weights' mass ``sum P s``
    floor = 64.0 * U32 * mass
    num_err = (num.double() - want_num).abs()
    allowed = num_bound * envelope + floor + 1e-30
    _record(routes, accumulate, "num", num_err / allowed)
    assert (num_err <= allowed).all(), (
        f"{int((num_err > allowed).sum())} num slots off; worst {(num_err / allowed).max().item():.3g} of the allowance")
    den_err = (den.double() - want_den).abs()
    allowed = den_bound * want_den + 1e-30
    _record(routes, accumulate, "den", den_err / allowed)
    assert (den_err <= allowed).all(), f"den off by {(den_err / allowed).max().item():.3g} of the allowance"
    return num, den


@pytest.mark.parametrize("accumulate", ACCUMULATE)
@pytest.mark.parametrize(("L", "H", "keep"), [(1024, 2, 0.3), (2048, 1, 1.0), (1536, 3, 0.05)])
def test_the_intra_is_the_bundle_arithmetic_per_slot(L, H, keep, accumulate):
    _check(*_bundle(L, H, keep), accumulate)


@pytest.mark.parametrize("accumulate", ACCUMULATE)
@pytest.mark.parametrize("tail", ["a partial tile alone", "whole tiles", "a partial tile", "a token short"])
def test_a_partial_last_window_is_the_bundle_arithmetic(tail, accumulate):
    """A call whose last window is partial: its pairs up to the call's end, every row past it neither read nor
    written -- under a tile, whole tiles past a window, a partial tile past one, one token short of two windows."""
    W = window_of()
    L = {"a partial tile alone": 37, "whole tiles": W + 64, "a partial tile": W + 100, "a token short": 2 * W - 1}[tail]
    _check(*_bundle(L, 2, 0.3, seed=23), accumulate)


def test_a_partial_last_window_reads_its_write_head():
    """Grouped queries over a partial last window, the union table's last tile partial."""
    _check(*_bundle(window_of() + 161, 4, 0.3, Hw=2, seed=29), "fp16")


@pytest.mark.parametrize("accumulate", ACCUMULATE)
@pytest.mark.parametrize("read_dense", SIDE_PATTERNS)
@pytest.mark.parametrize("write_dense", SIDE_PATTERNS)
def test_every_level_form_is_the_bundle_arithmetic(read_dense, write_dense, accumulate):
    """The nine read/write dense patterns at depth 2: each level's every form -- u8 or f16 on each side."""
    _check(*_bundle(1024, 2, 0.3, read_dense=read_dense, write_dense=write_dense, seed=17), accumulate)


#: THE WIDTHS THE INTRA READS: one or two levels, each a whole number of 16-digit granules -- a stage short of its
#: slabs, a slab half past a level's digits; depth 1 is dense on both sides (the format's one dense level a side)
@pytest.mark.parametrize("accumulate", ACCUMULATE)
@pytest.mark.parametrize("widths", [(256,), (64,), (16,), (48,), (64, 64), (128, 32), (16, 256), (48, 80), (96, 160)])
def test_every_width_is_the_bundle_arithmetic(widths, accumulate):
    dense = ((True,), (True,)) if len(widths) == 1 else ((False, True), (True, False))
    _check(*_bundle(1024, 2, 0.3, widths=widths, read_dense=dense[0], write_dense=dense[1], seed=11), accumulate)


@pytest.mark.parametrize("accumulate", ACCUMULATE)
@pytest.mark.parametrize(("H", "Hw"), [(4, 1), (4, 2), (6, 3)])
def test_grouped_queries_read_their_write_head(H, Hw, accumulate):
    """Each write head's rows, values and factors serve its ``H / H_w`` consecutive query heads."""
    _check(*_bundle(1024, H, 0.3, Hw=Hw, seed=19), accumulate)


@pytest.mark.parametrize("widths", [(64, 64), (48, 80)])
def test_the_slab_gate_is_an_exact_certificate_at_any_width(widths):
    """A sparse draw at narrow and uneven widths: the gated run is the ungated one's sums."""
    routes, values = _bundle(1024, 2, 0.02, widths=widths, read_dense=(False, True), write_dense=(False, True),
                             seed=13)
    for accumulate in ACCUMULATE:
        gated = _run(routes, values, accumulate=accumulate)
        assert all(_same(a, b) for a, b in zip(gated, _run(routes, values, accumulate=accumulate, gated=False),
                                               strict=True))


def test_the_slab_gate_is_an_exact_certificate():
    """Skipped pairs and slabs contribute exact zeros: the gated run is the bundle arithmetic per slot and the
    ungated run's sums (to the pieces' reassociation)."""
    routes, values = _clustered(2048, 2)
    gated = _check(routes, values)
    ungated = _run(routes, values, gated=False)
    assert _same(gated[0], ungated[0]) and _same(gated[1], ungated[1])


@pytest.mark.parametrize("draw", ("random", "clustered"))
def test_the_intra_is_reproducible_across_launches(draw):
    """The pipeline's copies are claimed by whichever warp gets there first, so a race would show as run-to-run
    variation: 40 launches over a grid of many CTAs, causal and gated, all within fp32 reassociation of the first."""
    routes, values = _bundle(4096, 4, 0.3, seed=7) if draw == "random" else _clustered(4096, 4, seed=7)
    for accumulate in ACCUMULATE:
        first = _run(routes, values, accumulate=accumulate)
        for _ in range(19):
            again = _run(routes, values, accumulate=accumulate)
            assert _same(again[0], first[0]) and _same(again[1], first[1])


def test_the_intra_reduces_into_the_planes_it_is_given():
    """A RED, not a store: what the planes held before stays under what the intra adds."""
    routes, values = _bundle(1024, 2, 0.3)
    plain_num, plain_den = _run(routes, values)
    num = torch.full_like(plain_num, 0.5)
    den = torch.full_like(plain_den, 2.0)
    _run(routes, values, num=num, den=den)
    assert _same(num, plain_num + 0.5) and _same(den, plain_den + 2.0)


@pytest.mark.parametrize("widths", [(256, 256), (64, 128)])
def test_the_producers_bundle(widths):
    """A real producer's bundle -- head-major packed planes, the alternating routing's statics -- per slot."""
    import rola

    torch.manual_seed(0)
    routing = rola.Routing([rola.split_routing(widths[0], sparse_duty="write"),
                            rola.split_routing(widths[1], sparse_duty="read")])
    layer = rola.RoLA(routing, hidden_size=128, num_heads=2, d_v=64).cuda().eval()
    with torch.no_grad():
        routes, values = layer._produce(torch.randn(1, 1024, 128, device="cuda"))
    for accumulate in ACCUMULATE:
        _check(routes, values, accumulate)


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    routes, values = _bundle(1024, 2, 0.3)
    plain = _run(routes, values)
    ledger = torch.zeros((intra_ops.LEDGER_WARPS, len(intra_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    intra_ops.ledger_bind(ledger)
    try:
        booked = _run(routes, values)
    finally:
        intra_ops.ledger_bind(None)
    assert _same(plain[0], booked[0]) and _same(plain[1], booked[1])
    assert (ledger > 0).all(), ledger


def test_the_seam_refuses_what_it_does_not_read():
    routes, values = _bundle(1024, 1, 0.3)
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    facts = facts_ops.facts(routes, values, layout, *statics, _LISTS)
    num, den = torch.zeros(1, 1024, 64, device="cuda"), torch.zeros(1, 1024, device="cuda")
    with pytest.raises(KeyError):
        intra_ops.intra_forward(routes, values, facts, layout, *statics, num, den, accumulate="bf16")
    short = RouteFactors(topology=routes.topology, amp_read=tuple(r[:, :1000] for r in routes.amp_read),
                         amp_write=tuple(r[:, :1000] for r in routes.amp_write),
                         scale_read=tuple(s[:, :1000] for s in routes.scale_read),
                         scale_write=tuple(s[:, :1000] for s in routes.scale_write),
                         sparse=routes.sparse)
    #: any token count is served (a partial last window included); operands that disagree on it are not
    with pytest.raises(RuntimeError, match="values is"):
        intra_ops.intra_forward(short, values, facts, layout, *statics, num, den)
    with pytest.raises(RuntimeError, match="past the layout's 32"):
        narrow = lv.LivenessLayout(D=2, B=(32, 256), L=1024)
        intra_ops.intra_forward(routes, values, facts, narrow, *statics, num, den)
    assert not intra_ops.reads((24,)) and not intra_ops.reads((64, 64, 64))
    assert not intra_ops.admits((64,), 128) and intra_ops.admits((48, 80), 64)


def test_the_loaded_binary_states_the_footprint_it_carries():
    """The extension-trap check: the device's own block rows and fixed tenants are the footprint law's -- a reader row
    of every stage's slot, at least three 128-byte stages and room for every warp's deposit staging, past them (that
    the widest rows fit the card is the walk's compile-time assertion, not a runtime read)."""
    stamp = intra_ops.build_stamp() // 1000003
    rows, tenants = stamp // 1000003, stamp % 1000003
    assert rows == intra_ops.block_rows(torch.cuda.current_device())
    staging = -(-intra_ops.LEDGER_WARPS * 16 * 72 * 4 // rows)
    least = -(-max(3 * 128, staging) // 128) * 128
    #: (256, 256) f16 on both sides: four 128-byte stages a level, each a 128-byte reader slot
    assert intra_ops.footprint((256, 256)) == tenants + rows * 8 * 128
    #: u8 read on an f16 write: a stage of 64 digits, its reader slot 64 bytes; f16 read on a u8 write: 256
    from rola.ops._ext import extension
    mixed = extension().intra_footprint([256, 256], [2, 1], torch.cuda.current_device())
    assert mixed == tenants + rows * (4 * 64 + 2 * 256)
    assert intra_ops.footprint((64,)) == tenants + rows * least
