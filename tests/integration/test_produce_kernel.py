"""The fused producer's kernel (csrc/rola/src/produce/), at the op, one gate a landed component.

The skeleton's claim is the composition's SHAPE -- every element of every output is written, whatever the routing,
and the arrival counters are back at zero after the call. The projection's is its logits: every job's columns for
every token, against the fp64 product of the same bf16 operands, per slot within the envelope
(docs/internals/produce/ampere/produce_kernel.md#accumulate). The solve's is its levels: every route job's sides against the fp64
reference solve on the kernel's own logits (docs/internals/produce/ampere/produce_kernel.md#solve). The bundle's is the routes the
fp64 composition makes from the same bf16 operands: every row's digits, every scale, the read mass and the write gain
(docs/internals/produce/ampere/produce_kernel.md#bundle). The values': every channel's f16 unit and every scale, against the same
(docs/internals/produce/ampere/produce_kernel.md#value-rows).
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.ops.produce import (
    ProducerPlan,
    ProducerWeights,
    _job_tensor,
    _write_gain,
    dense_levels,
    hidden_contiguous,
    job_table,
    produce,
    route_logits,
    row_layout,
)
from rola.routing import quant
from rola.routing.entmax.production import _torch_activation, _torch_union_split

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is a CUDA kernel")

#: EVERY SIDE HAS A DENSE LEVEL (docs/api.md section 2.1), so a topology of entmax levels declares a softmax one: the
#: union and sparsemax rows begin with a dense level, the split rows alternate which duty is sparse.
_ROUTINGS = {
    "union": [rola.dense_routing(16), rola.union_routing(64), rola.union_routing(32)],
    "split": [rola.split_routing(64, sparse_duty="write"), rola.split_routing(64, sparse_duty="read")],
    "tied-deep": [rola.tied_routing(16), rola.dense_routing(32), rola.union_routing(16)],
    "wide": [rola.split_routing(256, sparse_duty="write"), rola.split_routing(200, sparse_duty="read")],
    #: unions past 128 columns: the pair form's rows, a warp pair a token group
    "wide-union": [rola.dense_routing(16), rola.union_routing(256), rola.union_routing(160)],
    "sparsemax": [rola.dense_routing(16), rola.union_routing(64, 2.0), rola.tied_routing(32, rola.entmax(2.0))],
}
_MAX_COLUMNS = 512


def _projection_rtol(hidden: int) -> float:
    """The projection's per-slot bound, relative to the envelope sum |x| |W| (+ |bias|): the bf16 products are exact
    in fp32, so the error is the fp32 accumulation's -- ``hidden + 1`` adds of at most one unit in the last place
    each, in any order, truncating or not. Measured: at most 2^-22.5 of the envelope at hidden 256, 2^-20.7 at 1024."""
    return (hidden + 1) * 2.0 ** -23


def _operands(name: str, *, B=2, T=5, hidden=48, H=3, d_v=64, bias=True, gain=True):
    torch.manual_seed(0)
    plan = ProducerPlan(routing=rola.Routing(_ROUTINGS[name], gain=gain), num_heads=H, hidden_size=hidden, d_v=d_v)
    C = plan.routing.columns
    route_W = hidden_contiguous(torch.randn(H, hidden, C, device="cuda").to(torch.bfloat16))
    route_bias = torch.randn(H, C, device="cuda") if bias else None
    gain_bias = torch.zeros(H, plan.routing.gain_columns, device="cuda")
    v_W = torch.randn(H * d_v, hidden, device="cuda").to(torch.bfloat16)
    x = torch.randn(B, T, hidden, device="cuda").to(torch.bfloat16)
    return plan, x, route_W, route_bias, gain_bias, v_W


def _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill: int, probe: bool = False):
    """One launch into buffers poisoned with ``fill``; the outputs and the probes after it."""
    from rola.ops._ext import extension

    B, T, hidden = x.shape
    H, D, d_v = plan.num_heads, plan.routing.topology.D, plan.d_v
    dense = {side: dense_levels(plan, side) for side in ("read", "write")}

    def poisoned(*shape, dtype=torch.float32):
        out = torch.empty(B, H, T, *shape, dtype=dtype, device="cuda")
        out.view(torch.uint8).fill_(fill)
        return out

    out = {"amp_read": poisoned(row_layout(plan, "read")[1], dtype=torch.uint8),
           "amp_write": poisoned(row_layout(plan, "write")[1], dtype=torch.uint8),
           **{f"dense_{side}": poisoned(row_layout(plan, side, dense=True)[1], dtype=quant.DENSE_DTYPE)
              if dense[side] else None for side in ("read", "write")},
           "scale_read": poisoned(D), "scale_write": poisoned(D), "v": poisoned(d_v, dtype=torch.float16),
           "v_scale": poisoned()}
    jobs, shapes = _job_tensor(plan, "cuda:0")
    probes = [torch.full((jobs.shape[0], B * T, _MAX_COLUMNS), float("nan"), device="cuda") if probe else None
              for _ in range(2)]
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).contiguous(), route_W, route_bias, gain_bias, v_W, jobs,
        *out.values(), *probes, shapes, D,
        *(sum(1 << level for level in dense[side]) for side in ("read", "write")))
    torch.cuda.synchronize()
    return {key: None if t is None else t.transpose(1, 2) for key, t in out.items()}, probes


def _expected_logits(plan, x, route_W, route_bias, v_W):
    """Per job, ``(product, envelope)`` over its columns: fp64 sums of the bf16 operands, and of their magnitudes."""
    xs = x.reshape(-1, x.shape[-1]).double()
    out = []
    for kind, head, heads, _solve, width, in0, in1, *_ in job_table(plan):
        if kind == 1:
            W = v_W[head * plan.d_v:(head + heads) * plan.d_v].double().t()
            bias = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
        else:
            cols = list(range(in0, in0 + width)) + (list(range(in1, in1 + width)) if in1 >= 0 else [])
            W = route_W[head][:, cols].double()
            bias = (route_bias[head, cols].double() if route_bias is not None
                    else torch.zeros(len(cols), dtype=torch.float64, device=W.device))
        out.append((xs @ W + bias, xs.abs() @ W.abs() + bias.abs()))
    return out


@pytest.mark.parametrize("name", list(_ROUTINGS))
def test_every_output_is_written(name):
    operands = _operands(name)
    first, _ = _launch(*operands, fill=0x55)
    second, _ = _launch(*operands, fill=0xAA)
    for key in (key for key in first if first[key] is not None):
        assert torch.equal(first[key].contiguous().view(torch.uint8), second[key].contiguous().view(torch.uint8)), (
            f"{name}: `{key}` holds a byte the kernel did not write")


@pytest.mark.parametrize(("name", "B", "T", "hidden", "bias"), [
    ("union", 2, 5, 48, True),      # a partial k-block, one live token group
    ("split", 1, 71, 256, True),    # three token tiles, the last with one live group
    ("tied-deep", 3, 7, 64, False), # sliced jobs (16 columns: eight warps across the hidden axis), no bias
    ("wide", 1, 1, 256, True),      # the decode shape: one token, jobs of 512 and 200 columns
    ("wide", 4, 40, 1024, True),    # a wide hidden axis: 32 k-blocks, two in flight
])
def test_the_projection_matches_the_fp32_product(name, B, T, hidden, bias):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=hidden, bias=bias)
    _, (logits, _) = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0, probe=True)
    for index, (product, envelope) in enumerate(_expected_logits(plan, x, route_W, route_bias, v_W)):
        got = logits[index, :, :product.shape[1]].double()
        excess = (got - product).abs() - _projection_rtol(hidden) * envelope
        assert excess.max() <= 0, (
            f"{name}: job {index}'s logit (token, column) {divmod(int(excess.argmax()), product.shape[1])} is off by "
            f"{float((got - product).abs().flatten()[excess.argmax()]):.3g} against an envelope of "
            f"{float(envelope.flatten()[excess.argmax()]):.3g}")


#: the job table's solve codes (produce.cuh `SolveKind`) as the reference's activations
_ACTIVATIONS = {0: rola.softmax(), 1: rola.entmax(1.5), 2: rola.entmax(2.0)}
_UNION_ALPHA = {3: 1.5, 4: 2.0}


def _solve_rtol(width: int) -> float:
    """The solve's per-slot bound on a level (every level is at most 1): the fp32 threshold's prefix sums over at most
    ``width`` sorted members of the shifted row, and the exponentials of softmax and the union's write side, each a few
    units in the last place a member. Measured below it by the factor the test prints on failure."""
    return (width + 16) * 2.0 ** -23


def _solved(row, solve, width):
    """``(read, write)`` the reference split of one job's logit row, fp64 ``[tokens, width]``."""
    if solve in _UNION_ALPHA:
        return _torch_union_split(row[:, :width], row[:, width:2 * width], _UNION_ALPHA[solve])
    level = _torch_activation(row[:, :width], _ACTIVATIONS[solve])
    return level, level


def _conditioning(row, solve, width):
    """THE CONDITIONING TERM J.eps of a solve (2026-10-10): the fp64 reference's response to the rounding of its own
    inputs. ``J`` is the derivative of each output entry (and of each level's max, the scale) with respect to each logit,
    one fp64 central difference a logit, and ``eps_c = 2^-24 |logit_c|`` the half-ulp of the fp32 logit; the term is
    ``sum_c |J_c| eps_c`` a row. A union split's threshold couples the duties, so a row of three live members amplifies a
    rounding by 20 to 130 (a well-conditioned row has a term under 1e-7). Returns ``{side: (entries, peak)}`` of
    ``[tokens, width]`` and ``[tokens]``, fp64."""
    columns = 2 * width if solve in _UNION_ALPHA else width
    step = 1e-5
    entry = [torch.zeros_like(row[:, :width]) for _ in range(2)]
    peak = [torch.zeros_like(row[:, 0]) for _ in range(2)]
    for column in range(columns):
        rounding = row[:, column].abs() * 2.0 ** -24
        sides = []
        for sign in (1.0, -1.0):
            moved = row[:, :columns].clone()
            moved[:, column] += sign * step
            sides.append(_solved(moved, solve, width))
        for side in range(2):
            entry[side] += ((sides[0][side] - sides[1][side]).abs() / (2 * step)) * rounding[:, None]
            peak[side] += ((sides[0][side].amax(-1) - sides[1][side].amax(-1)).abs() / (2 * step)) * rounding
    return {"read": (entry[0], peak[0]), "write": (entry[1], peak[1])}


@pytest.mark.parametrize(("name", "B", "T"), [("union", 2, 5), ("split", 1, 40), ("tied-deep", 3, 7),
                                              ("wide", 1, 33), ("sparsemax", 2, 9), ("wide-union", 1, 33)])
def test_the_solve_matches_the_fp64_reference_on_the_kernels_logits(name, B, T):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=64)
    _, (logits, levels) = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0, probe=True)
    for index, (kind, _head, _heads, solve, width, _in0, in1, out_read, out_write, *_rest) in enumerate(
            job_table(plan)):
        if kind != 0:
            continue
        row = logits[index].double()
        read, write = _solved(row, solve, width)
        conditioning = _conditioning(row, solve, width)
        for side, expected, out in (("read", read, out_read), ("write", write, out_write)):
            if out < 0:
                continue
            got = levels[index, :, (0 if side == "read" else width):][:, :width].double()
            #: the solve's bound plus the conditioning term J.eps of its own inputs' rounding (`_conditioning`, 2026-10-10)
            excess = (got - expected).abs() - _solve_rtol(width) - conditioning[side][0]
            assert float(excess.max()) <= 0, (
                f"{name}: job {index} (solve {solve}, width {width}) {side} side is off by "
                f"{float((got - expected).abs().max()):.3g} past the solve's {_solve_rtol(width):.3g} and its "
                f"conditioning term")


#: the bundle's per-slot bounds. THE ROWS AND SCALES ARE GRADED AT THE SEAM: against the reference split solved on the
#: KERNEL'S OWN logits, as the row-level solve test does, because the logits differ from the fp64 product by the fp32
#: accumulation (the projection test bounds that) and a union split amplifies a logit difference by its conditioning
#: (measured: 19 on a row of three live members, a 3.9e-6 logit difference moving a level by 5.7e-5). A u8 digit is a
#: level within the solve's bound of the reference's, rounded: the same digit or its neighbour. A scale is a row's max:
#: twice the solve's bound. Write level 0's scale carries the write gain (docs/CONSTITUTION.md#r23), the gain its own
#: projection, graded against the fp64 composition's within `_GAIN_RTOL` relative.
_GAIN_RTOL = 2.0 ** -12


@pytest.mark.parametrize(("name", "B", "T", "bias", "gain"), [
    ("union", 2, 5, True, True), ("split", 1, 40, True, True), ("tied-deep", 3, 7, False, True),
    ("wide", 1, 33, True, True), ("sparsemax", 2, 9, True, False),   # no gain column: the gain is 1
    ("wide-union", 1, 33, True, True),
])
def test_the_bundle_matches_the_reference_on_the_kernels_logits(name, B, T, bias, gain):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=64, bias=bias, gain=gain)
    gain_bias = torch.randn_like(gain_bias)
    out, (logits, _) = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0x55, probe=True)
    dense = {side: dense_levels(plan, side) for side in ("read", "write")}
    reference = ProducerWeights(route_W=route_W.double(), gain_bias=gain_bias.double(), v_W=v_W.double(),
                                route_bias=None if route_bias is None else route_bias.double())
    #: THE fp64 COMPOSITION'S WRITE GAIN, through its own pieces (the gain logit, then `softplus`)
    gain = _write_gain(route_logits(x.double(), plan, reference)[2], reference.gain_bias, B=B, T=T, H=plan.num_heads,
                       dtype=torch.float64, device=x.device)

    for index, (kind, head, _heads, solve, width, _in0, _in1, out_read, out_write, level) in enumerate(
            job_table(plan)):
        if kind != 0:
            continue
        levels = dict(zip(("read", "write"), _solved(logits[index].double(), solve, width), strict=True))
        conditioning = _conditioning(logits[index].double(), solve, width)
        for side, written in (("read", out_read), ("write", out_write)):
            if written < 0:
                continue
            expected = levels[side]
            peak = expected.amax(-1, keepdim=True)
            unit = expected / peak
            scale = out[f"scale_{side}"][:, :, head, level].double().reshape(-1, 1)
            #: twice the solve's bound plus the conditioning term of the level's max (`_conditioning`, 2026-10-10)
            bound = (width + 16) * 2.0 ** -22 + conditioning[side][1][:, None]
            if side == "write" and level == 0:
                #: the gain folded in: `g` times the max's bound, plus the max times the gain's own
                g = gain[:, :, head].reshape(-1, 1)
                peak, bound = g * peak, g * bound + g * peak * _GAIN_RTOL
            assert bool(((scale - peak).abs() <= bound).all()), (
                f"{name}: level {level} {side} scale off by {float((scale - peak).abs().max()):.3g}")
            if level in dense[side]:
                #: a dense level's f16 rows: one u8 digit's worth (the solves' fp32 against fp64), the floor exact
                start = row_layout(plan, side, dense=True)[0][level]
                got = out[f"dense_{side}"][:, :, head, start:start + width].reshape(-1, width)
                assert got.dtype is quant.DENSE_DTYPE
                off = float((got.double() - unit.clamp_min(quant.DENSE_FLOOR)).abs().max())
                assert off <= 1 / quant.AMP_LEVELS, f"{name}: level {level} {side} f16 rows off by {off:.3g}"
                assert bool((got.float() >= quant.DENSE_FLOOR).all()), f"{name}: a dense {side} coefficient under the floor"
            else:
                offset = row_layout(plan, side)[0][level]
                got = out[f"amp_{side}"][:, :, head, offset:offset + width].reshape(-1, width)
                digit = int((got.int() - torch.round(unit * quant.AMP_LEVELS).int()).abs().max())
                assert digit <= 1, f"{name}: level {level} {side} digits off by {digit}"


@pytest.mark.parametrize(("name", "B", "T"), [("union", 2, 5), ("wide", 1, 33), ("split", 1, 40)])
def test_the_values_match_the_fp64_composition(name, B, T):
    """Per slot, one f16 unit in the last place (at the larger of the two units) plus the projection's fp32 error,
    ``(hidden + 1) 2^-23`` of the envelope sum |x| |W|, the channel's and the scale's, over the token's scale."""
    hidden, H, d_v = 64, 3, 64
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=hidden, H=H, d_v=d_v)
    out, _ = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0x55)
    reference = ProducerWeights(route_W=route_W.double(), gain_bias=gain_bias.double(), v_W=v_W.double(),
                                route_bias=None if route_bias is None else route_bias.double())
    _, values = produce(x.double(), plan, reference)

    envelope = (x.double().abs() @ v_W.double().abs().t()).reshape(B, T, H, d_v)
    error = (hidden + 1) * 2.0 ** -23 * envelope
    scale = values.scale.double()[..., None]
    got, want = out["v"].double(), values.v.double()
    top = torch.maximum(got.abs(), want.abs()).clamp_min(2.0 ** -14)
    ulp = torch.exp2(torch.floor(torch.log2(top)) - 10)
    excess = (got - want).abs() - (ulp + (error + error.amax(-1, keepdim=True)) / scale)
    assert excess.max() <= 0, f"{name}: a value off by {float(excess.max()):.3g} past its bound"
    scale_excess = (out["v_scale"].double() - values.scale.double()).abs() - error.amax(-1)
    assert scale_excess.max() <= 0, f"{name}: a value scale off by {float(scale_excess.max()):.3g} past its bound"


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/produce/ampere/produce_kernel.md#phase-ledger): every warp of every job books, a route job
    every route phase and a value job its own, and the outputs are the same bytes with it bound or not."""
    from rola.ops import produce as producer

    operands = _operands("union", B=2, T=40, hidden=64)
    plain, _ = _launch(*operands, fill=0x55)
    jobs = job_table(operands[0])
    ledger = torch.zeros(len(jobs), producer.LEDGER_WARPS, len(producer.LEDGER_PHASES), dtype=torch.int64,
                         device="cuda")
    producer.ledger_bind(ledger)
    try:
        booked, _ = _launch(*operands, fill=0x55)
    finally:
        producer.ledger_bind(None)
    for key in (key for key in plain if plain[key] is not None):
        same = torch.equal(plain[key].contiguous().view(torch.uint8), booked[key].contiguous().view(torch.uint8))
        assert same, f"binding moved `{key}`"
    phases = dict(zip(producer.LEDGER_PHASES, range(len(producer.LEDGER_PHASES)), strict=True))
    route = [name for name in producer.LEDGER_PHASES if name != "values"]
    for index, job in enumerate(jobs):
        #: a softmax solve (the dense level's job, solve 0) has no threshold to find
        names = ["job", "project", "values"] if job[0] else [
            name for name in route if name != "threshold" or job[3] != 0]
        for name in names:
            assert bool((ledger[index, :, phases[name]] > 0).all()), f"job {index}: a warp booked no `{name}`"
