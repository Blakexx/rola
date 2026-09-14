"""What the oracle returns and what it differentiates -- both stated as tests.

`naive_rola` states one gradient fact outright: `c[t,s] = stop_gradient(prod_l
p_write[...])`. The write levels feed BOTH the deposit and the clock, so under decay
the oracle's gradient is DELIBERATELY not the derivative of its own forward -- the
clock term is absent by definition. That is not graph trickery to be cleaned up; it is
the specification, and the reference backward IS autograd through this oracle, so
every backward gate anchored to it inherits the choice.

The contract, in four claims:

1. **THE CLOCK IS NOT DIFFERENTIATED**, and it is a DIFFERENCE claim: a central
   difference on the forward must DISAGREE with autograd under decay, and AGREE
   without it.
2. **THE DETACHMENT IS LOAD-BEARING** -- the write levels do move the forward through
   the clock, so claim 1 is about the gradient and not about an unused path.
3. **EVERYTHING THE RESULT DEPENDS ON IS DIFFERENTIATED**, with a gradient that is
   actually nonzero. A test that only checked `is not None` would pass against a
   manufactured edge no matter what.
4. **BOTH OUTPUTS ARE fp64**, whatever the inputs were. `y` used to be cast back to
   `v.dtype` while the final state stayed fp64 (audit finding F2), so "the fp64
   oracle" handed an fp32 caller two claims about one run in two precisions -- and an
   accuracy gate comparing against it would have been comparing against a reference
   that had itself been rounded. The cast belongs to the caller who knows why.

No device: the oracle is fp64 torch.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
)

_ROUTING = IndependentRouting(
    width=1, read=SoftmaxActivation(), write=SoftmaxActivation())

WIDTHS = (4, 4)
B, T, H, D_V = 2, 5, 2, 8


def _case(*, decay_on: bool, seed: int = 7):
    """One small differentiable cell; every operand is a distinct leaf tensor."""
    g = torch.Generator().manual_seed(seed)
    topology = Topology(levels=tuple(_ROUTING.at(w) for w in WIDTHS))

    def simplex(width):
        x = torch.rand(B, T, H, width, generator=g, dtype=torch.float64) + 1e-3
        return (x / x.sum(dim=-1, keepdim=True)).requires_grad_(True)

    operands = {
        "v": torch.randn(B, T, H, D_V, generator=g, dtype=torch.float64).requires_grad_(True),
        "g_write": (torch.rand(B, T, H, generator=g, dtype=torch.float64) + 0.5).requires_grad_(True),
    }
    read = tuple(simplex(w) for w in WIDTHS)
    write = tuple(simplex(w) for w in WIDTHS)
    for index, (r, w) in enumerate(zip(read, write)):
        operands[f"read{index}"], operands[f"write{index}"] = r, w
    decay = None
    if decay_on:
        dials = tuple((torch.rand(H, w, generator=g, dtype=torch.float64) * 0.3 + 0.1
                       ).requires_grad_(True) for w in WIDTHS)
        decay = LeafMassDecay(dials=dials)
        for index, dial in enumerate(dials):
            operands[f"dial{index}"] = dial

    def run(write_override=None):
        return naive_rola(
            operands["v"], read, write if write_override is None else write_override,
            operands["g_write"], topology, decay,
            output_final_state=True)

    return operands, write, run


def _grads(operands, y, state):
    keys = sorted(operands)
    grads = torch.autograd.grad(
        outputs=(y, state), inputs=[operands[k] for k in keys],
        grad_outputs=(torch.ones_like(y), torch.ones_like(state)), allow_unused=True)
    return dict(zip(keys, grads))


def _directional(run, write, direction, *, step=1e-5):
    """The forward's OWN directional derivative of `y.sum() + state.sum()` along
    `direction`, by a central difference. Directional rather than per-entry because the
    clock's contribution is spread over every token that has a future to decay."""
    out = []
    for delta in (+step, -step):
        moved = tuple(level.detach() + delta * d for level, d in zip(write, direction))
        y, state = run(write_override=moved)
        out.append(float(y.sum()) + float(state.sum()))
    return (out[0] - out[1]) / (2 * step)


@pytest.mark.parametrize("decay_on", [False, True])
def test_the_clock_is_detached_so_autograd_is_deliberately_not_the_forwards_derivative(
        decay_on):
    """Claim 1, and it is a DIFFERENCE claim, not an agreement one.

    The write levels feed BOTH the deposit and the clock, so the true derivative of
    this forward carries a clock term and the oracle's gradient deliberately does not.
    A central difference is what can see the omission: with decay ON the two must
    DISAGREE, and with decay OFF -- no clock in the recurrence at all -- they must
    agree, which is what proves the disagreement is the clock and not the step size.
    """
    operands, write, run = _case(decay_on=decay_on)
    y, state = run()
    analytic = torch.autograd.grad(
        outputs=(y, state), inputs=list(write),
        grad_outputs=(torch.ones_like(y), torch.ones_like(state)))
    g = torch.Generator().manual_seed(31)
    direction = tuple(torch.randn(level.shape, generator=g, dtype=level.dtype)
                      for level in write)
    exact = sum(float((a * d).sum()) for a, d in zip(analytic, direction))
    numeric = _directional(run, write, direction)
    scale = max(abs(exact), 1.0)
    if decay_on:
        assert abs(numeric - exact) > 1e-4 * scale, (
            f"the forward's own derivative ({numeric}) matches autograd's ({exact}) "
            "under decay, so the clock's stop-gradient is not in the graph")
    else:
        assert abs(numeric - exact) <= 1e-6 * scale, (
            f"with no clock in the recurrence the two must agree; got {numeric} "
            f"against {exact}, so the difference step is what the decay arm measures")


@pytest.mark.parametrize("decay_on", [False, True])
def test_every_operand_the_result_depends_on_still_gets_a_real_gradient(decay_on):
    """Claim 2, with the NONZERO half that `is not None` alone would not give."""
    operands, _write, run = _case(decay_on=decay_on)
    y, state = run()
    grads = _grads(operands, y, state)
    expected = list(operands)
    assert len(expected) >= 6, "the fixture stopped covering the operands it names"
    for key in expected:
        grad = grads[key]
        assert grad is not None, f"{key} lost its gradient"
        assert torch.isfinite(grad).all(), f"{key} has a non-finite gradient"
        assert float(grad.abs().max()) > 0, (
            f"{key}'s gradient is identically zero -- either the operand stopped "
            f"reaching the output or the manufactured edge is back")


def test_the_clock_does_move_the_forward_so_the_detachment_claim_is_not_vacuous():
    """Claim 2: the detachment is load-bearing, not a statement about an unused path."""
    operands, write, run = _case(decay_on=True)
    y_base, _ = run()
    y_moved, _ = run(write_override=tuple(level.detach().flip(-1) for level in write))
    assert not torch.equal(y_base, y_moved), (
        "permuting the write levels left the forward unchanged, so the claim above is "
        "vacuous for this cell")


def test_the_oracle_declares_the_none_answer_in_its_docstring():
    """The behaviour is a contract, so it is written down where a caller reads it."""
    assert "None" in naive_rola.__doc__ and "stop_gradient" in naive_rola.__doc__


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_both_outputs_are_fp64_whatever_the_inputs_were(dtype):
    """Claim 4 (audit F2): the fp64 oracle returns fp64, and the two agree."""
    operands, _write, run = _case(decay_on=True)
    lowered = {}
    for key, tensor in operands.items():
        lowered[key] = tensor.detach().to(dtype)
    topology = Topology(levels=tuple(_ROUTING.at(w) for w in WIDTHS))
    read = tuple(lowered[f"read{i}"] for i in range(len(WIDTHS)))
    write = tuple(lowered[f"write{i}"] for i in range(len(WIDTHS)))
    dials = tuple(lowered[f"dial{i}"] for i in range(len(WIDTHS)))
    y, state = naive_rola(
        lowered["v"], read, write,
        lowered["g_write"], topology,
        LeafMassDecay(dials=dials), output_final_state=True)
    assert y.dtype is torch.float64, (
        f"y came back as {y.dtype} for {dtype} inputs; the oracle returns fp64")
    assert state.dtype is torch.float64, f"the final state came back as {state.dtype}"


def test_the_dtype_policy_is_stated_where_a_caller_reads_it():
    """A contract nobody can find is a convention. It is in the docstring."""
    assert "float64" in naive_rola.__doc__


