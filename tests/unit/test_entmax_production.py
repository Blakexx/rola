"""Focused CUDA tests for the production entmax solve."""

import pytest
import torch

from rola.routing.entmax.production import (
    production_factor,
)
from rola.routing.entmax.reference import entmax
from rola.routing.types import (
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    UnionRouting,
)
from tests.oracle.fixtures import assert_slots_close
from tests.oracle.tolerances import ENTMAX_GRADIENTS, ENTMAX_VALUES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="production entmax requires CUDA")


def _assert_matches_oracle(actual, expected, *, gradient=False):
    output = ENTMAX_GRADIENTS if gradient else ENTMAX_VALUES
    assert_slots_close(actual, expected, output=output, what=f"the production solve ({output.name})")


def _unpack_token_support(words, length):
    tokens = torch.arange(length, device=words.device, dtype=torch.long)
    selected = words.index_select(-2, torch.div(tokens, 32, rounding_mode="floor"))
    shifts = tokens.remainder(32).view(*((1,) * (selected.ndim - 2)), length, 1)
    return ((selected.to(torch.int64) >> shifts) & 1).bool()


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_production_factor_mask_ties_support_values_and_vjp(alpha):
    z = torch.tensor([[4.0, 4.0, 0.25, -8.0, 11.0]], device="cuda", dtype=torch.float32)
    mask = torch.tensor([[True, True, True, False, True]], device="cuda")
    coeff = torch.tensor([[0.4, -0.7, 1.2, 0.0, -0.2]], device="cuda", dtype=torch.float32)
    zp = z.detach().clone().requires_grad_()
    p, support_words = production_factor(zp, "entmax", alpha=alpha, mask=mask)
    zr = z.detach().clone().to(torch.float64).requires_grad_()
    ref = entmax(zr, alpha, dim=-1, mask=mask)
    support = _unpack_token_support(support_words, z.shape[-2])
    assert support_words.dtype == torch.int32
    assert support_words.shape == (1, z.shape[-1])
    assert torch.equal(support, p != 0)
    assert p.dtype == torch.float32
    _assert_matches_oracle(p, ref)
    (p * coeff).sum().backward()
    (ref * coeff).sum().backward()
    _assert_matches_oracle(zp.grad, zr.grad, gradient=True)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 65])
def test_production_factor_packs_bit31_and_zeroes_tail_words(alpha, length):
    z = torch.full((2, length, 3), -10.0, device="cuda", dtype=torch.float32)
    z[..., 0] = 5.0
    p, support_words = production_factor(z, "entmax", alpha=alpha)

    assert torch.equal(_unpack_token_support(support_words, length), p != 0)
    assert torch.all(p[..., 0] != 0)
    raw_words = support_words.to(torch.int64) & 0xFFFFFFFF
    if length >= 32:
        assert torch.all((raw_words[:, 0, 0] & (1 << 31)) != 0)
    remainder = length % 32
    if remainder:
        tail_bits = 0xFFFFFFFF ^ ((1 << remainder) - 1)
        assert torch.count_nonzero(raw_words[:, -1, :] & tail_bits) == 0
    _assert_matches_oracle(p, entmax(z.to(torch.float64), alpha, dim=-1))


def test_production_factor_packs_masked_arbitrary_leading_dimensions():
    gen = torch.Generator(device="cuda").manual_seed(68)
    z = torch.randn(2, 3, 33, 4, device="cuda", dtype=torch.float32, generator=gen)
    batch = torch.arange(2, device="cuda")[:, None, None, None]
    token = torch.arange(33, device="cuda")[None, None, :, None]
    branch = torch.arange(4, device="cuda")[None, None, None, :]
    mask = ((batch + token + branch) % 3 != 0).expand(2, 1, 33, 4).clone()
    mask[..., 0] = True

    p, support_words = production_factor(z, "entmax", alpha=1.5, mask=mask)
    ref = entmax(z.to(torch.float64), 1.5, dim=-1, mask=mask)

    assert support_words.shape == (2, 3, 2, 4)
    assert torch.equal(_unpack_token_support(support_words, z.shape[-2]), p != 0)
    _assert_matches_oracle(p, ref)


def test_production_factor_empty_rows_preserve_autograd_contract():
    z = torch.empty(0, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    p, support_words = production_factor(z, "entmax", alpha=1.5)
    assert p.shape == z.shape
    assert support_words.shape == (0, z.shape[-1])
    assert support_words.dtype == torch.int32
    p.sum().backward()
    assert z.grad is not None and z.grad.shape == z.shape


def test_production_routing_width_limit_rejects_before_jit():
    boundary = torch.empty(0, 256, device="cuda", dtype=torch.float32)
    values, support_words = production_factor(boundary, "entmax", alpha=1.5)
    assert values.shape == boundary.shape
    assert support_words.shape == (0, 256)

    too_wide = torch.empty(0, 257, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="branch width must be <= 256"):
        production_factor(too_wide, "entmax", alpha=1.5)

    with pytest.raises(ValueError, match="width must be <= 256"):
        IndependentRouting(width=257, read=SoftmaxActivation(), write=SoftmaxActivation())


@pytest.mark.parametrize("alpha", [1.25, 1.8])
def test_production_factor_rejects_unsupported_alpha(alpha):
    logits = torch.zeros(1, 4, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly alpha=1.5.*alpha=2.0"):
        production_factor(logits, "entmax", alpha=alpha)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_the_solve_reads_a_packed_projections_half_slice_where_it_lies(alpha):
    """Every solve entry takes an explicit stride triple, so the logits need not be
    contiguous (P73 S4, inventory §B3).

    An untied packed projection returns each side as a half-slice of one GEMM output,
    and the claim is that the entries address it AS IT LIES: the levels off the slice
    must be the levels off a materialized copy of it BIT FOR BIT, forward and backward
    alike. A row that only checked closeness would pass on a stride bug that read the
    wrong columns of a symmetric fixture.
    """
    from rola.routing.entmax.production import production_routing_factor_levels

    torch.manual_seed(5)
    routing = ResolvedRouting(levels=(
        UnionRouting(width=8, alpha=alpha),
        IndependentRouting(width=8, read=SoftmaxActivation(), write=SoftmaxActivation())))
    both = torch.randn(4, 64, 2 * routing.packed_logit_width, device="cuda",
                       requires_grad=True)
    sides = (both[:, :, routing.packed_logit_width:], both[:, :, :routing.packed_logit_width])
    assert not sides[0].is_contiguous()

    def run(read, write):
        levels = production_routing_factor_levels(read, write, routing)
        out = sum(level.square().sum() for side in levels for level in side)
        return levels, torch.autograd.grad(out, both, retain_graph=True)[0]

    strided, d_strided = run(*sides)
    packed, d_packed = run(*(side.contiguous() for side in sides))
    for side_a, side_b in zip(strided, packed):
        for level_a, level_b in zip(side_a, side_b):
            assert torch.equal(level_a, level_b)
    assert torch.equal(d_strided, d_packed)


@pytest.mark.parametrize("width", [3, 8, 64])
def test_every_solve_writes_every_element_of_a_dense_level(width):
    """The licence for allocating a dense level UNINITIALIZED (P73 S4, inventory §B5).

    A dense level is exactly `[BH, T, width]` -- no tile padding, unlike the
    tile-major arena whose tail lanes past `T` no kernel touches -- so nothing may
    survive a solve. Each shipped solve writes into a buffer poisoned two different
    ways; if any element were left alone the two results would differ there. Widths
    that are not powers of two are the ones a masked lane could hide in.
    """
    from rola.routing.entmax.production import _Plan

    torch.manual_seed(6)
    BH, T = 4, 96
    logits = torch.randn(BH, T, width, device="cuda")
    other = logits + 1
    words = torch.empty((BH, (T + 31) // 32, width), dtype=torch.int32, device="cuda")

    def buffers(fill, count):
        return [torch.full((BH, T, width), fill, device="cuda") for _ in range(count)]

    def solves(fill, alpha):
        union, factor, soft = buffers(fill, 3), buffers(fill, 2), buffers(fill, 2)
        plan = _Plan()
        plan.add("union", alpha, (logits, other, union[2], words, union[0], union[1]),
                 width, logit_off=0)
        plan.add("entmax", alpha, (logits, factor[0], factor[1], words), width, logit_off=0)
        plan.add("softmax", 0.0, (logits, soft[0], soft[1]), width, logit_off=0)
        plan.issue()
        return union + factor + soft

    for alpha in (1.5, 2.0):
        for cold, hot in zip(solves(0.0, alpha), solves(7.0, alpha)):
            assert torch.equal(cold, hot), (
                "a solve left an element of its dense output alone")


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("kind", ["union", "entmax", "softmax"])
def test_the_batched_solve_is_the_per_level_loop_bit_for_bit(alpha, kind):
    """P73 the STOP CLAUSE, as a gate: batching may change the launch SHAPE and nothing
    else.

    One launch over four levels of one width class, against four launches of one level
    each, on the same inputs and into the same layout -- asserted with `torch.equal`,
    because the claim is that each row performs the arithmetic it performed before, not
    that it lands nearby. A tolerance here would pass on the failure this gate exists to
    catch: a level silently run in a WIDER arm, where `LW`/`IPT` change how the sort
    permutes and how the prefix scan associates, moving `tau` by an ulp and with it a
    support boundary.

    Widths differ INSIDE the class (3 and 4 both pad to 4) because a table that carried
    only equal widths would not exercise the per-level `width` the kernel reads.
    """
    from rola.routing.entmax.production import _Plan

    torch.manual_seed(11)
    BH, T = 3, 70
    widths = [3, 4, 4, 3]
    offsets = [0, 3, 7, 11]
    total = 14
    read = torch.randn(BH, T, total, device="cuda")
    write = torch.randn(BH, T, total, device="cuda")
    n_words = (T + 31) // 32

    def run(batched):
        values = torch.zeros(BH, T, total, device="cuda")
        second = torch.zeros_like(values)
        midpoints = torch.zeros_like(values)
        words = torch.zeros((BH, n_words, total), dtype=torch.int32, device="cuda")
        plans = [_Plan()] if batched else [_Plan() for _ in widths]
        for index, (width, offset) in enumerate(zip(widths, offsets)):
            plan = plans[0] if batched else plans[index]
            if kind == "union":
                plan.add("union", alpha,
                         (read, write, midpoints, words, values, second), width,
                         logit_off=offset, value_off=offset, midpoint_off=offset,
                         support_off=offset)
            elif kind == "entmax":
                plan.add("entmax", alpha, (read, values, second, words), width,
                         logit_off=offset, value_off=offset, support_off=offset)
            else:
                plan.add("softmax", 0.0, (read, values, second), width,
                         logit_off=offset, value_off=offset)
        launches = sum(plan.issue() for plan in plans)
        return launches, values, second, midpoints, words

    batched_count, *batched = run(True)
    loop_count, *loop = run(False)
    assert batched_count == 1 and loop_count == len(widths), (
        f"the shapes under comparison are wrong: batched issued {batched_count} launch(es) "
        f"and the loop {loop_count}; the point of the gate is 1 against {len(widths)}")
    for name, got, want in zip(("values", "second", "midpoints", "support"), batched, loop):
        assert torch.equal(got, want), (
            f"the batched solve's {name} differs from the per-level loop's in "
            f"{int((got != want).sum())} elements -- batching changed the arithmetic, "
            "not just the launch shape")

    # THE TOOTH. The equality above is only worth reading if the class rule it rests on
    # is ENFORCED rather than assumed, and the enforcement is not in this file: a table
    # carrying two width classes is refused at the entry, so the failure the `torch.equal`
    # could never see (a narrow level silently padded into a wider arm) cannot be built
    # in the first place. `_Plan` is what keeps such a table from being assembled; this
    # asserts the layer under it refuses one anyway.
    from rola.ops._ext import extension

    values = torch.zeros(BH, T, 68, device="cuda")
    with pytest.raises(RuntimeError, match="one width class"):
        extension().routing_softmax_forward(
            torch.randn(BH, T, 68, device="cuda"), values, None, [0, 4], [0, 4], [4, 64], 1)


def test_a_uniform_plan_solves_all_its_levels_in_one_launch():
    """`producer.py` has claimed "one packed projection and one batched solve" since it
    was written, and until P73 S5 the second half was false -- a python `for` over
    `routing.levels`, one launch and three allocations per level. This is the claim made
    checkable.

    A level joins a launch only when it selects the same template arm, so the gate also
    pins the NEGATIVE: two levels of different width CLASSES (4 and 64) stay two
    launches, because running the narrow one in the wide arm would not be the loop it
    replaces.
    """
    from rola.routing.entmax.production import _Plan, _ProductionLevelsFn

    def launches(routing, widths):
        counted = []
        original = _Plan.issue

        def counting(self):
            counted.append(len(self._issued))
            return original(self)

        _Plan.issue = counting
        try:
            read = torch.randn(2, 64, routing.packed_logit_width, device="cuda")
            _ProductionLevelsFn.apply(read, torch.randn_like(read), routing, torch.bfloat16)
        finally:
            _Plan.issue = original
        return counted[0]

    uniform = ResolvedRouting((UnionRouting(width=4, alpha=1.5), UnionRouting(width=4, alpha=1.5),
                              UnionRouting(width=4, alpha=1.5)))
    assert launches(uniform, (4, 4, 4)) == 1, (
        "three union levels of one width class must solve in ONE launch")

    mixed = ResolvedRouting((UnionRouting(width=4, alpha=1.5), UnionRouting(width=64, alpha=1.5)))
    assert launches(mixed, (4, 64)) == 2, (
        "levels of different width classes must NOT share a launch: the sort and the "
        "prefix scan associate differently per class")
