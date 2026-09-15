"""Gates for the dense per-level production routing solve (the producer path).

`rola.routing.entmax.production.production_routing_factor_levels` is
what `rola/routing/producer.py` calls. It has two lowerings of one semantics -- a DEVICE
lowering at CUDA fp32 and the identical sorted closed form in torch on CPU/fp64 -- and
both are gated here against the ONE canonical ground truth this branch authorizes for
the routing primitive: the fp64 oracle in `routing/reference.py`
(`entmax/reference.py:entmax`), which no runtime path touches. The oracle is a closed
form at both alphas this file tests (`entmax15` at 1.5, `sparsemax` at 2.0 -- the third
party's own exact forms, inverted algebraically for `tau`) -- there is no bisection
anywhere in this file's ground truth.

**THE DEVICE LOWERING IS NOT ONE KERNEL**, and it is no longer any part Triton. The UNION solve is a hand-CUDA + CUB kernel (`csrc/rola/src/entmax/entmax.cu`);
the independent per-side solves and the softmax pair are `csrc/rola/src/entmax/factor.cu`,
ported off Triton for the closed-world AOT policy and for P65, which trains through
them. Everything below that says "the two lowerings" still means exactly what it
meant, device against torch; what changed over time is WHICH BINARY the rows were
measured on, and the tie table was
re-measured on the new one (see `_TIE_DIVERGENCE_BOUND`).

Support is checked for **exact** set equality, never by threshold: entmax support is a
discrete object (sparsity comes from entmax, not from a floor), so "the same zeros" is a hard
assertion and only the amplitudes carry a tolerance. Two scopes are deliberately
excluded from that hard assertion, and neither is a tolerance in disguise:

* **softmax sides.** A dense activation has no support; its "zeros" are fp underflow
  and differ between fp32 and fp64 by construction. Exact support is asserted for
  exactly the zero-capable sides (`ResolvedRouting.read/write_level_can_zero`).
* **the exact-tie fixture**, which is not gated against the oracle at all. Where a
  logit sits exactly on the threshold, an oracle comparison is a comparison of TWO
  INDEPENDENT closed forms -- production's fp32 sorted solve and the oracle's fp64
  `entmax_tau` -- at an exact equality, each free to land on either side of the same
  strict `z_i > tau` boundary. They are not obligated to agree there, and measured
  directly they do not always (alpha=1.5, the `union` shape, level 0 disagrees). One
  extra support member at that boundary moves `den` in `s*(dp - num/den)`, so the
  GRADIENT effect of a boundary disagreement is O(1) even though the two sides'
  amplitudes are close -- so nothing about that fixture can be gated on the oracle
  without gating one boundary comparison against a second one that has no obligation
  to agree with it. It is instead gated on the two PRODUCTION lowerings against each
  other -- the SAME closed form computed by two different code paths (the fused
  CUDA/CUB kernels vs the pure-torch lowering), both handed the SAME fp32
  inputs -- which must therefore agree on support exactly
  (`test_the_two_lowerings_agree_on_exact_ties`).
"""

from __future__ import annotations

import pytest
import torch
from conftest import fuse_gain_columns, produce_routing, resolved_routing_from_topology

import rola.routing.producer as production_path
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.producer import RouteProducer
from rola.routing.reference import routing_factor_levels as oracle_factor_levels
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
)
from tests.oracle.fixtures import assert_planted_errors_fail
from tests.oracle.tolerances import BF16_RTOL

# Identical to `tests/ops/test_rola_entmax_production.py`'s -- the same solve against
# the same oracle, so the same numbers. Never loosened.
_PRODUCTION_RTOL = 2e-5
_PRODUCTION_ATOL = 2e-6
_PRODUCTION_GRAD_RTOL = 5e-5
_PRODUCTION_GRAD_ATOL = 5e-6

#: THE SEAM: a comparison that crosses the STORAGE boundary carries the bf16 operand
#: term and nothing else. `u_bf16 = 2^-8 = 3.9e-3` per requantization; `BF16_RTOL`
#: is that budget, derived once in `tests/oracle/tolerances.py`, and it is the same
#: number every kernel-vs-oracle gate in the tree uses. The ABSOLUTE floor is that
#: relative bound applied to a simplex entry's own scale (<= 1). Measured worst here:
#: 3.58e-3 relative, 1.91e-3 absolute, both at alpha=1.5. SUPPORT carries no
#: tolerance across this seam at all -- the solve writes exact zeros as zeros, so the
#: set is identical or the storage broke the self-masking theorem's precondition.
_STORED_RTOL = BF16_RTOL
_STORED_ATOL = BF16_RTOL

#: Union tie-gradient divergence bound: the measured worst case is 4.44% of
#: elements (write side, alpha=1.5, `union`), and this carries ~1.35x headroom.
#: Tightening is the correct direction; loosening requires a re-measurement in
#: `test_the_two_lowerings_agree_on_exact_ties`'s own table.
#:
#: RE-MEASURED 2026-08-02 (entmax kernel merge), because the thing on the CUDA side of
#: this comparison CHANGED. Track E replaced the union solve's fused-Triton lowering
#: with a hand-CUDA + CUB kernel and DELETED the Triton one, so every number in the
#: table below was a measurement of an implementation that no longer exists -- a stale
#: provenance, not a stale bound. Re-run against the new kernel, the table reproduces to
#: the recorded digits: union 1.5 read 31/720 (4.31%) / write 32/720 (4.44%), union 2.0
#: read 2/720 (0.28%) / write 4/720 (0.56%), mixed_per_level 1.5 28/1800 (1.56%) /
#: 33/1800 (1.83%), mixed_per_level 2.0 6/1800 (0.33%) both sides; complement worst case
#: 2.673e-6 of scale, divergence floor 4.857e-6.
#:
#: That it reproduces is a fact about the divergence's CAUSE rather than a coincidence,
#: and it is the reason these bounds are carried forward unchanged rather than merely
#: re-asserted: the divergence is an ill-conditioned division by a value that is exactly
#: 0 at a tie, so what it measures is fp32 rounding UPSTREAM of that division. Two
#: independent closed forms round independently and land the same distance away.
_TIE_DIVERGENCE_BOUND = 0.06

#: The tie test's complement bound, applied to the elements the production
#: tolerance SELECTS -- deliberately TIGHTER than that selector, which is what makes
#: the assertion a different predicate rather than a restatement of its own
#: selection. Read as a fraction of the tensor's own scale (`max|oracle|`), the same
#: scale form `test_producer_conditioned_path_backward_matches_its_oracle_path`
#: already uses and for the same reason: these are composite gradients with
#: near-zero entries, where a bare elementwise relative bound is not the meaningful
#: quantity.
#:
#: MEASURED, with its headroom stated: across all 8 excluded cases (2 shapes x 2
#: alphas x read/write) the largest complement deviation is 2.67e-6 of scale
#: (union, alpha=1.5, write; the other seven are ~2e-7), while the SMALLEST
#: deviation among the diverging elements is 4.86e-6 of scale. The divergence is
#: therefore separated from the agreement by a real gap, and this bound sits inside
#: it with ~3.7x headroom over the measured worst case.
_TIE_COMPLEMENT_SCALE_BOUND = 1e-5

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused production solve needs CUDA")


def _routings(alpha: float) -> dict[str, ResolvedRouting]:
    """Every structural level shape the V3 producer can hand the solve."""
    return {
        "union": ResolvedRouting((UnionRouting(width=6, alpha=alpha),)),
        # A second, untied level alongside the tied one, so `read_logits` is genuinely
        # used somewhere in the graph. A standalone tied level never reads
        # `read_logits` at all (by design: the read factor IS the write factor), which
        # makes `production_read` an unused `autograd.grad` input on its own.
        "tied": ResolvedRouting(
            (
                TiedRouting(width=4, op=EntmaxActivation(alpha)),
                IndependentRouting(width=5, read=SoftmaxActivation(), write=SoftmaxActivation()),
            ),
        ),
        "mixed_per_level": ResolvedRouting(
            (
                IndependentRouting(width=3, read=EntmaxActivation(alpha), write=SoftmaxActivation()),
                IndependentRouting(width=5, read=SoftmaxActivation(),
                                   write=EntmaxActivation(alpha)),
                UnionRouting(width=4, alpha=alpha),
            ),
        ),
        "dense_only": ResolvedRouting(
            (IndependentRouting(width=4, read=SoftmaxActivation(), write=SoftmaxActivation()),)),
    }


def _stress_logits(routing: ResolvedRouting, kind: str, *, device, seed: int) -> torch.Tensor:
    """Stress fixtures for the threshold solve, in the packed ``[BH, T, packed_logit_width]`` layout."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (3, 40, routing.packed_logit_width)
    base = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if kind == "random":
        return base

    if kind == "near_boundary":
        # Quantizing to halves clusters logits onto the candidate thresholds; the jitter
        # then separates them by ~1e-4, which is far inside any tolerance here but still
        # a value the closed-form oracle resolves exactly. This is the case where
        # production's greatest-valid-k rule and the oracle's closed-form tau are most
        # likely to disagree about where the support boundary falls.
        jitter = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return torch.round(base * 2.0) / 2.0 + 1.0e-4 * jitter
    if kind == "exact_tie":
        # Logits landing exactly ON a threshold. See the module docstring: support
        # equality is not asserted against the oracle here, because the oracle is the
        # side that cannot answer.
        return torch.round(base * 2.0) / 2.0
    if kind == "collapsed_support":
        # One dominant digit per row: the support collapses to a single element, the
        # smallest support entmax can produce (a genuinely empty support is
        # unreachable -- the argmax is always in it).
        collapsed = torch.full(shape, -20.0, device=device, dtype=torch.float32)
        # ONE dominant digit per LEVEL per row. In the packed layout the dominant index
        # must land inside the level's own span (`level_logit_slice`) -- a single index
        # drawn over the whole packed width would leave every other level a constant row,
        # whose entmax is uniform, i.e. the opposite of the collapsed support this
        # fixture exists to produce.
        for level, width in enumerate(routing.branches):
            span = routing.level_logit_slice(level)
            index = torch.randint(0, width, shape[:-1], device=device, generator=generator)
            collapsed[..., span] = collapsed[..., span].scatter(-1, index.unsqueeze(-1), 20.0)
        return collapsed
    if kind == "large_magnitude":
        # 5x the router's natural logit scale: the solve's support collapses hard, so
        # the row-max shift is what keeps it finite. The scale is deliberately not pushed
        # further because union's read side is `entmax_value * sigmoid(read - write)`, and
        # once a support member's entmax value falls to ~1e-18 its `log` in the write
        # softmax is fp32-ill-conditioned -- a conditioning property of the fixture, not
        # of the solve (both production lowerings agree with each other to 2e-7 there,
        # and disagree with fp64 together). Union's extreme-magnitude contract (the
        # overflow-free delta) is gated at the fp32 maximum by
        # `test_rola_entmax_production.py::test_production_union_handles_near_max_logits_and_vjp`.
        return base * 5.0
    raise ValueError(kind)


def _assert_levels_match(actual, expected, *, gradient=False, stored=False, extra_atol=0.0):
    """``extra_atol`` is a DERIVED term the caller states, never a loosened constant."""
    if stored:
        rtol, atol = _STORED_RTOL, _STORED_ATOL
    elif gradient:
        rtol, atol = _PRODUCTION_GRAD_RTOL, _PRODUCTION_GRAD_ATOL
    else:
        rtol, atol = _PRODUCTION_RTOL, _PRODUCTION_ATOL
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol + extra_atol,
                               check_dtype=False)


def _assert_support_identical(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert torch.equal(actual != 0, expected != 0), f"{label}: support sets differ"


def _count_exact_ties(logits: torch.Tensor, alpha: float) -> int:
    """Count elements landing exactly on the sorted-solve threshold ``z == tau``.

    Independent reimplementation of `_sorted_solve`'s tau (production.py), not an
    import of it: the point is to positively confirm the "exact_tie" fixture actually
    produces the tied-digit case `test_the_two_lowerings_agree_on_exact_ties` claims to
    exercise, so it must not share a bug with the code it is checking.
    """
    z = logits - logits.amax(dim=-1, keepdim=True)
    width = z.shape[-1]
    sorted_z, _ = torch.sort(z, dim=-1, descending=True)
    prefix = sorted_z.cumsum(dim=-1)
    k = torch.arange(1, width + 1, device=z.device, dtype=z.dtype)
    if alpha == 2.0:
        tau_candidates = (prefix - 1.0) / k
        candidate = sorted_z > tau_candidates
    else:
        prefix_sq = (sorted_z * sorted_z).cumsum(dim=-1)
        discriminant = prefix * prefix - k * (prefix_sq - 4.0)
        tau_candidates = (prefix - discriminant.clamp(min=0.0).sqrt()) / k
        candidate = (discriminant >= 0.0) & (sorted_z > tau_candidates)
    columns = torch.arange(width, device=z.device)
    chosen = torch.where(candidate, columns, torch.full_like(columns, -1)).amax(dim=-1, keepdim=True)
    tau = torch.gather(tau_candidates, -1, chosen.clamp(min=0))
    tau = torch.where(chosen >= 0, tau, torch.zeros_like(tau))
    return int((z == tau).sum().item())


def _check_against_oracle(
    routing: ResolvedRouting, logits_read, logits_write, *, label: str, exact_support: bool = True,
):
    production_read = logits_read.detach().clone().requires_grad_()
    production_write = logits_write.detach().clone().requires_grad_()
    oracle_read = logits_read.detach().clone().to(torch.float64).requires_grad_()
    oracle_write = logits_write.detach().clone().to(torch.float64).requires_grad_()

    reads, writes = production_routing_factor_levels(
        production_read, production_write, routing, stored_dtype=torch.float32)
    oracle_reads, oracle_writes = oracle_factor_levels(oracle_read, oracle_write, routing)

    assert len(reads) == len(writes) == routing.D
    for level in range(routing.D):
        assert reads[level].shape == (
            logits_read.shape[0], logits_read.shape[1], routing.branches[level])
        if exact_support and routing.read_level_can_zero[level]:
            _assert_support_identical(
                reads[level], oracle_reads[level], f"{label} read level {level}")
        if exact_support and routing.write_level_can_zero[level]:
            _assert_support_identical(
                writes[level], oracle_writes[level], f"{label} write level {level}")
        _assert_levels_match(reads[level], oracle_reads[level])
        _assert_levels_match(writes[level], oracle_writes[level])

    generator = torch.Generator(device=logits_read.device).manual_seed(9109)
    cotangents = [
        torch.randn(level.shape, device=level.device, dtype=torch.float32, generator=generator)
        for level in (*reads, *writes)
    ]
    loss = sum(
        (level * cotangent).sum()
        for level, cotangent in zip((*reads, *writes), cotangents))
    oracle_loss = sum(
        (level * cotangent.to(torch.float64)).sum()
        for level, cotangent in zip((*oracle_reads, *oracle_writes), cotangents))
    production_vjp = torch.autograd.grad(loss, (production_read, production_write))
    oracle_vjp = torch.autograd.grad(oracle_loss, (oracle_read, oracle_write))
    _assert_levels_match(production_vjp[0], oracle_vjp[0], gradient=True)
    _assert_levels_match(production_vjp[1], oracle_vjp[1], gradient=True)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_tied_routing_read_factor_is_the_write_factor_on_both_lowerings(alpha):
    """`TiedRouting`'s read factor IS its write factor -- the same tensor, not merely
    numerically equal -- on the fp64 oracle lowering, and bit-identical on the torch
    production lowering (`_ProductionLevelsFn` always allocates read/write as distinct
    autograd outputs, even for a tied level, so identity does not survive there)."""
    tied = ResolvedRouting((TiedRouting(width=6, op=EntmaxActivation(alpha)),))
    generator = torch.Generator().manual_seed(4242)
    read = torch.randn(2, 10, 6, dtype=torch.float64, generator=generator)
    write = torch.randn(2, 10, 6, dtype=torch.float64, generator=generator)

    oracle_reads, oracle_writes = oracle_factor_levels(read, write, tied)
    assert oracle_reads[0] is oracle_writes[0]  # the read factor IS the write factor

    prod_reads, prod_writes = production_routing_factor_levels(
        read.to(torch.float32), write.to(torch.float32), tied, stored_dtype=torch.float32)
    torch.testing.assert_close(prod_reads[0], prod_writes[0], rtol=0, atol=0)


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_cuda_tied_routing_read_factor_is_the_write_factor(alpha):
    """The CUDA-kernel-dispatch path (`_ProductionLevelsFn`, GPU logits) takes a
    standalone `TiedRouting` level through one kernel launch, one solve. Mirrors
    `test_tied_routing_read_factor_is_the_write_factor_on_both_lowerings` (CPU/fp64)
    but forces the CUDA lowering by handing it CUDA fp32 logits."""
    tied = ResolvedRouting((TiedRouting(width=6, op=EntmaxActivation(alpha)),))
    generator = torch.Generator(device="cuda").manual_seed(4242)
    read = torch.randn(2, 10, 6, device="cuda", dtype=torch.float32, generator=generator)
    write = torch.randn(2, 10, 6, device="cuda", dtype=torch.float32, generator=generator)

    tied_reads, tied_writes = production_routing_factor_levels(
        read, write, tied, stored_dtype=torch.float32)
    torch.testing.assert_close(tied_reads[0], tied_writes[0], rtol=0, atol=0)

    oracle_read = read.to(torch.float64)
    oracle_write = write.to(torch.float64)
    oracle_reads, oracle_writes = oracle_factor_levels(oracle_read, oracle_write, tied)
    assert oracle_reads[0] is oracle_writes[0]  # the read factor IS the write factor
    _assert_levels_match(tied_reads[0], oracle_reads[0])
    _assert_levels_match(tied_writes[0], oracle_writes[0])


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "tied", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_torch_lowering_matches_the_fp64_oracle(alpha, shape, kind):
    """CPU/fp64 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cpu", seed=17)
    write = _stress_logits(routing, kind, device="cpu", seed=18)
    _check_against_oracle(routing, read, write, label=f"cpu/{shape}/{kind}/{alpha}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize(
    "shape", ["union", "tied", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_cuda_lowering_matches_the_fp64_oracle(alpha, shape, kind):
    """CUDA fp32 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cuda", seed=17)
    write = _stress_logits(routing, kind, device="cuda", seed=18)
    _check_against_oracle(routing, read, write, label=f"cuda/{shape}/{kind}/{alpha}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "tied", "mixed_per_level"])
def test_the_two_lowerings_agree_on_exact_ties(alpha, shape):
    """Exact ties, where the oracle cannot arbitrate: the closed forms must still agree.

    Support is asserted as an identical set (a closed form has no convergence slack to
    hide behind) and the amplitudes and the VJP at the production tolerances.
    """
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, "exact_tie", device="cuda", seed=23).requires_grad_()
    write = _stress_logits(routing, "exact_tie", device="cuda", seed=24).requires_grad_()

    # Soundness gate on the fixture itself: this test is the SOLE gate for tie
    # semantics (explicitly excluded from the oracle gate, see module docstring), so it
    # must not go vacuous under a seed/width change without saying so loudly.
    ties = sum(
        _count_exact_ties(read.detach()[..., routing.level_logit_slice(index)], alpha)
        + _count_exact_ties(write.detach()[..., routing.level_logit_slice(index)], alpha)
        for index, width in enumerate(routing.branches)
    )
    assert ties > 0, (
        f"exact_tie fixture for alpha={alpha} shape={shape!r} contains no exact ties -- "
        "the tie-semantics gate below would be exercising nothing"
    )

    cpu_read = read.detach().cpu().requires_grad_()
    cpu_write = write.detach().cpu().requires_grad_()
    cuda_reads, cuda_writes = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.float32)
    cpu_reads, cpu_writes = production_routing_factor_levels(cpu_read, cpu_write, routing)
    for level in range(routing.D):
        if routing.read_level_can_zero[level]:
            _assert_support_identical(cuda_reads[level].cpu(), cpu_reads[level], f"read {level}")
        if routing.write_level_can_zero[level]:
            _assert_support_identical(cuda_writes[level].cpu(), cpu_writes[level], f"write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(5501)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator)
        for level in (*cpu_reads, *cpu_writes)
    ]
    cuda_vjp = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        (read, write))
    cpu_vjp = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        (cpu_read, cpu_write))
    if any(isinstance(level, UnionRouting) for level in routing.levels):
        # V9 item 3 (compliance-audit): this used to `return`, leaving union tie
        # gradients with NO gate at all. The divergence itself is real and is NOT
        # weakened here -- what is added is a BOUND, so the hole becomes a pinned
        # quantity instead of an early exit.
        #
        # CONFIRMED, not assumed (fix-pass re-check of the adversarial review's
        # NOTE): removing the exclusion and comparing elementwise genuinely fails --
        # union-1.5, union-2.0, mixed_per_level-1.5, mixed_per_level-2.0 all
        # mismatch, up to 0.70 relative / 0.23 absolute on 6/1800 elements, orders
        # of magnitude past the 5e-5/5e-6 production tolerance. Root cause: union's
        # VJP divides the write-softmax adjoint by the entmax value
        # (`d_log_weight / midpoint_values`); at a tie that value is exactly 0 in
        # exact arithmetic, so fp32 rounding upstream of the division (row-max
        # shift, sort, prefix sums -- computed independently by the two lowerings,
        # which are not required to round identically) changes which side of
        # zero-ish the divisor lands on, and the division amplifies that into an
        # O(1) divergence in specific elements. This is not a tolerance gap that a
        # looser bound would close: the two lowerings compute genuinely different
        # fp32 numbers from an ill-conditioned division, not the same number to
        # different precision.
        #
        # What IS gateable, and is gated: the divergence is CONFINED. It touches a
        # small fraction of elements and the rest agree at the unmodified
        # production gradient tolerance. A change that widened it -- more diverging
        # elements, or divergence spreading off the tie -- fails here instead of
        # being invisible behind a `return`.
        #
        # CORRECTION to the number the superseded note carried. It quoted
        # "6/1800 elements" as though that were the divergence; measured across
        # every excluded case it is the alpha=2.0 figure only, and alpha=1.5 is an
        # order of magnitude larger:
        #
        #     shape            alpha  read           write
        #     union             1.5   31/720  4.31%  32/720  4.44%
        #     union             2.0    2/720  0.28%   4/720  0.56%
        #     mixed_per_level   1.5   28/1800 1.56%  33/1800 1.83%
        #     mixed_per_level   2.0    6/1800 0.33%   6/1800 0.33%
        #
        # The bound below is the worst of those (4.44%) with ~1.35x headroom.
        # TIGHTENING it is the correct direction for any future change.
        #
# THE COMPLEMENT ASSERTION IS WHAT MAKES THE BOUND MEAN SOMETHING, and it
        # took three goes to write. Recording the two rejected forms, because each
        # was wrong in a way that reads as right:
        #
        # 1. `_assert_levels_match(cuda_grad[close], cpu_grad[close])` -- selecting
        #    with `torch.isclose` at the production tolerance and then asserting the
        #    SAME predicate on the subset that predicate defines.
        #    `|a-b| <= atol + rtol*|b|` holds by construction on every element of
        #    `close`, so it could not fail for ANY input (adversarial review
        #    2026-07-29). It read as "and the complement agrees" while gating
        #    nothing.
        # 2. Selecting at a COARSE tolerance and asserting at the production one --
        #    i.e. claiming the divergence is BIMODAL. That is a genuinely different
        #    predicate, and it FAILS: measured, the diverging elements run down to
        #    9.8e-4 relative (union, alpha=1.5, read) and 5.1e-4 (mixed_per_level,
        #    alpha=1.5, write), so there is no coarse relative threshold with both
        #    the whole divergence above it and the whole agreement below it. The
        #    claim was false and the test said so. It is recorded rather than
        #    quietly dropped, because "the divergence is bimodal in relative error"
        #    is exactly the kind of thing a future reader would assume.
        #
        # What IS true, and is what this asserts: the separation is in the deviation
        # measured against the TENSOR'S SCALE, not elementwise-relative. Diverging
        # elements deviate by >= 4.86e-6 of scale; agreeing ones by <= 2.67e-6.
        #
        # THE BOUND IS A MARGIN, NOT A CONTAINMENT (corrected 2026-07-30,
        # adversarial review). `_TIE_COMPLEMENT_SCALE_BOUND` is 1e-5, which is
        # 2.06x ABOVE the top of that gap (4.86e-6) -- it does not sit inside it,
        # and the commit message and this comment all said it
        # did. What IS true: it sits 3.7x above the worst MEASURED complement
        # (2.67e-6), so it is a regression bound with headroom, and the injection
        # below is what says it still catches the drift that matters. Tightening to
        # 4e-6 would make the containment claim literally true and is the correct
        # direction for any future change; it is not taken here because the headroom
        # over 2.67e-6 is what absorbs a machine change, and a bound that fails on a
        # new GPU for a reason unrelated to the tie teaches nothing.
        #
        # So the complement is bounded strictly TIGHTER than the selector that chose
        # it -- the reviewer's first suggested form -- and the two predicates are
        # different in the direction that bites: a systematic drift of ~2e-5, small
        # enough to keep most elements inside the production `rtol` and so invisible
        # to the fraction bound above, moves this bound by 2x and fails here. A
        # larger drift (the review's injected 1e-3) fails the fraction bound
        # instead, so between them the two assertions have no blind band.
        for cuda_grad, cpu_grad, side in ((cuda_vjp[0].cpu(), cpu_vjp[0], "read"),
                                          (cuda_vjp[1].cpu(), cpu_vjp[1], "write")):
            close = torch.isclose(cuda_grad, cpu_grad,
                                  rtol=_PRODUCTION_GRAD_RTOL, atol=_PRODUCTION_GRAD_ATOL)
            diverging = int((~close).sum())
            fraction = diverging / close.numel()
            assert fraction <= _TIE_DIVERGENCE_BOUND, (
                f"union tie-gradient divergence WIDENED on the {side} side: "
                f"{diverging}/{close.numel()} = {fraction:.4f} of elements disagree "
                "past the production gradient tolerance, against a measured worst "
                "case of 4.44%. The known divergence is an ill-conditioned division "
                "at a tie; this bound exists so that it staying confined is checked "
                "rather than assumed.")
            scale = max(1e-30, float(cpu_grad.abs().max()))
            complement = float((cuda_grad[close] - cpu_grad[close]).abs().max()) / scale
            assert complement <= _TIE_COMPLEMENT_SCALE_BOUND, (
                f"the {side}-side complement -- the elements the production tolerance "
                f"ACCEPTS -- now deviates by {complement:.3e} of the tensor's scale, past "
                f"the {_TIE_COMPLEMENT_SCALE_BOUND} bound (measured worst case 2.67e-6, "
                "so this bound carries 3.7x MARGIN over the measurement). The known tie "
                "divergence is separated from the agreement by a real gap: agreement "
                "tops out at 2.67e-6 of scale and divergence starts at 4.86e-6. This "
                "bound sits 2.06x ABOVE the top of that gap -- it is a margin, not a "
                "containment -- and it still fails on a systematic drift too small to "
                "move the fraction bound above (verified by injection at 2e-5).")
        return
    _assert_levels_match(cuda_vjp[0].cpu(), cpu_vjp[0], gradient=True)
    _assert_levels_match(cuda_vjp[1].cpu(), cpu_vjp[1], gradient=True)


def _forbid_every_reference_entry_point(monkeypatch) -> None:
    """Patch all THREE of the reference module's public entry points
    (``__all__ = ["entmax", "entmax_forward", "entmax_tau"]``), not just the one this
    solve happens to call today. Patching only ``entmax_tau`` proves the solve does not
    reach that one function; it says nothing about ``entmax`` or ``entmax_forward``, and
    a caller that reached either of those would have passed the narrower guard."""
    import rola.routing.entmax.reference as entmax_reference

    def _forbidden(*args, **kwargs):
        raise AssertionError("production routing called the fp64 reference")

    for name in entmax_reference.__all__:
        monkeypatch.setattr(entmax_reference, name, _forbidden)


@_CUDA
def test_the_solve_never_reaches_the_reference_cuda(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CUDA half)."""
    _forbid_every_reference_entry_point(monkeypatch)
    routing = _routings(1.5)["mixed_per_level"]
    read = _stress_logits(routing, "random", device="cuda", seed=31).requires_grad_()
    write = _stress_logits(routing, "random", device="cuda", seed=32).requires_grad_()
    reads, writes = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.float32)
    sum(level.sum() for level in (*reads, *writes)).backward()
    assert read.grad is not None and write.grad is not None


def test_the_solve_never_reaches_the_reference_cpu(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CPU half).

    Split from the CUDA half (adversarial review NOTE): the torch lowering needs no GPU,
    so this must run on a CPU-only runner instead of being skipped along with the CUDA
    half under one `@_CUDA`-marked test.
    """
    _forbid_every_reference_entry_point(monkeypatch)
    routing = _routings(1.5)["mixed_per_level"]
    read = _stress_logits(routing, "random", device="cpu", seed=31).requires_grad_()
    write = _stress_logits(routing, "random", device="cpu", seed=32).requires_grad_()

    cpu_reads, cpu_writes = production_routing_factor_levels(read, write, routing)
    sum(level.sum() for level in (*cpu_reads, *cpu_writes)).backward()
    assert read.grad is not None and write.grad is not None


# ---------------------------------------------------------------------------
# End-to-end: the V3 producer on the production path against the V3 producer on
# the fp64 oracle path. Same producer, same conditioning arithmetic, only the
# routing solve swapped -- so any difference is the port's.
# ---------------------------------------------------------------------------


def _producer_topology(alpha: float) -> Topology:
    return Topology(
        levels=(
            TiedRouting(width=4, op=EntmaxActivation(alpha)),
            UnionRouting(width=6, alpha=alpha),
            IndependentRouting(width=3, read=SoftmaxActivation(), write=SoftmaxActivation()),
        ),
    )


#: the fixtures' value width; the topology carries none, so these gates state theirs.
_D_V = 5


def _producer_params(topology: Topology, *, dtype, device, sides, B=2, L=48, H=2, dm=16,
                     seed=404):
    resolved = resolved_routing_from_topology(topology)
    generator = torch.Generator(device=device).manual_seed(seed)

    def _randn(*shape):
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(dtype)

    return {
        "x": _randn(B, L, dm),
        "route_W": _randn(H, dm, resolved.packed_router_width),
        "gain_W": _randn(H, sides, dm),
        "gain_bias": torch.zeros(H, sides, device=device, dtype=dtype),
        "v": _randn(B, L, H, _D_V),
    }


def _run_producer(params, topology):
    return produce_routing(
        params["x"], None, params["v"], params["route_W"], None, params["gain_W"],
        params["gain_bias"], topology)


@_CUDA
def test_the_projection_lands_inside_its_bf16_output_envelope():
    """THE PROJECTION half of the seam split, gated numerically against fp64, PER LOGIT.

    cuBLAS's mainloop is opaque, so this does not mirror its arithmetic -- it bounds the
    DECLARED class, which is bf16 operands / fp32 accumulate / bf16 OUT. Two terms of the
    same order, and the fp32 accumulation (`2^-24` per step) is not one of them. On each
    logit slot:

        dz <= 2^-8 * (2 * sum_k |x_k| |W_k|  +  |z|)    (the operands before cancellation, the output)

    The operand sum is the same projection run in fp64 on the parameters' magnitudes, so it
    lands in the logits' own layout (`tests.oracle.fixtures.assert_slots_close`).
    `_logit_floor` is this bound's max over the slots, the `dz` the downstream amplitude
    envelope is built from. MEASURED 2026-09-14: the worst slot sits at 0.40 of its
    allowance. The planted errors (a wiped logit, the smallest allowances exceeded) must
    fail on the production logits themselves.
    """
    topology = _producer_topology(1.5)
    params = _producer_params(topology, dtype=torch.float32, device="cuda", sides=1)
    read_logits, write_logits = _production_logits(topology, params)
    assert read_logits.dtype is torch.bfloat16

    oracle_params = {name: tensor.to(torch.float64) for name, tensor in params.items()}
    read64, write64 = _production_logits(topology, oracle_params)
    assert read64.dtype is torch.float64

    magnitudes = {name: tensor.to(torch.float64).abs() for name, tensor in params.items()}
    read_terms, write_terms = _production_logits(topology, magnitudes)
    for side, got, want, terms in (("read", read_logits, read64, read_terms),
                                   ("write", write_logits, write64, write_terms)):
        assert_planted_errors_fail(_flat64(got), want, envelope=2.0 * terms + want.abs(), rtol=2.0 ** -8,
                                   what=f"the {side} logits")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_the_solve_seam_is_exact_on_the_production_logits(alpha):
    """THE SOLVE GATE, at the seam the oracle-precision doctrine puts it at.

    The projection is a bf16-operand tensor-core GEMM whose arithmetic no reference can
    mirror, so it is gated NUMERICALLY elsewhere and the solve is gated EXACTLY here:
    the reference solve is fed the PRODUCTION LOGITS -- the very tensor the kernel read --
    and the support sets must agree BITWISE. bf16 -> fp32 and bf16 -> fp64 are both exact
    widenings, so the two solves see the same real numbers and nothing but solve
    precision can separate them; entmax support is a discrete object, so "the same
    numbers in" has to mean "the same set out".
    """
    topology = _producer_topology(alpha)
    resolved = resolved_routing_from_topology(topology)
    params = _producer_params(topology, dtype=torch.float32, device="cuda", sides=1)

    H, hidden_size, _ = params["route_W"].shape
    producer = RouteProducer(topology.levels, hidden_size=hidden_size, num_heads=H)
    parameters = {"route_W": fuse_gain_columns(params["route_W"], params["gain_W"]),
                  "side_gain.gain_bias": params["gain_bias"]}
    read_logits, write_logits, _ = _project_under(producer, parameters, params["x"])
    assert read_logits.dtype is torch.bfloat16, (
        "the shipped projection emits bf16 logits; this gate is meaningless against any "
        "other dtype")

    kernel_read, kernel_write = production_routing_factor_levels(
        read_logits, write_logits, producer.routing)
    reference_read, reference_write = oracle_factor_levels(
        _flat64(read_logits), _flat64(write_logits), producer.routing)

    for level in range(topology.D):
        if resolved.read_level_can_zero[level]:
            _assert_support_identical(
                kernel_read[level], reference_read[level], f"read {level}")
        if resolved.write_level_can_zero[level]:
            _assert_support_identical(
                kernel_write[level], reference_write[level], f"write {level}")
        _assert_levels_match(kernel_read[level], reference_read[level], stored=True)
        _assert_levels_match(kernel_write[level], reference_write[level], stored=True)


def _project_under(producer, parameters, x):
    """`RouteProducer._project` alone, under the caller's parameter tensors.

    The seam gate needs the LOGITS the shipped forward computes and nothing that happens
    after them, so the projection is run as its own module rather than reconstructed --
    reconstructing it is how a gate ends up measuring a second implementation.
    """
    from torch.func import functional_call

    class _Projection(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, stream):
            return self.inner._project(stream, None)

    named = {f"inner.{name}": value for name, value in parameters.items()}
    return functional_call(_Projection(producer), named, (x,))


def _logit_floor(params, read64, write64) -> float:
    """The declared class's bound on the logit error: operand term + output term."""
    x = params["x"].double().abs()
    weight = fuse_gain_columns(params["route_W"], params["gain_W"]).double().abs()
    operands = float(torch.einsum("bld,hdc->blhc", x, weight).max())
    scale = max(float(read64.abs().max()), float(write64.abs().max()))
    return 2.0 * 2.0 ** -8 * operands + 2.0 ** -8 * scale


def _production_logits(topology, params):
    """The `(read, write)` logits the shipped producer computes for these parameters."""
    H, hidden_size, _ = params["route_W"].shape
    producer = RouteProducer(topology.levels, hidden_size=hidden_size, num_heads=H)
    parameters = {"route_W": fuse_gain_columns(params["route_W"], params["gain_W"]),
                  "side_gain.gain_bias": params["gain_bias"]}
    read_logits, write_logits, _ = _project_under(producer, parameters, params["x"])
    return read_logits, write_logits


def _flat64(logits: torch.Tensor) -> torch.Tensor:
    """The token-major plane the kernel reads, as the flat fp64 plane a reference takes.

    Both steps are EXACT: a permute moves no bits, and bf16 -> fp64 is a widening.
    """
    return logits.permute(0, 2, 1, 3).reshape(
        logits.shape[0] * logits.shape[2], logits.shape[1], logits.shape[3]).double()


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_producer_production_path_matches_its_oracle_path(monkeypatch, alpha, capsys):
    """`produce_routing` end to end: the shipped path against a wholly fp64 one.

    THIS IS PURPOSE-2 ACCOUNTING, and the doctrine says so. The two arms differ in the
    PROJECTION's operand class (bf16 operands on the tensor cores against fp64), so a
    logit near a threshold can land on either side of it and the support sets are not
    the same object. The flip count is therefore REPORTED, not asserted -- the exact
    claim about the solve lives in
    `test_the_solve_seam_is_exact_on_the_production_logits`, where the reference is fed
    the production logits.

    What IS asserted is the amplitudes at the stored tolerance, and that assertion
    bounds the flips it does not count: a member crossing the threshold enters with an
    amplitude of ~0 (entmax's `((z - tau)/2)^2` is continuous at the boundary), so a
    flip large enough to matter is a flip large enough to fail this.
    """
    topology = _producer_topology(alpha)
    resolved = resolved_routing_from_topology(topology)

    production_params = _producer_params(
        topology, dtype=torch.float32, device="cuda", sides=1)
    oracle_params = {name: tensor.to(torch.float64) for name, tensor in production_params.items()}

    read_logits, write_logits = _production_logits(topology, production_params)
    production = _run_producer(production_params, topology)
    #: The solve is patched WHERE THE PRODUCER RESOLVES IT -- `rola.routing.producer`
    #: binds the name at import -- so the oracle arm really runs the reference solve.
    monkeypatch.setattr(
        production_path, "production_routing_factor_levels", oracle_factor_levels)
    oracle = _run_producer(oracle_params, topology)

    #: THE ENVELOPE RE-DERIVES with the projection's operand class, it is not widened.
    #: The two arms are handed logits differing by the projection's declared class
    #: (`_logit_floor`); `tau` is a mean of supported logits so it moves by at most
    #: the same, and both entmax specializations have `|dp/dz| <= 1` on support
    #: (`d/dz ((z-tau)/2)^2 = sqrt(p) <= 1` at alpha 1.5, `= 1` at alpha 2). So the
    #: amplitude envelope is `2*dz` on top of the stored form's own tolerance --
    #: ON ROWS WHOSE SUPPORT SET DID NOT MOVE.
    #:
    #: A row whose support DID move is excluded from the envelope and counted instead,
    #: and that is not a dodge: `read_simplex()` divides by the row's own mass, so a
    #: member entering at a raw amplitude of ~0 can normalize to any fraction at all when
    #: the whole row's mass is comparably small. The bound on such a row is a bound on
    #: the FLIPPED MASS, which is what the accounting below reports.
    envelope = 2.0 * _logit_floor(production_params, read_logits.double(),
                                  write_logits.double())

    flips = elements = 0
    flipped_mass = 0.0
    for level in range(topology.D):
        for got, want, zeroable in (
            (production.read_levels[level], oracle.read_levels[level],
             resolved.read_level_can_zero[level]),
            (production.write_levels[level], oracle.write_levels[level],
             resolved.write_level_can_zero[level]),
        ):
            want = want.to(got.dtype)
            if not zeroable:
                _assert_levels_match(got, want, stored=True, extra_atol=envelope)
                continue
            moved = (got != 0) != (want != 0)
            flips += int(moved.sum())
            elements += got.numel()
            flipped_mass = max(flipped_mass, float((got * moved).abs().max()),
                               float((want * moved).abs().max()))
            settled = ~moved.any(dim=-1, keepdim=True).expand_as(got)
            _assert_levels_match(got[settled], want[settled], stored=True,
                                 extra_atol=envelope)

    #: `softplus` is 1-Lipschitz and the gain logit is a column of the SAME projection,
    #: so the gain's envelope is `dz` exactly -- half the amplitudes'.
    _assert_levels_match(production.g_write, oracle.g_write, extra_atol=envelope / 2.0)
    assert not hasattr(production, "g_read") and not hasattr(oracle, "g_read")
    with capsys.disabled():
        print(f"\n  FLIP ACCOUNTING alpha={alpha}: {flips} support flips in {elements} "
              f"zero-capable elements ({flips / max(elements, 1):.3e}), greatest flipped "
              f"amplitude {flipped_mass:.4f}, between the shipped bf16-operand projection "
              f"and a wholly fp64 one")


# ---------------------------------------------------------------------------
# V9 (compliance-audit) -- gate gaps named by adversarial review, closed here.
# ---------------------------------------------------------------------------


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "tied", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_the_two_lowerings_agree_directly(alpha, shape, kind):
    """V9 item 2: torch vs the CUDA kernels, DIRECTLY, on the ordinary fixtures.

    Both lowerings are separately gated against the fp64 oracle above, so
    their mutual drift was only TRANSITIVELY bounded, at twice the oracle tolerance
    (producer-port-review Claim 2). Two lowerings can each sit at the edge of that
    envelope on opposite sides and still pass, which is exactly the drift a port
    needs to be gated on. This asserts the pair directly, from the SAME fp32 inputs,
    at the unmodified production tolerances.

    Support is an EXACT set assertion: both are closed forms, so neither has
    convergence slack to hide behind (sparsity comes from entmax, not from a floor -- entmax
    support is a discrete object, not a thresholded one). The exact-TIE fixture is
    excluded because it has its own gate,
    ``test_the_two_lowerings_agree_on_exact_ties``, with its own measured bound.
    """
    routing = _routings(alpha)[shape]
    cuda_read = _stress_logits(routing, kind, device="cuda", seed=17).requires_grad_()
    cuda_write = _stress_logits(routing, kind, device="cuda", seed=18).requires_grad_()
    cpu_read = cuda_read.detach().cpu().requires_grad_()
    cpu_write = cuda_write.detach().cpu().requires_grad_()
    assert torch.equal(cuda_read.detach().cpu(), cpu_read.detach()), (
        "the two lowerings must be handed bit-identical inputs, or this measures the "
        "fixture rather than the port")

    cuda_reads, cuda_writes = production_routing_factor_levels(
        cuda_read, cuda_write, routing, stored_dtype=torch.float32)
    cpu_reads, cpu_writes = production_routing_factor_levels(cpu_read, cpu_write, routing)
    for level in range(routing.D):
        if routing.read_level_can_zero[level]:
            _assert_support_identical(
                cuda_reads[level].cpu(), cpu_reads[level], f"{shape}/{kind} read {level}")
        if routing.write_level_can_zero[level]:
            _assert_support_identical(
                cuda_writes[level].cpu(), cpu_writes[level], f"{shape}/{kind} write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(6607)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator)
        for level in (*cpu_reads, *cpu_writes)
    ]
    cuda_vjp = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        (cuda_read, cuda_write))
    cpu_vjp = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        (cpu_read, cpu_write))
    _assert_levels_match(cuda_vjp[0].cpu(), cpu_vjp[0], gradient=True)
    _assert_levels_match(cuda_vjp[1].cpu(), cpu_vjp[1], gradient=True)


def test_production_imports_nothing_from_the_reference_package():
    """The STRONGER, STATIC claim, checked directly rather than inferred from a
    monkeypatch. `pyproject.toml` and this module's own docstring say production
    "imports nothing" from the reference package -- a claim about the IMPORT GRAPH,
    stronger than and independent of `test_the_solve_never_reaches_the_reference_*`
    above (those prove the three entry points are never CALLED at runtime; a module
    can import a name and still never call it). This is the trivially checkable form
    the adversarial review found asserted nowhere: read `production.py` as an AST and
    confirm no import statement names `rola.routing.entmax.reference` or
    `rola.routing.reference` (the oracle-facing module of the same shape), by module
    or by symbol.
    """
    import ast
    import inspect

    import rola.routing.entmax.production as production_module

    source = inspect.getsource(production_module)
    tree = ast.parse(source, filename=production_module.__file__)
    forbidden_modules = {"rola.routing.entmax.reference", "rola.routing.reference"}
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            offending.append(node.module)
    assert not offending, (
        f"production.py imports from the reference package: {offending} -- the "
        "producer's own contract (module docstring, pyproject.toml) is that it "
        "imports nothing from it")


def test_the_package_imports_no_triton_and_declares_none():
    """P73 the RECEIPT, as a static fact about the tree rather than a claim in a doc.

    `docs/build.md` says the dependencies are "`torch` and `einops`. Not
    `transformers`, not `fla`, not `triton` -- the kernels are CUDA." That sentence
    was FALSE for every level routed `IndependentRouting`, from the module that owned
    the four Triton kernels this stage ported to `csrc/rola/src/entmax/factor.cu`.

    Checked two ways, because they can fail independently: the IMPORT GRAPH (an AST
    walk over every module under `rola/`, so a `triton` reachable at runtime cannot
    hide behind a lazy import inside a function -- `ast.walk` sees those too), and the
    DECLARATION (`pyproject.toml`'s `dependencies`, so a dependency nothing imports
    cannot linger either). Revival of the kernels is git; the ledger row is
    `docs/internals/DELETIONS.md`.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    offending: list[str] = []
    for path in sorted((root / "rola").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "triton" or name.startswith("triton."):
                    offending.append(f"{path.relative_to(root)}:{node.lineno} imports {name}")
    assert not offending, (
        "the package still imports Triton, so `docs/build.md`'s no-triton claim is "
        f"false: {offending}")

    text = (root / "pyproject.toml").read_text()
    declared = text.split("dependencies    = ", 1)[1].split("\n", 1)[0]
    assert "triton" not in declared, (
        f"pyproject.toml still declares a triton dependency: {declared}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "tied", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_the_stored_form_preserves_the_support_exactly(alpha, shape, kind):
    """THE STORAGE BOUNDARY'S ONE OBLIGATION, and it carries no tolerance.

    The solve WRITES bf16 rather than being rounded into it afterwards, so an exact
    structural zero is written as a zero and a supported member is written as a nonzero
    bf16. If either direction failed, the self-masking theorem's precondition
    (`docs/architecture/self-masking-theorem.md`, the storage precondition) would break:
    a manufactured nonzero gives a leaf a clock with no deposit, and a flushed member
    silently leaves the support the routing chose.

    The AMPLITUDES are not compared here -- that is the seam the bf16 budget covers, and
    it is gated in this file's oracle rows. This row is the discrete half alone.
    """
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cuda", seed=17).requires_grad_()
    write = _stress_logits(routing, kind, device="cuda", seed=18).requires_grad_()
    exact_reads, exact_writes = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.float32)
    stored_reads, stored_writes = production_routing_factor_levels(
        read, write, routing, stored_dtype=torch.bfloat16)
    for level in range(routing.D):
        assert stored_reads[level].dtype is torch.bfloat16
        _assert_support_identical(
            stored_reads[level], exact_reads[level], f"{shape}/{kind} read {level}")
        _assert_support_identical(
            stored_writes[level], exact_writes[level], f"{shape}/{kind} write {level}")
