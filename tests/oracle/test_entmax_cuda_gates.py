"""Track E gates: the hand-CUDA union entmax kernels, judged ONLY against the fp64 oracle.

RATIFIED GATE PLAN (user ruling 2026-08-02, recorded in
docs/internals/entmax/entmax.md). The Triton kernel is NOT a
reference here and is never asserted against -- "matching the old is not a virtue"; the
baseline is the independent fp64 ground truth in `entmax/reference.py`, whose own header
says the same thing ("never kernel-vs-kernel"). the matching-the-naive antipattern.

FOUR GATES, split at the seam the fold-vs-solve measurement identified:

* **A1 (the fold)** -- `midpoint = 0.5*read + 0.5*write`, its row-max shift, and the
  overflow-free `delta`, vs a plain fp32 torch expression, at ``rtol=0``. These are pure
  elementwise ops; no reduction and therefore no reassociation is possible, so
  bit-identity is structural rather than lucky.
* **A2 (the solve)** -- the SUPPORT SET vs the fp64 oracle *consuming the fp32
  midpoint*, at ``rtol=0``. Splitting here is not a convenience: measured on 528 hostile
  cells, an fp64-folded oracle disagrees on exactly ONE element, and that element was
  attributed to the fold's 6-ulp fp32 rounding, NOT to the solve. Given
  the kernel's own fp32 midpoint the oracle agrees exactly, 0/528, both fixture classes.
  Support is the product invariant (sparsity comes from entmax, not from a floor).
* **B (values)** -- `midpoint_values`/`read`/`write` vs the fp64 oracle at the repo's
  ratified production tolerance.
* **C (gradients)** -- vs the fp64 oracle's autograd at the ratified gradient tolerance,
  PLUS the dead-directions assertion at ``rtol=0``: off-support logit gradients are
  EXACTLY zero. That is structural (multiplication by ``s`` /
  ``read_values`` / ``write_values``, each exactly zero off support), so it asserts the
  FORM of the VJP, not a magnitude.

I6 ORDERING. These gates are committed BEFORE the kernel, and every one of them is
demonstrated to FAIL against a deliberately perturbed solve -- see the four
``test_teeth_*`` cases. A gate that has never failed is not evidence. The perturbation
chosen for A2 is a support-cardinality off-by-one (the solve picking ``k*+1`` instead of
``k*``), which is exactly the failure mode an fp32 threshold solve can actually have; a
blind "nudge tau by one ulp" usually flips nothing and would prove the gate is asleep.
"""
from __future__ import annotations

import pytest
import torch

from rola.routing.entmax.production import _overflow_free_delta
from rola.routing.entmax.reference import entmax_tau

#: The repo's ratified production tolerances. Values duplicated rather than imported
#: because they are module-private in `tests/integration/test_production_levels.py:55-58`;
#: that file is the provenance and the two must not drift.
_PRODUCTION_RTOL = 2e-5
_PRODUCTION_ATOL = 2e-6
_PRODUCTION_GRAD_RTOL = 5e-5
_PRODUCTION_GRAD_ATOL = 5e-6

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA entmax kernels")

#: Every width class boundary (2,4,8,16,32,64,128,256) AND
#: non-power-of-2 padding inside them (3,7,17,96,129,255). Widths 2-17 are the
#: sub-warp-packed classes, which nothing else in the suite exercises.
_WIDTHS = (2, 3, 4, 7, 8, 16, 17, 32, 64, 96, 128, 129, 255, 256)
#: Ragged token counts: below/at/above one 32-token support word, and a long ragged run.
_TOKENS = (1, 31, 32, 33, 65, 129, 517)
_SEEDS = (0xC0FFEE, 0xBADF00D, 7)
_ALPHAS = (1.5, 2.0)


def _fixture(width, tokens, seed, *, ties, device="cuda", dtype=torch.float32):
    """Hostile read/write logit pair, ``[2, tokens, width]``.

    ``ties=True`` rounds 40% of entries to quarter-integers, manufacturing exact ties so
    the threshold solve's greatest-valid-k selection is genuinely contested; the other
    60% are spread 8x to widen the dynamic range. Every cell carries the saturating
    ``+/-finfo.max``-class pair in slot (0,0,0), which is what exercises the
    overflow-free `delta` fold's clamp.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + width * 131 + tokens * 7)
    shape = (2, tokens, width)
    read = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    write = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    if ties:
        pick = torch.rand(shape, generator=gen, device=device) < 0.4
        read = torch.where(pick, (read * 4).round() / 4, read * 8)
        write = torch.where(pick, (write * 4).round() / 4, write * 8)
    read[0, 0, 0] = 1e30
    write[0, 0, 0] = -1e30
    return read.contiguous(), write.contiguous()


# ---------------------------------------------------------------------------
# A1 -- the fold, in plain fp32 torch. No reduction, so no reassociation exists.
# ---------------------------------------------------------------------------
def fold_reference(read_logits, write_logits):
    """``(midpoint_shifted, delta)`` exactly as the kernel must compute them."""
    midpoint = 0.5 * read_logits + 0.5 * write_logits
    midpoint = midpoint - midpoint.amax(dim=-1, keepdim=True)
    return midpoint, _overflow_free_delta(read_logits, write_logits)


def assert_fold_bit_identical(got_midpoint, got_delta, read_logits, write_logits, ctx=""):
    """GATE A1. rtol=0 -- `torch.equal` on the raw bit patterns."""
    want_midpoint, want_delta = fold_reference(read_logits, write_logits)
    if not torch.equal(got_midpoint, want_midpoint):
        n = int((got_midpoint != want_midpoint).sum())
        raise AssertionError(
            f"GATE A1 (fold) FAILED{ctx}: midpoint differs from the plain fp32 torch "
            f"fold in {n} of {want_midpoint.numel()} entries. This fold is pure "
            f"elementwise arithmetic -- no reduction, no reassociation is possible -- so "
            f"any difference is a real arithmetic divergence, not an ordering artifact.")
    if not torch.equal(got_delta, want_delta):
        n = int((got_delta != want_delta).sum())
        raise AssertionError(
            f"GATE A1 (fold) FAILED{ctx}: overflow-free `delta` differs in {n} entries.")


# ---------------------------------------------------------------------------
# The fp64 oracle, consuming the kernel's own fp32 midpoint
# ---------------------------------------------------------------------------
def oracle_solve(midpoint_fp32, alpha):
    """``(support, values)`` in fp64 from the independent reference.

    The threshold is SOLVED, not approached: the reference inverts it from the
    DeepSPIN package's closed-form values, so there is no iteration depth to choose
    and no question of the oracle being looser than the fp32 solve it judges.
    """
    z = midpoint_fp32.double()
    tau = entmax_tau(z, float(alpha), dim=-1)
    support = z > tau
    base = ((float(alpha) - 1.0) * (z - tau)).clamp(min=0.0)
    values = torch.where(support, base if alpha == 2.0 else base * base,
                         torch.zeros_like(base))
    return support, values


def oracle_union(read_logits, write_logits, alpha):
    """The full fp64 union split: ``(support, midpoint_values, read, write)``.

    Mirrors the production semantics (`_torch_union_split`) in fp64 -- but derived from
    the reference's tau, not from the production solve, so it shares no threshold code
    with the thing under test.
    """
    midpoint, _ = fold_reference(read_logits, write_logits)
    support, values = oracle_solve(midpoint, alpha)
    delta = _overflow_free_delta(read_logits.double(), write_logits.double())
    read = values * torch.sigmoid(delta)
    safe = torch.where(support, values, torch.ones_like(values))
    log_write = (safe.log() + torch.nn.functional.logsigmoid(-delta)).masked_fill(
        ~support, float("-inf"))
    has = support.any(dim=-1, keepdim=True)
    write = torch.softmax(torch.where(has, log_write, torch.zeros_like(log_write)), dim=-1)
    return support, values, read, torch.where(has & support, write, torch.zeros_like(write))


def assert_support_exact(got_support, midpoint_fp32, alpha, ctx=""):
    """GATE A2. rtol=0 vs the fp64 oracle fed the kernel's OWN fp32 midpoint."""
    want, _ = oracle_solve(midpoint_fp32, alpha)
    if not torch.equal(got_support, want):
        bad = (got_support != want).nonzero()
        n = bad.shape[0]
        s, t, c = bad[0].tolist()
        z = midpoint_fp32[s, t, c].double()
        tau = entmax_tau(midpoint_fp32.double(), float(alpha), dim=-1)[s, t, 0]
        ulp = torch.finfo(torch.float32).eps * max(abs(float(z)), 1.0)
        raise AssertionError(
            f"GATE A2 (support) FAILED{ctx}: {n} of {want.numel()} membership bits "
            f"disagree with the fp64 oracle. First at (stream={s}, token={t}, col={c}): "
            f"kernel says in-support={bool(got_support[s, t, c])}, oracle says "
            f"{bool(want[s, t, c])}; z={float(z):.10g} tau={float(tau):.10g}, "
            f"|z-tau| = {abs(float(z) - float(tau)) / ulp:.2f} ulp. Support is the "
            f"product invariant -- this gate is rtol=0 by ruling.")


def assert_values_close(got, want, ctx="", *, gradient=False):
    """GATES B / C -- the ratified production tolerances vs the fp64 oracle."""
    rtol = _PRODUCTION_GRAD_RTOL if gradient else _PRODUCTION_RTOL
    atol = _PRODUCTION_GRAD_ATOL if gradient else _PRODUCTION_ATOL
    torch.testing.assert_close(got.double(), want.double(), rtol=rtol, atol=atol,
                               msg=lambda m: f"GATE {'C' if gradient else 'B'} FAILED{ctx}: {m}")


def assert_off_support_zero(d_read_logits, d_write_logits, support, ctx=""):
    """GATE C's structural half: dead directions are EXACTLY zero, rtol=0.

    Not a magnitude claim. Off support the kernel multiplies by ``s``, ``read_values``
    and ``write_values``, each of which is exactly 0.0 there, so a nonzero here means the
    VJP's multiplicative FORM was broken -- a gradient leaking into a direction the
    forward solve declared dead.
    """
    for name, grad in (("d_read_logits", d_read_logits), ("d_write_logits", d_write_logits)):
        off = grad[~support]
        if off.numel() and bool((off != 0).any()):
            n = int((off != 0).sum())
            raise AssertionError(
                f"GATE C (dead directions) FAILED{ctx}: {name} has {n} NONZERO entries "
                f"off support (max |g| = {float(off.abs().max()):.3e}). Off-support "
                f"gradients must be exactly 0.0 by construction, not merely small.")


# ---------------------------------------------------------------------------
# I6 TEETH -- every gate above is shown to REJECT a deliberately wrong solve.
# These run with no kernel present; they gate the gates.
# ---------------------------------------------------------------------------
@_CUDA
def test_teeth_A1_fold_rejects_one_ulp():
    """A1 must reject a single-ulp perturbation of the fold (it is rtol=0)."""
    read, write = _fixture(64, 65, 7, ties=True)
    midpoint, delta = fold_reference(read, write)
    nudged = midpoint.clone()
    nudged[0, 0, 0] = torch.nextafter(nudged[0, 0, 0], torch.tensor(float("inf"), device=nudged.device))
    with pytest.raises(AssertionError, match="GATE A1"):
        assert_fold_bit_identical(nudged, delta, read, write)
    assert_fold_bit_identical(midpoint, delta, read, write)  # unperturbed passes


@_CUDA
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_teeth_A2_support_rejects_cardinality_off_by_one(alpha):
    """A2 must reject the solve taking ``k*+1``: the largest out-of-support member
    admitted. That is the real failure mode of an fp32 threshold solve; a blind one-ulp
    tau nudge usually flips nothing and would prove only that the gate is asleep."""
    read, write = _fixture(64, 65, 7, ties=True)
    midpoint, _ = fold_reference(read, write)
    support, _ = oracle_solve(midpoint, alpha)
    z = midpoint.double().masked_fill(support, float("-inf"))
    perturbed = support.clone()
    perturbed.scatter_(-1, z.argmax(dim=-1, keepdim=True), True)
    assert not torch.equal(perturbed, support), "the perturbation must actually differ"
    with pytest.raises(AssertionError, match="GATE A2"):
        assert_support_exact(perturbed, midpoint, alpha)
    assert_support_exact(support, midpoint, alpha)  # unperturbed passes


@_CUDA
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_teeth_B_values_reject_beyond_tolerance(alpha):
    """B must reject a perturbation just outside the ratified tolerance, and accept one
    just inside it -- i.e. the tolerance is doing work in both directions."""
    read, write = _fixture(64, 65, 7, ties=True)
    _, values, _, _ = oracle_union(read, write, alpha)
    assert_values_close(values * (1.0 + 0.1 * _PRODUCTION_RTOL), values)
    with pytest.raises(AssertionError, match="GATE B"):
        assert_values_close(values * (1.0 + 10.0 * _PRODUCTION_RTOL) + 10.0 * _PRODUCTION_ATOL,
                            values)


@_CUDA
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_teeth_C_dead_directions_reject_any_leak(alpha):
    """C's rtol=0 half must reject even a denormal-scale leak into a dead direction."""
    read, write = _fixture(64, 65, 7, ties=True)
    midpoint, _ = fold_reference(read, write)
    support, _ = oracle_solve(midpoint, alpha)
    clean = torch.zeros_like(midpoint)
    assert_off_support_zero(clean, clean, support)
    leaked = clean.clone()
    off = (~support).nonzero()
    assert off.shape[0], "fixture must have off-support entries for this to mean anything"
    s, t, c = off[0].tolist()
    leaked[s, t, c] = 1e-38
    with pytest.raises(AssertionError, match="GATE C"):
        assert_off_support_zero(leaked, clean, support)


@_CUDA
def test_teeth_oracle_is_tighter_than_fp32():
    """The oracle must be unambiguously tighter than what it judges, or the gate is
    measuring the oracle. Its own tau must satisfy the entmax normalization in fp64 to
    far better than the fp32 tolerance the values gate uses."""
    read, write = _fixture(128, 65, 7, ties=True)
    midpoint, _ = fold_reference(read, write)
    for alpha in _ALPHAS:
        _, values = oracle_solve(midpoint, alpha)
        mass = values.sum(dim=-1)
        assert float((mass - 1.0).abs().max()) < 1e-12, (
            f"fp64 oracle tau at alpha={alpha} does not normalize to 1 within 1e-12; "
            f"it is not fit to judge an fp32 solve at rtol={_PRODUCTION_RTOL}")
