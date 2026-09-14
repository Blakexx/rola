"""Property-based validation of the ``LeafMassDecay`` contract.

The migration-equivalence suite (``test_rola_v3_migration_equivalence.py``) only
exercises ``decay=None`` positively; its sole ``LeafMassDecay`` test is negative
(records that this clock cannot reproduce the earlier matrix-level decay law -- spec
). As of that suite, the decay path has **no positive validation** at all.

This file closes that gap by testing the recurrence's invariants directly against
``naive_rola`` (and, where noted, its own leaf-materializing primitive
``_leaf_product``) -- never against a second, independently written reference
implementation of the recurrence. A second reference would only prove
self-consistency between two implementations of the same idea; these tests
instead pin down algebraic identities of the one authorized oracle:

    c[t,s]    = stop_gradient(prod_l p_write_stored[l,t,d_l(s)])
    rate[s]   = prod_l delta_l[d_l(s)]
    keep[t,s] = (1 - rate[s]) ** c[t,s]
    M[t,s,:]  = keep[t,s] * M[t-1,s,:] + W[t,s] * v[t,:]

Covered:
    - write-conditioned quiescence: a leaf with zero stored write factor at
      some level has keep == 1 exactly and an unchanged (bit-identical) state
      across that step;
    - keep > 0 always under 0 <= delta < 1 (probed near delta -> 1), and the
      config contract's own loud rejection of delta >= 1 (no clamping);
    - c == 1 (complete unit write) gives keep == 1 - rate exactly;
    - rate == 0 gives keep == 1 for any c;
    - monotonicity of keep in c and in rate;
    - affine composability: splitting a sequence and continuing
      the recurrence from the boundary state reproduces the unsplit result,
      exactly, across several split points -- this is the practical form of
      `` o = (A2*A1, A2*B1 + B2)`` since the oracle's own
      state-continuation *is* that composition (the recurrence is affine in
      ``M``, the "no delta correction" consequence);
    - the same ``0 <= delta < 1`` invariant re-checked in the dtype the KERNEL
      consumes: ``rola.ops.decay.leaf_rate_dials`` casts to fp32, where a
      value strictly below 1 in fp64 can round to exactly ``1.0f``.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.decay import leaf_rate_dials
from rola.ops.naive import _leaf_product, naive_rola
from rola.ops.paging import bytes_equal
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
)


def _topology(widths) -> Topology:
    routing = IndependentRouting(
        width=1, read=SoftmaxActivation(), write=SoftmaxActivation())
    return Topology(levels=tuple(routing.at(w) for w in widths))


def _one_hot(width: int, digit: int, *, dtype=torch.float64) -> torch.Tensor:
    vec = torch.zeros(width, dtype=dtype)
    vec[digit] = 1.0
    return vec


def _simplex(width: int, *, seed: int, dtype=torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(width, generator=gen, dtype=dtype)
    return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 1. Write-conditioned quiescence: state bit-identical across a write-free
#    interval, which is only possible if keep == 1 exactly throughout it
#    ("Survival within a write-free interval is identically 1").
# ---------------------------------------------------------------------------


def test_write_conditioned_quiescence_state_bit_identical_across_write_free_interval():
    # D=2, widths (3,2); one head, one batch, d_v=2, T=5.
    B, H, dv, T = 1, 1, 2, 5
    widths = (3, 2)
    levels = [
        IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation())
        for w in widths
    ]
    topology = Topology(levels=tuple(levels))

    # Leaf under test: level0 digit 0, level1 digit 0 -> leaf index 0.
    # Written (write factor 1 at every level, i.e. c=1) at tokens 0 and 4 only;
    # zero write mass at level0's digit 0 at tokens 1,2,3 forces c=0 there
    # regardless of level1, since c is a product over levels.
    write_l0 = torch.stack(
        [
            _one_hot(3, 0, dtype=torch.float64),  # t=0: leaf written
            torch.tensor([0.0, 0.6, 0.4], dtype=torch.float64),  # t=1: digit0 factor 0
            torch.tensor([0.0, 0.3, 0.7], dtype=torch.float64),  # t=2
            torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64),  # t=3
            _one_hot(3, 0, dtype=torch.float64),  # t=4: leaf written again
        ]
    )
    write_l1 = torch.stack([_one_hot(2, 0, dtype=torch.float64) for _ in range(T)])
    write_levels = (
        write_l0.view(B, T, H, 3),
        write_l1.view(B, T, H, 2),
    )
    read_l0 = torch.stack([_simplex(3, seed=100 + t) for t in range(T)]).view(B, T, H, 3)
    read_l1 = torch.stack([_simplex(2, seed=200 + t) for t in range(T)]).view(B, T, H, 2)
    read_levels = (read_l0, read_l1)

    v = torch.randn(B, T, H, dv, dtype=torch.float64)
    g_write = torch.rand(B, T, H, dtype=torch.float64) + 0.5

    dial0 = torch.tensor([[0.9, 0.5, 0.2]], dtype=torch.float64)  # [H, 3]
    dial1 = torch.tensor([[0.8, 0.1]], dtype=torch.float64)  # [H, 2]
    decay = LeafMassDecay(dials=(dial0, dial1))

    states = []
    for prefix in range(1, T + 1):
        _, final_state = naive_rola(
            v[:, :prefix], (read_levels[0][:, :prefix], read_levels[1][:, :prefix]),
            (write_levels[0][:, :prefix], write_levels[1][:, :prefix]),
            g_write[:, :prefix], topology, decay,
            output_final_state=True,
        )
        states.append(final_state[0, 0, 0, :].clone())  # leaf 0's value state

    # Leaf 0 is write-conditioned-quiescent at tokens 1,2,3 (c==0 there): its
    # state after token 0 must equal its state after tokens 1, 2, and 3,
    # bit-for-bit -- not merely close.
    state_after_t0 = states[0]
    for t in (1, 2, 3):
        assert bytes_equal(states[t], state_after_t0), (
            f"leaf 0's state changed at write-free token {t}: "
            f"{states[t]} != {state_after_t0} (quiescence violated)"
        )
    # And it must actually change again once genuinely re-written at token 4.
    assert not bytes_equal(states[4], state_after_t0)


def test_write_conditioned_quiescence_keep_equals_one_exactly_via_leaf_product():
    """Directly pins ``keep = (1-rate)**c == 1`` when ``c == 0``, using the
    oracle's own leaf-materializing primitive (``_leaf_product``), not a
    reimplementation of it. Probes ``rate`` across the full reachable range
    (including near-1, the ``delta -> 1`` edge) to show ``c == 0`` alone
    forces ``keep == 1`` regardless of how large the decay rate is.
    """
    strides = (1,)
    N = 4
    for rate_value in (0.0, 0.5, 1 - 1e-9, 1 - 1e-15):
        dial = (torch.full((1, 4), rate_value, dtype=torch.float64),)
        rate = _leaf_product(dial, strides, N)  # [1,4], one rate per leaf (BH=1)
        c = torch.zeros((1, 1, N), dtype=torch.float64)  # c==0 at every leaf
        keep = torch.pow(1.0 - rate[:, None, :], c)
        assert torch.equal(keep, torch.ones_like(keep)), (
            f"keep != 1 exactly at c=0 for rate={rate_value}: {keep}"
        )


# ---------------------------------------------------------------------------
# 2. keep > 0 always under 0 <= delta < 1 (probed near delta -> 1); and the
#    config contract's own loud rejection of the unreachable boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta_value", [0.0, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-12])
def test_keep_is_strictly_positive_near_the_delta_to_one_edge(delta_value):
    strides = (1,)
    N = 2
    dial = (torch.full((1, 2), delta_value, dtype=torch.float64),)
    rate = _leaf_product(dial, strides, N)
    for c_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = torch.full((1, 1, N), c_value, dtype=torch.float64)
        keep = torch.pow(1.0 - rate[:, None, :], c)
        assert torch.all(keep > 0), f"keep <= 0 at delta={delta_value}, c={c_value}: {keep}"


@pytest.mark.parametrize("bad_delta", [1.0, 1.0 + 1e-9, 1.5, -1e-9])
def test_leaf_mass_decay_rejects_delta_outside_the_strict_open_interval(bad_delta):
    """``0 <= delta < 1`` is strict and load-bearing: confirms the
    config contract raises loudly rather than clamping ``delta == 1`` (which
    would make ``rate == 1`` reachable and ``keep == 0`` reachable, breaking
    the "keep=0 unreachable by construction" invariant the kernel design
    relies on).
    """
    with pytest.raises(ValueError):
        LeafMassDecay(dials=(torch.full((1, 3), bad_delta, dtype=torch.float64),))


# ---------------------------------------------------------------------------
# 3. c == 1 (complete unit write) gives keep == 1 - rate exactly.
# ---------------------------------------------------------------------------


def test_complete_write_gives_keep_equals_one_minus_rate_exactly():
    # D=1, width=3, single leaf under test = digit 0. Seed a nonzero initial
    # state at that leaf via `initial_state`, then take one step where the
    # leaf is completely written (c=1: write factor 1 at the only level).
    # M1 = keep*M0 + W1*v1  =>  keep = (M1 - W1*v1) / M0, solved from the
    # oracle's own output, and compared against 1-rate computed from the same
    # dial via the oracle's own `_leaf_product`.
    B, H, dv, T = 1, 1, 1, 1
    width = 3
    delta0 = 0.37
    topology = Topology(
        levels=(IndependentRouting(width=width, read=SoftmaxActivation(), write=SoftmaxActivation()),),
    )
    dial = torch.tensor([[delta0, 0.6, 0.05]], dtype=torch.float64)
    decay = LeafMassDecay(dials=(dial,))

    m0 = 5.0
    initial_state = torch.zeros(B, H, width, dv + 1, dtype=torch.float64)  # +1: global's mass column
    initial_state[0, 0, 0, 0] = m0

    write_levels = (_one_hot(width, 0).view(1, 1, 1, width),)  # leaf 0 fully written
    read_levels = (_one_hot(width, 0).view(1, 1, 1, width),)  # irrelevant to state, only to y
    v = torch.tensor([[[[2.0]]]], dtype=torch.float64)  # v1 = 2.0
    g_write_value = 1.7
    g_write = torch.full((B, T, H), g_write_value, dtype=torch.float64)

    _, final_state = naive_rola(
        v, read_levels, write_levels, g_write, topology, decay,
        initial_state=initial_state, output_final_state=True,
    )
    m1 = final_state[0, 0, 0, 0].item()
    w1 = g_write_value * 1.0  # write factor is exactly 1 (one-hot)
    keep_recovered = (m1 - w1 * 2.0) / m0

    rate = _leaf_product((dial,), topology.radix_strides, topology.N)[0, 0].item()
    expected_keep = 1.0 - rate
    assert keep_recovered == pytest.approx(expected_keep, rel=0, abs=1e-12), (
        f"recovered keep {keep_recovered} != 1-rate {expected_keep} at c=1"
    )


# ---------------------------------------------------------------------------
# 4. rate == 0 gives keep == 1 for any c.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_value", [0.0, 0.1, 0.37, 0.6, 0.9, 1.0])
def test_zero_rate_gives_keep_equals_one_for_any_c(c_value):
    # rate==0 requires delta==0 at the addressed digit (allowed: 0 <= delta).
    B, H, T = 1, 1, 2
    width = 2
    topology = Topology(
        levels=(IndependentRouting(width=width, read=SoftmaxActivation(), write=SoftmaxActivation()),),
    )
    dial = torch.tensor([[0.0, 0.8]], dtype=torch.float64)  # rate at digit0 == 0
    decay = LeafMassDecay(dials=(dial,))

    # write factor at digit0 == c_value (partial write mass), rest at digit1.
    write_l0 = torch.tensor(
        [[c_value, 1 - c_value], [c_value, 1 - c_value]], dtype=torch.float64
    ).view(B, T, H, width)
    read_l0 = write_l0.clone()
    v = torch.tensor([[[[3.0]], [[7.0]]]], dtype=torch.float64)
    g_write = torch.ones(B, T, H, dtype=torch.float64)

    _, final_state = naive_rola(
        v, (read_l0,), (write_l0,), g_write, topology, decay,
        output_final_state=True,
    )
    # With keep==1 always at digit0 (rate==0 regardless of c), the leaf-0
    # value state is a plain running sum: M = c_value*3.0 + c_value*7.0.
    expected = c_value * 3.0 + c_value * 7.0
    m_final = final_state[0, 0, 0, 0].item()
    assert m_final == pytest.approx(expected, rel=0, abs=1e-12), (
        f"expected pure accumulation {expected} (keep==1 at rate==0), got {m_final}"
    )


# ---------------------------------------------------------------------------
# 5. Monotonicity of keep in c and in rate.
# ---------------------------------------------------------------------------


def test_keep_is_nonincreasing_in_c():
    strides = (1,)
    N = 5
    dial = (torch.tensor([[0.1, 0.3, 0.6, 0.8, 0.95]], dtype=torch.float64),)
    rate = _leaf_product(dial, strides, N)  # [1,5]
    c_values = torch.linspace(0.0, 1.0, steps=11, dtype=torch.float64)
    keeps = torch.stack(
        [torch.pow(1.0 - rate, c.expand_as(rate)) for c in c_values], dim=0
    )  # [11, 1, 5]
    diffs = keeps[1:] - keeps[:-1]
    assert torch.all(diffs <= 1e-15), f"keep increased as c grew: {diffs}"


def test_keep_is_nonincreasing_in_rate():
    N = 1
    strides = (1,)
    rate_values = torch.linspace(0.0, 1 - 1e-9, steps=11, dtype=torch.float64)
    dials = (torch.stack([torch.full((1,), r, dtype=torch.float64) for r in rate_values], dim=0),)
    # dials[0] has shape [11, 1]; treat the 11 "heads" as independent probe points.
    rate = _leaf_product((dials[0],), strides, N)  # [11, 1]
    for c_value in (0.2, 0.5, 0.8, 1.0):
        c = torch.full_like(rate, c_value)
        keeps = torch.pow(1.0 - rate, c).squeeze(-1)  # [11]
        diffs = keeps[1:] - keeps[:-1]
        assert torch.all(diffs <= 1e-15), (
            f"keep increased as rate grew (c={c_value}): {diffs}"
        )


# ---------------------------------------------------------------------------
# 6. Affine composability: split-then-continue reproduces the
#    unsplit result exactly. Load-bearing for sequence parallelism.
# ---------------------------------------------------------------------------


def _random_case(seed: int, *, B, T, H, dv, widths):
    gen = torch.Generator().manual_seed(seed)

    def simplex(width):
        logits = torch.randn(B, T, H, width, generator=gen, dtype=torch.float64)
        return torch.softmax(logits, dim=-1)

    levels = [
        IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation())
        for w in widths
    ]
    topology = Topology(levels=tuple(levels))
    read_levels = tuple(simplex(w) for w in widths)
    write_levels = tuple(simplex(w) for w in widths)
    v = torch.randn(B, T, H, dv, generator=gen, dtype=torch.float64)
    g_write = torch.rand(B, T, H, generator=gen, dtype=torch.float64) + 0.5
    dials = tuple(
        (torch.rand(H, w, generator=gen, dtype=torch.float64) * 0.9) for w in widths
    )
    decay = LeafMassDecay(dials=dials)
    return topology, read_levels, write_levels, v, g_write, decay


@pytest.mark.parametrize("split_point", [1, 2, 4, 6])
def test_split_then_continue_reproduces_unsplit_result_exactly(split_point):
    B, T, H, dv, widths = 2, 7, 2, 3, (3, 4)
    topology, read_levels, write_levels, v, g_write, decay = _random_case(
        seed=42, B=B, T=T, H=H, dv=dv, widths=widths
    )

    y_full, state_full = naive_rola(
        v, read_levels, write_levels, g_write, topology, decay,
        output_final_state=True,
    )

    def _slice(levels, lo, hi):
        return tuple(level[:, lo:hi] for level in levels)

    y1, state1 = naive_rola(
        v[:, :split_point], _slice(read_levels, 0, split_point), _slice(write_levels, 0, split_point),
        g_write[:, :split_point], topology, decay,
        output_final_state=True,
    )
    y2, state2 = naive_rola(
        v[:, split_point:], _slice(read_levels, split_point, T), _slice(write_levels, split_point, T),
        g_write[:, split_point:], topology, decay,
        initial_state=state1, output_final_state=True,
    )

    y_chunked = torch.cat([y1, y2], dim=1)
    assert torch.equal(y_full, y_chunked), (
        f"chunked output diverges from unsplit at split_point={split_point}: "
        f"max abs diff {(y_full - y_chunked).abs().max().item()}"
    )
    assert bytes_equal(state_full, state2), (
        f"chunked final state diverges from unsplit at split_point={split_point}: "
        f"max abs diff {(state_full - state2).abs().max().item()}"
    )


def test_affine_composition_algebra_matches_direct_recurrence():
    """Pure-arithmetic check of `` o = (A2*A1, A2*B1+B2)``
    for a single scalar leaf, independent of the oracle: builds a
    known ``keep``/``W*v`` sequence, computes the direct recurrence, then
    computes it via per-chunk ``(A,B)`` summaries composed with the stated
    rule, and checks they agree exactly (associativity holds regardless of
    where the sequence is cut).
    """
    keep_seq = [0.9, 0.4, 1.0, 0.7, 0.2, 0.6]
    wv_seq = [1.0, 0.0, 3.0, 0.0, 2.0, 5.0]  # W[t]*v[t]
    m0 = 2.5

    def direct(keep_seq, wv_seq, m_init):
        m = m_init
        for keep, wv in zip(keep_seq, wv_seq):
            m = keep * m + wv
        return m

    def chunk_summary(keep_seq, wv_seq):
        """Return (A,B) with M_end = A*M_0 + B for this sub-sequence."""
        A, B_ = 1.0, 0.0
        for keep, wv in zip(keep_seq, wv_seq):
            A = keep * A
            B_ = keep * B_ + wv
        return A, B_

    def compose(ab1, ab2):
        a1, b1 = ab1
        a2, b2 = ab2
        return a2 * a1, a2 * b1 + b2

    m_direct_full = direct(keep_seq, wv_seq, m0)

    for split in range(1, len(keep_seq)):
        ab1 = chunk_summary(keep_seq[:split], wv_seq[:split])
        ab2 = chunk_summary(keep_seq[split:], wv_seq[split:])
        a_c, b_c = compose(ab1, ab2)
        m_composed = a_c * m0 + b_c
        assert m_composed == pytest.approx(m_direct_full, rel=0, abs=1e-15), (
            f"composed (A,B) at split={split} gives {m_composed}, direct recurrence gives "
            f"{m_direct_full}"
        )


# ---------------------------------------------------------------------------
# 7. The dial invariant AT THE CAST BOUNDARY the kernel reads it across
#     (`rola.ops.decay.leaf_rate_dials`; docs/internals/decode/decode.md)
# ---------------------------------------------------------------------------


def test_leaf_rate_dials_rejects_fp64_dial_that_rounds_to_1_0f():
    """B1 cliff, reject side: an fp64 dial in ``[1 - 3e-8, 1)`` passes
    ``LeafMassDecay``'s own-dtype check (strictly ``< 1`` in fp64) but rounds to
    exactly ``1.0f`` under the ``.to(torch.float32)`` cast the kernel consumes,
    which NaNs the kernel (``log1pf(-1.0f) = -inf``, then ``expf(0 * -inf)`` for
    any leaf with a zero clock). ``leaf_rate_dials``'s cast-boundary re-check
    must reject it loudly rather than let it through to the kernel."""
    widths = (4, 4)
    topology = _topology(widths)
    dial_val = 1.0 - 1e-9  # strictly < 1 in fp64
    assert float(torch.tensor(dial_val, dtype=torch.float64).float()) == 1.0, (
        "fixture assumption broken: this fp64 value no longer rounds to 1.0f")
    dials = tuple(torch.full((2, w), dial_val, dtype=torch.float64) for w in widths)
    decay = LeafMassDecay(dials=dials)  # passes the fp64 __post_init__ check
    with pytest.raises(ValueError, match="1\\.0f"):
        leaf_rate_dials(decay, topology)


def test_leaf_rate_dials_accepts_fp64_dial_that_survives_the_cast():
    """B1 cliff, survive side: a dial far enough from 1 (``1 - 1e-6``) rounds to
    a value strictly less than ``1.0f``, so the cast-boundary check must not
    raise, and the flattened dial tensor must be usable by the kernel."""
    widths = (4, 4)
    topology = _topology(widths)
    dial_val = 1.0 - 1e-6
    assert float(torch.tensor(dial_val, dtype=torch.float64).float()) < 1.0, (
        "fixture assumption broken: this fp64 value no longer survives the fp32 cast")
    dials = tuple(torch.full((2, w), dial_val, dtype=torch.float64) for w in widths)
    decay = LeafMassDecay(dials=dials)
    flat = leaf_rate_dials(decay, topology)
    assert torch.all(flat < 1.0)
