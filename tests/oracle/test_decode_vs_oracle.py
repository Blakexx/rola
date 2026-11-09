# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""TIER 1 — the T=1 decode path against the canonical fp64 oracle.

Gates G1-G7 of the decode design (docs/internals/decode/decode.md).

**WHY "BIT-IDENTITY VS THE PREFILL CONSUMER" IS NOT ONE OF THEM, and this has to be
said before the tolerances make sense.** Three independent facts make byte-equality
between decode and the prefill kernel unattainable, so a gate asking for it would be
asking for something no implementation can deliver:

1. The prefill consumer's ``y`` is not reproducible against ITSELF wherever a ``bh``
   holds more than one owner block — it fans partials in by ``atomicAdd`` and CUDA
   fixes no inter-CTA order (Track K measured 384 of 976 fixture cases differing
   between two runs of one binary). Decode's ``y`` is summed the same way now: its warps
   add their partials into a batch-head's sums atomically, in whatever order the work
   queue hands out positions (Blake, 2026-09-29: determinism is not a requirement).
2. Summation order differs by construction: the prefill kernel contracts inside an
   MMA fragment tree per owner then sums owners by atomic; decode contracts in leaf
   order in fp32 FMA chains and adds warps' partials by atomic.
3. The precision path differs: the prefill readout narrows its operands to bf16 for
   the MMA; decode has no MMA and stays fp32. Decode is the *more* accurate of the
   two, which is the safe direction but is still a difference.

So the gates are: byte-equality where decode is *supposed* to be exact -- the STATE, which
no reduction touches (G1, G2, the BC-blindness row in tier 2) -- and the ORACLE band
everywhere else, ``y`` included.

**P67 D2-b / K31.** The prefill leg here is the fp64 ORACLE now: the tiled consumer
left at D2, and the chunk arm left with the K31 deletion batch (baseline = tag
``baseline/pre-k31``).
That took one gate with it and improved another:

* **G5** — decode against the TILED consumer at ``T = 1`` — RETIRES. Its subject was
  the second prefill kernel as a cross-check, and the retired chunk arm could not run ``T = 1``
  at all (``T`` had to be a whole number of its 32-token chunks). What that gate
  actually established beyond the oracle rows was consumer-to-consumer agreement, and
  after D2 there is one prefill consumer.
* **G7** — the prefill/decode seam — is now graded against the fp64 ORACLE over the
  whole ``L + 1`` sequence rather than against a second prefill of ``L + 1``. It has
  to be: ``L`` and ``L + 1`` cannot both be chunk multiples. The replacement is an
  INDEPENDENT reference for the seam instead of the same family of kernel twice,
  which is the direction the standing rule points.

Run: ``bazel test //tests:oracle --test_filter=test_decode_vs_oracle``
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from rola.ops.decode import _decode_step, derive_decode_geometry
from rola.ops.lattice import permutation, to_canonical, to_lattice
from rola.ops.paging import bytes_equal
from rola.routing import quant
from rola.routing.factors import RouteFactors
from rola.routing.types import LeafMassDecay
from rola.values import Values
from tests.oracle.fixtures import (
    assert_planted_errors_fail,
    assert_slots_close,
    bundle_operands,
    logical_state,
    oracle_run,
    state_pages,
    state_quantum,
    state_rounded,
    step_bundles,
    with_gain,
)
from tests.oracle.oracle_fixtures import _decay_dials, _simplex, _topology
from tests.oracle.tolerances import DECODE_STATE, DECODE_Y

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]


def _chunk_tokens() -> int:
    """The oracle prefill's length unit, in tokens."""
    return 32


#: THE TOPOLOGY AXIS, written out rather than reduced because the lattice ``(k, m)`` each
#: row derives is what the walk is, and the axis has to carry every shape of it:
#:
#:   (16,)         D = 1, s = (16,)          -- the whole level is one owner run
#:   (33,)         D = 1, k = 1              -- no k divides it: every leaf its own owner
#:   (8, 16)       s = (8, 16)               -- unequal per-level shares, pi NOT the identity
#:   (7, 9)        k = 1                     -- coprime widths, N = 63
#:   (4, 4, 4)     s = (4, 4, 4)             -- the sub-box IS the whole level product
#:   (3, 5, 7)     k = 1                     -- D = 3 with no admissible span
#:   (2, 3, 4, 4)  s = (1, 1, 2, 4)          -- D = 4, capacity without a sub-box
#:   (64, 64)          s = (8, 16)          -- the flagship box, N = 4096
#:   (16, 16, 16)      s = (2, 4, 16)        -- the deep-3 box, atom = ONE level
#:   (16, 16, 16, 16)  s = (2, 2, 2, 16)     -- the deep-4 box, N = 65536; atom = ONE level too,
#:                                              same as the other two rows (K53: below the
#:                                              width-16 floor the deep-4 box's innermost span
#:                                              was narrower than the atom and spanned two levels
#:                                              -- (8, 8, 8, 8) is unlawful now, and moving its
#:                                              width to the floor also moves this row's span
#:                                              to exactly the atom)
#:
#: The order is positional in the parametrize ids and does not change.
#: PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29; K53): every row here is a
#: descriptor `StateFormat`/decode's own `_admit` both accept -- depth two, three and four,
#: each with every level at or above the floor. The ragged ones this list used to carry --
#: (33,), (7, 9), (3, 5, 7), (2, 3, 4, 4) -- are gone with the ragged state: no descriptor can
#: name them.
_TOPOLOGIES = [(64, 64), (16, 16, 16), (16, 16, 16, 16)]


def _lattice_mask(mask, config):
    """A ``[BH, N]`` CANONICAL leaf mask, re-read in LATTICE order."""
    pi = permutation(config.widths, config.lattice_k, config.lattice_m, device=mask.device)
    out = torch.empty_like(mask)
    out.index_copy_(1, pi, mask)
    return out


def _leaf_support(levels, widths, B, H, device):
    """``[BH, N]`` bool: leaf ``s`` is live iff EVERY level's digit is live (``RouteFactors.live``).

    This is `schedule.py`'s own definition of the support bitmap
    (``read_mask = tuple(level != 0 ...)``) taken to leaf granularity, written here in
    torch so the kernel's expansion is checked against the definition rather than
    against a second copy of the kernel.
    """
    D = len(widths)
    N = math.prod(widths)
    strides = [math.prod(widths[l + 1:]) for l in range(D)]
    idx = torch.arange(N, device=device)
    mask = torch.ones(B * H, N, dtype=torch.bool, device=device)
    for level, width in enumerate(widths):
        digit = (idx // strides[level]) % width
        folded = levels[level].permute(0, 2, 1, 3).reshape(B * H, width)
        mask &= folded[:, digit] != 0
    return mask


def _kernel_realized_support(routes, side, widths, B, H, device, d_v=64):
    """The step kernel's OWN touched leaf set, as ``[BH, N]`` bool in LATTICE order.

    The walk is ONE device path and it does not know which side it is serving, so the
    amplitudes go in through the WRITE port, where the leaf set it reaches is exactly the
    set of state rows the step deposits into. Membership is then read off the STATE DIFF
    -- observable at any topology, and bit for bit, which the kernel's internal factors
    are not.

    The read side is left empty so the walk's union is the set under test, the state
    starts at zero and ``v`` is all ones, so a deposited row is a nonzero row and the
    correspondence is exact rather than statistical. ``side``'s rows are the bundle's own stored rows, carried with
    its sparse flags (a dense level's zeros read as the floor), so the support is the one the kernel is handed.
    """
    topology = _topology(widths)
    N = math.prod(widths)
    config = derive_decode_geometry(topology, d_v=d_v, decay=False)
    f32 = dict(device=device, dtype=torch.float32)
    ones = torch.ones(B, 1, H, **f32)
    rows = routes.read_rows if side == "read" else routes.write_rows
    #: the read side is empty, but it must declare a dense level: level 0, which the probe ignores (the state diff
    #: shows only what the step WROTE, and a read deposits nothing), the others sparse and empty
    read_flags = (False,) + (True,) * (len(widths) - 1)
    flags = routes.sparse[0] if side == "read" else routes.sparse[1]
    #: a dense level is f16 at the floor on either side: the rows move to the write side in their own form
    empty = tuple(quant.aligned_rows(*r.shape, r.device, quant.ROW_DTYPE if sparse else quant.DENSE_DTYPE)
                  for r, sparse in zip(rows, read_flags))
    routes = RouteFactors.from_rows(topology, empty, rows, (ones,) * len(widths), (ones,) * len(widths),
                                    sparse=(read_flags, flags))
    values = Values.from_projection(torch.ones(B, 1, H, d_v, **f32))
    state = state_pages(torch.zeros(B, H, N, config.cols, **f32))
    _y, state, _ws = _decode_step(routes, values, config, state)
    return (logical_state(state, d_v) != 0).any(-1)


def _case(widths, d_v, B, H, norm, p_read, p_write, *, seed, decay_dials=None,
          ctas=None, nonzero_m0=True, steps=1, flushed=False):
    """Build one fixture, run the oracle and decode, return everything both produced."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths)
    N = math.prod(widths)
    cols = d_v + (1 if norm == "global" else 0)

    bundles, drawn_write = [], []
    for _ in range(steps):
        read = tuple(_simplex((B, 1, H, w), p_read, generator, device) for w in widths)
        write = tuple(_simplex((B, 1, H, w), p_write, generator, device) for w in widths)
        if flushed:
            write = tuple(_flush_some(t, generator) for t in write)
        drawn_write.append(write)
        g_write = torch.rand(B, 1, H, device=device, dtype=torch.float64, generator=generator) + 0.5
        v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float64, generator=generator)
        #: ONE representation, and it is the STORED one: the kernel reads the bundle and the oracle its values.
        bundles.append(step_bundles(v, read, with_gain(write, g_write), topology))

    initial_state = None
    if nonzero_m0:
        initial_state = 0.1 * torch.randn(B, H, N, cols, device=device, dtype=torch.float64,
                                          generator=generator)
        if norm == "global":
            initial_state[..., d_v] = initial_state[..., d_v].abs()
        #: the entry state IS a stored one: both sides start from the int8 format's values.
        initial_state = state_rounded(initial_state)

    decay = None if decay_dials is None else LeafMassDecay(dials=decay_dials)

    # ---- the oracle, chained one token at a time (which is what decode does) ----
    #: each step's readout and its envelope; the states are carried, not kept (64 fp64 planes at N = 65536 would not fit).
    y_refs, ref = [], initial_state
    for routes, values in bundles:
        ref = oracle_run(*bundle_operands(routes, values), topology, decay, entry=ref)
        y_refs.append((ref.y, ref.y_envelope))
        #: THE DECLARED STORE, mirrored: the step rounds every row it wrote to its group's s24 quantum.
        ref = ref._replace(state=state_rounded(ref.state))

    # ---- decode ---------------------------------------------------------------
    config = derive_decode_geometry(topology, d_v=d_v, decay=decay is not None)
    if ctas is not None:
        config = dataclasses.replace(config, ctas=ctas)
    #: THE SEAM. The oracle's plane is canonical forever; the kernel's is lattice-ordered.
    #: THE STORED FORM IS THE INT8 STATE PAGE: the seam packs after the lattice
    #: permutation and unpacks before the inverse, because a page is 16 leaves OF THAT
    #: ORDER and the kernel reads the bytes, not the logical view.
    state = state_pages(
        torch.zeros(B, H, N, cols, device=device, dtype=torch.float32)
        if initial_state is None
        else to_lattice(initial_state.to(torch.float32), widths,
                        config.lattice_k, config.lattice_m))
    workspace = None
    y_outs = []
    for routes, values in bundles:
        y, state, workspace = _decode_step(routes, values, config, state, decay=decay, workspace=workspace)
        y_outs.append(y)
    state = to_canonical(logical_state(state, d_v).reshape(B, H, N, cols), widths, config.lattice_k, config.lattice_m)

    return dict(y=y_outs, state=state, y_refs=y_refs, ref=ref, config=config, workspace=workspace,
                routes=[routes for routes, _ in bundles], drawn_write=drawn_write,
                widths=widths, B=B, H=H, N=N, device=device)


def _flush_some(level, generator):
    """A simplex with a third of its live non-max digits moved below half a u8 step of the row's max, so the stored
    row holds a zero where the drawn level does not."""
    peak = level.amax(-1, keepdim=True)
    tiny = (level > 0) & (level < peak) & (torch.rand(level.shape, device=level.device, generator=generator) < 1 / 3)
    level = torch.where(tiny, peak * 1e-4, level)
    return level / level.sum(-1, keepdim=True)


def _assert_oracle(out):
    """Every step's ``y`` and the final state, per slot against the chained oracle's envelopes."""
    for step, (y, (y_ref, y_envelope)) in enumerate(zip(out["y"], out["y_refs"])):
        assert_slots_close(y, y_ref, output=DECODE_Y, envelope=y_envelope, what=f"decode y at step {step}")
    #: the store's own rounding, on both sides from one fp32 update apart: a tie may land a quantum away.
    assert_slots_close(out["state"], out["ref"].state, output=DECODE_STATE, envelope=out["ref"].state_envelope,
                       what="decode state", derived_atol=2 * state_quantum(out["ref"].state))


# ---------------------------------------------------------------------------
# G3 -- the support authority, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g3_expansion_is_the_exact_leaf_support(widths, p_read, p_write):
    """The realized touched set IS the per-level exact-nonzero support, leaf for leaf.

    This is the one place the decode path's "membership is ``amp != 0``" claim is
    checked rather than argued, and the one place ``pi`` is gated end to end: the
    canonical support is derived from the level definition, permuted into lattice order,
    and matched bit for bit against the rows the kernel's own lattice walk moved. A leaf
    the walk wrongly includes may carry a zero amplitude and contribute nothing to ``y``,
    so nothing downstream would necessarily notice.

    Both sides go through the SAME device path, so both are stated here.
    """
    out = _case(widths, 64, 2, 3, "global", p_read, p_write, seed=17)
    B, H, device, config = out["B"], out["H"], out["device"], out["config"]
    routes = out["routes"][0]
    want_r = _lattice_mask(_leaf_support(routes.live("read"), widths, B, H, device), config)
    want_w = _lattice_mask(_leaf_support(routes.live("write"), widths, B, H, device), config)
    assert torch.equal(_kernel_realized_support(routes, "read", widths, B, H, device),
                       want_r), "the read walk is not the exact read support"
    assert torch.equal(_kernel_realized_support(routes, "write", widths, B, H, device),
                       want_w), "the write walk is not the exact write support"


# ---------------------------------------------------------------------------
# G4 -- the fp64 oracle, over the instantiation matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_matches_the_canonical_oracle(widths, p_read, p_write):
    _assert_oracle(_case(widths, 64, 2, 3, "global", p_read, p_write, seed=1234))


@pytest.mark.parametrize("d_v", [32, 64])
def test_g4_both_value_widths(d_v):
    _assert_oracle(_case((64, 64), d_v, 2, 2, "global", 0.5, 0.5, seed=77))


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on decode's ``y`` and state: on a sixteen-step chain, a wiped median slot and the smallest
    slots moved past their allowance fail."""
    out = _case((64, 64), 64, 2, 3, "global", 0.5, 0.4, seed=64, steps=16)
    y_ref, y_envelope = out["y_refs"][-1]
    assert_planted_errors_fail(out["y"][-1], y_ref, output=DECODE_Y, envelope=y_envelope, what="decode y")
    assert_planted_errors_fail(out["state"], out["ref"].state, output=DECODE_STATE, envelope=out["ref"].state_envelope,
                               what="decode state", derived_atol=2 * state_quantum(out["ref"].state))


def test_the_empty_support_corner_is_inert_rather_than_undefined():
    """``M = 0``: no leaf is touched, ``y`` is zero, and the state is byte-unchanged.

    ``M = 0`` is some level with no live digit on EITHER side -- an all-zero level, a legal
    input since no level need sum to 1 (docs/CONSTITUTION.md#r23). The oracle's answer is
    exact arithmetic: no write deposits, so the state is the entry's, and the readout is a
    sum over no terms, ``0 / (0 + eps) = 0``. Those identities are what this asserts, bit
    for bit. The empty UNIT RANGE it exercises is the path a head whose support is empty
    takes -- its one position is claimed and walks nothing -- and the degenerate forms of
    the queue and the split-K combine live on it.
    """
    device = "cuda"
    widths, d_v, B, H = (64, 64), 64, 1, 2
    topology = _topology(widths)
    zeros = tuple(torch.zeros(B, 1, H, w, device=device, dtype=torch.float32) for w in widths)
    g_write = torch.ones(B, 1, H, device=device, dtype=torch.float32)
    v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float32)
    config = derive_decode_geometry(topology, d_v=d_v, decay=False)
    m0 = state_pages(0.1 * torch.randn(B, H, config.N, config.cols, device=device, dtype=torch.float32))
    state = m0.clone()
    routes, values = step_bundles(v, zeros, with_gain(zeros, g_write), topology)
    y, state, _workspace = _decode_step(routes, values, config, state)
    assert not _leaf_support(routes.live("write"), widths, B, H, device).any(), (
        "the fixture did not produce M = 0")
    assert torch.equal(y, torch.zeros_like(y))
    assert bytes_equal(state, m0), "an untouched state must be byte-unchanged"


@pytest.mark.parametrize("ctas", [7, 64])
def test_more_ctas_than_rows_is_inert(ctas):
    """``ctas > positions``: the surplus CTAs claim nothing and the answer does not move.

    The reachable half of the corner above, and what makes a FROZEN grid safe: the grid is
    sized before the step's work is known, so a step whose realized support is small
    launches CTAs that find every head's queue empty. They must add nothing to a head's
    sums and finish no head they did not walk.
    """
    ref = _case((64, 64), 64, 1, 2, "global", 0.3, 0.3, seed=4, ctas=1)
    got = _case((64, 64), 64, 1, 2, "global", 0.3, 0.3, seed=4, ctas=ctas)
    assert torch.equal(ref["state"], got["state"])
    assert_slots_close(got["y"][-1], ref["y"][-1], output=DECODE_Y, envelope=ref["ref"].y_envelope,
                       what=f"y at ctas={ctas} against ctas=1")


def test_the_widest_staged_topology():
    """``N = 262,144``: the widest topology the staged carrier reaches, on the oracle.

    The lattice is walked from its factors, so ``N`` costs the walk nothing structurally
    -- which is exactly why it has to be gated rather than assumed. The widths are
    LAWFUL (powers of two at or above 16, one admission law with prefill), and the
    capacity share is still unequal across the three levels, so ``pi`` here is not a
    uniform digit rotation.
    """
    widths = (64, 64, 64)
    #: `p = 0.5` rather than something thinner, and `B = 2, H = 3` rather than `1, 2`:
    #: `_simplex` refuses a vacuously dense fixture, and the width-4 level is narrow
    #: enough that a thin Bernoulli mask zeroes every entry and the empty-row repair
    #: hands back a uniform (dense) row. The sparsity that matters for this row is the
    #: TOPOLOGY's size, not the support's.
    _assert_oracle(_case(widths, 32, 2, 3, "global", 0.5, 0.5, seed=262144))


# ---------------------------------------------------------------------------
# G4 -- decay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_decay_matches_the_canonical_oracle(widths, p_read, p_write):
    """Survival is inclusive at ``j = t``: the deposit at ``t`` is read at ``t`` with
    survival exactly 1.

    Decode realizes that by UPDATING the row and then reading it back, which is where
    the tiled kernel's causal intra Gram collapses to at ``T = 1``. Both off-by-ones —
    decaying after the deposit, or reading the row before the deposit — are O(1) errors
    rather than rounding, so this discriminates them at the fp32 tolerance without
    needing a tighter one. (An earlier revision of the kernel had exactly the second
    one; this row is what would have caught it.)
    """
    generator = torch.Generator(device="cuda").manual_seed(4242)
    dials = _decay_dials(widths, 3, generator, "cuda")
    _assert_oracle(_case(widths, 64, 2, 3, "global", p_read, p_write, seed=1234, decay_dials=dials))


@pytest.mark.parametrize("widths", _TOPOLOGIES)
def test_g4_decay_hot_dials(widths):
    """Near-unit retention, where ``1 - rate`` cancels catastrophically and only the
    ``log1pf`` composition recovers the bits."""
    generator = torch.Generator(device="cuda").manual_seed(5150)
    dials = _decay_dials(widths, 3, generator, "cuda", hot=True)
    _assert_oracle(_case(widths, 64, 2, 3, "global", 0.6, 0.5, seed=606, decay_dials=dials))


def test_g4_decay_on_the_stored_write_representation():
    """The clock and the deposit read ONE representation, and it is the stored u8 row.

    An amplitude that flushed to zero in storage is absent from the deposit AND from
    the clock, so ``keep == 1`` there with nothing deposited — the regime the shipped
    producer actually hands decode, run against the oracle on the same tensors.
    """
    generator = torch.Generator(device="cuda").manual_seed(31337)
    dials = _decay_dials((64, 64), 3, generator, "cuda")
    out = _case((64, 64), 64, 2, 3, "global", 0.5, 0.4, seed=808, decay_dials=dials, flushed=True)
    flushed = any(bool(((drawn > 0) & (row == 0)).any())
                  for drawn, row in zip(out["drawn_write"][0], out["routes"][0].write_rows))
    assert flushed, "no drawn amplitude flushed in storage: the row is vacuous"
    _assert_oracle(out)


# ---------------------------------------------------------------------------
# G1 / G2 -- determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctas", [1, 2, 7, 64])
@pytest.mark.parametrize("widths", _TOPOLOGIES)
def test_g1_the_state_is_byte_reproducible_against_itself(ctas, widths):
    """TWO RUNS OF THIS BINARY GIVE A BYTE-IDENTICAL ``state``, and ``y`` inside the oracle band.

    The state is not a reduction: every touched leaf is walked by exactly one warp, whichever
    CTA claims its position, so a failure means the positions do not partition the leaves.
    ``y`` is summed by atomics in queue order, so it is only graded against the oracle.
    """
    a = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=2024, ctas=ctas)
    b = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=2024, ctas=ctas)
    assert torch.equal(a["state"], b["state"]), "decode's state is not run-reproducible"
    assert_slots_close(b["y"][-1], a["y_refs"][-1][0], output=DECODE_Y, envelope=a["y_refs"][-1][1],
                       what=f"y at ctas={ctas}")


@pytest.mark.parametrize("widths", _TOPOLOGIES)
def test_g2_state_is_invariant_across_the_grid(widths):
    """``state`` is byte-identical whatever the resident grid, UNCONDITIONALLY.

    Unlike ``y``, the state is not a reduction: the positions hand every touched leaf to
    exactly one warp, so how many CTAs claim them cannot change a single stored bit. ``y``
    is a reduction and is checked at the oracle band across grids.
    """
    ref = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=555, ctas=1)
    for ctas in (2, 7, 64):
        got = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=555, ctas=ctas)
        assert torch.equal(ref["state"], got["state"]), (
            f"state moved with ctas={ctas}; the positions do not partition the leaves")
        assert_slots_close(got["y"][-1], ref["y"][-1], output=DECODE_Y, envelope=ref["ref"].y_envelope,
                           what=f"y at ctas={ctas} against ctas=1")


# ---------------------------------------------------------------------------
# G6 -- the multi-step recurrence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("decay", [False, True])
def test_g6_sixty_four_chained_steps(widths, decay):
    """64 decode steps against 64 chained oracle steps.

    A single step hides a decay sign error whose effect is second order in one
    application and first order in sixty-four, and it hides a mass-column drift
    entirely. This is also the only row that exercises workspace REUSE across steps and
    the ``ctr`` reset the last-arriving CTA performs — a counter that failed to reset
    would deadlock or double-count on step 2, not step 1.
    """
    dials = None
    if decay:
        generator = torch.Generator(device="cuda").manual_seed(11)
        dials = _decay_dials(widths, 3, generator, "cuda")
    #: `B = 2, H = 3` rather than `1, 2`: `_simplex` refuses a vacuously dense fixture,
    #: and a narrow level drawn only twice reliably survives the Bernoulli mask intact.
    #: The sparsity is the point of the row, so the draw count has to support it.
    out = _case(widths, 64, 2, 3, "global", 0.5, 0.4, seed=64, steps=64, decay_dials=dials)
    _assert_oracle(out)


# ---------------------------------------------------------------------------
# G7 -- the prefill/decode seam, graded against the oracle
# ---------------------------------------------------------------------------


def _oracle_prefill(widths, d_v, B, T, H, generator, device):
    """The prefill leg on the fp64 ORACLE; return what decode needs.

    The PRODUCTION prefill arm ran this leg until the K31 deletion batch removed
    it (baseline = tag `baseline/pre-k31`); the seam this gate owns -- decode
    continuing from the oracle's own entry state, crossed into lattice order,
    under the frozen carrier config -- survives it, because the oracle's `output_final_state` IS that
    state's definition. A hand-built bundle and not the producer: the fixture
    plants a specific density on each side, which a producer cannot be asked for.
    """
    topology = _topology(widths)
    read_levels = tuple(_simplex((B, T, H, w), 0.5, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 0.4, generator, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    write_levels = with_gain(write_levels, g_write)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    ref = oracle_run(v, read_levels, write_levels, topology)
    return dict(topology=topology, read_levels=read_levels, write_levels=write_levels, v=v, y=ref.y,
                state=ref.state.float())


@pytest.mark.parametrize("widths", [(16, 16), (16, 16, 16)],
                         ids=lambda w: "x".join(map(str, w)))
def test_g7_prefill_decode_seam(widths):
    """Prefill ``L`` on the fp64 oracle, decode token ``L + 1`` on the KERNEL,
    and grade the pair against the fp64 oracle over the whole ``L + 1`` sequence.

    This is the gate on the frozen carrier being the RIGHT config: an ``eps``
    that disagreed with the readout's, a head fold that disagreed with
    ``h = bh % H``, or a leaf order that was not the config's lattice would each
    show up here and nowhere else. The prefill leg ran on the production chunk arm until
    the K31 deletion (its charge was measured at 5.2e-3 / 6.4e-3 against the
    1e-2 band, with an oracle CONTROL to keep the blame assignable); the seam's
    entry state is now the oracle's own, narrowed to the kernel's fp32, so the
    number is decode's alone.
    """
    device = "cuda"
    d_v, B, H = 64, 2, 2
    L = 4 * _chunk_tokens()
    seed = 909
    pre = _oracle_prefill(widths, d_v, B, L, H,
                          torch.Generator(device=device).manual_seed(seed), device)

    # the step's own token: drawn from the same generator so the fixture is one
    # sequence of L + 1 tokens rather than a prefill and an unrelated step.
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    step_read = tuple(_simplex((B, 1, H, w), 0.5, gen, device) for w in widths)
    step_write = tuple(_simplex((B, 1, H, w), 0.4, gen, device) for w in widths)
    step_gw = torch.rand(B, 1, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    step_v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float64, generator=gen)

    config = derive_decode_geometry(pre["topology"], d_v=d_v, decay=False)
    #: THE SEAM ITSELF: the entry state is the oracle's, so it crosses into lattice order
    #: here and its answer crosses back.
    entry = state_pages(to_lattice(pre["state"], widths, config.lattice_k, config.lattice_m))
    routes, values = step_bundles(step_v, step_read, with_gain(step_write, step_gw), pre["topology"])
    y_dec, state_dec, _ = _decode_step(routes, values, config, entry)
    step_v, step_read, step_write = bundle_operands(routes, values)
    state_dec = to_canonical(logical_state(state_dec, d_v).reshape(B, H, config.N, d_v + 1), widths,
                             config.lattice_k, config.lattice_m)

    #: THE WHOLE SEQUENCE, on the oracle: prefill and step concatenated, so the
    #: reference sees exactly what the two kernel legs together saw.
    cat = lambda a, b: tuple(torch.cat([x, s], dim=1) for x, s in zip(a, b))  # noqa: E731
    ref = oracle_run(torch.cat([pre["v"], step_v], dim=1), cat(pre["read_levels"], step_read),
                     cat(pre["write_levels"], step_write), pre["topology"])

    assert_slots_close(y_dec, ref.y[:, L:L + 1], output=DECODE_Y, envelope=ref.y_envelope[:, L:L + 1], what="y across the seam")
    #: the entry state crossed the seam as a stored one (half a quantum), and the step stored it again.
    assert_slots_close(state_dec, ref.state, output=DECODE_STATE, envelope=ref.state_envelope,
                       what="the state across the seam", derived_atol=2 * state_quantum(ref.state))
    assert pre["y"].shape[1] == L
