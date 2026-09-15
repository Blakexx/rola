# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMBINED OPERATOR against the canonical fp64 oracle.

THE REFERENCE IS `rola.ops.naive.naive_rola` DIRECTLY, and that is the point of this
file. Each family's own conformance would pass on two different window grids; here the
two kernels run as one call and the whole answer is compared against the recurrence
itself, so a window-grid disagreement, a double-counted diagonal or a dropped tile pair
has nowhere to hide.

RED BY DESIGN (card `development/queue/C_CLEAN_SLATE.md`, "TEST-DRIVEN"): a cell whose
carry arm this line does not build yet fails, by name. The window is no longer a cell's to name -- the axis law made ``W`` a kernel constant (`rola.ops.carry.WINDOW`) and the intra family's
arm must be built at the same number, because the two kernels run ONE grid. Neither is
``(k, m)``: the box is derived at launch from the descriptor and the launch shape.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``W`` = the kernel's fixed window, ``modes`` = the
per-level support-mode DECLARATION the intra's zero certificate reads (a statement about
the routing, never a switch: an undeclared side reads all-ones and cannot move a number).
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import carry_cells, conservative_activity, liveness_words, realize
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops.constants import READOUT_EPS
from rola.ops.naive import naive_rola
from rola.ops.paging import bytes_equal
from rola.ops.prefill import prefill
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from tests.oracle.fixtures import canonical_from_plane, relative, require_arm
from tests.oracle.oracle_fixtures import _output_charge
from tests.oracle.tolerances import BF16_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: THE TOPOLOGY EACH DEPTH'S INTRA ARM IS INSTANTIATED AT, ``D -> B_l``. A combined cell
#: needs BOTH families at its topology; one only the carry carries is an inter-only cell.
INTRA_TOPOLOGY = {2: intra_ops.LEVEL_WIDTH, 3: intra_ops.LEVEL_WIDTH_DEEP3,
                  4: intra_ops.LEVEL_WIDTH_DEEP4}

#: The registry's cells the combined operator can run: the dense backing, a fresh state,
#: the shipped value width, and a topology both families carry. A cell whose carry arm no
#: build carries yet, or whose window is partial, stays selected and red (TEST-DRIVEN).
CELLS = tuple(c for c in carry_cells("oracle")
              if c.backing == "dense" and c.state == "fresh" and c.dv == 64
              and len(set(c.widths)) == 1
              and INTRA_TOPOLOGY.get(c.D) == c.widths[0])
IDS = [c.name for c in CELLS]


def modes_of(spec):
    """The cell's DRAW as the intra's per-level declaration.

    ``alt``/``cohort`` alternate the sparse side by level (even write-sparse, odd
    read-sparse); ``both`` is sparse on both sides at level zero; ``dense`` declares
    nothing. It is DERIVED from the draw rather than carried in the record because the
    declaration IS a statement about the draw, and two places to state one fact is the
    drift shape this family has already paid for once.
    """
    if spec.draw == "dense":
        return (intra_ops.DENSE_BOTH,) * spec.D
    if spec.draw == "both":
        return tuple(intra_ops.BOTH_SPARSE if level == 0 else intra_ops.DENSE_BOTH
                     for level in range(spec.D))
    return tuple(intra_ops.READ_SPARSE if level % 2 else intra_ops.WRITE_SPARSE
                 for level in range(spec.D))


def fresh_plane(spec, bh: int = 1):
    """the plane a fresh sequence hands out: the op allocates none."""
    from rola.ops.prefill import _descriptor

    return carry_ops.state_plane(_descriptor(list(spec.widths), spec.dv, bh), bh)


def call(spec, state_in=None):
    """One combined call on the FINAL surface: no window, no ``(k, m)``, no carve."""
    require_arm(*spec.arm)
    drawn = realize(spec)
    return drawn, prefill(drawn.read, drawn.write, drawn.gain, drawn.v, spec.widths,
                          modes=modes_of(spec),
                          sread=intra_ops.pack_support(drawn.read),
                          swrite=intra_ops.pack_support(drawn.write),
                          state_in=state_in,
                          state_out=fresh_plane(spec) if state_in is None else state_in)


def oracle(drawn, state_in=None):
    read, write, gain, v = drawn.doubles()
    routing = IndependentRouting(width=1, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(w) for w in drawn.spec.widths))
    return naive_rola(v, read, write, gain, topo, None, initial_state=state_in,
                      output_final_state=True)


def readout(num, den, spec):
    """``[BH, L, DV]`` / ``[BH, L]`` seen as the oracle's ``[B, L, H, DV]``."""
    y = num / (den[..., None].double() + READOUT_EPS)
    return y.reshape(1, 1, spec.tokens, spec.dv).permute(0, 2, 1, 3)


# ------------------------------------------- the grid, and the surface it runs on

def test_the_two_kernels_run_one_window_grid():
    """``W`` IS A CONSTANT AND IT IS SHARED. The carry's window left the surface at K50;
    the intra's is still an arm axis, so what must hold is that the intra family is BUILT
    at the carry's constant. If it were not, the combined operator would have no grid and
    the cells below would be comparing two different decompositions."""
    built = (intra_ops.WINDOW, intra_ops.WINDOW_WIDE, intra_ops.WINDOW_SMALL)
    assert carry_ops.WINDOW in built, (
        f"the carry's fixed window {carry_ops.WINDOW} is not among the intra family's "
        f"built windows {built}; one operator needs one grid")


def test_the_combined_operator_reaches_the_carry_launch_surface():
    """THE ONE CONTRACT ITEM THIS FILE STILL OWES (recorded at C1c): the combined
    operator's carry leg must go through `rola.ops.carry.carry_forward`.

    `rola.ops.prefill` still names the pre-C0 bindings directly, so its carry calls are
    written against a signature the extension no longer carries and against axes K50
    deleted. The cells below therefore fail at prefill's own seam rather than at the
    kernel boundary -- a different fact reported as the same red. Re-pointing those three
    call sites is what makes this file's reds the ONE reason the carry oracle tier
    already reports."""
    import inspect

    from rola.ops import prefill as prefill_mod

    body = inspect.getsource(prefill_mod)
    stale = [name for name in ("carry_forward_inter", "carry_backward_inter")
             if name in body]
    assert not stale, (
        f"rola/ops/prefill.py still calls {stale} directly; the combined operator's "
        f"carry leg is rola.ops.carry.carry_forward/carry_backward, which is the only "
        f"place the descriptor, the geometry block, the liveness words and the launch "
        f"shape are checked")


# ------------------------------------------------- the arithmetic, cell by cell

@pytest.mark.parametrize("spec", CELLS, ids=IDS)
def test_prefill_matches_the_fp64_oracle(spec):
    drawn, (num, den, plane) = call(spec)
    y_ref, s_ref = oracle(drawn)
    err = _output_charge(readout(num, den, spec), y_ref)
    assert err < BF16_RTOL, f"{spec.name}: a segment of the readout leaves the bf16 band at {err:.3e}"
    s_err = relative(canonical_from_plane(plane), s_ref.reshape(1, spec.N, spec.dv + 1))
    assert s_err < BF16_RTOL, f"{spec.name}: the final state leaves the band at {s_err:.3e}"
    del num, den, plane, y_ref, s_ref
    torch.cuda.empty_cache()


def test_the_readout_charge_sees_a_drift_the_global_max_cannot():
    """The metric the cells above are graded in has to be the right one (docs/testing.md, rule 6).

    The readout is a ratio whose denominator is accumulated write mass, so at `t = 0` `|y|` is orders of magnitude
    above the rest of the sequence, and one global normalizer is set by the token whose state has been updated zero
    times: that metric cannot see an error that ACCUMULATES. A `sqrt(t)` drift ten times the band is planted on the
    kernel's own readout, and the two forms must disagree about it -- the global form passing it is the premise, the
    per-segment charge catching it is the claim.
    """
    spec = next(c for c in CELLS if c.name == "flagship-dense")
    drawn, (num, den, _plane) = call(spec)
    y_ref, _s_ref = oracle(drawn)
    y = readout(num, den, spec)
    assert relative(y, y_ref) < BF16_RTOL and _output_charge(y, y_ref) < BF16_RTOL, (
        f"the unperturbed control must pass under both forms (global {relative(y, y_ref):.3e}, per-segment "
        f"{_output_charge(y, y_ref):.3e})")
    ramp = (torch.arange(spec.tokens, device=y.device, dtype=torch.float64) / (spec.tokens - 1)).sqrt()
    mutant = y.double() * (1.0 + 10.0 * BF16_RTOL * ramp[None, :, None, None])
    assert relative(mutant, y_ref) < BF16_RTOL, (
        f"the planted drift moved the global form to {relative(mutant, y_ref):.3e}: the premise is that it passes "
        "this mutant -- re-derive the mutant's size rather than deleting the claim")
    assert _output_charge(mutant, y_ref) > BF16_RTOL, (
        f"the planted drift moved the per-segment charge only to {_output_charge(mutant, y_ref):.3e}: it is as "
        "blind as the global form")


def test_a_chained_call_equals_one_long_call():
    """THE STATE IS THE WHOLE CONTRACT of a chunked prefill: two calls over the halves,
    the first's ``state_out`` seeding the second, must be the one long call. The split
    sits on a WINDOW line, which is where a chunk boundary is legal at all -- the window
    grid is global, so a split inside a window would ask the second call to carry pairs
    neither term owns.

    THE STATE IS BIT-IDENTICAL and that is the claim that matters: an owner writes its
    own leaves and nothing reduces across CTAs there. The OUTPUT is not, and cannot be:
    ``num``/``den`` are accumulated by a RED at the top, so the order the owners'
    contributions land in is the scheduler's -- for the whole call as much as for the
    chain. Agreement to fp32 rounding is what a reordered fp32 sum can promise, and it is
    far tighter than the bf16 band."""
    spec = next(c for c in CELLS if c.draw == "dense")
    drawn = realize(spec)
    modes = modes_of(spec)
    sread = intra_ops.pack_support(drawn.read)
    swrite = intra_ops.pack_support(drawn.write)
    whole = prefill(drawn.read, drawn.write, drawn.gain, drawn.v, spec.widths,
                    modes=modes, sread=sread, swrite=swrite, state_out=fresh_plane(spec))

    cut = carry_ops.WINDOW
    words = cut // intra_ops.WORD_TOKENS
    head = prefill(tuple(x[:, :cut] for x in drawn.read),
                   tuple(x[:, :cut] for x in drawn.write), drawn.gain[:, :cut],
                   drawn.v[:, :cut], spec.widths, modes=modes, sread=sread[:, :words],
                   swrite=swrite[:, :words], state_out=fresh_plane(spec))
    tail = prefill(tuple(x[:, cut:] for x in drawn.read),
                   tuple(x[:, cut:] for x in drawn.write), drawn.gain[:, cut:],
                   drawn.v[:, cut:], spec.widths, modes=modes,
                   sread=sread[:, words:].contiguous(),
                   swrite=swrite[:, words:].contiguous(), state_in=head[2], state_out=head[2])

    assert bytes_equal(tail[2], whole[2])
    num = torch.cat((head[0], tail[0]), dim=1)
    den = torch.cat((head[1], tail[1]), dim=1)
    assert relative(num, whole[0]) < 1e-5
    assert relative(den, whole[1]) < 1e-5
    y_ref, _ = oracle(drawn)
    err = relative(readout(num, den, spec), y_ref)
    assert err < BF16_RTOL, f"the chained readout leaves the bf16 band at {err:.3e}"


def test_the_intra_term_is_actually_present():
    """NON-VACUITY. The carry alone must NOT pass the oracle gate: if it did, every cell
    above would be testing the carry and calling it the operator. This is also the one
    cell in this file that drives the carry LAUNCH SURFACE directly, so it is where the
    combined operator's carry leg and the kernel boundary are the same call."""
    spec = next(c for c in CELLS if c.draw == "dense")
    drawn = realize(spec)
    desc = spec.descriptor()
    launch = spec.launch()
    num, den = carry_ops.carry_forward(
        carry_ops.route_planes(drawn.read, drawn.write, drawn.gain),
        drawn.v.permute(0, 2, 1, 3).reshape(1, spec.tokens, spec.dv).contiguous(),
        descriptor=desc, geometry=carry_ops.geometry_block(desc, launch),
        liveness=liveness_words(drawn, desc), activity=conservative_activity(desc),
        launch=launch, state_out=carry_ops.state_plane(desc, 1))
    y_ref, _ = oracle(drawn)
    err = relative(readout(num, den, spec), y_ref)
    assert err > 10 * BF16_RTOL, (
        f"the INTER term alone is within {err:.3e} of the whole recurrence; the intra "
        f"term this operator adds is not being tested by the cells above")


def test_the_combined_operator_refuses_what_it_has_no_kernel_for():
    """The refusal is STRUCTURAL, and stating it is how a gap stays a gap rather than
    becoming a host-side second code path: a topology only the carry family carries is an
    inter-only cell, and the combined entry says so instead of half-serving it."""
    spec = next(c for c in CELLS if c.draw == "dense")
    drawn = realize(spec)
    with pytest.raises(ValueError, match="level width"):
        prefill(drawn.read, drawn.write, drawn.gain, drawn.v, (64, 64),
                modes=modes_of(spec), sread=intra_ops.pack_support(drawn.read),
                swrite=intra_ops.pack_support(drawn.write))
    #: A width another depth ships is not this depth's: depth 3's 16 passes a check that asks
    #: membership in every depth's widths, and then the kernel refuses the support words' shape.
    with pytest.raises(ValueError, match="at depth 2"):
        prefill(drawn.read, drawn.write, drawn.gain, drawn.v, (16, 16),
                modes=modes_of(spec), sread=intra_ops.pack_support(drawn.read),
                swrite=intra_ops.pack_support(drawn.write))
