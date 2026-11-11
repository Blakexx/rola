# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 the cells: the routing draws, the plane crossings, and the build stamp gate.

``k_tok`` (a token's nonzero digits per level, the routing sparsity) is a property
of the DRAW here and of nothing in the kernel -- which is the design's whole claim,
so the fixtures state it explicitly and the kernel is never told.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import NamedTuple

import torch

from rola.ops import carry as carry_ops
from rola.ops.naive import naive_rola
from rola.ops.paging import decode_pages, encode_pages
from rola.routing.factors import RouteFactors
from rola.values import Values
from tests.oracle import tolerances


def simplex(shape, k_tok, gen, device, live=None):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense.

    ``live`` truncates the row to its FIRST ``live`` digits -- STRUCTURED support, as
    against ``k_tok``'s unstructured one. The two are different questions: unstructured
    sparsity at long ``T`` still reaches every atom, so only a truncation can produce an
    atom the routing never touches (the idle-resident cell). It is applied after both
    draws so a cell's RNG stream does not depend on it.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    if live is not None and live < shape[-1]:
        x[..., live:] = 0.0
    return x / x.sum(-1, keepdim=True)


def reordered_close(got, want) -> bool:
    """Is ``got`` the same step's readout as ``want`` up to the order its partials were summed?

    A decode step adds its warps' partials into a batch-head's sums atomically, in the order the
    work queue hands out positions (docs/internals/decode/decode.md#queue-finish), so two runs of
    one step agree to fp32 reordering of a sum over the step's leaves, not to the bit: relative
    ``2^-14`` and absolute ``2^-16`` of the readout's largest magnitude.
    """
    scale = float(want.abs().max()) or 1.0
    return bool(torch.allclose(got, want, rtol=2.0 ** -14, atol=scale * 2.0 ** -16))


def assert_reordered_close(got, want, what: str) -> None:
    """:func:`reordered_close`, as a gate."""
    worst = float((got - want).abs().max())
    assert reordered_close(got, want), f"{what}: off by up to {worst:.3g} beyond summation reordering"


def with_gain(write, g_write):
    """A fixture's write levels with its drawn gain folded into level 0, as a producer folds its own
    (docs/CONSTITUTION.md#r23): ``W = g prod_l k_l`` either way, and each level's share is unchanged."""
    write = tuple(write)
    return (write[0] * g_write[..., None].to(write[0].dtype), *write[1:])


def step_bundles(v, read, write, topology):
    """Floating step operands in the internal format: ``(RouteFactors, Values)`` as the framework's epilogue writes
    them (u8 rows, bf16 values). A drawn level with exact zeros is declared SPARSE, as a production whose activation
    has exact zeros is. Every side has a dense level (docs/api.md section 2.1): a side whose drawn levels ALL have
    zeros declares the ALTERNATING level dense (the read side's level 0, the write side's level 1, level 0 at depth
    one) and keeps the others sparse."""
    sparse = tuple(tuple(bool((level == 0).any()) for level in side) for side in (read, write))
    depth = len(read)
    sparse = tuple(flags if not all(flags) else tuple(i != dense for i in range(depth))
                   for flags, dense in zip(sparse, (0, 1 % depth)))
    routes = RouteFactors.from_levels(topology, tuple(t.float() for t in read), tuple(t.float() for t in write),
                                      sparse=sparse)
    return routes, Values.from_projection(v.float())


def bundle_operands(routes, values):
    """What the kernel is handed, in fp64 -- the op oracle's inputs: ``(v, read, write)``, the bundle's levels as
    stored (docs/api.md section 2.1)."""
    read, write = routes.dequantized()
    return values.dequantized().double(), tuple(r.double() for r in read), tuple(w.double() for w in write)


def state_pages(logical):
    """A logical ``[..., N, d_v + 1]`` state as its stored int8-format pages ``[BH, N/16, page_bytes]``: every leading
    axis folds into the batch-head axis the decode kernel indexes (`rola.ops.paging.encode_pages`)."""
    n, cols = logical.shape[-2], logical.shape[-1]
    return encode_pages(logical.float().reshape(-1, n // 16, 16, cols))


def logical_state(pages, d_v: int):
    """Stored pages ``[BH, N/16, page_bytes]`` as the logical ``[BH, N, d_v + 1]`` fp32 state (`decode_pages`)."""
    return decode_pages(pages, d_v).reshape(pages.shape[0], -1, d_v + 1)


def state_quantum(state):
    """Each slot's storage quantum under the int8 state format, ``2^e`` with ``e = max(k - 23, -126)`` from its (leaf,
    32-channel group)'s largest magnitude ``f * 2^k``; zero on the fp32 mass column (`encode_pages`)."""
    d_v = state.shape[-1] - 1
    groups = state[..., :d_v].double().reshape(*state.shape[:-1], d_v // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    _, k = torch.frexp(amax)
    quantum = torch.ldexp(torch.ones_like(groups), (k - 23).clamp_min(-126).expand_as(groups))
    quantum = torch.where(amax == 0, torch.zeros_like(quantum), quantum)
    return torch.cat([quantum.reshape(*state.shape[:-1], d_v), torch.zeros_like(state[..., d_v:]).double()], -1)


def state_rounded(state):
    """THE DECLARED STORE, in fp64: every leaf-channel rounded once to its group's s24 quantum (`state_quantum`), the
    mass untouched -- what the decode kernel's store does to a row it writes, so a chained oracle that applies it after
    each step is the kernel's declared arithmetic (`oracle-two-purposes-doctrine`: a fidelity gate mirrors it)."""
    quantum = state_quantum(state)
    d_v = state.shape[-1] - 1
    out = state.double().clone()
    q = quantum[..., :d_v]
    out[..., :d_v] = torch.where(q == 0, out[..., :d_v],
                                 torch.round(out[..., :d_v] / q.clamp_min(2.0 ** -149)).clamp(-(2 ** 23 - 1), 2 ** 23 - 1) * q)
    return out.to(state.dtype)


def built_arms() -> set:
    """The carry arm keys THIS binary carries (`rola.ops.carry.arm_key`).

    `arms()` is the binary's own answer (`rola.ops.carry.arms()`), so this is a fact
    about the `.so` under test and never about the source tree that produced it.
    """
    return {tuple(row) for row in carry_ops.arms()}


def assert_fresh_binary():
    """DEVICE-SIDE, because a path or hash check cannot catch an extension trap: the carry family's build stamp is
    a fact only a built kernel produces (`rola.ops.carry.build_stamp`)."""
    return carry_ops.build_stamp()


def allowances(reference, output, envelope=None, derived_atol=0.0):
    """Each slot's allowed error under `output` (`tolerances.Output`), and which term sets it: a clause ``(r, a)`` fails
    a slot whose relative AND absolute errors both pass it, so under clause ``i`` a slot of size ``s = |oracle|`` may be
    off by ``max(r·s, a + derived_atol)``; a multilinear output adds its envelope term ``rtol·envelope + derived_atol``
    (a declared upstream error adds to whichever budget binds); the allowance is the smallest. Returns ``(allowance, binding)``, ``binding`` the index of the tightest clause per slot, or
    ``len(clauses)`` where the envelope binds."""
    s = reference.double().abs()
    terms = [torch.clamp(r * s, min=a + derived_atol) for r, a in output.clauses]
    if output.rtol is not None:
        assert envelope is not None, f"{output.name} is multilinear: its gate states the envelope"
        terms.append(output.rtol * envelope.double() + derived_atol)
    else:
        assert envelope is None, f"{output.name} states no envelope budget, so a gate cannot hand it one"
    if not terms:
        return torch.full_like(s, math.inf), torch.zeros_like(s, dtype=torch.long)
    allowance, binding = torch.stack(terms).min(dim=0)
    return allowance, binding


def assert_slots_close(actual, reference, *, output, what, envelope=None, derived_atol=0.0):
    """THE ORACLE RULE, PER SLOT: the kernel and the oracle produce the same shape, and each slot's error is inside its
    allowance (`allowances`): every clause of the output kind (`tolerances.py`, a list of ``(rtol, atol)`` pairs with
    the measurements that set them), and on a multilinear output also ``rtol`` times the slot's ENVELOPE, the sum of the
    sizes of the terms the slot adds up (`oracle_run` computes it by running the same fp64 reference on ``|v|``).
    Rounding each term costs a share of that term's size whether or not the terms cancel, so a slot that is small
    because its terms cancel keeps their rounding, and a slot that is small because its terms are small keeps nothing.
    ``derived_atol`` is an absolute term the caller derives and states (an upstream seam's declared error), added to
    every clause's atol and to the envelope term, never a loosened constant. A slot whose allowance is zero must be exactly zero; a slot that is
    not finite on either side fails. The failure names how many slots failed and the worst one: its error over its
    allowance, and the clause or envelope that set the allowance.
    """
    a, r = actual.double(), reference.double()
    assert a.shape == r.shape, f"{what}: the kernel's shape {tuple(a.shape)} is not the oracle's {tuple(r.shape)}"
    if envelope is not None:
        assert envelope.shape == r.shape, f"{what}: the envelope's shape {tuple(envelope.shape)} is not the oracle's"
    allowance, binding = allowances(r, output, envelope, derived_atol)
    err = (a - r).abs()
    bad = ~(err <= allowance)
    ratio = torch.where(err == 0, 0.0, err / allowance).nan_to_num(nan=math.inf)
    _record_margins(output, what, err, r, ratio, bad, binding)
    if bool(bad.any()):
        i = int(torch.argmax(ratio.flatten()))
        where = tuple(int(x) for x in torch.unravel_index(torch.tensor(i), a.shape))
        term = int(binding.flatten()[i])
        named = "the envelope" if term == len(output.clauses) else f"clause {output.clauses[term]}"
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {err.numel()} slots are off the oracle ({output.name}); the worst, at {where}, "
            f"is off by {float(err.flatten()[i]):.4g} against an allowance of {float(allowance.flatten()[i]):.4g} set by "
            f"{named} (kernel {float(a.flatten()[i]):.6g}, oracle {float(r.flatten()[i]):.6g})")


class OracleRun(NamedTuple):
    """One fp64 recurrence and the envelope of each of its slots (`assert_slots_close`)."""

    y: torch.Tensor
    state: torch.Tensor
    y_envelope: torch.Tensor
    state_envelope: torch.Tensor


def oracle_run(v, read_levels, write_levels, topology, decay=None, entry=None) -> OracleRun:
    """`rola.ops.naive.naive_rola`, with the same recurrence run on ``|v|`` beside it: every weight is non-negative, so
    that run's slots are the sums of the term sizes. ``entry`` is None (a fresh state), a state of the caller's own
    (its envelope is ``|entry|``) or a previous `OracleRun`, whose state and envelope a chain continues. A readout
    ``num / (den + eps)`` rounds its denominator as well, so its envelope adds ``|y|``."""
    if isinstance(entry, OracleRun):
        state, envelope = entry.state, entry.state_envelope
    else:
        state, envelope = entry, None if entry is None else entry.double().abs()
    y, out = naive_rola(v, read_levels, write_levels, topology, decay, initial_state=state, output_final_state=True)
    y_magnitudes, out_envelope = naive_rola(v.abs(), read_levels, write_levels, topology, decay,
                                            initial_state=envelope, output_final_state=True)
    return OracleRun(y, out, y_magnitudes + y.abs(), out_envelope)


def assert_planted_errors_fail(actual, reference, *, output, what, envelope=None, derived_atol=0.0):
    """THE RULE HAS TEETH ON THIS OUTPUT. The kernel's own output passes, and mutants of it fail: the slot of median
    size wiped to zero (a dropped contribution the allowance must not swallow), the sixteen smallest allowances exceeded
    by half (the smallest slots are held, not waved through), and for each clause of the output, the slot of median
    size among those that clause holds tightest moved half past that clause's own allowance -- judged by the clauses
    alone, so a clause the envelope out-binds everywhere is still shown to hold what it declares."""
    assert_slots_close(actual, reference, output=output, what=what, envelope=envelope, derived_atol=derived_atol)
    a, r = actual.double().contiguous(), reference.double().contiguous()
    allowance, _ = allowances(r, output, envelope, derived_atol)
    nonzero = (r != 0).flatten().nonzero().squeeze(1)
    median = nonzero[torch.argsort(r.abs().flatten()[nonzero])[nonzero.numel() // 2]]
    wiped = a.clone()
    wiped.view(-1)[median] = 0.0
    small = a.clone()
    smallest = torch.topk(allowance.flatten(), min(16, allowance.numel()), largest=False).indices
    small.view(-1)[smallest] = r.view(-1)[smallest] + 1.5 * allowance.view(-1)[smallest] + 1e-300
    mutants = [(wiped, "the median slot wiped", output, envelope), (small, "the smallest allowances exceeded by half",
                                                                     output, envelope)]
    clauses_only = dataclasses.replace(output, rtol=None)
    clause_allowance, clause_binding = allowances(r, clauses_only, None, derived_atol)
    for i, clause in enumerate(output.clauses):
        held = (clause_binding.flatten() == i).nonzero().squeeze(1)
        if held.numel() == 0:
            continue
        slot = held[torch.argsort(r.abs().flatten()[held])[held.numel() // 2]]
        moved = a.clone()
        moved.view(-1)[slot] = r.view(-1)[slot] + 1.5 * clause_allowance.view(-1)[slot] + 1e-300
        mutants.append((moved, f"clause {clause} exceeded by half where it binds", clauses_only, None))
    for mutant, name, judged_by, env in mutants:
        try:
            assert_slots_close(mutant, r, output=judged_by, what=f"{what}, {name}", envelope=env,
                               derived_atol=derived_atol)
        except AssertionError:
            continue
        raise AssertionError(f"{what}: {name} passes the rule, which therefore cannot see it")


#: THE RELATIVE ERRORS A MARGINS LINE READS ITS FRONTIER AT: a decade grid, and the output's own declared rtols -- its
#: envelope budget and each clause's -- because no clause of it may be tighter than the precision contract it states. At
#: each, the largest absolute error among the slots whose relative error exceeds it: the smallest atol a clause at that
#: rtol could declare and still pass the comparison.
FRONTIER_DECADES = tuple(10.0 ** k for k in range(-8, 3))


def frontier_rtols(output):
    return tuple(sorted({*FRONTIER_DECADES, *(r for r, _a in output.clauses), *((output.rtol,) if output.rtol else ())}))


def _record_margins(output, what, err, reference, ratio, bad, binding):
    """``pytest --oracle-margins=<file>`` appends one JSON line per comparison: the output kind, the worst passing error
    over its allowance (``worst``), how many slots each clause and the envelope bound (``binding``), and the FRONTIER --
    for each relative error in `FRONTIER_RTOLS`, the largest absolute error among the slots past it. `tolerances.py`'s
    clauses are read off these lines."""
    out = tolerances.MARGINS_FILE
    if not out:
        return
    rel = torch.where(err == 0, 0.0, err / reference.abs()).nan_to_num(nan=math.inf, posinf=math.inf)
    passing = ratio[~bad]
    counts = torch.bincount(binding.flatten(), minlength=len(output.clauses) + 1).tolist()
    frontier = {f"{r:g}": float(err[rel > r].max()) if bool((rel > r).any()) else 0.0 for r in frontier_rtols(output)}
    row = {"output": output.name, "what": what, "slots": reference.numel(), "failed": int(bad.sum()),
           "worst": float(passing.max()) if passing.numel() else None, "binding": counts, "frontier": frontier}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def relative(actual, reference):
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference.double()).abs().max()) / scale
