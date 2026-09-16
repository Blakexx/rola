# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""WHAT is timed: one lean callable per kernel this line carries.

A subject takes a REGISTRY cell, builds everything the launch does not pay for, and
returns a :class:`Launch` whose call issues exactly the launch being priced.
The roster mirrors the oracle roster one for one -- a bench roster that does not match
the correctness roster is a roster with kernels nobody measures -- and it includes the
carry. A subject that is absent from the roster while its kernel exists is the defect the
mirroring exists to catch; a subject whose kernel is unbuilt refuses by name, which is an
honest report rather than a number for something that did not run.

**TWO CELL KINDS, ONE REGISTRY** (rola-devtools' central cells). A subject declares which kind it takes.
``kind="carry"`` is a drawn cell (`rola_devtools.cells.carry`): amplitudes put directly on the simplex, with the state
the sequence enters with, which is what the carry family's oracle and probe run -- read through `benchmarks.cells`.
``kind="layer"`` is a layer input (`rola_devtools.cells.layer`) under one of RoLA's constructions
(`benchmarks.cells.layer`): a producer, a routing template and a gain, from which the amplitudes are PRODUCED -- the
only way to price the producer's own solve or a decode step through the layer.

**EVERYTHING OUTSIDE THE LAUNCH IS BUILT ONCE, HERE.** The packed sides, the support
words, the state plane, the liveness words, the routing: all of it happens before the
callable is returned, so a timed span wraps the launch and nothing else.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Launch:
    """One thing to time: a name and the zero-argument callable that issues it. `reset`, for a launch that changes what
    its next call reads (a carried state, a decode step), restores exactly what the first call saw, and is run untimed
    before every call, so every call does the same work on the same data. `outside_allocator`, for a launch that holds
    device memory torch's caching allocator does not see, returns those bytes now."""

    name: str
    call: Callable[[], object]
    reset: Callable[[], None] | None = None
    outside_allocator: Callable[[], int] | None = None


def _restore(state_in):
    """The reset of a launch whose state plane is read and advanced in place: the plane's entry, copied back."""
    if state_in is None:
        return None
    entry = state_in.clone()
    return lambda: state_in.copy_(entry)


@dataclass(frozen=True)
class Subject:
    """One registered bench: what it prices, at which LEVEL, which cell kind it takes, how it builds."""

    name: str
    kind: str
    build: Callable[[dict], Launch]
    what: str
    #: THE LEVEL IT PRICES (Blake, 2026-09-15), recorded with every sample so a reading never crosses two of them:
    #: `kernel` is one family's launches and nothing around them; `op` is everything the op does over bare operands and
    #: a state -- the facts, the paging, the packing and the kernels -- as `rola_op` calls it; `layer` adds the
    #: projections that produce the routing, and is the ONLY level at which this library is compared with another.
    level: str
    #: THE BODY'S OWN NAME, for `ncu -k regex:`. A PREFIX would capture whichever kernel
    #: of the family launches first, and that is not the same kernel on two binaries -- a
    #: tip whose family launches a preliminary pass ahead of the body matches that pass
    #: while its parent matches the body, and every ratio taken from the pair is
    #: meaningless. Naming the body makes the two tips capture the same kernel or capture
    #: nothing.
    symbol: str
    #: The device-side stamp entry that proves THIS binary carries the family (a path or
    #: a hash cannot: only a fact the device produces can catch a stale binary).
    stamp: str
    #: THE CALL COUNTS it is priced at: the cell's sequence as that many carried calls. One is
    #: the whole sequence in one launch set; a count above one prices what each call's fixed
    #: cost adds, and the difference between the two rows IS the per-call price.
    calls: tuple[int, ...] = (1,)
    #: THE ARM DIALS it reads, each a property of one arm and never of a run (`bench.provider` names an arm by them):
    #: `schedule`, the carry order policy (`rola.ops.carry.ORDER_POLICIES`). A dial a subject does not read has one
    #: value, its default, so it cannot be set where it would reach nothing. The state a call binds is the CELL's
    #: (`benchmarks.cells.state_binding`), never a dial.
    dials: tuple[str, ...] = ()


# ------------------------------------------------------------------ carry cells

def _carry_operands(fx):
    from benchmarks.cells import carry_call

    return carry_call(fx["cell"], 1)


def _binding(fx, drawn, descriptor):
    from benchmarks.cells import state_binding

    return state_binding(fx["cell"], drawn, descriptor, 1)


def _carry_schedule(name: str, spec):
    """``first`` | ``identity`` -> the launch's dial: the ORDER POLICY (`CarrySchedule`).
    ``first`` is the first-live-box sort, the reap; ``identity`` keeps token order and tiles
    every token, the sort's A/B."""
    from rola.ops import carry as carry_ops

    del spec
    #: the cross-binary probe hands one schedule name to both binaries; the baseline's
    #: vocabulary (`box`, `sparse-gN`: its dense and reaping schedules) names the reap here.
    if name == "box" or name.startswith("sparse"):
        name = carry_ops.ORDER_FIRST_BOX
    if name not in carry_ops.ORDER_POLICIES:
        raise ValueError(f"schedule {name!r} is one of {tuple(carry_ops.ORDER_POLICIES)}")
    return carry_ops.CarrySchedule(order=name)


def carry_forward(fx) -> Launch:
    """The inter term: one carry launch over a whole cell, state advanced in place.

    The operands are the registry's own `carry_call`, so this prices the same tensors
    the oracle tier checks.
    """
    from rola.ops import carry as carry_ops

    spec = fx["cell"]
    drawn, call = _carry_operands(fx)
    routes, v = call.pop("routes"), call.pop("v")
    state_in, state_out, page_table = _binding(fx, drawn, call["descriptor"])
    #: THE SCHEDULE ARM. The runtime dial the launch carries (`CarrySchedule`, the order
    #: policy), so a probe's two arms can be the SAME binary differing in exactly the order
    #: -- which is what the reap is measured as. `first` is what a caller who names nothing
    #: gets.
    schedule = _carry_schedule(fx.get("schedule", "first"), spec)

    def run():
        return carry_ops.carry_forward(routes, v, state_in=state_in, state_out=state_out,
                                       page_table=page_table, schedule=schedule, **call)

    return Launch(name=f"carry_forward|{spec.name}", call=run, reset=_restore(state_in))


def intra_forward(fx) -> Launch:
    """The within-window term on the same cell, at the window the two kernels share.

    THE WINDOW IS THE CARRY'S CONSTANT, not the intra arm's default: the two kernels run
    ONE grid, so pricing the intra at a window the carry does not run would be pricing a
    different decomposition. A cell whose L is not a whole number of that window has no
    grid for this subject and is not applicable to it.
    """
    from benchmarks.cells import realize
    from rola.ops import carry as carry_ops
    from rola.ops import intra as intra_ops

    spec = fx["cell"]
    drawn = realize(spec)
    modes = _modes_of(spec)
    gwrite = drawn.gain.permute(0, 2, 1).reshape(-1, spec.tokens).contiguous()
    #: PACKED ONCE, HERE: the padding and packing are the producer's (host-side, by the
    #: intra op's own contract) and pricing them per call priced glue -- a fixed ~5 ms that
    #: read as the kernel at every L (2026-09-08). The launch is what this arm prices, as
    #: the carry arm prices its launch over the registry's packed operands.
    D = len(drawn.read)
    width = intra_ops._shipped_level_width(D, carry_ops.WINDOW)
    read = intra_ops.pad_routes(drawn.read, (width,) * D)
    write = intra_ops.pad_routes(drawn.write, (width,) * D)
    v = intra_ops.pad_v(drawn.v.contiguous(), intra_ops.VALUE_WIDTH)
    pread, pwrite = intra_ops.pack_side(read), intra_ops.pack_side(write)
    sread, swrite = intra_ops.pack_support(read), intra_ops.pack_support(write)

    def run():
        return intra_ops.intra_forward(pread, pwrite, gwrite, v, sread, swrite, modes,
                                       window=carry_ops.WINDOW)

    return Launch(name=f"intra_forward|{spec.name}", call=run)


def _atom_bits(read_levels, write_levels, widths):
    """the per-atom activity byte of a call, as the engine folds it from both planes."""
    from rola.engine.facts.planes import atom_bits
    from rola.ops.carry import pack_side

    return atom_bits(pack_side(write_levels), widths, pack_side(read_levels))


def carry_intra(fx) -> Launch:
    """THE TWO KERNELS AS ONE OPERATOR: `rola.ops.prefill` over the cell's drawn routes, gain and values -- the
    liveness pass, the carry and the intra, and that seam's own packing of its per-level inputs.

    A KERNEL-LEVEL SUBJECT, not the op (Blake, 2026-09-15). The support words are the producer's output and the activity
    bits are the facts pass's, so both are built once here, outside the timed call, exactly as `rola.ops.prefill`'s
    contract says they are the caller's. What an OP-level number would add -- producing those facts, planning and
    committing the pages, the packing the op owns -- is not timed here and has no live path to time until the prefill
    arm is rebuilt (`rola.interface.rola_op`).

    AT `calls` ABOVE ONE the same sequence runs as that many calls over whole windows, the
    state carried through one plane zeroed per launch set: what a caller that prefills in
    chunks pays -- every call's fixed cost (packing, liveness, the sweeps in and out of the
    state) that many times over the windows one call runs. A carried chain binds its own
    plane, so the cell's state binding is the single call's.
    """
    from benchmarks.cells import realize
    from rola.ops import carry as carry_ops
    from rola.ops import intra as intra_ops
    from rola.ops import prefill as prefill_ops

    spec = fx["cell"]
    calls = fx.get("calls", 1)
    if not _splits(spec.tokens, calls):
        raise ValueError(f"{spec.name}: {spec.tokens} tokens is not {calls} calls of whole windows")
    drawn = realize(spec)
    modes = _modes_of(spec)
    D = len(drawn.read)
    width = intra_ops._shipped_level_width(D, prefill_ops.WINDOW)
    step = spec.tokens // calls
    parts = []
    for c in range(calls):
        sl = slice(c * step, (c + 1) * step)
        read = [x[:, sl] for x in drawn.read]
        write = [x[:, sl] for x in drawn.write]
        #: THE ACTIVITY BYTES gate the state sweeps to the atoms this call touches; they are
        #: the engine's fold of the liveness words (`facts.planes.atom_bits`), a producer-side
        #: fact, so built once here. Passing none states every atom active.
        parts.append((read, write, drawn.gain[:, sl].contiguous(), drawn.v[:, sl].contiguous(),
                      intra_ops.pack_support(intra_ops.pad_routes(read, (width,) * D)),
                      intra_ops.pack_support(intra_ops.pad_routes(write, (width,) * D)),
                      _atom_bits(read, write, spec.widths)))
    descriptor = prefill_ops._descriptor(list(spec.widths), spec.dv, 1)
    if calls == 1:
        chain = None
        state_in, state_out, page_table = _binding(fx, drawn, descriptor)
        name = f"carry_intra|{spec.name}"
    else:
        chain = carry_ops.state_plane(descriptor, 1)
        state_in, state_out, page_table = chain, chain, None
        name = f"carry_intra|{spec.name}|calls={calls}"

    def run():
        if chain is not None:
            chain.zero_()
        out = None
        for read, write, gain, v, sread, swrite, bits in parts:
            out = prefill_ops.prefill(read, write, gain, v, list(spec.widths), modes=modes,
                                      sread=sread, swrite=swrite, atom_bits=bits,
                                      state_in=state_in, state_out=state_out,
                                      page_table=page_table)
        return out

    return Launch(name=name, call=run, reset=_restore(state_in) if chain is None else None)


def liveness_pass(fx) -> Launch:
    """The class-1 liveness words for both sides -- the pass every carry call reads."""
    from benchmarks.cells import realize
    from rola.engine.facts import liveness as lv
    from rola.ops.carry import pack_side
    from rola.ops.liveness import liveness_words

    spec = fx["cell"]
    drawn = realize(spec)
    read, write = pack_side(drawn.read), pack_side(drawn.write)
    layout = lv.LivenessLayout(D=spec.D, B=spec.widths, L=spec.tokens)
    statics = lv.side_statics(layout, (False,) * spec.D, layout.B)

    def run():
        return liveness_words(read, write, layout, statics, statics)

    return Launch(name=f"liveness_pass|{spec.name}", call=run)


def _modes_of(spec):
    """The cell's DRAW as the per-level support declaration.

    DERIVED from the draw rather than carried in the record, because the declaration IS
    a statement about the draw and two places to state one fact is the drift this family
    has already paid for once. The oracle tier derives it the same way.
    """
    from rola.ops import intra as intra_ops

    if spec.draw == "dense":
        return (intra_ops.DENSE_BOTH,) * spec.D
    if spec.draw == "both":
        return tuple(intra_ops.BOTH_SPARSE if level == 0 else intra_ops.DENSE_BOTH
                     for level in range(spec.D))
    return tuple(intra_ops.READ_SPARSE if level % 2 else intra_ops.WRITE_SPARSE
                 for level in range(spec.D))


# ------------------------------------------------------------------ layer cells

def entmax_solve(fx) -> Launch:
    """The producer's batched multi-level solve, on this cell's own routing.

    The logits are drawn once from the cell's seed and held: what is priced is the
    SOLVE, not the projection that produced them.
    """
    from rola.routing.entmax.production import production_routing_factor_levels

    spec = fx["cell"]
    producer = fx["layer"].routes
    routing = producer.routing
    gen = torch.Generator(device="cuda").manual_seed(spec.seed + 900)
    shape = (spec.B * spec.H, spec.tokens, routing.packed_logit_width)
    read_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    write_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)

    def run():
        return production_routing_factor_levels(read_logits, write_logits, routing)

    return Launch(name=f"entmax_solve|{spec.name}|{fx['construction'].name}", call=run)


def decode_step(fx) -> Launch:
    """ONE carried single-token step, on a state the seeding call populated.

    The state is MUTATED by the call, which is what a decode step is, so the launch's reset
    seeds a fresh state from the prefill before every call: each timed call is the same
    step, at the same context position, over the same residency.
    """
    import rola
    from rola.engine.dags.decode_dag import decode_forward
    from rola.engine.facts import planes
    from rola.ops.carry import box_leaves

    spec, widths = fx["cell"], fx["construction"].widths
    if spec.decode_steps == 0:
        raise ValueError(f"{spec.name} is a prefill-only cell; it has no decode subject")
    #: The seed binds through the facade's own entry and commits exactly the pages the
    #: routing writes -- a decode step's cost is a function of residency and routing,
    #: not of the numbers stored, so nothing has to have computed them.
    producer = fx["layer"].routes
    routes = fx["routes"]
    bits = planes.atom_bits(planes.pack_side(routes.write), widths)
    seeded: dict = {}

    def reset():
        seeded.clear()
        state = rola.state()
        with torch.no_grad():
            _s, arena = state._kernel_entry(routes, bits, d_v=spec.dv, paging=True, BC=box_leaves(spec.dv, 8))
            arena.wait()
        seeded.update(state=state, arena=arena)

    reset()
    token = fx["x"][:, spec.tokens:spec.tokens + 1]
    v_token = fx["v"][:, spec.tokens:spec.tokens + 1].contiguous()

    def run():
        with torch.no_grad():
            y, _ = decode_forward(producer(token), v_token, seeded["state"])
        return y

    #: THE ARENA'S PHYSICAL PAGES: under the VMM backing the driver maps them outside the caching allocator, so a memory
    #: measurement adds its own receipt; the dense bridge's one plane is a torch tensor the allocator already counts.
    return Launch(name=f"decode_step|{spec.name}|{fx['construction'].name}", call=run, reset=reset,
                  outside_allocator=lambda: seeded["arena"].committed_bytes if seeded["arena"].backing == "vmm" else 0)


#: THE ROSTER: one entry per kernel this line carries, plus the carry, whose arm refuses.
SUBJECTS = {
    "carry_forward": Subject("carry_forward", "carry", carry_forward,
                             "the inter term: one carry launch over a whole cell",
                             level="kernel", symbol="carry_kernel", stamp="carry_build_stamp", dials=("schedule",)),
    "intra_forward": Subject("intra_forward", "carry", intra_forward,
                             "the within-window term at the shared window",
                             level="kernel", symbol="intra_kernel", stamp="intra_build_stamp"),
    "carry_intra": Subject("carry_intra", "carry", carry_intra,
                           "the carry and the intra as one operator, over facts and support words the caller built: "
                           "the two kernels and the pair's own packing, never the op's facts or paging",
                           level="kernel", symbol="carry_kernel", stamp="carry_build_stamp", calls=(1, 4)),
    "liveness_pass": Subject("liveness_pass", "carry", liveness_pass,
                             "both sides' class-1 liveness words",
                             level="kernel", symbol="liveness_pass", stamp="csrc_build_stamp"),
    "entmax_solve": Subject("entmax_solve", "layer", entmax_solve,
                            "the producer's batched multi-level solve",
                            level="kernel", symbol="(union|factor|softmax)_forward_kernel",
                            stamp="csrc_build_stamp"),
    "decode_step": Subject("decode_step", "layer", decode_step,
                           "one carried single-token step through the engine: the facts pass, the pages it commits "
                           "and the kernels, as the op runs them",
                           level="op", symbol="decode_step_kernel", stamp="rola_decode_build_stamp"),
}

#: THE LEVELS A SUBJECT MAY PRICE, and what each owes (`Subject.level`). rola has no `layer` subject on this line: the
#: prefill arm (`rola.interface.rola_op`) refuses since the rebuild deleted the shipped chunk consumer, so neither the
#: op's prefill path nor a layer around it has anything to time. Decode is the one op-level path that runs.
LEVELS = ("kernel", "op", "layer")


def _shared_window() -> int:
    from rola.ops import carry as carry_ops

    return carry_ops.WINDOW


def _splits(tokens: int, calls: int) -> bool:
    """Whether a sequence runs as `calls` carried calls: one call takes any length, and a
    chain hands its state off on window boundaries, so each of its calls is whole windows."""
    return calls == 1 or tokens % (calls * _shared_window()) == 0


def applicable(spec, kind: str, calls: int = 1) -> list[str]:
    """Which subjects a cell can carry at a call count: its KIND, the counts a subject is
    priced at, the chain's whole windows, and each subject's own condition."""
    out = []
    for name, subject in SUBJECTS.items():
        if subject.kind != kind or calls not in subject.calls or not _splits(spec.tokens, calls):
            continue
        if name == "decode_step" and getattr(spec, "decode_steps", 0) == 0:
            continue
        if name == "intra_forward" and spec.tokens % _shared_window():
            continue
        #: the kernel pair runs the intra kernel's tile grid (whole windows) at one uniform level width
        if name == "carry_intra" and (spec.tokens % _shared_window() or len(set(spec.widths)) != 1):
            continue
        out.append(name)
    return out
