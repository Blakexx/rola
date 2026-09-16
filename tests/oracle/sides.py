# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CROWN JEWELS AS DIFF SIDES: what a comparison of two checkouts runs, one cell at a time.

The oracle pair -- `rola/ops/naive.py` and the routing references it composes with -- is what every Tier 1 gate is
anchored to, and a change to it cannot be reviewed by reading the diff: the question is never "is the new code
reasonable" but "does it compute the same function". So both versions are RUN, over the central cells, and compared
(`rola_devtools.diff`, declared in `declare.py`).

Each function here is a SIDE: it takes one cell's record and returns the quantities that side has to show. Nothing
here compares anything or states a tolerance -- the rule is the diff node's (`bit-identical` for the oracle,
`support-equal` for the producer) and the two checkouts' functions are the only things that differ.

THE OLD SIDE IS A SECOND CHECKOUT, not an import alias: the build system runs each side in its own environment, and
the side declares the library it must be running (`binds="rola"`), which the node proves rather than arranges for.
Everything here is CPU and fp64 -- the references are pure torch, so no build is involved and no device is touched.

    producer    the routing reference: the solve's amplitudes, its SUPPORT SET, tau, and the VJP through the public
                entry point. The support set is the claim with teeth: a threshold that moves by one entry is a routing
                change, not rounding, which is why the rule here is support equality and not bit-identity.
    oracle      `naive_rola` itself, FORWARD AND BACKWARD, on a carry cell's own draw. Compared BY BIT-IDENTITY: both
                sides run fp64 on identical inputs, so a rewrite that only nearly reproduces the old one has changed
                the definition.

THE BACKWARD HALF IS NOT OPTIONAL. A seeded VJP per cell over every operand, with "no gradient" recorded as an EMPTY
tensor rather than a zero one: the oracle's `.detach()` at the clock is its literal spec, and the difference between
"no gradient" and "a zero gradient" is precisely what a manufactured graph edge hides. The cotangents are `linspace`
and never random, so a generator's state cannot depend on how many cells ran before this one.
"""
from __future__ import annotations

import torch
from rola_devtools.cells import build as build_cell
from rola_devtools.cells import carry as carry_cells
from rola_devtools.cells import producer as producer_cells

#: "this input received NO gradient", as a tensor: an EMPTY one, never a zero one, because `None` and `0.0` are
#: different answers and a shape mismatch is how the comparison reports the difference
NO_GRAD = torch.zeros(0, dtype=torch.float64)


def _flatten(**named) -> dict:
    """The operands as `{key: tensor}`, each marked for grad. A tuple is expanded per level; a `None` operand simply
    does not appear, so a cell's key set states which operands that cell HAS."""
    flat = {}
    for key, value in named.items():
        if value is None:
            continue
        items = value if isinstance(value, tuple) else (value,)
        for index, tensor in enumerate(items):
            tensor.requires_grad_(True)
            flat[key if not isinstance(value, tuple) else f"{key}{index}"] = tensor
    return flat


def producer(record: dict, **_params) -> dict:
    """The routing reference on one producer cell: the solve, its support, and the VJP through the public entry."""
    from rola.routing.entmax.reference import entmax, entmax_forward
    from rola.routing.reference import _union_split_with_support, union_split

    cell = build_cell(record)
    drawn = producer_cells.realize(cell)
    if cell.pairs:
        read_logits, write_logits, mask = drawn
        read, write = union_split(read_logits=read_logits, write_logits=write_logits, alpha=cell.alpha, mask=mask)
        #: THE PACKED SUPPORT through the private entry on purpose: it is the reference's wire format, and a packing
        #: change the semantic outputs cannot see is exactly what the kernel would then be gated against
        packed = _union_split_with_support(read_logits=read_logits, write_logits=write_logits, alpha=cell.alpha,
                                           mask=mask)[2]
        return {"read": read, "write": write, "packed": packed}

    logits, mask = drawn
    p, support, tau = entmax_forward(z=logits, alpha=cell.alpha, dim=-1, mask=mask)
    grad_in = logits.clone().requires_grad_(True)
    values = entmax(z=grad_in, alpha=cell.alpha, dim=-1, mask=mask)
    cotangent = torch.linspace(-1.0, 1.0, values.numel(), dtype=torch.float64).reshape(values.shape)
    (dz,) = torch.autograd.grad(values, grad_in, grad_outputs=cotangent)
    return {"p": p, "support": support.to(torch.float64), "tau": tau, "dz": dz}


def oracle(record: dict, *, batch: int = 1, heads: int = 2, **_params) -> dict:
    """`naive_rola` on one carry cell, forward and backward, in fp64 on the CPU.

    The cell's own draw is the input -- the same record the kernel's own oracle gate runs on -- upcast to fp64,
    because what is being compared here is the REFERENCE's arithmetic and not a kernel's rounding.
    """
    from rola.ops.naive import naive_rola
    from rola.routing.types import IndependentRouting, LeafMassDecay, SoftmaxActivation, Topology

    cell = build_cell(record)
    drawn = carry_cells.realize(cell, B=batch, H=heads, device="cpu")
    read = tuple(x.double() for x in drawn.read)
    write = tuple(x.double() for x in drawn.write)
    v = drawn.v.double()
    g_write = drawn.gain.double()
    decay = None
    if getattr(cell, "decay", None):
        gen = torch.Generator().manual_seed(cell.seed + 1)
        decay = LeafMassDecay(dials=tuple(torch.rand(heads, w, generator=gen, dtype=torch.float64) * 0.3 + 0.1
                                          for w in cell.widths))
    #: THE SAME ROUTING THE ORACLE TIER BUILDS OVER -- untied softmax on both sides -- so this comparison and the
    #: kernel-vs-oracle gates are asking about one object
    routing = IndependentRouting(width=1, read=SoftmaxActivation(), write=SoftmaxActivation())
    topology = Topology(levels=tuple(routing.at(w) for w in cell.widths))
    operands = _flatten(v=v, g_write=g_write, read=read, write=write,
                        dials=None if decay is None else decay.dials)
    #: a CARRIED cell enters with its own state, which is the cell's claim and not this comparison's
    entry = None if drawn.entry is None else drawn.entry.double()
    y, state = naive_rola(v, read, write, g_write, topology, decay, initial_state=entry, output_final_state=True)
    out = {"y": y.detach(), "state": state.detach()}
    dy = torch.linspace(-1.0, 1.0, y.numel(), dtype=torch.float64).reshape(y.shape)
    dstate = torch.linspace(0.25, -0.75, state.numel(), dtype=torch.float64).reshape(state.shape)
    keys = sorted(operands)
    grads = torch.autograd.grad(outputs=(y, state), inputs=[operands[k] for k in keys],
                                grad_outputs=(dy, dstate), allow_unused=True)
    out.update({f"d{key}": (NO_GRAD if grad is None else grad.detach()) for key, grad in zip(keys, grads, strict=True)})
    return out


__all__ = ["NO_GRAD", "oracle", "producer"]


def carry_kernel(record: dict, **_params) -> dict:
    """THE CARRY KERNEL on one carry cell, its slots in the reference's layout -- the left side of rola's own
    kernel-vs-oracle diff, the same call `test_carry_vs_oracle.py` makes."""
    from tests.oracle.test_carry_vs_oracle import kernel_slots, run

    spec = build_cell(record)
    drawn, num, den, plane = run(spec)
    got_num, got_den, got_state = kernel_slots(spec, num, den, plane)
    return {"num": got_num, "den": got_den, "state": got_state}


def carry_reference(record: dict, **_params) -> dict:
    """THE fp64 REFERENCE on the same cell, with each slot's ENVELOPE beside it: the reference run again on `|v|` and
    `|state_in|`, which is the sum of the sizes of the terms each slot adds up (`tests.oracle.fixtures`)."""
    from benchmarks.cells import carry_call
    from tests.oracle.test_carry_vs_oracle import entry, ref

    spec = build_cell(record)
    drawn, _call = carry_call(spec, bh=1)
    want, env = ref(drawn, entry(spec, drawn)), ref(drawn, entry(spec, drawn), magnitudes=True)
    shape = (1, spec.N, spec.dv + 1)
    return {"num": want[0], "den": want[1], "state": want[2].reshape(shape),
            "num_envelope": env[0], "den_envelope": env[1], "state_envelope": env[2].reshape(shape)}


__all__ += ["carry_kernel", "carry_reference"]
