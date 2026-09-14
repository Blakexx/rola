"""Canonical fp64 reference implementation for the routing-native RoLA oracle.

**THE CORRECTNESS REFERENCE IS THE TEXTBOOK, BECAUSE THE REDUCTION THEOREM SAYS
IT MUST BE.** RoLA's consumer is a gated linear-attention recurrence over a
routed leaf address space, and once the routing allocations are materialized it
IS textbook gated LA: one state per leaf, a diagonal gate, a rank-one deposit,
a linear readout. So the recurrence below is written as that and nothing else --
the plain-torch few-line form a reader can check against any linear-attention
paper -- and the parts that are OURS are pushed to the edges where they can be
audited: the leaf address map (``_leaf_product``), the gate values
(``keep = (1 - rate) ** c``) and the ratio readout.

Normalization uses the ONES-COLUMN trick rather than a second recurrence. The
mass state is what you get by running the same recurrence on a value vector of
all ones, so it is not a separate object: ``v`` gains a column of ones, the
state carries ``d_v + 1`` columns, and the last one IS the mass. One recurrence,
one state, one readout -- and the state's layout is then literally the
``[B,H,N,d_v+1]`` cache both backends exchange.

Implements the recurrence exactly. There is no Q/K, no feature map, and
no privileged routing level here -- every level is an ordinary mixed-radix
routing level, and the oracle consumes already-materialized per-level, per-side
routing allocations (``p_read``, ``p_write``) plus the positive write-side gain
(``g_write``). Computing those allocations from a
projected hidden state is PRODUCER scope and
is out of scope here.

This module intentionally does not import or reuse anything from the Q/K-era
oracle it replaces (``ResolvedRouting``, ``routing_factor_levels``,
``packed_router_logits_reference``, ``rola.ops.decay`` --
all tied to the packed-router / matrix-state ABI). The old oracle survives only
in git history at ``d6f73e97`` and is used there, read-only, by the migration
equivalence test.
"""

from __future__ import annotations

import torch

from rola.ops._oracle_contract import initial_cache, validate_inputs, validate_simplex
from rola.ops.constants import READOUT_EPS
from rola.routing.types import DecayConfig, Topology


def _leaf_capacity(widths: tuple[int, ...]) -> int:
    """``N = prod_l width_l`` -- the number of leaves the address space has.

    Derived here, from the widths alone. See :func:`_radix_strides` for why the
    oracle re-derives its own address map instead of importing production's.
    """
    capacity = 1
    for width in widths:
        capacity *= width
    return capacity


def _radix_strides(widths: tuple[int, ...]) -> tuple[int, ...]:
    """MSB-first mixed-radix strides: ``stride_l = prod(width_{l+1..D-1})``.

    **THE ORACLE DERIVES ITS OWN LEAF ADDRESSES, AND THAT IS THE POINT.** This is
    three lines that `rola.routing.types.Topology.radix_strides` also computes, and
    the duplication is deliberate: it is the difference between a reference and a
    second opinion. Every Tier 1 gate compares the kernel against this file, and the
    kernel's addressing comes from that production property; importing it here would
    make a wrong stride convention agree with itself in every gate the repository
    has. Two independent derivations disagree when one is wrong, which is the only
    reason to have a reference at all.

    The agreement is then a TESTED CLAIM rather than an unexamined shared import:
    `tests/oracle/test_oracle_independence.py` asserts these strides equal the
    production property over the whole supported width space, and is allowed to fail.

    The convention, stated so the reader can check it without leaving the file: leaf
    ``s`` decomposes MSB-first, level 0 being the most significant digit, so level
    ``l``'s digit is ``(s // stride_l) % width_l`` with ``stride_l`` the product of
    every width BELOW it (`ROLA_V3_SPEC.md` section 2). Level ``D-1`` therefore has
    stride 1 and varies fastest.
    """
    strides = []
    for index in range(len(widths)):
        stride = 1
        for width in widths[index + 1:]:
            stride *= width
        strides.append(stride)
    return tuple(strides)


def _leaf_product(
    levels: tuple[torch.Tensor, ...], radix_strides: tuple[int, ...], N: int
) -> torch.Tensor:
    """Materialize the complete per-leaf product from per-level per-digit factors.

    ``levels[l]`` has shape ``[..., width_l]``. Returns ``[..., N]`` where leaf
    ``s``'s entry is ``prod_l levels[l][..., d_l(s)]`` under the MSB-first
    mixed-radix digit decomposition ``d_l(s) = (s // radix_strides[l]) % width_l``
    This is the oracle's one leaf-materializing step;
    it is deliberately simple and auditable rather than fast.
    """
    device = levels[0].device
    state_ids = torch.arange(N, device=device, dtype=torch.long)
    leaves = None
    for level_factors, stride in zip(levels, radix_strides):
        width = level_factors.shape[-1]
        digit = (state_ids // stride) % width
        selected = level_factors.index_select(-1, digit)
        leaves = selected if leaves is None else leaves * selected
    return leaves


def naive_rola(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_write: torch.Tensor,
    topology: Topology,
    decay: DecayConfig,
    *,
    eps: float = READOUT_EPS,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the canonical routing-native RoLA reference in fp64 -- the executable spec.

    There is no ``q``, no ``k``, and no feature map anywhere in this function.
    Inputs are per-level per-side routing allocations (already normalized
    simplices, one tensor per level, shape ``[B,T,H,width_l]``), the positive
    write-side gain (``g_write``, shape ``[B,T,H]``; there is no read-side gain
    at all -- the readout is a ratio, which fixes it to 1 identically and
    production has no read-gain projection), a ``Topology``, and a ``DecayConfig``
    (``None`` or ``LeafMassDecay``).

    ``write_levels`` feeds the deposit ``W`` and trains normally, AND is what the
    ``LeafMassDecay`` clock reads (``c[t,s] = stop_gradient(prod_l p_write[...])``).
    There is no second, separately quantized clock input: the producer emits the
    stored form, so the levels handed here ARE the stored representation and a
    parameter for emulating one would be a second spelling of the same tensor.

    Returns ``(y, final_state)`` with ``y: [B,T,H,d_v]`` and, if
    ``output_final_state``, ``final_state: [B,H,N,d_v+1]`` (the last column
    carries the mass state the ratio readout divides by).

    BOTH OUTPUTS ARE ``float64``, WHATEVER THE INPUTS WERE. Inputs of any dtype
    are accepted and cast up; nothing is cast back down. This is the fp64 oracle,
    and a function that returned an fp32 ``y`` beside an fp64 state would be
    making its two claims about one run in two precisions -- an accuracy gate
    would then be comparing against a reference that had itself been rounded.
    Casting the result to a caller's working dtype is the caller's business, and
    it is the caller who knows why.

    AN INPUT THE OUTPUT DOES NOT DEPEND ON GETS ``None`` FOR ITS GRADIENT, which
    is autograd's ordinary answer and this function states nothing else. A test that
    wants a zero asserts ``grad is None`` and supplies the zero itself.
    """
    write_levels = tuple(write_levels)
    B, T, H, d_v = validate_inputs(
        v, read_levels, write_levels, g_write, topology, decay
    )
    #: DERIVED HERE, not read off `topology`. The oracle's address map is its own
    #: (see `_radix_strides`); `topology` supplies the SHAPE (`widths`), which is
    #: configuration the caller chose, never the interpretation of it.
    widths = topology.widths
    N = _leaf_capacity(widths)
    strides = _radix_strides(widths)

    v64 = v.to(torch.float64)
    read64 = tuple(level.to(torch.float64) for level in read_levels)
    write64 = tuple(level.to(torch.float64) for level in write_levels)
    validate_simplex("read_levels", read64)
    validate_simplex("write_levels", write64)

    g_write64 = g_write.to(torch.float64)

    #: THE ONES COLUMN. The mass state is the same recurrence run on a value of
    #: all ones, so it is a COLUMN of the value, not a second state. `[B,T,H,d_v+1]`.
    deposit_value = torch.cat(
        (v64, torch.ones((B, T, H, 1), device=v.device, dtype=torch.float64)), dim=-1)

    #: The read and write allocations, materialized per leaf. `[B,T,H,N]`. The
    #: readout is a ratio, so the read side carries no gain column at all.
    R = _leaf_product(read64, strides, N)
    W = g_write64[..., None] * _leaf_product(write64, strides, N)

    #: The gate. `keep[t,s] = (1 - rate[s]) ** c[t,s]` -- a DIAGONAL gate, which is
    #: what makes this the gated-LA recurrence and not something new. `rate` trains
    #: through the dials; `c`, the clock, is stop-gradient by definition.
    if decay is None:
        keep = torch.ones((B, T, H, N), device=v.device, dtype=torch.float64)
    else:
        dials = tuple(dial.to(device=v.device, dtype=torch.float64) for dial in decay.dials)
        rate = _leaf_product(dials, strides, N)                       # [H,N]
        c = _leaf_product(write64, strides, N).detach()               # [B,T,H,N]
        keep = torch.pow(1.0 - rate, c)

    #: The state, in the layout it is handed back in: one row per leaf, `d_v`
    #: columns of value plus the ones column's mass.
    state = initial_cache(
        initial_state, B=B, H=H, N=N, d_v=d_v, device=v.device
    )

    outputs = []
    for t in range(T):
        # S[t,s,:] = keep[t,s] * S[t-1,s,:] + W[t,s] * v[t,:]
        state = keep[:, t, ..., None] * state + W[:, t, ..., None] * deposit_value[:, t, :, None, :]
        # o[t,:] = sum_s R[t,s] * S[t,s,:], inclusive read-after-write
        readout = torch.einsum("bhn,bhnv->bhv", R[:, t], state)
        outputs.append(readout[..., :d_v] / (readout[..., d_v:] + eps))

    y = (torch.stack(outputs, dim=1) if outputs
         else v64.new_zeros((B, 0, H, d_v)))
    return y, (state if output_final_state else None)


__all__ = ["naive_rola"]
