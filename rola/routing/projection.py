"""Packed router projection for a :class:`ResolvedRouting` plan."""

from __future__ import annotations

import torch

from rola.routing.types import ResolvedRouting


def _packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
    projection_dtype: torch.dtype,
    token_major: bool = False,
    gain_columns: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project packed router weights in one explicitly selected arithmetic domain.

    Returns ``(read_logits, write_logits)``, each ``[B*H, T, routing.packed_logit_width]``:
    one side's levels laid out end to end on ``routing.level_offsets``, so level ``l`` is
    ``logits[:, :, routing.level_logit_slice(l)]``. There is no per-level padding to
    ``max(branches)`` -- see ``ResolvedRouting.packed_logit_width``.

    ``token_major`` returns the same numbers as ``[B, T, H, packed_logit_width]``
    instead, WITHOUT the transpose that flattening ``(B, H)`` needs. The sgemm's own
    output is already this layout, so the flat form costs a full pass over the plane
    that the flat form is the only reason for; the solve reads either through its stride
    tuple. BIT-IDENTICAL, and that is checkable rather than argued: a column of the
    output is a dot product over the full hidden dim, so it does not depend on where the
    column is written -- gated by
    ``test_the_two_projection_layouts_are_one_gemm_and_agree_at_the_bit``.

    ``route_W`` is ``[H, hidden_size, routing.packed_router_width + gain_columns]``: the
    stored parameter IS the packed layout, one slot's columns per level it feeds, then
    the side-gain columns. The gain rides whichever projection reads ``h`` -- it is a
    function of the READ stream by definition -- so on the shared-stream path it is
    columns of the one GEMM the routing already pays for, and it never becomes a launch
    that would not have happened anyway.

    Returns ``(read_logits, write_logits, gain_logits)``; the third is ``None`` when
    ``gain_columns`` is 0 and ``[..., gain_columns]`` otherwise, in the same layout.
    """
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if type(gain_columns) is not int or gain_columns < 0:
        raise ValueError(f"gain_columns must be a nonnegative exact int, got {gain_columns!r}")
    routed = routing.packed_router_width
    heads = routing.validate_packed_router(
        route_W[:, :, :routed] if gain_columns else route_W,
        None if route_bias is None or not gain_columns else route_bias[:, :routed])
    if route_W.shape[2] != routed + gain_columns:
        raise ValueError(
            f"route_W must carry {routed} routing columns and {gain_columns} gain "
            f"columns, got {route_W.shape[2]}")

    def validate_hidden(hidden: torch.Tensor, name: str) -> None:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim not in (3, 4):
            shape = getattr(hidden, "shape", None)
            raise ValueError(f"{name} must have rank 3 or 4, got {shape}")
        hidden_width = hidden.shape[-1]
        if hidden_width != route_W.shape[1]:
            raise ValueError(
                f"{name} hidden width must match route_W hidden width {route_W.shape[1]}, "
                f"got {hidden_width}")
        if hidden.ndim == 4 and hidden.shape[2] != heads:
            raise ValueError(
                f"{name} rank-4 head count must match route_W heads {heads}, "
                f"got {hidden.shape[2]}")

    validate_hidden(h, "h")
    if h_w is not None:
        validate_hidden(h_w, "h_w")
        if h_w.shape != h.shape or h_w.device != h.device:
            raise ValueError("h_w must match h shape and device")

    # THE PARAMETER IS THE PACKED LAYOUT. `route_W` is stored `[H, hidden,
    # packed_router_width]` with each slot owning exactly the `branches[level]` columns
    # its level consumes (`ResolvedRouting.packed_slot_offsets`). There are no padded
    # columns to strip, so gathering a side's slots is span arithmetic on ONE tensor,
    # and `packed_column_runs` merges abutting spans: the write side is always
    # `range(D)`, one run, so its weight is a SLICE and no copy happens at all. A fully
    # tied plan's read side is that same run; a fully untied plan's read side is the
    # trailing run. Only a mixed-tie plan interleaves the two regions and still copies.
    #
    # BIT-IDENTICAL, and that is the whole safety argument: a column of the output is a
    # dot product over the full hidden dim and is unaffected by which OTHER columns share
    # the GEMM, or by whether the weight it reads is a view or a copy of the same values.
    # So `test_projection_tf32_support_gate.py` -- the gate that exists because effd23c
    # moved this GEMM's arithmetic and flipped 14-95 entmax support entries -- sees
    # byte-for-byte the logits it sees today, as do the fp64 oracle gates.
    def pack_weight(slots: tuple[int, ...], *, with_gain: bool = False):
        """One side's slot columns, as a view when they are one run and a copy when not.

        ``with_gain`` appends the trailing gain span, MERGED into the last run when it
        abuts it -- which is every case the shipped configurations hit, so the fused
        weight is `route_W` itself and no gather happens anywhere on the step.
        """
        runs = list(routing.packed_column_runs(slots))
        if with_gain:
            span = (routed, routed + gain_columns)
            if runs and runs[-1][1] == span[0]:
                runs[-1] = (runs[-1][0], span[1])
            else:
                runs.append(span)
        runs = tuple(runs)
        if len(runs) == 1:
            (start, stop), = runs
            weight = route_W[:, :, start:stop]
            bias = None if route_bias is None else route_bias[:, start:stop]
            return weight, bias
        weight = torch.cat([route_W[:, :, start:stop] for start, stop in runs], dim=2)
        if route_bias is None:
            return weight, None
        bias = torch.cat([route_bias[:, start:stop] for start, stop in runs], dim=1)
        return weight, bias

    def project(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                *, trailing: int = 0):
        """``(logits, trailing_columns)`` -- the routing logits in the layout
        ``token_major`` selects, and the last ``trailing`` columns always as
        ``[B, T, H, trailing]``, which is the layout the gain's consumer reads."""
        #: `blhc` is the sgemm's OWN output order, so the einsum returns a view of it and
        #: the head axis costs nothing; `bhlc` is that plus a full-plane transpose, which
        #: exists only to let `(B, H)` flatten into one stream axis.
        order = "blhc" if token_major else "bhlc"
        equation = f"bld,hdc->{order}" if hidden.ndim == 3 else f"blhd,hdc->{order}"
        logits = torch.einsum(equation, hidden.to(projection_dtype), weight.to(projection_dtype))
        if bias is not None:
            logits = logits + bias.to(projection_dtype)[None, None, :, :] if token_major \
                else logits + bias.to(projection_dtype)[None, :, None, :]
        tail = None
        if trailing:
            tail = logits[..., -trailing:]
            if not token_major:
                tail = tail.permute(0, 2, 1, 3)
            logits = logits[..., :-trailing]
        if token_major:
            return logits, tail
        return logits.reshape(hidden.shape[0] * heads, hidden.shape[1], logits.shape[3]), tail

    write_slots = tuple(range(routing.D))
    # `packed_read_slot_lookup` already resolves "which slot feeds this level's READ
    # logits" for tied, untied and mixed plans, so the three-branch slot arithmetic this
    # function carries collapses into one lookup. Fully tied is exactly
    # `read_slots == write_slots`, stated structurally rather than as `not untied_levels`.
    read_slots = routing.packed_read_slot_lookup
    fully_tied = read_slots == write_slots

    #: THE GAIN RIDES THE READ STREAM'S PROJECTION, in every arm that has one. The gain
    #: is `f(h)` by definition, so the only arm that cannot fuse it is the one where no
    #: projection reads `h` at all -- fully tied over SPLIT streams -- and there it pays
    #: exactly what it paid before.
    fuse = gain_columns > 0 and (h_w is None or not fully_tied)

    tail = gain_columns if fuse else 0

    def standalone_gain():
        """The one arm with no projection of `h` to ride: fully tied over SPLIT streams."""
        if fuse or gain_columns == 0:
            return None
        return project(h, *pack_weight((), with_gain=True), trailing=gain_columns)[1]

    if fully_tied:
        # ONE solve serves both duties, and callers (the arena's `fully_shared`) rely
        # on getting the SAME TENSOR OBJECT back twice.
        tied_logits, gain_logits = project(
            h if h_w is None else h_w, *pack_weight(write_slots, with_gain=fuse),
            trailing=tail)
        return tied_logits, tied_logits, gain_logits if fuse else standalone_gain()

    if h_w is None:
        # One stream, so both sides are columns of ONE GEMM. Packed as ONE slot list
        # rather than two packs concatenated: a fully untied plan's write and read runs
        # abut, so the merged run is the whole of `route_W` and the GEMM reads the
        # parameter itself with no gather anywhere on the step.
        both_weight, both_bias = pack_weight(write_slots + read_slots, with_gain=fuse)
        both, gain_logits = project(h, both_weight, both_bias, trailing=tail)
        width = routing.packed_logit_width
        return both[..., width:], both[..., :width], gain_logits

    # Split read/write streams: the write side is projected from `h_w`, and a TIED level
    # takes its read logits from that write projection (the write stream is its source),
    # so only the untied levels are projected from `h`.
    write_logits, _ = project(h_w, *pack_weight(write_slots))
    untied = routing.untied_levels
    if len(untied) == routing.D:
        read_weight, read_bias = pack_weight(read_slots, with_gain=fuse)
        read_logits, gain_logits = project(h, read_weight, read_bias, trailing=tail)
        return read_logits, write_logits, gain_logits
    untied_weight, untied_bias = pack_weight(
        tuple(read_slots[level] for level in untied), with_gain=fuse)
    untied_logits, gain_logits = project(h, untied_weight, untied_bias, trailing=tail)
    untied_offsets = {}
    cursor = 0
    for level in untied:
        untied_offsets[level] = cursor
        cursor += routing.branches[level]
    read_logits = torch.cat(
        [
            untied_logits[..., untied_offsets[level]:untied_offsets[level] + routing.branches[level]]
            if level in untied_offsets
            else write_logits[..., routing.level_logit_slice(level)]
            for level in range(routing.D)
        ],
        dim=-1,
    )
    return read_logits, write_logits, gain_logits


def packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project production router logits in fp32."""
    read, write, _ = _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float32)
    return read, write


def packed_router_logits_reference(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project canonical reference router logits in fp64."""
    read, write, _ = _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float64)
    return read, write
