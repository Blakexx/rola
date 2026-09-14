"""Canonical mixed-radix routing topology."""


MAX_ROUTING_DEPTH = 4
MAX_BRANCH_WIDTH = 256

#: THE KERNEL-FACING FLOOR (padding happens at the producer; the floor is B_l >= 16):
#: every level the state descriptor addresses is a power of two at or above this --
#: ``rola._state.MIN_LEVEL_WIDTH`` is the same number, asserted equal in
#: ``tests/unit/test_topology_padding.py`` so the two never drift apart. It lives here,
#: not imported from ``rola._state``, because this module has no reason to import the
#: state/paging stack for one integer, and ``rola._state`` already predates this file's
#: padding concern: ``B`` already carries only the lawful, padded width.
PADDED_LEVEL_FLOOR = 16


def padded_level_width(width: int, *, floor: int = PADDED_LEVEL_FLOOR) -> int:
    """The next power of two at or above ``floor`` that admits ``width`` real digits.

    PADDING AT THE PRODUCER: a caller's level width ``b_l`` (any int >= 1, the
    API contract) is served by the SHIPPED, KERNEL-FACING width ``B_l`` this function
    derives -- never a second input a caller supplies, so a call can never present a
    ``B_l`` that disagrees with its own ``b_l``. ``b_l <= floor`` still pads to
    ``floor`` ("dissolves the D=1/N<32 corner -- a toy state pads to one warp
    with dead slots"); a ``b_l`` already a power of two at or above ``floor`` is its
    own answer, so an already-ideal model pays nothing through this function (though
    the producer still calls it, uniformly -- KERNEL_STANDARDS §R9: a structure branch
    is a derived fact, never a data-dependent shortcut).
    """
    if type(width) is not int or width < 1:
        raise ValueError(f"a level width must be a positive exact int, got {width!r}")
    needed = max(width, floor)
    return 1 << (needed - 1).bit_length()


def validate_level_width(width, *, owner="routing level") -> int:
    """Validate ONE level's branch width, where the level's own spec declares it.

    The per-level bound and the whole-list bound are separate checks because the width
    now arrives with the level object, one at a time, before any list of them exists:
    a level config that cannot be built is refused at its own constructor rather than
    at the assembly that would have collected it. ``validate_routing_shape`` remains
    the depth check and the list-wide restatement of this one.
    """
    if type(width) is not int:
        raise TypeError(f"{owner}.width must be an exact int, got {width!r}")
    if width < 1:
        # The contract states only the upper bound; a width < 1 is not a modeling
        # choice, it makes N and the radix strides ill-defined.
        raise ValueError(f"{owner}.width must be >= 1, got {width}")
    if width > MAX_BRANCH_WIDTH:
        raise ValueError(f"{owner}.width must be <= {MAX_BRANCH_WIDTH}, got {width}")
    return width


def validate_routing_shape(branches, *, owner="routing") -> tuple[int, ...]:
    """Validate the production routing-tree depth and per-level width limits."""
    branches = tuple(branches)
    if not branches:
        raise ValueError(f"{owner} must contain at least one routing level")
    if len(branches) > MAX_ROUTING_DEPTH:
        raise ValueError(
            f"{owner} routing depth D must be <= {MAX_ROUTING_DEPTH}; got D={len(branches)}")
    for level, width in enumerate(branches):
        if type(width) is not int:
            raise TypeError(f"{owner} branch widths must be exact ints, got {branches!r}")
        if width > MAX_BRANCH_WIDTH:
            raise ValueError(
                f"{owner} branch width must be <= {MAX_BRANCH_WIDTH}; "
                f"got level {level} width {width}")
    return branches


def resolve_b_spec(b, levels) -> tuple[int, ...]:
    """Normalize an integer or mixed-radix branch specification."""
    if type(levels) is not int:
        raise TypeError(f"levels must be an exact int, got {levels!r}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if levels > MAX_ROUTING_DEPTH:
        raise ValueError(
            f"routing depth D must be <= {MAX_ROUTING_DEPTH}; got D={levels}")
    if type(b) is int:
        branches = [b] * levels
    else:
        try:
            branches = list(b)
        except TypeError as error:
            raise TypeError(
                "states_per_level must be an exact int or an iterable of exact ints"
            ) from error
        if any(type(width) is not int for width in branches):
            raise TypeError(f"states_per_level entries must be exact ints, got {branches!r}")
    if len(branches) != levels:
        raise ValueError(f"states_per_level must have exactly D={levels} entries, got {len(branches)}")
    if any(width < 2 for width in branches):
        raise ValueError(f"states_per_level entries must be >= 2, got {branches}")
    validate_routing_shape(branches, owner="states_per_level")
    return tuple(branches)


def canonical_radix_strides(branches) -> tuple[int, ...]:
    """Return MSB-first state-id to digit divisors without materializing digits."""
    branches = tuple(branches)
    if not branches or any(type(width) is not int or width < 2 for width in branches):
        raise ValueError("branches must be a nonempty sequence of exact ints >= 2")
    validate_routing_shape(branches)
    strides = []
    suffix = 1
    for width in reversed(branches):
        strides.append(suffix)
        suffix *= width
    return tuple(reversed(strides))
