"""The compliant twin of drift_bad.py, for tools/lint/drift_guards.py. Not imported."""

DECAY_KINDS = ("none", "scalar")


def bytes_equal(a, b):
    """A byte comparison, which is what a storage identity claim is about."""
    return a.view("uint8").tobytes() == b.view("uint8").tobytes()


def declared_arms():
    """The arm table, read from the declaration rather than restated here."""
    import gen_shards

    return gen_shards.CARRY_ARMS


def check(state_before, state_after):
    """Storage identity is a claim about bytes."""
    return bytes_equal(state_before, state_after)
