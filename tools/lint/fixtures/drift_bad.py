"""A DELIBERATELY NON-COMPLIANT fixture for tools/lint/drift_guards.py. Not imported."""

NORMALIZATION_KINDS = ("global",)

CARRY_ARMS = ((2, 64, 8), (2, 128, 4))


def check(state_before, state_after):
    # The pass used to fold this the other way round.
    import torch

    return torch.equal(state_before, state_after)
