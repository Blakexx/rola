"""Fixture: a knob nothing moves. `constant_parameters.py` MUST fire on `dead_knob`."""


def fixture_entry(operand, dead_knob=None):
    return operand, dead_knob


def caller_one():
    return fixture_entry(1, dead_knob=None)


def caller_two():
    return fixture_entry(2, dead_knob=None)
