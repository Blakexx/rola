"""Fixture: two knobs the rule must stay quiet about -- one moved, one declared."""


def fixture_moved(operand, live_knob=None):
    return operand, live_knob


def fixture_held(operand, held_knob=8):  #: reserved: the two-CTA launch shape
    return operand, held_knob


def caller_one():
    return fixture_moved(1, live_knob=None)


def caller_two():
    return fixture_moved(2, live_knob="a different value")


def caller_three():
    return fixture_held(3, held_knob=8)
