"""The ratchet gates on what a commit adds: a new finding fails, a fixed finding must leave the baseline, and a baseline
never gains an entry over HEAD's (tools/lint/ratchet.py). No GPU."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "lint"))

import ratchet  # noqa: E402

A, B = ("a.py", "unused function 'f'", "0" * 16), ("b.py", "names the work code 'P1'", "1" * 16)


def test_the_tree_at_its_baseline_passes():
    assert ratchet.verdict(Counter({A: 2}), Counter({A: 2}), Counter({A: 2})) == (Counter(), Counter(), Counter())


def test_a_new_finding_and_a_second_copy_of_an_old_one_are_new():
    new, fixed, grown = ratchet.verdict(Counter({A: 3, B: 1}), Counter({A: 2}), Counter({A: 2}))
    assert new == Counter({A: 1, B: 1}) and not fixed and not grown


def test_a_fixed_finding_must_leave_the_baseline():
    new, fixed, grown = ratchet.verdict(Counter({A: 1}), Counter({A: 2}), Counter({A: 2}))
    assert fixed == Counter({A: 1}) and not new and not grown


def test_a_baseline_that_gains_over_head_fails_even_when_it_matches_the_tree():
    new, fixed, grown = ratchet.verdict(Counter({A: 1, B: 1}), Counter({A: 1, B: 1}), Counter({A: 1}))
    assert grown == Counter({B: 1}) and not new and not fixed


def test_the_baseline_round_trips():
    entries = Counter({A: 2, B: 1})
    assert ratchet._decode(ratchet._encode("x", entries)) == entries
