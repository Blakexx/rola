"""The judgement over measurements (the record itself is `rola_results`, docs/measurement.md).

Two modules, because judging has two jobs and neither is a fixture:

* :mod:`bench.stats` -- the statistics. The paired signed-rank verdict and its
  committed manifest, so a claim of "faster" is a test and not a pair of medians.
* :mod:`bench.regression` -- the judgement. Per-cell thresholds derived from that
  cell's own measured IQR, and the clock-settling precondition a comparable number
  requires.

WHAT is measured is not here: the fixtures and the timing driver belong to the arm
being measured, and the chunk arm's harness is its own card.
"""
