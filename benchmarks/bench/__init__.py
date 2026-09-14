"""The bench library: what rola's kernels are timed as. The method, the statistics and the record are docs/measurement.md.

* :mod:`bench.subjects` -- the roster: one lean callable per kernel this line carries, over a registry cell.
* :mod:`bench.provider` -- the roster as arms of rola-devtools' interleaving driver, which `tools/compare.py` runs.
* :mod:`bench.carry_model` -- what the design says each component of the carry must cost on a cell.
* :mod:`bench.hw_profile` -- a hardware profile, never a hostname.

The cells are `benchmarks/cells`.
"""
