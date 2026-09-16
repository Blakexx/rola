"""The bench library: what rola's kernels are timed as. The method, the statistics and the record are docs/measurement.md.

* :mod:`measure.subjects` -- the roster: one lean callable per kernel this line carries, over a registry cell.
* :mod:`measure.provider` -- the roster as timed arms for rola-devtools' timing system (`declare.py`'s timing entries).
* :mod:`measure.carry_model` -- what the design says each component of the carry must cost on a cell.
* :mod:`measure.hw_profile` -- a hardware profile, never a hostname.

The cells are rola-devtools' central registry, read through `measure/cells`.
"""
