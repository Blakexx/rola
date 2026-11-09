# `common/atoms/clock.cuh` — the clock

Mirrors `csrc/rola/src/common/atoms/clock.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): the SM's cycle counter for the phase ledgers.

- <a id="cycles"></a>`cycles()`: `%clock64` by `mov`, for the phase ledgers; the `clock64` intrinsic is not declared
  in every host-pass compilation of a header both a launch and a harness include.
  <a id="cycles-after"></a>`cycles_after(after)` takes `after` as an input it does not use, so the read cannot issue
  before `after` arrives: ptxas deletes a clock read whose value is unused, so a dependency is the only way to order
  one.
