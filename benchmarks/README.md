# `benchmarks/` — rola's reading of the central cells, the benches over them, and the measurement registry

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**THE CELLS ARE ROLA-DEVTOOLS' (`rola_devtools.cells`), and there is one registry for every package.** A cell is an
input -- a carry cell's routing amplitudes, gain, values and the state it enters with; a layer cell's hidden states and
values -- with its draw and the regime it proves, so a name means one input everywhere it appears, in rola and in the
libraries rola is compared against. A cell names nothing rola runs it with. `benchmarks/cells/` is rola's reading:

* `cells/__init__.py` — what the carry kernel is given for a carry cell: its state descriptor, the launch shape
  (`WARPS_PER_CTA`), the arm key `(D, DV, warps_per_cta)`, the mode word of its declared sparsity, its liveness words and
  activity bytes, the state its `state` and `backing` bind, and `carry_call` — one cell's WHOLE call, the same operands in
  a correctness run and a measured one.
* `cells/layer.py` — RoLA's layer CONSTRUCTIONS (a routing template, widths, a gain), each named with the central layer
  inputs it was declared for, from which the amplitudes are PRODUCED. It is the only way to price the producer's own
  solve or a decode step through the layer.

**`benchmarks/registry.py` — THE MEASUREMENT REGISTRY.** This checkout's units for rola-devtools' measurement service
(`rola_devtools.measure`): its build, its clock reader, its instruments (SASS, register walk, phase clock, pipe counters,
stall census, timeline) and every bench subject as a timed arm, each accepting the central cells this binary runs it on.
`python -m rola_devtools.measure run benchmarks.registry:registry ...` runs it from here; rola-bench's composer runs it
once per checkout it measures.

**`benchmarks/bench/` — the library.** `subjects.py` is the roster: one lean callable per
kernel this line carries, each taking a registry cell and returning the launch to time,
with everything the launch does not pay for built outside the timed callable. The roster
mirrors the oracle roster one for one — a bench roster that does not match the correctness
roster is a roster with kernels nobody measures — and it includes the carry, whose body
does not exist on this line and whose arm therefore refuses by name rather than reporting
a number for something that did not run. `provider.py` is rola's runner: given a central cell
it offers the subjects this binary runs on it, with their dials
(`carry_forward@schedule=identity`) and on a layer cell their constructions
(`decode_step@layer=chunk-decode-w16`), or refuses the cell by name; arms build in their own
checkout's venv.

**`tools/compare.py` — THE COMPARISON.** Arms of rola checkouts, and of other libraries, on
the cells of one point, interleaved call by call under one stopwatch
(`docs/internals/tools/compare.md`). It takes the GPU lock and the clock lock itself and is
invoked BARE; never wrap it in an external `flock` on the same path (it self-deadlocks).
Whether a stored difference is a regression is `python -m rola_results verdict`.

```bash
python tools/compare.py --cells flagship-alt-k4 --arm label:first,arm:carry_forward \
    --arm label:identity,arm:carry_forward@schedule=identity
```

**`benchmarks/unit/` — the carry's part harness and its calibration** (`bench_carry_parts.py`,
`bench_carry_calib.py`), gated against `bench/carry_model.py`'s budgets.

**`benchmarks/bench_intra.py`** adds the ROOFLINE to the registered intra bench: the same
cells and the same timed callable, with the device's own mma.sync ceiling beside them.

