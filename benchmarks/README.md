# `benchmarks/` — the cell registry, the benches over it, and the record

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**`benchmarks/cells/` — THE REGISTRY, and there is only one.** Every cell any test,
bench or tool runs is a record here, so a name means one shape and one draw everywhere it
appears. Two kinds, both data:

* `carry_cells.json` — the KERNEL cells: a record declares a SHAPE and a DRAW
  (amplitudes put directly on the simplex) and never an arm, since the arm key
  `(D, DV, warps_per_cta)` is derived from the record. `cells/__init__.py` validates the
  records, realizes a draw from the cell's own name (the seed is `crc32(name)`, so a
  failure reproduces from the name alone) and builds `carry_call` — one cell's WHOLE call,
  the same operands in a correctness run and a measured one.
* `layer_cells.json` — the LAYER cells: a record declares a CONSTRUCTOR (a producer, a
  routing template, a gain), from which the amplitudes are PRODUCED. It is the only way to
  price the producer's own solve or a decode step through the layer. `cells/layer.py`
  builds the fixture and checks `layer_manifest.json`, the committed INTEGER statistics
  every constructor realizes — a statistic is not a fixture, and the manifest exists so a
  drift in the RNG, the solver or the producer announces itself as a diff.
* `registry.py` — every cell by name with its kind, and `registry()`: these files as cells in rola-devtools' sense (each
  names its data provider, `carry_cell` or `layer_cell`) with any other registry files, cells and the points that group
  them by runner.

**`benchmarks/bench/` — the library.** `subjects.py` is the roster: one lean callable per
kernel this line carries, each taking a registry cell and returning the launch to time,
with everything the launch does not pay for built outside the timed callable. The roster
mirrors the oracle roster one for one — a bench roster that does not match the correctness
roster is a roster with kernels nobody measures — and it includes the carry, whose body
does not exist on this line and whose arm therefore refuses by name rather than reporting
a number for something that did not run. `provider.py` is rola's runner: given a cell's
data it offers the subjects this binary runs on it, with their dials
(`carry_forward@schedule=identity`), or refuses the cell by name; arms build in their own
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

**`tools/probe_cells.py` — the A/B orchestrator, and the only cross-binary instrument.**
It defines no step, no cell and no shape: it runs a REGISTERED BENCH over REGISTERED
CELLS on two binaries, each in its own worktree's venv, interleaved round by round, with
a device-side symbol assert before it times anything and an optional `ncu` pass whose
filter names the body. A bench that cannot be A/B'd through it is the defect — it means a
second definition of a launch exists somewhere.

**`benchmarks/bench_intra.py`** adds the ROOFLINE to the registered intra bench: the same
cells and the same timed callable, with the device's own mma.sync ceiling beside them.

**`benchmarks/bench_layer.py`, `bench_scaling.py` and `bench_paging.py`** are curve and
attribution studies rather than gated cells — a length sweep against attention, and the
residency question — and they state their own usage lines.
