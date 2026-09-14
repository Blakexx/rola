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

**`benchmarks/bench/` — the library.** `subjects.py` is the roster: one lean callable per
kernel this line carries, each taking a registry cell and returning the launch to time,
with everything the launch does not pay for built outside the timed callable. The roster
mirrors the oracle roster one for one — a bench roster that does not match the correctness
roster is a roster with kernels nobody measures — and it includes the carry, whose body
does not exist on this line and whose arm therefore refuses by name rather than reporting
a number for something that did not run. `driver.py` is one driver's whole body;
`pairing.py` is the interleaved paired-ratio engine; `discipline.py` is the
preconditions, enforced; `stats.py` and `regression.py` are whether a difference is real;
`store.py` and `ledger.py` are where a number lives afterwards.

**`benchmarks/unit/` — one driver per registered bench.** Each is `bench/driver.py` plus
the name of its subject, so the discipline cannot be acquired in five slightly different
ways. An arm spec is `subject@cell`, both halves names from the registry, so a command
line is citable and a row's `arm` field reads the same way.

**THE LOCK: this harness takes it itself, and every driver is invoked BARE.**
`bench.discipline.disciplined` holds `tools/gpu_lock.py`'s `gpu_lock()` for its whole
body; never wrap a driver in an external `flock` on the same path (it self-deadlocks).

```bash
python benchmarks/unit/bench_liveness.py --tier landing
python benchmarks/unit/bench_intra_forward.py --tier landing --cells flagship-alt-k4
python benchmarks/unit/bench_entmax.py --tier landing
python benchmarks/unit/bench_decode_step.py --tier landing
python benchmarks/unit/bench_carry_forward.py --tier landing   # refuses: no body yet
```

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
