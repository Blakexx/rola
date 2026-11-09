# `measure/` — rola's reading of the central cells, the benches over them, and the executors of its declarations

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**THE CELLS ARE ROLA-DEVTOOLS' (`rola_devtools.cells`), and there is one registry for every package.** A cell is an
input -- a carry cell's routing amplitudes, gain, values and the state it enters with; a layer cell's hidden states and
values -- with its draw and the regime it proves, so a name means one input everywhere it appears, in rola and in the
libraries rola is compared against. A cell names nothing rola runs it with. `measure/cells/` is rola's reading:

* `cells/__init__.py` — a central cell's typed record by name (`by_name`), the carry cells by tier (`carry_cells`) and
  a cell's draw (`realize`). What the carry kernel is given for a cell is built where it is timed
  (`subjects.carry_operands`).
* `cells/layer.py` — RoLA's layer CONSTRUCTIONS (a routing template, widths, a gain), each named with the central layer
  inputs it was declared for, from which the amplitudes are PRODUCED. It is the only way to price the producer's own
  solve or a decode step through the layer.

**`measure/executors.py` — WHAT ROLA'S DECLARATIONS RUN.** The repository root's `declare.py` declares this
checkout's targets for rola-devtools' build system (`rola_devtools.build`): its build, the machine facts, its instruments
(SASS, register walk, phase clock, pipe counters, stall census, timeline, the intra roofline), every bench subject as a
timing entry on the central cells it takes, and its clock reader. `executors.py` is what those targets run, in this
checkout's venv. `bazel run //:measure -- run declare.py:all` measures this checkout by itself; rola-bench's root
loads the same `declare.py` from every checkout it measures.

**`measure/lab/` — THE LAB.** Small CUDA experiments on the machine itself (what a pipe costs beside the tensor pipe,
what ptxas does to a pattern), each a `experiments/<name>.cu` on the kit `lab.cuh`, compiled, SASS-gated, timed and
recorded by `tools/lab.py` into `results/<name>/<sm_XX>.json`; a KERNEL_STANDARDS rule about the machine cites one
(docs/internals/tools/lab.md).

**`measure/` — the library.** `subjects.py` is the roster: one lean callable per
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

**A timing session** interleaves timing entries of rola checkouts, and of other libraries, call by call under one
stopwatch and the GPU and clock locks (`rola_devtools.timing`, `docs/measurement.md`), and stores the raw ordered
samples; whether a stored difference is a regression is `bazel run //:results -- verdict`.

```bash
bazel run //:measure -- run declare.py:all --only 'rola/phases' --only 'rola/sass'
```

**`measure/harness/` — the carry's calibration** (`bench_carry_calib.py`).

The ROOFLINE instrument moved to `tools/roofline.py`, where every other instrument lives: it is one of
`declare.py`'s instruments and never was a bench, and being the only one outside `tools/` is what kept it off the
tools map (`docs/internals/tools/README.md`).

