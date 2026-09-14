# `tools/compare.py` — the comparison

Mirrors `tools/compare.py` and `benchmarks/bench/provider.py`. A comparison times arms of rola checkouts, and of other
libraries, on the cells of one point, interleaved call by call. The method is rola-devtools' interleaving driver
(`rola_devtools.interleave`) and its registry of cells and points (`rola_devtools.cells`); this page says what rola adds.

## Cells, points and runners

A **cell** is a data provider and its constructor parameters. rola's two registry files are cells: `carry_cells.json`
names `benchmarks.cells:carry_cell` (a shape and a draw, validated into a `CellSpec`), `layer_cells.json` names
`benchmarks.cells.layer:layer_cell` (a constructor, a `LayerCellSpec`). Another repository's registry (`--registry`)
adds its own cells, such as rola-bench's attention cells.

A **point** is a named group of cells by **runner**, with what it holds equal (`holds`, recorded; `equal`, the parameters
every cell must share, checked on load). rola's runner is `rola`; a foreign arm names the runner the point addresses.
`--point NAME` takes a registered point, `--cells a,b` narrows it by name, and `--cells` alone is a point of rola cells for
the rola runner.

**rola's runner** (`bench.provider:arms`) receives each cell's data and returns the arms this checkout runs on it, or
refuses the cell by name:

| refusal | when |
|---|---|
| an iteration build | the binary lacks an arm `tools/manifests/shipped_set.json` ships (`ROLA_CARRY_ARMS`) |
| no carry arm | a carry cell whose (D, DV, warps_per_cta) the binary does not carry |
| not rola's data | a cell whose data is neither a carry nor a layer cell |

An arm is offered where the subject applies to the cell's own facts (`bench.subjects.applicable`: its kind, call counts,
whole windows, decode steps) and the binary carries the subject's kernel at the cell's shape: the intra arm at the cell's
depth and window for `intra_forward` and `prefill_op`, the decode arm for `decode_step`.

## The arms

A **rola arm** is a bench subject (`bench.subjects.SUBJECTS`) with its dials, and its name spells them:

| name | what it times |
|---|---|
| `carry_forward` | the subject at its defaults: one call, the `first` carry order, the `fresh` state arm |
| `carry_forward@schedule=identity` | the carry in token order |
| `prefill_op@calls=4@state=continuation` | the op as four carried calls over one plane |

A dial belongs to the arm, never to the run, and exists only where the subject reads it (`Subject.calls`,
`Subject.dials`); a dial the subject does not read has no name to set it by. A run-level schedule once reached no
binary and a whole week of records said `first` over master's unreaped dense order, which is why the dial lives in the
arm's name.

The runner builds rola's arms inside a worker run by the arm's checkout's own venv, from its own directory, so two rola
arms differ in exactly what their names say. Building an arm refuses it by name unless the device reports the subject's
family stamp (a path or a hash cannot catch a stale binary) and `import rola` resolved inside that checkout. What an arm
reports is the cell's facts, the dials, and the binary's: the manifest digest, the family stamp, the device, torch, the
assembler, and the SM clock read when the arm was built. Each sample is one launch between two CUDA events.

A **foreign arm** names a runner (`module:function`), the runner the point addresses (`runner:`, default its label), the
arm, the python that runs it and the directory it runs in: rola-bench's attention reference is one
(`rola_bench.measure.attention:arms`, arm `flash`).

## The run

Every arm runs on every cell the point sends its runner; a row is `label|cell`. The tool holds the GPU lock exclusively
for every timed call and engages the host's clock lock, proven by the device's clock read before the first call and
after the last; a second read off the lock refuses the comparison. The driver warms each row past its floor of 10
launches and then calls every row once per rep in a fresh random order, `--reps` (odd) per round for `--rounds` rounds.

Ratios pair within a cell. `--reference` is a label (the first arm's by default): its row on a cell pairs with every
other row on that cell, and a row on a cell the reference does not run (the attention arm's) pairs with each of its rows.
A `--reference` naming one row pairs every other row with it. The tool prints each row's median over the round medians,
their interquartile range, and the median of each pairing's per-rep ratios. `--out` writes the whole result: the point
with its cell records, every row's cell, what the runner built, raw samples in the order taken, pairings, and the clock.
`--record` stores it through `rola_results` at `compare`, keyed by the point and each arm's commit and diff digest.

```bash
python tools/compare.py --cells flat-small-alt-k16,flat-small-dense \
    --arm label:first,arm:carry_forward --arm label:identity,arm:carry_forward@schedule=identity --rounds 8
```
