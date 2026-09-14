# `tools/compare.py` — the comparison

Mirrors `tools/compare.py` and `benchmarks/bench/provider.py`. A comparison times arms of rola checkouts, and of other
libraries, at one registered cell, interleaved call by call. The method is rola-devtools' interleaving driver
(`rola_devtools.interleave`); this page says what rola adds to it.

## The point and the arms

The **point** is a registered cell (`benchmarks/cells`) with the facts another library's arms read from it: its tokens
and value width. The point is what the comparison holds equal. With an arm from another library, `--matching` states
the rule that makes the point fair (for attention, capacity at N = L; `docs/measurement.md`), because that is a claim.

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

`benchmarks/bench/provider.py` builds rola's arms inside a worker run by the arm's checkout's own venv, from its own
directory, so two rola arms differ in exactly what their names say. Building an arm refuses it by name unless the
device reports the subject's family stamp (a path or a hash cannot catch a stale binary) and `import rola` resolved
inside that checkout. The arm's cell is the registry cell's facts, the dials, and the binary's: the manifest digest, the
family stamp, the device, torch, the assembler, and the SM clock read when the arm was built. Each sample is one launch
between two CUDA events.

A **foreign arm** names a provider (`module:function`), the arm, the python that runs it and the directory it runs in:
rola-bench's attention reference is one (`rola_bench.measure.attention:arms`, arm `flash`).

## The run

The tool holds the GPU lock exclusively for every timed call and engages the host's clock lock, proven by the device's
clock read before the first call and after the last; a second read off the lock refuses the comparison. The driver
warms each arm past its floor of 10 launches and then calls every arm once per rep in a fresh random order, `--reps`
(odd) per round for `--rounds` rounds. It prints each arm's median over the round medians, their interquartile range,
and the median of its per-rep ratios to the reference (the first arm unless `--reference`). `--out` writes the whole
result: the point, the matching rule, every arm's cell and raw samples in the order taken, and the clock. `--record`
stores it through `rola_results` at `compare`, keyed by the point, the matching rule and each arm's commit and diff
digest.

```bash
python tools/compare.py --cell flat-small-alt-k16 \
    --arm label:first,arm:carry_forward --arm label:identity,arm:carry_forward@schedule=identity --rounds 8
```
