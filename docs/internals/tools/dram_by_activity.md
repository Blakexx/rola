# `tools/experiments/dram_by_activity.py` — DRAM by activity byte

The committed form of the ad hoc "plane-count instrument" whose numbers journal `## G1`
reported (read-only atoms at `0.657x` DRAM on `k16_w384`; written atoms at `+9.6%` on
`k4_w512_record`). That instrument had no committed invocation, so its written column was
not reproducible — `## G1b-pad` spent a whole stage on a fix for a cause it could not
re-measure. This file exists so the experiment has one definition.

## What it holds fixed, and what it varies

Everything except one byte per atom is the shipped measured path: the frozen cell draw,
the arm resolution and its refusal, the packing, the full-coverage page table and the
launch all come from `tools/probe_cells.py` by import, not by copy.

What varies is the ACTIVITY byte the facts pass publishes per atom
(`rola.engine.facts.planes.atom_bits`, P80b — bit 0 `WRITTEN`, bit 1 `READ`), which the
kernel's state sweeps read as *load iff resident and (read|written), store iff written*
(`docs/internals/state.md`):

| variant | the byte handed to the kernel | what the sweep does |
|---|---|---|
| `asis` | what the facts pass emits for this cell | the cell's own state I/O |
| `read` | `ATOM_READ` on every touched atom | loads only; on a split-plane page, the hi plane only |
| `written` | `ATOM_READ \| ATOM_WRITTEN` on every touched atom | both planes, both ways |
| `none` | — (`s_out`/`page_tbl`/bits all null) | no state I/O at all: the NULL-STATE control |

The override changes an atom's ACTIVITY and never the RESIDENT set: an atom the routing
does not touch stays untouched in every variant.

`--continuation` is the second axis, and it is not optional for a read-side question.
By default the call is made the way `probe_cells.make_call` makes it — with a NULL
`s_in` — so there is no entry sweep at all and a "read-only" atom moves no bytes
whatsoever: the split planes' read-side saving is unobservable by construction, and so
is any read-side regression. `--continuation` passes the exit plane as the entry plane
(the in-place law: the op seam refuses two distinct planes under an activity bitmap,
`docs/internals/state.md#activity`), which is what a real continuation does and the only
shape in which the entry sweep runs. Measured at `k16_w384`: the entry sweep adds
252 K global-load sectors on the split binary, the hi plane's 4096 x 2080 B.

Two controls are built in. `none` is the noise floor — the two binaries run identical
work there, so their difference is this host's DRAM run-to-run drift (`## G1b-pad`
measured ~0.6%). And on every frozen cell the read set and the write set are both *every*
atom, so `asis` and `written` request the same work: any spread between those
two columns is measurement, not mechanism.

## How it measures

One `ncu` process per (binary, variant, cell) profiles `--launches` steady-state launches
after `--warmup` skipped ones (`--launch-skip`), under `gpu_lock()` — never an external
`flock`, which self-deadlocks against the tool's own acquisition. The reported
figure per metric is the MEDIAN across profiled launches, with min/max kept beside it.
Binaries are interleaved per variant (A, B, A, B — the K36 lesson), never all of A then
all of B.

Counters are asked in READ and WRITE halves at every level, and in REQUESTS as well as
SECTORS: a DRAM byte total alone cannot separate "more bytes asked for" from "the same
bytes in worse sectors", and sectors-per-request is the coalescing/alignment question
`## G1b-pad`'s fixed metric list could not ask.

`--source` adds one SourceCounters capture per (binary, cell, variant), aggregated by
source line — how the three blocks of a split page (hi values, lo values, the two 16-bit
mass runs; `docs/internals/common/state_page.md`) are told apart, since they are three
distinct loops in `csrc/rola/src/common/state_page.cuh`.

Rows are stored through `rola_results` at `dram_by_activity`, one sample a run, its stage in
the provenance.

## Invocation

```bash
python tools/experiments/dram_by_activity.py \
  --binary worktree:/path/to/split,venv:/path/to/venv,label:split \
  --binary worktree:/path/to/parent,venv:/path/to/venv,label:parent \
  --cells k4_w512_record,k16_w384 --variants asis,read,written,none \
  --launches 6 --warmup 3 --stage G1c-dram
```

A cross-tip run invokes THIS file (one source) with each worktree's own venv python and
that worktree as cwd, exactly as `probe_cells.py` does: the only thing that differs
between the two columns is which compiled extension answers `rola.ops.carry`.
