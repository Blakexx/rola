# `tools/dashboard.py` — the kernel and its variants over the cells, on one page

The measurement suite stores every number it takes. The dashboard reads those stored numbers and renders
them as one static HTML page. It measures nothing. The same inputs always give the same bytes: no clock,
no randomness, inputs read in sorted order, data serialized with sorted keys.

    python tools/dashboard.py            # writes the scratch directory's dashboard/dashboard.html
    python tools/dashboard.py --check    # renders twice; refuses unless the bytes agree and no absolute path is inlined

`tools/build_ledger.py` and `tools/compose_ledger.py` render it after writing their reports, so the page is
current after every ledger or composer run. The page is `tools/dashboard_page.html` with the data inlined as
JSON. Its header carries the hash of that data.

**Inputs.** Every input is read through `rola_results` (`docs/measurement.md`), each successful sample of a location:

| location | read for |
|---|---|
| `build_ledger` | standings, builds |
| `compose_ledger` | part ladders |
| `pipe_timeline` | tensor and ALU pipes over a launch |
| `pipe_timeline.scale` | the note under the timelines |
| `probe_cells` | variants against a baseline |

A capture made outside a ledger run joins the page when `tools/pipe_timeline.py --rep <report>` imports it.

**Sections.**
- *Standings*: each gate cell's newest stored value of each build ledger measurement. That is the time
  against the baseline, true utilization from the timeline, HMMA-count utilization, and cycles a warp a
  CTA-window in total and per phase. Each value names the commit of the report it came from, so a stale
  value shows as stale.
- *Variants over the cells*: one row for each probed binary and stage, one column for each cell. A value is
  the binary's median launch time divided by the baseline's median in the same interleaved session. When a
  binary ran in several sessions of one stage, the newest session per cell is shown.
- *Pipe timelines*: every stored capture of a cell, newest first, on the same true-utilization scale.
- *Composer*: a composer report's ladders. Each rung's phase cycles, the step its part adds, the total, the
  rung's true utilization when captured with `--timeline`, and the rung's checks.
- *Builds*: every build ledger report, with total cycles a warp a window per gate cell plotted across builds.

**Reading the variants.**
- *Session.* A probe session is one `tools/probe_cells.py` invocation. Its binaries run interleaved under
  the clock lock, so only ratios within a session are comparable. A stored session is one sample of a
  `probe_cells` record, every binary's rows together.
- *Baseline.* The baseline is the session's run on the first branch in `--baselines` that is present
  (default `master`, then `k31/carry-clean`). A session with several runs on `master` takes, per cell, the
  run that followed master's order policy (`box` dense, `sparse-gN` sparse). A session with no such run
  shows milliseconds only.
- *Flags.* A master baseline that did not run its order policy is flagged, and the value is dashed. Its
  time is not master's, so the ratio against it is not a ratio against master. The known cases are master
  running its dense order at sparse cells and a row recording `first`, a schedule master does not have
  (the probe's schedule rule, `tools/probe_cells.py`). An unlocked clock is flagged too. **Hide flagged**
  removes flagged values.

**Limits.**
- Everything the page reads is in the measurements store (`store.root`), shared by every worktree and the container;
  the page itself is written to the scratch directory.
- Standings read build ledger reports only. A probe run outside the ledger does not change the standings,
  because the probe record does not say which binary is the tree's kernel.
