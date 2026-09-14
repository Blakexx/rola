# `tools/compose_ledger.py` — a phase's time attributed to its parts by composition

KERNEL_STANDARDS §22 (9). The carry kernel's MMA phases are built from parts that a build may stub
(`docs/internals/carry/carry_kernel.md#parts`). A LADDER over a phase builds the kernel with none of
that phase's parts real and every other part real, then adds the phase's parts back in the order
named, one rung a build; the last rung is the kernel itself.

    python tools/compose_ledger.py --cells flagship-dense,nl64k-alt-k4 \
        --ladder readout:stream,loads,drain --ladder fold:pool,ring,loads

Each rung is read in wall time by `tools/phase_ledger.py` (one warm-up launch, then `--launches`):
a part's cost is the step between the rung without it and the rung with it. No constant enters and
no stall sample is interpreted. A rung is VALID only when two checks pass. The profiler's HMMA
count for each cell must equal the kernel's, since a stub that changed the workload measures a
different kernel. The build's device instructions must hash differently from the kernel's, since a
build that did not take the mask measures nothing. The kernel is rebuilt with every part real at
the end, and its instructions must hash to the value taken before the first rung. The kernel is then
read again: its before-and-after difference is the run's drift bound, and a rung step smaller than
that bound is not a reading. The first run needed it: another process had the GPU at 36-38%, and the
sparse cell read 45% slow with rungs that contradicted each other. Nothing in a single run's numbers
flagged that except the kernel disagreeing with the day's readings.

Why it exists: on 2026-09-12 five kernel changes were built on inferences from warp-side
instruments (stall samples, the phase clock, a utilization computed from an HMMA count and a
32-cycle constant), and their misses could not separate a wrong fix from a wrong premise. An
MMA-only variant then measured the phases' floor directly (readout 38.2K, fold 35.7K cycles a warp
a window at flagship-dense against 52.4K and 47.0K). The composer makes that measurement a
mechanical, per-part ladder.

Every rung records, beside the phases: every phase's per-warp row (imbalance per composition); the
launch's resource counters a CTA-window (shared loads and stores, `ldmatrix` loads, global reductions
and their translation requests, shared wavefronts and bank conflicts for loads and stores apart,
issue-slot occupancy, warps active); the STALL CENSUS (every warp-stall sample of a launch by reason,
and by component and reason, as cycles a warp a window: the sample share times the phase total); and
the arm's peak live registers from its life ranges. The report's PART STEPS table gives, for each
added part, its step in the total, in its top stall reasons, in the counters, and in peak live: how
much of the part fails to overlap, why it stalls, what it spends, and what registers it takes.
Measured on the kernel at flagship-dense, the census reads the 110K-cycle window as 36.8K pipe-throttled
(the HMMA floor is 36.9K), 31.5K fixed-latency dependency, 16.3K shared scoreboard, 13.0K issuing.

With `--timeline`, every rung also records its pipe timeline (`docs/internals/tools/pipe_timeline.md`): the
tensor pipe's true utilization on silicon over full-occupancy samples, and its tensor series in 5 µs bins. A part
then shows in the report both as a wall-time step and as the change in how the pipe was kept fed.

The report lands in `<store>/ledgers/<utc>-<sha7>-compose.{json,md}` (the measurements store), and the dashboard is rendered
after it (`tools/dashboard.py`, [dashboard.md](dashboard.md)). Builds run
serially (the host's memory watchdog kills parallel ones). Every build is an iteration build and
never ships.
