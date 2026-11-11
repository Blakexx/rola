# KERNEL_STANDARDS — the laws

**THE CONSTITUTION** is `docs/CONSTITUTION.md`: Blake's standing rulings, changed only by Blake, above every
law here.

**THE LAWS** are this document: the coordinator's procedural and incident-derived rules. Dated, adaptive, revisable
by the coordinator; each carries the incident that made it and is repealed when the situation changes. A law never
contradicts the constitution; where a law is an inference from one moment of development it is marked as such and
re-examined at the next milestone.

**THE CARDS.** This line's kernels are written for two families, `hopper` (sm_90a) and `blackwell` (sm_100). The
`ampere` family (sm_80 to sm_89) is the branch `ampere`, frozen: its schedules, its atoms and the labs that measured
them are that branch's, and its cards stay rows of the hardware table (`csrc/rola/src/common/hardware.json`),
tabulated and not built. A kernel with no family schedule (the facts pass, decode, paging, entmax) compiles for sm_86
as it is written for every card, so the development card runs it; nothing is written for sm_86 alone
(`docs/CONSTITUTION.md#r8`).

**THREE KINDS OF RULE.** Every rule here is one of three kinds, and its kind says what a new card does to it.
- PROCESS: how a kernel is designed, budgeted, gated, measured and committed. It holds on every card, and it is the
  kind of every rule that carries no mark.
- `[COMPILER]`: what ptxas does to a source pattern. Each was measured on sm_86 under CUDA 13.0, by a lab of the
  branch `ampere` where it names one this tree does not hold; it stands here as the default shape of the code, and its
  re-check on sm_90a is owed (§26.1) -- on the instruction this line issues, where it was measured around `mma.sync`.
- `[MACHINE]`: what a card's pipes cost, and the form that cost forces. A card's rates are its rows of the hardware
  table, each citing its lab record: a number quoted beside a rule here is the measurement the rule was drawn from,
  and the row is the rate. A rule drawn from them holds for the family it was measured on. The ampere family's are the branch's. The few that stand here were measured on sm_86 on work every card
  does the same way (a stream of rows from DRAM, a reduction's L2 requests) and are owed their sm_90a run; hopper's
  are written here as its labs run.

**OWED TO THE FIRST HOPPER SCHEDULE.** No op runs on this line until the carry, the intra and the producer have hopper
schedules. These land with the first of them, each red or an owed entry until then (§25.3, §32.5):
1. the SASS gate's hopper machine model and its step cover (§32.4), and each kernel's budget file (§19, §29);
2. the family's part harness, phase ledger, composer and calibrations (§19, §22);
3. hopper's rows of the hardware table and of the price list, from its labs (§26, §29.11, §29.14);
4. every `[COMPILER]` rule's re-check and every `[MACHINE]` rule's run on sm_90a;
5. the ratified sm_90a manifest (§16).

# ROLA KERNEL STANDARDS — how we write CUDA here (draft for Blake's read, 2026-08-26)

SCOPE. This document is about CRAFT: how a designed piece is realized at the hardware's grain. It is not
about design. Design authority is Blake's rulings and `docs/ARCHITECTURE.md`; nothing cited here is an
argument for or against a design choice. References are the local copies under a named sibling checkout,
`rola_references/` (`flash-attention`, `cutlass`, `ThunderKittens`); citations are `path:line`.

Every brief points here. Every review applies it line by line. A check in `ratify` (§9) is the backstop
and should catch nothing.

## 1. Register-resident data exists only as FRAGMENTS
The tensor core's unit in registers is the fragment: one MMA tile spread across the warp's 32 lanes in a
fixed thread→value layout. Anything held in registers across a loop is a stack of fragments, and code
addresses it by fragment modes — never by element, never by leaf.
- FA2 allocates the accumulator as a fragment shaped by the tiled MMA and keeps it across the whole K
  loop: `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:194` (`partition_fragment_C(tiled_mma, …)  // MMA, MMA_M, MMA_K`).
- Its only inner loop is over a fragment mode: `flash-attention/csrc/flash_attn/src/utils.h:152-158` (`for (i < size<2>(tCrA)) … cute::gemm(…)`); softmax touches the accumulator through fragment modes only: `softmax.h:28-31, 100-113`.
- CuTe states the principle: an MMA atom IS three thread-value layouts — `cutlass/media/docs/cpp/cute/0t_mma_atom.md:168-172`; per-thread views fall out of the layout (`include/cute/atom/mma_atom.hpp:467-503`).
- ThunderKittens makes it a type: a register tile is a 2-D array of MMA-shaped base tiles
  (`ThunderKittens/include/types/register/rt.cuh:104`, `rt_base.cuh:3`); user code never writes element loops (`README.md:211, 217`).
RULE: a `static_for`/`#pragma unroll` nest over register-resident state iterates fragments (m-tiles × n-tiles ×
k-steps) with any finer index (leaf → slot, channel → row) folded into constexpr address math. The nest's
BODY COUNT must be stated as a function of the arm in the code's decl block and must be O(fragments).
COUNTER-EXAMPLE: `page_regs` unrolled per LEAF × atom × channel-group → 4,096 bodies at the
D=4 8-warp arm; compile 5 → 50 min/arch; cicc at 6 GB segfaulting. Per-leaf iteration below the fragment
grain is a defect, not a style choice.

## 2. Reuse an accumulator as the next GEMM's operand by RELAYOUT, not by data movement
The fp32 accumulator of one GEMM becomes the A operand of the next by reinterpreting its layout in place
and narrowing the type — no store, no shuffle.
- FA2: `flash_fwd_kernel.h:378-380` (`make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()))` → `gemm_rs`); the layout math `utils.h:199-211`.
- FA3: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1156-1160` (same, plus an explicit register permute for fp8, and a documented fallback through SMEM when the register-sourced GMMA is unavailable, `:1161-1162`).
RoLA: the fold's fp32 accumulators become the readout's bf16 operand by this relayout, narrowed in place
(`docs/ARCHITECTURE.md`, section 9) -- unbuilt, its instruction count a lab's to measure before a hopper carry is
designed; a phase change that moves state is the transfer the design names, never an ad-hoc copy.

## 3. Shared memory: copy atoms, swizzles by composition, transposition as a layout operation
- `ldmatrix` in both variants is a declared copy atom with an arch fallback chosen at the atom, not inline
  PTX at call sites: `flash-attention/csrc/flash_attn/src/kernel_traits.h:40-44` (`SM75_U32x4_LDSM_N`, `SM75_U16x8_LDSM_T`).
- The transposed atom feeds V as the B operand with no transpose pass: `flash_fwd_kernel.h:210`; copy and
  MMA thread mappings are derived from the SAME tiled MMA (`make_tiled_copy_B(…, tiled_mma)`) so they cannot drift.
- Bank-conflict-free layouts are a swizzle composed with a small atom and tiled to shape:
  `kernel_traits.h:79-90` (`composition(Swizzle<kSwizzle,3,3>{}, Layout<Shape<_8, kBlockKSmem>>)`), the swizzle
  parameter derived from the row width. A transposed view is a COMPOSITION of the same layout
  (`kernel_traits.h:92-95`) — zero data movement.
RULE: every SMEM tile has a named layout with its swizzle derived from its access pattern (writer's and
reader's), and a ROUND-TRIP test proving bit-exactness and conflict-freedom (the existing plane gate).
Changing an access pattern (e.g. 2-byte scattered stores) re-derives the swizzle and re-runs the gate.
Primitives (a matrix load, an asynchronous copy, an MMA, a reduction) are wrapped ONCE, in their concern's atom
(`csrc/rola/src/common/atoms/`, all included by `common/ops.cuh`); call sites use the wrapper.

## 4. Pipelines: explicit stage state, release immediately after consumption, two stages in flight
- FA3's producer acquires a stage before issuing, the consumer waits, issues the GEMM, waits for it, and
  releases the stage at once: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:750, 1140-1143`; pipeline types per
  operand `:283-286`; the invariant is documented where it bites (`:787`).
RULE: overlap requires two stages in flight in ANY form: the second stage is a second SMEM slot (a ring's next
stage), or registers where the operand is register-resident. Sync carries exactly two facts, "landed" and "drained",
by mbarrier: a stage's full and empty barriers (`docs/CONSTITUTION.md#r21`). No pipeline state is implicit in loop
parity without a comment naming the invariant.

## 5. Warp specialization only as register economy
- FA3 gives the producer warpgroup ~24-56 registers and the consumers 160-256 via `setmaxnreg`
  (`hopper/flash_fwd_kernel_sm90.h:82-83, 308-309, 360-361`), with the budgets `constexpr` per configuration.
RULE: a warp is specialized only where its registers move with its role -- `setmaxnreg` hands a producer's to the
consumers -- and only where the accumulator binds: a producer that holds a consumer's slice and folds nothing is
waste. `[MACHINE]` `setmaxnreg` around a live state is unmeasured here (a hopper lab's).
THE IDEAL REGIME IS FEW WARPS WITH FILLED REGISTERS (Blake, restated 2026-08-08: "If that doesn't work it means
something is wrong with the kernel in another way we need to fix. not intentionally avoid the ideal regime"). Warp
count and the register budget are one knob, swept jointly as a surface and never marginally; occupancy is an output,
not a constraint. A loss in the ideal regime at a correctly sized budget is a defect to locate (a spill, a
serialization, a missed overlap), never a reason to leave the regime: a "fewer warps is worse" reading taken with the
budget sized for more warps measured spill, not the regime.

## 6. Unroll discipline
- CuTe pairs `CUTE_UNROLL` with `CUTE_NO_UNROLL` (`cutlass/include/cute/config.hpp:51-55`): saying "do not
  unroll" is as deliberate as saying "unroll". Unrolls are over fragment modes with compile-time bounds
  (`include/cute/algorithm/gemm.hpp:293-320`); FA2 unrolls a loop only when its trip count is `constexpr` (`flash_fwd_kernel.h:308`).
RULE: unroll = compile-time-bounded loops over fragment modes, nothing else. Dynamic loops (windows,
batches, tokens) carry `#pragma unroll 1` explicitly. `static_for` is a fragment-mode tool; a `static_for`
whose bound is a leaf count, a channel count, or a product of them is wrong.

## 7. Instantiation hygiene: one instantiation per translation unit, generated, prunable, budgeted
- FA2: each `.cu` holds ONE explicit specialization and says why: `flash-attention/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu:2-3`
  ("Splitting the different head dimensions to different files to speed up compilation. This file is
  auto-generated. See generate_kernels.py"); the generator `generate_kernels.py:12-50`; shared sub-trees are
  `extern template` so they are not re-instantiated (`:44-48`).
- FA3: 451 generated TUs under `hopper/instantiations/`; feature axes deliberately COLLAPSED to cut the
  cross-product (`hopper/generate_kernels.py:99, 108`); every TU guarded by `#ifndef FLASHATTENTION_DISABLE_*`
  so a subset builds.
- FA's build sizes parallelism to memory because instantiation cost is the dominant resource:
  `setup.py:708-715` ("peak_threads * 5GB <= free_memory").
- CUTLASS emits one `.cu` per configuration with an aggregator that only declares initializers
  (`cutlass/python/cutlass_library/manifest.py:290-330`) and prunes the instantiated set for compile cost
  (`media/docs/cpp/profiler.md:14`; `quickstart.md:609`).
RULE: one arm per TU, generated from the closed-world arm list; an ARM-SUBSET build knob (declared list,
default all) for iteration; iteration builds are single-arch AND arm-subset; every-arch all-arm builds
once per stage. Each arm has a COMPILE BUDGET (wall time, cicc peak RSS) recorded in the manifest.

## 8. Arch mapping at the atom, closed-world at the arm
Feature selection lives where the primitive is defined (`kernel_traits.h:40-44`'s `DefaultCopy` fallback),
never as `#if` at call sites. RoLA adds: every structural decision states its hopper and blackwell mapping on the
card (§4 of the architecture); only measured binaries run (pinned ptxas, per-arch ratified manifests, device stamp).

TWO LEVELS, ONE HOME EACH (2026-11-07, the family foundation; constitution R8). An INSTRUCTION difference is an atom:
one name, its body selected per family where it is defined, in `common/atoms/`. A STRUCTURE difference is a schedule in
`<kernel>/<family>/`, which holds only what differs in structure -- anything differing by an instruction alone moves to
an atom. A kernel's semantics live in `<kernel>/`, shared by its schedules. THE SELECTION is one layer every kernel
plugs into: the cards are data (hardware.json's arch and product rows, generated), a kernel's sizes come from its plan
alone (`Plan<build, arm>`: the card's build traits and the declaration's arm), and one host call selects the card and
the arm, refusing by name an untabulated architecture, a build the binary does not carry and an unbuilt arm; a device
with no product row runs its architecture's build. Products equal in their build traits share their kernels. THE
CAPABILITY ROWS hold only what is read -- the numeric facts plans and the admission read, the instruction flags atoms
read; a fact nothing reads is not tabulated, and a family is declared on its row, never derived from flags. The design
in one page: [`architecture/families.md`](architecture/families.md).

## 9. The backstop (checks that should catch nothing)
`ratify` records per arm and arch: registers, stack, spills, residency (existing) + compile wall time,
cicc peak RSS, and the body count of each named `static_for` nest as a function of the arm (new). A budget
breach fails the gate exactly as a spill does. The device stamp is asserted before any gate.

## 10. Correctness instruments (unchanged, restated so they sit beside the craft rules)
The fp64 oracle is the truth. Bit-identity against a previous binary is a REFACTOR instrument, never a
design constraint (amended 2026-08-26). No fallbacks, no dead code, no special-case paths; comments
are file-header/decl blocks only, prose goes to `docs/internals/` mirroring `csrc/`. Build the whole design
or stop and name the gap; efficiency is correctness.

## 11. Debugging a device hang or race (recorded 2026-08-26 after a stage burned 2.5 h reasoning blind)
`cuda-gdb` works on this machine when the process is LAUNCHED under it; ATTACHING (`-p`) is refused by
Yama `ptrace_scope=1` (a Linux setting, liftable with `sudo sysctl kernel.yama.ptrace_scope=0`).
Recipe: `<venv>/bin/python -m rola_devtools.locks.gpu -- cuda-gdb -q --args <venv>/bin/python repro.py` (the GPU
lock's command form: the lock is the files beside `host.gpu_lock`, so a raw `flock` on its path serializes against
nothing) → `run` → Ctrl-C on a
hang → `info cuda kernels` / `info cuda warps` / `cuda kernel 0 block (..) thread (..)` / `bt` / `print`.
`compute-sanitizer --tool synccheck|racecheck|memcheck` for barrier misuse, shared-memory hazards and
addressing (it found the decode pool race). Device `printf` flushes only on kernel completion — useless
for a hang; a bounded loop + `__trap()` converts a hang into a fault but skips barriers, so prefer the
debugger. RULE: a hang or a run-to-run-varying miscount is diagnosed with the debugger or the sanitizer
within the first hour, never by inspection alone; the finding goes in FINDINGS with the debugger's output.

## 12. Warp collectives and fences (recorded 2026-08-26)
- NEVER place a warp-collective (`__shfl_sync`, `__ballot_sync`, `ldmatrix`, `__syncwarp`) inside a
  short-circuit or a lane-divergent predicate: `part && lane_broadcast(x)` splits the warp at an
  instruction that needs every lane, and the split never completes (a hang that survives removing every
  explicit wait). Hoist the collective; predicate its RESULT.
- A fence orders only the FENCING THREAD's own accesses. In any publish/consume pair (data -> counter,
  counter -> data), EVERY lane that issued the data accesses executes the fence before the `__syncwarp`
  that precedes the one-lane counter update; and EVERY lane that reads the data after a one-lane
  counter check executes a fence after the broadcast and before its loads. A leader-only fence covers
  the leader only (the ring hang: seven of eight lanes unordered).
- Debugging recipe under WSL (cuda-gdb launch mode, attach is refused by Yama): run gdb with a FIFO on
  stdin (`mkfifo in; cuda-gdb -q --args <venv>/bin/python repro.py < in &`), `echo run > in`, wait for
  the hang, `kill -INT <gdb pid>`, then `echo 'info cuda kernels' > in`, `info cuda threads`, `bt` —
  it names the split warp and the PC in one shot.

## 13. A mechanism's brief carries the WHOLE protocol (recorded 2026-08-26 after the ring's four patches)
When a synchronization protocol or data-path mechanism changes, the brief restates the ENTIRE mechanism
as pseudocode and the implementer rebuilds that function from it — deleting the previous one — never
editing around the previous protocol's skeleton. Scaffolding that only made sense for a superseded
protocol (a deferral rule, a sentinel state, a counter) must not survive; the review checks the function
against the pseudocode line by line. "Build the whole design" applies at mechanism grain.
Addendum: a whole rebuild is diffed against the BUG LIST of the thing it replaces, not only against
the pseudocode — a rebuild reintroduced the `part && lane_broadcast(...)` short-circuit hang while
rewriting the acquire, three lines below its own warning. The previous pass's findings are a checklist
the rebuild is walked against before its first build.

## <a id="s14"></a>14. Waiting on a build (recorded 2026-08-26 from the agent time ledger: 327 min of `until pgrep … sleep` loops)
A long build is launched as a BACKGROUND task writing to a log file, and the agent then ENDS ITS TURN —
the harness re-invokes it when the command exits, with the exit status. Never wait by polling: no
`until pgrep … sleep` loops, no `pgrep -f` on a process name (it self-matches, it matches other tracks'
builds in the same venv, and a live waiter re-wakes a stopped agent into a worktree it no longer owns).
Never edit a source file while a build of it is in flight. One gate build per stage; iteration builds
single-arch and arm-subset.
§14 addendum (Blake: background tasks have died silently before): the launch records the build's PID and
appends an exit marker (`echo BUILD_EXIT=$? >> log`) as the command's last step; on re-invocation for ANY
reason the agent reads the marker first, never trusts the notification alone. The wait itself is a
BOUNDED TIMER with early exit — `timeout 600 tail --pid=$PID -f /dev/null` (≤10 min, returns at once when
the build ends) — after which the agent INSPECTS ITS OWN STATE (the marker, the build dir) and decides;
repeat if still building. Never trust the pid or the notification alone; the marker is the fact. Name-matched polling (`pgrep -f`) stays forbidden: it self-matches, matches other tracks, and
re-wakes stopped agents.

## 15. Stage cycle budget (recorded 2026-08-27 from the ledger: one stage ran 63 builds and 22 ratifies in 346 min)
- RATIFY ONCE per stage, after the last source edit. It recompiles every TU on every architecture; it is a gate,
  never an iteration tool.
- A stage has a budget of THREE full build+gate cycles. Iteration is single-arch, arm-subset, oracle-first.
  When the budget is spent and a gate is still red, the stage STOPS and reports the residual — it does not
  keep chasing green.
- MID-STAGE RE-SCOPING IS ALLOWED when it changes WHAT is built (a design or mechanism ruling) — Blake:
  "I'd rather re-scope than build something wrong and fix it later." A re-scoped stage restarts its
  cycle budget from the amended spec. Amendments that do not change the design (count formats, report
  shape, housekeeping) go in the NEXT brief instead.
- Every kernel family carries an arm-subset knob (the carry's `--//:carry_arms`); a family without one gets it
  before its next build.
§15 addendum (build parallelism, 2026-08-27; re-measured since): a cicc peaks at ~2.9 GB in a unit that parses torch's
extension header and ~0.33 GB in a torch-free arm unit (docs/build.md#compile-ledger), on a 16-CPU / 24 GB machine.
A build's jobs are the host-budget slots the front door holds, divided by `host.nvcc_threads` (2: one cicc an
architecture), every build under the same machine-wide pool (§23); an iteration build is single-arch and arm-subset.
Re-derive the budget if cicc's peak changes (a per-leaf nest can push it back to 6 GB — §1).

## <a id="s16"></a>§16 — Ratify at milestones, not stages (Blake, 2026-08-28)

A stage's gate is SMALL: an iteration build (single arch, arm subset), the fp64 oracle on the arms
it touched, the census on those arms, and the model tests. A stage does NOT run the every-arch
all-arm gate build and does NOT ratify. Manifests go stale between milestones by design; a stage
commit says "unratified" in its title. RATIFICATION — the full closed-world measurement — runs
ONCE per cohesive product: at the milestone that merges (the receipt stage), and after the last
edit of any kind. Rationale: a 40-minute ratify (plus a 20-35 minute gate build) after every
small piece was 1-2 hours of waiting per stage; the closed-world guarantee is a property of what
SHIPS, and nothing ships between milestones. The stage-level guarantee is the oracle.
Corollary: a stage that needs the SASS-body/instantiation gates for its OWN claim (a byte-identity
refactor) runs `ratify --check` against the LAST milestone's manifest on its subset only.
§16 addendum (Blake, 2026-08-28) — EVERY STAGE GATE INCLUDES A PERF PROBE. Since the stage compiles
anyway, its gate runs the fixed two-cell probe (dense flagship k_tok=None + k16 scattered; L=768;
the stage's touched arm; ncu carry `gpu__time_duration` 3 warm + the step wall) and records it in
FINDINGS as a running per-stage ledger next to the previous stage's numbers and the mainline
reference. A >1.10x regression on either cell vs the previous stage is a RED like a spill: the stage
reports it, does not rationalize it, and Blake rules. This is the instrument that would have caught
the 3-9x loss PROBE-K35 found after nine stages.
§16 addendum 2 (2026-08-28): AN INSTRUCTION LEDGER IS NECESSARY, NOT SUFFICIENT. One measured span
cut instructions 5.6% and cost 22% of time (IPC -27%, long-scoreboard x3.6, bank
conflicts x2.6). The stage probe is TIME; every ledger table carries IPC and the top stall reasons
(ncu `smsp__warp_issue_stalled_*`) beside instruction counts, and a stage whose instructions fall while
its time rises is RED, not a win.
§16 addendum 3 (Blake, 2026-08-28) — THE COMPILE LEDGER. Build time is a gated number like the census.
Every stage gate records, per built arm TU, the compile wall and cicc peak RSS beside the previous
stage's; >1.2x on any arm is RED (report, do not rationalize; diagnose against §7 first). Arm TUs never
include torch headers (the ~34 s header floor lives in the dispatch TU only). Arms are added as reviewed
rows in the one arm list (TEST-tagged unless shipped). Ratify stays incremental per arm.
§16 addendum 4 (Blake, 2026-11-09) — A GATE THAT READS PROFILER COUNTERS RUNS ON A HOST THAT GRANTS THEM. The probe's
and the ledgers' counters (ncu) are the device's host's to grant, and a rented card's may not: such a gate runs on a
host that does, or its row carries a written exception naming the host and the reading that stands in (the kernel's
own clocks, the SASS's counts). Whether a rented H100's host grants them is unmeasured.

## §R addenda — LAWS (coordinator, 2026-08-28; incident-derived, revisable)
R2, R3. `[MACHINE]` The ampere family's fan-in rules (a global reduction's segments an instruction; its native
    global RED from an aligned layout), the branch's. A reduction's requests on this line: §26.3 q.
R4. NO PER-WARP WORK THAT ANOTHER WARP ALSO DOES: CTA-identical derivations (union lists, owner
    ballots) are derived once and published behind a barrier the kernel already pays, or split by
    executor share — never recomputed per warp to avoid a barrier.
R5. `[MACHINE]` The ampere carry's three-part schedule, the branch's.
R6. `[COMPILER]` A KERNEL'S SASS DEPENDS ON ITS TU NEIGHBORS: byte-identity is a claim about a partition, not a body;
    per-arm TUs make each body canonical (7 movers, proven inert and free).
R7. STAGED-vs-STRAIGHT IS DECIDED BY LATENCY, NOT COUNTS: "identical sectors and issues" lost 9.5% on
    long-scoreboard (e390764). Time, IPC and stall reasons decide; counts describe.

## <a id="s17"></a>§17 — Portability: what the repo points at (Blake, 2026-08-28)
The repository is SELF-CONTAINED: anything it DEPENDS ON — a tool it runs, a doc it cites, a record it
reads, a baseline it compares against — lives in the repo or in a NAMED sibling repository with a stable
layout (the bench/results repo), never in an ad-hoc place (a scratch directory, a home path, a session
scratchpad, a campaign journal, a worktree path). Ad-hoc material never enters the repo either — no
script blowup: briefs, transcripts, per-session scripts and campaign journals stay outside and are
cited FROM the queue, not from repo files. Absolute paths under a home directory are a lint finding.
Committed records carry repo-relative or host-keyed locations. Skills that ship with the repo point at
repo docs and the measurements record, never at a journal.

## §14 addendum (law, 2026-08-28): CONTAINERS AND THE GPU LOCK
A lock file is per-filesystem: a lock taken inside a container serializes nothing on the host or in another
container unless both see the same files. Every container that touches the GPU mounts the host's lock directory at
the one lock path (`/tmp/rola-locks`, every machine's default), so one set of locks governs all. One gate container at a time: stop the previous run before
launching the next (three concurrent in-container probes turned a minutes-long gate into an hour).

## §18 — Tools refuse, they do not choose (law, 2026-08-28, from Blake's "why keep two linkers")
A tool or build step that could run one of two ways — a linker present or not, a cache pinned or not,
a lint with or without its compdb, a lock taken or a slot — REFUSES with the command that fixes it; it
never silently takes the other path. Legitimate MODES (the §16 strict-manifest tiering; a comparison
arm kept as a yardstick) are keyed on an explicit declared variable and carry their exit condition on
a card. Applied: mold as the only linker; a unit lint with no verdict fails; one build-lock discipline (slots as the
parameter).

## <a id="comment-density"></a>§10 addendum (law, 2026-08-28): THE COMMENT-DENSITY GATE
The constitution's self-documenting-code clause is gated mechanically: `tools/lint/lint_standards.py`
measures comment lines / total lines per file under csrc/ and is RED above the mandate's threshold
(named constant; decl-block level, well under 15%). A `//:` prose block outside a file header is a
finding. A stage never raises a file's ratio; existing prose moves to the mirrored docs
(byte-identical SASS gate). Measured 2026-08-28 before that move: carry.cu 48%, ops.cuh 43%, box.cuh 41%,
carry_kernel.cuh 35%, decode.cu 32%, facts_kernel.cuh 32%.

## §14 addendum 2 (law, 2026-08-28, after Blake had to shut WSL down): ONE MACHINE, ONE COMPILE BUDGET
Three tooling agents each ran a full rebuild concurrently (MAX_JOBS=4 each -> 12+ cicc + 3 links) and
pegged the host at 100% CPU for 10+ minutes. The compile budget is MACHINE-WIDE and TOOL-ENFORCED, not
brief-enforced: the graph's front door takes machine-wide slots for every build (`rola_devtools.graph.front_door` over
`rola_devtools.locks.host`, §23) and waits when they are full — a brief cannot forget it. An agent checks
`pgrep -c cicc` before launching and never wraps a full build in hyperfine (bounded `--runs 3`, on the
LINK step only). Coordinator error recorded: two briefs omitted the build lock.

## <a id="no-verify"></a>§18 addendum (law, 2026-08-29): `--no-verify` is never a precedent
A commit bypasses the hooks only with an explicit, per-commit coordinator authorization naming the
pre-existing hook failure and the stage that fixes it. "Another stage did it" is not authorization. Red
hooks are fixed (the hook, or the tree), never routed around.

## <a id="sanitizer"></a>THE SANITIZER LANE (law; its cadence ruled by Blake, 2026-10-12)
The lane (`bazel test //tests:sanitizer`, which runs `tools/sanitize_oracle.py` with the toolkit's own compute-sanitizer,
`docs/testing.md#the-sanitizer-lane`) runs memcheck, synccheck and initcheck on
each kernel's one-CTA cells -- `tools/coverage.json` names every kernel's row -- when a component lands, before a
ratify, and on a schedule; never per commit (Blake: "every commit too noisy ... it nearly crashed the computer
sometimes though, racecheck especially"). Each run is guarded by a kill timeout and the host budget. It reports zero
hazards, and a target killed under the sanitizer is a failing gate, never a resource problem to work around: racecheck
exhausting the host on `cp.async` was a hazard's own signature. Benign races (every warp storing the same word) are not
tolerated either: one warp writes, the readers stand behind an edge (`docs/CONSTITUTION.md#r21`). Racecheck is deferred
for every kernel (Blake, 2026-10-01): it does not finish on the mbarrier pipelines.

## <a id="s19"></a>§19 — THE COMPONENT BUDGET GATE (law, 2026-09-09, from the carry rebuild's four reference-grade forms)
Rule 12 said "state the floor before building"; nothing enforced it, and a whole kernel was built four
times to correctness with its schedule machinery at 3x master's instruction count before the floor was
ever compared. From here the floor is a GATE, per component, before composition:
- A COMPONENT is a device function with one job (a head, a ring's issue, a step's body, a re-layout, a sweep). Each has
  a BUDGET LINE in the brief before code: its analytic floors (tensor: its MMAs x the card's cycles an MMA /
  schedulers; bandwidth: bytes / BW; latency: chain depth x latency at the stated prefetch distance), its instruction
  budget derived from the design's own operation count, and WHO EXECUTES IT HOW MANY TIMES (a per-window job is done
  once per CTA, by one warp or split across warps -- never replicated on every warp).
- Each component is built into its family's PART HARNESS (one `__global__` driver per component over the real device
  functions, inputs shaped by a cell's liveness)
  and measured there -- cycles at the locked clock, instructions, the binding counter -- against its
  floors. It is accepted at <= 1.3x its binding floor and within its instruction budget; the numbers
  enter the regression store, and a later change that moves a ratio is red.
- The COMPOSED kernel carries a phase ledger (a debug build's per-warp `clock64` per phase) and is
  compared to the sum of its parts; the difference is the composition cost and is reported as such.
- The per-component instruction count of the shipped binary is attributed by `tools/region_ledger.py`
  (SourceCounters + lineinfo, functions and struct methods) against `tools/budgets/<kernel>.json`; a
  component over budget is red, the same as a spill. Every kernel has its budget file, or an owed or exempt
  entry in `tools/coverage.json` saying why not (§25).
- SKELETON FIRST (Blake, 2026-09-09): the kernel is composed from STUB components before any is
  implemented -- each stub has the real signature, the real shared-memory tenants and edges, and does
  the cheapest thing that keeps the composition correct in shape (a zero fold, a pass-through head) --
  so the final form, its ledger and its phase structure are fixed and compile before the first real
  component lands. Components are then implemented one at a time into the skeleton, each gated in the
  part harness and in the composed phase ledger before the next begins. A component is never written
  outside the form it will live in.
- A design verdict is never drawn from a build whose components have not passed their gates
  ("slop numbers get misread as design verdicts" is now a gate, not a memory).
- EVERY PHASE AT ITS FLOOR BEFORE COMPOSITION (Blake, 2026-11-01: "we should of course ensure our individual phases
  are lean / at their floor / individually understood, analyzed, optimized, before we start composing them"): a
  cross-phase lever (an overlap, a reorder around a barrier, a parallel form) is taken only after every phase it
  composes passes its gate at the cell's own step count, sparse and dense; what remains above the phases' floors is
  then the composition's to win, stated as such. A parallel form (more warps, a split, a scan) is reached for only
  after the core is at its floor.

## <a id="s20"></a>§20 — ptxas SIGNATURES (law, 2026-09-09): what the SASS gate reads, and the rule behind each
`tools/sass_gate.py` runs after every build (the library's gate action, iteration builds included), on
every function of every cubin, and fails on any of these -- and on a kernel the source defines that no cubin
carries; each was re-learned by trial this week and costs a rebuild when missed.
`[COMPILER]` Every signature below was read from sm_86 cubins, the MMA an `mma.sync`'s HMMA; each is re-read on this
line's `wgmma` steps (owed).
- LDL/STL > 0 or a stack frame: LOCAL MEMORY. Causes: a runtime index into a register array or
  struct member array (the layout struct indexed by a runtime level); a `constexpr` helper with a
  runtime argument (`span_base(D, BC, l)` with `l` from a variable evaluates the span rule's arrays);
  a capturing lambda that was not inlined (its `[&]` captures go through memory). Rule: static indices
  only, compile-time arguments to constexpr helpers, lambdas only where inlining is verified.
- `CALL.REL.NOINC $__internal_*` : a CONVERGENCE SUBROUTINE. ptxas outlines a collective it cannot
  prove convergent: under a branch on a loaded value, after a loop whose trip count it cannot prove
  uniform, inside a lambda with an early `return`. Rule: collectives at the top level of loops whose
  bounds come from constants, REDUX results or uniform registers; predicate RESULTS, never the
  collective; a quad sum past a branchy loop goes through shared memory, not shuffles.
  ADDENDUM (2026-09-29): the gate read NONE of these until this date -- it matched the call's operands
  against its opcode alone. Reading them found 10,258 calls into `WARPSYNC` + collective subroutines
  across the entmax family, the decode arms and five intra arms, and EVERY ONE GUARDED: this ptxas (CUDA
  13) emits the inline collective for a converged warp and reaches the subroutine only through a
  `BRA.DIV` into a fallback block or past a `BRA.CONV` (probed: a branch on a loaded value and a loop
  whose bound it cannot prove uniform compile to inline shuffles or to guarded fallbacks, never to an
  unguarded call). A guarded call is cold code, not a cost a collective: the gate counts it (`guarded`)
  and reds only an UNGUARDED call, a converged warp's. The rule's reason is now code size: control a
  branch around collectives reads belongs in the uniform datapath (kernel parameters indexed by block
  index, constants), which keeps the fallbacks out; a data-dependent loop exits on a vote.
  ADDENDUM (2026-11-01/02): two causes found since. (1) An OUTLINED LAMBDA: a lambda's operator is not
  force-inlined; past the inliner's threshold NVVM outlines it, its by-reference captures go to memory and
  every collective inside becomes a call -- so where the gate shows fallbacks or a stack frame with no spills,
  first read the `CALL.REL.NOINC` targets' names (an outlined lambda shows by its mangled name); a large body
  dispatches through force-inlined frames (`uniform_switch`), and a kernel parameter captured by reference is
  `const __grid_constant__`. (2) A LOOP BOUNDED BY A PER-THREAD VALUE (`threadIdx`-derived, or a value ptxas
  cannot see is uniform) with shuffles inside: ptxas loses convergence for everything after it in the
  enclosing loop -- the drain's first facts form cost 84 more guarded collectives and a fold MMA step
  duplicated into a fallback; its bound taken from lane 0 by a shuffle restored the 9 the kernel had
  (`docs/internals/common/atoms/warp.md#uniform-warp`).
- `@P HMMA` or a pair body whose non-tensor instruction count equals its `if`-guarded block's:
  IF-CONVERSION. A warp-uniform `if` (dealt work, a runtime flag) is predicated, not branched, and runs
  predicated-off on every other warp. Rule: a one-trip `for` (`#pragma unroll 1`) forces the branch;
  better, keep dealt or optional work out of pair bodies.
- BSSY/BSYNC per item in an unrolled loop where every item is live: the PREDICATED-CHAIN form; use a
  straight-line block when all items are live (ptxas schedules across items) and a jump table (BRX)
  when one is. Both, selected by a uniform test, never one for all.
- A convergence barrier or a forward conditional branch inside the innermost loop that issues MMAs: A BRANCH
  INSIDE A STEP (`mma_loop_branches`, red since 2026-10-06; §27). The loop's own exit followed only by register
  copies and an unconditional back-edge is its latch, the step's end, not inside it (2026-11-06,
  `docs/internals/tools/sass_gate.md#step-branches`). No kernel built before the rule still
  branches in a step, so the gate names no branch debt; a lab form that keeps its branches declares them (an
  expectation on BSSY), or its number is refused, and an instrument's probe declares the step signatures it keeps
  beside its source, red when undeclared or unused (`docs/internals/tools/sass_gate.md#instrument`).
- A warp's copy tasks mapped so its lanes touch different rows: one L2 request per copy. Rule: tasks
  CONTIGUOUS in memory across a warp's lanes; coalescing is checked by `l1tex__t_requests` per
  sector, not assumed.
- Instructions executed an MMA (`smsp__inst_executed` over the MMA count) above the arm's ceiling -- what the card's
  issue rate leaves beside its tensor pipe at the design's warps a scheduler, stated in the budget: the kernel is
  issue-bound; the budget ledger says which component.

## <a id="s21"></a>§21 — THE MEASUREMENT PROTOCOL (law, 2026-09-09, after a dense verdict was drawn against the baseline's sparse schedule)
A comparison is valid only when: the same grid and block (read `launch__grid_size`), the baseline run
with ITS OWN schedule for the cell, `smsp__inst_executed`, `sm__cycles_active` and `sm__cycles_elapsed`
beside the time (idle SMs and tail waves are reported, not inferred), the GPU quiet (no sanitizer or
profiler child alive: a timed-out `compute-sanitizer` leaves its python child on the device), and the
probe's clock rows accepted. "Issue active at X%" is a ratio over active cycles and says nothing about
the instruction count; the count is read, never inferred from the ratio.
A change to a shared body is TIMED AT BOTH ENDS AT SCALE before it is called a win: the sparse and the dense cell,
in the measurement suite's at-scale sessions (2026-11-02: four commits each faster at sparse cost the dense carry
+3% at 64K x 4 heads, one of them +3.7% on an identical address; the local dense ledger swings ~2% between runs of
one binary and could not see it, the sessions repeat within ~0.3%). A design comparison quotes every family this line
builds, from the same counts priced at each card's rates (`tools/pipes.py`); a form that wins on one card and loses on
another is flagged and taken only with the trade written down (the constitution's LEAN ON EVERY AXIS, ON EVERY CARD).

## <a id="s22"></a>§22 — THE MEASUREMENT SUITE, AND HOW A MEASUREMENT IS READ (law, 2026-09-12, after a week in which a register cap, a "spin", a pipe metric and a schedule default were each misattributed)
WHAT IS MEASURED, on every build, before any component is judged, as one report diffed against the
previous build's: (1) the SASS composition (§20) and the REGISTER LIFE RANGES (`nvdisasm
--print-life-ranges`): peak live registers per source region beside the allocation -- a budget
argument cites the peak live at the program point, never the allocation; (2) the scoped oracle; (3)
the phase ledger on the GATE CELLS (the N = L cells, §21's baseline rules) with per-warp rows, so an
imbalance is seen, not inferred; (4) the region ledger with per-line stalls; (5) tensor utilization as
the MMA count times the measured cycles an MMA holds the pipe, over scheduler cycles -- never a
profiler's pipe-active ratio, which is normalized to a rate this kernel does not run at; (6) the A/B
against the baseline RUN WITH ITS OWN SCHEDULE, named per binary (a spec without one ran the dense
order and recorded the reap); (7) THE PHASE CENSUS: the MMAs a phase COUNTED from the profiler's
per-instruction dump (never the model's), utilization a phase against the phase ledger's cycles, a
warp's cycles at its MMA instructions against its other work a unit (a fragment, a pair) -- the
number that says whether two warps a scheduler can cover each other (the other work must fit under
the partner's burst), and the instructions a component against its budget with the launch's
CTA-window divisor (a budget is per CTA-window; a per-launch total against it flagged everything
red and meant nothing); (8) THE WAVEFRONT CENSUS: shared-memory wavefronts above the profiler's
ideal, by source line -- on a shared store or load, bank conflicts on the store side of a layout,
which the load side's conflict-free `ldmatrix` hides from every other instrument (an 8-way conflict
sat in the snapshot for a week as "wider stores don't pay"); on an asynchronous copy (`LDGSTS`) the
count is the GLOBAL LINES the copy touches, not a bank conflict (sm_86, calibrated 2026-09-18: a line a lane
reads 32 of ideal 4, the fill's sixteen token rows 16 of 4, whole lines 4 of 4, and the rows and the
lines forms copy at the same rate -- the sectors are what a gather pays); (9) THE COMPOSER (a family's own, built
with its schedule): a phase's
time attributed to its PARTS by building compositions in which a part is real or a stub (the stub keeps
the part's MMAs, same count, atom and accumulators, and drops its other work), read in wall time rung by
rung, each composition checked by its MMA count against the kernel's and by its device instructions
against the kernel's hash. A per-part cost is the difference of two measured times. It exists because
warp-side instruments cannot say whether the pipe is idle, and the MMA-only composition is the phase's
measured floor; (10) THE CALIBRATIONS (a family's own, built with its schedule):
each operation the parts are made of (the kernel's MMA atom, shared loads and stores, matrix loads,
asynchronous copies, global reductions, CTA and shared-memory barriers, warp syncs) alone on every warp,
as cycles an operation a warp on this card -- reference rows every reading uses in place of a constant (an
MMA's cycles was one such constant for eight days); (11) THE PIPE TIMELINE (a family's own, built with its schedule): the
tensor pipe's true utilization over a launch, measured on silicon by the profiler's PM sampling (1 us samples), on a
scale taken from the MMA-only composition's plateau -- the one instrument that says when the pipe idled rather
than how much in total; validated against the MMA-count utilization at three points on sm_86 (66.8 / 21.0 /
33.9% against 65-66 / 20-21 / 33.7%).
HOW IT IS READ: a component's THEORY -- what the form must do to the ledger, and why -- is written as
its budget line before the build; the measurement is read against that theory. A miss is explained
by a measurement that ISOLATES the mechanism (a line, a counter, a life range, a variant that removes
one thing) before anything is reverted; "it did not work" is not an explanation and is not a verdict.
A form that loses is a form: the idea stays open on the card until a mechanism-level explanation
closes it, and the card records which. An instruction count is time only when the sampler agrees
(a spin that is 10% of instructions and 0.1% of samples is issue slots, not time), and a BANK CONFLICT
is time only when the line's sample share agrees (the drain's 4-way store conflict was 3,072 excess
wavefronts a window and 0.4% of samples: removing it bought nothing, because the drain's path is its
global reduction at 3.4% -- the wavefront census is read beside the per-line samples, never alone),
and a BARRIER'S WAIT belongs to whichever phase issues the first instruction that depends on it: a
deferred-blocking barrier lets the clock read of a lap placed after it issue at arrival, so the wait
is charged to the next phase, and a wait is read only from per-warp arrival stamps taken before the
barrier (the 1,738-cycle "scans" were ~460 of scans and the readout's imbalance, 2026-09-12). A probe that can
report a stale artifact (a PTX assembled after a failed compile, a worker default standing in for a
schedule) is a defect in the suite and is fixed before its numbers are used again.

## §22 addendum (Blake, 2026-09-12): THE GATE IS A DECISION TREE THAT ONLY GROWS
"we should never point towards or assume from data that isn't definitive. with our reusable gating
system, we keep expanding it to add more metrics and data, and any time we find we need more info we
add more profilers, metrics, etc. to the gate -- so that it becomes a decision tree: any time the
kernel isn't working, go look at this metric, it deterministically tells you what is going on."
Consequence: the measurement suite (rola-bench's `measure`) is that tree's trunk. Every number computed by hand in a debug
session becomes a permanent step of it the same day; a node that cannot decide the question it was
read for (two mechanisms fit the same numbers) is the signal to add the node that splits them, and
until that node exists the question is OPEN, not answered by the likelier story.

## <a id="s23"></a>§23 — THE DEV ENVIRONMENT IS ONE SYSTEM (law, 2026-09-12, from Blake's "one unified dev config ... everything reads that one file", "consolidate everything into one dev env init script, and also ensure docker stays maintained")
A tree that measures itself is only as reproducible as the machine it runs on. Before this law, 26 environment variables,
three per-machine files and fourteen hardcoded paths decided that machine, set by five scattered scripts and a page of
manual steps, and the dev container had drifted for two weeks unnoticed.
1. ONE LOCATION. Every value that depends on the machine is declared in `tools/dev_config.py` and read from the dev
   config directory (`~/.config/rola/`, one JSON file a section) through that loader. A tool never reads a setting from
   the process environment and never spells a machine path; the environment variables that remain are the loader's
   listed HANDOFFS (a parent process passing a resolved value to a child); what one build is, is the graph's settings.
   `tools/lint/lint_standards.py` refuses any other read and any CUDA or Windows path spelled in tools or benchmarks. A replaced variable is deleted, never kept as a fallback (§18).
2. ONE INIT. `tools/dev.py`'s commands, run as the graph's targets of the same name (`bazel run //:init`, `//:check`,
   `//:clock`, `//:dev_env`, `//:worktree`), are the only thing that sets up a host (config, pinned tools, commit gate,
   clock, each checkout's venv, worktrees) and the only thing that checks one (`check`: every dependency a row, exit 1 on
   a red, the build's own gates reused); outside the graph is only `python3 tools/bootstrap.py`, the pinned bazelisk the
   targets run under. The devbox container runs the same loop inside. A new machine-dependent value lands as a schema
   key with its `init` detection and its `check` row in the same commit.
3. THE CONTAINER IS KEPT CURRENT BY A KEYED GATE. The devbox image is a supported environment, not a document: a graph
   target (`//:image`: the pinned base, the pinned OS packages) and a thin wrapper around the bootstrap, whose identity
   is its content digest, named in `tools/devbox_image.json`. `bazel run //:container_check` runs the host's own loop
   inside it on fresh volumes of its own -- the bootstrap, `init`, the checkout's venv, `check`, the commit gate, the
   GPU and reference library, a host-held lock seen as held, an iteration build imported -- each step timed, and
   records the result under the digest. The commit gate refuses a tree whose image is not the one it names, and a
   change to the name without a passing record for the new digest. The loop's own inputs (a pin, the lock, `dev.py`)
   are proven by every host that runs the same loop; the container shares only the host's locks, the image's digest
   (read-only, which ratify records) and, under WSL, the driver's libraries.

## §24 — TWO TIERS: DECIDE COARSELY, BURST STATICALLY (law, 2026-09-17, from the fold's and readout's forms)

`[MACHINE]` The ampere family's rule, and the branch's: a synchronous `mma.sync` holds its warp at the pipe, so a
phase that issues MMAs was split in the source into a DECIDE tier (everything data-dependent, run at a boundary into
a list and a count) and a BURST tier (a counted loop over the list with no vote, wait or data branch inside it). On
hopper the split is the hardware's -- an asynchronous `wgmma`, TMA, mbarriers: producer warps decide and stage, a
consumer warpgroup issues -- and what it asks of the source is unmeasured. No hopper schedule is designed yet: its
design states its two tiers in its steady-state timeline (§32.1), and this section is rewritten from its labs.

What holds on every card is the shape of a step (§27): one block, no data-dependent branch; sparsity a shorter list,
never a skip; a decision needed mid-way two loops with the decision between them; the list built by the lanes at
once, never by a serial scan. The source lint of a function that issues MMAs (`tools/lint/burst_tier.py`, a ratchet:
`//: @burst` in its declaration block, and inside it no vote, warp sync, shared-memory barrier, rendezvous, `while`,
`if` that is not `constexpr`, or `#pragma unroll 1`; `//: @burst-exempt <reason>` names a debt the ratchet holds)
keeps that shape, and reads hopper's MMA atoms when they exist (owed).

Precedent: CUTLASS's tile scheduler against its collective mainloop; FlashAttention-2's unrolled, branch-free KV loop
with masking by predication; ThunderKittens' producer and consumer warpgroups.

## §23 addendum (law, 2026-09-18, from the research behind the burst: the coordinator's research journal, SASS guidelines)

1. `[COMPILER]` THE EMITTED DISTANCE IS THE LATENCY BUDGET. ptxas sets an instruction's stall count to the producer's
   latency minus the instructions it placed between producer and first consumer (Huerta et al. 2025, §4, validated on
   GA102), so a dependence is hidden only by what sits between its ends in the SASS, never by anything in the source.
   A family's step cover reads that accounting once its model exists (§32.4).
2. `[COMPILER]` ORDER BY DEPENDENCY, NEVER BY HOPE. There is no scheduling knob (nvcc §4.2.9.1) and no supported
   ordering; the one sanctioned device is a data dependency, one primitive in a common header (§32.6). No fence for
   ordering (it orders memory operations and costs runtime), no clock trick. The burst-tier lint refuses a fence in a
   burst.
3. `[COMPILER]` LOADS FIRST, ALL OF THEM, THEN THE ARITHMETIC ON THEM. The canonical hand schedule spaces the loads and
   pushes the stores down (Gray, SGEMM: 98.5% useful cycles); the measured good attention kernel keeps about eight
   instructions between an `ldmatrix` and its consumer, the bad one one or two, and the cause was register pressure
   (lubits.ch, A100). The gate's exposed-latency reading is this rule's number.
4. REGISTERS ARE THE PIPELINING BUDGET, AND THE THRESHOLDS ARE NUMBERS. Allocation is in eights a thread against a
   scheduler's register partition, so the warps a scheduler a count buys is arithmetic (`[MACHINE]` GA102's 16,384:
   248 a thread buys two warps, 168 three -- its whitepaper; Jia et al.; sm_90a's partition is a lab's to read).
   A component states its live-register cost; the gate prints the arm's distance to the next threshold.
5. CONTROL BITS ARE THE COMPILER'S UNTIL THE GATE SAYS OTHERWISE. Patching yield and reuse bits gave Hopper 10%
   once and was retired when the compiler learned the interleave (DeepGEMM). Nothing below ptxas is patched
   unless the coverage reading shows the order right and only the bits wrong, and then as a ratified artifact
   with its patch script under the same gate.
## §23 addendum 2 (law, 2026-09-18, from Blake's "either the simulator is wrong, or something is binding the kernel that shouldn't be. we need a way to find that out")

1. A MODEL READS AGAINST THE PROBE BEFORE IT READS THE KERNEL. A pipe model simulates every
   calibration row's loop as the probe runs it and prints it beside the row's latest clocked measurement; a floor
   is ratcheted on a loop only while the model reproduces the rows that exercise what the loop does (the tensor
   pipe, the memory pipe, a gather chain). The first model passed one point and was called trusted; its own
   chain row already refuted it, and it budgeted a form that lost.
2. ONE POINT IS NOT VALIDATION. A model's reading of the kernel is trusted after two forms of the same loop have
   been measured and both reproduced, never after one; a second point that misses names what the model lacks, and
   the miss is worked before the model reads another form.
3. A PROBE ROW MEASURES WHAT ITS ADDRESSES SAY. A row is named by its access pattern, and the pattern is the
   kernel's: the `matrix_load` rows were broadcasts (one wavefront a load) for a week and read as the kernel's
   four-wavefront loads; `matrix_load_rows` is the kernel's pattern. Every row's header states its pattern.
4. AN INSTRUMENT'S DENOMINATOR IS READ OFF THE KERNEL. The phase trace divided a chunk's fragment phase by one
   fragment's HMMAs and reported the fold at 3.8x its pipe time; the phase holds a chunk's four fragments. A tool
   that compares a measurement to an ideal states where the ideal's count comes from.
5. A STALL CENSUS IS READ PER LINE. Per component it says a warp waits; per line it says on what. The isolating
   measurement for a model's miss is the per-line census of the two forms, not a reading of the model.
6. `[COMPILER]` ONE BODY, A REGISTER SELECT FOR A VARIANT. A large inlined body instantiated twice (a tile called with
   its two outputs swapped) with four statically named variants inside (a drain pass a compile-time half)
   put twelve copies of one body in the kernel; ptxas outlined the shared body into a real subroutine
   (`CALL.REL.NOINC` from twelve sites), allocated 255 registers and spilled 117 times, and `nvdisasm`
   refused the cubin's life ranges (2026-09-19). One body with the variant a runtime value, its register
   chosen by a select (`hsel ? y[j][2] : y[j][0]`, never an index, which is local memory), took the same
   kernel to one copy, 254 registers, no spill. A form's register cost is read at one arm's compile
   (its registers, its `CALL` count) before its chain runs.
7. `[MACHINE]` What an instruction between two `mma.sync` MMAs costs its warp: the ampere family's, the branch's.
8. `[COMPILER]` A PROBE'S PLACEMENT IS READ OFF ITS SASS. `asm volatile` orders nothing below ptxas: a chain written
   after a burst was interleaved among its HMMAs by ptxas all the same (the queue rows, 2026-09-18). What a row
   measures is the placement in its SASS, which the row's reading states; a row whose placement differs from the
   kernel's measures a different thing.

9. A STUB PRICES ONLY ITS OWN ACCESS PATTERN (2026-09-22, four negative builds on the fold and readout bodies): a
   drain stub that writes coalesced lines cannot price a drain that writes the accumulator's scatter -- the stage
   that turns the layout into lines earns its round trip. Before a part switch's trivial form is read as a win, what
   it actually writes is read. Likewise compaction: splitting tokens into compacted lists removes dead rows but pads
   every list to a whole fragment, so at a short chunk the padding equals what it removed; it is priced against its
   padding first and pays only on long lists.

## §25 — EVERY GATE COVERS EVERY KERNEL (law, 2026-09-29, from Blake's "expand our gates and standard to cover all our kernels and source generically")
An inventory found the kernel gates had each narrowed to the kernels their author had in mind: the SASS gate read one
cubin, the device clang-tidy one unit, ratify a hand-written list that had lost three shipped kernels, and the budgets,
phase-ledger tool and declared arm set named the carry alone. Nothing had decided any of it. So:
1. ONE KERNEL LIST, DERIVED. `tools/kernels.py` is the translation units the build compiles and every `__global__`
   the source defines, with its family and the units that compile it. The build, ratify and every gate read it; no
   tool keeps a second list.
2. A GATE TAKES ITS TARGETS FROM THE LIST, and a gate that finds no target fails -- it never passes by absence. A
   narrowing (`--kernel`, `--member`) is a measurement, never a gate.
3. EVERY KERNEL MEETS EVERY OBLIGATION OR SAYS WHY. `tools/coverage.json` declares, per kernel, the oracle test, the
   sanitizer row, the budget and the phase ledger -- a pointer, an exemption with its reason, or an owed entry --
   and `ratchet.py coverage` gates it: a kernel not declared, or a pointer to nothing, fails the commit; owed entries
   are the reviewed backlog, printed with their reasons (a new kernel lands owing its budget and its ratification).
   Ratified is declared "manifest" and checked against the committed manifests, or owed until the milestone.
4. A GATE RUNS WHERE ITS INPUTS EXIST: the source gates in the commit gate (`bazel test //:gate`); the binary gates
   (the SASS gate, ratify) after every build.
5. A TOOL THAT CANNOT PARSE A UNIT HAS NOT GATED IT. The analyzer is pinned to a release that parses every unit
   (`tools/clang_tidy_pin.json`), and a parse error is a gate failure: the distro's clang 14 reported used
   parameters as unused on a unit it could not parse.
6. A REGION DECLARES ITS BARRIER FREEDOM (Blake, 2026-09-29, generalizing the decode prologue law of 2026-08-27).
   Any device file may mark a region `//: @barrier-free <reason>` ... `//: @barrier-free-end`; a CTA or grid barrier
   inside one fails the standards lint, and so does a region that never closes. The decode step's walk is one; the
   work queue's visit folds and switches heads behind barriers after it.

## <a id="s26"></a>§26 — A RULE ABOUT THE MACHINE CITES A LAB EXPERIMENT (law, 2026-10-04, from Blake's "create a system where we can experiment with different cuda structures and get information we can eventually record as rules")
Two days of fold work rested on a per-instruction cost table that a later experiment overturned, and two rows of the
probe that replaced it measured instructions ptxas had folded or turned into calls. So:
1. A CLAIM ABOUT THE HARDWARE OR THE COMPILER — a rate, an overlap, a cost beside the tensor pipe, what ptxas does to a
   pattern — is a rule only when a lab experiment measured it (`measure/lab/experiments/`, run by `tools/lab.py`;
   docs/internals/tools/lab.md), and the rule names the experiment, its record (`measure/lab/results/<name>/<sm_XX>.json`)
   and the architectures it holds on. An architecture not measured is stated as unmeasured; a new card or toolkit
   re-runs the experiments its rules cite. A RECORD OF A CARD THIS LINE DOES NOT BUILD (sm_80, sm_86) is kept in the
   tree where a row of the hardware table cites it or its experiment measures a kernel every card runs; an experiment
   on the ampere family's own instructions is the branch `ampere`'s -- source, record and page -- and a rule that
   stands here on one names it and is owed its run on this line.
2. AN EXPERIMENT'S NUMBER COUNTS ONLY AFTER ITS SASS GATE: the declared instructions present, no local memory, no
   undeclared convergence call. The driver refuses to time otherwise.
3. THE RULES THAT STAND (each measured on sm_86 under CUDA 13.0, its sm_90a run owed). The letters a, c, e to p, r
   and t to z are the ampere family's machine rules -- what an instruction costs beside `mma.sync`, its pass
   pipelines, pools, lists and readout steps -- and stay the branch's; a card's rates are its rows of the hardware
   table (item 1). Hopper's rules take new letters as its labs run.
   b. `[COMPILER]` COLLECTIVES (`collective_branch`): a warp collective under control flow that depends on the thread —
      a branch on the warp id, or a trip count the warp's role sets — compiles into a CALL into a convergence
      subroutine (~1.65x the inline shuffle's cycles); under a branch on the block index or a kernel argument it stays
      inline. A collective's enclosing control flow derives from kernel arguments, the block index or constants.
   d. `[COMPILER]` INTEGER ADDS (`tensor_tax`'s SASS): ptxas folds repeated adds, rewrites add recurrences, merges
      two-input adds into `IADD3` and issues adds as `IMAD.IADD` on the FMA pipe on its own; the source does not choose
      the integer pipe.
   q. `[MACHINE]` `[COMPILER]` A GLOBAL REDUCTION COSTS ITS L2 LINE REQUESTS (`drain`): ~5 SM cycles a line request
      with every SM draining (the hardware table's `red_request`), so a row is reduced aligned to its lines and a
      request past what a union holds is not issued (the loop's warp-uniform bound). Inside a warp's last trip a step
      past it is issued as an exact zero, never branched around: a branch around an add compiles to a predicated
      `red.global`, which ptxas wraps in a convergence region a request (removing it took a drain 2,758 -> 1,623
      cycles a warp-window). A dead lane or step adds its zero at a row of its CTA's own, never at one row every CTA
      shares (a shared dummy line serialized every CTA ~6x).
   s. `[COMPILER]` A STEP'S BODY IS ONE BASIC BLOCK (`read_step` modes 6-8, the branch's): ptxas schedules within a
      block and does not speculate a memory operation, so `in ? load : 0`, `if (lane) atomic` and a predicated atomic
      in inline asm each compile to a branch with a convergence barrier, and a runtime `switch` selecting
      register-resident state or an `if` around a prefetch splits the block between the next step's loads and this
      step's MMAs. On the ampere readout's step: 3.2x its MMA floor with the branches; 2.1x with the loads
      unconditional (a pad's slot a real row, its value masked) and the adds of a value masked to zero; 1.56x with the
      whole sequence unrolled so its rings sit at fixed registers. Rotating a ring by copies in a rolled loop costs
      ~20 moves a step.
   aa. `[MACHINE]` A STREAM READS 128 OR MORE CONTIGUOUS BYTES OF A ROW (`row_span`): a work item owning a 32-byte
      piece of each row streams at half the DRAM peak (half a burst), 64 bytes at 0.81, 128 and up at the peak, at any
      block shape (2 x 8, 1 x 16 or 1 x 8 warps an SM). A level's work item spans 128 digits or more of its rows.
   bb. `[MACHINE]` BYTES KEPT FOR A LATER STAGE ARE APPENDED TO ONE PACKED STREAM, NEVER SCATTERED TO PER-KEY REGIONS
      (`vote_stash`): each byte stored at its key's own region dirtied a sector that left L2 half written -- 93 MB of
      DRAM writes for a 17 MB stash, the stage at 0.25x; appended lane by lane after a scan of the lanes' counts, the
      writes are the bytes and the stage costs 3% (k 4 of 256) to 8% (k 16). A lane's serial walk over its entries is
      bounded by the warp's busiest lane (0.6x at 25% density): dense key sets want a lane-parallel extraction.
   cc. `[MACHINE]` A WARP'S BIT WORDS LEAVE THROUGH A TILE AS 64-BYTE RUNS (`facts_block`): a warp storing one 4-byte
      word to each of 32 rows took ~4K cycles an item -- the facts pass's vote's whole cost; written to a shared tile
      and stored as each row's 64 contiguous bytes after the item's barrier, the vote runs at 4x its DRAM floor's rate.
      The 128-byte rule holds even for a run a row the next item re-reads: 128-byte runs read a whole sweep apart still
      stream at 0.98 of the peak, 64-byte ones fall to 0.59 (`row_span`, spans apart).
   dd. `[MACHINE]` AN sm_86 MBARRIER WAIT POLLS, SO A WAITING WARP SLEEPS BETWEEN POLLS (`facts_block`): eight warps
      waiting on eight builders spun 26% of the kernel's warp instructions (30.7M of 116.5M) in their LDS/LOP3/branch
      poll, taking the builders' issue slots; `__nanosleep(64)` between polls returns them. A wait whose producer is
      busy computing sleeps; only a wait sure to be short spins.
   ee. A STREAMING PASS'S FLOOR COUNTS ITS WRITES (`facts_block`): the facts pass writes its bitlists (an eighth of
      what it reads), union words and lists; against the read bytes alone its rate looked 0.81 of the peak, against
      the bytes it moves 0.92. Every streaming lab and gate prices the floor on read plus write bytes at a flat
      stream's rate measured in the same run.
   ff. `[MACHINE]` HOLD ONLY WHAT A LATER PHASE READS, STREAM THE REST (`facts_block`): a pass whose build needs one
      level's bytes after the vote cannot double-buffer every level's whole window in 99 KB -- holding the write level
      through its build left nothing in flight (TILED computes at 0.72x its floor at k 4 and streams at 1.28x); holding
      that level alone and streaming the others through a ring of small stages freed as voted takes it to 1.16x
      (resident 0.54x).

## <a id="s27"></a>§27 — A STEP IS ONE BLOCK, INTERLEAVED BY CONSTRUCTION (law, 2026-10-06, from Blake's "we should have lints and sass checks that enforce we properly interleave loops by default, and only override if justified")
The per-list readout ran 3.2x its MMA floor and the cause was the shape of its code, not the hardware: branches the
source never meant as branches, a block split between a prefetch and the MMAs it should hide under, and a ring
rotated by copies. The 2026-10-05 `list_fold` finding had said it ("each entry's pad check a branch that fences the
scheduler") and was not applied. So the step's shape is a rule, the default, enforced, with every exception named.
`[COMPILER]` The six items are what ptxas does to a step's source, measured on `mma.sync` steps (sm_86): the default
shape of a step on this line, re-read on its `wgmma` steps (owed).
1. A STEP BODY IS ONE BASIC BLOCK: no branch between a stage's prefetch and its MMAs, nor around its adds. A padded
   entry is made safe -- its slot a real row, its value masked to zero -- never skipped by a branch.
2. A LANE-CONDITIONAL SIDE EFFECT IS A MASKED VALUE: never an `if` around a load, store or atomic, and never a
   predicated memory operation (ptxas turns those into branches too).
3. REGISTER-RESIDENT STATE IS INDEXED ONLY AT COMPILE TIME: the loop over the indexing dimension is unrolled; never a
   runtime `switch` to select registers.
4. A REGISTER RING OF DEPTH D LIVES AT FIXED REGISTERS: its stages are unrolled by D (or the whole sequence, when its
   length is compile-time); never rotated by copies in a rolled loop.
5. THE OUTER DYNAMIC LOOP IS `#pragma unroll 1` AND ITS BODY IS THE UNROLLED STAGES: a remainder is padded to the
   ring period with a masked step, or peeled outside the steady-state block.
6. AN INTERLEAVE IS A PROPERTY OF THE SASS: read it there, never from the source. ptxas places register-only
   instructions; a data dependency is the only lever that places an MMA, and a zero-valued one costs an instruction.
ENFORCED: `tools/sass_gate.py` reds a convergence barrier or a forward conditional branch in the innermost loop that
issues MMAs (§20), on every function of every build; no kernel built before the rule still branches in a step, so
it names no debt, and an instrument's probe keeps one only where it declares it (§20). `tools/lab.py` refuses a lab number whose step branches unless
the experiment declares them (an expectation on BSSY: a comparison baseline kept on purpose). The burst tier's lint
(§24) holds the source side: no data-dependent `if` inside a burst function, `@burst-exempt <reason>` the only way
out. Register moves per MMA in a step (the copy-rotation signature) and the MMA gap (the longest MMA-free run in the
step loop's body, and around it) are reported; the gate on what they cost is §32.4's step cover, which runs once a family's model exists. Measured: §26.3 s.

## <a id="s28"></a>§28 — NO PHASE IDLES THE BINDING STREAM (law, 2026-10-07, from Blake's "any kernel we write should be going for maximal pipelining, ilp, etc. we shouldn't be compromising our standards, and if linters are letting us we should be making our linters more powerful and standards more clear")
The facts pass's list build was priced by instruction count (about 10 an entry, "inside the ~40% issue headroom")
and built as its own phase after the vote's barrier: no rows streamed while it ran, about 25% issue with one eligible
warp a scheduler, and the pass landed at 1.48x its DRAM floor -- near the SUM of its floors, not their max. The
coordinator then proposed lowering the line. No gate would have flagged the design: §19 asks for floors and a
latency chain, but nothing checked that one phase runs under another's stream.
1. A KERNEL IS PRICED AT THE LARGEST OF ITS FLOORS (issue, tensor, shared, DRAM), and its budget line names the
   overlap that makes it so: which stream binds, and how every other phase runs under it -- the next stage's loads
   issued before it, a second item in flight, steps interleaved. Never instructions alone, and never "fits in the
   headroom" without the structure that puts it there.
2. A PHASE BETWEEN BARRIERS THAT IDLES THE BINDING STREAM IS A DESIGN DEFECT: the design overlaps it, or the budget
   line proves no overlapped form exists within the resources (shared memory, registers) and prices it serial.
3. A LINE IS NEVER LOWERED TO ADMIT A SERIAL PHASE: a time above the line is red, and the design changes.
4. THE SAME DEFECT AT THE WARP LEVEL: a phase whose warps stall on one dependent chain (low issue, about one eligible
   warp a scheduler) interleaves independent work into it -- the next entry, the next item's loads.
ENFORCED: `tools/overlap_gate.py` times each kernel's perf rows and reds elapsed / max(floors) above 1.15; every
kernel names its row, or why it has none, in `tools/coverage.json`'s `overlap` obligation (the §25 ratchet). The
instrument, checked against a probe row: a pure 512 MB read stream (a torch reduction) reaches 96% of the profiler's
sustained DRAM peak on the RTX 3080 Ti (1.04x), so the DRAM floor is reachable; the tensor floor takes the product's
calibrated MMA rate (§22), its probe row owed with the first MMA kernel's row. Per phase, item 4 is §32's: each
phase against its predicted cycles (§32.2) and, once a family's model exists, every MMA step loop's pipe cover (§32.4).

## <a id="s29"></a>§29 — EVERY INSTRUCTION IS BUDGETED BY CLASS BEFORE CODE, AND HOISTED BY FREQUENCY (law, 2026-10-11, from Blake's "is there any kind of linting / sass checks we can add to stop us from writing such bad code? ... we do not take a standard analytical / deterministic approach to writing kernels when i feel the patterns are quite easy to formalize")
The readout lab (`read_image`) ran 18.7 instructions an MMA where its design needs about 9: each panel add
recomputed its token's row address and its pad select (~5 an MMA), every output converted on the conversion unit
(F2I) where one FFMA does it, and each digit's bookkeeping recomputed per use. Each was found after the fact by reading
counts; none was budgeted before the code, and the lessons were relearned after every context compaction. So the
method is a procedure with a gate, not a memory.
1. BEFORE CODE, A BUDGET BY OPCODE CLASS: the step's operation count per MMA in `tools/sass.py`'s classes (mma, shared
   load, shared store, shared atomic, global, shuffle, pack, conversion unit, float, select/compare, integer/address,
   control), derived from the design at the declared regime: per step, plus each per-digit (per-list) and per-window
   cost divided by the steps it amortizes over there (at alt-k4 a read digit is one step a window, so its bookkeeping
   lands on every step). The lab states it with `Bench::budget`; a production component in `tools/budgets/<kernel>.json`.
2. HOIST BY FREQUENCY: every value is computed at the outermost level its inputs allow -- once a kernel (a lane's
   fragment offsets), a window, a digit or list, a token, a step, an element -- and never again below it, WITHIN THE
   REGISTER LINE (§32.3): a hoisted value is a register for its whole level, and one that would pass a phase's line
   is re-derived at its use instead.
3. AN ADDRESS IS A BASE PLUS A COMPILE-TIME OFFSET: the base computed once at its level (a token's panel row, a
   digit's plane), each access an immediate or one add; a swizzle is folded into the base (§3) or applied by one XOR
   with a lane constant.
4. PADDING IS ROUTED BY A BASE, ONCE: a pad token's base is the lane's dummy row, chosen once a token a step; never a
   select per element or per access (with §27.1: never a branch).
5. THE FACTS PASS IS AN INDEXER, AND AN ENTRY IS A SLOT AND A BYTE (Blake, 2026-11-09): the facts pass emits what is
   the data's alone -- the liveness words and unions, each sparse level's lists (tokens in order, ranks, boundaries,
   totals), the per-token factor products -- and nothing of a card's: no exponent, no coefficient in an operand's
   format, no layout offset. What a card's step needs beyond the entry it forms where it expands the byte, once at
   the outermost level its inputs allow (item 2).
6. `[MACHINE]` CONVERSIONS RUN ON THE FMA PIPE: an fp32 to integer conversion is one FFMA adding 1.5 x 2^23 (the
   rounded integer in the low bits), a pair packed by one PRMT; the conversion unit (F2I/I2F/F2F/MUFU; on sm_86 16
   lanes a cycle an SM) never in a step.
7. GLOBAL MEMORY NEVER IN A STEP: a step's operands are staged ahead by the family's copy (TMA on hopper); a reduction
   to global memory is a drain outside the steps.
8. A LAB'S NUMBERS ARE A CORRECT KERNEL'S: every recorded variant checks its outputs against a host reference
   (`Bench::check`; inputs chosen so the reference is exact), or states why it has none (`Bench::unchecked`: an
   isolation whose outputs fold into a sink); every MMA kernel in a lab has a budget or a stated exemption
   (`Bench::exempt`: a rate probe, an isolation, a baseline). Found 2026-10-11: the readout lab's pad words overran
   into a warp's heads -- in bounds, so memcheck passed, and even a zero atomic add is a read-modify-write that raced
   the next window's copy -- and the corrupted counts were measured as a form's cost; its exact check fails on it.
9. `[COMPILER]` WHAT PTXAS ADDS IS PREVENTED AT THE SOURCE (measured 2026-10-11 in `read_image`, 13.2 -> 8.2
   instructions an MMA): a lane's constants are held in registers (`lab::held`, an empty asm; ptxas otherwise
   re-derives them from `S2R` at every digit); every branch on loaded data is a vote (`__any_sync`), so ptxas sees it
   warp-uniform and puts no convergence barrier (BSSY/BSYNC, WARPSYNC, BRA.DIV) around the collectives it guards; a
   loop whose trip count is data is bottom-tested on a vote, its entry test outside it; a prefetch is issued
   unconditionally into a pad past the last window, never under a guard; a lane-predicated copy in a loop becomes a
   branch, so data a few lanes need travels inside what every lane copies. Inside any loop (a block's item or pass
   loop) no loop sits inside a lane-divergent region and no exit tests a value read from shared memory without a vote:
   ptxas then cannot prove the warp converged at the enclosing loop's back-edge, and every collective of the body
   compiles guarded (2026-11-01, the producer: 345 guarded sites from one mass loop under a lane test; 45 from a Newton
   exit on the partner warp's vote read back). What remains is named in the budget: three never-executed `LDS` before
   every `LDGSTS`, a loop's exit `CALL`, alignment NOPs, `UIADD3` fillers between dependent MMAs.
10. A FREE-RUNNING CTA IS TIMED BY ITS SPAN (`WarpSpan`) against the free floor, the busiest scheduler over all
   windows. The mean warp's cycles undercount it, and a per-window floor overstates what it must take.
11. A BUDGET IS PRICED, NOT GUESSED (law, 2026-10-12, from Blake's "why do we always seem to fail to hit our budgets"):
   each line item is its trips times the SASS price of its primitives, from the price list
   (docs/internals/tools/budgets.md#prices) for the architecture. A primitive with no price is compiled alone and its
   SASS read before it is used, and enters the list. The coloring labs missed their stated lines 1.4-2.4x on operations
   counted where instructions are paid: a `redux.sync`'s move back from a uniform register, a runtime-indexed counter
   as a compare-select-add a word, both sides of a lane-dependent select.
12. TRIPS COME FROM THE MIRROR, AT THE WARP'S MAXIMUM: an experiment's mirror (its `.inputs.py`) or the facts it
   stands for counts each phase's trips per cell, and a lane-divergent loop is priced at the warp's longest lane, never
   the mean (alt-k16's tokens average 2.3 digits; the warp paid ~5-6). The fixed part is itemized phase by phase, never
   a lump.
13. A MISS NAMES ITS PHASE AND ITS PRICE: `tools/lab.py ledger <experiment>` attributes executed instructions to the
   kernel's phases (its `//:` headings) against the line items. A restated budget names the price that was wrong and
   corrects the list in the same commit; the stated-against-measured record is
   docs/internals/tools/budgets.md#ledger. Convergence regions inside a budgeted kernel's loops (`BSSY`/`BSYNC`,
   `BRA.DIV`) are removed by item 9's source patterns or declared (`LOOP_CONVERGENCE`); `tools/lab.py run` refuses an
   undeclared one.
14. EVERY RUN PRICES EVERY PIPE (amended 2026-10-18, from Blake's approval of the process changes after the lean
   hand-off): a form's floor is its binding pipe, not its instruction count. `tools/lab.py run` prints, for every
   kernel and per card, each pipe's cycles a unit -- tensor, the shared-memory pipe priced by kind of traffic (load
   wavefronts; stores and atomics at the larger of their instructions and wavefronts; copies and global loads by
   their bytes), issue, the integer, FMA and transcendental pipes at their cycles an instruction, the SM's global path
   (L2's lines and atomic requests, every SM busy) -- and on the local
   card each pipe's busy share, L2 sectors and the stall reasons ([`tools/pipes.py`](internals/tools/pipes.md); its
   prices checked against `smem_rates` to the percent). The ampere carry's hand-off had been measured only inside the
   composed window and was bound by the integer pipe, which no gate printed. A budget states its binding
   pipe's cycles beside its instructions; a kernel over 1.3x its binding pipe names the stall that holds it.
15. A BUDGET LINE NAMES ITS FORM (amended 2026-10-18): each line item cites the cheapest known form for its job from
   docs/internals/tools/budgets.md#forms (integer keys for scaled compares, an atomic a lane for a cross-warp
   reduction, held bases, the widest store, `.cg` copies, scalars in the record) or says why it departs; a form found
   cheaper replaces its row in the commit that measures it.
ENFORCED: `tools/lab.py run` reads each budgeted kernel's executed instructions by class (ncu's per-opcode counts),
prints them against the budget, and refuses to record a variant over it (`--force` to record a known-over variant on
purpose). `tools/sass_gate.py` reds, on every function of every build, conversion-unit instructions and global
accesses other than the family's staged copy inside an innermost MMA loop; a kernel built before the rule whose steps
still read global memory is a named debt with what replaces it, and the list only shrinks; no kernel keeps the
conversion unit in a step, so it names no such debt. A
lab variant that keeps one on purpose declares STEP_XU or STEP_GLOBAL, and an instrument's probe declares it beside its
source (§20).
`tools/lab.py run` also records no variant whose output check failed, and (past the experiments recorded before the
rule, its named debts) none without a check or a stated reason, and builds no MMA kernel without a budget or exemption.
Measured: `read_image` (sm_86, the branch's).

## <a id="s30"></a>§30 — SHARED MEMORY IS CONFLICT-FREE BY CONSTRUCTION (law, 2026-10-11, from Blake's "Should we create principles to ensure things are consistently designed to minimize bank conflicts always from ground up")
The readout lab's bank conflicts were designed in by accident: the panel's pitch suited one access shape and not the
other form's, pads went to whatever word was spare, and slot numbers were left to the data, so a digit's tokens
collided on banks at random. Where shared memory binds (the A100), a conflict costs more than an instruction. Fixing all
three cut conflicts from 1.6-2.1 an MMA to 0.02-0.03 and wavefronts 25%, with no instruction added (`read_image`).
1. EVERY SHARED BUFFER HAS AN ACCESS TABLE BEFORE CODE: each instruction that touches it (loads, matrix loads, atomics,
   a copy's stores) by lanes, addresses and banks; its layout (pitch, swizzle) is chosen to satisfy all its readers and
   writers, and states the instruction shapes it is conflict-free for. A new access shape on an existing layout gets
   its own placement rule or layout, never an assumption (§3 extends from fixed tensors to every buffer).
2. WHEN THE DATA PICKS ADDRESSES, A PRODUCER'S FORMAT CARRIES CONFLICT-FREEDOM ONLY WHERE THE COMPOSED CONSUMER EARNS
   IT (amended 2026-10-13): gathered rows and atomic targets indexed by slot can get slots the producer places so each
   instruction's set is conflict-free (a union's slots colored by residue, a slot's residue mod 8 its row's bank group,
   each token at the position its residue requires), but a placement constraint enters a producer's output format
   (§29.5) only after the composed consumer is timed with and without it and the producer's cost and the pads it
   forces are priced against the saving; a part's wavefronts are a hypothesis until then. Measured 2026-10-13: on
   sm_86 the carry ran as fast on rank slots as on the best residue coloring, its time following its lists' places,
   while the coloring was most of the facts pass's time, so the facts pass emits rank slots
   ([`facts.md`](internals/facts/facts.md#ranks)).
3. A PAD SITS WHERE A REAL ELEMENT WOULD: a free residue's row for a gather, a dummy row aligned to that residue for
   an add -- never a spare word that may share a bank with real traffic.
4. WAVEFRONTS ARE BUDGETED LIKE INSTRUCTIONS: each §29 budget has a `wavefronts` line, the design's bytes a 128-byte
   wavefront plus what the counter is known to add, and `tools/lab.py` gates ncu's measured count against it (and
   prints the bank conflicts in it). It is the floor of a card whose shared memory binds, checked on every local run.
5. A BANK MODEL RUNS BEFORE THE GPU: `tools/bank_model.py` gives an instruction's wavefronts from its lanes'
   addresses; the producer (the facts pass, or a lab's inputs standing in for it) runs it on what it emits and reports
   wavefronts by part, so a format change that adds conflicts is seen without hardware. It is checked per operation
   against ncu's counters once a layout (sm_86, 2026-10-11: matrix loads and atomics exact, entry loads within 0.03).
6. CONFLICTS ARE PRICED ACROSS DENSITIES: dense, sparse and the W = 1024 lists each, since a data-dependent conflict
   rate moves with the distribution (CONSTITUTION §R10).
Measured: `read_image`, `read_w1024` (sm_86, the branch's); the model: docs/internals/tools/bank_model.md.

## <a id="first-run"></a>NEW DEVICE CODE RUNS FIRST UNDER A KILL TIMEOUT (law, 2026-10-07, after a hang froze the host)
A warp shuffle placed as an operand of a short-circuit `||` ran on only the lanes that evaluated it; the warp hung, the
display driver stopped answering, and the host had to be restarted.
1. A warp collective is never an operand of `&&`, `||` or `?:`, nor under a lane-dependent branch: compute it on every
   lane, then combine.
2. The first run of new or changed device code is under `timeout -s KILL <seconds>`, its waits bounded; a hang is
   killed, never waited on.

## <a id="s31"></a>§31 — SKILLS SHIP WITH THE REPO, PROCEDURE ONLY, ROUTED BY HOOKS (law, 2026-10-12, from Blake's "have our agent skills point to docs so they stay consistent with code" and "repo portability, work tree support, and the ability to ship and change skills with our repo as it changes")
Of the fourteen skills on the coordinator's machine, seven were links into a deleted worktree, the repo's own copies
loaded only in sessions started inside it, and their text had rotted: dead paths, a copy of these standards stopping at
§21. A skill that already existed for per-section attribution went unused while its method was rebuilt by hand.
1. A REPO ACTION'S SKILL LIVES IN THE REPO (`.claude/skills/<name>/SKILL.md`), versioned with the code it drives, so
   every worktree carries its own branch's skills; a skill reaches an agent through the action it takes (4). Only
   cross-repo coordination (cards, the queue, a site's records) is a user-level skill.
2. A SKILL HOLDS PROCEDURE, THE DOCS HOLD THE TRUTH: a skill is steps and commands; each rule, fact, number and format
   it relies on is a reference to its doc section, never a copy. `tools/docrefs.py` holds a skill's references strictly:
   a skill with a pending or broken reference fails the commit, so a skill changes in the commit that changes what it
   cites.
3. PORTABLE: nothing under `.claude/` names a machine path. Hooks resolve through `$CLAUDE_PROJECT_DIR`; a site's own
   locations come from the dev config or the environment (`ROLA_SCRATCH`), as §17 has it for the code.
   `tools/lint/portable_claude.py` fails a commit that names one.
4. ROUTED BY THE ACTION, FROM THE CHECKOUT IT TOUCHES: `.claude/skill-routes.json` maps actions -- a path edited, a
   command run -- to skills. The PreToolUse router (`.claude/hooks/route_skills.py`) finds the checkout a call touches
   (the edited file's; a command's leading `cd` or `git -C`; else the session's) and puts THAT checkout's skill text
   in the acting agent's context the first time the agent takes the action there. Any number of worktrees on any
   branches route from any session, subagents included, and nothing is registered globally. A session started inside
   the repo is routed by the repo's own hook entry; one started elsewhere by the user-level bootstrap the repo ships
   (`.claude/hooks/skill_router_bootstrap.py`), which `bazel run //:init` installs and `bazel run //:check` verifies. `CLAUDE.md`
   shows the table. Skills make the right path the easy one; lints and gates make the wrong one fail.

## <a id="s32"></a>§32 — THE STEADY STATE IS DESIGNED FIRST, AND EVERY GATE SEES IT (law, 2026-10-16, from Blake's "why we keep failing to properly overlap and write good kernels. What about our standards are causing failure and can be codified into our docs, skills, lints" and "we should be software pipelining rather than relying on more warps")
The carry's rebuild passed every component gate and composed into a kernel whose step loops keep the tensor pipe
busy 40-44% of the time at its two warps a scheduler: the readout's tile loops issue each `ldmatrix` one instruction
before the MMA that reads it, about 200 registers stay live through every phase (so no phase has room for a second
stage's operands), and four query heads a write head bought no time per query head. The instruments to see it existed
and none was in the path: the pipe model read only loops named by function, whose names the
rebuild deleted -- its floors pointed at nothing and nothing failed; the MMA gap was reported "until a lab fixes the
bound"; §28's per-phase reading and the carry's overlap row were owed; the two builds of a change were compared by
unlocked times across runs; and a helper the common header had (`ops::fresh`) was reinvented in the carry. Components
pass alone; the time goes in the composition, and no gate looked there. So the composition is designed first and
gated, and an instrument's missing target is a red.
1. THE STEADY STATE FIRST: before a kernel's components are written (§19's skeleton), its design states ONE WARP's
   steady-state window (or item) as a timeline: each phase, the stream it runs under, every load issued a stage ahead
   of its use (its load-to-use distance in issue cycles against `common/hardware.json`'s latencies, lab
   `issue_latency`), and the registers live in each phase. Components get interfaces split at those stages (an issue
   half and a wait half), so the composition overlaps by construction, never by a later rebuild. More warps is never
   the overlap mechanism (§5, CONSTITUTION R15): a warp's own steps cover its pipe at the warps a scheduler its
   registers admit.
2. PREDICT, THEN MEASURE, AND A MISS STOPS: each phase's cycles a warp a window are predicted (its budget line, the
   pipe model) and written to the budget (`phase_cycles` per cell) before a schedule or composition change is built;
   the family's phase ledger reds a phase over 1.3x its prediction or without one. A miss stops work on the
   kernel until the price or the model that was wrong is named and corrected (§29.13) -- §22's abandonment rule
   applied to a phase.
3. REGISTERS ARE BUDGETED PER PHASE: the budget's `registers` line states each component's peak live registers, under
   the allocation of the warps a scheduler the design wants; `tools/life_ranges.py --budget` reds a component past
   its line or without one. §29.2's hoisting stops at the line: a value whose hoisting would pass it is re-derived at
   its use (`ops::opaque`, docs/internals/common/atoms/warp.md#opaque), a few instructions there instead of a register held
   across the window. A spill is never the signal (§20 reds it); the line is. A RE-DERIVATION IS A PRICED LINE ITEM
   (amended 2026-10-18): its instructions at its trips (§29.11), in the budget beside the register it saves -- the
   ampere carry's ledger miss (3,884 instructions a warp-window against ~2,020) was mostly tokens, sources and
   swizzles re-derived at each use and priced nowhere.
4. EVERY STEP LOOP COVERS ITS PIPE: `tools/sass_gate.py` simulates every innermost MMA loop of every function, on every
   build, from ptxas's control words and the family's machine model (its MMA's pipe time, its latencies and its
   shared-memory rates the hardware table's rows, each a lab's), and reds a loop whose COVER -- its pipe share with
   the warps a scheduler its registers admit running it together, sharing the scheduler's issue port and the SM's
   memory pipe, at most its share alone times those warps and 100% -- is under 1/1.15 (§28's line): a loop under it
   cannot be covered by the warps the design has (an instrument's probe that sets the pipe's share on purpose declares
   it, §20). A kernel under it is a named debt with what replaces it, each function held at its floor by
   architecture, and the list only shrinks; the floors are the debt's ratchet, never its target -- a kernel is clean
   only at the line on every architecture this line builds (Blake, 2026-11-03). The cover is a model's reading: a
   floor is lowered only in a commit whose locked comparison shows the kernel no slower on its cells, the numbers in
   the commit, or in the commit that corrects the model, every floor re-read on an unchanged kernel (the first model
   read one warp alone times its warps, blind to the shared issue port). A floor of an architecture the host cannot
   run is lowered only on that architecture's own rented run. `[MACHINE]` Hopper's model -- an asynchronous `wgmma`'s
   pipe time, TMA's rates -- is built from its labs with the first hopper schedule (owed); the gate runs no cover until
   a family's model exists.
5. A TARGET THAT DISAPPEARS IS RED: a floor, budget line or gate row naming a function, loop or cell the tree no longer
   has fails; it never reads as nothing.
6. ONE PRIMITIVE A JOB: before writing a helper, the common headers (`common/atoms/*.cuh`, `common/*.cuh`) are read for
   it; a kernel-local copy of a common primitive is a defect, and an unused common primitive is dead code.
7. A/B IS THE DECLARED SUITE: two builds are compared only in one interleaved, clock-locked session -- rola-bench's
   suite with the baseline as a reference checkout, pruned to the sessions (`rola-bench`) -- or by ncu's cycles of one
   launch each; times from separate unlocked runs are never compared.
ENFORCED: items 3, 4 and 5 by the tools named; the SASS gate runs inside every build (the library's gate action), so
an under-line step fails the build. Item 2's predictions are a kernel's design's to state, and its ledger check is red
until they exist.

## <a id="s33"></a>§33 — DATA STAYS IN REGISTERS UNLESS SOMETHING STAGES IT (law, 2026-10-18, from Blake's "'Staging rule as a kernel standard' ok")
Shared memory is a pipe a kernel can bind on (it bound the ampere family's sparse carry), so nothing passes through it
without a reason. A value is staged in shared memory only when:
1. other warps read it (a value shared across warps: a stream's stage, the panel);
2. it must be rearranged on the way (a transpose or relayout an MMA operand needs);
3. it must be buffered beyond what the warp's registers hold (a ring deep enough to cover a stream's latency); or
4. the MMA reads it from shared memory (`wgmma`'s B operand by descriptor; sm_100's `tcgen05` operands).
Otherwise it stays in the registers that produce and consume it (`wgmma`'s A operand, where it is the warp's own
accumulators: §2). Staging is not a pipe saving to be had by skipping it: a load from L2 into registers is written into
the banks and read out as a staged copy and its read are, and it holds registers for the loads in flight (`[MACHINE]`
sm_86, `smem_rates`: ~1.6 pipe cycles a 128 bytes against a copy's 0.5 and its matrix load's 1.0) -- so rows an MMA
reads from global memory are staged by the family's copy (TMA's bulk copy on hopper), never loaded to registers
direct. sm_90: item 4 stages B; sm_100: accumulators in tensor memory, operands as item 4.
