# KERNEL_STANDARDS — two tiers (Blake, 2026-08-28)

**CONSTITUTION** = Blake's standing rulings. Stable; changed only by Blake; they guide every law below.
They are: §10's correctness instruments (oracle = truth; bit-identity = refactor instrument); efficiency is
correctness; closed world (only measured binaries run); one form per mechanism, no fallbacks, no dead code;
build the whole design or stop and name the gap; the ancestral-code standard; §17 portability; ONE ATOMIC PRODUCT
(Blake, 2026-08-28: the repository in published form is ONE self-contained kernel product — never several kernels
being separately gated; every test lives in exactly one of the three tiers, `tests/unit`, `tests/integration`,
`tests/oracle` (docs/testing.md), plus `measure/`; the whole tree is gated holistically at its current state;
no per-campaign test directories, and no campaign names in the tree — names are roles); SELF-DOCUMENTING
CODE (Blake, 2026-08-04: code is legible without comments; comments are sparse — a file header and 1-3-line decl
blocks naming purpose and the one invariant; derivations, histories and why-not-the-other-form prose live in
`docs/internals/` mirroring `csrc/`); and §R8-R10
(all architectures; no fuzzy-threshold branching — the kernel handles any distribution; design and probe
for ALL regimes at once — dense is the worst case and a test of the kernel, never a regime to concede).

**LAWS** = the coordinator's procedural and incident-derived rules (§1-§9 craft, §11-§16 and their addenda,
§R2-R7). Dated, adaptive, revisable by the coordinator; each carries the incident that made it and is
repealed when the situation changes. A law never contradicts the constitution; where a law is an inference
from one moment of development it is marked as such and re-examined at the next milestone.
§R1 ("price overheads against the sparse cell") is STRUCK 2026-08-28 — subsumed by §R10 and wrong in
direction: overhead concentrating at sparse may be the design paying off; dense remains the worst case.

# ROLA KERNEL STANDARDS — how we write CUDA here (draft for Blake's read, 2026-08-26)

SCOPE. This document is about CRAFT: how a designed piece is realized at the hardware's grain. It is not
about design. Design authority is `development/queue/P76_*.md` and Blake's rulings; nothing cited here is
an argument for or against a design choice. Where RoLA has patterns with no analogue in the reference
kernels (routed ownership, the two carves, the exchange, the fan-in ring, the cooperative expansion), §R
gives the RoLA rule directly. References are the local copies under a named sibling checkout,
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
RoLA: the readout's A fragments ARE the fold accumulator's top halves. Any phase change that
moves state must be the exchange the design names (§R3), never an ad-hoc copy.

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
Primitives (`ldmatrix`, `cp.async`, `mma`, `red`) are wrapped ONCE (`csrc/rola/src/common/ops.cuh`); call sites use the wrapper.

## 4. Pipelines: explicit stage state, release immediately after consumption, two stages in flight
- FA3's producer acquires a stage before issuing, the consumer waits, issues the GEMM, waits for it, and
  releases the stage at once: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:750, 1140-1143`; pipeline types per
  operand `:283-286`; the invariant is documented where it bites (`:787`).
RULE: overlap requires two stages in flight in ANY form: the second stage is registers
(single-slot segments: drain by `ldmatrix`, refill, compute from fragments) or a second SMEM slot. Sync
carries exactly two facts, "landed" and "drained"; on sm_80 by the per-batch barrier (form (b)), on
sm_90 by mbarrier. No pipeline state is implicit in loop parity without a comment naming the invariant.

## 5. Warp specialization only as register economy
- FA3 gives the producer warpgroup ~24-56 registers and the consumers 160-256 via `setmaxnreg`
  (`hopper/flash_fwd_kernel_sm90.h:82-83, 308-309, 360-361`), with the budgets `constexpr` per configuration.
RULE: on sm_80 there is no register redistribution, so a fetcher warp burns a full slice and folds nothing —
form (b) (role-alternating) everywhere; specialization is the sm_90+ mapping and only where the
accumulator binds.

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
default all) for iteration; iteration builds are single-arch AND arm-subset; both-arch all-arm builds
once per stage. Each arm has a COMPILE BUDGET (wall time, cicc peak RSS) recorded in the manifest.

## 8. Arch mapping at the atom, closed-world at the arm
Feature selection lives where the primitive is defined (`kernel_traits.h:40-44`'s `DefaultCopy` fallback),
never as `#if` at call sites. RoLA adds: every structural decision states its sm_80/90/100 mapping on the
card (§4 of the architecture); only measured binaries run (pinned ptxas, per-arch ratified manifests, device stamp).

## 9. The backstop (checks that should catch nothing)
`ratify` records per arm and arch: registers, stack, spills, residency (existing) + compile wall time,
cicc peak RSS, and the body count of each named `static_for` nest as a function of the arm (new). A budget
breach fails the gate exactly as a spill does. The device stamp is asserted before any gate.

## 10. Correctness instruments (unchanged, restated so they sit beside the craft rules)
The fp64 oracle is the truth. Bit-identity against a previous binary is a REFACTOR instrument, never a
design constraint (amended 2026-08-26). No fallbacks, no dead code, no special-case paths; comments
are file-header/decl blocks only, prose goes to `docs/internals/` mirroring `csrc/`. Build the whole design
or stop and name the gap; efficiency is correctness.

## R. RoLA-specific patterns (no reference analogue) — the rule comes from the architecture
- R1 OWNERSHIP CARVES: a stream owns an affine rectangle of digits per level (per-side span vectors);
  the private slot order puts carved (sparse) levels outermost; liveness is the exact AND over levels.
  Executors ≠ streams (an executor folds the union of its streams' boxes).
- R2 COOPERATIVE FETCH + EXPANSION: one union walk per batch; writer-side placement into per-stream
  segments of value rows AND fully expanded coefficient rows; memo internal to the fetcher; consumers are
  pure MMA. The per-tile liveness mask is produced by the expansion (it knows which slots it wrote nonzero).
- R3 THE EXCHANGE (order law): every phase change re-lays the state into the other side's order on every
  arch; publisher places each leaf's row at the READER's slot (per-reader `[leaf][channel]` segments,
  16-bit stores at compile-time addresses); the reader pulls its contiguous segment with `ldmatrix.trans`
  and ejects. No pair-level or run-alignment constraint on carves.
- R4 THE FAN-IN RING (§2.6-C): union rank keys a C-row SMEM ring; first writer stores, others `red.shared`;
  the NEXT writer reclaims a complete slot by CAS and flushes it; spin on incomplete; drain at the phase barrier.
- R5 LO PARK: hi in registers, lo parked into the transit's drained slots, freed registers become
  leaves (box size); a per-arch/width declaration.

## 11. Debugging a device hang or race (recorded 2026-08-26 after a stage burned 2.5 h reasoning blind)
`cuda-gdb` works on this machine when the process is LAUNCHED under it; ATTACHING (`-p`) is refused by
Yama `ptrace_scope=1` (a Linux setting, liftable with `sudo sysctl kernel.yama.ptrace_scope=0`).
Recipe: `flock /tmp/rola_gpu.lock cuda-gdb -q --args <venv>/bin/python repro.py` → `run` → Ctrl-C on a
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

## 14. Waiting on a build (recorded 2026-08-26 from the agent time ledger: 327 min of `until pgrep … sleep` loops)
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
- RATIFY ONCE per stage, after the last source edit. It recompiles every TU on both arches; it is a gate,
  never an iteration tool.
- A stage has a budget of THREE full build+gate cycles. Iteration is single-arch, arm-subset, oracle-first.
  When the budget is spent and a gate is still red, the stage STOPS and reports the residual — it does not
  keep chasing green.
- MID-STAGE RE-SCOPING IS ALLOWED when it changes WHAT is built (a design or mechanism ruling) — Blake:
  "I'd rather re-scope than build something wrong and fix it later." A re-scoped stage restarts its
  cycle budget from the amended spec. Amendments that do not change the design (count formats, report
  shape, housekeeping) go in the NEXT brief instead.
- Every kernel family carries an arm-subset knob (the carry's ROLA_CARRY_ARMS); a family without one gets it
  before its next build.
§15 addendum (build parallelism, 2026-08-27): cicc peaks ~1 GB post-S1, on a 16-CPU / 24 GB machine.
Gate builds: exclusive lock, MAX_JOBS=8, NVCC_THREADS=2. Iteration builds: one of two parallel slots
(then tools/build/build_lock.sh; now the host budget `setup.py` takes, §23), MAX_JOBS=4, single-arch, arm-subset. Re-derive these numbers if
cicc's peak changes (a per-leaf nest can push it back to 6 GB — §1).

## §16 — Ratify at milestones, not stages (Blake, 2026-08-28)

A stage's gate is SMALL: an iteration build (single arch, arm subset), the fp64 oracle on the arms
it touched, the census on those arms, and the model tests. A stage does NOT run the both-arch
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

## §R addenda — LAWS (coordinator, 2026-08-28; incident-derived, revisable; R1 struck)
R1. [STRUCK 2026-08-28 — subsumed by §R10; see the preamble.] PRICE OVERHEADS AGAINST THE SPARSE CELL. A fixed per-window / per-contribution cost that is "cheap"
    next to a dense window's MMA becomes the kernel once sparsity removes that MMA (the ring: 230 M
    instructions at dense AND at k16; the replicated schedule; the per-lane gate). The design's payoff and
    its overheads scale with different variables; state both per mechanism.
R2. THE FAN-IN'S FIGURE OF MERIT IS 128 B SEGMENTS PER RED INSTRUCTION, not sector fullness and not
    instruction count: one token's 32 contiguous channels per issue (the staged panel layout) = 1 segment;
    4 rows x 8 channels = 4 segments at identical RED count = +93%; fragment-direct = 8 half sectors.
R3. FAN-IN ON AMPERE IS NATIVE GLOBAL RED FROM AN ALIGNED LAYOUT. Shared-memory fp32 atomics are a CAS
    loop; a locked SMEM ring paid 18x its floor to halve DRAM. The DRAM question belongs to the A100
    parity-curve receipt, not to the local cell.
R4. NO PER-WARP WORK THAT ANOTHER WARP ALSO DOES: CTA-identical derivations (union lists, owner
    ballots) are derived once and published behind a barrier the kernel already pays, or split by
    executor share — never recomputed per warp to avoid a barrier.
R5. THE SCHEDULE HAS THREE PARTS (fetch over TOKENS once; per-stream SENDS from each stream's own ballot;
    seal-and-consume). A union over (token, stream) pairs collapses the first two and re-fetches k times.
R6. A KERNEL'S SASS DEPENDS ON ITS TU NEIGHBORS: byte-identity is a claim about a partition, not a body;
    per-arm TUs make each body canonical (7 movers, proven inert and free).
R7. STAGED-vs-STRAIGHT IS DECIDED BY LATENCY, NOT COUNTS: "identical sectors and issues" lost 9.5% on
    long-scoreboard (e390764). Time, IPC and stall reasons decide; counts describe.

## §17 — Portability: what the repo points at (Blake, 2026-08-28)
The repository is SELF-CONTAINED: anything it DEPENDS ON — a tool it runs, a doc it cites, a record it
reads, a baseline it compares against — lives in the repo or in a NAMED sibling repository with a stable
layout (the bench/results repo), never in an ad-hoc place (a scratch directory, a home path, a session
scratchpad, a campaign journal, a worktree path). Ad-hoc material never enters the repo either — no
script blowup: briefs, transcripts, per-session scripts and campaign journals stay outside and are
cited FROM the queue, not from repo files. Absolute paths under a home directory are a lint finding.
Committed records carry repo-relative or host-keyed locations. Skills that ship with the repo point at
repo docs and the measurements record, never at a journal.

## §R — CONSTITUTION (Blake, 2026-08-28)
R8. DESIGN MECHANISMS THAT EXPAND TO ALL ARCHITECTURES. Every mechanism states its sm_80 / sm_90 /
    sm_100 form at design time (wgmma/tcgen05 SMEM- and TMEM-sourced operands, TMA box loads, clusters
    and DSM, vector RED); a mechanism that only exists on one arch is a workaround, not a design.
R9. NEVER BRANCH ON A FUZZY THRESHOLD. No dispatch, launch policy or in-kernel path keys on a measured
    statistic (sparsity fraction, density, live count, "small L"). Branch on STRUCTURE the declaration
    fixes (tile counts, modes, arm axes). The kernel itself handles any data distribution.
R10. DESIGN AND CODE FOR ALL REGIMES AT ONCE — dense, sparse, mixed, clustered, small and large L, small
    and large N. Never overfit a mechanism, a measurement or a verdict to one cell: every design states
    its cost in each regime, every stage is probed at both ends of the distribution, and a mechanism
    that wins one regime by losing another is a finding to rule on, not a win.

## CONSTITUTION addendum (Blake, 2026-08-28): NO FALLBACKS AT THE ATOM
A primitive may select DIFFERENT NATIVE INSTRUCTIONS per architecture; it may never select a wider-type
emulation on any target architecture. An operation an architecture genuinely lacks is a declared design
fact with its cost on the card, never an `#if` branch. (Origin: mul_bf16x2's fp32 emulation on sm_80
lived for months behind a passing oracle; fma.rn.bf16x2 was native all along.)

## §14 addendum (law, 2026-08-28): CONTAINERS AND THE GPU LOCK
`flock /tmp/rola_gpu.lock` is per-filesystem: a lock taken inside a container serializes nothing on the
host or in another container. Every container that touches the GPU bind-mounts the host's lock
directory so one flock governs all. One gate container at a time: stop the previous run before
launching the next (three concurrent in-container probes turned a minutes-long gate into an hour).

## §18 — Tools refuse, they do not choose (law, 2026-08-28, from Blake's "why keep two linkers")
A tool or build step that could run one of two ways — a linker present or not, a cache pinned or not,
a lint with or without its compdb, a lock taken or a slot — REFUSES with the command that fixes it; it
never silently takes the other path. Legitimate MODES (the §16 strict-manifest tiering; a comparison
arm kept as a yardstick) are keyed on an explicit declared variable and carry their exit condition on
a card. Applied: mold as the only linker; sccache pin miss refuses; run_clang_tidy never skips; one
build-lock discipline (slots as the parameter).

## §10 addendum (law, 2026-08-28): THE COMMENT-DENSITY GATE
The constitution's self-documenting-code clause is gated mechanically: `tools/lint/lint_standards.py`
measures comment lines / total lines per file under csrc/ and is RED above the mandate's threshold
(named constant; decl-block level, well under 15%). A `//:` prose block outside a file header is a
finding. A stage never raises a file's ratio; existing prose moves to the mirrored docs
(byte-identical SASS gate). Measured 2026-08-28 before that move: carry.cu 48%, ops.cuh 43%, box.cuh 41%,
carry_kernel.cuh 35%, decode.cu 32%, facts_kernel.cuh 32%.

## §14 addendum 2 (law, 2026-08-28, after Blake had to shut WSL down): ONE MACHINE, ONE COMPILE BUDGET
Three tooling agents each ran a full rebuild concurrently (MAX_JOBS=4 each -> 12+ cicc + 3 links) and
pegged the host at 100% CPU for 10+ minutes. The compile budget is MACHINE-WIDE and TOOL-ENFORCED, not
brief-enforced: setup.py (via the one build-lock discipline) takes a machine-wide slot before
compiling and REFUSES/waits when the slots are full — a brief cannot forget it. Until that lands: at
most TWO agents that build at once; every build goes through tools/build/build_lock.sh (since landed in setup.py and `rola_devtools.locks.host`, §23); an agent checks
`pgrep -c cicc` before launching and never wraps a full build in hyperfine (bounded `--runs 3`, on the
LINK step only). Coordinator error recorded: two briefs omitted the build lock.

## §18 addendum (law, 2026-08-29): `--no-verify` is never a precedent
A commit bypasses the hooks only with an explicit, per-commit coordinator authorization naming the
pre-existing hook failure and the stage that fixes it. "Another stage did it" is not authorization. Red
hooks are fixed (the hook, or the tree), never routed around.

## CONSTITUTION addendum (Blake, 2026-08-29): THE FOUNDATION LAWS
- R11 THE AXIS LAW: a compile-time axis is legitimate only as a deterministic function of user input the
  selector can pick; everything else is derived or a constant. Free axes = D and DV. MO and per-level B
  are runtime addressing; BC, K, M, kWarps, S are derived; W and C are constants.
- R12 SHAPE COMPILE-TIME, ADDRESSING RUNTIME: only the register shape (leaves-per-warp × DV) is compiled;
  every SMEM tenant is launch-sized from the state's descriptor; residency is asserted at the seam as a
  refusal; no runtime trip count in a register-shaped loop; never max-size SMEM.
- R13 THE STATE OWNS ITS FORMAT: a descriptor (D, per-level B_l powers of two ≥ 16, canonical order,
  the page rectangle = trailing 4 canonical bits, DV, split bf16 planes) bound at first call and CHECKED
  after; the warp sub-box's spans are compile-time from (D, DV) and never exceed a level's width under
  the floor (D=1 requires B ≥ 32); prefill and decode share ONE
  admission law; N is a multiple of 16 (ragged N refused, never padded); pages are [hi][lo][mass] bf16
  planes; readout operands are the hi plane in every kernel; a reading token moves hi only.
- R14 OWNERSHIP GRAINS NEST ON THE PAGE, nothing more: decode's unit is the atom; prefill's box owns whole
  rows; the kernel consumes activity bits at its own (warp-box) grain and the page table derives its own.
- R15 THE CTA IS THE SM (target): one CTA per SM with the warps the register file admits (8 × 256 regs),
  streams partition inside it; latency hiding is the kernel's job (covers, split-phase edges) — exposed
  latency is a coverage finding, never grounds to split the CTA; ONE CTA per SM is the goal, not a
  measured choice — a two-CTA launch is only a diagnostic of how far the body is from the target.
- R16 SHIP WITHOUT ARM REGISTRATION: a Python-package user never registers an arm; the shipped set is
  D × DV per arch; unknown descriptors are admitted by refusal-checked derivation, not by enumeration.
- R17 BF16 EVERYTHING: operands are bf16 (one form; float operand paths deleted); state accumulates fp32
  as split planes; truncation at the readout operand is the declared arithmetic.
- R18 ONE BODY PER OP FAMILY: the split (channel-carve) body is deleted; dense is the S = 1 point of the
  one body; a tied carve is a transit with the identity map, never a second body.
- R19 LAWS LAND WITH THEIR STAGE: a ruling that changes a law is written here in the same stage that
  implements it, never only on a card; the milestone carries the drift guards (lints, docs rev, skills).

## CONSTITUTION addendum (Blake, 2026-08-29): R9 ENFORCEMENT — A STRUCTURE BRANCH FAILS LOUD AT EVERY LAYER
A CTA-uniform branch on structure (e.g. a switch over admissible (span, shift) pairs) is legal only with:
(1) its cases GENERATED at compile time from the arm's shape, never hand-enumerated, with a static_assert
on the count; (2) a compile-time exhaustiveness assert (the derivation's possible outputs ⊆ the covered
set) so an arm that could receive an uncovered case does not build; (3) a launch-time refusal at the seam
naming the uncovered case; (4) a trapping default (`__trap()`, never a fallthrough) and a post-build
coverage sweep (every arm at every admissible case on a tiny cell).

## CONSTITUTION addendum (Blake, 2026-08-29): R13 — TWO CONTRACTS, ONE TRANSLATOR
The API accepts any shape (any b_l ≥ 2, any d_v, any L) and serves it by padding; the KERNEL accepts only
the descriptor's strict shape (powers of two ≥ 16, a shipped DV, N a multiple of 16, bf16 operands) and
REFUSES the rest — it never pads or masks. The seam is the only translator. Both boundaries are tested:
API tests prove padded runs bit-identical; kernel-boundary tests call the launch surface with unpadded
shapes and assert the refusal; a lint keeps logical-width fields out of every kernel entry signature.

## CONSTITUTION addendum (Blake, 2026-08-29): R9 ENFORCEMENT IS ONE PRIMITIVE
Every CTA-uniform structure branch in device code is a `uniform_switch<Set>(value, body)`
(`common/structure_switch.cuh`): the four layers — generated Set with a count assert, compile-time
exhaustiveness (derivation outputs ⊆ Set), the seam's launch refusal reading the same Set, a `__trap()`
default with the post-build coverage sweep — are the primitive's contract, never re-implemented per site.
A hand-written device `switch` on a geometry field is a lint finding.

## CONSTITUTION addendum (Blake, 2026-08-29): R20 — THE SM OWNS THE STATE; WARPS PARTITION IT; CTAs GROUP WARPS
On every architecture the unit of state ownership is the multiprocessor: its box is derived on its own
from the architecture's state capacity (`leaves_per_sm`, from DV) balanced over the D levels. Warps
partition that box — each side's level order carves it, sparse levels first, until every warp owns a
segment — and a CTA is only a grouping of warps holding a contiguous subset of those segments (how many
is a launch parameter). Where the accumulator lives (registers, tensor memory) changes how a warp touches
its segment, never who owns what. Every mechanism — the carve, the transit, the activity bits, the
launch shape, the schedule — is stated in these terms, and a design that couples ownership to the CTA
or to the accumulator's home is misfiled.

## CONSTITUTION amendment (Blake, 2026-08-29): R20 — THE OWNERSHIP UNIT IS THE CTA's BOX
R20 read "the SM owns the state; CTAs group warps". Corrected by the arithmetic: a CTA loads and
stores its box at entry and exit, so its read-side warps' union and its write-side warps' union must be
the SAME leaves; carving the SM's box first and handing a CTA half the segments gives the two sides
different halves at two CTAs per SM. Therefore the ownership unit is the CTA's box — carved per side
among the CTA's warps — and it equals the SM's box exactly when there is one CTA per SM (the goal).
The shape set is f(D, leaves_per_warp, kWarps); the architecture law (the SM's box) is asserted at
kWarps = warps_per_sm.

## CONSTITUTION addendum (Blake, 2026-08-30): SOURCE COMMENTS DESCRIBE THE CODE AS IT IS
A comment in csrc/ or rola/ states what the code does and why, in the present tense, in the library's
own terms. It never describes a past state of the kernel, a superseded mechanism, a stage or campaign
identifier, or a comparison to what was ("the previous pass replaced…", "the old fp32 rows…"). History,
justification and background live in docs/ (DELETIONS.md, ARCHITECTURE's status marks, the internals'
"why" sections) and in the journal outside the repo. The kernel library references only itself (§17);
the lint that refuses work codes in source is the mechanical half; reviewers refuse the rest.

## CONSTITUTION addendum (Blake, 2026-09-08): R21 — EVERY ASYNCHRONOUS OPERATION STATES TWO EDGES
An asynchronous write -- `cp.async` today; TMA, `cp.async.bulk`, `wgmma` on sm_90 and after -- has
two hazards, and a wait covers only one of them. Its COMPLETION edge (RAW) is the wait or barrier
phase behind which the first read of the new contents stands. Its ISSUE edge (WAR) is the barrier
after the last read of the OLD contents by ANY warp: a warp's wait orders that warp against its own
copies and never against another warp's reads of the bytes it is about to overwrite. Every shared
tenant an asynchronous operation writes states both edges in the ledger (`smem_ledger.md`), and a
review refuses a tenant that names only the wait. On sm_90 the two edges are a pipeline stage's
empty and full barriers; on sm_80 they are the CTA edges the frame already has, and the issue is
placed past the edge that ends the last read. Born of the box-words hazard of 2026-09-08 (journal
section 39): the issue sat before a slot's edge, the outer-level rows are shared between warps,
and the slower walks read the other side's words -- intermittently, invisibly at window 0 and on
dense draws, and under the one-ratio tolerance when it fired.

## R21 addendum (2026-09-08): THE SANITIZER LANE IS A GATE, AND ITS OOM IS A VERDICT
`compute-sanitizer --tool racecheck` models `cp.async`: on the box-words hazard it did not
report, it exhausted the host (23 GB in fifty seconds) and the target was killed -- the
"open K2 item" read as a host-budget limit was the race's own signature from the day the box
words landed. With the hazard gone the lane runs the one-CTA cells in seconds at under 1 GB.
So: the lane runs at every commit of the carry kernel, on `one-box-short` and
`one-box-multiwindow` at both grains through `tools/sanitize_cell.py`, and must report zero
hazards; a target killed under the sanitizer is a failing gate, never a resource problem to
be worked around. Benign races (every warp storing the same word) are not tolerated either:
a lane with known noise is a lane nobody reads. One warp writes, the readers stand behind an
edge.


## §19 — THE COMPONENT BUDGET GATE (law, 2026-09-09, from the P81 rebuild's four reference-grade forms)
Rule 12 said "state the floor before building"; nothing enforced it, and a whole kernel was built four
times to correctness with its schedule machinery at 3x master's instruction count before the floor was
ever compared. From here the floor is a GATE, per component, before composition:
- A COMPONENT is a device function with one job (the head, the operand ring's issue, the fold pair body,
  the readout tile, the re-layout, the sweeps). Each has a BUDGET LINE in the brief before code: its
  analytic floors (tensor: HMMAs x 32 cycles / schedulers; bandwidth: bytes / BW; latency: chain depth
  x latency at the stated prefetch distance), its instruction budget derived from the design's own
  operation count, and WHO EXECUTES IT HOW MANY TIMES (a per-window job is done once per CTA, by one
  warp or split across warps -- never replicated on every warp).
- Each component is built into the PART HARNESS (`measure/harness/bench_carry_parts.py`: one
  `__global__` driver per component over the real device functions, inputs shaped by a cell's liveness)
  and measured there -- cycles at the locked clock, instructions, the binding counter -- against its
  floors. It is accepted at <= 1.3x its binding floor and within its instruction budget; the numbers
  enter the regression store, and a later change that moves a ratio is red.
- The COMPOSED kernel carries a phase ledger (a debug build's per-warp `clock64` per phase) and is
  compared to the sum of its parts; the difference is the composition cost and is reported as such.
- The per-component instruction count of the shipped binary is attributed by `tools/region_ledger.py`
  (SourceCounters + lineinfo, functions and struct methods) against `tools/budgets/<kernel>.json`; a
  component over budget is red, the same as a spill.
- SKELETON FIRST (Blake, 2026-09-09): the kernel is composed from STUB components before any is
  implemented -- each stub has the real signature, the real shared-memory tenants and edges, and does
  the cheapest thing that keeps the composition correct in shape (a zero fold, a pass-through head) --
  so the final form, its ledger and its phase structure are fixed and compile before the first real
  component lands. Components are then implemented one at a time into the skeleton, each gated in the
  part harness and in the composed phase ledger before the next begins. A component is never written
  outside the form it will live in.
- A design verdict is never drawn from a build whose components have not passed their gates
  ("slop numbers get misread as design verdicts" is now a gate, not a memory).

## §20 — ptxas SIGNATURES (law, 2026-09-09): what the SASS gate reads, and the rule behind each
`tools/sass_gate.py` runs on every iteration build and fails on any of these; each was re-learned by
trial this week and costs a rebuild when missed.
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
- `@P HMMA` or a pair body whose non-tensor instruction count equals its `if`-guarded block's:
  IF-CONVERSION. A warp-uniform `if` (dealt work, a runtime flag) is predicated, not branched, and runs
  predicated-off on every other warp. Rule: a one-trip `for` (`#pragma unroll 1`) forces the branch;
  better, keep dealt or optional work out of pair bodies.
- BSSY/BSYNC per item in an unrolled loop where every item is live: the PREDICATED-CHAIN form; use a
  straight-line block when all items are live (ptxas schedules across items) and a jump table (BRX)
  when one is. Both, selected by a uniform test, never one for all.
- A warp's copy tasks mapped so its lanes touch different rows: one L2 request per copy. Rule: tasks
  CONTIGUOUS in memory across a warp's lanes; coalescing is checked by `l1tex__t_requests` per
  sector, not assumed.
- `smsp__inst_executed` / HMMA count above the arm's ceiling (this arm: 16 per HMMA per warp before
  issue binds the tensor pipe at two warps a scheduler): the kernel is issue-bound; the budget ledger
  says which component.

## §21 — THE MEASUREMENT PROTOCOL (law, 2026-09-09, after a dense verdict was drawn against the baseline's sparse schedule)
A comparison is valid only when: the same grid and block (read `launch__grid_size`), the baseline run
with ITS OWN schedule for the cell, `smsp__inst_executed`, `sm__cycles_active` and `sm__cycles_elapsed`
beside the time (idle SMs and tail waves are reported, not inferred), the GPU quiet (no sanitizer or
profiler child alive: a timed-out `compute-sanitizer` leaves its python child on the device), and the
probe's clock rows accepted. "Issue active at X%" is a ratio over active cycles and says nothing about
the instruction count; the count is read, never inferred from the ratio.

## §22 — THE MEASUREMENT SUITE, AND HOW A MEASUREMENT IS READ (law, 2026-09-12, after a week in which a register cap, a "spin", a pipe metric and a schedule default were each misattributed)
WHAT IS MEASURED, on every build, before any component is judged, as one report diffed against the
previous build's: (1) the SASS composition (§20) and the REGISTER LIFE RANGES (`nvdisasm
--print-life-ranges`): peak live registers per source region beside the allocation -- a budget
argument cites the peak live at the program point, never the allocation; (2) the scoped oracle; (3)
the phase ledger on the GATE CELLS (the N = L cells, §21's baseline rules) with per-warp rows, so an
imbalance is seen, not inferred; (4) the region ledger with per-line stalls; (5) tensor utilization as
the HMMA count times the measured cycles an HMMA holds the pipe, over scheduler cycles -- never a
profiler's pipe-active ratio, which is normalized to a rate this kernel does not run at; (6) the A/B
against the baseline RUN WITH ITS OWN SCHEDULE, named per binary (a spec without one ran the dense
order and recorded the reap); (7) THE PHASE CENSUS: the HMMAs a phase COUNTED from the profiler's
per-instruction dump (never the model's), utilization a phase against the phase ledger's cycles, a
warp's cycles at its HMMA instructions against its other work a unit (a fragment, a pair) -- the
number that says whether two warps a scheduler can cover each other (the other work must fit under
the partner's burst), and the instructions a component against its budget with the launch's
CTA-window divisor (a budget is per CTA-window; a per-launch total against it flagged everything
red and meant nothing); (8) THE WAVEFRONT CENSUS: shared-memory wavefronts above the profiler's
ideal, by source line -- on a shared store or load, bank conflicts on the store side of a layout,
which the load side's conflict-free `ldmatrix` hides from every other instrument (an 8-way conflict
sat in the snapshot for a week as "wider stores don't pay"); on an asynchronous copy (`LDGSTS`) the
count is the GLOBAL LINES the copy touches, not a bank conflict (calibrated 2026-09-18: a line a lane
reads 32 of ideal 4, the fill's sixteen token rows 16 of 4, whole lines 4 of 4, and the rows and the
lines forms copy at the same rate -- the sectors are what a gather pays); (9) THE COMPOSER (`tools/compose_ledger.py`): a phase's
time attributed to its PARTS by building compositions in which a part is real or a stub (the stub keeps
the part's HMMAs, same count, atom and accumulators, and drops its other work), read in wall time rung by
rung, each composition checked by its HMMA count against the kernel's and by its device instructions
against the kernel's hash. A per-part cost is the difference of two measured times. It exists because
warp-side instruments cannot say whether the pipe is idle, and the MMA-only composition is the phase's
measured floor; (10) THE CALIBRATIONS (`measure/harness/bench_carry_calib.py`, `docs/internals/carry/calibration.md`):
each operation the parts are made of (the kernel's HMMA atom, shared loads and stores, `ldmatrix` loads,
asynchronous copies, global reductions, CTA and shared-memory barriers, warp syncs) alone on every warp,
as cycles an operation a warp on this card -- reference rows every reading uses in place of a constant (the
HMMA's 32 cycles was one such constant for eight days); (11) THE PIPE TIMELINE (`tools/pipe_timeline.py`): the
tensor pipe's true utilization over a launch, measured on silicon by the profiler's PM sampling (1 us samples), on a
scale taken from the MMA-only composition's plateau -- the one instrument that says when the pipe idled rather
than how much in total; validated against the HMMA-count utilization at three points (66.8 / 21.0 / 33.9%
against 65-66 / 20-21 / 33.7%).
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

## §23 — THE DEV ENVIRONMENT IS ONE SYSTEM (law, 2026-09-12, from Blake's "one unified dev config ... everything reads that one file", "consolidate everything into one dev env init script, and also ensure docker stays maintained")
A tree that measures itself is only as reproducible as the machine it runs on. Before this law, 26 environment variables,
three per-machine files and fourteen hardcoded paths decided that machine, set by five scattered scripts and a page of
manual steps, and the dev container had drifted for two weeks unnoticed.
1. ONE LOCATION. Every value that depends on the machine is declared in `tools/dev_config.py` and read from the dev
   config directory (`~/.config/rola/`, one JSON file a section) through that loader. A tool never reads a setting from
   the process environment and never spells a machine path; the environment variables that remain are the loader's
   listed HANDOFFS (a parent process passing a resolved value to a child) and BUILD_PARAMETERS (what one build is).
   `tools/lint/lint_standards.py` refuses any other read and any CUDA or Windows path spelled in tools, benchmarks or
   setup.py. A replaced variable is deleted, never kept as a fallback (§18).
2. ONE INIT. `python tools/dev.py` is the only thing that sets up a host or a container (config, pinned tools, commit
   gate, clock, worktrees) and the only thing that checks one (`check`: every dependency a row, exit 1 on a red, the
   build's own gates reused). A new machine-dependent value lands as a schema key with its `init` detection and its
   `check` row in the same commit.
3. THE CONTAINER IS KEPT CURRENT BY A KEYED GATE. The dev image is a supported environment, not a document. It is
   built by `python tools/dev.py container build` from the environment's inputs (`ENV_INPUTS`: the Dockerfile, the
   devcontainer and compose files, `requirements.lock`, the tool pins, `tools/dev.py` and the modules it imports) and
   its last step is the same `init` a host runs, whose `check` fails the image build on a red. Their hash is the
   environment key. `python tools/dev.py container check` proves the image with the host's mounts (the environment
   check inside, the GPU and reference library, a host-held lock seen as held, an iteration build imported) and records
   the result under that key; a commit changing any input is refused by the commit gate unless a passing record exists
   for the key of what it stages. A host-side toolchain change that the image does not share (a driver, a toolkit
   path) is not an input and needs no container record; a pin, a lock or a Dockerfile line always is.

## §24 — TWO TIERS: DECIDE COARSELY, BURST STATICALLY (law, 2026-09-17, from the fold's and readout's forms)

The facts, each a calibration or trace row (`docs/internals/carry/calibration.md`, `carry_kernel.md#phase-trace`):

1. A warp issues in order and runs one or two HMMAs ahead of the tensor pipe. Loads issued ahead of a burst are
   free (four `ldmatrix` ahead of nine HMMAs: +0.3 a HMMA alone); a chain placed after a burst is exposed whole on a
   warp whose partner is not issuing (the fragment's gather: +18% alone, 0 paired).
2. With two warps a scheduler, every cycle either warp spends outside a burst costs the pipe half a cycle: a lone warp
   drives it at ~50%. The window equals the pipe work plus one warp's non-MMA work (nl64k-dense: 70.7K + 32.5K).
3. ptxas schedules within a basic block. A vote, a barrier poll, a data-dependent branch or a rolled loop's back-edge
   ends the block; under register pressure it sinks loads to their uses and regroups bursts. Registers are the
   pipelining budget: a set of operands a unit of depth.

The rule: a phase that issues MMAs is two tiers. The DECIDE tier is everything data-dependent -- which units, in what
order, a barrier's state -- run at a boundary, into a list and a count. The BURST tier is `Burst<Ops>::run`
(`common/burst.cuh`, `docs/internals/common/burst.md`): a counted loop over the list, two operand sets alternating,
each unit's gather ordered under the burst before it by a data dependency; no vote, wait or data branch inside it;
sparsity a shorter list, never a skip. A decision needed mid-way is two bursts with the decision between them. The
list is built by the lanes at once, never by a serial scan (the readout's sixteen-iteration list cost a tile 12%).

What it forbids, mechanically (`tools/lint/burst_tier.py`, a ratchet): an MMA issued by a function without
`//: @burst` in its declaration block (`//: @burst-exempt <reason>` names a debt the ratchet holds); inside a burst
function a vote, a warp sync, a shared-memory barrier, a rendezvous, a `while`, an `if` that is not `constexpr`, or
`#pragma unroll 1`. The primitive's own pair loop is the one loop a burst runs and is not in a burst function. What it
measures: the burst's lone-warp rate on the trace, and (when built) the static pipe floor per burst loop in the SASS
gate.

Precedent: CUTLASS's tile scheduler against its collective mainloop; FlashAttention-2's unrolled, branch-free KV loop
with masking by predication; ThunderKittens' producer and consumer warpgroups. On Hopper the tier split is the
hardware (`wgmma` async, TMA, mbarriers); on Ampere it is this rule.

## §23 addendum (law, 2026-09-18, from the research behind the burst: the coordinator's research journal, SASS guidelines)

1. THE EMITTED DISTANCE IS THE LATENCY BUDGET. ptxas sets an instruction's stall count to the producer's latency
   minus the instructions it placed between producer and first consumer (Huerta et al. 2025, §4, validated on
   GA102), so a dependence is hidden only by what sits between its ends in the SASS, never by anything in the
   source. The burst gate reads that accounting (`docs/internals/tools/burst_gate.md#coverage`).
2. ORDER BY DEPENDENCY, NEVER BY HOPE. There is no scheduling knob (nvcc §4.2.9.1) and no supported ordering; the
   one sanctioned device is a data dependency, `rola::burst::after`, in one header. No fence for ordering (it
   orders memory operations and costs runtime), no clock trick. The burst-tier lint refuses a fence in a burst.
3. LOADS FIRST, ALL OF THEM, THEN THE ARITHMETIC ON THEM. The canonical hand schedule spaces the loads and pushes
   the stores down (Gray, SGEMM: 98.5% useful cycles); the measured good attention kernel keeps about eight
   instructions between an `ldmatrix` and its consumer, the bad one one or two, and the cause was register
   pressure (lubits.ch, A100). The gate's exposed-latency reading is this rule's number.
4. REGISTERS ARE THE PIPELINING BUDGET, AND THE THRESHOLDS ARE NUMBERS. Allocation is in eights a thread against
   a 16,384-register partition: 248 a thread buys two warps a scheduler, 168 three (GA102 whitepaper; Jia et al.).
   A component states its live-register cost; the gate prints the arm's distance to the next threshold.
5. CONTROL BITS ARE THE COMPILER'S UNTIL THE GATE SAYS OTHERWISE. Patching yield and reuse bits gave Hopper 10%
   once and was retired when the compiler learned the interleave (DeepGEMM). Nothing below ptxas is patched
   unless the coverage reading shows the order right and only the bits wrong, and then as a ratified artifact
   with its patch script under the same gate.
Facts the laws rest on, with their rows: one warp can saturate this card, completion latency (~25) under the
issue interval (32, the GA102 whitepaper's arithmetic), so a lone warp at half rate is always the chain; a warp's
MMA throughput is flat to three in flight on A100 (Sun et al.) and one or two on GA102 by our calibration, the
only measurement there is; the sub-core instruction buffer is three deep, the memory queue four (Huerta et al.).

## §23 addendum 2 (law, 2026-09-18, from Blake's "either the simulator is wrong, or something is binding the kernel that shouldn't be. we need a way to find that out")

1. A MODEL READS AGAINST THE PROBE BEFORE IT READS THE KERNEL. `tools/pipe_sim.py --calibrate` simulates every
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
6. A PROBE'S PLACEMENT IS READ OFF ITS SASS. `asm volatile` orders nothing below ptxas: a chain written after a
   burst was interleaved among its HMMAs by ptxas all the same (the queue rows, 2026-09-18). What a row measures
   is the placement in its SASS, which the row's reading states; a row whose placement differs from the kernel's
   measures a different thing.

