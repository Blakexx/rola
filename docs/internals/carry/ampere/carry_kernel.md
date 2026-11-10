# `carry/ampere/carry_kernel.cuh` — the carry kernel

Mirrors `csrc/rola/src/carry/ampere/carry_kernel.cuh`: the window loop over the components (`parts.md`, `fold.md`) under each
window's plan record (`schedule.md`) for one arm's plan (`plan.md`). The carry is one planned pipeline (ruled
2026-10-14): shared memory holds what is active and what lands next, sized by each window's actual counts, the planner
choosing chunk sizes the card's area holds. Its window runs in ONE FIXED ORDER (2026-10-25, from the window lab's
form 5, its record `measure/lab/results/carry_window`): the step machine that ran the record word by word, re-reading the window's
statics at each use, cost ~6.3K cycles a CTA-window at either density -- its per-step glue, its barriers and waits --
and the same parts in a fixed order recovered it (0.73x production on alt-k4 and its 1/16-thinned twin).

THE STATE WORK IS A FIXED WINDOW TERM. The hand-off's image stores, the digit maxima and their barriers run on every
list every window: a window's union touches nearly every list at any density short of extreme (Blake, 2026-10-14:
"every list is going to be touched at least once unless sparsity is extreme"), so sizing them by the lists touched buys
nothing. They are made cheap, hidden under the neighbouring phase, and measured exposed on the sparsest cell, where
their share caps the `N = L` ratio.

<a id="kernel"></a>
## The kernel

One CTA a box and batch and write head (its query heads' reads with it, `../carry.md#sides`), 8 warps, one CTA an SM,
after the plan pass (`schedule.md#pass`). The fold's fp32 state
(`kStateRegs` = 128 registers a lane) is the C fragments of the fold's MMAs and stays in registers for the whole call.
Every part in the window loop takes the warp and lane opaque (`wid`, `lid`: `ops::opaque` of the thread index), so
each forms its lane addresses where it runs; held from before the loop, every phase's were live through every other
(measured 2026-10-16 at nl64k-alt-k4, arm 0 on sm_86: the readout's peak 251 -> 239 live, ptxas's allocation 255 ->
246, every other part 13-42 fewer; one launch 0.9-2.8% faster on the three nl64k cells, `num` bit-identical).
Window 0's inputs and record are staged before the loop (one copy group, waited). A window, its plan record read once
(`stream_of`: the mode, the chunks, the counts) and each chunk's place from its step word (`schedule.md#steps`):

1. **The maxima's keys**: each warp's digit maxima posted (`handoff.md#maxima`) and each digit's largest den
   (`handoff.md#den`), before any copy of the window is issued -- the keys' shared loads and shuffles would otherwise
   queue behind the copies in the shared pipe (measured 2026-10-25 at nl64k-alt-k4: the maxima 1,952 -> 1,708 cycles a
   warp-window, the call 2.9% faster; issued after the keys' barrier instead, the copies land late and the dense cells
   run 13-15% slower).
2. **The first read chunk's fill**, issued into its place (one copy group) with its runs worked out first
   (`parts.md#entries`: a chunk's runs once, for its fill and its readout), then **the next window's inputs
   staged** (a second group): both land under the keys' rendezvous, the exponents and the first store. Then the
   rendezvous, each digit's exponent recorded by every warp (the same values) for its stores and the readout, and the
   den's factors and image (`handoff.md#den`).
3. **The read chunks**, each in order: its fill where it is not the first (issued after the chunk before is drained,
   into its own place; where that place takes the panel still being drained -- the step word's `reuse` -- the chunk
   before ends with a rendezvous after its drain, predicated: `../common/atoms/barrier.md#rendezvous-if`); its image -- whole, stored
   once before the first chunk, or in halves, each chunk storing both
   -- its copies landed (`cp.async.wait_group`: every group but the staging's, which may still fly), a rendezvous,
   its readout over the image (`readout.md`, at the call's reduce dtype and the card's accumulate form; a dense read
   side's `readout_dense.md`: its factors where a listed side's entry factors are, its den once a chunk, fp32
   accumulate; a rendezvous before the second half's store), then a rendezvous and its panel drained to `num` and
   `den` (`panel.md#drain`).
4. **The fold chunks**, each in order: a rendezvous where the chunk before's rows or the drained panel may be taken
   (every chunk after the first; the first where its step word's `reuse` says, predicated: `../common/atoms/barrier.md#rendezvous-if`),
   its fill, every group landed, a rendezvous, the window's units stepped at the first (`fold.md#units`), the fold
   (`fold.md#chunk`); a window folding nothing steps its units at its end.

Where a window has one chunk a side -- every alt-k4 window -- this is the lab's form 5 with its fold's rendezvous
where the plan says the fill takes the drained panel. Where it has several, the chunks run in sequence at the plan's
places; of the record's other bits the carry reads `reuse` (a fill on the panel being drained waits for the drain),
not filled ahead or next (`schedule.md#steps`).

The work is an exclusive chain over the entry's side and the image mode (R9's case marker, a trapping last arm), not
`uniform_switch`: the primitive's independent tests made a body 14% slower at nl64k-alt-k4 (31.7K against 27.9K
cycles a warp-window, 2026-10-14). A warp moves on as soon as its own work is done: a FOLD needs only the warp's own
stores (its registers are the state), so warps flow from their reads into their folds; only a rendezvous waits for the
other warps.

THE REGISTERS (255 a thread, the state 128 of them): what lives across a window is the window's stream (its mode,
chunks and counts, eight integers) and the first chunk's entry, runs and place; a side's statics (volatile) and the offsets
formed from the window (`parts.md#window`: its first token and parity opaque) are read at each use, so a
window-invariant address is formed where it is used rather than hoisted and held across the readout, where the peak
is.

<a id="staging"></a>
## The staging

Every async copy lands on a COMMIT GROUP (`cp.async.commit_group` / `cp.async.wait_group`, `ops::stage_commit`,
`ops::stage_wait<N>`), the carry's groups in a fixed order a window: the first read chunk's, the next window's inputs'
(`stage_window`: both sides' lists, the facts and the window's plan record for the CTA's box class, into the other
parity), then a group a further read chunk and a group a fold chunk. A read chunk's landing waits for every group but
the staging's where that was committed after it (`wait_group 1`), else for every group; a fold chunk's for every
group. The window lab measured the groups' form (its record `measure/lab/results/carry_window`); mbarriers, which the step machine landed
its slots on (an arrive fences the warp's outstanding atomics, the waiters poll; ~2x a hardware barrier a round,
the lab image_ring), are gone from the carry.

<a id="barriers"></a>
## The barriers

Every CTA-wide rendezvous is the hardware barrier (`ops::rendezvous`): every warp runs the same order, so a barrier
where the consumer starts means every warp is done with what came before. A window's: at its top (every warp done
folding the window before, whose rows may be where the first chunk lands); after the maxima (the keys posted); after each read chunk's copies land (its rows and entries visible, its panel zeroed, the first half stored);
before the second half's store (the first half read out); after the second half's store; after the chunk's tiles
(its panel complete, drained next); after a read chunk's drain where the next read's fill takes its panel's bytes,
and before the first fold chunk's fill where it does (`reuse`); before every fold chunk's after the first (the chunk
before's rows free); after a fold chunk's copies land. A warp moves on as soon as its own work is done: a FOLD needs
only the warp's own stores
(its registers are the state), so warps flow from their reads into their folds; only a rendezvous waits for the
other warps.

<a id="budget"></a>
## The budget

`tools/budgets/carry.json` (KERNEL_STANDARDS §29): each component's instructions a CTA-window on the gating cell
nl64k-alt-k4, its price a trip times its trips, the trips at each warp's maximum from the facts pass's lists and the
plans' model (`tests/oracle/carry_plan.py`). A warp-window there: a union of ~216 slots and ~256 entries a side, the
image in halves, one read chunk and one fold chunk; 7 readout tiles over 4 digits, 4 eight-token fold steps over 4
lists.

| component | a trip's price (sm_86 SASS) | trips a warp-window | design a warp-window |
|---|---|---|---|
| readout (`read_chunk`) | the value carve's (the design-floor model): a tile 23 (16 entries a warp: rows, 2 MMAs, the panel's adds), a live digit 8, the den's tile 9 | every digit a warp, the token group's entries in 16s | 1,045 |
| fold (`fold_chunk`) | an 8-token step ~30 (8 MMAs, 3 loads, 4 shuffles, 4 HMUL2); a list ~12 and its rescale 32 FMUL | 4 steps, 4 lists | 296 |
| units (`fold_units`) | once a window | 1 | 30 |
| stores (`store_batches`) | a batch: 4 lists x (a scale 4, 8 FMUL, 4 packs, one 16-byte store) + 2; a call ~20 | 4 batches, 2 calls | 318 |
| maxima (`post_keys`) | a (list, digit row): 7 FMNMX with abs, its key ~4; 16 of them, the quad's joins 16, the units and top ~18, one atomic | 1 | 217 |
| exponents (`post_exps`) | the lane's key to its exponent ~4, the two parities' addresses ~5, the reset, the record, the warp's sync | 1 | 14 |
| drain (`drain`) | the value carve's (the design-floor model): a full line of `num` a request, a den request a 16 tokens | the chunk's slots | 63 |
| read fill (`read_fill`) | a 16-byte copy ~13; an entry run ~10 | 6.9 copies a thread, 4 runs | 130 |
| fold fill (`fold_fill`) | the same | 6.75 copies a thread, 4 runs | 128 |
| staging (`stage_window`) | its ~240 copies over 256 threads, and its tests | 1 | 40 |
| barriers (the flow's edges) | an arrival ~3, a satisfied wait ~3 | ~27 | 81 |
| glue (the kernel's body) | the plan's fields and the windows' statics read once, loop control | 1 | 150 |

The design is ~2,540 instructions a warp-window (`tools/budgets/carry.json`: 20,328 a CTA-window), ~5.1K cycles of issue
a window over 8 warps on 4 schedulers; the value carve's readout and drain are priced by the design-floor model, the
other components by their 2026-10-14 trip prices. Measured (2026-10-14, carry-k5 839dadb5) for those components: the
fills 2.3-3.0x their lines (each copy re-derives its token, its source and its swizzle; each run its list's row), the
glue 2.6x and the barriers' own instructions 2.8x (fields re-read at each use), the fold 1.4x, the maxima 1.3x, the
stores 1.09x -- each class (b), the mechanism right and its form above its floor.

<a id="check"></a>
## The check

The state's fp32 sum, the skeleton's `check` (`../carry_api.md#carry-forward`): each lane's sum of its state, the warp's
lanes' joined by five shuffles, one atomic add a warp. One add a thread put 32,768 adds a call on one address; they
serialized at its L2 slice for ~47 us a call on sm_86 (the window lab's record, `measure/lab/results/carry_window`),
most of the call's fixed cost.

<a id="phases"></a>
## The phase ledger's phases

`CarryPhase` is the ledger's column order: `landed` (an entry's copies waited), `maxima`, `store`, `readout` (with a
read chunk's entry factors, `readout.md#factors`), `drain`,
`fill` (a fill's copies issued), `fold` (with the fold's units), `wait` (a step's rendezvous and the staging, every
warp's). A lap books the cycles since the last into its phase, so the columns sum to the warp's whole run.
`rola.ops.carry.LEDGER_PHASES` names the same order.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` (`docs/internals/common/phase_clock.md`) keeps each warp's accumulators in shared memory, lane 0 laps,
and `publish` adds them into the bound ledger `[ctas][warps][phases]` at the end. Unbound, the clock is gated off
and its laps issue nothing.
