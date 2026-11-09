# `tools/carve_model.py` — the carry's mechanics for any carve

KERNEL_STANDARDS §32.2: a schedule or composition change is predicted before it is built. This model predicts the
carry's work for a carve -- a box partitioned among warps, each phase's warp tile, streamed or resident rows, the
hand-off route -- on a card, over a real cell's lists, and ranks nothing until it reproduces the measured forms.

<a id="statistics"></a>
## The cells' statistics (`stats`)

Every box-window of a registry cell as the carry reads it: each side's union and each list's entries by the union's
64-slot segments, from the facts pass's prefix rows (`pre[l][j]` the boundaries of lists below `l`; a list's entries in
a segment are differences of differences, modulo 2^16), and the window's plan word from the plan pass's model
(`tests/oracle/carry_plan.py`). A dense side's lists are implied: every slot in every list.

<a id="events"></a>
## The events (`events`)

Today's carve's work a CTA-window from the plan words and the lists: readout tile-steps (a warp's digit pair `w + 8i`,
two at a time, as many tiles as the pair's longer run; a dense side a digit at a time over every slot), fold steps (a
list's 16-token steps while more than 8 entries remain, then an 8-token tail), images stored (a half counts a half),
chunks, rows filled, entries, panel slots drained, the window's steps. The MMAs these imply equal the measured MMAs by
shape on every cell measured (nl64k alt-k4, dense, g4; flagship alt-k4, alt-k16, cohort-k4, dense).

<a id="measured"></a>
## The measured side (`measured`)

One launch's `ncu --section SourceCounters` dump: each SASS line's executed instructions and shared wavefronts,
attributed to the innermost MMA loop holding it (or none) and its innermost carry function (the extension's line
information), a CTA-window's worth. Attributing by innermost function (the region ledger's rule, `owner_of`) keeps the fold's and the
readout's instructions apart where their code is the same.

<a id="prices"></a>
## The prices

Each function has one driving event (`DRIVERS`: a readout tile-step, a digit, a fold step, an image stored, a row, a
staged run, a chunk, a drained slot); the window's bookkeeping (`GLUE`: the plan's and the flow's reads, the staging,
the places, the instruments) is priced per step. A function in neither is refused (`unmapped`, over 0.5% of a launch's
instructions): unnamed, a renamed or new function priced as bookkeeping, silently -- on 2026-10-19 37% of a launch's
instructions, the rewritten hand-off's among them, and a stream build's fold 10x high. A price is a function's
instructions and wavefronts per event on a calibration cell: nl64k-alt-k4 for the listed side's code, nl64k-dense for
the dense side's (its readout and fold run other code paths). A staged list's entries (`entries_fill`) price by its
RUNS: each list's run in a chunk (its shuffles) and a warp trip a 32 entries, 20 instructions each on every listed cell
(cohort-k4 15) -- priced an entry they read 5.0 at alt-k4 and 2.0 at alt-k16.

<a id="validation"></a>
## Validation (2026-10-19, carry-k5 at 4ae60265, sm_86)

Instructions and wavefronts a CTA-window, predicted from the calibration cells' prices, against held-out cells:

| cell | instructions | wavefronts |
|---|---|---|
| flagship-alt-k4 | -4.8% | -2.0% |
| flagship-alt-k16 | -4.4% | -8.5% |
| flagship-dense | -0.5% | -0.2% |
| flagship-cohort-k4 | -1.9% | +17% |
| nl64k-alt-k4-g4 | -4.4% | -21% |

By phase: the readout within 2%, the drain 4%, the fold 5%, the fills 2% (cohort-k4 +10%), the hand-off 3% (alt-k16
-8%, g 4 -11%), the bookkeeping -8 to -18%. The g 4 wavefront miss is its prefix-row reads (bank conflicts in the grouped
prefix rows, a defect) and tables that grow with the query heads, which the events do not carry.

What it shows of today's carry at nl64k-alt-k4 (a CTA-window): 33.1K warp instructions, 74% outside the fold and the
readout (bookkeeping 9.3K, fills 6.7K, hand-off 4.9K, drain 3.4K).

<a id="cycles"></a>
## Cycles (`validate --ledgers DIR`)

A phase's cycles a CTA-window are its ISSUE floor -- its instructions over the four schedulers -- times its EFFICIENCY,
with its shared wavefronts at one a clock and its MMAs' pipe time over the schedulers beneath it (`cycles_of`); the
efficiency is the measured cycles over the issue floor on the calibration cell of its side (the phase ledger, clock
locked, the GPU quiet). Each warp's phase is a dependent chain, two warps a scheduler: measured, a phase runs at a
steady 5-9 cycles a warp instruction from alt-k4 to dense -- the fold at 7.1-9.3 on every cell, the tensor floor binding
only the dense fold -- so the schedulers issue a quarter to a third of the time. Stretching a phase's LARGEST floor
instead (the form until 2026-10-19) priced the MMA-bound fold of alt-k16 from the latency-bound fold of alt-k4: +43%
there, -21% at cohort-k4. Each model phase is ledger columns: the readout, the fold, the hand-off (maxima and store),
the fills (fill and landed: the wait for their copies), the drain, the bookkeeping (wait). A phase ends at a
rendezvous, so the warps other than its slowest IDLE through the slowest warp's extra work, booked in the ledger's wait:
the phase's cycles times the share of its work that differs between warps (its tile-steps or fold steps, not its
per-digit or per-list work) times its slowest warp's work over its mean warp's, less one.

A LEDGER IS THE BUILD'S: `ledger_of` refuses one whose `build_stamp` is not this binary's (the SourceCounters dumps are
keyed by the binary's mtime; until 2026-10-19 the ledgers were read against whatever binary the tree held). A GPU just
locked reads ~11% slow for its first launches (nl64k-alt-k4 25,460 cycles a warp-window against 22,657): the phase
ledger warms up for 500 ms (`--warmup-ms`).

Held-out window cycles (2026-10-19, clock locked, ledgers on this build): nl64k g 4 -0.9%, cohort-k4 +5.6%, flagship
alt-k4 -9.3%, alt-k16 -5.7%, flagship-dense -1.8%. By phase the fold -17% to +1%, the readout -4% to +14%, the fills
+1% to +14% (the flagship cells' smaller footprints: their scattered rows translate faster), the bookkeeping -51% to
+39% (the flagship cells run two windows a CTA, a call's first window's waits weighing on them).

<a id="forms"></a>
## Forms (`forms`), the window (`window`) and streamed rows

A FORCED FORM is planned by the plan model's `forced_plan` and priced the same way. Against measured launches at
nl64k-alt-k4 (ncu, 2026-10-17's planner, 5.473M cycles): the whole image with 128-slot read chunks predicted 1.155x,
measured 1.156x; halves with 128-slot read chunks 1.271x against 1.371x; halves with 128-slot fold chunks 1.121x
against 1.233x -- an extra chunk's cost read ~7-9% low.

THE WINDOW AT k x 512 TOKENS (`merged`): consecutive windows of a box merged exactly -- unions added, each list's
entries carried over (a segment's entries spread over its slots and re-binned at the merged union's 64-slot boundaries,
since the prefix rows hold no finer place) -- planned by the same rule in the area the larger window tables leave
(7.2 KB less at g 1).

STREAMED ROWS, a side at a time (`streamed`: the read ring's depth, the fold's ring): a streamed listed side's chunks
hold no rows or entries (a read chunk is its panel), every entry fetches its row into its warp's ring, and each warp's
entries arrive as its window's stage record; the read ring (576 bytes a stage, 8 rows and their entry words) sits beside
the image, the fold's takes the image's place (a window's read chunks all run before its fold) and its prologue's
tenant takes 2,176 bytes of the area (`FOLD_TENANT`); a dense side stays resident. A streamed side's cost beyond its MMA
phase, booked in `fills`: the FOLD's 1.05x its phase, from the built fold stream (2026-10-19, nl64k-alt-k4 at equal
area: fold +898, fills and landed -960 cycles a warp-window, against the 3,320 of fills the model removes; on the
earlier planner the same factor puts flagship-alt-k16-L16384 and -cohort-k16 at +3.0% and +1.8% against +0.2% and
+1.3% measured); the READOUT's with its drain 0.22x, the [`read_stream`](../../lab/read_stream.md) lab's 1.34x of its
bare phase rescaled to the model's (0.648) -- unvalidated: the fold's lab factor
([`row_stream`](../../lab/row_stream.md), 0.375 of the phase) was 2.8x short of its build (the lab kept its ring across
windows and copied half the slice), so `window` also prints the read stream at 2.8x its factor (`LAB_MISS`). A 4-stage
read ring adds 264 cycles a CTA-window.

The read stream at ring 4 against today, a CTA-window (2026-10-19, at the lab's factor / 2.8x it): flagship-alt-k16-L16384
0.81x / 0.94x (its read side one chunk, the halves stored once), nl64k-alt-k4-g4 0.89x / 1.05x, flagship-cohort-k16-L16384
0.94x / 1.07x, nl64k-alt-k4 0.92x / 1.02x (one chunk already: the fills alone). At ring 5 the ring's 4.6 KB splits the
read chunks again at g 4 and alt-k16 (0.98x and 0.92x at the lab's factor).

## The forms ranked (`rank`)

Every form at a cell -- resident, the read stream at each ring, the fold stream, both -- at W 512 and 1024, best first,
against today at the calibration cells' efficiencies and with every phase at its floor (the phases still in turn; their
overlap is the kernel's next work). On sm_86, a 512 tokens (2026-10-19, the read stream at the lab's factor):

| cell | resident W 1024 | read stream W 512 (ring 4) | fold stream W 512 | both W 512 (ring 4) | best at the floors |
|---|---|---|---|---|---|
| nl64k-alt-k4 | 0.79x | 0.92x | 1.00x | 0.92x | 0.28x (read stream, W 1024) |
| nl64k-alt-k4-g4 | 1.43x | 0.89x | 1.13x | 0.88x | 0.39x (read stream, W 512) |
| flagship-alt-k16 | 0.88x | 0.82x | 1.03x | 0.86x | 0.39x (read stream, W 512) |
| flagship-cohort-k4 | 0.76x | 0.96x | 0.99x | 0.95x | 0.32x (read stream, W 1024) |
| nl64k-dense | 0.97x | 1.00x | 1.00x | 1.00x | 0.64x |

The fold stream wins nowhere (its tenant's area costs g 4 a read chunk: 1.13x). Under the fewest chunks a resident
1,024-token window gains on three cells (it lost 1.11-1.53x under the two-in-flight rule), but not at g 4, whose ~1,700-
slot read union splits; W = 512 stands on the intra's cost, which doubles with W (measured 2026-10-11), and these carry
numbers are an input to its revisit. Only the leaf carve's forms are ranked: the value carve and the 64-channel box were
dropped on measured grounds (their fold shared-memory bound, a 16-channel tile's warps each reloading the shared
operand; the box's halved lists doubling the readout's partials), and pricing them here needs their readout and
hand-off, which no lab has measured.

<a id="cards"></a>
## The forms on each card, the global path in (`rank-cards`)

Every form at a cell at its floors on sm_86 and sm_80, a 512 tokens: `in turn` the phases' binding floors summed, the
global path a floor beside them; `overlapped` the largest of the window's totals -- issue, shared wavefronts, the
tensor pipe at the card's MMA rates, the SM's global path ([`pipes.md`](pipes.md): L2 lines and atomic requests at
`common/hardware.json`'s prices, every SM busy). A form's global path: a resident side its union's rows once and its
staged entries; a streamed listed side a row an entry (pads not copied) and its records (64 bytes an 8-entry read
stage, 96 a fold step); the drain a request a slot. A streamed side's producer (20 warp instructions a read stage, 27
a fold step: the staged labs) and every copy's write (0.5 shared-pipe cycles a 128 bytes) join the prices. sm_80's L2
price is announced from sm_86, so it is ranked at that and at half (`L2_BRACKET`).

At today's prices (2026-10-18; the carry 2-3x its instruction floor) issue binds every sparse form and the ranking
stands: nl64k-alt-k4 a CTA-window streamed W 512 7,424 against resident 8,674 on both cards, the global path 4,165
(430 lines, 216 atomics) and 3,527 (340 lines); flagship-alt-k16 streamed W 512 ring 4 global-bound at 14,187 (1,663
lines against the resident form's 751) against resident 19,633 (issue); flagship-cohort-k4 streamed W 1024 5,744
against resident 8,062; nl64k-dense tensor-bound on sm_86, issue on sm_80, alike in every form. The global path
binds where the stream refetches most -- a token's row once for each digit or list that reads it (alt-k16's read side
~2.7 a union token) -- and at the design floor (the staged labs' forms, the 16-byte hand-off, the drain's best form,
a 231-instruction warp-window of glue: at alt-k4 issue ~3.3K, shared ~3.9K) it is the streamed form's largest pipe
on both cards. Its levers: fewer bytes a union token -- groups of digits fetching a token's row once (the grouped
readout's unified form fetched ~30% fewer L2 lines at m 2 and 4 than at m 1), the stream or the union staged chosen by
the planner per side and window from exact counts -- and sm_80's own L2 rate.

OWED: the alternative carves' prices, sm_80's L2 rate (the rental's first measurement), (each step's primitives by its warp tile, checked against the readout and fold
labs), registers, sm_80's calibration (the A100), and labs for the prices it lacks.
