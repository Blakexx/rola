# `tools/replay.py` — the whole-CTA replay: one CTA's traced streams through the calibrated machine

KERNEL_STANDARDS §22 (12). The warp timeline says what each warp did and how long each activity took on the real run;
the replay says WHY, by issuing the same CTA's traced instruction streams through the calibrated machine model and
reading the model's own stamps against the real ones, interval by interval, and the model's delay reasons against the
census's stall reasons, activity by activity.

    python tools/warp_timeline.py nl16k-dense --addrs --keep-trace --json t.json   # the trace the replay reads
    python tools/replay.py nl16k-dense --census auto [--learn-only]

## The synchronization, learned from the trace

On sm_86 an mbarrier is a 64-bit shared word: `mbarrier.init`, `arrive` and `test_wait` compile to plain shared-memory
instructions (`STS.64`, `ATOMS.ARRIVE.64`, an `LDS` of the high half and one `LOP3` that tests bit 31), and only the
copy's arrive is its own opcode (`ARRIVES.LDGSTSBAR.64`). `learn` finds the protocol with no knowledge of the kernel:
- BARRIER WORDS are the addresses the arrives name (the tracer records every memory operand's lane addresses).
- POLLS are the shared loads of a barrier word. The tracer records, for the first predicate-writing instruction after
  every `LDS` that reads its value, the value it read (a 16-byte value record), and every record carries the warp's
  predicate registers, so a poll's OUTCOME is the predicate its compare wrote, read in the warp's next record.
- THE WORD IS HARDWARE STATE: its high half is the phase bit (bit 31, the completed phases' parity) and the pending
  arrivals as a negative count, so a barrier's COUNT is the pending count right after a flip (256, 8 and 32 for the
  pool's full and empty barriers and the rings, 2026-09-19), and a successful poll's needed COMPLETIONS are the largest
  number with the phase bit's parity whose lanes the arrivals issued by then cover.
- POLARITY per compare form: a SPIN is the same poll instruction repeated by one warp with no stamp between; its last
  poll completed and the rest did not (614 of 614 and 14,005 of 14,005 agree on `nl16k-dense`).

## The replay

Each warp's path is issued in order. In the default mode (`--order real`) the stream is cut at its clock reads into
activities, keyed by (window, event, ordinal), and issued in the REAL run's order: an activity's content does not depend
on timing (the k-th walk is chunk k's), its place does -- the traced run is ~1000x slower, so its fills fire at other
polls and it waits where the real run did not (253 of 256 warp-windows differ at `nl16k-dense`). A successful poll
licenses the activity after it, so its gate travels with that activity; a wait the real run had and the traced run did
not is issued as its own clock read. A spin collapses to its successful poll; a word ONE warp arrives on (a warp's own
ring) gates on that warp's own arrivals so far, whatever tiles it took.

The machine is `sass_control.MACHINES` (latencies from the dependent-chain rows) plus: scoreboards from the control bits
that COUNT their producers (the fill's copy and an `R2UR` share scoreboard 1, and the consumer waits for both), a memory
instruction reading its registers when the memory pipe takes it, the stall count as the fixed-latency wait, the tensor
pipe per scheduler (`HMMA_PIPE`), one memory pipe per SM costed from the census's measured wavefronts per execution (a
shared access) or `COPY_BASE + COPY_LINE` a distinct global line the live lanes read (an asynchronous copy), a copy
landing `COPY_LATENCY` (the L2 round-trip row) after it issues, a copy's arrive counting when the warp's prior copies
land, a taken branch's target issuing `TAKEN_BRANCH` after it and holding the scheduler's branch path `BRANCH_PORT`, a global reduction's issue cost (12) holding the issuing warp, not the scheduler's port (a drain's partner is blocked
2% of the time in the census; on the port the drain read 15% long), a CTA barrier releasing at its last arrival, `BAR.SYNC.DEFER_BLOCKING` blocking at the first convergence, control or
memory instruction after it (every barrier site's census wait sits on the first `BSSY`; the kernel's `edges` stamp
after the fold's rendezvous is read BEFORE the barrier releases, so the timeline's `snapshot` holds the fold-end
wait), and the tensor pipe's arbitration: an HMMA tie goes to the warp whose HMMA the pipe took last, any other tie to
the warp that issued least recently (calibration.md, the arbitration rows: a pair settles one burst apart; a fair rule
kept the replay's pairs in step, 87% overlapped in the fold against 74%, and a last-issuer rule starved the partner).

`--json out.json` writes the replay in the timeline's record format; the page (`timeline_page.render` of the real
record and the replay's) draws the replay beneath each warp under "beneath each warp", same window, same scale.

`--per-instruction N` prints the N instructions whose modelled cycles a warp differ most from the census's (a sample is
a warp-cycle waiting at an instruction, which is what the replay books at each instruction), with both sides' reasons:
the instrument that found the uniform move's latency, the counting scoreboards and the taken branches.

## Validation, 2026-09-19

THE CORE AGAINST THE CALIBRATION ROWS: the replay's event loop on every probe loop the loop simulator knows, 35 rows:
27 within 3%. The misses are named: floating-point work beside a partner's HMMAs (`hmma_alu_*_2w` -8 and -16%,
`hmma_queue_40_2w` -6%: on the hardware one warp's arithmetic stretch is NOT hidden by its partner's HMMAs, the model
hides it), divergent reductions (`hmma_reduce_div_*` -45 and -71%: a reduction's sectors are not yet costed from its
lane addresses), and a lone warp's matrix loads (+10%).

THE KERNEL, real order, one CTA, pipe-owner arbitration, reductions on the warp:

| cell | whole run, replay / real | windows | within 10% | outside |
|---|---|---|---|---|
| `nl16k-dense` | 0.99 | -2 to -4% after the first | fragment 1.04, drain 1.01, tile 0.95, walk 1.02, head words 1.07, head scans 0.92 | snapshot 1.22, fill 0.88, head 0.82, issue 0.82 |
| `nl16k-alt-k4` | 0.98 | -9 to 0% | fragment 0.99, drain 1.08, walk 1.05, snapshot 1.05, head scans 0.92 | tile 0.89, fill 0.90, head words 1.20, head 0.82, issue 0.76 |

THE PAIR: a fold run's partner is in an HMMA activity 78% of the run in the replay against 74% on the hardware (dense),
57% against 56% (sparse), and the fold's partners start each walk a median 1,181 cycles apart against 1,172. What
remains to model, each from a row: work beside a partner's HMMAs slowing the tensor pipe (the floating-point rows
above, and `hmma_reduce_8`: eight reductions per nine HMMAs 76.4 against the model's 65.6), a reduction's sectors, the instruction cache (the census's `no_inst` at branch targets), and the 18-HMMA arbitration row's
drift, which the pipe-owner rule does not reproduce.
