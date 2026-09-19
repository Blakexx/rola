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

Each warp's traced path is issued in order: a spin collapses to its successful poll, which issues once the word's
completions reach what that poll observed; a test that failed stays as traced. The machine is `sass_control.MACHINES`
plus: scoreboards from the control bits, the stall count as the fixed-latency wait, the tensor pipe per scheduler
(`HMMA_PIPE`), one memory pipe per SM costed from the census's measured wavefronts per execution (a shared access) or
`COPY_BASE + COPY_LINE` a distinct global line the live lanes read (an asynchronous copy: calibration.md's
sixteen-a-group copy rows, so a dead lane's zero fill is free), a copy landing `COPY_LATENCY` after it issues (a
choice, not yet a probe row), a copy's arrive counting when the warp's prior copies land, a CTA barrier releasing at
its last arrival, `BAR.SYNC.DEFER_BLOCKING` blocking one instruction late (the census samples its wait there: the
kernel's `edges` stamp after the fold's rendezvous is taken BEFORE the barrier releases, so the timeline's `snapshot`
holds the fold-end wait), and the earliest ready warp issuing with ties to the warp that issued least recently (the
first form always favoured the lower warp and put warps 4-7 11.6K behind at the head's barrier).

## Validation, 2026-09-19

| cell | whole run, replay / real | windows | within 20% | outside |
|---|---|---|---|---|
| `nl16k-dense` | 0.99 | within 0.5% after the first | tile 0.94, drain 0.94, fill 0.84, head words 0.85 | fragment 1.20, walk 0.81, head 0.71, issue 0.76, snapshot 0.44 |
| `nl16k-alt-k4` | 0.93 | 4-12% fast | tile 0.92, drain 1.00, head words 0.93, snapshot 0.80 | fragment 1.14, walk 0.83, fill 0.82, head 0.71, issue 0.79 |

What the model lacks, named by the census beside it: BRANCH RESOLUTION and INSTRUCTION FETCH (`branch_resolving`
4-13% and `no_inst` 2-6% of the decide tier's samples: the walk, the head, the issue and the fill are 15-30% fast in
the model on both cells), and the pair's overlap in the fold (the fragment is 14-20% slow: the model puts both warps
of a scheduler in their fragments at once more than the real run does). Each is closed by a probe row, not a fitted
constant, and a closing is read off both cells.
