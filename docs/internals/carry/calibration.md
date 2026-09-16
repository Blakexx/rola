# The calibrations: what one operation costs on this card

`measure/harness/bench_carry_calib.py` runs the kernels in `measure/harness/carry_parts/carry_calib.cuh`
(built into the part harness's module, `ROLA_BUILD_PARTS=1`). Each kernel isolates one cost the
carry kernel's parts are made of and runs it back to back on every warp of 80 CTAs, one per SM. The
host times it under the locked clock. A row is the median launch's seconds, times the clock's cycles
per second, divided by the operations a warp issued: **cycles an operation a warp**. The rows are
reference data for this card (`<store>/<hw-profile>/calibration-<utc>.json` in the measurements store). The composer's and
the tracer's readings use them in place of assumed constants (KERNEL_STANDARDS §22 (10)).

## RTX 3080 Ti (sm_86), 2026-09-12, 1.665 GHz locked

| calibration | cycles an op a warp | reading |
|---|---|---|
| HMMA, the kernel's atom (`aligned m16n8k16 row.col`, bf16 A and B, fp32 accumulate), two warps a scheduler | 65.0 | 32.5 a scheduler: the pipe's rate, shared by the scheduler's warps |
| the same HMMA, one warp a scheduler | 32.8 | one warp saturates its scheduler |
| scalar shared loads, bursts of 4 / 16 / 64 | 6.9 / 3.0 / 2.3 | cheap, and a burst's loads overlap |
| shared stores, bursts of 4 / 16 / 64 | 17.4 / 16.3 / 16.3 | serialized at ~16 cycles |
| two-n-tile `ldmatrix.trans` loads, bursts of 4 / 16 | 32.7 / 32.5 | serialized at ~32 cycles: as costly as the HMMA it feeds |
| a global f32 reduction (a 1.3 MB output) | 85.6 | serialized at ~86 cycles |
| a CTA barrier | 50.8 | |
| a shared-memory barrier, arrive and wait | 123.9 | |
| a warp sync between one lane's store and every lane's load | 57.5 | |

**Not yet valid:** the asynchronous-copy rows (347.3 bank-free, 92.7 "aliased" a 16-byte run). The aliased
variant writes every lane's run to the same destination, so it measures overwriting, not bank conflicts.
The time also includes each group's landed wait. The kernel is being redesigned: distinct destinations
sharing one bank group, group sizes swept, and the wait timed apart.

## What the rows already explain

- **The readout's drain.** At flagship-dense a warp issues 136 global reductions a window. At ~86 cycles
  each that is ~11.6K cycles, against the drain's composer step of 9.8K. The drain is reduction-bound:
  the lever is the reduction count and width (the card's open "bf16 partial outputs" question), not the
  drain's placement.
- **The two loads parts.** An `ldmatrix` load costs one HMMA's pipe time and the loads of a burst do not
  overlap each other. The readout's box issues 4 per 9 HMMAs and the fold's fragment 6 per 18, which is
  why both parts step the short scoreboard and push their HMMAs back.
- **The HMMA floor.** The kernel's own atom confirms the 32-cycle constant: 32.5 a scheduler.

The reduction row uses a small output. The kernel's output at N = L alt-k4 is 16.7 MB, so a large-output
row is needed before the drain's number is read at scale.
