# `tools/warp_timeline.py` — one CTA's warps, window by window: real cycles, real instructions

KERNEL_STANDARDS §22 (12). The instrument that shows WHAT EACH WARP IS DOING WHEN, on the real binary. It is two
launches of the same binary joined:

- **the stamps** (`tools/phase_trace.py`'s `take_stamps`): an uninstrumented launch with the kernel's phase trace
  bound, lane 0 of every warp storing `clock64 << 8 | event` at each activity change. Exact cycles on the SM's counter;
  ~110 stamps a warp a window; measured free (the same binary bound against unbound, clock locked: +0.8% on
  `nl16k-dense`, -0.1% on `nl64k-dense`, +0.2% on `nl64k-alt-k4`, 2026-09-19).
- **the warp trace** (`tools/warp_trace.py` and the NVBit tool `tools/nvbit/warp_trace`): a second launch in a child
  process with NVIDIA's binary instrumentation injected (`CUDA_INJECTION64_PATH`; a preload loads the tool into a
  torch process and sees none of its launches). Every dynamic instruction of one CTA's warps -- offset, active mask,
  predicate ballot, opcode, logical warp, and with `--addrs` the 32 lane addresses of each memory operand -- goes
  through NVBit's channel to a file, with a SASS listing keyed by offset. The child also binds the stamps, so its
  instruction stream and its stamps are one run.

The instrumented launch is ~1000x slower, so its TIMING IS DISCARDED: polls spin a different number of times, waits
appear and disappear, a fill fires at a different poll. Its instruction CONTENT is exact (the HMMA count reproduces to
the instruction), and `warp_trace.match` gives each real interval the mix of the same activity in the traced run by
(window, event, ordinal): the k-th walk of a window is chunk k's, the k-th fragment event run k's, the k-th fill chunk
k + 2's, the k-th tile tile k's. An interval the real run had and the traced run did not (a wait, a tile another warp
took there) carries no mix and is drawn hatched.

    python tools/warp_timeline.py nl16k-dense --json timeline.json --html timeline.html [--also other.json] [--cta 0]

The JSON is the record the `warp-timeline` target stores: per warp, per window, every interval's event, cycles and
instruction counts by class (`warp_trace.KEYS`: instructions, HMMA, ldmatrix, copies, global loads and stores, shared
loads and stores, shuffles, barriers, mbarrier arrives, reductions, backward branches), and per window each
scheduler's PIPE-FED share and the cycles, events and instructions by activity. The pipe-fed share is an estimate: an
interval's HMMAs times the pipe's cycles (`calibration.md`) spread evenly over it, the scheduler's two warps summed,
capped at one -- it says where the pipe had nothing, not exactly when.

The page (`tools/timeline_page.py`): four schedulers, each its two warps and its pipe strip; a window selector; drag
to zoom; hover an interval for its cycles, instructions, HMMAs against the pipe floor (alone and paired), loads,
copies, barriers and spins; a table of the window by activity. Several cells on one page through `--also`.

## What it is read for, and the rules

An activity's cycles beside its instructions is the question "what did this warp do for these cycles" answered without
a model. The pipe strip is where both warps of a scheduler were outside their bursts at once. A tile that is 128 HMMAs
in 8,900 cycles is 1.07x the paired pipe time and pipe-bound; a drain that is 344 instructions and no HMMA for 2,500
cycles, four times a warp a window, is where the readout's pipe idles; an empty second fold run (88 instructions, no
HMMA, 500 cycles) is a one-fragment chunk's bare loop. The first page (2026-09-19) said all three on its first window.

Rules: content from the trace, cycles from the stamps, never timing from the trace; a stamp site is named by the event
the same run recorded (the clock reads are inlined and carry no line info; the clock's constructor mark is a read with
no record and is dropped); the page is read before any form is judged, and a change is read as two pages side by side.

## The tool and its pin

`tools/nvbit_pin.json` pins the NVBit release (version, asset, sha256); `warp_trace.nvbit_root` fetches it once into
the scratch and refuses a digest that is not the pin's. `warp_trace.tool_so` builds the tracer against it with the
toolkit's nvcc for the device's arch, keyed by the sources' and the pin's hash. NVBit's own `get_warpid()` is the
hardware warp slot, not the CTA's warp: the tracer records `threadIdx.x >> 5`. The raw records (tens of MB a cell for
one CTA) are deleted after matching unless `--keep-trace`; large traces belong on the E drive, not the WSL disk.

What the instrument cannot say: the timing INSIDE an interval. That is the replay model's (`pipe_sim.md`), which
replays a traced interval's stream through the calibrated machine and is validated interval by interval against these
stamps; and the profiler's stall census (`stall_census.md`), whose per-instruction samples join these intervals by
address. Accel-Sim was evaluated for the same purpose (2026-09-19) and does not model this card's synchronization: its
mbarrier model is Hopper's, and sm_86 compiles mbarrier init, arrive and test to plain shared-memory instructions.
