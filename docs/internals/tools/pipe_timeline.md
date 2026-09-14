# `tools/pipe_timeline.py` — the pipes over one launch, measured on silicon

KERNEL_STANDARDS §22 (11). Every other instrument in the suite aggregates over a launch, so none of them can
say when the tensor pipe sat idle or what else was happening then. This one reads the tensor pipe's
activity over time on the real card, through the profiler's PM sampling. Each gate cell gets a
timeline of true utilization and a chart.

    python tools/pipe_timeline.py --cell flagship-dense            # capture, then read with the stored scale
    python tools/pipe_timeline.py --cell flagship-dense --rep r.ncu-rep
    python tools/pipe_timeline.py --cell flagship-dense --calibrate  # on a ROLA_CARRY_PARTS=none build

**The capture.** It uses the Nsight Compute CLI 2025.3 or later, found at `/usr/local/cuda-13.0/bin/ncu`.
The 2024.1 CLI that ships with CUDA 12.4 has no PM sampling. The capture runs `--section PmSampling` over
one launch of the cell through the probe worker. The profiler replays the kernel over several passes
(five here). Each pass samples one single-pass counter group at a fixed interval, 1 µs on this card
(about 1,665 cycles), and the groups are aligned by timestamp. The interval flag does not go below 1 µs
here. NVIDIA's CUPTI sample program samples one single-pass group only. On this card no such group holds
a tensor-pipe metric, which is why the profiler's multi-pass section is used instead.

**The series.** They are read from `--page raw --print-metric-instances details --csv`. Each carries a
timestamp and a value per sample, averaged over the SMs, placed on the time axis of its pass group's
workload start:
- the tensor pipe's active cycles (`sm__pipe_tensor_cycles_active_realtime`);
- the ALU, XU and FMA pipes;
- warps active and SM cycles active;
- the shared LSU's wavefronts.

**The scale.** The tensor series is a percent of the profiler's modeled peak, not of this atom's
saturation. The scale comes from the MMA-only composition (every readout and fold part stubbed): its MMA
phases issue HMMAs as fast as the pipe takes them. Its full-occupancy busy samples form a flat plateau.
On 2026-09-12 at flagship-dense that plateau was 49.62% of the modeled peak (p10 48.93, p90 50.11, 336
samples), stored as `<store>/<hw-profile>/pipe-timeline-scale-<utc>.json` in the measurements store. A reading divided
by the plateau is the pipe's true utilization at that moment.

**Validation.** The same scale gives the utilization computed independently from HMMA counts at 32 cycles
an HMMA, at three points:

| capture | timeline | count-based |
|---|---|---|
| kernel, flagship-dense | 66.8% | 65-66% |
| kernel, nl64k-alt-k4 | 21.0% | 20-21% |
| MMA-only, nl64k-alt-k4 | 33.9% | 33.7% |

**What a flagship-dense launch shows** (653 µs):
- all 8 warps of every SM active for 490 µs, then 1.6 warps per SM for the last 160 µs (256 CTAs over 80
  SMs leaves a partial fourth wave of 16 CTAs);
- in each wave: the prologue and state sweep at 0% tensor with the ALU pipe busy;
- each window's readout jittery between about 60% and 93% true utilization;
- each window's fold flat near 80%;
- short dips at the head and at the window edge.

The MMA-only composition holds its plateau through every MMA phase.

**Limits.**
- The values are averaged over SMs, so the phase structure shows only because the CTAs of a wave run in
  step. There is no per-warp detail.
- Sparse cells' windows (about 8-13 µs at nl64k-alt-k4) are only a few samples long, so their phases blur.
  Folding the samples on the window period is the finer view, not yet built.
- The capture replays the kernel, so it costs several launches' time.

**Where it runs.** `tools/build_ledger.py` runs it as the `timeline` step for every gate cell.
`tools/compose_ledger.py --timeline` records every rung's true utilization, and a tensor series in 5 µs
bins. Outputs are `<stem>.json` (series and summary) and `<stem>.html` (the chart).
