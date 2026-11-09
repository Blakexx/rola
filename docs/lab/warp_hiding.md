# `warp_hiding`

Mirrors `measure/lab/experiments/warp_hiding.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. The carry's window is latency-bound at two warps a scheduler (its register-resident state allows no
more): would more warps hide that latency, or does a pipe bind each step's instruction mix? Asked 2026-10-25 after the
fixed-work track left the window at ~18.4K cycles a CTA-window (alt-k4) against a physics floor near 4K, with the
per-phase stall census (fixed-latency and shared-memory waits 50-78% of every compute phase's samples, barriers ~8%
kernel-wide) saying where the warps wait but not whether a third and fourth warp would fill the gaps.

THE METHOD. Each of the carry's chains, rebuilt from the primitives with the production dependency structure on
synthetic shared rows, run by 4, 8 and 16 warps a CTA (1, 2 and 4 a scheduler; one CTA an SM by 80 KB of dynamic
shared memory), every warp the same loop of 2,000 steps:

- the READOUT tile pair (`read_chunk`'s `Pairs` body): two entries' shared loads, the units' bytes, four shuffles and
  one `ldmatrix` each, four fp16-accumulate MMAs into a fresh fragment, two exact pairs, four shared atomics a digit;
- the FOLD's 8-token step (`fold_chunk`'s `step8`): an entry, a shuffle and a packed multiply, a shuffle, two
  `ldmatrix.trans`, four packed multiplies, eight k8 fp32-accumulate MMAs into live accumulators, two lists
  alternating;
- the read FILL (`fill`): a token's shared load, a 64-bit address, two predicated 16-byte `cp.async` a slot, four
  slots a thread ahead, a wait every eight steps;
- the HAND-OFF (`post_keys` and `store_batches`): each list's digit maxima over its 32 fp32 accumulators a lane,
  the quad's reduce-scatter, the leaf byte stored, one shared atomic a lane, then every value scaled, packed to f16 and
  stored 16 bytes a row -- 4 lists a warp at 4 and 8 warps, 2 at 16 (the same box: 32 registers a list).

A warp's cycles a step is the chain under the other warps' load; the warps a scheduler over it is the scheduler's
throughput. Flat cycles a step across warp counts would mean the latency hides (the state's two warps are the bound);
cycles growing with the warps means a pipe binds (the mix is). sm_86 (RTX 3080 Ti), median of 7, every sample within
0.1% but one:

| chain | 1 warp a scheduler | 2 | 4 | 2 -> 4 throughput | the pipe (`lab.py run`'s report) |
|---|---|---|---|---|---|
| readout tile pair, cycles a step a warp | 351 | 463 | 875 | +7% | shared 79% busy at 2, 84% at 4 (mio throttle 30%) |
| fold step8 | 204 | 265 | 450 | +19% | tensor 96% at 2, 85% at 4 (math throttle 52 -> 69%) |
| read fill | 622 | 732 | 1,137 | +30% | global (LSU) 43% at 2, 58% at 4; long scoreboard 85% |
| hand-off, 4 / 4 / 2 lists a warp | 742 | 1,206 | 1,295 | the box 0.93x | ALU 75% busy and issue 70% at 2, 81% / 74% at 4 |

Throughput a scheduler in steps a 1K cycles: readout 2.9 / 4.3 / 4.6; fold 4.9 / 7.5 / 8.9; fill 1.6 / 2.7 / 3.5. The
hand-off's 16-warp form does half the work a warp, so its box (32 lists) takes 1,295 cycles against the 8-warp form's
1,206: more warps make it slower.

WHAT IT SETTLES.

1. More warps would not reach the floor. Three of the four chains saturate a pipe at two warps a scheduler: the
   readout the shared-memory pipe (its 8 shuffles, 8 atomics, 2 `ldmatrix` and 4 loads a pair-step are ~46 shared
   cycles a warp against 128 tensor cycles: shared-bound 1.44x at two warps), the fold the tensor pipe (1.03x its
   floor at two warps, as `fold_step` found), the hand-off the ALU pipe (FMNMX, LOP3, SHFL and F2FP at 16 lanes a
   cycle on this card). Only the fill, waiting on L2, gains a third (+30% at four warps). The regime ruling -- few
   warps, filled registers -- costs the carry nothing on these chains on sm_86.
2. The gap is not in the steps. The production window at alt-k4 runs its readout phase in 3.3K cycles for ~2.5
   pair-steps (~1.2K here), its fold in 3.1K for ~5 step8s (~0.7K here), its maxima and store in 3.3K for one hand-off
   (1.2K here), its fill in 4.1K for ~20 slot-steps (~1.8K here). Each phase runs 2-3x the steady-state cost of its
   own chain: in production a phase is a few iterations behind a head of setup (exponents, runs, votes, the state
   fragments' 16 `ldmatrix`) and a tail (a wait, a barrier), and never reaches the throughput the tight loop shows. At
   W = 512 and alt-k4 a window has too few steps a phase to fill its pipeline, eight phases a window.
3. What that leaves: a longer window (W = 1024 doubles the steps a phase behind the same head and tail: more than
   the 5-12% the fixed-cost count gave, nearer -15 to -20%), fewer phases (the hand-off's barriers are the state's
   partition across warps), and de-phasing the warps so the shared, tensor, ALU and LSU pipes run at once instead of
   in turn (every phase today uses one pipe while the others idle; the barriers hold every warp in the same phase).
   None of these is more warps, and none reaches the bytes floor: the readout's shared traffic and the hand-off's ALU
   work are per-window costs the bytes floor does not count.

Record: `measure/lab/results/warp_hiding/sm_86.json` (commit d1ad1aba's tree, 2026-10-25).
