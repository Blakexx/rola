# `tools/pipes.py` — the pipes a launch takes: one counter set, priced per card

KERNEL_STANDARDS §29.14. One set of ncu counters for any kernel's launch (`METRICS`, `STALLS`), read by the lab
(`tools/lab.py run`, every experiment kernel) and by the production capture ([`pipe_counters.py`](pipe_counters.md),
one carry launch), and priced the same way for both: each pipe's cycles a unit at `common/hardware.json`'s rates,
the largest binding.

<a id="price"></a>

## What it prices

Per card (the local product, sm_80's defaults), each pipe's cycles a unit an SM:
- TENSOR: a declared MMA's kind and K at the card's rate; undeclared, the m16n8k16s the tensor-pipe counter counted
  (`TENSOR_COUNTED`: 16 cycles a scheduler for one on sm_86 whatever the accumulate -- a count of the work, not its
  time: `read_image`'s `HMMA.16816.F32.BF16` reads 16.0 against its 32-cycle rate) at the card's rate for the one MMA
  kind the SASS issues. The profiler's tensor-pipe ratio is never read as utilization (§22).
- SHARED: the shared-memory pipe priced by kind at [hardware.md#shared](../common/hardware.md#shared)'s rates -- load
  wavefronts; stores and atomics each at the larger of their instructions' least and their wavefronts; `cp.async`
  copies and global loads by their bytes (a load from L2 into registers is written into the banks and read out).
- ISSUE, and the integer (`alu`), FMA and transcendental (`xu`) pipes: ncu's per-pipe instruction counts at each
  pipe's cycles an instruction a scheduler ([hardware.md#pipes](../common/hardware.md#pipes)).
- GLOBAL: the SM's path to L2, priced with every SM busy ([hardware.md#global](../common/hardware.md#global)) -- each
  128-byte line read from or written to L2 (`l2_line`, ~18 bytes a cycle an SM) and each global atomic request
  (`red_request`, ~5 cycles: the drain's floor).

On the card the launch ran on: tensor and shared priced as busy shares of the elapsed cycles, issue and the integer
and FMA pipes measured (ncu's busy counters), L2 sectors read and written, and each stall reason's share of the active
warps' cycles.

A form's floor is its binding pipe; a kernel well over it is held by what its stalls name, not by its instruction
count. The hand-off's lean form (2026-10-18) takes 2,248 cycles a window against an integer pipe of 1,310.

<a id="checked"></a>

## Checked against the probes

| instrument | the probe | priced or predicted | measured |
|---|---|---|---|
| the shared pipe, priced | `smem_rates`' 14 pipe-bound kernels (loads, `ldmatrix`, atomics, stores of 4/8/16 bytes, each global stream beside `ldmatrix`) | | 96-105% busy |
| the integer pipe at 2 cycles an instruction | `handoff`'s six forms, `smem_rates`' `pipe<0>`, `pipe<6>`, `share<0,1>` | 12-53% | the same to the percent |
| issue, FMA-heavy at 2 | the same | | the same to the percent |
| the tensor counter's count | `read_image` (declared kind) | 16 a scheduler an MMA | 16.0 |
| tensor priced | `read_image`'s readout (near tensor-bound) | | 85-97% busy (the profiler's ratio read 34-48%) |
| the global path, priced | `smem_rates`' L2 streams alone and beside `ldmatrix` (loads into registers, `.cg` and `.ca` copies) | | 84-99% busy |
| the global path's atomics | `drain`'s skipping form (its price's source) and the forms beside copies | | its drain phase ~85%, 93% with a copy every add |

Records: `measure/lab/results/smem_rates/sm_86.json` (2026-10-18, sm_86 locked); the hand-off's and the readout
lab's are the branch `ampere`'s.

<a id="limits"></a>

## Known limits

- Stores of 8 bytes a lane price 5-14% high (one store rate serves 8 and 16 bytes); a kernel mixing store widths is
  priced at the larger of its totals, a floor under its true cost.
- `cp.async.ca` beside `ldmatrix` prices its shared pipe at 34%: the global path binds it (88%), not the shared pipe.
- The global path's prices are sm_86's with every SM busy; sm_80's are announced the same until a rented A100 measures
  them (its L2 serves more bytes a cycle an SM, so its global path likely prices high there).
- ncu's elapsed cycles include the launch's ramp (~10% over a lab's own timing of a short kernel).
- A kernel whose run length depends on other warps' timing (`smem_rates`' role 3) reads different work in ncu's replay
  passes.
- `TENSOR_COUNTED` holds sm_86 only: on another architecture an undeclared MMA kernel prints no tensor line until its
  count is read there (a lab declaring its kind, `read_image`).
