# `smem_rates`

Mirrors `measure/lab/experiments/smem_rates.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What an SM's shared-memory pipe sustains and which global traffic takes its cycles, for the hardware configuration
(`common/hardware.json`): one CTA an SM, 8 or 16 warps, each issuing independent operations at addresses that change
every iteration (volatile asm; an address that repeats each iteration lets ptxas hoist a plain `ld.shared` out of the
loop -- the first build did, and timed a loop without its loads). The block's span over the wavefronts the SM issued,
the wavefronts confirmed by ncu (`l1tex__data_pipe_lsu_wavefronts_mem_shared`). sm_86, cycles a wavefront (8 warps;
16 the same):

| operation | wavefronts an instruction | cycles a wavefront | cycles an instruction |
|---|---|---|---|
| `ld.shared.v4` (LDS.128) | 4 | 1.01 | 4.0 |
| `ldmatrix.x4` (LDSM) | 4 | 1.00 | 4.0 |
| `red.shared.add.u32` (ATOMS.ADD), a lane a bank | 1 | 2.00 | 2.0 |
| the same, two lanes a bank | 2 | 1.00 | 2.0 |
| the same, four lanes a bank | 4 | 1.00 | 4.0 |
| `st.shared.u32`, a warp's 128 contiguous bytes | 1 | 2.00 | 2.0 |
| `st.shared.v2`, 256 | 2 | 1.00-1.06 | 2.0-2.1 |
| `st.shared.v4`, 512 | 4 | 1.14 | 4.6 |

Loads run at the documented bank rate, a wavefront a cycle. A shared atomic takes the larger of two cycles an
instruction and a cycle a wavefront: a two-way conflict between the warp's halves is free. A plain store does the
same: 4 bytes a lane costs 2 cycles for its one wavefront, 8 or 16 bytes a lane about a cycle a wavefront -- the
same bytes stored 8 or 16 to a lane take about half the pipe's cycles of 4 to a lane.

WHAT GLOBAL TRAFFIC TAKES FROM THE PIPE: 16 warps, 8 issuing `ldmatrix.x4` (alone 1.011 cycles a wavefront) while 8
stream global data, 512 bytes a warp-instruction, in one form -- from a 16 KB table (L1 hits) or a 4 MB one (L2 hits)
-- the stream's own rate beside them, and the pipe cycles it took: `1 - 1.011 / (ldmatrix's cycles a wavefront)` over
the stream's 128-byte units a cycle, a sample a repetition (recorded as `pipe cycles a 128 B: <stream>`, the rows
`common/hardware.json`'s `shared` prices cite), and what a line from L2 adds to a load over an L1 hit (the `.cg` row
less the L1 hits'). Bank reads and writes a 128 bytes streamed are ncu's (`l1tex__data_bank_reads`, `_writes`, the
stream alone; 32 a full wavefront).

| stream | alone, B/cycle | beside, B/cycle | ldmatrix beside, cycles/wf | pipe cycles/128 B | bank R/W per 128 B |
|---|---|---|---|---|---|
| `ld.global.nc.v4` (`__ldg`), L1 hits | 119.7 | 53.7 | 2.028 | 1.19 | 32 / 0 |
| `ld.global.nc.v4`, L2 hits (allocated in L1) | 20.7 | 16.3 | 1.300 | 1.74 | 32 / 25 |
| `ld.global.cg.v4`, L2 hits | 16.3 | 16.3 | 1.266 | 1.58 | 32 / 32 |
| `cp.async.cg` into shared memory, L2 hits | 18.0 | 17.9 | 1.087 | 0.50 | 0 / 32 |
| `cp.async.ca` into shared memory, L2 hits | 22.3 | 18.6 | 4.487 | 5.34 | 35 / 58 |

A line from L2 adds 0.39 pipe cycles a 128 bytes to a load over an L1 hit (its write into the banks). The
`cp.async.cg` stream alone, every SM streaming, takes 7.12 cycles a 128-byte line (~18 bytes a cycle an SM): the SM's
global path's price of a line in `common/hardware.json`.

THE PRICES CHECKED: [`tools/pipes.py`](../internals/tools/pipes.md) prices each kernel's shared-memory traffic by kind
at these rates; on this lab's pipe-bound kernels (every form above at 8 and 16 warps, and each stream beside
`ldmatrix`) the priced pipe reads 96-105% of the elapsed cycles, except `cp.async.ca` beside `ldmatrix` (34%: its
cost there is not additive) ([pipes.md#checked](../internals/tools/pipes.md#checked)).

What it settles:

- A load from L2 into registers passes through the shared-memory banks: its data is written into them and read out (32
  and 32 a 128 bytes), ~1.6 pipe cycles a 128 bytes -- as much as a `cp.async.cg` copy (0.5, its write) and the
  `ldmatrix` that reads it (1.0). Loading straight into MMA fragments saves no pipe cycles over staging; it saves the
  copier's instructions and the ring, and holds registers for the loads in flight.
- `cp.async.cg` is the cheapest way in: its write into shared memory takes half a pipe cycle a 128 bytes.
- `cp.async.ca` fills L1, reads the line back and writes shared memory, and holds the pipe beyond its bank passes: 5.3
  cycles a 128 bytes, the `ldmatrix` beside it 4.4x slower. Never where the pipe binds.

Records: `measure/lab/results/smem_rates/sm_86.json`, `sm_86.counters.json`.
