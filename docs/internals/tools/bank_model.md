# The bank model

`tools/bank_model.py` gives the shared-memory wavefronts one warp instruction costs, given the byte address each lane
touches (KERNEL_STANDARDS §30). Shared memory has 32 banks of 4 bytes and serves 128 bytes per wavefront. An access of
`width` bytes (4, 8 or 16) is served in phases of `128 / width` lanes: all 32 lanes for 4 bytes, 16 for 8, 8 for 16.
Each phase costs as many wavefronts as the most distinct bank rows any single bank holds; lanes reading the same word
share it. An `ldmatrix` costs one phase per 8x8 matrix, with its 8 row addresses of 16 bytes each.

It runs where addresses are chosen. A producer that picks slots (the facts pass, or a lab's inputs standing in for it)
runs the model on what it emits, so a conflict shows up before any hardware run.
`measure/lab/experiments/read_image.inputs.py` writes `banks.json`: the wavefronts per MMA of a CTA-window's
gathers, adds, entry loads and state loads, for each step form.

## Checked against ncu

ncu's per-operation counters (`sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_{ldsm,ld,atom,st,ldgsts}`),
on `read_image`'s tokens-on-M kernel at alt-k4 in its earlier layout (2026-10-11):

| part | model | ncu |
|---|---|---|
| `ldmatrix` (gathers and state) | 3.65 | 3.65 |
| shared atomics (panel adds) | 1.91 | 1.91 |
| entry loads (8-byte, a half-warp a wavefront) | 0.25 | 0.28 |

The total counter (`l1tex__data_pipe_lsu_wavefronts_mem_shared`) runs about 4-6% above the sum of the per-operation
counters. That remainder goes with no named operation and grows with the shuffle count (about 0.6 per shuffle), so a
budget adds it rather than modeling it. An `LDGSTS` copy counts twice its minimum: 8 wavefronts per 512 bytes it
writes.
