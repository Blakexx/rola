# `read_value`

Mirrors `measure/lab/experiments/read_value.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION: what does the value-carve readout of a listed side cost a window against its floor, when the CTA's 8
warps walk the same digits in lockstep -- each warp its 8 channels of a 64-channel box over every leaf -- at alt-k4's
~1 tile a (digit, warp) a window? It is the one quantity `tools/design_floor.py` does not model: the readout's
latency (the design's build order, component 3).

THE FORM: one CTA an SM, 8 warps. A pass is a window: 8 digits (half a 16-digit box: sm_86's area) of 32 lists, each
`ENTRIES` entries at distinct slots of 128. Each digit: each warp's slice (32 lists x 8 channels, one
`ldmatrix.x4.trans`). Each tile of 16 entries (tokens on M): the rows from the swizzled pool (two `ldmatrix.x4`), two
m16n8k16 MMAs, the entries' coefficients on the accumulator, the f16 panel's load-add-store at each of the lane's two
tokens (no atomics; a warp's 16 bytes of a slot XORed by the slot's residue); the den's tile (two more MMAs on the den
image's slice) by warp `tile mod 8`, its fp32 word a token a warp. A barrier ends the pass. The PIPELINED form flattens
the window's tiles and loads each tile's slices, entries and rows while the one before it computes (two stages).

THE RECORD: `measure/lab/results/read_value/sm_86.json` (RTX 3080 Ti), cycles a window a CTA:

| entries a digit a window | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| lockstep, fp32 accumulate, the den's tile | 2,273 | 2,480 | 4,412 | 8,516 |
| pipelined a tile ahead | 1,987 | 2,283 | 4,538 | -- |
| pipelined, the k-steps' accumulators split | 2,067 | 2,330 | -- | -- |

At 16 entries: fp16 accumulate 2,286; no den 1,977 (pipelined 1,861) -- the den's tile is +25%, as priced.

WHAT IT DECIDES: shared memory binds the form on both cards (10.1 shared-pipe cycles an MMA at 16 entries -- the
rows each channel group gathers, its slice, the entries and the panel's load and store -- against the A100's 2 for the
tensor pipe), and the form runs ~1.6x that floor alone: its stalls are fixed-latency waits and shared-load scoreboards
in each tile's chain, the panel's read-add-write and the den's turn (splitting the MMA chain does not move it). So the
readout component's budget assumes composition hides that latency -- each scheduler's two warps in different phases
(the role-alternating form, KERNEL_STANDARDS §5) -- and its part harness measures the readout alone at this record's
cost; the merge width m (several digits a tile) is the in-phase lever to price there.
