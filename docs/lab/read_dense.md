# `read_dense`

Mirrors `measure/lab/experiments/read_dense.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does the value-carve readout of a DENSE read side cost a warp-window alone -- production's
`dense_factors`, `dense_den` and `read_dense` ([`readout_dense.md`](../internals/carry/ampere/readout_dense.md)), nothing else
in the window -- against its tensor floor (fp32 accumulate: one accumulator a token over the box's 512 leaves)?

THE METHOD. One CTA an SM with the arm `[16, 2, 16, 2, 64, 1]`'s plan tenants (32 digits x K 16 x 64 channels); each of
64 windows a 128-slot read chunk of dense rows (every token's K rows and 32 digits' coefficients, small f16 values drawn
a window), the image in halves as sm_86's area runs a dense side; the timed span the factors, a rendezvous, the den,
then both halves' readouts with a rendezvous between. The draw is stored before two barriers (read_alone's lining-up).
THE FLOOR, a warp-window: a scheduler's two warps' MMAs -- a tile a digit (8 tiles x 32 digits a warp) and the den's (a
tile a block over the warps) -- at fp32 accumulate (32 cycles a scheduler on sm_86, 8 on sm_80).

THE MEASUREMENT (sm_86, cycles a warp-window, median of 5):

| form | whole | the factors | the den |
|---|---|---|---|
| first built (a warp a token, a lane a digit, the warp's max by REDUX) | 27,679 | 7,523 | 1,106 |
| four tokens a step, the exit words by predicated stores | 25,607 | 5,562 | 1,106 |
| every term by selects (ptxas had made branches of the ternaries: a convergence region before each max) | 24,023 | 4,049 | 1,107 |
| four lanes a token: `y = a 2^(e_d)`, `e_t = ceil_exp(max |y|)` by two shuffles | 21,578 | 1,444 | 1,108 |

Tensor floor 16,640; the fp32 panel 21,262; the span's empty control 2.

WHAT IT DECIDES. The nest is at its line (1.30x the whole, the nest alone ~1.15x); the factors' first forms were a
serial per-token chain behind a warp collective, the four-lane form a quarter of the work with no collective. The
den's 1.1K is a tile a warp's chain: a later trim. The component goes into the carry at this form.
