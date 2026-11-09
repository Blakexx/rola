# `row_span`

Mirrors `measure/lab/experiments/row_span.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

How fast the facts pass's row stream reads DRAM when a work item owns a wider span of a level's digits. A level's
rows are `[T, H, 256]` u8 (a token's head's row contiguous, heads side by side); the facts pass's units own 32 digits,
one 32-byte piece of each token's row. The facts pass's ring (four 1 KB stages a warp, three ahead, two quanta a lane
a stage, eight lanes reading one token's run), items taken in order with the span innermost (as the pass deals its
units), every item 64 KB; read only. sm_86 (RTX 3080 Ti), bytes a chip cycle against a flat stream of the same bytes:

| span | 2 x 8 warps an SM | 1 x 16 | 1 x 8 |
|---|---|---|---|
| flat stream | 541.7 (1.00) | 539.8 (1.00) | 525.0 (1.00) |
| 32 digits (32 B a token) | 268.0 (0.50) | 243.0 (0.45) | |
| 64 digits | 436.3 (0.81) | 438.0 (0.81) | |
| 128 digits | 528.0 (0.98) | 524.3 (0.97) | 533.2 (1.02) |
| 256 digits (the whole row) | 538.9 (0.99) | 537.3 (1.00) | 524.7 (1.00) |

The flat stream is the card's DRAM peak (~900 GB/s). A 32-byte piece of a row is half a DRAM burst: the stream reads
at half rate unless the row's neighbouring piece is read by the same warp at the same time -- the facts pass's vote
items (four consecutive units of a level, their warps reading together) recover it, its 64-digit list items do not.
From 128 digits on the stream is at the peak at any block shape: neither the second block an SM nor sixteen warps are
needed to stream. SPANS APART (the span outermost in the item order, so a row's other spans are read a whole sweep
later, as a facts pass whose blocks drift apart reads them): 128 digits 527.8 (0.98), the same as innermost; 64 digits
319.8 (0.59) against 0.81 innermost -- a 64-byte run is a whole burst only when its row's neighbour is read about the
same time, a 128-byte run always. Record: `measure/lab/results/row_span/sm_86.json`.
