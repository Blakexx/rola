# `read_alone`

Mirrors `measure/lab/experiments/read_alone.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does the value-carve readout cost a warp-window alone -- production's `read_factors` and
`read_chunk` ([`readout.md`](../internals/carry/ampere/readout.md)), nothing else in the window -- against its tensor and
shared-memory floors and against the composed kernel's readout phase? The readout component's instrument for the
iteration Blake ruled 2026-11-06 (the readout to its floor without changing the ruled design).

THE METHOD. One CTA an SM with a box's plan tenants. At the 32 x 32 x 32 box each of 64 windows a 216-slot union whose
32 digits hold ~8 entries each (slot `s` in digit `d`'s list with probability 10 / 256; or COLORED, the coloring's best
case, a list's entry `k` on residue `k mod 8`; two denser draws, 18 / 256 -- alt-k16's density, 2.2 reads a token --
and 40 / 256 past it, the listed regime's other end). At the 32 x 16 x 64 box (one token group by eight channel
groups; the planner's box at nl64k-alt-k4 and flagship-alt-k16) a read chunk as the cell's: nl64k-alt-k4's 120 slots,
each slot one digit's and any other's at 1 / 256 (the cell: a 117-slot union a box-window, ~8 entries a digit, 1.09
reads a token), flagship-alt-k16's 192 at 9 / 256 (its chunk of a 333-slot union, 1.5 reads a token). The image in
halves as sm_86 runs it (the planner's rule at both cells): the factors, a rendezvous, both halves' readouts with a
rendezvous between. The draw is stored to shared memory before two barriers and read inside the span: after one
barrier the empty span read 19.4K cycles a window (the warps not lined up by it, measured), after two 38. The floors are
computed on the host from the same draw: TENSOR each scheduler's two warps' MMAs (4 a tile at the first form's den on
every warp, 16 cycles at fp16 accumulate on sm_86, 32 at fp32, 8 on sm_80), SHARED the CTA's wavefront-cycles at the
first form's traffic (a load one, a store two). `--blocks 1` runs it at one CTA for racecheck.

THE MEASUREMENT (sm_86, cycles a warp-window, median of 7, the f16 panel at fp16 accumulate; random / colored slots):

| form | random | colored |
|---|---|---|
| as first built (the den on every warp, each token's factor a channel group) | 23,860 | 23,530 |
| the den on its own channel group | 21,412 | 22,002 |
| the entries' factors once a chunk, the tiles a tile ahead | 14,478 | 13,890 |
| the two tokens' adds overlapped | 13,482 | 13,157 |
| a pass's next digit a lane's, the den in its digits' num tiles, the panel's word one XOR of the row key, a tile its run's remainder | 10,746 | 10,387 |
| two halves a tile (rows 0-7 one digit's up to 8, rows 8-15 the next's), the den block a channel group's | 8,200 | 7,157 |
| the empty span (the harness's zero) | 38 | 38 |

At the denser draws, the one-digit tile against two halves a tile: 18 / 256 11,602 -> 11,139; 40 / 256 17,893 ->
18,766 (a run of 16 or more is a tile of one digit's two halves, its MMAs and slices twice). A table of the pass's
halves in place of the successor shuffle: 9,614 / 8,510 / 20,478 at 10 / 256 random and colored and 40 / 256 (its
serial build). Floors at the first form's traffic: tensor 4,023, shared 8,091. Bank conflicts are not what binds: the colored draw
has none (104K conflicted wavefronts against 6.4M random, ncu) and costs the same. RACECHECK at one CTA over every
variant: 0 hazards.

THE 32 x 16 x 64 BOX (sm_86, cycles a warp-window, median of 7; floors as above, a channel group's tiles 16 rows of a
digit):

| form | alt-k4's chunk, sm_86's form | the A100's form | fp32 panel | flagship-alt-k16's chunk |
|---|---|---|---|---|
| the den digits `d mod 8` (each image half one of a warp's two) | 7,513 | 7,981 | 8,014 | 11,160 |
| the den digits a group's two contiguous (one half's, one den pass a chunk) | 6,343 | 6,418 | 6,649 | 10,594 |
| floors (tensor / shared) | 2,052 / 4,102 | 4,104 / 4,102 | 4,104 / 4,615 | 3,408 / 5,966 |

At `d mod 8` the box's work fell (the slots a warp-window's tiles 16 -> ~11) but not its time: every pass runs whole
two-tile trips, and a den pass of one digit a half is a 269-instruction trip for about half a tile (the census at
nl64k-alt-k4: 124 tiles a CTA-window against ~92 the runs need). The ALU pipe binds (the lab's pipes: ~32-38 ALU
cycles an MMA against 9-10 shared and 4 tensor).

WHAT IT DECIDES. The readout's own form was 3x its shared floor from instructions and its tile's dependent chain, not
from conflicts; each change above was taken into the kernel and gated (the per-slot oracle, the SASS gate), the
composed readout phase following it (24.9K -> 13.1K -> 11.0K at nl64k-alt-k4). Halves a tile win at the headline
densities and lose 5% at twice alt-k16's: a pass of whole tiles for runs of 16 or more is the dense end's lever.
