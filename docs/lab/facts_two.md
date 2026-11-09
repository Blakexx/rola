# `facts_two`

Mirrors `measure/lab/experiments/facts_two.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Whether the facts pass as two plain kernels (a STREAM kernel that reads every row once, a BUILD kernel that ranks from
its intermediates, the call chunked and the two overlapped on two CUDA streams by events alone) comes near its DRAM
floor, and what the intermediates cost. The `facts_block` cell (rows `[4 level-sides, 65536, 4, 256]` u8, 268 MB, k live
digits a token-head), RTX 3080 Ti. The stream kernel's unit is (window of 512 tokens, level-side, head, 32-digit group):
a CTA of eight warps, a 16 KB window of 32-byte runs by `cp.async`, the vote, the tile unions, and on a list level-side
the group's union tokens, each live entry's slot and digit and its final coefficient (write: the f16 pair at 2^(4 - E),
E from the unit's own maxima; read: f16 byte / 256) in (digit, token) order, and the write lists' exponents. The build
is a warp an eighth of a unit: a lane an entry, `match.any` on (digit, pass, residue), one shared atomic a key, ranks by
lane order, round-robin places after a counting pass. A fused variant places in the stream CTA itself. Checked against a
naive per-list reference and the live count; floors at a flat read's rate (855 GB/s): required bytes 338 us (k 4) and
386 us (k 16), the pair's own floor (the stash written and read) 363 and 482.

| k | stream alone | build alone | pair, 4 chunks | pair, 1 chunk | fused unit kernel |
|---|---|---|---|---|---|
| 4 | 379 us | 130 us (101 at one chunk) | 497 us, 1.47x required | 456 us, 1.35x | 394-418 us, 1.17-1.24x |
| 16 | 423 us | 251 us (210 at one chunk) | 645 us, 1.67x | 604 us, 1.56x | 484-528 us, 1.25-1.37x |

Findings (each measured on the way; the record is `measure/lab/results/facts_two/sm_86.json`, the narrative the findings
document):
- THE BUILD'S INSTRUCTIONS: a lane a digit's words walked 156-ish instructions an entry and 33% of the lanes; a lane an
  entry from the flat list is 17 instructions an entry at k 4 and 7 at k 16, issue-bound at 77% eligible once a launch
  has 2000 warps.
- OVERLAP DOES NOT PAY AT FOUR CHUNKS: a stream chunk beside a build takes 117 us against 98 alone, each launch gap is
  4-8 us and the last build is a 30 us drain; the pair is no faster than the stream then the build in one chunk each
  (447 against 497). Persistent streams, capped build grids and geometric chunks all lost.
- THE STASH COMES BACK FROM DRAM: the build's L2 read hit rate is 10-12% at k 4 and 23% at k 16 -- a chunk's 67 MB of
  rows pass through the 6 MB L2 between the stash's write and its read.
- A COLLECTIVE IN A LOOP WHOSE COUNT THE COMPILER CANNOT PROVE UNIFORM is wrapped in a convergence call (the lab refuses
  it); one broadcast shuffle of the count fixes it.
