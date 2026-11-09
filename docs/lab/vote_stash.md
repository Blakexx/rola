# `vote_stash`

Mirrors `measure/lab/experiments/vote_stash.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What the facts pass's first stage pays to keep each digit's live bytes (a STASH, one byte an entry, which a second
stage reads to form the lists' coefficients), beside the vote. Rows `[level-side, T, H, 256]`, each token's head live in
k random digits; one block of eight warps an SM, each warp taking (window, level-side, head, 128-digit span) items in
order and streaming each as 16 blocks of 32 tokens x 128 bytes through a ring of three 4 KB stages (two ahead, issued
on into the next item, eight lanes a token's run); each block's four 32-digit sub-words voted as the facts pass votes a
word and stored word-major. A STASH A DIGIT: each lane's digit's live bytes stored at the digit's own region (512 bytes
a (window, digit)) at its running count. PACKED: a sub-word's live bytes appended to the item's one stash, lane by
lane after a scan of the lanes' counts, each lane's in token order. Bytes a chip cycle (the flat stream 541):

| k of 256 | vote | + stash a digit | + stash packed |
|---|---|---|---|
| 4 | 454.4 | 190.2 | 440.6 (-3.0%) |
| 16 | 454.7 | 111.9 | 419.7 (-7.7%) |
| 64 | 455.2 | 80.9 | 272.4 (-40%) |

A stash a digit amplifies its writes: at k 16, 93 MB of DRAM writes for 17 MB of stash, each byte store dirtying a
sector of one of 128 regions that leaves L2 before its neighbours are written. Packed, the writes are the stash's
bytes. What remains is the lanes' serial walk, ~2.4 warp instructions an entry at k 16: a warp walks as long as its
busiest lane, so at 25% density the stage halves; a lane-parallel extraction is owed for dense list levels. The vote's
own 84% of the flat stream is this lab's one block of eight warps (issue 10%, DRAM 93% at k 16 under ncu). TILED (each
warp's item bits in an 8 KB shared tile, its rows out as 64-byte runs at the item's end: the on-chip footprint a
second stage needs), packed stash: 6 warps an SM 424.0 (k 4) / 398.1 (k 16), 4 warps with three stages 407.8 / 297.5,
against 8 warps untiled 441.6 at k 4 -- the tile's footprint costs warps, and four warps cannot hide the walk at k 16.
Record: `measure/lab/results/vote_stash/sm_86.json`.
