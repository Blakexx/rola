# `facts_block`

Mirrors `measure/lab/experiments/facts_block.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What shape lets the facts pass's list build hide under its row stream, at or under 1.15x its DRAM floor
(KERNEL_STANDARDS §28). Rows `[level-side, T, H, 256]` u8, each token's head live in k random digits; level-side 0
the write list level (8-byte records whose coefficient needs the byte, the token's `(F, m)` pair and the window's
maxima of b x gains), 1 the read list level (2-byte slot lists from its bits), 2-3 voted only. One block of sixteen
warps an SM, warp w the window's w-th 32 tokens; the build is the facts pass's walk build (four warps a 32-digit
group). Time against the DRAM floor -- the rows, the gains, the bits and union words, the lists and their pass starts
-- at a flat read's rate measured in the same run (539 bytes a chip cycle); each build's lists checked against the
live count and, by checksum, bit for bit against the first form's (they agree). sm_86, x the DRAM floor:

| form | vote only | k 4 | k 16 | resident k 4 / k 16 |
|---|---|---|---|---|
| BULK: 64-digit items over two whole-window buffers, three barriers an item, bit words stored from the vote | 1.18 | 1.29 | 1.60 | 1.11 / 1.40 |
| TILED: the same, bits through a shared tile as 64-byte runs, the maxima in the build's first walk | 1.10 | 1.28 | 1.27 | 0.72 / 1.08 |
| SPLIT: the write level two 64-digit halves in one held window, the others 128-digit items through a per-warp ring of 8-token stages, one barrier an item | 1.07 | 1.16 | 1.20 | 0.54 / 0.88 |

What each step found (each measured in this lab on the way; the dropped forms are in its history):
- THE VOTE'S BIT STORES WERE ITS WHOLE COST: each warp storing one 4-byte word to each of 32 rows took ~4K cycles an
  item; through a tile and out as 64-byte runs after the barrier the vote runs at 4x the floor's rate (resident 0.23x).
- THE MAXIMA OVER LIVE ENTRIES: the write level's per-digit maxima of b x gains over every byte (64 f16 products a lane
  a group in the vote) move to the build's first walk, which visits only the live entries.
- RESIDENT IS NOT ENOUGH: TILED's build computes at 0.72x the floor at k 4 yet streams at 1.28x -- its two buffers hold
  the write level's window through the build, so nothing is in flight under it. A third 32 KB buffer does not fit beside
  the tiles and tables in sm_86's 99 KB. Only the write level needs its bytes after the vote, so SPLIT holds that level
  alone (64 digits, 32 KB) and streams the rest as 128-byte runs through a small ring, each stage free once voted.
- AN sm_86 MBARRIER WAIT IS A POLLED SHARED WORD: in SPLIT the eight warps waiting on the write level's eight builders
  spun 30.7M of 116.5M warp instructions (26%); polls with `__nanosleep(64)` between them return the slots (resident
  0.56 -> 0.54 at k 4, 0.90 -> 0.88 at k 16).
- FORMS THAT LOST: popcount counts a (digit, word) with independent per-entry ranks (73M warp instructions at k 4
  against the walk build's 28M: per-cell work is paid at every density); one barrier a quad of level-sides on two
  slots (resident 0.55 / 0.84 but streaming 1.50 / 1.45, converted from bytes a cycle by the same cells' bytes moved:
  the held window leaves one slot to stream through); the read level built on
  eight warps beside the others' votes (k 16 1.20 -> 1.31: its latency doubles); a lane's walk batched four entries with
  predicated effects (resident k 16 0.88 -> 0.99: more instructions and registers than the overlap returns).
- OPEN: k 16 is 1.20x with its compute at 0.88x the floor; the build's ~156 instructions a live entry (both walks) and
  its latency at eight to sixteen warps are what is left to cut. The vote-only 1.07x is this geometry's floor on sm_86:
  one level-side in four reads 64-byte runs.

Record: `measure/lab/results/facts_block/sm_86.json`.
