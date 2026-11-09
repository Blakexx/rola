# `read_red`

Mirrors `measure/lab/experiments/read_red.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION: can the sparse readout's exits go straight to `num` by global reductions -- each lane's outputs scaled,
packed as f16 pairs and added by `red.global.add.noftz.f16x2`, with no shared panel and no drain -- and what does that
cost beside the readout's MMAs against exact s16 pairs added into the CTA's panel and drained after each pass?
(Blake, 2026-11-06: the sparse carry rebuilt with the panel gone.)

THE FORM: one CTA an SM, 8 warps, 16 CTAs sharing a head's `num` rows (the boxes of one head add into the same
tokens), 1,024 tokens a head, a CTA one 32-channel half of a row. A pass: each warp's tiles -- 8 tokens on N, one
`ldmatrix.x4` of their read rows and 4 bf16 MMAs against a held state -- then the exit. The panel form adds each
lane's two tokens' channel pairs `(g, g + 8)` as exact s16 pairs at the token's slot and drains after each pass (a
barrier, a warp its 32 slots two a request, each word read, cleared and added into `num`, a barrier). The direct
forms add each pair into `num` at the token's row: in the MMA fragment's order (4 requests a lane a tile, each warp
request four tokens' 32 bytes), as fp32 adds, or through the warp's own staging tile in shared memory (plain stores, a
`__syncwarp`, then each request two tokens' 64 contiguous bytes). A sample is the CTA's span over 256 passes.

THE RECORD: `measure/lab/results/read_red/sm_86.json` (RTX 3080 Ti), cycles a tile a warp:

| exit | 4 tiles a warp a pass (alt-k4's ~3.3 a window) | 16 tiles a pass |
|---|---|---|
| none (the tiles alone) | 260 | 260 |
| exact s16 pairs into the panel, drained a pass | 525 | 334 |
| f16 pairs straight to `num`, the fragment's order | 573 | 574 |
| fp32 straight to `num` | 1,112 | 1,115 |
| **f16 pairs through the warp's staging tile, 64 bytes a token a request** | **412** | 412 |
| f16 pairs relaid out in registers (four `shfl.sync` and a select a request) | 459 | 448 |
| a 64-channel box, staged, a token's 128 bytes a request (a tile 8 MMAs: twice the work) | 685 | 686 |
| f16 pairs straight, the fragment's order, one CTA a head (no shared rows) | 890 | -- |

The A100 (SXM4 40 GB, rented 2026-11-06; counters closed, the rows printed before the ncu pass), cycles a tile a
warp at 4 / 16 tiles a pass: tiles alone 138 / 130; panel and drain 395 / 233; the fragment's order 472 / 474; fp32
894 / 894; **staged 350 / 350**; shuffles 363 / 362; the 64-channel box staged 601 / 601 (~300 a 32-channel tile).
Its MMAs take half sm_86's cycles and its exits about as many (~5.5 cycles a line request), so the exit is most of a
sparse tile there: staged 11% under the panel at 4 tiles a pass, the 64-channel box 24% under, and the panel 33%
under the staged form at 16.

WHAT IT SETTLES (sm_86): the direct exit's cost is the SM's global request path -- about 5 cycles a line request (the
`drain` lab's floor) -- so its address order decides it. In the MMA fragment's order every request touches four
tokens' rows (four lines), twice the drain's two, and it runs 9% over the panel and drain at sparse tile counts;
through the warp's own staging tile each request is two tokens' 64 bytes and it runs 21% under them (412 against
525). Its cost is a constant a tile, where the panel's drain is shared by every tile that touched a token in the
pass: past about 8 tiles a warp a pass the panel is cheaper (16 tiles: 334 against 412), so a denser window wants a
token's tiles combined before its request. fp32 adds take twice f16's requests.

WHY THE FRAGMENT'S ORDER SCATTERS (Blake asked whether the results could be made batchable natively): an MMA
accumulator register spans an 8 x 8 block across the warp -- rows `g` (8 lanes' worth) by column pairs `2t, 2t + 1`
(4) -- so with channels on M and tokens on N one register, packed as the pair `(g, g + 8)`, is 16 channels of each of
4 tokens: 4 rows of 32 bytes, 4 lines a request. Tokens on M is worse (8 tokens' 16 bytes). No fragment holds 1-2
token rows in one register; a relayout across the 8 x 8 blocks is what batches it, and `movmatrix` (an in-register 8 x
8 transpose) only transposes within a block. The shared staging tile is the cheaper relayout on sm_86 (4 stores and 4
loads a lane a tile against 16 shuffles); a wider box makes each request a full line: the 64-channel box's exit costs
~165 cycles a tile over its MMAs (~2 x 260) for 64 channels against the 32-channel box's ~152 for 32 -- half a
channel's requests, ~343 a 32-channel tile, under the panel even at 16 tiles a pass (334). On sm_90 the relayout is
native: `stmatrix` stores fragments by rows and the bulk reduction (`cp.reduce.async.bulk`) adds a staged tile into
global memory in one instruction. Spreading the rows over more heads (no
two CTAs on a row) runs slower, not faster: L2 serves the shared rows' atomics. The A100's rates are owed (the next
rented session).
