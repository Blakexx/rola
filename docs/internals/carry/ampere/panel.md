# `carry/ampere/panel.cuh` — the panel and the drain

Mirrors `csrc/rola/src/carry/ampere/panel.cuh`. A read chunk's PANEL is its slots' `num` words, the readout's adds landing
there (`readout.md#chunk`), and its DEN WORDS beside it; the DRAIN takes both to HBM once every warp has read the chunk.
The read buffer holds them after the chunk's rows (`read_panel`, `read_dens`; `schedule.md#buffers`), and the chunk's
fill clears them (`parts.md#fill`).

<a id="words"></a>
## The words

A slot's row holds the box's 32 channels at the reduce dtype (`panel_row_bytes`): f16 pairs, 16 words (64 bytes), word
`k` channels `2 k` and `2 k + 1`; or fp32, 32 words (128 bytes, a word a channel). A readout add touches a tile's 8
tokens (rows `g`) at a channel group's words: f16 its 4 pair words, one 4-byte load and store a lane; fp32 its 8
channels, one 8-byte load and store a lane. The row's words are ROTATED by the slot as an XOR of its 16-byte units
(`panel_word`): f16 `(64 s + 16 ((s >> 1) & 3)) ^ 4 w` -- two slots share a 128-byte line, and the rotation takes a
pair's two slots to its halves and four pairs to distinct words of them -- fp32 `(128 s + 16 (s & 7)) ^ 4 w`, a slot a
line. So 8 slots of distinct residues (a dense chunk's consecutive slots) meet 32 banks an add on either dtype; a list's
rank slots meet them where their ranks run consecutively, as the pool rows do (`parts.md#swizzle`). The f16 row's base
`64 s + 16 ((s >> 1) & 3)` is the slot's 64-byte pool row's byte offset (`readout.md#factors`), so a readout add's word
is one XOR of the offset its entry holds; the XOR leaves bits 0-3 alone, so an fp32 lane's two channels stay adjacent.

AT 64 CHANNELS the rows double: f16 32 words (128 bytes) rotated as the 32-channel fp32 row, `(128 s + 16 (s & 7)) ^ 4
w`; fp32 64 words (256 bytes), `(256 s + 32 (s & 7)) ^ 4 w`. The read row stays 64 bytes (the readout's K is 32 lists),
so an entry's key is no longer the panel's base: it holds the base in 16-byte units, `8 s + (s & 7)`
(`readout.md#factors`), and an add's address is two operations from it -- f16 `(panel + 4 tq) + 16 (key ^ cg)`, fp32
`(panel + 8 tq) + 32 (key ^ cg)`, the lane's word `4 cg + tq` (f16) or `8 cg + 2 tq` (fp32) of the rotation above. 8
slots of distinct residues meet 32 banks on either dtype, as at 32 channels.

THE DEN WORDS are `kDenWords` fp32 a slot, one a readout channel group (the group that reads a digit's den adds it at
its own word `cg`, `cg` the box's channel groups; `handoff.md#den`), 16 bytes at 32 channels and 32 at 64: no two warps add into one
word, so the den needs no atomics either; the drain sums them.

<a id="drain"></a>
## The drain

`drain<P, F32>`, after every warp's adds (the caller's rendezvous): warp `w` takes slots `w + 8 i`, two steps a trip so
their loads overlap. F16 PAIRS: two slots a step, a lane a word (its half of the warp a slot), each added by
`red.global.add.noftz.f16x2` at the slot's row of `num` (`token_row`: the slot's query head and window token) -- the
word's channels in their natural order, so `num`'s f16 pairs read as its f16 values (`rola.ops.prefill.unpack_num`).
FP32: a slot a step, a lane a channel, `red.global.add.f32`. At 64 channels both dtypes take a warp a slot, a lane its
word (f16) or its two words 32 apart (fp32). Then THE DEN: a thread a slot, its words summed (four, or eight at 64
channels: two 16-byte loads), one `red.global.add.f32` into `den`. Each address is a 32-bit word offset from the head's base (`word_at`, one widening
multiply-add; the host holds a write head's rows under 2^32 words). The CTAs join only here (R22: independent units,
joined by atomics): the cross-CTA adds (`red_add_h2`, `red_add_f32`) are unpredicated, each one `red.global`.

NO BRANCH AROUND AN ADD (lab [`drain`](../../../lab/drain.md): a predicated `red.global` is a convergence region around
each add). A GAP slot (the union's tail, `kGap`) holds no token; the union is padded to whole 8-token steps
(`facts.cu`), so its gaps lie in its last step and a chunk's in its last 8 slots (a dense read side stages no union:
its slots are the window's tokens, a slot its token, no gap -- the fill's unlisted case): one load a lane, a ballot and a
popcount give the chunk's token count `n`, a warp-uniform bound. A step past `n` (the warp's last trip) adds zero at the
trip's first slot's row -- a token's own, never one row every CTA's dead lanes would share (the lab's serialized
dummy row). The f16 pairs meet across CTAs at one scale, the token's byte, which every CTA reads the same.

MEASURED (sm_86, two sparse cells; no dense or sm_80 reading, `tools/phase_ledger.py`): the branch-free form 2,758 ->
1,623 cycles a warp-window at nl64k-alt-k4 (its floor ~1.1K: ~215 line requests at ~5 cycles, every SM draining),
5,811 -> 3,456 at flagship-alt-k16. Its loop is ~23
instructions an add (a slot's select, its panel word, its row); the budget's ~4 (the design-floor model) is a
token's row precomputed, which the union does not carry.
