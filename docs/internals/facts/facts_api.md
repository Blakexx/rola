# `csrc/rola/src/facts/facts_api.cuh` — the facts pass's host seam

Mirror doc for the pass's host entries and for the refusals they make before a launch
exists. The kernel is [`facts.md`](facts.md); the words' shape and meaning are
[`liveness_contract.md`](liveness_contract.md). `rola/ops/facts.py` is the Python
caller: `call_layout` and `call_statics` read a bundle's layout and its two sides'
static operands off its `sparse` flags and topology, and `facts` packs them into this argument list and names what
comes back.

<a id="rola-facts-forward"></a>

## `rola_facts_forward`

    rola_facts_forward(read[], write[], scale_write[], v_scale, widths, dense_read, mask_read, dense_write,
                       mask_write, write_list_level, read_list_level, read_list_group,
                       write_list_group) -> (words, unions, fm, write_lists[5], read_lists[5])

Each side's rows a level (read `[B, T, H, b_l]`, write `[B, T, H_w, b_l]` u8, `H_w` dividing `H` and `g = H / H_w`
at most 16 -- grouped queries, [`api.md`](../../api.md#grouped-queries); a dense level's are not read, so not checked),
the write levels' scales (their product the write side's shared scale) and the value scale (`[B, T, H_w]` fp32, any
strides). The entry finds its
card from the rows' device and ALLOCATES its outputs there at the card's window W (the pass's plan,
[`facts.md`](facts.md#window)), so no caller holds a window. They come back as one flat list, a write head's each
(`B·H_w`): `words` the contiguous int32 `[B·H_w, g + 1, rows, ceil(T/32)]`,
`unions` the contiguous int32 `[B·H_w, g + 1, groups, ceil(T/64)]` (slots `0 .. g - 1` its query heads' read side, `g`
its write side; [`liveness_contract.md`](liveness_contract.md#tile-unions)), `fm` the contiguous fp32
`[B·H_w, T, 2]` factor products ([`facts.md`](facts.md#columns)), and each side's five lists at its list level, the read side's
over its write head's `h = g` query heads' rows (the write side's `h = 1`), `groups = ceil(B / G)` at the side's list
group `G` -- entries `[B·H_w, windows, B, W h]` int32 (`slot | byte << 16`), the boundaries' prefix rows
`[B·H_w, windows, groups, G + 1, W h / 64]` int16, the groups' totals
`[B·H_w, windows, groups, W h / 64]` int32, the union by slot `[B·H_w, windows, groups, W h]` int16 and the unions' sizes
`[B·H_w, windows, groups]` int32 ([`facts.md`](facts.md#lists-out)); width 0 for a side that lists nothing. The pass writes `fm` in
full, the prefix rows and totals, the unions by slot and their sizes in full, each list's
entries to its end, and of `words` and `unions`
every sparse level's rows and groups; a dense level's are left as allocated (its words are the width mask, which a
reader takes from the statics, [`liveness_contract.md`](liveness_contract.md#statics)). `widths` are the layout's PADDED
widths; `dense_*` is a bitmask over levels and `mask_*` one bit a row, packed by
`rola.engine.facts.liveness.side_statics` -- the only place `b_l` appears. `read_list_group` and `write_list_group`
are each side's list group (`rola.ops.facts.ListsForm`), each union and its prefix rows spanning `G` digits
([`facts.md`](facts.md#list-groups)).

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* the depth is one to four levels, each side holds a row set a level and the write side a scale a level; a level is 1
  to 256 digits wide;
* a side's list level is a sparse level of the depth, or -1, and its list group 16 or 32 digits;
* the read side's heads are a whole number, at most 16, of the write side's;
* a sparse level's rows are u8 `[B, T, H, width]` with the first level's `B, T, H`, digits contiguous, under the row
  alignment law ([`liveness_contract.md`](liveness_contract.md#alignment));
* each side's dense bitmask names only levels inside the depth, and its digit mask is exactly `mask_words(rows)` words;
* **a mask bit names a digit inside its level's rows.** The kernel reads a digit iff its bit is set (on a sparse
  level), so a mask wider than the rows would read the next level's digits, or past the tensor; it is caught here, not
  by a wrong answer;
* the columns are CUDA fp32 `[B, T, H]`.

<a id="rola-facts-ledger-bind"></a>

## `rola_facts_ledger_bind`

Binds the `[8][7]` int64 phase ledger the next launches book into, or unbinds it with
`None` ([`facts.md`](facts.md#phase-ledger)); it issues no launch.
