# `digit_box`, `digit_box32`, `list_box`

Mirrors `measure/lab/experiments/digit_box.cu`, `digit_box32.cu` and `list_box.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Which box keeps the carry's hand-off from the fold's warps to the readout's cheapest on sm_86 (99 KB of shared
memory). The candidates:
- 32 lists x 16 digits (`digit_box`), its whole image resident (32 KB);
- 16 lists x 32 digits (`list_box`), its whole image resident (32 KB);
- 32 x 32 (`digit_box32`), its 64 KB image in two parts, since the whole does not fit beside the rest.

All three are one file, `digit_box.cu`, built per shape. The readout is `read_image.cu`'s 32-channel tokens-on-N form;
at 16 lists it takes one k-tile on 32-byte read rows, and each list box adds its partial output. The fold stand-in
carries the fold's shared traffic: per step an entry, the write rows by `ldmatrix.trans`, the values from a zero row
(addressed as a real row), the coefficients, and m16n8k8 MMAs. Both skip a digit with no readers and a list with no
writes. The image's 16-byte units are swizzled `(4 list + (c ^ s(list))) ^ (digit & 7)`, so stores and loads are both
conflict-free.

The hand-offs:
- RESIDENT: the readout, a named barrier, the fold storing each list it folded as its last step lands, a named
  barrier. A list with no writes keeps its rows.
- TWO-PART: the fold, then for each half of the digits a barrier, the half's stores (every list), a barrier, its
  readout.

Two cells (`*.inputs.py`): nl64k-alt-k4 as drawn, and the same with each entry kept with probability 1/16, so lists
average ~0.5 writes a window. The thinned cell emulates ~1M leaves for the same fan-out (no central cell is that
large): a box's state, and so its transfer, keeps its size while its work drops 16x. All checked outputs are exact.

Per 512 tokens over 32 lists x 32 digits x 32 channels (two CTA-windows for a halved box), sm_86:

| | alt-k4 | thinned 1/16 |
|---|---|---|
| 32 x 32, a static image (no transfer: the ideal) | 3929 | 1694 |
| 32 x 32, the two-part hand-off | 5926 (1.51x) | 3475 (2.05x) |
| 32 x 16, a static image / the resident hand-off | 5025 / 6795 (1.73x) | 2302 / 3736 (2.21x) |
| 16 x 32, a static image / the resident hand-off | 5080 / 6700 (1.71x) | 2326 / 3720 (2.20x) |

Residency does cut the hand-off itself:
- alt-k4: 1.62K-1.77K against the two parts' 2.0K;
- thinned: 1.39K-1.43K against 1.78K (unwritten lists store nothing).

But a halved box costs 28-37% before any transfer:
- 32 x 16: the fold loads each write token's values in both digit boxes (fold 2 x 1470 against 1939);
- 16 x 32: the readout runs each entry in both list boxes and adds its partial twice (readout 2 x 1577 against 2128).

So the two-part 32 x 32 is the cheapest at both cells. The transfer outweighs the work at the thinned cell (1.78K
against 1.69K): it scales with the state, not the writes, and its two barriers a window stay. Records:
`measure/lab/results/digit_box/sm_86.json`, `measure/lab/results/digit_box32/sm_86.json`,
`measure/lab/results/list_box/sm_86.json`.
