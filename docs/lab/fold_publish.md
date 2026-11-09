# `fold_publish`

Mirrors `measure/lab/experiments/fold_publish.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The hand-off with one image, at a 16-channel box (a 32 KB image, so the whole image fits sm_86). One image means two
coupling points a window: every warp finishes reading before any overwrites, and finishes writing before any reads.
So the window runs: the readout, a named barrier, the fold storing each list's rows into the image as its last step
lands, a named barrier. Separating the phases lets the read and fold pools go single-buffered, ~92.5 KB in all.

The readout is `read_image.cu`'s tokens-on-N form built for 16 channels (`read_image.cuh` with `READ_IMAGE_CHANNELS`
16): one m-tile and 12-word panel rows (`fold_publish.inputs.py`). The image has 32 digit planes of [list][16
channels], its 16-byte units swizzled `(2 list + (c ^ s(list))) ^ (digit & 7)`, so the fold's stores and the
readout's loads are both conflict-free. The fold stand-in carries the fold's shared traffic on the same lists:
- an entry, the write rows by `ldmatrix.trans` and the values;
- the coefficients by HMUL2;
- 4 m16n8k8 MMAs per 8 tokens into 64 fp32 state registers.

Its values are zero, so the outputs stay exact.

| sm_86, alt-k4, a CTA-window (16 channels) | cycles |
|---|---|
| a static image, free-running | 2538 |
| the hand-off | 3485 (+0.95K, +37%) |
| the hand-off, the readout's digits balanced over the warps | 3437 |
| the hand-off's barriers without its stores | 3101 (+0.56K) |
| a static image, digits balanced, free-running | 2662 |
| the readout alone | 1399 |

The hand-off's parts: two barriers +0.56K, the stores +0.38K, not all hidden in the fold's tail. Balancing the
readout's digits per window recovers only 0.05K. It cannot balance the fold, whose state stays in its warps'
registers, and free-running warps already average across windows.

The box is the larger cost. A 16-channel readout takes 1399 cycles for 16 channels, against `read_image`'s 1896 for 32,
1.48x per channel: each tile's read-row gather, entry load and shuffles serve half the channels. The variants' budgets
ran low on the votes and per-list bookkeeping (select 1.07 against 0.59, integer 5.2 against 4.1); the record is forced
and the form is not taken. Record: `measure/lab/results/fold_publish/sm_86.json`.
