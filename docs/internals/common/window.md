# `common/window.cuh` -- the window W, a build's

Mirrors `csrc/rola/src/common/window.cuh`.

A call's token axis is cut into windows of W tokens, global to every box and every kernel: the carry reads each
window's state out to the window's readers and folds its writes, the intra takes the pairs inside one window, the facts
pass emits each window's lists. W is not a data size. A window's resident rows
follow its read and write unions (the plan's chunks, multiples of 64 union slots); W sets how many tokens share one
window's fixed work -- the hand-off of the state into the image, the staging of the next window's inputs, the barriers.
A card with more shared memory amortizes that work over more tokens (ruled 2026-10-31: W and the other structural
constants are per-card configuration).

<a id="window-for"></a>
## Where W comes from

W is a build's trait, chosen in `hardware.json`'s `build` block ([`hardware.md#traits`](hardware.md#traits)): each
kernel's plan reads its build's (the facts pass's `FactsPlan<B>::kWindowTokens`), and each entry sizes its outputs and grid from its card's (`card_of(device).traits.window`,
[`select.md`](select.md)) -- one value both ways, so a host shape and the kernel it launches cannot disagree. Python
reads no window to shape a call: the layer's segment length is the binary's answer too
([`arch_runtime.md#segment-tokens`](arch_runtime.md#segment-tokens)).

The A100 (sm_80) is 1,024, every other card 512 (2026-11-04: interleaved on one A100 against 512, L 65,536 x 4 heads --
the sparse carry -35%, the op's sparse prefill -17.5% (0.72x FlashAttention), the layer at g 16 / 4 -21% / -16%; the
near-dense layer at g 1 +5.4% and the dense op +0.8%, the intra's work a token doubling with W). A call is a whole
number of the device's windows (the intra's check), so the A100 takes multiples of 1,024 tokens where sm_86 takes
multiples of 512, until the intra takes a partial last window.

<a id="token-field"></a>
## `window_bits(w)`, `window_heads(w)`

A union slot is 16 bits: its token within the window in the low `window_bits(w)` bits (log2 W), its query head above
them, bit 15 a gap slot. A write head's query heads are at most `window_heads(w)`: 64 at W = 512, 32 at W = 1,024.
