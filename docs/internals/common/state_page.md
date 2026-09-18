# The state page's split bf16 planes

Mirror doc for `csrc/rola/src/common/state_page.cuh`. `Blocks<DV>` — the page's byte
geometry below — is the ONE definition every body's state I/O is laid out against: the
carry kernel's `box.cuh` (`BP::Page = Blocks<DV_>`) and the decode step both read its
offsets (`kBytes`, `kLo`, `kMassHi`, `kMassLo`) directly. `load`/`store`, the two
functions below that gather and scatter a whole page through those offsets, are NOT
currently called by either: both bodies inline their own sweep over the same layout
(the carry kernel's `state_copy`, `carry_kernel.md#state-io`; decode's per-leaf-row
access), so `load`/`store` document the layout's intended access pattern rather than a
function every consumer shares today. The channel-split carry body and the two-scan
backward pass this page once served were deleted whole in the 2026-08-30 carry-family
rebuild (`docs/internals/DELETIONS.md`, commit ef567f9) and have not been rebuilt; the
current carry kernel is the single body that replaced them.

## <a id="geometry"></a>The page, in bytes

Ruled by K50 on 2026-08-29 ("THE PAGE STORES SPLIT bf16 PLANES"). A page is one atom
= 16 leaves, and it holds

    [16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]

`hi || lo` is the fp32 word EXACTLY -- `hi` is the truncation `bits >> 16`, `lo` the
low half -- so the page carries the same bits an fp32 page did, in the same
`16 * (DV + 1) * 4` bytes, and `Blocks<DV>::kBytes` static-asserts that equality.
Nothing about the arena, the page table, the id space, the VMM commitment or the
32-bit displacement bound moves with this: it is a re-layout INSIDE the page.

What the re-layout buys is two things the fp32 rows could not give:

1. **A hi row is one line.** At `DV = 64` a hi row is 128 B and a page's hi plane is
   eight lines; the fp32 row was `(DV + 1) * 4` = 260 B, which is 8 sectors plus 4 --
   never line-aligned, and 2x the bytes.
2. **A read-only atom moves half the bytes.** An atom this call READS but does not
   WRITE has a lo plane no arithmetic can observe (see below), so `load` skips it.

The mass rides its OWN two blocks rather than as column `DV` of each row, which is
what keeps the value row a power-of-two stride in every access and closes the row
alignment question K50 left open (no padding).

## <a id="load"></a>`load`: the page into its SMEM image

The destination is the body's `[16][LDO]` fp32 staging block with the mass in column
`DV`: the split planes are rejoined on the way in, so a consumer of the staging block
indexes fp32 rows and never learns that the storage is split.

`with_lo` is the ACTIVITY fact, CTA-uniform per atom: an atom the call writes is
loaded from both planes and rejoined exactly; one it only reads is loaded `hi || 0`.
That is not an approximation. Under F1b the read side's A operand IS the hi plane
(`carry_kernel.cuh` builds `sa`/`ma` as `hi_bf16x2` of the accumulator and of the
mass), and the same declaration now holds in decode: **the readout operand is the hi
plane in every kernel**. A lo plane is therefore observable only through an atom's
own continued ACCUMULATION, which is exactly the written case.

One 32-bit access carries a PAIR of value columns, so the loop is over pairs and the
rejoin is one `PRMT` per pair per plane -- `__byte_perm(lo, hi, 0x5410)` and
`__byte_perm(lo, hi, 0x7632)`, one integer op per element. The hi-only case is a
shift and a mask of the same word, so it is one op per element as well. Every access
is 4 B aligned because `DV` is even (`Blocks` static-asserts it).

## <a id="store"></a>`store`: the accumulator back into both planes

A store is always BOTH planes: a page is stored only when the call WROTE it, and what
it writes is the exact fp32 accumulator, so dropping the lo half would lose the
sequence's carried precision rather than an unobservable half. The split is the
mirror `PRMT` pair -- `__byte_perm(v0, v1, 0x7632)` for hi and `0x5410` for lo -- and
the mass is two 16-bit stores per row.

The store is the reason the ENTRY load of a written atom must take both planes: the
exit sweep writes back what the accumulator holds, and the accumulator's low half is
whatever the entry load put there.
