# `intra_partial.cu`, `intra_partial.cuh`

Mirrors [`csrc/rola/src/intra/ampere/intra_partial.cu`](../../../../csrc/rola/src/intra/ampere/intra_partial.cu) and its header.

THE PARTIAL CALL'S ARMS: `intra_kernel<BuildTag<B>, D, kF16, false>` at each depth and accumulation, and
`launch_partial<B, D, kF16>`, the launch the host entry (`intra.cu`) calls at the device's build for a call whose length
is not a whole number of the device's windows. They are instantiated at every build the binary carries (`ROLA_BUILDS`,
[`../../common/select.md#carried`](../../common/select.md#carried)), each build's kernels in its own image (`in_image`); a
build outside the image returns `cudaErrorInvalidDeviceFunction`, which the card's admission has refused before.
They compile here, in a translation unit of their own, so `intra.cu`'s unit -- the whole-window arms with the torch
seam -- compiles as it did (KERNEL_STANDARDS §16 addendum 3's compile ledger, per unit); this unit includes no torch
header and returns the launch's error for the seam to check. What the arms bound and why they are arms:
[`intra_kernel.md#partial-window`](intra_kernel.md#partial-window).

The launch raises the arm's dynamic shared memory attribute once a footprint, the whole-window launch's rule
([`../intra.md`](../intra.md#the-translation-unit)), on its own high-water mark, and launches the call's `(windows × pieces,
B·H)` grid, `windows` its tokens over W rounded up.
